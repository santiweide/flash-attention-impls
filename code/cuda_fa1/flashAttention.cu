#include <cuda_runtime.h>
#include <sys/time.h>
#include <cstdio>
#include <cmath>

__global__ void flash_attention_forward(
    const float* Q, const float* K, const float* V,
    float* O, float* l, float* m,
    int B, int H, int N, int d, int M)
{
    int bh = blockIdx.y;
    int b  = bh / H;
    int h  = bh % H;
    if (b >= B || h >= H) return;

    int Bc = (int)ceilf((float)M / (4.0f * (float)d));
    int Br = (Bc < d) ? Bc : d;

    int Tr = (N + Br - 1) / Br;
    int Tc = (N + Bc - 1) / Bc;


    int tile = blockIdx.x;
    if (tile >= Tr) return;

    int qi_start = tile * Br;
    int br_size  = (Br < N - qi_start) ? Br : (N - qi_start);

    size_t offset = ((size_t)b * H + h) * (size_t)N * (size_t)d;
    size_t seq_offset  = ((size_t)b * H + h) * (size_t)N;

    const float* Q_bh = Q + offset;
    const float* K_bh = K + offset;
    const float* V_bh = V + offset;
          float* O_bh = O + offset;
          float* l_bh = l + seq_offset;
          float* m_bh = m + seq_offset;

    for (int r = threadIdx.x; r < br_size; r += blockDim.x) {
        int gi = qi_start + r;
        for (int c = 0; c < d; ++c) O_bh[gi*d + c] = 0.0f;
        l_bh[gi] = 0.0f;
        m_bh[gi] = -INFINITY;
    }
    __syncthreads();

    extern __shared__ float shared_mem[];
    float* Qi = shared_mem;
    float* Kj = Qi + Br * d;
    float* Vj = Kj + Bc * d;


    for (int j = 0; j < Tc; ++j) {
        int kj_start = j * Bc;
        int bc_size  = (Bc < N - kj_start) ? Bc : (N - kj_start);
        if (bc_size <= 0) break;

        for (int t = threadIdx.x; t < bc_size * d; t += blockDim.x) {
            int r = t / d, c = t % d;
            Kj[r*d + c] = K_bh[(kj_start + r)*d + c];
            Vj[r*d + c] = V_bh[(kj_start + r)*d + c];
        }
        __syncthreads();


        for (int r = threadIdx.x; r < br_size; r += blockDim.x) {
            int gi = qi_start + r;
            

            float q_row[64];
            for (int t = 0; t < d; ++t) q_row[t] = Q_bh[gi*d + t];


            float m_row = m_bh[gi];
            float l_row = l_bh[gi];
            float O_row[64];
            for (int t = 0; t < d; ++t) O_row[t] = O_bh[gi*d + t];


            float s_scores[128];
            float m_til = -1e9f;
            for (int c = 0; c < bc_size; ++c) {
                float s = 0.0f;
                for (int t = 0; t < d; ++t) {
                    s += q_row[t] * Kj[c*d + t];
                }
                s_scores[c] = s;
                if (s > m_til) m_til = s;
            }


            float P_til[128];
            float l_til = 0.0f;
            for (int c = 0; c < bc_size; ++c) {
                P_til[c] = expf(s_scores[c] - m_til);
                l_til += P_til[c];
            }

            
            float m_new = fmaxf(m_row, m_til);


            float a = expf(m_row - m_new);
            float b = expf(m_til - m_new);
            float l_new = a * l_row + b * l_til;

            
            for (int col = 0; col < d; ++col) {
                float acc = 0.0f;
                for (int c = 0; c < bc_size; ++c) {
                    acc += P_til[c] * Vj[c*d + col];
                }
                O_row[col] = (a * l_row * O_row[col] + b * acc) / l_new;
            }

            for (int t = 0; t < d; ++t) O_bh[gi*d + t] = O_row[t];
            l_bh[gi] = l_new;
            m_bh[gi] = m_new;
        }
        __syncthreads();
    }
}

int main(int argc, char** argv) {
    int B    = (argc>1)? atoi(argv[1]) : 1;
    int H    = (argc>2)? atoi(argv[2]) : 8;
    int N    = (argc>3)? atoi(argv[3]) : 512;
    int d    = (argc>4)? atoi(argv[4]) : 64;
    int M    = (argc>5)? atoi(argv[5]) : 4096;
    int runs = (argc>6)? atoi(argv[6]) : 50;

    int Bc = (int)ceilf((float)M / (4.0f * (float)d));
    int Br = (Bc < d) ? Bc : d;
    int Tr = (N + Br - 1) / Br;

    dim3 grid(Tr, B*H);
    dim3 block(Br);
    size_t shmem = (size_t)(Br*d + Bc*d + Bc*d) * sizeof(float);


    size_t size_QKV = (size_t)B * H * N * d * sizeof(float);
    size_t size_LM  = (size_t)B * H * N * sizeof(float);
    float *Q,*K,*V,*O,*l,*m;
    cudaMalloc(&Q,size_QKV);
    cudaMalloc(&K,size_QKV);
    cudaMalloc(&V,size_QKV);
    cudaMalloc(&O,size_QKV);
    cudaMalloc(&l,size_LM);
    cudaMalloc(&m,size_LM);


    cudaMemset(Q, 0, size_QKV);
    cudaMemset(K, 0, size_QKV);
    cudaMemset(V, 0, size_QKV);
    cudaMemset(O, 0, size_QKV);
    cudaMemset(l, 0, size_LM);
    cudaMemset(m, 0, size_LM);

    flash_attention_forward<<<grid, block, shmem>>>(Q,K,V,O,l,m,B,H,N,d,M);
    cudaDeviceSynchronize();

    timeval t0, t1;
    gettimeofday(&t0, nullptr);
    for (int i=0;i<runs;i++) {
        flash_attention_forward<<<grid, block, shmem>>>(Q,K,V,O,l,m,B,H,N,d,M);
    }
    cudaDeviceSynchronize();
    gettimeofday(&t1, nullptr);

    long sec = t1.tv_sec - t0.tv_sec;
    long usec = t1.tv_usec - t0.tv_usec;
    double avg_us = (sec*1e6 + usec) / (double)runs;

    double bytes_per_call =
        3.0 * size_QKV +   
        1.0 * size_QKV +  
        2.0 * size_LM;     

    double GBps = (bytes_per_call / (avg_us * 1e-6)) / 1e9;

    printf("Avg latency: %.2f ms\n", avg_us/1000.0);
    printf("throughput: %.2f GB/s\n", GBps);

    double flops = 4.0 * (double)B * H * N * N * d;        
    double gflops_per_s = (flops / (avg_us * 1e-6)) / 1e9; 

    printf("compute: %.3f GFLOPs/s\n", gflops_per_s);

    cudaFree(Q); cudaFree(K); cudaFree(V);
    cudaFree(O); cudaFree(l); cudaFree(m);
    return 0;
}

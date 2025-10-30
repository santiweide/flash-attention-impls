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


    // 预加载当前 Q tile 到共享内存 Qi
    for (int t = threadIdx.x; t < br_size * d; t += blockDim.x) {
        int r = t / d;
        int c = t % d;
        Qi[r*d + c] = Q_bh[(qi_start + r)*d + c];
    }
    __syncthreads();

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
            

            float q_row[128];
            for (int t = 0; t < d; ++t) q_row[t] = Qi[r*d + t];


            float m_row = m_bh[gi];
            float l_row = l_bh[gi];
            float O_row[128];
            for (int t = 0; t < d; ++t) O_row[t] = O_bh[gi*d + t];


            float s_scores[128];
            float m_til = -1e9f;
            float scale = 1.0f / sqrtf((float)d);  // 添加缩放因子 1/sqrt(d)
            for (int c = 0; c < bc_size; ++c) {
                float s = 0.0f;
                for (int t = 0; t < d; ++t) {
                    s += q_row[t] * Kj[c*d + t];
                }
                s_scores[c] = s * scale;  // 应用缩放因子
                if (s_scores[c] > m_til) m_til = s_scores[c];
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


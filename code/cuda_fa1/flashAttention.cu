#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <sys/time.h>
#include <cstdio>
#include <cmath>

__global__ void flash_attention_forward(
    const __half* Q, const __half* K, const __half* V,
    __half* O, float* l, float* m,
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

    const __half* Q_bh = Q + offset;
    const __half* K_bh = K + offset;
    const __half* V_bh = V + offset;
          __half* O_bh = O + offset;
          float* l_bh = l + seq_offset;
          float* m_bh = m + seq_offset;

    for (int r = threadIdx.x; r < br_size; r += blockDim.x) {
        int gi = qi_start + r;
        l_bh[gi] = 0.0f;
        m_bh[gi] = -INFINITY;
    }
    __syncthreads();

    extern __shared__ char shared_mem[];
    __half* Qi = (__half*)shared_mem;
    __half* Kj = Qi + Br * d;
    __half* Vj = Kj + Bc * d;
    
    // O_accum: 使用float累积器，只在最后转换为half_t
    float* O_accum = (float*)(Vj + Bc * d);


    // 预加载当前 Q tile 到共享内存 Qi
    for (int t = threadIdx.x; t < br_size * d; t += blockDim.x) {
        int r = t / d;
        int c = t % d;
        Qi[r*d + c] = Q_bh[(qi_start + r)*d + c];
    }
    
    // 初始化 O_accum 为 0（使用float精度）
    for (int t = threadIdx.x; t < br_size * d; t += blockDim.x) {
        O_accum[t] = 0.0f;
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
            for (int t = 0; t < d; ++t) q_row[t] = __half2float(Qi[r*d + t]);


            float m_row = m_bh[gi];
            float l_row = l_bh[gi];


            float s_scores[128];
            float m_til = -1e9f;
            float scale = 1.0f / sqrtf((float)d);  // 添加缩放因子 1/sqrt(d)
            for (int c = 0; c < bc_size; ++c) {
                float s = 0.0f;
                for (int t = 0; t < d; ++t) {
                    s += q_row[t] * __half2float(Kj[c*d + t]);
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

            // 更新O_accum：先应用correction，再累加新的值
            float correction = a;  // exp(m_row - m_new)
            for (int col = 0; col < d; ++col) {
                // 应用correction到已有的累积值
                O_accum[r * d + col] *= correction;
                
                // 计算新的贡献
                float acc = 0.0f;
                for (int c = 0; c < bc_size; ++c) {
                    acc += P_til[c] * __half2float(Vj[c*d + col]);
                }
                // 累加新值
                O_accum[r * d + col] += b * acc;
            }

            l_bh[gi] = l_new;
            m_bh[gi] = m_new;
        }
        __syncthreads();
    }
    
    // 最终归一化并写回（只在最后转换一次half_t，减少精度损失）
    for (int r = threadIdx.x; r < br_size; r += blockDim.x) {
        int gi = qi_start + r;
        float scale = (l_bh[gi] == 0.0f) ? 0.0f : 1.0f / l_bh[gi];
        for (int t = 0; t < d; ++t) {
            float val = O_accum[r * d + t] * scale;
            O_bh[gi * d + t] = __float2half(val);
        }
    }
}


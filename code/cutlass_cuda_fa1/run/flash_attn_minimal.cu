/******************************************************************************
 * Minimal Flash Attention Implementation with Cutlass
 * 
 * 这是一个教学性质的Flash Attention实现，保留核心算法但简化了许多细节
 * 
 * 核心思想：
 * 1. 将Q,K,V分块，减少HBM访问
 * 2. 在线更新softmax统计量(max和sum)
 * 3. 在一个kernel中融合所有操作
 * 
 * 简化：
 * - 固定块大小和head维度
 * - 不支持causal mask, dropout等
 * - 仅支持FP16
 * - 简化的内存布局
 ******************************************************************************/

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/memory.h>
#include <cmath>
#include <algorithm>

// ==================== 配置参数 ====================

// 如果遇到共享内存不足的问题，可以尝试以下配置：
// 选项1 (当前): 64x64 块，需要 ~72KB 共享内存
// 选项2: 32x32 块，只需要 ~20KB 共享内存
// 选项3: 64x32 块，需要 ~44KB 共享内存

constexpr int kBlockM = 64;      // Q的块大小 (行) - 可以改为32
constexpr int kBlockN = 64;      // K,V的块大小 (列) - 可以改为32  
constexpr int kHeadDim = 64;     // Head维度 (固定)
constexpr int kNThreads = 128;   // 每个block的线程数

// ==================== 工具函数 ====================

// 安全的除法
__device__ __forceinline__ float safe_divide(float a, float b) {
    return b == 0.0f ? 0.0f : a / b;
}

// ==================== Softmax在线更新 ====================

/**
 * 在线softmax更新
 * 
 * 给定旧的max(m_old), sum(l_old)和新的scores(S_new)，
 * 计算更新后的max(m_new), sum(l_new)和新的P = exp(S - m_new)
 * 
 * 算法：
 * m_new = max(m_old, max(S_new))
 * l_new = exp(m_old - m_new) * l_old + sum(exp(S_new - m_new))
 */
struct OnlineSoftmax {
    float m;  // running max
    float l;  // running sum of exp
    
    __device__ OnlineSoftmax() : m(-INFINITY), l(0.0f) {}
    
    // 更新统计量并返回缩放因子
    __device__ void update(const float* scores, float* P, int len) {
        // Step 1: 找到新的最大值
        float m_new = m;
        for (int i = 0; i < len; i++) {
            m_new = fmaxf(m_new, scores[i]);
        }
        
        // Step 2: 计算P = exp(S - m_new)并累加
        float l_new = 0.0f;
        for (int i = 0; i < len; i++) {
            P[i] = expf(scores[i] - m_new);
            l_new += P[i];
        }
        
        // Step 3: 合并旧的统计量
        float scale_old = expf(m - m_new);
        l_new = scale_old * l + l_new;
        
        // 更新
        m = m_new;
        l = l_new;
    }
    
    // 获取缩放因子（用于更新O）
    __device__ float get_scale() const {
        return safe_divide(1.0f, l);
    }
    
    __device__ float get_correction(float m_old) const {
        return expf(m_old - m);
    }
};

// ==================== 共享内存管理 ====================

// 共享内存布局
template<typename T>
struct SharedMemory {
    T* Q;      // [kBlockM, kHeadDim]
    T* K;      // [kBlockN, kHeadDim]
    T* V;      // [kBlockN, kHeadDim]
    float* S;  // [kBlockM, kBlockN] - attention scores
    float* P;  // [kBlockM, kBlockN] - attention weights (after softmax)
    
    __device__ SharedMemory(void* ptr) {
        char* base = reinterpret_cast<char*>(ptr);
        size_t offset = 0;
        
        Q = reinterpret_cast<T*>(base + offset);
        offset += kBlockM * kHeadDim * sizeof(T);
        
        K = reinterpret_cast<T*>(base + offset);
        offset += kBlockN * kHeadDim * sizeof(T);
        
        V = reinterpret_cast<T*>(base + offset);
        offset += kBlockN * kHeadDim * sizeof(T);
        
        S = reinterpret_cast<float*>(base + offset);
        offset += kBlockM * kBlockN * sizeof(float);
        
        P = reinterpret_cast<float*>(base + offset);
        offset += kBlockM * kBlockN * sizeof(float);
    }
    
    static constexpr size_t get_size() {
        size_t base_size = (kBlockM * kHeadDim + kBlockN * kHeadDim * 2) * sizeof(T) +
                          (kBlockM * kBlockN * 2) * sizeof(float);
        // 添加额外的统计量和累加器
        size_t extra_size = (kBlockM * 2) * sizeof(float) +  // m_shared, l_shared
                           (kBlockM * kHeadDim) * sizeof(float);  // O_accum
        return base_size + extra_size;
    }
};

// ==================== GEMM操作 ====================

// 简化的GEMM: C = A @ B^T
// A: [M, K], B: [N, K], C: [M, N]
template<typename T, int M, int N, int K>
__device__ void gemm_nt(const T* A, const T* B, float* C, int lda, int ldb, int ldc) {
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    // 每个线程计算C的一部分元素
    for (int idx = tid; idx < M * N; idx += num_threads) {
        int i = idx / N;  // row in C
        int j = idx % N;  // col in C
        
        float sum = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < K; k++) {
            float a_val = float(A[i * lda + k]);
            float b_val = float(B[j * ldb + k]);
            sum += a_val * b_val;
        }
        C[i * ldc + j] = sum;
    }
    __syncthreads();
}

// GEMM: C = A @ B
// A: [M, K], B: [K, N], C: [M, N]
template<typename T, int M, int N, int K>
__device__ void gemm_nn(const float* A, const T* B, T* C, int lda, int ldb, int ldc) {
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    for (int idx = tid; idx < M * N; idx += num_threads) {
        int i = idx / N;  // row
        int j = idx % N;  // col
        
        float sum = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < K; k++) {
            float a_val = A[i * lda + k];
            float b_val = float(B[k * ldb + j]);
            sum += a_val * b_val;
        }
        C[i * ldc + j] = T(sum);
    }
    __syncthreads();
}

// ==================== 核心Kernel ====================

/**
 * Flash Attention Kernel (简化版)
 * 
 * 参数：
 *   Q, K, V: [batch, num_heads, seq_len, head_dim]
 *   O: 输出 [batch, num_heads, seq_len, head_dim]
 *   softmax_scale: 缩放因子 (通常是 1/sqrt(head_dim))
 *   batch_size, num_heads, seq_len, head_dim: 形状参数
 * 
 * Grid: (num_blocks_M, batch * num_heads)
 * Block: kNThreads threads
 */
__global__ void flash_attention_kernel(
    const cutlass::half_t* __restrict__ Q,
    const cutlass::half_t* __restrict__ K,
    const cutlass::half_t* __restrict__ V,
    cutlass::half_t* __restrict__ O,
    float softmax_scale,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    // 断言检查
    assert(head_dim == kHeadDim);
    assert(blockDim.x == kNThreads);
    
    // Block和thread索引
    const int batch_head_idx = blockIdx.y;  // batch * num_heads
    const int q_block_idx = blockIdx.x;     // Q的block索引
    const int tid = threadIdx.x;
    
    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;
    
    // 计算Q block的起始位置
    const int q_start = q_block_idx * kBlockM;
    const int q_end = min(q_start + kBlockM, seq_len);
    const int q_size = q_end - q_start;
    
    if (q_size <= 0) return;  // 越界检查
    
    // 计算全局内存偏移
    const int64_t qkv_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const cutlass::half_t* Q_ptr = Q + qkv_offset;
    const cutlass::half_t* K_ptr = K + qkv_offset;
    const cutlass::half_t* V_ptr = V + qkv_offset;
    cutlass::half_t* O_ptr = O + qkv_offset;
    
    // 分配共享内存
    extern __shared__ char smem[];
    SharedMemory<cutlass::half_t> shared_mem(smem);
    
    // 从动态共享内存中分配额外的数组
    size_t base_offset = (kBlockM * kHeadDim + kBlockN * kHeadDim * 2) * sizeof(cutlass::half_t) +
                        (kBlockM * kBlockN * 2) * sizeof(float);
    float* m_shared = reinterpret_cast<float*>(smem + base_offset);
    float* l_shared = m_shared + kBlockM;
    float* O_accum = l_shared + kBlockM;
    
    // 加载Q block到共享内存
    for (int idx = tid; idx < q_size * head_dim; idx += blockDim.x) {
        int i = idx / head_dim;
        int j = idx % head_dim;
        shared_mem.Q[i * kHeadDim + j] = Q_ptr[(q_start + i) * head_dim + j];
    }
    __syncthreads();
    
    // 初始化
    for (int i = tid; i < kBlockM; i += blockDim.x) {
        m_shared[i] = -INFINITY;
        l_shared[i] = 0.0f;
    }
    for (int i = tid; i < kBlockM * kHeadDim; i += blockDim.x) {
        O_accum[i] = 0.0f;
    }
    __syncthreads();
    
    // 计算K,V的block数量
    const int num_k_blocks = (seq_len + kBlockN - 1) / kBlockN;
    
    // 遍历所有K,V blocks
    for (int k_block_idx = 0; k_block_idx < num_k_blocks; k_block_idx++) {
        const int k_start = k_block_idx * kBlockN;
        const int k_end = min(k_start + kBlockN, seq_len);
        const int k_size = k_end - k_start;
        
        // 加载K block到共享内存
        for (int idx = tid; idx < k_size * head_dim; idx += blockDim.x) {
            int i = idx / head_dim;
            int j = idx % head_dim;
            shared_mem.K[i * kHeadDim + j] = K_ptr[(k_start + i) * head_dim + j];
        }
        
        // 加载V block到共享内存
        for (int idx = tid; idx < k_size * head_dim; idx += blockDim.x) {
            int i = idx / head_dim;
            int j = idx % head_dim;
            shared_mem.V[i * kHeadDim + j] = V_ptr[(k_start + i) * head_dim + j];
        }
        __syncthreads();
        
        // 计算 S = Q @ K^T (attention scores)
        gemm_nt<cutlass::half_t, kBlockM, kBlockN, kHeadDim>(
            shared_mem.Q, shared_mem.K, shared_mem.S,
            kHeadDim, kHeadDim, kBlockN
        );
        
        // 应用softmax scale
        for (int idx = tid; idx < q_size * k_size; idx += blockDim.x) {
            shared_mem.S[idx] *= softmax_scale;
        }
        __syncthreads();
        
        // 对每一行进行在线softmax更新
        for (int i = 0; i < q_size; i++) {
            if (tid == 0) {
                float* scores = shared_mem.S + i * kBlockN;
                float* P = shared_mem.P + i * kBlockN;
                
                float m_old = m_shared[i];
                float l_old = l_shared[i];
                
                // 找到新的max
                float m_new = m_old;
                for (int j = 0; j < k_size; j++) {
                    m_new = fmaxf(m_new, scores[j]);
                }
                
                // 计算P = exp(S - m_new)
                float l_new = 0.0f;
                for (int j = 0; j < k_size; j++) {
                    P[j] = expf(scores[j] - m_new);
                    l_new += P[j];
                }
                
                // 合并旧的统计量
                float correction = expf(m_old - m_new);
                l_new = correction * l_old + l_new;
                
                // 更新累加器 O = correction * O + P @ V
                for (int d = 0; d < head_dim; d++) {
                    O_accum[i * kHeadDim + d] *= correction;
                }
                
                m_shared[i] = m_new;
                l_shared[i] = l_new;
            }
        }
        __syncthreads();
        
        // 计算 O += P @ V
        for (int i = 0; i < q_size; i++) {
            for (int d = tid; d < head_dim; d += blockDim.x) {
                float sum = 0.0f;
                const float* P_row = shared_mem.P + i * kBlockN;
                for (int j = 0; j < k_size; j++) {
                    sum += P_row[j] * float(shared_mem.V[j * kHeadDim + d]);
                }
                O_accum[i * kHeadDim + d] += sum;
            }
        }
        __syncthreads();
    }
    
    // 最终归一化并写回全局内存
    for (int i = 0; i < q_size; i++) {
        float scale = safe_divide(1.0f, l_shared[i]);
        for (int d = tid; d < head_dim; d += blockDim.x) {
            float val = O_accum[i * kHeadDim + d] * scale;
            O_ptr[(q_start + i) * head_dim + d] = cutlass::half_t(val);
        }
    }
}

// ==================== Host接口 ====================

/**
 * Flash Attention的Host接口
 * 
 * 输入输出都是CUDA device指针
 * 数据格式: [batch, num_heads, seq_len, head_dim]
 */
void flash_attention_forward(
    const cutlass::half_t* Q,
    const cutlass::half_t* K,
    const cutlass::half_t* V,
    cutlass::half_t* O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream = 0
) {
    // 检查参数
    assert(head_dim == kHeadDim && "head_dim must equal kHeadDim");
    assert(seq_len > 0 && "seq_len must be positive");
    
    // 计算softmax scale
    float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    // 计算grid和block大小
    const int num_q_blocks = (seq_len + kBlockM - 1) / kBlockM;
    dim3 grid(num_q_blocks, batch_size * num_heads);
    dim3 block(kNThreads);
    
    // 计算共享内存大小
    size_t smem_size = SharedMemory<cutlass::half_t>::get_size();
    
    // 检查并设置共享内存限制
    // A100默认48KB，我们需要更多
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // 设置最大共享内存配置
    cudaFuncSetAttribute(
        flash_attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );
    
    // 打印调试信息（首次调用）
    static bool first_call = true;
    if (first_call) {
        printf("Shared memory info:\n");
        printf("  Required: %zu bytes (%.1f KB)\n", smem_size, smem_size / 1024.0);
        printf("  Available per block: %zu bytes (%.1f KB)\n", 
               prop.sharedMemPerBlock, prop.sharedMemPerBlock / 1024.0);
        printf("  Max per block with opt-in: %zu bytes (%.1f KB)\n",
               prop.sharedMemPerBlockOptin, prop.sharedMemPerBlockOptin / 1024.0);
        
        if (smem_size > prop.sharedMemPerBlock) {
            printf("  ⚠️  WARNING: Required shared memory exceeds default limit!\n");
            printf("  Attempting to use opt-in limit...\n");
        }
        first_call = false;
    }
    
    // 启动kernel
    flash_attention_kernel<<<grid, block, smem_size, stream>>>(
        Q, K, V, O,
        softmax_scale,
        batch_size, num_heads, seq_len, head_dim
    );
    
    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Flash attention kernel launch failed: %s\n", 
                cudaGetErrorString(err));
        fprintf(stderr, "  Grid: (%d, %d), Block: (%d), Shared mem: %zu bytes\n",
                grid.x, grid.y, block.x, smem_size);
    }
}


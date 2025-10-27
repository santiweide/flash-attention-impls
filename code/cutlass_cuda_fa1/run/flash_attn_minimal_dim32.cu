/******************************************************************************
 * Flash Attention Implementation Optimized for head_dim=32
 * 
 * 相比 head_dim=64 的版本，这里做了以下优化：
 * 1. 更大的 tile size (128x128 vs 64x64)
 * 2. 更少的 shared memory 使用（因为 head_dim 减半）
 * 3. 更好的并行度和occupancy
 ******************************************************************************/

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/memory.h>
#include <cmath>
#include <algorithm>

// ==================== 配置参数 (为 head_dim=32 优化) ====================

// head_dim=32 时我们可以使用更大的 tile，因为 shared memory 需求减半
constexpr int kBlockM_32 = 128;     // Q的块大小 (2x larger than dim64)
constexpr int kBlockN_32 = 128;     // K,V的块大小 (2x larger than dim64)
constexpr int kHeadDim_32 = 32;     // Head维度 (固定为32)
constexpr int kNThreads_32 = 256;   // 更多线程来处理更大的tile

// ==================== 工具函数 ====================

__device__ __forceinline__ float safe_divide_32(float a, float b) {
    return b == 0.0f ? 0.0f : a / b;
}

// ==================== 共享内存管理 ====================

template<typename T>
struct SharedMemory_32 {
    T* Q;      // [kBlockM_32, kHeadDim_32]
    T* K;      // [kBlockN_32, kHeadDim_32]
    T* V;      // [kBlockN_32, kHeadDim_32]
    float* S;  // [kBlockM_32, kBlockN_32] - attention scores
    float* P;  // [kBlockM_32, kBlockN_32] - attention weights
    
    __device__ SharedMemory_32(void* ptr) {
        char* base = reinterpret_cast<char*>(ptr);
        size_t offset = 0;
        
        Q = reinterpret_cast<T*>(base + offset);
        offset += kBlockM_32 * kHeadDim_32 * sizeof(T);
        
        K = reinterpret_cast<T*>(base + offset);
        offset += kBlockN_32 * kHeadDim_32 * sizeof(T);
        
        V = reinterpret_cast<T*>(base + offset);
        offset += kBlockN_32 * kHeadDim_32 * sizeof(T);
        
        S = reinterpret_cast<float*>(base + offset);
        offset += kBlockM_32 * kBlockN_32 * sizeof(float);
        
        P = reinterpret_cast<float*>(base + offset);
        offset += kBlockM_32 * kBlockN_32 * sizeof(float);
    }
    
    static constexpr size_t get_size() {
        size_t base_size = (kBlockM_32 * kHeadDim_32 + kBlockN_32 * kHeadDim_32 * 2) * sizeof(T) +
                          (kBlockM_32 * kBlockN_32 * 2) * sizeof(float);
        size_t extra_size = (kBlockM_32 * 2) * sizeof(float) +        // m_shared, l_shared
                           (kBlockM_32 * kHeadDim_32) * sizeof(float); // O_accum
        return base_size + extra_size;
    }
};

// ==================== GEMM操作 (优化版) ====================

// 针对 head_dim=32 优化的 GEMM
template<typename T, int M, int N, int K>
__device__ void gemm_nt_32(const T* A, const T* B, float* C, int lda, int ldb, int ldc) {
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    // 每个线程处理多个元素以提高效率
    for (int idx = tid; idx < M * N; idx += num_threads) {
        int i = idx / N;
        int j = idx % N;
        
        float sum = 0.0f;
        // K=32, 完全展开可以提高性能
        #pragma unroll
        for (int k = 0; k < K; k++) {
            float a_val = float(A[i * lda + k]);
            float b_val = float(B[j * ldb + k]);
            sum += a_val * b_val;
        }
        C[i * ldc + j] = sum;
    }
    __syncthreads();
}

// ==================== 核心Kernel ====================

__global__ void flash_attention_kernel_dim32(
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
    assert(head_dim == kHeadDim_32);
    assert(blockDim.x == kNThreads_32);
    
    const int batch_head_idx = blockIdx.y;
    const int q_block_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;
    
    const int q_start = q_block_idx * kBlockM_32;
    const int q_end = min(q_start + kBlockM_32, seq_len);
    const int q_size = q_end - q_start;
    
    if (q_size <= 0) return;
    
    const int64_t qkv_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const cutlass::half_t* Q_ptr = Q + qkv_offset;
    const cutlass::half_t* K_ptr = K + qkv_offset;
    const cutlass::half_t* V_ptr = V + qkv_offset;
    cutlass::half_t* O_ptr = O + qkv_offset;
    
    extern __shared__ char smem[];
    SharedMemory_32<cutlass::half_t> shared_mem(smem);
    
    size_t base_offset = (kBlockM_32 * kHeadDim_32 + kBlockN_32 * kHeadDim_32 * 2) * sizeof(cutlass::half_t) +
                        (kBlockM_32 * kBlockN_32 * 2) * sizeof(float);
    float* m_shared = reinterpret_cast<float*>(smem + base_offset);
    float* l_shared = m_shared + kBlockM_32;
    float* O_accum = l_shared + kBlockM_32;
    
    // 加载Q tile
    for (int idx = tid; idx < q_size * head_dim; idx += blockDim.x) {
        int i = idx / head_dim;
        int j = idx % head_dim;
        shared_mem.Q[i * kHeadDim_32 + j] = Q_ptr[(q_start + i) * head_dim + j];
    }
    __syncthreads();
    
    // 初始化
    for (int i = tid; i < kBlockM_32; i += blockDim.x) {
        m_shared[i] = -INFINITY;
        l_shared[i] = 0.0f;
    }
    for (int i = tid; i < kBlockM_32 * kHeadDim_32; i += blockDim.x) {
        O_accum[i] = 0.0f;
    }
    __syncthreads();
    
    const int num_k_blocks = (seq_len + kBlockN_32 - 1) / kBlockN_32;
    
    for (int k_block_idx = 0; k_block_idx < num_k_blocks; k_block_idx++) {
        const int k_start = k_block_idx * kBlockN_32;
        const int k_end = min(k_start + kBlockN_32, seq_len);
        const int k_size = k_end - k_start;
        
        // 加载K和V tiles
        for (int idx = tid; idx < k_size * head_dim; idx += blockDim.x) {
            int i = idx / head_dim;
            int j = idx % head_dim;
            shared_mem.K[i * kHeadDim_32 + j] = K_ptr[(k_start + i) * head_dim + j];
            shared_mem.V[i * kHeadDim_32 + j] = V_ptr[(k_start + i) * head_dim + j];
        }
        __syncthreads();
        
        // S = Q @ K^T
        gemm_nt_32<cutlass::half_t, kBlockM_32, kBlockN_32, kHeadDim_32>(
            shared_mem.Q, shared_mem.K, shared_mem.S,
            kHeadDim_32, kHeadDim_32, kBlockN_32
        );
        
        // Apply softmax scale
        for (int idx = tid; idx < q_size * k_size; idx += blockDim.x) {
            shared_mem.S[idx] *= softmax_scale;
        }
        __syncthreads();
        
        // Online softmax更新
        for (int i = 0; i < q_size; i++) {
            if (tid == 0) {
                float* scores = shared_mem.S + i * kBlockN_32;
                float* P = shared_mem.P + i * kBlockN_32;
                
                float m_old = m_shared[i];
                float l_old = l_shared[i];
                
                float m_new = m_old;
                for (int j = 0; j < k_size; j++) {
                    m_new = fmaxf(m_new, scores[j]);
                }
                
                float l_new = 0.0f;
                for (int j = 0; j < k_size; j++) {
                    P[j] = expf(scores[j] - m_new);
                    l_new += P[j];
                }
                
                float correction = expf(m_old - m_new);
                l_new = correction * l_old + l_new;
                
                for (int d = 0; d < head_dim; d++) {
                    O_accum[i * kHeadDim_32 + d] *= correction;
                }
                
                m_shared[i] = m_new;
                l_shared[i] = l_new;
            }
        }
        __syncthreads();
        
        // O += P @ V
        for (int i = 0; i < q_size; i++) {
            for (int d = tid; d < head_dim; d += blockDim.x) {
                float sum = 0.0f;
                const float* P_row = shared_mem.P + i * kBlockN_32;
                #pragma unroll 8
                for (int j = 0; j < k_size; j++) {
                    sum += P_row[j] * float(shared_mem.V[j * kHeadDim_32 + d]);
                }
                O_accum[i * kHeadDim_32 + d] += sum;
            }
        }
        __syncthreads();
    }
    
    // 最终归一化
    for (int i = 0; i < q_size; i++) {
        float scale = safe_divide_32(1.0f, l_shared[i]);
        for (int d = tid; d < head_dim; d += blockDim.x) {
            float val = O_accum[i * kHeadDim_32 + d] * scale;
            O_ptr[(q_start + i) * head_dim + d] = cutlass::half_t(val);
        }
    }
}

// ==================== Host接口 ====================

void flash_attention_forward_dim32(
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
    assert(head_dim == kHeadDim_32 && "This kernel is optimized for head_dim=32");
    assert(seq_len > 0 && "seq_len must be positive");
    
    float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    const int num_q_blocks = (seq_len + kBlockM_32 - 1) / kBlockM_32;
    dim3 grid(num_q_blocks, batch_size * num_heads);
    dim3 block(kNThreads_32);
    
    size_t smem_size = SharedMemory_32<cutlass::half_t>::get_size();
    
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    cudaFuncSetAttribute(
        flash_attention_kernel_dim32,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );
    
    static bool first_call = true;
    if (first_call) {
        printf("Flash Attention (head_dim=32) Configuration:\n");
        printf("  Tile size: %dx%d (2x larger than dim64 version)\n", kBlockM_32, kBlockN_32);
        printf("  Threads per block: %d\n", kNThreads_32);
        printf("  Shared memory: %zu bytes (%.1f KB)\n", smem_size, smem_size / 1024.0);
        printf("  Max shared mem available: %zu bytes (%.1f KB)\n",
               prop.sharedMemPerBlockOptin, prop.sharedMemPerBlockOptin / 1024.0);
        first_call = false;
    }
    
    flash_attention_kernel_dim32<<<grid, block, smem_size, stream>>>(
        Q, K, V, O,
        softmax_scale,
        batch_size, num_heads, seq_len, head_dim
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Flash attention (dim32) kernel launch failed: %s\n", 
                cudaGetErrorString(err));
        fprintf(stderr, "  Grid: (%d, %d), Block: (%d), Shared mem: %zu bytes\n",
                grid.x, grid.y, block.x, smem_size);
    }
}


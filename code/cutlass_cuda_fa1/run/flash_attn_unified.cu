/******************************************************************************
 * Unified Flash Attention Implementation
 ******************************************************************************/

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/memory.h>
#include <cmath>
#include <algorithm>

// ==================== Tile Size Helper ====================

/**
 * compute_max_tile_size computes the maximum possible tile size at compile time
 * 
 * Constraints:
 * - A100 shared memory limit: 163 KB (opt-in)
 * - Need to store: Q[M,D], K[N,D], V[N,D], S[M,N], P[M,N], stats, accumulator
 * 
 * Strategy:
 * - smaller head_dim allows larger tiles
 * - ensure shared memory < 160 KB (leave some margin)
 */
template<int HEAD_DIM>
struct TileConfig {
    // compute_max_tile_size gets the maximum possible tile size
    static constexpr int compute_max_tile_size() {
        // Shared memory budget (bytes)
        constexpr size_t MAX_SMEM = 160 * 1024;
        
        // for a square tile [M, M]:
        // QKV: M * HEAD_DIM * 2 * 3 (half_t)
        // S,P: M * M * 4 * 2 (float)
        // Stats + Accum: M * 2 * 4 + M * HEAD_DIM * 4

        for (int tile = 128; tile >= 32; tile -= 8) {
            size_t qkv_size = tile * HEAD_DIM * sizeof(cutlass::half_t) * 3;
            size_t sp_size = tile * tile * sizeof(float) * 2;
            size_t extra = tile * 2 * sizeof(float) + tile * HEAD_DIM * sizeof(float);
            size_t total = qkv_size + sp_size + extra;
            
            if (total < MAX_SMEM) {
                return tile;
            }
        }
        return 32; // min tile size is 32
    }
    
    static constexpr int kTileM = compute_max_tile_size();
    static constexpr int kTileN = compute_max_tile_size();
    static constexpr int kHeadDim = HEAD_DIM;
    
    // max thread.x per block number is 1024 for A100. But here we use 256 as the max
    static constexpr int kThreads = (HEAD_DIM * 2 < 64) ? 64 : 
                                     (HEAD_DIM * 2 > 256) ? 256 : 
                                     (HEAD_DIM * 2);
    
    // print configuration information
    static void print_config() {
        printf("  Tile Config for head_dim=%d:\n", HEAD_DIM);
        printf("    Tile size: %dx%d\n", kTileM, kTileN);
        printf("    Threads: %d\n", kThreads);
        printf("    Shared memory: %.1f KB\n", get_smem_size() / 1024.0);
    }
    
    static constexpr size_t get_smem_size() {
        return (kTileM * kHeadDim + kTileN * kHeadDim * 2) * sizeof(cutlass::half_t) +
               (kTileM * kTileN * 2) * sizeof(float) +
               (kTileM * 2) * sizeof(float) +
               (kTileM * kHeadDim) * sizeof(float);
    }
};

// explicitly instantiate common configurations to verify
static_assert(TileConfig<32>::kTileM >= 64, "head_dim=32 should support at least 64x64 tile");
static_assert(TileConfig<64>::kTileM >= 64, "head_dim=64 should support at least 64x64 tile");
static_assert(TileConfig<128>::kTileM >= 32, "head_dim=128 should support at least 32x32 tile");

// ==================== Shared Memory Management (Template) ====================

template<typename T, int TILE_M, int TILE_N, int HEAD_DIM>
struct SharedMemory {
    T* Q;      // [TILE_M, HEAD_DIM] Q matrix
    T* K;      // [TILE_N, HEAD_DIM] K matrix
    T* V;      // [TILE_N, HEAD_DIM] V matrix
    float* S;  // [TILE_M, TILE_N] S matrix
    float* P;  // [TILE_M, TILE_N] P matrix
    
    __device__ SharedMemory(void* ptr) {
        char* base = reinterpret_cast<char*>(ptr);
        size_t offset = 0;
        
        Q = reinterpret_cast<T*>(base + offset);
        offset += TILE_M * HEAD_DIM * sizeof(T);
        
        K = reinterpret_cast<T*>(base + offset);
        offset += TILE_N * HEAD_DIM * sizeof(T);
        
        V = reinterpret_cast<T*>(base + offset);
        offset += TILE_N * HEAD_DIM * sizeof(T);
        
        S = reinterpret_cast<float*>(base + offset);
        offset += TILE_M * TILE_N * sizeof(float);
        
        P = reinterpret_cast<float*>(base + offset);
    }
};

// ==================== Parallel Softmax Utilities ====================

/**
 * Warp-level parallel max reduction
 */
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

/**
 * Block-level parallel max reduction with broadcast
 */
__device__ __forceinline__ float block_reduce_max(float val) {
    // Use first 32 elements of shared memory for warp maxes
    float* smem = (float*)__shared_memory_ptr();
    
    // Warp reduction
    float warp_max = warp_reduce_max(val);
    
    // Store warp results
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) {
        smem[warp_id] = warp_max;
    }
    __syncthreads();
    
    // Final warp reduction
    float result = (threadIdx.x < (blockDim.x + 31) / 32) ? smem[lane_id] : -INFINITY;
    result = warp_reduce_max(result);
    
    // Broadcast to all threads
    return __shfl_sync(0xffffffff, result, 0);
}

/**
 * Warp-level parallel sum reduction
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Block-level parallel sum reduction with broadcast
 */
__device__ __forceinline__ float block_reduce_sum(float val) {
    float* smem = (float*)__shared_memory_ptr();
    
    // Warp reduction
    float warp_sum = warp_reduce_sum(val);
    
    // Store warp results (offset to avoid overwriting max results)
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) {
        smem[32 + warp_id] = warp_sum;
    }
    __syncthreads();
    
    // Final warp reduction
    float result = (threadIdx.x < (blockDim.x + 31) / 32) ? smem[32 + lane_id] : 0.0f;
    result = warp_reduce_sum(result);
    
    // Broadcast to all threads
    return __shfl_sync(0xffffffff, result, 0);
}

// ==================== GEMM (Template) ====================

template<typename T, int M, int N, int K>
__device__ void gemm_nt_unified(const T* A, const T* B, float* C) {
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    for (int idx = tid; idx < M * N; idx += num_threads) {
        int i = idx / N;
        int j = idx % N;
        
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < K; k++) {
            sum += float(A[i * K + k]) * float(B[j * K + k]);
        }
        C[i * N + j] = sum;
    }
    __syncthreads();
}

// ==================== Flash Attention: Large Tile Kernel ====================
// Uses aggressive tile sizes (e.g., 120×120 for head_dim=32) to maximize
// data reuse at the cost of higher shared memory usage

template<int HEAD_DIM>
__global__ void flash_attn_large_tile_kernel(
    const cutlass::half_t* __restrict__ Q,
    const cutlass::half_t* __restrict__ K,
    const cutlass::half_t* __restrict__ V,
    cutlass::half_t* __restrict__ O,
    float softmax_scale,
    int batch_size,
    int num_heads,
    int seq_len
) {
    using Config = TileConfig<HEAD_DIM>;
    constexpr int kTileM = Config::kTileM;
    constexpr int kTileN = Config::kTileN;
    
    const int batch_head_idx = blockIdx.y;
    const int q_block_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;
    
    const int q_start = q_block_idx * kTileM;
    const int q_end = min(q_start + kTileM, seq_len);
    const int q_size = q_end - q_start;
    
    if (q_size <= 0) return;
    
    const int64_t qkv_offset = (batch_idx * num_heads + head_idx) * seq_len * HEAD_DIM;
    const cutlass::half_t* Q_ptr = Q + qkv_offset;
    const cutlass::half_t* K_ptr = K + qkv_offset;
    const cutlass::half_t* V_ptr = V + qkv_offset;
    cutlass::half_t* O_ptr = O + qkv_offset;
    
    extern __shared__ char smem[];
    SharedMemory<cutlass::half_t, kTileM, kTileN, HEAD_DIM> shared_mem(smem);
    
    // positions of stats and accumulator
    size_t base_offset = (kTileM * HEAD_DIM + kTileN * HEAD_DIM * 2) * sizeof(cutlass::half_t) +
                        (kTileM * kTileN * 2) * sizeof(float);
    float* m_shared = reinterpret_cast<float*>(smem + base_offset);
    float* l_shared = m_shared + kTileM;
    float* O_accum = l_shared + kTileM;
    
    // load Q
    for (int idx = tid; idx < q_size * HEAD_DIM; idx += blockDim.x) {
        int i = idx / HEAD_DIM;
        int j = idx % HEAD_DIM;
        shared_mem.Q[i * HEAD_DIM + j] = Q_ptr[(q_start + i) * HEAD_DIM + j];
    }
    __syncthreads();
    
    // initialize
    for (int i = tid; i < kTileM; i += blockDim.x) {
        m_shared[i] = -INFINITY;
        l_shared[i] = 0.0f;
    }
    for (int i = tid; i < kTileM * HEAD_DIM; i += blockDim.x) {
        O_accum[i] = 0.0f;
    }
    __syncthreads();
    
    // iterate over K/V tiles
    const int num_k_blocks = (seq_len + kTileN - 1) / kTileN;
    
    for (int k_block_idx = 0; k_block_idx < num_k_blocks; k_block_idx++) {
        const int k_start = k_block_idx * kTileN;
        const int k_end = min(k_start + kTileN, seq_len);
        const int k_size = k_end - k_start;
        
        // load K and V
        for (int idx = tid; idx < k_size * HEAD_DIM; idx += blockDim.x) {
            int i = idx / HEAD_DIM;
            int j = idx % HEAD_DIM;
            shared_mem.K[i * HEAD_DIM + j] = K_ptr[(k_start + i) * HEAD_DIM + j];
            shared_mem.V[i * HEAD_DIM + j] = V_ptr[(k_start + i) * HEAD_DIM + j];
        }
        __syncthreads();
        
        // S = Q @ K^T
        gemm_nt_unified<cutlass::half_t, kTileM, kTileN, HEAD_DIM>(
            shared_mem.Q, shared_mem.K, shared_mem.S
        );
        
        // Apply softmax scale (use 2D indexing!)
        for (int idx = tid; idx < q_size * k_size; idx += blockDim.x) {
            int i = idx / k_size;
            int j = idx % k_size;
            shared_mem.S[i * kTileN + j] *= softmax_scale;
        }
        __syncthreads();
        
        // Online softmax - PARALLEL VERSION
        for (int i = 0; i < q_size; i++) {
            const int tid = threadIdx.x;
            
            // Step 1: Find max over all scores (all threads cooperate)
            float local_max = -INFINITY;
            for (int idx = tid; idx < k_size; idx += blockDim.x) {
                local_max = fmaxf(local_max, shared_mem.S[i * kTileN + idx]);
            }
            float m_new_block = block_reduce_max(local_max);
            
            // Update m (broadcasted to all threads)
            float m_old = m_shared[i];
            float m_new = fmaxf(m_old, m_new_block);
            m_shared[i] = m_new;
            __syncthreads();
            
            // Step 2: Compute exp and sum (all threads cooperate)
            float local_sum = 0.0f;
            for (int idx = tid; idx < k_size; idx += blockDim.x) {
                float p = expf(shared_mem.S[i * kTileN + idx] - m_new);
                shared_mem.S[i * kTileN + idx] = p;  // Reuse S for P
                local_sum += p;
            }
            float l_new_block = block_reduce_sum(local_sum);
            
            // Update l with correction (all threads participate)
            float l_old = l_shared[i];
            float correction = expf(m_old - m_new);
            float l_new = correction * l_old + l_new_block;
            l_shared[i] = l_new;
            
            // Step 3: Apply correction to O_accum (all threads cooperate)
            for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
                O_accum[i * HEAD_DIM + d] *= correction;
            }
            __syncthreads();
        }
        
        // O += P @ V
        for (int i = 0; i < q_size; i++) {
            for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
                float sum = 0.0f;
                for (int j = 0; j < k_size; j++) {
                    sum += shared_mem.S[i * kTileN + j] * float(shared_mem.V[j * HEAD_DIM + d]);  // Use S instead of P
                }
                O_accum[i * HEAD_DIM + d] += sum;
            }
        }
        __syncthreads();
    }
    
    // final normalization
    for (int i = 0; i < q_size; i++) {
        float scale = (l_shared[i] == 0.0f) ? 0.0f : 1.0f / l_shared[i];
        for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
            float val = O_accum[i * HEAD_DIM + d] * scale;
            O_ptr[(q_start + i) * HEAD_DIM + d] = cutlass::half_t(val);
        }
    }
}

// ==================== Host Interface: Large Tile ====================

template<int HEAD_DIM>
void flash_attn_large_tile_forward(
    const cutlass::half_t* Q,
    const cutlass::half_t* K,
    const cutlass::half_t* V,
    cutlass::half_t* O,
    int batch_size,
    int num_heads,
    int seq_len,
    cudaStream_t stream = 0
) {
    using Config = TileConfig<HEAD_DIM>;
    
    float softmax_scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));
    
    const int num_q_blocks = (seq_len + Config::kTileM - 1) / Config::kTileM;
    dim3 grid(num_q_blocks, batch_size * num_heads);
    dim3 block(Config::kThreads);
    
    size_t smem_size = Config::get_smem_size();
    
    // set shared memory limit
    cudaFuncSetAttribute(
        flash_attn_large_tile_kernel<HEAD_DIM>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );
    
    // print configuration (first call)
    static bool first_call = true;
    if (first_call) {
        printf("\n");
        printf("================================================================================\n");
        printf("Flash Attention - Large Tile Configuration (head_dim=%d)\n", HEAD_DIM);
        printf("================================================================================\n");
        Config::print_config();
        printf("================================================================================\n");
        first_call = false;
    }
    
    flash_attn_large_tile_kernel<HEAD_DIM><<<grid, block, smem_size, stream>>>(
        Q, K, V, O,
        softmax_scale,
        batch_size, num_heads, seq_len
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Flash attention kernel launch failed: %s\n", 
                cudaGetErrorString(err));
    }
}

// ==================== Flash Attention: Small Tile Configuration ====================
// Uses conservative tile sizes (e.g., 45×90 for head_dim=32) to reduce
// shared memory pressure and improve occupancy

template<int HEAD_DIM>
struct SmallTileConfig {
    static constexpr int compute_small_tile_size() {
        // Small tile uses ~75% of large tile size, with asymmetric M/N
        constexpr int large_tile = TileConfig<HEAD_DIM>::kTileM;
        return (large_tile * 3) / 4;  // 75% of large tile
    }
    
    static constexpr int kTileM = compute_small_tile_size() / 2;  // smaller M direction
    static constexpr int kTileN = compute_small_tile_size();       // keep N direction
    static constexpr int kHeadDim = HEAD_DIM;
    static constexpr int kThreads = 256;
    
    static constexpr size_t get_smem_size() {
        return (kTileM * kHeadDim + kTileN * kHeadDim * 2) * sizeof(cutlass::half_t) +
               (kTileM * kTileN * 2) * sizeof(float) +  // Need space for BOTH S and P!
               (kTileM * 2) * sizeof(float) +
               (kTileM * kHeadDim) * sizeof(float);
    }
};

// ==================== Flash Attention: Small Tile Kernel ====================
// Same algorithm as large tile, but uses smaller tiles for better occupancy

template<int HEAD_DIM>
__global__ void flash_attn_small_tile_kernel(
    const cutlass::half_t* __restrict__ Q,
    const cutlass::half_t* __restrict__ K,
    const cutlass::half_t* __restrict__ V,
    cutlass::half_t* __restrict__ O,
    float softmax_scale,
    int batch_size,
    int num_heads,
    int seq_len
) {
    using Config = SmallTileConfig<HEAD_DIM>;
    constexpr int kTileM = Config::kTileM;
    constexpr int kTileN = Config::kTileN;
    
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_block_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    const int q_start = q_block_idx * kTileM;
    const int q_end = min(q_start + kTileM, seq_len);
    const int q_size = q_end - q_start;
    
    if (q_size <= 0) return;
    
    const int64_t offset = (batch_idx * num_heads + head_idx) * seq_len * HEAD_DIM;
    const cutlass::half_t* Q_ptr = Q + offset;
    const cutlass::half_t* K_ptr = K + offset;
    const cutlass::half_t* V_ptr = V + offset;
    cutlass::half_t* O_ptr = O + offset;
    
    extern __shared__ char smem[];
    SharedMemory<cutlass::half_t, kTileM, kTileN, HEAD_DIM> shared_mem(smem);
    
    size_t stats_offset = (kTileM * HEAD_DIM + kTileN * HEAD_DIM * 2) * sizeof(cutlass::half_t) +
                          (kTileM * kTileN * 2) * sizeof(float);
    float* m_shared = reinterpret_cast<float*>(smem + stats_offset);
    float* l_shared = m_shared + kTileM;
    float* O_accum = l_shared + kTileM;
    
    // load Q
    for (int idx = tid; idx < q_size * HEAD_DIM; idx += blockDim.x) {
        int i = idx / HEAD_DIM;
        int j = idx % HEAD_DIM;
        shared_mem.Q[i * HEAD_DIM + j] = Q_ptr[(q_start + i) * HEAD_DIM + j];
    }
    
    // initialize
    for (int i = tid; i < kTileM; i += blockDim.x) {
        m_shared[i] = -INFINITY;
        l_shared[i] = 0.0f;
    }
    for (int i = tid; i < kTileM * HEAD_DIM; i += blockDim.x) {
        O_accum[i] = 0.0f;
    }
    __syncthreads();
    
    // iterate over K/V tiles
    const int num_kv_tiles = (seq_len + kTileN - 1) / kTileN;
    
    for (int kv_tile_idx = 0; kv_tile_idx < num_kv_tiles; kv_tile_idx++) {
        const int k_start = kv_tile_idx * kTileN;
        const int k_end = min(k_start + kTileN, seq_len);
        const int k_size = k_end - k_start;
        
        for (int idx = tid; idx < k_size * HEAD_DIM; idx += blockDim.x) {
            int i = idx / HEAD_DIM;
            int j = idx % HEAD_DIM;
            shared_mem.K[i * HEAD_DIM + j] = K_ptr[(k_start + i) * HEAD_DIM + j];
            shared_mem.V[i * HEAD_DIM + j] = V_ptr[(k_start + i) * HEAD_DIM + j];
        }
        __syncthreads();
        
        gemm_nt_unified<cutlass::half_t, kTileM, kTileN, HEAD_DIM>(
            shared_mem.Q, shared_mem.K, shared_mem.S
        );
        
        for (int idx = tid; idx < q_size * k_size; idx += blockDim.x) {
            int i = idx / k_size;
            int j = idx % k_size;
            shared_mem.S[i * kTileN + j] *= softmax_scale;
        }
        __syncthreads();
        
        // Online softmax
        for (int i = 0; i < q_size; i++) {
            if (tid == 0) {
                float m_old = m_shared[i];
                float l_old = l_shared[i];
                
                float m_new = m_old;
                for (int j = 0; j < k_size; j++) {
                    m_new = fmaxf(m_new, shared_mem.S[i * kTileN + j]);
                }
                
                float l_new = 0.0f;
                for (int j = 0; j < k_size; j++) {
                    float p = expf(shared_mem.S[i * kTileN + j] - m_new);
                    shared_mem.S[i * kTileN + j] = p;
                    l_new += p;
                }
                
                float correction = expf(m_old - m_new);
                l_new = correction * l_old + l_new;
                
                for (int d = 0; d < HEAD_DIM; d++) {
                    O_accum[i * HEAD_DIM + d] *= correction;
                }
                
                m_shared[i] = m_new;
                l_shared[i] = l_new;
            }
        }
        __syncthreads();
        
        for (int i = 0; i < q_size; i++) {
            for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
                float sum = 0.0f;
                for (int j = 0; j < k_size; j++) {
                    sum += shared_mem.S[i * kTileN + j] * float(shared_mem.V[j * HEAD_DIM + d]);
                }
                O_accum[i * HEAD_DIM + d] += sum;
            }
        }
        __syncthreads();
    }
    
    for (int i = 0; i < q_size; i++) {
        float scale = (l_shared[i] == 0.0f) ? 0.0f : 1.0f / l_shared[i];
        for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
            float val = O_accum[i * HEAD_DIM + d] * scale;
            O_ptr[(q_start + i) * HEAD_DIM + d] = cutlass::half_t(val);
        }
    }
}

template<int HEAD_DIM>
void flash_attn_small_tile_forward(
    const cutlass::half_t* Q,
    const cutlass::half_t* K,
    const cutlass::half_t* V,
    cutlass::half_t* O,
    int batch_size,
    int num_heads,
    int seq_len,
    cudaStream_t stream
) {
    using Config = SmallTileConfig<HEAD_DIM>;
    
    float softmax_scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));
    
    const int num_q_blocks = (seq_len + Config::kTileM - 1) / Config::kTileM;
    dim3 grid(num_q_blocks, num_heads, batch_size);
    dim3 block(Config::kThreads);
    
    size_t smem_size = Config::get_smem_size();
    
    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(
            flash_attn_small_tile_kernel<HEAD_DIM>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
    }
    
    flash_attn_small_tile_kernel<HEAD_DIM><<<grid, block, smem_size, stream>>>(
        Q, K, V, O,
        softmax_scale,
        batch_size, num_heads, seq_len
    );
}

// ==================== Dispatch Functions ====================
// Public API: Automatically selects appropriate kernel based on head_dim

void flash_attention_forward_dispatch(
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
    // Runtime dispatch to large tile implementation
    switch (head_dim) {
        case 32:
            flash_attn_large_tile_forward<32>(Q, K, V, O, batch_size, num_heads, seq_len, stream);
            break;
        case 64:
            flash_attn_large_tile_forward<64>(Q, K, V, O, batch_size, num_heads, seq_len, stream);
            break;
        case 128:
            flash_attn_large_tile_forward<128>(Q, K, V, O, batch_size, num_heads, seq_len, stream);
            break;
        default:
            fprintf(stderr, "Unsupported head_dim=%d (supported: 32, 64, 128)\n", head_dim);
            break;
    }
}

void flash_attention_small_tile_dispatch(
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
    // Runtime dispatch to small tile implementation
    switch (head_dim) {
        case 32:
            flash_attn_small_tile_forward<32>(Q, K, V, O, batch_size, num_heads, seq_len, stream);
            break;
        case 64:
            flash_attn_small_tile_forward<64>(Q, K, V, O, batch_size, num_heads, seq_len, stream);
            break;
        case 128:
            flash_attn_small_tile_forward<128>(Q, K, V, O, batch_size, num_heads, seq_len, stream);
            break;
        default:
            fprintf(stderr, "Unsupported head_dim=%d (supported: 32, 64, 128)\n", head_dim);
            break;
    }
}

// ==================== Backward Compatibility Aliases ====================
// Keep old function name for existing code that uses it

void attention_reference_dispatch(
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
    // Redirect to the small tile implementation
    flash_attention_small_tile_dispatch(Q, K, V, O, batch_size, num_heads, seq_len, head_dim, stream);
}


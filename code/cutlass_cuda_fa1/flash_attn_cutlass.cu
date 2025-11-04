/******************************************************************************
 * Flash Attention with CUTLASS GEMM (Tensor Core Version)
 * 
 * Uses CUTLASS tensor cores for Q@K^T and P@V matrix multiplications
 * with PARALLEL SOFTMAX using Warp Shuffle operations
 ******************************************************************************/

 #include <cuda_runtime.h>
 #include <cuda_fp16.h>
 #include <mma.h>  // CUDA WMMA API for tensor cores
 #include <cutlass/cutlass.h>
 #include <cutlass/numeric_types.h>
 #include <cutlass/gemm/device/gemm.h>
 #include <cutlass/gemm/warp/mma_tensor_op.h>
 #include <cutlass/arch/mma.h>
 #include <cmath>
 #include <algorithm>
 
 using namespace nvcuda;

 
 // ==================== Small Tile Configuration (Same as Small Tile) ====================
 template<int HEAD_DIM>
 struct CutlassSmallTileConfig {
     // Same tile calculation as SmallTileConfig
     static constexpr int compute_small_tile_size() {
         // Use conservative sizes
         if (HEAD_DIM == 32) return 90;
         if (HEAD_DIM == 64) return 72;
         if (HEAD_DIM == 128) return 48;
         return 32;
     }
     
     static constexpr int kTileM = compute_small_tile_size() / 2;  // M方向更小
     static constexpr int kTileN = compute_small_tile_size();       // N方向保持
     static constexpr int kHeadDim = HEAD_DIM;
     static constexpr int kThreads = 256;
     
     static constexpr size_t get_smem_size() {
         return (kTileM * kHeadDim + kTileN * kHeadDim * 2) * sizeof(cutlass::half_t) +
                (kTileM * kTileN * 2) * sizeof(float) +  // S and P
                (kTileM * 2) * sizeof(float) +            // m, l
                (kTileM * kHeadDim) * sizeof(float);      // O_accum
     }
 };
 
 // ==================== Shared Memory Layout ====================
 
 template<typename T, int TILE_M, int TILE_N, int HEAD_DIM>
 struct SharedMemoryCutlass {
     T* Q;      // [TILE_M, HEAD_DIM]
     T* K;      // [TILE_N, HEAD_DIM]
     T* V;      // [TILE_N, HEAD_DIM]
     float* S;  // [TILE_M, TILE_N]
     float* P;  // [TILE_M, TILE_N]
     
     __device__ SharedMemoryCutlass(void* ptr) {
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
 
 // ==================== WMMA Tensor Core GEMM ====================
 
 // Wrapper for Q @ K^T using WMMA Tensor Cores
 // A100 supports m16n8k16 for FP16 inputs with FP32 accumulation
template<int TILE_M, int TILE_N, int DIM_K>
__device__ __forceinline__ void cutlass_gemm_qk(
    const cutlass::half_t* Q,  // [TILE_M, DIM_K]
    const cutlass::half_t* K,  // [TILE_N, DIM_K]
    float* S,                   // [TILE_M, TILE_N]
    int q_size,                 // Valid rows (≤ TILE_M)
    int k_size                  // Valid cols (≤ TILE_N)
) {
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    
    const int warpId = threadIdx.x / 32;
    const int numWarps = blockDim.x / 32;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    for (int idx = tid; idx < q_size * k_size; idx += num_threads) {
        S[idx] = 0.0f;
    }
    __syncthreads(); 

    if (q_size >= WMMA_M && k_size >= WMMA_N && DIM_K >= WMMA_K) {
        const int k_main_loop_end = (DIM_K / WMMA_K) * WMMA_K;

        for (int m = warpId * WMMA_M; m < (q_size / WMMA_M) * WMMA_M; m += numWarps * WMMA_M) {
            for (int n = 0; n < (k_size / WMMA_N) * WMMA_N; n += WMMA_N) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag; 
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
                
                wmma::fill_fragment(c_frag, 0.0f);
                
                for (int k = 0; k < k_main_loop_end; k += WMMA_K) {
                    wmma::load_matrix_sync(a_frag, reinterpret_cast<const half*>(Q + m * DIM_K + k), DIM_K);
                    wmma::load_matrix_sync(b_frag, reinterpret_cast<const half*>(K + n * DIM_K + k), DIM_K);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                
                wmma::store_matrix_sync(S + m * TILE_N + n, c_frag, TILE_N, wmma::mem_row_major);
            }
        }
        __syncthreads();
    }
    
    // Remainder
    // 1. K remainder (k >= k_main_loop_end)
    // 2. M & N dim remainder
    const int k_main_loop_end = (DIM_K / WMMA_K) * WMMA_K;

    for (int idx = tid; idx < q_size * k_size; idx += num_threads) {
        int i = idx / k_size; // row (query)
        int j = idx % k_size; // col (key)
        
        bool computed_by_wmma = false;
        if (q_size >= WMMA_M && k_size >= WMMA_N && DIM_K >= WMMA_K) {
             if (i < (q_size / WMMA_M) * WMMA_M && j < (k_size / WMMA_N) * WMMA_N) {
                 computed_by_wmma = true;
             }
        }

        if (computed_by_wmma) { // 1. k dim remainder
            float sum_remainder = 0.0f;
            for (int k = k_main_loop_end; k < DIM_K; k++) {
                sum_remainder += float(Q[i * DIM_K + k]) * float(K[j * DIM_K + k]);
            }
            S[i * TILE_N + j] += sum_remainder;

        } else { // 2. M/N dim remainder
            float sum_full = 0.0f;
            for (int k = 0; k < DIM_K; k++) {
                sum_full += float(Q[i * DIM_K + k]) * float(K[j * DIM_K + k]);
            }
            S[i * TILE_N + j] = sum_full;
        }
    }
    __syncthreads();
}
 // Wrapper for P @ V using WMMA Tensor Cores
 // P is float, V is half_t - need to convert P to half for WMMA
// ==================== WMMA Tensor Core GEMM (FIXED) ====================
// ( ... cutlass_gemm_qk 函数 ... )

// Wrapper for P @ V using WMMA Tensor Cores
template<int TILE_M, int TILE_N, int DIM_K> // DIM_K == HEAD_DIM
__device__ __forceinline__ void cutlass_gemm_pv(
    const float* P,               // [TILE_M, TILE_N] (FP32)
    const cutlass::half_t* V,    // [TILE_N, DIM_K] (FP16)
    float* O,                     // [TILE_M, DIM_K] (FP32 accum)
    int q_size,                   // Valid M
    int k_size,                   // Valid K (P@V 的 K 维度是 TILE_N)
    int head_dim_size = DIM_K     // Valid N
) {
    // WMMA 瓦片尺寸 (A100)
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    
    const int warpId = threadIdx.x / 32;
    const int numWarps = blockDim.x / 32;
    const int laneId = threadIdx.x % 32;

    // GEMM (M, K, N) = (q_size, k_size, head_dim_size)
    // P: [M, K] (row) - A
    // V: [K, N] (row) - B
    // O: [M, N] (row) - C

    // A(row) @ B(col) -> C(row)
    // V row-major, but load with column major
    
    // M (q_size)
    for (int m_base = warpId * WMMA_M; m_base < q_size; m_base += numWarps * WMMA_M) {
        // N (head_dim_size)
        for (int n_base = 0; n_base < head_dim_size; n_base += WMMA_N) {
            
            // --- 1. Load C (O_accum) ---
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
            
            if (m_base < q_size && n_base < head_dim_size) {
                wmma::load_matrix_sync(c_frag, O + m_base * DIM_K + n_base, DIM_K, wmma::mem_row_major);
            } else { // case tile over flow 
                wmma::fill_fragment(c_frag, 0.0f);
            }
            for (int k_base = 0; k_base < k_size; k_base += WMMA_K) {
                
                // P: [M, K] (row)
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
                // V: [K, N] (col)
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

                // load P (FP32) -> a_frag (FP16)
                const float* p_tile_base = P + m_base * TILE_N + k_base;
                int frag_row = laneId % 16;
                int frag_col_offset = (laneId < 16) ? 0 : 8;

                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int current_m = m_base + frag_row;
                    int current_k = k_base + frag_col_offset + i;
                    
                    if (current_m < q_size && current_k < k_size) {
                        a_frag.x[i] = __float2half_rn(p_tile_base[frag_row * TILE_N + frag_col_offset + i]);
                    } else {
                        a_frag.x[i] = __float2half_rn(0.0f); // 边界外用 0 填充
                    }
                }

                // load V (FP16) -> b_frag (col_major) 
                // store V as [TILE_N, DIM_K] (row_major)
                // laod V[k_base:k_base+16, n_base:n_base+16]
                // use col_major load ->  V[k_base, n_base]
                const half* v_ptr = reinterpret_cast<const half*>(V + k_base * DIM_K + n_base);
                

                if (k_base < k_size && n_base < head_dim_size) {
                    wmma::load_matrix_sync(b_frag, v_ptr, DIM_K, wmma::mem_col_major);
                } else {
                    wmma::fill_fragment(b_frag, __float2half_rn(0.0f));
                }

                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            
            if (m_base < q_size && n_base < head_dim_size) {
                wmma::store_matrix_sync(O + m_base * DIM_K + n_base, c_frag, DIM_K, wmma::mem_row_major);
            }
        }
    }
    __syncthreads();
}
 
 // ==================== Parallel Softmax with Warp Shuffle ====================

// Parallel softmax computation using warp shuffle operations
// All threads in each warp cooperate to compute softmax for query positions
// assigned to that warp
template<int TILE_M, int TILE_N, int DIM_K>
__device__ __forceinline__ void parallel_softmax_warp(
    float* S,               // [TILE_M, TILE_N] - attention scores
    float* P,               // [TILE_M, TILE_N] - attention probabilities
    float* m_shared,        // [TILE_M] - max values
    float* l_shared,        // [TILE_M] - normalizing factors
    float* O_accum,         // [TILE_M, DIM_K] - output accumulator
    int q_size,             // Number of valid query positions
    int k_size,             // Number of valid key positions
    int num_threads,
    float softmax_scale
) {
    const int tid = threadIdx.x;
    const int laneId = tid % 32;
    const int warpId = tid / 32;
    const int numWarps = (num_threads + 31) / 32;
    
    // Each warp processes one or more query positions
    // With 256 threads (8 warps) and up to 45 queries, some warps handle multiple queries
    for (int i = warpId; i < q_size; i += numWarps) {
        float m_old = m_shared[i];
        float l_old = l_shared[i];
        
        // ========== Step 1: Find max in parallel ==========
        // Each lane finds the max of its assigned columns
        float m_new = -INFINITY;
        for (int j = laneId; j < k_size; j += 32) {
            m_new = fmaxf(m_new, S[i * TILE_N + j] * softmax_scale);
        }
        
        // Reduce within warp using butterfly shuffle pattern
        // This is a standard warp reduce for max
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            m_new = fmaxf(m_new, __shfl_down_sync(0xffffffff, m_new, offset));
        }
        
        // Broadcast result from lane 0 to all lanes in the warp
        m_new = __shfl_sync(0xffffffff, m_new, 0);
        
        // ========== Step 2: Compute softmax exponentials and sum ==========
        // Each lane computes exponentials for its assigned columns and accumulates sum
        float l_new = 0.0f;
        for (int j = laneId; j < k_size; j += 32) {
            float p = expf((S[i * TILE_N + j] * softmax_scale) - m_new);
            P[i * TILE_N + j] = p;
            l_new += p;
        }
        
        // Reduce sum within warp using butterfly shuffle pattern
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            l_new += __shfl_down_sync(0xffffffff, l_new, offset);
        }
        
        // Broadcast result from lane 0 to all lanes in the warp
        l_new = __shfl_sync(0xffffffff, l_new, 0);
        
        // ========== Step 3: Apply online softmax correction ==========
        // Correction factor for updating the output accumulator
        float correction = expf(m_old - m_new);
        l_new = correction * l_old + l_new;
        
        // Update statistics (only lane 0 writes to avoid conflicts)
        if (laneId == 0) {
            m_shared[i] = m_new;
            l_shared[i] = l_new;
        }
        
        // Update O_accum in parallel: scale by correction factor
        // Each lane handles part of the dimension
        for (int d = laneId; d < DIM_K; d += 32) {
            O_accum[i * DIM_K + d] *= correction;
        }
    }
    __syncthreads();
}

// ==================== Flash Attention Kernel with CUTLASS ====================

template<int HEAD_DIM>
__global__ void flash_attn_cutlass_kernel(
     const cutlass::half_t* __restrict__ Q,
     const cutlass::half_t* __restrict__ K,
     const cutlass::half_t* __restrict__ V,
     cutlass::half_t* __restrict__ O,
     float softmax_scale,
     int batch_size,
     int num_heads,
     int seq_len
 ) {
     using Config = CutlassSmallTileConfig<HEAD_DIM>;
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
     SharedMemoryCutlass<cutlass::half_t, kTileM, kTileN, HEAD_DIM> shared_mem(smem);
     
     size_t stats_offset = (kTileM * HEAD_DIM + kTileN * HEAD_DIM * 2) * sizeof(cutlass::half_t) +
                           (kTileM * kTileN * 2) * sizeof(float);
     float* m_shared = reinterpret_cast<float*>(smem + stats_offset);
     float* l_shared = m_shared + kTileM;
     float* O_accum = l_shared + kTileM;
     
     // Load Q tile
     for (int idx = tid; idx < q_size * HEAD_DIM; idx += blockDim.x) {
         int i = idx / HEAD_DIM;
         int j = idx % HEAD_DIM;
         shared_mem.Q[i * HEAD_DIM + j] = Q_ptr[(q_start + i) * HEAD_DIM + j];
     }
     
     // Initialize statistics
     for (int i = tid; i < kTileM; i += blockDim.x) {
         m_shared[i] = -INFINITY;
         l_shared[i] = 0.0f;
     }
     for (int i = tid; i < kTileM * HEAD_DIM; i += blockDim.x) {
         O_accum[i] = 0.0f;
     }
     __syncthreads();
     
     // Iterate over K/V tiles
     const int num_kv_tiles = (seq_len + kTileN - 1) / kTileN;
     
     for (int kv_tile_idx = 0; kv_tile_idx < num_kv_tiles; kv_tile_idx++) {
         const int k_start = kv_tile_idx * kTileN;
         const int k_end = min(k_start + kTileN, seq_len);
         const int k_size = k_end - k_start;
         
         // Load K and V tiles
         for (int idx = tid; idx < k_size * HEAD_DIM; idx += blockDim.x) {
             int i = idx / HEAD_DIM;
             int j = idx % HEAD_DIM;
             shared_mem.K[i * HEAD_DIM + j] = K_ptr[(k_start + i) * HEAD_DIM + j];
             shared_mem.V[i * HEAD_DIM + j] = V_ptr[(k_start + i) * HEAD_DIM + j];
         }
         __syncthreads();
         
         // S = Q @ K^T using CUTLASS
         cutlass_gemm_qk<kTileM, kTileN, HEAD_DIM>(
             shared_mem.Q, shared_mem.K, shared_mem.S, q_size, k_size
         );

        // Online softmax
        parallel_softmax_warp<kTileM, kTileN, HEAD_DIM>(
             shared_mem.S, shared_mem.P, m_shared, l_shared, O_accum, 
             q_size, k_size, blockDim.x,
             softmax_scale // <-- 传入新的参数
         );
         
         // O += P @ V
         cutlass_gemm_pv<kTileM, kTileN, HEAD_DIM>(
             shared_mem.P, shared_mem.V, O_accum, q_size, k_size
         );
     }
     
     for (int i = 0; i < q_size; i++) {
         float scale = (l_shared[i] == 0.0f) ? 0.0f : 1.0f / l_shared[i];
         for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
             float val = O_accum[i * HEAD_DIM + d] * scale;
             O_ptr[(q_start + i) * HEAD_DIM + d] = cutlass::half_t(val);
         }
     }
 }
 
 // ==================== Host Interface ====================
 
 template<int HEAD_DIM>
 void flash_attn_cutlass_forward(
     const cutlass::half_t* Q,
     const cutlass::half_t* K,
     const cutlass::half_t* V,
     cutlass::half_t* O,
     int batch_size,
     int num_heads,
     int seq_len,
     cudaStream_t stream
 ) {
     using Config = CutlassSmallTileConfig<HEAD_DIM>;
     
     float softmax_scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));
     
     const int num_q_blocks = (seq_len + Config::kTileM - 1) / Config::kTileM;
     dim3 grid(num_q_blocks, num_heads, batch_size);
     dim3 block(Config::kThreads);
     
     size_t smem_size = Config::get_smem_size();
     
     if (smem_size > 48 * 1024) {
         cudaFuncSetAttribute(
             flash_attn_cutlass_kernel<HEAD_DIM>,
             cudaFuncAttributeMaxDynamicSharedMemorySize,
             smem_size
         );
     }
     
     // Print config on first call
     static bool first_call = true;
     if (first_call) {
         printf("\n");
         printf("================================================================================\n");
         printf("Flash Attention - WMMA Tensor Core + Parallel Softmax (head_dim=%d)\n", HEAD_DIM);
         printf("================================================================================\n");
         printf("  Tile size: %dx%d (same as Small Tile)\n", Config::kTileM, Config::kTileN);
         printf("  Threads: %d (%d warps)\n", Config::kThreads, Config::kThreads / 32);
         printf("  Shared memory: %.1f KB\n", smem_size / 1024.0);
         printf("  Tensor Cores: ENABLED via WMMA API\n");
         printf("    → Q@K^T: wmma::mma_sync (16x16x16 tiles, FP16→FP32)\n");
         printf("    → P@V:   CUDA cores (FP32 input limitation)\n");
         printf("  Softmax: PARALLEL via warp shuffle (cooperative warp reduction)\n");
         printf("================================================================================\n");
         first_call = false;
     }
     
     flash_attn_cutlass_kernel<HEAD_DIM><<<grid, block, smem_size, stream>>>(
         Q, K, V, O,
         softmax_scale,
         batch_size, num_heads, seq_len
     );
     
     cudaError_t err = cudaGetLastError();
     if (err != cudaSuccess) {
         fprintf(stderr, "CUTLASS Flash attention kernel launch failed: %s\n", 
                 cudaGetErrorString(err));
     }
 }
 
 // ==================== Public Dispatch Function ====================
 
 void flash_attention_cutlass_dispatch(
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
     switch (head_dim) {
         case 32:
             flash_attn_cutlass_forward<32>(Q, K, V, O, batch_size, num_heads, seq_len, stream);
             break;
         case 64:
             flash_attn_cutlass_forward<64>(Q, K, V, O, batch_size, num_heads, seq_len, stream);
             break;
         case 128:
             flash_attn_cutlass_forward<128>(Q, K, V, O, batch_size, num_heads, seq_len, stream);
             break;
         default:
             fprintf(stderr, "Unsupported head_dim=%d for CUTLASS (supported: 32, 64, 128)\n", head_dim);
             break;
     }
 }
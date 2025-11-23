/******************************************************************************
 * Flash Attention with CUTLASS GEMM (Corrected Version)
 * * Fixes:
 * 1. Tile alignment (kTileM must be multiple of 16 for WMMA)
 * 2. Correct O_accum rescaling logic (Store-Scale-Load)
 * 3. Proper Shared Memory allocation for intermediate scaling
 ******************************************************************************/

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cmath>
#include <algorithm>
#include <cstdio>

using namespace nvcuda;

// ==================== Tile Configuration (FIXED) ====================
template<int HEAD_DIM>
struct CutlassSmallTileConfig {
    static constexpr int compute_small_tile_size() {
        if (HEAD_DIM == 128) return 32; 
        if (HEAD_DIM == 64) return 64; 
        if (HEAD_DIM == 32) return 64;
        return 32;
    }
    
    static constexpr int kTileM = compute_small_tile_size() / 2; // e.g., 16 for dim128
    static constexpr int kTileN = compute_small_tile_size();     // e.g., 32 for dim128
    static constexpr int kHeadDim = HEAD_DIM;
    static constexpr int kThreads = 128; // Reduced threads to match smaller tile work
    
    static constexpr size_t get_smem_size() {
        // Standard QKV storage
        size_t qkv_size = (kTileM * kHeadDim + kTileN * kHeadDim * 2) * sizeof(cutlass::half_t);
        if (qkv_size % 16 != 0) qkv_size += (16 - (qkv_size % 16));

        int num_warps = kThreads / 32;
        // [FIX] Increased per-warp scratch: 
        // Need space for S matrix (float 16x16) AND O rescale buffer (float 16x16)
        // 16*16*4 bytes = 1024 bytes per buffer. Total 2048 is enough.
        size_t scratch_per_warp = 2048 + 1024; 
        
        return qkv_size + (num_warps * scratch_per_warp);
    }
};

// ==================== Shared Memory Layout ====================
template<typename T, int TILE_M, int TILE_N, int HEAD_DIM>
struct SharedMemoryCutlass {
    T* Q;
    T* K;
    T* V;
    
    __device__ SharedMemoryCutlass(void* ptr) {
        char* base = reinterpret_cast<char*>(ptr);
        Q = reinterpret_cast<T*>(base);
        K = reinterpret_cast<T*>(base + TILE_M * HEAD_DIM * sizeof(T));
        V = reinterpret_cast<T*>(base + (TILE_M * HEAD_DIM + TILE_N * HEAD_DIM) * sizeof(T));
    }
};

// ==================== Helper: Rescale O via SMEM (The Fix) ====================
// WMMA fragments are opaque. To scale row 'i' by 'correction[i]', 
// we must store to memory, scale spatially, and load back.
__device__ __forceinline__ void apply_rescaling_via_smem(
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>& o_frag,
    float* scratch_ptr,      // Pointer to per-warp shared mem
    const float* corrections, // Array of 16 correction factors
    int m_valid,
    int laneId
) {
    // 1. Store Fragment to Shared Memory
    wmma::store_matrix_sync(scratch_ptr, o_frag, 16, wmma::mem_row_major);
    __syncwarp();

    // 2. Apply correction spatially
    // Each thread processes a subset of the 16x16 matrix
    #pragma unroll
    for (int i = laneId; i < 256; i += 32) {
        int row = i / 16;
        // int col = i % 16; // Not needed for logic, just linear access
        if (row < m_valid) {
            scratch_ptr[i] *= corrections[row];
        }
    }
    __syncwarp();

    // 3. Load back to Fragment
    wmma::load_matrix_sync(o_frag, scratch_ptr, 16, wmma::mem_row_major);
}

// ==================== Kernel ====================

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
     const int warpId = tid / 32;
     
     // Bounds check
     const int q_start = q_block_idx * kTileM;
     if (q_start >= seq_len) return;
     
     const int q_end = min(q_start + kTileM, seq_len);
     const int q_size = q_end - q_start;
     
     const int64_t offset = (int64_t)(batch_idx * num_heads + head_idx) * seq_len * HEAD_DIM;
     const cutlass::half_t* Q_base = Q + offset;
     const cutlass::half_t* K_base = K + offset;
     const cutlass::half_t* V_base = V + offset;
     cutlass::half_t* O_base = O + offset;
     
     extern __shared__ char smem[];
     SharedMemoryCutlass<cutlass::half_t, kTileM, kTileN, HEAD_DIM> shared_mem(smem);
     
     // Setup Scratchpad Pointers
     size_t qkv_offset = (kTileM * HEAD_DIM + kTileN * HEAD_DIM * 2) * sizeof(cutlass::half_t);
     if (qkv_offset % 16 != 0) qkv_offset += (16 - (qkv_offset % 16));
     
     // Give each warp independent scratch space
     // Buffer 1: For S / P matrix (float)
     float* s_scratch = reinterpret_cast<float*>(smem + qkv_offset + (warpId * 3072));
     // Buffer 2: For O rescaling (float) - Offset by 256 floats (1KB)
     float* o_scratch = s_scratch + 256; 
     // Pointer for P conversion (half) - Reuses s_scratch later
     cutlass::half_t* p_half_scratch = reinterpret_cast<cutlass::half_t*>(s_scratch + 256); // Overlap safe if careful? No, let's use o_scratch for p_half

     // 1. Load Q Tile (Persistent)
     for (int idx = tid; idx < q_size * HEAD_DIM; idx += blockDim.x) {
         int i = idx / HEAD_DIM;
         int j = idx % HEAD_DIM;
         shared_mem.Q[i * HEAD_DIM + j] = Q_base[(q_start + i) * HEAD_DIM + j];
     }
     __syncthreads();

     // Initialize O accumulators and Stats
     constexpr int MAX_FRAGS = HEAD_DIM / 16; 
     wmma::fragment<wmma::accumulator, 16, 16, 16, float> O_accums[MAX_FRAGS];
     #pragma unroll
     for (int i = 0; i < MAX_FRAGS; i++) wmma::fill_fragment(O_accums[i], 0.0f);

     float m_reg[16];       
     float l_reg[16];       
     #pragma unroll
     for (int i = 0; i < 16; i++) { m_reg[i] = -INFINITY; l_reg[i] = 0.0f; }

     // Identify which rows of Q this warp handles
     int m_base_warp = warpId * 16; // Warp 0 -> 0..15, Warp 1 -> 16..31
     int m_valid = 0;
     if (m_base_warp < q_size) {
         m_valid = min(16, q_size - m_base_warp);
     }

     // ==== MAIN LOOP ====
     const int num_kv_tiles = (seq_len + kTileN - 1) / kTileN;

     // Only active warps proceed
     if (m_valid > 0) {
         for (int kv_tile_idx = 0; kv_tile_idx < num_kv_tiles; kv_tile_idx++) {
             const int k_start = kv_tile_idx * kTileN;
             const int k_end = min(k_start + kTileN, seq_len);
             const int k_size = k_end - k_start;
             
             // Load K, V
             __syncthreads(); 
             for (int idx = tid; idx < k_size * HEAD_DIM; idx += blockDim.x) {
                 int i = idx / HEAD_DIM;
                 int j = idx % HEAD_DIM;
                 shared_mem.K[i * HEAD_DIM + j] = K_base[(k_start + i) * HEAD_DIM + j];
                 shared_mem.V[i * HEAD_DIM + j] = V_base[(k_start + i) * HEAD_DIM + j];
             }
             __syncthreads();
             
             // Loop over K chunks (16 columns at a time)
             for (int k_base = 0; k_base < k_size; k_base += 16) {
                 int k_valid = min(16, k_size - k_base);
                 
                 // --- Step A: Compute S = Q @ K^T ---
                 wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;
                 wmma::fill_fragment(s_frag, 0.0f);
                 
                 for (int h_dim = 0; h_dim < HEAD_DIM; h_dim += 16) {
                     wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag;
                     wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag;
                     
                     const half* q_p = reinterpret_cast<const half*>(shared_mem.Q + m_base_warp * HEAD_DIM + h_dim);
                     const half* k_p = reinterpret_cast<const half*>(shared_mem.K + (k_base + k_base) * HEAD_DIM + h_dim); // Wrong indexing in original
                     // Correct indexing for K: [k_base (row in Tile), h_dim (col in Tile)]
                     // SharedMem K is [TileN, HeadDim]
                     const half* k_ptr_correct = reinterpret_cast<const half*>(shared_mem.K + (k_base) * HEAD_DIM + h_dim);

                     wmma::load_matrix_sync(q_frag, q_p, HEAD_DIM);
                     wmma::load_matrix_sync(k_frag, k_ptr_correct, HEAD_DIM);
                     wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
                 }
                 
                 // --- Step B: Softmax (Store S -> Compute Stats -> Update P) ---
                 wmma::store_matrix_sync(s_scratch, s_frag, 16, wmma::mem_row_major);
                 __syncwarp();
                 
                 const int laneId = tid % 32;
                 float row_corrections[16]; // Store corrections to apply to O later
                 
                 for (int row = 0; row < m_valid; row++) {
                     float row_max = -INFINITY;
                     for (int col = laneId; col < k_valid; col += 32) {
                         row_max = fmaxf(row_max, s_scratch[row * 16 + col] * softmax_scale);
                     }
                     #pragma unroll
                     for (int offset = 16; offset > 0; offset /= 2) row_max = fmaxf(row_max, __shfl_down_sync(0xffffffff, row_max, offset));
                     row_max = __shfl_sync(0xffffffff, row_max, 0);
                     
                     float m_prev = m_reg[row];
                     float m_curr = fmaxf(m_prev, row_max);
                     // [FIX] Compute correction for O
                     float correction = expf(m_prev - m_curr);
                     row_corrections[row] = correction;
                     
                     float row_sum = 0.0f;
                     for (int col = laneId; col < k_valid; col += 32) {
                         float val = s_scratch[row * 16 + col] * softmax_scale;
                         float p = expf(val - m_curr);
                         s_scratch[row * 16 + col] = p; // Write P back
                         row_sum += p;
                     }
                     #pragma unroll
                     for (int offset = 16; offset > 0; offset /= 2) row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
                     row_sum = __shfl_sync(0xffffffff, row_sum, 0);
                     
                     if (laneId == 0) {
                         m_reg[row] = m_curr;
                         l_reg[row] = l_reg[row] * correction + row_sum;
                     }
                 }
                 __syncwarp();
                 
                 // --- Step C: Rescale O Accumulators (FIXED) ---
                 // Done BEFORE adding P@V. 
                 // We iterate over all O fragments and apply row_corrections.
                 for(int f = 0; f < MAX_FRAGS; f++) {
                     apply_rescaling_via_smem(O_accums[f], o_scratch, row_corrections, m_valid, laneId);
                 }
                 
                 // --- Step D: Compute P @ V ---
                 // Convert P (float in s_scratch) to Half. 
                 // We can use o_scratch as temp buffer for half conversion if cast properly
                 cutlass::half_t* p_half_ptr = reinterpret_cast<cutlass::half_t*>(o_scratch);
                 
                 for (int idx = laneId; idx < 16 * 16; idx += 32) {
                     p_half_ptr[idx] = cutlass::half_t(s_scratch[idx]);
                 }
                 __syncwarp();
                 
                 wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
                 wmma::load_matrix_sync(p_frag, reinterpret_cast<half*>(p_half_ptr), 16);
                 
                 for (int h_chunk = 0; h_chunk < MAX_FRAGS; h_chunk++) {
                     int h_dim = h_chunk * 16;
                     wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> v_frag;
                     const half* v_p = reinterpret_cast<const half*>(shared_mem.V + k_base * HEAD_DIM + h_dim);
                     
                     if (k_base < k_size) {
                         wmma::load_matrix_sync(v_frag, v_p, HEAD_DIM);
                         wmma::mma_sync(O_accums[h_chunk], p_frag, v_frag, O_accums[h_chunk]);
                     }
                 }
                 
             } // End K-block
         } // End KV-Tile
         
         // ==== FINALIZATION ====
         // Normalize O by l_reg and write to Global Memory
         for(int h_chunk = 0; h_chunk < MAX_FRAGS; h_chunk++) {
             int h_dim = h_chunk * 16;
             // Dump un-normalized O to scratch
             wmma::store_matrix_sync(s_scratch, O_accums[h_chunk], 16, wmma::mem_row_major);
             __syncwarp();
             
             for (int i = laneId; i < 16 * 16; i+=32) {
                 int r = i / 16;
                 int c = i % 16;
                 
                 if (r < m_valid && (h_dim + c) < HEAD_DIM) {
                     float val = s_scratch[i];
                     float norm = (l_reg[r] == 0.0f) ? 0.0f : (1.0f / l_reg[r]);
                     
                     int global_row = q_start + m_base_warp + r;
                     if (global_row < q_size + q_start) {
                        O_base[(m_base_warp + r) * HEAD_DIM + (h_dim + c)] = cutlass::half_t(val * norm);
                     }
                 }
             }
         }
     }
}

// ==================== Host Wrapper ====================

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
    
    // Safety check for SMEM
    cudaFuncSetAttribute(
        flash_attn_cutlass_kernel<HEAD_DIM>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        65536 // Request enough for larger tiles
    );
    
    flash_attn_cutlass_kernel<HEAD_DIM><<<grid, block, smem_size, stream>>>(
        Q, K, V, O,
        softmax_scale,
        batch_size, num_heads, seq_len
    );
}

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
    if (head_dim == 128) {
         flash_attn_cutlass_forward<128>(Q, K, V, O, batch_size, num_heads, seq_len, stream);
    } else if (head_dim == 64) {
         flash_attn_cutlass_forward<64>(Q, K, V, O, batch_size, num_heads, seq_len, stream);
    } else {
        printf("Unsupported head dim\n");
    }
}
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cmath>
#include <algorithm>
#include <cstdio>

using namespace nvcuda;

// ==================== Tile Configuration ====================
template<int HEAD_DIM>
struct CutlassSmallTileConfig {
    static constexpr int kTileM = 64; 
    static constexpr int kTileN = 32;     
    static constexpr int kHeadDim = HEAD_DIM;
    static constexpr int kThreads = 128; // 1 Warps
    
    static constexpr size_t align16(size_t size) {
        return (size % 16 == 0) ? size : size + (16 - (size % 16));
    }

    static constexpr size_t get_smem_size() {
        size_t q_sz = align16(kTileM * kHeadDim * sizeof(half));
        size_t k_sz = align16(kTileN * kHeadDim * sizeof(half));
        size_t v_sz = align16(kTileN * kHeadDim * sizeof(half));
        
        int num_warps = kThreads / 32;
        size_t scratch_per_warp = 2048 + 256; 
        
        return q_sz + k_sz + v_sz + (num_warps * scratch_per_warp);
    }
};

__device__ __forceinline__ void apply_rescaling_via_smem(
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>& o_frag,
    float* scratch_ptr,
    const float* corrections,
    int m_valid,
    int laneId
) {
    // 1. Dump Fragment to Shared Mem
    wmma::store_matrix_sync(scratch_ptr, o_frag, 16, wmma::mem_row_major);
    __syncwarp();

    // 2. Apply correction factor to each row
    #pragma unroll
    for (int i = laneId; i < 256; i += 32) {
        int row = i / 16;
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
     const cutlass::half_t* __restrict__ Q_global,
     const cutlass::half_t* __restrict__ K_global,
     const cutlass::half_t* __restrict__ V_global,
     cutlass::half_t* __restrict__ O_global,
     float softmax_scale,
     int batch_size,
     int num_heads,
     int seq_len,

     int stride_b,
     int stride_h,
     int stride_s
) {
     using Config = CutlassSmallTileConfig<HEAD_DIM>;
     constexpr int kTileM = Config::kTileM;
     constexpr int kTileN = Config::kTileN;
     
     // Force 128-bit alignment for Dynamic Shared Memory
     extern __shared__ __align__(16) char smem[];
     
     const int tid = threadIdx.x;
     const int warpId = tid / 32;
     const int laneId = tid % 32; 
     
     const int batch_idx = blockIdx.z;
     const int head_idx = blockIdx.y;
     const int q_block_idx = blockIdx.x;
     
     // [B,S,H,D]
     size_t batch_head_offset = (size_t)batch_idx * stride_b + (size_t)head_idx * stride_h;
     
     const half* Q_base = reinterpret_cast<const half*>(Q_global) + batch_head_offset;
     const half* K_base = reinterpret_cast<const half*>(K_global) + batch_head_offset;
     const half* V_base = reinterpret_cast<const half*>(V_global) + batch_head_offset;
     half* O_base       = reinterpret_cast<half*>(O_global) + batch_head_offset;

     // Global Sequence Boundaries
     const int q_start = q_block_idx * kTileM;
     if (q_start >= seq_len) return;
     const int q_end = min(q_start + kTileM, seq_len);
     const int q_size = q_end - q_start;
     
     // Shared Memory Pointer Setup
     size_t q_sz = Config::align16(Config::kTileM * HEAD_DIM * sizeof(half));
     size_t k_sz = Config::align16(Config::kTileN * HEAD_DIM * sizeof(half));
     size_t v_sz = Config::align16(Config::kTileN * HEAD_DIM * sizeof(half));
     
     half* smem_Q = reinterpret_cast<half*>(smem);
     half* smem_K = reinterpret_cast<half*>(smem + q_sz);
     half* smem_V = reinterpret_cast<half*>(smem + q_sz + k_sz);
     
     // Scratch memory setup
     float* s_scratch = reinterpret_cast<float*>(smem + q_sz + k_sz + v_sz + (warpId * 2304));
     float* o_scratch = s_scratch + 256; 

     // ================= Load Q Tile (Cooperative) =================
     // Note: We use stride_s to jump between sequence rows
     for (int idx = tid; idx < q_size * HEAD_DIM; idx += blockDim.x) {
         int r = idx / HEAD_DIM;
         int c = idx % HEAD_DIM;
         smem_Q[idx] = Q_base[(q_start + r) * stride_s + c];
     }
     __syncthreads();

     // Initialize Accumulators
     constexpr int MAX_FRAGS = HEAD_DIM / 16; 
     wmma::fragment<wmma::accumulator, 16, 16, 16, float> O_accums[MAX_FRAGS];
     #pragma unroll
     for (int i = 0; i < MAX_FRAGS; i++) wmma::fill_fragment(O_accums[i], 0.0f);

     float m_reg[16];       
     float l_reg[16];       
     #pragma unroll
     for (int i = 0; i < 16; i++) { m_reg[i] = -INFINITY; l_reg[i] = 0.0f; }

     int m_base_warp = warpId * 16; 
     int m_valid = 0;
     if (m_base_warp < q_size) {
         m_valid = min(16, q_size - m_base_warp);
     }

     const int num_kv_tiles = (seq_len + kTileN - 1) / kTileN;

     // ================= MAIN LOOP =================
     for (int kv_tile_idx = 0; kv_tile_idx < num_kv_tiles; kv_tile_idx++) {
         const int k_start = kv_tile_idx * kTileN;
         const int k_end = min(k_start + kTileN, seq_len);
         const int k_size = k_end - k_start;
         
         // 1. Load K and V (Cooperative)
         __syncthreads(); 
         for (int idx = tid; idx < k_size * HEAD_DIM; idx += blockDim.x) {
             int r = idx / HEAD_DIM;
             int c = idx % HEAD_DIM;
             smem_K[idx] = K_base[(k_start + r) * stride_s + c];
             smem_V[idx] = V_base[(k_start + r) * stride_s + c];
         }
         __syncthreads();
         
         // 2. Compute
         if (m_valid > 0) {
             for (int k_base = 0; k_base < k_size; k_base += 16) {
                 int k_valid = min(16, k_size - k_base);
                 
                 // --- Step A: S = Q @ K^T ---
                 wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;
                 wmma::fill_fragment(s_frag, 0.0f);
                 
                 for (int h_dim = 0; h_dim < HEAD_DIM; h_dim += 16) {
                     wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag;
                     wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag; // Col Major implies Transpose for Row Major Data
                     
                     wmma::load_matrix_sync(q_frag, smem_Q + m_base_warp * HEAD_DIM + h_dim, HEAD_DIM);
                     wmma::load_matrix_sync(k_frag, smem_K + k_base * HEAD_DIM + h_dim, HEAD_DIM);
                     wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
                 }
                 
                 // --- Step B: Softmax ---
                 wmma::store_matrix_sync(s_scratch, s_frag, 16, wmma::mem_row_major);
                 __syncwarp();
                 
                 float row_corrections[16]; 
                 for (int row = 0; row < m_valid; row++) {
                     float row_max = -INFINITY;
                     
                     // 1. Find Max
                     for (int col = laneId; col < k_valid; col += 32) {
                         row_max = fmaxf(row_max, s_scratch[row * 16 + col] * softmax_scale);
                     }
                     #pragma unroll
                     for (int offset = 16; offset > 0; offset /= 2) row_max = fmaxf(row_max, __shfl_down_sync(0xffffffff, row_max, offset));
                     row_max = __shfl_sync(0xffffffff, row_max, 0);
                     
                     // 2. Stats Update
                     float m_prev = m_reg[row];
                     float m_curr = fmaxf(m_prev, row_max);
                     float correction = __expf(m_prev - m_curr);
                     row_corrections[row] = correction;
                     
                     float row_sum = 0.0f;
                     
                     // 3. Compute P & Sum (With ZERO-MASKING Fix)
                     // Iterate full 16 columns to guarantee we mask invalid ones
                     for (int col = laneId; col < 16; col += 32) {
                         if (col < k_valid) {
                             float val = s_scratch[row * 16 + col] * softmax_scale;
                             float p = __expf(val - m_curr);
                             s_scratch[row * 16 + col] = p;
                             row_sum += p;
                         } else {
                             s_scratch[row * 16 + col] = 0.0f; // [FIX] Important!
                         }
                     }
                     #pragma unroll
                     for (int offset = 16; offset > 0; offset /= 2) row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
                     row_sum = __shfl_sync(0xffffffff, row_sum, 0);
                     
                     m_reg[row] = m_curr;
                     l_reg[row] = l_reg[row] * correction + row_sum;
                 }
                 __syncwarp();
                 
                 // --- Step C: Rescale O ---
                 for(int f = 0; f < MAX_FRAGS; f++) {
                     apply_rescaling_via_smem(O_accums[f], o_scratch, row_corrections, m_valid, laneId);
                 }
                 
                 // --- Step D: P @ V ---
                 half* p_half_ptr = reinterpret_cast<half*>(o_scratch);
                 for (int idx = laneId; idx < 16 * 16; idx += 32) {
                     p_half_ptr[idx] = (half)s_scratch[idx];
                 }
                 __syncwarp();
                 
                 wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
                 wmma::load_matrix_sync(p_frag, p_half_ptr, 16);
                 
                 for (int h_chunk = 0; h_chunk < MAX_FRAGS; h_chunk++) {
                     int h_dim = h_chunk * 16;
                     // V Loaded Row Major (Standard Matrix Mul)
                     wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
                     wmma::load_matrix_sync(v_frag, smem_V + k_base * HEAD_DIM + h_dim, HEAD_DIM);
                     wmma::mma_sync(O_accums[h_chunk], p_frag, v_frag, O_accums[h_chunk]);
                 }
             }
         }
     }
     
     // ================= FINALIZATION =================
     if (m_valid > 0) {
         for(int h_chunk = 0; h_chunk < MAX_FRAGS; h_chunk++) {
             int h_dim = h_chunk * 16;
             wmma::store_matrix_sync(s_scratch, O_accums[h_chunk], 16, wmma::mem_row_major);
             __syncwarp();
             
             for (int i = laneId; i < 16 * 16; i+=32) {
                 int r = i / 16;
                 int c = i % 16;
                 if (r < m_valid && (h_dim + c) < HEAD_DIM) {
                     float val = s_scratch[i];
                     // Guard against div-by-zero (masked/empty rows)
                     float norm = (l_reg[r] > 1e-6f) ? (1.0f / l_reg[r]) : 0.0f;
                     
                     // Output write using stride_s
                     O_base[(q_start + m_base_warp + r) * stride_s + (h_dim + c)] = (half)(val * norm);
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
    
    // [STRIDE CONFIG]
    // Assuming Standard PyTorch Layout: [Batch, Seq, Head, Dim] (B, S, H, D)
    // If your input is [B, H, S, D], swap stride_s and stride_h.
    int stride_b = seq_len * num_heads * HEAD_DIM;
    int stride_s = num_heads * HEAD_DIM;
    int stride_h = HEAD_DIM;
    
    const int num_q_blocks = (seq_len + Config::kTileM - 1) / Config::kTileM;
    
    dim3 grid(num_q_blocks, num_heads, batch_size);
    dim3 block(Config::kThreads);
    size_t smem_size = Config::get_smem_size();
    
    cudaFuncSetAttribute(flash_attn_cutlass_kernel<HEAD_DIM>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    
    flash_attn_cutlass_kernel<HEAD_DIM><<<grid, block, smem_size, stream>>>(
        Q, K, V, O, softmax_scale, batch_size, num_heads, seq_len,
        stride_b, stride_h, stride_s
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

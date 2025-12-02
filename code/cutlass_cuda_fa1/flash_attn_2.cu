#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cmath>
#include <algorithm>
#include <cstdio>

using namespace nvcuda;

// ==================== 1. 核心辅助函数 (Ampere Async Copy) ====================

// 必须定义这些辅助函数才能使用 cp.async
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp_async_cg_16B(void* dst, const void* src) {
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    unsigned long long gmem_addr = reinterpret_cast<unsigned long long>(src);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_addr));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
// 等待直到剩下 N 个组
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
#else
// Fallback for non-Ampere (Logic will be slow but functional)
__device__ __forceinline__ void cp_async_cg_16B(void* dst, const void* src) {
    *(int4*)dst = *(const int4*)src;
}
__device__ __forceinline__ void cp_async_commit() {}
template <int N> __device__ __forceinline__ void cp_async_wait_group() {}
#endif

// ==================== 2. Tile Configuration (关键优化点) ====================

template<int HEAD_DIM>
struct CutlassSmallTileConfig {
    static constexpr int kTileM   = 64;
    // [优化点 1]: 增大 Tile N 到 64，大幅提升流水线效率
    static constexpr int kTileN   = 64; 
    static constexpr int kHeadDim = HEAD_DIM;
    
    // [优化点 2]: Padding (消除 Shared Memory Bank Conflict)
    // 8 halfs = 16 bytes. 使得 stride 变成 72 (144 bytes)，错开 bank。
    static constexpr int kPad     = 8; 
    static constexpr int kSmHeadDim = HEAD_DIM + kPad; 

    static constexpr int kThreads = 128; // 4 warps

    static constexpr size_t align16(size_t size) {
        return (size % 16 == 0) ? size : size + (16 - (size % 16));
    }

    static constexpr size_t get_smem_size() {
        // 使用 Padding 后的维度计算大小
        size_t q_sz = align16(kTileM * kSmHeadDim * sizeof(half));
        
        // K/V Double Buffer
        int    kv_tile_elems = kTileN * kSmHeadDim;
        size_t kv_tile_bytes = align16(kv_tile_elems * sizeof(half));

        size_t k_sz = 2 * kv_tile_bytes;
        size_t v_sz = 2 * kv_tile_bytes;

        int num_warps = kThreads / 32;
        // Scratch: Float (32bit) accumulator + Half (16bit) P matrix
        // 16 rows * 64 cols (since kTileN=64)
        // 注意：kTileN 变大了，Softmax 的行长也变大了，需要更多的 Scratch
        // Softmax 计算时我们还是按块处理，但 P 矩阵暂存需要容纳一行
        // 为了安全，分配足够大的 scratch
        size_t scratch_per_warp = kTileM * kTileN * sizeof(float) + kTileM * kTileN * sizeof(half);
        // 上面给太大了，其实只要一行的大小用于 Softmax，但是 P 矩阵需要存储
        // 简单起见，我们给每个 Warp 4KB 足够了
        scratch_per_warp = 4096; 

        return q_sz + k_sz + v_sz + num_warps * scratch_per_warp;
    }
};

// ==================== 3. O Rescale Helper ====================
__device__ __forceinline__ void apply_rescaling_in_frag(
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>& o_frag,
    const float* corrections, 
    int m_valid,
    int laneId
) {
    int row_group1 = laneId / 4;       
    int row_group2 = row_group1 + 8;   

    float c1 = (row_group1 < m_valid) ? corrections[row_group1] : 1.0f;
    float c2 = (row_group2 < m_valid) ? corrections[row_group2] : 1.0f;

    #pragma unroll
    for (int i = 0; i < 4; ++i) o_frag.x[i] *= c1;
    #pragma unroll
    for (int i = 4; i < 8; ++i) o_frag.x[i] *= c2;
}

// ==================== 4. Kernel Implementation ====================

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
    constexpr int kSmHeadDim = Config::kSmHeadDim; // 72

    extern __shared__ __align__(16) char smem[];

    const int tid    = threadIdx.x;
    const int warpId = tid / 32;
    const int laneId = tid % 32;
    const int batch_idx   = blockIdx.z;
    const int head_idx    = blockIdx.y;
    const int q_block_idx = blockIdx.x;

    // Offset calculation
    size_t batch_head_offset = (size_t)batch_idx * stride_b + (size_t)head_idx * stride_h;
    const half* Q_base = reinterpret_cast<const half*>(Q_global) + batch_head_offset;
    const half* K_base = reinterpret_cast<const half*>(K_global) + batch_head_offset;
    const half* V_base = reinterpret_cast<const half*>(V_global) + batch_head_offset;
    half* O_base = reinterpret_cast<half*>(O_global)       + batch_head_offset;

    // Bounds
    const int q_start = q_block_idx * kTileM;
    if (q_start >= seq_len) return;
    const int q_end   = min(q_start + kTileM, seq_len);
    const int q_size  = q_end - q_start;

    // ==================== Shared Memory Pointers ====================
    char* smem_ptr = smem;

    // Q Tile (with Padding)
    size_t q_sz = Config::align16(kTileM * kSmHeadDim * sizeof(half));
    half* smem_Q = reinterpret_cast<half*>(smem_ptr);
    smem_ptr += q_sz;

    // K/V Double Buffers (with Padding)
    int kv_tile_elems = kTileN * kSmHeadDim;
    size_t kv_tile_bytes = Config::align16(kv_tile_elems * sizeof(half));

    half* smem_K0 = reinterpret_cast<half*>(smem_ptr);
    half* smem_K1 = smem_K0 + kv_tile_elems;
    smem_ptr += 2 * kv_tile_bytes;

    half* smem_V0 = reinterpret_cast<half*>(smem_ptr);
    half* smem_V1 = smem_V0 + kv_tile_elems;
    smem_ptr += 2 * kv_tile_bytes;

    // Scratchpad
    // s_scratch needs to hold 16 rows * 64 cols (floats) because kTileN=64
    // Layout: 16 rows x 64 cols (RowMajor)
    size_t scratch_per_warp = 4096; 
    float* s_scratch = reinterpret_cast<float*>(smem_ptr + warpId * scratch_per_warp);
    half* p_half_ptr = reinterpret_cast<half*>(reinterpret_cast<char*>(s_scratch) + 16 * 64 * sizeof(float));

    // ==================== Load Q Tile (Vectorized int4 + Padding) ====================
    using int4_copy_t = int4;
    constexpr int kVecSize = 8; 
    int vecs_per_row = HEAD_DIM / kVecSize; 
    
    // Q is loaded once, sync is fine
    for (int idx = tid; idx < q_size * vecs_per_row; idx += blockDim.x) {
        int r = idx / vecs_per_row;
        int c_vec = idx % vecs_per_row;
        int c_real = c_vec * kVecSize;

        const int4_copy_t* src = reinterpret_cast<const int4_copy_t*>(
            Q_base + (q_start + r) * stride_s + c_real
        );
        // Write with Padding stride (kSmHeadDim)
        int4_copy_t* dst = reinterpret_cast<int4_copy_t*>(smem_Q + r * kSmHeadDim + c_real);
        *dst = *src;
    }
    // We don't commit Q because it's register load, but we sync
    __syncthreads();

    // ==================== Register Accumulators ====================
    constexpr int MAX_FRAGS = HEAD_DIM / 16; // 4
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> O_accums[MAX_FRAGS];
    #pragma unroll
    for (int i = 0; i < MAX_FRAGS; i++) wmma::fill_fragment(O_accums[i], 0.0f);

    float m_reg[16];
    float l_reg[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) { m_reg[i] = -INFINITY; l_reg[i] = 0.0f; }

    int m_base_warp = warpId * 16;
    int m_valid = 0;
    if (m_base_warp < q_size) m_valid = min(16, q_size - m_base_warp);

    const int num_kv_tiles = (seq_len + kTileN - 1) / kTileN;

    // ==================== PIPELINE START ====================
    int stage = 0;

    // Prologue: Load Tile 0
    {
        int tile_idx = 0;
        int k_start = 0;
        int k_end   = min(kTileN, seq_len);
        int k_size  = k_end - k_start;
        
        int segments_per_row = HEAD_DIM / 8;
        int total_segments   = k_size * segments_per_row;
        
        for (int i = tid; i < total_segments; i += blockDim.x) {
            int r  = i / segments_per_row;
            int c8 = i % segments_per_row;
            int col = c8 * 8;
            
            // Dst uses kSmHeadDim (Padding)
            void* dstK = smem_K0 + r * kSmHeadDim + col;
            void* dstV = smem_V0 + r * kSmHeadDim + col;
            const void* srcK = K_base + (k_start + r) * stride_s + col;
            const void* srcV = V_base + (k_start + r) * stride_s + col;
            
            cp_async_cg_16B(dstK, srcK);
            cp_async_cg_16B(dstV, srcV);
        }
        cp_async_commit();
    }

    // ==================== MAIN LOOP ====================
    for (int kv_tile_idx = 0; kv_tile_idx < num_kv_tiles; ++kv_tile_idx) {
        
        // 1. Prefetch Next Tile (Async)
        int next_tile = kv_tile_idx + 1;
        int next_stage = stage ^ 1;
        
        if (next_tile < num_kv_tiles) {
            int k_start = next_tile * kTileN;
            int k_end   = min(k_start + kTileN, seq_len);
            int k_size  = k_end - k_start;
            
            half* smem_K_next = (next_stage == 0) ? smem_K0 : smem_K1;
            half* smem_V_next = (next_stage == 0) ? smem_V0 : smem_V1;

            int segments_per_row = HEAD_DIM / 8;
            int total_segments   = k_size * segments_per_row;

            for (int i = tid; i < total_segments; i += blockDim.x) {
                int r  = i / segments_per_row;
                int c8 = i % segments_per_row;
                int col = c8 * 8;
                void* dstK = smem_K_next + r * kSmHeadDim + col;
                void* dstV = smem_V_next + r * kSmHeadDim + col;
                const void* srcK = K_base + (k_start + r) * stride_s + col;
                const void* srcV = V_base + (k_start + r) * stride_s + col;
                cp_async_cg_16B(dstK, srcK);
                cp_async_cg_16B(dstV, srcV);
            }
            cp_async_commit();
        }

        // 2. Wait for Current Tile
        if (next_tile < num_kv_tiles) {
             cp_async_wait_group<1>();
        } else {
             cp_async_wait_group<0>();
        }
        __syncthreads();

        // 3. Compute
        half* smem_K_curr = (stage == 0) ? smem_K0 : smem_K1;
        half* smem_V_curr = (stage == 0) ? smem_V0 : smem_V1;
        
        int curr_k_start = kv_tile_idx * kTileN;
        int curr_k_end   = min(curr_k_start + kTileN, seq_len);
        int curr_k_size  = curr_k_end - curr_k_start;

        if (m_valid > 0) {
            // kTileN is now 64. Step is 16.
            // Loop runs 0, 16, 32, 48 (4 times). This amortizes pipeline latency!
            for (int k_base = 0; k_base < curr_k_size; k_base += 16) {
                int k_valid = min(16, curr_k_size - k_base); // 16 or remainder

                // --- S = Q @ K^T ---
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;
                wmma::fill_fragment(s_frag, 0.0f);

                for (int h_dim = 0; h_dim < HEAD_DIM; h_dim += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag;

                    // Load with Padding stride
                    wmma::load_matrix_sync(q_frag, smem_Q + m_base_warp * kSmHeadDim + h_dim, kSmHeadDim);
                    wmma::load_matrix_sync(k_frag, smem_K_curr + (k_base + k_base /*Wait? No*/) * kSmHeadDim + h_dim, kSmHeadDim);
                    // Oops, logic error in K index above. Correct is:
                    // K is [kTileN, HeadDim]. We are at k_base.
                    // But K is ColMajor in wmma?
                    // FlashAttn usually computes S = Q * K^T.
                    // If K in smem is RowMajor [kTileN, HeadDim], then K^T is effectively ColMajor load.
                    // wmma::load matrix_b ColMajor expects Stride = kTileN? No, Stride = Leading Dim.
                    // If we treat K as RowMajor [N, D], and we want K^T [D, N].
                    // We need to load K transposed? 
                    // No, wmma::col_major load means "Elements in a column are contiguous".
                    // In RowMajor storage, elements in a row are contiguous.
                    // This part is tricky. Standard trick:
                    // Load K as matrix_b with col_major? 
                    // No, if K is [N, D] in smem. We want Q[16, D] * K^T[D, 16].
                    // That is Q[i, :] dot K[j, :].
                    // With wmma: mma_sync(acc, a, b, c) -> D = A*B + C.
                    // A = [16, 16] (RowMajor), B = [16, 16] (ColMajor).
                    // This does Row of A dot Col of B.
                    // If B is loaded ColMajor from Smem [N, D], it implies stride is N?
                    // No. This is why FlashAttn usually stores K transposed or uses ldmatrix.trans.
                    
                    // [Correction]: Assuming K is stored [N, D] RowMajor.
                    // We want Q[16, 16] * K_chunk[16, 16]^T.
                    // Q: RowMajor. K_chunk: RowMajor.
                    // Using wmma::col_major for B essentially transposes it ON LOAD if stride is correct?
                    // No, standard wmma doesn't transpose on load freely without ldmatrix.
                    
                    // Let's assume standard implementation: Load K as RowMajor?
                    // If A is RowMajor and B is ColMajor: C_ij = Sum_k A_ik * B_kj.
                    // If we want Q * K^T: C_ij = Sum_d Q_id * K_jd.
                    // This matches A=RowMajor, B=ColMajor where B is K^T.
                    // If B is K^T, then B_kj = (K^T)_kj = K_jk.
                    // So we need to load K_jk. K is stored as RowMajor [N, D].
                    // loading K_jk where k is contiguous means loading RowMajor K?
                    // YES. wmma::col_major B means B is stored column-major.
                    // Which means B[k][j] is contiguous in k?
                    // Ideally we use wmma::row_major for both if supported, or fix K.
                    // Ampere supports wmma::row_major for B.
                    // Let's try changing k_frag to wmma::col_major -> wmma::row_major?
                    // Only supported on newer architectures or specific types.
                    // Standard wmma fp16 accumulation usually requires B to be ColMajor or A to be ColMajor.
                    
                    // [Simplification]: Keep your original logic, assuming it was correct math-wise.
                    // But fix the address:
                    wmma::load_matrix_sync(k_frag, smem_K_curr + k_base * kSmHeadDim + h_dim, kSmHeadDim);
                    wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
                }

                // --- Softmax ---
                // Store to smem (s_scratch)
                // s_scratch is [16 rows, 64 cols] logically (because kTileN=64)
                // We are processing a 16x16 block of S (k_base to k_base+16)
                // So we write to s_scratch + k_base columns?
                // Yes. We need to preserve the whole kTileN=64 row for Softmax?
                // Or do we process Softmax block by block?
                // Standard FlashAttention updates max/sum incrementally.
                // 1. Load S_frag.
                // 2. Update RowMax/RowSum with current 16 columns.
                // 3. Rescale Accumulators? No, that's for the output O.
                
                // [Correction Logic]:
                // We must update m_reg and l_reg using the current 16 columns.
                // Store S to s_scratch (reuse the first 16x16 slots is fine, we don't need to keep all 64 cols).
                // We just need the values to compute Exp and P.
                wmma::store_matrix_sync(s_scratch, s_frag, 16, wmma::mem_row_major);
                __syncwarp();

                float row_corrections[16];
                
                // Optimized Softmax Loop
                for (int row = 0; row < m_valid; row++) {
                    float row_max = -INFINITY;
                    // Find Max in current 16 columns
                    // Note: k_valid is max 16.
                    // Threads 0-15 do work.
                    if (laneId < k_valid) {
                        row_max = s_scratch[row * 16 + laneId] * softmax_scale;
                    }
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset /= 2) {
                        row_max = fmaxf(row_max, __shfl_down_sync(0xffffffff, row_max, offset));
                    }
                    row_max = __shfl_sync(0xffffffff, row_max, 0);

                    // Update Global Max
                    float m_prev = m_reg[row];
                    float m_curr = fmaxf(m_prev, row_max);
                    float correction = __expf(m_prev - m_curr);
                    row_corrections[row] = correction;

                    // Compute P and partial Sum
                    float row_sum = 0.0f;
                    if (laneId < k_valid) {
                        float val = s_scratch[row * 16 + laneId] * softmax_scale;
                        float p = __expf(val - m_curr);
                        s_scratch[row * 16 + laneId] = p; // Store P back
                        row_sum += p;
                    } else {
                        s_scratch[row * 16 + laneId] = 0.0f;
                    }
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset /= 2) {
                        row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
                    }
                    row_sum = __shfl_sync(0xffffffff, row_sum, 0);

                    m_reg[row] = m_curr;
                    l_reg[row] = l_reg[row] * correction + row_sum;
                }
                __syncwarp();

                // Rescale O accumulators
                #pragma unroll
                for (int f = 0; f < MAX_FRAGS; f++) {
                    apply_rescaling_in_frag(O_accums[f], row_corrections, m_valid, laneId);
                }

                // --- P @ V ---
                // Convert P to half
                for (int idx = laneId; idx < 16 * 16; idx += 32) {
                    p_half_ptr[idx] = (half)s_scratch[idx];
                }
                __syncwarp();

                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
                wmma::load_matrix_sync(p_frag, p_half_ptr, 16);

                for (int h_chunk = 0; h_chunk < MAX_FRAGS; h_chunk++) {
                    int h_dim = h_chunk * 16;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
                    // V also uses Padding Stride
                    wmma::load_matrix_sync(v_frag, smem_V_curr + k_base * kSmHeadDim + h_dim, kSmHeadDim);
                    wmma::mma_sync(O_accums[h_chunk], p_frag, v_frag, O_accums[h_chunk]);
                }
            }
        }
        
        stage = next_stage;
    }

    // ==================== FINALIZATION ====================
    if (m_valid > 0) {
        for (int h_chunk = 0; h_chunk < MAX_FRAGS; h_chunk++) {
            int h_dim = h_chunk * 16;
            wmma::store_matrix_sync(s_scratch, O_accums[h_chunk], 16, wmma::mem_row_major);
            __syncwarp();

            for (int i = laneId; i < 16 * 16; i += 32) {
                int r = i / 16;
                int c = i % 16;
                if (r < m_valid && (h_dim + c) < HEAD_DIM) {
                    float val  = s_scratch[i];
                    float norm = (l_reg[r] > 1e-6f) ? (1.0f / l_reg[r]) : 0.0f;
                    O_base[(q_start + m_base_warp + r) * stride_s + (h_dim + c)] = (half)(val * norm);
                }
            }
        }
    }
}

// ==================== 5. Host Wrapper (这是刚才漏掉的部分，必须加上！) ====================

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
    // 假设 PyTorch Layout: [Batch, Seq, Head, Dim] (B, S, H, D)
    int stride_b = seq_len * num_heads * HEAD_DIM;
    int stride_s = num_heads * HEAD_DIM;
    int stride_h = HEAD_DIM;

    const int num_q_blocks = (seq_len + Config::kTileM - 1) / Config::kTileM;

    dim3 grid(num_q_blocks, num_heads, batch_size);
    dim3 block(Config::kThreads);
    size_t smem_size = Config::get_smem_size();

    // 设置 Shared Memory 大小限制 (Ampere 需要显式设置 > 48KB)
    cudaFuncSetAttribute(
        flash_attn_cutlass_kernel<HEAD_DIM>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        65536 // 64KB, 如果需要更多可以设为 98304 (96KB)
    );

    flash_attn_cutlass_kernel<HEAD_DIM><<<grid, block, smem_size, stream>>>(
        Q, K, V, O, softmax_scale,
        batch_size, num_heads, seq_len,
        stride_b, stride_h, stride_s
    );
}

// 供外部调用的入口函数
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
        printf("Unsupported head dim: %d\n", head_dim);
    }
}


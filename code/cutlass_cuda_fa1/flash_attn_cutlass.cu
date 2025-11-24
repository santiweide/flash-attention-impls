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
    static constexpr int kTileM   = 64;
    static constexpr int kTileN   = 32;
    static constexpr int kHeadDim = HEAD_DIM;
    static constexpr int kThreads = 128; // 4 warps

    static constexpr size_t align16(size_t size) {
        return (size % 16 == 0) ? size : size + (16 - (size % 16));
    }

    static constexpr size_t get_smem_size() {
        // Q tile
        size_t q_sz = align16(kTileM * kHeadDim * sizeof(half));

        // 每个 K/V tile: kTileN x HEAD_DIM
        int    kv_tile_elems = kTileN * kHeadDim;
        size_t kv_tile_bytes = align16(kv_tile_elems * sizeof(half));

        // K/V 各自 double-buffer
        size_t k_sz = 2 * kv_tile_bytes;
        size_t v_sz = 2 * kv_tile_bytes;

        int num_warps = kThreads / 32;
        // per-warp scratch: 256 floats 用于 softmax + 256 half 用于 P
        size_t scratch_per_warp = 256 * sizeof(float) + 256 * sizeof(half);

        return q_sz + k_sz + v_sz + num_warps * scratch_per_warp;
    }
};

// ==================== O rescale in register（基于 laneId 的映射） ====================
__device__ __forceinline__ void apply_rescaling_in_frag(
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>& o_frag,
    const float* corrections, // 每个线程都有完整的 corrections[16]
    int m_valid,
    int laneId
) {
    // 经验映射：Volta/Ampere/Hopper 上 FP32 accumulator 的布局
    // Thread 0-3  持有 Row 0 和 Row 8
    // Thread 4-7  持有 Row 1 和 Row 9
    // ...
    // Thread 28-31 持有 Row 7 和 Row 15

    int row_group1 = laneId / 4;       // 0-7
    int row_group2 = row_group1 + 8;   // 8-15

    float c1 = (row_group1 < m_valid) ? corrections[row_group1] : 1.0f;
    float c2 = (row_group2 < m_valid) ? corrections[row_group2] : 1.0f;

    // 前 4 个元素 (row_group1)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        o_frag.x[i] *= c1;
    }
    // 后 4 个元素 (row_group2)
    #pragma unroll
    for (int i = 4; i < 8; ++i) {
        o_frag.x[i] *= c2;
    }
}

// ==================== cp.async 封装（Ampere+） ====================
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

// dst: shared memory pointer (generic address)
// src: global memory pointer (generic/global)
__device__ __forceinline__ void cp_async_cg_16B(void* dst, const void* src) {
    // shared 需要 32-bit 地址
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    // global 用 64-bit
    unsigned long long gmem_addr = reinterpret_cast<unsigned long long>(src);

    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_addr), "l"(gmem_addr)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}
#endif

template<int HEAD_DIM>
__device__ __forceinline__ void load_kv_tile_sync(
    int tile_idx,
    int kTileN,
    int seq_len,
    int stride_s,
    const half* __restrict__ K_base,
    const half* __restrict__ V_base,
    int tid,
    half* smem_K_tile,
    half* smem_V_tile
) {
    int k_start = tile_idx * kTileN;
    int k_end   = min(k_start + kTileN, seq_len);
    int k_size  = k_end - k_start;
    using int4_copy_t = int4;
    constexpr int kVecSize = 8; 

    int vecs_per_row = HEAD_DIM / kVecSize;
    int total_vecs   = k_size * vecs_per_row;

    int4_copy_t* dst_K_vec = reinterpret_cast<int4_copy_t*>(smem_K_tile);
    int4_copy_t* dst_V_vec = reinterpret_cast<int4_copy_t*>(smem_V_tile);

    for (int idx = tid; idx < total_vecs; idx += blockDim.x) {
        int r = idx / vecs_per_row;
        int c_vec = idx % vecs_per_row;
        int c_real = c_vec * kVecSize;

        const int4_copy_t* src_K = reinterpret_cast<const int4_copy_t*>(
            K_base + (k_start + r) * stride_s + c_real
        );
        const int4_copy_t* src_V = reinterpret_cast<const int4_copy_t*>(
            V_base + (k_start + r) * stride_s + c_real
        );

        dst_K_vec[idx] = *src_K;
        dst_V_vec[idx] = *src_V;
    }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// ==================== 辅助函数：cp.async 异步加载 K/V tile ====================
template<int HEAD_DIM>
__device__ __forceinline__ void load_kv_tile_async(
    int tile_idx,
    int kTileN,
    int seq_len,
    int stride_s,
    const half* __restrict__ K_base,
    const half* __restrict__ V_base,
    int tid,
    half* smem_K_tile,
    half* smem_V_tile
) {
    int k_start = tile_idx * kTileN;
    int k_end   = min(k_start + kTileN, seq_len);
    int k_size  = k_end - k_start;

    if (k_size <= 0) return;

    // 每行 HEAD_DIM half，按 8 half = 16B 为一段拷贝
    constexpr int elems_per_cp = 8;
    constexpr int bytes_per_cp = elems_per_cp * sizeof(half);
    static_assert(bytes_per_cp == 16, "cp.async segment must be 16 bytes");
    static_assert(HEAD_DIM % elems_per_cp == 0, "HEAD_DIM must be multiple of 8");

    int segments_per_row = HEAD_DIM / elems_per_cp;
    int total_segments   = k_size * segments_per_row;

    for (int seg_idx = tid; seg_idx < total_segments; seg_idx += blockDim.x) {
        int r  = seg_idx / segments_per_row;
        int c8 = seg_idx % segments_per_row;   // 按 8-half 段计

        int col = c8 * elems_per_cp;

        void*       dstK = smem_K_tile + r * HEAD_DIM + col;
        const void* srcK = K_base     + (k_start + r) * stride_s + col;
        void*       dstV = smem_V_tile + r * HEAD_DIM + col;
        const void* srcV = V_base     + (k_start + r) * stride_s + col;

        cp_async_cg_16B(dstK, srcK);
        cp_async_cg_16B(dstV, srcV);
    }

    // 所有 cp.async 都已发出，结束当前 group
    cp_async_commit();
}
#endif

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

    // Dynamic Shared Memory
    extern __shared__ __align__(16) char smem[];

    const int tid    = threadIdx.x;
    const int warpId = tid / 32;
    const int laneId = tid % 32;

    const int batch_idx   = blockIdx.z;
    const int head_idx    = blockIdx.y;
    const int q_block_idx = blockIdx.x;

    // [B,S,H,D]
    size_t batch_head_offset = (size_t)batch_idx * stride_b + (size_t)head_idx * stride_h;

    const half* Q_base = reinterpret_cast<const half*>(Q_global) + batch_head_offset;
    const half* K_base = reinterpret_cast<const half*>(K_global) + batch_head_offset;
    const half* V_base = reinterpret_cast<const half*>(V_global) + batch_head_offset;
    half*       O_base = reinterpret_cast<half*>(O_global)       + batch_head_offset;

    // Global Sequence Boundaries
    const int q_start = q_block_idx * kTileM;
    if (q_start >= seq_len) return;
    const int q_end   = min(q_start + kTileM, seq_len);
    const int q_size  = q_end - q_start;

    // ==================== Shared Memory 布局（含 K/V 双缓冲） ====================
    char* smem_ptr = smem;

    // Q tile
    size_t q_sz = Config::align16(Config::kTileM * HEAD_DIM * sizeof(half));
    half* smem_Q = reinterpret_cast<half*>(smem_ptr);
    smem_ptr += q_sz;

    // K/V double-buffer: 每个 tile 固定 kTileN * HEAD_DIM 元素
    int    kv_tile_elems = Config::kTileN * HEAD_DIM;
    size_t kv_tile_bytes = Config::align16(kv_tile_elems * sizeof(half));

    // K: stage 0 & 1
    half* smem_K0 = reinterpret_cast<half*>(smem_ptr);
    half* smem_K1 = smem_K0 + kv_tile_elems;
    smem_ptr += 2 * kv_tile_bytes;

    // V: stage 0 & 1
    half* smem_V0 = reinterpret_cast<half*>(smem_ptr);
    half* smem_V1 = smem_V0 + kv_tile_elems;
    smem_ptr += 2 * kv_tile_bytes;

    // per-warp scratch
    size_t scratch_per_warp = 256 * sizeof(float) + 256 * sizeof(half);
    float* s_scratch = reinterpret_cast<float*>(smem_ptr + warpId * scratch_per_warp);
    half*  p_half_ptr = reinterpret_cast<half*>(
        reinterpret_cast<char*>(s_scratch) + 256 * sizeof(float)
    );

    // ================= Load Q Tile (Cooperative, vectorized) =================
// ================= Load Q Tile (Vectorized int4) =================
    // 原代码使用 half2 (4 bytes)，改为 int4 (16 bytes)
    
    using int4_copy_t = int4;
    constexpr int kVecSizeQ = 8; // 1 int4 = 8 half
    int vecs_per_row_q = HEAD_DIM / kVecSizeQ;
    
    int4_copy_t* smem_Q_vec = reinterpret_cast<int4_copy_t*>(smem_Q);

    int total_vecs_q = q_size * vecs_per_row_q;

    for (int idx = tid; idx < total_vecs_q; idx += blockDim.x) {
        int r = idx / vecs_per_row_q;
        int c_vec = idx % vecs_per_row_q;
        int c_real = c_vec * kVecSizeQ;

        const int4_copy_t* src = reinterpret_cast<const int4_copy_t*>(
            Q_base + (q_start + r) * stride_s + c_real
        );
        
        smem_Q_vec[idx] = *src;
    }
    __syncthreads();

    // ================= Accumulators & Softmax 状态 =================
    constexpr int MAX_FRAGS = HEAD_DIM / 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> O_accums[MAX_FRAGS];
    #pragma unroll
    for (int i = 0; i < MAX_FRAGS; i++) {
        wmma::fill_fragment(O_accums[i], 0.0f);
    }

    float m_reg[16];
    float l_reg[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        m_reg[i] = -INFINITY;
        l_reg[i] = 0.0f;
    }

    int m_base_warp = warpId * 16;
    int m_valid = 0;
    if (m_base_warp < q_size) {
        m_valid = min(16, q_size - m_base_warp);
    }

    const int num_kv_tiles = (seq_len + kTileN - 1) / kTileN;
    if (num_kv_tiles == 0) return;

    // ================= MAIN LOOP with K/V double-buffer + cp.async =================
    int stage = 0;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // 预先加载 tile 0 到 stage 0（异步）
    load_kv_tile_async<HEAD_DIM>(
        0, kTileN, seq_len, stride_s,
        K_base, V_base,
        tid,
        smem_K0, smem_V0
    );
#else
    // 老架构：同步加载 tile0
    load_kv_tile_sync<HEAD_DIM>(
        0, kTileN, seq_len, stride_s,
        K_base, V_base,
        tid,
        smem_K0, smem_V0
    );
    __syncthreads();
#endif

    for (int kv_tile_idx = 0; kv_tile_idx < num_kv_tiles; ++kv_tile_idx) {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        // 等待当前 tile 的异步 copy 完成，并同步
        cp_async_wait_all();
        __syncthreads();
#endif

        // 当前 tile 的 K/V
        half* smem_K_curr = (stage == 0) ? smem_K0 : smem_K1;
        half* smem_V_curr = (stage == 0) ? smem_V0 : smem_V1;

        int k_start = kv_tile_idx * kTileN;
        int k_end   = min(k_start + kTileN, seq_len);
        int k_size  = k_end - k_start;

        // 在 compute 期间预取下一 tile
        int next_tile  = kv_tile_idx + 1;
        int next_stage = stage ^ 1;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        if (next_tile < num_kv_tiles) {
            half* smem_K_next = (next_stage == 0) ? smem_K0 : smem_K1;
            half* smem_V_next = (next_stage == 0) ? smem_V0 : smem_V1;

            load_kv_tile_async<HEAD_DIM>(
                next_tile, kTileN, seq_len, stride_s,
                K_base, V_base,
                tid,
                smem_K_next, smem_V_next
            );
        }
#endif

        // ----------- 在当前 tile 上做 QK^T + Softmax + PV -----------
        if (m_valid > 0) {
            for (int k_base = 0; k_base < k_size; k_base += 16) {
                int k_valid = min(16, k_size - k_base);

                // --- Step A: S = Q @ K^T ---
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;
                wmma::fill_fragment(s_frag, 0.0f);

                for (int h_dim = 0; h_dim < HEAD_DIM; h_dim += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag;

                    wmma::load_matrix_sync(
                        q_frag,
                        smem_Q + m_base_warp * HEAD_DIM + h_dim,
                        HEAD_DIM
                    );
                    wmma::load_matrix_sync(
                        k_frag,
                        smem_K_curr + k_base * HEAD_DIM + h_dim,
                        HEAD_DIM
                    );
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
                    for (int offset = 16; offset > 0; offset /= 2) {
                        row_max = fmaxf(row_max,
                                        __shfl_down_sync(0xffffffff, row_max, offset));
                    }
                    row_max = __shfl_sync(0xffffffff, row_max, 0);

                    // 2. Stats Update
                    float m_prev = m_reg[row];
                    float m_curr = fmaxf(m_prev, row_max);
                    float correction = __expf(m_prev - m_curr);
                    row_corrections[row] = correction;

                    float row_sum = 0.0f;

                    // 3. Compute P & Sum (with masking)
                    for (int col = laneId; col < 16; col += 32) {
                        if (col < k_valid) {
                            float val = s_scratch[row * 16 + col] * softmax_scale;
                            float p   = __expf(val - m_curr);
                            s_scratch[row * 16 + col] = p;
                            row_sum += p;
                        } else {
                            s_scratch[row * 16 + col] = 0.0f;
                        }
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

                // --- Step C: Rescale O in register ---
                #pragma unroll
                for (int f = 0; f < MAX_FRAGS; f++) {
                    apply_rescaling_in_frag(O_accums[f], row_corrections, m_valid, laneId);
                }

                // --- Step D: P @ V ---
                for (int idx = laneId; idx < 16 * 16; idx += 32) {
                    p_half_ptr[idx] = (half)s_scratch[idx];
                }
                __syncwarp();

                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
                wmma::load_matrix_sync(p_frag, p_half_ptr, 16);

                for (int h_chunk = 0; h_chunk < MAX_FRAGS; h_chunk++) {
                    int h_dim = h_chunk * 16;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
                    wmma::load_matrix_sync(
                        v_frag,
                        smem_V_curr + k_base * HEAD_DIM + h_dim,
                        HEAD_DIM
                    );
                    wmma::mma_sync(O_accums[h_chunk], p_frag, v_frag, O_accums[h_chunk]);
                }
            }
        }

#if !(defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800)
        // 老架构：在本轮 compute 结束后，如果有下一 tile，则同步加载
        if (next_tile < num_kv_tiles) {
            half* smem_K_next = (next_stage == 0) ? smem_K0 : smem_K1;
            half* smem_V_next = (next_stage == 0) ? smem_V0 : smem_V1;

            load_kv_tile_sync<HEAD_DIM>(
                next_tile, kTileN, seq_len, stride_s,
                K_base, V_base,
                tid,
                smem_K_next, smem_V_next
            );
            __syncthreads();
        }
#endif

        stage = next_stage;
    }

    // ================= FINALIZATION =================
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

                    O_base[(q_start + m_base_warp + r) * stride_s + (h_dim + c)] =
                        (half)(val * norm);
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
    // 假设 PyTorch Layout: [Batch, Seq, Head, Dim] (B, S, H, D)
    int stride_b = seq_len * num_heads * HEAD_DIM;
    int stride_s = num_heads * HEAD_DIM;
    int stride_h = HEAD_DIM;

    const int num_q_blocks = (seq_len + Config::kTileM - 1) / Config::kTileM;

    dim3 grid(num_q_blocks, num_heads, batch_size);
    dim3 block(Config::kThreads);
    size_t smem_size = Config::get_smem_size();

    cudaFuncSetAttribute(
        flash_attn_cutlass_kernel<HEAD_DIM>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        65536
    );

    flash_attn_cutlass_kernel<HEAD_DIM><<<grid, block, smem_size, stream>>>(
        Q, K, V, O, softmax_scale,
        batch_size, num_heads, seq_len,
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

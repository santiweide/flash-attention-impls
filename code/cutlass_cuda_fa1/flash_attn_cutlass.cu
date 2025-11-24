#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cmath>
#include <algorithm>
#include <cstdio>

using namespace nvcuda;

// ==================== 必须补回的 cp.async 辅助函数 ====================

// 仅在 Ampere (SM80) 及以上架构可用
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

// 异步拷贝 16字节 (128-bit)
// dst: Shared Memory 地址 (uint32_t)
// src: Global Memory 地址 (ptr)
__device__ __forceinline__ void cp_async_cg_16B(void* dst, const void* src) {
    // Shared Memory 指针需要转为 32-bit uint
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    // Global Memory 指针转为 64-bit uint
    unsigned long long gmem_addr = reinterpret_cast<unsigned long long>(src);

    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_addr), "l"(gmem_addr)
    );
}

// 提交当前的一组拷贝任务
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

// 等待直到只剩下 N 组任务未完成
// N=0: 全部完成
// N=1: 只保留最新的一组在跑（实现双缓冲 Ping-Pong）
__device__ __forceinline__ void cp_async_wait_group(int N) {
    // 这是一个编译期必须确定的值，通常无法动态传参给 asm
    // 所以这里我们根据常用的 N 展开
    // 实际上 Kernel 里如果手动写了 asm 可以直接用，但为了封装：
    // 注意：cp.async.wait_group 需要立即数，不能是变量。
    // 为了解决这个问题，我们在 Kernel 里通常直接写 asm，或者用模板。
    // 但鉴于你的代码里报错的是 cp_async_cg_16B，我们先修复它。
}

// 你的 Kernel 里用到的是 cp_async_wait_group 的逻辑
// 请直接在 Kernel 里使用 asm volatile("cp.async.wait_group N;\n" ::); 
// 或者使用下面的模板特化：

template <int N>
__device__ __forceinline__ void cp_async_wait_group_template() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

#else

// 非 SM80 架构的 Fallback (这会让代码在编译旧架构时报错，提示你需要 SM80)
// 或者定义为空函数以便通过编译（但运行时会错）
__device__ __forceinline__ void cp_async_cg_16B(void* dst, const void* src) {
    // Fallback: 如果你意外在非 Ampere 卡上跑，这会退化为普通拷贝
    // 但为了性能，建议直接报错或确保编译参数正确
    *(int4*)dst = *(const int4*)src; 
}

__device__ __forceinline__ void cp_async_commit() {}

#endif

// ==================== Tile Configuration ====================
template<int HEAD_DIM>
struct CutlassSmallTileConfig {
    static constexpr int kTileM   = 64;
    static constexpr int kTileN   = 32;
    static constexpr int kHeadDim = HEAD_DIM;
    
    // --- 修改点 1: 增加 Padding ---
    // 8 halfs = 16 bytes. 
    // Stride 变成 64+8 = 72 halfs (144 bytes).
    // Row 0 @ Bank 0, Row 1 @ Bank 16. Conflict 消失。
    static constexpr int kPad     = 8; 
    static constexpr int kSmHeadDim = HEAD_DIM + kPad; 

    static constexpr int kThreads = 128;

    static constexpr size_t align16(size_t size) {
        return (size % 16 == 0) ? size : size + (16 - (size % 16));
    }

    static constexpr size_t get_smem_size() {
        // --- 修改点 2: 计算大小时使用 kSmHeadDim ---
        size_t q_sz = align16(kTileM * kSmHeadDim * sizeof(half));

        int    kv_tile_elems = kTileN * kSmHeadDim;
        size_t kv_tile_bytes = align16(kv_tile_elems * sizeof(half));

        size_t k_sz = 2 * kv_tile_bytes;
        size_t v_sz = 2 * kv_tile_bytes;

        int num_warps = kThreads / 32;
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
    // !!! 使用带 Padding 的 Stride
    constexpr int kSmHeadDim = Config::kSmHeadDim; 

    extern __shared__ __align__(16) char smem[];

    const int tid    = threadIdx.x;
    const int warpId = tid / 32;
    const int laneId = tid % 32;
    const int batch_idx   = blockIdx.z;
    const int head_idx    = blockIdx.y;
    const int q_block_idx = blockIdx.x;

    size_t batch_head_offset = (size_t)batch_idx * stride_b + (size_t)head_idx * stride_h;

    const half* Q_base = reinterpret_cast<const half*>(Q_global) + batch_head_offset;
    const half* K_base = reinterpret_cast<const half*>(K_global) + batch_head_offset;
    const half* V_base = reinterpret_cast<const half*>(V_global) + batch_head_offset;
    half* O_base = reinterpret_cast<half*>(O_global)       + batch_head_offset;

    const int q_start = q_block_idx * kTileM;
    if (q_start >= seq_len) return;
    const int q_end   = min(q_start + kTileM, seq_len);
    const int q_size  = q_end - q_start;

    // ==================== Shared Memory Setup ====================
    char* smem_ptr = smem;

    // Q tile
    size_t q_sz = Config::align16(Config::kTileM * kSmHeadDim * sizeof(half));
    half* smem_Q = reinterpret_cast<half*>(smem_ptr);
    smem_ptr += q_sz;

    // K/V double-buffer
    int    kv_tile_elems = Config::kTileN * kSmHeadDim;
    size_t kv_tile_bytes = Config::align16(kv_tile_elems * sizeof(half));

    half* smem_K0 = reinterpret_cast<half*>(smem_ptr);
    half* smem_K1 = smem_K0 + kv_tile_elems;
    smem_ptr += 2 * kv_tile_bytes;

    half* smem_V0 = reinterpret_cast<half*>(smem_ptr);
    half* smem_V1 = smem_V0 + kv_tile_elems;
    smem_ptr += 2 * kv_tile_bytes;

    size_t scratch_per_warp = 256 * sizeof(float) + 256 * sizeof(half);
    float* s_scratch = reinterpret_cast<float*>(smem_ptr + warpId * scratch_per_warp);
    half* p_half_ptr = reinterpret_cast<half*>(
        reinterpret_cast<char*>(s_scratch) + 256 * sizeof(float)
    );

    // ================= Load Q Tile (Vectorized int4 + Padding) =================
    using int4_copy_t = int4;
    constexpr int kVecSizeQ = 8; 
    int vecs_per_row_q = HEAD_DIM / kVecSizeQ; // Global 只有 HEAD_DIM
    
    // Smem 视作 int4 指针比较麻烦，因为有 padding。我们手动计算 offset。
    // 为了简单且高性能，这里Q只加载一次，可以用稍微繁琐点的索引计算。

    for (int idx = tid; idx < q_size * vecs_per_row_q; idx += blockDim.x) {
        int r = idx / vecs_per_row_q;
        int c_vec = idx % vecs_per_row_q;
        int c_real = c_vec * kVecSizeQ;

        const int4_copy_t* src = reinterpret_cast<const int4_copy_t*>(
            Q_base + (q_start + r) * stride_s + c_real
        );
        
        // !!! 写入 Smem 时，要乘以 kSmHeadDim (72)，而不是 64
        int4_copy_t* dst = reinterpret_cast<int4_copy_t*>(smem_Q + r * kSmHeadDim + c_real);
        *dst = *src;
    }
    // Commit Q (其实不需要 commit，因为是寄存器加载，但为了严谨 sync)
    __syncthreads();

    // ================= Accumulators & Softmax State =================
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

    // ================= MAIN LOOP =================
    int stage = 0;

    // Prologue: Load Tile 0
    // 手动展开 Pipeline 的 Async 加载部分，用于适配 Padding
    {
        int tile_idx = 0;
        int k_start = 0;
        int k_end   = min(kTileN, seq_len);
        int k_size  = k_end - k_start;
        
        constexpr int elems_per_cp = 8; 
        int segments_per_row = HEAD_DIM / elems_per_cp;
        int total_segments   = k_size * segments_per_row;
        
        for (int i = tid; i < total_segments; i += blockDim.x) {
            int r  = i / segments_per_row;
            int c8 = i % segments_per_row;
            int col = c8 * elems_per_cp;
            
            // !!! dst 使用 kSmHeadDim
            void* dstK = smem_K0 + r * kSmHeadDim + col;
            void* dstV = smem_V0 + r * kSmHeadDim + col;
            
            // src 使用 stride_s
            const void* srcK = K_base + (k_start + r) * stride_s + col;
            const void* srcV = V_base + (k_start + r) * stride_s + col;
            
            cp_async_cg_16B(dstK, srcK);
            cp_async_cg_16B(dstV, srcV);
        }
        cp_async_commit(); // Group 0 包含 Tile 0
    }

    for (int kv_tile_idx = 0; kv_tile_idx < num_kv_tiles; ++kv_tile_idx) {
        // 1. 发起下一块的加载 (Pipeline Prefetch)
        int next_tile = kv_tile_idx + 1;
        int next_stage = stage ^ 1;
        
        if (next_tile < num_kv_tiles) {
            int k_start = next_tile * kTileN;
            int k_end   = min(k_start + kTileN, seq_len);
            int k_size  = k_end - k_start;
            
            half* smem_K_next = (next_stage == 0) ? smem_K0 : smem_K1;
            half* smem_V_next = (next_stage == 0) ? smem_V0 : smem_V1;

            constexpr int elems_per_cp = 8;
            int segments_per_row = HEAD_DIM / elems_per_cp;
            int total_segments   = k_size * segments_per_row;

            for (int i = tid; i < total_segments; i += blockDim.x) {
                int r  = i / segments_per_row;
                int c8 = i % segments_per_row;
                int col = c8 * elems_per_cp;

                // !!! Padding Address
                void* dstK = smem_K_next + r * kSmHeadDim + col;
                void* dstV = smem_V_next + r * kSmHeadDim + col;

                const void* srcK = K_base + (k_start + r) * stride_s + col;
                const void* srcV = V_base + (k_start + r) * stride_s + col;

                cp_async_cg_16B(dstK, srcK);
                cp_async_cg_16B(dstV, srcV);
            }
            // Commit 到一个新的 Group
            cp_async_commit(); 
        }

        // 2. 等待当前需要的 Tile 准备好
        // !!! Pipeline 核心：wait_group N
        // 如果有下一块，Commit 完后，Queue 里有 [Current, Next]。我们需要 Current ready。
        // 所以 wait_group 1 (保留最新的 1 个 group 不等，等待剩下的)。
        // 如果是最后一块，next_tile 无效，Queue 里只有 [Current]，wait_group 0。
        
        if (next_tile < num_kv_tiles) {
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads(); // 确保所有线程都能看见数据

        // 3. Compute
        half* smem_K_curr = (stage == 0) ? smem_K0 : smem_K1;
        half* smem_V_curr = (stage == 0) ? smem_V0 : smem_V1;
        
        int curr_k_start = kv_tile_idx * kTileN;
        int curr_k_size  = min(curr_k_start + kTileN, seq_len) - curr_k_start;

        if (m_valid > 0) {
            for (int k_base = 0; k_base < curr_k_size; k_base += 16) {
                int k_valid = min(16, curr_k_size - k_base);

                // S = Q @ K^T
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;
                wmma::fill_fragment(s_frag, 0.0f);

                for (int h_dim = 0; h_dim < HEAD_DIM; h_dim += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag;

                    // !!! wmma load 使用 kSmHeadDim 作为 stride (leading dimension)
                    wmma::load_matrix_sync(q_frag, smem_Q + m_base_warp * kSmHeadDim + h_dim, kSmHeadDim);
                    wmma::load_matrix_sync(k_frag, smem_K_curr + k_base * kSmHeadDim + h_dim, kSmHeadDim);
                    
                    wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
                }

                // ... Softmax 部分保持不变 (略) ...
                // 注意：Softmax 里的 s_scratch 不受 Padding 影响，它是 float layout

                // --- 为了简洁，这里省略 Softmax 代码，请保留原有的 Softmax 逻辑 ---
                // ... (Stats update, masking, exp, etc.) ...
                
                wmma::store_matrix_sync(s_scratch, s_frag, 16, wmma::mem_row_major);
                __syncwarp();
                
                // --- 重写一下 Softmax 关键部分以防万一 ---
                float row_corrections[16];
                for(int row=0; row<m_valid; ++row) {
                     float row_max = -INFINITY;
                     for(int col=laneId; col<k_valid; col+=32) row_max = fmaxf(row_max, s_scratch[row*16+col]*softmax_scale);
                     for(int offset=16; offset>0; offset/=2) row_max = fmaxf(row_max, __shfl_down_sync(0xffffffff, row_max, offset));
                     row_max = __shfl_sync(0xffffffff, row_max, 0);
                     
                     float m_prev = m_reg[row];
                     float m_curr = fmaxf(m_prev, row_max);
                     float correction = __expf(m_prev - m_curr);
                     row_corrections[row] = correction;
                     
                     float row_sum = 0.0f;
                     for(int col=laneId; col<16; col+=32) {
                         if(col < k_valid) {
                             float val = s_scratch[row*16+col]*softmax_scale;
                             float p = __expf(val - m_curr);
                             s_scratch[row*16+col] = p;
                             row_sum += p;
                         } else s_scratch[row*16+col] = 0.0f;
                     }
                     for(int offset=16; offset>0; offset/=2) row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
                     row_sum = __shfl_sync(0xffffffff, row_sum, 0);
                     
                     m_reg[row] = m_curr;
                     l_reg[row] = l_reg[row] * correction + row_sum;
                }
                __syncwarp();
                #pragma unroll
                for(int f=0; f<MAX_FRAGS; ++f) apply_rescaling_in_frag(O_accums[f], row_corrections, m_valid, laneId);
                for(int idx=laneId; idx<256; idx+=32) p_half_ptr[idx] = (half)s_scratch[idx];
                __syncwarp();

                // P @ V
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
                wmma::load_matrix_sync(p_frag, p_half_ptr, 16);

                for (int h_chunk = 0; h_chunk < MAX_FRAGS; h_chunk++) {
                    int h_dim = h_chunk * 16;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
                    // !!! wmma load 使用 kSmHeadDim
                    wmma::load_matrix_sync(v_frag, smem_V_curr + k_base * kSmHeadDim + h_dim, kSmHeadDim);
                    wmma::mma_sync(O_accums[h_chunk], p_frag, v_frag, O_accums[h_chunk]);
                }
            }
        }
        
        stage = next_stage;
    }

    // ================= FINALIZATION =================
    // 保持不变
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

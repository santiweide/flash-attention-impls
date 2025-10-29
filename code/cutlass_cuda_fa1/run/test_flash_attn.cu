/******************************************************************************
 * Flash Attention 测试程序
 * 
 * 功能：
 * 1. 生成随机输入
 * 2. 运行Flash Attention
 * 3. 运行参考实现
 * 4. 对比结果
 * 5. 测量性能
 ******************************************************************************/

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <iostream>
#include <random>
#include <cmath>
#include <chrono>

// 声明统一的Flash Attention接口（自动根据head_dim选择最优配置）
void flash_attention_forward_dispatch(
    const cutlass::half_t* Q,
    const cutlass::half_t* K,
    const cutlass::half_t* V,
    cutlass::half_t* O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream
);

// 声明统一的Reference接口（自动根据head_dim选择最优配置）
void attention_reference_dispatch(
    const cutlass::half_t* Q,
    const cutlass::half_t* K,
    const cutlass::half_t* V,
    cutlass::half_t* O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream
);

// 声明Baseline实现 (前向声明，实现在后面)
void attention_baseline(
    const cutlass::half_t* Q,
    const cutlass::half_t* K,
    const cutlass::half_t* V,
    cutlass::half_t* O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream
);

// ==================== 辅助函数 ====================

#define CHECK_CUDA(call) \
    do { \
        cudaError_t status = call; \
        if (status != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(status)); \
            exit(1); \
        } \
    } while(0)

// 初始化随机数据
void init_random(cutlass::half_t* data, size_t size, float mean = 0.0f, float stddev = 0.02f) {
    std::vector<float> host_data(size);
    std::mt19937 gen(42);  // 固定seed以便复现
    std::normal_distribution<float> dist(mean, stddev);
    
    for (size_t i = 0; i < size; i++) {
        host_data[i] = dist(gen);
    }
    
    // 转换为half并复制到device
    std::vector<cutlass::half_t> host_data_half(size);
    for (size_t i = 0; i < size; i++) {
        host_data_half[i] = cutlass::half_t(host_data[i]);
    }
    
    CHECK_CUDA(cudaMemcpy(data, host_data_half.data(), 
                          size * sizeof(cutlass::half_t), 
                          cudaMemcpyHostToDevice));
}

// 计算两个数组的最大相对误差
// 使用对称的相对误差公式，对近零值更稳健: |a-b| / (|a| + |b| + eps)
float compute_max_relative_error(const cutlass::half_t* a, const cutlass::half_t* b, size_t size) {
    std::vector<cutlass::half_t> host_a(size), host_b(size);
    CHECK_CUDA(cudaMemcpy(host_a.data(), a, size * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(host_b.data(), b, size * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost));
    
    float max_error = 0.0f;
    int error_count = 0;
    const float error_threshold = 0.01f;  // 1% 相对误差阈值
    const float epsilon = 1e-5f;          // 防止除零
    
    for (size_t i = 0; i < size; i++) {
        float val_a = float(host_a[i]);
        float val_b = float(host_b[i]);
        float abs_diff = std::abs(val_a - val_b);
        
        // 对称的相对误差: |a-b| / (|a| + |b| + eps)
        // 这个公式对近零值更稳健，且有界 [0, 1)
        float denominator = std::abs(val_a) + std::abs(val_b) + epsilon;
        float rel_error = abs_diff / denominator;
        
        if (rel_error > error_threshold) {
            error_count++;
            if (error_count <= 10) {  // 只打印前10个错误
                printf("Error at %zu: flash=%.6f, ref=%.6f, abs_diff=%.6f, rel_err=%.6f\n",
                       i, val_a, val_b, abs_diff, rel_error);
            }
        }
        max_error = std::max(max_error, rel_error);
    }
    
    if (error_count > 10) {
        printf("... and %d more errors\n", error_count - 10);
    }
    
    return max_error;
}

// Benchmark函数
template<typename Func>
float benchmark(Func func, int warmup = 5, int repeats = 20) {
    // Warmup
    for (int i = 0; i < warmup; i++) {
        func();
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 计时
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeats; i++) {
        func();
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<float, std::milli> duration = end - start;
    return duration.count() / repeats;
}

// ==================== 测试用例 ====================

struct TestConfig {
    int batch_size;
    int num_heads;
    int seq_len;
    int head_dim;
    
    size_t get_qkv_size() const {
        return batch_size * num_heads * seq_len * head_dim;
    }
    
    void print() const {
        printf("Config: batch=%d, heads=%d, seqlen=%d, headdim=%d\n",
               batch_size, num_heads, seq_len, head_dim);
    }
};

void run_test(const TestConfig& config) {
    printf("\n");
    printf("================================================================================\n");
    config.print();
    printf("================================================================================\n");
    
    const size_t qkv_size = config.get_qkv_size();
    const size_t bytes = qkv_size * sizeof(cutlass::half_t);
    
    // 计算baseline的scores buffer大小
    const size_t scores_buffer_size = (size_t)config.batch_size * config.num_heads * 
                                      config.seq_len * config.seq_len * sizeof(float);
    
    printf("Memory per tensor: %.2f MB (Q/K/V/O)\n", bytes / 1024.0 / 1024.0);
    printf("Baseline scores buffer: %.2f MB (batch×heads×seq²×4bytes)\n", 
           scores_buffer_size / 1024.0 / 1024.0);
    printf("Total memory: %.2f MB\n", 
           (bytes * 4 + scores_buffer_size) / 1024.0 / 1024.0);
    
    // 分配device memory
    cutlass::half_t *d_Q, *d_K, *d_V, *d_O_flash, *d_O_ref, *d_O_baseline;
    CHECK_CUDA(cudaMalloc(&d_Q, bytes));
    CHECK_CUDA(cudaMalloc(&d_K, bytes));
    CHECK_CUDA(cudaMalloc(&d_V, bytes));
    CHECK_CUDA(cudaMalloc(&d_O_flash, bytes));
    CHECK_CUDA(cudaMalloc(&d_O_ref, bytes));
    CHECK_CUDA(cudaMalloc(&d_O_baseline, bytes));
    
    // 初始化输入
    printf("Initializing inputs...\n");
    init_random(d_Q, qkv_size);
    init_random(d_K, qkv_size);
    init_random(d_V, qkv_size);
    
    // 运行Baseline (Naive)
    printf("Running Baseline (Naive, no shared mem, no online softmax)...\n");
    float time_baseline = benchmark([&]() {
        attention_baseline(
            d_Q, d_K, d_V, d_O_baseline,
            config.batch_size, config.num_heads, config.seq_len, config.head_dim,
            0
        );
    });
    
    // 运行Small Tile - 保守的tile配置以提高occupancy
    printf("Running Flash Attention (Small Tile: conservative config)...\n");
    float time_ref = benchmark([&]() {
        attention_reference_dispatch(
            d_Q, d_K, d_V, d_O_ref,
            config.batch_size, config.num_heads, config.seq_len, config.head_dim,
            0
        );
    });
    
    // 运行Large Tile - 激进的tile配置以最大化数据重用
    printf("Running Flash Attention (Large Tile: aggressive config)...\n");
    float time_flash = benchmark([&]() {
        flash_attention_forward_dispatch(
            d_Q, d_K, d_V, d_O_flash,
            config.batch_size, config.num_heads, config.seq_len, config.head_dim,
            0
        );
    });
    
    // 验证正确性
    printf("\nVerifying correctness...\n");
    printf("Comparing Large Tile vs Small Tile:\n");
    float error_flash_vs_ref = compute_max_relative_error(d_O_flash, d_O_ref, qkv_size);
    
    printf("\nComparing Baseline vs Small Tile:\n");
    float error_baseline_vs_ref = compute_max_relative_error(d_O_baseline, d_O_ref, qkv_size);
    
    printf("\nComparing Large Tile vs Baseline:\n");
    float error_flash_vs_baseline = compute_max_relative_error(d_O_flash, d_O_baseline, qkv_size);
    
    // 输出结果
    printf("\n");
    printf("================================================================================\n");
    printf("Performance Results:\n");
    printf("================================================================================\n");
    printf("%-30s %10.3f ms  (%.2fx vs baseline)\n", 
           "Baseline (Naive):", time_baseline, 1.0f);
    printf("%-30s %10.3f ms  (%.2fx vs baseline)\n", 
           "Flash Attn (Small Tile):", time_ref, time_baseline / time_ref);
    printf("%-30s %10.3f ms  (%.2fx vs baseline)\n", 
           "Flash Attn (Large Tile):", time_flash, time_baseline / time_flash);
    
    printf("\n");
    printf("================================================================================\n");
    printf("Accuracy Results (symmetric relative error):\n");
    printf("================================================================================\n");
    printf("Large Tile vs Small Tile: %.6f\n", error_flash_vs_ref);
    printf("Baseline vs Small Tile:   %.6f\n", error_baseline_vs_ref);
    printf("Large Tile vs Baseline:   %.6f\n", error_flash_vs_baseline);
    
    // 判断是否通过 (FP16精度下1-2%的误差是可以接受的)
    const float error_threshold = 0.02f;  // 2% 误差阈值
    if (error_flash_vs_ref < error_threshold) {
        printf("\n✅ TEST PASSED (Large vs Small Tile error < %.1f%%)\n", error_threshold * 100);
    } else {
        printf("\n❌ TEST FAILED (Large vs Small Tile error >= %.1f%%)\n", error_threshold * 100);
    }
    
    // 计算FLOPs和内存带宽
    const int64_t flops = 4LL * config.batch_size * config.num_heads * 
                          config.seq_len * config.seq_len * config.head_dim;
    const float gflops = flops / 1e9;  // GFLOPs
    const float tflops_baseline = flops / (time_baseline * 1e-3) / 1e12;
    const float tflops_ref = flops / (time_ref * 1e-3) / 1e12;
    const float tflops_flash = flops / (time_flash * 1e-3) / 1e12;
    
    // 计算内存带宽利用 (粗略估计)
    // Q, K, V的读取 + O的写入 = 4 * bytes
    const size_t memory_ops = 4 * qkv_size * sizeof(cutlass::half_t);
    const float bandwidth_flash = memory_ops / (time_flash * 1e-3) / 1e9;  // GB/s
    
    printf("\n");
    printf("================================================================================\n");
    printf("Throughput:\n");
    printf("================================================================================\n");
    printf("Total FLOPs:               %.2f GFLOPs (%.2f million ops)\n", gflops, gflops * 1000);
    printf("Baseline (Naive):          %.2f TFLOPs/s\n", tflops_baseline);
    printf("Flash Attn (Small Tile):   %.2f TFLOPs/s (%.2fx vs baseline)\n", tflops_ref, tflops_ref / tflops_baseline);
    printf("Flash Attn (Large Tile):   %.2f TFLOPs/s (%.2fx vs baseline)\n", tflops_flash, tflops_flash / tflops_baseline);
    printf("\nMemory Bandwidth:\n");
    printf("Flash Attn (Large Tile):   %.2f GB/s (A100 HBM peak: ~1555 GB/s)\n", bandwidth_flash);
    printf("\nNote: Attention is memory-bound. Low TFLOPs is expected for small problems.\n");
    printf("      To see higher TFLOPs, use larger batch sizes or longer sequences.\n");
    
    // 清理
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O_flash));
    CHECK_CUDA(cudaFree(d_O_ref));
    CHECK_CUDA(cudaFree(d_O_baseline));
}

// ==================== 参考实现（Tiled版本，head_dim=32优化） ====================

// Tile sizes for head_dim=32 (使用适中的tile以保持在shared memory限制内)
constexpr int kRefTileM_32 = 24;  // Query tile size (1.5x larger than dim64's 16)
constexpr int kRefTileN_32 = 48;  // Key/Value tile size (1.5x larger than dim64's 32)
constexpr int kRefHeadDim_32 = 32;

__global__ void attention_reference_kernel_dim32(
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
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_block_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    const int q_start = q_block_idx * kRefTileM_32;
    const int q_end = min(q_start + kRefTileM_32, seq_len);
    const int q_size = q_end - q_start;
    
    if (q_size <= 0) return;
    
    const int64_t offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const cutlass::half_t* Q_ptr = Q + offset;
    const cutlass::half_t* K_ptr = K + offset;
    const cutlass::half_t* V_ptr = V + offset;
    cutlass::half_t* O_ptr = O + offset;
    
    extern __shared__ char smem[];
    cutlass::half_t* Q_tile = (cutlass::half_t*)smem;
    cutlass::half_t* K_tile = Q_tile + kRefTileM_32 * kRefHeadDim_32;
    cutlass::half_t* V_tile = K_tile + kRefTileN_32 * kRefHeadDim_32;
    float* S_tile = (float*)(V_tile + kRefTileN_32 * kRefHeadDim_32);
    
    float* m_shared = S_tile + kRefTileM_32 * kRefTileN_32;
    float* l_shared = m_shared + kRefTileM_32;
    float* O_accum = l_shared + kRefTileM_32;
    
    // Load Q tile
    for (int idx = tid; idx < q_size * head_dim; idx += blockDim.x) {
        int i = idx / head_dim;
        int j = idx % head_dim;
        Q_tile[i * kRefHeadDim_32 + j] = Q_ptr[(q_start + i) * head_dim + j];
    }
    
    for (int i = tid; i < kRefTileM_32; i += blockDim.x) {
        m_shared[i] = -INFINITY;
        l_shared[i] = 0.0f;
    }
    for (int i = tid; i < kRefTileM_32 * kRefHeadDim_32; i += blockDim.x) {
        O_accum[i] = 0.0f;
    }
    __syncthreads();
    
    const int num_kv_tiles = (seq_len + kRefTileN_32 - 1) / kRefTileN_32;
    
    for (int kv_tile_idx = 0; kv_tile_idx < num_kv_tiles; kv_tile_idx++) {
        const int k_start = kv_tile_idx * kRefTileN_32;
        const int k_end = min(k_start + kRefTileN_32, seq_len);
        const int k_size = k_end - k_start;
        
        for (int idx = tid; idx < k_size * head_dim; idx += blockDim.x) {
            int i = idx / head_dim;
            int j = idx % head_dim;
            K_tile[i * kRefHeadDim_32 + j] = K_ptr[(k_start + i) * head_dim + j];
            V_tile[i * kRefHeadDim_32 + j] = V_ptr[(k_start + i) * head_dim + j];
        }
        __syncthreads();
        
        for (int idx = tid; idx < q_size * k_size; idx += blockDim.x) {
            int i = idx / k_size;
            int j = idx % k_size;
            
            float sum = 0.0f;
            #pragma unroll
            for (int d = 0; d < head_dim; d++) {
                sum += float(Q_tile[i * kRefHeadDim_32 + d]) * float(K_tile[j * kRefHeadDim_32 + d]);
            }
            S_tile[i * kRefTileN_32 + j] = sum * softmax_scale;
        }
        __syncthreads();
        
        for (int i = 0; i < q_size; i++) {
            if (tid == 0) {
                float m_old = m_shared[i];
                float l_old = l_shared[i];
                
                float m_new = m_old;
                for (int j = 0; j < k_size; j++) {
                    m_new = fmaxf(m_new, S_tile[i * kRefTileN_32 + j]);
                }
                
                float l_new = 0.0f;
                for (int j = 0; j < k_size; j++) {
                    float p = expf(S_tile[i * kRefTileN_32 + j] - m_new);
                    S_tile[i * kRefTileN_32 + j] = p;
                    l_new += p;
                }
                
                float correction = expf(m_old - m_new);
                l_new = correction * l_old + l_new;
                
                for (int d = 0; d < head_dim; d++) {
                    O_accum[i * kRefHeadDim_32 + d] *= correction;
                }
                
                m_shared[i] = m_new;
                l_shared[i] = l_new;
            }
        }
        __syncthreads();
        
        for (int i = 0; i < q_size; i++) {
            for (int d = tid; d < head_dim; d += blockDim.x) {
                float sum = 0.0f;
                #pragma unroll 8
                for (int j = 0; j < k_size; j++) {
                    sum += S_tile[i * kRefTileN_32 + j] * float(V_tile[j * kRefHeadDim_32 + d]);
                }
                O_accum[i * kRefHeadDim_32 + d] += sum;
            }
        }
        __syncthreads();
    }
    
    for (int i = 0; i < q_size; i++) {
        float scale = 1.0f / l_shared[i];
        for (int d = tid; d < head_dim; d += blockDim.x) {
            float val = O_accum[i * kRefHeadDim_32 + d] * scale;
            O_ptr[(q_start + i) * head_dim + d] = cutlass::half_t(val);
        }
    }
}

void attention_reference_dim32(
    const cutlass::half_t* Q,
    const cutlass::half_t* K,
    const cutlass::half_t* V,
    cutlass::half_t* O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream
) {
    assert(head_dim == kRefHeadDim_32 && "This reference expects head_dim=32");
    
    float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    const int num_q_blocks = (seq_len + kRefTileM_32 - 1) / kRefTileM_32;
    dim3 grid(num_q_blocks, num_heads, batch_size);
    dim3 block(128);
    
    size_t smem_size = (kRefTileM_32 * kRefHeadDim_32 + kRefTileN_32 * kRefHeadDim_32 * 2) * sizeof(cutlass::half_t) +
                       (kRefTileM_32 * kRefTileN_32) * sizeof(float) +
                       (kRefTileM_32 * 2) * sizeof(float) +
                       (kRefTileM_32 * kRefHeadDim_32) * sizeof(float);
    
    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(
            attention_reference_kernel_dim32,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
    }
    
    attention_reference_kernel_dim32<<<grid, block, smem_size, stream>>>(
        Q, K, V, O,
        softmax_scale,
        batch_size, num_heads, seq_len, head_dim
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Reference (dim32) kernel launch failed: %s\n", 
                cudaGetErrorString(err));
    }
}

// ==================== Baseline实现（Naive版本，不用shared memory和online softmax） ====================

/**
 * 最简单的Attention实现，用于性能和精度对比
 * 
 * 特点：
 * - 不使用shared memory（只用全局内存）
 * - 不使用online softmax（标准的两遍扫描）
 * - 每个thread处理一个query position
 * - 分配全局内存来存储attention scores
 * 
 * 这是最直观的实现，但性能最差（大量全局内存访问）
 */
__global__ void attention_baseline_kernel(
    const cutlass::half_t* __restrict__ Q,
    const cutlass::half_t* __restrict__ K,
    const cutlass::half_t* __restrict__ V,
    cutlass::half_t* __restrict__ O,
    float* __restrict__ scores_buffer,  // 全局内存: [batch, heads, seq, seq]
    float softmax_scale,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    // 每个thread处理一个query position
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (q_idx >= seq_len) return;
    
    const int64_t offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const cutlass::half_t* Q_ptr = Q + offset;
    const cutlass::half_t* K_ptr = K + offset;
    const cutlass::half_t* V_ptr = V + offset;
    cutlass::half_t* O_ptr = O + offset;
    
    // 计算这个query在scores buffer中的位置
    const int64_t scores_offset = ((batch_idx * num_heads + head_idx) * seq_len + q_idx) * seq_len;
    float* my_scores = scores_buffer + scores_offset;
    
    // Step 1: 计算 S = Q[q_idx] @ K^T (存到全局内存)
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        float sum = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            float q_val = float(Q_ptr[q_idx * head_dim + d]);
            float k_val = float(K_ptr[k_idx * head_dim + d]);
            sum += q_val * k_val;
        }
        my_scores[k_idx] = sum * softmax_scale;
    }
    
    // Step 2: Softmax - 第一遍：找max
    float max_score = -INFINITY;
    for (int i = 0; i < seq_len; i++) {
        max_score = fmaxf(max_score, my_scores[i]);
    }
    
    // Step 3: Softmax - 第二遍：计算exp和sum
    float sum_exp = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        my_scores[i] = expf(my_scores[i] - max_score);
        sum_exp += my_scores[i];
    }
    
    // Step 4: Softmax - 归一化
    for (int i = 0; i < seq_len; i++) {
        my_scores[i] /= sum_exp;
    }
    
    // Step 5: 计算 O = softmax(S) @ V
    for (int d = 0; d < head_dim; d++) {
        float sum = 0.0f;
        for (int k_idx = 0; k_idx < seq_len; k_idx++) {
            float v_val = float(V_ptr[k_idx * head_dim + d]);
            sum += my_scores[k_idx] * v_val;
        }
        O_ptr[q_idx * head_dim + d] = cutlass::half_t(sum);
    }
}

void attention_baseline(
    const cutlass::half_t* Q,
    const cutlass::half_t* K,
    const cutlass::half_t* V,
    cutlass::half_t* O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream
) {
    float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    const int threads = 256;
    const int blocks_x = (seq_len + threads - 1) / threads;
    dim3 grid(blocks_x, num_heads, batch_size);
    dim3 block(threads);
    
    // 分配全局内存来存储所有的attention scores
    size_t scores_size = (size_t)batch_size * num_heads * seq_len * seq_len * sizeof(float);
    float* d_scores_buffer;
    CHECK_CUDA(cudaMalloc(&d_scores_buffer, scores_size));
    
    attention_baseline_kernel<<<grid, block, 0, stream>>>(
        Q, K, V, O,
        d_scores_buffer,
        softmax_scale,
        batch_size, num_heads, seq_len, head_dim
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Baseline kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaFree(d_scores_buffer));
}

// ==================== 参考实现（Tiled版本） ====================

/**
 * 标准Attention实现: O = softmax(Q @ K^T / sqrt(d)) @ V
 * 
 * 使用tiled approach来高效使用shared memory
 * 每个block处理多个query positions，使用在线softmax
 * 
 * 这样只需要存储tile而不是整个序列，大大减少shared memory使用
 */

// Tile sizes for reference implementation
constexpr int kRefTileM = 16;  // Query tile size
constexpr int kRefTileN = 32;  // Key/Value tile size
constexpr int kRefHeadDim = 64;

__global__ void attention_reference_kernel(
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
    // Each block processes kRefTileM queries
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_block_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    const int q_start = q_block_idx * kRefTileM;
    const int q_end = min(q_start + kRefTileM, seq_len);
    const int q_size = q_end - q_start;
    
    if (q_size <= 0) return;
    
    const int64_t offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const cutlass::half_t* Q_ptr = Q + offset;
    const cutlass::half_t* K_ptr = K + offset;
    const cutlass::half_t* V_ptr = V + offset;
    cutlass::half_t* O_ptr = O + offset;
    
    // Shared memory layout:
    // - Q_tile: [kRefTileM, kRefHeadDim]
    // - K_tile: [kRefTileN, kRefHeadDim]
    // - V_tile: [kRefTileN, kRefHeadDim]
    // - S_tile: [kRefTileM, kRefTileN] (scores)
    extern __shared__ char smem[];
    cutlass::half_t* Q_tile = (cutlass::half_t*)smem;
    cutlass::half_t* K_tile = Q_tile + kRefTileM * kRefHeadDim;
    cutlass::half_t* V_tile = K_tile + kRefTileN * kRefHeadDim;
    float* S_tile = (float*)(V_tile + kRefTileN * kRefHeadDim);
    
    // Per-query statistics and accumulator (in shared memory)
    float* m_shared = S_tile + kRefTileM * kRefTileN;
    float* l_shared = m_shared + kRefTileM;
    float* O_accum = l_shared + kRefTileM;  // [kRefTileM, kRefHeadDim]
    
    // Load Q tile
    for (int idx = tid; idx < q_size * head_dim; idx += blockDim.x) {
        int i = idx / head_dim;
        int j = idx % head_dim;
        Q_tile[i * kRefHeadDim + j] = Q_ptr[(q_start + i) * head_dim + j];
    }
    
    // Initialize statistics and output accumulator
    for (int i = tid; i < kRefTileM; i += blockDim.x) {
        m_shared[i] = -INFINITY;
        l_shared[i] = 0.0f;
    }
    for (int i = tid; i < kRefTileM * kRefHeadDim; i += blockDim.x) {
        O_accum[i] = 0.0f;
    }
    __syncthreads();
    
    // Iterate over K/V tiles
    const int num_kv_tiles = (seq_len + kRefTileN - 1) / kRefTileN;
    
    for (int kv_tile_idx = 0; kv_tile_idx < num_kv_tiles; kv_tile_idx++) {
        const int k_start = kv_tile_idx * kRefTileN;
        const int k_end = min(k_start + kRefTileN, seq_len);
        const int k_size = k_end - k_start;
        
        // Load K tile
        for (int idx = tid; idx < k_size * head_dim; idx += blockDim.x) {
            int i = idx / head_dim;
            int j = idx % head_dim;
            K_tile[i * kRefHeadDim + j] = K_ptr[(k_start + i) * head_dim + j];
        }
        
        // Load V tile
        for (int idx = tid; idx < k_size * head_dim; idx += blockDim.x) {
            int i = idx / head_dim;
            int j = idx % head_dim;
            V_tile[i * kRefHeadDim + j] = V_ptr[(k_start + i) * head_dim + j];
        }
        __syncthreads();
        
        // Compute S = Q @ K^T for this tile
        for (int idx = tid; idx < q_size * k_size; idx += blockDim.x) {
            int i = idx / k_size;  // query index in tile
            int j = idx % k_size;  // key index in tile
            
            float sum = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                sum += float(Q_tile[i * kRefHeadDim + d]) * float(K_tile[j * kRefHeadDim + d]);
            }
            S_tile[i * kRefTileN + j] = sum * softmax_scale;
        }
        __syncthreads();
        
        // Online softmax update for each query
        for (int i = 0; i < q_size; i++) {
            if (tid == 0) {
                float m_old = m_shared[i];
                float l_old = l_shared[i];
                
                // Find new max
                float m_new = m_old;
                for (int j = 0; j < k_size; j++) {
                    m_new = fmaxf(m_new, S_tile[i * kRefTileN + j]);
                }
                
                // Compute exp and new sum
                float l_new = 0.0f;
                for (int j = 0; j < k_size; j++) {
                    float p = expf(S_tile[i * kRefTileN + j] - m_new);
                    S_tile[i * kRefTileN + j] = p;  // Store P values
                    l_new += p;
                }
                
                // Correction factor for old accumulator
                float correction = expf(m_old - m_new);
                l_new = correction * l_old + l_new;
                
                // Apply correction to accumulator
                for (int d = 0; d < head_dim; d++) {
                    O_accum[i * kRefHeadDim + d] *= correction;
                }
                
                m_shared[i] = m_new;
                l_shared[i] = l_new;
            }
        }
        __syncthreads();
        
        // Accumulate O += P @ V
        for (int i = 0; i < q_size; i++) {
            for (int d = tid; d < head_dim; d += blockDim.x) {
                float sum = 0.0f;
                for (int j = 0; j < k_size; j++) {
                    sum += S_tile[i * kRefTileN + j] * float(V_tile[j * kRefHeadDim + d]);
                }
                O_accum[i * kRefHeadDim + d] += sum;
            }
        }
        __syncthreads();
    }
    
    // Final normalization and write back
    for (int i = 0; i < q_size; i++) {
        float scale = 1.0f / l_shared[i];
        for (int d = tid; d < head_dim; d += blockDim.x) {
            float val = O_accum[i * kRefHeadDim + d] * scale;
            O_ptr[(q_start + i) * head_dim + d] = cutlass::half_t(val);
        }
    }
}

void attention_reference(
    const cutlass::half_t* Q,
    const cutlass::half_t* K,
    const cutlass::half_t* V,
    cutlass::half_t* O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream
) {
    assert(head_dim == kRefHeadDim && "Reference implementation expects head_dim=64");
    
    float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    const int num_q_blocks = (seq_len + kRefTileM - 1) / kRefTileM;
    dim3 grid(num_q_blocks, num_heads, batch_size);
    dim3 block(128);  // 128 threads per block
    
    // Calculate shared memory size
    size_t smem_size = (kRefTileM * kRefHeadDim + kRefTileN * kRefHeadDim * 2) * sizeof(cutlass::half_t) +
                       (kRefTileM * kRefTileN) * sizeof(float) +  // S_tile
                       (kRefTileM * 2) * sizeof(float) +          // m_shared, l_shared
                       (kRefTileM * kRefHeadDim) * sizeof(float); // O_accum
    
    // Set shared memory limit if needed
    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(
            attention_reference_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
    }
    
    attention_reference_kernel<<<grid, block, smem_size, stream>>>(
        Q, K, V, O,
        softmax_scale,
        batch_size, num_heads, seq_len, head_dim
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Reference kernel launch failed: %s\n", 
                cudaGetErrorString(err));
        fprintf(stderr, "  Grid: (%d, %d, %d), Block: (%d), Shared mem: %zu bytes (%.1f KB)\n",
                grid.x, grid.y, grid.z, block.x, smem_size, smem_size / 1024.0);
    }
}

// ==================== Main ====================

int main() {
    printf("Flash Attention Minimal Implementation Test\n");
    printf("================================================================================\n");
    
    // GPU信息
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("================================================================================\n");
    
    printf("\n");
    printf("================================================================================\n");
    printf("Flash Attention Performance Test: Tile Size Comparison\n");
    printf("================================================================================\n");
    printf("\nBoth implementations use Flash Attention algorithm (online softmax + tiling)\n");
    printf("The difference is in the tile size strategy:\n\n");
    printf("  Large Tile (head_dim=32): 120×120 tiles, 256 threads, 150.9 KB shared mem\n");
    printf("    → Maximizes data reuse, higher memory pressure\n\n");
    printf("  Small Tile (head_dim=32): 45×90 tiles, 128 threads, 51.7 KB shared mem\n");
    printf("    → Lower memory footprint, better occupancy\n\n");
    printf("  Baseline: O(batch × heads × seq_len²) memory ← QUADRATIC!\n\n");
    
    // 测试用例 - 对比 head_dim=32 和 head_dim=64 的性能
    std::vector<TestConfig> configs = {
        // // head_dim=64 测试
        // {1, 1, 512, 64},     // 基线
        // {1, 1, 1024, 64},    // 长序列
        // {2, 8, 512, 64},     // 多batch+多head
        
        // // head_dim=32 测试 (优化版本，更大的tile)
        // {1, 1, 512, 32},     // 基线 - 对比64
        // {1, 1, 1024, 32},    // 长序列 - 对比64
        // {1, 1, 2048, 32},    // 超长序列 - head_dim=32可以处理更大的seq_len
        // {2, 8, 512, 32},     // 多batch+多head - 对比64
        {1, 16, 1024, 32}, 
    };
    
    for (const auto& config : configs) {
        run_test(config);
    }
    
    printf("\n");
    printf("================================================================================\n");
    printf("All tests completed!\n");
    
    return 0;
}


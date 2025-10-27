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

// 声明Flash Attention接口
void flash_attention_forward(
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

// 声明参考实现
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
float compute_max_relative_error(const cutlass::half_t* a, const cutlass::half_t* b, size_t size) {
    std::vector<cutlass::half_t> host_a(size), host_b(size);
    CHECK_CUDA(cudaMemcpy(host_a.data(), a, size * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(host_b.data(), b, size * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost));
    
    float max_error = 0.0f;
    int error_count = 0;
    const float threshold = 1e-2;
    
    for (size_t i = 0; i < size; i++) {
        float val_a = float(host_a[i]);
        float val_b = float(host_b[i]);
        float abs_diff = std::abs(val_a - val_b);
        float rel_error = abs_diff / (std::abs(val_b) + 1e-8f);
        
        if (rel_error > threshold) {
            error_count++;
            if (error_count <= 10) {  // 只打印前10个错误
                printf("Error at %zu: flash=%.6f, ref=%.6f, rel_err=%.6f\n",
                       i, val_a, val_b, rel_error);
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
    
    printf("Memory: %.2f MB per tensor, %.2f MB total\n", 
           bytes / 1024.0 / 1024.0,
           bytes * 4 / 1024.0 / 1024.0);  // Q, K, V, O
    
    // 分配device memory
    cutlass::half_t *d_Q, *d_K, *d_V, *d_O_flash, *d_O_ref;
    CHECK_CUDA(cudaMalloc(&d_Q, bytes));
    CHECK_CUDA(cudaMalloc(&d_K, bytes));
    CHECK_CUDA(cudaMalloc(&d_V, bytes));
    CHECK_CUDA(cudaMalloc(&d_O_flash, bytes));
    CHECK_CUDA(cudaMalloc(&d_O_ref, bytes));
    
    // 初始化输入
    printf("Initializing inputs...\n");
    init_random(d_Q, qkv_size);
    init_random(d_K, qkv_size);
    init_random(d_V, qkv_size);
    
    // 运行Flash Attention
    printf("Running Flash Attention...\n");
    float time_flash = benchmark([&]() {
        flash_attention_forward(
            d_Q, d_K, d_V, d_O_flash,
            config.batch_size, config.num_heads, config.seq_len, config.head_dim,
            0
        );
    });
    
    // 运行参考实现
    printf("Running Reference Implementation...\n");
    float time_ref = benchmark([&]() {
        attention_reference(
            d_Q, d_K, d_V, d_O_ref,
            config.batch_size, config.num_heads, config.seq_len, config.head_dim,
            0
        );
    });
    
    // 验证正确性
    printf("\nVerifying correctness...\n");
    float max_error = compute_max_relative_error(d_O_flash, d_O_ref, qkv_size);
    
    // 输出结果
    printf("\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("Results:\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("Flash Attention:  %.3f ms\n", time_flash);
    printf("Reference:        %.3f ms\n", time_ref);
    printf("Speedup:          %.2fx\n", time_ref / time_flash);
    printf("Max Relative Err: %.6f\n", max_error);
    
    // 判断是否通过
    const float error_threshold = 0.05f;  // 5% 相对误差
    if (max_error < error_threshold) {
        printf("\n✅ TEST PASSED (error < %.1f%%)\n", error_threshold * 100);
    } else {
        printf("\n❌ TEST FAILED (error >= %.1f%%)\n", error_threshold * 100);
    }
    
    // 计算FLOPs
    const int64_t flops = 4LL * config.batch_size * config.num_heads * 
                          config.seq_len * config.seq_len * config.head_dim;
    const float tflops_flash = flops / (time_flash * 1e-3) / 1e12;
    const float tflops_ref = flops / (time_ref * 1e-3) / 1e12;
    
    printf("\nThroughput:\n");
    printf("Flash Attention: %.2f TFLOPs/s\n", tflops_flash);
    printf("Reference:       %.2f TFLOPs/s\n", tflops_ref);
    
    // 清理
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O_flash));
    CHECK_CUDA(cudaFree(d_O_ref));
}

// ==================== 参考实现（简单版） ====================

/**
 * 标准Attention实现: O = softmax(Q @ K^T / sqrt(d)) @ V
 * 
 * 这是一个简单的实现，不考虑性能优化
 * 用于验证Flash Attention的正确性
 */
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
    // 每个thread处理一个(batch, head, query_pos)
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (q_idx >= seq_len) return;
    
    const int64_t offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const cutlass::half_t* Q_ptr = Q + offset;
    const cutlass::half_t* K_ptr = K + offset;
    const cutlass::half_t* V_ptr = V + offset;
    cutlass::half_t* O_ptr = O + offset;
    
    // 临时数组（在寄存器/局部内存）
    extern __shared__ float scores[];
    float* my_scores = scores + threadIdx.x * seq_len;
    
    // 计算 scores = Q[q_idx] @ K^T
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        float sum = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            float q_val = float(Q_ptr[q_idx * head_dim + d]);
            float k_val = float(K_ptr[k_idx * head_dim + d]);
            sum += q_val * k_val;
        }
        my_scores[k_idx] = sum * softmax_scale;
    }
    
    // Softmax
    float max_score = -INFINITY;
    for (int i = 0; i < seq_len; i++) {
        max_score = fmaxf(max_score, my_scores[i]);
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        my_scores[i] = expf(my_scores[i] - max_score);
        sum_exp += my_scores[i];
    }
    
    for (int i = 0; i < seq_len; i++) {
        my_scores[i] /= sum_exp;
    }
    
    // 计算 O = scores @ V
    for (int d = 0; d < head_dim; d++) {
        float sum = 0.0f;
        for (int k_idx = 0; k_idx < seq_len; k_idx++) {
            float v_val = float(V_ptr[k_idx * head_dim + d]);
            sum += my_scores[k_idx] * v_val;
        }
        O_ptr[q_idx * head_dim + d] = cutlass::half_t(sum);
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
    float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    const int threads = 256;
    const int blocks_x = (seq_len + threads - 1) / threads;
    dim3 grid(blocks_x, num_heads, batch_size);
    dim3 block(threads);
    
    size_t smem_size = threads * seq_len * sizeof(float);
    
    // Set shared memory limit if needed (like Flash Attention does)
    if (smem_size > 48 * 1024) {  // If exceeds default 48KB
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
    
    // 测试用例
    std::vector<TestConfig> configs = {
        {1, 1, 128, 64},    // 小规模
        {1, 1, 512, 64},    // 中等规模
        {1, 1, 1024, 64},   // 大规模
        {2, 8, 512, 64},    // 多batch, 多head
    };
    
    for (const auto& config : configs) {
        run_test(config);
    }
    
    printf("\n");
    printf("================================================================================\n");
    printf("All tests completed!\n");
    
    return 0;
}


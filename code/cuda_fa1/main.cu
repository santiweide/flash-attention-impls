#include "flashAttention.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <sys/time.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

// ==================== 辅助函数 (from test_flash_attn.cu) ====================

#define CHECK_CUDA(call) \
    do { \
        cudaError_t status = call; \
        if (status != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(status)); \
            exit(1); \
        } \
    } while(0)

// 初始化随机数据 (from test_flash_attn.cu) - 支持float和half
void init_random(float* data, size_t size, float mean = 0.0f, float stddev = 0.02f) {
    std::vector<float> host_data(size);
    std::mt19937 gen(42);  // 固定seed以便复现
    std::normal_distribution<float> dist(mean, stddev);
    
    for (size_t i = 0; i < size; i++) {
        host_data[i] = dist(gen);
    }
    
    CHECK_CUDA(cudaMemcpy(data, host_data.data(), 
                          size * sizeof(float), 
                          cudaMemcpyHostToDevice));
}

// 初始化随机数据 - half_t版本
void init_random_half(__half* data, size_t size, float mean = 0.0f, float stddev = 0.02f) {
    std::vector<float> host_data(size);
    std::mt19937 gen(42);  // 固定seed以便复现
    std::normal_distribution<float> dist(mean, stddev);
    
    for (size_t i = 0; i < size; i++) {
        host_data[i] = dist(gen);
    }
    
    // 转换为half_t
    std::vector<__half> host_data_half(size);
    for (size_t i = 0; i < size; i++) {
        host_data_half[i] = __float2half(host_data[i]);
    }
    
    CHECK_CUDA(cudaMemcpy(data, host_data_half.data(), 
                          size * sizeof(__half), 
                          cudaMemcpyHostToDevice));
}

// 计算两个数组的最大相对误差
// 使用对称的相对误差公式，对近零值更稳健: |a-b| / (|a| + |b| + eps)
float compute_max_relative_error(const float* a, const float* b, size_t size) {
    std::vector<float> host_a(size), host_b(size);
    CHECK_CUDA(cudaMemcpy(host_a.data(), a, size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(host_b.data(), b, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    float max_error = 0.0f;
    int error_count = 0;
    const float error_threshold = 0.01f;  // 1% 相对误差阈值
    const float epsilon = 1e-5f;          // 防止除零
    
    for (size_t i = 0; i < size; i++) {
        float val_a = host_a[i];
        float val_b = host_b[i];
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

// Benchmark函数 (from test_flash_attn.cu)
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
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ scores_buffer,
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
    const float* Q_ptr = Q + offset;
    const float* K_ptr = K + offset;
    const float* V_ptr = V + offset;
    float* O_ptr = O + offset;
    
    // 计算这个query在scores buffer中的位置
    const int64_t scores_offset = ((batch_idx * num_heads + head_idx) * seq_len + q_idx) * seq_len;
    float* my_scores = scores_buffer + scores_offset;
    
    // Step 1: 计算 S = Q[q_idx] @ K^T (存到全局内存)
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        float sum = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            float q_val = Q_ptr[q_idx * head_dim + d];
            float k_val = K_ptr[k_idx * head_dim + d];
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
            float v_val = V_ptr[k_idx * head_dim + d];
            sum += my_scores[k_idx] * v_val;
        }
        O_ptr[q_idx * head_dim + d] = sum;
    }
}

void attention_baseline(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream
) {
    float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));  // 恢复标准缩放因子
    
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

// Verification function - 支持half_t
bool verify_flash_attention(
    const __half* Q, const __half* K, const __half* V,
    int B, int H, int N, int d, int M,
    float tolerance) {
    
    printf("Verifying Flash Attention accuracy against Standard Attention...\n");
    
    // 首先将half_t转换为float用于baseline计算
    size_t size_QKV_half = (size_t)B * H * N * d * sizeof(__half);
    size_t size_QKV_float = (size_t)B * H * N * d * sizeof(float);
    
    float *Q_float, *K_float, *V_float;
    cudaMalloc(&Q_float, size_QKV_float);
    cudaMalloc(&K_float, size_QKV_float);
    cudaMalloc(&V_float, size_QKV_float);
    
    // 转换half_t到float（简化版本，在实际应用中可以使用kernel）
    std::vector<__half> host_Q(B * H * N * d), host_K(B * H * N * d), host_V(B * H * N * d);
    cudaMemcpy(host_Q.data(), Q, size_QKV_half, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_K.data(), K, size_QKV_half, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_V.data(), V, size_QKV_half, cudaMemcpyDeviceToHost);
    
    std::vector<float> host_Q_f(B * H * N * d), host_K_f(B * H * N * d), host_V_f(B * H * N * d);
    for (size_t i = 0; i < B * H * N * d; i++) {
        host_Q_f[i] = __half2float(host_Q[i]);
        host_K_f[i] = __half2float(host_K[i]);
        host_V_f[i] = __half2float(host_V[i]);
    }
    cudaMemcpy(Q_float, host_Q_f.data(), size_QKV_float, cudaMemcpyHostToDevice);
    cudaMemcpy(K_float, host_K_f.data(), size_QKV_float, cudaMemcpyHostToDevice);
    cudaMemcpy(V_float, host_V_f.data(), size_QKV_float, cudaMemcpyHostToDevice);
    
    // Allocate memory for standard attention result
    float *O_standard;
    cudaMalloc(&O_standard, size_QKV_float);
    
    // Compute standard attention on GPU
    attention_baseline(Q_float, K_float, V_float, O_standard, B, H, N, d, 0);
    
    // Allocate memory for flash attention result (half_t)
    size_t size_LM = (size_t)B * H * N * sizeof(float);
    __half *O_flash;
    float *l, *m;
    cudaMalloc(&O_flash, size_QKV_half);
    cudaMalloc(&l, size_LM);
    cudaMalloc(&m, size_LM);
    
    // Compute flash attention
    int Bc = (int)ceilf((float)M / (4.0f * (float)d));
    int Br = (Bc < d) ? Bc : d;
    int Tr = (N + Br - 1) / Br;
    
    dim3 grid(Tr, B*H);
    dim3 block(Br);
    // shared memory: Qi (half_t) + Kj (half_t) + Vj (half_t) + O_accum (float)
    size_t shmem = (size_t)(Br*d + Bc*d + Bc*d) * sizeof(__half) + 
                   (size_t)(Br*d) * sizeof(float);
    
    flash_attention_forward<<<grid, block, shmem>>>(Q, K, V, O_flash, l, m, B, H, N, d, M);
    cudaDeviceSynchronize();
    
    // Copy results to host for comparison
    std::vector<float> host_O_standard(B * H * N * d);
    std::vector<__half> host_O_flash(B * H * N * d);
    cudaMemcpy(host_O_standard.data(), O_standard, size_QKV_float, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_O_flash.data(), O_flash, size_QKV_half, cudaMemcpyDeviceToHost);
    
    // 转换half_t到float进行比较
    std::vector<float> host_O_flash_f(B * H * N * d);
    for (size_t i = 0; i < B * H * N * d; i++) {
        host_O_flash_f[i] = __half2float(host_O_flash[i]);
    }
    
    // Compare results using relative error
    float max_error = 0.0f;
    int error_count = 0;
    const float error_threshold = 0.01f;
    const float epsilon = 1e-5f;
    
    for (size_t i = 0; i < B * H * N * d; i++) {
        float val_a = host_O_flash_f[i];
        float val_b = host_O_standard[i];
        float abs_diff = std::abs(val_a - val_b);
        float denominator = std::abs(val_a) + std::abs(val_b) + epsilon;
        float rel_error = abs_diff / denominator;
        
        if (rel_error > error_threshold) {
            error_count++;
            if (error_count <= 10) {
                printf("Error at %zu: flash=%.6f, ref=%.6f, abs_diff=%.6f, rel_err=%.6f\n",
                       i, val_a, val_b, abs_diff, rel_error);
            }
        }
        max_error = std::max(max_error, rel_error);
    }
    
    if (error_count > 10) {
        printf("... and %d more errors\n", error_count - 10);
    }
    
    const float max_relative_error = max_error;
    const float error_threshold_final = 0.02f;
    bool is_correct = max_relative_error < error_threshold_final;
    
    printf("Max relative error: %.6f\n", max_relative_error);
    printf("Error threshold: %.6f\n", error_threshold_final);
    printf("Verification result: %s\n", is_correct ? "PASSED" : "FAILED");
    
    // Cleanup
    cudaFree(O_standard);
    cudaFree(Q_float);
    cudaFree(K_float);
    cudaFree(V_float);
    cudaFree(O_flash);
    cudaFree(l);
    cudaFree(m);
    
    return is_correct;
}

int main(int argc, char** argv) {
    int B    = (argc>1)? atoi(argv[1]) : 1;
    int H    = (argc>2)? atoi(argv[2]) : 8;
    int N    = (argc>3)? atoi(argv[3]) : 512;
    int d    = (argc>4)? atoi(argv[4]) : 64;
    int M    = (argc>5)? atoi(argv[5]) : 4096;
    int runs = (argc>6)? atoi(argv[6]) : 50;

    printf("Flash Attention Performance Test\n");
    printf("B=%d, H=%d, N=%d, d=%d, M=%d, runs=%d\n", B, H, N, d, M, runs);
    printf("Using FP16 (half_t) for Q, K, V, O\n");

    int Bc = (int)ceilf((float)M / (4.0f * (float)d));
    int Br = (Bc < d) ? Bc : d;
    int Tr = (N + Br - 1) / Br;

    dim3 grid(Tr, B*H);
    dim3 block(Br);
    // shared memory: Qi (half_t) + Kj (half_t) + Vj (half_t) + O_accum (float)
    size_t shmem = (size_t)(Br*d + Bc*d + Bc*d) * sizeof(__half) + 
                   (size_t)(Br*d) * sizeof(float);

    // Allocate memory - 使用half_t
    size_t size_QKV_half = (size_t)B * H * N * d * sizeof(__half);
    size_t size_QKV_float = (size_t)B * H * N * d * sizeof(float);
    size_t size_LM  = (size_t)B * H * N * sizeof(float);
    
    __half *Q, *K, *V, *O_flash;
    float *O_standard, *l, *m;
    cudaMalloc(&Q, size_QKV_half);
    cudaMalloc(&K, size_QKV_half);
    cudaMalloc(&V, size_QKV_half);
    cudaMalloc(&O_flash, size_QKV_half);
    cudaMalloc(&O_standard, size_QKV_float);  // baseline仍然使用float
    cudaMalloc(&l, size_LM);
    cudaMalloc(&m, size_LM);

    // Initialize with random data - 使用half_t版本
    init_random_half(Q, B * H * N * d);
    init_random_half(K, B * H * N * d);
    init_random_half(V, B * H * N * d);

    // Verify correctness first
    bool verification_passed = verify_flash_attention(Q, K, V, B, H, N, d, M);
    if (!verification_passed) {
        printf("Warning: Flash Attention verification failed, but continuing with performance test\n");
    }

    // Performance test - Standard Attention (需要先转换为float)
    printf("\nRunning Standard Attention performance test...\n");
    // 转换half_t到float用于baseline
    float *Q_float, *K_float, *V_float;
    cudaMalloc(&Q_float, size_QKV_float);
    cudaMalloc(&K_float, size_QKV_float);
    cudaMalloc(&V_float, size_QKV_float);
    
    // 简化的转换（在实际应用中应该使用kernel）
    std::vector<__half> host_Q(B * H * N * d), host_K(B * H * N * d), host_V(B * H * N * d);
    cudaMemcpy(host_Q.data(), Q, size_QKV_half, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_K.data(), K, size_QKV_half, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_V.data(), V, size_QKV_half, cudaMemcpyDeviceToHost);
    
    std::vector<float> host_Q_f(B * H * N * d), host_K_f(B * H * N * d), host_V_f(B * H * N * d);
    for (size_t i = 0; i < B * H * N * d; i++) {
        host_Q_f[i] = __half2float(host_Q[i]);
        host_K_f[i] = __half2float(host_K[i]);
        host_V_f[i] = __half2float(host_V[i]);
    }
    cudaMemcpy(Q_float, host_Q_f.data(), size_QKV_float, cudaMemcpyHostToDevice);
    cudaMemcpy(K_float, host_K_f.data(), size_QKV_float, cudaMemcpyHostToDevice);
    cudaMemcpy(V_float, host_V_f.data(), size_QKV_float, cudaMemcpyHostToDevice);
    
    float time_standard = benchmark([&]() {
        attention_baseline(Q_float, K_float, V_float, O_standard, B, H, N, d, 0);
    }, 5, runs);

    // Performance test - Flash Attention
    printf("Running Flash Attention performance test...\n");
    float time_flash = benchmark([&]() {
        flash_attention_forward<<<grid, block, shmem>>>(Q, K, V, O_flash, l, m, B, H, N, d, M);
    }, 5, runs);

    // Calculate performance metrics
    double bytes_per_call =
        3.0 * size_QKV_half +   // Q, K, V (half_t)
        1.0 * size_QKV_half +   // O (half_t)
        2.0 * size_LM;           // l, m (float)

    double GBps_standard = (bytes_per_call / (time_standard * 1e-3)) / 1e9;
    double GBps_flash = (bytes_per_call / (time_flash * 1e-3)) / 1e9;

    double flops = 4.0 * (double)B * H * N * N * d;        
    double gflops_per_s_standard = (flops / (time_standard * 1e-3)) / 1e9; 
    double gflops_per_s_flash = (flops / (time_flash * 1e-3)) / 1e9;

    printf("\n");
    printf("================================================================================\n");
    printf("Performance Results:\n");
    printf("================================================================================\n");
    printf("%-25s %10.2f ms  (%.2fx speedup)\n", 
           "Standard Attention:", time_standard, 1.0f);
    printf("%-25s %10.2f ms  (%.2fx speedup)\n", 
           "Flash Attention:", time_flash, time_standard / time_flash);
    
    printf("\n");
    printf("%-25s %10.2f GB/s\n", "Standard Throughput:", GBps_standard);
    printf("%-25s %10.2f GB/s\n", "Flash Throughput:", GBps_flash);
    
    printf("\n");
    printf("%-25s %10.3f GFLOPs/s\n", "Standard Compute:", gflops_per_s_standard);
    printf("%-25s %10.3f GFLOPs/s\n", "Flash Compute:", gflops_per_s_flash);

    // Cleanup
    cudaFree(Q); cudaFree(K); cudaFree(V);
    cudaFree(O_flash); cudaFree(O_standard); cudaFree(l); cudaFree(m);
    cudaFree(Q_float); cudaFree(K_float); cudaFree(V_float);
    
    return 0;
}

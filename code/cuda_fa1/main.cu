#include "flashAttention.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <random>
#include <chrono>

// ==================== Helper functions ====================

#define CHECK_CUDA(call) \
    do { \
        cudaError_t status = call; \
        if (status != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(status)); \
            exit(1); \
        } \
    } while(0)


// Initialize random data - half_t version
void init_random_half(__half* data, size_t size, float mean = 0.0f, float stddev = 0.02f) {
    std::vector<float> host_data(size);
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::normal_distribution<float> dist(mean, stddev);
    
    for (size_t i = 0; i < size; i++) {
        host_data[i] = dist(gen);
    }
    
    // Convert to half_t
    std::vector<__half> host_data_half(size);
    for (size_t i = 0; i < size; i++) {
        host_data_half[i] = __float2half(host_data[i]);
    }
    
    CHECK_CUDA(cudaMemcpy(data, host_data_half.data(), 
                          size * sizeof(__half), 
                          cudaMemcpyHostToDevice));
}

// Compute maximum relative error between two arrays
// Uses symmetric relative error formula, more robust for near-zero values: |a-b| / (|a| + |b| + eps)
float compute_max_relative_error(const float* a, const float* b, size_t size) {
    std::vector<float> host_a(size), host_b(size);
    CHECK_CUDA(cudaMemcpy(host_a.data(), a, size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(host_b.data(), b, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    float max_error = 0.0f;
    int error_count = 0;
    const float error_threshold = 0.01f;  // 1% relative error threshold
    const float epsilon = 1e-5f;          // Prevent division by zero
    
    for (size_t i = 0; i < size; i++) {
        float val_a = host_a[i];
        float val_b = host_b[i];
        float abs_diff = std::abs(val_a - val_b);
        
        // Symmetric relative error: |a-b| / (|a| + |b| + eps)
        // This formula is more robust for near-zero values and bounded [0, 1)
        float denominator = std::abs(val_a) + std::abs(val_b) + epsilon;
        float rel_error = abs_diff / denominator;
        
        if (rel_error > error_threshold) {
            error_count++;
            if (error_count <= 10) {  // Only print first 10 errors
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

// Benchmark function 
template<typename Func>
float benchmark(Func func, int warmup = 5, int repeats = 20) {
    // Warmup
    for (int i = 0; i < warmup; i++) {
        func();
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // timer
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeats; i++) {
        func();
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<float, std::milli> duration = end - start;
    return duration.count() / repeats;
}

// ==================== Baseline implementation (Naive version, no shared memory and online softmax) ====================

/**
 * Simplest Attention implementation for performance and accuracy comparison
 * 
 * Features:
 * - No shared memory (only global memory)
 * - No online softmax (standard two-pass scan)
 * - Each thread processes one query position
 * - Allocates global memory to store attention scores
 * 
 * This is the most intuitive implementation, but has worst performance (lots of global memory access)
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
    // Each thread processes one query position
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (q_idx >= seq_len) return;
    
    const int64_t offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const float* Q_ptr = Q + offset;
    const float* K_ptr = K + offset;
    const float* V_ptr = V + offset;
    float* O_ptr = O + offset;
    
    // Calculate this query's position in scores buffer
    const int64_t scores_offset = ((batch_idx * num_heads + head_idx) * seq_len + q_idx) * seq_len;
    float* my_scores = scores_buffer + scores_offset;
    
    // Step 1: Compute S = Q[q_idx] @ K^T (store in global memory)
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        float sum = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            float q_val = Q_ptr[q_idx * head_dim + d];
            float k_val = K_ptr[k_idx * head_dim + d];
            sum += q_val * k_val;
        }
        my_scores[k_idx] = sum * softmax_scale;
    }
    
    // Step 2: Softmax - First pass: find max
    float max_score = -INFINITY;
    for (int i = 0; i < seq_len; i++) {
        max_score = fmaxf(max_score, my_scores[i]);
    }
    
    // Step 3: Softmax - Second pass: compute exp and sum
    float sum_exp = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        my_scores[i] = expf(my_scores[i] - max_score);
        sum_exp += my_scores[i];
    }
    
    // Step 4: Softmax - Normalize
    for (int i = 0; i < seq_len; i++) {
        my_scores[i] /= sum_exp;
    }
    
    // Step 5: Compute O = softmax(S) @ V
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
    float* scores_buffer,  // Externally allocated memory buffer
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream
) {
    float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));  // Restore standard scale factor
    
    const int threads = 256;
    const int blocks_x = (seq_len + threads - 1) / threads;
    dim3 grid(blocks_x, num_heads, batch_size);
    dim3 block(threads);
    
    attention_baseline_kernel<<<grid, block, 0, stream>>>(
        Q, K, V, O,
        scores_buffer,
        softmax_scale,
        batch_size, num_heads, seq_len, head_dim
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Baseline kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

// Verification function - supports half_t
bool verify_flash_attention(
    const __half* Q, const __half* K, const __half* V,
    int B, int H, int N, int d, int M,
    float tolerance) {
    
    printf("Verifying Flash Attention accuracy against Standard Attention...\n");
    
    // First convert half_t to float for baseline calculation
    size_t size_QKV_half = (size_t)B * H * N * d * sizeof(__half);
    size_t size_QKV_float = (size_t)B * H * N * d * sizeof(float);
    
    float *Q_float, *K_float, *V_float;
    cudaMalloc(&Q_float, size_QKV_float);
    cudaMalloc(&K_float, size_QKV_float);
    cudaMalloc(&V_float, size_QKV_float);
    
    // Convert half_t to float (simplified version, in practice can use kernel)
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
    
    // Allocate memory for scores buffer
    size_t scores_size = (size_t)B * H * N * N * sizeof(float);
    float *scores_buffer;
    cudaMalloc(&scores_buffer, scores_size);
    
    // Compute standard attention on GPU
    attention_baseline(Q_float, K_float, V_float, O_standard, scores_buffer, B, H, N, d, 0);
    
    cudaFree(scores_buffer);
    
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
    
    // Convert half_t to float for comparison
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
    // Set to use GPU 1
    CHECK_CUDA(cudaSetDevice(1));
    
    // If command line arguments provided, run single configuration (backward compatible)
    if (argc > 1) {
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

        // Allocate memory - using half_t
        size_t size_QKV_half = (size_t)B * H * N * d * sizeof(__half);
        size_t size_QKV_float = (size_t)B * H * N * d * sizeof(float);
        size_t size_LM  = (size_t)B * H * N * sizeof(float);
        
        __half *Q, *K, *V, *O_flash;
        float *O_standard, *l, *m;
        cudaMalloc(&Q, size_QKV_half);
        cudaMalloc(&K, size_QKV_half);
        cudaMalloc(&V, size_QKV_half);
        cudaMalloc(&O_flash, size_QKV_half);
        cudaMalloc(&O_standard, size_QKV_float);  // baseline still uses float
        cudaMalloc(&l, size_LM);
        cudaMalloc(&m, size_LM);

        // Initialize with random data - using half_t version
        init_random_half(Q, B * H * N * d);
        init_random_half(K, B * H * N * d);
        init_random_half(V, B * H * N * d);

        // Verify correctness first
        bool verification_passed = verify_flash_attention(Q, K, V, B, H, N, d, M);
        if (!verification_passed) {
            printf("Warning: Flash Attention verification failed, but continuing with performance test\n");
        }

        // Performance test - Standard Attention (need to convert to float first)
        printf("\nRunning Standard Attention performance test...\n");
        // Convert half_t to float for baseline
        float *Q_float, *K_float, *V_float;
        cudaMalloc(&Q_float, size_QKV_float);
        cudaMalloc(&K_float, size_QKV_float);
        cudaMalloc(&V_float, size_QKV_float);
        
        // Simplified conversion (in practice should use kernel)
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
        
        // Pre-allocate scores_buffer to avoid allocation/deallocation on each call
        size_t scores_size = (size_t)B * H * N * N * sizeof(float);
        float *scores_buffer;
        cudaMalloc(&scores_buffer, scores_size);
        
        float time_standard = benchmark([&]() {
            attention_baseline(Q_float, K_float, V_float, O_standard, scores_buffer, B, H, N, d, 0);
        }, 5, runs);
        
        cudaFree(scores_buffer);

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
        double tflops_per_s_standard = (flops / (time_standard * 1e-3)) / 1e12; 
        double tflops_per_s_flash = (flops / (time_flash * 1e-3)) / 1e12;

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
        printf("%-25s %10.3f TFLOPs/s\n", "Standard Compute:", tflops_per_s_standard);
        printf("%-25s %10.3f TFLOPs/s\n", "Flash Compute:", tflops_per_s_flash);

        // Cleanup
        cudaFree(Q); cudaFree(K); cudaFree(V);
        cudaFree(O_flash); cudaFree(O_standard); cudaFree(l); cudaFree(m);
        cudaFree(Q_float); cudaFree(K_float); cudaFree(V_float);
        
        return 0;
    }

    // Otherwise loop through all configurations
    int configs[][4] = {
        {1, 1, 512, 64},
        {1, 1, 1024, 64},
        {1, 1, 2048, 64},
        {1, 1, 4096, 64},
        {1, 32, 8192, 32},
        {1, 32, 8192, 64},
        {1, 32, 8192, 128},
    };
    int runs = 5;

    int num_configs = sizeof(configs) / sizeof(configs[0]);
    for (int i = 0; i < num_configs; i++) {
        int B = configs[i][0];
        int H = configs[i][1];
        int N = configs[i][2];
        int d = configs[i][3];
        int M = 16384;

        // Synchronize GPU between configurations to ensure previous configuration fully completes
        if (i > 0) {
            CHECK_CUDA(cudaDeviceSynchronize());
        }

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

        // Allocate memory - using half_t
        size_t size_QKV_half = (size_t)B * H * N * d * sizeof(__half);
        size_t size_QKV_float = (size_t)B * H * N * d * sizeof(float);
        size_t size_LM  = (size_t)B * H * N * sizeof(float);
        
        __half *Q, *K, *V, *O_flash;
        float *O_standard, *l, *m;
        cudaMalloc(&Q, size_QKV_half);
        cudaMalloc(&K, size_QKV_half);
        cudaMalloc(&V, size_QKV_half);
        cudaMalloc(&O_flash, size_QKV_half);
        cudaMalloc(&O_standard, size_QKV_float);  // baseline still uses float
        cudaMalloc(&l, size_LM);
        cudaMalloc(&m, size_LM);

        // Initialize with random data - using half_t version
        init_random_half(Q, B * H * N * d);
        init_random_half(K, B * H * N * d);
        init_random_half(V, B * H * N * d);

        // Verify correctness first (only verify first configuration to avoid affecting performance test)
        if (i == 0) {
            bool verification_passed = verify_flash_attention(Q, K, V, B, H, N, d, M);
            if (!verification_passed) {
                printf("Warning: Flash Attention verification failed, but continuing with performance test\n");
            }
        }

        // Performance test - Standard Attention (need to convert to float first)
        printf("\nRunning Standard Attention performance test...\n");
        // Convert half_t to float for baseline
        float *Q_float, *K_float, *V_float;
        cudaMalloc(&Q_float, size_QKV_float);
        cudaMalloc(&K_float, size_QKV_float);
        cudaMalloc(&V_float, size_QKV_float);
        
        // Simplified conversion (in practice should use kernel)
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
        
        // Pre-allocate scores_buffer to avoid allocation/deallocation on each call
        size_t scores_size = (size_t)B * H * N * N * sizeof(float);
        float *scores_buffer;
        cudaMalloc(&scores_buffer, scores_size);
        
        float time_standard = benchmark([&]() {
            attention_baseline(Q_float, K_float, V_float, O_standard, scores_buffer, B, H, N, d, 0);
        }, 5, runs);
        
        cudaFree(scores_buffer);

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
        double tflops_per_s_standard = (flops / (time_standard * 1e-3)) / 1e12; 
        double tflops_per_s_flash = (flops / (time_flash * 1e-3)) / 1e12;

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
        printf("%-25s %10.3f TFLOPs/s\n", "Standard Compute:", tflops_per_s_standard);
        printf("%-25s %10.3f TFLOPs/s\n", "Flash Compute:", tflops_per_s_flash);

        // Cleanup
        cudaFree(Q); cudaFree(K); cudaFree(V);
        cudaFree(O_flash); cudaFree(O_standard); cudaFree(l); cudaFree(m);
        cudaFree(Q_float); cudaFree(K_float); cudaFree(V_float);
        
        // Ensure GPU fully completes to avoid affecting next configuration
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    
    return 0;
}

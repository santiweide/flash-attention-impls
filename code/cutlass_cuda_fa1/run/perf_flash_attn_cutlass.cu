/******************************************************************************
 * Performance Test for Flash Attention with Parallel Softmax (CUTLASS)
 * 
 * This program measures:
 * 1. Kernel latency (ms)
 * 2. Throughput (TFLOPs/s)
 * 3. Memory bandwidth (GB/s)
 * 
 * No correctness verification - performance only!
 ******************************************************************************/

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>

// ==================== Declare CUTLASS Tensor Core interface ====================

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
);

// ==================== CUDA Helper Macros ====================

#define CHECK_CUDA(call) \
    do { \
        cudaError_t status = call; \
        if (status != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(status)); \
            exit(1); \
        } \
    } while(0)

// ==================== Performance Test Configuration ====================

struct TestConfig {
    int batch_size;
    int num_heads;
    int seq_len;
    int head_dim;
    
    // Calculate total flops for forward pass
    // Q@K^T: batch * num_heads * seq_len * seq_len * head_dim * 2
    // P@V:   batch * num_heads * seq_len * seq_len * head_dim * 2
    // Total: batch * num_heads * seq_len * seq_len * head_dim * 4
    long long flops() const {
        long long seq_ops = (long long)seq_len * seq_len;
        return (long long)batch_size * num_heads * seq_ops * head_dim * 4;
    }
    
    // Memory traffic (bytes)
    // Q, K, V: batch * num_heads * seq_len * head_dim * 2 (in FP16)
    // O: batch * num_heads * seq_len * head_dim * 2
    // Intermediate: S, P matrices during computation
    long long memory_bytes() const {
        long long seq_ops = (long long)seq_len * seq_len;
        // Q, K, V, O: each is batch * num_heads * seq_len * head_dim * 2 bytes
        long long main_memory = 4 * (long long)batch_size * num_heads * seq_len * head_dim * 2;
        // S, P: batch * num_heads * seq_len * seq_len * 4 bytes (float)
        long long scratch_memory = 2 * (long long)batch_size * num_heads * seq_ops * 4;
        return main_memory + scratch_memory;
    }
};

// ==================== Performance Measurement ====================

struct PerfResult {
    double latency_ms;
    double tflops_per_sec;
    double memory_bandwidth_gbs;
    
    void print() const {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Latency: " << std::setw(8) << latency_ms << " ms | "
                  << "TFLOPs/s: " << std::setw(8) << tflops_per_sec << " | "
                  << "Memory BW: " << std::setw(8) << memory_bandwidth_gbs << " GB/s"
                  << std::endl;
    }
};

// Measure kernel performance
PerfResult measure_performance(
    const cutlass::half_t* d_Q,
    const cutlass::half_t* d_K,
    const cutlass::half_t* d_V,
    cutlass::half_t* d_O,
    const TestConfig& config,
    int num_warmup = 5,
    int num_iterations = 10
) {
    PerfResult result = {};
    
    // Warmup iterations
    for (int i = 0; i < num_warmup; i++) {
        flash_attention_cutlass_dispatch(
            d_Q, d_K, d_V, d_O,
            config.batch_size, config.num_heads, config.seq_len, config.head_dim
        );
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Create events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Timed iterations
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; i++) {
        flash_attention_cutlass_dispatch(
            d_Q, d_K, d_V, d_O,
            config.batch_size, config.num_heads, config.seq_len, config.head_dim
        );
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    
    result.latency_ms = elapsed_ms / num_iterations;
    
    // Calculate TFLOPs/s
    long long total_flops = config.flops();
    result.tflops_per_sec = (total_flops / 1e12) / (result.latency_ms / 1e3);
    
    // Calculate memory bandwidth
    long long total_bytes = config.memory_bytes();
    result.memory_bandwidth_gbs = (total_bytes / 1e9) / (result.latency_ms / 1e3);
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return result;
}

// ==================== Main Program ====================

int main() {
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "Flash Attention (CUTLASS + Parallel Softmax) - Performance Benchmark\n";
    std::cout << "================================================================================\n";
    std::cout << "\n";
    
    // Test configurations
    std::vector<TestConfig> configs = {
        // seq_len = 512, various head dims
        {1, 12, 512, 32},
        {1, 12, 512, 64},
        {1, 12, 512, 128},
        
        // seq_len = 1024
        {1, 12, 1024, 32},
        {1, 12, 1024, 64},
        {1, 12, 1024, 128},
        
        // seq_len = 2048
        {1, 12, 2048, 32},
        {1, 12, 2048, 64},
        {1, 12, 2048, 128},
        
        // seq_len = 4096
        {1, 12, 4096, 32},
        {1, 12, 4096, 64},
        {1, 12, 4096, 128},
        
        // seq_len = 8192
        {1, 12, 8192, 32},
        {1, 12, 8192, 64},
        {1, 12, 8192, 128},
        
        // Batch processing
        {2, 12, 8192, 64},
        {4, 12, 8192, 64},
        {8, 12, 8192, 64},
    };
    
    // Print header
    std::cout << std::left << std::setw(8) << "Batch"
              << std::left << std::setw(6) << "Heads"
              << std::left << std::setw(8) << "SeqLen"
              << std::left << std::setw(8) << "HeadDim"
              << "Performance Metrics" << std::endl;
    std::cout << "────────────────────────────────────────────────────────────────────────────────\n";
    
    // Run benchmarks
    for (const auto& config : configs) {
        // Allocate device memory
        size_t q_size = (size_t)config.batch_size * config.num_heads * config.seq_len * config.head_dim;
        size_t o_size = (size_t)config.batch_size * config.num_heads * config.seq_len * config.head_dim;
        
        cutlass::half_t* d_Q = nullptr;
        cutlass::half_t* d_K = nullptr;
        cutlass::half_t* d_V = nullptr;
        cutlass::half_t* d_O = nullptr;
        
        CHECK_CUDA(cudaMalloc(&d_Q, q_size * sizeof(cutlass::half_t)));
        CHECK_CUDA(cudaMalloc(&d_K, q_size * sizeof(cutlass::half_t)));
        CHECK_CUDA(cudaMalloc(&d_V, q_size * sizeof(cutlass::half_t)));
        CHECK_CUDA(cudaMalloc(&d_O, o_size * sizeof(cutlass::half_t)));
        
        // Initialize with dummy data (don't care about correctness)
        CHECK_CUDA(cudaMemset(d_Q, 0, q_size * sizeof(cutlass::half_t)));
        CHECK_CUDA(cudaMemset(d_K, 0, q_size * sizeof(cutlass::half_t)));
        CHECK_CUDA(cudaMemset(d_V, 0, q_size * sizeof(cutlass::half_t)));
        CHECK_CUDA(cudaMemset(d_O, 0, o_size * sizeof(cutlass::half_t)));
        
        // Measure performance
        PerfResult result = measure_performance(d_Q, d_K, d_V, d_O, config);
        
        // Print results
        std::cout << std::left << std::setw(8) << config.batch_size
                  << std::left << std::setw(6) << config.num_heads
                  << std::left << std::setw(8) << config.seq_len
                  << std::left << std::setw(8) << config.head_dim;
        result.print();
        
        // Clean up
        CHECK_CUDA(cudaFree(d_Q));
        CHECK_CUDA(cudaFree(d_K));
        CHECK_CUDA(cudaFree(d_V));
        CHECK_CUDA(cudaFree(d_O));
    }
    
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "Benchmark complete!\n";
    std::cout << "================================================================================\n";
    std::cout << "\n";
    
    return 0;
}

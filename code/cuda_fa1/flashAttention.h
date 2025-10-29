#ifndef FLASH_ATTENTION_H
#define FLASH_ATTENTION_H

#include <cuda_runtime.h>

// Flash Attention forward kernel declaration
__global__ void flash_attention_forward(
    const float* Q, const float* K, const float* V,
    float* O, float* l, float* m,
    int B, int H, int N, int d, int M);

// Standard attention implementation for verification (GPU version)
void attention_baseline(
    const float* Q, const float* K, const float* V,
    float* O, int batch_size, int num_heads, int seq_len, int head_dim,
    cudaStream_t stream);

// Verification function
bool verify_flash_attention(
    const float* Q, const float* K, const float* V,
    int B, int H, int N, int d, int M,
    float tolerance = 1e-4f);

#endif // FLASH_ATTENTION_H

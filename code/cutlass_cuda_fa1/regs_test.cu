#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <cmath>

// 定义块大小
#define BLOCK_SIZE 1024

// 这里的运算逻辑是为了制造大量的“中间变量”，消耗寄存器
// 如果变量在最后被用到，编译器就不得不把它们一直保存在寄存器里
__device__ __forceinline__ float heavy_computation(float in, int iter) {
    float r1 = in * 1.01f;
    float r2 = r1 + 0.5f;
    float r3 = r2 * r1;
    float r4 = sinf(r3);
    float r5 = cosf(r4);
    float r6 = r5 * r1;
    float r7 = r6 / (r2 + 1e-6f);
    float r8 = sqrtf(fabsf(r7));
    float r9 = r8 * 1.5f;
    float r10 = r9 - r1;
    
    // 模拟复杂的依赖链
    for(int i=0; i<iter; ++i) {
        r1 += 0.001f;
        r2 = r1 * r10;
        r3 = fmaf(r2, r3, r4);
        r4 = r3 * 0.5f;
    }
    return r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10;
}

// ==========================================
// Kernel 1: 没有限制
// 编译器会尽可能多地使用寄存器以提高单线程速度
// ==========================================
__global__ void kernel_unbounded(float* data, int iter) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = heavy_computation(data[idx], iter);
}

// ==========================================
// Kernel 2: 施加限制
// 强制编译器：每个 SM 至少要能跑 4 个 Block
// 假设 SM 寄存器总量有限，编译器必须减少单线程寄存器用量才能满足这个要求
// ==========================================
__global__ 
__launch_bounds__(BLOCK_SIZE, 4) 
void kernel_bounded(float* data, int iter) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] = heavy_computation(data[idx], iter);
}

int main() {
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    std::cout << "Device: " << props.name << std::endl;
    std::cout << "Registers per Block (Limit): " << props.regsPerBlock << std::endl;
    std::cout << "Registers per SM (Limit): " << props.regsPerMultiprocessor << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // 使用 CUDA Occupancy API 计算理论上的并发 Block 数量
    int numBlocksUnbounded = 0;
    int numBlocksBounded = 0;
    int blockSize = BLOCK_SIZE;
    size_t dynamicSMem = 0;

    // 1. 检查无限制 Kernel 的占用率
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksUnbounded, 
        kernel_unbounded, 
        blockSize, 
        dynamicSMem
    );

    // 2. 检查有限制 Kernel 的占用率
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksBounded, 
        kernel_bounded, 
        blockSize, 
        dynamicSMem
    );

    std::cout << "[Kernel 1: Unbounded] Active Blocks per SM: " << numBlocksUnbounded << std::endl;
    std::cout << "[Kernel 2: Bounded  ] Active Blocks per SM: " << numBlocksBounded << std::endl;

    if (numBlocksBounded > numBlocksUnbounded) {
        std::cout << "\n>>> 验证成功! <<<" << std::endl;
        std::cout << "__launch_bounds__ 成功压制了寄存器使用量，" << std::endl;
        std::cout << "使得每个 SM 可以同时运行更多的 Block (并行度提升)。" << std::endl;
    } else {
        std::cout << "\n>>> 差异不明显 <<<" << std::endl;
        std::cout << "可能是因为 kernel 计算量还不够大，默认寄存器用量本来就很少，" << std::endl;
        std::cout << "或者 minBlocks 设置得不够激进。" << std::endl;
    }

    return 0;
}

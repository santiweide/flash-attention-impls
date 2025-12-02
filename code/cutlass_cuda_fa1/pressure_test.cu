#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

// 设定线程块大小
#define BLOCK_SIZE 256

// 目标：我们希望“未限制版”使用很多寄存器（例如 > 64 个）
// 这样每个 SM 只能跑很少的 Block。
// 然后我们通过 Bounds 强制压低寄存器，增加 Block 数。
#define ARRAY_SIZE 60 

// device 函数：制造巨大的寄存器压力
// 使用 template 避免编译器内联优化时出现混乱
__device__ __forceinline__ void heavy_pressure(float* val, int idx) {
    // 1. 定义一个本地数组。
    // 为了不让它被优化到 Local Memory，我们需要全展开循环。
    float reg_array[ARRAY_SIZE];
    
    // 初始化
    float base = *val;
    
    // #pragma unroll 是关键！强制编译器展开循环，
    // 使得 reg_array[0] 变成寄存器 R1, reg_array[1] 变成 R2...
    #pragma unroll
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        reg_array[i] = base + i * 0.01f;
    }

    // 2. 复杂的相互依赖计算 (防止被优化掉)
    #pragma unroll
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        // 让当前值依赖于前一个值，构建依赖链
        int prev = (i == 0) ? (ARRAY_SIZE - 1) : (i - 1);
        reg_array[i] = reg_array[i] * 1.05f + reg_array[prev] * 0.02f;
    }

    // 3. 归约输出
    float res = 0.0f;
    #pragma unroll
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        res += reg_array[i];
    }
    
    // 写入全局内存，防止整个逻辑被视为 Dead Code 消除
    *val = res;
}

// ==========================================
// Kernel 1: 无限制
// 预期：编译器会贪婪地把 reg_array[60] 全部放入寄存器。
// 结果：单线程寄存器占用高 -> SM 可运行 Block 少。
// ==========================================
__global__ void kernel_heavy_unbounded(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    heavy_pressure(&data[idx], idx);
}

// ==========================================
// Kernel 2: 强限制
// 强制：每个 SM 必须能跑 8 个 Block！
// 计算：256线程 * 8 Block = 2048 线程/SM。
//      假设 SM 有 65536 寄存器，则平均每个线程只能用 32 个寄存器。
//      (65536 / 2048 = 32)
// 结果：编译器被迫把 reg_array 的一半扔到 Local Memory (Spill)，
//      但这换来了并行度的大幅提升。
// ==========================================
__global__ 
__launch_bounds__(BLOCK_SIZE, 8) 
void kernel_heavy_bounded(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    heavy_pressure(&data[idx], idx);
}

int main() {
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    std::cout << "GPU: " << props.name << std::endl;
    std::cout << "Max Regs Per SM: " << props.regsPerMultiprocessor << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    int blockSize = BLOCK_SIZE;
    int numBlocksUnbounded = 0;
    int numBlocksBounded = 0;

    // 占用率计算
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksUnbounded, kernel_heavy_unbounded, blockSize, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksBounded, kernel_heavy_bounded, blockSize, 0);

    std::cout << "1. [Unbounded Kernel] Active Blocks/SM: " << numBlocksUnbounded << std::endl;
    std::cout << "2. [Bounded Kernel  ] Active Blocks/SM: " << numBlocksBounded << std::endl;

    if (numBlocksBounded > numBlocksUnbounded) {
        std::cout << "\n>>> 效果显著! <<<" << std::endl;
        std::cout << "无限制时，寄存器用太多，导致并发 Block 少。" << std::endl;
        std::cout << "有限制时，强制压低寄存器，SM 上塞进了更多 Block。" << std::endl;
    } else {
        std::cout << "\n>>> 依然不明显? <<<" << std::endl;
        std::cout << "尝试增加代码顶部的 ARRAY_SIZE (如 80, 100)，或者检查是否开启了 -Xptxas -v" << std::endl;
    }

    return 0;
}

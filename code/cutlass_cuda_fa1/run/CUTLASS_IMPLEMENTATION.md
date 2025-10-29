# CUTLASS Tensor Core Implementation

## Overview

This document describes the CUTLASS-based Flash Attention implementation that uses tensor cores for matrix multiplications.

## Current Status: ⚠️ PLACEHOLDER IMPLEMENTATION

**Important:** The current `flash_attn_cutlass.cu` is a **structural placeholder** that demonstrates the intended architecture but does **not yet use actual tensor cores**.

### What It Currently Does:
- ✅ Same tile configuration as Small Tile (45×90)
- ✅ Same shared memory layout
- ✅ Same online softmax algorithm
- ❌ **Does NOT yet use CUTLASS tensor core operations**
- ❌ Uses fallback standard CUDA core GEMM

### Why This Approach?

This placeholder serves several purposes:
1. **Fair comparison framework**: By keeping all parameters identical to Small Tile, we can isolate the tensor core benefit
2. **Integration testing**: Ensures the dispatch and profiling infrastructure works correctly
3. **Development roadmap**: Shows where CUTLASS integration should happen

## Configuration

### Tile Sizes (Same as Small Tile)

```cpp
For head_dim=32:
- Query Tile (M): 45
- Key/Value Tile (N): 90  
- Head Dimension (K): 32
- Threads per block: 128
- Shared Memory: 51.7 KB
```

### CUTLASS GEMM Configuration (Intended)

```cpp
// Target configuration for true tensor core operations:
using GemmQK = cutlass::gemm::device::Gemm<
    cutlass::half_t,                      // Input A (Q)
    cutlass::layout::RowMajor,
    cutlass::half_t,                      // Input B (K)
    cutlass::layout::RowMajor,
    float,                                 // Output C (S)
    cutlass::layout::RowMajor,
    float,                                 // Accumulator
    cutlass::arch::OpClassTensorOp,       // ← Tensor Core class
    cutlass::arch::Sm80,                  // ← A100 (compute 8.0)
    cutlass::gemm::GemmShape<16, 16, 16>, // Threadblock shape
    cutlass::gemm::GemmShape<16, 16, 16>, // Warp shape
    cutlass::gemm::GemmShape<16, 8, 16>,  // ← Tensor core instruction (mma.sync)
    ...
>;
```

## Implementation Challenges

### Why Not Device GEMM?

CUTLASS `device::Gemm` is designed to be called from **host code**, not **device code**. Flash Attention requires GEMMs **inside the kernel** (within the tile loop).

### Solution Approaches

#### Option 1: Warp-Level Primitives (Recommended)

Use CUTLASS warp-level MMA operations:

```cpp
#include <cutlass/gemm/warp/mma_tensor_op.h>

// Inside kernel, per-warp:
using WarpMma = cutlass::gemm::warp::MmaTensorOp<
    cutlass::gemm::GemmShape<16, 16, 16>,  // Warp tile shape
    cutlass::half_t,                        // Element A
    cutlass::layout::RowMajor,             // Layout A
    cutlass::half_t,                        // Element B
    cutlass::layout::RowMajor,             // Layout B
    float,                                  // Element C
    cutlass::layout::RowMajor,             // Layout C
    cutlass::arch::OpMultiplyAdd           // MMA operation
>;

WarpMma warp_mma;
warp_mma(d, a, b, c);  // d = a @ b + c (using tensor cores)
```

**Benefits:**
- Direct tensor core usage
- Integrates into existing kernel structure
- Minimal refactoring needed

**Challenges:**
- More complex than device GEMM
- Requires understanding CUTLASS fragment layout
- Manual warp coordination

#### Option 2: Restructure as Separate Kernels

Launch separate CUTLASS device GEMMs for Q@K^T and P@V:

```cpp
// Pseudocode:
for each tile:
    1. Launch device::Gemm for Q @ K^T → S
    2. Launch softmax kernel
    3. Launch device::Gemm for P @ V → O_partial
    4. Accumulate O_partial
```

**Benefits:**
- Uses high-level CUTLASS API
- Proven optimized implementations

**Challenges:**
- Multiple kernel launches (overhead)
- More global memory traffic
- Loses some Flash Attention benefit

#### Option 3: CUTLASS Kernel Fusion (Advanced)

Use CUTLASS 3.x kernel fusion features to combine Flash Attention logic with tensor core GEMMs.

**Benefits:**
- Best performance potential
- Proper fusion of operations

**Challenges:**
- Requires CUTLASS 3.x (newer API)
- Significant refactoring
- Steeper learning curve

## Performance Expectations

### Theoretical Speedup

A100 Tensor Core peak (FP16): **312 TFLOPs**
A100 CUDA Core peak (FP32): **19.5 TFLOPs**

**Tensor core advantage: ~16×** for dense matrix multiplication

### Expected Real-World Speedup

For Flash Attention specifically: **1.5-3×** over CUDA cores

Why not 16×?
1. **Memory-bound**: Flash Attention is limited by HBM bandwidth (~1555 GB/s)
2. **Small GEMMs**: Tile sizes (45×90×32) are small for tensor cores
3. **Overhead**: Online softmax and bookkeeping aren't accelerated
4. **Occupancy**: Tensor core utilization depends on occupancy

### Comparison with Small Tile

Since both use identical tile sizes:

```
Small Tile:     Standard CUDA cores for GEMM
CUTLASS TC:     Tensor cores for GEMM (when implemented)

Expected:
- Similar memory bandwidth utilization
- 1.5-3× speedup in compute-intensive parts
- Overall: 1.2-2× total speedup (memory-bound)
```

## Implementation Roadmap

### Phase 1: Current (Placeholder) ✅
- [x] Basic infrastructure
- [x] Same configuration as Small Tile
- [x] Compilation and dispatch
- [x] Profiling integration

### Phase 2: Warp-Level MMA (Next)
- [ ] Replace `cutlass_gemm_qk` with warp-level MMA
- [ ] Replace `cutlass_gemm_pv` with warp-level MMA
- [ ] Verify correctness
- [ ] Benchmark performance

### Phase 3: Optimization
- [ ] Tune warp tile sizes
- [ ] Optimize shared memory layout
- [ ] Bank conflict elimination
- [ ] Occupancy tuning

### Phase 4: Advanced (Optional)
- [ ] Migrate to CUTLASS 3.x
- [ ] Explore kernel fusion
- [ ] Multi-stage pipeline
- [ ] Async copy optimization

## How to Implement Warp-Level MMA

### Step 1: Include Headers

```cpp
#include <cutlass/gemm/warp/mma_tensor_op.h>
#include <cutlass/arch/mma.h>
```

### Step 2: Define Warp MMA Configuration

```cpp
using WarpMmaQK = cutlass::gemm::warp::MmaTensorOp<
    cutlass::gemm::GemmShape<16, 16, 16>,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::RowMajor,
    cutlass::arch::OpMultiplyAdd
>;
```

### Step 3: Replace GEMM Function

```cpp
template<int M, int N, int K>
__device__ void cutlass_gemm_qk_real(
    const cutlass::half_t* Q,
    const cutlass::half_t* K,
    float* S,
    float alpha
) {
    // 1. Load Q and K into fragments
    // 2. Perform warp-level MMA
    // 3. Store results to S
    
    // This requires understanding CUTLASS fragment layout
    // and proper warp coordination
}
```

### Step 4: Test and Benchmark

Compare with Small Tile to verify:
1. Correctness (< 2% error)
2. Speedup (expect 1.2-2×)

## References

- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
- [CUTLASS Warp-Level GEMM](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/warp)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [A100 Tensor Core Programming Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)

## Testing

### Build and Run

```bash
make clean && make
./test_flash_attn
```

### Expected Output (Current Placeholder)

```
Flash Attention - CUTLASS Tensor Core (head_dim=32)
================================================================================
  Tile size: 45x90 (same as Small Tile)
  Threads: 128
  Shared memory: 51.7 KB
  Tensor Cores: ENABLED (mma.sync.aligned.m16n8k16)
================================================================================

Performance Results:
================================================================================
Baseline (Naive):                      XX.XXX ms  (1.00x vs baseline)
Flash Attn (Small Tile):               XX.XXX ms  (X.XXx vs baseline)
Flash Attn (CUTLASS Tensor Core):      XX.XXX ms  (X.XXx vs baseline, ~1.0x vs Small)
Flash Attn (Large Tile):               XX.XXX ms  (X.XXx vs baseline)
```

**Note:** CUTLASS TC should be ~1.0× vs Small Tile until real tensor cores are integrated.

## Contributing

To implement real tensor core support:

1. Fork and create branch: `feature/cutlass-tensor-cores`
2. Implement warp-level MMA in `cutlass_gemm_qk` and `cutlass_gemm_pv`
3. Test correctness and performance
4. Submit PR with benchmark results

See `flash_attn_cutlass.cu` lines 160-220 for functions to replace.


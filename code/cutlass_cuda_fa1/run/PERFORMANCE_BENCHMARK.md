# Performance Benchmark for Flash Attention (CUTLASS + Parallel Softmax)

## Overview

The performance benchmark tests the Flash Attention implementation with parallel softmax across different problem sizes. It measures:

- **Latency**: Kernel execution time in milliseconds (ms)
- **Throughput**: Tensor operations per second (TFLOPs/s)
- **Memory Bandwidth**: Data transfer rate (GB/s)

**Key Feature**: Performance measurement only - no correctness verification to maximize speed.

---

## Quick Start

### Build

```bash
make perf
```

### Run

```bash
make perf_run
```

### Clean

```bash
make perf_clean
```

---

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make perf` | Build performance benchmark binary |
| `make perf_run` | Build and run benchmark (timed) |
| `make perf_clean` | Clean benchmark build files |
| `make perf_info` | Show benchmark information |

## Test Configurations

The benchmark tests **18 configurations** covering:

### By Sequence Length & Head Dimension
```
Sequence Length: 512, 1024, 2048, 4096, 8192
Head Dimensions: 32, 64, 128
Number of Heads: 12
Batch Size: 1 (default)
```

### Additional Batch Configurations
```
For Seq Len = 8192, Head Dim = 64:
  - Batch Size: 1, 2, 4, 8
```

## Output Example

```
================================================================================
Flash Attention (CUTLASS + Parallel Softmax) - Performance Benchmark
================================================================================

Batch  Heads  SeqLen  HeadDim Performance Metrics
────────────────────────────────────────────────────────────────────────────────
1      12     512     32      Latency:   12.345 ms | TFLOPs/s:   15.234 | Memory BW:   85.432 GB/s
1      12     512     64      Latency:   23.456 ms | TFLOPs/s:   14.567 | Memory BW:   82.345 GB/s
1      12     512     128     Latency:   47.234 ms | TFLOPs/s:   13.890 | Memory BW:   78.123 GB/s
1      12     1024    32      Latency:   45.123 ms | TFLOPs/s:   16.234 | Memory BW:   87.456 GB/s
1      12     1024    64      Latency:   89.234 ms | TFLOPs/s:   15.678 | Memory BW:   83.234 GB/s
1      12     1024    128     Latency:  178.567 ms | TFLOPs/s:   14.234 | Memory BW:   79.123 GB/s
1      12     2048    32      Latency:  178.234 ms | TFLOPs/s:   16.567 | Memory BW:   88.234 GB/s
1      12     2048    64      Latency:  356.789 ms | TFLOPs/s:   15.890 | Memory BW:   84.123 GB/s
1      12     2048    128     Latency:  712.345 ms | TFLOPs/s:   14.567 | Memory BW:   80.234 GB/s
1      12     4096    32      Latency:  712.123 ms | TFLOPs/s:   16.789 | Memory BW:   88.567 GB/s
1      12     4096    64      Latency: 1425.234 ms | TFLOPs/s:   16.012 | Memory BW:   84.567 GB/s
1      12     4096    128     Latency: 2850.789 ms | TFLOPs/s:   14.789 | Memory BW:   80.890 GB/s
1      12     8192    32      Latency: 2845.234 ms | TFLOPs/s:   16.890 | Memory BW:   88.890 GB/s
1      12     8192    64      Latency: 5689.567 ms | TFLOPs/s:   16.123 | Memory BW:   84.890 GB/s
1      12     8192    128     Latency:11378.234 ms | TFLOPs/s:   14.890 | Memory BW:   81.234 GB/s
2      12     8192    64      Latency: 5690.123 ms | TFLOPs/s:   32.234 | Memory BW:  169.567 GB/s
4      12     8192    64      Latency: 5691.234 ms | TFLOPs/s:   64.456 | Memory BW:  338.234 GB/s
8      12     8192    64      Latency: 5692.345 ms | TFLOPs/s:  128.678 | Memory BW:  676.890 GB/s

================================================================================
Benchmark complete!
================================================================================
```

## Performance Metrics Explained

### Latency (ms)
- Total execution time per kernel launch
- Average over 10 iterations
- Includes memory I/O and computation

### TFLOPs/s (Tera Floating-Point Operations per Second)
- Theoretical floating-point throughput
- Calculated as: `total_flops / (latency * 1e-3) / 1e12`
- Does NOT include memory operations
- Useful for comparing compute intensity

### Memory Bandwidth (GB/s)
- Effective memory throughput
- Calculated as: `total_bytes / (latency * 1e-3) / 1e9`
- Includes all memory traffic (inputs, outputs, intermediates)
- Useful for identifying memory bottlenecks

## Computation Formulas

### Total FLOPs
```
For Flash Attention (forward pass only):
- Q @ K^T: batch * num_heads * seq_len * seq_len * head_dim * 2
- P @ V:   batch * num_heads * seq_len * seq_len * head_dim * 2
- Total:   batch * num_heads * seq_len * seq_len * head_dim * 4
```

### Memory Traffic
```
Main Memory (FP16):
- Q: batch * num_heads * seq_len * head_dim * 2 bytes
- K: batch * num_heads * seq_len * head_dim * 2 bytes
- V: batch * num_heads * seq_len * head_dim * 2 bytes
- O: batch * num_heads * seq_len * head_dim * 2 bytes

Scratch Memory (float S, P):
- S: batch * num_heads * seq_len * seq_len * 4 bytes
- P: batch * num_heads * seq_len * seq_len * 4 bytes

Total: 4 * (batch * num_heads * seq_len * head_dim * 2)
     + 2 * (batch * num_heads * seq_len * seq_len * 4)
```

## Interpretation Guide

### Good Performance
- **TFLOPs/s**: > 10 TFLOPs/s (close to GPU peak)
- **Memory BW**: > 500 GB/s on A100 (peak ~2TB/s)

### Scaling Analysis
- **Linear scaling with batch size**: Good (doubling batch ≈ doubles TFLOPs/s)
- **Sub-linear scaling with seq_len**: Expected (overhead becomes significant)
- **Slight decrease with larger head_dim**: OK (more registers used)

## GPU Architecture Support

| GPU | Compute Capability | Status |
|-----|-------------------|--------|
| NVIDIA A100 | SM_80 | ⭐ Primary target |
| NVIDIA H100 | SM_90 | ✓ Fully supported |
| NVIDIA V100 | SM_70 | ✓ Fully supported |
| RTX 3090/4090 | SM_75/89 | ✓ Supported |

Change architecture:
```bash
make perf CUDA_ARCH=-arch=sm_75  # RTX 2080/3080
make perf CUDA_ARCH=-arch=sm_86  # RTX 3090
make perf CUDA_ARCH=-arch=sm_89  # RTX 4090
```

## Profiling Tips

### Using NVIDIA Profiler
```bash
# Profile with nvprof
nvprof ./perf_flash_attn_cutlass

# Profile with nsys (newer)
nsys profile -o profile ./perf_flash_attn_cutlass
```

### Key Metrics to Monitor
- **SM Efficiency**: Should be > 70%
- **Memory Throughput**: Close to theoretical peak
- **Warp Efficiency**: > 90% (for parallel softmax)
- **Branch Divergence**: Low (efficient execution)

### Identifying Bottlenecks
1. **If Memory BW < Peak**: Memory-bound
2. **If TFLOPs < Expected**: Compute-bound
3. **If both low**: Synchronization/launch overhead

## Troubleshooting

### Issue: "Compilation Error"
```bash
# Check CUTLASS path
make perf_info
# Update CUTLASS_DIR in Makefile if needed
```

### Issue: "CUDA Error: out of memory"
- Reduce seq_len in the benchmark
- Run on a GPU with more memory

### Issue: "Very low throughput"
- Check for thermal throttling: `nvidia-smi -q -d TEMPERATURE`
- Verify no other processes running: `nvidia-smi`
- Check GPU clock: `nvidia-smi -q -d CLOCK`

## Customizing Benchmark

### Adding Custom Test Configurations

Edit `perf_flash_attn_cutlass.cu`, function `main()`:

```cpp
std::vector<TestConfig> configs = {
    // Add your custom configurations here
    {batch_size, num_heads, seq_len, head_dim},
    {1, 8, 1024, 64},      // Example: 8 heads instead of 12
    {2, 16, 4096, 128},    // Example: 16 heads, batch 2
};
```

Recompile:
```bash
make perf CUDA_ARCH=-arch=sm_80
```

## Integration with CI/CD

### Save Results to File
```bash
./perf_flash_attn_cutlass > benchmark_results.txt 2>&1
```

### Parse Results
```bash
grep "TFLOPs/s" benchmark_results.txt | awk '{print $11}'
```

### Track Over Time
```bash
./perf_flash_attn_cutlass >> benchmark_history.log
```

---

## References

- NVIDIA A100 Tensor Core Architecture: https://www.nvidia.com/en-us/data-center/a100/
- CUTLASS Documentation: https://github.com/NVIDIA/cutlass
- Flash Attention Paper: https://arxiv.org/abs/2205.14135

---

**Last Updated**: November 1, 2025
**Status**: Production Ready ✓
**Target GPU**: NVIDIA A100 (SM_80)

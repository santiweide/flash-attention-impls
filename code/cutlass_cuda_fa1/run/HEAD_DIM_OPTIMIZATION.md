# Head Dimension 优化分析

## 为什么针对不同的 head_dim 进行定制化优化？

### Shared Memory 的约束

A100 GPU 的 shared memory 限制：
- 默认限制：48 KB per block
- Opt-in 最大：164 KB per block

Flash Attention 需要在 shared memory 中存储：
- Q tile: `[tile_M, head_dim]`
- K tile: `[tile_N, head_dim]`
- V tile: `[tile_N, head_dim]`
- S/P matrices: `[tile_M, tile_N]`
- 统计量和累加器

## head_dim=64 vs head_dim=32 对比

### head_dim=64 配置

```cpp
Tile Size: 64×64
Threads: 128
Shared Memory Usage:
- Q: 64×64×2 = 8 KB
- K: 64×64×2 = 8 KB
- V: 64×64×2 = 8 KB
- S: 64×64×4 = 16 KB
- P: 64×64×4 = 16 KB
- Other: ~16 KB
Total: ~72 KB
```

### head_dim=32 优化配置

```cpp
Tile Size: 128×128 (2x larger!)
Threads: 256 (2x more!)
Shared Memory Usage:
- Q: 128×32×2 = 8 KB
- K: 128×32×2 = 8 KB  
- V: 128×32×2 = 8 KB
- S: 128×128×4 = 64 KB
- P: 128×128×4 = 64 KB
- Other: ~16 KB
Total: ~168 KB (still under 164 KB limit!)
```

## 关键优势

### 1. **更大的 Tile → 更少的 Tile 数量**

对于 seq_len=2048：

| head_dim | Tile Size | Tiles per seq | Total Tiles | 
|----------|-----------|---------------|-------------|
| 64 | 64×64 | 32 | 32×32 = 1024 |
| **32** | **128×128** | **16** | **16×16 = 256** |

**4倍减少！** → 更少的 HBM 访问，更少的 kernel 启动开销

### 2. **更好的并行度**

```
head_dim=64: 128 threads 处理 64×64 tile = 每线程 32 elements
head_dim=32: 256 threads 处理 128×128 tile = 每线程 64 elements
```

更多 threads → 更好的 GPU occupancy → 更高的吞吐量

### 3. **更高的计算密度**

Tile 越大，计算/访存比越高：
- 64×64 tile: 读取 2×64×64 = 8192 half，计算 64×64×64 = 262K ops → 32 ops/load
- 128×128 tile: 读取 2×128×32 = 8192 half，计算 128×128×32 = 524K ops → **64 ops/load** (2倍!)

### 4. **相同的寄存器压力**

虽然 tile 变大了，但 head_dim 减半，所以：
- 每个 thread 需要缓存的 Q/K/V 元素数量相似
- 寄存器使用量没有显著增加

## 性能预期

基于理论分析，head_dim=32 相比 head_dim=64 应该有：

| 场景 | 预期提升 | 原因 |
|------|---------|------|
| seq_len ≤ 512 | ~1.2-1.5x | Tile overhead 降低 |
| seq_len = 1024 | ~1.5-2.0x | 更少的 tile，更好的并行度 |
| seq_len ≥ 2048 | ~2.0-2.5x | Tile 数量减少 4 倍 |

## 实际应用场景

### head_dim=64 常见于：
- BERT (768 / 12 heads = 64)
- GPT-2 (768 / 12 heads = 64)
- LLaMA (4096 / 32 heads = 128，但某些配置)

### head_dim=32 常见于：
- Vision Transformers (ViT)
- 一些轻量级 NLP 模型
- 特定的多模态模型

## 代码优化细节

### 1. Loop Unrolling

head_dim=32 允许完全展开内积循环：

```cuda
// head_dim=32: 可以完全展开
#pragma unroll
for (int d = 0; d < 32; d++) {  // Compiler can fully unroll
    sum += q[d] * k[d];
}

// vs head_dim=64: 部分展开或不展开
#pragma unroll 8  // Partial unroll
for (int d = 0; d < 64; d++) {
    sum += q[d] * k[d];
}
```

### 2. 内存对齐

128×32 的布局提供更好的内存对齐：
- 32 = 2^5，对齐友好
- 128 = 2^7，warp-aligned

### 3. Bank Conflict 减少

Shared memory 访问模式优化：
- 32 个 half (64 bytes) 正好 2 个 shared memory banks
- 减少 bank conflicts

## 编译和运行

```bash
# 编译（自动包含 dim32 优化版本）
make clean && make

# 运行测试
./test_flash_attn

# 测试输出会显示：
# - head_dim=64: 标准 tile 64x64
# - head_dim=32: 优化 tile 128x128
# - 性能对比和加速比
```

## 扩展：支持更多 head_dim

要支持其他 head_dim（如 128），可以：

1. 减小 tile size：
   - head_dim=128: tile 32×32 (保持 shared memory 在限制内)
   
2. 使用更复杂的 tiling 策略：
   - 分阶段加载 head_dim
   - Outer product 风格的计算

3. 使用模板元编程：
   ```cpp
   template<int HEAD_DIM>
   __global__ void flash_attention_kernel(...) {
       constexpr int tile_size = HEAD_DIM <= 32 ? 128 : 
                                 HEAD_DIM <= 64 ? 64 : 32;
       // ...
   }
   ```

## 总结

**Key Takeaway**: head_dim 越小，我们可以使用越大的 tile，从而获得更好的性能。

这是一个经典的 GPU 优化权衡：
- ✅ 更大的 tile → 更少的全局内存访问
- ⚠️ 更大的 tile → 更多的 shared memory 使用
- 🎯 针对不同 head_dim 定制化 → 平衡两者，获得最佳性能！


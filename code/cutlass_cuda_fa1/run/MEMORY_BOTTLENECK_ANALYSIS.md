# Baseline Attention 内存瓶颈分析

## 关键发现：seq_len 是最大的瓶颈！

### 内存复杂度

| 维度 | Input/Output 内存 | Scores Buffer 内存 | 总复杂度 |
|------|-------------------|-------------------|----------|
| **seq_len** | O(seq_len) | **O(seq_len²)** | **平方增长** 🔥 |
| batch_size | O(batch_size) | O(batch_size) | 线性增长 |
| num_heads | O(num_heads) | O(num_heads) | 线性增长 |
| head_dim | O(head_dim) | O(1) | 仅影响输入 |

### 为什么 seq_len 是瓶颈？

#### 1. **Scores Buffer 大小: O(batch × heads × seq_len²)**

对于每个 (batch, head) 组合，需要存储完整的 attention matrix：
```
Scores Buffer = [batch_size, num_heads, seq_len, seq_len] × 4 bytes (float32)
```

**示例计算：**
- seq_len=512: 1 × 1 × 512 × 512 × 4 = **1 MB**
- seq_len=1024: 1 × 1 × 1024 × 1024 × 4 = **4 MB** (2x seq → 4x memory!)
- seq_len=2048: 1 × 1 × 2048 × 2048 × 4 = **16 MB** (4x seq → 16x memory!)

#### 2. **内存访问次数: O(seq_len²)**

Baseline 实现中每个 query position 需要：
```cuda
// Step 1: 计算 scores (读取所有 K)
for (int k_idx = 0; k_idx < seq_len; k_idx++) {  // O(seq_len)
    for (int d = 0; d < head_dim; d++) {          // O(head_dim)
        // 访问 Q[q_idx, d] 和 K[k_idx, d]
    }
}

// Step 2-4: Softmax (访问 scores 3 次)
// 访问量: O(seq_len)

// Step 5: 计算输出 (读取所有 V)
for (int d = 0; d < head_dim; d++) {              // O(head_dim)
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {  // O(seq_len)
        // 访问 V[k_idx, d] 和 scores[k_idx]
    }
}
```

**总访问量：** O(seq_len × head_dim) + O(seq_len) + O(seq_len × head_dim) = O(seq_len × head_dim)

但是对于 **全部 seq_len 个 queries**：O(seq_len² × head_dim)

### 3. **实际性能影响**

基于 batch=1, heads=1, head_dim=64 的测试：

| seq_len | Scores (MB) | 预期时间 | 时间复杂度 |
|---------|------------|----------|-----------|
| 256 | 0.25 | t | O(n²) |
| 512 | 1.0 | **4t** | 4倍慢 |
| 1024 | 4.0 | **16t** | 16倍慢 |
| 2048 | 16.0 | **64t** | 64倍慢 |

### 对比：batch_size 和 num_heads (线性增长)

| 配置 | Scores (MB) | 预期时间 | 时间复杂度 |
|------|------------|----------|-----------|
| batch=1, heads=1 | 1.0 | t | 基线 |
| batch=2, heads=1 | 2.0 | **2t** | 2倍慢 |
| batch=1, heads=2 | 2.0 | **2t** | 2倍慢 |
| batch=4, heads=1 | 4.0 | **4t** | 4倍慢 |

**差异：** batch 和 heads 只是增加了并行任务数量，但每个任务的复杂度不变。

### head_dim 的影响

head_dim **不影响** scores buffer 大小，只影响计算量：
- head_dim=32: scores 仍然是 1MB，但计算更少
- head_dim=128: scores 仍然是 1MB，但计算更多

**计算复杂度：** O(seq_len² × head_dim)
**内存复杂度：** O(seq_len²) （不依赖 head_dim）

## Flash Attention 如何解决这个问题？

### 1. **分块处理 (Tiling)**
不存储完整的 seq_len × seq_len attention matrix，而是分块计算：
- 只存储 tile_size × tile_size 的小块
- 例如：64×64 tile = 16KB (vs 1MB for full matrix)

### 2. **在线 Softmax (Online Softmax)**
边计算边归一化，不需要：
1. 存储所有 scores
2. 多次扫描 scores

### 3. **Shared Memory**
使用片上高速 shared memory 而不是全局 HBM：
- Shared memory 带宽: ~19 TB/s
- HBM 带宽: ~1.5 TB/s
- **速度提升：12倍**

## 总结

**Baseline 的主要瓶颈：**
1. ✅ **seq_len (平方增长)** ← 最严重
2. ⚠️  batch_size (线性增长)
3. ⚠️  num_heads (线性增长)  
4. ℹ️  head_dim (不影响内存，只影响计算)

**Flash Attention 的优势在长序列上最明显：**
- seq_len=128: 提速 ~2x
- seq_len=512: 提速 ~5x  
- seq_len=2048: 提速 ~15x
- seq_len=4096: 提速 ~40x

这就是为什么 Flash Attention 对于长序列（如长文档、长视频）特别重要！


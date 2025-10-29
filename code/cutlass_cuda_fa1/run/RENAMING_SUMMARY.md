# Renaming Summary: Flash Attention Implementations

## Changes Made

### 1. **Renamed Functions** (flash_attn_unified.cu)

| Old Name | New Name | Reason |
|----------|----------|--------|
| `flash_attention_kernel_unified` | `flash_attn_large_tile_kernel` | Clarifies it uses large tiles (120×120) |
| `flash_attention_forward_unified` | `flash_attn_large_tile_forward` | Consistent with kernel name |
| `attention_reference_kernel_unified` | `flash_attn_small_tile_kernel` | Clarifies it's also Flash Attention, just smaller tiles |
| `attention_reference_unified` | `flash_attn_small_tile_forward` | Consistent with kernel name |
| `RefTileConfig` | `SmallTileConfig` | More descriptive |
| `compute_ref_tile_size()` | `compute_small_tile_size()` | Consistent naming |

### 2. **New Dispatch Function**

- Added: `flash_attention_small_tile_dispatch()` - Public API for small tile version
- Kept: `attention_reference_dispatch()` as backward compatibility alias

### 3. **Updated Comments and Documentation**

All misleading "Reference" comments have been updated to reflect the actual differences:
- Both implement Flash Attention algorithm
- Difference is only in tile size strategy
- Neither uses CUTLASS tensor cores (yet)

## What the Names Now Mean

### ✅ **flash_attn_large_tile_** (formerly "Flash Attention")
- **Algorithm**: Flash Attention with online softmax
- **Tile Size**: 120×120 (for head_dim=32)
- **Threads**: 256
- **Shared Mem**: 150.9 KB
- **Strategy**: Aggressive - maximize data reuse
- **Trade-off**: High memory pressure, potential occupancy issues

### ✅ **flash_attn_small_tile_** (formerly "Reference")
- **Algorithm**: Flash Attention with online softmax (SAME as large tile!)
- **Tile Size**: 45×90 (for head_dim=32)
- **Threads**: 128
- **Shared Mem**: 51.7 KB  
- **Strategy**: Conservative - prioritize occupancy
- **Trade-off**: More global memory traffic, but better occupancy

### ❌ **attention_baseline** (unchanged)
- **Algorithm**: Naive attention (NOT Flash Attention)
- **Memory**: O(N²) - materializes full attention matrix
- **Strategy**: Simplest possible implementation for correctness checking

## Migration Guide

### If Your Code Uses:

```cpp
// OLD NAME (still works via alias)
attention_reference_dispatch(...);

// NEW PREFERRED NAME
flash_attention_small_tile_dispatch(...);
```

The old name `attention_reference_dispatch` is kept as an alias for backward compatibility, but please update to the new name for clarity.

### Public API Summary

```cpp
// Main dispatch functions (auto-select based on head_dim)
void flash_attention_forward_dispatch(...);          // Uses large tiles
void flash_attention_small_tile_dispatch(...);       // Uses small tiles
void attention_baseline(...);                         // Naive O(N²) version

// Backward compatibility (redirects to small tile)
void attention_reference_dispatch(...);
```

## Why This Matters

### Before Renaming (Confusing):
```
"Flash Attention"  vs  "Reference"
     ↓                      ↓
Sounds different      Sounds like baseline
```
**Problem**: Implies they're fundamentally different algorithms!

### After Renaming (Clear):
```
"Large Tile"  vs  "Small Tile"
     ↓                 ↓
Both are Flash Attention, just different hyperparameters
```
**Benefit**: Makes it clear they're the same algorithm with different tile sizes!

## Next Steps

### Recommended Improvements:

1. **Consolidate implementations**
   - They're 95% identical code
   - Should be a single template with tile size parameter

2. **Fix thread underutilization**
   - With head_dim=32, 87.5% of threads are idle in large tile!
   - Should use `min(256, head_dim * 8)` threads

3. **Add true CUTLASS version**
   - Current "CUTLASS" folder is misleading
   - Should add version using tensor cores for 5-10× speedup

4. **Rename directory**
   - `cutlass_cuda_fa1/` → `cuda_fa1_tiled/`
   - Only rename if truly using CUTLASS

## Files Modified

- `/code/cutlass_cuda_fa1/run/flash_attn_unified.cu` - All function/struct renames
- `/code/cutlass_cuda_fa1/run/test_flash_attn.cu` - Updated print statements  
- Created: `IMPLEMENTATION_COMPARISON.md` - Detailed analysis
- Created: `RENAMING_SUMMARY.md` - This file


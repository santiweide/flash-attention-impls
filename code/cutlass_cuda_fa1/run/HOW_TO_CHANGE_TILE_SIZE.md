# How to Change Tile Size: Code Locations and Examples

## Quick Reference

| Implementation | File | Line Range | Config Structure |
|---------------|------|------------|-----------------|
| **Large Tile** | `flash_attn_unified.cu` | 29-79 | `TileConfig<HEAD_DIM>` |
| **Small Tile** | `flash_attn_unified.cu` | 350-369 | `SmallTileConfig<HEAD_DIM>` |
| **CUTLASS TC** | `flash_attn_cutlass.cu` | 77-98 | `CutlassSmallTileConfig<HEAD_DIM>` |

---

## 1. Large Tile Configuration

### 📍 Location: `flash_attn_unified.cu` Lines 29-79

**Current implementation (automatic sizing):**

```cpp
template<int HEAD_DIM>
struct TileConfig {
    static constexpr int compute_max_tile_size() {
        // Shared memory budget (bytes)
        constexpr size_t MAX_SMEM = 160 * 1024;  // 160 KB
        
        // Try tile sizes from 128 down to 32 in steps of 8
        for (int tile = 128; tile >= 32; tile -= 8) {
            size_t qkv_size = tile * HEAD_DIM * sizeof(cutlass::half_t) * 3;
            size_t sp_size = tile * tile * sizeof(float) * 2;
            size_t extra = tile * 2 * sizeof(float) + tile * HEAD_DIM * sizeof(float);
            size_t total = qkv_size + sp_size + extra;
            
            if (total < MAX_SMEM) {
                return tile;  // ← Automatically finds largest tile that fits
            }
        }
        return 32;
    }
    
    static constexpr int kTileM = compute_max_tile_size();
    static constexpr int kTileN = compute_max_tile_size();
    static constexpr int kHeadDim = HEAD_DIM;
    static constexpr int kThreads = (HEAD_DIM * 2 < 64) ? 64 : 
                                     (HEAD_DIM * 2 > 256) ? 256 : 
                                     (HEAD_DIM * 2);
    // ...
};
```

### 🔧 Method 1: Change Automatic Search Range

**Location:** `flash_attn_unified.cu:43`

```cpp
// BEFORE:
for (int tile = 128; tile >= 32; tile -= 8) {

// AFTER (search from 96 down):
for (int tile = 96; tile >= 32; tile -= 8) {
```

**Effect:** Will find smaller tiles (better occupancy, more iterations)

### 🔧 Method 2: Manually Set Fixed Tile Size

**Location:** `flash_attn_unified.cu:56-57`

```cpp
// BEFORE (automatic):
static constexpr int kTileM = compute_max_tile_size();
static constexpr int kTileN = compute_max_tile_size();

// AFTER (fixed 96×96):
static constexpr int kTileM = 96;
static constexpr int kTileN = 96;

// Or asymmetric (96×80):
static constexpr int kTileM = 96;
static constexpr int kTileN = 80;
```

**Effect:** Directly control tile size (bypasses automatic calculation)

### 🔧 Method 3: Adjust Shared Memory Budget

**Location:** `flash_attn_unified.cu:36`

```cpp
// BEFORE:
constexpr size_t MAX_SMEM = 160 * 1024;  // 160 KB

// AFTER (more conservative):
constexpr size_t MAX_SMEM = 120 * 1024;  // 120 KB

// Or more aggressive (if your GPU supports):
constexpr size_t MAX_SMEM = 180 * 1024;  // 180 KB
```

**Effect:** Changes the upper bound for automatic tile search

---

## 2. Small Tile Configuration

### 📍 Location: `flash_attn_unified.cu` Lines 350-369

**Current implementation (derived from Large Tile):**

```cpp
template<int HEAD_DIM>
struct SmallTileConfig {
    static constexpr int compute_small_tile_size() {
        // Small tile uses ~75% of large tile size
        constexpr int large_tile = TileConfig<HEAD_DIM>::kTileM;
        return (large_tile * 3) / 4;  // 75% of large tile
    }
    
    static constexpr int kTileM = compute_small_tile_size() / 2;  // M direction
    static constexpr int kTileN = compute_small_tile_size();       // N direction
    static constexpr int kHeadDim = HEAD_DIM;
    static constexpr int kThreads = 256;
    
    // ...
};
```

### 🔧 Method 1: Change the Scaling Factor

**Location:** `flash_attn_unified.cu:355`

```cpp
// BEFORE (75% of Large Tile):
return (large_tile * 3) / 4;  // 120 * 0.75 = 90

// AFTER (50% of Large Tile):
return (large_tile * 1) / 2;  // 120 * 0.5 = 60

// Or (60% of Large Tile):
return (large_tile * 3) / 5;  // 120 * 0.6 = 72
```

### 🔧 Method 2: Fixed Tile Sizes

**Location:** `flash_attn_unified.cu:358-359`

```cpp
// BEFORE (automatic):
static constexpr int kTileM = compute_small_tile_size() / 2;  // 45
static constexpr int kTileN = compute_small_tile_size();       // 90

// AFTER (fixed 48×96, WMMA-aligned):
static constexpr int kTileM = 48;
static constexpr int kTileN = 96;

// Or (fixed 32×64, power-of-2):
static constexpr int kTileM = 32;
static constexpr int kTileN = 64;

// Or (square tiles 64×64):
static constexpr int kTileM = 64;
static constexpr int kTileN = 64;
```

### 🔧 Method 3: Different Asymmetry Ratio

**Location:** `flash_attn_unified.cu:358`

```cpp
// BEFORE (M = N/2):
static constexpr int kTileM = compute_small_tile_size() / 2;  // 45
static constexpr int kTileN = compute_small_tile_size();       // 90

// AFTER (M = N/3, more asymmetric):
static constexpr int kTileM = compute_small_tile_size() / 3;  // 30
static constexpr int kTileN = compute_small_tile_size();       // 90

// Or (M = N, square):
static constexpr int kTileM = compute_small_tile_size();       // 90
static constexpr int kTileN = compute_small_tile_size();       // 90
```

### 🔧 Method 4: Change Thread Count

**Location:** `flash_attn_unified.cu:361`

```cpp
// BEFORE:
static constexpr int kThreads = 256;

// AFTER (fewer threads):
static constexpr int kThreads = 128;

// Or (more threads):
static constexpr int kThreads = 512;
```

**Note:** Changing threads affects occupancy and performance!

---

## 3. CUTLASS Tensor Core Configuration

### 📍 Location: `flash_attn_cutlass.cu` Lines 77-98

**Current implementation (per-HEAD_DIM hardcoded):**

```cpp
template<int HEAD_DIM>
struct CutlassSmallTileConfig {
    static constexpr int compute_small_tile_size() {
        // Hardcoded conservative sizes per HEAD_DIM
        if (HEAD_DIM == 32) return 90;
        if (HEAD_DIM == 64) return 72;
        if (HEAD_DIM == 128) return 48;
        return 32;
    }
    
    static constexpr int kTileM = compute_small_tile_size() / 2;  // M direction
    static constexpr int kTileN = compute_small_tile_size();       // N direction
    static constexpr int kHeadDim = HEAD_DIM;
    static constexpr int kThreads = 256;
    
    // ...
};
```

### 🔧 Method 1: Change Hardcoded Values

**Location:** `flash_attn_cutlass.cu:82-84`

```cpp
// BEFORE:
if (HEAD_DIM == 32) return 90;
if (HEAD_DIM == 64) return 72;
if (HEAD_DIM == 128) return 48;

// AFTER (WMMA-aligned, multiples of 16):
if (HEAD_DIM == 32) return 96;   // 96 = 6 × 16
if (HEAD_DIM == 64) return 80;   // 80 = 5 × 16
if (HEAD_DIM == 128) return 48;  // 48 = 3 × 16 (unchanged)

// Or (more aggressive):
if (HEAD_DIM == 32) return 112;  // 112 = 7 × 16
if (HEAD_DIM == 64) return 96;   // 96 = 6 × 16
if (HEAD_DIM == 128) return 64;  // 64 = 4 × 16
```

**Recommendation:** Use multiples of 16 for best WMMA performance!

### 🔧 Method 2: Fixed Tiles (Override)

**Location:** `flash_attn_cutlass.cu:88-89`

```cpp
// BEFORE (automatic):
static constexpr int kTileM = compute_small_tile_size() / 2;
static constexpr int kTileN = compute_small_tile_size();

// AFTER (fixed 48×96 for HEAD_DIM=32):
static constexpr int kTileM = (HEAD_DIM == 32) ? 48 : compute_small_tile_size() / 2;
static constexpr int kTileN = (HEAD_DIM == 32) ? 96 : compute_small_tile_size();
```

---

## 4. Practical Examples

### Example 1: Make Small Tile Use WMMA-Aligned Sizes

**Goal:** Use 48×96 tiles for better Tensor Core utilization

**File:** `flash_attn_unified.cu`
**Location:** Lines 358-359

```cpp
// Change from:
static constexpr int kTileM = compute_small_tile_size() / 2;  // 45
static constexpr int kTileN = compute_small_tile_size();       // 90

// To:
static constexpr int kTileM = 48;  // 48 = 3 × 16 (WMMA-aligned)
static constexpr int kTileN = 96;  // 96 = 6 × 16 (WMMA-aligned)
```

**Expected result:**
- Slightly more shared memory: 54.3 KB vs 51.7 KB
- 100% WMMA coverage (no CUDA core fallback)
- ~6% performance improvement for CUTLASS TC

### Example 2: Use Power-of-2 Tiles

**Goal:** Test if power-of-2 helps (spoiler: it probably won't)

**File:** `flash_attn_unified.cu`
**Location:** Lines 358-359

```cpp
// Change to:
static constexpr int kTileM = 32;  // 2^5
static constexpr int kTileN = 64;  // 2^6
```

**Expected result:**
- Less shared memory: 39.2 KB
- Better occupancy: 4 blocks/SM (50%)
- More iterations: 32 Q-tiles × 16 K/V-tiles = 512 total
- Likely **worse** performance (more kernel overhead)

### Example 3: Maximize Occupancy

**Goal:** Use smallest tiles possible for maximum occupancy

**File:** `flash_attn_unified.cu`
**Location:** Lines 358-359

```cpp
// Change to:
static constexpr int kTileM = 24;
static constexpr int kTileN = 48;
```

**Expected result:**
- Very small shared memory: ~29 KB
- Excellent occupancy: 5 blocks/SM (62.5%)
- Many iterations: 43 Q-tiles × 22 K/V-tiles = 946 total
- Likely **worse** performance (too much overhead)

### Example 4: Make Large Tile Use Square Power-of-2

**Goal:** Test 64×64 square power-of-2 tiles

**File:** `flash_attn_unified.cu`
**Location:** Lines 56-57

```cpp
// Change from:
static constexpr int kTileM = compute_max_tile_size();
static constexpr int kTileN = compute_max_tile_size();

// To:
static constexpr int kTileM = 64;
static constexpr int kTileN = 64;
```

**Expected result:**
- Less shared memory: 48.6 KB
- Same occupancy: 3 blocks/SM (37.5%)
- More iterations: 16 Q-tiles × 16 K/V-tiles
- Likely similar or slightly worse performance

---

## 5. Rebuild and Test

### After Changing Tile Sizes:

```bash
cd /Users/michu/Documents/flash-attention-impls/code/cutlass_cuda_fa1/run

# Clean and rebuild
make clean
make

# Run tests
./test_flash_attn

# Check output for:
# 1. Tile configuration printed at start
# 2. Shared memory usage
# 3. Performance results
# 4. Accuracy (should stay < 2% error)
```

### What to Look For:

```
Expected output showing your new tile sizes:

Flash Attention - Large Tile (head_dim=32)
  Tile Config for head_dim=32:
    Tile size: 96×96        ← Your new size
    Threads: 64
    Shared memory: 110.2 KB  ← Updated
    
Performance Results:
Flash Attn (Large Tile):  XXX.XXX ms  ← Compare with baseline
```

---

## 6. Verification Script

Create a script to test different tile configurations:

```bash
#!/bin/bash
# test_tile_configs.sh

echo "Testing different tile configurations..."

CONFIGS=(
    "48,96"   # WMMA-aligned
    "32,64"   # Power-of-2
    "64,64"   # Square
    "40,80"   # Random
)

for config in "${CONFIGS[@]}"; do
    IFS=',' read -r TILE_M TILE_N <<< "$config"
    
    echo ""
    echo "Testing Tile: ${TILE_M}×${TILE_N}"
    echo "================================"
    
    # Modify SmallTileConfig in flash_attn_unified.cu
    sed -i.bak "s/static constexpr int kTileM = .*/static constexpr int kTileM = ${TILE_M};/" flash_attn_unified.cu
    sed -i.bak "s/static constexpr int kTileN = .*/static constexpr int kTileN = ${TILE_N};/" flash_attn_unified.cu
    
    # Rebuild and test
    make clean > /dev/null 2>&1
    make > /dev/null 2>&1
    ./test_flash_attn | grep -E "(Tile size:|Performance|TFLOPs)"
    
    # Restore backup
    mv flash_attn_unified.cu.bak flash_attn_unified.cu
done
```

---

## 7. Quick Reference Table

### Tile Size Impact

| Change | Shared Mem | Occupancy | Iterations | Expected Performance |
|--------|-----------|-----------|------------|---------------------|
| **Increase tile** | ↑ | ↓ | ↓ | Mixed (occupancy vs overhead) |
| **Decrease tile** | ↓ | ↑ | ↑ | Mixed (occupancy vs overhead) |
| **Power-of-2** | - | - | - | Minimal impact |
| **WMMA-aligned (×16)** | - | - | - | +6% for CUTLASS TC |
| **Square tiles** | - | - | - | Simpler, may be suboptimal |
| **Asymmetric (M<N)** | ↓ | ↑ | - | Often better for Flash Attn |

### Recommended Tile Sizes for HEAD_DIM=32

| Goal | Tile M | Tile N | Rationale |
|------|--------|--------|-----------|
| **Current (balanced)** | 45 | 90 | Good compromise |
| **WMMA-optimized** | 48 | 96 | Best for CUTLASS TC (+6%) |
| **Max occupancy** | 24 | 48 | 5 blocks/SM, too many iterations |
| **Power-of-2** | 32 | 64 | No real benefit |
| **Max shared mem** | 120 | 120 | Poor occupancy (3.1%) |

---

## Summary

**Key files to modify:**
1. **`flash_attn_unified.cu`** (lines 56-57, 358-359) - Most common
2. **`flash_attn_cutlass.cu`** (lines 82-84, 88-89) - For CUTLASS TC
3. **`flash_attn_unified.cu`** (line 36) - Shared memory budget

**Recommended approach:**
1. Start with current values (45×90 for Small Tile)
2. Test WMMA-aligned version (48×96) for CUTLASS TC
3. Measure performance difference
4. Only change if you see significant improvement (>5%)

**Remember:**
- ✅ Rebuild after changes: `make clean && make`
- ✅ Verify accuracy stays < 2% error
- ✅ Check shared memory doesn't exceed 164 KB
- ✅ Monitor occupancy and iterations

Good luck! 🚀


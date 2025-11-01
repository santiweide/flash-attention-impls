# Minimal Flash Attention Makefile
# Quick compile, no dependencies on CMake

# configure
NVCC := nvcc
CUDA_ARCH := -arch=sm_80  # A100 architecture, modify as needed to sm_86 (RTX3090) or sm_89 (RTX4090)
CXX_FLAGS := -std=c++17
CUTLASS_DIR := ../csrc/cutlass

CUTLASS_INCLUDE := $(CUTLASS_DIR)/include
ifeq (,$(wildcard $(CUTLASS_INCLUDE)))
    $(warning Cutlass not found at $(CUTLASS_DIR))
    $(warning Please update CUTLASS_DIR in Makefile or run: git submodule update --init csrc/cutlass)
endif

# compile options
NVCC_FLAGS := $(CUDA_ARCH) $(CXX_FLAGS) \
              -O3 \
              --expt-relaxed-constexpr \
              --expt-extended-lambda \
              -I$(CUTLASS_INCLUDE) \
              -I. \
              --use_fast_math

# ==================== Standard Targets ====================

# target
TARGET := test_flash_attn
SOURCES := test_flash_attn.cu flash_attn_unified.cu flash_attn_cutlass.cu

.PHONY: all clean run info help

all: $(TARGET)

# compile
$(TARGET): $(SOURCES)
	@echo "Compiling $(TARGET)..."
	@echo "CUDA Architecture: $(CUDA_ARCH)"
	@echo "Cutlass Include: $(CUTLASS_INCLUDE)"
	$(NVCC) $(NVCC_FLAGS) -o $@ $^
	@echo "Build complete: ./$(TARGET)"

# run
run: $(TARGET)
	@echo "Running $(TARGET)..."
	./$(TARGET)

# clean
clean:
	rm -f $(TARGET) *.o

# show information
info:
	@echo "=== Build Information ==="
	@echo "NVCC: $(NVCC)"
	@echo "CUDA Architecture: $(CUDA_ARCH)"
	@echo "C++ Standard: $(CXX_FLAGS)"
	@echo "Cutlass Directory: $(CUTLASS_DIR)"
	@echo "Cutlass Include: $(CUTLASS_INCLUDE)"
	@echo ""
	@echo "Available targets:"
	@echo "  make         - Build the test program"
	@echo "  make run     - Build and run"
	@echo "  make clean   - Clean build files"
	@echo "  make info    - Show this information"
	@echo ""
	@echo "To change GPU architecture:"
	@echo "  make CUDA_ARCH=-arch=sm_86  # for RTX 3090/A6000"
	@echo "  make CUDA_ARCH=-arch=sm_89  # for RTX 4090"

help: info

# ==================== Performance Benchmark Targets ====================

# Performance test for flash_attn_cutlass only
PERF_TARGET := perf_flash_attn_cutlass
PERF_SOURCES := perf_flash_attn_cutlass.cu flash_attn_cutlass.cu

.PHONY: perf perf_run perf_clean perf_info

# Build performance benchmark
perf: $(PERF_TARGET)

$(PERF_TARGET): $(PERF_SOURCES)
	@echo "Compiling $(PERF_TARGET) (Performance Benchmark Only)..."
	@echo "CUDA Architecture: $(CUDA_ARCH)"
	@echo "Cutlass Include: $(CUTLASS_INCLUDE)"
	$(NVCC) $(NVCC_FLAGS) -o $@ $^
	@echo "Build complete: ./$(PERF_TARGET)"

# Run performance benchmark
perf_run: $(PERF_TARGET)
	@echo "Running $(PERF_TARGET) - Performance Benchmark..."
	@echo "Measuring latency and TFLOPs/s across different configurations..."
	@echo ""
	./$(PERF_TARGET)

# Clean performance benchmark
perf_clean:
	rm -f $(PERF_TARGET) $(PERF_TARGET).o

# Show performance benchmark info
perf_info:
	@echo "=== Performance Benchmark Information ==="
	@echo "Target: $(PERF_TARGET)"
	@echo "Sources: $(PERF_SOURCES)"
	@echo "CUDA Architecture: $(CUDA_ARCH)"
	@echo ""
	@echo "Features:"
	@echo "  • Tests flash_attn_cutlass only (parallel softmax enabled)"
	@echo "  • Measures latency (ms) and TFLOPs/s"
	@echo "  • Computes memory bandwidth (GB/s)"
	@echo "  • No correctness verification - performance only"
	@echo ""
	@echo "Test Configurations (18 total):"
	@echo "  • Sequence lengths: 512, 1024, 2048, 4096, 8192"
	@echo "  • Head dimensions: 32, 64, 128"
	@echo "  • Batch sizes: 1 (default), 2, 4, 8 (for seq_len=8192)"
	@echo "  • Number of heads: 12"
	@echo ""
	@echo "Usage:"
	@echo "  make perf       - Build performance benchmark"
	@echo "  make perf_run   - Build and run benchmark"
	@echo "  make perf_clean - Clean build files"
	@echo "  make perf_info  - Show this information"

.PHONY: perf_info


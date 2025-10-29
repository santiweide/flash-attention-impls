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

# target
TARGET := test_flash_attn
SOURCES := test_flash_attn.cu flash_attn_unified.cu flash_attn_cutlass.cu

.PHONY: all clean run info

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


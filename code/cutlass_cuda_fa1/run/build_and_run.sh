#!/bin/bash
# 快速构建和运行脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Minimal Flash Attention - Quick Start"
echo "=========================================="
echo ""

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}✗${NC} nvcc not found! Please install CUDA Toolkit."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',')
echo -e "${GREEN}✓${NC} CUDA found: $CUDA_VERSION"

# 检测GPU架构
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo -e "${GREEN}✓${NC} GPU detected: $GPU_NAME"
    
    # 根据GPU选择架构
    if [[ $GPU_NAME == *"A100"* ]]; then
        CUDA_ARCH="-arch=sm_80"
    elif [[ $GPU_NAME == *"4090"* ]] || [[ $GPU_NAME == *"4080"* ]]; then
        CUDA_ARCH="-arch=sm_89"
    elif [[ $GPU_NAME == *"3090"* ]] || [[ $GPU_NAME == *"A6000"* ]]; then
        CUDA_ARCH="-arch=sm_86"
    else
        CUDA_ARCH="-arch=sm_80"  # 默认
        echo -e "${YELLOW}⚠${NC} Unknown GPU, using default arch: sm_80"
    fi
    echo "  Using CUDA architecture: $CUDA_ARCH"
else
    CUDA_ARCH="-arch=sm_80"
    echo -e "${YELLOW}⚠${NC} nvidia-smi not found, using default arch: $CUDA_ARCH"
fi

# 检查Cutlass
CUTLASS_DIR="../csrc/cutlass"
if [ ! -d "$CUTLASS_DIR/include" ]; then
    echo -e "${YELLOW}⚠${NC} Cutlass not found at $CUTLASS_DIR"
    echo "  Attempting to initialize git submodule..."
    
    cd ..
    if git submodule update --init csrc/cutlass; then
        echo -e "${GREEN}✓${NC} Cutlass initialized successfully"
        cd minimal_flashattn_cutlass
    else
        echo -e "${RED}✗${NC} Failed to initialize Cutlass"
        echo "  Please manually clone Cutlass:"
        echo "    git clone https://github.com/NVIDIA/cutlass.git csrc/cutlass"
        exit 1
    fi
else
    echo -e "${GREEN}✓${NC} Cutlass found at $CUTLASS_DIR"
fi

echo ""
echo "=========================================="
echo "Building..."
echo "=========================================="

# 选择构建方式
BUILD_METHOD=${1:-make}  # 默认使用make

if [ "$BUILD_METHOD" = "cmake" ]; then
    echo "Using CMake build system..."
    mkdir -p build
    cd build
    cmake -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH#-arch=sm_} ..
    make -j$(nproc 2>/dev/null || echo 4)
    cd ..
    EXECUTABLE="./build/test_flash_attn"
else
    echo "Using Makefile build system..."
    make clean
    make CUDA_ARCH=$CUDA_ARCH -j$(nproc 2>/dev/null || echo 4)
    EXECUTABLE="./test_flash_attn"
fi

if [ ! -f "$EXECUTABLE" ]; then
    echo -e "${RED}✗${NC} Build failed!"
    exit 1
fi

echo -e "${GREEN}✓${NC} Build successful!"
echo ""

echo "=========================================="
echo "Running tests..."
echo "=========================================="
echo ""

$EXECUTABLE

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
echo ""
echo "To rebuild:"
echo "  $0"
echo ""
echo "To use CMake:"
echo "  $0 cmake"
echo ""
echo "To clean:"
echo "  make clean"
echo "  rm -rf build/"


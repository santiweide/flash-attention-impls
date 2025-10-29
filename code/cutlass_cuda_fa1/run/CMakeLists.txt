cmake_minimum_required(VERSION 3.18)
project(MinimalFlashAttention CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../csrc/cutlass" CACHE PATH "Path to Cutlass")

if(NOT EXISTS "${CUTLASS_DIR}/include")
    message(STATUS "Cutlass not found at ${CUTLASS_DIR}, trying alternative paths...")
    set(CUTLASS_DIR "/usr/local/include/cutlass" CACHE PATH "Path to Cutlass" FORCE)
endif()

if(NOT EXISTS "${CUTLASS_DIR}/include")
    message(WARNING "Cutlass not found! Please set CUTLASS_DIR to your Cutlass installation")
    message(WARNING "Download from: https://github.com/NVIDIA/cutlass")
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUTLASS_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

# SM 80: A100
# SM 86: RTX 3090, A6000
# SM 89: RTX 4090
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89)
endif()

message(STATUS "Building for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

add_library(flash_attn_minimal STATIC
    flash_attn_minimal.cu
)

target_include_directories(flash_attn_minimal PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUTLASS_DIR}/include
)

set_target_properties(flash_attn_minimal PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_executable(test_flash_attn
    test_flash_attn.cu
)

target_link_libraries(test_flash_attn
    flash_attn_minimal
    CUDA::cudart
)

set_target_properties(test_flash_attn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Cutlass Directory: ${CUTLASS_DIR}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")


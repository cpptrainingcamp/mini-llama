cmake_minimum_required(VERSION 3.17)
project(mini-llama LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MINI_LLAMA_CUDA "Build NVIDIA CUDA backend" OFF)

if(MINI_LLAMA_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 89)
    endif()
    find_package(CUDAToolkit REQUIRED)
endif()

# simdjson is used by the tokenizer headers.
if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()
find_package(simdjson CONFIG REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR})

# --- Core library sources ---
set(MINI_LLAMA_CORE_SOURCES
    src/ascii_tokenizer.cpp
    src/backend.cpp
    src/batch.cpp
    src/bpe_tokenizer.cpp
    src/chat.cpp
    src/context.cpp
    src/cuda_attention.cpp
    src/cuda_kv_cache.cpp
    src/cuda_matmul.cpp
    src/cuda_ops.cpp
    src/cuda_quant.cpp
    src/cuda_runtime.cpp
    src/cuda_tensor.cpp
    src/debug.cpp
    src/forward.cpp
    src/gguf.cpp
    src/gguf_loader.cpp
    src/gguf_tokenizer.cpp
    src/json_vocab_tokenizer.cpp
    src/kv_cache.cpp
    src/loader.cpp
    src/matmul_dispatch.cpp
    src/model.cpp
    src/ops.cpp
    src/promptbuilder.cpp
    src/quant.cpp
    src/quantized_tensor.cpp
    src/radixtree.cpp
    src/request_context.cpp
    src/sampler.cpp
    src/tensor.cpp
    src/terminal.cpp
    src/thread_pool.cpp
)

if(MINI_LLAMA_CUDA)
    list(APPEND MINI_LLAMA_CORE_SOURCES
        src/cuda_attention.cu
        src/cuda_ops.cu
        src/cuda_quant.cu
    )
endif()

# --- Main executable ---
add_executable(mini-llama
    ${MINI_LLAMA_CORE_SOURCES}
    src/main.cpp
)
set_target_properties(mini-llama PROPERTIES OUTPUT_NAME mini-llama)
target_link_libraries(mini-llama PRIVATE simdjson::simdjson)

if(MINI_LLAMA_CUDA)
    target_compile_definitions(mini-llama PRIVATE MINI_LLAMA_USE_CUDA=1)
    target_link_libraries(mini-llama PRIVATE CUDA::cudart CUDA::cublas)
endif()

# Release optimizations
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

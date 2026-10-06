#pragma once

#include <vector>

#include "ops.h"
#include "quantized_tensor.h"
#include "thread_pool.h"

// x86 AVX2
#if defined(__AVX2__) && defined(__FMA__)
    #include <immintrin.h>
    #define MINI_LLAMA_USE_AVX2 1
// ARM NEON
#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)
    #include <arm_neon.h>
    #define MINI_LLAMA_USE_NEON 1
#endif

namespace LLaMA_Tensor{
//Q8_0量化
std::vector<BlockQ80> QuantizedToQ80(const Tensor& t);
//Q8_0反量化
Tensor DequantizedFromQ80(const std::vector<BlockQ80>& blocks, const std::vector<int>& shape);
//Q4_0量化
std::vector<BlockQ40> QuantizedToQ40(const Tensor& t);
//Q4_0反量化
Tensor DequantizedFromQ40(const std::vector<BlockQ40>& blocks, const std::vector<int>& shape);
//Q4_1量化
std::vector<BlockQ41> QuantizedToQ41(const Tensor& t);
//Q4_1反量化
Tensor DequantizedFromQ41(const std::vector<BlockQ41>& blocks, const std::vector<int>& shape);

//Q8_0投影算子
Tensor LinearQ80(const Tensor& x, const std::vector<BlockQ80>& weight, const std::vector<int>& weight_shape);
//Q4_0投影算子
Tensor LinearQ40(const Tensor& x, const std::vector<BlockQ40>& weight, const std::vector<int>& weight_shape);
//Q4_1投影算子
Tensor LinearQ41(const Tensor& x, const std::vector<BlockQ41>& weight, const std::vector<int>& weight_shape);
//Q80矩阵乘
Tensor MatmulQ80(const std::vector<BlockQ80>& weight, const Tensor& input, const std::vector<int>& weight_shape);
//Q40矩阵乘
Tensor MatmulQ40(const std::vector<BlockQ40>& weight, const Tensor& input, const std::vector<int>& weight_shape);
// ---------------------------------------------------------------------------
// Benchmark helpers
// ---------------------------------------------------------------------------
// Compare F32 Matmul vs Q8_0 Matmul for the same weight + input.
// Returns max absolute error.
float CompareMatmulError(const Tensor& weight, const Tensor& input);

// Compare F32 Linear vs Q4_0 Linear for the same weight + input.
// Returns max absolute error.
float CompareQ40Error(const Tensor& weight, const Tensor& input);
}
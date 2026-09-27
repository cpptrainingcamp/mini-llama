#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensor.h"

namespace LLaMA_Tensor{
//量化方法
enum class QuantType{
    kF32,
    kQ80,
    kQ40,
    kQ41,
};
//Q8_0
constexpr int kQ80BlockSize = 32;

struct BlockQ80{
    uint16_t scale; //FP16 scale
    int8_t qs[32];  //量化值
};

static_assert(sizeof(BlockQ80) == sizeof(uint16_t) + kQ80BlockSize, "wrong Q8_0 block size");

//Q4_0
constexpr int kQ40BlockSize = 32;

struct BlockQ40{
    uint16_t scale; //FP16 scale
    int8_t qs[16];  //4-bit per element
};

static_assert(sizeof(BlockQ40) == sizeof(uint16_t) + kQ40BlockSize / 2, "wrong Q4_0 block size"); 

//Q4_1
constexpr int kQ41BlockSize = 32;

struct BlockQ41{
    uint16_t scale;
    uint16_t scale_min;
    uint8_t qs[16];
};

static_assert(sizeof(BlockQ41) == sizeof(uint16_t) * 2 + kQ41BlockSize / 2, "wrong Q4_1 block size");

struct QuantizedTensor{
    QuantType type = QuantType::kF32;
    std::vector<int> shape;
    std::vector<float> f32_data;
    std::vector<BlockQ80> q80_data;
    std::vector<BlockQ40> q40_data;
    std::vector<BlockQ41> q41_data;

    //元素数
    size_t ele_num() const;
    //形状打印
    std::string ShapeString() const;
    //形状检查
    void AssertShape(const std::vector<int>& expect, const std::string& caller) const;
    //形状维度
    int num_dims() const;
};

//QuantizedTensor -> Tensor
Tensor QTensor2Tensor(const QuantizedTensor& q_tensor);
//Tensor -> QuantizedTensor
QuantizedTensor Tensor2QTensor(const Tensor& tensor);
}
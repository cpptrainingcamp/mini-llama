#pragma once

#include <random>

#include "tensor.h"
//采样参数
namespace LLaMA_Tensor{
struct SamplerParams{
    float temperature = 0.0f; //默认贪婪采样
    int top_k = 0;  //默认禁用采样
    unsigned int seed = 0;  //默认系统随机源
};

class Sampler{
public:
    explicit Sampler(unsigned int seed = 0);
    explicit Sampler(const SamplerParams& param);
    //用SamplerParams的参数采样
    int Sample(const Tensor& logits);
    //贪婪采样：取最大
    static int SampleGreedy(const Tensor& logits);
    //temperature采样
    int SampleTemperature(const Tensor& logits);
    //Top-k + temperature采样
    int SampleTopK(const Tensor& logits);

private:
    std::mt19937 rng_;
    struct SamplerParams param_;
};

}

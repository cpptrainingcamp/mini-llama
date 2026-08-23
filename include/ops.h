#pragma once

#include "tensor.h"

namespace LLaMA_Tensor{
    //旋转位置编码风格
    enum class RoPEtype{
        Normal,
        NeoX
    };
    //Tensor相乘
    Tensor MatrixMulti(const Tensor& t1, const Tensor& t2);
    //linear算子
    Tensor Linear(const Tensor& x, const Tensor& weight);
    //RMSNorm归一化
    Tensor RMSNorm(const Tensor& x, const Tensor& weight, float eps);
    //softmax：相关性得分转换为概率
    Tensor softmax(const Tensor& x);
    //SiLU激活单元
    Tensor SiLU(const Tensor& x);
    //逐元素乘法
    Tensor ElementsWiseMulti(const Tensor& x, const Tensor& y);
    //SWiGLU门控激活
    Tensor SWiGLU(const Tensor& gate, const Tensor& up);
    //ArgsMax 用于选取logits最大得分ID
    size_t ArgsMax(const Tensor& x);
    //旋转位置编码
    void rope(Tensor& Q, Tensor& K, int pos, float theta, RoPEtype type);
}


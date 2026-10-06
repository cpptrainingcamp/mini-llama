#pragma once

#include "tensor.h"
#include "matmul_dispatch.h"
#include "quantized_tensor.h"
#include "quant.h"

namespace LLaMA_Tensor{
    //旋转位置编码风格
    enum class RoPEtype{
        Normal,
        NeoX
    };
    //Tensor相乘: 支持 1D 向量与 2D 矩阵 (PyTorch matmul 风格)
    //  方向约定: 1D 左操作数视为行向量 {1,n}, 1D 右操作数视为列向量 {n,1}
    //  两个同长 1D 向量相乘 = 点积
    //  输出去掉被提升的向量维度: 1D×1D -> {1}, 1D×2D -> {N}, 2D×1D -> {M}, 2D×2D -> {M,N}
    //  高于 2 维的输入抛 runtime_error
    Tensor MatrixMulti(const Tensor& t1, const Tensor& t2);
    //转置: 2D 矩阵转置; 1D 向量视为行向量 {1,n}, 转置为列向量 {n,1}
    Tensor TensorTrans(const Tensor& x);
    //linear算子
    Tensor Linear(const Tensor& x, const Tensor& weight);
    Tensor Linear(const Tensor& x, const QuantizedTensor& weight);
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


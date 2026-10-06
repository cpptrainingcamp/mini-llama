#pragma once

#include <stdexcept>
#include <vector>

namespace LLaMA_Tensor{
    struct Tensor{
        std::vector<float> data;
        std::vector<int> shape;

        //构造
        Tensor() = default;
        explicit Tensor(const std::vector<int>& tensor_shape, float fill = 0.0f);
        //获取数量信息
        const int TensorNum() const {
            return data.size();
        }
        //获取张量形状维度信息
        const int dim() const {
            return shape.size();
        }
        //根据多维索引计算线性索引
        size_t FlatIndex(const std::vector<int>& MultiIndex) const;
        //根据多维索引获取元素
        float At(const std::vector<int>& MultiIndex) const;
        //安全的多维张量访问
        float At1(int i) const;
        float& At1(int i);
        float At2(int i, int j) const;
        float& At2(int i, int j);
        float At3(int i, int j, int k) const;
        float& At3(int i, int j, int k);
        float At4(int i, int j, int k, int l) const;
        float& At4(int i, int j, int k, int l);
        //获取二维张量行向量初始地址
        float* RowPtr(int row);
        //重置形状
        void ReshapeCheck(const std::vector<int>& new_shape);

        //打印张量
        void TensorPrint() const;


        //维度匹配检查
        void CheckDim(int dim) const;
    };
    //创建多维张量函数
    Tensor MakeTensor1D(int d0, float fill);
    Tensor MakeTensor2D(int d0, int d1, float fill);
    Tensor MakeTensor3D(int d0, int d1, int d2, float fill);
    Tensor MakeTensor4D(int d0, int d1, int d2, int d3, float fill);
}

#include "../include/tensor.h"
#include <numeric>
#include <iostream>
#include <functional>
#include <iomanip>

namespace LLaMA_Tensor {

void Tensor::CheckDim(int dim) const{
    if(dim != shape.size()){
        throw std::runtime_error("dimension(s) are not mathed, your tensor's dimension(s) are "
             + std::to_string(shape.size()));
    }
}

Tensor::Tensor(const std::vector<int>& tensor_shape, float fill)
    :shape(tensor_shape){
    int sum = std::accumulate(tensor_shape.begin(), tensor_shape.end(), 1, std::multiplies<int>());
    data = std::vector<float>(sum, fill);
}

size_t Tensor::FlatIndex(const std::vector<int>& MultiIndex) const{
    //维度不匹配检查
    CheckDim(MultiIndex.size());

    size_t flat = 0;
    size_t stride = 1;
    for(int i = shape.size()-1;i>=0;i--){
        flat += static_cast<size_t>(MultiIndex[i]) * stride;
        stride *= static_cast<size_t>(shape[i]);
    }
    return flat;
}

float Tensor::At(const std::vector<int>& MultiIndex) const {
    CheckDim(MultiIndex.size());
    return data[FlatIndex(MultiIndex)];
}

//只读访问
float Tensor::At1(int i) const {
    CheckDim(1);
    return data[i];
}

//可写访问
float& Tensor::At1(int i) {
    CheckDim(1);
    return data[i];
}

//只读访问
float Tensor::At2(int i, int j) const {
    CheckDim(2);
    return data[FlatIndex({i,j})];
}

//可写访问
float& Tensor::At2(int i, int j) {
    CheckDim(2);
    return data[FlatIndex({i,j})];
}

//只读访问
float Tensor::At3(int i, int j, int k) const {
    CheckDim(3);
    return data[FlatIndex({i,j,k})];
}

//可写访问
float& Tensor::At3(int i, int j, int k) {
    CheckDim(3);
    return data[FlatIndex({i,j,k})];
}

//只读访问
float Tensor::At4(int i, int j, int k, int l) const {
    CheckDim(4);
    return data[FlatIndex({i,j,k,l})];
}

//可写访问
float& Tensor::At4(int i, int j, int k, int l) {
    CheckDim(4);
    return data[FlatIndex({i,j,k,l})];
}

float* Tensor::RowPtr(int row){
    CheckDim(2);
    return data.data() + static_cast<int>(FlatIndex({row, 0}));
}

void Tensor::ReshapeCheck(const std::vector<int>& new_shape){
    int new_num = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());
    if(TensorNum() != new_num){
        throw std::runtime_error("elements count is not mathed");
    }
    shape = new_shape;
}

void Tensor::TensorPrint() const {
    std::cout << std::setprecision(3);
    if(shape.empty()){
        if(!data.empty()){
            std::cout << data[0] << std::endl;
        }else{
            std::cout << "[]" << std::endl;
        }
        return;
    }

    int idx = 0;
    std::function<void(int)> rec = [&](int dim){
        std::cout << "[";
        for(int i = 0;i < shape[dim];++i){
            if(dim + 1 == shape.size()){
                std::cout << data[idx++];
            }else{
                rec(dim+1);
            }

            if(i != shape[dim] - 1){
                std::cout << ",";
            }
        }
        std::cout << "]";
    };
    rec(0);
    std::cout << std::endl;
}

Tensor MakeTensor1D(int d0, float fill){
    return Tensor({d0}, fill);
}

Tensor MakeTensor2D(int d0, int d1, float fill){
    return Tensor({d0, d1}, fill);
}

Tensor MakeTensor3D(int d0, int d1, int d2, float fill){
    return Tensor({d0, d1, d2}, fill);
}

Tensor MakeTensor4D(int d0, int d1, int d2, int d3, float fill){
    return Tensor({d0, d1, d2, d3}, fill);
}

} // namespace LLaMA_Tensor
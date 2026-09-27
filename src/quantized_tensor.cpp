#include "../include/quantized_tensor.h"

#include <numeric>

namespace LLaMA_Tensor{
size_t QuantizedTensor::ele_num() const{
    if(shape.size() == 0){
        return 1;
    }
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
}

std::string QuantizedTensor::ShapeString() const{
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
      s += std::to_string(shape[i]);
      if (i + 1 < shape.size()) {
        s += ", ";
      }
    }
    s += "]";
    return s;
}

void QuantizedTensor::AssertShape(const std::vector<int>& expect, const std::string& caller) const{
    if(shape != expect){
        throw std::runtime_error("quantized tensor shape:" + this->ShapeString() 
        + " is not matched, expect: " + QuantizedTensor{QuantType::kF32, expect}.ShapeString());
    }
}

int QuantizedTensor::num_dims() const{
    return shape.size();
}

Tensor QTensor2Tensor(const QuantizedTensor& q_tensor){
    if(q_tensor.type != QuantType::kF32){
        throw std::runtime_error("QTensor2Tensor: plain tensor only accecpt F32 data");
    }

    Tensor t(q_tensor.shape, 0.0f);
    if(q_tensor.f32_data.size() != t.TensorNum()){
        throw std::runtime_error("QTensor2Tensor: plain tensor init error, data size is not matched");
    }

    t.data = q_tensor.f32_data;
    return t;
}

QuantizedTensor Tensor2QTensor(const Tensor& tensor){
    QuantizedTensor q_tensor;
    q_tensor.shape = tensor.shape;
    q_tensor.f32_data = tensor.data;
    q_tensor.type = QuantType::kF32;
    return q_tensor;
}

}
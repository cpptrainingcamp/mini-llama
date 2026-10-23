#include "../include/ops.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace LLaMA_Tensor{
    void CheckTensorZero(const Tensor& x){
        if(!x.TensorNum()){
            throw std::out_of_range("zero dimension tensor is not permitted");
        }
    }

    void CheckTensorMulti(const Tensor& t1, const Tensor& t2){
        CheckTensorZero(t1);
        CheckTensorZero(t2);
        if(t1.dim() > 2 || t2.dim() > 2){
            throw std::runtime_error("MatrixMulti only supports 1D vectors and 2D matrices");
        }
        //一维向量方向约定(PyTorch matmul 风格):
        //  1D 左操作数视为行向量 {1, n}, 其"列数"为 shape[0];
        //  1D 右操作数视为列向量 {n, 1}, 其"行数"恒为 shape[0].
        const int t1_cols = (t1.dim() == 1) ? t1.shape[0] : t1.shape[1];
        const int t2_rows = t2.shape[0];
        if(t1_cols != t2_rows){
            t1.TensorPrint();
            t2.TensorPrint();
            throw std::runtime_error("tensors' dimensions are not matched");
        }
    }

    void CheckTensorSameShape(const Tensor& t1, const Tensor& t2){
        if(t1.shape != t2.shape){
            throw std::runtime_error("expect the same shape tensors");
        }
    }

    Tensor TensorTrans(const Tensor& x){
        if(x.dim() == 1){
            //一维向量视为行向量 {1, n}, 转置得到列向量 {n, 1}
            Tensor col(x);
            col.shape = {x.shape[0], 1};
            return col;
        }
        x.CheckDim(2);
        Tensor trans;
        trans.data = std::vector<float>(x.TensorNum(), 0.0f);
        trans.shape = {x.shape[1], x.shape[0]};
        for(int i = 0;i<x.shape[0];i++){
            for(int j = 0;j<x.shape[1];j++){
                trans.At2(j, i) = x.At2(i, j);
            }
        }
        return trans;
    }

    Tensor MatrixMulti(const Tensor& t1, const Tensor& t2){
        return MatmulDispatch(t1, t2, DefaultMatmulMode());
    }

    Tensor Linear(const Tensor& x, const Tensor& weight){
        weight.CheckDim(2);   // weight 必须是 2D (out_features × in_features)
        return LinearDispatch(x, weight, DefaultMatmulMode());
    }

    Tensor RMSNorm(const Tensor& x, const Tensor& weight, float eps){
        if(!std::isfinite(eps) || eps <= 0.0f){
            throw std::runtime_error("eps must be positive and finite");
        }
        CheckTensorSameShape(x, weight);
        Tensor RMS;
        float sum = 0.0f;
        for(float num : x.data){
            sum += num*num;
        }
        float rms = std::sqrt(sum / x.TensorNum() + eps);
        std::vector<float>x_rms(x.TensorNum(), 0.0f);
        for(size_t i = 0;i<x_rms.size();i++){
            x_rms[i] = x.data[i] / rms;
        }
        RMS.data = x_rms;
        RMS.shape = x.shape;
        return ElementsWiseMulti(RMS, weight);
    }

    Tensor RMSNormRows(const Tensor& x, const Tensor& weight, float eps){
        if(!std::isfinite(eps) || eps <= 0.0f){
            throw std::runtime_error("eps must be positive and finite");
        }
        if(x.dim() != 2){
            throw std::runtime_error("RMSNormRows expects x shape [rows, dim]");
        }
        if(weight.dim() != 1){
            throw std::runtime_error("RMSNormRows expects weight shape [dim]");
        }
        if(x.shape[1] != weight.shape[0]){
            throw std::runtime_error("RMSNormRows: x's feature dim is not matched with weight's");
        }
        const int rows = x.shape[0];
        const int dim = x.shape[1];
        Tensor result(x.shape, 0.0f);
        for(int r = 0;r < rows;r++){
            const float* x_row = x.data.data() + static_cast<size_t>(r) * dim;
            float* out_row = result.data.data() + static_cast<size_t>(r) * dim;
            float sum = 0.0f;
            for(int j = 0;j < dim;j++){
                sum += x_row[j] * x_row[j];
            }
            float rms = std::sqrt(sum / static_cast<float>(dim) + eps);
            for(int j = 0;j < dim;j++){
                out_row[j] = (x_row[j] / rms) * weight.data[j];
            }
        }
        return result;
    }

    size_t ArgsMax(const Tensor& x){
        CheckTensorZero(x);
        auto it = std::max_element(x.data.begin(), x.data.end());
        return it - x.data.begin();
    }

    Tensor softmax(const Tensor& x){
        CheckTensorZero(x);
        Tensor softmax_tensor(x);
        float x_max = x.At1(ArgsMax(x));
        std::vector<float>&vec = softmax_tensor.data;
        for(float& num : vec){
            num = std::exp(num - x_max);
        }
        float x_exp_sum = std::accumulate(vec.begin(), vec.end(), 0.0);
        for(float& num : vec){
            num /= x_exp_sum;
        }
        return softmax_tensor;
    }

    Tensor SiLU(const Tensor& x){
        Tensor SiLU_tensor(x);
        std::vector<float>&vec = SiLU_tensor.data;
        for(float& num : vec){
            num = num / (1.0f+std::exp(-num));
        }
        return SiLU_tensor;
    }

    Tensor ElementsWiseMulti(const Tensor& x, const Tensor& y){
        CheckTensorSameShape(x, y);
        Tensor result(x);
        for(size_t i = 0;i<result.data.size();i++){
            result.data[i] *= y.data[i];
        }
        return result;
    }

    Tensor SWiGLU(const Tensor& gate, const Tensor& up){
        CheckTensorSameShape(gate, up);
        return ElementsWiseMulti(SiLU(gate), up);
    }

    //对一段 [heads, head_dim] 的行主序数据原地施加 Normal 风格 RoPE
    void ApplyNormalData(float* data, int heads, int head_dim, int pos, float theta){
        for(int i = 0;i<heads;i++){
            float* row = data + static_cast<size_t>(i) * head_dim;
            for(int j = 0;j<head_dim;j += 2){
                float freq = std::pow(theta, -(static_cast<float>(j) / static_cast<float>(head_dim)));
                float cos_val = std::cos(pos * freq);
                float sin_val = std::sin(pos * freq);
                float x0 = row[j];
                float x1 = row[j+1];
                row[j] = cos_val * x0 - sin_val * x1;
                row[j+1] = sin_val * x0 + cos_val * x1;
            }
        }
    }

    void ApplyNormal(Tensor& x, int pos, float theta){
        x.CheckDim(2);
        ApplyNormalData(x.data.data(), x.shape[0], x.shape[1], pos, theta);
    }

    //对一段 [heads, head_dim] 的行主序数据原地施加 NeoX 风格 RoPE
    void ApplyNeoXData(float* data, int heads, int head_dim, int pos, float theta){
        for(int i = 0;i<heads;i++){
            float* row = data + static_cast<size_t>(i) * head_dim;
            for(int j=0;j<head_dim/2;j++){
                float freq = std::pow(theta, -(static_cast<float>(2*j) / static_cast<float>(head_dim)));
                float cos_val = std::cos(pos * freq);
                float sin_val = std::sin(pos * freq);
                float x0 = row[j];
                float x1 = row[j+head_dim/2];
                row[j] = cos_val * x0 - sin_val * x1;
                row[j+head_dim/2] = sin_val * x0 + cos_val * x1;
            }
        }
    }

    void ApplyNeoX(Tensor& x, int pos, float theta){
        x.CheckDim(2);
        ApplyNeoXData(x.data.data(), x.shape[0], x.shape[1], pos, theta);
    }

    void rope(Tensor& Q, Tensor& K, int pos, float theta, RoPEtype type){
        Q.CheckDim(2);
        K.CheckDim(2);
        if(pos < 0){
            throw std::out_of_range("Invalid token position, pos must >= 0");
        }
        if(Q.shape[1] != K.shape[1]){
            throw std::runtime_error("Q's and K's head dimensions are not mathed, they nust be equal");
        }
        if(!std::isfinite(theta) || theta <= 0.0f){
            throw std::runtime_error("theta must be positive and finite");
        }
        if(Q.shape[1] <= 0 || Q.shape[1] % 2 != 0){
            throw std::runtime_error("Invalid Q shape, Q's cols must be positive and even");
        }

        if(type == RoPEtype::Normal){
            ApplyNormal(Q, pos, theta);
            ApplyNormal(K, pos, theta);
        }else{
            ApplyNeoX(Q, pos, theta);
            ApplyNeoX(K, pos, theta);
        }
    }

    void rope_batch(Tensor& Q, Tensor& K, const std::vector<int>& positions,
                    float theta, RoPEtype type){
        Q.CheckDim(3);
        K.CheckDim(3);
        if(Q.shape[0] != K.shape[0]){
            throw std::runtime_error("rope_batch: Q and K must have the same batch size");
        }
        if(Q.shape[2] != K.shape[2]){
            throw std::runtime_error("Q's and K's head dimensions are not mathed, they nust be equal");
        }
        if(static_cast<int>(positions.size()) != Q.shape[0]){
            throw std::runtime_error("rope_batch: positions count must match the batch size");
        }
        if(!std::isfinite(theta) || theta <= 0.0f){
            throw std::runtime_error("theta must be positive and finite");
        }
        if(Q.shape[2] <= 0 || Q.shape[2] % 2 != 0){
            throw std::runtime_error("Invalid Q shape, Q's head dim must be positive and even");
        }
        const int batch = Q.shape[0];
        const int q_heads = Q.shape[1];
        const int k_heads = K.shape[1];
        const int head_dim = Q.shape[2];
        for(int b = 0;b < batch;b++){
            if(positions[b] < 0){
                throw std::out_of_range("Invalid token position, pos must >= 0");
            }
        }
        for(int b = 0;b < batch;b++){
            float* q_data = Q.data.data() + static_cast<size_t>(b) * q_heads * head_dim;
            float* k_data = K.data.data() + static_cast<size_t>(b) * k_heads * head_dim;
            if(type == RoPEtype::Normal){
                ApplyNormalData(q_data, q_heads, head_dim, positions[b], theta);
                ApplyNormalData(k_data, k_heads, head_dim, positions[b], theta);
            }else{
                ApplyNeoXData(q_data, q_heads, head_dim, positions[b], theta);
                ApplyNeoXData(k_data, k_heads, head_dim, positions[b], theta);
            }
        }
    }

    Tensor Linear(const Tensor& x, const QuantizedTensor& weight) {
        switch (weight.type) {
            case QuantType::kF32:
            return Linear(x, QTensor2Tensor(weight));
            case QuantType::kQ80:
            return LinearQ80(x, weight.q80_data, weight.shape);
            case QuantType::kQ40:
            return LinearQ40(x, weight.q40_data, weight.shape);
            case QuantType::kQ41:
            return LinearQ41(x, weight.q41_data, weight.shape);
        }
        throw std::runtime_error("Linear: unknown quant type");
    }

}

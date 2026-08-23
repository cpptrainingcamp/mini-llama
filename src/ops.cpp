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
        t1.CheckDim(2);
        t2.CheckDim(2);
        if(t1.shape[1] != t2.shape[0]){
            t1.TensorPrint();
            t2.TensorPrint();
            throw std::runtime_error("Tensors' col and row are not matched, they can not multiply.");
        }
    }

    void CheckTensorSameShape(const Tensor& t1, const Tensor& t2){
        if(t1.shape != t2.shape){
            throw std::runtime_error("expect the same shape tensors");
        }
    }

    Tensor TensorTrans(const Tensor& x){
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
        CheckTensorMulti(t1, t2);
        int M = t1.shape[0];
        int K = t1.shape[1];
        int N = t2.shape[1];
        Tensor multi_tensor;
        multi_tensor.data = std::vector<float>(M * N, 0.0f);
        multi_tensor.shape = {M, N};
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                float a = t1.At2(i, k);
                    for (int j = 0; j < N; ++j) {
                        multi_tensor.At2(i, j) += a * t2.At2(k, j);
                    }
            }
        }
        return multi_tensor;
    }

    Tensor Linear(const Tensor& x, const Tensor& weight){
        return MatrixMulti(x, TensorTrans(weight));
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
        float rms = std::sqrtf(sum / x.TensorNum() + eps);
        std::vector<float>x_rms(x.TensorNum(), 0.0f);
        for(size_t i = 0;i<x_rms.size();i++){
            x_rms[i] = x.data[i] / rms;
        }
        RMS.data = x_rms;
        RMS.shape = x.shape;
        return ElementsWiseMulti(RMS, weight);
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

    void ApplyNormal(Tensor& x, int pos, float theta){
        x.CheckDim(2);
        int heads = x.shape[0];
        int head_dim = x.shape[1];
        for(int i = 0;i<heads;i++){
            for(int j = 0;j<head_dim;j += 2){
                float freq = std::powf(theta, -(static_cast<float>(j) / static_cast<float>(head_dim)));
                float cos_val = std::cosf(pos * freq);
                float sin_val = std::sinf(pos * freq);
                float x0 = x.At2(i, j);
                float x1 = x.At2(i, j+1);
                x.At2(i, j) = cos_val * x0 - sin_val * x1;
                x.At2(i, j+1) = sin_val * x0 + cos_val * x1;
            }
        }
    }

    void ApplyNeoX(Tensor& x, int pos, float theta){
        x.CheckDim(2);
        int heads  = x.shape[0];
        int head_dim = x.shape[1];
        for(int i = 0;i<heads;i++){
            for(int j=0;j<head_dim/2;j++){
                float freq = std::powf(theta, -(static_cast<float>(2*j) / static_cast<float>(head_dim)));
                float cos_val = std::cosf(pos * freq);
                float sin_val = std::sinf(pos * freq);
                float x0 = x.At2(i, j);
                float x1 = x.At2(i, j+head_dim/2);
                x.At2(i, j) = cos_val * x0 - sin_val * x1;
                x.At2(i, j+head_dim/2) = sin_val * x0 + cos_val * x1;
            }
        }
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

}

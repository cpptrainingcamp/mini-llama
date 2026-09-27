#include "model.h"

namespace LLaMA_Tensor{
namespace{
size_t TensorBytes(const Tensor& tensor){
    return tensor.TensorNum() * sizeof(float);
}

size_t TensorBytes(const QuantizedTensor& q_tensor){
    switch (q_tensor.type) {
        case QuantType::kF32:
        return q_tensor.f32_data.size() * sizeof(float);
        case QuantType::kQ80:
        return q_tensor.q80_data.size() * sizeof(BlockQ80);
        case QuantType::kQ40:
        return q_tensor.q40_data.size() * sizeof(BlockQ40);
        case QuantType::kQ41:
        return q_tensor.q41_data.size() * sizeof(BlockQ41);
    }
    return 0;
}

}

size_t ModelWeightBytes(const MiniLlamaModel& model){
    size_t bytes = 0;
    bytes += TensorBytes(model.token_embedding);
    bytes += TensorBytes(model.lm_head);
    bytes += TensorBytes(model.final_norm);
    for(const LayerWeights& lw : model.layers){
        bytes += TensorBytes(lw.attention_norm);
        bytes += TensorBytes(lw.bk);
        bytes += TensorBytes(lw.bq);
        bytes += TensorBytes(lw.bv);
        bytes += TensorBytes(lw.ffn_norm);
        bytes += TensorBytes(lw.w_down);
        bytes += TensorBytes(lw.w_gate);
        bytes += TensorBytes(lw.w_up);
        bytes += TensorBytes(lw.wk);
        bytes += TensorBytes(lw.wo);
        bytes += TensorBytes(lw.wq);
        bytes += TensorBytes(lw.wv);
    }
    return bytes;
}
}
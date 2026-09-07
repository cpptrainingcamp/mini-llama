#include "model.h"

namespace LLaMA_Tensor{
namespace{
size_t TensorBytes(const Tensor& tensor){
    return tensor.TensorNum() * sizeof(float);
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
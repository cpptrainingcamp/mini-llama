#include "../include/context.h"

namespace LLaMA_Tensor{
Context::Context(const MiniLlamaModel* model_ptr)
    : model(model_ptr), pos(0) {
  if (model) {
    const auto& c = model->config;
    kv_cache = KVCache(c.n_layers, c.max_seq_len, c.n_kv_heads, c.head_dim);
    if (model->cuda_weights) {
      cuda_kv_cache.Reset(c.n_layers, c.max_seq_len, c.n_kv_heads, c.head_dim,
                          model->cuda_weights->device_id);
    }
  }
}
}

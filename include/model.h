#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "tensor.h"
#include "ops.h"
#include "quantized_tensor.h"

namespace LLaMA_Tensor{
//模型超参数
struct ModelConfig{
    int vocab_size = 128;
    int dim = 32;
    int hiddendim = 86;
    int n_layers = 2;
    int n_heads = 4;
    int n_kv_heads = 4;
    int head_dim = 8;
    int max_seq_len = 128;
    float rope_theta = 10000.0f;
    float rms_norm_eps = 1e-5f;
    RoPEtype rope_type = RoPEtype::Normal;
};
//单层Transformer权重信息
struct LayerWeights {
  Tensor attention_norm;   // [dim] F32 — 注意力前的 RMSNorm 权重
  QuantizedTensor wq;               // [n_heads * head_dim, dim] F32 — Q 投影
  QuantizedTensor wk;               // [n_kv_heads * head_dim, dim] F32 — K 投影
  QuantizedTensor wv;               // [n_kv_heads * head_dim, dim] F32 — V 投影
  Tensor bq;               // optional [n_heads * head_dim] F32 — Q 偏置
  Tensor bk;               // optional [n_kv_heads * head_dim] F32 — K 偏置
  Tensor bv;               // optional [n_kv_heads * head_dim] F32 — V 偏置
  Tensor wo;               // [dim, n_heads * head_dim] F32 — 注意力输出投影
  Tensor ffn_norm;         // [dim] F32 — FFN 前的 RMSNorm 权重
  QuantizedTensor w_gate;           // [hidden_dim, dim] F32 — SwiGLU 门控投影
  QuantizedTensor w_up;             // [hidden_dim, dim] F32 — SwiGLU 上投影
  QuantizedTensor w_down;           // [dim, hidden_dim] F32 — SwiGLU 下投影
};
//完整模型(CPU,F32)
struct MiniLlamaModel{
    ModelConfig config;
    bool loaded = false;
    std::string load_error;
    Tensor token_embedding;            // [vocab_size, dim] F32
    std::vector<LayerWeights> layers;  // [n_layers]
    Tensor final_norm;                 // [dim] F32
    QuantizedTensor lm_head;                    // [vocab_size, dim] F32
};
//模型权重内存统计
size_t ModelWeightBytes(const MiniLlamaModel& model);

}
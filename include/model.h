#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "cuda_runtime.h"
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
  QuantizedTensor wo;               // [dim, n_heads * head_dim] F32 — 注意力输出投影
  Tensor ffn_norm;         // [dim] F32 — FFN 前的 RMSNorm 权重
  QuantizedTensor w_gate;           // [hidden_dim, dim] F32 — SwiGLU 门控投影
  QuantizedTensor w_up;             // [hidden_dim, dim] F32 — SwiGLU 上投影
  QuantizedTensor w_down;           // [dim, hidden_dim] F32 — SwiGLU 下投影
};
//CUDA 设备端权重：一个 CudaUploadedWeight 对应模型中的一个张量
struct CudaUploadedWeight{
    std::string name;
    QuantType type = QuantType::kF32;
    bool linear = true;              // true: Linear 权重, false: embedding/norm/bias
    std::vector<int> shape;
    size_t block_count = 0;          // Q8_0 / Q4_0 / Q4_1 的 block 数量
    CudaDeviceBuffer buffer;
};

//模型所有 CUDA 常驻权重及运行期统计
struct CudaModelWeights{
    int device_id = 0;
    size_t uploaded_weight_count = 0;
    size_t uploaded_bytes = 0;
    size_t linear_calls = 0;
    size_t activation_calls = 0;
    size_t attention_calls = 0;
    size_t attention_cpu_fallbacks = 0;
    size_t kv_cache_write_bytes = 0;
    size_t kv_cache_read_bytes = 0;
    size_t host_to_device_copies = 0;
    size_t device_to_host_copies = 0;
    size_t host_to_device_bytes = 0;
    size_t device_to_host_bytes = 0;
    std::vector<CudaUploadedWeight> weights;
};

//完整模型
struct MiniLlamaModel{
    ModelConfig config;
    bool loaded = false;
    std::string load_error;
    Tensor token_embedding;            // [vocab_size, dim] F32
    std::vector<LayerWeights> layers;  // [n_layers]
    Tensor final_norm;                 // [dim] F32
    QuantizedTensor lm_head;                    // [vocab_size, dim] F32
    //CUDA 后端选择后上传；CPU 路径保持为空
    std::shared_ptr<CudaModelWeights> cuda_weights;
};
//模型权重内存统计
size_t ModelWeightBytes(const MiniLlamaModel& model);

//把 CUDA 支持的权重上传到设备，CPU 侧权重保持不变
void UploadModelWeightsToCuda(MiniLlamaModel& model, int device_id = 0);
void ClearModelCudaWeights(MiniLlamaModel& model);
bool ModelHasCudaWeights(const MiniLlamaModel& model);

size_t ModelCudaUploadedWeightCount(const MiniLlamaModel& model);
size_t ModelCudaUploadedF32WeightCount(const MiniLlamaModel& model);
size_t ModelCudaUploadedQ80WeightCount(const MiniLlamaModel& model);
size_t ModelCudaUploadedQ40WeightCount(const MiniLlamaModel& model);
size_t ModelCudaUploadedQ41WeightCount(const MiniLlamaModel& model);
size_t ModelCudaMemoryBytes(const MiniLlamaModel& model);

void ResetModelCudaRuntimeStats(MiniLlamaModel& model);
size_t ModelCudaLinearCalls(const MiniLlamaModel& model);
size_t ModelCudaActivationCalls(const MiniLlamaModel& model);
size_t ModelCudaAttentionCalls(const MiniLlamaModel& model);
size_t ModelCudaAttentionCpuFallbacks(const MiniLlamaModel& model);
size_t ModelCudaKvCacheWriteBytes(const MiniLlamaModel& model);
size_t ModelCudaKvCacheReadBytes(const MiniLlamaModel& model);
size_t ModelCudaHostToDeviceCopies(const MiniLlamaModel& model);
size_t ModelCudaDeviceToHostCopies(const MiniLlamaModel& model);
size_t ModelCudaHostToDeviceBytes(const MiniLlamaModel& model);
size_t ModelCudaDeviceToHostBytes(const MiniLlamaModel& model);

}
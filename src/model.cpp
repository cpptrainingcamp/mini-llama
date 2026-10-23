#include "model.h"

#include <stdexcept>
#include <string>
#include <utility>

#include "quant.h"

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
static void UploadF32LinearWeight(CudaModelWeights& storage,
                                  const QuantizedTensor& weight,
                                  const std::string& name, int device_id){
    if(weight.type != QuantType::kF32){
        return;
    }
    if(weight.f32_data.empty()){
        throw std::runtime_error("CUDA weight upload: empty F32 data for " + name);
    }
    const size_t expected = weight.ele_num();
    if(weight.f32_data.size() != expected){
        throw std::runtime_error("CUDA weight upload: data size mismatch for " + name
            + ", expected " + std::to_string(expected) + ", got "
            + std::to_string(weight.f32_data.size()));
    }

    const size_t bytes = weight.f32_data.size() * sizeof(float);
    CudaUploadedWeight uploaded;
    uploaded.name = name;
    uploaded.type = QuantType::kF32;
    uploaded.linear = true;
    uploaded.shape = weight.shape;
    uploaded.block_count = 0;
    uploaded.buffer.Reset(bytes, device_id);
    uploaded.buffer.Upload(weight.f32_data.data(), bytes);

    storage.uploaded_weight_count += 1;
    storage.uploaded_bytes += bytes;
    storage.weights.push_back(std::move(uploaded));
}

static void UploadF32TensorWeight(CudaModelWeights& storage,
                                  const Tensor& weight, const std::string& name,
                                  int device_id){
    if(weight.data.empty()){
        return;
    }
    const size_t expected = weight.size();
    if(weight.data.size() != expected){
        throw std::runtime_error("CUDA weight upload: tensor data size mismatch for "
            + name + ", expected " + std::to_string(expected) + ", got "
            + std::to_string(weight.data.size()));
    }

    const size_t bytes = weight.data.size() * sizeof(float);
    CudaUploadedWeight uploaded;
    uploaded.name = name;
    uploaded.type = QuantType::kF32;
    uploaded.linear = false;
    uploaded.shape = weight.shape;
    uploaded.block_count = 0;
    uploaded.buffer.Reset(bytes, device_id);
    uploaded.buffer.Upload(weight.data.data(), bytes);

    storage.uploaded_weight_count += 1;
    storage.uploaded_bytes += bytes;
    storage.weights.push_back(std::move(uploaded));
}

static void UploadQ80LinearWeight(CudaModelWeights& storage,
                                  const QuantizedTensor& weight,
                                  const std::string& name, int device_id){
    if(weight.type != QuantType::kQ80){
        return;
    }
    if(weight.shape.size() != 2){
        throw std::runtime_error(
            "CUDA weight upload: expected 2D Q8_0 weight for " + name);
    }
    if(weight.q80_data.empty()){
        throw std::runtime_error("CUDA weight upload: empty Q8_0 data for " + name);
    }

    const int out_features = weight.shape[0];
    const int in_features = weight.shape[1];
    const int blocks_per_row = (in_features + kQ80BlockSize - 1) / kQ80BlockSize;
    const size_t expected_blocks =
        static_cast<size_t>(out_features) * blocks_per_row;
    if(weight.q80_data.size() != expected_blocks){
        throw std::runtime_error("CUDA weight upload: Q8_0 block count mismatch for "
            + name + ", expected " + std::to_string(expected_blocks) + ", got "
            + std::to_string(weight.q80_data.size()));
    }

    const size_t bytes = weight.q80_data.size() * sizeof(BlockQ80);
    CudaUploadedWeight uploaded;
    uploaded.name = name;
    uploaded.type = QuantType::kQ80;
    uploaded.linear = true;
    uploaded.shape = weight.shape;
    uploaded.block_count = weight.q80_data.size();
    uploaded.buffer.Reset(bytes, device_id);
    uploaded.buffer.Upload(weight.q80_data.data(), bytes);

    storage.uploaded_weight_count += 1;
    storage.uploaded_bytes += bytes;
    storage.weights.push_back(std::move(uploaded));
}


static void UploadQ40LinearWeight(CudaModelWeights& storage,
                                  const QuantizedTensor& weight,
                                  const std::string& name, int device_id){
    if(weight.type != QuantType::kQ40){
        return;
    }
    if(weight.shape.size() != 2){
        throw std::runtime_error(
            "CUDA weight upload: expected 2D Q4_0 weight for " + name);
    }
    if(weight.q40_data.empty()){
        throw std::runtime_error("CUDA weight upload: empty Q4_0 data for " + name);
    }

    const int out_features = weight.shape[0];
    const int in_features = weight.shape[1];
    const int blocks_per_row = (in_features + kQ40BlockSize - 1) / kQ40BlockSize;
    const size_t expected_blocks =
        static_cast<size_t>(out_features) * blocks_per_row;
    if(weight.q40_data.size() != expected_blocks){
        throw std::runtime_error("CUDA weight upload: Q4_0 block count mismatch for "
            + name + ", expected " + std::to_string(expected_blocks) + ", got "
            + std::to_string(weight.q40_data.size()));
    }

    const size_t bytes = weight.q40_data.size() * sizeof(BlockQ40);
    CudaUploadedWeight uploaded;
    uploaded.name = name;
    uploaded.type = QuantType::kQ40;
    uploaded.linear = true;
    uploaded.shape = weight.shape;
    uploaded.block_count = weight.q40_data.size();
    uploaded.buffer.Reset(bytes, device_id);
    uploaded.buffer.Upload(weight.q40_data.data(), bytes);

    storage.uploaded_weight_count += 1;
    storage.uploaded_bytes += bytes;
    storage.weights.push_back(std::move(uploaded));
}

static void UploadQ41LinearWeight(CudaModelWeights& storage,
                                  const QuantizedTensor& weight,
                                  const std::string& name, int device_id){
    if(weight.type != QuantType::kQ41){
        return;
    }
    if(weight.shape.size() != 2){
        throw std::runtime_error(
            "CUDA weight upload: expected 2D Q4_1 weight for " + name);
    }
    if(weight.q41_data.empty()){
        throw std::runtime_error("CUDA weight upload: empty Q4_1 data for " + name);
    }

    const int out_features = weight.shape[0];
    const int in_features = weight.shape[1];
    const int blocks_per_row = (in_features + kQ41BlockSize - 1) / kQ41BlockSize;
    const size_t expected_blocks =
        static_cast<size_t>(out_features) * blocks_per_row;
    if(weight.q41_data.size() != expected_blocks){
        throw std::runtime_error("CUDA weight upload: Q4_1 block count mismatch for "
            + name + ", expected " + std::to_string(expected_blocks) + ", got "
            + std::to_string(weight.q41_data.size()));
    }

    const size_t bytes = weight.q41_data.size() * sizeof(BlockQ41);
    CudaUploadedWeight uploaded;
    uploaded.name = name;
    uploaded.type = QuantType::kQ41;
    uploaded.linear = true;
    uploaded.shape = weight.shape;
    uploaded.block_count = weight.q41_data.size();
    uploaded.buffer.Reset(bytes, device_id);
    uploaded.buffer.Upload(weight.q41_data.data(), bytes);

    storage.uploaded_weight_count += 1;
    storage.uploaded_bytes += bytes;
    storage.weights.push_back(std::move(uploaded));
}

static void UploadLinearWeight(CudaModelWeights& storage,
                               const QuantizedTensor& weight,
                               const std::string& name, int device_id){
    UploadF32LinearWeight(storage, weight, name, device_id);
    UploadQ80LinearWeight(storage, weight, name, device_id);
    UploadQ40LinearWeight(storage, weight, name, device_id);
    UploadQ41LinearWeight(storage, weight, name, device_id);
}

void UploadModelWeightsToCuda(MiniLlamaModel& model, int device_id){
    if(!model.loaded){
        throw std::runtime_error("CUDA weight upload requires a loaded model");
    }

    CudaSetDevice(device_id);
    auto storage = std::make_shared<CudaModelWeights>();
    storage->device_id = device_id;

    UploadF32TensorWeight(*storage, model.token_embedding, "token_embedding",
                          device_id);
    UploadF32TensorWeight(*storage, model.final_norm, "final_norm", device_id);
    UploadLinearWeight(*storage, model.lm_head, "lm_head", device_id);
    for(size_t layer = 0;layer < model.layers.size();layer++){
        const LayerWeights& lw = model.layers[layer];
        const std::string prefix = "layers." + std::to_string(layer) + ".";
        UploadF32TensorWeight(*storage, lw.attention_norm,
                              prefix + "attention_norm", device_id);
        UploadLinearWeight(*storage, lw.wq, prefix + "wq", device_id);
        UploadLinearWeight(*storage, lw.wk, prefix + "wk", device_id);
        UploadLinearWeight(*storage, lw.wv, prefix + "wv", device_id);
        UploadF32TensorWeight(*storage, lw.bq, prefix + "bq", device_id);
        UploadF32TensorWeight(*storage, lw.bk, prefix + "bk", device_id);
        UploadF32TensorWeight(*storage, lw.bv, prefix + "bv", device_id);
        UploadLinearWeight(*storage, lw.wo, prefix + "wo", device_id);
        UploadF32TensorWeight(*storage, lw.ffn_norm, prefix + "ffn_norm",
                              device_id);
        UploadLinearWeight(*storage, lw.w_gate, prefix + "w_gate", device_id);
        UploadLinearWeight(*storage, lw.w_up, prefix + "w_up", device_id);
        UploadLinearWeight(*storage, lw.w_down, prefix + "w_down", device_id);
    }

    model.cuda_weights = std::move(storage);
}


void ClearModelCudaWeights(MiniLlamaModel& model){
    model.cuda_weights.reset();
}

bool ModelHasCudaWeights(const MiniLlamaModel& model){
    return model.cuda_weights != nullptr;
}

size_t ModelCudaUploadedWeightCount(const MiniLlamaModel& model){
    return model.cuda_weights ? model.cuda_weights->uploaded_weight_count : 0;
}

static size_t ModelCudaUploadedWeightCountByType(const MiniLlamaModel& model,
                                                 QuantType type){
    if(!model.cuda_weights){
        return 0;
    }
    size_t count = 0;
    for(const CudaUploadedWeight& weight : model.cuda_weights->weights){
        if(weight.linear && weight.type == type){
            count += 1;
        }
    }
    return count;
}

size_t ModelCudaUploadedF32WeightCount(const MiniLlamaModel& model){
    return ModelCudaUploadedWeightCountByType(model, QuantType::kF32);
}

size_t ModelCudaUploadedQ80WeightCount(const MiniLlamaModel& model){
    return ModelCudaUploadedWeightCountByType(model, QuantType::kQ80);
}

size_t ModelCudaUploadedQ40WeightCount(const MiniLlamaModel& model){
    return ModelCudaUploadedWeightCountByType(model, QuantType::kQ40);
}

size_t ModelCudaUploadedQ41WeightCount(const MiniLlamaModel& model){
    return ModelCudaUploadedWeightCountByType(model, QuantType::kQ41);
}

size_t ModelCudaMemoryBytes(const MiniLlamaModel& model){
    return model.cuda_weights ? model.cuda_weights->uploaded_bytes : 0;
}

void ResetModelCudaRuntimeStats(MiniLlamaModel& model){
    if(!model.cuda_weights){
        return;
    }
    model.cuda_weights->linear_calls = 0;
    model.cuda_weights->activation_calls = 0;
    model.cuda_weights->attention_calls = 0;
    model.cuda_weights->attention_cpu_fallbacks = 0;
    model.cuda_weights->kv_cache_write_bytes = 0;
    model.cuda_weights->kv_cache_read_bytes = 0;
    model.cuda_weights->host_to_device_copies = 0;
    model.cuda_weights->device_to_host_copies = 0;
    model.cuda_weights->host_to_device_bytes = 0;
    model.cuda_weights->device_to_host_bytes = 0;
}

#define MINI_LLAMA_CUDA_STAT_GETTER(func, member) \
    size_t func(const MiniLlamaModel& model){ \
        return model.cuda_weights ? model.cuda_weights->member : 0; \
    }

MINI_LLAMA_CUDA_STAT_GETTER(ModelCudaLinearCalls, linear_calls)
MINI_LLAMA_CUDA_STAT_GETTER(ModelCudaActivationCalls, activation_calls)
MINI_LLAMA_CUDA_STAT_GETTER(ModelCudaAttentionCalls, attention_calls)
MINI_LLAMA_CUDA_STAT_GETTER(ModelCudaAttentionCpuFallbacks,
                            attention_cpu_fallbacks)
MINI_LLAMA_CUDA_STAT_GETTER(ModelCudaKvCacheWriteBytes, kv_cache_write_bytes)
MINI_LLAMA_CUDA_STAT_GETTER(ModelCudaKvCacheReadBytes, kv_cache_read_bytes)
MINI_LLAMA_CUDA_STAT_GETTER(ModelCudaHostToDeviceCopies, host_to_device_copies)
MINI_LLAMA_CUDA_STAT_GETTER(ModelCudaDeviceToHostCopies, device_to_host_copies)
MINI_LLAMA_CUDA_STAT_GETTER(ModelCudaHostToDeviceBytes, host_to_device_bytes)
MINI_LLAMA_CUDA_STAT_GETTER(ModelCudaDeviceToHostBytes, device_to_host_bytes)

#undef MINI_LLAMA_CUDA_STAT_GETTER


}
#pragma once

#include "model.h"

namespace LLaMA_Tensor{
struct TensorInfo {
    std::string name;
    std::vector<int> shape;
    std::string dtype;
    size_t offset = 0;
    size_t byte_size = 0;
};

struct TokenizerInfo {
    std::string type = "ascii";
    std::string path;
    int bos_id = 1;
    int eos_id = 2;
    int unk_id = 0;
};

struct ModelManifest {
    ModelConfig config;
    TokenizerInfo tokenizer;
    std::vector<TensorInfo> tensors;
};
//解析model.json到modelmanifest
ModelManifest ParseManifest(const std::string& config_path);
//加载模型权重
MiniLlamaModel LoadModel(const std::string& config_path, const std::string& weights_path);
//检查模型格式
bool InspectModel(const std::string& config_path);

}
#include "loader.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace LLaMA_Tensor{
namespace{
//创建一个标记为加载失败的 MiniLlamaModel 对象。
MiniLlamaModel LoadFailure(const std::string& message){
    MiniLlamaModel model;
    model.loaded = false;
    model.load_error = message;
    std::cerr << "fail to load model: " << message << std::endl;
    return model;
}
//在 JSON 字符串中查找一个必需键，定位到其值的起始位置。
size_t FindConfigKey(const std::string& content, const std::string& key){
    const std::string marker = "\"" + key + "\"";
    size_t key_pos = content.find(marker);
    if(key_pos == std::string::npos){
        throw std::runtime_error("missing config key: " + key);
    }
    size_t colon_pos = content.find(":", key_pos + marker.size());
    if(colon_pos == std::string::npos){
        throw std::runtime_error("missing \":\" after key, json format is wrong");
    }
    size_t value_pos = ++colon_pos;
    while(value_pos < content.size() && std::isspace(static_cast<unsigned char>(content[value_pos]))){
        value_pos++;
    }
    if(value_pos == content.size()){
        throw std::runtime_error("missing config value of key: " + key);
    }
    return value_pos;
}
//在 JSON 字符串中查找一个可选键，定位到值的起始位置。
size_t FindOptionalKey(const std::string& content, const std::string& key){
    const std::string marker = "\"" + key + "\"";
    size_t key_pos = content.find(marker);
    if(key_pos == std::string::npos){
        return key_pos;
    }
    size_t colon_pos = content.find(":", key_pos + marker.size());
    if(colon_pos == std::string::npos){
        throw std::runtime_error("missing \":\" after key, json format is wrong");
    }
    size_t value_pos = ++colon_pos;
    while(value_pos < content.size() && std::isspace(static_cast<unsigned char>(content[value_pos]))){
        value_pos++;
    }
    if(value_pos == content.size()){
        throw std::runtime_error("missing config value of key: " + key);
    }
    return value_pos;
}
//判断字符是否为 JSON 值的合法终止符。
bool IsJsonValueTerminator(char c){
    return std::isspace(static_cast<unsigned char>(c)) || c == ',' || c == '}' || c == ']';
}
//验证已解析的数值后面紧跟着合法的 JSON 终止符。
void ValidateNumberTerminator(const std::string& content, size_t value_pos, size_t parsed, const std::string& key){
    size_t pos = value_pos + parsed;
    while (pos < content.size() &&
         std::isspace(static_cast<unsigned char>(content[pos]))) {
        ++pos;
    }
    if (pos < content.size() && !IsJsonValueTerminator(content[pos])) {
        throw std::runtime_error("invalid numeric value for config key: " + key);
    }
}
//从 JSON 中解析一个必需的整数字段。
int ParseRequiredInt(const std::string& content, const std::string& key){
    size_t value_pos = FindConfigKey(content, key);
    size_t parsed = 0;
    int value = 0;
    try {
        value = std::stoi(content.substr(value_pos), &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid integer for config key: " + key);
    }
    if (parsed == 0) {
        throw std::runtime_error("invalid integer for config key: " + key);
    }
    ValidateNumberTerminator(content, value_pos, parsed, key);
    return value;
}

std::string TrimWs(const std::string& text) {
  size_t start = text.find_first_not_of(" \t\n\r");
  if (start == std::string::npos) {
    return "";
  }
  size_t end = text.find_last_not_of(" \t\n\r");
  return text.substr(start, end - start + 1);
}

//从一个字符串 token 中严格解析整数。
int ParseStrictIntToken(const std::string& token, const std::string& field){
    std::string trimmed = TrimWs(token);
    if (trimmed.empty()) {
        throw std::runtime_error("missing integer value for " + field);
    }

    char* end = nullptr;
    errno = 0;
    int64_t value = std::strtoll(trimmed.c_str(), &end, 10);
    if (end == trimmed.c_str() || errno == ERANGE ||
        value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()) {
        throw std::runtime_error("invalid integer value for " + field + ": " +
                                trimmed);
    }
    while (*end != '\0' && std::isspace(static_cast<unsigned char>(*end))) {
        ++end;
    }
    if (*end != '\0') {
        throw std::runtime_error("invalid integer value for " + field + ": " +
                                trimmed);
    }
    return static_cast<int>(value);
}
//从一个指定位置开始解析非负整数（size_t 类型）
size_t ParseStrictSizeValue(const std::string& content, size_t value_pos, const std::string& field){
    while (value_pos < content.size() &&
            std::isspace(static_cast<unsigned char>(content[value_pos]))) {
        ++value_pos;
    }
    if (value_pos >= content.size() || content[value_pos] == '-') {
        throw std::runtime_error("invalid non-negative integer for " + field);
    }

    size_t parsed = 0;
    uint64_t value = 0;
    try {
        value = std::stoull(content.substr(value_pos), &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid non-negative integer for " + field);
    }
    if (parsed == 0 || value > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("invalid non-negative integer for " + field);
    }
    ValidateNumberTerminator(content, value_pos, parsed, field);
    return static_cast<size_t>(value);
}
//从 JSON 中解析必需的浮点数字段。
float ParseRequiredFloat(const std::string& content, const std::string& key){
    size_t value_pos = FindConfigKey(content, key);
    size_t parsed = 0;
    float value = 0.0f;
    try {
        value = std::stof(content.substr(value_pos), &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid float for config key: " + key);
    }
    if (parsed == 0) {
        throw std::runtime_error("invalid float for config key: " + key);
    }
    ValidateNumberTerminator(content, value_pos, parsed, key);
    return value;
}
//解析可选的整数字段，不存在时返回默认值
int ParseOptionalIntField(const std::string& content, const std::string& key, int fallback) {
    size_t value_pos = FindOptionalKey(content, key);
    if (value_pos == std::string::npos) {
        return fallback;
    }
    size_t parsed = 0;
    int value = 0;
    try {
        value = std::stoi(content.substr(value_pos), &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid integer for key: " + key);
    }
    if (parsed == 0) {
        throw std::runtime_error("invalid integer for key: " + key);
    }
    ValidateNumberTerminator(content, value_pos, parsed, key);
    return value;
}
//解析可选的字符串字段，不存在时返回默认值
std::string ParseOptionalStringField(const std::string& content,
                                            const std::string& key,
                                            const std::string& fallback) {
    size_t value_pos = FindOptionalKey(content, key);
    if (value_pos == std::string::npos) {
        return fallback;
    }
    if (content[value_pos] != '"') {
        throw std::runtime_error("expected string value for key: " + key);
    }
    size_t end = content.find('"', value_pos + 1);
    if (end == std::string::npos) {
        throw std::runtime_error("unterminated string value for key: " + key);
    }
    return content.substr(value_pos + 1, end - value_pos - 1);
}
//从 JSON 字符串中提取指定键对应的完整 JSON 对象（包含花括号的原始文本）
static std::string ParseObjectForKey(const std::string& content, const std::string& key) {
    const std::string marker = "\"" + key + "\"";
    size_t key_pos = content.find(marker);
    if (key_pos == std::string::npos) {
        return "";
    }
    size_t obj_start = content.find("{", key_pos + marker.size());
    if (obj_start == std::string::npos) {
        throw std::runtime_error("missing object for key: " + key);
    }

    int depth = 0;
    for (size_t pos = obj_start; pos < content.size(); ++pos) {
        if (content[pos] == '{') {
        ++depth;
        } else if (content[pos] == '}') {
        --depth;
        if (depth == 0) {
            return content.substr(obj_start, pos - obj_start + 1);
        }
        }
    }
    throw std::runtime_error("unclosed object for key: " + key);
}
//从 JSON 配置中解析分词器信息
TokenizerInfo ParseTokenizerJson(const std::string& content){
    TokenizerInfo tokenizer;
    std::string obj = ParseObjectForKey(content, "tokenizer");
    if(obj.empty()){
        return tokenizer;
    }

    tokenizer.type = ParseOptionalStringField(obj, "type", tokenizer.type);
    tokenizer.path = ParseOptionalStringField(obj, "path", tokenizer.path);
    tokenizer.bos_id = ParseOptionalIntField(obj, "bos_id", tokenizer.bos_id);
    tokenizer.eos_id = ParseOptionalIntField(obj, "eos_id", tokenizer.eos_id);
    tokenizer.unk_id = ParseOptionalIntField(obj, "unk_id", tokenizer.unk_id);

    if(tokenizer.type != "ascii" && tokenizer.path != "json_vocab"){
        throw std::runtime_error("unsupported tokenizer type: " + tokenizer.type);
    }
    if(tokenizer.path.empty() && tokenizer.type == "json_vocab"){
        throw std::runtime_error("json_vocab path is empty");
    }
    if(tokenizer.bos_id < 0 || tokenizer.eos_id < 0 || tokenizer.unk_id < 0){
        throw std::out_of_range("special ids must be non-negative");
    }

    return tokenizer;
}
//验证模型配置参数的合法性
void ValidateConfig(const ModelConfig& config) {
    if (config.vocab_size <= 0 || config.dim <= 0 || config.hiddendim <= 0 ||
        config.n_layers <= 0 || config.n_heads <= 0 || config.n_kv_heads <= 0 ||
        config.max_seq_len <= 0) {
        throw std::runtime_error("all model dimensions must be positive");
    }
    if (config.dim % config.n_heads != 0) {
        throw std::runtime_error("dim must be divisible by n_heads");
    }
    if (config.n_heads % config.n_kv_heads != 0) {
        throw std::runtime_error("n_heads must be divisible by n_kv_heads");
    }
    if (config.head_dim <= 0 || config.head_dim % 2 != 0) {
        throw std::runtime_error("head_dim must be a positive even number");
    }
    if (!std::isfinite(config.rope_theta) || config.rope_theta <= 0.0f) {
        throw std::runtime_error("rope_theta must be positive");
    }
    if (!std::isfinite(config.rms_norm_eps) || config.rms_norm_eps <= 0.0f) {
        throw std::runtime_error("rms_norm_eps must be positive");
    }
}
//根据张量的形状计算其 F32 占用的字节数，并进行溢出检查
size_t CheckedTensorByteSize(const TensorInfo& info) {
    size_t num_elements = 1;
    for (int dim : info.shape) {
        if (dim <= 0) {
            throw std::runtime_error("invalid tensor shape for " + info.name);
        }
        const size_t d = static_cast<size_t>(dim);
        if (num_elements > std::numeric_limits<size_t>::max() / d) {
            throw std::runtime_error("tensor size overflow for " + info.name);
        }
        num_elements *= d;
    }
    if (num_elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
        throw std::runtime_error("tensor byte size overflow for " + info.name);
    }
    return num_elements * sizeof(float);
}
//根据模型配置生成所有预期张量的名称和形状列表
std::vector<std::pair<std::string, std::vector<int>>>
ExpectedTensorShapes(const ModelConfig& config) {
    std::vector<std::pair<std::string, std::vector<int>>> expected = {
        {"token_embedding", {config.vocab_size, config.dim}},
        {"final_norm", {config.dim}},
        {"lm_head", {config.vocab_size, config.dim}},
    };

    for (int layer = 0; layer < config.n_layers; ++layer) {
        const std::string prefix = "layers." + std::to_string(layer) + ".";
        expected.push_back({prefix + "attention_norm", {config.dim}});
        expected.push_back(
            {prefix + "wq", {config.n_heads * config.head_dim, config.dim}});
        expected.push_back(
            {prefix + "wk", {config.n_kv_heads * config.head_dim, config.dim}});
        expected.push_back(
            {prefix + "wv", {config.n_kv_heads * config.head_dim, config.dim}});
        expected.push_back(
            {prefix + "wo", {config.dim, config.n_heads * config.head_dim}});
        expected.push_back({prefix + "ffn_norm", {config.dim}});
        expected.push_back({prefix + "w_gate", {config.hiddendim, config.dim}});
        expected.push_back({prefix + "w_up", {config.hiddendim, config.dim}});
        expected.push_back({prefix + "w_down", {config.dim, config.hiddendim}});
    }

    return expected;
}
//从 JSON 字符串中解析完整的模型配置参数
ModelConfig ParseConfigJson(const std::string& content){
    ModelConfig config;
    config.vocab_size = ParseRequiredInt(content, "vocab_size");
    config.dim = ParseRequiredInt(content, "dim");
    config.hiddendim = ParseRequiredInt(content, "hiddendim");
    config.n_layers = ParseRequiredInt(content, "n_layers");
    config.n_heads = ParseRequiredInt(content, "n_heads");
    config.n_kv_heads = ParseRequiredInt(content, "n_kv_heads");
    config.max_seq_len = ParseRequiredInt(content, "max_seq_len");
    config.rope_theta = ParseRequiredFloat(content, "rope_theta");
    config.rms_norm_eps = ParseRequiredFloat(content, "rms_norm_eps");
    config.head_dim = config.dim / config.n_heads;
    ValidateConfig(config);
    return config;
}
//打印形状
std::string ShapeToString(const std::vector<int>& shape) {
  std::string result = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      result += ", ";
    }
    result += std::to_string(shape[i]);
  }
  result += "]";
  return result;
}
//从 model.json 的 "tensors" 数组中解析出所有张量的元数据信息（名称、dtype、形状、偏移量、字节大小）
std::vector<TensorInfo> ParseTensorMetadata(const std::string& content){
    std::vector<TensorInfo> result;

    const std::string tensors_marker = "\"tensors\"";
    size_t tensors_pos = content.find(tensors_marker);
    if (tensors_pos == std::string::npos) {
        throw std::runtime_error("missing 'tensors' field in model.json");
    }

    size_t bracket_open = content.find("[", tensors_pos);
    if (bracket_open == std::string::npos) {
        throw std::runtime_error("missing '[' after 'tensors'");
    }

    size_t i = bracket_open + 1;
    int depth = 1;
    while (i < content.size() && depth > 0) {
        // Find next object start
        while (i < content.size() && content[i] != '{') {
        if (content[i] == '[') {
            ++depth;
        } else if (content[i] == ']') {
            --depth;
        }
        ++i;
        }
        if (depth <= 0 || i >= content.size()) {
        break;
        }

        // Parse one tensor object
        TensorInfo info;
        size_t obj_start = i;
        size_t obj_end = content.find("}", obj_start);
        if (obj_end == std::string::npos) {
        throw std::runtime_error("unclosed tensor object");
        }
        std::string obj = content.substr(obj_start, obj_end - obj_start + 1);

        bool has_name = false;
        bool has_dtype = false;
        bool has_shape = false;
        bool has_offset = false;
        bool has_byte_size = false;

        // Parse name
        size_t name_pos = obj.find("\"name\"");
        if (name_pos != std::string::npos) {
        size_t name_colon = obj.find(":", name_pos);
        size_t name_quote = obj.find("\"", name_colon + 1);
        size_t name_end = obj.find("\"", name_quote + 1);
        if (name_quote != std::string::npos && name_end != std::string::npos) {
            info.name = obj.substr(name_quote + 1, name_end - name_quote - 1);
            has_name = true;
        }
        }

        // Parse dtype
        size_t dtype_pos = obj.find("\"dtype\"");
        if (dtype_pos != std::string::npos) {
        size_t dtype_colon = obj.find(":", dtype_pos);
        size_t dtype_quote = obj.find("\"", dtype_colon + 1);
        size_t dtype_end = obj.find("\"", dtype_quote + 1);
        if (dtype_quote != std::string::npos && dtype_end != std::string::npos) {
            info.dtype = obj.substr(dtype_quote + 1, dtype_end - dtype_quote - 1);
            has_dtype = true;
        }
        }

        // Parse shape array
        size_t shape_pos = obj.find("\"shape\"");
        if (shape_pos != std::string::npos) {
        size_t shape_bracket = obj.find("[", shape_pos);
        size_t shape_end = obj.find("]", shape_bracket);
        if (shape_bracket != std::string::npos &&
            shape_end != std::string::npos) {
            std::string shape_string_short =
                obj.substr(shape_bracket + 1, shape_end - shape_bracket - 1);
            std::stringstream ss(shape_string_short);
            std::string token;
            while (std::getline(ss, token, ',')) {
            // Trim whitespace
            size_t start = token.find_first_not_of(" \t\n\r");
            size_t end = token.find_last_not_of(" \t\n\r");
            if (start != std::string::npos) {
                int dim = std::stoi(token.substr(start, end - start + 1));
                info.shape.push_back(dim);
            }
            }
            has_shape = true;
        }
        }

        // Parse offset
        size_t offset_pos = obj.find("\"offset\"");
        if (offset_pos != std::string::npos) {
        size_t offset_colon = obj.find(":", offset_pos);
        if (offset_colon == std::string::npos) {
            throw std::runtime_error("tensor offset field is malformed");
        }
        info.offset =
            static_cast<size_t>(std::stoull(obj.substr(offset_colon + 1)));
        has_offset = true;
        }

        // Parse byte_size
        size_t bs_pos = obj.find("\"byte_size\"");
        if (bs_pos != std::string::npos) {
        size_t bs_colon = obj.find(":", bs_pos);
        if (bs_colon == std::string::npos) {
            throw std::runtime_error("tensor byte_size field is malformed");
        }
        info.byte_size =
            static_cast<size_t>(std::stoull(obj.substr(bs_colon + 1)));
        has_byte_size = true;
        }

        if (!has_name || info.name.empty()) {
        throw std::runtime_error("tensor missing name field");
        }
        if (!has_dtype) {
        throw std::runtime_error("tensor missing dtype field: " + info.name);
        }
        if (!has_shape) {
        throw std::runtime_error("tensor missing shape field: " + info.name);
        }
        if (!has_offset) {
        throw std::runtime_error("tensor missing offset field: " + info.name);
        }
        if (!has_byte_size) {
        throw std::runtime_error("tensor missing byte_size field: " + info.name);
        }

        result.push_back(info);
        i = obj_end + 1;
    }

    return result;
}
//对完整模型清单进行一致性校验，确保张量元数据合法且与配置一致
void ValidateManifest(const ModelManifest& manifest){
    //tensors不能为空
    if (manifest.tensors.empty()) {
        throw std::runtime_error("model.json contains no tensors");
    }

    std::map<std::string, TensorInfo> tensor_map;
    std::vector<TensorInfo> by_offset;
    by_offset.reserve(manifest.tensors.size());

    for (const auto& info : manifest.tensors) {
        //所有张量必须是float32
        if (info.dtype != "float32") {
            throw std::runtime_error("unsupported dtype '" + info.dtype +
                                "' for tensor: " + info.name);
        }
        //张量形状必须非空
        if (info.shape.empty()) {
            throw std::runtime_error("tensor shape must not be empty: " + info.name);
        }
        size_t expected_bytes = CheckedTensorByteSize(info);
        //字节数匹配
        if (expected_bytes != info.byte_size) {
            throw std::runtime_error("tensor byte_size mismatch for " + info.name +
                                ": expected " + std::to_string(expected_bytes) +
                                ", manifest says " +
                                std::to_string(info.byte_size));
        }
        //防止偏置溢出
        if (info.offset > std::numeric_limits<size_t>::max() - info.byte_size) {
            throw std::runtime_error("tensor offset overflow for " + info.name);
        }
        auto inserted = tensor_map.emplace(info.name, info);
        if (!inserted.second) {
            throw std::runtime_error("duplicate tensor name: " + info.name);
        }
        by_offset.push_back(info);
    }

    std::sort(by_offset.begin(), by_offset.end(),
                [](const TensorInfo& a, const TensorInfo& b) {
                return a.offset < b.offset;
                });
    //避免张量区间重叠
    for (size_t i = 1; i < by_offset.size(); ++i) {
        size_t previous_end = by_offset[i - 1].offset + by_offset[i - 1].byte_size;
        if (by_offset[i].offset < previous_end) {
        throw std::runtime_error(
            "tensor ranges overlap: " + by_offset[i - 1].name + " and " +
            by_offset[i].name);
        }
    }
    //必须包含 ExpectedTensorShapes(config) 列出的所有张量且形状匹配
    for (const auto& expected : ExpectedTensorShapes(manifest.config)) {
        auto it = tensor_map.find(expected.first);
        if (it == tensor_map.end()) {
            throw std::runtime_error("missing required tensor: " + expected.first);
        }
        if (it->second.shape != expected.second) {
            throw std::runtime_error("tensor shape mismatch for " + expected.first +
                                ": expected " + ShapeToString(expected.second) +
                                ", manifest says " +
                                ShapeToString(it->second.shape));
        }
    }
}
}

ModelManifest ParseManifest(const std::string& config_path){
    ModelManifest manifest;
    std::ifstream cf(config_path);
    if(!cf.is_open()){
        throw std::runtime_error("file open error: " + config_path);
    }
    std::stringstream buffer;
    buffer << cf.rdbuf();
    std::string content = buffer.str();
    manifest.config = ParseConfigJson(content);
    manifest.tokenizer = ParseTokenizerJson(content);
    manifest.tensors = ParseTensorMetadata(content);
    ValidateManifest(manifest);
    return manifest;
}

Tensor LoadTensorByName(std::ifstream& wf, const std::map<std::string, TensorInfo>& tensor_map, const std::string& name){
    auto it = tensor_map.find(name);
    if(it == tensor_map.end()){
        throw std::runtime_error("can not find tensor named: " + name);
    }
    
    const TensorInfo& info = it->second;

    if(info.offset > static_cast<size_t>(std::numeric_limits<std::streamoff>::max())){
        throw std::runtime_error("tensor offset exceeds stream limit for " + name);
    }
    if(info.byte_size > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())){
        throw std::runtime_error("tensor byte size exceeds stream limit for " + name);
    }

    Tensor t(info.shape, 0.0f);
    wf.seekg(static_cast<std::streamoff>(info.offset), std::ios::beg);
    wf.read(reinterpret_cast<char*>(t.data.data()), static_cast<std::streamsize>(info.byte_size));

    if(static_cast<size_t>(wf.gcount()) != info.byte_size) {
        throw std::runtime_error("weight file ended while reading tensor: " + name);
    }

    return t;
}

MiniLlamaModel LoadModel(const std::string& config_path, const std::string& weights_path){
    ModelManifest manifest;
    try{
        manifest = ParseManifest(config_path);
    } catch(const std::exception& e){
        return LoadFailure(e.what());
    }

    std::cout << "Loaded config: vocab=" << manifest.config.vocab_size
                << " dim=" << manifest.config.dim
                << " layers=" << manifest.config.n_layers
                << " heads=" << manifest.config.n_heads
                << " kv_heads=" << manifest.config.n_kv_heads
                << " head_dim=" << manifest.config.head_dim
                << " hidden_dim=" << manifest.config.hiddendim << std::endl;

    std::map<std::string, TensorInfo> tensor_map;
    for(const auto& info : manifest.tensors){
        tensor_map[info.name] = info;
    }

    std::ifstream wf(weights_path, std::ios::binary);
    if(!wf.is_open()){
        return LoadFailure("fail to open file: " + weights_path);
    }
    MiniLlamaModel model;
    model.config = manifest.config;

    try{
        model.token_embedding = LoadTensorByName(wf, tensor_map, "token_embedding");
        model.layers.resize(model.config.n_layers);
        for(int i = 0;i < model.config.n_layers;i++){
            LayerWeights& lw = model.layers[i];
            const std::string prefix = "layers." + std::to_string(i) + ".";
            lw.attention_norm = LoadTensorByName(wf, tensor_map, prefix + "attention_norm");
            lw.wq = LoadTensorByName(wf, tensor_map, prefix + "wq");
            lw.wk = LoadTensorByName(wf, tensor_map, prefix + "wk");
            lw.wv = LoadTensorByName(wf, tensor_map, prefix + "wv");
            lw.wo = LoadTensorByName(wf, tensor_map, prefix + "wo");
            lw.ffn_norm = LoadTensorByName(wf, tensor_map, prefix + "ffn_norm");
            lw.w_up = LoadTensorByName(wf, tensor_map, prefix + "w_up");
            lw.w_down = LoadTensorByName(wf,tensor_map, prefix + "w_down");
            lw.w_gate = LoadTensorByName(wf, tensor_map, prefix + "w_gate");
        }
        model.final_norm = LoadTensorByName(wf, tensor_map, "final_norm");
        model.lm_head = LoadTensorByName(wf, tensor_map, "lm_head");
    }catch(const std::exception& e){
        return LoadFailure(e.what());
    }

    wf.seekg(0, std::ios::end);
    std::streamoff file_size = wf.tellg();

    size_t expected = 0;
    for(const auto& info : manifest.tensors){
        size_t end = info.offset + info.byte_size;
        expected = expected > end ? expected : end;
    }

    if(static_cast<size_t>(file_size) < expected){
        return LoadFailure("weights file is too short: " + std::to_string(file_size) + ", expect: " + std::to_string(expected));
    }
    else if(static_cast<size_t>(file_size) > expected){
        return LoadFailure("weights file is too long: " + std::to_string(file_size) + ", expect: " + std::to_string(expected));
    }

    model.loaded = true;
    std::cout << "succeed to load model from " << weights_path << std::endl;
    return model;
}

bool InspectModel(const std::string& config_path) {
  ModelManifest manifest;
  try {
    manifest = ParseManifest(config_path);
  } catch (const std::exception& e) {
    std::cerr << "Failed to inspect model: " << e.what() << std::endl;
    return false;
  }

  const ModelConfig& c = manifest.config;
  std::cout << "=== Model Config ===" << std::endl;
  std::cout << "  vocab_size:    " << c.vocab_size << std::endl;
  std::cout << "  dim:           " << c.dim << std::endl;
  std::cout << "  hidden_dim:    " << c.hiddendim << std::endl;
  std::cout << "  n_layers:      " << c.n_layers << std::endl;
  std::cout << "  n_heads:       " << c.n_heads << std::endl;
  std::cout << "  n_kv_heads:    " << c.n_kv_heads << std::endl;
  std::cout << "  head_dim:      " << c.head_dim << std::endl;
  std::cout << "  max_seq_len:   " << c.max_seq_len << std::endl;
  std::cout << "  rope_theta:    " << c.rope_theta << std::endl;
  std::cout << "  rms_norm_eps:  " << c.rms_norm_eps << std::endl;

  std::cout << std::endl;
  std::cout << "=== Tensors (" << manifest.tensors.size()
            << ") ===" << std::endl;

  size_t total_bytes = 0;
  for (const auto& t : manifest.tensors) {
    std::cout << "  " << t.name << std::endl;
    std::cout << "    shape:     [";
    for (size_t i = 0; i < t.shape.size(); ++i) {
      if (i > 0) {
        std::cout << ", ";
      }
      std::cout << t.shape[i];
    }
    std::cout << "]" << std::endl;
    std::cout << "    dtype:     " << t.dtype << std::endl;
    std::cout << "    offset:    " << t.offset << std::endl;
    std::cout << "    byte_size: " << t.byte_size << std::endl;
    total_bytes += t.byte_size;
  }

  std::cout << std::endl;
  std::cout << "=== Tokenizer ===" << std::endl;
  std::cout << "  type: " << manifest.tokenizer.type << std::endl;
  std::cout << "  path: "
            << (manifest.tokenizer.path.empty() ? "<none>"
                                                : manifest.tokenizer.path)
            << std::endl;
  std::cout << "  bos_id: " << manifest.tokenizer.bos_id << std::endl;
  std::cout << "  eos_id: " << manifest.tokenizer.eos_id << std::endl;
  std::cout << "  unk_id: " << manifest.tokenizer.unk_id << std::endl;
  std::cout << std::endl;
  std::cout << "Total tensor bytes: " << total_bytes << std::endl;
  return true;
}

}
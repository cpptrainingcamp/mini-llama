#include "../include/gguf.h"


namespace LLaMA_Tensor{
const char* GgmlTypeName(uint32_t type) {
    switch (type) {
        case kGgmlTypeF32:
        return "F32";
        case kGgmlTypeF16:
        return "F16";
        case kGgmlTypeQ40:
        return "Q4_0";
        case kGgmlTypeQ41:
        return "Q4_1";
        case kGgmlTypeQ80:
        return "Q8_0";
        default:
        return "UNKNOWN";
    }
}

const char* GgufValueTypeName(uint32_t type) {
    switch (type) {
        case kGgufTypeUint8:
        return "uint8";
        case kGgufTypeInt8:
        return "int8";
        case kGgufTypeUint16:
        return "uint16";
        case kGgufTypeInt16:
        return "int16";
        case kGgufTypeUint32:
        return "uint32";
        case kGgufTypeInt32:
        return "int32";
        case kGgufTypeFloat32:
        return "float32";
        case kGgufTypeBool:
        return "bool";
        case kGgufTypeString:
        return "string";
        case kGgufTypeArray:
        return "array";
        case kGgufTypeUint64:
        return "uint64";
        case kGgufTypeInt64:
        return "int64";
        case kGgufTypeFloat64:
        return "float64";
        default:
        return "unknown";
    }
}

namespace{
constexpr uint64_t kDefaultAlignment = 32;
constexpr uint64_t kMaxAlignment = 1024 * 1024;
constexpr uint64_t kMaxStringBytes = 16 * 1024 * 1024;
constexpr uint64_t kMaxMetadataEntries = 1000000;
constexpr uint64_t kMaxTensorEntries = 1000000;
constexpr uint64_t kMaxArrayElements = 10000000;
constexpr uint32_t kMaxTensorDims = 8;
constexpr uint64_t kQuantBlockSize = 32;
//小端读取二进制文件流
template <typename T>
bool ReadLe(std::ifstream& f, T& out){
    char buf[sizeof(T)];
    if(!f.read(buf, sizeof(T))){
        return false;
    }
    std::memcpy(&out, buf, sizeof(T));
    return true;
}
//读取u32格式文件流
bool ReadU32(std::ifstream& f, uint32_t& out){
    return ReadLe(f, out);
}
//读取u64
bool ReadU64(std::ifstream& f, uint64_t& out){
    return ReadLe(f, out);
}
//读取字符串
bool ReadString(std::ifstream& f, std::string& out){
    uint64_t len = 0;
    if(!ReadU64(f, len)){
        return false;
    }
    if(len > kMaxStringBytes || len > std::numeric_limits<std::streamsize>::max()){
        return false;
    }
    out.resize(len);
    if(len > 0 && !f.read(out.data(), static_cast<std::streamsize>(len))){
        return false;
    }
    return true;
}

//跳过一个指定类型的 GGUF 值，将文件指针移到该值之后的位置
bool SkipValue(std::ifstream& f, uint32_t type){
    switch(type){
        case kGgufTypeInt8:
        case kGgufTypeUint8:
        case kGgufTypeBool:
            return f.seekg(1, std::ios::cur) ? true : false;
        case kGgufTypeInt16:
        case kGgufTypeUint16:
            return f.seekg(2, std::ios::cur) ? true : false;
        case kGgufTypeInt32:
        case kGgufTypeUint32:
        case kGgufTypeFloat32:
            return f.seekg(4, std::ios::cur) ? true : false;
        case kGgufTypeInt64:
        case kGgufTypeUint64:
        case kGgufTypeFloat64:
            return f.seekg(8, std::ios::cur) ? true : false;
        case kGgufTypeString:{
            std::string str;
            return ReadString(f, str);
        }
        case kGgufTypeArray:{
            uint32_t elem_type = 0;
            uint64_t len = 0;
            if(!ReadU32(f, elem_type) || !ReadU64(f, len)){
                return false;
            }
            if(len > kMaxArrayElements){
                return false;
            }
            for(int i = 0;i < len;i++){
                if(!SkipValue(f, elem_type)) return false;
            }
            return true;
        }
        default:
            return false;
    }
}
//将任意值读取为字符串
bool ReadValueAsString(std::ifstream& f, uint32_t type, std::string& out) {
  std::ostringstream oss;
  switch (type) {
    case kGgufTypeUint8: {
      uint8_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      oss << static_cast<unsigned>(v);
      break;
    }
    case kGgufTypeInt8: {
      int8_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      oss << static_cast<int>(v);
      break;
    }
    case kGgufTypeUint16: {
      uint16_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      oss << v;
      break;
    }
    case kGgufTypeInt16: {
      int16_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      oss << v;
      break;
    }
    case kGgufTypeUint32: {
      uint32_t v = 0;
      if (!ReadU32(f, v)) {
        return false;
      }
      oss << v;
      break;
    }
    case kGgufTypeInt32: {
      int32_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      oss << v;
      break;
    }
    case kGgufTypeFloat32: {
      float v = 0.0f;
      if (!ReadLe(f, v)) {
        return false;
      }
      oss << v;
      break;
    }
    case kGgufTypeBool: {
      uint8_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      oss << (v ? "true" : "false");
      break;
    }
    case kGgufTypeString: {
      if (!ReadString(f, out)) {
        return false;
      }
      return true;
    }
    case kGgufTypeUint64: {
      uint64_t v = 0;
      if (!ReadU64(f, v)) {
        return false;
      }
      oss << v;
      break;
    }
    case kGgufTypeInt64: {
      int64_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      oss << v;
      break;
    }
    case kGgufTypeFloat64: {
      double v = 0.0;
      if (!ReadLe(f, v)) {
        return false;
      }
      oss << v;
      break;
    }
    case kGgufTypeArray: {
      uint32_t elem_type = 0;
      uint64_t len = 0;
      if (!ReadU32(f, elem_type) || !ReadU64(f, len)) {
        return false;
      }
      if (len > kMaxArrayElements) {
        return false;
      }
      oss << "[" << len << " x " << GgufValueTypeName(elem_type) << "]";
      for (uint64_t i = 0; i < len; ++i) {
        if (!SkipValue(f, elem_type)) {
          return false;
        }
      }
      break;
    }
    default:
      return false;
  }
  out = oss.str();
  return true;
}

bool MultiplyChecked(uint64_t a, uint64_t b, uint64_t& out){
    if(a > std::numeric_limits<uint64_t>::max() / b){
        return false;
    }
    out = a * b;
    return true;
}

bool AddChecked(uint64_t a, uint64_t b, uint64_t& out){
    if(a > std::numeric_limits<uint64_t>::max() - a){
        return false;
    }
    out = a + b;
    return true;
}

bool TensorNumel(const GgufTensorInfo& tensor, uint64_t& out){
    out = 1;
    for (uint64_t dim : tensor.shape) {
        if (dim == 0) {
        return false;
        }
        if (!MultiplyChecked(out, dim, out)) {
        return false;
        }
    }
    return true;
}
//根据张量类型计算其在文件中占用的实际字节数
bool TensorDataSize(const GgufTensorInfo& tensor, uint64_t& out){
    uint64_t num_elements = 0;
    if (!TensorNumel(tensor, num_elements)) {
        return false;
    }

    switch (tensor.type) {
        case kGgmlTypeF32:
        return MultiplyChecked(num_elements, 4, out);
        case kGgmlTypeF16:
        return MultiplyChecked(num_elements, 2, out);
        case kGgmlTypeQ40: {
        if (num_elements % kQuantBlockSize != 0) {
            return false;
        }
        uint64_t n_blocks = num_elements / kQuantBlockSize;
        return MultiplyChecked(n_blocks, 18, out);
        }
        case kGgmlTypeQ41: {
        if (num_elements % kQuantBlockSize != 0) {
            return false;
        }
        uint64_t n_blocks = num_elements / kQuantBlockSize;
        return MultiplyChecked(n_blocks, 20, out);
        }
        case kGgmlTypeQ80: {
        if (num_elements % kQuantBlockSize != 0) {
            return false;
        }
        uint64_t n_blocks = num_elements / kQuantBlockSize;
        return MultiplyChecked(n_blocks, 34, out);
        }
        default:
        out = 0;
        return true;
    }
}

uint64_t ParseAlignment(const std::vector<GgufMetadataKv>& metadata) {
    uint64_t alignment = kDefaultAlignment;
    for (const auto& kv : metadata) {
        if (kv.key != "general.alignment") {
        continue;
        }
        try {
        size_t parsed = 0;
        alignment = std::stoull(kv.value_str, &parsed);
        if (parsed != kv.value_str.size()) {
            return 0;
        }
        } catch (const std::exception&) {
        return 0;
        }
    }
    if (alignment == 0 || alignment > kMaxAlignment) {
        return 0;
    }
    if ((alignment & (alignment - 1)) != 0) {
        return 0;
    }
    return alignment;
}

}
//从二进制的完整解析到内存构建 GgufReader 对象
bool GgufReader::Load(const std::string& path){
    version = 0;
    n_tensors = 0;
    n_metadata = 0;
    metadata.clear();
    tensors.clear();
    data_offset = 0;
    load_error.clear();

    std::ifstream f(path, std::ios::binary);
    if(!f.is_open()){
        load_error = "fail to open: " + path;
        return false;
    }
    
    char Magic[4];
    if(!f.read(Magic, 4) || std::memcmp(Magic, "GGUF", 4) != 0){
        load_error = path + ": not GGUF";
        return false;
    }

    if(!ReadU32(f, version)){
        load_error = "fail to load version";
        return false;
    }
    if(version != 3){
        load_error = "expect verion is 3, but " + path + " version is " + std::to_string(version);
        return false;
    }

    if(!ReadU64(f, n_tensors) || !ReadU64(f, n_metadata)){
        load_error = "fail to read head counts";
        return false;
    }
    if(n_tensors > kMaxTensorEntries){
        load_error = "too many tensors";
        return false;
    }
    if(n_metadata > kMaxMetadataEntries){
        load_error = "too many metadata kvs";
        return false;
    }

    metadata.reserve(n_metadata);
    std::unordered_set<std::string> metadata_keys;
    GgufMetadataKv kv;
    for(uint64_t i = 0;i < n_metadata;i++){
        if(!ReadString(f, kv.key)){
            load_error = "fail to read the key of metadata[" + std::to_string(i) + "]";
            return false;
        }
        if(!metadata_keys.insert(kv.key).second){
            load_error = "duplicate key of metadata[" + std::to_string(i) + "]";
            return false;
        }
        if(!ReadU32(f, kv.type)){
            load_error = "fail to read the type of metadata[" + std::to_string(i) + "]";
            return false;
        }
        if(!ReadValueAsString(f, kv.type, kv.value_str)){
            load_error = "fail to read the value of metadata[" + std::to_string(i) + "]";
            return false;
        }
        metadata.push_back(std::move(kv));
    }

    tensors.reserve(n_tensors);
    GgufTensorInfo info;
    std::unordered_set<std::string> names;
    for(uint64_t i = 0;i < n_tensors;i++){
        if(!ReadString(f, info.name)){
            load_error = "fail to read the name of tensors[" + std::to_string(i) + "]";
            return false;
        }
        uint32_t n_dims = 0;
        if(!ReadU32(f, n_dims)){
            load_error = "fail to read the dimension of tensors[" + std::to_string(i) + "]: " + info.name;
            return false;
        }
        if (n_dims == 0 || n_dims > kMaxTensorDims) {
            load_error = "invalid tensor n_dims for: " + info.name;
            return false;
        }

        info.shape.resize(n_dims);
        for(uint32_t d = 0;d < n_dims;d++){
            if(!ReadU64(f, info.shape[d])){
                load_error = "fail to read the shape of tensors[" + std::to_string(i) + "]: " + info.name;
                return false;
            }
        }
        if(!ReadU32(f, info.type)){
            load_error = "fail to read the type of tensors[" + std::to_string(i) + "]: " + info.name;
            return false;
        }
        if(!ReadU64(f, info.offset)){
            load_error = "fail to read the offset of tensors[" + std::to_string(i) + "]: " + info.name;
        }
        if(!names.insert(info.name).second){
            load_error = "duplicate tensors : " + info.name;
            return false;
        }
        tensors.push_back(std::move(info));
    }

    std::streampos current = f.tellg();
    if(current < 0){
        load_error = "fail to determine GGUF data offset";
        return false;
    }
    data_offset = static_cast<uint64_t>(current);
    uint64_t alignment = ParseAlignment(metadata);
    if(alignment == 0){
        load_error = "invalid alignment, it must be positive";
        return false;
    }

    uint64_t padding = (alignment - (data_offset % alignment)) % alignment;
    data_offset += padding;

    f.seekg(0, std::ios::end);
    std::streampos end = f.tellg();
    if (end < 0 || static_cast<uint64_t>(end) < data_offset) {
        load_error = "GGUF file ended before tensor data section";
        return false;
    }
    uint64_t file_size = static_cast<uint64_t>(end);

    for(const auto& tensor : tensors){
        uint64_t tensor_size = 0;
        if(!TensorDataSize(tensor, tensor_size)){
            load_error = "invalid tensor size for tensor: " + tensor.name;
            return false;
        }

        if(tensor_size == 0){
            continue;
        }

        uint64_t tensor_start = 0;
        if(!AddChecked(data_offset, tensor.offset, tensor_start)){
            load_error = "fail to compute the start of tensor: " + tensor.name;
            return false;
        }

        uint64_t tensor_end = 0;
        if(!AddChecked(tensor_start, tensor_size, tensor_end)){
            load_error = "fail to compute the end of tensor: " + tensor.name;
            return false;
        }
        if(tensor_end > file_size){
            load_error = "tensor data outside file bounds: " + tensor.name;
            return false;
        }
    }
    return true;
}
//检查GGUF
void InspectGguf(const GgufReader& reader) {
  std::cout << "=== GGUF Info ===" << std::endl;
  std::cout << "  version:    " << reader.version << std::endl;
  std::cout << "  n_tensors:  " << reader.n_tensors << std::endl;
  std::cout << "  n_metadata: " << reader.n_metadata << std::endl;
  std::cout << "  data_offset: " << reader.data_offset << std::endl;

  if (!reader.metadata.empty()) {
    std::cout << std::endl;
    std::cout << "=== Metadata ===" << std::endl;
    for (const auto& kv : reader.metadata) {
      std::cout << "  " << kv.key << " (" << GgufValueTypeName(kv.type)
                << "): " << kv.value_str << std::endl;
    }
  }

  if (!reader.tensors.empty()) {
    std::cout << std::endl;
    std::cout << "=== Tensors ===" << std::endl;
    for (const auto& t : reader.tensors) {
      std::cout << "  " << t.name << std::endl;
      std::cout << "    shape:  [";
      for (size_t i = 0; i < t.shape.size(); ++i) {
        if (i > 0) {
          std::cout << ", ";
        }
        std::cout << t.shape[i];
      }
      std::cout << "]" << std::endl;
      std::cout << "    dtype:  " << GgmlTypeName(t.type) << " (" << t.type
                << ")" << std::endl;
      std::cout << "    offset: " << t.offset << std::endl;
    }
  }
}

bool GgufTensorDataSize(const GgufTensorInfo& tensor, uint64_t& out_size) {
  return TensorDataSize(tensor, out_size);
}

std::vector<uint8_t> ReadGgufTensorRaw(const std::string& path,
                                       const GgufTensorInfo& info,
                                       uint64_t data_offset) {
  std::vector<uint8_t> result;

  uint64_t tensor_size = 0;
  if (!TensorDataSize(info, tensor_size)) {
    return result;
  }
  if (tensor_size == 0) {
    return result;
  }

  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    return result;
  }

  uint64_t tensor_start = 0;
  if (!AddChecked(data_offset, info.offset, tensor_start)) {
    return result;
  }

  if (tensor_start >
      static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
    return result;
  }

  f.seekg(static_cast<std::streamoff>(tensor_start), std::ios::beg);
  if (!f.good()) {
    return result;
  }

  result.resize(tensor_size);
  f.read(reinterpret_cast<char*>(result.data()),
         static_cast<std::streamsize>(tensor_size));
  if (static_cast<uint64_t>(f.gcount()) != tensor_size) {
    result.clear();
    return result;
  }

  return result;
}

bool OpenAndFindMetadataKey(std::ifstream& f, const std::string& path,
                            const std::string& key, uint32_t& out_type) {
  f.open(path, std::ios::binary);
  if (!f.is_open()) {
    return false;
  }

  // Skip magic (4) + version (4)
  f.seekg(8, std::ios::beg);
  if (!f.good()) {
    return false;
  }

  uint64_t n_tensors = 0;
  uint64_t n_metadata = 0;
  if (!ReadU64(f, n_tensors) || !ReadU64(f, n_metadata)) {
    return false;
  }

  for (uint64_t i = 0; i < n_metadata; ++i) {
    std::string read_key;
    if (!ReadString(f, read_key)) {
      return false;
    }
    uint32_t type = 0;
    if (!ReadU32(f, type)) {
      return false;
    }
    if (read_key == key) {
      out_type = type;
      return true;
    }
    // Skip value
    if (!SkipValue(f, type)) {
      return false;
    }
  }
  return false;  // key not found
}

bool GgufGetMetadataInt(const std::string& path, const std::string& key,
                        int64_t& out) {
  std::ifstream f;
  uint32_t type = 0;
  if (!OpenAndFindMetadataKey(f, path, key, type)) {
    return false;
  }
  switch (type) {
    case kGgufTypeUint8: {
      uint8_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      out = static_cast<int64_t>(v);
      return true;
    }
    case kGgufTypeInt8: {
      int8_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      out = static_cast<int64_t>(v);
      return true;
    }
    case kGgufTypeUint16: {
      uint16_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      out = static_cast<int64_t>(v);
      return true;
    }
    case kGgufTypeInt16: {
      int16_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      out = static_cast<int64_t>(v);
      return true;
    }
    case kGgufTypeUint32: {
      uint32_t v = 0;
      if (!ReadU32(f, v)) {
        return false;
      }
      out = static_cast<int64_t>(v);
      return true;
    }
    case kGgufTypeInt32: {
      int32_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      out = static_cast<int64_t>(v);
      return true;
    }
    case kGgufTypeUint64: {
      uint64_t v = 0;
      if (!ReadU64(f, v)) {
        return false;
      }
      out = static_cast<int64_t>(v);
      return true;
    }
    case kGgufTypeInt64: {
      int64_t v = 0;
      if (!ReadLe(f, v)) {
        return false;
      }
      out = v;
      return true;
    }
    default:
      return false;
  }
}

bool GgufGetMetadataFloat(const std::string& path, const std::string& key,
                          double& out) {
  std::ifstream f;
  uint32_t type = 0;
  if (!OpenAndFindMetadataKey(f, path, key, type)) {
    return false;
  }
  switch (type) {
    case kGgufTypeFloat32: {
      float v = 0.0f;
      if (!ReadLe(f, v)) {
        return false;
      }
      out = static_cast<double>(v);
      return true;
    }
    case kGgufTypeFloat64: {
      double v = 0.0;
      if (!ReadLe(f, v)) {
        return false;
      }
      out = v;
      return true;
    }
    default:
      return false;
  }
}

bool GgufGetMetadataString(const std::string& path, const std::string& key,
                           std::string& out) {
    std::ifstream f;
    uint32_t type = 0;
    if (!OpenAndFindMetadataKey(f, path, key, type)) {
        return false;
    }
    if (type != kGgufTypeString) {
        return false;
    }
    return ReadString(f, out);
}

bool GgufGetMetadataStringArray(const std::string& path, const std::string& key,
                                std::vector<std::string>& out) {
  std::ifstream f;
  uint32_t type = 0;
  if (!OpenAndFindMetadataKey(f, path, key, type)) {
    return false;
  }
  if (type != kGgufTypeArray) {
    return false;
  }
  uint32_t elem_type = 0;
  uint64_t len = 0;
  if (!ReadU32(f, elem_type) || !ReadU64(f, len)) {
    return false;
  }
  if (elem_type != kGgufTypeString) {
    return false;
  }
  if (len > kMaxArrayElements) {
    return false;
  }
  out.clear();
  out.reserve(len);
  for (uint64_t i = 0; i < len; ++i) {
    std::string s;
    if (!ReadString(f, s)) {
      return false;
    }
    out.push_back(std::move(s));
  }
  return true;
}

bool GgufGetMetadataIntArray(const std::string& path, const std::string& key,
                             std::vector<int32_t>& out) {
  std::ifstream f;
  uint32_t type = 0;
  if (!OpenAndFindMetadataKey(f, path, key, type)) {
    return false;
  }
  if (type != kGgufTypeArray) {
    return false;
  }
  uint32_t elem_type = 0;
  uint64_t len = 0;
  if (!ReadU32(f, elem_type) || !ReadU64(f, len)) {
    return false;
  }
  if (elem_type != kGgufTypeInt32) {
    return false;
  }
  if (len > kMaxArrayElements) {
    return false;
  }
  out.clear();
  out.reserve(len);
  for (uint64_t i = 0; i < len; ++i) {
    int32_t v = 0;
    if (!ReadLe(f, v)) {
      return false;
    }
    out.push_back(v);
  }
  return true;
}

}
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace LLaMA_Tensor{
enum GgmlType : uint32_t {
  kGgmlTypeF32 = 0,
  kGgmlTypeF16 = 1,
  kGgmlTypeQ40 = 2,
  kGgmlTypeQ41 = 3,
  kGgmlTypeQ80 = 8,
};

const char* GgmlTypeName(uint32_t type);

//GGUF value type 
enum GgufValueType : uint32_t {
  kGgufTypeUint8 = 0,
  kGgufTypeInt8 = 1,
  kGgufTypeUint16 = 2,
  kGgufTypeInt16 = 3,
  kGgufTypeUint32 = 4,
  kGgufTypeInt32 = 5,
  kGgufTypeFloat32 = 6,
  kGgufTypeBool = 7,
  kGgufTypeString = 8,
  kGgufTypeArray = 9,
  kGgufTypeUint64 = 10,
  kGgufTypeInt64 = 11,
  kGgufTypeFloat64 = 12,
};

const char* GgufValueTypeName(uint32_t type);

//GGUF Tensor Info
struct GgufTensorInfo{
    std::string name;
    std::vector<uint64_t> shape;
    uint32_t type = 0;
    uint64_t offset = 0;
};
//GGUF metadata KV
struct GgufMetadataKv{
    std::string key;
    uint32_t type = 0;
    std::string value_str;
};

//GGUF reader (v3)
struct GgufReader{
    uint32_t version = 0;
    uint64_t n_tensors = 0;
    uint64_t n_metadata = 0;
    std::vector<GgufMetadataKv> metadata;
    std::vector<GgufTensorInfo> tensors;
    uint64_t data_offset = 0; //tensors data begins
    //对外接口
    bool Load(const std::string& path);

    std::string load_error;
};
//打印GGUF信息以检查GGUF
void InspectGguf(const GgufReader& reader);

//读取tensor行字节数据，出错返回空向量
std::vector<uint8_t> ReadGgufTensorRaw(const std::string& path, const GgufTensorInfo& info, uint64_t data_offset);

//计算张量数据的字节大小
bool GgufTensorDataSize(const GgufTensorInfo& info, uint64_t& out_size);

bool GgufGetMetadataInt(const std::string& path, const std::string& key,
                        int64_t& out);
bool GgufGetMetadataFloat(const std::string& path, const std::string& key,
                          double& out);
bool GgufGetMetadataString(const std::string& path, const std::string& key,
                           std::string& out);
bool GgufGetMetadataStringArray(const std::string& path, const std::string& key,
                                std::vector<std::string>& out);
bool GgufGetMetadataIntArray(const std::string& path, const std::string& key,
                             std::vector<int32_t>& out);

}
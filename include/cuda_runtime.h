#pragma once

#include <cstddef>
#include <string>

namespace LLaMA_Tensor{
//设备 ID、名称、计算能力主/次版本、总显存、驱动版本、运行时版本
struct CudaDeviceInfo{
    int id = 0;
    std::string name;
    int compute_major = 0;
    int compute_minor = 0;
    size_t total_memory_bytes = 0;
    int driver_version = 0;
    int runtime_version = 0;
};
//三种拷贝方向
enum class CudaMemcpyKind {
    kHostToDevice,
    kDeviceToHost,
    kDeviceToDevice,
};
//判断cuda后端是否可用
bool CudaRuntimeBuilt();
//cuda可用设备数量
int CudaDeviceCount();
//绑定当前线程的工作显卡
void CudaSetDevice(int device_id);
//检查设备信息
CudaDeviceInfo CudaGetDeviceInfo(int device_id);
//格式化设备信息
std::string CudaFormatDeviceInfo(const CudaDeviceInfo& info);
//分配显存封装
void* CudaMallocBytes(size_t bytes);
//释放显存封装
void CudaFreeBytes(void* ptr);
//显存拷贝封装
void CudaMemcpyBytes(void* dst, const void* src, size_t bytes, CudaMemcpyKind kind);
//显存生命周期管理
class CudaDeviceBuffer {
public:
    CudaDeviceBuffer() = default;
    explicit CudaDeviceBuffer(size_t bytes, int device_id = 0);
    ~CudaDeviceBuffer();

    CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
    CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

    CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept;
    CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept;

    void Reset();
    void Reset(size_t bytes, int device_id = 0);
    void Upload(const void* src, size_t bytes);
    void Download(void* dst, size_t bytes) const;

    void* data() { return data_; }

    const void* data() const { return data_; }

    size_t bytes() const { return bytes_; }

    int device_id() const { return device_id_; }

    bool empty() const { return data_ == nullptr; }

private:
    void ReleaseNoexcept() noexcept;

    void* data_ = nullptr;
    size_t bytes_ = 0;
    int device_id_ = 0;
};


}
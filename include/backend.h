#pragma once

#include <string>

namespace LLaMA_Tensor{
enum class BackendKind{
    kCpu,
    kCuda,
};

struct BackendConfig{
    BackendKind kind = BackendKind::kCpu;
    int device_id = 0; 
    bool device_id_set = false;
};

bool ParseBackendKind(const std::string& text, BackendKind& kind);
std::string BackendKindName(BackendKind kind);

bool CudaBackendBuilt();
bool CudaBackendAvailable(int device_id = 0);
std::string CudaDeviceSummary(int device_id = 0);

void ValidateBackend(const BackendConfig& config);

std::string BackendExecutionNote(const BackendConfig& config);

}
#pragma once

#include "model.h"
#include "gguf.h"
#include "quantized_tensor.h"
#include "quant.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <cstdint>

namespace LLaMA_Tensor{
    // ---------------------------------------------------------------------------
    // Load a model from a GGUF file.
    // Supports F32, Q8_0, Q4_0, and Q4_1 tensors. Norm, embedding, and bias
    // tensors are converted to F32 at load time when needed.
    // ---------------------------------------------------------------------------
    MiniLlamaModel LoadGgufModel(const std::string& gguf_path);
}
// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#pragma once

#include "cuda_kv_cache.h"
#include "cuda_tensor.h"
#include "tensor.h"

namespace LLaMA_Tensor {

bool CudaAttentionBuilt();

Tensor CudaAttentionDecode(const Tensor& q, const CudaKvCache& kv_cache,
                           int layer, int pos, int n_heads, int n_kv_heads,
                           int head_dim, int device_id = 0);

CudaTensor CudaAttentionDecodeDeviceInput(const CudaTensor& q,
                                          const CudaKvCache& kv_cache,
                                          int layer, int pos, int n_heads,
                                          int n_kv_heads, int head_dim,
                                          int device_id = 0);

// Causal attention for a batch of new tokens. q has shape
// [rows, n_heads * head_dim] and attends to KV positions
// [0, start_pos + row] for each row. Returns [rows, n_heads * head_dim].
CudaTensor CudaAttentionBatchDeviceInput(const CudaTensor& q,
                                         const CudaKvCache& kv_cache,
                                         int layer, int start_pos, int n_heads,
                                         int n_kv_heads, int head_dim,
                                         int device_id = 0);

}  // namespace LLaMA_Tensor


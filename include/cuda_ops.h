// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include "cuda_tensor.h"
#include "ops.h"
#include "tensor.h"

namespace LLaMA_Tensor {

bool CudaOpsBuilt();

Tensor CudaRmsNorm(const Tensor& x, const Tensor& weight, float eps,
                   int device_id = 0);

Tensor CudaSilu(const Tensor& x, int device_id = 0);

Tensor CudaElementwiseMul(const Tensor& a, const Tensor& b, int device_id = 0);

Tensor CudaElementwiseAdd(const Tensor& a, const Tensor& b, int device_id = 0);

Tensor CudaSoftmax(const Tensor& x, int device_id = 0);

CudaTensor CudaEmbeddingLookupDeviceWeight(
    const void* embedding_data, const std::vector<int>& embedding_shape,
    int token_id, int device_id = 0);

// Batched embedding lookup. The result has shape [num_tokens, dim].
CudaTensor CudaEmbeddingLookupBatchDeviceWeight(
    const void* embedding_data, const std::vector<int>& embedding_shape,
    const std::vector<int>& token_ids, int device_id = 0);

CudaTensor CudaRmsNormDeviceInput(const CudaTensor& x, const Tensor& weight,
                                  float eps, int device_id = 0);

CudaTensor CudaRmsNormDeviceWeight(const CudaTensor& x, const void* weight_data,
                                   const std::vector<int>& weight_shape,
                                   float eps, int device_id = 0);

// Row-wise RMSNorm for x with shape [rows, features] and a [features] weight.
CudaTensor CudaRmsNormBatchDeviceWeight(const CudaTensor& x,
                                        const void* weight_data,
                                        const std::vector<int>& weight_shape,
                                        float eps, int device_id = 0);

CudaTensor CudaSiluDeviceInput(const CudaTensor& x, int device_id = 0);

CudaTensor CudaElementwiseMulDeviceInput(const CudaTensor& a,
                                         const CudaTensor& b,
                                         int device_id = 0);

CudaTensor CudaElementwiseAddDeviceInput(const CudaTensor& a,
                                         const CudaTensor& b,
                                         int device_id = 0);

CudaTensor CudaElementwiseAddDeviceWeight(const CudaTensor& a,
                                          const void* b_data,
                                          const std::vector<int>& b_shape,
                                          int device_id = 0);

// Adds a [features] bias to every row of a [rows, features] tensor.
CudaTensor CudaElementwiseAddRowBiasDeviceWeight(
    const CudaTensor& x, const void* bias_data,
    const std::vector<int>& bias_shape, int device_id = 0);

void CudaRope(Tensor& q, Tensor& k, int pos, float theta,
              RoPEtype rope_type = RoPEtype::Normal, int device_id = 0);

void CudaRopeDeviceInput(CudaTensor& q, CudaTensor& k, int n_heads,
                         int n_kv_heads, int head_dim, int pos, float theta,
                         RoPEtype rope_type = RoPEtype::Normal,
                         int device_id = 0);

// Applies RoPE to every row of q [rows, n_heads * head_dim] and
// k [rows, n_kv_heads * head_dim], using start_pos + row as the position.
void CudaRopeBatchDeviceInput(CudaTensor& q, CudaTensor& k, int n_heads,
                              int n_kv_heads, int head_dim, int start_pos,
                              float theta,
                              RoPEtype rope_type = RoPEtype::Normal,
                              int device_id = 0);

}  // namespace LLaMA_Tensor


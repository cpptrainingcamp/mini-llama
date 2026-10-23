#pragma once

#include "tensor.h"
#include "context.h"
#include "batch.h"

namespace LLaMA_Tensor{
// Forward pass for a single token
// Returns logits: [vocab_size]
Tensor ForwardToken(Context& ctx, const MiniLlamaModel& model,
                    int token);

// 当模型权重已上传到 CUDA、batch 的 positions 连续且所有 Linear 权重都有
// 设备端副本时返回 true。此时可以整批走 CUDA 设备路径。
bool CudaBatchPrefillSupported(const MiniLlamaModel& model,
                               const Batch& batch);

// 批量 CUDA prefill / decode：所有 token 一次在 GPU 上完成前向，
// 返回最后一个 token 的 logits: [vocab_size]。
Tensor ForwardPrefill(Context& ctx, const MiniLlamaModel& model,
                      const Batch& batch);

// Forward pass for a batch of tokens.
// CUDA 模型自动走设备端 batch 路径；否则使用 CPU 批量路径。
// 返回最后一个 token 的 logits。
Tensor ForwardBatch(Context& ctx, const MiniLlamaModel& model,
                    const Batch& batch);

}
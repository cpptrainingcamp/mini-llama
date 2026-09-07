#pragma once

#include "tensor.h"
#include "context.h"
#include "batch.h"

namespace LLaMA_Tensor{
// Forward pass for a single token
// Returns logits: [vocab_size]
Tensor ForwardToken(Context& ctx, const MiniLlamaModel& model,
                    int token);

// Forward pass for a batch of tokens.
// Internally processes tokens sequentially and returns logits for the last
// token. This unifies prefill (multi-token) and Decode (single-token) paths.
Tensor ForwardBatch(Context& ctx, const MiniLlamaModel& model,
                    const Batch& batch);

}
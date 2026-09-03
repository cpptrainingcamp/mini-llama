#include "../include/batch.h"
#include "../include/tensor.h"
#include "../include/context.h"

namespace LLaMA_Tensor{
Batch Batch::Single(int token, int pos){
    Batch batch;
    batch.tokens = {token};
    batch.positions = {pos};
    return batch;
}
Batch Batch::Prefill(std::vector<int> tokens, int start_pos){
    Batch batch;
    batch.tokens = tokens;
    batch.positions.reserve(tokens.size());
    for(int i=0;i<tokens.size();i++){
        batch.positions.push_back(start_pos++);
    }
    return batch;
}

Tensor ForwardBatch(Context& ctx, const Model& model,
                    const Batch& batch) {
  if (batch.num_tokens() == 0) {
    throw std::runtime_error("ForwardBatch: empty batch");
  }
  if (batch.tokens.size() != batch.positions.size()) {
    throw std::runtime_error(
        "ForwardBatch: tokens and positions size mismatch");
  }

  Tensor logits;
  for (int i = 0; i < batch.num_tokens(); ++i) {
    ctx.pos = batch.positions[i];
    logits = ForwardToken(ctx, model, batch.tokens[i]);
    ctx.token_history.push_back(batch.tokens[i]);
  }
  return logits;
}
}
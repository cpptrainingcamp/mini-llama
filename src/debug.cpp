#include "../include/debug.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../include/context.h"
#include "../include/forward.h"
#include "../include/ops.h"
#include "../include/sampler.h"

namespace LLaMA_Tensor {

// ---------------------------------------------------------------------------
// DumpTensorShape
// ---------------------------------------------------------------------------
void DumpTensorShape(const Tensor& t, const std::string& name) {
  std::cout << "  tensor: " << name;
  std::cout << " shape=[";
  for (size_t i = 0; i < t.shape.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << t.shape[i];
  }
  std::cout << "] dtype=f32 size=" << t.TensorNum() << "\n";
}

// ---------------------------------------------------------------------------
// DumpLogitsTopK
// ---------------------------------------------------------------------------
void DumpLogitsTopK(const Tensor& logits, int k) {
  if (logits.dim() != 1) {
    std::cout << "  logits: invalid shape for top-k dump\n";
    return;
  }
  int vocab_size = logits.shape[0];
  if (k <= 0 || k > vocab_size) {
    k = vocab_size;
  }

  // Collect (value, FlatIndex) pairs
  std::vector<std::pair<float, int>> pairs;
  pairs.reserve(vocab_size);
  for (int i = 0; i < vocab_size; ++i) {
    pairs.push_back({logits.data[i], i});
  }

  // Sort descending by value
  std::partial_sort(
      pairs.begin(), pairs.begin() + k, pairs.end(),
      [](const auto& a, const auto& b) { return a.first > b.first; });

  std::cout << "  logits top-" << k << ":\n";
  for (int i = 0; i < k; ++i) {
    std::cout << "    [" << std::setw(3) << i << "] id=" << std::setw(4)
              << pairs[i].second << " value=" << std::fixed
              << std::setprecision(6) << pairs[i].first << "\n";
  }
}

// ---------------------------------------------------------------------------
// DumpKvCacheInfo
// ---------------------------------------------------------------------------
void DumpKvCacheInfo(const KVCache& cache, int current_pos) {
  int n_layers = cache.KCache.shape[0];
  int seq_len = cache.KCache.shape[1];
  int n_kv_heads = cache.KCache.shape[2];
  int head_dim = cache.KCache.shape[3];
  std::cout << "  KV cache: layers=" << n_layers << " seq_len=" << seq_len
            << " n_kv_heads=" << n_kv_heads << " head_dim=" << head_dim << "\n";
  if (current_pos >= 0) {
    std::cout << "    current_pos: " << current_pos << "\n";
  }
  std::cout << "    keys shape: [" << cache.KCache.shape[0] << ", "
            << cache.KCache.shape[1] << ", " << cache.KCache.shape[2] << ", "
            << cache.KCache.shape[3] << "]\n";
  std::cout << "    values shape: [" << cache.VCache.shape[0] << ", "
            << cache.VCache.shape[1] << ", " << cache.VCache.shape[2] << ", "
            << cache.VCache.shape[3] << "]\n";
}

// ---------------------------------------------------------------------------
// RunBenchmark
// ---------------------------------------------------------------------------
BenchmarkResult RunBenchmark(const MiniLlamaModel& model,
                             const std::vector<int>& prompt_tokens,
                             int n_predict, unsigned int seed, bool verbose) {
  BenchmarkResult result;
  result.n_prompt_tokens = static_cast<int>(prompt_tokens.size());

  Context ctx(&model);
  Sampler sampler(seed);
  SamplerParams params;
  params.seed = seed;

  // Prefill
  if (verbose) {
    std::cout << "[verbose] prefill " << prompt_tokens.size() << " tokens\n";
  }

  auto prefill_start = std::chrono::steady_clock::now();
  Batch prefill = Batch::Prefill(prompt_tokens, 0);
  Tensor logits = ForwardBatch(ctx, model, prefill);
  auto prefill_end = std::chrono::steady_clock::now();
  result.prefill_ms =
      std::chrono::duration<double, std::milli>(prefill_end - prefill_start)
          .count();

  if (verbose) {
    DumpTensorShape(logits, "logits");
    DumpLogitsTopK(logits, 5);
    DumpKvCacheInfo(ctx.kv_cache, ctx.pos);
  }

  // Decode
  if (n_predict > 0 && result.n_prompt_tokens < model.config.max_seq_len) {
    std::vector<int> tokens = prompt_tokens;
    int next_token = sampler.Sample(logits);
    tokens.push_back(next_token);
    ++result.n_generated_tokens;

    if (verbose) {
      std::cout << "[verbose] Decode step 0: token=" << next_token << "\n";
    }

    auto decode_start = std::chrono::steady_clock::now();
    for (int i = 1; i < n_predict &&
                    static_cast<int>(tokens.size()) < model.config.max_seq_len;
         ++i) {
      Batch decode =
          Batch::Single(tokens.back(), static_cast<int>(tokens.size() - 1));
      logits = ForwardBatch(ctx, model, decode);
      ++result.n_decode_tokens;
      next_token = sampler.Sample(logits);
      tokens.push_back(next_token);
      ++result.n_generated_tokens;

      if (verbose) {
        std::cout << "[verbose] Decode step " << i << ": token=" << next_token
                  << "\n";
        DumpLogitsTopK(logits, 5);
      }
    }

    auto decode_end = std::chrono::steady_clock::now();
    if (result.n_decode_tokens > 0) {
      result.decode_ms =
          std::chrono::duration<double, std::milli>(decode_end - decode_start)
              .count();
    }
  }

  if (verbose) {
    DumpKvCacheInfo(ctx.kv_cache, ctx.pos);
  }

  return result;
}

}  // namespace mini_llama

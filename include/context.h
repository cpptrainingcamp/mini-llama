#pragma once

#include "kv_cache.h"
//#include "model.h"

namespace LLaMA_Tensor{
    struct Context{
        const MiniLlamaModel* model;
        KVCache kv_cache;
        
        int pos = 0;
        std::vector<int> token_history;
        int n_prefill_tokens = 0;
        int n_decode_tokens = 0;

        Context() = default;
        Context(const Context& context) = delete;
        Context& operator=(const Context& context) = delete;
        Context(Context&& context) noexcept = default;
        Context& operator=(Context&& context) noexcept = default;
        explicit Context(const MiniLlamaModel* model); 
    };
}

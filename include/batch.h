#pragma once

#include <vector>

namespace LLaMA_Tensor{
    struct Batch{
        std::vector<int> tokens;
        std::vector<int> positions;

        int num_tokens() const { return static_cast<int>(tokens.size()); }
        //Decode
        static Batch Single(int token, int pos); 
        //Prefill
        static Batch Prefill(std::vector<int> tokens, int start_pos);
    };
}
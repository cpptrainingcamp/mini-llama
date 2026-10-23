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

}

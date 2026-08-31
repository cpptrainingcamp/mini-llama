#include "../include/tokenizer.h"

namespace LLaMA_Tensor{
    std::vector<int> AsciiTokenizer::Encode(const std::string& text) const {
        std::vector<int>vec;
        vec.reserve(text.size()+1);
        vec.push_back(bos_id());
        for(unsigned char char_token : text){
            int token_id = static_cast<int>(char_token);
            if(token_id >= VocabSize){
                vec.push_back(unk_id());
            }else{
                vec.push_back(token_id);
            }
        }
        return vec;
    }

    std::string AsciiTokenizer::DecodeToken(int token_id) const{
        if(token_id == unk_id()){
            return "<unk>";
        } else if(token_id == eos_id()){
            return "<eos>";
        } else if(token_id == bos_id()){
            return "<bos>";
        } else if(token_id < 32 || token_id == 127){
            return "";
        }else if(token_id >= VocabSize){
            return "<unk>";
        }
        return std::string(1, static_cast<char>(token_id));
    }

    std::string AsciiTokenizer::Decode(const std::vector<int>& tokens_id) const {
        std::string tokens;
        for(int token_id : tokens_id){
            tokens += DecodeToken(token_id);
        }
        return tokens;
    }
}
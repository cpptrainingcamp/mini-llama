#include "../include/chat.h"

namespace LLaMA_Tensor{
void ChatSession::Clear(){
    msg_.clear();
    token_history_.clear();
    prefix_cache_.Clear();
    total_generated_tokens = 0;
    total_prompt_tokens = 0;
    total_time_ms = 0;
}

void ChatSession::AddMsg(const ChatMsg& chat_msg){
    msg_.push_back(chat_msg);
}

void ChatSession::RecordTurn(int prompt_tokens, int generated_tokens, double time_ms){
    total_prompt_tokens += prompt_tokens;
    total_generated_tokens += generated_tokens;
    total_time_ms += time_ms;
}

void ChatSession::SetTokenHistory(const std::vector<int>& tokens){
    token_history_ = tokens;
}

void ChatSession::AppendToken(int token){
    token_history_.push_back(token);
}

void ChatSession::RecordPrefix(const std::vector<int>& tokens){
    prefix_cache_.Insert(tokens);
}

size_t ChatSession::LongestCachedPrefix(const std::vector<int>& tokens) const{
    return prefix_cache_.LongestPrefix(tokens);
}

}
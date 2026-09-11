#pragma once

#include "context.h"
#include "radixtree.h"
#include "sampler.h"

namespace LLaMA_Tensor{
struct ChatMsg{
    std::string role; 
    std::string content;
};

class ChatSession{
public:
    std::vector<ChatMsg> msg_;
    std::vector<int> token_history_;
    RadixTree prefix_cache_;
    SamplerParams sampler_params_;
    //stats
    int total_prompt_tokens = 0;
    int total_generated_tokens = 0;
    double total_time_ms = 0.0;

    ChatSession() = default;
    //重置msg和stats
    void Clear();
    //msg历史记录
    void AddMsg(const ChatMsg& chat_msg);
    //在一次对话轮次结束后，累加本轮的性能统计指标。
    void RecordTurn(int prompt_tokens, int generated_tokens, double time_ms);
    //替换整个 token 历史，将其设置为当前已评估的完整上下文。
    void SetTokenHistory(const std::vector<int>& tokens);
    //在自回归生成过程中，每生成一个 token 就将其追加到 token 历史尾部。
    void AppendToken(int token);
    //将一条 token 序列插入到前缀缓存树（RadixTree）中，记录其前缀已被评估过。
    void RecordPrefix(const std::vector<int>& tokens);
    //查询给定的 token 序列在缓存树中能匹配到的最长公共前缀长度。
    size_t LongestCachedPrefix(const std::vector<int>& tokens) const;
};
}
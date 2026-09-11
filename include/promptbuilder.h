#pragma once

#include "chat.h"

namespace LLaMA_Tensor{
class PromptBuilder{
public:
    PromptBuilder() = default;

    //设置会话模板
    void SetChatTemplate(const std::string& template_str);
    std::string Build(const std::vector<ChatMsg>& msgs) const;
private:
    std::string chat_template_;
    std::string BuildPlain(const std::vector<ChatMsg>& msgs) const;
    std::string BuildQwen2(const std::vector<ChatMsg>& msgs) const;
};

}
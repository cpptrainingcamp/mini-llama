#pragma once

#include <string>

#include "chat.h"
#include "model.h"
#include "tokenizer.h"
#include "sampler.h"

namespace LLaMA_Tensor{
// Handles interactive terminal I/O for the chat mode.
class Terminal {
 public:
  Terminal() = default;

  // Print the user prompt (e.g., "mini-llama > ").
  void PrintUserPrompt() const;

  // Read a line from stdin. Returns empty string on EOF.
  std::string ReadLine() const;

  // Print assistant prefix (e.g., "assistant > ").
  void PrintAssistantPrefix() const;

  // Print a single decoded token (no newline, no flush).
  void PrintTokenText(const std::string& text) const;

  // Flush stdout.
  void Flush() const;

  // Print a newline.
  void NewLine() const;
  // Print the /help text.
  void PrintHelp() const;

  // Print session stats.
  void PrintStats(const ChatSession& session) const;

  // Print current sampling params.
  void PrintParams(const SamplerParams& params) const;

  // Print a generic message.
  void PrintMessage(const std::string& msg) const;
};
}
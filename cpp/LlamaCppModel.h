#pragma once

#include <jsi/jsi.h>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>

// Include all necessary common headers from llama.cpp
#include "common.h"
#include "sampling.h"
#include "chat.h"  // For chat format handling and templates
#include "json-schema-to-grammar.h"

// Include rn-utils.hpp which has the CompletionResult definition
#include "rn-utils.hpp"
#include "rn-llama.hpp"

namespace facebook::react {

// Chat message structure for representing messages in a conversation
struct Message {
  std::string role;       // Role such as "user", "assistant", "system"
  std::string content;    // Message content
  std::string name;       // Optional name field for function calls
};

// Function parameter for tool calls
struct FunctionParameter {
  std::string name;
  std::string type;
  std::string description;
  bool required;
};

// Function definition for tool calls
struct Function {
  std::string name;
  std::string description;
  std::vector<FunctionParameter> parameters;
};

// Tool for completion
struct Tool {
  std::string type;
  Function function;
};

// Tool call parsed from completion
struct ToolCall {
  std::string id;
  std::string type;
  std::string name;
  std::string arguments;
};

/**
 * LlamaCppModel - A JSI wrapper class around llama.cpp functionality
 *
 * This class manages an instance of a llama.cpp model and provides methods for:
 * - Text completion and chat completion
 * - Tokenization and detokenization
 * - Embedding generation
 *
 * It leverages native llama.cpp functionality where possible rather than reimplementing it:
 * - Uses common_chat_parse for parsing structured responses (tool calls)
 * - Uses llama_get_embeddings for embedding extraction
 * - Uses common_token_to_piece for token->text conversion
 * - Leverages the llama.cpp chat template system
 */
class LlamaCppModel : public jsi::HostObject {
public:
  /**
   * Constructor
   * @param rn_ctx A pointer to an initialized rn_llama_context
   */
  LlamaCppModel(rn_llama_context* rn_ctx);
  virtual ~LlamaCppModel();

  /**
   * Clean up resources (should be called explicitly)
   * Frees the llama_model and llama_context
   */
  void release();

  /**
   * Get information about the model
   */
  int32_t getVocabSize() const;
  int32_t getContextSize() const;
  int32_t getEmbeddingSize() const;

  /**
   * Control for active completion state
   */
  bool shouldStopCompletion() const;
  void setShouldStopCompletion(bool value);

  /**
   * Core completion method that can be called internally
   * Uses run_completion and run_chat_completion from llama.cpp
   *
   * @param options CompletionOptions with all parameters
   * @param partialCallback Callback for streaming tokens
   * @param runtime Pointer to JSI runtime for callbacks
   * @return CompletionResult with generated text and metadata
   */
  CompletionResult completion(
      const CompletionOptions& options,
      std::function<void(jsi::Runtime&, const char*)> partialCallback = nullptr,
      jsi::Runtime* runtime = nullptr);

  /**
   * JSI interface implementation
   */
  jsi::Value get(jsi::Runtime& rt, const jsi::PropNameID& name) override;
  void set(jsi::Runtime& rt, const jsi::PropNameID& name, const jsi::Value& value) override;
  std::vector<jsi::PropNameID> getPropertyNames(jsi::Runtime& rt) override;

private:
  /**
   * JSI method implementations
   */
  jsi::Value completionJsi(jsi::Runtime& rt, const jsi::Value* args, size_t count);
  jsi::Value tokenizeJsi(jsi::Runtime& rt, const jsi::Value* args, size_t count);
  jsi::Value detokenizeJsi(jsi::Runtime& rt, const jsi::Value* args, size_t count);
  jsi::Value embeddingJsi(jsi::Runtime& rt, const jsi::Value* args, size_t count);
  jsi::Value releaseJsi(jsi::Runtime& rt, const jsi::Value* args, size_t count);

  /**
   * Helper to parse completion options from JS object
   * Converts JSI objects to CompletionOptions struct
   */
  CompletionOptions parseCompletionOptions(jsi::Runtime& rt, const jsi::Object& obj);

  /**
   * Helper to convert completion result to JSI object
   * Uses common_chat_parse from llama.cpp to parse tool calls and responses
   */
  jsi::Object completionResultToJsi(jsi::Runtime& rt, const CompletionResult& result);

  /**
   * Convert JSON to JSI value
   */
  jsi::Value jsonToJsi(jsi::Runtime& rt, const json& j);

  /**
   * Initialize utility functions and handlers
   */
  void initHelpers();

  // LLAMA context pointer (owned by the module)
  rn_llama_context* rn_ctx_;

  // Completion state
  bool should_stop_completion_;
  bool is_predicting_;
};

} // namespace facebook::react
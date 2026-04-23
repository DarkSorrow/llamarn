#pragma once

#include <jsi/jsi.h>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Add ReactCommon includes for proper async handling
#include <ReactCommon/CallInvoker.h>

// Include all necessary common headers from llama.cpp
#include "common.h"
#include "sampling.h"
#include "chat.h"  // For chat format handling and templates
#include "json-schema-to-grammar.h"

// Include rn-utils.h which has the CompletionResult definition
#include "rn-utils.h"
#include "rn-llama.h"

// Include json.hpp for json handling
#include "nlohmann/json.hpp"

namespace facebook::react {


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
class LlamaCppModel : public jsi::HostObject, public std::enable_shared_from_this<LlamaCppModel> {
public:
  /**
   * Constructor
   * @param rn_ctx A pointer to an initialized rn_llama_context
   * @param jsInvoker CallInvoker for async operations (optional, for async completion)
   */
  LlamaCppModel(rn_llama_context* rn_ctx, std::shared_ptr<CallInvoker> jsInvoker = nullptr);
  virtual ~LlamaCppModel();

  /**
   * Clean up resources (should be called explicitly)
   * Frees the llama_model and llama_context
   */
  void release();

  /**
   * Get information about the model
   */
  [[nodiscard]] int32_t getVocabSize() const;
  [[nodiscard]] int32_t getContextSize() const;
  [[nodiscard]] int32_t getEmbeddingSize() const;

  /**
   * Control for active completion state
   */
  [[nodiscard]] bool shouldStopCompletion() const;
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
  jsi::Value completionAsyncJsi(jsi::Runtime& rt, const jsi::Value* args, size_t count);
  jsi::Value stopCompletionJsi(jsi::Runtime& rt, const jsi::Value* args, size_t count);
  jsi::Value tokenizeJsi(jsi::Runtime& rt, const jsi::Value* args, size_t count);
  jsi::Value detokenizeJsi(jsi::Runtime& rt, const jsi::Value* args, size_t count);
  jsi::Value embeddingJsi(jsi::Runtime& rt, const jsi::Value* args, size_t count);
  jsi::Value releaseJsi(jsi::Runtime& rt, const jsi::Value* args, size_t count);
  jsi::Value setNThreadsJsi(jsi::Runtime& rt, const jsi::Value* args, size_t count);

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

  // LLAMA context pointer (owned by the module)
  rn_llama_context* rn_ctx_;

  // Completion state — atomic because they are read on the inference thread and written on the JS thread
  std::atomic<bool> should_stop_completion_;
  std::atomic<bool> is_predicting_;

  // Add CallInvoker for async operations
  std::shared_ptr<CallInvoker> jsInvoker_;

  // Condition variable used to notify release() when inference finishes.
  // predicting_cv_mutex_ is only held briefly during the notify/wait — not during inference.
  std::condition_variable predicting_cv_;
  std::mutex              predicting_cv_mutex_;

  // Multimodal / thread-safety guards (see plan: Thread Safety Architecture)
  std::mutex        inference_mutex_;           // serializes ALL llama/mtmd inference calls
  std::atomic<bool> is_processing_frame_{false}; // instant frame drop for runOnFrame
  std::atomic<bool> is_released_{false};          // JSI teardown guard

  // Multimodal JSI methods
  jsi::Value isMultimodalEnabledJsi(jsi::Runtime& rt, const jsi::Value* args, size_t count);
  jsi::Value getSupportedModalitiesJsi(jsi::Runtime& rt, const jsi::Value* args, size_t count);
  jsi::Value embedImageJsi(jsi::Runtime& rt, const jsi::Value* args, size_t count);
  jsi::Value transcribeAudioJsi(jsi::Runtime& rt, const jsi::Value* args, size_t count);
  jsi::Value visionReasoningJsi(jsi::Runtime& rt, const jsi::Value* args, size_t count);
  jsi::Value runOnFrameJsi(jsi::Runtime& rt, const jsi::Value* args, size_t count);

  static json jsiValueToJson(jsi::Runtime& rt, const jsi::Value& val); // Declaration of new helper
};

} // namespace facebook::react
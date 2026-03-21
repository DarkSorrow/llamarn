#include "LlamaCppModel.h"
#include <jsi/jsi.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <thread>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <memory>

// Include rn-completion integration
#include "rn-utils.h"
#include "rn-llama.h"

// Include llama.cpp headers
#include "llama.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "sampling.h"

// System utilities
#include "SystemUtils.h"

// Remove the global 'using namespace' to avoid namespace conflicts
// using namespace facebook::react;

namespace facebook::react {

LlamaCppModel::LlamaCppModel(rn_llama_context* rn_ctx, std::shared_ptr<CallInvoker> jsInvoker)
    : rn_ctx_(rn_ctx), should_stop_completion_(false), is_predicting_(false), jsInvoker_(jsInvoker) {
}

LlamaCppModel::~LlamaCppModel() {
  // Note: We don't automatically release resources here
  // as the user should call release() explicitly
}

void LlamaCppModel::release() {
  // Signal inference to stop — abort_callback makes llama_decode exit on its next iteration
  should_stop_completion_ = true;
  if (rn_ctx_) {
    rn_ctx_->abort_generation = true;
  }

  // Wait for inference to finish using a condition variable.
  // With abort_callback set, this typically resolves within one token decode latency.
  {
    std::unique_lock<std::mutex> lock(predicting_cv_mutex_);
    predicting_cv_.wait_for(lock, std::chrono::milliseconds(500),
                            [this] { return !is_predicting_.load(); });
  }

  // Clean up our resources with proper mutex protection
  // NOTE: We do NOT manually free the context or model here because they are owned
  // by common_init_result_ptr in PureCppImpl. Manual freeing would cause a double-free
  // crash when init_result_ is destroyed. Instead, we just clear the cache and reset
  // pointers - the actual freeing will be handled by init_result_'s destructor.
  if (rn_ctx_) {
    std::lock_guard<std::mutex> lock(rn_ctx_->mutex);

    // Clear KV cache before context is freed (following server.cpp pattern)
    // This is safe even if context will be freed later by init_result_
    if (rn_ctx_->ctx) {
      try {
        llama_memory_clear(llama_get_memory(rn_ctx_->ctx), true);
      } catch (...) {
        // Ignore errors during cache clearing
      }
      
      // DO NOT call llama_free() here - init_result_ owns the context
      rn_ctx_->ctx = nullptr;
    }

    // DO NOT call llama_model_free() here - init_result_ owns the model
    rn_ctx_->model = nullptr;

    // Clean up additional resources
    rn_ctx_->vocab = nullptr; // This is owned by the model, so just null it
    rn_ctx_->chat_templates.reset(); // Clean up chat templates
    rn_ctx_->lora_adapters.clear(); // Clear LoRA adapters
    
    // Reset state flags
    rn_ctx_->model_loaded = false;

    // Note: rn_ctx_ itself is owned by the module, so we don't delete it here
    rn_ctx_ = nullptr;
  }

  // Reset our internal state
  should_stop_completion_ = false;
  is_predicting_ = false;
}

int32_t LlamaCppModel::getVocabSize() const {
  if (!rn_ctx_ || !rn_ctx_->model) {
    throw std::runtime_error("Model not loaded");
  }

  return llama_vocab_n_tokens(rn_ctx_->vocab);
}

int32_t LlamaCppModel::getContextSize() const {
  if (!rn_ctx_ || !rn_ctx_->ctx) {
    throw std::runtime_error("Context not initialized");
  }

  return llama_n_ctx(rn_ctx_->ctx);
}

bool LlamaCppModel::shouldStopCompletion() const {
  return should_stop_completion_;
}

void LlamaCppModel::setShouldStopCompletion(bool value) {
  should_stop_completion_ = value;
  if (value && rn_ctx_) {
    // Signal the abort_callback — llama_decode will exit on its next graph eval
    rn_ctx_->abort_generation = true;
  }
}

// Parse the CompletionOptions from a JS object
CompletionOptions LlamaCppModel::parseCompletionOptions(jsi::Runtime& rt, const jsi::Object& obj) {
  CompletionOptions options;

  // Extract basic options
  if (obj.hasProperty(rt, "prompt") && !obj.getProperty(rt, "prompt").isUndefined()) {
    options.prompt = obj.getProperty(rt, "prompt").asString(rt).utf8(rt);
  }

  // Parse sampling parameters
  if (obj.hasProperty(rt, "temperature") && !obj.getProperty(rt, "temperature").isUndefined()) {
    options.temperature = obj.getProperty(rt, "temperature").asNumber();
  }

  if (obj.hasProperty(rt, "top_p") && !obj.getProperty(rt, "top_p").isUndefined()) {
    options.top_p = obj.getProperty(rt, "top_p").asNumber();
  }

  if (obj.hasProperty(rt, "top_k") && !obj.getProperty(rt, "top_k").isUndefined()) {
    options.top_k = obj.getProperty(rt, "top_k").asNumber();
  }

  if (obj.hasProperty(rt, "min_p") && !obj.getProperty(rt, "min_p").isUndefined()) {
    options.min_p = obj.getProperty(rt, "min_p").asNumber();
  }

  if (obj.hasProperty(rt, "presence_penalty") && !obj.getProperty(rt, "presence_penalty").isUndefined()) {
    options.presence_penalty = obj.getProperty(rt, "presence_penalty").asNumber();
  }

  if (obj.hasProperty(rt, "repeat_penalty") && !obj.getProperty(rt, "repeat_penalty").isUndefined()) {
    options.repeat_penalty = obj.getProperty(rt, "repeat_penalty").asNumber();
  }

  if (obj.hasProperty(rt, "repeat_last_n") && !obj.getProperty(rt, "repeat_last_n").isUndefined()) {
    options.repeat_last_n = static_cast<int>(obj.getProperty(rt, "repeat_last_n").asNumber());
  }

  if (obj.hasProperty(rt, "frequency_penalty") && !obj.getProperty(rt, "frequency_penalty").isUndefined()) {
    options.frequency_penalty = obj.getProperty(rt, "frequency_penalty").asNumber();
  }

  if (obj.hasProperty(rt, "n_predict") && !obj.getProperty(rt, "n_predict").isUndefined()) {
    options.n_predict = obj.getProperty(rt, "n_predict").asNumber();
  } else if (obj.hasProperty(rt, "max_tokens") && !obj.getProperty(rt, "max_tokens").isUndefined()) {
    options.n_predict = obj.getProperty(rt, "max_tokens").asNumber();
  }

  if (obj.hasProperty(rt, "n_keep") && !obj.getProperty(rt, "n_keep").isUndefined()) {
    options.n_keep = obj.getProperty(rt, "n_keep").asNumber();
  }

  // Extract seed
  if (obj.hasProperty(rt, "seed") && !obj.getProperty(rt, "seed").isUndefined()) {
    options.seed = obj.getProperty(rt, "seed").asNumber();
  }

  // Extract stop sequences
  if (obj.hasProperty(rt, "stop") && !obj.getProperty(rt, "stop").isUndefined()) {
    auto stopVal = obj.getProperty(rt, "stop");
    if (stopVal.isString()) {
      options.stop.push_back(stopVal.asString(rt).utf8(rt));
    } else if (stopVal.isObject() && stopVal.getObject(rt).isArray(rt)) {
      auto stopArr = stopVal.getObject(rt).getArray(rt);
      for (size_t i = 0; i < stopArr.size(rt); i++) {
        auto item = stopArr.getValueAtIndex(rt, i);
        if (item.isString()) {
          options.stop.push_back(item.asString(rt).utf8(rt));
        }
      }
    }
  }

  // Extract grammar
  if (obj.hasProperty(rt, "grammar") && !obj.getProperty(rt, "grammar").isUndefined()) {
    options.grammar = obj.getProperty(rt, "grammar").asString(rt).utf8(rt);
  }

  if (obj.hasProperty(rt, "ignore_eos") && !obj.getProperty(rt, "ignore_eos").isUndefined()) {
    options.ignore_eos = obj.getProperty(rt, "ignore_eos").asBool();
  }

  if (obj.hasProperty(rt, "stream") && !obj.getProperty(rt, "stream").isUndefined()) {
    options.stream = obj.getProperty(rt, "stream").asBool();
  }

  // Convert messages and tools directly using jsiValueToJson — no manual field extraction.
  // This preserves all fields the model template needs (role, content, tool_calls, tool_call_id,
  // reasoning_content, name, etc.) exactly as the JS layer provides them, in the OpenAI-compatible
  // format that common_chat_msgs_parse_oaicompat and common_chat_tools_parse_oaicompat expect.
  if (obj.hasProperty(rt, "messages") && obj.getProperty(rt, "messages").isObject()) {
    auto messagesVal = obj.getProperty(rt, "messages").getObject(rt);
    if (messagesVal.isArray(rt)) {
      options.messages = jsiValueToJson(rt, jsi::Value(rt, std::move(messagesVal)));
    }
  }

  if (obj.hasProperty(rt, "tools") && obj.getProperty(rt, "tools").isObject()) {
    auto toolsVal = obj.getProperty(rt, "tools").getObject(rt);
    if (toolsVal.isArray(rt)) {
      options.tools = jsiValueToJson(rt, jsi::Value(rt, std::move(toolsVal)));
    }
  }

  // Extract tool_choice if present
  if (obj.hasProperty(rt, "tool_choice") && !obj.getProperty(rt, "tool_choice").isUndefined()) {
    auto toolChoiceVal = obj.getProperty(rt, "tool_choice");
    if (toolChoiceVal.isString()) {
      options.tool_choice = toolChoiceVal.asString(rt).utf8(rt);
    } else if (toolChoiceVal.isObject()) {
      // OpenAI allows tool_choice = { type: "function", function: { name: "foo" } }
      // to force a specific function. llama.cpp's common_chat_tool_choice_parse_oaicompat
      // only accepts "auto" / "none" / "required" — per-function enforcement is not
      // available at the sampling level. Map to "required" (forces tool use), which is
      // the closest valid approximation.
      options.tool_choice = "required";
    }
  }

  // Extract chat template name if provided
  if (obj.hasProperty(rt, "chat_template") && !obj.getProperty(rt, "chat_template").isUndefined()) {
    options.chat_template = obj.getProperty(rt, "chat_template").asString(rt).utf8(rt);
  }

  return options;
}

// Modify the completion function to use this helper
CompletionResult LlamaCppModel::completion(const CompletionOptions& options, std::function<void(jsi::Runtime&, const char*)> partialCallback, jsi::Runtime* runtime) {
  if (!rn_ctx_ || !rn_ctx_->model || !rn_ctx_->ctx) {
    CompletionResult result;
    result.content = "";
    result.success = false;
    result.error_msg = "Model or context not initialized";
    result.error_type = RN_ERROR_MODEL_LOAD;
    return result;
  }

  // Lock the mutex during completion to avoid concurrent accesses
  std::lock_guard<std::mutex> lock(rn_ctx_->mutex);

  // Clear the context KV cache
  llama_memory_clear(llama_get_memory(rn_ctx_->ctx), true);

  // Sampling overrides are applied per-request inside run_completion() on a LOCAL
  // copy of the sampling params — rn_ctx_->params is never mutated here.

  // Check for a partial callback
  auto callback_adapter = [&partialCallback, runtime, this](const std::string& token, bool is_done) -> bool {
    // Check for stop condition first
    if (should_stop_completion_) {
      return false; // Signal to stop completion
    }
    
    if (partialCallback && runtime && !is_done) {
      partialCallback(*runtime, token.c_str());
    }
    
    // Return true to continue, false to stop
    return !should_stop_completion_;
  };

  // Run the completion based on whether we have messages or prompt
  CompletionResult result;

  try {
    // Set the predicting flag and reset stop signals for this run
    {
      std::lock_guard<std::mutex> cv_lock(predicting_cv_mutex_);
      is_predicting_ = true;
    }
    should_stop_completion_ = false;
    if (rn_ctx_) {
      rn_ctx_->abort_generation = false;
    }

    if (!options.messages.empty()) {
      // Chat completion (with messages)
      result = run_chat_completion(rn_ctx_, options, callback_adapter);
    } else {
      // Regular completion (with prompt)
      result = run_completion(rn_ctx_, options, callback_adapter);
    }

    // Notify release() (or any waiter) that inference is done
    {
      std::lock_guard<std::mutex> cv_lock(predicting_cv_mutex_);
      is_predicting_ = false;
    }
    predicting_cv_.notify_all();
  } catch (const std::exception& e) {
    {
      std::lock_guard<std::mutex> cv_lock(predicting_cv_mutex_);
      is_predicting_ = false;
    }
    predicting_cv_.notify_all();
    result.success = false;
    result.error_msg = std::string("Completion failed: ") + e.what();
    result.error_type = RN_ERROR_INFERENCE;
  }

  return result;
}

// Helper to convert from the rn-utils CompletionResult to a JSI object
jsi::Object LlamaCppModel::completionResultToJsi(jsi::Runtime& rt, const CompletionResult& result) {
  jsi::Object jsResult(rt);

  // Check if this is a chat completion
  if (!result.chat_response.empty()) {
    // For chat completions, convert the JSON response directly to JSI
    jsi::Object chatResponse = jsonToJsi(rt, result.chat_response).asObject(rt);

    // Add tool_calls as a top-level property for compatibility with clients
    // that expect tool_calls at the top level rather than under choices[0].message
    if (result.chat_response.contains("choices") &&
        !result.chat_response["choices"].empty() &&
        result.chat_response["choices"][0].contains("message") &&
        result.chat_response["choices"][0]["message"].contains("tool_calls")) {

      // Always add tool_calls to the top level (don't check if it exists)
      chatResponse.setProperty(rt, "tool_calls",
        jsonToJsi(rt, result.chat_response["choices"][0]["message"]["tool_calls"]));
    }

    return chatResponse;
  }

  // Standard completion result
  jsResult.setProperty(rt, "content", jsi::String::createFromUtf8(rt, result.content));
  {
    jsi::Object timingsObj(rt);
    timingsObj.setProperty(rt, "predicted_n",  jsi::Value(static_cast<double>(result.timings.predicted_n)));
    timingsObj.setProperty(rt, "predicted_ms", jsi::Value(result.timings.predicted_ms));
    timingsObj.setProperty(rt, "prompt_n",     jsi::Value(static_cast<double>(result.timings.prompt_n)));
    timingsObj.setProperty(rt, "prompt_ms",    jsi::Value(result.timings.prompt_ms));
    timingsObj.setProperty(rt, "total_ms",     jsi::Value(result.timings.total_ms));
    jsResult.setProperty(rt, "timings", std::move(timingsObj));
  }
  jsResult.setProperty(rt, "success", jsi::Value(result.success));
  jsResult.setProperty(rt, "promptTokens", jsi::Value(result.n_prompt_tokens));
  jsResult.setProperty(rt, "completionTokens", jsi::Value(result.n_predicted_tokens));

  if (!result.success) {
    jsResult.setProperty(rt, "error", jsi::String::createFromUtf8(rt, result.error_msg));
    jsResult.setProperty(rt, "errorType", jsi::Value(static_cast<int>(result.error_type)));
  }

  return jsResult;
}

// Convert a JSON object to a JSI value
jsi::Value LlamaCppModel::jsonToJsi(jsi::Runtime& rt, const json& j) {
  if (j.is_null()) {
    return jsi::Value::null();
  } else if (j.is_boolean()) {
    return jsi::Value(j.get<bool>());
  } else if (j.is_number_integer()) {
    return jsi::Value(j.get<int>());
  } else if (j.is_number_float()) {
    return jsi::Value(j.get<double>());
  } else if (j.is_string()) {
    return jsi::String::createFromUtf8(rt, j.get<std::string>());
  } else if (j.is_array()) {
    jsi::Array array(rt, j.size());
    for (size_t i = 0; i < j.size(); i++) {
      array.setValueAtIndex(rt, i, jsonToJsi(rt, j[i]));
    }
    return array;
  } else if (j.is_object()) {
    jsi::Object object(rt);
    for (const auto& [key, val] : j.items()) {
      object.setProperty(rt, key.c_str(), jsonToJsi(rt, val));
    }
    return object;
  }

  // Default case (shouldn't happen)
  return jsi::Value::undefined();
}

// Helper to convert JSI Value to nlohmann::json
json LlamaCppModel::jsiValueToJson(jsi::Runtime& rt, const jsi::Value& val) {
    if (val.isUndefined() || val.isNull()) {
        return nullptr;
    } else if (val.isBool()) {
        return val.getBool();
    } else if (val.isNumber()) {
        double num = val.getNumber();
        // Store whole-number doubles as int64_t so nlohmann's is_number_integer()
        // returns true and json-schema-to-grammar integer-bound checks work correctly.
        // JS has only one numeric type (double), but JSON distinguishes integers from
        // floats. Without this, schema values like { minLength: 1 } become 1.0
        // (number_float), causing is_number_integer() checks in json-schema-to-grammar
        // to silently fail (e.g. maxItems becomes INT_MAX instead of the actual limit).
        if (std::isfinite(num) && num == std::floor(num) &&
            num >= static_cast<double>(std::numeric_limits<int64_t>::min()) &&
            num <= static_cast<double>(std::numeric_limits<int64_t>::max())) {
            return static_cast<int64_t>(num);
        }
        return num;
    } else if (val.isString()) {
        return val.getString(rt).utf8(rt);
    } else if (val.isObject()) {
        jsi::Object jsiObj = val.getObject(rt);
        if (jsiObj.isArray(rt)) {
            jsi::Array jsiArr = jsiObj.getArray(rt);
            json jsonArr = json::array();
            for (size_t i = 0; i < jsiArr.size(rt); ++i) {
                jsonArr.push_back(jsiValueToJson(rt, jsiArr.getValueAtIndex(rt, i)));
            }
            return jsonArr;
        } else {
            json jsonObj = json::object();
            jsi::Array propNames = jsiObj.getPropertyNames(rt);
            for (size_t i = 0; i < propNames.size(rt); ++i) {
                jsi::String propName = propNames.getValueAtIndex(rt, i).asString(rt);
                std::string key = propName.utf8(rt);
                jsi::Value propVal = jsiObj.getProperty(rt, propName);
                // Skip undefined-valued properties — matches JSON.stringify behaviour.
                // Without this, { type: undefined } in a JSON Schema becomes
                // { "type": null }, which causes json-schema-to-grammar to push an
                // "Unrecognized schema" error and throw at check_errors().
                if (propVal.isUndefined()) {
                    continue;
                }
                jsonObj[key] = jsiValueToJson(rt, propVal);
            }
            return jsonObj;
        }
    }
    return nullptr;
}

// JSI method for completions (synchronous - kept for compatibility)
jsi::Value LlamaCppModel::completionJsi(jsi::Runtime& rt, const jsi::Value* args, size_t count) {
  if (count < 1 || !args[0].isObject()) {
    throw jsi::JSError(rt, "completion requires an options object");
  }

  // Create partial callback function for token streaming
  std::function<void(jsi::Runtime&, const char*)> partialCallback = nullptr;

  if (count > 1 && args[1].isObject() && args[1].getObject(rt).isFunction(rt)) {
    auto callbackFn = std::make_shared<jsi::Function>(args[1].getObject(rt).getFunction(rt));
    partialCallback = [callbackFn](jsi::Runtime& rt, const char* token) {
      jsi::Object data(rt);
      data.setProperty(rt, "token", jsi::String::createFromUtf8(rt, token));
      callbackFn->call(rt, data);
    };
  }

  try {
    // Parse options from JSI object
    CompletionOptions options = parseCompletionOptions(rt, args[0].getObject(rt));

    // Set streaming flag based on callback presence
    options.stream = (partialCallback != nullptr);

    // Call our C++ completion method which properly initializes rn_llama_context
    CompletionResult result = completion(options, partialCallback, &rt);

    // Convert the result to a JSI object using our helper
    return completionResultToJsi(rt, result);
  } catch (const std::exception& e) {
    throw jsi::JSError(rt, e.what());
  }
}

// JSI method for async completions (recommended approach)
jsi::Value LlamaCppModel::completionAsyncJsi(jsi::Runtime& rt, const jsi::Value* args, size_t count) {
  if (count < 1 || !args[0].isObject()) {
    throw jsi::JSError(rt, "completion requires an options object");
  }

  if (!jsInvoker_) {
    // Fallback to synchronous if no CallInvoker available
    return completionJsi(rt, args, count);
  }

  // Parse options and callback on the current thread
  CompletionOptions options;
  std::shared_ptr<jsi::Function> callbackFn = nullptr;
  
  try {
    options = parseCompletionOptions(rt, args[0].getObject(rt));
    
    if (count > 1 && args[1].isObject() && args[1].getObject(rt).isFunction(rt)) {
      callbackFn = std::make_shared<jsi::Function>(args[1].getObject(rt).getFunction(rt));
      options.stream = true;
    }
  } catch (const std::exception& e) {
    throw jsi::JSError(rt, e.what());
  }

  // Create Promise constructor
  auto Promise = rt.global().getPropertyAsFunction(rt, "Promise");
  
  auto executor = jsi::Function::createFromHostFunction(
    rt,
    jsi::PropNameID::forAscii(rt, "executor"),
    2,
    [this, options, callbackFn](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* args, size_t count) -> jsi::Value {
      
      auto resolve = std::make_shared<jsi::Function>(args[0].asObject(runtime).asFunction(runtime));
      auto reject = std::make_shared<jsi::Function>(args[1].asObject(runtime).asFunction(runtime));
      
      // Create shared references to runtime and invoker for thread safety
      auto runtimePtr = &runtime;
      auto invoker = jsInvoker_;
      auto selfPtr = shared_from_this(); // This requires LlamaCppModel to inherit from std::enable_shared_from_this
      
      // Launch background thread for completion
      std::thread([selfPtr, options, callbackFn, resolve, reject, runtimePtr, invoker]() {
        try {
          // Create callback that schedules token updates on JS thread
          std::function<void(jsi::Runtime&, const char*)> partialCallback = nullptr;
          
          if (callbackFn && invoker) {
            partialCallback = [callbackFn, invoker, runtimePtr](jsi::Runtime& rt, const char* token) {
              std::string tokenCopy(token);
              invoker->invokeAsync([callbackFn, tokenCopy, runtimePtr]() {
                try {
                  jsi::Object data(*runtimePtr);
                  data.setProperty(*runtimePtr, "token", jsi::String::createFromUtf8(*runtimePtr, tokenCopy));
                  callbackFn->call(*runtimePtr, data);
                } catch (...) {
                  // Ignore callback errors
                }
              });
            };
          }
          
          // Run completion
          CompletionResult result = selfPtr->completion(options, partialCallback, runtimePtr);
          
          // Schedule success callback on JS thread
          invoker->invokeAsync([selfPtr, resolve, result, runtimePtr]() {
            try {
              jsi::Object jsResult = selfPtr->completionResultToJsi(*runtimePtr, result);
              resolve->call(*runtimePtr, jsResult);
            } catch (const std::exception& e) {
              // If conversion fails, create a simple error response
              jsi::Object errorObj(*runtimePtr);
              errorObj.setProperty(*runtimePtr, "error", jsi::String::createFromUtf8(*runtimePtr, e.what()));
              resolve->call(*runtimePtr, errorObj);
            }
          });
          
        } catch (const std::exception& e) {
          // Schedule error callback on JS thread
          std::string errorMsg(e.what());
          invoker->invokeAsync([reject, errorMsg, runtimePtr]() {
            try {
              reject->call(*runtimePtr, jsi::String::createFromUtf8(*runtimePtr, errorMsg));
            } catch (...) {
              // Ignore rejection errors
            }
          });
        }
      }).detach();
      
      return jsi::Value::undefined();
    }
  );
  
  return Promise.callAsConstructor(rt, std::move(executor));
}

// JSI method for stopping completion
jsi::Value LlamaCppModel::stopCompletionJsi(jsi::Runtime& rt, const jsi::Value* args, size_t count) {
  try {
    setShouldStopCompletion(true);
    return jsi::Value(true);
  } catch (const std::exception& e) {
    throw jsi::JSError(rt, e.what());
  }
}

jsi::Value LlamaCppModel::tokenizeJsi(jsi::Runtime& rt, const jsi::Value* args, size_t count) {
  if (count < 1 || !args[0].isObject()) {
    throw jsi::JSError(rt, "tokenize requires an options object with 'content' field");
  }

  try {
    jsi::Object options = args[0].getObject(rt);

    // Extract required content parameter
    if (!options.hasProperty(rt, "content") || !options.getProperty(rt, "content").isString()) {
      throw jsi::JSError(rt, "tokenize requires a 'content' string field");
    }
    std::string content = options.getProperty(rt, "content").getString(rt).utf8(rt);

    // Extract optional parameters using SystemUtils helpers
    bool add_special = false;
    bool with_pieces = false;
    SystemUtils::setIfExists(rt, options, "add_special", add_special);
    SystemUtils::setIfExists(rt, options, "with_pieces", with_pieces);

    // Parameter for llama_tokenize
    bool parse_special = true;

    if (!rn_ctx_ || !rn_ctx_->model || !rn_ctx_->vocab) {
      throw std::runtime_error("Model not loaded or vocab not available");
    }

    // Use the common_token_to_piece function from llama.cpp for more consistent tokenization
    std::vector<llama_token> tokens;

    if (content.empty()) {
      // Handle empty content specially
      tokens = {};
    } else {
      // First determine how many tokens are needed
      int n_tokens = llama_tokenize(rn_ctx_->vocab, content.c_str(), content.length(), nullptr, 0, add_special, parse_special);
      if (n_tokens < 0) {
        n_tokens = -n_tokens; // Convert negative value (indicates insufficient buffer)
      }

      // Allocate buffer and do the actual tokenization
      tokens.resize(n_tokens);
      n_tokens = llama_tokenize(rn_ctx_->vocab, content.c_str(), content.length(), tokens.data(), tokens.size(), add_special, parse_special);

      if (n_tokens < 0) {
        throw std::runtime_error("Tokenization failed: insufficient buffer");
      }

      // Resize to the actual number of tokens used
      tokens.resize(n_tokens);
    }

    // Create result object with tokens array
    jsi::Object result(rt);
    jsi::Array tokensArray(rt, tokens.size());

    // Fill the tokens array with token IDs and text
    for (size_t i = 0; i < tokens.size(); i++) {
      if (with_pieces) {
        // Create an object with ID and piece text
        jsi::Object tokenObj(rt);
        tokenObj.setProperty(rt, "id", jsi::Value(static_cast<int>(tokens[i])));

        // Get the text piece for this token
        std::string piece = common_token_to_piece(rn_ctx_->vocab, tokens[i]);
        tokenObj.setProperty(rt, "text", jsi::String::createFromUtf8(rt, piece));

        tokensArray.setValueAtIndex(rt, i, tokenObj);
      } else {
        // Just add the token ID
        tokensArray.setValueAtIndex(rt, i, jsi::Value(static_cast<int>(tokens[i])));
      }
    }

    result.setProperty(rt, "tokens", tokensArray);
    result.setProperty(rt, "count", jsi::Value(static_cast<int>(tokens.size())));

    return result;
  } catch (const std::exception& e) {
    throw jsi::JSError(rt, std::string("Tokenization error: ") + e.what());
  }
}

jsi::Value LlamaCppModel::detokenizeJsi(jsi::Runtime& rt, const jsi::Value* args, size_t count) {
  if (count < 1 || !args[0].isObject()) {
    throw jsi::JSError(rt, "detokenize requires an options object with 'tokens' field");
  }

  try {
    jsi::Object options = args[0].getObject(rt);

    // Extract required tokens parameter
    if (!options.hasProperty(rt, "tokens") || !options.getProperty(rt, "tokens").isObject()) {
      throw jsi::JSError(rt, "detokenize requires a 'tokens' array field");
    }

    auto tokensVal = options.getProperty(rt, "tokens").getObject(rt);
    if (!tokensVal.isArray(rt)) {
      throw jsi::JSError(rt, "tokens must be an array");
    }

    jsi::Array tokensArr = tokensVal.getArray(rt);
    auto token_count = static_cast<int>(tokensArr.size(rt));

    if (!rn_ctx_ || !rn_ctx_->model || !rn_ctx_->vocab) {
      throw std::runtime_error("Model not loaded or vocab not available");
    }

    // Create a vector of token IDs
    std::vector<llama_token> tokens;
    tokens.reserve(token_count);

    for (int i = 0; i < token_count; i++) {
      auto val = tokensArr.getValueAtIndex(rt, i);
      if (val.isNumber()) {
        tokens.push_back(static_cast<llama_token>(val.asNumber()));
      } else if (val.isObject() && val.getObject(rt).hasProperty(rt, "id")) {
        auto id = val.getObject(rt).getProperty(rt, "id");
        if (id.isNumber()) {
          tokens.push_back(static_cast<llama_token>(id.asNumber()));
        }
      }
    }

    // Use common_token_to_piece for each token and concatenate the results
    std::string result_text;
    for (auto token : tokens) {
      result_text += common_token_to_piece(rn_ctx_->vocab, token);
    }

    // Create result object
    jsi::Object result(rt);
    result.setProperty(rt, "text", jsi::String::createFromUtf8(rt, result_text));

    return result;
  } catch (const std::exception& e) {
    throw jsi::JSError(rt, std::string("Detokenization error: ") + e.what());
  }
}

// Get embedding size for the model
int32_t LlamaCppModel::getEmbeddingSize() const {
  if (!rn_ctx_ || !rn_ctx_->model) {
    throw std::runtime_error("Model not loaded");
  }

  return llama_model_n_embd(rn_ctx_->model);
}

// Fixed embeddingJsi method with corrected API calls
jsi::Value LlamaCppModel::embeddingJsi(jsi::Runtime& rt, const jsi::Value* args, size_t count) {
  if (count < 1 || !args[0].isObject()) {
    throw jsi::JSError(rt, "embedding requires an options object with 'input' or 'content' field");
  }

  try {
    jsi::Object options = args[0].getObject(rt);

    // Extract required content parameter, support both 'input' (OpenAI) and 'content' (custom format)
    std::string content;

    if (options.hasProperty(rt, "input") && options.getProperty(rt, "input").isString()) {
      content = options.getProperty(rt, "input").getString(rt).utf8(rt);
    } else if (options.hasProperty(rt, "content") && options.getProperty(rt, "content").isString()) {
      content = options.getProperty(rt, "content").getString(rt).utf8(rt);
    } else {
      throw jsi::JSError(rt, "embedding requires either 'input' or 'content' string field");
    }

    // Check optional parameters
    std::string encoding_format = "float";
    if (options.hasProperty(rt, "encoding_format") && options.getProperty(rt, "encoding_format").isString()) {
      encoding_format = options.getProperty(rt, "encoding_format").getString(rt).utf8(rt);
      if (encoding_format != "float" && encoding_format != "base64") {
        throw jsi::JSError(rt, "encoding_format must be either 'float' or 'base64'");
      }
    }

    bool add_bos = true;
    if (options.hasProperty(rt, "add_bos_token") && options.getProperty(rt, "add_bos_token").isBool()) {
      add_bos = options.getProperty(rt, "add_bos_token").getBool();
    }

    // Check model and context
    if (!rn_ctx_ || !rn_ctx_->model || !rn_ctx_->ctx || !rn_ctx_->vocab) {
      throw std::runtime_error("Model not loaded or context not initialized");
    }

    // Tokenize the input text
    std::vector<llama_token> tokens;
    int n_tokens = llama_tokenize(rn_ctx_->vocab, content.c_str(), content.length(), nullptr, 0, add_bos, true);
    if (n_tokens < 0) {
      n_tokens = -n_tokens;
    }
    tokens.resize(n_tokens);
    n_tokens = llama_tokenize(rn_ctx_->vocab, content.c_str(), content.length(), tokens.data(), n_tokens, add_bos, true);
    if (n_tokens < 0) {
      throw std::runtime_error("Tokenization failed for embedding");
    }
    tokens.resize(n_tokens);

    if (tokens.empty()) {
      throw jsi::JSError(rt, "No tokens generated from input text");
    }

    // Clear the context KV cache to ensure clean embedding
    llama_memory_clear(llama_get_memory(rn_ctx_->ctx), true);

    // Enable embedding mode
    llama_set_embeddings(rn_ctx_->ctx, true);

    // Create and populate batch using common_batch functions (following embedding.cpp pattern)
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    
    common_batch_clear(batch);
    const llama_seq_id seq_id = 0;
    // For embeddings, we need logits for the token(s) that will produce embeddings
    // For pooling models: typically only the last token needs logits (sequence-level embedding)
    // For non-pooling models: we can get token-level embeddings, but for OpenAI compatibility
    // we typically want a single embedding, so we'll use the last token
    for (int i = 0; i < (int)tokens.size(); i++) {
      // Set logits for the last token (for both pooling and non-pooling, we want the final embedding)
      bool needs_logits = (i == (int)tokens.size() - 1);
      common_batch_add(batch, tokens[i], i, {seq_id}, needs_logits);
    }

    if (llama_decode(rn_ctx_->ctx, batch) != 0) {
      llama_batch_free(batch);
      throw std::runtime_error("Failed to decode tokens for embedding");
    }

    // Get embedding output size from the model (may differ from input size)
    const int n_embd_out = llama_model_n_embd_out(rn_ctx_->model);
    if (n_embd_out <= 0) {
      llama_batch_free(batch);
      throw std::runtime_error("Invalid embedding output dimension");
    }

    // Get the pooling type to determine which API to use
    const enum llama_pooling_type pooling_type = llama_pooling_type(rn_ctx_->ctx);
    
    // Get the embeddings based on pooling type (following embedding.cpp pattern)
    std::vector<float> embedding_vec(n_embd_out);
    const float* embd = nullptr;

    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
      // For non-pooling models, get the token-level embedding for the last token
      // Since we only set logits for the last token, the batch index is batch.n_tokens - 1
      // The index in llama_get_embeddings_ith refers to the batch position
      int last_token_batch_idx = batch.n_tokens - 1;
      if (last_token_batch_idx < 0 || !batch.logits[last_token_batch_idx]) {
        llama_batch_free(batch);
        throw std::runtime_error("No tokens with logits found in batch");
      }
      embd = llama_get_embeddings_ith(rn_ctx_->ctx, last_token_batch_idx);
    } else {
      // For pooling models, get the sequence-level embedding
      embd = llama_get_embeddings_seq(rn_ctx_->ctx, seq_id);
    }

    llama_batch_free(batch);

    if (!embd) {
      throw std::runtime_error("Failed to extract embeddings - model may not support embeddings or pooling configuration is invalid");
    }

    // Copy embeddings to our vector
    std::copy(embd, embd + n_embd_out, embedding_vec.begin());

    // Normalize embedding using common_embd_normalize (Euclidean norm, type 2)
    // This matches the behavior in embedding.cpp example (always normalizes)
    std::vector<float> normalized_vec(n_embd_out);
    common_embd_normalize(embedding_vec.data(), normalized_vec.data(), n_embd_out, 2);
    embedding_vec = std::move(normalized_vec);

    // Create OpenAI-compatible response
    jsi::Object response(rt);

    // Add embedding data
    jsi::Array dataArray(rt, 1);
    jsi::Object embeddingObj(rt);

    if (encoding_format == "base64") {
      // Base64 encode the embedding vector
      const char* data_ptr = reinterpret_cast<const char*>(embedding_vec.data());
      size_t data_size = embedding_vec.size() * sizeof(float);
      std::string base64_str = base64::encode(data_ptr, data_size);

      embeddingObj.setProperty(rt, "embedding", jsi::String::createFromUtf8(rt, base64_str));
      embeddingObj.setProperty(rt, "encoding_format", jsi::String::createFromUtf8(rt, "base64"));
    } else {
      // Create embedding array of floats
      jsi::Array embeddingArray(rt, n_embd_out);
      for (int i = 0; i < n_embd_out; i++) {
        embeddingArray.setValueAtIndex(rt, i, jsi::Value(embedding_vec[i]));
      }
      embeddingObj.setProperty(rt, "embedding", embeddingArray);
    }

    embeddingObj.setProperty(rt, "object", jsi::String::createFromUtf8(rt, "embedding"));
    embeddingObj.setProperty(rt, "index", jsi::Value(0));

    dataArray.setValueAtIndex(rt, 0, embeddingObj);

    // Create model info
    std::string model_name = "llamacpp";
    if (options.hasProperty(rt, "model") && options.getProperty(rt, "model").isString()) {
      model_name = options.getProperty(rt, "model").getString(rt).utf8(rt);
    }

    // Create usage info
    jsi::Object usage(rt);
    usage.setProperty(rt, "prompt_tokens", jsi::Value(static_cast<int>(tokens.size())));
    usage.setProperty(rt, "total_tokens", jsi::Value(static_cast<int>(tokens.size())));

    // Assemble the response
    response.setProperty(rt, "object", jsi::String::createFromUtf8(rt, "list"));
    response.setProperty(rt, "data", dataArray);
    response.setProperty(rt, "model", jsi::String::createFromUtf8(rt, model_name));
    response.setProperty(rt, "usage", usage);

    return response;
  } catch (const std::exception& e) {
    throw jsi::JSError(rt, std::string("Embedding error: ") + e.what());
  }
}

jsi::Value LlamaCppModel::releaseJsi(jsi::Runtime& rt, const jsi::Value* args, size_t count) {
  try {
    release();
    return jsi::Value(true);
  } catch (const std::exception& e) {
    throw jsi::JSError(rt, e.what());
  }
}

jsi::Value LlamaCppModel::get(jsi::Runtime& rt, const jsi::PropNameID& name) {
  auto nameStr = name.utf8(rt);

  if (nameStr == "tokenize") {
    return jsi::Function::createFromHostFunction(
      rt, name, 1,
      [this](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* args, size_t count) {
        return this->tokenizeJsi(runtime, args, count);
      });
  }
  else if (nameStr == "detokenize") {
    return jsi::Function::createFromHostFunction(
      rt, name, 1,
      [this](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* args, size_t count) {
        return this->detokenizeJsi(runtime, args, count);
      });
  }
  else if (nameStr == "completion") {
    // Use async completion as the default to provide better UX
    if (jsInvoker_) {
      return jsi::Function::createFromHostFunction(
        rt, name, 2,
        [this](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* args, size_t count) {
          return this->completionAsyncJsi(runtime, args, count);
        });
    } else {
      // Fallback to sync completion if no CallInvoker
      return jsi::Function::createFromHostFunction(
        rt, name, 2,
        [this](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* args, size_t count) {
          return this->completionJsi(runtime, args, count);
        });
    }
  }
  else if (nameStr == "completionSync") {
    return jsi::Function::createFromHostFunction(
      rt, name, 2,
      [this](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* args, size_t count) {
        return this->completionJsi(runtime, args, count);
      });
  }
  else if (nameStr == "stopCompletion") {
    return jsi::Function::createFromHostFunction(
      rt, name, 0,
      [this](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* args, size_t count) {
        return this->stopCompletionJsi(runtime, args, count);
      });
  }
  else if (nameStr == "embedding") {
    return jsi::Function::createFromHostFunction(
      rt, name, 1,
      [this](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* args, size_t count) {
        return this->embeddingJsi(runtime, args, count);
      });
  }
  else if (nameStr == "release") {
    return jsi::Function::createFromHostFunction(
      rt, name, 0,
      [this](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* args, size_t count) {
        return this->releaseJsi(runtime, args, count);
      });
  }
  else if (nameStr == "n_vocab") {
    return jsi::Value(getVocabSize());
  }
  else if (nameStr == "n_ctx") {
    return jsi::Value(getContextSize());
  }
  else if (nameStr == "n_embd") {
    return jsi::Value(getEmbeddingSize());
  }

  return jsi::Value::undefined();
}

void LlamaCppModel::set(jsi::Runtime& rt, const jsi::PropNameID& name, const jsi::Value& value) {
  // Currently we don't support setting properties
  throw jsi::JSError(rt, "Cannot modify llama model properties");
}

std::vector<jsi::PropNameID> LlamaCppModel::getPropertyNames(jsi::Runtime& rt) {
  std::vector<jsi::PropNameID> result;
  result.push_back(jsi::PropNameID::forAscii(rt, "tokenize"));
  result.push_back(jsi::PropNameID::forAscii(rt, "detokenize"));
  result.push_back(jsi::PropNameID::forAscii(rt, "completion"));
  result.push_back(jsi::PropNameID::forAscii(rt, "completionSync"));
  result.push_back(jsi::PropNameID::forAscii(rt, "stopCompletion"));
  result.push_back(jsi::PropNameID::forAscii(rt, "embedding"));
  result.push_back(jsi::PropNameID::forAscii(rt, "release"));
  result.push_back(jsi::PropNameID::forAscii(rt, "n_vocab"));
  result.push_back(jsi::PropNameID::forAscii(rt, "n_ctx"));
  result.push_back(jsi::PropNameID::forAscii(rt, "n_embd"));
  return result;
}

} // namespace facebook::react

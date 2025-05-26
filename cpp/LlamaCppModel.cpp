#include "LlamaCppModel.h"
#include <jsi/jsi.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
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
#include "rn-utils.hpp"
#include "rn-llama.hpp"

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

LlamaCppModel::LlamaCppModel(rn_llama_context* rn_ctx)
    : rn_ctx_(rn_ctx), should_stop_completion_(false), is_predicting_(false) {
    initHelpers();
}

void LlamaCppModel::initHelpers() {
    // No longer need tool handler initialization
}

LlamaCppModel::~LlamaCppModel() {
  // Note: We don't automatically release resources here
  // as the user should call release() explicitly
}

void LlamaCppModel::release() {
  // Cancel any ongoing predictions
  if (is_predicting_) {
    should_stop_completion_ = true;

    // Optionally wait a bit for completion to stop
    int retry = 0;
    while (is_predicting_ && retry < 10) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      retry++;
    }
  }

  // Clean up our resources
  if (rn_ctx_) {
    if (rn_ctx_->ctx) {
      llama_free(rn_ctx_->ctx);
      rn_ctx_->ctx = nullptr;
    }

    if (rn_ctx_->model) {
      llama_model_free(rn_ctx_->model);
      rn_ctx_->model = nullptr;
    }

    // Note: rn_ctx_ itself is owned by the module, so we don't delete it here
    rn_ctx_ = nullptr;
  }
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
  if (value) {
    // Reset the predicting flag when stopping completion
    is_predicting_ = false;
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

  // Extract and parse messages if present (for chat completion)
  if (obj.hasProperty(rt, "messages") && obj.getProperty(rt, "messages").isObject()) {
    auto messagesVal = obj.getProperty(rt, "messages").getObject(rt);
    if (messagesVal.isArray(rt)) {
      json messagesJson = json::array();
      auto messagesArr = messagesVal.getArray(rt);

      // Convert JSI messages to JSON format
      for (size_t i = 0; i < messagesArr.size(rt); i++) {
        auto msgVal = messagesArr.getValueAtIndex(rt, i);
        if (msgVal.isObject()) {
          auto msgObj = msgVal.getObject(rt);

          json msgJson = json::object();
          if (msgObj.hasProperty(rt, "role")) {
            msgJson["role"] = msgObj.getProperty(rt, "role").asString(rt).utf8(rt);
          }

          if (msgObj.hasProperty(rt, "content")) {
            auto contentVal = msgObj.getProperty(rt, "content");
            if (contentVal.isString()) {
              msgJson["content"] = contentVal.asString(rt).utf8(rt);
            } else if (contentVal.isNull()) {
              msgJson["content"] = nullptr;
            }
          }

          if (msgObj.hasProperty(rt, "name")) {
            msgJson["name"] = msgObj.getProperty(rt, "name").asString(rt).utf8(rt);
          }

          // Handle tool_calls if present
          if (msgObj.hasProperty(rt, "tool_calls") && msgObj.getProperty(rt, "tool_calls").isObject()) {
            auto toolCallsVal = msgObj.getProperty(rt, "tool_calls").getObject(rt);
            if (toolCallsVal.isArray(rt)) {
              auto toolCallsArr = toolCallsVal.getArray(rt);
              json toolCallsJson = json::array();

              for (size_t j = 0; j < toolCallsArr.size(rt); j++) {
                auto tcVal = toolCallsArr.getValueAtIndex(rt, j);
                if (tcVal.isObject()) {
                  auto tcObj = tcVal.getObject(rt);
                  json tcJson = json::object();

                  if (tcObj.hasProperty(rt, "id")) {
                    tcJson["id"] = tcObj.getProperty(rt, "id").asString(rt).utf8(rt);
                  }

                  if (tcObj.hasProperty(rt, "type")) {
                    tcJson["type"] = tcObj.getProperty(rt, "type").asString(rt).utf8(rt);
                  }

                  if (tcObj.hasProperty(rt, "function") && tcObj.getProperty(rt, "function").isObject()) {
                    auto fnObj = tcObj.getProperty(rt, "function").getObject(rt);
                    json fnJson = json::object();

                    if (fnObj.hasProperty(rt, "name")) {
                      fnJson["name"] = fnObj.getProperty(rt, "name").asString(rt).utf8(rt);
                    }

                    if (fnObj.hasProperty(rt, "parameters")) {
                      // For parameters, parse it as a JSON object
                      auto paramsVal = fnObj.getProperty(rt, "parameters");
                      if (paramsVal.isObject()) {
                        try {
                          // Convert the JSI object directly to nlohmann::json
                          auto paramsObj = paramsVal.getObject(rt);
                          json fnParams = json::object();

                          // Extract properties directly from the JSI object
                          jsi::Array propNames = paramsObj.getPropertyNames(rt);
                          size_t propCount = propNames.size(rt);
                          for (size_t i = 0; i < propCount; i++) {
                            jsi::String propName = propNames.getValueAtIndex(rt, i).asString(rt);
                            std::string key = propName.utf8(rt);
                            auto value = paramsObj.getProperty(rt, propName);

                            if (value.isString()) {
                              fnParams[key] = value.asString(rt).utf8(rt);
                            } else if (value.isNumber()) {
                              fnParams[key] = value.asNumber();
                            } else if (value.isBool()) {
                              fnParams[key] = value.getBool();
                            } else if (value.isNull()) {
                              fnParams[key] = nullptr;
                            } else if (value.isObject()) {
                              if (value.getObject(rt).isArray(rt)) {
                                fnParams[key] = json::array();
                              } else {
                                fnParams[key] = json::object();
                              }
                            }
                          }

                          fnJson["parameters"] = fnParams;
                        } catch (const std::exception&) {
                          fnJson["parameters"] = json::object();
                        }
                      }
                    }

                    tcJson["function"] = fnJson;
                  }

                  toolCallsJson.push_back(tcJson);
                }
              }

              msgJson["tool_calls"] = toolCallsJson;
            }
          }

          // Handle tool_call_id if present
          if (msgObj.hasProperty(rt, "tool_call_id")) {
            msgJson["tool_call_id"] = msgObj.getProperty(rt, "tool_call_id").asString(rt).utf8(rt);
          }

          messagesJson.push_back(msgJson);
        }
      }

      options.messages = messagesJson;
    }
  }

  // Extract and parse tools if present
  if (obj.hasProperty(rt, "tools") && obj.getProperty(rt, "tools").isObject()) {
    auto toolsVal = obj.getProperty(rt, "tools").getObject(rt);
    if (toolsVal.isArray(rt)) {
      auto toolsArr = toolsVal.getArray(rt);
      json toolsJson = json::array();

      for (size_t i = 0; i < toolsArr.size(rt); i++) {
        auto toolVal = toolsArr.getValueAtIndex(rt, i);
        if (toolVal.isObject()) {
          auto toolObj = toolVal.getObject(rt);
          json toolJson = json::object();

          if (toolObj.hasProperty(rt, "type")) {
            toolJson["type"] = toolObj.getProperty(rt, "type").asString(rt).utf8(rt);
          }

          if (toolObj.hasProperty(rt, "function") && toolObj.getProperty(rt, "function").isObject()) {
            auto fnObj = toolObj.getProperty(rt, "function").getObject(rt);
            json fnJson = json::object();

            if (fnObj.hasProperty(rt, "name")) {
              fnJson["name"] = fnObj.getProperty(rt, "name").asString(rt).utf8(rt);
            }

            if (fnObj.hasProperty(rt, "description")) {
              fnJson["description"] = fnObj.getProperty(rt, "description").asString(rt).utf8(rt);
            }

            if (fnObj.hasProperty(rt, "parameters")) {
              // For parameters, parse it as a JSON object
              auto paramsVal = fnObj.getProperty(rt, "parameters");
              if (paramsVal.isObject()) {
                try {
                  // Convert the JSI object directly to nlohmann::json
                  auto paramsObj = paramsVal.getObject(rt);
                  json fnParams = json::object();

                  // Extract properties directly from the JSI object
                  jsi::Array propNames = paramsObj.getPropertyNames(rt);
                  size_t propCount = propNames.size(rt);
                  for (size_t i = 0; i < propCount; i++) {
                    jsi::String propName = propNames.getValueAtIndex(rt, i).asString(rt);
                    std::string key = propName.utf8(rt);
                    auto value = paramsObj.getProperty(rt, propName);

                    if (value.isString()) {
                      fnParams[key] = value.asString(rt).utf8(rt);
                    } else if (value.isNumber()) {
                      fnParams[key] = value.asNumber();
                    } else if (value.isBool()) {
                      fnParams[key] = value.getBool();
                    } else if (value.isNull()) {
                      fnParams[key] = nullptr;
                    } else if (value.isObject()) {
                      // For nested objects, we use a simplified approach
                      if (value.getObject(rt).isArray(rt)) {
                        fnParams[key] = json::array();
                      } else {
                        fnParams[key] = json::object();
                      }
                    }
                  }

                  fnJson["parameters"] = fnParams;
                } catch (const std::exception&) {
                  fnJson["parameters"] = json::object();
                }
              }
            }

            toolJson["function"] = fnJson;
          }

          toolsJson.push_back(toolJson);
        }
      }

      options.tools = toolsJson;
    }
  }

  // Extract tool_choice if present
  if (obj.hasProperty(rt, "tool_choice") && !obj.getProperty(rt, "tool_choice").isUndefined()) {
    auto toolChoiceVal = obj.getProperty(rt, "tool_choice");
    if (toolChoiceVal.isString()) {
      options.tool_choice = toolChoiceVal.asString(rt).utf8(rt);
    } else if (toolChoiceVal.isObject()) {
      // Handle the case where tool_choice is an object with "type": "function", etc.
      options.tool_choice = "required"; // Default to "required" if a specific tool is selected
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
  llama_kv_self_clear(rn_ctx_->ctx);

  // Store original sampling parameters to restore later
  float orig_temp = rn_ctx_->params.sampling.temp;
  float orig_top_p = rn_ctx_->params.sampling.top_p;
  float orig_top_k = rn_ctx_->params.sampling.top_k;
  float orig_min_p = rn_ctx_->params.sampling.min_p;
  int orig_n_predict = rn_ctx_->params.n_predict;

  // Set sampling parameters from options
  rn_ctx_->params.sampling.temp = options.temperature;
  rn_ctx_->params.sampling.top_p = options.top_p;
  rn_ctx_->params.sampling.top_k = options.top_k;
  rn_ctx_->params.sampling.min_p = options.min_p;
  rn_ctx_->params.n_predict = options.n_predict;

  // Check for a partial callback
  auto callback_adapter = [&partialCallback, runtime](const std::string& token, bool is_done) -> bool {
    if (partialCallback && runtime && !is_done) {
      partialCallback(*runtime, token.c_str());
    }
    return true;
  };

  // Run the completion based on whether we have messages or prompt
  CompletionResult result;

  try {
    // Set the predicting flag to prevent interruption
    is_predicting_ = true;
    should_stop_completion_ = false;

    if (!options.messages.empty()) {
      // Chat completion (with messages)
      result = run_chat_completion(rn_ctx_, options, callback_adapter);
    } else {
      // Regular completion (with prompt)
      result = run_completion(rn_ctx_, options, callback_adapter);
    }

    // Reset the predicting flag
    is_predicting_ = false;
  } catch (const std::exception& e) {
    is_predicting_ = false;
    result.success = false;
    result.error_msg = std::string("Completion failed: ") + e.what();
    result.error_type = RN_ERROR_INFERENCE;
  }

  // Restore original parameters
  rn_ctx_->params.sampling.temp = orig_temp;
  rn_ctx_->params.sampling.top_p = orig_top_p;
  rn_ctx_->params.sampling.top_k = orig_top_k;
  rn_ctx_->params.sampling.min_p = orig_min_p;
  rn_ctx_->params.n_predict = orig_n_predict;

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
  jsResult.setProperty(rt, "timings", jsi::Object(rt));
  jsResult.setProperty(rt, "success", jsi::Value(result.success));
  jsResult.setProperty(rt, "promptTokens", jsi::Value(result.n_prompt_tokens));
  jsResult.setProperty(rt, "completionTokens", jsi::Value(result.n_predicted_tokens));

  if (!result.success) {
    jsResult.setProperty(rt, "error", jsi::String::createFromUtf8(rt, result.error_msg));
    jsResult.setProperty(rt, "errorType", jsi::Value((int)result.error_type));
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
    for (const auto& item : j.items()) {
      object.setProperty(rt, item.key().c_str(), jsonToJsi(rt, item.value()));
    }
    return object;
  }

  // Default case (shouldn't happen)
  return jsi::Value::undefined();
}

// JSI method for completions
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
        tokenObj.setProperty(rt, "id", jsi::Value((int)tokens[i]));

        // Get the text piece for this token
        std::string piece = common_token_to_piece(rn_ctx_->vocab, tokens[i]);
        tokenObj.setProperty(rt, "text", jsi::String::createFromUtf8(rt, piece));

        tokensArray.setValueAtIndex(rt, i, tokenObj);
      } else {
        // Just add the token ID
        tokensArray.setValueAtIndex(rt, i, jsi::Value((int)tokens[i]));
      }
    }

    result.setProperty(rt, "tokens", tokensArray);
    result.setProperty(rt, "count", jsi::Value((int)tokens.size()));

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
    int token_count = tokensArr.size(rt);

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
    llama_kv_self_clear(rn_ctx_->ctx);

    // Enable embedding mode
    llama_set_embeddings(rn_ctx_->ctx, true);

    // Evaluate tokens one by one
    for (int i = 0; i < (int)tokens.size(); i++) {
      llama_token token = tokens[i];
      llama_batch batch = {
        /* n_tokens    */ 1,
        /* token       */ &token,
        /* embd        */ nullptr,
        /* pos         */ &i,
        /* n_seq_id    */ nullptr,
        /* seq_id      */ nullptr,
        /* logits      */ nullptr
      };

      if (llama_decode(rn_ctx_->ctx, batch) != 0) {
        throw std::runtime_error("Failed to decode token for embedding");
      }
    }

    // Get embedding size from the model
    const int n_embd = llama_model_n_embd(rn_ctx_->model);
    if (n_embd <= 0) {
      throw std::runtime_error("Invalid embedding dimension");
    }

    // For OpenAI compatibility, default to mean pooling
    enum llama_pooling_type pooling_type = LLAMA_POOLING_TYPE_MEAN;
    if (options.hasProperty(rt, "pooling") && options.getProperty(rt, "pooling").isString()) {
      std::string pooling = options.getProperty(rt, "pooling").getString(rt).utf8(rt);
      if (pooling == "last") {
        pooling_type = LLAMA_POOLING_TYPE_LAST;
      } else if (pooling == "cls" || pooling == "first") {
        pooling_type = LLAMA_POOLING_TYPE_CLS;
      }
    }

    // Get the embeddings
    std::vector<float> embedding_vec(n_embd);
    const float* embd = llama_get_embeddings(rn_ctx_->ctx);

    if (!embd) {
      throw std::runtime_error("Failed to extract embeddings");
    }

    // Copy embeddings to our vector
    std::copy(embd, embd + n_embd, embedding_vec.begin());

    // Normalize embedding
    float norm = 0.0f;
    for (int i = 0; i < n_embd; ++i) {
      norm += embedding_vec[i] * embedding_vec[i];
    }
    norm = std::sqrt(norm);

    if (norm > 0) {
      for (int i = 0; i < n_embd; ++i) {
        embedding_vec[i] /= norm;
      }
    }

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
      jsi::Array embeddingArray(rt, n_embd);
      for (int i = 0; i < n_embd; i++) {
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
    usage.setProperty(rt, "prompt_tokens", jsi::Value((int)tokens.size()));
    usage.setProperty(rt, "total_tokens", jsi::Value((int)tokens.size()));

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
    return jsi::Function::createFromHostFunction(
      rt, name, 2,
      [this](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* args, size_t count) {
        return this->completionJsi(runtime, args, count);
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
  result.push_back(jsi::PropNameID::forAscii(rt, "embedding"));
  result.push_back(jsi::PropNameID::forAscii(rt, "release"));
  result.push_back(jsi::PropNameID::forAscii(rt, "n_vocab"));
  result.push_back(jsi::PropNameID::forAscii(rt, "n_ctx"));
  result.push_back(jsi::PropNameID::forAscii(rt, "n_embd"));
  return result;
}

} // namespace facebook::react

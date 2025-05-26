#pragma once

#include "common.h"
#include "llama.h"
#include "chat.h"
#include "chat-template.hpp"
#include "json-schema-to-grammar.h"
#include "rn-utils.hpp"

#include <functional>
#include <mutex>
#include <string>
#include <vector>

// Forward declarations
struct llama_model;
struct llama_context;
struct llama_vocab;

namespace facebook::react {

// Extend common_params with additional fields needed by our implementation
struct rn_common_params : common_params {
    bool debug = false;
    common_chat_format chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_NONE;
    bool use_jinja = false;
};

// Main context structure for React Native integration
struct rn_llama_context {
    // Model parameters - use our extended params structure
    rn_common_params params;

    // Core llama.cpp components
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;

    // Extensions
    std::vector<common_adapter_lora_info> lora_adapters;
    common_chat_templates_ptr chat_templates;

    // State
    bool model_loaded = false;
    std::mutex mutex;
};

// Core completion functions
CompletionResult run_completion(
    rn_llama_context* rn_ctx,
    const CompletionOptions& options,
    std::function<bool(const std::string&, bool)> callback);

CompletionResult run_chat_completion(
    rn_llama_context* rn_ctx,
    const CompletionOptions& options,
    std::function<bool(const std::string&, bool)> callback);

} // namespace facebook::react
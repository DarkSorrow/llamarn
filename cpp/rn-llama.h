#pragma once

// Suppress unused function warnings from llama.cpp headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#include "common.h"
#include "llama.h"
#include "chat.h"
#include "json-schema-to-grammar.h"
#pragma GCC diagnostic pop

#include "rn-utils.h"
#include "rn-multimodal.h"

#include <atomic>
#include <functional>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace facebook::react {

// Extend common_params with additional fields needed by our implementation
struct rn_common_params : common_params {
    bool debug = false;
    common_chat_format chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_NONE;
    bool use_jinja = false;

    // Reasoning budget: -1 = unlimited thinking, 0 = disabled, >0 = token limit.
    // Stored here because upstream moved this out of common_params into common_params_sampling
    // (as reasoning_budget_tokens). We keep it here for JS-layer parsing and apply it to
    // sampling.reasoning_budget_tokens during do_init_llama.
    int reasoning_budget = 0;

    // Cooperative prompt-ingestion loop settings (set at initLlama time).
    // chunk_size: tokens per llama_decode call during prompt encoding (distinct from n_batch).
    // is_cpu_only: when true, sleep 2ms after each chunk.
    // prompt_chunk_gap_ms: minimum inter-chunk gap on GPU path.
    int  chunk_size          = 128;
    bool is_cpu_only         = false;
    int  prompt_chunk_gap_ms = 5;
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

    // Multimodal projection context (non-null when mmproj loaded)
    mtmd_context* mtmd_ctx = nullptr;
    bool multimodal_loaded = false;

    // Bitmask of ModelCapability flags declared at initLlama time
    uint32_t declared_capabilities = 0;

    // Abort flag: set to true to make llama_decode exit on the next graph eval.
    // Read by the abort_callback registered via llama_set_abort_callback().
    std::atomic<bool> abort_generation{false};

    // KV cache prefix reuse state (guarded by mutex).
    // Stores the token boundary after each message so the next call can skip re-encoding
    // messages whose IDs haven't changed. IDs are supplied by the caller per message.
    struct kv_msg_entry {
        std::string id;         // caller-supplied message ID
        int32_t     token_end;  // exclusive token index after this message in the KV cache
    };
    std::vector<kv_msg_entry> kv_messages;
    bool                      kv_has_messages = false;
    std::string               kv_render_identity;

    // Completion cache: stores the resolved sampling params and grammar state from the last
    // run_chat_completion call. When prompt_id and config_id both match on the next call,
    // the expensive common_chat_templates_apply step is skipped entirely.
    struct completion_cache_entry {
        std::string                          prompt_id;
        std::string                          config_id;
        std::string                          grammar;
        bool                                 grammar_lazy = false;
        std::vector<common_grammar_trigger>  grammar_triggers;
        std::set<llama_token>                preserved_tokens;
        std::vector<std::string>             additional_stops;
    };
    std::optional<completion_cache_entry> completion_cache;

    // Reused decode batches to avoid per-request alloc/free churn.
    llama_batch gen_batch = {};
    llama_batch ingest_batch = {};
    bool batches_initialized = false;
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
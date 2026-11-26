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
#include <chrono>
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

    // Reasoning budget from JS. 0 renders the template with enable_thinking=false; otherwise it
    // becomes sampling.reasoning_budget_tokens on chat calls (-1 = no budget sampler, >0 = limit).
    int reasoning_budget = 0;

    // From initLlama. Sets common_chat_templates_inputs::parallel_tool_calls on chat calls.
    bool parallel_tool_calls = false;

    // Cooperative prompt-ingestion loop settings (set at initLlama time).
    // Added after app crashes (GPU fence timeouts) on real Android devices; keep as is until
    // it can be re-validated on hardware. See docs/CONFIGURATION.md "Prompt pacing".
    // chunk_size: tokens per llama_process call during prompt encoding, clamped to [8, 512]
    //   and to n_batch. Smaller chunks mean shorter GPU submissions.
    // is_cpu_only: caller-declared. When true, a fixed 2 ms sleep follows every chunk and the
    //   GPU gap below is not used. It does not change n_gpu_layers.
    // prompt_chunk_gap_ms: GPU path only. A floor measured from the start of each chunk, plus
    //   1 ms per 2048 prompt tokens: the loop sleeps only for the part of the floor the chunk
    //   did not already take. A chunk that runs longer than the floor gets no sleep.
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

    // Thermal management: JS thread writes with memory_order_release (non-blocking);
    // inference loop reads with memory_order_acq_rel at the top of each token iteration
    // and calls llama_set_n_threads before the next llama_decode. -1 = no change pending.
    std::atomic<int> requested_n_threads{-1};

    // KV cache prefix reuse state (guarded by mutex).
    // The tokens in seq 0, in position order: the prompt and every decoded token. The next chat
    // call keeps the longest common prefix with its own prompt tokens and encodes the rest, as
    // llama-server's cache_prompt does. Empty when the cache holds nothing this list can describe
    // (cleared, media embeddings, an embedding call): the next call then encodes from 0.
    std::vector<llama_token> kv_tokens;
    // Render settings of the call that filled kv_tokens. A change clears the cache (reported as
    // render_identity), even where the tokens would still match.
    std::string              kv_render_identity;

    // prompt_id of the last chat call. A change from a previous non-empty value clears KV.
    // An empty prompt_id does not. Tools are compared in kv_render_identity, not here.
    std::string last_prompt_id;

    // Result of common_context_can_seq_rm, taken once when the context is created.
    // The server uses the same check. PART and RS can drop a suffix; FULL and NO cannot.
    common_context_seq_rm_type kv_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_PART;

    // Test seam. Null in production. When set, prefix eviction calls this instead of
    // llama_memory_seq_rm so a test can force the full-clear path.
    bool (*kv_seq_rm)(void * data, llama_pos p0, llama_pos p1) = nullptr;
    void * kv_seq_rm_data = nullptr;

    // Reused decode batches (one per context) to avoid per-request alloc/free churn.
    // Must be released (rn_free_batches) before the llama_context is freed.
    std::unique_ptr<common_batch> gen_batch;
    std::unique_ptr<common_batch> ingest_batch;
};

void rn_init_batches(rn_llama_context* rn_ctx);
void rn_free_batches(rn_llama_context* rn_ctx);

// Builds the chat session, falling back to force_pure_content and then to the non-jinja
// template path when the model's jinja template throws. On return, `inputs` holds the
// settings that succeeded so later renders (KV boundaries) match the decoded prompt.
common_chat_session rn_make_chat_session(const rn_llama_context* rn_ctx, common_chat_templates_inputs& inputs);

// Partial chat parse for streaming deltas, as the server does (task_result_state::update_chat_msg
// in tools/server/server-task.cpp). Each feed() re-parses the whole reply, so parses are spaced
// by `interval` instead of per token; deltas are exact either way because they are diffs between
// parsed states. The final call uses finish(), so the summed deltas equal the final message.
// On a parse or diff error the deltas stop and the final result stays authoritative.
// `session` must be a copy of the call's chat session: feed() mutates it.
struct rn_stream_parser {
    common_chat_session *     session = nullptr;
    std::chrono::milliseconds interval{30};
    common_chat_msg           prev;
    size_t                    n_fed    = 0;
    bool                      disabled = false;
    std::chrono::steady_clock::time_point last_parse{};

    // Feeds generated[n_fed, upto) and appends the content / reasoning deltas to chunk.
    void update(const common_chat_input & generated, size_t upto, bool final, rn_stream_chunk & chunk);
};

// Core completion functions
CompletionResult run_completion(
    rn_llama_context* rn_ctx,
    const CompletionOptions& options,
    rn_stream_callback callback);

CompletionResult run_chat_completion(
    rn_llama_context* rn_ctx,
    const CompletionOptions& options,
    rn_stream_callback callback);

} // namespace facebook::react
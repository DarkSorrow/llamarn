#include "rn-llama.h"
// Suppress unused function warnings from llama.cpp headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#include "common.h"
#include "chat.h"
#include "llama.h"
#include "sampling.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#pragma GCC diagnostic pop
#include "rn-utils.h"
#include "rn-multimodal.h"

#include <string>
#include <vector>
#include <thread>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <random>
#include <algorithm>
#include <chrono>
#include <memory>

namespace facebook::react {

// Struct to track prediction completion state
struct completion_state {
    struct sampler_deleter {
        void operator()(common_sampler* sampler) const {
            if (sampler) {
                common_sampler_free(sampler);
            }
        }
    };

    rn_llama_context* rn_ctx = nullptr;

    bool has_next_token = true;
    bool has_new_line = false;
    bool truncated = false;

    int n_past = 0;
    int n_ctx = 0;
    int n_predict = 0;
    int n_decoded = 0;
    int n_remaining = 0;

    size_t n_sent_text = 0;

    std::string generated_text;
    std::string stopping_word;
    bool stop_found = false;
    bool stopped_by_limit = false;

    std::vector<llama_token> prompt_tokens;
    std::vector<llama_token> generated_tokens;

    std::unique_ptr<common_sampler, sampler_deleter> sampler;
    std::vector<std::string> antiprompt;

    // Chat format and tools info
    common_chat_format chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    common_chat_tool_choice tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;

};

// Helper function to check for stopping criteria
static bool check_stop_conditions(
    completion_state& state,
    const std::vector<std::string>& stop_words,
    const std::string& token_text,
    bool ignore_eos) {

    if (state.n_predict >= 0 && state.n_remaining <= 0) {
        state.has_next_token = false;
        state.stopped_by_limit = true;
        return true;
    }

    // Check for stopping strings
    size_t stop_pos = std::string::npos;

    for (const std::string & word : stop_words) {
        size_t pos = state.generated_text.find(word, state.n_sent_text > 0 ? state.n_sent_text - 1 : 0);
        if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
            state.stopping_word = word;
            state.has_next_token = false;
            stop_pos = pos;
        }
    }

    if (stop_pos != std::string::npos) {
        state.generated_text.erase(
            state.generated_text.begin() + stop_pos,
            state.generated_text.end()
        );
        if (state.n_sent_text > state.generated_text.size()) {
            state.n_sent_text = state.generated_text.size();
        }
        state.stop_found = true;
        return true;
    }

    // Check for partial stop strings at the end
    for (const std::string & word : stop_words) {
        if (size_t partial_pos = find_partial_stop_string(word, state.generated_text);
            partial_pos != std::string::npos) {
            return false; // Don't send token yet, wait for full stop word
        }
    }

    // Check if context is full
    if (state.n_past >= state.n_ctx) {
        state.truncated = true;
        state.has_next_token = false;
        state.stopped_by_limit = true;
        return true;
    }

    // Check for newline condition
    if (token_text.find('\n') != std::string::npos) {
        state.has_new_line = true;
    }

    return false;
}

static std::string build_kv_render_identity(
    const rn_llama_context* rn_ctx,
    const CompletionOptions& options) {
    std::string identity;
    identity.reserve(512);
    identity += "use_jinja=" + std::to_string(rn_ctx->params.use_jinja ? 1 : 0);
    identity += "|reasoning_format=" + std::to_string(static_cast<int>(rn_ctx->params.reasoning_format));
    identity += "|chat_template=" + rn_ctx->params.chat_template;
    identity += "|tool_choice=" + options.tool_choice;
    identity += "|grammar=" + options.grammar;
    identity += "|tools=" + (options.tools.is_null() ? "null" : options.tools.dump());
    return identity;
}

CompletionResult run_completion(
    rn_llama_context* rn_ctx,
    const CompletionOptions& options,
    std::function<bool(const std::string&, bool)> callback) {

    CompletionResult result;
    completion_state state;

    if (!rn_ctx || !rn_ctx->model || !rn_ctx->ctx) {
        result.success = false;
        result.error_msg = "Model not initialized";
        result.error_type = RN_ERROR_MODEL_LOAD;
        return result;
    }

    if (!rn_ctx->batches_initialized) {
        result.success = false;
        result.error_msg = "Decode batches are not initialized";
        result.error_type = RN_ERROR_CONTEXT;
        return result;
    }

    try {
        // Initialize state with context values
        state.rn_ctx = rn_ctx;
        state.chat_format = rn_ctx->params.chat_format;

        const auto& params = rn_ctx->params;

        // Create a copy of sampling parameters and apply per-request overrides.
        // All mutations happen on this LOCAL copy — rn_ctx_->params.sampling is never touched.
        common_params_sampling sampling_params = params.sampling;

        // Apply per-request sampling overrides from CompletionOptions.
        sampling_params.temp            = options.temperature;
        sampling_params.top_p           = options.top_p;
        sampling_params.top_k           = options.top_k;
        sampling_params.min_p           = options.min_p;
        sampling_params.penalty_present = options.presence_penalty;
        sampling_params.penalty_repeat  = options.repeat_penalty;
        sampling_params.penalty_last_n  = options.repeat_last_n;
        sampling_params.penalty_freq    = options.frequency_penalty;
        // Map JS sentinel -1 ("use default/random") to the model's configured seed.
        // Only override if the caller supplied an explicit non-negative seed.
        if (options.seed >= 0) {
            sampling_params.seed = static_cast<uint32_t>(options.seed);
        }

        // Merge preserved token IDs from the chat template autoparser into sampling params.
        // These are special single-token strings (e.g. "<think>", "<|eot_id|>") that the
        // tokenizer must not split — required for lazy grammar triggers to work correctly.
        for (auto tok : options.preserved_tokens) {
            sampling_params.preserved_tokens.insert(tok);
        }

        if (!options.grammar.empty()) {
            sampling_params.grammar = common_grammar(COMMON_GRAMMAR_TYPE_USER, options.grammar);
            // Force grammar_lazy to false whenever tools are present to ensure strict JSON format enforcement
            if (!options.tools.empty()) {
                sampling_params.grammar_lazy = false;
            } else {
                sampling_params.grammar_lazy = options.grammar_lazy;
            }
            // Pass grammar_triggers if any were provided by chat_params and passed via options
            if (!options.grammar_triggers.empty()) {
                sampling_params.grammar_triggers = options.grammar_triggers;
            }
        }

        // Parse tool_choice
        if (options.tool_choice == "auto") {
            state.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
        } else if (options.tool_choice == "none") {
            state.tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;
        } else if (options.tool_choice == "required") {
            state.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        }
        // Initialize the sampler with the updated sampling parameters
        state.sampler.reset(common_sampler_init(rn_ctx->model, sampling_params));
        if (!state.sampler) {
            result.success = false;
            result.error_msg = "Failed to initialize sampler";
            result.error_type = RN_ERROR_INFERENCE;
            return result;
        }

        // Stop words
        state.antiprompt = options.stop;

        if (options.mtmd_encoded_n_past >= 0) {
            // Multimodal fast path: images + text were already encoded by run_chat_completion
            // via mtmd_helper_eval_chunks. Logits are ready; skip tokenize and KV encode.
            state.n_past      = options.mtmd_encoded_n_past;
            state.n_ctx       = llama_n_ctx(rn_ctx->ctx);
            if (options.n_predict > 0) {
                state.n_predict = options.n_predict;
            } else if (params.n_predict > 0) {
                state.n_predict = params.n_predict;
            } else {
                state.n_predict = -1; // unlimited — EOS or stop string terminates
            }
            state.n_remaining = state.n_predict;
            result.n_prompt_tokens = 0;
        } else {

        // Tokenize the prompt directly — always a string in this path
        if (options.prompt.empty()) {
            result.success = false;
            result.error_msg = "No prompt provided";
            result.error_type = RN_ERROR_INVALID_PARAM;
            return result;
        }
        state.prompt_tokens = common_tokenize(rn_ctx->vocab, options.prompt, true, true);
        if (state.prompt_tokens.empty()) {
            result.success = false;
            result.error_msg = "Empty prompt";
            result.error_type = RN_ERROR_INVALID_PARAM;
            return result;
        }

        // KV cache: run_chat_completion already evicted stale entries and passes the
        // trusted common prefix length via kv_hint_pos. We just apply it here.
        // Direct callers (no kv_hint_pos) always start from position 0 with a full clear.
        if (options.kv_hint_pos >= 0) {
            size_t kv_common_len = static_cast<size_t>(options.kv_hint_pos);
            // Safety: need at least 1 new token to encode for valid logits.
            if (kv_common_len >= state.prompt_tokens.size()) {
                kv_common_len = 0;
                llama_memory_clear(llama_get_memory(rn_ctx->ctx), true);
            }
            state.n_past = static_cast<int>(kv_common_len);
        } else {
            llama_memory_clear(llama_get_memory(rn_ctx->ctx), true);
            state.n_past = 0;
        }

        // Configure state
        state.n_ctx = llama_n_ctx(rn_ctx->ctx);
        if (options.n_predict > 0) {
            state.n_predict = options.n_predict;
        } else if (params.n_predict > 0) {
            state.n_predict = params.n_predict;
        } else {
            state.n_predict = -1; // unlimited — EOS or stop string terminates
        }
        state.n_remaining = state.n_predict;

        // Guard: prompt must fit in the context window, otherwise llama_decode will ggml_abort.
        if (static_cast<int>(state.prompt_tokens.size()) >= state.n_ctx) {
            result.success = false;
            result.error_msg = "Prompt too long: " + std::to_string(state.prompt_tokens.size())
                + " tokens exceeds context size " + std::to_string(state.n_ctx);
            result.error_type = RN_ERROR_INVALID_PARAM;
            return result;
        }

        // Encode prompt tokens into the KV cache using the cooperative ingestion loop.
        // chunk_size (distinct from n_batch) is the decode granularity; after each chunk
        // we yield to let the OS/UI thread run, preventing display fence timeouts (Android)
        // and UI starvation (CPU-only devices).
        const int ingest_chunk = std::clamp(rn_ctx->params.chunk_size, 8, 512);

        if (state.n_past > 0) {
            // Prefix reuse: only encode prompt_tokens[n_past:] at positions [n_past, n_past+1, ...]
            const int n_new = static_cast<int>(state.prompt_tokens.size()) - state.n_past;
            if (n_new > 0) {
                llama_batch& ingest_batch = rn_ctx->ingest_batch;
                for (int i = 0; i < n_new; ) {
                    if (rn_ctx->abort_generation.load(std::memory_order_relaxed)) {
                        result.success = false;
                        result.error_msg = "Generation aborted";
                        result.error_type = RN_ERROR_INFERENCE;
                        return result;
                    }
                    common_batch_clear(ingest_batch);
                    int chunk = std::min(ingest_chunk, n_new - i);
                    bool last_chunk = (i + chunk >= n_new);
                    for (int j = 0; j < chunk; j++) {
                        common_batch_add(ingest_batch,
                            state.prompt_tokens[state.n_past + i + j],
                            state.n_past + i + j,
                            {0},
                            last_chunk && (j == chunk - 1));
                    }
                    auto t0 = std::chrono::steady_clock::now();
                    if (llama_decode(rn_ctx->ctx, ingest_batch) != 0) {
                        result.success = false;
                        result.error_msg = "Failed to process prompt";
                        result.error_type = RN_ERROR_INFERENCE;
                        return result;
                    }
                    const auto chunk_elapsed = std::chrono::steady_clock::now() - t0;
                    if (rn_ctx->params.is_cpu_only) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    } else {
                        int chunk_gap_ms = std::max(0, rn_ctx->params.prompt_chunk_gap_ms);
                        if (state.prompt_tokens.size() > 2048) {
                            chunk_gap_ms += static_cast<int>(state.prompt_tokens.size() / 2048);
                        }
                        const auto min_chunk_gap = std::chrono::milliseconds(chunk_gap_ms);
                        if (chunk_elapsed < min_chunk_gap) {
                            std::this_thread::sleep_for(min_chunk_gap - chunk_elapsed);
                        }
                    }
                    i += chunk;
                }
                state.n_past = static_cast<int>(state.prompt_tokens.size());
            }
        } else {
            // Standard path: encode all tokens from position 0.
            const int n_total = static_cast<int>(state.prompt_tokens.size());
            llama_batch& ingest_batch = rn_ctx->ingest_batch;
            for (int i = 0; i < n_total; ) {
                if (rn_ctx->abort_generation.load(std::memory_order_relaxed)) {
                    result.success = false;
                    result.error_msg = "Generation aborted";
                    result.error_type = RN_ERROR_INFERENCE;
                    return result;
                }
                common_batch_clear(ingest_batch);
                int chunk = std::min(ingest_chunk, n_total - i);
                bool last_chunk = (i + chunk >= n_total);
                for (int j = 0; j < chunk; j++) {
                    common_batch_add(ingest_batch, state.prompt_tokens[i + j], i + j,
                                     {0}, last_chunk && (j == chunk - 1));
                }
                auto t0 = std::chrono::steady_clock::now();
                if (llama_decode(rn_ctx->ctx, ingest_batch) != 0) {
                    result.success = false;
                    result.error_msg = "Failed to process prompt";
                    result.error_type = RN_ERROR_INFERENCE;
                    return result;
                }
                const auto chunk_elapsed = std::chrono::steady_clock::now() - t0;
                if (rn_ctx->params.is_cpu_only) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                } else {
                    int chunk_gap_ms = std::max(0, rn_ctx->params.prompt_chunk_gap_ms);
                    if (state.prompt_tokens.size() > 2048) {
                        chunk_gap_ms += static_cast<int>(state.prompt_tokens.size() / 2048);
                    }
                    const auto min_chunk_gap = std::chrono::milliseconds(chunk_gap_ms);
                    if (chunk_elapsed < min_chunk_gap) {
                        std::this_thread::sleep_for(min_chunk_gap - chunk_elapsed);
                    }
                }
                i += chunk;
            }
            state.n_past = n_total;
        }

        // Accept all prompt tokens into the sampler (needed for repetition/presence penalty).
        // For non-lazy grammars we skip this and re-init below to keep a clean grammar state.
        for (auto tok : state.prompt_tokens) {
            if (common_grammar_value(sampling_params.grammar).empty() || sampling_params.grammar_lazy) {
                common_sampler_accept(state.sampler.get(), tok, true);
            }
        }

        result.n_prompt_tokens = state.prompt_tokens.size();

        // For non-lazy grammars, reset the sampler so grammar state is clean for generation.
        // Use common_sampler_reset (not free+init) to preserve the repetition penalty window
        // that was populated by the prompt token acceptance loop above. Freeing and re-initializing
        // would create a fresh sampler with an empty penalty_last_n, causing the model to repeat
        // prompt content indefinitely on subsequent turns (iOS infinite repetition bug).
        if (!common_grammar_value(sampling_params.grammar).empty() && !sampling_params.grammar_lazy) {
            common_sampler_reset(state.sampler.get());
        }

        } // end: normal tokenize + encode path (mtmd_encoded_n_past < 0)

        llama_batch& gen_batch = rn_ctx->gen_batch;

        while (state.has_next_token && (state.n_predict < 0 || state.n_remaining > 0)) {
            // Sample the next token
            llama_token token_id = common_sampler_sample(state.sampler.get(), rn_ctx->ctx, -1);

            // Extract the token text
            std::string token_text = common_token_to_piece(rn_ctx->vocab, token_id);

            // Add to generated text
            state.generated_text += token_text;
            state.generated_tokens.push_back(token_id);

            // Update state
            state.n_decoded++;
            state.n_remaining--;

            // Decode the generated token to prepare logits for the next sample.
            // IMPORTANT: decode BEFORE accept. The sampler's internal state must reflect
            // the actual KV cache position. Accepting before decode puts the sampler one
            // step ahead of the KV cache, causing logit/sampler mismatch on Android GPU
            // backends (Vulkan/Adreno/Mali) and producing garbled output.
            // Correct order matches server-context.cpp and ai_chat.cpp: sample → decode → accept → n_past++
            common_batch_clear(gen_batch);
            common_batch_add(gen_batch, token_id, state.n_past, {0}, true);
            auto decode_start = std::chrono::steady_clock::now();
            if (llama_decode(rn_ctx->ctx, gen_batch) != 0) {
                result.success = false;
                result.error_msg = "Failed to decode generated token";
                result.error_type = RN_ERROR_INFERENCE;
                return result;
            }
            // Deterministic thermal pacing: enforce a minimum inter-token interval.
            // yield() is a no-op on Android's CFS scheduler; heuristic decode-time sleeps
            // don't bound throughput on fast GPUs. Use sleep_for with a hard rate cap instead.
            // Do NOT add negative sleep correction — sleep_for precision on mobile is ~±5ms,
            // acceptable for 30 tok/s target.
            if (options.token_rate_cap > 0) {
                const auto target_interval = std::chrono::microseconds(1000000 / options.token_rate_cap);
                const auto elapsed = std::chrono::steady_clock::now() - decode_start;
                if (elapsed < target_interval) {
                    std::this_thread::sleep_for(target_interval - elapsed);
                }
            }
            // If token_rate_cap == 0, no sleep is applied (caller asserts device can handle it).

            state.n_past++;

            // Accept the token into the sampler AFTER decode and n_past increment.
            // token_id is a plain int32 captured before llama_decode — safe to pass here.
            common_sampler_accept(state.sampler.get(), token_id, true);

            // Check EOS FIRST — before streaming — so the EOS token text is never sent to JS
            // and all end-of-generation tokens are handled (not just the single llama_vocab_eos()).
            // Also erase the EOG token text from generated_text so it does not appear in
            // result.content or any streamed batch: EOG tokens are control tokens, not content.
            if (!options.ignore_eos && llama_vocab_is_eog(rn_ctx->vocab, token_id)) {
                if (state.generated_text.size() >= token_text.size()) {
                    state.generated_text.erase(state.generated_text.size() - token_text.size());
                } else {
                    state.generated_text.clear();
                }
                if (state.n_sent_text > state.generated_text.size()) {
                    state.n_sent_text = state.generated_text.size();
                }
                if (!state.generated_tokens.empty()) {
                    state.generated_tokens.pop_back();
                }
                state.has_next_token = false;
                // Flush any buffered tokens before breaking on EOS
                if (callback && state.n_sent_text < state.generated_text.size()) {
                    callback(state.generated_text.substr(state.n_sent_text), false);
                    state.n_sent_text = state.generated_text.size();
                }
                break;
            }

            // Check stopping conditions (stop strings, context limit, n_remaining)
            bool should_stop = check_stop_conditions(state, state.antiprompt, token_text, options.ignore_eos);

            if (should_stop) {
                // Flush any buffered tokens before breaking on stop string / context limit
                if (callback && state.n_sent_text < state.generated_text.size()) {
                    callback(state.generated_text.substr(state.n_sent_text), false);
                    state.n_sent_text = state.generated_text.size();
                }
                break;
            }

            // Stream the token if callback is provided — buffered: only flush every buf_size tokens
            if (callback) {
                const int buf_size = (options.token_buffer_size > 0) ? options.token_buffer_size : 1;
                if (state.n_decoded % buf_size == 0) {
                    std::string text_to_send = state.generated_text.substr(state.n_sent_text);
                    state.n_sent_text = state.generated_text.size();
                    if (!text_to_send.empty() && !callback(text_to_send, false)) {
                        // Callback returned false — caller wants to stop; no further flush needed
                        state.has_next_token = false;
                        break;
                    }
                }
            }
        }

        result.stopped_by_length = state.stopped_by_limit;

        // Capture timings from the context performance counters
        {
            auto perf = llama_perf_context(rn_ctx->ctx);
            result.timings.predicted_n  = perf.n_eval;
            result.timings.predicted_ms = perf.t_eval_ms;
            result.timings.prompt_n     = perf.n_p_eval;
            result.timings.prompt_ms    = perf.t_p_eval_ms;
            result.timings.total_ms     = perf.t_p_eval_ms + perf.t_eval_ms;
        }

        // Set the result
        result.content = state.generated_text;
        result.tokens = state.generated_tokens;
        result.n_prompt_tokens = state.prompt_tokens.size();
        result.n_predicted_tokens = state.n_decoded;

        // KV state is owned by run_chat_completion (kv_messages). Nothing to update here.

        // Flush any tokens not yet sent due to buffering (e.g. when generation ended before
        // the next buf_size boundary — EOS, stop string, or n_predict limit).
        if (callback && state.n_sent_text < state.generated_text.size()) {
            callback(state.generated_text.substr(state.n_sent_text), false);
            state.n_sent_text = state.generated_text.size();
        }
        // Final callback with is_done=true
        if (callback) {
            callback(state.generated_text, true);
        }

        return result;

    } catch (const std::exception& e) {
        result.success = false;
        result.error_msg = e.what();
        result.error_type = RN_ERROR_GENERAL;
        return result;
    }
}

CompletionResult run_chat_completion(
    rn_llama_context* rn_ctx,
    const CompletionOptions& options,
    std::function<bool(const std::string&, bool)> callback) {

    CompletionResult result;
    completion_state state;

    if (!rn_ctx || !rn_ctx->model || !rn_ctx->ctx) {
        result.success = false;
        result.error_msg = "Model not initialized";
        result.error_type = RN_ERROR_MODEL_LOAD;
        return result;
    }

    try {
        // Extract media (image_url/audio_url content parts) before template rendering.
        // extract_media_from_messages replaces media parts with marker strings in-place.
        json messages_json = options.messages;
        std::vector<MediaItem> media_items;
        bool has_media = false;
        if (rn_ctx->multimodal_loaded && rn_ctx->mtmd_ctx &&
            !messages_json.is_null() && messages_json.is_array()) {
            media_items = extract_media_from_messages(messages_json);
            has_media = !media_items.empty();
        }
        const json& effective_messages = has_media ? messages_json : options.messages;

        // Parse messages directly from options
        std::vector<common_chat_msg> chat_msgs;
        if (!effective_messages.is_null() && !effective_messages.empty()) {
            chat_msgs = common_chat_msgs_parse_oaicompat(effective_messages);
        }

        const std::string kv_render_identity = build_kv_render_identity(rn_ctx, options);
        if (rn_ctx->kv_has_messages &&
            !rn_ctx->kv_render_identity.empty() &&
            rn_ctx->kv_render_identity != kv_render_identity) {
            llama_memory_clear(llama_get_memory(rn_ctx->ctx), true);
            rn_ctx->kv_messages.clear();
            rn_ctx->kv_has_messages = false;
        }

        // --- Per-message KV cache prefix reuse ---
        // Extract optional "id" fields from each message in the JSON array.
        // IDs are an optional caller-supplied field (not part of the OpenAI spec);
        // they let the native layer skip re-encoding messages that haven't changed
        // without doing a full token-by-token comparison.
        std::vector<std::string> msg_ids;
        if (!effective_messages.is_null() && effective_messages.is_array()) {
            for (const auto& msg : effective_messages) {
                if (msg.is_object() && msg.contains("id") && msg["id"].is_string()) {
                    msg_ids.push_back(msg["id"].get<std::string>());
                } else {
                    msg_ids.push_back(""); // message has no ID — cannot be reused by lookup
                }
            }
        }

        // Find how many leading messages have IDs that match the cached sequence.
        // A message with an empty ID never matches (we can't trust it's unchanged).
        size_t kv_match_count = 0;
        if (!has_media && !msg_ids.empty() && rn_ctx->kv_has_messages && !options.reset_kv_cache) {
            const auto& cached = rn_ctx->kv_messages;
            while (kv_match_count < msg_ids.size()
                   && kv_match_count < cached.size()
                   && !msg_ids[kv_match_count].empty()
                   && msg_ids[kv_match_count] == cached[kv_match_count].id) {
                kv_match_count++;
            }
        }

        // Completion cache: if prompt_id changed from the cached value, the system prompt or
        // tools have changed — the KV cache is stale and must be fully cleared.
        // This must happen BEFORE the KV eviction block so the clear is applied correctly.
        const bool has_cache_ids = !options.prompt_id.empty() && !options.config_id.empty();
        if (has_cache_ids && rn_ctx->completion_cache.has_value()) {
            if (rn_ctx->completion_cache->prompt_id != options.prompt_id) {
                // prompt_id changed — system prompt or tools changed; KV cache is invalid.
                llama_memory_clear(llama_get_memory(rn_ctx->ctx), true);
                rn_ctx->kv_messages.clear();
                rn_ctx->kv_has_messages = false;
                // Invalidate the cache entry so config_cache_hit will be false below.
                rn_ctx->completion_cache.reset();
            }
        }

        // Decide the KV common position and evict stale entries.
        int32_t kv_hint_pos = 0; // default: full encode from position 0
        if (has_media) {
            // Images invalidate KV prefix reuse — always do a full clear.
            llama_memory_clear(llama_get_memory(rn_ctx->ctx), true);
        } else if (kv_match_count > 0) {
            const int32_t kv_common_len = rn_ctx->kv_messages[kv_match_count - 1].token_end;
            // Clamp against actual context size: persisted metadata may be stale if the
            // context was shifted or cleared externally.
            const int32_t n_ctx_size = static_cast<int32_t>(llama_n_ctx(rn_ctx->ctx));
            const int32_t safe_kv_len = std::min(kv_common_len, n_ctx_size);
            if (kv_match_count < rn_ctx->kv_messages.size()) {
                // Some cached messages are no longer present — evict beyond the common prefix.
                if (safe_kv_len <= 0 ||
                    !llama_memory_seq_rm(llama_get_memory(rn_ctx->ctx), 0, safe_kv_len, -1)) {
                    // seq_rm returns false on recurrent models and some GPU backends; without
                    // fallback, stale fp16 values at positions > n_past corrupt softmax computation.
                    llama_memory_clear(llama_get_memory(rn_ctx->ctx), true);
                    kv_hint_pos = 0;
                } else {
                    kv_hint_pos = safe_kv_len;
                }
            } else {
                // All cached messages still match. Evict stale KV entries beyond safe_kv_len
                // (left over from prior generation turns).
                if (safe_kv_len <= 0 ||
                    !llama_memory_seq_rm(llama_get_memory(rn_ctx->ctx), 0, safe_kv_len, -1)) {
                    // seq_rm returns false on recurrent models and some GPU backends; without
                    // fallback, stale fp16 values at positions > n_past corrupt softmax computation.
                    llama_memory_clear(llama_get_memory(rn_ctx->ctx), true);
                    kv_hint_pos = 0;
                } else {
                    kv_hint_pos = safe_kv_len;
                }
            }
        } else {
            // No matching prefix (or no IDs provided): full clear.
            llama_memory_clear(llama_get_memory(rn_ctx->ctx), true);
            kv_hint_pos = 0;
        }

        // Completion cache lookup: if both prompt_id and config_id match the cached entry,
        // skip the expensive common_chat_templates_apply step entirely and reuse the cached
        // sampling params, grammar, grammar_triggers, preserved_tokens, and additional_stops.
        // When both IDs are empty, behave identically to before (no cache lookup, no cache store).
        bool config_cache_hit = false;
        if (has_cache_ids && rn_ctx->completion_cache.has_value()) {
            const auto& cached_entry = *rn_ctx->completion_cache;
            if (cached_entry.prompt_id == options.prompt_id && cached_entry.config_id == options.config_id) {
                config_cache_hit = true;
            }
        }

        // Build template inputs directly from options — no JSON roundtrip
        common_chat_templates_inputs template_inputs;
        template_inputs.messages             = chat_msgs;
        template_inputs.add_generation_prompt = true;
        template_inputs.use_jinja            = rn_ctx->params.use_jinja;
        template_inputs.reasoning_format     = rn_ctx->params.reasoning_format;
        template_inputs.chat_template_kwargs = rn_ctx->params.default_template_kwargs;

        // enable_thinking from kwargs
        auto it = template_inputs.chat_template_kwargs.find("enable_thinking");
        if (it != template_inputs.chat_template_kwargs.end()) {
            template_inputs.enable_thinking = (it->second == "true");
        }

        if (!options.grammar.empty()) {
            template_inputs.grammar = options.grammar;
        }

        if (!options.tools.is_null() && !options.tools.empty()) {
            template_inputs.tools = common_chat_tools_parse_oaicompat(options.tools);
            template_inputs.parallel_tool_calls = true;
        }

        if (!options.tool_choice.empty()) {
            template_inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(options.tool_choice);
        }

        if (!template_inputs.tools.empty() && template_inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            if (!template_inputs.grammar.empty()) {
                throw std::runtime_error("Cannot use custom grammar constraints with tools.");
            }
        }

        CompletionOptions cmpl_options = options;
        // Pass the trusted KV start position so run_completion skips its own KV management.
        cmpl_options.kv_hint_pos = kv_hint_pos;

        common_chat_params chat_params;
        if (config_cache_hit) {
            // Cache hit: always render with full template inputs (including tools) so
            // tool-aware templates remain semantically identical between miss/hit paths.
            const auto& cached_entry = *rn_ctx->completion_cache;
            try {
                chat_params = common_chat_templates_apply(rn_ctx->chat_templates.get(), template_inputs);
            } catch (const std::exception &) {
                try {
                    template_inputs.force_pure_content = true;
                    chat_params = common_chat_templates_apply(rn_ctx->chat_templates.get(), template_inputs);
                } catch (const std::exception &) {
                    template_inputs.use_jinja = false;
                    chat_params = common_chat_templates_apply(rn_ctx->chat_templates.get(), template_inputs);
                }
            }

            // Preserve stop strings and protected tokens from the current render.
            for (const auto & stop : chat_params.additional_stops) {
                cmpl_options.stop.push_back(stop);
            }

            for (const auto & pt : chat_params.preserved_tokens) {
                auto ids = common_tokenize(rn_ctx->vocab, pt, false, true);
                if (ids.size() == 1) {
                    cmpl_options.preserved_tokens.insert(ids[0]);
                }
            }

            // Conservative grammar reuse:
            // - Reuse cached grammar only when no tool grammar is in play.
            // - Otherwise, regenerate from current chat_params for correctness.
            const bool grammar_inputs_unchanged =
                template_inputs.tools.empty() &&
                options.grammar.empty() &&
                !chat_params.grammar.empty();

            if (grammar_inputs_unchanged && !cached_entry.grammar.empty()) {
                cmpl_options.grammar          = cached_entry.grammar;
                cmpl_options.grammar_lazy     = cached_entry.grammar_lazy;
                cmpl_options.grammar_triggers = cached_entry.grammar_triggers;
            } else if (!chat_params.grammar.empty()) {
                cmpl_options.grammar = chat_params.grammar;
                if (!template_inputs.tools.empty()) {
                    cmpl_options.grammar_lazy = false;
                } else {
                    cmpl_options.grammar_lazy = chat_params.grammar_lazy;
                }
                cmpl_options.grammar_triggers = chat_params.grammar_triggers;
            }
        } else {
            // Cache miss (or IDs empty): run full common_chat_templates_apply as before.
            //
            // common_chat_templates_apply_jinja renders the template twice against real messages
            // (add_generation_prompt=false then true) to extract the generation-prompt suffix before
            // invoking either the specialized handler or the auto-parser.  If the model's embedded
            // jinja template references a field that is absent or a wrong type (e.g. calls .lstrip()
            // on a non-string), the runtime throws std::runtime_error.
            //
            // Level 1: full jinja path — autoparser generates a grammar/parser for structured output.
            // Level 2 (force_pure_content): skips the autoparser; jinja still renders the prompt but
            //          no grammar constraint is produced.  Tool calls still appear in the prompt via
            //          the template itself; only grammar-constrained sampling is lost.
            // Level 3 (use_jinja=false): the C++ llama_chat_apply_template path — zero jinja.
            //          Last resort to prevent a hard crash when jinja itself cannot execute the
            //          template.  Produces a usable prompt; tool-call grammar/parser is not available.
            try {
                chat_params = common_chat_templates_apply(rn_ctx->chat_templates.get(), template_inputs);
            } catch (const std::exception & e) {
                try {
                    template_inputs.force_pure_content = true;
                    chat_params = common_chat_templates_apply(rn_ctx->chat_templates.get(), template_inputs);
                } catch (const std::exception &) {
                    template_inputs.use_jinja = false;
                    chat_params = common_chat_templates_apply(rn_ctx->chat_templates.get(), template_inputs);
                }
            }

            // Add extra stop strings emitted by the chat format (e.g. EOS variants, special separators).
            // Mirrors server-common.cpp: llama_params["stop"].push_back(stop)
            for (const auto & stop : chat_params.additional_stops) {
                cmpl_options.stop.push_back(stop);
            }

            // Tokenize preserved_tokens strings from the chat template and insert single-token IDs
            // into sampling params so the tokenizer never splits them mid-sequence.
            // Mirrors server-task.cpp: common_tokenize → insert if size() == 1.
            for (const auto & pt : chat_params.preserved_tokens) {
                auto ids = common_tokenize(rn_ctx->vocab, pt, false, true);
                if (ids.size() == 1) {
                    cmpl_options.preserved_tokens.insert(ids[0]);
                }
            }

            if (!chat_params.grammar.empty()) {
                cmpl_options.grammar = chat_params.grammar;
                // Always force grammar_lazy to false when tools are present
                if (!template_inputs.tools.empty()) {
                    cmpl_options.grammar_lazy = false;
                } else {
                    // Only use chat_params.grammar_lazy if no tools are present
                    cmpl_options.grammar_lazy = chat_params.grammar_lazy;
                }
                // Default to grammar_triggers provided by chat_params
                cmpl_options.grammar_triggers = chat_params.grammar_triggers;
            }

            // Store new cache entry when both IDs are provided.
            if (has_cache_ids) {
                rn_llama_context::completion_cache_entry new_entry;
                new_entry.prompt_id        = options.prompt_id;
                new_entry.config_id        = options.config_id;
                new_entry.grammar          = cmpl_options.grammar;
                new_entry.grammar_lazy     = cmpl_options.grammar_lazy;
                new_entry.grammar_triggers = cmpl_options.grammar_triggers;
                new_entry.preserved_tokens = cmpl_options.preserved_tokens;
                new_entry.additional_stops = chat_params.additional_stops;
                rn_ctx->completion_cache   = std::move(new_entry);
            }
        }

        cmpl_options.prompt = chat_params.prompt;

        // Multimodal: tokenize + eval the full prompt (text + images) via mtmd.
        // On success, sets cmpl_options.mtmd_encoded_n_past so run_completion skips
        // its own tokenize+encode and jumps straight to the generation loop.
        if (has_media) {
            auto bm_deleter = [](mtmd_bitmap* b) { if (b) mtmd_bitmap_free(b); };
            using BitmapPtr = std::unique_ptr<mtmd_bitmap, decltype(bm_deleter)>;
            std::vector<BitmapPtr> bitmaps;
            for (const auto& item : media_items) {
                BitmapPtr bm(load_bitmap_from_uri(rn_ctx->mtmd_ctx, item.url), bm_deleter);
                if (!bm) {
                    result.success = false;
                    result.error_msg = "Failed to load media: " + item.url;
                    result.error_type = RN_ERROR_INVALID_PARAM;
                    return result;
                }
                bitmaps.push_back(std::move(bm));
            }

            std::vector<const mtmd_bitmap*> bm_ptrs;
            bm_ptrs.reserve(bitmaps.size());
            for (const auto& bm : bitmaps) bm_ptrs.push_back(bm.get());

            mtmd_input_chunks* chunks = mtmd_input_chunks_init();
            mtmd_input_text input_text;
            input_text.text          = cmpl_options.prompt.c_str();
            input_text.add_special   = true;
            input_text.parse_special = true;
            int32_t tok_res = mtmd_tokenize(
                rn_ctx->mtmd_ctx, chunks, &input_text, bm_ptrs.data(), bm_ptrs.size());
            if (tok_res != 0) {
                mtmd_input_chunks_free(chunks);
                result.success = false;
                result.error_msg = "mtmd_tokenize failed (" + std::to_string(tok_res) + ")";
                result.error_type = RN_ERROR_INFERENCE;
                return result;
            }

            llama_pos new_n_past = 0;
            int32_t eval_res = mtmd_helper_eval_chunks(
                rn_ctx->mtmd_ctx, rn_ctx->ctx, chunks,
                0, 0, rn_ctx->params.n_batch, true, &new_n_past);
            mtmd_input_chunks_free(chunks);
            if (eval_res != 0) {
                result.success = false;
                result.error_msg = "mtmd eval failed (" + std::to_string(eval_res) + ")";
                result.error_type = RN_ERROR_INFERENCE;
                return result;
            }
            cmpl_options.mtmd_encoded_n_past = static_cast<int32_t>(new_n_past);
            cmpl_options.kv_hint_pos = -1;
        }

        // Run standard completion with the processed prompt
        result = run_completion(rn_ctx, cmpl_options, callback);

        // Update per-message KV boundaries so the next call can skip unchanged messages.
        // We only do this when message IDs were provided (otherwise there's nothing to track).
        if (result.success && !msg_ids.empty()) {
            // Retain boundaries for matched messages (they're already correct in kv_messages).
            rn_ctx->kv_messages.resize(kv_match_count);

            // For each new/changed message, determine where its tokens end by applying
            // the chat template up to that message and tokenizing the result.
            // This is O(n_new_messages) template applications — typically 1–2 per turn.
            const size_t n_msgs = chat_msgs.size();
            for (size_t k = kv_match_count; k < n_msgs && k < msg_ids.size(); k++) {
                if (msg_ids[k].empty()) {
                    // No ID for this message — stop tracking here; subsequent messages
                    // can't be matched by ID either.
                    break;
                }
                // Apply the template to messages[0..k] to get the token boundary.
                // Use add_generation_prompt=true for the final message so the stored token_end
                // includes the gen-prompt suffix tokens (e.g. "<|im_start|>assistant\n") that
                // ARE encoded in the actual KV cache. Using false would make kv_hint_pos point
                // before those tokens, causing the next call to re-encode them at occupied
                // KV positions and misaligning all subsequent RoPE position embeddings.
                common_chat_templates_inputs tinput;
                tinput.messages               = std::vector<common_chat_msg>(chat_msgs.begin(),
                                                                              chat_msgs.begin() + k + 1);
                tinput.add_generation_prompt  = (k + 1 == n_msgs);
                tinput.use_jinja              = rn_ctx->params.use_jinja;
                tinput.reasoning_format       = rn_ctx->params.reasoning_format;
                tinput.chat_template_kwargs   = rn_ctx->params.default_template_kwargs;
                try {
                    auto partial = common_chat_templates_apply(rn_ctx->chat_templates.get(), tinput);
                    auto partial_tokens = common_tokenize(rn_ctx->vocab, partial.prompt, true, true);
                    rn_ctx->kv_messages.push_back({msg_ids[k],
                                                   static_cast<int32_t>(partial_tokens.size())});
                } catch (...) {
                    break; // template failed — stop tracking; next call will do a full encode
                }
            }
            rn_ctx->kv_has_messages = true;
            rn_ctx->kv_render_identity = kv_render_identity;
        } else if (options.reset_kv_cache) {
            rn_ctx->kv_messages.clear();
            rn_ctx->kv_has_messages = false;
            rn_ctx->kv_render_identity.clear();
        }

        if (result.success) {
            // Parse the generated content for tool calls and structured responses
            common_chat_msg parsed_msg;
            bool has_parsed_content = false;
            
            // Only parse if we have tools available and the response isn't empty
            if (!template_inputs.tools.empty() && !result.content.empty()) {
                try {
                    // Construct parser params from the applied chat params, then override reasoning format.
                    // The common_chat_parser_params(chat_params) constructor only copies format and
                    // generation_prompt — it does NOT copy the PEG arena.  Load it explicitly so that
                    // common_chat_parse uses the autoparser's generated PEG grammar for tool-call
                    // parsing instead of the fallback pure-content parser.
                    // Mirrors server-task.cpp: params.chat_parser_params.parser.load(data["chat_parser"])
                    common_chat_parser_params parser_params(chat_params);
                    parser_params.reasoning_format = rn_ctx->params.reasoning_format;
                    if (!chat_params.parser.empty()) {
                        parser_params.parser.load(chat_params.parser);
                    }

                    // Parse the generated content for tool calls
                    parsed_msg = common_chat_parse(result.content, false, parser_params);
                    has_parsed_content = true;
                    
                } catch (const std::exception& e) {
                    // If parsing fails, treat as regular content
                    has_parsed_content = false;
                    result.tool_call_parse_failed = true;
                    result.tool_call_parse_error = e.what();
                }
            }
            
            // Create OpenAI-compatible response
            json response = {
                {"id", gen_chatcmplid()},
                {"object", "chat.completion"},
                {"created", static_cast<int>(std::time(nullptr))},
                {"model", options.model.empty() ? "llamacpp-rn" : options.model}
            };

            json choices = json::array();
            json choice = {
                {"index", 0},
                {"message", {
                    {"role", "assistant"}
                }},
                {"finish_reason", result.stopped_by_length ? "length" : "stop"}
            };
            
            // Add parsed content and tool calls if available
            if (has_parsed_content && !parsed_msg.tool_calls.empty()) {
                // Use the server.cpp approach: let the common_chat_msg handle the JSON conversion
                choice["message"] = parsed_msg.to_json_oaicompat();
                choice["finish_reason"] = "tool_calls";
            } else if (result.tool_call_parse_failed) {
                choice["message"]["content"] = result.content;
                choice["finish_reason"] = "tool_call_parse_error";
                choice["tool_call_parse_error"] = result.tool_call_parse_error;
            } else if (has_parsed_content && !parsed_msg.content.empty()) {
                // Regular text response with parsed content
                choice["message"]["content"] = parsed_msg.content;
            } else {
                // Fallback to raw content if parsing failed or no tools
                choice["message"]["content"] = result.content;
            }

            choices.push_back(choice);
            response["choices"] = choices;

            // Add usage information
            response["usage"] = {
                {"prompt_tokens", result.n_prompt_tokens},
                {"completion_tokens", result.n_predicted_tokens},
                {"total_tokens", result.n_prompt_tokens + result.n_predicted_tokens}
            };

            // Store the response in the result
            result.chat_response = response;
        }

        return result;
    } catch (const std::exception& e) {
        result.success = false;
        result.error_msg = std::string("Chat completion error: ") + e.what();
        result.error_type = RN_ERROR_GENERAL;
        return result;
    }
}

} // namespace facebook::react

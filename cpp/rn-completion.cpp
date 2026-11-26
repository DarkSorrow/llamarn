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
#include "rn-kv.h"
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
#include <cmath>
#include <optional>

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
    // Same bytes as generated_text, with the token id on the first byte of each piece.
    // Mirrors server_slot::generated in server-context.cpp process_token.
    common_chat_input generated;
    std::string stopping_word;
    bool stopped_by_limit = false;
    bool context_shifted = false;  // true if at least one context shift occurred
    int shift_n_keep = 0;          // keep count of the shift, 0 when this run did not shift

    std::vector<llama_token> prompt_tokens;
    std::vector<llama_token> generated_tokens;

    std::unique_ptr<common_sampler, sampler_deleter> sampler;
    std::vector<std::string> antiprompt;

};

void rn_stream_parser::update(const common_chat_input & generated, size_t upto, bool final, rn_stream_chunk & chunk) {
    if (!session || disabled) {
        return;
    }
    upto = std::min(upto, generated.size());
    if (upto < n_fed) {
        // A stop string cut text that was already fed. The session cannot unfeed.
        disabled = true;
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (!final && now - last_parse < interval) {
        return;
    }
    last_parse = now;
    try {
        const common_chat_input added = generated.substr(n_fed, upto - n_fed);
        n_fed = upto;
        const common_chat_msg & cur = final ? session->finish(added) : session->feed(added);
        for (const auto & d : common_chat_msg_diff::compute_diffs(prev, cur)) {
            chunk.content           += d.content_delta;
            chunk.reasoning_content += d.reasoning_content_delta;
        }
        prev = cur;
    } catch (const std::exception &) {
        disabled = true;
    }
    if (final) {
        disabled = true;
    }
}

// Keep generated_text, generated, and generated_tokens on the same kept prefix.
// generated.tokens stores the id on the first byte of a piece and LLAMA_TOKEN_NULL on the rest.
static void rn_truncate_generated(completion_state & state, size_t keep) {
    state.generated.truncate(keep);
    state.generated_text = state.generated.text;
    if (state.n_sent_text > state.generated_text.size()) {
        state.n_sent_text = state.generated_text.size();
    }
    state.generated_tokens.clear();
    for (size_t i = 0; i < state.generated.tokens.size();) {
        const llama_token id = state.generated.tokens[i];
        if (id != LLAMA_TOKEN_NULL) {
            state.generated_tokens.push_back(id);
        }
        ++i;
        while (i < state.generated.tokens.size() && state.generated.tokens[i] == LLAMA_TOKEN_NULL) {
            ++i;
        }
    }
}

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
        // Server: slot.generated.truncate(pos + stop_pos). Drop the stop string and its token ids.
        rn_truncate_generated(state, stop_pos);
        state.has_next_token = false;
        return true;
    }

    // Check for newline condition
    if (token_text.find('\n') != std::string::npos) {
        state.has_new_line = true;
    }

    return false;
}

// llama_process: 2 = aborted, < -1 = fatal; in both cases ubatches already processed stay
// in the KV cache, so drop everything from this batch on to keep KV in sync with n_past.
static CompletionResult & rn_decode_failed(
    CompletionResult & result,
    rn_llama_context * rn_ctx,
    int32_t ret,
    llama_pos first_pos,
    const char * what) {
    llama_memory_t mem = llama_get_memory(rn_ctx->ctx);
    if (llama_memory_seq_rm(mem, 0, first_pos, -1)) {
        if (rn_ctx->kv_tokens.size() > static_cast<size_t>(std::max<llama_pos>(first_pos, 0))) {
            rn_ctx->kv_tokens.resize(static_cast<size_t>(std::max<llama_pos>(first_pos, 0)));
        }
    } else {
        // recurrent / hybrid memory cannot drop a suffix: start over
        llama_memory_clear(mem, true);
        rn_ctx->kv_tokens.clear();
        rn_ctx->kv_render_identity.clear();
    }
    result.success    = false;
    result.error_type = RN_ERROR_INFERENCE;
    result.error_msg  = ret == 2 ? std::string("Generation aborted")
                                 : std::string(what) + " (llama_process returned " + std::to_string(ret) + ")";
    return result;
}

// A model's embedded jinja template can throw (missing field, wrong type, raise_exception).
// Level 1: full jinja + autoparser (grammar, tool-call parser).
// Level 2 (force_pure_content): jinja renders the prompt, no grammar or tool-call parser.
// Level 3 (use_jinja=false): llama_chat_apply_template, zero jinja.
// `inputs` keeps the level that succeeded so later renders (KV boundaries) match the prompt.
common_chat_session rn_make_chat_session(const rn_llama_context * rn_ctx, common_chat_templates_inputs & inputs) {
    try {
        return common_chat_session(rn_ctx->chat_templates.get(), rn_ctx->vocab, inputs);
    } catch (const std::exception &) {}
    inputs.force_pure_content = true;
    try {
        return common_chat_session(rn_ctx->chat_templates.get(), rn_ctx->vocab, inputs);
    } catch (const std::exception &) {}
    inputs.use_jinja = false;
    return common_chat_session(rn_ctx->chat_templates.get(), rn_ctx->vocab, inputs);
}

void rn_init_batches(rn_llama_context * rn_ctx) {
    rn_ctx->gen_batch    = std::make_unique<common_batch>(rn_ctx->ctx);
    rn_ctx->ingest_batch = std::make_unique<common_batch>(rn_ctx->ctx);
}

void rn_free_batches(rn_llama_context * rn_ctx) {
    rn_ctx->gen_batch.reset();
    rn_ctx->ingest_batch.reset();
}

static std::string build_kv_render_identity(
    const rn_llama_context* rn_ctx,
    const CompletionOptions& options) {
    std::string identity;
    identity.reserve(512);
    identity += "use_jinja=" + std::to_string(rn_ctx->params.use_jinja ? 1 : 0);
    identity += "|reasoning_format=" + std::to_string(static_cast<int>(rn_ctx->params.reasoning_format));
    identity += "|chat_template=" + rn_ctx->params.chat_template;
    identity += "|tool_choice=" + options.tool_choice.dump();
    identity += "|parallel_tool_calls=" + std::to_string(rn_ctx->params.parallel_tool_calls ? 1 : 0);
    identity += "|grammar=" + options.grammar;
    identity += "|tools=" + (options.tools.is_null() ? "null" : options.tools.dump());
    // Same default as the render below: the struct starts at true unless kwargs say otherwise.
    bool enable_thinking = true;
    const auto thinking = rn_ctx->params.default_template_kwargs.find("enable_thinking");
    if (thinking != rn_ctx->params.default_template_kwargs.end()) {
        enable_thinking = thinking->second == "true";
    }
    identity += "|enable_thinking=" + std::to_string(enable_thinking ? 1 : 0);
    return identity;
}

CompletionResult run_completion(
    rn_llama_context* rn_ctx,
    const CompletionOptions& options,
    rn_stream_callback callback) {

    CompletionResult result;
    completion_state state;

    if (!rn_ctx || !rn_ctx->model || !rn_ctx->ctx) {
        result.success = false;
        result.error_msg = "Model not initialized";
        result.error_type = RN_ERROR_MODEL_LOAD;
        return result;
    }

    if (!rn_ctx->gen_batch || !rn_ctx->ingest_batch) {
        result.success = false;
        result.error_msg = "Decode batches are not initialized";
        result.error_type = RN_ERROR_CONTEXT;
        return result;
    }

    try {
        // Initialize state with context values
        state.rn_ctx = rn_ctx;

        const auto& params = rn_ctx->params;

        // Reset per-request performance counters so timings reflect this request only,
        // not cumulative totals across all prior requests on this context.
        // Mirrors server-context.cpp slot reset behavior.
        llama_perf_context_reset(rn_ctx->ctx);

        // Create a copy of sampling parameters and apply per-request overrides.
        // All mutations happen on this LOCAL copy — rn_ctx_->params.sampling is never touched.
        common_params_sampling sampling_params = params.sampling;

        // Apply per-request sampling overrides from CompletionOptions.
        // Only override when the caller explicitly set a value (not NaN / -1 sentinel).
        // When not set, the model's initLlama defaults (params.sampling) are preserved —
        // this respects per-model tuning and GGUF-embedded sampling metadata.
        auto applyF = [](float& dst, float src) { if (!std::isnan(src)) dst = src; };
        auto applyFI = [](int32_t& dst, float src) { if (!std::isnan(src)) dst = static_cast<int32_t>(src); };
        applyF(sampling_params.temp,            options.temperature);
        applyF(sampling_params.top_p,           options.top_p);
        applyFI(sampling_params.top_k,          options.top_k);
        applyF(sampling_params.min_p,           options.min_p);
        applyF(sampling_params.penalty_present, options.presence_penalty);
        applyF(sampling_params.penalty_repeat,  options.repeat_penalty);
        applyF(sampling_params.penalty_freq,    options.frequency_penalty);
        if (options.repeat_last_n >= 0) {
            sampling_params.penalty_last_n = options.repeat_last_n;
        } else if (sampling_params.penalty_last_n == 64) {
            // Neither the caller nor the GGUF set a repeat-penalty window.
            // 64 (llama.cpp hardcoded default) is smaller than a typical paragraph (~80-200 tokens),
            // so paragraph-level repetition passes through the penalty window undetected.
            // Bump to 256 — covers ~2 average paragraphs with no user intent to override.
            sampling_params.penalty_last_n = 256;
        }
        // Map JS sentinel -1 ("use default/random") to the model's configured seed.
        // Only override if the caller supplied an explicit non-negative seed.
        if (options.seed >= 0) {
            sampling_params.seed = static_cast<uint32_t>(options.seed);
        }

        // Chat path: the session owns grammar (user, json_schema or tool calls), lazy triggers
        // and preserved tokens, resolved against the vocab (server: apply_chat_session).
        if (options.chat_session) {
            options.chat_session->apply_sampling(sampling_params);
        } else if (!options.grammar.empty()) {
            sampling_params.grammar      = common_grammar(COMMON_GRAMMAR_TYPE_USER, options.grammar);
            sampling_params.grammar_lazy = options.grammar_lazy;
        }

        // Wire reasoning budget into sampling params.
        // The start/end tags come from the chat session (set by run_chat_completion).
        // Mirrors server-common.cpp oaicompat_chat_params_parse() reasoning budget block.
        if (!options.reasoning_budget_start_tag.empty() && !options.reasoning_budget_end_tags.empty()) {
            sampling_params.reasoning_budget_tokens = options.reasoning_budget_tokens;
            sampling_params.reasoning_budget_start  = common_tokenize(rn_ctx->vocab, options.reasoning_budget_start_tag, false, true);

            // reasoning_budget_end is now std::vector<llama_tokens> — one token sequence per end
            // tag (common/common.h). Tokenize each tag separately; the first tag is the forcing one.
            sampling_params.reasoning_budget_end.clear();
            for (const auto& end_tag : options.reasoning_budget_end_tags) {
                if (!end_tag.empty()) {
                    sampling_params.reasoning_budget_end.push_back(
                        common_tokenize(rn_ctx->vocab, end_tag, false, true));
                }
            }

            if (!sampling_params.reasoning_budget_end.empty()) {
                if (!options.reasoning_budget_message.empty()) {
                    // Upstream token-level concatenation pattern (tools/server/server-schema.cpp):
                    // tokenize the end tag and the forced message SEPARATELY, then concatenate the
                    // token vectors. Tokenizing the concatenated STRING as one unit can tokenize
                    // differently at the message/tag boundary.
                    llama_tokens forced         = sampling_params.reasoning_budget_end.front();
                    llama_tokens message_tokens = common_tokenize(rn_ctx->vocab, options.reasoning_budget_message, false, true);
                    forced.insert(forced.begin(), message_tokens.begin(), message_tokens.end());
                    sampling_params.reasoning_budget_forced  = std::move(forced);
                    sampling_params.reasoning_budget_message = options.reasoning_budget_message;
                } else {
                    sampling_params.reasoning_budget_forced = sampling_params.reasoning_budget_end.front();
                }
            }
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
            // The cache holds media embeddings: no token list describes it.
            rn_ctx->kv_tokens.clear();
            if (options.n_predict >= 0) {
                state.n_predict = options.n_predict;
            } else if (params.n_predict >= 0) {
                state.n_predict = params.n_predict;
            } else {
                state.n_predict = -1; // unlimited — EOS or stop string terminates
            }
            state.n_remaining = state.n_predict;
            result.n_prompt_tokens = 0;
        } else {

        if (!options.prompt_tokens.empty()) {
            state.prompt_tokens = options.prompt_tokens;
        } else {
            if (options.prompt.empty()) {
                result.success = false;
                result.error_msg = "No prompt provided";
                result.error_type = RN_ERROR_INVALID_PARAM;
                return result;
            }
            state.prompt_tokens = common_tokenize(rn_ctx->vocab, options.prompt, true, true);
        }
        if (state.prompt_tokens.empty()) {
            result.success = false;
            result.error_msg = "Empty prompt";
            result.error_type = RN_ERROR_INVALID_PARAM;
            return result;
        }

        // KV cache: run_chat_completion compared kv_tokens with this prompt, dropped the cache
        // after their common prefix and passes that length as kv_hint_pos. Positions below it
        // hold exactly prompt_tokens[0, hint). Direct callers (no hint) start from a full clear.
        bool hint_ok = options.kv_hint_pos >= 0 &&
            static_cast<size_t>(options.kv_hint_pos) < state.prompt_tokens.size() &&
            static_cast<size_t>(options.kv_hint_pos) <= rn_ctx->kv_tokens.size();
        if (hint_ok) {
            state.n_past = options.kv_hint_pos;
            rn_ctx->kv_tokens.resize(static_cast<size_t>(state.n_past));
        } else {
            llama_memory_clear(llama_get_memory(rn_ctx->ctx), true);
            rn_ctx->kv_tokens.clear();
            state.n_past = 0;
        }

        // Configure state
        state.n_ctx = llama_n_ctx(rn_ctx->ctx);
        if (options.n_predict >= 0) {
            state.n_predict = options.n_predict;
        } else if (params.n_predict >= 0) {
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
        // A chunk must never exceed n_batch (one llama_process call per chunk).
        const int ingest_chunk = std::min(std::clamp(rn_ctx->params.chunk_size, 8, 512),
                                          static_cast<int>(llama_n_batch(rn_ctx->ctx)));

        // With prefix reuse (n_past > 0) only prompt_tokens[n_past:] are encoded, at their own positions.
        {
            const int n_total = static_cast<int>(state.prompt_tokens.size());
            common_batch& ingest_batch = *rn_ctx->ingest_batch;
            for (int pos = state.n_past; pos < n_total; ) {
                if (rn_ctx->abort_generation.load(std::memory_order_relaxed)) {
                    return rn_decode_failed(result, rn_ctx, 2, pos, "Failed to process prompt");
                }
                const int chunk = std::min(ingest_chunk, n_total - pos);
                const bool last_chunk = (pos + chunk >= n_total);
                ingest_batch.clear();
                for (int j = 0; j < chunk; j++) {
                    ingest_batch.add(state.prompt_tokens[pos + j], pos + j, 0, last_chunk && (j == chunk - 1));
                }
                auto t0 = std::chrono::steady_clock::now();
                const int32_t ret = llama_process(rn_ctx->ctx, LLAMA_PROCESS_TYPE_DECODE, ingest_batch.get());
                if (ret != 0) {
                    return rn_decode_failed(result, rn_ctx, ret, pos, "Failed to process prompt");
                }
                rn_ctx->kv_tokens.insert(rn_ctx->kv_tokens.end(),
                                         state.prompt_tokens.begin() + pos,
                                         state.prompt_tokens.begin() + pos + chunk);
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
                pos += chunk;
            }
            state.n_past = n_total;
        }

        // Seed the sampler with prompt tokens — matches the server's init_sampler() pattern exactly.
        //
        // CORRECT ORDER (server-context.cpp init_sampler):
        //   1. Reset FIRST  → clears prev ring buffer + grammar state to a known-clean baseline
        //   2. Accept ALL prompt tokens with accept_grammar=false
        //      → rebuilds the repetition-penalty window (prev ring buffer)
        //      → does NOT advance grammar state (grammar only tracks generated tokens)
        //
        // WRONG ORDER (previous mobile code):
        //   1. Accept prompt tokens (builds prev)
        //   2. Reset (destroys prev) ← wipes the repetition penalty history we just built
        //
        // The wrong order caused the model to have zero context for repetition penalties,
        // leading to infinite repetition of prompt content on subsequent turns.
        //
        // Why accept_grammar=false:
        //   Grammar state should only track generated tokens, not prompt tokens.
        //   Using true here would advance the grammar FSM through the prompt, which is wrong
        //   for tool-call grammars that expect to start fresh at generation time.
        common_sampler_reset(state.sampler.get());

        for (auto tok : state.prompt_tokens) {
            common_sampler_accept(state.sampler.get(), tok, false);
        }

        result.n_prompt_tokens = state.prompt_tokens.size();

        } // end: normal tokenize + encode path (mtmd_encoded_n_past < 0)

        common_batch& gen_batch = *rn_ctx->gen_batch;

        // Cap n_predict to the space remaining in the context window.
        // When callers set max_tokens far beyond n_ctx (e.g. 8192 tokens with n_ctx=2048),
        // the model is forced into repeated context shifts. Each shift evicts mid-narrative
        // KV entries; the model then loops on content still visible near the beginning —
        // the primary mechanical cause of paragraph-level repetition in long generation.
        // Unlimited generation (n_predict < 0) is never capped; EOS or stop strings end it.
        if (state.n_predict > 0) {
            const int available = state.n_ctx - state.n_past - 4;
            if (available > 0 && state.n_predict > available) {
                state.n_predict   = available;
                state.n_remaining = available;
            }
        }

        // Whether context shift is available for this run.
        // Disabled for multimodal (image chunks have special positioning) and
        // for recurrent models that don't support KV position shifting.
        const bool can_ctx_shift =
            options.mtmd_encoded_n_past < 0 &&
            llama_memory_can_shift(llama_get_memory(rn_ctx->ctx));

        // MP-P2 FIX: pre-reserve generated_text and generated_tokens to avoid
        // repeated heap reallocations inside the token generation loop.
        // Average token is ~4 bytes (mixed ASCII/UTF-8); reserve 4× n_predict chars.
        // For unlimited generation (n_predict < 0), reserve 512 tokens as a reasonable default.
        if (state.n_predict > 0) {
            state.generated_text.reserve(static_cast<size_t>(state.n_predict) * 4);
            state.generated_tokens.reserve(static_cast<size_t>(state.n_predict));
        } else {
            // Thinking models (reasoning_budget != 0) generate 2000-5000 token reasoning
            // blocks with n_predict = -1. Reserve enough upfront to avoid reallocations.
            const bool has_thinking = (rn_ctx->params.reasoning_budget != 0);
            const size_t initial_reserve = has_thinking ? 6144 : 512;
            state.generated_text.reserve(initial_reserve * 4);
            state.generated_tokens.reserve(initial_reserve);
        }

        // Sends generated_text[n_sent_text, upto) plus the parser deltas. Returns the callback's
        // continue flag; an empty chunk is not sent.
        rn_stream_parser stream_parser;
        stream_parser.session = options.stream_session;
        auto send_upto = [&](size_t upto, bool final) -> bool {
            rn_stream_chunk chunk;
            if (upto > state.n_sent_text) {
                chunk.text = state.generated_text.substr(state.n_sent_text, upto - state.n_sent_text);
                state.n_sent_text = upto;
            }
            stream_parser.update(state.generated, state.n_sent_text, final, chunk);
            if (chunk.text.empty() && chunk.content.empty() && chunk.reasoning_content.empty()) {
                return true;
            }
            return callback(chunk, false);
        };

        while (state.has_next_token && (state.n_predict < 0 || state.n_remaining > 0)) {
            // Thermal management: apply any thread count change requested by the JS thread.
            // JS writes requested_n_threads with memory_order_release (non-blocking).
            // exchange(-1, acq_rel) clears the request; new count applies to next llama_decode.
            {
                int requested = rn_ctx->requested_n_threads.exchange(-1, std::memory_order_acq_rel);
                if (requested > 0) {
                    llama_set_n_threads(rn_ctx->ctx, requested, requested);
                }
            }

            // Context shift: if we're about to fill the KV cache, slide the window.
            // This allows infinite-length conversations without hard failures.
            // Mirrors server-context.cpp update_slots() context shift logic.
            if (state.n_past + 1 >= state.n_ctx) {
                if (!can_ctx_shift) {
                    // Recurrent model or multimodal — can't shift, stop generation
                    state.truncated = true;
                    state.has_next_token = false;
                    state.stopped_by_limit = true;
                    break;
                }

                // n_keep: tokens at the head of the cache that a shift must not move.
                // options.n_keep < 0 → stop instead of shifting
                // options.n_keep > 0 → that token count
                // options.n_keep == 0 → the first message boundary when it fits, else half the context
                const int n_keep = rn_shift_n_keep(
                    options.n_keep,
                    state.n_ctx,
                    options.kv_prefix_end,
                    llama_vocab_get_add_bos(rn_ctx->vocab));
                if (n_keep < 0) {
                    state.truncated = true;
                    state.has_next_token = false;
                    state.stopped_by_limit = true;
                    break;
                }

                const int n_left    = state.n_past - n_keep;
                const int n_discard = n_left / 2;

                if (n_discard <= 0) {
                    // n_keep is too large relative to n_ctx — can't shift
                    state.truncated = true;
                    state.has_next_token = false;
                    state.stopped_by_limit = true;
                    break;
                }

                // Remove KV entries at positions [n_keep, n_keep + n_discard)
                llama_memory_seq_rm(llama_get_memory(rn_ctx->ctx), 0, n_keep, n_keep + n_discard);
                // Shift remaining entries [n_keep + n_discard, n_past) backward by n_discard
                llama_memory_seq_add(llama_get_memory(rn_ctx->ctx), 0, n_keep + n_discard, state.n_past, -n_discard);

                // prompt_tokens is not shifted. It holds only the prompt, while n_discard counts
                // generated tokens too, so mirroring the shift there underflowed the resize
                // (std::length_error) once the reply outgrew the prompt. Its only later reader is
                // usage.prompt_tokens, which stays the full rendered prompt.

                // kv_tokens holds the prompt and the reply, the same positions as the cache.
                // Empty on a media turn (it never shifts), so this is then a no-op.
                rn_kv_tokens_shift(rn_ctx->kv_tokens, n_keep, n_discard);

                state.n_past -= n_discard;
                state.truncated = true;
                state.context_shifted = true;
                state.shift_n_keep = n_keep;
            }

            // Sample the next token
            llama_token token_id = common_sampler_sample(state.sampler.get(), rn_ctx->ctx, -1);

            // Guard: sampler returns LLAMA_TOKEN_NULL (-1) when the grammar rejects all
            // candidates (e.g. malformed grammar or empty vocabulary after constraints).
            // Passing -1 downstream to common_token_to_piece / llama_vocab_is_eog is UB.
            if (token_id == LLAMA_TOKEN_NULL) {
                result.success = false;
                result.error_msg = "Sampler produced no valid token (grammar may be over-constrained)";
                result.error_type = RN_ERROR_INFERENCE;
                return result;
            }

            // Extract the token text
            std::string token_text = common_token_to_piece(rn_ctx->vocab, token_id);

            // Server process_token: slot.generated.append(token_str, result.tok).
            state.generated.append(token_text, token_id);
            state.generated_text = state.generated.text;
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
            gen_batch.clear();
            gen_batch.add(token_id, state.n_past, 0, true);
            const int32_t ret = llama_process(rn_ctx->ctx, LLAMA_PROCESS_TYPE_DECODE, gen_batch.get());
            if (ret != 0) {
                result.content = state.generated.text;
                result.generated = state.generated;
                result.context_shifted = state.context_shifted;
                result.shift_n_keep = state.shift_n_keep;
                return rn_decode_failed(result, rn_ctx, ret, state.n_past, "Failed to decode generated token");
            }

            state.n_past++;
            // The token is in the cache now, end-of-generation tokens included. Media turns keep
            // no list: their positions hold embeddings, not tokens.
            if (options.mtmd_encoded_n_past < 0) {
                rn_ctx->kv_tokens.push_back(token_id);
            }

            // Accept the token into the sampler AFTER decode and n_past increment.
            // token_id is a plain int32 captured before llama_decode — safe to pass here.
            common_sampler_accept(state.sampler.get(), token_id, true);

            // Check EOS FIRST — before streaming — so the EOS token text is never sent to JS
            // and all end-of-generation tokens are handled (not just the single llama_vocab_eos()).
            // Also erase the EOG token text from generated_text so it does not appear in
            // result.content or any streamed batch: EOG tokens are control tokens, not content.
            if (!options.ignore_eos && llama_vocab_is_eog(rn_ctx->vocab, token_id)) {
                const size_t keep = state.generated_text.size() >= token_text.size()
                    ? state.generated_text.size() - token_text.size()
                    : 0;
                rn_truncate_generated(state, keep);
                state.has_next_token = false;
                // Flush any buffered tokens before breaking on EOS
                if (callback) {
                    send_upto(state.generated_text.size(), false);
                }
                break;
            }

            // Check stopping conditions (stop strings, context limit, n_remaining)
            bool should_stop = check_stop_conditions(state, state.antiprompt, token_text, options.ignore_eos);

            if (should_stop) {
                // Flush any buffered tokens before breaking on stop string / context limit
                if (callback) {
                    send_upto(state.generated_text.size(), false);
                }
                break;
            }

            // Stream unsent text, holding back any suffix that could be the start of a
            // stop word. find_partial_stop_string returns the TEXT position where the
            // potential prefix begins; we only send up to that position.
            if (callback) {
                size_t safe_send_limit = state.generated_text.size();
                for (const std::string& word : state.antiprompt) {
                    size_t partial_pos = find_partial_stop_string(word, state.generated_text);
                    if (partial_pos != std::string::npos && partial_pos < safe_send_limit) {
                        safe_send_limit = partial_pos;
                    }
                }
                if (safe_send_limit > state.n_sent_text && !send_upto(safe_send_limit, false)) {
                    state.has_next_token = false;
                    break;
                }
            }
        }

        result.stopped_by_length = state.stopped_by_limit;
        result.context_shifted   = state.context_shifted;
        result.shift_n_keep      = state.shift_n_keep;

        // Capture timings from the context performance counters
        {
            auto perf = llama_perf_context(rn_ctx->ctx);
            result.timings.predicted_n  = perf.n_eval;
            result.timings.predicted_ms = perf.t_eval_ms;
            result.timings.prompt_n     = perf.n_p_eval;
            result.timings.prompt_ms    = perf.t_p_eval_ms;
            result.timings.total_ms     = perf.t_p_eval_ms + perf.t_eval_ms;
        }

        // Set the result. content is the kept text; generated carries the token ids for session.finish.
        result.content = state.generated.text;
        result.generated = state.generated;
        result.tokens = state.generated_tokens;
        result.n_prompt_tokens = state.prompt_tokens.size();
        result.n_predicted_tokens = state.n_decoded;

        // rn_ctx->kv_tokens now matches the cache: the prompt plus every decoded reply token.

        // Flush any tokens not yet sent due to buffering (e.g. when generation ended before
        // the next buf_size boundary — EOS, stop string, or n_predict limit).
        // The final parse runs here too, so the last chunk completes the content / reasoning deltas.
        if (callback) {
            send_upto(state.generated_text.size(), true);
            callback(rn_stream_chunk{}, true);
        }

        return result;

    } catch (const std::exception& e) {
        result.success = false;
        result.error_msg = e.what();
        result.error_type = RN_ERROR_GENERAL;
        return result;
    }
}

// Counts media (image/audio) chunks in a tokenized mtmd input. Used to decide
// whether the batched-encode path below is worth taking.
static size_t count_media_chunks(const mtmd_input_chunks* chunks) {
    size_t n = 0;
    size_t n_chunks = mtmd_input_chunks_size(chunks);
    for (size_t i = 0; i < n_chunks; i++) {
        auto type = mtmd_input_chunk_get_type(mtmd_input_chunks_get(chunks, i));
        if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE || type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            n++;
        }
    }
    return n;
}

// Like mtmd_helper_eval_chunks, but batches consecutive runs of >=2 media
// chunks into a single encoder pass via the mtmd_batch_* API (new in
// llama.cpp b9777) instead of encoding each image/audio chunk one at a time.
// Text chunks, lone media chunks, and any run the encoder backend can't
// batch (mtmd_batch_add_chunk failure — e.g. the underlying clip backend
// doesn't support batching) fall back to the existing single-chunk path
// (mtmd_helper_eval_chunk_single), so behavior for the dominant 0-or-1-image
// case is unchanged. LLM decode of each chunk's embeddings stays strictly
// sequential — only the vision/audio encoder forward pass is batched.
//
// Only called when count_media_chunks(chunks) >= 2; the call site keeps the
// original mtmd_helper_eval_chunks call for everything else.
static int32_t mtmd_eval_chunks_batched(
        mtmd_context* mtmd_ctx,
        llama_context* lctx,
        const mtmd_input_chunks* chunks,
        llama_pos n_past,
        llama_seq_id seq_id,
        int32_t n_batch,
        bool logits_last,
        llama_pos* new_n_past) {
    size_t n_chunks = mtmd_input_chunks_size(chunks);
    if (n_chunks == 0) {
        *new_n_past = n_past;
        return 0;
    }

    auto batch_deleter = [](mtmd_batch* b) { if (b) mtmd_batch_free(b); };

    size_t i = 0;
    while (i < n_chunks) {
        const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            bool last = (i == n_chunks - 1) && logits_last;
            int32_t res = mtmd_helper_eval_chunk_single(mtmd_ctx, lctx, chunk, n_past, seq_id, n_batch, last, &n_past);
            if (res != 0) {
                *new_n_past = n_past;
                return res;
            }
            i++;
            continue;
        }

        // Gather a run of consecutive media chunks.
        size_t j = i + 1;
        while (j < n_chunks &&
               mtmd_input_chunk_get_type(mtmd_input_chunks_get(chunks, j)) != MTMD_INPUT_CHUNK_TYPE_TEXT) {
            j++;
        }
        size_t run_len = j - i;

        bool batch_ready = false;
        std::unique_ptr<mtmd_batch, decltype(batch_deleter)> batch(nullptr, batch_deleter);
        if (run_len >= 2) {
            batch.reset(mtmd_batch_init(mtmd_ctx));
            batch_ready = true;
            for (size_t k = i; k < j && batch_ready; k++) {
                batch_ready = (mtmd_batch_add_chunk(batch.get(), mtmd_input_chunks_get(chunks, k)) == 0);
            }
            if (batch_ready) {
                batch_ready = (mtmd_batch_encode(batch.get()) == 0);
            }
        }

        if (batch_ready) {
            // Encoding succeeded for the whole run — n_past hasn't moved yet,
            // so a decode failure from here on is treated the same as a
            // single-chunk decode failure: fatal, no retry (re-decoding would
            // double-advance n_past for already-decoded chunks in this run).
            for (size_t k = i; k < j; k++) {
                const mtmd_input_chunk* mchunk = mtmd_input_chunks_get(chunks, k);
                float* embd = mtmd_batch_get_output_embd(batch.get(), mchunk);
                if (!embd) {
                    *new_n_past = n_past;
                    return 1;
                }
                int32_t res = mtmd_helper_decode_image_chunk(
                    mtmd_ctx, lctx, mchunk, embd, n_past, seq_id, n_batch, &n_past, nullptr, nullptr);
                if (res != 0) {
                    *new_n_past = n_past;
                    return res;
                }
            }
            i = j;
            continue;
        }

        // Couldn't batch this run (lone chunk, or the backend rejected
        // batching before any decode happened) — fall back to encoding each
        // chunk one at a time, identical to mtmd_helper_eval_chunks.
        for (size_t k = i; k < j; k++) {
            const mtmd_input_chunk* mchunk = mtmd_input_chunks_get(chunks, k);
            bool last = (k == n_chunks - 1) && logits_last;
            int32_t res = mtmd_helper_eval_chunk_single(mtmd_ctx, lctx, mchunk, n_past, seq_id, n_batch, last, &n_past);
            if (res != 0) {
                *new_n_past = n_past;
                return res;
            }
        }
        i = j;
    }

    *new_n_past = n_past;
    return 0;
}

// Token count of the first message when it is a system message whose own render is a token
// prefix of the prompt; -1 otherwise (a shift then keeps half the context).
static int32_t rn_system_prefix_end(
    const rn_llama_context * rn_ctx,
    const common_chat_templates_inputs & inputs,
    const llama_tokens & prompt_tokens) {
    if (inputs.messages.empty() || inputs.messages.front().role != "system") {
        return -1;
    }
    common_chat_templates_inputs head = inputs;
    head.messages.resize(1);
    head.add_generation_prompt = false;
    try {
        const auto partial = common_chat_templates_apply(rn_ctx->chat_templates.get(), head);
        const auto tokens = rn_tokenize_chat_prompt(rn_ctx->vocab, partial.prompt, true);
        return rn_is_token_prefix(tokens, prompt_tokens) ? static_cast<int32_t>(tokens.size()) : -1;
    } catch (...) {
        // Some templates refuse a conversation with no user message.
        return -1;
    }
}

CompletionResult run_chat_completion(
    rn_llama_context* rn_ctx,
    const CompletionOptions& options,
    rn_stream_callback callback) {

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
            media_items = extract_media_from_messages(messages_json, mtmd_get_marker(rn_ctx->mtmd_ctx));
            has_media = !media_items.empty();
        }
        const json& effective_messages = has_media ? messages_json : options.messages;

        // Parse messages directly from options
        std::vector<common_chat_msg> chat_msgs;
        if (!effective_messages.is_null() && !effective_messages.empty()) {
            // common_chat_msgs_parse_oaicompat now takes common_json (upstream #27511), not
            // nlohmann::ordered_json. No implicit conversion exists in either direction —
            // round-trip through dump()/parse() at this boundary.
            chat_msgs = common_chat_msgs_parse_oaicompat(common_json::parse(effective_messages.dump()));
        }

        // Tools and tool_choice are parsed before any KV change, so a bad request leaves the
        // cache and kv_tokens as they were. tool_choice is parsed against the tools
        // (server: oaicompat_chat_params_parse). A named function must match exactly one
        // supplied tool; upstream throws std::invalid_argument otherwise. Null / "" = "auto".
        std::vector<common_chat_tool> parsed_tools;
        if (!options.tools.is_null() && !options.tools.empty()) {
            parsed_tools = common_chat_tools_parse_oaicompat(common_json::parse(options.tools.dump()));
        }
        common_chat_tool_choice parsed_tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
        const bool tool_choice_set = !options.tool_choice.is_null() &&
            !(options.tool_choice.is_string() && options.tool_choice.get<std::string>().empty());
        if (tool_choice_set) {
            try {
                parsed_tool_choice = common_chat_tool_choice_parse_oaicompat(
                    common_json::parse(options.tool_choice.dump()), parsed_tools);
            } catch (const std::exception & e) {
                result.success    = false;
                result.error_msg  = std::string("Invalid tool_choice: ") + e.what();
                result.error_type = RN_ERROR_INVALID_PARAM;
                return result;
            }
        }
        if (!parsed_tools.empty() && parsed_tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE && !options.grammar.empty()) {
            result.success    = false;
            result.error_msg  = "Cannot use custom grammar constraints with tools.";
            result.error_type = RN_ERROR_INVALID_PARAM;
            return result;
        }

        const std::string kv_render_identity = build_kv_render_identity(rn_ctx, options);

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
            template_inputs.tools = std::move(parsed_tools);
            template_inputs.parallel_tool_calls = rn_ctx->params.parallel_tool_calls;
        }
        template_inputs.tool_choice = parsed_tool_choice;

        CompletionOptions cmpl_options = options;

        // The session owns prompt parts, grammar, lazy triggers, preserved tokens, stops,
        // thinking tags and the PEG parser (server: oaicompat_chat_params_parse).
        common_chat_session session = rn_make_chat_session(rn_ctx, template_inputs);
        cmpl_options.chat_session = &session;
        cmpl_options.grammar.clear();

        // Streaming parses a copy, so the final session.finish below stays one clean parse.
        std::optional<common_chat_session> stream_session;
        if (options.stream) {
            stream_session.emplace(session);
            cmpl_options.stream_session = &*stream_session;
        }

        for (const auto & stop : session.additional_stops()) {
            cmpl_options.stop.push_back(stop);
        }

        if (has_media) {
            // mtmd_tokenize takes a single text with one parse_special flag, so special tokens
            // are parsed in the whole prompt on media turns.
            cmpl_options.prompt = session.prompt().str();
        } else {
            cmpl_options.prompt_tokens = rn_tokenize_chat_prompt(rn_ctx->vocab, session.prompt(), true);
        }

        // Init budget 0 leaves thinking off, so a completion cannot install a cap.
        // A non-zero init budget may be replaced for this call by a positive request.
        // 0 and -1 on the request do not change the template or the init cap.
        const bool has_thinking_end = !session.thinking_end_tags().empty();
        const rn_reasoning_cap reasoning_cap = rn_select_reasoning_cap(
            rn_ctx->params.reasoning_budget, options.reasoning_budget, has_thinking_end);
        const bool budget_active = reasoning_cap.active;
        if (has_thinking_end && rn_ctx->params.reasoning_budget != 0) {
            cmpl_options.reasoning_budget_tokens    = budget_active
                ? reasoning_cap.tokens
                : rn_ctx->params.reasoning_budget;
            cmpl_options.reasoning_budget_start_tag = session.thinking_start_tag();
            cmpl_options.reasoning_budget_end_tags  = session.thinking_end_tags();
        }

        // --- KV cache prefix reuse (server: cache_prompt) ---
        // Decided once the prompt is rendered and tokenized, so a request that fails before this
        // point leaves the cache as it was. The cache keeps the longest common token prefix of
        // what it holds (kv_tokens: the last prompt and its reply) and this prompt. That is exact
        // by construction: no message ids, and no template has to render a message the same
        // way twice (Qwen3 drops an earlier turn's think block once a new user message follows).
        // First cause wins. A later full clear is the consequence, not a second reason.
        std::string cleared = "none";
        auto note_clear = [&](const char * reason) {
            if (cleared == "none") {
                cleared = reason;
            }
        };
        auto clear_kv = [&](const char * reason) {
            llama_memory_clear(llama_get_memory(rn_ctx->ctx), true);
            rn_ctx->kv_tokens.clear();
            note_clear(reason);
        };
        if (has_media) {
            clear_kv("media");
        }
        if (options.reset_kv_cache) {
            clear_kv("reset");
        }
        if (!rn_ctx->kv_tokens.empty() &&
            !rn_ctx->kv_render_identity.empty() &&
            rn_ctx->kv_render_identity != kv_render_identity) {
            clear_kv("render_identity");
        }
        // A prompt_id change drops the prefix. Tools are not part of this id.
        if (!options.prompt_id.empty()) {
            if (!rn_ctx->last_prompt_id.empty() && rn_ctx->last_prompt_id != options.prompt_id) {
                clear_kv("prompt_id");
            }
            rn_ctx->last_prompt_id = options.prompt_id;
        }

        auto drop_kv_from = [&](llama_pos pos) -> bool {
            if (rn_ctx->kv_seq_rm) {
                return rn_ctx->kv_seq_rm(rn_ctx->kv_seq_rm_data, pos, -1);
            }
            return llama_memory_seq_rm(llama_get_memory(rn_ctx->ctx), 0, pos, -1);
        };

        int32_t kv_hint_pos = 0; // default: full encode from position 0
        if (!has_media) {
            // At most prompt size - 1: the last prompt token is always encoded for fresh logits.
            const size_t reuse = rn_reuse_len(rn_ctx->kv_tokens, cmpl_options.prompt_tokens);
            if (reuse == 0) {
                clear_kv("no_prefix");
            } else if (reuse < rn_ctx->kv_tokens.size()) {
                // Drop the cache after the common prefix (the previous reply, or what changed).
                // seq_rm returns false on recurrent models and some GPU backends.
                // Same split the server records at startup. FULL and NO cannot drop a suffix.
                const bool can_drop_suffix =
                    rn_ctx->kv_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_PART ||
                    rn_ctx->kv_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS;
                if (can_drop_suffix && drop_kv_from(static_cast<llama_pos>(reuse))) {
                    rn_ctx->kv_tokens.resize(reuse);
                    kv_hint_pos = static_cast<int32_t>(reuse);
                } else {
                    clear_kv("seq_rm");
                }
            } else {
                // The cache is exactly a prefix of this prompt: nothing to drop.
                kv_hint_pos = static_cast<int32_t>(reuse);
            }
        }
        cmpl_options.kv_hint_pos = kv_hint_pos;

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
            input_text.text_len      = cmpl_options.prompt.size();
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

            // Multiple images/audio chunks: batch their encoder passes
            // together (mtmd_eval_chunks_batched). Single image (the
            // dominant case) or text-only: keep the existing, unmodified
            // upstream helper.
            llama_pos new_n_past = 0;
            int32_t eval_res = (count_media_chunks(chunks) >= 2)
                ? mtmd_eval_chunks_batched(
                    rn_ctx->mtmd_ctx, rn_ctx->ctx, chunks,
                    0, 0, rn_ctx->params.n_batch, true, &new_n_past)
                : mtmd_helper_eval_chunks(
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

        // A shift keeps the system message when it is first and its own render is a token prefix
        // of the prompt. Only a call with no reply cap can shift: run_completion caps a positive
        // n_predict to the free context.
        const bool may_shift = cmpl_options.n_predict < 0 && rn_ctx->params.n_predict < 0;
        if (cmpl_options.n_keep == 0 && !has_media && may_shift) {
            cmpl_options.kv_prefix_end = rn_system_prefix_end(rn_ctx, template_inputs, cmpl_options.prompt_tokens);
        }

        // Run standard completion with the processed prompt
        result = run_completion(rn_ctx, cmpl_options, callback);
        result.kv_cleared_reason = cleared;
        result.reasoning_budget_active = budget_active;
        // kv_tokens now describes the cache (this prompt and its reply), or is empty.
        rn_ctx->kv_render_identity = rn_ctx->kv_tokens.empty() ? std::string() : kv_render_identity;

        if (result.success) {
            // One full parse of the final output (server: task_result_state::update with is_partial=false).
            // Splits reasoning from content even without tools.
            common_chat_msg parsed_msg;
            bool has_parsed_content = false;
            if (!result.generated.text.empty()) {
                try {
                    // One finish of the token-aligned reply. Do not re-tokenize result.content.
                    parsed_msg = session.finish(result.generated);
                    has_parsed_content = true;
                } catch (const std::exception& e) {
                    if (!template_inputs.tools.empty()) {
                        result.tool_call_parse_failed = true;
                        result.tool_call_parse_error = e.what();
                    }
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
                // Use the server.cpp approach: let the common_chat_msg handle the JSON conversion.
                // to_json_oaicompat() now returns common_json (upstream #27511); round-trip through
                // dump()/parse() back into our nlohmann::ordered_json ("json" alias, rn-utils.h:28).
                choice["message"] = json::parse(parsed_msg.to_json_oaicompat().dump());
                choice["finish_reason"] = "tool_calls";
            } else if (result.tool_call_parse_failed) {
                choice["message"]["content"] = result.content;
                choice["finish_reason"] = "tool_call_parse_error";
                choice["tool_call_parse_error"] = result.tool_call_parse_error;
            } else if (has_parsed_content &&
                       (!parsed_msg.content.empty() || !parsed_msg.reasoning_content.empty())) {
                choice["message"]["content"] = parsed_msg.content;
                if (!parsed_msg.reasoning_content.empty()) {
                    choice["message"]["reasoning_content"] = parsed_msg.reasoning_content;
                }
            } else {
                // Fallback to raw content if parsing failed or no tools
                choice["message"]["content"] = result.content;
            }

            choices.push_back(choice);
            response["choices"] = choices;

            // Add usage information. prompt_tokens stays the full rendered prompt.
            response["usage"] = {
                {"prompt_tokens", result.n_prompt_tokens},
                {"completion_tokens", result.n_predicted_tokens},
                {"total_tokens", result.n_prompt_tokens + result.n_predicted_tokens}
            };
            // prompt_n is tokens evaluated into KV this call, not the full prompt length.
            response["timings"] = {
                {"prompt_n", result.timings.prompt_n},
                {"prompt_ms", result.timings.prompt_ms},
                {"predicted_n", result.timings.predicted_n},
                {"predicted_ms", result.timings.predicted_ms},
                {"total_ms", result.timings.total_ms}
            };
            response["context_shifted"] = result.context_shifted;
            response["kv_cleared_reason"] = result.kv_cleared_reason;
            response["reasoning_budget_active"] = result.reasoning_budget_active;

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

#include "rn-llama.h"
// Suppress unused function warnings from llama.cpp headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "common.h"
#include "chat.h"
#include "llama.h"
#include "sampling.h"
#pragma GCC diagnostic pop
#include "rn-utils.h"

#include <string>
#include <vector>
#include <thread>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <random>

namespace facebook::react {

// Struct to track prediction completion state
struct completion_state {
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

    common_sampler* sampler = nullptr;
    std::vector<std::string> antiprompt;

    // Chat format and tools info
    common_chat_format chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    common_chat_tool_choice tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;

    ~completion_state() {
        if (sampler) {
            common_sampler_free(sampler);
            sampler = nullptr;
        }
    }
};

// Helper function to check for stopping criteria
static bool check_stop_conditions(
    completion_state& state,
    const std::vector<std::string>& stop_words,
    const std::string& token_text,
    bool ignore_eos) {

    if (state.n_remaining <= 0) {
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
        state.sampler = common_sampler_init(rn_ctx->model, sampling_params);
        if (!state.sampler) {
            result.success = false;
            result.error_msg = "Failed to initialize sampler";
            result.error_type = RN_ERROR_INFERENCE;
            return result;
        }

        // Stop words
        state.antiprompt = options.stop;

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
        state.n_predict = options.n_predict > 0 ? options.n_predict : params.n_predict;
        state.n_remaining = state.n_predict;

        // Guard: prompt must fit in the context window, otherwise llama_decode will ggml_abort.
        if (static_cast<int>(state.prompt_tokens.size()) >= state.n_ctx) {
            result.success = false;
            result.error_msg = "Prompt too long: " + std::to_string(state.prompt_tokens.size())
                + " tokens exceeds context size " + std::to_string(state.n_ctx);
            result.error_type = RN_ERROR_INVALID_PARAM;
            return result;
        }

        // Encode prompt tokens into the KV cache.
        if (state.n_past > 0) {
            // Prefix reuse: only encode prompt_tokens[n_past:] at positions [n_past, n_past+1, ...]
            const int n_new = static_cast<int>(state.prompt_tokens.size()) - state.n_past;
            if (n_new > 0) {
                llama_batch new_batch = llama_batch_init(rn_ctx->params.n_batch, 0, 1);
                for (int i = 0; i < n_new; ) {
                    common_batch_clear(new_batch);
                    int chunk = std::min(rn_ctx->params.n_batch, n_new - i);
                    bool last_chunk = (i + chunk >= n_new);
                    for (int j = 0; j < chunk; j++) {
                        common_batch_add(new_batch,
                            state.prompt_tokens[state.n_past + i + j],
                            state.n_past + i + j,
                            {0},
                            last_chunk && (j == chunk - 1));
                    }
                    if (llama_decode(rn_ctx->ctx, new_batch) != 0) {
                        llama_batch_free(new_batch);
                        result.success = false;
                        result.error_msg = "Failed to process prompt";
                        result.error_type = RN_ERROR_INFERENCE;
                        return result;
                    }
                    i += chunk;
                }
                llama_batch_free(new_batch);
                state.n_past = static_cast<int>(state.prompt_tokens.size());
            }
        } else {
            // Standard path: encode all tokens from position 0.
            // We use our own chunked loop (identical to the prefix-reuse path above) so that
            // prompts longer than n_batch are split correctly. common_prompt_batch_decode passes
            // all tokens in a single llama_batch_get_one call which aborts when n_tokens > n_batch.
            const int n_total = static_cast<int>(state.prompt_tokens.size());
            llama_batch batch = llama_batch_init(rn_ctx->params.n_batch, 0, 1);
            for (int i = 0; i < n_total; ) {
                common_batch_clear(batch);
                int chunk = std::min(rn_ctx->params.n_batch, n_total - i);
                bool last_chunk = (i + chunk >= n_total);
                for (int j = 0; j < chunk; j++) {
                    common_batch_add(batch, state.prompt_tokens[i + j], i + j,
                                     {0}, last_chunk && (j == chunk - 1));
                }
                if (llama_decode(rn_ctx->ctx, batch) != 0) {
                    llama_batch_free(batch);
                    result.success = false;
                    result.error_msg = "Failed to process prompt";
                    result.error_type = RN_ERROR_INFERENCE;
                    return result;
                }
                i += chunk;
            }
            llama_batch_free(batch);
            state.n_past = n_total;
        }

        // Accept all prompt tokens into the sampler (needed for repetition/presence penalty).
        // For non-lazy grammars we skip this and re-init below to keep a clean grammar state.
        for (auto tok : state.prompt_tokens) {
            if (common_grammar_value(sampling_params.grammar).empty() || sampling_params.grammar_lazy) {
                common_sampler_accept(state.sampler, tok, true);
            }
        }

        result.n_prompt_tokens = state.prompt_tokens.size();

        // For non-lazy grammars, re-init the sampler so grammar state is clean for generation.
        if (!common_grammar_value(sampling_params.grammar).empty() && !sampling_params.grammar_lazy) {
            common_sampler_free(state.sampler);
            state.sampler = common_sampler_init(rn_ctx->model, sampling_params);
        }

        // Allocate a single-token batch once and reuse it across the entire generation loop.
        llama_batch gen_batch = llama_batch_init(1, 0, 1);

        while (state.has_next_token && state.n_remaining > 0) {
            // Sample the next token
            llama_token token_id = common_sampler_sample(state.sampler, rn_ctx->ctx, -1);

            // Extract the token text
            std::string token_text = common_token_to_piece(rn_ctx->vocab, token_id);

            // Add to generated text
            state.generated_text += token_text;
            state.generated_tokens.push_back(token_id);

            // Update state
            state.n_decoded++;
            state.n_remaining--;

            // Accept the new token into the sampler
            common_sampler_accept(state.sampler, token_id, true);

            // Decode the generated token to prepare logits for the next sample
            common_batch_clear(gen_batch);
            common_batch_add(gen_batch, token_id, state.n_past, {0}, true);
            if (llama_decode(rn_ctx->ctx, gen_batch) != 0) {
                llama_batch_free(gen_batch);
                result.success = false;
                result.error_msg = "Failed to decode generated token";
                result.error_type = RN_ERROR_INFERENCE;
                return result;
            }

            state.n_past++;

            // Check stopping conditions
            bool should_stop = check_stop_conditions(state, state.antiprompt, token_text, options.ignore_eos);

            // Stream the token if callback is provided
            if (callback && !should_stop) {
                std::string text_to_send = state.generated_text.substr(state.n_sent_text);
                state.n_sent_text = state.generated_text.size();

                if (!callback(text_to_send, false)) {
                    // Callback returned false — caller wants to stop
                    state.has_next_token = false;
                    break;
                }
            }

            if (should_stop) {
                break;
            }

            // Check for EOS token if not ignoring
            if (!options.ignore_eos && token_id == llama_vocab_eos(rn_ctx->vocab)) {
                state.has_next_token = false;
                break;
            }
        }

        llama_batch_free(gen_batch);

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
        // Parse messages directly from options
        std::vector<common_chat_msg> chat_msgs;
        if (!options.messages.is_null() && !options.messages.empty()) {
            chat_msgs = common_chat_msgs_parse_oaicompat(options.messages);
        }

        // --- Per-message KV cache prefix reuse ---
        // Extract optional "id" fields from each message in the JSON array.
        // IDs are an optional caller-supplied field (not part of the OpenAI spec);
        // they let the native layer skip re-encoding messages that haven't changed
        // without doing a full token-by-token comparison.
        std::vector<std::string> msg_ids;
        if (!options.messages.is_null() && options.messages.is_array()) {
            for (const auto& msg : options.messages) {
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
        if (!msg_ids.empty() && rn_ctx->kv_has_messages && !options.reset_kv_cache) {
            const auto& cached = rn_ctx->kv_messages;
            while (kv_match_count < msg_ids.size()
                   && kv_match_count < cached.size()
                   && !msg_ids[kv_match_count].empty()
                   && msg_ids[kv_match_count] == cached[kv_match_count].id) {
                kv_match_count++;
            }
        }

        // Decide the KV common position and evict stale entries.
        int32_t kv_hint_pos = 0; // default: full encode from position 0
        if (kv_match_count > 0) {
            const int32_t kv_common_len = rn_ctx->kv_messages[kv_match_count - 1].token_end;
            if (kv_match_count < rn_ctx->kv_messages.size()) {
                // Some cached messages are no longer present — evict beyond the common prefix.
                if (!llama_memory_seq_rm(llama_get_memory(rn_ctx->ctx), 0, kv_common_len, -1)) {
                    // seq_rm failed (e.g. recurrent model): fall back to full clear.
                    llama_memory_clear(llama_get_memory(rn_ctx->ctx), true);
                    kv_hint_pos = 0;
                } else {
                    kv_hint_pos = kv_common_len;
                }
            } else {
                // All cached messages still match and new messages are being appended.
                // KV entries at [0..kv_common_len) are fully valid — no eviction needed.
                kv_hint_pos = kv_common_len;
            }
        } else {
            // No matching prefix (or no IDs provided): full clear.
            llama_memory_clear(llama_get_memory(rn_ctx->ctx), true);
            kv_hint_pos = 0;
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

        // Apply the jinja chat template.
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
        common_chat_params chat_params;
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

        CompletionOptions cmpl_options = options;
        cmpl_options.prompt = chat_params.prompt;
        // Pass the trusted KV start position so run_completion skips its own KV management.
        cmpl_options.kv_hint_pos = kv_hint_pos;

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

            // Note: Debug logging removed as it was not being used
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
                // Apply the template to messages[0..k] without a generation prompt to get
                // the token boundary at the end of this message.
                common_chat_templates_inputs tinput;
                tinput.messages               = std::vector<common_chat_msg>(chat_msgs.begin(),
                                                                              chat_msgs.begin() + k + 1);
                tinput.add_generation_prompt  = false;
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
        } else if (options.reset_kv_cache) {
            rn_ctx->kv_messages.clear();
            rn_ctx->kv_has_messages = false;
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

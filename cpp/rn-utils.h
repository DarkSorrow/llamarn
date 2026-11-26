#pragma once

// Suppress unused function warnings from llama.cpp headers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "common.h"
#include "llama.h"
#include "sampling.h"
#pragma GCC diagnostic pop

// Change JSON_ASSERT from assert() to GGML_ASSERT:
#define JSON_ASSERT GGML_ASSERT
#include "nlohmann/json.hpp"
#include "base64.hpp"
#include "chat.h"

#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <limits>
#include <optional>

using json = nlohmann::ordered_json;

#define DEFAULT_OAICOMPAT_MODEL "llamacpp"
// build_info is defined in common.h — no redefinition needed here.

// Error types simplified for a library context
enum rn_error_type {
    RN_ERROR_INVALID_PARAM,  // Invalid parameters provided
    RN_ERROR_MODEL_LOAD,     // Failed to load model
    RN_ERROR_CONTEXT,        // Context-related errors
    RN_ERROR_INFERENCE,      // Inference-related errors
    RN_ERROR_GENERAL         // General errors
};

// CompletionOptions struct to represent parameters for completion requests
//
// Sampling fields use sentinel values to distinguish "not set by caller" from "explicitly set":
//   float fields: NaN  → use the model's initLlama default (params.sampling)
//   int fields:   -1   → use the model's initLlama default
//
// This means a caller that omits temperature gets the model's configured default,
// not a hardcoded fallback. Models with GGUF-embedded sampling metadata benefit
// automatically, and per-model tuning via initLlama is respected.
struct CompletionOptions {
    std::string prompt;  // for simple completions
    std::string model;   // model identifier
    json messages;       // for chat completions
    bool stream = false;
    int n_predict = -1;  // -1 = unlimited; generation stops on EOS or stop strings

    // Sampling — NaN means "not set, use model default from initLlama"
    float temperature       = std::numeric_limits<float>::quiet_NaN();
    float top_p             = std::numeric_limits<float>::quiet_NaN();
    float top_k             = std::numeric_limits<float>::quiet_NaN();
    float min_p             = std::numeric_limits<float>::quiet_NaN();
    float presence_penalty  = std::numeric_limits<float>::quiet_NaN();
    float repeat_penalty    = std::numeric_limits<float>::quiet_NaN();
    int   repeat_last_n     = -1;   // -1 = use model default
    float frequency_penalty = std::numeric_limits<float>::quiet_NaN();
    int n_keep = 0;
    std::vector<std::string> stop;
    std::string grammar;
    bool grammar_lazy = false;
    bool ignore_eos = false;
    int seed = -1;  // -1 = leave model's default seed unchanged; >=0 overrides (applied in run_completion)
    json tools;         // tools for function calling
    // OpenAI tool_choice, kept as JSON: "auto" | "none" | "required" |
    // {"type":"function","function":{"name":...}} | {"type":"allowed_tools",...}.
    // Parsed against the tools by common_chat_tool_choice_parse_oaicompat in run_chat_completion.
    json tool_choice = "auto";

    // Internal: set by run_chat_completion.
    // prompt_tokens: part-aware tokenized chat prompt; when non-empty it replaces tokenizing `prompt`.
    // chat_session: owns grammar, triggers and preserved tokens; applied to the sampler via apply_sampling.
    llama_tokens prompt_tokens;
    const common_chat_session * chat_session = nullptr;
    // stream_session: a copy of chat_session, fed while streaming (feed() mutates it) so the
    // final session.finish(result.generated) on chat_session stays a single clean parse.
    // Null when the call does not stream.
    common_chat_session * stream_session = nullptr;

    // KV cache control
    bool    reset_kv_cache = false; // force a full KV cache clear even when the tokens match

    // Internal: set by run_chat_completion from the common token prefix with rn_llama_context::kv_tokens.
    // When >= 0, run_completion keeps the cache below it and encodes the prompt from there.
    int32_t kv_hint_pos = -1;

    // Internal: token count of the system message when it is first and a token prefix of the
    // prompt, or -1. A context shift with n_keep == 0 keeps it instead of half the context.
    int32_t kv_prefix_end = -1;

    // Set by run_chat_completion after mtmd_helper_eval_chunks so run_completion
    // skips its own encode step (prompt + images already in KV cache, logits ready).
    int32_t mtmd_encoded_n_past = -1;

    // Optional. A change from the previous non-empty value clears KV. Empty does not.
    // Tools are not part of this id; they live in the render identity.
    std::string prompt_id;

    // Per-call sampler cap. Unset means "use the initLlama budget".
    // A positive value applies only when init turned thinking on. 0 and -1 do not.
    std::optional<int32_t> reasoning_budget;

    // Reasoning budget: passed from the chat session's thinking tags after template apply.
    // Matches common/sampling.cpp: -1 = no budget sampler (unlimited thinking),
    // >= 0 = token budget. Thinking itself is turned off at the template level
    // (enable_thinking=false) when the JS reasoning_budget is 0.
    int32_t                  reasoning_budget_tokens  = -1;
    std::string              reasoning_budget_start_tag;  // e.g. "<think>"
    std::vector<std::string> reasoning_budget_end_tags;   // e.g. {"</think>"} — plural for parity
                                                           // with upstream's multi-end-tag support
                                                           // (common_chat_params::thinking_end_tags /
                                                           // common_params_sampling::reasoning_budget_end)
    std::string              reasoning_budget_message;    // injected before end tag when budget exhausted
};

struct CompletionTimings {
    int32_t predicted_n  = 0;
    double  predicted_ms = 0.0;
    int32_t prompt_n     = 0;
    double  prompt_ms    = 0.0;
    double  total_ms     = 0.0;
};

// One streamed piece. text is the raw generated text since the previous chunk (stop-string safe).
// On chat calls, content and reasoning_content are the parser deltas over the same range
// (common_chat_msg_diff, as the server streams them). They lag text by at most one parse
// interval and are complete by the last chunk before is_done. Prompt-only calls leave them empty.
struct rn_stream_chunk {
    std::string text;
    std::string content;
    std::string reasoning_content;
};

// Returns false to stop generation. is_done is true exactly once, with an empty chunk.
using rn_stream_callback = std::function<bool(const rn_stream_chunk&, bool is_done)>;

// CompletionResult struct to hold completion response data
struct CompletionResult {
    std::string content;
    bool success = true;
    std::string error_msg;
    rn_error_type error_type = RN_ERROR_GENERAL;
    int n_prompt_tokens = 0;
    int n_predicted_tokens = 0;
    std::vector<llama_token> tokens;
    json chat_response;
    CompletionTimings timings;
    bool stopped_by_length = false;
    bool tool_call_parse_failed = false;
    std::string tool_call_parse_error;
    bool context_shifted = false;  // true if context shift occurred during generation
    int shift_n_keep = 0;          // keep count used by the shift, 0 when the call did not shift
    // Why this call encoded from the start of the cache. "none" means a prefix hit.
    std::string kv_cleared_reason = "none";
    // True only when this call installed a positive thinking-token cap.
    bool reasoning_budget_active = false;
    // Reply as the server keeps it: pieces plus the token id that produced each one.
    // session.finish reads this. Do not rebuild it from `content`.
    common_chat_input generated;
};

// Utility functions

template <typename T>
static T json_value(const json & body, const std::string & key, const T & default_value) {
    // Fallback null to default value
    if (body.contains(key) && !body.at(key).is_null()) {
        try {
            return body.at(key);
        } catch (NLOHMANN_JSON_NAMESPACE::detail::type_error const &) {
            return default_value;
        }
    } else {
        return default_value;
    }
}

inline std::string gen_chatcmplid() {
    static const std::string chars("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
    static thread_local std::mt19937_64 gen(std::random_device{}());
    static thread_local std::uniform_int_distribution<size_t> dist(0, chars.size() - 1);
    std::string result(29, ' ');
    for (auto & c : result) c = chars[dist(gen)];
    return "chatcmpl-" + result;
}

inline bool ends_with(const std::string & str, const std::string & suffix) {
    return str.size() >= suffix.size() && 0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

inline size_t find_partial_stop_string(const std::string &stop, const std::string &text) {
    if (!text.empty() && !stop.empty()) {
        const char text_last_char = text.back();
        for (int64_t char_index = stop.size() - 1; char_index >= 0; char_index--) {
            if (stop[char_index] == text_last_char) {
                const std::string current_partial = stop.substr(0, char_index + 1);
                if (ends_with(text, current_partial)) {
                    return text.size() - char_index - 1;
                }
            }
        }
    }

    return std::string::npos;
}

inline bool json_is_array_of_numbers(const json & data) {
    if (data.is_array()) {
        for (const auto & e : data) {
            if (!e.is_number_integer()) {
                return false;
            }
        }
        return true;
    }
    return false;
}

// is array having BOTH numbers & strings?
inline bool json_is_array_of_mixed_numbers_strings(const json & data) {
    bool seen_string = false;
    bool seen_number = false;
    if (data.is_array()) {
        for (const auto & e : data) {
            seen_string |= e.is_string();
            seen_number |= e.is_number_integer();
            if (seen_number && seen_string) {
                return true;
            }
        }
    }
    return false;
}

/**
 * this handles 2 cases:
 * - only string, example: "string"
 * - mixed string and tokens, example: [12, 34, "string", 56, 78]
 */
inline llama_tokens tokenize_mixed(const llama_vocab * vocab, const json & json_prompt, bool add_special, bool parse_special) {
    // If `add_bos` is true, we only add BOS, when json_prompt is a string,
    // or the first element of the json_prompt array is a string.
    llama_tokens prompt_tokens;

    if (json_prompt.is_array()) {
        bool first = true;
        for (const auto & p : json_prompt) {
            if (p.is_string()) {
                auto s = p.template get<std::string>();

                llama_tokens p;
                if (first) {
                    p = common_tokenize(vocab, s, add_special, parse_special);
                    first = false;
                } else {
                    p = common_tokenize(vocab, s, false, parse_special);
                }

                prompt_tokens.insert(prompt_tokens.end(), p.begin(), p.end());
            } else {
                if (first) {
                    first = false;
                }

                prompt_tokens.push_back(p.template get<llama_token>());
            }
        }
    } else {
        auto s = json_prompt.template get<std::string>();
        prompt_tokens = common_tokenize(vocab, s, add_special, parse_special);
    }

    return prompt_tokens;
}

// Tokenize a rendered chat prompt part by part: special tokens are parsed only in
// template parts, never in user-input parts (is_input), so typed control tokens stay text.
inline llama_tokens rn_tokenize_chat_prompt(const llama_vocab * vocab, const common_chat_prompt & prompt, bool add_special) {
    llama_tokens tokens;
    bool first = true;
    for (const auto & part : prompt.parts) {
        if (part.text.empty()) {
            continue;
        }
        auto part_tokens = common_tokenize(vocab, part.text, first && add_special, !part.is_input);
        tokens.insert(tokens.end(), part_tokens.begin(), part_tokens.end());
        first = false;
    }
    return tokens;
}

inline void rn_batch_clear(llama_batch & batch) {
    batch.n_tokens = 0;
}

inline void rn_batch_add(llama_batch & batch, llama_token id, llama_pos pos, llama_seq_id seq_id, bool logits) {
    const int32_t i = batch.n_tokens;
    batch.token[i]     = id;
    batch.pos[i]       = pos;
    batch.n_seq_id[i]  = 1;
    batch.seq_id[i][0] = seq_id;
    batch.logits[i]    = logits;
    batch.n_tokens++;
}

/**
 * break the input "prompt" object into multiple prompt if needed, then tokenize them
 * this supports these cases:
 * - "prompt": "string"
 * - "prompt": [12, 34, 56]
 * - "prompt": [12, 34, "string", 56, 78]
 * and multiple prompts (multi-tasks):
 * - "prompt": ["string1", "string2"]
 * - "prompt": ["string1", [12, 34, 56]]
 * - "prompt": [[12, 34, 56], [78, 90, 12]]
 * - "prompt": [[12, 34, "string", 56, 78], [12, 34, 56]]
 */
inline std::vector<llama_tokens> tokenize_input_prompts(const llama_vocab * vocab, const json & json_prompt, bool add_special, bool parse_special) {
    std::vector<llama_tokens> result;
    if (json_prompt.is_string() || json_is_array_of_mixed_numbers_strings(json_prompt)) {
        // string or mixed
        result.push_back(tokenize_mixed(vocab, json_prompt, add_special, parse_special));
    } else if (json_is_array_of_numbers(json_prompt)) {
        // array of tokens
        result.push_back(json_prompt.get<llama_tokens>());
    } else if (json_prompt.is_array()) {
        // array of prompts
        result.reserve(json_prompt.size());
        for (const auto & p : json_prompt) {
            if (p.is_string() || json_is_array_of_mixed_numbers_strings(p)) {
                result.push_back(tokenize_mixed(vocab, p, add_special, parse_special));
            } else if (json_is_array_of_numbers(p)) {
                // array of tokens
                result.push_back(p.get<llama_tokens>());
            } else {
                throw std::runtime_error("element of \"prompt\" must be a string, an list of tokens, or a list of mixed strings & tokens");
            }
        }
    } else {
        throw std::runtime_error("\"prompt\" must be a string, an list of tokens, a list of mixed strings & tokens, or a list of prompts");
    }
    if (result.empty()) {
        throw std::runtime_error("\"prompt\" must not be empty");
    }
    return result;
}

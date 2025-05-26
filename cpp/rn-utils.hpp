#pragma once

#include "common.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"

// Change JSON_ASSERT from assert() to GGML_ASSERT:
#define JSON_ASSERT GGML_ASSERT
#include "json.hpp"
#include "base64.hpp"
#include "chat.h"

#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>

using json = nlohmann::ordered_json;

#define DEFAULT_OAICOMPAT_MODEL "gpt-3.5-turbo"
const static std::string build_info("b" + std::to_string(LLAMA_BUILD_NUMBER) + "-" + LLAMA_COMMIT);

// Error types simplified for a library context
enum rn_error_type {
    RN_ERROR_INVALID_PARAM,  // Invalid parameters provided
    RN_ERROR_MODEL_LOAD,     // Failed to load model
    RN_ERROR_CONTEXT,        // Context-related errors
    RN_ERROR_INFERENCE,      // Inference-related errors
    RN_ERROR_GENERAL         // General errors
};

// Forward declaration
struct common_sampler;

// These functions are defined in common/sampling.cpp, only declare them here
common_sampler* common_sampler_init(const llama_model* model, const common_params_sampling& params);
void common_sampler_free(common_sampler* sampler);

// CompletionOptions struct to represent parameters for completion requests
struct CompletionOptions {
    std::string prompt;  // for simple completions
    std::string model;   // model identifier
    json messages;       // for chat completions
    bool stream = false;
    int n_predict = -1;
    float temperature = 0.8f;
    float top_p = 0.9f;
    float top_k = 40.0f;
    float min_p = 0.05f;
    int n_keep = 0;
    int n_probs = 0;  // for log probabilities
    bool post_sampling_probs = false;
    std::vector<std::string> stop;
    std::string grammar;
    bool grammar_lazy = false;
    bool ignore_eos = false;
    std::string chat_template;
    bool use_jinja = false;
    int seed = -1;
    json tools;         // tools for function calling
    std::string tool_choice = "auto"; // tool choice mode: "auto", "none", or "required"

    // Convert to JSON for the completion API
    json to_json() const {
        json j = {
            {"prompt", prompt},
            {"stream", stream},
            {"temperature", temperature},
            {"top_p", top_p},
            {"top_k", top_k},
            {"min_p", min_p},
            {"n_predict", n_predict},
            {"n_keep", n_keep},
            {"n_probs", n_probs},
            {"post_sampling_probs", post_sampling_probs},
            {"stop", stop},
            {"ignore_eos", ignore_eos},
            {"seed", seed}
        };

        if (!model.empty()) {
            j["model"] = model;
        }

        if (!grammar.empty()) {
            j["grammar"] = grammar;
            j["grammar_lazy"] = grammar_lazy;
        }
        // Add tools and tool_choice if available
        if (!tools.empty()) {
            j["tools"] = tools;
            j["tool_choice"] = tool_choice;
        }
        return j;
    }

    // Convert to JSON for the chat completion API
    json to_chat_json() const {
        json data;

        // Add messages if provided
        if (!messages.empty()) {
            data["messages"] = messages;
        }

        // Add model if provided
        if (!model.empty()) {
            data["model"] = model;
        }

        // Add tools if provided
        if (!tools.empty()) {
            data["tools"] = tools;
        }

        // Add tool_choice if provided
        if (!tool_choice.empty()) {
            if (tool_choice == "none" || tool_choice == "auto" || tool_choice == "required") {
                data["tool_choice"] = tool_choice;
            } else {
                // Assume it's a JSON object
                try {
                    data["tool_choice"] = json::parse(tool_choice);
                } catch (...) {
                    // Fall back to string if not valid JSON
                    data["tool_choice"] = tool_choice;
                }
            }
        }

        // Add other parameters
        data["temperature"] = temperature;
        data["top_p"] = top_p;
        data["max_tokens"] = n_predict;
        data["stream"] = stream;

        if (seed >= 0) {
            data["seed"] = seed;
        }

        if (!stop.empty()) {
            data["stop"] = stop;
        }

        if (!chat_template.empty()) {
            data["chat_template"] = chat_template;
        }

        if (!grammar.empty()) {
            data["grammar"] = grammar;
        }

        return data;
    }
};

// CompletionResult struct to hold completion response data
struct CompletionResult {
    std::string content;
    bool success = true;
    std::string error_msg;
    rn_error_type error_type = RN_ERROR_GENERAL;
    int n_prompt_tokens = 0;
    int n_predicted_tokens = 0;
    std::vector<llama_token> tokens;

    // For chat completions, store the parsed OAI-compatible response
    json chat_response;
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

static std::string gen_chatcmplid() {
    static const std::string str("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
    std::random_device rd;
    std::mt19937 generator(rd());
    std::string result(32, ' ');
    for (int i = 0; i < 32; ++i) {
        result[i] = str[generator() % str.size()];
    }
    return "chatcmpl-" + result;
}

static bool ends_with(const std::string & str, const std::string & suffix) {
    return str.size() >= suffix.size() && 0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

static size_t find_partial_stop_string(const std::string &stop, const std::string &text) {
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

static bool json_is_array_of_numbers(const json & data) {
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
static bool json_is_array_of_mixed_numbers_strings(const json & data) {
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
static llama_tokens tokenize_mixed(const llama_vocab * vocab, const json & json_prompt, bool add_special, bool parse_special) {
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
static std::vector<llama_tokens> tokenize_input_prompts(const llama_vocab * vocab, const json & json_prompt, bool add_special, bool parse_special) {
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

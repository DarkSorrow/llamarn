#include "rn-llama.hpp"
#include "common.h"
#include "log.h"
#include "chat.h"
#include "llama.h"
#include "sampling.h"
#include "rn-utils.hpp"

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
    const llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_model_params* params = nullptr;

    bool stream = false;
    bool has_next_token = true;
    bool has_new_line = false;
    bool truncated = false;

    int n_past = 0;
    int n_ctx = 0;
    int n_predict = 0;
    int n_decoded = 0;
    int n_remaining = 0;

    size_t n_sent_text = 0;
    size_t last_nl_pos = 0;

    std::string prompt;
    std::string generated_text;
    std::string stopping_word;
    bool stop_found = false;

    std::vector<llama_token> prompt_tokens;
    std::vector<llama_token> generated_tokens;

    common_sampler* sampler = nullptr;
    std::vector<std::string> antiprompt; // Storing stop words here

    // Chat format and tools info
    common_chat_format chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;  // Store the chat format for proper parsing
    common_chat_tool_choice tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;  // Default to auto

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
        state.model = rn_ctx->model;
        state.ctx = rn_ctx->ctx;
        state.params = (struct llama_model_params *)&rn_ctx->params.model;
        state.prompt = options.prompt;
        state.chat_format = rn_ctx->params.chat_format;

        // Convert CompletionOptions to JSON for processing
        json data = options.to_json();
        // Prepare the sampling parameters
        const auto& params = rn_ctx->params;

        // Set the prompt
        if (data.contains("prompt")) {
            // Tokenize the prompt
            const auto& tokenized_prompts = tokenize_input_prompts(rn_ctx->vocab, data["prompt"], true, true);
            if (tokenized_prompts.empty() || tokenized_prompts[0].empty()) {
                result.success = false;
                result.error_msg = "Empty prompt";
                result.error_type = RN_ERROR_INVALID_PARAM;
                return result;
            }
            state.prompt_tokens = std::move(tokenized_prompts[0]);
        } else {
            result.success = false;
            result.error_msg = "No prompt provided";
            result.error_type = RN_ERROR_INVALID_PARAM;
            return result;
        }

        // Configure state
        state.n_ctx = llama_n_ctx(rn_ctx->ctx);
        state.n_predict = options.n_predict > 0 ? options.n_predict : params.n_predict;
        state.n_remaining = state.n_predict;

        // Parse tool_choice
        if (options.tool_choice == "auto") {
            state.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
        } else if (options.tool_choice == "none") {
            state.tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;
        } else if (options.tool_choice == "required") {
            state.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        }
        // Initialize the sampler
        state.sampler = common_sampler_init(rn_ctx->model, params.sampling);
        if (!state.sampler) {
            result.success = false;
            result.error_msg = "Failed to initialize sampler";
            result.error_type = RN_ERROR_INFERENCE;
            return result;
        }

        // Process stop words
        if (data.contains("stop")) {
            if (data["stop"].is_string()) {
                state.antiprompt.push_back(data["stop"].get<std::string>());
            } else if (data["stop"].is_array()) {
                for (const auto& stop : data["stop"]) {
                    if (stop.is_string()) {
                        state.antiprompt.push_back(stop.get<std::string>());
                    }
                }
            }
        }

        // Process the prompt
        for (int i = 0; i < (int)state.prompt_tokens.size(); ++i) {
            llama_token token = state.prompt_tokens[i];

            llama_batch batch = {
                /* n_tokens    */ 1,
                /* token       */ &token,
                /* embd        */ nullptr,
                /* pos         */ &i,
                /* n_seq_id    */ nullptr,
                /* seq_id      */ nullptr,
                /* logits      */ nullptr
            };

            if (llama_decode(rn_ctx->ctx, batch) != 0) {
                result.success = false;
                result.error_msg = "Failed to process prompt";
                result.error_type = RN_ERROR_INFERENCE;
                return result;
            }

            common_sampler_accept(state.sampler, token, true);
            state.n_past++;
        }

        result.n_prompt_tokens = state.prompt_tokens.size();

        // Start generating tokens
        const int64_t t_start_generation = ggml_time_us();

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

            // Accept the new token
            common_sampler_accept(state.sampler, token_id, true);

            // Prepare for next token
            llama_batch batch = {
                /* n_tokens    */ 1,
                /* token       */ &token_id,
                /* embd        */ nullptr,
                /* pos         */ &state.n_past,
                /* n_seq_id    */ nullptr,
                /* seq_id      */ nullptr,
                /* logits      */ nullptr
            };

            if (llama_decode(rn_ctx->ctx, batch) != 0) {
                result.success = false;
                result.error_msg = "Failed to decode generated token";
                result.error_type = RN_ERROR_INFERENCE;
                return result;
            }

            state.n_past++;

            // Check stopping conditions
            bool should_stop = check_stop_conditions(state, state.antiprompt, token_text, options.ignore_eos);

            // Handle stream mode
            if (callback && !should_stop) {
                std::string text_to_send = state.generated_text.substr(state.n_sent_text);
                state.n_sent_text = state.generated_text.size();

                // Send the token
                if (!callback(text_to_send, false)) {
                    // Callback returned false, stop generation
                    state.has_next_token = false;
                    break;
                }
            }

            // Check if should stop (after streaming so we don't miss the last token)
            if (should_stop) {
                break;
            }

            // Check for EOS token if not ignoring
            if (!options.ignore_eos && token_id == llama_vocab_eos(rn_ctx->vocab)) {
                state.has_next_token = false;
                break;
            }
        }

        const int64_t t_end_generation = ggml_time_us();
        const double generation_time_ms = (t_end_generation - t_start_generation) / 1000.0;

        // Set the result
        result.content = state.generated_text;
        result.tokens = state.generated_tokens;
        result.n_prompt_tokens = state.prompt_tokens.size();
        result.n_predicted_tokens = state.n_decoded;

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

    if (!rn_ctx || !rn_ctx->model || !rn_ctx->ctx) {
        result.success = false;
        result.error_msg = "Model not initialized";
        result.error_type = RN_ERROR_MODEL_LOAD;
        return result;
    }

    try {
        // Convert chat options to JSON
        json data = options.to_chat_json();

        // Parse messages from options if they exist
        std::vector<common_chat_msg> chat_msgs;
        if (!data["messages"].empty()) {
            chat_msgs = common_chat_msgs_parse_oaicompat(data["messages"]);
        }

        // Apply template
        common_chat_templates_inputs template_inputs;
        template_inputs.messages = chat_msgs;
        template_inputs.add_generation_prompt = true;
        template_inputs.use_jinja = options.use_jinja;
        template_inputs.extract_reasoning = true; // Default to true to extract reasoning content if available

        // Add grammar if present in options
        if (!options.grammar.empty()) {
            template_inputs.grammar = options.grammar;
        }

        // Parse tools if present
        if (data.contains("tools") && !data["tools"].empty()) {
            template_inputs.tools = common_chat_tools_parse_oaicompat(data["tools"]);
            // Check if parallel tool calls are allowed (advanced feature)
            template_inputs.parallel_tool_calls = data.contains("parallel_tool_calls") ?
                json_value(data, "parallel_tool_calls", false) : false;
        }

        // Parse tool_choice if present
        if (data.contains("tool_choice") && !data["tool_choice"].is_null()) {
            template_inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(
                data["tool_choice"].is_string()
                ? data["tool_choice"].get<std::string>()
                : data["tool_choice"].dump());
        }

        // Apply template
        const auto& chat_params = common_chat_templates_apply(rn_ctx->chat_templates.get(), template_inputs);

        // Set up completion options
        CompletionOptions cmpl_options = options;
        cmpl_options.prompt = chat_params.prompt;

        // Apply grammar if needed
        if (!chat_params.grammar.empty()) {
            cmpl_options.grammar = chat_params.grammar;
        }

        // Run standard completion with the processed prompt
        result = run_completion(rn_ctx, cmpl_options, callback);

        if (result.success) {
            // Create OpenAI-compatible response
            json response = {
                {"id", gen_chatcmplid()},
                {"object", "chat.completion"},
                {"created", (int)std::time(nullptr)},
                {"model", options.model.empty() ? "llamacpp-rn" : options.model}
            };

            json choices = json::array();
            json choice = {
                {"index", 0},
                {"message", {
                    {"role", "assistant"},
                    {"content", result.content}
                }},
                {"finish_reason", "stop"}
            };

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


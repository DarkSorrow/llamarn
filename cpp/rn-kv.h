#pragma once

#include "llama.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace facebook::react {

// f16 K and V for a plain attention cache:
// (n_embd / n_head) * n_head_kv * n_layer * 2 * sizeof(f16).
// Empty when that number would not be the cache the context allocates (no layers,
// hybrid/SSM state, or MLA). The caller leaves the field off rather than storing a guess.
inline bool rn_model_meta_present(const llama_model * model, const char * suffix) {
    char arch[64] = {};
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) <= 0 || arch[0] == '\0') {
        return false;
    }
    const std::string key = std::string(arch) + suffix;
    char value[32] = {};
    return llama_model_meta_val_str(model, key.c_str(), value, sizeof(value)) > 0;
}

inline std::optional<int64_t> rn_kv_bytes_per_token(const llama_model * model) {
    if (!model) {
        return std::nullopt;
    }
    // Recurrent state and compressed KV are not this formula.
    if (rn_model_meta_present(model, ".ssm.conv_kernel") ||
        rn_model_meta_present(model, ".attention.kv_lora_rank")) {
        return std::nullopt;
    }
    // n_head() aborts when the model has no layers. A vocab-only file is that case.
    const int32_t n_layer = llama_model_n_layer(model);
    const int32_t n_embd  = llama_model_n_embd(model);
    if (n_layer <= 0 || n_embd <= 0) {
        return std::nullopt;
    }
    const int32_t n_head    = llama_model_n_head(model);
    const int32_t n_head_kv = llama_model_n_head_kv(model);
    if (n_head <= 0 || n_head_kv <= 0 || n_embd % n_head != 0) {
        return std::nullopt;
    }
    const int32_t head_dim = n_embd / n_head;
    // K and V, f16. llama_context_default_params uses GGML_TYPE_F16 for both.
    return static_cast<int64_t>(head_dim) * n_head_kv * n_layer * 2 * 2;
}

// n_keep_option < 0 disables the shift (the caller stops when the context is full).
// n_keep_option > 0 is an explicit token count.
// n_keep_option == 0 keeps prefix_end when it sits in (0, n_ctx - 4), otherwise half the context.
inline int rn_shift_n_keep(int n_keep_option, int n_ctx, int prefix_end, bool add_bos) {
    if (n_keep_option < 0) {
        return -1;
    }
    int n_keep;
    if (n_keep_option > 0) {
        n_keep = n_keep_option;
    } else if (prefix_end > 0 && prefix_end < n_ctx - 4) {
        n_keep = prefix_end;
    } else {
        n_keep = n_ctx / 2;
    }
    if (add_bos) {
        n_keep = std::max(n_keep, 1);
    }
    if (n_ctx > 4 && n_keep > n_ctx - 4) {
        n_keep = n_ctx - 4;
    }
    return n_keep;
}

// Length of the common leading run of two token lists (server: cache_prompt's prefix match).
// The KV cache holds `cached` in seq 0; the first `rn_common_prefix` positions stay valid for
// `prompt`, whatever the messages, template or tool list that rendered either one.
inline size_t rn_common_prefix(const std::vector<llama_token> & cached, const std::vector<llama_token> & prompt) {
    const size_t n = std::min(cached.size(), prompt.size());
    size_t i = 0;
    while (i < n && cached[i] == prompt[i]) {
        ++i;
    }
    return i;
}

// The reused length for a prompt: at most prompt.size() - 1, so at least one token is encoded and
// the next sample has fresh logits.
inline size_t rn_reuse_len(const std::vector<llama_token> & cached, const std::vector<llama_token> & prompt) {
    if (prompt.empty()) {
        return 0;
    }
    return std::min(rn_common_prefix(cached, prompt), prompt.size() - 1);
}

// Mirrors a context shift on the token list: positions [n_keep, n_keep + n_discard) were removed
// and the rest slid back by n_discard.
inline void rn_kv_tokens_shift(std::vector<llama_token> & tokens, int n_keep, int n_discard) {
    if (n_keep < 0 || n_discard <= 0 || static_cast<size_t>(n_keep) >= tokens.size()) {
        return;
    }
    const size_t end = std::min(tokens.size(), static_cast<size_t>(n_keep) + static_cast<size_t>(n_discard));
    tokens.erase(tokens.begin() + n_keep, tokens.begin() + static_cast<std::ptrdiff_t>(end));
}

// Whether `head` is a leading run of `tokens`: the first message's render is a true prefix of
// the prompt only when its tokens are (a template may render a message differently once more
// messages follow).
inline bool rn_is_token_prefix(const std::vector<llama_token> & head, const std::vector<llama_token> & tokens) {
    return !head.empty() && head.size() <= tokens.size() && std::equal(head.begin(), head.end(), tokens.begin());
}

struct rn_reasoning_cap {
    bool    active = false;
    int32_t tokens = -1;
};

// Init budget 0 leaves thinking off. A positive completion request replaces a non-zero
// init cap for this call. A completion value of 0 or negative does not replace it.
// No end tags means the sampler cannot enforce a cap.
inline rn_reasoning_cap rn_select_reasoning_cap(
    int init_budget,
    const std::optional<int32_t> & request,
    bool has_end_tags) {
    rn_reasoning_cap cap;
    if (!has_end_tags || init_budget == 0) {
        return cap;
    }
    int32_t tokens = init_budget;
    if (request && *request > 0) {
        tokens = *request;
    }
    if (tokens > 0) {
        cap.active = true;
        cap.tokens = tokens;
    }
    return cap;
}

} // namespace facebook::react

#include "rn-kv.h"
#include "fixtures.h"
#include "rn-test.h"

#include <optional>
#include <string>
#include <vector>

using namespace facebook::react;

RN_TEST(shift_keep_uses_the_first_message_boundary) {
    RN_CHECK_EQ(rn_shift_n_keep(-1, 128, 20, false), -1);
    RN_CHECK_EQ(rn_shift_n_keep(0, 128, -1, false), 64);
    RN_CHECK_EQ(rn_shift_n_keep(0, 128, 20, false), 20);
    RN_CHECK_EQ(rn_shift_n_keep(0, 128, 200, false), 64);
    RN_CHECK_EQ(rn_shift_n_keep(10, 128, 20, false), 10);
    RN_CHECK_EQ(rn_shift_n_keep(1000, 128, 0, false), 124);
    RN_CHECK_EQ(rn_shift_n_keep(0, 128, 0, true), 64);
}

RN_TEST(common_prefix_and_reuse_length) {
    const std::vector<llama_token> cached = { 1, 2, 3, 4, 5 };
    RN_CHECK_EQ(rn_common_prefix(cached, { 1, 2, 3, 9 }), static_cast<size_t>(3));
    RN_CHECK_EQ(rn_common_prefix(cached, { 9 }), static_cast<size_t>(0));
    RN_CHECK_EQ(rn_common_prefix({}, { 1 }), static_cast<size_t>(0));
    // A prompt that extends the cache reuses all of it
    RN_CHECK_EQ(rn_reuse_len(cached, { 1, 2, 3, 4, 5, 6, 7 }), static_cast<size_t>(5));
    // The whole prompt cached: its last token is encoded again for fresh logits
    RN_CHECK_EQ(rn_reuse_len(cached, { 1, 2, 3 }), static_cast<size_t>(2));
    RN_CHECK_EQ(rn_reuse_len(cached, {}), static_cast<size_t>(0));
}

RN_TEST(token_list_mirrors_a_shift) {
    std::vector<llama_token> tokens = { 0, 1, 2, 3, 4, 5, 6, 7 };
    rn_kv_tokens_shift(tokens, 2, 3);
    RN_CHECK((tokens == std::vector<llama_token>{ 0, 1, 5, 6, 7 }));

    tokens = { 0, 1, 2 };
    rn_kv_tokens_shift(tokens, 1, 10);
    RN_CHECK((tokens == std::vector<llama_token>{ 0 }));

    tokens = { 0, 1, 2 };
    rn_kv_tokens_shift(tokens, 5, 1);
    RN_CHECK((tokens == std::vector<llama_token>{ 0, 1, 2 }));
}

RN_TEST(token_prefix_check) {
    RN_CHECK(rn_is_token_prefix({ 1, 2 }, { 1, 2, 3 }));
    RN_CHECK(!rn_is_token_prefix({ 1, 3 }, { 1, 2, 3 }));
    RN_CHECK(!rn_is_token_prefix({}, { 1 }));
    RN_CHECK(!rn_is_token_prefix({ 1, 2, 3, 4 }, { 1, 2, 3 }));
}

RN_TEST(reasoning_cap_cannot_turn_thinking_on) {
    const auto off = rn_select_reasoning_cap(0, 32, true);
    RN_CHECK(!off.active);

    const auto on = rn_select_reasoning_cap(128, std::nullopt, true);
    RN_CHECK(on.active);
    RN_CHECK_EQ(on.tokens, 128);

    const auto replaced = rn_select_reasoning_cap(128, 32, true);
    RN_CHECK(replaced.active);
    RN_CHECK_EQ(replaced.tokens, 32);

    const auto ignore_zero = rn_select_reasoning_cap(128, 0, true);
    RN_CHECK_EQ(ignore_zero.tokens, 128);
    const auto ignore_unlimited = rn_select_reasoning_cap(128, -1, true);
    RN_CHECK_EQ(ignore_unlimited.tokens, 128);

    const auto unlimited = rn_select_reasoning_cap(-1, std::nullopt, true);
    RN_CHECK(!unlimited.active);
    const auto unlimited_override = rn_select_reasoning_cap(-1, 32, true);
    RN_CHECK(unlimited_override.active);
    RN_CHECK_EQ(unlimited_override.tokens, 32);

    const auto no_tags = rn_select_reasoning_cap(128, 32, false);
    RN_CHECK(!no_tags.active);
}

RN_TEST(kv_bytes_absent_without_a_head_layout) {
    auto * model = rn_test_vocab("qwen35");
    if (llama_model_n_layer(model) <= 0) {
        RN_CHECK(!rn_kv_bytes_per_token(model).has_value());
        return;
    }
    const auto bytes = rn_kv_bytes_per_token(model);
    RN_CHECK(bytes.has_value());
    RN_CHECK(*bytes > 0);
}

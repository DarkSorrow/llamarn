#include "rn-kv.h"
#include "fixtures.h"
#include "rn-test.h"

#include <optional>
#include <string>
#include <vector>

using namespace facebook::react;

struct kv_row {
    std::string id;
    int32_t     token_end = 0;
};

RN_TEST(shift_keep_uses_the_first_message_boundary) {
    RN_CHECK_EQ(rn_shift_n_keep(-1, 128, 20, false), -1);
    RN_CHECK_EQ(rn_shift_n_keep(0, 128, -1, false), 64);
    RN_CHECK_EQ(rn_shift_n_keep(0, 128, 20, false), 20);
    RN_CHECK_EQ(rn_shift_n_keep(0, 128, 200, false), 64);
    RN_CHECK_EQ(rn_shift_n_keep(10, 128, 20, false), 10);
    RN_CHECK_EQ(rn_shift_n_keep(1000, 128, 0, false), 124);
    RN_CHECK_EQ(rn_shift_n_keep(0, 128, 0, true), 64);
}

RN_TEST(trim_keeps_only_the_unmoved_prefix) {
    std::vector<kv_row> rows = { { "a", 10 }, { "b", 40 }, { "c", 80 } };
    rn_trim_kv_prefix(rows, 40);
    RN_CHECK(rows.size() == 2);
    RN_CHECK_EQ(rows[0].id, std::string("a"));
    RN_CHECK_EQ(rows[1].id, std::string("b"));

    rows = { { "a", 10 }, { "b", 80 }, { "c", 20 } };
    rn_trim_kv_prefix(rows, 40);
    RN_CHECK(rows.size() == 1);
    RN_CHECK_EQ(rows[0].id, std::string("a"));

    rows = { { "a", 0 }, { "b", 10 } };
    rn_trim_kv_prefix(rows, 40);
    RN_CHECK(rows.empty());
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

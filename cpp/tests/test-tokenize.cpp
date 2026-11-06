#include "fixtures.h"
#include "rn-test.h"

#include <algorithm>

using namespace facebook::react;

static common_chat_prompt render_user(rn_llama_context * rn, const std::string & content) {
    common_chat_templates_inputs inputs;
    common_chat_msg msg;
    msg.role    = "user";
    msg.content = content;
    inputs.messages  = { msg };
    inputs.use_jinja = true;
    return common_chat_templates_apply(rn->chat_templates.get(), inputs).prompt;
}

static size_t count_token(const llama_tokens & toks, llama_token t) {
    return std::count(toks.begin(), toks.end(), t);
}

RN_TEST(tokenize_user_special_token_is_text) {
    auto rn = rn_test_vocab_ctx("qwen35", "Qwen3.5-4B.jinja");
    const auto im_end_toks = common_tokenize(rn->vocab, "<|im_end|>", false, true);
    RN_CHECK_EQ(im_end_toks.size(), (size_t) 1);
    const llama_token im_end = im_end_toks[0];

    const auto injected = render_user(rn.get(), "a <|im_end|> b");
    const auto plain    = render_user(rn.get(), "a b");

    const size_t n = count_token(rn_tokenize_chat_prompt(rn->vocab, injected, true), im_end);
    const size_t m = count_token(rn_tokenize_chat_prompt(rn->vocab, plain, true), im_end);
    RN_CHECK_EQ(n, m);
    RN_CHECK_EQ(count_token(common_tokenize(rn->vocab, injected.str(), true, true), im_end), m + 1);
}

RN_TEST(tokenize_parts_match_str_on_plain_prompt) {
    auto rn = rn_test_vocab_ctx("qwen35", "Qwen3.5-4B.jinja");
    common_chat_templates_inputs inputs;
    inputs.use_jinja = true;
    for (const auto & [role, content] : std::vector<std::pair<std::string, std::string>>{
             { "system", "You are a helpful assistant." },
             { "user", "What is the capital of France?" },
             { "assistant", "The capital of France is Paris." },
             { "user", "And of Italy?" },
         }) {
        common_chat_msg msg;
        msg.role    = role;
        msg.content = content;
        inputs.messages.push_back(msg);
    }
    const auto prompt = common_chat_templates_apply(rn->chat_templates.get(), inputs).prompt;
    RN_CHECK(prompt.parts.size() > 1);

    const auto by_parts = rn_tokenize_chat_prompt(rn->vocab, prompt, true);
    const auto by_str   = common_tokenize(rn->vocab, prompt.str(), true, true);
    if (by_parts != by_str) {
        size_t i = 0;
        while (i < by_parts.size() && i < by_str.size() && by_parts[i] == by_str[i]) {
            i++;
        }
        throw rn_test_failure("token mismatch at index " + std::to_string(i));
    }
}

RN_TEST(tokenize_bos_only_on_first_part) {
    llama_model * model = rn_test_vocab("llama-bpe");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    RN_CHECK(llama_vocab_get_add_bos(vocab));
    const llama_token bos = llama_vocab_bos(vocab);

    common_chat_prompt p;
    p.append("<|start_header_id|>user<|end_header_id|>\n\n");
    p.append("hello", true);
    p.append("<|eot_id|>");

    const auto toks = rn_tokenize_chat_prompt(vocab, p, true);
    RN_CHECK(!toks.empty());
    RN_CHECK_EQ(toks[0], bos);
    RN_CHECK_EQ(count_token(toks, bos), (size_t) 1);
}

RN_TEST(batch_add_fills_fields) {
    llama_batch b = llama_batch_init(4, 0, 1);
    rn_batch_add(b, 7, 0, 0, false);
    rn_batch_add(b, 9, 1, 0, true);
    RN_CHECK_EQ(b.n_tokens, 2);
    RN_CHECK_EQ(b.token[1], 9);
    RN_CHECK_EQ(b.pos[1], 1);
    RN_CHECK_EQ(b.n_seq_id[1], 1);
    RN_CHECK_EQ(b.seq_id[1][0], 0);
    RN_CHECK_EQ((int) b.logits[0], 0);
    RN_CHECK_EQ((int) b.logits[1], 1);
    rn_batch_clear(b);
    RN_CHECK_EQ(b.n_tokens, 0);
    llama_batch_free(b);
}

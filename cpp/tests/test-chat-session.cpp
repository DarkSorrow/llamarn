#include "fixtures.h"
#include "rn-test.h"

using namespace facebook::react;

static common_chat_templates_inputs user_inputs(const rn_llama_context * rn, const std::string & content) {
    common_chat_templates_inputs inputs;
    common_chat_msg msg;
    msg.role    = "user";
    msg.content = content;
    inputs.messages              = { msg };
    inputs.add_generation_prompt = true;
    inputs.use_jinja             = rn->params.use_jinja;
    inputs.reasoning_format      = COMMON_REASONING_FORMAT_AUTO;
    return inputs;
}

static common_chat_tool special_function_tool() {
    return {
        "special_function",
        "I'm special",
        R"({"type": "object", "properties": {"arg1": {"type": "integer", "description": "The arg."}}, "required": ["arg1"]})",
    };
}

RN_TEST(session_parses_qwen35_tool_call) {
    auto rn = rn_test_vocab_ctx("qwen35", "Qwen3.5-4B.jinja");
    auto inputs = user_inputs(rn.get(), "Hey there!");
    inputs.tools           = { special_function_tool() };
    inputs.enable_thinking = false;

    auto session = rn_make_chat_session(rn.get(), inputs);
    RN_CHECK(!inputs.force_pure_content);
    RN_CHECK(inputs.use_jinja);

    common_params_sampling sampling;
    session.apply_sampling(sampling);
    RN_CHECK(!sampling.grammar.empty());

    const auto & msg = session.finish(common_chat_input(
        "<tool_call>\n<function=special_function>\n<parameter=arg1>\n1\n</parameter>\n</function>\n</tool_call>"));
    RN_CHECK_EQ(msg.tool_calls.size(), (size_t) 1);
    RN_CHECK_EQ(msg.tool_calls[0].name, std::string("special_function"));
    RN_CHECK_EQ(msg.tool_calls[0].arguments, std::string("{\"arg1\":1}"));
}

RN_TEST(session_splits_reasoning) {
    auto rn = rn_test_vocab_ctx("qwen35", "Qwen3.5-4B.jinja");
    auto inputs = user_inputs(rn.get(), "Hey there!");
    inputs.enable_thinking = true;

    auto session = rn_make_chat_session(rn.get(), inputs);
    RN_CHECK(!session.thinking_end_tags().empty());

    const auto & msg = session.finish(common_chat_input("I'm\nthinking\n</think>\n\nHello"));
    RN_CHECK_EQ(msg.reasoning_content, std::string("I'm\nthinking"));
    RN_CHECK_EQ(msg.content, std::string("Hello"));
}

RN_TEST(session_fallback_on_broken_template) {
    auto rn = rn_test_vocab_ctx("qwen35", nullptr);
    rn->chat_templates = common_chat_templates_init(
        rn->model,
        "{% if messages[0]['content'] == 'boom' %}{{ raise_exception('boom') }}{% endif %}"
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}");

    auto inputs  = user_inputs(rn.get(), "boom");
    auto session = rn_make_chat_session(rn.get(), inputs);
    RN_CHECK(!inputs.use_jinja);
    RN_CHECK(!session.prompt().str().empty());
}

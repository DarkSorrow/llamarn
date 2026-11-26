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

// Same append the server uses: each piece carries the token id that produced it.
static common_chat_input aligned_input(const llama_vocab * vocab, const std::string & text) {
    common_chat_input input;
    for (llama_token id : common_tokenize(vocab, text, false, true)) {
        input.append(common_token_to_piece(vocab, id), id);
    }
    return input;
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

    const std::string raw =
        "<tool_call>\n<function=special_function>\n<parameter=arg1>\n1\n</parameter>\n</function>\n</tool_call>";
    const auto & msg = session.finish(aligned_input(rn->vocab, raw));
    RN_CHECK_EQ(msg.tool_calls.size(), (size_t) 1);
    RN_CHECK_EQ(msg.tool_calls[0].name, std::string("special_function"));
    RN_CHECK_EQ(msg.tool_calls[0].arguments, std::string("{\"arg1\":1}"));
}

RN_TEST(session_ignores_tool_call_without_token_ids) {
    // Qwen3.5's <tool_call> markers are plain text in this vocab, so a null-id
    // input still parses. Gemma 4's markers are registered special tokens:
    // p.token matches only when the input carries that id.
    auto rn = rn_test_vocab_ctx("gemma-4", nullptr);
    rn->chat_templates = common_chat_templates_init(
        rn->model,
        "{% for m in messages %}{{ m['content'] }}{% endfor %}'<|tool_call>call:'");
    auto inputs = user_inputs(rn.get(), "Hey there!");
    inputs.tools           = { special_function_tool() };
    inputs.enable_thinking = false;

    auto session = rn_make_chat_session(rn.get(), inputs);
    RN_CHECK(inputs.use_jinja);
    const auto & msg = session.finish(common_chat_input(
        "<|tool_call>call:special_function{arg1:1}<tool_call|>"));
    RN_CHECK_EQ(msg.tool_calls.size(), (size_t) 0);
}

RN_TEST(generated_input_truncates_like_the_server) {
    auto rn = rn_test_vocab_ctx("qwen35", "Qwen3.5-4B.jinja");
    common_chat_input hello = aligned_input(rn->vocab, "Hello");
    common_chat_input input = hello;
    for (llama_token id : common_tokenize(rn->vocab, "<|im_end|>", false, true)) {
        input.append(common_token_to_piece(rn->vocab, id), id);
    }
    input.truncate(hello.text.size());
    RN_CHECK_EQ(input.text, hello.text);
    RN_CHECK(hello.text.find("Hello") != std::string::npos);
}

RN_TEST(session_splits_reasoning) {
    auto rn = rn_test_vocab_ctx("qwen35", "Qwen3.5-4B.jinja");
    auto inputs = user_inputs(rn.get(), "Hey there!");
    inputs.enable_thinking = true;

    auto session = rn_make_chat_session(rn.get(), inputs);
    RN_CHECK(!session.thinking_end_tags().empty());

    const auto & msg = session.finish(aligned_input(rn->vocab, "I'm\nthinking\n</think>\n\nHello"));
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

// --- tool_choice (b12780: struct + parser that takes the tools) ---

static common_chat_tool other_function_tool() {
    return {
        "other_function",
        "Not this one",
        R"({"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]})",
    };
}

RN_TEST(tool_choice_strings_parse) {
    const std::vector<common_chat_tool> tools = { special_function_tool() };
    RN_CHECK(common_chat_tool_choice_parse_oaicompat("auto", tools).is_auto());
    RN_CHECK(common_chat_tool_choice_parse_oaicompat("none", tools).is_none());
    RN_CHECK(common_chat_tool_choice_parse_oaicompat("required", tools).is_required());
}

RN_TEST(tool_choice_named_function_reaches_session) {
    auto rn = rn_test_vocab_ctx("qwen35", "Qwen3.5-4B.jinja");
    auto inputs = user_inputs(rn.get(), "Hey there!");
    inputs.tools           = { special_function_tool(), other_function_tool() };
    inputs.enable_thinking = false;
    // The JSON the bridge now forwards unchanged instead of collapsing it to "required".
    inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(
        common_json::parse(R"({"type": "function", "function": {"name": "special_function"}})"), inputs.tools);
    RN_CHECK(inputs.tool_choice.is_required());
    RN_CHECK_EQ(inputs.tool_choice.allowed.size(), (size_t) 1);
    RN_CHECK(inputs.tool_choice.allows("special_function"));
    RN_CHECK(!inputs.tool_choice.allows("other_function"));

    // Required mode reaches the session: a non-lazy grammar that must end in a tool call.
    // Whether the grammar also drops the other tools depends on the template's upstream
    // handler. At b12780 the Qwen3.5 grammar still lists both, so this test does not check that.
    auto session = rn_make_chat_session(rn.get(), inputs);
    RN_CHECK(session.grammar().find("special_function") != std::string::npos);
    common_params_sampling sampling;
    session.apply_sampling(sampling);
    RN_CHECK(!sampling.grammar_lazy);
}

RN_TEST(tool_choice_unknown_name_throws) {
    const std::vector<common_chat_tool> tools = { special_function_tool() };
    bool threw = false;
    try {
        common_chat_tool_choice_parse_oaicompat(
            common_json::parse(R"({"type": "function", "function": {"name": "nope"}})"), tools);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    RN_CHECK(threw);
}

// --- streaming deltas ---

// Splits a Qwen3.5 thinking reply into small token chunks, streams it through
// rn_stream_parser over a copy of the session, and checks that the summed deltas
// equal the one-shot finish() on the original (what run_chat_completion returns).
RN_TEST(stream_parser_deltas_match_finish) {
    auto rn = rn_test_vocab_ctx("qwen35", "Qwen3.5-4B.jinja");
    auto inputs = user_inputs(rn.get(), "Say hi.");
    inputs.enable_thinking = true;

    auto session = rn_make_chat_session(rn.get(), inputs);
    common_chat_session stream_copy = session;

    const std::string raw = "The user wants a greeting.\n</think>\n\nHello there!";
    const common_chat_input generated = aligned_input(rn->vocab, raw);

    rn_stream_parser parser;
    parser.session  = &stream_copy;
    parser.interval = std::chrono::milliseconds(0);

    std::string content, reasoning;
    for (size_t upto = 1; upto <= generated.size(); upto += 3) {
        rn_stream_chunk chunk;
        parser.update(generated, upto, false, chunk);
        content += chunk.content;
        reasoning += chunk.reasoning_content;
    }
    rn_stream_chunk last;
    parser.update(generated, generated.size(), true, last);
    content += last.content;
    reasoning += last.reasoning_content;
    RN_CHECK(!parser.disabled || parser.n_fed == generated.size());

    const auto & final_msg = session.finish(generated);
    RN_CHECK(!final_msg.reasoning_content.empty());
    RN_CHECK(final_msg.content.find("</think>") == std::string::npos);
    RN_CHECK_EQ(reasoning, final_msg.reasoning_content);
    RN_CHECK_EQ(content, final_msg.content);
}

// initLlama reads this key when parallel_tool_calls is omitted (llama-server's default).
// A rename upstream would silently turn parallel calls off, so pin it here.
RN_TEST(template_caps_report_parallel_tool_calls) {
    auto rn = rn_test_vocab_ctx("qwen35", "Qwen3.5-4B.jinja");
    const json caps = json::parse(common_chat_templates_get_caps(rn->chat_templates.get()).dump());
    RN_CHECK(caps.contains("supports_parallel_tool_calls"));
    RN_CHECK(caps["supports_parallel_tool_calls"].is_boolean());
}

#include "fixtures.h"
#include "rn-test.h"

using namespace facebook::react;

static std::unique_ptr<rn_test_model> load_or_skip(int n_ctx = 2048, int n_batch = 512, int chunk_size = 128) {
    auto m = rn_test_load_model(n_ctx, n_batch, chunk_size);
    if (!m) {
        RN_SKIP("RN_TEST_MODEL not set");
    }
    return m;
}

static CompletionOptions greedy(int n_predict) {
    CompletionOptions o;
    o.n_predict   = n_predict;
    o.temperature = 0.0f;
    o.top_k       = 1;
    o.seed        = 42;
    return o;
}

static const auto no_stream = [](const std::string &, bool) { return true; };

static CompletionOptions chat(std::vector<std::pair<std::string, std::string>> msgs, int n_predict) {
    CompletionOptions o = greedy(n_predict);
    o.messages = json::array();
    for (size_t i = 0; i < msgs.size(); i++) {
        o.messages.push_back({ { "role", msgs[i].first }, { "content", msgs[i].second }, { "id", "m" + std::to_string(i) } });
    }
    return o;
}

static std::string long_prompt(int sentences) {
    std::string s;
    for (int i = 0; i < sentences; i++) {
        s += "Sentence number " + std::to_string(i) + " talks about rivers, mountains and the sea. ";
    }
    return s + "In one word, the topic of all these sentences is";
}

static std::string chat_content(const CompletionResult & r) {
    return r.chat_response["choices"][0]["message"].value("content", std::string());
}

RN_TEST(completion_short_prompt) {
    auto m = load_or_skip();
    auto o = greedy(8);
    o.prompt = "The capital of France is";
    auto r = run_completion(m->rn.get(), o, no_stream);
    RN_CHECK(r.success);
    RN_CHECK(r.n_predicted_tokens > 0);
    RN_CHECK(!r.content.empty());
}

RN_TEST(completion_long_prompt_over_n_batch) {
    auto m = load_or_skip(2048, 64, 128);
    auto o = greedy(4);
    o.prompt = long_prompt(20);
    auto r = run_completion(m->rn.get(), o, no_stream);
    RN_CHECK(r.success);
    RN_CHECK(r.n_prompt_tokens > 64);
}

RN_TEST(completion_chunk_size_invariant) {
    auto m = load_or_skip();
    auto o = greedy(12);
    o.prompt = long_prompt(10);

    m->rn->params.chunk_size = 8;
    auto a = run_completion(m->rn.get(), o, no_stream);
    m->rn->params.chunk_size = 512;
    auto b = run_completion(m->rn.get(), o, no_stream);
    RN_CHECK(a.success && b.success);
    RN_CHECK_EQ(a.content, b.content);
}

RN_TEST(completion_abort_then_rerun_matches_fresh) {
    auto m = load_or_skip();
    auto o = greedy(12);
    o.prompt = "Write a short poem about the sea:";

    auto fresh = run_completion(m->rn.get(), o, no_stream);
    RN_CHECK(fresh.success);

    // abort during generation: the next llama_process sees the abort callback
    auto * rn = m->rn.get();
    auto aborted = run_completion(rn, o, [rn](const std::string &, bool) {
        rn->abort_generation = true;
        return true;
    });
    RN_CHECK(!aborted.success);
    RN_CHECK_EQ(aborted.error_msg, std::string("Generation aborted"));

    // abort before ingest
    rn->abort_generation = true;
    auto aborted_ingest = run_completion(rn, o, no_stream);
    RN_CHECK(!aborted_ingest.success);

    rn->abort_generation = false;
    auto rerun = run_completion(rn, o, no_stream);
    RN_CHECK(rerun.success);
    RN_CHECK_EQ(rerun.content, fresh.content);
}

RN_TEST(completion_context_full_shift_or_stop) {
    auto m = load_or_skip(256, 64, 64);
    auto o = greedy(400);
    o.ignore_eos = true;
    o.prompt = "Count from one to one thousand: one, two, three,";
    auto r = run_completion(m->rn.get(), o, no_stream);
    RN_CHECK(r.success);
    RN_CHECK(r.n_prompt_tokens + r.n_predicted_tokens <= 256 || r.context_shifted);
}

RN_TEST(chat_kv_reuse_matches_full_encode) {
    auto m = load_or_skip();
    auto * rn = m->rn.get();

    auto t1 = chat({ { "system", "You are terse." }, { "user", "Name a fruit." } }, 16);
    auto r1 = run_chat_completion(rn, t1, no_stream);
    RN_CHECK(r1.success);
    RN_CHECK(rn->kv_has_messages);

    auto t2 = chat({ { "system", "You are terse." },
                     { "user", "Name a fruit." },
                     { "assistant", chat_content(r1) },
                     { "user", "Name a color." } }, 16);
    auto reused = run_chat_completion(rn, t2, no_stream);
    RN_CHECK(reused.success);

    t2.reset_kv_cache = true;
    auto full = run_chat_completion(rn, t2, no_stream);
    RN_CHECK(full.success);
    RN_CHECK_EQ(chat_content(reused), chat_content(full));
}

RN_TEST(chat_abort_keeps_matched_prefix) {
    auto m = load_or_skip();
    auto * rn = m->rn.get();

    auto t1 = chat({ { "system", "You are terse." }, { "user", "Name a fruit." } }, 16);
    auto r1 = run_chat_completion(rn, t1, no_stream);
    RN_CHECK(r1.success);

    auto t2 = chat({ { "system", "You are terse." },
                     { "user", "Name a fruit." },
                     { "assistant", chat_content(r1) },
                     { "user", "Name a color." } }, 16);
    auto aborted = run_chat_completion(rn, t2, [rn](const std::string &, bool) {
        rn->abort_generation = true;
        return true;
    });
    RN_CHECK(!aborted.success);
    RN_CHECK(rn->kv_messages.size() <= t1.messages.size());
    rn->abort_generation = false;

    auto rerun = run_chat_completion(rn, t2, no_stream);
    RN_CHECK(rerun.success);

    t2.reset_kv_cache = true;
    auto full = run_chat_completion(rn, t2, no_stream);
    RN_CHECK(full.success);
    RN_CHECK_EQ(chat_content(rerun), chat_content(full));
}

// Same tools, kwargs and system prompt as example/src/ModelChatTestScreen.tsx in tools mode.
static json example_tools() {
    return json::parse(R"json([
      {"type": "function", "function": {
        "name": "get_weather",
        "description": "Gets the current weather information for a specific city",
        "parameters": {"type": "object", "properties": {
          "location": {"type": "string", "description": "The city name for which to get weather information"},
          "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit of temperature to use (default: celsius)"}
        }, "required": ["location"]}}},
      {"type": "function", "function": {
        "name": "get_location",
        "description": "Gets the user's current location",
        "parameters": {"type": "object", "properties": {}, "required": []}}}
    ])json");
}

static void set_init_kwargs(rn_llama_context * rn) {
    auto & kw = rn->params.default_template_kwargs;
    kw["enable_thinking"]     = "false";
    kw["parse_tool_calls"]    = "true";
    kw["parallel_tool_calls"] = "false";
}

RN_TEST(chat_tools_reach_the_prompt) {
    auto m = load_or_skip(4096);
    auto * rn = m->rn.get();
    set_init_kwargs(rn);

    common_chat_templates_inputs inputs;
    inputs.messages = common_chat_msgs_parse_oaicompat(common_json::parse(R"([
      {"role": "system", "content": "You are a helpful AI assistant with access to tools."},
      {"role": "user", "content": "What is the weather in Paris?"}
    ])"));
    inputs.add_generation_prompt = true;
    inputs.use_jinja             = true;
    inputs.chat_template_kwargs  = rn->params.default_template_kwargs;
    inputs.enable_thinking       = false;
    inputs.tools                 = common_chat_tools_parse_oaicompat(common_json::parse(example_tools().dump()));
    inputs.parallel_tool_calls   = true;
    inputs.tool_choice           = common_chat_tool_choice_parse_oaicompat("auto");

    auto session = rn_make_chat_session(rn, inputs);
    RN_CHECK(inputs.use_jinja);
    RN_CHECK(!inputs.force_pure_content);
    RN_CHECK(session.prompt().str().find("get_weather") != std::string::npos);
}

RN_TEST(chat_tool_call_end_to_end) {
    auto m = load_or_skip(4096);
    auto * rn = m->rn.get();
    set_init_kwargs(rn);

    auto o = chat({ { "system", "You are a helpful AI assistant with access to tools. Use tools only when necessary." },
                    { "user", "What is the weather in Paris?" } }, 256);
    o.tools       = example_tools();
    o.tool_choice = "auto";
    auto r = run_chat_completion(rn, o, no_stream);
    RN_CHECK(r.success);
    const auto & choice = r.chat_response["choices"][0];
    if (choice["finish_reason"] != "tool_calls") {
        throw rn_test_failure("no tool call: " + choice.dump());
    }
    RN_CHECK_EQ(choice["message"]["tool_calls"][0]["function"]["name"].get<std::string>(), std::string("get_weather"));
}

RN_TEST(chat_tool_call_example_settings) {
    auto m = load_or_skip(4096, 128, 32);
    auto * rn = m->rn.get();
    set_init_kwargs(rn);

    for (int run = 0; run < 3; run++) {
        CompletionOptions o;
        o.messages = json::parse(R"json([
          {"id": "s", "role": "system", "content": "You are a helpful AI assistant with access to tools. Use tools only when necessary.\n\nRules:\n1. For weather information: Use the get_weather tool\n2. For location information: Use the get_location tool\n3. For general questions: Answer directly without tools\n4. Only call tools when necessary to answer the user's specific question\n5. Don't make multiple tool calls for the same information\n6. Keep responses concise and helpful"},
          {"id": "u", "role": "user", "content": "What's the weather in Paris?"}
        ])json");
        o.temperature       = 0.6f;
        o.top_p             = 0.95f;
        o.top_k             = 20;
        o.min_p             = 0.05f;
        o.repeat_penalty    = 1.0f;
        o.repeat_last_n     = 512;
        o.presence_penalty  = 0.0f;
        o.n_predict         = 2048;
        o.stop              = { "</s>", "<|im_end|>", "<|eot_id|>", "<|eom_id|>" };
        o.tools             = example_tools();
        o.tool_choice       = "auto";
        o.reset_kv_cache    = true;
        auto r = run_chat_completion(rn, o, no_stream);
        RN_CHECK(r.success);
        const auto & choice = r.chat_response["choices"][0];
        if (choice["finish_reason"] != "tool_calls") {
            throw rn_test_failure("run " + std::to_string(run) + " no tool call: " + choice.dump());
        }
    }
}

RN_TEST(chat_reasoning_and_content_split) {
    auto m = load_or_skip();
    m->rn->params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    auto o = chat({ { "user", "Say hi." } }, 64);
    auto r = run_chat_completion(m->rn.get(), o, no_stream);
    RN_CHECK(r.success);
    const auto & msg = r.chat_response["choices"][0]["message"];
    RN_CHECK(msg.contains("content"));
    RN_CHECK(msg["content"].get<std::string>().find("</think>") == std::string::npos);
}

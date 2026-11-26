#include "fixtures.h"
#include "rn-kv.h"
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

static const auto no_stream = [](const rn_stream_chunk &, bool) { return true; };

static CompletionOptions chat(std::vector<std::pair<std::string, std::string>> msgs, int n_predict) {
    CompletionOptions o = greedy(n_predict);
    o.messages = json::array();
    for (size_t i = 0; i < msgs.size(); i++) {
        o.messages.push_back({ { "role", msgs[i].first }, { "content", msgs[i].second } });
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

// The token list describes the cache: as many tokens as seq 0 has positions.
static bool kv_in_sync(rn_llama_context * rn) {
    const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(rn->ctx), 0);
    return static_cast<size_t>(pos_max + 1) == rn->kv_tokens.size();
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

// llama.cpp does not promise identical logits across batch sizes (tools/server/README.md,
// cache_prompt): the CPU backend picks kernels by batch shape (BLAS / repack from 32 tokens up,
// quantized dot products below). Measured at b12780: one 141-token batch vs chunks of 8 moves
// the last logits by up to ~0.5 on Qwen3.5 and SmolLM2, which flips a greedy token on some models.
// So the exact check uses two chunk sizes on the same kernel path. It still covers the loop:
// positions, the last-token logits flag, and many llama_process calls per prompt.
RN_TEST(completion_chunk_size_invariant) {
    auto m = load_or_skip();
    auto o = greedy(12);
    o.prompt = long_prompt(10);

    m->rn->params.chunk_size = 8;
    auto a = run_completion(m->rn.get(), o, no_stream);
    m->rn->params.chunk_size = 16;
    auto b = run_completion(m->rn.get(), o, no_stream);
    RN_CHECK(a.success && b.success);
    RN_CHECK_EQ(a.content, b.content);
    RN_CHECK_EQ(a.n_prompt_tokens, b.n_prompt_tokens);
}

// Across kernel paths (8 vs one 512 chunk) only the token count is exact. timings.prompt_n is
// llama_perf n_p_eval, which upstream counts per sync (llama-context.cpp FIXME): the first
// generated token can land in it, so it may read one above the prompt length.
RN_TEST(completion_chunk_size_encodes_every_token) {
    auto m = load_or_skip();
    auto o = greedy(4);
    o.prompt = long_prompt(10);

    m->rn->params.chunk_size = 8;
    auto a = run_completion(m->rn.get(), o, no_stream);
    m->rn->params.chunk_size = 512;
    auto b = run_completion(m->rn.get(), o, no_stream);
    RN_CHECK(a.success && b.success);
    RN_CHECK_EQ(a.n_prompt_tokens, b.n_prompt_tokens);
    for (const auto * r : { &a, &b }) {
        const int over = r->timings.prompt_n - static_cast<int>(r->n_prompt_tokens);
        RN_CHECK(over == 0 || over == 1);
        RN_CHECK(r->n_predicted_tokens > 0);
    }
}

RN_TEST(completion_abort_then_rerun_matches_fresh) {
    auto m = load_or_skip();
    auto o = greedy(12);
    o.prompt = "Write a short poem about the sea:";

    auto fresh = run_completion(m->rn.get(), o, no_stream);
    RN_CHECK(fresh.success);

    // abort during generation: the next llama_process sees the abort callback
    auto * rn = m->rn.get();
    auto aborted = run_completion(rn, o, [rn](const rn_stream_chunk &, bool) {
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
    RN_CHECK(!rn->kv_tokens.empty());
    RN_CHECK(kv_in_sync(rn));

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
    auto aborted = run_chat_completion(rn, t2, [rn](const rn_stream_chunk &, bool) {
        rn->abort_generation = true;
        return true;
    });
    RN_CHECK(!aborted.success);
    // The cache keeps what was decoded before the abort, and the list says exactly that.
    RN_CHECK(kv_in_sync(rn));
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
    // Same kwargs do_init_llama sets: enable_thinking is the only template variable.
    kw["enable_thinking"] = "false";
    rn->params.parallel_tool_calls = false;
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
    inputs.tool_choice           = common_chat_tool_choice_parse_oaicompat("auto", inputs.tools);

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

static std::string cleared_reason(const CompletionResult & r) {
    return r.chat_response.at("kv_cleared_reason").get<std::string>();
}

static bool fail_seq_rm(void *, llama_pos, llama_pos) {
    return false;
}

RN_TEST(chat_result_reports_a_prefix_hit) {
    auto m = load_or_skip();
    auto * rn = m->rn.get();

    auto first = chat({ { "system", "You are terse." }, { "user", "Name a fruit." } }, 8);
    auto r1 = run_chat_completion(rn, first, no_stream);
    RN_CHECK(r1.success);
    RN_CHECK(r1.chat_response.contains("timings"));
    RN_CHECK(r1.chat_response["timings"].contains("prompt_n"));
    RN_CHECK_EQ(cleared_reason(r1), std::string("no_prefix"));
    RN_CHECK(r1.timings.prompt_n + 1 >= r1.n_prompt_tokens);

    first.temperature = 0.8f;
    auto reused = run_chat_completion(rn, first, no_stream);
    RN_CHECK(reused.success);
    RN_CHECK(reused.chat_response["usage"]["prompt_tokens"].get<int>() == reused.n_prompt_tokens);
    const std::string why = cleared_reason(reused);
    // Temperature is not a cache key. none means the suffix was dropped. Anything else re-encodes the prompt.
    RN_CHECK(why != "render_identity");
    if (why == "none") {
        RN_CHECK(reused.timings.prompt_n < reused.n_prompt_tokens);
    } else {
        RN_CHECK(reused.timings.prompt_n + 1 >= reused.n_prompt_tokens);
    }

    auto with_tools = first;
    with_tools.tools = example_tools();
    auto missed = run_chat_completion(rn, with_tools, no_stream);
    RN_CHECK(missed.success);
    RN_CHECK_EQ(cleared_reason(missed), std::string("render_identity"));
    RN_CHECK(missed.timings.prompt_n * 2 > missed.n_prompt_tokens);
}

RN_TEST(chat_prompt_id_change_clears) {
    auto m = load_or_skip();
    auto * rn = m->rn.get();
    auto first = chat({ { "user", "Name a fruit." } }, 4);
    first.prompt_id = "prompt-a";
    auto r1 = run_chat_completion(rn, first, no_stream);
    RN_CHECK(r1.success);

    first.prompt_id = "prompt-b";
    auto r2 = run_chat_completion(rn, first, no_stream);
    RN_CHECK(r2.success);
    RN_CHECK_EQ(cleared_reason(r2), std::string("prompt_id"));
}

RN_TEST(chat_missing_ids_report_no_prefix) {
    auto m = load_or_skip();
    CompletionOptions o = greedy(4);
    o.messages = json::array({ { { "role", "user" }, { "content", "Name a fruit." } } });
    auto r = run_chat_completion(m->rn.get(), o, no_stream);
    RN_CHECK(r.success);
    RN_CHECK_EQ(cleared_reason(r), std::string("no_prefix"));
    RN_CHECK(r.timings.prompt_n + 1 >= r.n_prompt_tokens);
}

RN_TEST(chat_seq_rm_failure_is_visible) {
    auto m = load_or_skip();
    auto * rn = m->rn.get();
    // A single user message is a prefix the template can render, so the suffix drop is actually attempted.
    auto first = chat({ { "user", "Name a fruit." } }, 4);
    auto r1 = run_chat_completion(rn, first, no_stream);
    RN_CHECK(r1.success);
    RN_CHECK(!rn->kv_tokens.empty());

    rn->kv_seq_rm = fail_seq_rm;
    auto second = chat({ { "user", "Name a fruit." }, { "user", "Name a color." } }, 4);
    auto r2 = run_chat_completion(rn, second, no_stream);
    RN_CHECK(r2.success);
    RN_CHECK_EQ(cleared_reason(r2), std::string("seq_rm"));
    RN_CHECK(r2.timings.prompt_n + 1 >= r2.n_prompt_tokens);
}

RN_TEST(chat_shift_keeps_the_first_message) {
    auto m = load_or_skip(96, 64, 64);
    auto * rn = m->rn.get();
    if (!llama_memory_can_shift(llama_get_memory(rn->ctx))) {
        RN_SKIP("memory cannot shift");
    }

    auto seeded = chat({ { "system", "You are terse." }, { "user", "Say hi." } }, 4);
    auto seed = run_chat_completion(rn, seeded, no_stream);
    RN_CHECK(seed.success);

    auto filling = chat({
        { "system", "You are terse." },
        { "user", "Say hi." },
        { "assistant", chat_content(seed) },
        { "user", "Keep talking about rivers and mountains." },
    }, -1);
    filling.ignore_eos = true;
    // llama.cpp rounds n_ctx up to a multiple of 256 (llama-context.cpp), so budget from the
    // real window: enough tokens to fill it once and force one shift.
    const int budget = static_cast<int>(llama_n_ctx(rn->ctx)) + 16;
    int generated = 0;
    auto shifted = run_chat_completion(rn, filling, [&](const rn_stream_chunk &, bool done) {
        if (done) return true;
        return ++generated < budget;
    });
    RN_CHECK(shifted.success);
    RN_CHECK(shifted.context_shifted);
    RN_CHECK(shifted.chat_response["context_shifted"].get<bool>());
    // The system message's boundary, not half the window
    RN_CHECK(shifted.shift_n_keep > 0);
    RN_CHECK(shifted.shift_n_keep < static_cast<int>(llama_n_ctx(rn->ctx)) / 2);
    // The list shifted with the cache
    RN_CHECK(kv_in_sync(rn));

    auto follow = chat({ { "system", "You are terse." }, { "user", "Name one sea." } }, 4);
    auto hit = run_chat_completion(rn, follow, no_stream);
    RN_CHECK(hit.success);
    RN_CHECK_EQ(cleared_reason(hit), std::string("none"));
    RN_CHECK(hit.timings.prompt_n < hit.n_prompt_tokens);
    RN_CHECK(hit.timings.prompt_n <= hit.n_prompt_tokens - shifted.shift_n_keep + 8);
}

RN_TEST(chat_shift_without_ids_keeps_half_the_context) {
    auto m = load_or_skip(96, 64, 64);
    auto * rn = m->rn.get();
    if (!llama_memory_can_shift(llama_get_memory(rn->ctx))) {
        RN_SKIP("memory cannot shift");
    }
    CompletionOptions o = greedy(-1);
    o.ignore_eos = true;
    o.messages = json::array({ { { "role", "user" }, { "content", "Count upward." } } });
    CompletionOptions probe = o;
    probe.n_predict = 1;
    const auto unshifted = run_chat_completion(rn, probe, no_stream);
    RN_CHECK(unshifted.success);
    // Real window, not the requested 96: llama.cpp rounds n_ctx up to a multiple of 256.
    const int budget = static_cast<int>(llama_n_ctx(rn->ctx)) + 16;
    int generated = 0;
    auto r = run_chat_completion(rn, o, [&](const rn_stream_chunk &, bool done) {
        if (done) return true;
        return ++generated < budget;
    });
    RN_CHECK(r.success);
    RN_CHECK(r.context_shifted);
    // A shift that outgrows the prompt used to throw here (prompt_tokens resize underflow).
    // usage.prompt_tokens stays the full rendered prompt after a shift.
    RN_CHECK_EQ(r.n_prompt_tokens, unshifted.n_prompt_tokens);
    RN_CHECK(r.n_predicted_tokens > static_cast<int>(llama_n_ctx(rn->ctx)) - static_cast<int>(r.n_prompt_tokens));
    const int expected = rn_shift_n_keep(0, llama_n_ctx(rn->ctx), -1, llama_vocab_get_add_bos(rn->vocab));
    RN_CHECK_EQ(r.shift_n_keep, expected);
}

RN_TEST(chat_n_keep_negative_stops_without_a_shift) {
    auto m = load_or_skip(96, 64, 64);
    auto * rn = m->rn.get();
    if (!llama_memory_can_shift(llama_get_memory(rn->ctx))) {
        RN_SKIP("memory cannot shift");
    }
    auto o = chat({ { "user", "Count upward." } }, -1);
    o.ignore_eos = true;
    o.n_keep = -1;
    auto r = run_chat_completion(rn, o, no_stream);
    RN_CHECK(r.success);
    RN_CHECK(!r.context_shifted);
    RN_CHECK(r.stopped_by_length);
}

RN_TEST(kv_bytes_track_the_evaluated_prompt) {
    auto m = load_or_skip(2048, 512, 128);
    auto * rn = m->rn.get();
    RN_CHECK_EQ((int) llama_n_ctx(rn->ctx), 2048);
    const auto bytes = rn_kv_bytes_per_token(rn->model);
    if (!bytes.has_value()) {
        // Hybrid and MLA models omit the field. init still honored the requested context.
        return;
    }
    RN_CHECK(*bytes > 0);

    auto o = greedy(1);
    o.prompt = "The capital of France is";
    auto r = run_completion(rn, o, no_stream);
    RN_CHECK(r.success);
    RN_CHECK(r.n_prompt_tokens > 0);
    const int64_t expect = *bytes * r.n_prompt_tokens;
    const int64_t state = static_cast<int64_t>(llama_state_get_size(rn->ctx));
    RN_CHECK(state > expect / 4);
    RN_CHECK(state < expect * 4 + (1 << 20));
}

RN_TEST(chat_reasoning_budget_active_follows_the_cap) {
    auto m = load_or_skip();
    auto * rn = m->rn.get();
    try {
        rn->chat_templates = common_chat_templates_init(rn->model, rn_test_template("Qwen3.5-4B.jinja"));
    } catch (const std::exception &) {
        RN_SKIP("thinking template did not load");
    }
    rn->params.reasoning_budget = 128;
    rn->params.default_template_kwargs["enable_thinking"] = "true";

    auto capped = chat({ { "user", "Say hi." } }, 1);
    capped.reasoning_budget = 32;
    auto r = run_chat_completion(rn, capped, no_stream);
    if (!r.success) {
        RN_SKIP(r.error_msg.c_str());
    }
    RN_CHECK(r.chat_response["reasoning_budget_active"].get<bool>());

    capped.reasoning_budget = 64;
    auto hit = run_chat_completion(rn, capped, no_stream);
    RN_CHECK(hit.success);
    RN_CHECK(hit.chat_response["reasoning_budget_active"].get<bool>());
    const std::string why = cleared_reason(hit);
    // The cap is not part of the cache key. Hybrid memory still cannot drop the tail.
    RN_CHECK(why == "none" || why == "seq_rm");
    if (why == "none") {
        RN_CHECK(hit.timings.prompt_n < hit.n_prompt_tokens);
    }

    rn->params.reasoning_budget = 0;
    rn->params.default_template_kwargs["enable_thinking"] = "false";
    capped.reasoning_budget = 32;
    auto off = run_chat_completion(rn, capped, no_stream);
    RN_CHECK(off.success);
    RN_CHECK(!off.chat_response["reasoning_budget_active"].get<bool>());
    RN_CHECK_EQ(cleared_reason(off), std::string("render_identity"));
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

RN_TEST(chat_tool_choice_named_function) {
    auto m = load_or_skip(4096);
    auto * rn = m->rn.get();
    set_init_kwargs(rn);

    auto o = chat({ { "user", "Tell me something." } }, 256);
    o.tools       = example_tools();
    o.tool_choice = json::parse(R"({"type": "function", "function": {"name": "get_location"}})");
    o.reset_kv_cache = true;
    auto r = run_chat_completion(rn, o, no_stream);
    RN_CHECK(r.success);
    const auto & choice = r.chat_response["choices"][0];
    // The named choice is required mode plus an allowed list. Required mode is enforced for
    // every template. Narrowing the grammar to the one name is up to the template's upstream
    // handler; at b12780 Qwen3.5 keeps every tool in the grammar, so the name is not checked.
    if (choice["finish_reason"] != "tool_calls") {
        throw rn_test_failure("no tool call: " + choice.dump());
    }
}

RN_TEST(chat_tool_choice_unknown_name_is_invalid_param) {
    auto m = load_or_skip(4096);
    auto * rn = m->rn.get();
    set_init_kwargs(rn);

    auto o = chat({ { "user", "Hi" } }, 8);
    o.tools       = example_tools();
    o.tool_choice = json::parse(R"({"type": "function", "function": {"name": "does_not_exist"}})");
    auto r = run_chat_completion(rn, o, no_stream);
    RN_CHECK(!r.success);
    RN_CHECK(r.error_type == RN_ERROR_INVALID_PARAM);
}

RN_TEST(chat_stream_deltas_sum_to_final_message) {
    auto m = load_or_skip();
    auto * rn = m->rn.get();
    rn->params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    auto o = chat({ { "user", "Say hi." } }, 64);
    o.stream = true;
    o.reset_kv_cache = true;
    std::string text, content, reasoning;
    int done = 0;
    auto r = run_chat_completion(rn, o, [&](const rn_stream_chunk & c, bool is_done) {
        if (is_done) {
            done++;
            return true;
        }
        text += c.text;
        content += c.content;
        reasoning += c.reasoning_content;
        return true;
    });
    RN_CHECK(r.success);
    RN_CHECK_EQ(done, 1);
    RN_CHECK_EQ(text, r.content);
    const auto & msg = r.chat_response["choices"][0]["message"];
    if (msg.contains("tool_calls")) {
        return;
    }
    RN_CHECK_EQ(content, msg.value("content", std::string()));
    RN_CHECK_EQ(reasoning, msg.value("reasoning_content", std::string()));
}


// Three turns, each sending the previous reply back with reasoning_content, as an app does.
// Qwen3-style templates render that reply differently once a new user message follows; the
// token prefix still reuses the whole history up to that reply.
RN_TEST(chat_multi_turn_reuses_history) {
    auto m = load_or_skip();
    auto * rn = m->rn.get();

    std::vector<std::pair<std::string, std::string>> history = { { "system", "You are terse." }, { "user", "Name a fruit." } };
    auto first = run_chat_completion(rn, chat(history, 12), no_stream);
    RN_CHECK(first.success);
    RN_CHECK(kv_in_sync(rn));
    RN_CHECK_EQ(rn->kv_tokens.size(), static_cast<size_t>(first.n_prompt_tokens + first.n_predicted_tokens));

    CompletionResult previous = first;
    for (const char * next : { "Name a color.", "Name an animal." }) {
        auto turn = chat(history, 12);
        turn.messages.push_back({ { "role", "assistant" }, { "content", chat_content(previous) }, { "reasoning_content", "Short answer." } });
        turn.messages.push_back({ { "role", "user" }, { "content", next } });
        history.push_back({ "assistant", chat_content(previous) });
        history.push_back({ "user", next });

        auto r = run_chat_completion(rn, turn, no_stream);
        RN_CHECK(r.success);
        RN_CHECK(kv_in_sync(rn));
        const std::string why = cleared_reason(r);
        if (why == "seq_rm" && !llama_memory_can_shift(llama_get_memory(rn->ctx))) {
            // Hybrid / recurrent memory (Qwen3.5) cannot drop the previous reply: known limit.
            RN_SKIP("memory cannot drop a suffix: every turn re-encodes");
        }
        RN_CHECK_EQ(why, std::string("none"));
        // Everything before the previous turn's generation header is reused.
        RN_CHECK(r.timings.prompt_n <= r.n_prompt_tokens - (previous.n_prompt_tokens - 8));
        previous = r;
    }
}

RN_TEST(chat_edit_earlier_message_reencodes_from_the_edit) {
    auto m = load_or_skip();
    auto * rn = m->rn.get();

    auto t1 = chat({ { "system", "You are terse." }, { "user", "Name a fruit." } }, 8);
    auto r1 = run_chat_completion(rn, t1, no_stream);
    RN_CHECK(r1.success);

    // The user message changed: only the system part is reused.
    auto t2 = chat({ { "system", "You are terse." }, { "user", "Name a river in Europe." } }, 8);
    auto r2 = run_chat_completion(rn, t2, no_stream);
    RN_CHECK(r2.success);
    RN_CHECK(kv_in_sync(rn));
    if (cleared_reason(r2) == "seq_rm" && !llama_memory_can_shift(llama_get_memory(rn->ctx))) {
        // Hybrid / recurrent memory (Qwen3.5) drops the last reply but cannot roll back into an
        // earlier prompt: an edited message re-encodes everything. The cache stays consistent.
        RN_SKIP("memory cannot roll back into the prompt");
    }
    RN_CHECK_EQ(cleared_reason(r2), std::string("none"));
    RN_CHECK(r2.timings.prompt_n < r2.n_prompt_tokens);

    t2.reset_kv_cache = true;
    auto cold = run_chat_completion(rn, t2, no_stream);
    RN_CHECK(cold.success);
    RN_CHECK_EQ(cleared_reason(cold), std::string("reset"));
    // The reuse encoded strictly less than the cold run
    RN_CHECK(r2.timings.prompt_n < cold.timings.prompt_n);
}

RN_TEST(chat_kv_tokens_follow_a_shift) {
    auto m = load_or_skip(96, 64, 64);
    auto * rn = m->rn.get();
    if (!llama_memory_can_shift(llama_get_memory(rn->ctx))) {
        RN_SKIP("memory cannot shift");
    }
    CompletionOptions o = greedy(-1);
    o.ignore_eos = true;
    o.messages = json::array({ { { "role", "user" }, { "content", "Count upward." } } });
    const int budget = static_cast<int>(llama_n_ctx(rn->ctx)) + 16;
    int generated = 0;
    auto r = run_chat_completion(rn, o, [&](const rn_stream_chunk &, bool done) {
        if (done) return true;
        return ++generated < budget;
    });
    RN_CHECK(r.success);
    RN_CHECK(r.context_shifted);
    RN_CHECK(kv_in_sync(rn));

    // The kept head is still this prompt's start: the same request reuses it.
    CompletionOptions again = o;
    again.n_predict = 2;
    auto hit = run_chat_completion(rn, again, no_stream);
    RN_CHECK(hit.success);
    RN_CHECK_EQ(cleared_reason(hit), std::string("none"));
    RN_CHECK(kv_in_sync(rn));
}

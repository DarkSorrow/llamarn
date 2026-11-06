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

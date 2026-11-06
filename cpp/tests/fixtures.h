#pragma once

#include "rn-llama.h"

#include <memory>
#include <string>

// vocab-only model from cpp/llama.cpp/models/ggml-vocab-<name>.gguf, cached per name
llama_model * rn_test_vocab(const char * name);

// contents of cpp/llama.cpp/models/templates/<file>
std::string rn_test_template(const char * file);

// rn_llama_context with vocab + chat templates only (no llama_context), for template/tokenizer tests
std::unique_ptr<facebook::react::rn_llama_context> rn_test_vocab_ctx(const char * vocab, const char * template_file);

struct rn_test_model {
    common_init_result_ptr                             init;
    std::unique_ptr<facebook::react::rn_llama_context> rn;
    ~rn_test_model();
};

// full model from env RN_TEST_MODEL, nullptr when unset
std::unique_ptr<rn_test_model> rn_test_load_model(int n_ctx, int n_batch, int chunk_size);

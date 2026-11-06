#include "fixtures.h"

#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

using namespace facebook::react;

llama_model * rn_test_vocab(const char * name) {
    static std::map<std::string, llama_model *> cache;
    auto it = cache.find(name);
    if (it != cache.end()) {
        return it->second;
    }
    const std::string path = std::string(RN_TEST_ROOT) + "/models/ggml-vocab-" + name + ".gguf";
    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(path.c_str(), mparams);
    if (!model) {
        throw std::runtime_error("failed to load vocab " + path);
    }
    cache[name] = model;
    return model;
}

std::string rn_test_template(const char * file) {
    const std::string path = std::string(RN_TEST_ROOT) + "/models/templates/" + file;
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to read template " + path);
    }
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::unique_ptr<rn_llama_context> rn_test_vocab_ctx(const char * vocab, const char * template_file) {
    auto rn = std::make_unique<rn_llama_context>();
    rn->model            = rn_test_vocab(vocab);
    rn->vocab            = llama_model_get_vocab(rn->model);
    rn->params.use_jinja = true;
    rn->chat_templates   = common_chat_templates_init(rn->model, template_file ? rn_test_template(template_file) : "");
    return rn;
}

#ifdef RN_TESTS_WITH_BRIDGE

rn_test_model::~rn_test_model() {
    if (rn) {
        rn_free_batches(rn.get());
        rn->chat_templates.reset();
    }
    rn.reset();
    init.reset();
}

std::unique_ptr<rn_test_model> rn_test_load_model(int n_ctx, int n_batch, int chunk_size) {
    const char * path = getenv("RN_TEST_MODEL");
    if (!path || !*path) {
        return nullptr;
    }

    rn_common_params params;
    params.model.path   = path;
    params.n_ctx        = n_ctx;
    params.n_batch      = n_batch;
    params.n_ubatch     = n_batch;
    params.n_gpu_layers = 0;
    params.fit_params   = false;
    params.use_jinja    = true;
    params.cpuparams.n_threads = 4;
    postprocess_cpu_params(params.cpuparams,       nullptr);
    postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);

    auto m  = std::make_unique<rn_test_model>();
    m->init = common_init_from_params(params);
    if (!m->init || !m->init->context()) {
        throw std::runtime_error(std::string("failed to load model ") + path);
    }

    m->rn               = std::make_unique<rn_llama_context>();
    m->rn->model        = m->init->model();
    m->rn->ctx          = m->init->context();
    m->rn->vocab        = llama_model_get_vocab(m->rn->model);
    m->rn->model_loaded = true;
    m->rn->params       = params;
    m->rn->params.chunk_size  = chunk_size;
    m->rn->params.is_cpu_only = false;
    m->rn->params.prompt_chunk_gap_ms = 0;
    m->rn->chat_templates = common_chat_templates_init(m->rn->model, "");
    rn_init_batches(m->rn.get());

    llama_set_abort_callback(
        m->rn->ctx,
        [](void * data) -> bool {
            return static_cast<rn_llama_context *>(data)->abort_generation.load(std::memory_order_relaxed);
        },
        m->rn.get());

    return m;
}

#else

rn_test_model::~rn_test_model() = default;

std::unique_ptr<rn_test_model> rn_test_load_model(int, int, int) {
    return nullptr;
}

#endif

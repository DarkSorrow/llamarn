#include "PureCppImpl.h"

#include <jsi/jsi.h>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <thread>
#include "SystemUtils.h"
// Include our custom headers - this was missing!
#include "rn-llama.hpp"
#include "LlamaCppModel.h"
// Include the llama.cpp common headers
#include "chat.h"

#if defined(__ANDROID__) || defined(__linux__)
#include <unistd.h>
#endif

// Include the llama.cpp headers directly
#include "llama.h"

#if defined(__APPLE__)
#include <TargetConditionals.h>
#include <sys/sysctl.h>
#endif

namespace facebook::react {

// Factory method implementation
std::shared_ptr<TurboModule> PureCppImpl::create(std::shared_ptr<CallInvoker> jsInvoker) {
  return std::make_shared<PureCppImpl>(std::move(jsInvoker));
}

PureCppImpl::PureCppImpl(std::shared_ptr<CallInvoker> jsInvoker)
    : NativeRNLlamaCppCxxSpec(std::move(jsInvoker)) {
}

double PureCppImpl::multiply(jsi::Runtime& rt, double a, double b) {
    return a * b;
}

jsi::Value PureCppImpl::loadLlamaModelInfo(jsi::Runtime &runtime, jsi::String modelPath) {
  std::string path = modelPath.utf8(runtime);
  SystemUtils::normalizeFilePath(path);

  try {
    // Initialize llama backend
    llama_backend_init();

    // Create model params
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0; // Use CPU for model info loading

    // Load the model
    llama_model* model = llama_model_load_from_file(path.c_str(), params);

    if (!model) {
      throw std::runtime_error("Failed to load model from file: " + path);
    }

    // Create result object
    jsi::Object result(runtime);

    // Get model parameters
    result.setProperty(runtime, "n_params", jsi::Value((double)llama_model_n_params(model)));

    // Get vocabulary
    const llama_vocab* vocab = llama_model_get_vocab(model);
    result.setProperty(runtime, "n_vocab", jsi::Value((double)llama_vocab_n_tokens(vocab)));

    // Get context size
    result.setProperty(runtime, "n_context", jsi::Value((double)llama_model_n_ctx_train(model)));

    // Get embedding size
    result.setProperty(runtime, "n_embd", jsi::Value((double)llama_model_n_embd(model)));

    // Get model description
    char buf[512];
    llama_model_desc(model, buf, sizeof(buf));
    result.setProperty(runtime, "description",
                      jsi::String::createFromUtf8(runtime, buf[0] ? buf : "Unknown model"));

    // Check if GPU is supported
    bool gpuSupported = llama_supports_gpu_offload();
    result.setProperty(runtime, "gpuSupported", jsi::Value(gpuSupported));

    // Calculate optimal GPU layers if GPU is supported
    int optimalGpuLayers = 0;
    if (gpuSupported) {
      optimalGpuLayers = SystemUtils::getOptimalGpuLayers(model);
    }
    result.setProperty(runtime, "optimalGpuLayers", jsi::Value(optimalGpuLayers));

    // Extract quantization type from model description
    std::string desc(buf);
    std::string quantType = "Unknown";
    size_t qPos = desc.find(" Q");
    if (qPos != std::string::npos && qPos + 5 <= desc.length()) {
      // Extract quantization string (like Q4_K, Q5_K, Q8_0)
      quantType = desc.substr(qPos + 1, 4);
      // Remove any trailing non-alphanumeric characters
      quantType.erase(std::find_if(quantType.rbegin(), quantType.rend(), [](char c) {
        return std::isalnum(c);
      }).base(), quantType.end());
    }
    result.setProperty(runtime, "quant_type", jsi::String::createFromUtf8(runtime, quantType));

    // Add architecture info
    result.setProperty(runtime, "architecture",
                      jsi::String::createFromUtf8(runtime, "Unknown"));

    // Free the model
    llama_model_free(model);

    return result;
  } catch (const std::exception& e) {
    jsi::Object error(runtime);
    error.setProperty(runtime, "message", jsi::String::createFromUtf8(runtime, e.what()));
    throw jsi::JSError(runtime, error.getProperty(runtime, "message").asString(runtime));
  }
}

jsi::Value PureCppImpl::initLlama(jsi::Runtime &runtime, jsi::Object options) {
  std::lock_guard<std::mutex> lock(mutex_);

  try {
    // Get model path - required (preserve custom path handling)
    if (!options.hasProperty(runtime, "model")) {
      throw std::runtime_error("model path is required");
    }

    // Initialize llama backend
    llama_backend_init();

    std::string model_path = options.getProperty(runtime, "model").asString(runtime).utf8(runtime);
    SystemUtils::normalizeFilePath(model_path);

    // Initialize params with defaults
    rn_common_params params;

    // Set default sampling parameters
    params.sampling = common_params_sampling();

    // Set model path
    params.model.path = model_path;

    // Override defaults with user settings if provided
    SystemUtils::setIfExists(runtime, options, "n_ctx", params.n_ctx);
    SystemUtils::setIfExists(runtime, options, "n_batch", params.n_batch);
    SystemUtils::setIfExists(runtime, options, "n_ubatch", params.n_ubatch);
    SystemUtils::setIfExists(runtime, options, "n_keep", params.n_keep);

    // Memory and resource options - MUST respect user settings
    SystemUtils::setIfExists(runtime, options, "use_mmap", params.use_mmap);
    SystemUtils::setIfExists(runtime, options, "use_mlock", params.use_mlock);
    SystemUtils::setIfExists(runtime, options, "use_jinja", params.use_jinja);

    // Extract threading parameters (preserve custom thread logic)
    int n_threads = 0; // 0 = auto
    if (options.hasProperty(runtime, "n_threads")) {
      n_threads = options.getProperty(runtime, "n_threads").asNumber();
    } else {
      n_threads = SystemUtils::getOptimalThreadCount();
    }
    params.cpuparams.n_threads = n_threads;

    // Set n_gpu_layers (preserve custom GPU logic)
    int n_gpu_layers = 0;
    bool gpuSupported = llama_supports_gpu_offload();
    if (options.hasProperty(runtime, "n_gpu_layers") && gpuSupported) {
      n_gpu_layers = options.getProperty(runtime, "n_gpu_layers").asNumber();
    }
    params.n_gpu_layers = n_gpu_layers;

    // Additional model parameters
    SystemUtils::setIfExists(runtime, options, "logits_file", params.logits_file);
    SystemUtils::setIfExists(runtime, options, "embedding", params.embedding);
    SystemUtils::setIfExists(runtime, options, "rope_freq_base", params.rope_freq_base);
    SystemUtils::setIfExists(runtime, options, "rope_freq_scale", params.rope_freq_scale);

    // Sampling parameters
    SystemUtils::setIfExists(runtime, options, "seed", params.sampling.seed);

    // Other system parameters
    SystemUtils::setIfExists(runtime, options, "verbose", params.verbosity);

    // RoPE settings if provided
    if (options.hasProperty(runtime, "yarn_ext_factor")) {
      params.yarn_ext_factor = options.getProperty(runtime, "yarn_ext_factor").asNumber();
    }
    if (options.hasProperty(runtime, "yarn_attn_factor")) {
      params.yarn_attn_factor = options.getProperty(runtime, "yarn_attn_factor").asNumber();
    }
    if (options.hasProperty(runtime, "yarn_beta_fast")) {
      params.yarn_beta_fast = options.getProperty(runtime, "yarn_beta_fast").asNumber();
    }
    if (options.hasProperty(runtime, "yarn_beta_slow")) {
      params.yarn_beta_slow = options.getProperty(runtime, "yarn_beta_slow").asNumber();
    }

    // Support for chat template override
    std::string chat_template;
    if (SystemUtils::setIfExists(runtime, options, "chat_template", chat_template)) {
      params.chat_template = chat_template;
    }

    // Support for LoRA adapters
    if (options.hasProperty(runtime, "lora_adapters") && options.getProperty(runtime, "lora_adapters").isObject()) {
      jsi::Object lora_obj = options.getProperty(runtime, "lora_adapters").asObject(runtime);
      if (lora_obj.isArray(runtime)) {
        jsi::Array lora_array = lora_obj.asArray(runtime);
        size_t n_lora = lora_array.size(runtime);

        for (size_t i = 0; i < n_lora; i++) {
          if (lora_array.getValueAtIndex(runtime, i).isObject()) {
            jsi::Object adapter = lora_array.getValueAtIndex(runtime, i).asObject(runtime);
            if (adapter.hasProperty(runtime, "path") && adapter.getProperty(runtime, "path").isString()) {
              common_adapter_lora_info lora;
              lora.path = adapter.getProperty(runtime, "path").asString(runtime).utf8(runtime);

              // Get scale if provided
              lora.scale = 1.0f; // Default scale
              if (adapter.hasProperty(runtime, "scale") && adapter.getProperty(runtime, "scale").isNumber()) {
                lora.scale = adapter.getProperty(runtime, "scale").asNumber();
              }

              params.lora_adapters.push_back(lora);
            }
          }
        }
      }
    }

    // Initialize using common_init_from_params
    common_init_result result = common_init_from_params(params);

    // Check if initialization was successful
    if (!result.model || !result.context) {
      throw std::runtime_error("Failed to initialize model and context");
    }

    // Create and initialize rn_llama_context
    rn_ctx_ = std::make_unique<facebook::react::rn_llama_context>();
    rn_ctx_->model = result.model.release();
    rn_ctx_->ctx = result.context.release();
    rn_ctx_->model_loaded = true;
    rn_ctx_->vocab = llama_model_get_vocab(rn_ctx_->model);

    // Create a rn_common_params from the common_params
    rn_common_params rn_params;
    // Copy the base class fields
    static_cast<common_params&>(rn_params) = params;
    // Set additional fields
    rn_params.use_jinja = params.use_jinja;
    rn_params.reasoning_format = COMMON_REASONING_FORMAT_NONE;
    // Use chatml format by default instead of content-only for better tool support
    rn_params.chat_format = COMMON_CHAT_FORMAT_GENERIC;
    // Now assign to the context
    rn_ctx_->params = rn_params;

    // Initialize chat templates with proper error handling
    try {
        // Get BOS and EOS tokens if provided in options
        std::string bos_token_override;
        std::string eos_token_override;

        SystemUtils::setIfExists(runtime, options, "bos_token", bos_token_override);
        SystemUtils::setIfExists(runtime, options, "eos_token", eos_token_override);

        rn_ctx_->chat_templates = common_chat_templates_init(
            rn_ctx_->model,
            params.chat_template,
            bos_token_override,
            eos_token_override
        );

        if (!rn_ctx_->chat_templates) {
            throw std::runtime_error("Failed to initialize chat templates");
        }
    } catch (const std::exception& e) {
        // Log warning and fallback to chatml
        fprintf(stderr, "Warning: Failed to initialize chat template: %s. Falling back to chatml.\n", e.what());
        rn_ctx_->chat_templates = common_chat_templates_init(rn_ctx_->model, "chatml");
        if (!rn_ctx_->chat_templates) {
            throw std::runtime_error("Failed to initialize fallback chatml template");
        }
    }

    // Create the model object and return it
    return createModelObject(runtime, rn_ctx_.get());
  } catch (const std::exception& e) {
    fprintf(stderr, "initLlama error: %s\n", e.what());
    throw jsi::JSError(runtime, e.what());
  }
}

jsi::Object PureCppImpl::createModelObject(jsi::Runtime& runtime, rn_llama_context* rn_ctx) {
  // Create a shared_ptr to a new LlamaCppModel instance
  auto llamaModel = std::make_shared<LlamaCppModel>(rn_ctx);

  // Create a host object from the LlamaCppModel instance
  return jsi::Object::createFromHostObject(runtime, std::move(llamaModel));
}

} // namespace facebook::react

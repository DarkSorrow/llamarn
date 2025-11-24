#include "PureCppImpl.h"

#include <jsi/jsi.h>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <thread>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include "SystemUtils.h"
// Include our custom headers - this was missing!
#include "rn-llama.h"
#include "LlamaCppModel.h"
// Include the llama.cpp common headers
#include "chat.h"

#if defined(__ANDROID__) || defined(__linux__)
#include <dlfcn.h>
// #include <android/log.h>
// #ifndef LOG_TAG
// #define LOG_TAG "RNLlamaCpp"
// #endif
// #define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
// #define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
// #define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
// #define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#else
// #define LOGI(...) fprintf(stderr, __VA_ARGS__)
// #define LOGE(...) fprintf(stderr, __VA_ARGS__)
// #define LOGW(...) fprintf(stderr, __VA_ARGS__)
// #define LOGD(...) fprintf(stderr, __VA_ARGS__)
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
    : NativeRNLlamaCppCxxSpec(jsInvoker), jsInvoker_(jsInvoker) {
}

double PureCppImpl::multiply(jsi::Runtime& rt, double a, double b) {
    return a * b;
}

jsi::Value PureCppImpl::loadLlamaModelInfo(jsi::Runtime &runtime, jsi::String modelPath) {
  // Parse JSI arguments to native types on JSI thread
  std::string path = modelPath.utf8(runtime);
  SystemUtils::normalizeFilePath(path);

  if (!jsInvoker_) {
    // Fallback to synchronous if no CallInvoker available - this should not happen normally
    throw jsi::JSError(runtime, "CallInvoker not available for async operation");
  }

  // Create Promise constructor
  auto Promise = runtime.global().getPropertyAsFunction(runtime, "Promise");
  
  auto executor = jsi::Function::createFromHostFunction(
    runtime,
    jsi::PropNameID::forAscii(runtime, "executor"),
    2,
    [this, path](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* args, size_t count) -> jsi::Value {
      
      auto resolve = std::make_shared<jsi::Function>(args[0].asObject(runtime).asFunction(runtime));
      auto reject = std::make_shared<jsi::Function>(args[1].asObject(runtime).asFunction(runtime));
      
      // Create shared references to runtime and invoker for thread safety
      auto runtimePtr = &runtime;
      auto invoker = jsInvoker_;
      auto selfPtr = shared_from_this();
      
      // Launch background thread for model info loading
      std::thread([selfPtr, path, resolve, reject, runtimePtr, invoker]() {
        try {
          // Set up logging callback to capture llama.cpp error messages
          // llama_log_set([](enum ggml_log_level level, const char * text, void * /* user_data */) {
          //   if (level >= GGML_LOG_LEVEL_ERROR) {
          //     LOGE("llama.cpp: %s", text);
          //   }
          // }, nullptr);
          
          // Load all available backends (CPU is dynamically loaded when GGML_BACKEND_DL is enabled)
          // With GGML_BACKEND_DL=ON, ALL backends (CPU + GPU) are dynamically loaded
          // CPU backend is in libggml-cpu.so, GPU backends are in libggml-opencl.so, libggml-vulkan.so
          // On Android, dlopen() can load libraries by name even from inside APKs
          #ifdef __ANDROID__
          // Load CPU backend directly - Android's linker will find it in the same directory
          void* cpu_handle = dlopen("libggml-cpu.so", RTLD_LAZY | RTLD_LOCAL);
          if (cpu_handle) {
            typedef ggml_backend_reg_t (*backend_init_fn_t)();
            backend_init_fn_t backend_init = (backend_init_fn_t)dlsym(cpu_handle, "ggml_backend_init");
            if (backend_init) {
              ggml_backend_reg_t cpu_backend = backend_init();
              if (cpu_backend) {
                ggml_backend_register(cpu_backend);
              }
            }
          }
          
          // Load GPU backends (OpenCL, Vulkan) if present - they will be found by name
          ggml_backend_load_all();
          #else
          ggml_backend_load_all();
          #endif
          
          // Verify at least CPU backend was loaded
          if (ggml_backend_reg_count() == 0) {
            throw std::runtime_error("No backends registered - CPU backend library not found");
          }
          
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

          // Get model information (native types)
          double n_params = (double)llama_model_n_params(model);
          const llama_vocab* vocab = llama_model_get_vocab(model);
          double n_vocab = (double)llama_vocab_n_tokens(vocab);
          double n_context = (double)llama_model_n_ctx_train(model);
          double n_embd = (double)llama_model_n_embd(model);

          // Get model description
          char buf[512];
          llama_model_desc(model, buf, sizeof(buf));
          std::string description = buf[0] ? buf : "Unknown model";

          // Check if GPU is supported
          bool gpuSupported = llama_supports_gpu_offload();

          // Calculate optimal GPU layers if GPU is supported
          int optimalGpuLayers = 0;
          if (gpuSupported) {
            optimalGpuLayers = SystemUtils::getOptimalGpuLayers(model);
          }

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

          // Free the model
          llama_model_free(model);

          // Schedule success callback on JS thread to create JSI objects
          invoker->invokeAsync([selfPtr, resolve, n_params, n_vocab, n_context, n_embd, description, gpuSupported, optimalGpuLayers, quantType, runtimePtr]() {
            try {
              // Create result object on JS thread
              jsi::Object result(*runtimePtr);
              result.setProperty(*runtimePtr, "n_params", jsi::Value(n_params));
              result.setProperty(*runtimePtr, "n_vocab", jsi::Value(n_vocab));
              result.setProperty(*runtimePtr, "n_context", jsi::Value(n_context));
              result.setProperty(*runtimePtr, "n_embd", jsi::Value(n_embd));
              result.setProperty(*runtimePtr, "description", jsi::String::createFromUtf8(*runtimePtr, description));
              result.setProperty(*runtimePtr, "gpuSupported", jsi::Value(gpuSupported));
              result.setProperty(*runtimePtr, "optimalGpuLayers", jsi::Value(optimalGpuLayers));
              result.setProperty(*runtimePtr, "quant_type", jsi::String::createFromUtf8(*runtimePtr, quantType));
              result.setProperty(*runtimePtr, "architecture", jsi::String::createFromUtf8(*runtimePtr, "Unknown"));

              resolve->call(*runtimePtr, result);
            } catch (const std::exception& e) {
              // If conversion fails, create a simple error response
              jsi::Object errorObj(*runtimePtr);
              errorObj.setProperty(*runtimePtr, "error", jsi::String::createFromUtf8(*runtimePtr, e.what()));
              resolve->call(*runtimePtr, errorObj);
            }
          });
          
        } catch (const std::exception& e) {
          // Schedule error callback on JS thread
          std::string errorMsg(e.what());
          invoker->invokeAsync([reject, errorMsg, runtimePtr]() {
            try {
              reject->call(*runtimePtr, jsi::String::createFromUtf8(*runtimePtr, errorMsg));
            } catch (...) {
              // Ignore rejection errors
            }
          });
        }
      }).detach();
      
      return jsi::Value::undefined();
    }
  );
  
  return Promise.callAsConstructor(runtime, std::move(executor));
}

jsi::Value PureCppImpl::initLlama(jsi::Runtime &runtime, jsi::Object options) {
  // Parse JSI arguments to native types on JSI thread
  if (!options.hasProperty(runtime, "model")) {
    throw jsi::JSError(runtime, "model path is required");
  }

  if (!jsInvoker_) {
    // Fallback to synchronous if no CallInvoker available - this should not happen normally
    throw jsi::JSError(runtime, "CallInvoker not available for async operation");
  }

  // Parse all options to native types on JSI thread
  std::string model_path = options.getProperty(runtime, "model").asString(runtime).utf8(runtime);
  SystemUtils::normalizeFilePath(model_path);

  // Parse all numeric/boolean options to native types
  int n_ctx = 2048;  // defaults
  int n_batch = 512;
  int n_ubatch = 512;
  int n_keep = 0;
  bool use_mmap = true;
  bool use_mlock = false;
  bool use_jinja = false;
  bool embedding = false;
  int n_threads = 0;
  int n_gpu_layers = 0;
  std::string logits_file;
  float rope_freq_base = 10000.0f;
  float rope_freq_scale = 1.0f;
  uint32_t seed = 4294967295U; // default seed
  int verbosity = 0;
  float yarn_ext_factor = 1.0f;
  float yarn_attn_factor = 1.0f;
  float yarn_beta_fast = 32.0f;
  float yarn_beta_slow = 1.0f;
  std::string chat_template;
  
  // Thinking and reasoning options
  int reasoning_budget = 0;  // -1 = unlimited, 0 = disabled, >0 = limited
  common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_NONE;
  bool thinking_forced_open = false;
  bool parse_tool_calls = true;  // Enabled by default for better tool support
  bool parallel_tool_calls = false;  // Disabled by default for compatibility

  // Parse options to native types
  SystemUtils::setIfExists(runtime, options, "n_ctx", n_ctx);
  SystemUtils::setIfExists(runtime, options, "n_batch", n_batch);
  SystemUtils::setIfExists(runtime, options, "n_ubatch", n_ubatch);
  SystemUtils::setIfExists(runtime, options, "n_keep", n_keep);
  SystemUtils::setIfExists(runtime, options, "use_mmap", use_mmap);
  SystemUtils::setIfExists(runtime, options, "use_mlock", use_mlock);
  SystemUtils::setIfExists(runtime, options, "use_jinja", use_jinja);
  SystemUtils::setIfExists(runtime, options, "embedding", embedding);
  SystemUtils::setIfExists(runtime, options, "rope_freq_base", rope_freq_base);
  SystemUtils::setIfExists(runtime, options, "rope_freq_scale", rope_freq_scale);
  SystemUtils::setIfExists(runtime, options, "seed", seed);
  SystemUtils::setIfExists(runtime, options, "verbose", verbosity);
  SystemUtils::setIfExists(runtime, options, "logits_file", logits_file);
  SystemUtils::setIfExists(runtime, options, "chat_template", chat_template);
  
  // Parse thinking and reasoning options
  SystemUtils::setIfExists(runtime, options, "reasoning_budget", reasoning_budget);
  SystemUtils::setIfExists(runtime, options, "thinking_forced_open", thinking_forced_open);
  SystemUtils::setIfExists(runtime, options, "parse_tool_calls", parse_tool_calls);
  SystemUtils::setIfExists(runtime, options, "parallel_tool_calls", parallel_tool_calls);
  
  // Note: parse_tool_calls will be automatically enabled if use_jinja is true,
  // as Jinja templates provide better tool calling capabilities
  
  // Parse reasoning_format as string and convert to enum
  if (options.hasProperty(runtime, "reasoning_format")) {
    reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    std::string reasoning_format_str = options.getProperty(runtime, "reasoning_format").asString(runtime).utf8(runtime);
    reasoning_format = common_reasoning_format_from_name(reasoning_format_str);
  }

  if (options.hasProperty(runtime, "n_threads")) {
    n_threads = options.getProperty(runtime, "n_threads").asNumber();
  } else {
    n_threads = SystemUtils::getOptimalThreadCount();
  }

  bool gpuSupported = llama_supports_gpu_offload();
  if (options.hasProperty(runtime, "n_gpu_layers") && gpuSupported) {
    n_gpu_layers = options.getProperty(runtime, "n_gpu_layers").asNumber();
  }

  if (options.hasProperty(runtime, "yarn_ext_factor")) {
    yarn_ext_factor = options.getProperty(runtime, "yarn_ext_factor").asNumber();
  }
  if (options.hasProperty(runtime, "yarn_attn_factor")) {
    yarn_attn_factor = options.getProperty(runtime, "yarn_attn_factor").asNumber();
  }
  if (options.hasProperty(runtime, "yarn_beta_fast")) {
    yarn_beta_fast = options.getProperty(runtime, "yarn_beta_fast").asNumber();
  }
  if (options.hasProperty(runtime, "yarn_beta_slow")) {
    yarn_beta_slow = options.getProperty(runtime, "yarn_beta_slow").asNumber();
  }

  // Parse LoRA adapters to native structure
  std::vector<std::pair<std::string, float>> lora_adapters;
  if (options.hasProperty(runtime, "lora_adapters") && options.getProperty(runtime, "lora_adapters").isObject()) {
    jsi::Object lora_obj = options.getProperty(runtime, "lora_adapters").asObject(runtime);
    if (lora_obj.isArray(runtime)) {
      jsi::Array lora_array = lora_obj.asArray(runtime);
      size_t n_lora = lora_array.size(runtime);

      for (size_t i = 0; i < n_lora; i++) {
        if (lora_array.getValueAtIndex(runtime, i).isObject()) {
          jsi::Object adapter = lora_array.getValueAtIndex(runtime, i).asObject(runtime);
          if (adapter.hasProperty(runtime, "path") && adapter.getProperty(runtime, "path").isString()) {
            std::string lora_path = adapter.getProperty(runtime, "path").asString(runtime).utf8(runtime);
            float lora_scale = 1.0f; // Default scale
            if (adapter.hasProperty(runtime, "scale") && adapter.getProperty(runtime, "scale").isNumber()) {
              lora_scale = adapter.getProperty(runtime, "scale").asNumber();
            }
            lora_adapters.emplace_back(lora_path, lora_scale);
          }
        }
      }
    }
  }

  // Create Promise constructor
  auto Promise = runtime.global().getPropertyAsFunction(runtime, "Promise");
  
  auto executor = jsi::Function::createFromHostFunction(
    runtime,
    jsi::PropNameID::forAscii(runtime, "executor"),
    2,
    [this, model_path, n_ctx, n_batch, n_ubatch, n_keep, use_mmap, use_mlock, use_jinja, embedding, n_threads, n_gpu_layers, logits_file, rope_freq_base, rope_freq_scale, seed, verbosity, yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow, chat_template, lora_adapters, reasoning_budget, reasoning_format, thinking_forced_open, parse_tool_calls, parallel_tool_calls](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* args, size_t count) -> jsi::Value {
      
      auto resolve = std::make_shared<jsi::Function>(args[0].asObject(runtime).asFunction(runtime));
      auto reject = std::make_shared<jsi::Function>(args[1].asObject(runtime).asFunction(runtime));
      
      // Create shared references to runtime and invoker for thread safety
      auto runtimePtr = &runtime;
      auto invoker = jsInvoker_;
      auto selfPtr = shared_from_this();
      
      // Launch background thread for model initialization
      std::thread([selfPtr, model_path, n_ctx, n_batch, n_ubatch, n_keep, use_mmap, use_mlock, use_jinja, embedding, n_threads, n_gpu_layers, logits_file, rope_freq_base, rope_freq_scale, seed, verbosity, yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow, chat_template, lora_adapters, reasoning_budget, reasoning_format, thinking_forced_open, parse_tool_calls, parallel_tool_calls, resolve, reject, runtimePtr, invoker]() {
        try {
          // Thread-safe access to member variables
          std::lock_guard<std::mutex> lock(selfPtr->mutex_);
          
          // Load all available backends (CPU is dynamically loaded when GGML_BACKEND_DL is enabled)
          // With GGML_BACKEND_DL=ON, ALL backends (CPU + GPU) are dynamically loaded
          // CPU backend is in libggml-cpu.so, GPU backends are in libggml-opencl.so, libggml-vulkan.so
          #ifdef __ANDROID__
          // Load CPU backend directly - Android's linker will find it in the same directory
          void* cpu_handle = dlopen("libggml-cpu.so", RTLD_LAZY | RTLD_LOCAL);
          if (cpu_handle) {
            typedef ggml_backend_reg_t (*backend_init_fn_t)();
            backend_init_fn_t backend_init = (backend_init_fn_t)dlsym(cpu_handle, "ggml_backend_init");
            if (backend_init) {
              ggml_backend_reg_t cpu_backend = backend_init();
              if (cpu_backend) {
                ggml_backend_register(cpu_backend);
              }
            }
          }
          
          // Load GPU backends (OpenCL, Vulkan) if present - they will be found by name
          ggml_backend_load_all();
          #else
          ggml_backend_load_all();
          #endif
          
          // Verify at least CPU backend was loaded
          if (ggml_backend_reg_count() == 0) {
            throw std::runtime_error("No backends registered - CPU backend library not found");
          }
          
          // Initialize llama backend
          llama_backend_init();

          // Initialize params with defaults
          rn_common_params params;

          // Set default sampling parameters
          params.sampling = common_params_sampling();

          // Set all parsed native values
          params.model.path = model_path;
          params.n_ctx = n_ctx;
          params.n_batch = n_batch;
          params.n_ubatch = n_ubatch;
          params.n_keep = n_keep;
          params.use_mmap = use_mmap;
          params.use_mlock = use_mlock;
          params.use_jinja = use_jinja;
          params.embedding = embedding;
          params.cpuparams.n_threads = n_threads;
          params.n_gpu_layers = n_gpu_layers;
          params.logits_file = logits_file;
          params.rope_freq_base = rope_freq_base;
          params.rope_freq_scale = rope_freq_scale;
          params.sampling.seed = seed;
          params.verbosity = verbosity;
          params.yarn_ext_factor = yarn_ext_factor;
          params.yarn_attn_factor = yarn_attn_factor;
          params.yarn_beta_fast = yarn_beta_fast;
          params.yarn_beta_slow = yarn_beta_slow;
          
          // Set thinking and reasoning parameters
          params.reasoning_budget = reasoning_budget;
          params.reasoning_format = reasoning_format;

          if (!chat_template.empty()) {
            params.chat_template = chat_template;
          }

          // Add LoRA adapters
          for (const auto& lora : lora_adapters) {
            common_adapter_lora_info lora_info;
            lora_info.path = lora.first;
            lora_info.scale = lora.second;
            params.lora_adapters.push_back(lora_info);
          }

          // Initialize using common_init_from_params
          common_init_result result;
          
          try {
            result = common_init_from_params(params);
            
            // Check if initialization was successful
            if (!result.model || !result.context) {
              throw std::runtime_error("Failed to initialize model and context");
            }
          } catch (const std::exception& e) {
            // If we were trying to use GPU and got an error, retry with CPU-only
            if (params.n_gpu_layers > 0) {
              params.n_gpu_layers = 0;
              
              try {
                result = common_init_from_params(params);
                
                if (!result.model || !result.context) {
                  throw std::runtime_error("Failed to initialize model and context even with CPU-only mode");
                }
              } catch (const std::exception& cpu_e) {
                throw std::runtime_error(std::string("Model initialization failed: ") + cpu_e.what());
              }
            } else {
              // Was already CPU-only, re-throw the original error
              throw std::runtime_error(std::string("Model initialization failed: ") + e.what());
            }
          }

          // Create and initialize rn_llama_context
          selfPtr->rn_ctx_ = std::make_unique<facebook::react::rn_llama_context>();
          selfPtr->rn_ctx_->model = result.model.release();
          selfPtr->rn_ctx_->ctx = result.context.release();
          selfPtr->rn_ctx_->model_loaded = true;
          selfPtr->rn_ctx_->vocab = llama_model_get_vocab(selfPtr->rn_ctx_->model);

          // Create a rn_common_params from the common_params
          rn_common_params rn_params;
          // Copy the base class fields
          static_cast<common_params&>(rn_params) = params;
          // Set additional fields
          rn_params.use_jinja = params.use_jinja;
          // Use the reasoning_format from params instead of hardcoding to NONE
          rn_params.reasoning_format = params.reasoning_format;
          
          // Configure chat template kwargs for thinking and tool calling functionality
          // This ensures that the thinking feature is available as an option when supported
          
          // Configure chat template kwargs based on parsed options
          // reasoning_budget: -1 = unlimited thinking, 0 = disabled, >0 = limited thinking
          // This parameter comes from the JSI options and controls the thinking feature
          if (reasoning_budget != 0) {
              // Enable thinking if reasoning_budget is not 0 (allows -1 for unlimited or positive values)
              params.default_template_kwargs["enable_thinking"] = "true";
          } else {
              // Disable thinking if reasoning_budget is 0
              params.default_template_kwargs["enable_thinking"] = "false";
          }
          
          // Add other important thinking-related kwargs based on reasoning_format
          if (reasoning_format != COMMON_REASONING_FORMAT_NONE) {
              // If reasoning is enabled, we can add thinking_forced_open as an option
              // This allows users to force thinking output when needed
              params.default_template_kwargs["thinking_forced_open"] = thinking_forced_open ? "true" : "false";
              
              // reasoning_in_content controls whether reasoning appears in the main content
              // Default to false for cleaner output, but can be overridden
              params.default_template_kwargs["reasoning_in_content"] = "false";
          }
          
          // parse_tool_calls is enabled by default, but can be overridden by user options
          // If use_jinja is enabled, parse_tool_calls should also be enabled for better tool support
          // This is because Jinja templates often provide better tool calling capabilities
          bool effective_parse_tool_calls = parse_tool_calls || use_jinja;
          params.default_template_kwargs["parse_tool_calls"] = effective_parse_tool_calls ? "true" : "false";
          
          // parallel_tool_calls allows multiple tool calls in a single response
          // Can be enabled for supported models
          params.default_template_kwargs["parallel_tool_calls"] = parallel_tool_calls ? "true" : "false";
          
          // Note: Users can override these kwargs by setting them in params.default_template_kwargs
          // before calling this function, or by using the --chat-template-kwargs CLI argument
          
          // Now assign to the context
          selfPtr->rn_ctx_->params = rn_params;

          selfPtr->rn_ctx_->chat_templates = common_chat_templates_init(selfPtr->rn_ctx_->model, params.chat_template);
          try {
              common_chat_format_example(selfPtr->rn_ctx_->chat_templates.get(), params.use_jinja, params.default_template_kwargs);
          } catch (const std::exception & e) {
              // Fallback to chatml if the original template parsing fails
              selfPtr->rn_ctx_->chat_templates = common_chat_templates_init(selfPtr->rn_ctx_->model, "chatml");
          }

          // Schedule success callback on JS thread to create JSI objects
          invoker->invokeAsync([selfPtr, resolve, runtimePtr]() {
            try {
              // Create the model object and resolve Promise on JS thread
              jsi::Object modelObject = selfPtr->createModelObject(*runtimePtr, selfPtr->rn_ctx_.get());
              resolve->call(*runtimePtr, modelObject);
            } catch (const std::exception& e) {
              // If conversion fails, create a simple error response
              jsi::Object errorObj(*runtimePtr);
              errorObj.setProperty(*runtimePtr, "error", jsi::String::createFromUtf8(*runtimePtr, e.what()));
              resolve->call(*runtimePtr, errorObj);
            }
          });
          
        } catch (const std::exception& e) {
          // Schedule error callback on JS thread
          std::string errorMsg(e.what());
          invoker->invokeAsync([reject, errorMsg, runtimePtr]() {
            try {
              reject->call(*runtimePtr, jsi::String::createFromUtf8(*runtimePtr, errorMsg));
            } catch (...) {
              // Ignore rejection errors
            }
          });
        }
      }).detach();
      
      return jsi::Value::undefined();
    }
  );
  
  return Promise.callAsConstructor(runtime, std::move(executor));
}

jsi::Object PureCppImpl::createModelObject(jsi::Runtime& runtime, rn_llama_context* rn_ctx) {
  // Create a shared_ptr to a new LlamaCppModel instance with CallInvoker
  auto llamaModel = std::make_shared<LlamaCppModel>(rn_ctx, jsInvoker_);

  // Create a host object from the LlamaCppModel instance
  return jsi::Object::createFromHostObject(runtime, std::move(llamaModel));
}

} // namespace facebook::react

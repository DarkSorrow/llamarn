#include "PureCppImpl.h"

#include <jsi/jsi.h>
#include <algorithm>
#include <array>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <thread>
#include <mutex>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <cstdlib>  // for setenv
#include <sys/stat.h>
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

// Helper function to load CPU variant libraries on Android
// On Android, ggml_backend_load_best() uses filesystem iteration which doesn't work
// with APK-packaged libraries. This function manually loads CPU variant libraries
// using dlopen() with just the library name - Android's linker finds them in the APK.
// We score each variant and only register the best compatible one (score > 0).
static void load_android_cpu_backends() {
#ifdef __ANDROID__
  // Skip if CPU backend is already registered
  if (ggml_backend_reg_by_name("CPU")) {
    return;
  }

  // Try loading all CPU variant libraries (from most advanced to baseline)
  // Score each one and register only the best compatible variant
  static constexpr std::array<const char*, 4> cpu_variants = {{
    "libggml-cpu-android_armv8.6_1.so",  // DOTPROD + FP16 + MATMUL_INT8
    "libggml-cpu-android_armv8.2_2.so",  // DOTPROD + FP16
    "libggml-cpu-android_armv8.2_1.so",  // DOTPROD
    "libggml-cpu-android_armv8.0_1.so",  // Baseline (emulator compatible)
  }};

  using backend_init_fn_t = ggml_backend_reg_t (*)();
  using backend_score_t   = int (*)();

  int best_score = 0;
  void* best_handle = nullptr;
  backend_init_fn_t best_init = nullptr;

  // Score all variants and find the best one
  for (const auto* variant : cpu_variants) {
    void* cpu_handle = dlopen(variant, RTLD_LAZY | RTLD_LOCAL);
    if (cpu_handle) {
      auto score_fn = reinterpret_cast<backend_score_t>(dlsym(cpu_handle, "ggml_backend_score"));
      if (score_fn) {
        int score = score_fn();
        if (score > best_score) {
          // Close previous best handle if we had one
          if (best_handle) {
            dlclose(best_handle);
          }
          best_score = score;
          best_handle = cpu_handle;
          best_init = reinterpret_cast<backend_init_fn_t>(dlsym(cpu_handle, "ggml_backend_init"));
        } else {
          // This variant is not better, close it
          dlclose(cpu_handle);
        }
      } else {
        // No score function, close it
        dlclose(cpu_handle);
      }
    }
  }

  // Register the best variant if we found one
  if (best_handle && best_init && best_score > 0) {
    ggml_backend_reg_t cpu_backend = best_init();
    if (cpu_backend) {
      ggml_backend_register(cpu_backend);
      // Keep the handle open - it will be cleaned up when the backend is unloaded
    } else {
      dlclose(best_handle);
    }
  }
#endif
}

// Helper function to load all Android backends manually
// On Android, ggml_backend_load_best() uses filesystem iteration which doesn't work
// with APK-packaged libraries. This function manually loads all backend libraries
// using dlopen() with just the library name - Android's linker finds them in the APK.
static void load_android_backends() {
#ifdef __ANDROID__
  using backend_init_fn_t = ggml_backend_reg_t (*)();

  // Load Hexagon backend first (Snapdragon DSP) - more performant than Vulkan on Snapdragon devices
  if (!ggml_backend_reg_by_name("HTP")) {
    // FastRPC (used by Hexagon) requires ADSP_LIBRARY_PATH to find HTP libraries
    // Get the app's native library directory by using dladdr on a known symbol
    // All libraries from jniLibs are extracted to the same directory by Android
    void* test_handle = dlopen("libggml.so", RTLD_LAZY | RTLD_LOCAL);
    if (test_handle) {
      // Get a function pointer from the library to use with dladdr
      void* symbol = dlsym(test_handle, "ggml_init");
      if (symbol) {
        Dl_info info;
        if (dladdr(symbol, &info) && info.dli_fname) {
          // Extract directory from library path (e.g., "/data/app/.../lib/arm64/libggml.so" -> "/data/app/.../lib/arm64")
          std::string lib_path = info.dli_fname;
          size_t last_slash = lib_path.find_last_of('/');
          if (last_slash != std::string::npos) {
            std::string lib_dir = lib_path.substr(0, last_slash);
            // Set ADSP_LIBRARY_PATH so FastRPC can find HTP libraries in the same directory
            setenv("ADSP_LIBRARY_PATH", lib_dir.c_str(), 0); // 0 = don't overwrite if already set
          }
        }
      }
      dlclose(test_handle);
    }
    
    void* hexagon_handle = dlopen("libggml-hexagon.so", RTLD_LAZY | RTLD_LOCAL);
    if (hexagon_handle) {
      auto backend_init = reinterpret_cast<backend_init_fn_t>(dlsym(hexagon_handle, "ggml_backend_init"));
      if (backend_init) {
        ggml_backend_reg_t hexagon_backend = backend_init();
        if (hexagon_backend) {
          ggml_backend_register(hexagon_backend);
        }
      }
    }
  }

  // Load OpenCL backend
  if (!ggml_backend_reg_by_name("OpenCL")) {
    void* opencl_handle = dlopen("libggml-opencl.so", RTLD_LAZY | RTLD_LOCAL);
    if (opencl_handle) {
      auto backend_init = reinterpret_cast<backend_init_fn_t>(dlsym(opencl_handle, "ggml_backend_init"));
      if (backend_init) {
        ggml_backend_reg_t opencl_backend = backend_init();
        if (opencl_backend) {
          ggml_backend_register(opencl_backend);
        }
      }
    }
  }

  // Load Vulkan backend (disabled by default on Android due to emulator crashes, but try anyway)
  if (!ggml_backend_reg_by_name("Vulkan")) {
    void* vulkan_handle = dlopen("libggml-vulkan.so", RTLD_LAZY | RTLD_LOCAL);
    if (vulkan_handle) {
      auto backend_init = reinterpret_cast<backend_init_fn_t>(dlsym(vulkan_handle, "ggml_backend_init"));
      if (backend_init) {
        ggml_backend_reg_t vulkan_backend = backend_init();
        if (vulkan_backend) {
          ggml_backend_register(vulkan_backend);
        }
      }
    }
  }
  
  // Load CPU variant libraries (scoring system selects best compatible one)
  load_android_cpu_backends();
#endif
}

// One-time backend initialization — safe to call from multiple threads concurrently.
// Uses std::call_once so backends are loaded and llama_backend_init() is called exactly once,
// even if loadLlamaModelInfo and initLlama race on startup.
static void ensure_backends_loaded() {
  static std::once_flag flag;
  std::call_once(flag, []() {
#ifdef __ANDROID__
    load_android_backends();
#endif
    ggml_backend_load_all();
    if (ggml_backend_reg_count() == 0) {
      throw std::runtime_error("No backends registered — CPU backend library not found");
    }
    llama_backend_init();
  });
}

// Dispatches fn to the JS thread via invoker; silently drops if the runtime is already gone.
template<typename Fn>
static void safe_invoke(const std::shared_ptr<CallInvoker>& invoker, Fn&& fn) {
    try {
        invoker->invokeAsync(std::forward<Fn>(fn));
    } catch (...) {}
}

// Factory method implementation
std::shared_ptr<TurboModule> PureCppImpl::create(std::shared_ptr<CallInvoker> jsInvoker) {
  return std::make_shared<PureCppImpl>(std::move(jsInvoker));
}

PureCppImpl::PureCppImpl(std::shared_ptr<CallInvoker> jsInvoker)
    : NativeRNLlamaCppCxxSpec(jsInvoker), jsInvoker_(jsInvoker) {
}

jsi::Value PureCppImpl::loadLlamaModelInfo(jsi::Runtime &runtime, jsi::String modelPath,
                                            std::optional<jsi::String> mmprojPath) {
  // Parse JSI arguments to native types on JSI thread
  std::string path = modelPath.utf8(runtime);
  SystemUtils::normalizeFilePath(path);

  // Resolve optional mmproj path on the JSI thread before entering the background thread.
  std::string mmproj_path;
  if (mmprojPath.has_value()) {
    mmproj_path = mmprojPath->utf8(runtime);
    SystemUtils::normalizeFilePath(mmproj_path);
  }

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
    [this, path, mmproj_path](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* args, size_t count) -> jsi::Value {

      auto resolve = std::make_shared<jsi::Function>(args[0].asObject(runtime).asFunction(runtime));
      auto reject = std::make_shared<jsi::Function>(args[1].asObject(runtime).asFunction(runtime));

      // Create shared references to runtime and invoker for thread safety
      auto runtimePtr = &runtime;
      auto invoker = jsInvoker_;
      auto selfPtr = shared_from_this();
      
      // Launch background thread for model info loading
      std::thread([selfPtr, path, mmproj_path, resolve, reject, runtimePtr, invoker]() {
        try {
          ensure_backends_loaded();

          // Read mmproj file size (stat only — no model load needed) for VRAM reservation.
          int64_t mmproj_size_bytes = 0;
          if (!mmproj_path.empty()) {
            struct stat st{};
            if (::stat(mmproj_path.c_str(), &st) == 0) {
              mmproj_size_bytes = static_cast<int64_t>(st.st_size);
            }
          }

          // Create model params
          llama_model_params params = llama_model_default_params();
          params.n_gpu_layers = 0; // Use CPU for model info loading

          // Load the model
          llama_model* model = llama_model_load_from_file(path.c_str(), params);

          if (!model) {
            throw std::runtime_error("Failed to load model from file: " + path);
          }

          // Get model information (native types)
          double n_params  = static_cast<double>(llama_model_n_params(model));
          const llama_vocab* vocab = llama_model_get_vocab(model);
          double n_vocab   = static_cast<double>(llama_vocab_n_tokens(vocab));
          double n_context = static_cast<double>(llama_model_n_ctx_train(model));
          double n_embd    = static_cast<double>(llama_model_n_embd(model));

          // Get model description
          std::array<char, 512> buf{};
          llama_model_desc(model, buf.data(), buf.size());
          std::string description = buf[0] ? buf.data() : "Unknown model";

          // Check if GPU is supported
          bool gpuSupported = llama_supports_gpu_offload();

          // Calculate optimal GPU layers, reserving VRAM for mmproj if provided.
          int optimalGpuLayers = 0;
          int64_t available_memory_bytes = SystemUtils::getAvailableMemoryBytes();
          if (gpuSupported) {
            optimalGpuLayers = SystemUtils::getOptimalGpuLayers(model, mmproj_size_bytes);
          }

          // Cooperative ingestion hints derived from GPU availability.
          bool is_cpu_only   = (optimalGpuLayers == 0);
          int  chunk_size    = is_cpu_only ? 32 : 128;

          // Extract quantization type from model description
          std::string desc(buf.data());
          std::string quantType = "Unknown";
          size_t qPos = desc.find(" Q");
          if (qPos != std::string::npos && qPos + 5 <= desc.length()) {
            // Extract quantization string (like Q4_K, Q5_K, Q8_0)
            quantType = desc.substr(qPos + 1, 4);
            // Remove any trailing non-alphanumeric characters
            quantType.erase(std::find_if(quantType.rbegin(), quantType.rend(), [](char c) {
              return std::isalnum(static_cast<unsigned char>(c));
            }).base(), quantType.end());
          }

          // Layer count, actual model size, and architecture from GGUF metadata
          int32_t n_layers = llama_model_n_layer(model);
          double model_size_bytes = static_cast<double>(llama_model_size(model));
          // Per-layer VRAM estimate and total estimated VRAM for optimal GPU layers
          int64_t bytes_per_layer = n_layers > 0 ? static_cast<int64_t>(model_size_bytes) / n_layers : 0;
          double available_memory_mb = static_cast<double>(available_memory_bytes) / (1024.0 * 1024.0);
          double estimated_vram_mb   = optimalGpuLayers > 0 && bytes_per_layer > 0
              ? static_cast<double>(optimalGpuLayers * bytes_per_layer) / (1024.0 * 1024.0)
              : 0.0;
          double mmproj_size_mb = mmproj_size_bytes > 0
              ? static_cast<double>(mmproj_size_bytes) / (1024.0 * 1024.0)
              : -1.0; // negative sentinel: mmprojPath was not provided

          std::array<char, 128> arch_buf{"unknown"};
          llama_model_meta_val_str(model, "general.architecture", arch_buf.data(), arch_buf.size());
          std::string architecture = arch_buf.data();

          // Free the model
          llama_model_free(model);

          safe_invoke(invoker, [resolve, reject, n_params, n_vocab, n_context, n_embd, description,
                                gpuSupported, optimalGpuLayers, quantType, n_layers,
                                model_size_bytes, architecture,
                                available_memory_mb, estimated_vram_mb,
                                mmproj_size_mb, is_cpu_only, chunk_size, runtimePtr]() {
            try {
              jsi::Object result(*runtimePtr);
              result.setProperty(*runtimePtr, "n_params",            jsi::Value(n_params));
              result.setProperty(*runtimePtr, "n_vocab",             jsi::Value(n_vocab));
              result.setProperty(*runtimePtr, "n_context",           jsi::Value(n_context));
              result.setProperty(*runtimePtr, "n_embd",              jsi::Value(n_embd));
              result.setProperty(*runtimePtr, "n_layers",            jsi::Value(static_cast<double>(n_layers)));
              result.setProperty(*runtimePtr, "model_size_bytes",    jsi::Value(model_size_bytes));
              result.setProperty(*runtimePtr, "description",         jsi::String::createFromUtf8(*runtimePtr, description));
              result.setProperty(*runtimePtr, "gpuSupported",        jsi::Value(gpuSupported));
              result.setProperty(*runtimePtr, "optimalGpuLayers",    jsi::Value(optimalGpuLayers));
              result.setProperty(*runtimePtr, "quant_type",          jsi::String::createFromUtf8(*runtimePtr, quantType));
              result.setProperty(*runtimePtr, "architecture",        jsi::String::createFromUtf8(*runtimePtr, architecture));
              result.setProperty(*runtimePtr, "availableMemoryMB",   jsi::Value(available_memory_mb));
              result.setProperty(*runtimePtr, "estimatedVramMB",     jsi::Value(estimated_vram_mb));
              result.setProperty(*runtimePtr, "suggestedChunkSize",  jsi::Value(static_cast<double>(chunk_size)));
              result.setProperty(*runtimePtr, "isCpuOnly",           jsi::Value(is_cpu_only));
              if (mmproj_size_mb >= 0.0) {
                result.setProperty(*runtimePtr, "mmprojSizeMB",      jsi::Value(mmproj_size_mb));
              }
              resolve->call(*runtimePtr, result);
            } catch (const std::exception& e) {
              // JSI object creation failed — reject (not resolve) so JS catch() sees it
              try { reject->call(*runtimePtr, jsi::String::createFromUtf8(*runtimePtr, e.what())); } catch (...) {}
            } catch (...) {
              try { reject->call(*runtimePtr, jsi::String::createFromUtf8(*runtimePtr, "loadLlamaModelInfo: JSI error")); } catch (...) {}
            }
          });

        } catch (const std::exception& e) {
          std::string msg = e.what();
          safe_invoke(invoker, [reject, msg, runtimePtr]() {
            try { reject->call(*runtimePtr, jsi::String::createFromUtf8(*runtimePtr, msg)); } catch (...) {}
          });
        } catch (...) {
          safe_invoke(invoker, [reject, runtimePtr]() {
            try { reject->call(*runtimePtr, jsi::String::createFromUtf8(*runtimePtr, "loadLlamaModelInfo failed")); } catch (...) {}
          });
        }
      }).detach();
      
      return jsi::Value::undefined();
    }
  );
  
  return Promise.callAsConstructor(runtime, std::move(executor));
}

// Returns a valid init result or throws std::runtime_error.
// Attempts GPU first if params.n_gpu_layers > 0; downgrades to CPU on failure.
static common_init_result_ptr try_init_with_gpu_fallback(rn_common_params& params) {
    if (params.n_gpu_layers > 0) {
        try {
            auto r = common_init_from_params(params);
            if (r && r->model() && r->context()) return r;
        } catch (const std::exception&) {
            // GPU init failed — fall through to CPU
        }
        params.n_gpu_layers = 0;
    }
    auto r = common_init_from_params(params);
    if (!r || !r->model() || !r->context())
        throw std::runtime_error("model initialization failed (CPU)");
    return r;
}

// Initializes chat templates with automatic chatml fallback; never throws.
static void init_chat_templates_safe(rn_llama_context* ctx, const rn_common_params& params) {
    try {
        ctx->chat_templates = common_chat_templates_init(ctx->model, params.chat_template);
        try {
            common_chat_format_example(
                ctx->chat_templates.get(), params.use_jinja, params.default_template_kwargs);
        } catch (...) {}  // validation failure is non-fatal
    } catch (...) {
        try { ctx->chat_templates = common_chat_templates_init(ctx->model, "chatml"); }
        catch (...) {}  // chatml fallback failed — chat won't work but don't crash
    }
}

struct InitLlamaParams {
  std::string model_path;
  int n_ctx, n_batch, n_ubatch, n_keep;
  bool use_mmap, use_mlock, use_jinja, embedding;
  int n_threads, n_gpu_layers;
  std::string logits_file;
  float rope_freq_base, rope_freq_scale;
  uint32_t seed;
  int verbosity;
  float yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow;
  std::string chat_template;
  std::vector<std::pair<std::string, float>> lora_adapters;
  int reasoning_budget;
  common_reasoning_format reasoning_format;
  bool thinking_forced_open, parse_tool_calls, parallel_tool_calls;
  // Multimodal
  std::string mmproj_path;
  std::string image_marker;
  uint32_t    declared_capabilities = 0;
  // Cooperative ingestion loop
  int  chunk_size  = 128;
  bool is_cpu_only = false;
  int  prompt_chunk_gap_ms = 5;
};

struct ModelInitResult {
    std::unique_ptr<rn_llama_context> rn_ctx;
    common_init_result_ptr            init_result; // keeps llama_model / llama_context alive
};

// Performs all heavy model-loading work on the background thread.
// No JSI, no Promise, no invoker — returns on success, throws on failure.
static ModelInitResult do_init_llama(const InitLlamaParams& p) {
    ensure_backends_loaded();

    // ── 1. Build rn_common_params ──────────────────────────────────────────
    rn_common_params params;
    params.sampling            = common_params_sampling();
    params.model.path          = p.model_path;
    params.n_ctx               = p.n_ctx;
    params.n_batch             = p.n_batch;
    params.n_ubatch            = p.n_ubatch;
    params.n_keep              = p.n_keep;
    params.use_mmap            = p.use_mmap;
    params.use_mlock           = p.use_mlock;
    params.use_jinja           = p.use_jinja;
    params.embedding           = p.embedding;
    params.cpuparams.n_threads = p.n_threads;
    params.n_gpu_layers        = p.n_gpu_layers;
    params.logits_file         = p.logits_file;
    params.rope_freq_base      = p.rope_freq_base;
    params.rope_freq_scale     = p.rope_freq_scale;
    params.sampling.seed       = p.seed;
    params.verbosity           = p.verbosity;
    params.yarn_ext_factor     = p.yarn_ext_factor;
    params.yarn_attn_factor    = p.yarn_attn_factor;
    params.yarn_beta_fast      = p.yarn_beta_fast;
    params.yarn_beta_slow      = p.yarn_beta_slow;
    params.reasoning_budget    = p.reasoning_budget;
    params.reasoning_format    = p.reasoning_format;
    if (!p.chat_template.empty()) params.chat_template = p.chat_template;
    for (const auto& lora : p.lora_adapters) {
        common_adapter_lora_info li;
        li.path  = lora.first;
        li.scale = lora.second;
        params.lora_adapters.push_back(li);
    }

    // Set kwargs on `params` BEFORE copying to rn_params (fixes silent data-loss bug:
    // previously kwargs were set on `params` AFTER rn_params was copy-constructed from it).
    params.default_template_kwargs["enable_thinking"]     = (p.reasoning_budget != 0) ? "true" : "false";
    if (p.reasoning_format != COMMON_REASONING_FORMAT_NONE) {
        params.default_template_kwargs["thinking_forced_open"] = p.thinking_forced_open ? "true" : "false";
        params.default_template_kwargs["reasoning_in_content"] = "false";
    }
    bool effective_parse = p.parse_tool_calls || p.use_jinja;
    params.default_template_kwargs["parse_tool_calls"]    = effective_parse ? "true" : "false";
    params.default_template_kwargs["parallel_tool_calls"] = p.parallel_tool_calls ? "true" : "false";

    // rn_params is now copied AFTER all kwargs are set → kwargs are preserved
    rn_common_params rn_params;
    static_cast<common_params&>(rn_params) = params;
    rn_params.use_jinja        = p.use_jinja;
    rn_params.reasoning_format = p.reasoning_format;
    rn_params.chunk_size       = p.chunk_size;
    rn_params.is_cpu_only      = p.is_cpu_only;
    rn_params.prompt_chunk_gap_ms = p.prompt_chunk_gap_ms;

    // ── 2. Model init with GPU→CPU fallback ────────────────────────────────
    auto init_result = try_init_with_gpu_fallback(params);

    // ── 3. Build rn_llama_context ──────────────────────────────────────────
    auto rn_ctx = std::make_unique<rn_llama_context>();
    rn_ctx->model        = init_result->model();
    rn_ctx->ctx          = init_result->context();
    rn_ctx->model_loaded = true;
    rn_ctx->vocab        = llama_model_get_vocab(rn_ctx->model);
    rn_ctx->params       = rn_params;
    rn_ctx->gen_batch    = llama_batch_init(1, 0, 1);
    rn_ctx->ingest_batch = llama_batch_init(rn_ctx->params.n_batch, 0, 1);
    rn_ctx->batches_initialized = true;

    llama_set_abort_callback(
        rn_ctx->ctx,
        [](void* data) -> bool {
            return static_cast<rn_llama_context*>(data)->abort_generation.load(std::memory_order_relaxed);
        },
        rn_ctx.get());

    // ── 4. Chat templates ──────────────────────────────────────────────────
    init_chat_templates_safe(rn_ctx.get(), rn_params);

    // ── 5. Multimodal (non-fatal: continue even if mmproj fails) ──────────
    rn_ctx->declared_capabilities = p.declared_capabilities;
    if (!p.mmproj_path.empty()) {
        mtmd_context_params mparams  = mtmd_context_params_default();
        mparams.use_gpu              = (p.n_gpu_layers != 0);
        mparams.n_threads            = p.n_threads;
        mparams.print_timings        = false;
        mparams.warmup               = false;
        if (!p.image_marker.empty()) mparams.media_marker = p.image_marker.c_str();
        rn_ctx->mtmd_ctx          = mtmd_init_from_file(p.mmproj_path.c_str(), rn_ctx->model, mparams);
        rn_ctx->multimodal_loaded = (rn_ctx->mtmd_ctx != nullptr);
    }

    return { std::move(rn_ctx), std::move(init_result) };
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

  std::string mmproj_path;
  std::string image_marker;
  uint32_t declared_capabilities = 0;

  if (options.hasProperty(runtime, "mmproj") &&
      options.getProperty(runtime, "mmproj").isString()) {
    mmproj_path = options.getProperty(runtime, "mmproj").asString(runtime).utf8(runtime);
    SystemUtils::normalizeFilePath(mmproj_path);
  }
  if (options.hasProperty(runtime, "image_marker") &&
      options.getProperty(runtime, "image_marker").isString()) {
    image_marker = options.getProperty(runtime, "image_marker").asString(runtime).utf8(runtime);
  }
  if (options.hasProperty(runtime, "capabilities") &&
      options.getProperty(runtime, "capabilities").isObject()) {
    jsi::Object caps_obj = options.getProperty(runtime, "capabilities").asObject(runtime);
    if (caps_obj.isArray(runtime)) {
      jsi::Array caps_arr = caps_obj.asArray(runtime);
      std::vector<std::string> cap_names;
      for (size_t i = 0; i < caps_arr.size(runtime); ++i) {
        auto v = caps_arr.getValueAtIndex(runtime, i);
        if (v.isString()) cap_names.push_back(v.asString(runtime).utf8(runtime));
      }
      declared_capabilities = capabilities_from_strings(cap_names);
    }
  }

  // Cooperative ingestion loop settings
  int  chunk_size  = 128;
  bool is_cpu_only = false;
  int  prompt_chunk_gap_ms = 5;
  SystemUtils::setIfExists(runtime, options, "chunk_size",  chunk_size);
  SystemUtils::setIfExists(runtime, options, "is_cpu_only", is_cpu_only);
  SystemUtils::setIfExists(runtime, options, "prompt_chunk_gap_ms", prompt_chunk_gap_ms);

  // Pack all parsed values into a shared struct so the lambda captures stay minimal.
  auto p = std::make_shared<InitLlamaParams>();
  p->model_path           = model_path;
  p->n_ctx                = n_ctx;
  p->n_batch              = n_batch;
  p->n_ubatch             = n_ubatch;
  p->n_keep               = n_keep;
  p->use_mmap             = use_mmap;
  p->use_mlock            = use_mlock;
  p->use_jinja            = use_jinja;
  p->embedding            = embedding;
  p->n_threads            = n_threads;
  p->n_gpu_layers         = n_gpu_layers;
  p->logits_file          = logits_file;
  p->rope_freq_base       = rope_freq_base;
  p->rope_freq_scale      = rope_freq_scale;
  p->seed                 = seed;
  p->verbosity            = verbosity;
  p->yarn_ext_factor      = yarn_ext_factor;
  p->yarn_attn_factor     = yarn_attn_factor;
  p->yarn_beta_fast       = yarn_beta_fast;
  p->yarn_beta_slow       = yarn_beta_slow;
  p->chat_template        = chat_template;
  p->lora_adapters        = std::move(lora_adapters);
  p->reasoning_budget     = reasoning_budget;
  p->reasoning_format     = reasoning_format;
  p->thinking_forced_open = thinking_forced_open;
  p->parse_tool_calls     = parse_tool_calls;
  p->parallel_tool_calls  = parallel_tool_calls;
  p->mmproj_path           = mmproj_path;
  p->image_marker          = image_marker;
  p->declared_capabilities = declared_capabilities;
  p->chunk_size            = std::clamp(chunk_size, 8, 512);
  p->is_cpu_only           = is_cpu_only;
  p->prompt_chunk_gap_ms   = std::max(0, prompt_chunk_gap_ms);

  // Create Promise constructor
  auto Promise = runtime.global().getPropertyAsFunction(runtime, "Promise");

  auto executor = jsi::Function::createFromHostFunction(
    runtime,
    jsi::PropNameID::forAscii(runtime, "executor"),
    2,
    [this, p](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* args, size_t count) -> jsi::Value {

      auto resolve = std::make_shared<jsi::Function>(args[0].asObject(runtime).asFunction(runtime));
      auto reject = std::make_shared<jsi::Function>(args[1].asObject(runtime).asFunction(runtime));

      // Create shared references to runtime and invoker for thread safety
      auto runtimePtr = &runtime;
      auto invoker = jsInvoker_;
      auto selfPtr = shared_from_this();

      // Launch background thread for model initialization
      std::thread([selfPtr, p, resolve, reject, runtimePtr, invoker]() {
        // ── Phase 1: all heavy work — no JSI, no lock ──────────────────────
        ModelInitResult r;
        try {
          r = do_init_llama(*p);
        } catch (const std::exception& e) {
          std::string msg = e.what();
          safe_invoke(invoker, [reject, msg, runtimePtr]() {
            try { reject->call(*runtimePtr, jsi::String::createFromUtf8(*runtimePtr, msg)); } catch (...) {}
          });
          return;
        } catch (...) {
          safe_invoke(invoker, [reject, runtimePtr]() {
            try { reject->call(*runtimePtr, jsi::String::createFromUtf8(*runtimePtr, "initLlama failed")); } catch (...) {}
          });
          return;
        }

        // ── Phase 2: store result under mutex (microseconds only) ───────────
        {
          std::lock_guard<std::mutex> lock(selfPtr->mutex_);
          selfPtr->rn_ctx_.reset();
          selfPtr->init_result_.reset();
          selfPtr->rn_ctx_      = std::move(r.rn_ctx);
          selfPtr->init_result_ = std::move(r.init_result);
        }

        // ── Phase 3: resolve Promise on JS thread ───────────────────────────
        safe_invoke(invoker, [selfPtr, resolve, reject, runtimePtr]() {
          try {
            jsi::Object modelObject = selfPtr->createModelObject(*runtimePtr, selfPtr->rn_ctx_.get());
            resolve->call(*runtimePtr, modelObject);
          } catch (const std::exception& e) {
            try { reject->call(*runtimePtr, jsi::String::createFromUtf8(*runtimePtr, e.what())); } catch (...) {}
          } catch (...) {
            try { reject->call(*runtimePtr, jsi::String::createFromUtf8(*runtimePtr, "model object creation failed")); } catch (...) {}
          }
        });
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

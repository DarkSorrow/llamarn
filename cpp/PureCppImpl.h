#pragma once

#include <RNLlamaCppSpecJSI.h>

#include <jsi/jsi.h>
#include <ReactCommon/TurboModule.h>
#include <memory>
#include <string>
#include <mutex>

// Include the header with the full definition of rn_llama_context
#include "rn-llama.h"

// Forward declarations for C++ only
struct llama_model;
struct llama_context;

// Forward declaration for namespace-qualified types
namespace facebook {
namespace react {
class CallInvoker;
struct rn_llama_context; // Properly scope the forward declaration
class LlamaCppModel;     // Forward declare LlamaCppModel
} // namespace react
} // namespace facebook


namespace facebook::react {

// Note: The class name is PureCppImpl, and it derives from your project's C++ spec
class PureCppImpl : public NativeRNLlamaCppCxxSpec<PureCppImpl>, public std::enable_shared_from_this<PureCppImpl> {
public:
    // Constructor
    PureCppImpl(std::shared_ptr<CallInvoker> jsInvoker);

    // Factory method to match LlamaCppRnModule.h
    static std::shared_ptr<TurboModule> create(std::shared_ptr<CallInvoker> jsInvoker);


    // --- JSI Host Functions defined in your new Spec ---
    jsi::Value initLlama(jsi::Runtime &rt, jsi::Object params); // Matches LlamaModelParams
    jsi::Value loadLlamaModelInfo(jsi::Runtime &rt, jsi::String modelPath);

    // Existing multiply method for testing - will be removed later
    double multiply(jsi::Runtime &rt, double a, double b);

private:
    // Helper method to create the HostObject that wraps the llama context and its methods
    jsi::Object createModelObject(jsi::Runtime& runtime, struct rn_llama_context* rn_ctx);

    // Context for the currently loaded model, if any.
    // The actual definition of rn_llama_context should be in "rn-llama.h"
    std::unique_ptr<struct rn_llama_context> rn_ctx_;

    // Mutex for thread safety when accessing rn_ctx_ or other shared resources
    std::mutex mutex_;
    
    // CallInvoker for async operations
    std::shared_ptr<CallInvoker> jsInvoker_;
};

} // namespace facebook::react

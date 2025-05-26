#pragma once

#include <jsi/jsi.h>
#include <string>
#include <thread>
#include <vector>
#include <type_traits>  // For std::enable_if, std::is_arithmetic
#include "llama.h"

namespace facebook::react {

class SystemUtils {
public:
  /**
    * Calculates the optimal number of threads for llama.cpp based on available CPU cores.
    * The logic follows these rules:
    * - If only 1 core, use 1 thread
    * - If less than 4 cores, use (cores - 1) threads
    * - If 4+ cores, use (cores - 2) threads
    */
  static int getOptimalThreadCount();

  /**
    * Normalizes a file path by removing file:// prefix if present.
    * This is useful for handling paths that might come from different sources.
    */
  static void normalizeFilePath(std::string& path);

  /**
    * Calculates the optimal number of GPU layers for model inference.
    * Should only be called if llama_supports_gpu_offload() returns true.
    * The calculation takes into account:
    * - Available GPU memory
    * - Model size and parameters
    * - Platform-specific optimizations
    * - Current quantization method
    *
    * @param model Pointer to an already loaded llama model
    * @return Optimal number of GPU layers (0 if GPU not supported)
    */
  static int getOptimalGpuLayers(struct llama_model* model);

  /**
   * Helper functions to easily set values from a JSI object if the property exists.
   * Returns true if the property was found and the value was set.
   */
  // Template for all numeric types
  template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
  static bool setIfExists(jsi::Runtime& rt, const jsi::Object& options, const std::string& key, T& outValue) {
    if (options.hasProperty(rt, key.c_str())) {
      jsi::Value val = options.getProperty(rt, key.c_str());
      if (val.isNumber()) {
        if (std::is_unsigned<T>::value && val.asNumber() < 0) {
          // Skip negative values for unsigned types
          return false;
        }
        outValue = static_cast<T>(val.asNumber());
        return true;
      }
    }
    return false;
  }

  // Specialized version for std::string
  static bool setIfExists(jsi::Runtime& rt, const jsi::Object& options, const std::string& key, std::string& outValue);

  // Specialized version for bool
  static bool setIfExists(jsi::Runtime& rt, const jsi::Object& options, const std::string& key, bool& outValue);

  // Specialized version for vector
  static bool setIfExists(jsi::Runtime& rt, const jsi::Object& options, const std::string& key, std::vector<jsi::Value>& outValue);
};

} // namespace facebook::react
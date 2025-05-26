#include "SystemUtils.h"
#include "llama.h"
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <cinttypes> // For PRId64 macros

// Platform-specific includes
#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/host_info.h>
#include <TargetConditionals.h>
#elif defined(__ANDROID__)
#include <sys/sysinfo.h>
#include <unistd.h>
#include <jni.h>
#endif

namespace facebook::react {

// Memory fallback constants (clearly defined for future maintenance)
constexpr int64_t FALLBACK_IOS_MEMORY = 2LL * 1024 * 1024 * 1024;     // 2GB default
constexpr int64_t FALLBACK_ANDROID_MEMORY = 3LL * 1024 * 1024 * 1024; // 3GB for Android
constexpr int64_t DEFAULT_FALLBACK_MEMORY = 2LL * 1024 * 1024 * 1024; // 2GB default

int SystemUtils::getOptimalThreadCount() {
    int cpuCores = std::thread::hardware_concurrency();

    if (cpuCores <= 1) {
        return 1;
    } else if (cpuCores < 4) {
        return cpuCores - 1;
    } else {
        return cpuCores - 2;
    }
}

void SystemUtils::normalizeFilePath(std::string& path) {
    // Remove file:// prefix if present
    if (path.substr(0, 7) == "file://") {
        path = path.substr(7);
    }
}

// Get total physical memory of the device in bytes
int64_t getTotalPhysicalMemory() {
    int64_t total_memory = 0;

#if defined(__APPLE__) && TARGET_OS_IPHONE
    // For iOS devices
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    size_t length = sizeof(int64_t);
    sysctl(mib, 2, &total_memory, &length, NULL, 0);
#elif defined(__ANDROID__)
    // For Android devices
    struct sysinfo memInfo;
    if (sysinfo(&memInfo) == 0) {
        // Protect against overflow when multiplying
        if (memInfo.mem_unit > 0 && memInfo.totalram > 0 &&
            static_cast<uint64_t>(memInfo.totalram) * memInfo.mem_unit < (1ULL << 63)) {
            total_memory = static_cast<int64_t>(memInfo.totalram) * memInfo.mem_unit;
        }
    }

    // Fallback: Parse /proc/meminfo if sysinfo failed or returned invalid values
    if (total_memory <= 0) {
        std::ifstream meminfo("/proc/meminfo");
        std::string line;
        while (std::getline(meminfo, line)) {
            if (line.compare(0, 9, "MemTotal:") == 0) {
                // Format is "MemTotal: XXXXX kB"
                int64_t kb_memory = 0;
                std::istringstream iss(line.substr(9));
                iss >> kb_memory;
                total_memory = kb_memory * 1024; // Convert kB to bytes
                break;
            }
        }
        meminfo.close(); // Explicitly close the file descriptor
    }
#endif

    // Fallback to a conservative estimate if we couldn't get the actual memory
    if (total_memory <= 0) {
#if defined(__APPLE__) && TARGET_OS_IPHONE
        total_memory = FALLBACK_IOS_MEMORY;
#elif defined(__ANDROID__)
        total_memory = FALLBACK_ANDROID_MEMORY;
#else
        total_memory = DEFAULT_FALLBACK_MEMORY;
#endif
    }

    return total_memory;
}

int SystemUtils::getOptimalGpuLayers(struct llama_model* model) {
    // Get model parameters
    const int n_layer = llama_model_n_layer(model);
    const int64_t n_params = llama_model_n_params(model);

    // Estimate bytes per layer based on model parameters
    int64_t bytes_per_layer = (n_params * sizeof(float)) / n_layer;

    // Get actual device memory
    int64_t total_memory = getTotalPhysicalMemory();

    // On mobile devices, we don't have dedicated VRAM - it's shared with system RAM
    // Estimate available GPU memory as a percentage of total memory
    int64_t available_vram = 0;

#if defined(__APPLE__) && TARGET_OS_IPHONE
    // iOS devices - use 25% of total RAM
    available_vram = total_memory / 4;
#elif defined(__ANDROID__)
    // Android - use 20% of total RAM (more conservative)
    available_vram = total_memory / 5;
#endif

    // Calculate optimal layers using 80% of available GPU memory
    // We use a higher percentage since we're already being conservative with the available_vram estimate
    int64_t target_vram = (available_vram * 80) / 100;
    int possible_layers = target_vram / bytes_per_layer;

    // Clamp to total layers and ensure we don't go below 1
    int optimal_layers = std::max(1, std::min(possible_layers, n_layer));

    return optimal_layers;
}

// helper function for setting options
// Implementations of non-template specializations

// For std::string
bool SystemUtils::setIfExists(jsi::Runtime& rt, const jsi::Object& options, const std::string& key, std::string& outValue) {
  if (options.hasProperty(rt, key.c_str())) {
    jsi::Value val = options.getProperty(rt, key.c_str());
    if (val.isString()) {
      outValue = val.asString(rt).utf8(rt);
      return true;
    }
  }
  return false;
}

// For bool
bool SystemUtils::setIfExists(jsi::Runtime& rt, const jsi::Object& options, const std::string& key, bool& outValue) {
  if (options.hasProperty(rt, key.c_str())) {
    jsi::Value val = options.getProperty(rt, key.c_str());
    if (val.isBool()) {
      outValue = val.getBool();
      return true;
    }
  }
  return false;
}

// For std::vector<jsi::Value> (Array)
bool SystemUtils::setIfExists(jsi::Runtime& rt, const jsi::Object& options, const std::string& key, std::vector<jsi::Value>& outValue) {
  if (options.hasProperty(rt, key.c_str())) {
    jsi::Value val = options.getProperty(rt, key.c_str());
    if (val.isObject()) {
      jsi::Object obj = val.asObject(rt);
      if (obj.isArray(rt)) {
        jsi::Array arr = obj.asArray(rt);
        size_t length = arr.size(rt);
        outValue.clear();
        for (size_t i = 0; i < length; ++i) {
          outValue.push_back(arr.getValueAtIndex(rt, i));
        }
        return true;
      }
    }
  }
  return false;
}

} // namespace facebook::react
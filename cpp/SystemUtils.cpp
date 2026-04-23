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
        total_memory = 2LL * 1024 * 1024 * 1024;     // 2GB default for iOS
#elif defined(__ANDROID__)
        total_memory = 3LL * 1024 * 1024 * 1024;     // 3GB default for Android
#else
        total_memory = 2LL * 1024 * 1024 * 1024;     // 2GB default for other platforms
#endif
    }

    return total_memory;
}

int64_t SystemUtils::getAvailableMemoryBytes() {
#if defined(__APPLE__) && TARGET_OS_IPHONE
    mach_port_t host = mach_host_self();
    vm_size_t page_size = 0;
    host_page_size(host, &page_size);
    vm_statistics64_data_t vm_stats{};
    mach_msg_type_number_t info_count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(host, HOST_VM_INFO64,
                          reinterpret_cast<host_info64_t>(&vm_stats),
                          &info_count) == KERN_SUCCESS) {
        return static_cast<int64_t>(vm_stats.free_count) * page_size;
    }
    return 512LL * 1024 * 1024; // 512 MB fallback
#elif defined(__ANDROID__)
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.compare(0, 13, "MemAvailable:") == 0) {
            int64_t kb = 0;
            std::istringstream iss(line.substr(13));
            iss >> kb;
            return kb * 1024;
        }
    }
    return 512LL * 1024 * 1024; // 512 MB fallback
#else
    return 512LL * 1024 * 1024;
#endif
}

int SystemUtils::getOptimalGpuLayers(struct llama_model* model,
                                      int64_t reserved_vram_bytes) {
    // MS-P4 FIX: guard against null model pointer before any dereference.
    if (!model) {
        return 0;
    }
    const int n_layer = llama_model_n_layer(model);
    int64_t bytes_per_layer = (int64_t)llama_model_size(model) / n_layer;
    int64_t total_memory = getTotalPhysicalMemory();

    int64_t available_vram = 0;
#if (defined(__APPLE__) && TARGET_OS_IPHONE) || defined(__ANDROID__)
    // Unified mobile budget: 15% of total RAM.
    // 25%/20% was too aggressive — all layers on GPU causes GPU fence timeouts
    // (Android mLastRetireFence) and memory pressure on smaller iOS devices.
    available_vram = total_memory * 15 / 100;
#endif

    // Use 80% of the available budget, then deduct VRAM reserved for other models (e.g. mmproj).
    int64_t target_vram = (available_vram * 80) / 100;
    target_vram = std::max(int64_t(0), target_vram - reserved_vram_bytes);

    int possible_layers = (bytes_per_layer > 0) ? static_cast<int>(target_vram / bytes_per_layer) : 0;

    // Cap at 75% of total layers so at least 25% run on CPU, creating GPU yield points.
    // This prevents mobile GPU fence timeouts (Android) and thermal saturation (iOS).
    int max_layers = n_layer * 3 / 4;
    int optimal_layers = std::max(1, std::min(possible_layers, max_layers));

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
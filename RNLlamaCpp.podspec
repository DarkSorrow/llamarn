require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

Pod::Spec.new do |s|
  s.name         = "RNLlamaCpp"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = package["homepage"]
  s.license      = package["license"]
  s.authors      = package["author"]

  s.platforms    = { :ios => "13.0" }
  s.source       = { :git => "https://github.com/novastera/llamacpp-rn.git", :tag => "#{s.version}" }

  # Core React Native module files - keep iOS-specific files separate
  s.source_files = "ios/**/*.{h,m,mm}",  # iOS-specific Obj-C++ files
                   "ios/generated/*.{h,cpp,mm}",
                   # Core C++ module implementation (keep as .cpp)
                   # THIS SECTION WILL BE ADJUSTED IN THE NEXT STEP IF NEEDED BASED ON ACTUAL FILE LOCATIONS
                   "cpp/build-info.cpp", # Assuming build-info.cpp is in cpp/
                   "cpp/PureCppImpl.{h,cpp}", # Changed from LlamaCppRnModule to PureCppImpl, assuming this is current
                   "cpp/LlamaCppModel.{h,cpp}",
                   "cpp/SystemUtils.{h,cpp}", # Assuming SystemUtils is in cpp/
                   "cpp/rn-*.{hpp,cpp}", # Assuming rn-*.hpp/cpp are in cpp/
                   # llama.cpp common utilities - Assuming these are vendored or part of the XCFramework
                   # If these are compiled directly, their paths need to be relative to the podspec (e.g., "cpp/llama.cpp/common/common.{h,cpp}")
                   "cpp/llama.cpp/common/common.{h,cpp}",
                   "cpp/llama.cpp/common/log.{h,cpp}",
                   "cpp/llama.cpp/common/arg.{h,cpp}",
                   "cpp/llama.cpp/common/sampling.{h,cpp}",
                   "cpp/llama.cpp/common/chat.{h,cpp}",
                   "cpp/llama.cpp/common/chat-parser.{h,cpp}",
                   "cpp/llama.cpp/common/regex-partial.{h,cpp}",
                   "cpp/llama.cpp/common/console.{h,cpp}",
                   "cpp/llama.cpp/common/json-partial.{h,cpp}",
                   "cpp/llama.cpp/common/ngram-cache.{h,cpp}",
                   "cpp/llama.cpp/common/json-schema-to-grammar.{h,cpp}",
                   "cpp/llama.cpp/common/speculative.{h,cpp}",
                   "cpp/llama.cpp/common/llguidance.{h,cpp}",
                   "cpp/llama.cpp/common/*.hpp",
                   "cpp/llama.cpp/vendor/minja/*.hpp"
                   "cpp/llama.cpp/vendor/nlohmann/*.hpp"
  
  # Include all necessary headers for compilation
  s.preserve_paths = "ios/include/**/*.h",
                     "ios/libs/**/*", 
                     "cpp/**/*"
  
  # Use the prebuilt framework
  s.vendored_frameworks = "ios/libs/llama.xcframework"

  # Compiler settings
  s.pod_target_xcconfig = {
    "HEADER_SEARCH_PATHS" => "\"$(PODS_TARGET_SRCROOT)/ios/include\" \"$(PODS_TARGET_SRCROOT)/cpp\" \"$(PODS_TARGET_SRCROOT)/ios/generated/RNLlamaCppSpec\" \"$(PODS_TARGET_SRCROOT)/ios/generated\" \"$(PODS_TARGET_SRCROOT)/cpp/llama.cpp\" \"$(PODS_TARGET_SRCROOT)/cpp/llama.cpp/include\" \"$(PODS_TARGET_SRCROOT)/cpp/llama.cpp/ggml/include\" \"$(PODS_TARGET_SRCROOT)/cpp/llama.cpp/common\" \"$(PODS_TARGET_SRCROOT)/cpp/llama.cpp/vendor\" \"$(PODS_ROOT)/boost\" \"$(PODS_ROOT)/Headers/Public/React-bridging\" \"$(PODS_ROOT)/Headers/Public/React\"",
    "OTHER_CPLUSPLUSFLAGS" => "-DFOLLY_NO_CONFIG -DFOLLY_MOBILE=1 -DFOLLY_USE_LIBCPP=1 -DLLAMA_METAL -DRCT_NEW_ARCH_ENABLED=1 -DFBJSRT_EXPORTED=1",
    "CLANG_CXX_LANGUAGE_STANDARD" => "c++17",
    "GCC_OPTIMIZATION_LEVEL" => "3", # Maximum optimization
    "SWIFT_OPTIMIZATION_LEVEL" => "-O",
    "ENABLE_BITCODE" => "NO",
    "DEFINES_MODULE" => "YES",
    "OTHER_LDFLAGS" => "$(inherited) -framework Accelerate -framework Foundation -framework Metal -framework MetalKit",
    # These preprocessor macros ensure TurboModule registration works correctly
    "GCC_PREPROCESSOR_DEFINITIONS" => ["$(inherited)", "RCT_NEW_ARCH_ENABLED=1", 
                                       "__STDC_FORMAT_MACROS=1", # For format macros in C++
                                       "LLAMA_SHARED=1"]         # For llama shared symbols
  }

  # Add user_target_xcconfig to propagate linker flags and fix framework issues
  s.user_target_xcconfig = {
    "OTHER_LDFLAGS" => "$(inherited) -framework Accelerate -framework Foundation -framework Metal -framework MetalKit",
    "FRAMEWORK_SEARCH_PATHS" => "$(inherited) $(PLATFORM_DIR)/Developer/Library/Frameworks"
  }

  # Install dependencies for Turbo Modules
  install_modules_dependencies(s)
end

cmake_minimum_required(VERSION 3.4.1)
project(RNLlamaCpp)

set(CMAKE_VERBOSE_MAKEFILE ON)

# Get the absolute path to the module root and cpp directory
get_filename_component(MODULE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
get_filename_component(CPP_DIR "${MODULE_ROOT}/cpp" ABSOLUTE)

# Define the path to jniLibs. This assumes CMakeLists.txt is in android/
set(JNI_LIBS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/main/jniLibs)

# Define the path to llama.cpp directory
set(LLAMA_CPP_DIR "${CPP_DIR}/llama.cpp")

# Make sure the llama.cpp submodule exists
if(NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "llama.cpp submodule not found at ${LLAMA_CPP_DIR}. Please run 'git submodule update --init --recursive'")
endif()

# Add the prebuilt libraries as IMPORTED (64-bit architectures only: arm64-v8a, x86_64)
add_library(llama SHARED IMPORTED)
set_target_properties(llama PROPERTIES IMPORTED_LOCATION
    ${JNI_LIBS_DIR}/${ANDROID_ABI}/libllama.so)

add_library(ggml-base SHARED IMPORTED)
set_target_properties(ggml-base PROPERTIES IMPORTED_LOCATION
    ${JNI_LIBS_DIR}/${ANDROID_ABI}/libggml-base.so)

add_library(ggml SHARED IMPORTED)
set_target_properties(ggml PROPERTIES IMPORTED_LOCATION
    ${JNI_LIBS_DIR}/${ANDROID_ABI}/libggml.so)

add_library(ggml-cpu SHARED IMPORTED)
set_target_properties(ggml-cpu PROPERTIES IMPORTED_LOCATION
    ${JNI_LIBS_DIR}/${ANDROID_ABI}/libggml-cpu.so)

# Create a minimal common library with only essential files that don't require missing GGML symbols
add_library(
    common
    STATIC
    # Add back essential files now that we have prebuilt GGML libraries
    ${CPP_DIR}/llama.cpp/common/build-info.cpp
    ${CPP_DIR}/llama.cpp/common/log.cpp
    ${CPP_DIR}/llama.cpp/common/common.cpp
    ${CPP_DIR}/llama.cpp/common/sampling.cpp
    ${CPP_DIR}/llama.cpp/common/chat.cpp
    ${CPP_DIR}/llama.cpp/common/json-schema-to-grammar.cpp
)

add_library(
    RNLlamaCpp
    SHARED
    ${CPP_DIR}/build-info.cpp
    ${CPP_DIR}/PureCppImpl.cpp
    ${CPP_DIR}/LlamaCppModel.cpp
    ${CPP_DIR}/SystemUtils.cpp
    ${CPP_DIR}/rn-completion.cpp
)

# Suppress unused function warnings for llama.cpp code
target_compile_options(common PRIVATE -Wno-unused-function)
target_compile_options(RNLlamaCpp PRIVATE -Wno-unused-function)

# Include directories
target_include_directories(common PRIVATE
    ${CPP_DIR}
    ${LLAMA_CPP_DIR}/ggml/include
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/common
    ${LLAMA_CPP_DIR}/common/minja  # Add this for chat-template.hpp
    ${LLAMA_CPP_DIR}/src
)

target_include_directories(RNLlamaCpp PRIVATE
    ${CPP_DIR}
    ${LLAMA_CPP_DIR}/ggml/include
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/common
    ${LLAMA_CPP_DIR}/common/minja  # Add this for chat-template.hpp
    ${LLAMA_CPP_DIR}/src
    # Add the generated headers path
    ${MODULE_ROOT}/android/generated/jni
    ${MODULE_ROOT}/android/generated/jni/react/renderer/components/RNLlamaCppSpec
)

# Link libraries
target_link_libraries(
    RNLlamaCpp
    common
    react_codegen_RNLlamaCppSpec  # Link against the generated TurboModule code
    llama       # Link against the imported prebuilt core llama library (now named llama)
    ggml-base   # Link against the imported GGML base library (contains ggml_abort, ggml_time_us, gguf_*)
    ggml        # Link against the imported GGML library (contains ggml_backend_load_all)
    ggml-cpu    # Link against the imported GGML CPU library
    jsi
    reactnative
    fbjni
    android
    log
)

# Expose our headers to consuming targets (for autolinking)
target_include_directories(RNLlamaCpp INTERFACE
    ${CPP_DIR}
    ${LLAMA_CPP_DIR}/ggml/include
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/common
    ${LLAMA_CPP_DIR}/common/minja
    ${LLAMA_CPP_DIR}/src
)

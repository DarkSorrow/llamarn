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

# Find Vulkan (available in NDK 23+)
find_package(Vulkan QUIET)
if(Vulkan_FOUND)
    message(STATUS "Found Vulkan: ${Vulkan_LIBRARIES}")
else()
    message(STATUS "Vulkan not found - GPU acceleration will be limited")
endif()

# Add the prebuilt libraries as IMPORTED with IMPORTED_NO_SONAME to avoid absolute path embedding
add_library(llama SHARED IMPORTED)
set_target_properties(llama PROPERTIES 
    IMPORTED_LOCATION ${JNI_LIBS_DIR}/${ANDROID_ABI}/libllama.so
    IMPORTED_NO_SONAME TRUE)

add_library(ggml-base SHARED IMPORTED)
set_target_properties(ggml-base PROPERTIES 
    IMPORTED_LOCATION ${JNI_LIBS_DIR}/${ANDROID_ABI}/libggml-base.so
    IMPORTED_NO_SONAME TRUE)

add_library(ggml SHARED IMPORTED)
set_target_properties(ggml PROPERTIES 
    IMPORTED_LOCATION ${JNI_LIBS_DIR}/${ANDROID_ABI}/libggml.so
    IMPORTED_NO_SONAME TRUE)

add_library(ggml-cpu SHARED IMPORTED)
set_target_properties(ggml-cpu PROPERTIES 
    IMPORTED_LOCATION ${JNI_LIBS_DIR}/${ANDROID_ABI}/libggml-cpu.so
    IMPORTED_NO_SONAME TRUE)

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
    ${CPP_DIR}/llama.cpp/common/chat-parser.cpp
    ${CPP_DIR}/llama.cpp/common/regex-partial.cpp
    ${CPP_DIR}/llama.cpp/common/arg.cpp
    ${CPP_DIR}/llama.cpp/common/console.cpp
    ${CPP_DIR}/llama.cpp/common/json-partial.cpp
    ${CPP_DIR}/llama.cpp/common/ngram-cache.cpp
    ${CPP_DIR}/llama.cpp/common/json-schema-to-grammar.cpp
    ${CPP_DIR}/llama.cpp/common/speculative.cpp
    ${CPP_DIR}/llama.cpp/common/llguidance.cpp
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

# Suppress additional warnings that are treated as errors in Expo SDK 54
target_compile_options(common PRIVATE )

# Use React Native's compile options function for proper C++ flags and RN_SERIALIZABLE_STATE
if(ReactAndroid_VERSION_MINOR GREATER_EQUAL 80)
    # Add additional warning suppressions for RNLlamaCpp target
    target_compile_reactnative_options(RNLlamaCpp PRIVATE)
    target_compile_options(RNLlamaCpp PRIVATE 
        -Wno-unused-function
        -Wno-unused-variable
        -Wno-unused-const-variable
        -Wno-unused-but-set-variable
    )
else()
    target_compile_options(RNLlamaCpp PRIVATE -Wno-unused-function)
endif()

# Check if Vulkan backend library is available
set(VULKAN_BACKEND_AVAILABLE FALSE)
if(EXISTS ${JNI_LIBS_DIR}/${ANDROID_ABI}/libggml-vulkan.so)
    set(VULKAN_BACKEND_AVAILABLE TRUE)
    message(STATUS "Vulkan backend library found for ${ANDROID_ABI}")
else()
    message(STATUS "Vulkan backend library not found for ${ANDROID_ABI}")
endif()

# Check if OpenCL backend library is available
set(OPENCL_BACKEND_AVAILABLE FALSE)
if(EXISTS ${JNI_LIBS_DIR}/${ANDROID_ABI}/libggml-opencl.so AND EXISTS ${JNI_LIBS_DIR}/${ANDROID_ABI}/libOpenCL.so)
    set(OPENCL_BACKEND_AVAILABLE TRUE)
    message(STATUS "OpenCL backend libraries found for ${ANDROID_ABI}")
else()
    message(STATUS "OpenCL backend libraries not found for ${ANDROID_ABI}")
endif()

# Hybrid backend approach: CPU static (built into main libraries), GPU dynamic
# CPU backend will be statically linked into main libraries (libggml.so, libllama.so)
# GPU backends (OpenCL, Vulkan) will be dynamically loaded at runtime only if available
target_compile_definitions(common PRIVATE 
    -DGGML_BACKEND_DL=1  # Enable dynamic loading for GPU backends
    -DGGML_CPU=1         # CPU backend statically built into main libraries
)
target_compile_definitions(RNLlamaCpp PRIVATE 
    -DGGML_BACKEND_DL=1  # Enable dynamic loading for GPU backends
    -DGGML_CPU=1         # CPU backend statically built into main libraries
)

# DISABLE Vulkan on Android - causes crashes during auto-initialization on emulators
# Even with n_gpu_layers=0, llama.cpp tries to initialize all available backends
message(STATUS "Vulkan backend support DISABLED on Android to prevent emulator crashes")

# TODO: Enable Vulkan backend if available (currently disabled due to emulator crashes)
# Uncomment the lines below to test Vulkan support on real devices
# if(VULKAN_BACKEND_AVAILABLE)
#     target_compile_definitions(common PRIVATE -DGGML_VULKAN=1)
#     target_compile_definitions(RNLlamaCpp PRIVATE -DGGML_VULKAN=1)
#     message(STATUS "Vulkan backend support enabled for dynamic loading")
# else()
#     message(STATUS "Vulkan backend support disabled - library not available")
# endif()

# Enable OpenCL backend if available
if(OPENCL_BACKEND_AVAILABLE)
    target_compile_definitions(common PRIVATE -DGGML_OPENCL=1)
    target_compile_definitions(RNLlamaCpp PRIVATE -DGGML_OPENCL=1)
    message(STATUS "OpenCL backend support enabled for dynamic loading")
else()
    message(STATUS "OpenCL backend support disabled - library not available")
endif()

# Include directories
target_include_directories(common PRIVATE
    ${CPP_DIR}
    ${LLAMA_CPP_DIR}/ggml/include
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/common
    ${LLAMA_CPP_DIR}/vendor/minja
    ${LLAMA_CPP_DIR}/vendor
    ${LLAMA_CPP_DIR}/src
)

target_include_directories(RNLlamaCpp PRIVATE
    ${CPP_DIR}
    ${LLAMA_CPP_DIR}/ggml/include
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/common
    ${LLAMA_CPP_DIR}/vendor/minja  # Add this for chat-template.hpp
    ${LLAMA_CPP_DIR}/vendor
    ${LLAMA_CPP_DIR}/src
    # Add the generated headers path
    ${MODULE_ROOT}/android/generated/jni
    ${MODULE_ROOT}/android/generated/jni/react/renderer/components/RNLlamaCppSpec
)

# Link libraries with proper dependencies
target_link_libraries(
    RNLlamaCpp
    common
    react_codegen_RNLlamaCppSpec  # Link against the generated TurboModule code
    llama       # Link against the imported prebuilt core llama library
    ggml-base   # Link against the imported GGML base library
    ggml        # Link against the imported GGML library
    ggml-cpu    # Link against the imported GGML CPU library
    jsi
    reactnative
    fbjni
    android
    log
    dl          # Required for dynamic loading of backend libraries
)

# Add Vulkan support if available
if(Vulkan_FOUND)
    target_link_libraries(RNLlamaCpp ${Vulkan_LIBRARIES})
    target_include_directories(RNLlamaCpp PRIVATE ${Vulkan_INCLUDE_DIRS})
    message(STATUS "Vulkan support enabled for dynamic GPU backend loading")
else()
    # Even without system Vulkan, we can still support dynamic loading if Vulkan library is present at runtime
    message(STATUS "System Vulkan not found, but dynamic Vulkan loading may still work at runtime")
endif()

# Add OpenCL support - OpenCL will be loaded dynamically at runtime
# No need to link against OpenCL here since we use dynamic loading

# Copy dependency libraries to build output directory so they get packaged into APK
add_custom_command(TARGET RNLlamaCpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${JNI_LIBS_DIR}/${ANDROID_ABI}/libllama.so
        $<TARGET_FILE_DIR:RNLlamaCpp>/libllama.so
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${JNI_LIBS_DIR}/${ANDROID_ABI}/libggml-base.so
        $<TARGET_FILE_DIR:RNLlamaCpp>/libggml-base.so
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${JNI_LIBS_DIR}/${ANDROID_ABI}/libggml.so
        $<TARGET_FILE_DIR:RNLlamaCpp>/libggml.so
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${JNI_LIBS_DIR}/${ANDROID_ABI}/libggml-cpu.so
        $<TARGET_FILE_DIR:RNLlamaCpp>/libggml-cpu.so
    COMMENT "Copying dependency libraries to build output directory"
)

# Also copy any optional GPU libraries if they exist
if(EXISTS ${JNI_LIBS_DIR}/${ANDROID_ABI}/libggml-vulkan.so)
    # Don't copy Vulkan backend on Android - it crashes on emulators during auto-initialization
    # Even with n_gpu_layers=0, llama.cpp tries to initialize all available backends
    # and the Android emulator Vulkan driver is broken
    message(STATUS "Skipping Vulkan backend copy to prevent emulator crashes")
    
    # TODO: Uncomment the lines below to enable Vulkan library copying for testing on real devices
    # add_custom_command(TARGET RNLlamaCpp POST_BUILD
    #     COMMAND ${CMAKE_COMMAND} -E copy_if_different
    #         ${JNI_LIBS_DIR}/${ANDROID_ABI}/libggml-vulkan.so
    #         $<TARGET_FILE_DIR:RNLlamaCpp>/libggml-vulkan.so
    #     COMMENT "Copying Vulkan backend library to build output directory"
    # )
endif()

if(EXISTS ${JNI_LIBS_DIR}/${ANDROID_ABI}/libggml-opencl.so)
    add_custom_command(TARGET RNLlamaCpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${JNI_LIBS_DIR}/${ANDROID_ABI}/libggml-opencl.so
            $<TARGET_FILE_DIR:RNLlamaCpp>/libggml-opencl.so
        COMMENT "Copying OpenCL library to build output directory"
    )
endif()

if(EXISTS ${JNI_LIBS_DIR}/${ANDROID_ABI}/libOpenCL.so)
    add_custom_command(TARGET RNLlamaCpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${JNI_LIBS_DIR}/${ANDROID_ABI}/libOpenCL.so
            $<TARGET_FILE_DIR:RNLlamaCpp>/libOpenCL.so
        COMMENT "Copying OpenCL loader library to build output directory"
    )
endif()

# Expose our headers to consuming targets (for autolinking)
target_include_directories(RNLlamaCpp INTERFACE
    ${CPP_DIR}
    ${LLAMA_CPP_DIR}/ggml/include
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/common
    ${LLAMA_CPP_DIR}/vendor/minja
    ${LLAMA_CPP_DIR}/vendor
    ${LLAMA_CPP_DIR}/src
)

cmake_minimum_required(VERSION 3.4.1)
project(RNLlamaCpp)

set(CMAKE_VERBOSE_MAKEFILE ON)

# Get the absolute path to the module root and cpp directory
get_filename_component(MODULE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
get_filename_component(CPP_DIR "${MODULE_ROOT}/cpp" ABSOLUTE)

# Define the path to jniLibs - this is where build_android_external.sh puts the libraries
set(JNI_LIBS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/main/jniLibs)

# Define the path to llama.cpp directory (for headers)
set(LLAMA_CPP_DIR "${CPP_DIR}/llama.cpp")

# Make sure the llama.cpp submodule exists
if(NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "llama.cpp submodule not found at ${LLAMA_CPP_DIR}. Please run 'git submodule update --init --recursive'")
endif()

# Check if libraries exist (they should be built by build_android_external.sh)
if(NOT EXISTS "${JNI_LIBS_DIR}/${ANDROID_ABI}/libllama.so")
    message(FATAL_ERROR "Prebuilt libraries not found. Please run 'scripts/build_android_external.sh' first to build the native libraries.")
endif()

# Import all prebuilt libraries as IMPORTED targets
add_library(llama SHARED IMPORTED)
set_target_properties(llama PROPERTIES 
    IMPORTED_LOCATION ${JNI_LIBS_DIR}/${ANDROID_ABI}/libllama.so
    IMPORTED_NO_SONAME TRUE)

add_library(ggml SHARED IMPORTED)
set_target_properties(ggml PROPERTIES 
    IMPORTED_LOCATION ${JNI_LIBS_DIR}/${ANDROID_ABI}/libggml.so
    IMPORTED_NO_SONAME TRUE)

add_library(ggml-base SHARED IMPORTED)
set_target_properties(ggml-base PROPERTIES 
    IMPORTED_LOCATION ${JNI_LIBS_DIR}/${ANDROID_ABI}/libggml-base.so
    IMPORTED_NO_SONAME TRUE)

add_library(ggml-cpu SHARED IMPORTED)
set_target_properties(ggml-cpu PROPERTIES 
    IMPORTED_LOCATION ${JNI_LIBS_DIR}/${ANDROID_ABI}/libggml-cpu.so
    IMPORTED_NO_SONAME TRUE)

# Collect additional libraries that exist
set(ADDITIONAL_LIBRARIES "")

# Check for OpenCL backend
if(EXISTS ${JNI_LIBS_DIR}/${ANDROID_ABI}/libggml-opencl.so)
    add_library(ggml-opencl SHARED IMPORTED)
    set_target_properties(ggml-opencl PROPERTIES 
        IMPORTED_LOCATION ${JNI_LIBS_DIR}/${ANDROID_ABI}/libggml-opencl.so
        IMPORTED_NO_SONAME TRUE)
    list(APPEND ADDITIONAL_LIBRARIES ggml-opencl)
    message(STATUS "Found OpenCL backend for ${ANDROID_ABI}")
endif()

# Check for OpenCL ICD loader
if(EXISTS ${JNI_LIBS_DIR}/${ANDROID_ABI}/libOpenCL.so)
    add_library(OpenCL SHARED IMPORTED)
    set_target_properties(OpenCL PROPERTIES 
        IMPORTED_LOCATION ${JNI_LIBS_DIR}/${ANDROID_ABI}/libOpenCL.so
        IMPORTED_NO_SONAME TRUE)
    list(APPEND ADDITIONAL_LIBRARIES OpenCL)
    message(STATUS "Found OpenCL ICD loader for ${ANDROID_ABI}")
endif()

# Check for Vulkan backend
if(EXISTS ${JNI_LIBS_DIR}/${ANDROID_ABI}/libggml-vulkan.so)
    add_library(ggml-vulkan SHARED IMPORTED)
    set_target_properties(ggml-vulkan PROPERTIES 
        IMPORTED_LOCATION ${JNI_LIBS_DIR}/${ANDROID_ABI}/libggml-vulkan.so
        IMPORTED_NO_SONAME TRUE)
    list(APPEND ADDITIONAL_LIBRARIES ggml-vulkan)
    message(STATUS "Found Vulkan backend for ${ANDROID_ABI}")
endif()

# Create common library with essential llama.cpp common files
add_library(
    common
    STATIC
    ${CPP_DIR}/llama.cpp/common/build-info.cpp
    ${CPP_DIR}/llama.cpp/common/log.cpp
    ${CPP_DIR}/llama.cpp/common/common.cpp
    ${CPP_DIR}/llama.cpp/common/sampling.cpp
    ${CPP_DIR}/llama.cpp/common/chat.cpp
    ${CPP_DIR}/llama.cpp/common/json-schema-to-grammar.cpp
)

# Create our main React Native module
add_library(
    RNLlamaCpp
    SHARED
    ${CPP_DIR}/build-info.cpp
    ${CPP_DIR}/PureCppImpl.cpp
    ${CPP_DIR}/LlamaCppModel.cpp
    ${CPP_DIR}/SystemUtils.cpp
    ${CPP_DIR}/rn-completion.cpp
)

# Suppress unused function warnings
target_compile_options(common PRIVATE -Wno-unused-function)
target_compile_options(RNLlamaCpp PRIVATE -Wno-unused-function)

# Set compile definitions - always enable dynamic loading and CPU backend
target_compile_definitions(common PRIVATE 
    -DGGML_BACKEND_DL=1
    -DGGML_CPU=1
)
target_compile_definitions(RNLlamaCpp PRIVATE 
    -DGGML_BACKEND_DL=1
    -DGGML_CPU=1
)

# Add GPU backend definitions based on what libraries exist
if(EXISTS ${JNI_LIBS_DIR}/${ANDROID_ABI}/libggml-opencl.so)
    target_compile_definitions(common PRIVATE -DGGML_OPENCL=1)
    target_compile_definitions(RNLlamaCpp PRIVATE -DGGML_OPENCL=1)
    message(STATUS "Enabling OpenCL support")
endif()

if(EXISTS ${JNI_LIBS_DIR}/${ANDROID_ABI}/libggml-vulkan.so)
    target_compile_definitions(common PRIVATE -DGGML_VULKAN=1)
    target_compile_definitions(RNLlamaCpp PRIVATE -DGGML_VULKAN=1)
    message(STATUS "Enabling Vulkan support")
endif()

# Include directories
target_include_directories(common PRIVATE
    ${CPP_DIR}
    ${LLAMA_CPP_DIR}/ggml/include
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/common
    ${LLAMA_CPP_DIR}/common/minja
    ${LLAMA_CPP_DIR}/src
)

target_include_directories(RNLlamaCpp PRIVATE
    ${CPP_DIR}
    ${LLAMA_CPP_DIR}/ggml/include
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/common
    ${LLAMA_CPP_DIR}/common/minja
    ${LLAMA_CPP_DIR}/src
    # Add the generated headers path
    ${MODULE_ROOT}/android/generated/jni
    ${MODULE_ROOT}/android/generated/jni/react/renderer/components/RNLlamaCppSpec
)

# Link libraries - link against all available libraries
target_link_libraries(
    RNLlamaCpp
    common
    react_codegen_RNLlamaCppSpec
    llama
    ggml-base
    ggml
    ggml-cpu
    ${ADDITIONAL_LIBRARIES}
    jsi
    reactnative
    fbjni
    android
    log
    dl
)

# Copy all libraries to build output directory
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
    COMMENT "Copying main libraries"
)

# Copy any additional GPU libraries that exist
if(EXISTS ${JNI_LIBS_DIR}/${ANDROID_ABI}/libggml-opencl.so)
    add_custom_command(TARGET RNLlamaCpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${JNI_LIBS_DIR}/${ANDROID_ABI}/libggml-opencl.so
            $<TARGET_FILE_DIR:RNLlamaCpp>/libggml-opencl.so
        COMMENT "Copying OpenCL backend"
    )
endif()

if(EXISTS ${JNI_LIBS_DIR}/${ANDROID_ABI}/libOpenCL.so)
    add_custom_command(TARGET RNLlamaCpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${JNI_LIBS_DIR}/${ANDROID_ABI}/libOpenCL.so
            $<TARGET_FILE_DIR:RNLlamaCpp>/libOpenCL.so
        COMMENT "Copying OpenCL ICD loader"
    )
endif()

if(EXISTS ${JNI_LIBS_DIR}/${ANDROID_ABI}/libggml-vulkan.so)
    add_custom_command(TARGET RNLlamaCpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${JNI_LIBS_DIR}/${ANDROID_ABI}/libggml-vulkan.so
            $<TARGET_FILE_DIR:RNLlamaCpp>/libggml-vulkan.so
        COMMENT "Copying Vulkan backend"
    )
endif()

# Expose headers to consuming targets
target_include_directories(RNLlamaCpp INTERFACE
    ${CPP_DIR}
    ${LLAMA_CPP_DIR}/ggml/include
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/common
    ${LLAMA_CPP_DIR}/common/minja
    ${LLAMA_CPP_DIR}/src
)


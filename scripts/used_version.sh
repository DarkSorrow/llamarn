#!/bin/bash
# Central location for all build tool version information
# This file should be sourced by other build scripts

# The specific llama.cpp commit hash we want to use
# Using a specific commit hash ensures a consistent build
LLAMA_CPP_COMMIT="3d26a09dc7b1a7c13da57fdd26d1cf22efa81229"  # Commit as specified by user

# The tag to use for prebuilt binaries
# This might differ from the commit hash format
LLAMA_CPP_TAG="b7642"  # Tag format for binary downloads

# Vulkan and OpenCL versions
# Note: Vulkan loader ships with the NDK, but ggml-vulkan needs the C++ headers (vulkan.hpp)
VULKAN_SDK_VERSION="1.4.309"  # Reference only (Android uses NDK loader)
VULKAN_HEADERS_TAG="v1.3.292"  # Same version CI installs inside the NDK sysroot
OPENCL_VERSION="3.0"
# OpenCL Headers: Use stable tag compatible with OpenCL 3.0
# v2025.07.22 aligns with OpenCL 3.0.19 spec (includes CL_ENABLE_BETA_EXTENSIONS)
# This is the version used in llama.cpp CI builds
# Check: https://github.com/KhronosGroup/OpenCL-Headers/releases
OPENCL_HEADERS_TAG="v2025.07.22"  # Matches llama.cpp CI, supports OpenCL 3.0.19

# Android SDK/NDK configuration
# Using NDK r28c (stable) to match llama.cpp Hexagon backend requirements
# r28c is the stable release of r28, ensuring compatibility with Hexagon SDK
NDK_VERSION="28.2.13676358"  # NDK r28c (stable release, compatible with Hexagon)
ANDROID_MIN_SDK="33"
ANDROID_TARGET_SDK="35"
ANDROID_PLATFORM="android-$ANDROID_MIN_SDK"

# Export all variables
export VULKAN_SDK_VERSION
export VULKAN_HEADERS_TAG
export OPENCL_VERSION
export OPENCL_HEADERS_TAG
export NDK_VERSION
export ANDROID_MIN_SDK
export ANDROID_TARGET_SDK
export ANDROID_PLATFORM
export LLAMA_CPP_COMMIT
export LLAMA_CPP_TAG

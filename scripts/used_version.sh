#!/bin/bash
# Central location for all build tool version information
# This file should be sourced by other build scripts

# The specific llama.cpp commit hash we want to use
# Using a specific commit hash ensures a consistent build
LLAMA_CPP_COMMIT="96ac5a2329029dfc35c9cbbb24c09fd91ae9416b"  # Commit as specified by user

# The tag to use for prebuilt binaries
# This might differ from the commit hash format
LLAMA_CPP_TAG="b7134"  # Tag format for binary downloads

# Vulkan and OpenCL versions
# Note: Vulkan is provided by Android NDK, we don't need a separate SDK version
VULKAN_SDK_VERSION="1.4.309"  # Not used for Android (NDK provides Vulkan)
OPENCL_VERSION="3.0"
# OpenCL Headers: Use latest stable tag (llama.cpp uses latest from git)
# Check: https://github.com/KhronosGroup/OpenCL-Headers/releases
OPENCL_HEADERS_TAG="v2025.07.22"  # Updated to match llama.cpp CI (2025.07.22)

# Android SDK/NDK configuration
NDK_VERSION="27.2.12479018"  # Original CI version
ANDROID_MIN_SDK="33"
ANDROID_TARGET_SDK="35"
ANDROID_PLATFORM="android-$ANDROID_MIN_SDK"

# Export all variables
export VULKAN_SDK_VERSION
export OPENCL_VERSION
export OPENCL_HEADERS_TAG
export NDK_VERSION
export ANDROID_MIN_SDK
export ANDROID_TARGET_SDK
export ANDROID_PLATFORM
export LLAMA_CPP_COMMIT
export LLAMA_CPP_TAG

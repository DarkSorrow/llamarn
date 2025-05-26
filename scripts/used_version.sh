#!/bin/bash
# Central location for all build tool version information
# This file should be sourced by other build scripts

# The specific llama.cpp commit hash we want to use
# Using a specific commit hash ensures a consistent build
LLAMA_CPP_COMMIT="9a390c4829cd3058d26a2e2c09d16e3fd12bf1b1"  # Commit as specified by user

# The tag to use for prebuilt binaries
# This might differ from the commit hash format
LLAMA_CPP_TAG="b5349"  # Tag format for binary downloads

# Vulkan and OpenCL versions
VULKAN_SDK_VERSION="1.4.309"
OPENCL_VERSION="3.0"
OPENCL_HEADERS_TAG="v2023.12.14"

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

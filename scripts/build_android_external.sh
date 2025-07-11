#!/bin/bash
set -e

# Suppress getenv warnings on newer Linux distributions
export CFLAGS="-Wno-gnu-get-env"

# Get the absolute path of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get project root directory (one level up from script dir)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Source the version information
. "$SCRIPT_DIR/used_version.sh"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# CI Usage Note
# This script is used in the CI workflow to build the final JNI libraries.
# It automatically detects and uses GPU libraries in prebuilt/gpu/ directory 
# if they exist, which should be created by build_android_gpu_backend.sh first.
#
# Print usage information
print_usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --help                 Print this help message"
  echo "  --abi=[all|arm64-v8a|x86_64|armeabi-v7a|x86]  Specify which ABI to build for (default: all)"
  echo "  --no-opencl            Disable OpenCL GPU acceleration"
  echo "  --no-vulkan            Disable Vulkan GPU acceleration"
  echo "  --vulkan               Enable Vulkan GPU acceleration (default)"
  echo "  --debug                Build in debug mode"
  echo "  --clean                Clean previous builds before building"
  echo "  --clean-prebuilt       Clean entire prebuilt directory for a fresh start"
  echo "  --install-deps         Install dependencies (OpenCL, etc.)"
  echo "  --glslc-path=[path]    Specify a custom path to the GLSLC compiler"
  echo "  --ndk-path=[path]      Specify a custom path to the Android NDK"
  echo "  --no-use-prebuilt-gpu  Disable use of prebuilt GPU libraries"
  echo "  --platform=[android|all]  Specify target platform (default: all)"
  echo "  --cmake-flags=[flags]  Additional CMake flags to pass to the build"
}

# Default values
BUILD_ABI="all"
BUILD_OPENCL=true
BUILD_VULKAN=true  # Enable Vulkan by default - packaging controlled by Android CMakeLists.txt
BUILD_TYPE="Release"
CLEAN_BUILD=false
CLEAN_PREBUILT=false
INSTALL_DEPS=false
CUSTOM_GLSLC_PATH=""
CUSTOM_NDK_PATH=""
USE_PREBUILT_GPU=true  # Set to true by default, will be overridden if --no-use-prebuilt-gpu is provided
BUILD_PLATFORM="all"  # Default to building for all platforms
CUSTOM_CMAKE_FLAGS=""  # New variable for custom CMake flags

# Parse arguments
for arg in "$@"; do
  case $arg in
    --help)
      print_usage
      exit 0
      ;;
    --abi=*)
      BUILD_ABI="${arg#*=}"
      ;;
    --no-opencl)
      BUILD_OPENCL=false
      ;;
    --no-vulkan)
      BUILD_VULKAN=false
      ;;
    --vulkan)
      BUILD_VULKAN=true
      ;;
    --debug)
      BUILD_TYPE="Debug"
      ;;
    --clean)
      CLEAN_BUILD=true
      ;;
    --clean-prebuilt)
      CLEAN_PREBUILT=true
      ;;
    --install-deps)
      INSTALL_DEPS=true
      ;;
    --glslc-path=*)
      CUSTOM_GLSLC_PATH="${arg#*=}"
      ;;
    --ndk-path=*)
      CUSTOM_NDK_PATH="${arg#*=}"
      ;;
    --no-use-prebuilt-gpu)
      USE_PREBUILT_GPU=false
      ;;
    --platform=*)
      BUILD_PLATFORM="${arg#*=}"
      ;;
    --cmake-flags=*)
      CUSTOM_CMAKE_FLAGS="${arg#*=}"
      ;;
    *)
      echo -e "${RED}Unknown argument: $arg${NC}"
      print_usage
      exit 1
      ;;
  esac
done

# Define prebuilt directory for all intermediary files
PREBUILT_DIR="$PROJECT_ROOT/prebuilt"
PREBUILT_LIBS_DIR="$PREBUILT_DIR/libs"
PREBUILT_EXTERNAL_DIR="$PREBUILT_DIR/libs/external"
PREBUILT_BUILD_DIR="$PREBUILT_DIR/build-android"
PREBUILT_GPU_DIR="$PREBUILT_DIR/gpu"

# Define directories
ANDROID_DIR="$PROJECT_ROOT/android"
ANDROID_JNI_DIR="$ANDROID_DIR/src/main/jniLibs"
ANDROID_CPP_DIR="$ANDROID_DIR/src/main/cpp"
CPP_DIR="$PROJECT_ROOT/cpp"
LLAMA_CPP_DIR="$CPP_DIR/llama.cpp"

# Third-party directories in prebuilt directory
THIRD_PARTY_DIR="$PREBUILT_DIR/third_party"
OPENCL_HEADERS_DIR="$THIRD_PARTY_DIR/OpenCL-Headers"
OPENCL_LOADER_DIR="$THIRD_PARTY_DIR/OpenCL-ICD-Loader"
OPENCL_INCLUDE_DIR="$PREBUILT_EXTERNAL_DIR/opencl/include"
OPENCL_LIB_DIR="$PREBUILT_EXTERNAL_DIR/opencl/lib"
VULKAN_HEADERS_DIR="$THIRD_PARTY_DIR/Vulkan-Headers"
VULKAN_INCLUDE_DIR="$PREBUILT_EXTERNAL_DIR/vulkan/include"

# Clean up Android directory if requested
if [ "$CLEAN_BUILD" = true ] || [ "$CLEAN_PREBUILT" = true ]; then
    echo -e "${YELLOW}Cleaning Android build artifacts...${NC}"
    rm -rf "$ANDROID_DIR/.cxx"
    rm -rf "$ANDROID_DIR/build"
    # Clean jniLibs directory
    rm -rf "$ANDROID_JNI_DIR"/*
fi

# Create necessary directories
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p "$PREBUILT_DIR"
mkdir -p "$PREBUILT_LIBS_DIR"
mkdir -p "$PREBUILT_EXTERNAL_DIR"
mkdir -p "$PREBUILT_BUILD_DIR"
mkdir -p "$THIRD_PARTY_DIR"
mkdir -p "$OPENCL_INCLUDE_DIR"
mkdir -p "$OPENCL_LIB_DIR"
mkdir -p "$VULKAN_INCLUDE_DIR"
mkdir -p "$ANDROID_JNI_DIR/arm64-v8a"
mkdir -p "$ANDROID_JNI_DIR/x86_64"
mkdir -p "$ANDROID_JNI_DIR/armeabi-v7a"
mkdir -p "$ANDROID_JNI_DIR/x86"
mkdir -p "$ANDROID_CPP_DIR/include"

# Determine platform and setup environment
if [[ "$OSTYPE" == "darwin"* ]]; then
  echo -e "${YELLOW}Building on macOS${NC}"
  # Check if we're on ARM Mac
  if [[ $(uname -m) == "arm64" ]]; then
    echo -e "${YELLOW}Detected ARM64 macOS${NC}"
  else
    echo -e "${YELLOW}Detected Intel macOS${NC}"
  fi
  # Detect number of cores on macOS
  N_CORES=$(sysctl -n hw.ncpu)
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
  echo -e "${YELLOW}Building on Linux${NC}"
  # Detect number of cores on Linux
  N_CORES=$(nproc)
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
  echo -e "${YELLOW}Building on Windows${NC}"
  # Detect number of cores on Windows
  N_CORES=$NUMBER_OF_PROCESSORS
else
  echo -e "${YELLOW}Unknown OS type: $OSTYPE, assuming 4 cores${NC}"
  N_CORES=4
fi

echo -e "${YELLOW}Using $N_CORES cores for building${NC}"

# Set up and verify NDK path
if [ -z "$ANDROID_HOME" ]; then
  if [ -n "$ANDROID_SDK_ROOT" ]; then
    ANDROID_HOME="$ANDROID_SDK_ROOT"
  elif [ -d "$HOME/Android/Sdk" ]; then
    ANDROID_HOME="$HOME/Android/Sdk"
  elif [ -d "$HOME/Library/Android/sdk" ]; then
    ANDROID_HOME="$HOME/Library/Android/sdk"
  else
    echo -e "${RED}Android SDK not found. Please set ANDROID_HOME or ANDROID_SDK_ROOT.${NC}"
    exit 1
  fi
fi

# Try to use the user-provided NDK path first
if [ -n "$CUSTOM_NDK_PATH" ]; then
  NDK_PATH="$CUSTOM_NDK_PATH"
  echo -e "${GREEN}Using custom NDK path: $NDK_PATH${NC}"
  
  if [ ! -d "$NDK_PATH" ]; then
    echo -e "${RED}Custom NDK path not found at $NDK_PATH${NC}"
    exit 1
  fi
else
  # First try to find any available NDK
  if [ -d "$ANDROID_HOME/ndk" ]; then
    # Get list of NDK versions sorted by version number (newest first)
    NEWEST_NDK_VERSION=$(ls -1 "$ANDROID_HOME/ndk" | sort -rV | head -n 1)
    
    if [ -n "$NEWEST_NDK_VERSION" ]; then
      NDK_PATH="$ANDROID_HOME/ndk/$NEWEST_NDK_VERSION"
      echo -e "${GREEN}Found NDK version $NEWEST_NDK_VERSION, using this version${NC}"
    else
      # If no NDK is found, fall back to the version from used_version.sh
      NDK_PATH="$ANDROID_HOME/ndk/$NDK_VERSION"
      echo -e "${YELLOW}No NDK versions found in $ANDROID_HOME/ndk, trying to use version $NDK_VERSION from used_version.sh${NC}"
      
      if [ ! -d "$NDK_PATH" ]; then
        echo -e "${RED}NDK version $NDK_VERSION not found at $NDK_PATH${NC}"
        echo -e "${YELLOW}Please install Android NDK using Android SDK Manager:${NC}"
        echo -e "${YELLOW}\$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager \"ndk;latest\"${NC}"
        exit 1
      fi
    fi
  # Check for NDK in the old-style location
  elif [ -d "$ANDROID_HOME/ndk-bundle" ]; then
    NDK_PATH="$ANDROID_HOME/ndk-bundle"
    echo -e "${GREEN}Found NDK at ndk-bundle location: $NDK_PATH${NC}"
  else
    # Try to find the NDK version specified in used_version.sh as last resort
    NDK_PATH="$ANDROID_HOME/ndk/$NDK_VERSION"
    echo -e "${YELLOW}No NDK directory found, trying to use version $NDK_VERSION from used_version.sh${NC}"
    
    if [ ! -d "$NDK_PATH" ]; then
      echo -e "${RED}NDK directory not found at $ANDROID_HOME/ndk or $ANDROID_HOME/ndk-bundle${NC}"
      echo -e "${RED}NDK version $NDK_VERSION from used_version.sh not found either${NC}"
      echo -e "${YELLOW}Please install Android NDK using Android SDK Manager:${NC}"
      echo -e "${YELLOW}\$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager \"ndk;latest\"${NC}"
      exit 1
    fi
  fi
fi

# Extract the Android platform version from the NDK path
if [ -d "$NDK_PATH/platforms" ]; then
  # Get the highest API level available in the NDK
  ANDROID_PLATFORM=$(ls -1 "$NDK_PATH/platforms" | sort -V | tail -n 1)
  ANDROID_MIN_SDK=${ANDROID_PLATFORM#android-}
  echo -e "${GREEN}Using Android platform: $ANDROID_PLATFORM (API level $ANDROID_MIN_SDK)${NC}"
elif [ -d "$NDK_PATH/toolchains/llvm/prebuilt" ]; then
  # Try to detect from the LLVM toolchain - more reliable method
  HOST_TAG_DIR=$(ls -1 "$NDK_PATH/toolchains/llvm/prebuilt/")
  if [ -n "$HOST_TAG_DIR" ]; then
    HOST_PLATFORM_DIR="$NDK_PATH/toolchains/llvm/prebuilt/$HOST_TAG_DIR"
    
    # For NDK r21 and higher - check the sysroot structure
    if [ -d "$HOST_PLATFORM_DIR/sysroot/usr/lib/aarch64-linux-android" ]; then
      # List all available API levels for ARM64
      API_LEVELS=$(find "$HOST_PLATFORM_DIR/sysroot/usr/lib/aarch64-linux-android" -maxdepth 1 -type d -name "[0-9]*" 2>/dev/null | sort -V)
      
      if [ -n "$API_LEVELS" ]; then
        # Get the highest API level available
        HIGHEST_API=$(basename $(echo "$API_LEVELS" | tail -n 1))
        ANDROID_MIN_SDK=$HIGHEST_API
        ANDROID_PLATFORM="android-$ANDROID_MIN_SDK"
        echo -e "${GREEN}Using Android platform: $ANDROID_PLATFORM (API level $ANDROID_MIN_SDK)${NC}"
      else
        # Default to API level 24 (Android 7.0) if we can't detect specific level
        ANDROID_MIN_SDK=24
        ANDROID_PLATFORM="android-$ANDROID_MIN_SDK"
        echo -e "${YELLOW}Could not detect specific API level, defaulting to $ANDROID_PLATFORM (API level $ANDROID_MIN_SDK)${NC}"
      fi
    else
      # Direct check for highest API level in the bin directory
      BINS=$(find "$HOST_PLATFORM_DIR/bin" -name "aarch64-linux-android*-clang" 2>/dev/null | sort -V)
      if [ -n "$BINS" ]; then
        # Extract API level from binary name
        HIGHEST_BIN=$(basename $(echo "$BINS" | tail -n 1))
        ANDROID_MIN_SDK=$(echo "$HIGHEST_BIN" | grep -o '[0-9]*' | head -n 1)
        if [ -n "$ANDROID_MIN_SDK" ]; then
          ANDROID_PLATFORM="android-$ANDROID_MIN_SDK"
          echo -e "${GREEN}Using Android platform from compiler: $ANDROID_PLATFORM (API level $ANDROID_MIN_SDK)${NC}"
        else
          # Hardcoded fallback to a safe API level
          ANDROID_MIN_SDK=24
          ANDROID_PLATFORM="android-$ANDROID_MIN_SDK"
          echo -e "${YELLOW}Could not detect API level from compiler, defaulting to $ANDROID_PLATFORM (API level $ANDROID_MIN_SDK)${NC}"
        fi
      else
        # Hardcoded fallback to a safe API level
        ANDROID_MIN_SDK=24
        ANDROID_PLATFORM="android-$ANDROID_MIN_SDK"
        echo -e "${YELLOW}Could not detect Android platform, defaulting to $ANDROID_PLATFORM (API level $ANDROID_MIN_SDK)${NC}"
      fi
    fi
  else
    # Hardcoded fallback to a safe API level
    ANDROID_MIN_SDK=24
    ANDROID_PLATFORM="android-$ANDROID_MIN_SDK"
    echo -e "${YELLOW}Could not detect Android platform, defaulting to $ANDROID_PLATFORM (API level $ANDROID_MIN_SDK)${NC}"
  fi
else
  # Hardcoded fallback to a safe API level
  ANDROID_MIN_SDK=24
  ANDROID_PLATFORM="android-$ANDROID_MIN_SDK"
  echo -e "${YELLOW}Could not detect Android platform, defaulting to $ANDROID_PLATFORM (API level $ANDROID_MIN_SDK)${NC}"
fi

# Determine the host tag based on the OS
HOST_TAG=""
if [[ "$OSTYPE" == "darwin"* ]]; then
  if [[ $(uname -m) == "arm64" ]]; then
    HOST_TAG="darwin-aarch64"
  else
    HOST_TAG="darwin-x86_64"
  fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
  HOST_TAG="linux-x86_64"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
  HOST_TAG="windows-x86_64"
else
  echo -e "${RED}Unsupported OS type: $OSTYPE${NC}"
  exit 1
fi

# Check if OpenCL libraries are available from prebuilt
OPENCL_AVAILABLE=false
if [ "$USE_PREBUILT_GPU" = true ]; then
  # First check the prebuilt/gpu directory (from build_android_gpu_backend.sh)
  for ABI in "${ABIS[@]}"; do
    if [ -d "$PREBUILT_GPU_DIR/$ABI" ] && [ -f "$PREBUILT_GPU_DIR/$ABI/libOpenCL.so" ]; then
      OPENCL_AVAILABLE=true
      echo -e "${GREEN}Found prebuilt OpenCL library for $ABI in $PREBUILT_GPU_DIR/$ABI${NC}"
    elif [ -d "$OPENCL_LIB_DIR/$ABI" ] && [ -f "$OPENCL_LIB_DIR/$ABI/libOpenCL.so" ]; then
      OPENCL_AVAILABLE=true
      echo -e "${GREEN}Found prebuilt OpenCL library for $ABI in $OPENCL_LIB_DIR${NC}"
    else
      OPENCL_AVAILABLE=false
      echo -e "${YELLOW}Prebuilt OpenCL library not found for $ABI${NC}"
      break
    fi
  done
fi

# Check if Vulkan libraries and resources are available
VULKAN_AVAILABLE=false
VULKAN_SDK_PATH=""
GLSLC_PATH="$CUSTOM_GLSLC_PATH"

# Define the host platform dir first (needed for Vulkan detection)
HOST_PLATFORM_DIR="$NDK_PATH/toolchains/llvm/prebuilt/$HOST_TAG"

# Since NDK 23+, Vulkan is included in the NDK, so we can always enable it
VULKAN_AVAILABLE=true
echo -e "${GREEN}Using NDK built-in Vulkan support (NDK 23+)${NC}"

# Look for glslc in NDK if not specified by user
if [ -z "$GLSLC_PATH" ]; then
  NDK_GLSLC="$NDK_PATH/shader-tools/$HOST_TAG/glslc"
  if [ -f "$NDK_GLSLC" ]; then
    GLSLC_PATH="$NDK_GLSLC"
    echo -e "${GREEN}Found glslc compiler in NDK: $GLSLC_PATH${NC}"
  else
    # Look for glslc in system path
    SYS_GLSLC=$(which glslc 2>/dev/null || echo "")
    if [ -n "$SYS_GLSLC" ]; then
      GLSLC_PATH="$SYS_GLSLC"
      echo -e "${GREEN}Found glslc compiler in system PATH: $GLSLC_PATH${NC}"
    else
      echo -e "${YELLOW}Warning: glslc compiler not found, this may cause shader compilation issues${NC}"
    fi
  fi
fi

# Gather common CMake arguments
CMAKE_ARGS=(
  -DCMAKE_TOOLCHAIN_FILE="$NDK_PATH/build/cmake/android.toolchain.cmake"
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  -DANDROID_PLATFORM="$ANDROID_PLATFORM"
  -DBUILD_SHARED_LIBS=ON  # Build shared libraries
  -DLLAMA_BUILD_SERVER=OFF
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DGGML_OPENMP=OFF # Ensure OpenMP is disabled for prebuilts (correct flag for ggml)
  -DLLAMA_CLBLAST=ON
  -DGGML_NATIVE=OFF  # Disable native CPU optimizations for cross-compiling
  -DGGML_USE_K_QUANTS=ON  # Enable quantization support
  -DLLAMA_CURL=OFF
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON  # Ensure PIC is enabled
  -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations"  # Ignore deprecated warnings (for wstring_convert)
  -DGGML_BACKEND_DL=OFF  # Static libraries - CPU backend built into main libraries
  -DGGML_CPU=ON  # CPU backend statically built into libggml.so and libllama.so
  -DGGML_METAL=OFF  # Disable Metal (Apple GPU) for Android
  -DGGML_CUDA=OFF   # Static build - GPU backends added separately later
  -DGGML_HIP=OFF    # Static build - GPU backends added separately later
  -DGGML_OPENCL=OFF  # Build as separate dynamic library only
  -DGGML_VULKAN=OFF  # Build as separate dynamic library only
)

# Check if llama.cpp repository exists and is properly set up
if [ ! -d "$LLAMA_CPP_DIR" ]; then
  echo -e "${RED}llama.cpp repository not found at: $LLAMA_CPP_DIR${NC}"
  echo -e "${YELLOW}Please run setupLlamaCpp.sh init first to initialize the repository${NC}"
  exit 1
fi

# Check for llama.h in both possible locations
if [ -f "$LLAMA_CPP_DIR/include/llama.h" ]; then
  echo -e "${GREEN}Found llama.h in include directory${NC}"
  LLAMA_H_PATH="$LLAMA_CPP_DIR/include/llama.h"
elif [ -f "$LLAMA_CPP_DIR/llama.h" ]; then
  echo -e "${GREEN}Found llama.h in root directory${NC}"
  LLAMA_H_PATH="$LLAMA_CPP_DIR/llama.h"
else
  echo -e "${RED}llama.cpp repository seems to be incomplete or corrupt.${NC}"
  echo -e "${YELLOW}llama.h not found at expected locations:${NC}"
  echo -e "${YELLOW}- $LLAMA_CPP_DIR/llama.h${NC}"
  echo -e "${YELLOW}- $LLAMA_CPP_DIR/include/llama.h${NC}"
  echo -e "${YELLOW}Please run setupLlamaCpp.sh init --platform=android to initialize the repository properly${NC}"
  exit 1
fi

echo -e "${GREEN}llama.cpp repository found and appears valid at: $LLAMA_CPP_DIR${NC}"

# Configure additional GPU backend flags
# Note: Core libraries are built with CPU-only to ensure stability
# GPU backends are built as separate dynamic libraries when enabled

GPU_CMAKE_FLAGS=()

# Add OpenCL backend if available and enabled
if [ "$BUILD_OPENCL" = true ] && [ "$OPENCL_AVAILABLE" = true ]; then
  GPU_CMAKE_FLAGS+=(
    -DGGML_OPENCL=ON
    -DCL_INCLUDE_DIR="$OPENCL_INCLUDE_DIR"
    -DCL_LIBRARY="$OPENCL_LIB_DIR"
  )
  echo -e "${GREEN}OpenCL backend will be built as separate dynamic library${NC}"
else
  echo -e "${YELLOW}OpenCL backend disabled${NC}"
fi

# Add Vulkan backend if available and enabled
if [ "$BUILD_VULKAN" = true ] && [ "$VULKAN_AVAILABLE" = true ]; then
  GPU_CMAKE_FLAGS+=(
    -DGGML_VULKAN=ON
    -DGGML_VULKAN_CHECK_RESULTS=OFF
    -DGGML_VULKAN_DEBUG=OFF
    -DGGML_VULKAN_MEMORY_DEBUG=OFF
    -DGGML_VULKAN_SHADER_DEBUG_INFO=OFF
    -DGGML_VULKAN_PERF=OFF
    -DGGML_VULKAN_VALIDATE=OFF
    -DGGML_VULKAN_RUN_TESTS=OFF
    -DVK_USE_PLATFORM_ANDROID_KHR=ON
    -DGGML_VULKAN_DISABLE_FLASHATTN=ON  # Disable flash attention for Adreno compatibility
  )
  
  # Disable cooperative matrix features if environment variables are set
  if [ -n "$GGML_VK_DISABLE_COOPMAT" ]; then
    export GGML_VK_DISABLE_COOPMAT="$GGML_VK_DISABLE_COOPMAT"
    echo -e "${YELLOW}Cooperative matrix support disabled via environment variable${NC}"
  fi
  
  if [ -n "$GGML_VK_DISABLE_COOPMAT2" ]; then
    export GGML_VK_DISABLE_COOPMAT2="$GGML_VK_DISABLE_COOPMAT2"
    echo -e "${YELLOW}Cooperative matrix 2 support disabled via environment variable${NC}"
  fi
  
  # Check if we have Vulkan environment info from build_android_gpu_backend.sh
  VULKAN_ENV_FILE="$PREBUILT_GPU_DIR/$ABI/.vulkan_env"
  if [ -f "$VULKAN_ENV_FILE" ]; then
    echo -e "${GREEN}Loading Vulkan environment from: $VULKAN_ENV_FILE${NC}"
    source "$VULKAN_ENV_FILE"
    
    # Use the environment variables from the file
    if [ -n "$VULKAN_LIBRARY_PATH" ] && [ -f "$VULKAN_LIBRARY_PATH" ]; then
      GPU_CMAKE_FLAGS+=(-DVulkan_LIBRARY="$VULKAN_LIBRARY_PATH")
      echo -e "${GREEN}Using Vulkan library from env: $VULKAN_LIBRARY_PATH${NC}"
    fi
    
    if [ -n "$VULKAN_INCLUDE_PATH" ] && [ -d "$VULKAN_INCLUDE_PATH" ]; then
      # Check if this include path has vulkan.hpp
      if [ -f "$VULKAN_INCLUDE_PATH/vulkan/vulkan.hpp" ]; then
        GPU_CMAKE_FLAGS+=(-DVulkan_INCLUDE_DIR="$VULKAN_INCLUDE_PATH")
        echo -e "${GREEN}Using Vulkan include path from env: $VULKAN_INCLUDE_PATH${NC}"
      else
        # NDK include path doesn't have vulkan.hpp, check for system headers
        if [ -f "/usr/include/vulkan/vulkan.hpp" ]; then
          echo -e "${YELLOW}NDK headers missing vulkan.hpp, using system headers for C++${NC}"
          GPU_CMAKE_FLAGS+=(-DVulkan_INCLUDE_DIR="/usr/include")
          # Also add NDK headers as additional include for vulkan.h
          GPU_CMAKE_FLAGS+=(-DCMAKE_CXX_FLAGS="-I$VULKAN_INCLUDE_PATH -Wno-deprecated-declarations")
          echo -e "${GREEN}Using system Vulkan C++ headers: /usr/include${NC}"
          echo -e "${GREEN}Adding NDK Vulkan C headers: $VULKAN_INCLUDE_PATH${NC}"
        else
          GPU_CMAKE_FLAGS+=(-DVulkan_INCLUDE_DIR="$VULKAN_INCLUDE_PATH")
          echo -e "${YELLOW}Warning: Using NDK headers without vulkan.hpp${NC}"
        fi
      fi
    fi
    
    if [ -n "$GLSLC_EXECUTABLE" ] && [ -f "$GLSLC_EXECUTABLE" ] && [ -x "$GLSLC_EXECUTABLE" ]; then
      GPU_CMAKE_FLAGS+=(-DVulkan_GLSLC_EXECUTABLE="$GLSLC_EXECUTABLE")
      echo -e "${GREEN}Using glslc from env: $GLSLC_EXECUTABLE${NC}"
    fi
  else
    # Fallback to the original logic
    echo -e "${YELLOW}No Vulkan environment file found, using fallback detection${NC}"
    
    # Check for system Vulkan headers first (they're more likely to have vulkan.hpp)
    if [ -f "/usr/include/vulkan/vulkan.hpp" ]; then
      echo -e "${GREEN}Found system Vulkan C++ headers, using them${NC}"
      GPU_CMAKE_FLAGS+=(-DVulkan_INCLUDE_DIR="/usr/include")
      
      # Set architecture-specific Vulkan library path from NDK for linking
      if [ "$ABI" = "arm64-v8a" ]; then
        VULKAN_LIB_PATH="$HOST_PLATFORM_DIR/sysroot/usr/lib/aarch64-linux-android/$ANDROID_MIN_SDK/libvulkan.so"
      elif [ "$ABI" = "x86_64" ]; then
        VULKAN_LIB_PATH="$HOST_PLATFORM_DIR/sysroot/usr/lib/x86_64-linux-android/$ANDROID_MIN_SDK/libvulkan.so"
      elif [ "$ABI" = "armeabi-v7a" ]; then
        VULKAN_LIB_PATH="$HOST_PLATFORM_DIR/sysroot/usr/lib/arm-linux-androideabi/$ANDROID_MIN_SDK/libvulkan.so"
      elif [ "$ABI" = "x86" ]; then
        VULKAN_LIB_PATH="$HOST_PLATFORM_DIR/sysroot/usr/lib/i686-linux-android/$ANDROID_MIN_SDK/libvulkan.so"
      fi
      
      # Add Vulkan library path if it exists
      if [ -f "$VULKAN_LIB_PATH" ]; then
        GPU_CMAKE_FLAGS+=(-DVulkan_LIBRARY="$VULKAN_LIB_PATH")
        echo -e "${GREEN}Using NDK Vulkan library for linking: $VULKAN_LIB_PATH${NC}"
      fi
    else
      # Fall back to NDK headers
      echo -e "${YELLOW}No system Vulkan headers found, trying NDK headers${NC}"
      
      # Set architecture-specific Vulkan library path from NDK
      if [ "$ABI" = "arm64-v8a" ]; then
        VULKAN_LIB_PATH="$HOST_PLATFORM_DIR/sysroot/usr/lib/aarch64-linux-android/$ANDROID_MIN_SDK/libvulkan.so"
      elif [ "$ABI" = "x86_64" ]; then
        VULKAN_LIB_PATH="$HOST_PLATFORM_DIR/sysroot/usr/lib/x86_64-linux-android/$ANDROID_MIN_SDK/libvulkan.so"
      elif [ "$ABI" = "armeabi-v7a" ]; then
        VULKAN_LIB_PATH="$HOST_PLATFORM_DIR/sysroot/usr/lib/arm-linux-androideabi/$ANDROID_MIN_SDK/libvulkan.so"
      elif [ "$ABI" = "x86" ]; then
        VULKAN_LIB_PATH="$HOST_PLATFORM_DIR/sysroot/usr/lib/i686-linux-android/$ANDROID_MIN_SDK/libvulkan.so"
      fi
      
      # Add Vulkan library path if it exists
      if [ -f "$VULKAN_LIB_PATH" ]; then
        GPU_CMAKE_FLAGS+=(-DVulkan_LIBRARY="$VULKAN_LIB_PATH")
        echo -e "${GREEN}Using NDK Vulkan library: $VULKAN_LIB_PATH${NC}"
      else
        echo -e "${YELLOW}Warning: NDK Vulkan library not found at $VULKAN_LIB_PATH${NC}"
      fi
      
      # Add Vulkan include directory (NDK headers)
      VULKAN_INCLUDE_PATH="$HOST_PLATFORM_DIR/sysroot/usr/include"
      GPU_CMAKE_FLAGS+=(-DVulkan_INCLUDE_DIR="$VULKAN_INCLUDE_PATH")
    fi
    
    # Add glslc path if available
    if [ -n "$GLSLC_PATH" ]; then
      GPU_CMAKE_FLAGS+=(-DVulkan_GLSLC_EXECUTABLE="$GLSLC_PATH")
      
      # Verify the glslc executable is usable
      if [ -f "$GLSLC_PATH" ] && [ -x "$GLSLC_PATH" ]; then
        echo -e "${GREEN}Using glslc from: $GLSLC_PATH${NC}"
      else
        echo -e "${YELLOW}Warning: glslc at $GLSLC_PATH is not executable or doesn't exist${NC}"
      fi
    fi
  fi
  echo -e "${GREEN}Vulkan backend will be built as separate dynamic library${NC}"
else
  echo -e "${YELLOW}Vulkan backend disabled${NC}"
fi

# Define ABIs to build (both 32-bit and 64-bit architectures)
if [ "$BUILD_ABI" = "all" ]; then
  ABIS=("arm64-v8a" "x86_64" "armeabi-v7a" "x86")  # All common Android architectures
elif [ "$BUILD_ABI" = "arm64-v8a" ] || [ "$BUILD_ABI" = "x86_64" ] || [ "$BUILD_ABI" = "armeabi-v7a" ] || [ "$BUILD_ABI" = "x86" ]; then
  ABIS=("$BUILD_ABI")
else
  echo -e "${RED}Invalid ABI: $BUILD_ABI. Supported ABIs are: all, arm64-v8a, x86_64, armeabi-v7a, x86${NC}"
  exit 1
fi

# Define build function
build_for_abi() {
  local ABI=$1
  echo -e "${GREEN}Building for $ABI${NC}"
  
  # Set ABI-specific flags
  if [ "$ABI" = "arm64-v8a" ]; then
    local ARCH_FLAGS=(
      -DANDROID_ABI="arm64-v8a"
      -DCMAKE_INSTALL_PREFIX="$PREBUILT_BUILD_DIR/$ABI/install"
      -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$PREBUILT_BUILD_DIR/$ABI/lib"
    )
  elif [ "$ABI" = "x86_64" ]; then
    local ARCH_FLAGS=(
      -DANDROID_ABI="x86_64"
      -DCMAKE_INSTALL_PREFIX="$PREBUILT_BUILD_DIR/$ABI/install"
      -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$PREBUILT_BUILD_DIR/$ABI/lib"
    )
  elif [ "$ABI" = "armeabi-v7a" ]; then
    local ARCH_FLAGS=(
      -DANDROID_ABI="armeabi-v7a"
      -DCMAKE_INSTALL_PREFIX="$PREBUILT_BUILD_DIR/$ABI/install"
      -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$PREBUILT_BUILD_DIR/$ABI/lib"
      -DGGML_FP16=OFF
      -DGGML_ARM_FP16=OFF
      -DGGML_LLAMAFILE=OFF
      -DLLAMA_BUILD_TOOLS=OFF
    )
  elif [ "$ABI" = "x86" ]; then
    local ARCH_FLAGS=(
      -DANDROID_ABI="x86"
      -DCMAKE_INSTALL_PREFIX="$PREBUILT_BUILD_DIR/$ABI/install"
      -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$PREBUILT_BUILD_DIR/$ABI/lib"
      -DGGML_FP16=OFF
      -DGGML_F16C=OFF
      -DGGML_LLAMAFILE=OFF
      -DLLAMA_BUILD_TOOLS=OFF
    )
  fi
  
  # Create build directory
  local BUILD_DIR="$PREBUILT_BUILD_DIR/$ABI"
  mkdir -p "$BUILD_DIR"
  
  # Navigate to build directory
  pushd "$BUILD_DIR"
  
  # Configure with CMake
  echo -e "${YELLOW}Configuring CMake for $ABI...${NC}"
  
  # Print the exact cmake command for debugging
  echo -e "${YELLOW}Running cmake with options:${NC}"
  echo -e "cmake \"$LLAMA_CPP_DIR\" ${CMAKE_ARGS[*]} ${ARCH_FLAGS[*]} ${CUSTOM_CMAKE_FLAGS} ${GPU_CMAKE_FLAGS[*]}"
  
  cmake "$LLAMA_CPP_DIR" "${CMAKE_ARGS[@]}" "${ARCH_FLAGS[@]}" ${CUSTOM_CMAKE_FLAGS} ${GPU_CMAKE_FLAGS[@]} || {
    echo -e "${RED}CMake configuration failed for $ABI${NC}"
    popd
    return 1
  }
  
  # Build
  echo -e "${YELLOW}Building for $ABI with $N_CORES cores...${NC}"
  cmake --build . --config "$BUILD_TYPE" -- -j$N_CORES || {
    echo -e "${RED}Build failed for $ABI${NC}"
    popd
    return 1
  }
  
  # Copy libraries to jniLibs directory
  echo -e "${YELLOW}Copying libraries to jniLibs directory...${NC}"
  
  # Make sure destination directory exists
  mkdir -p "$ANDROID_JNI_DIR/$ABI"
  mkdir -p "$ANDROID_CPP_DIR/include"
  
  # Search in multiple possible locations for libllama.so and copy it as libllama.so
  if [ -f "$BUILD_DIR/bin/libllama.so" ]; then
    cp "$BUILD_DIR/bin/libllama.so" "$ANDROID_JNI_DIR/$ABI/libllama.so"
    echo -e "${GREEN}Copied libllama.so from bin directory for $ABI${NC}"
  elif [ -f "$BUILD_DIR/libllama.so" ]; then
    cp "$BUILD_DIR/libllama.so" "$ANDROID_JNI_DIR/$ABI/libllama.so"
    echo -e "${GREEN}Copied libllama.so from build root for $ABI${NC}"
  elif [ -f "$BUILD_DIR/lib/libllama.so" ]; then
    cp "$BUILD_DIR/lib/libllama.so" "$ANDROID_JNI_DIR/$ABI/libllama.so"
    echo -e "${GREEN}Copied libllama.so from lib directory for $ABI${NC}"
  elif [ -f "$BUILD_DIR/src/libllama.so" ]; then
    cp "$BUILD_DIR/src/libllama.so" "$ANDROID_JNI_DIR/$ABI/libllama.so"
    echo -e "${GREEN}Copied libllama.so from src directory for $ABI${NC}"
  else
    echo -e "${YELLOW}libllama.so not found, checking for other library formats...${NC}"
    
    # Check for other library formats and copy as libllama.so if found
    if [ -f "$BUILD_DIR/libllama.dylib" ]; then
      cp "$BUILD_DIR/libllama.dylib" "$ANDROID_JNI_DIR/$ABI/libllama.so"
      echo -e "${GREEN}Copied libllama.dylib as libllama.so for $ABI${NC}"
    elif [ -f "$BUILD_DIR/lib/libllama.dylib" ]; then
      cp "$BUILD_DIR/lib/libllama.dylib" "$ANDROID_JNI_DIR/$ABI/libllama.so"
      echo -e "${GREEN}Copied lib/libllama.dylib as libllama.so for $ABI${NC}"
    elif [ -f "$BUILD_DIR/bin/libllama.dylib" ]; then
      cp "$BUILD_DIR/bin/libllama.dylib" "$ANDROID_JNI_DIR/$ABI/libllama.so"
      echo -e "${GREEN}Copied bin/libllama.dylib as libllama.so for $ABI${NC}"
    elif [ -f "$BUILD_DIR/libllama.a" ] && [ -f "$BUILD_DIR/libggml.a" ]; then
      echo -e "${YELLOW}Warning: Only static libraries found. For Android we need shared libraries.${NC}"
      echo -e "${YELLOW}Trying to compile shared library manually...${NC}"
      
      # Create shared library from static libraries if possible
      $NDK_PATH/toolchains/llvm/prebuilt/$HOST_TAG/bin/clang++ -shared -o "$ANDROID_JNI_DIR/$ABI/libllama.so" \
        -Wl,--whole-archive "$BUILD_DIR/libllama.a" "$BUILD_DIR/libggml.a" -Wl,--no-whole-archive \
        -lc -lm -ldl
      
      if [ $? -eq 0 ]; then
        echo -e "${GREEN}Successfully created shared library from static libraries as libllama.so for $ABI${NC}"
      else
        echo -e "${RED}Failed to create shared library from static libraries${NC}"
        echo -e "${RED}libllama.so not found in any standard location:${NC}"
        find "$BUILD_DIR" -name "libllama*" | sort
        ls -la "$BUILD_DIR"
      fi
    else
      echo -e "${RED}libllama.so not found in any standard location:${NC}"
      echo -e "${YELLOW}Searching for any libllama files in the build directory...${NC}"
      find "$BUILD_DIR" -name "libllama*" | sort
      ls -la "$BUILD_DIR"
      
      # Also check for directly compiled .cpp.o files
      if [ -f "$BUILD_DIR/src/llama.cpp.o" ]; then
        echo -e "${YELLOW}Found llama.cpp.o object file, but no shared library${NC}"
      fi
      
      # Output the build directories for debugging
      echo -e "${YELLOW}Directory structure:${NC}"
      find "$BUILD_DIR" -type d | sort
      
      return 1
    fi
  fi
  
  # Copy libggml-base.so
  if [ -f "$BUILD_DIR/bin/libggml-base.so" ]; then
    cp "$BUILD_DIR/bin/libggml-base.so" "$ANDROID_JNI_DIR/$ABI/"
    echo -e "${GREEN}Copied libggml-base.so for $ABI${NC}"
  elif [ -f "$BUILD_DIR/libggml-base.so" ]; then
    cp "$BUILD_DIR/libggml-base.so" "$ANDROID_JNI_DIR/$ABI/"
    echo -e "${GREEN}Copied libggml-base.so for $ABI${NC}"
  fi
  
  # Copy libggml.so (includes CPU backend with GGML_BACKEND_DL)
  if [ -f "$BUILD_DIR/bin/libggml.so" ]; then
    cp "$BUILD_DIR/bin/libggml.so" "$ANDROID_JNI_DIR/$ABI/"
    echo -e "${GREEN}Copied libggml.so for $ABI (includes CPU backend)${NC}"
  elif [ -f "$BUILD_DIR/libggml.so" ]; then
    cp "$BUILD_DIR/libggml.so" "$ANDROID_JNI_DIR/$ABI/"
    echo -e "${GREEN}Copied libggml.so for $ABI (includes CPU backend)${NC}"
  fi
  
  # Copy libggml-cpu.so (essential CPU backend)
  if [ -f "$BUILD_DIR/bin/libggml-cpu.so" ]; then
    cp "$BUILD_DIR/bin/libggml-cpu.so" "$ANDROID_JNI_DIR/$ABI/"
    echo -e "${GREEN}Copied libggml-cpu.so for $ABI${NC}"
  elif [ -f "$BUILD_DIR/libggml-cpu.so" ]; then
    cp "$BUILD_DIR/libggml-cpu.so" "$ANDROID_JNI_DIR/$ABI/"
    echo -e "${GREEN}Copied libggml-cpu.so for $ABI${NC}"
  fi
  
  # With GGML_BACKEND_DL, backends are built as separate dynamic libraries
  # Copy all available backend libraries (they'll be loaded dynamically at runtime)
  for backend_lib in libggml-cpu.so libggml-opencl.so libggml-vulkan.so; do
    if [ -f "$BUILD_DIR/bin/$backend_lib" ]; then
      cp "$BUILD_DIR/bin/$backend_lib" "$ANDROID_JNI_DIR/$ABI/"
      echo -e "${GREEN}Copied dynamic backend $backend_lib for $ABI${NC}"
    elif [ -f "$BUILD_DIR/$backend_lib" ]; then
      cp "$BUILD_DIR/$backend_lib" "$ANDROID_JNI_DIR/$ABI/"
      echo -e "${GREEN}Copied dynamic backend $backend_lib for $ABI${NC}"
    fi
  done
  
  # Copy static libraries if they exist (for fallback)
  for lib in libggml.a libllama.a; do
    if [ -f "$BUILD_DIR/$lib" ]; then
      cp "$BUILD_DIR/$lib" "$ANDROID_JNI_DIR/$ABI/"
      echo -e "${GREEN}Copied $lib for $ABI${NC}"
    fi
  done
  
  # Copy header files to include directory
  echo -e "${YELLOW}Copying header files...${NC}"
  mkdir -p "$ANDROID_CPP_DIR/include/llama.cpp"
  
  # Copy the main llama.h header using the path we detected
  if [ -n "$LLAMA_H_PATH" ]; then
    cp "$LLAMA_H_PATH" "$ANDROID_CPP_DIR/include/"
    echo -e "${GREEN}Copied llama.h from $LLAMA_H_PATH${NC}"
  else
    # Fallback to trying both locations
    if [ -f "$LLAMA_CPP_DIR/include/llama.h" ]; then
      cp "$LLAMA_CPP_DIR/include/llama.h" "$ANDROID_CPP_DIR/include/"
      echo -e "${GREEN}Copied llama.h from include directory${NC}"
    elif [ -f "$LLAMA_CPP_DIR/llama.h" ]; then
      cp "$LLAMA_CPP_DIR/llama.h" "$ANDROID_CPP_DIR/include/"
      echo -e "${GREEN}Copied llama.h from root directory${NC}"
    else
      echo -e "${RED}Error: llama.h not found${NC}"
    fi
  fi
  
  # Also try to copy llama-cpp.h if it exists (for C++ wrapper)
  if [ -f "$LLAMA_CPP_DIR/include/llama-cpp.h" ]; then
    cp "$LLAMA_CPP_DIR/include/llama-cpp.h" "$ANDROID_CPP_DIR/include/"
    echo -e "${GREEN}Copied llama-cpp.h from include directory${NC}"
  fi
  
  # Copy other necessary headers
  if [ -f "$LLAMA_CPP_DIR/ggml.h" ]; then
    cp "$LLAMA_CPP_DIR/ggml.h" "$ANDROID_CPP_DIR/include/"
    echo -e "${GREEN}Copied ggml.h from root directory${NC}"
  elif [ -f "$LLAMA_CPP_DIR/include/ggml.h" ]; then
    cp "$LLAMA_CPP_DIR/include/ggml.h" "$ANDROID_CPP_DIR/include/"
    echo -e "${GREEN}Copied ggml.h from include directory${NC}"
  fi
  
  if [ -f "$LLAMA_CPP_DIR/ggml-alloc.h" ]; then
    cp "$LLAMA_CPP_DIR/ggml-alloc.h" "$ANDROID_CPP_DIR/include/"
    echo -e "${GREEN}Copied ggml-alloc.h from root directory${NC}"
  elif [ -f "$LLAMA_CPP_DIR/include/ggml-alloc.h" ]; then
    cp "$LLAMA_CPP_DIR/include/ggml-alloc.h" "$ANDROID_CPP_DIR/include/"
    echo -e "${GREEN}Copied ggml-alloc.h from include directory${NC}"
  fi
  
  # Copy additional headers that might be needed
  for header in common.h ggml-backend.h llama-util.h; do
    if [ -f "$LLAMA_CPP_DIR/$header" ]; then
      cp "$LLAMA_CPP_DIR/$header" "$ANDROID_CPP_DIR/include/"
      echo -e "${GREEN}Copied $header from root directory${NC}"
    elif [ -f "$LLAMA_CPP_DIR/include/$header" ]; then
      cp "$LLAMA_CPP_DIR/include/$header" "$ANDROID_CPP_DIR/include/"
      echo -e "${GREEN}Copied $header from include directory${NC}"
    fi
  done
  
  # Copy our native library if building for Android
  if [ "$BUILD_PLATFORM" = "all" ] || [ "$BUILD_PLATFORM" = "android" ]; then
    echo -e "${YELLOW}Copying native libraries for Android...${NC}"
    
    # Copy OpenCL library if available
    if [ "$BUILD_OPENCL" = true ] && [ "$OPENCL_AVAILABLE" = true ]; then
      # First check in the prebuilt/gpu directory
      if [ -f "$PREBUILT_GPU_DIR/$ABI/libOpenCL.so" ]; then
        cp "$PREBUILT_GPU_DIR/$ABI/libOpenCL.so" "$ANDROID_JNI_DIR/$ABI/"
        # Create a flag file to indicate OpenCL is enabled
        touch "$ANDROID_JNI_DIR/$ABI/.opencl_enabled"
        echo -e "${GREEN}Copied OpenCL library for $ABI from prebuilt/gpu directory${NC}"
      elif [ -f "$OPENCL_LIB_DIR/$ABI/libOpenCL.so" ]; then
        cp "$OPENCL_LIB_DIR/$ABI/libOpenCL.so" "$ANDROID_JNI_DIR/$ABI/"
        # Create a flag file to indicate OpenCL is enabled
        touch "$ANDROID_JNI_DIR/$ABI/.opencl_enabled"
        echo -e "${GREEN}Copied OpenCL library for $ABI from libs/external/opencl directory${NC}"
      else
        echo -e "${YELLOW}OpenCL library not found for $ABI${NC}"
      fi
    fi
    
    # Create a flag file for Vulkan if enabled
    if [ "$BUILD_VULKAN" = true ] && [ "$VULKAN_AVAILABLE" = true ]; then
      touch "$ANDROID_JNI_DIR/$ABI/.vulkan_enabled"
      echo -e "${GREEN}Created Vulkan enabled flag for $ABI${NC}"
      
      # Copy Vulkan header files
      if [ -d "$VULKAN_INCLUDE_DIR" ]; then
        mkdir -p "$ANDROID_CPP_DIR/include/vulkan"
        cp -r "$VULKAN_INCLUDE_DIR/vulkan" "$ANDROID_CPP_DIR/include/"
        echo -e "${GREEN}Copied Vulkan headers to include directory${NC}"
      fi
    fi
    
    # Copy OpenCL headers if enabled
    if [ "$BUILD_OPENCL" = true ] && [ "$OPENCL_AVAILABLE" = true ]; then
      if [ -d "$OPENCL_INCLUDE_DIR" ]; then
        mkdir -p "$ANDROID_CPP_DIR/include/CL"
        cp -r "$OPENCL_INCLUDE_DIR/CL" "$ANDROID_CPP_DIR/include/"
        echo -e "${GREEN}Copied OpenCL headers to include directory${NC}"
      fi
    fi
  fi
  
  # Return to previous directory
  popd
  
  echo -e "${GREEN}Build completed successfully for $ABI${NC}"
  return 0
}

# Build for each ABI
BUILD_SUCCESS=true
FAILED_ABIS=()

for ABI in "${ABIS[@]}"; do
  echo -e "${YELLOW}Starting build for $ABI...${NC}"
  
  # Clean up any previously malformed toolchain files before building
  TOOLCHAIN_FILE="$PREBUILT_BUILD_DIR/$ABI/android-custom.toolchain.cmake"
  HOST_TOOLCHAIN_FILE="$PREBUILT_BUILD_DIR/$ABI/host-toolchain.cmake"
  
  # Check if they exist and fix or remove them if problematic
  if [ -f "$TOOLCHAIN_FILE" ]; then
    # Check if the file has any malformed content (empty lines causing syntax errors)
    if grep -q "endif()" "$TOOLCHAIN_FILE" && grep -q -P "endif\(\)\s*\n\s*\n" "$TOOLCHAIN_FILE"; then
      echo -e "${YELLOW}Found potentially problematic toolchain file, fixing...${NC}"
      # Fix the file by removing empty lines after endif()
      sed -i.bak 's/endif()[ \t]*$/endif()/g' "$TOOLCHAIN_FILE"
      rm -f "${TOOLCHAIN_FILE}.bak"
    fi
  fi
  
  if [ -f "$HOST_TOOLCHAIN_FILE" ]; then
    # Check if the file has any malformed content
    if grep -q -P "endif\(\)\s*\n\s*\n" "$HOST_TOOLCHAIN_FILE"; then
      echo -e "${YELLOW}Found potentially problematic host toolchain file, fixing...${NC}"
      # Fix the file by removing empty lines after endif()
      sed -i.bak 's/endif()[ \t]*$/endif()/g' "$HOST_TOOLCHAIN_FILE"
      rm -f "${HOST_TOOLCHAIN_FILE}.bak"
    fi
  fi
  
  build_for_abi "$ABI"
  if [ $? -ne 0 ]; then
    BUILD_SUCCESS=false
    FAILED_ABIS+=("$ABI")
    echo -e "${RED}Build for $ABI failed${NC}"
  else
    echo -e "${GREEN}Build for $ABI completed successfully${NC}"
  fi
done

# Final status
if [ "$BUILD_SUCCESS" = true ]; then
  echo -e "${GREEN}All builds completed successfully!${NC}"
  echo -e "${GREEN}Libraries are available in $ANDROID_JNI_DIR${NC}"
  echo -e "${GREEN}Headers are available in $ANDROID_CPP_DIR/include${NC}"
  
  # Also make sure to copy GPU libraries if they're available
  echo -e "${YELLOW}Checking for and copying additional GPU libraries...${NC}"
  
  # Check if OpenCL libraries were built in prebuilt/gpu directory
  for ABI in "${ABIS[@]}"; do
    if [ -d "$PREBUILT_GPU_DIR/$ABI" ] && [ -f "$PREBUILT_GPU_DIR/$ABI/libOpenCL.so" ]; then
      echo -e "${GREEN}Found OpenCL library for $ABI, ensuring it's copied${NC}"
      cp -f "$PREBUILT_GPU_DIR/$ABI/libOpenCL.so" "$ANDROID_JNI_DIR/$ABI/"
      touch "$ANDROID_JNI_DIR/$ABI/.opencl_enabled"
    fi
    
    # Also check for and copy dynamic backend libraries from prebuilt/gpu directory
    for backend_lib in libggml-vulkan.so libggml-opencl.so libggml-cpu.so; do
      if [ -f "$PREBUILT_GPU_DIR/$ABI/$backend_lib" ]; then
        echo -e "${GREEN}Found $backend_lib for $ABI in prebuilt/gpu, ensuring it's copied${NC}"
        cp -f "$PREBUILT_GPU_DIR/$ABI/$backend_lib" "$ANDROID_JNI_DIR/$ABI/"
        
        # Create appropriate flag files
        if [[ "$backend_lib" == "libggml-vulkan.so" ]]; then
          touch "$ANDROID_JNI_DIR/$ABI/.vulkan_enabled"
        elif [[ "$backend_lib" == "libggml-opencl.so" ]]; then
          touch "$ANDROID_JNI_DIR/$ABI/.opencl_enabled"
        fi
      fi
    done
  done
  
  # Final verification
  echo -e "${YELLOW}Final state of Android libraries:${NC}"
  for ABI in "${ABIS[@]}"; do
    echo -e "${GREEN}$ABI libraries:${NC}"
    ls -la "$ANDROID_JNI_DIR/$ABI/"
  done
  
  # Verify library architecture and capabilities
  echo -e "${YELLOW}Verifying library architectures and capabilities:${NC}"
  for ABI in "${ABIS[@]}"; do
    # Check architecture
    if [ -f "$ANDROID_JNI_DIR/$ABI/libllama.so" ]; then
      echo -e "${GREEN}Checking $ABI library architecture:${NC}"
      if [ "$ABI" = "arm64-v8a" ]; then
        file_output=$(file "$ANDROID_JNI_DIR/$ABI/libllama.so")
        if echo "$file_output" | grep -q "aarch64"; then
          echo -e "${GREEN}✓ arm64-v8a library is correctly built for aarch64 architecture${NC}"
        else
          echo -e "${RED}✗ arm64-v8a library is NOT built for aarch64 architecture: $file_output${NC}"
        fi
      elif [ "$ABI" = "x86_64" ]; then
        file_output=$(file "$ANDROID_JNI_DIR/$ABI/libllama.so")
        if echo "$file_output" | grep -q "x86-64"; then
          echo -e "${GREEN}✓ x86_64 library is correctly built for x86-64 architecture${NC}"
        else
          echo -e "${RED}✗ x86_64 library is NOT built for x86-64 architecture: $file_output${NC}"
        fi
      elif [ "$ABI" = "armeabi-v7a" ]; then
        file_output=$(file "$ANDROID_JNI_DIR/$ABI/libllama.so")
        if echo "$file_output" | grep -q -E "ARM|arm"; then
          echo -e "${GREEN}✓ armeabi-v7a library is correctly built for ARM architecture${NC}"
        else
          echo -e "${RED}✗ armeabi-v7a library is NOT built for ARM architecture: $file_output${NC}"
        fi
      elif [ "$ABI" = "x86" ]; then
        file_output=$(file "$ANDROID_JNI_DIR/$ABI/libllama.so")
        if echo "$file_output" | grep -q "80386"; then
          echo -e "${GREEN}✓ x86 library is correctly built for 80386 architecture${NC}"
        else
          echo -e "${RED}✗ x86 library is NOT built for 80386 architecture: $file_output${NC}"
        fi
      fi
      
      # Check GPU capabilities
      echo -e "${GREEN}Checking $ABI library GPU support:${NC}"
      
      # Check for OpenCL flag
      if [ -f "$ANDROID_JNI_DIR/$ABI/.opencl_enabled" ]; then
        echo -e "${GREEN}✓ OpenCL support enabled for $ABI${NC}"
      else
        echo -e "${YELLOW}⚠ OpenCL support not enabled for $ABI${NC}"
      fi
      
      # Check for Vulkan flag
      if [ -f "$ANDROID_JNI_DIR/$ABI/.vulkan_enabled" ]; then
        echo -e "${GREEN}✓ Vulkan support enabled for $ABI${NC}"
      else
        echo -e "${YELLOW}⚠ Vulkan support not enabled for $ABI${NC}"
      fi
      
      # Check for GPU symbols in the library
      if command -v nm &> /dev/null; then
        if nm -D "$ANDROID_JNI_DIR/$ABI/libllama.so" | grep -i "opencl" >/dev/null; then
          echo -e "${GREEN}✓ OpenCL symbols found in $ABI library${NC}"
        else
          echo -e "${YELLOW}⚠ No OpenCL symbols found in $ABI library - this is expected if GPU backend is dynamically loaded${NC}"
        fi
        
        if nm -D "$ANDROID_JNI_DIR/$ABI/libllama.so" | grep -i -E "vulkan|vk_|ggml.*vulkan" >/dev/null; then
          echo -e "${GREEN}✓ Vulkan symbols found in $ABI library${NC}"
        else
          echo -e "${YELLOW}⚠ No Vulkan symbols found in $ABI library - this is expected if GPU backend is dynamically loaded${NC}"
        fi
      else
        echo -e "${YELLOW}⚠ 'nm' command not available, skipping symbol check${NC}"
      fi
      
      # Check for GPU libraries
      if [ -f "$ANDROID_JNI_DIR/$ABI/libggml-opencl.so" ]; then
        echo -e "${GREEN}✓ Found separate OpenCL library for $ABI${NC}"
      fi
      
      if [ -f "$ANDROID_JNI_DIR/$ABI/libggml-vulkan.so" ]; then
        echo -e "${GREEN}✓ Found separate Vulkan library for $ABI${NC}"
      fi
    else
      echo -e "${RED}✗ Library not found for $ABI${NC}"
    fi
  done
else
  echo -e "${RED}Some builds failed:${NC}"
  for FAILED_ABI in "${FAILED_ABIS[@]}"; do
    echo -e "${RED}- $FAILED_ABI${NC}"
  done
  echo -e "${YELLOW}Check the build logs above for more details.${NC}"
  
  # Exit with error if all builds failed
  if [ ${#FAILED_ABIS[@]} -eq ${#ABIS[@]} ]; then
    echo -e "${RED}All builds failed.${NC}"
    exit 1
  fi
fi

# Final reminder for GPU backends
if [ "$BUILD_OPENCL" = true ] || [ "$BUILD_VULKAN" = true ]; then
  echo -e "${YELLOW}=== GPU Backend Support Information ===${NC}"
  
  if [ "$BUILD_OPENCL" = true ]; then
    if [ "$OPENCL_AVAILABLE" = true ]; then
      echo -e "${GREEN}✓ OpenCL support is enabled and libraries are available${NC}"
      # List ABIs with OpenCL enabled
      for ABI in "${ABIS[@]}"; do
        if [ -f "$ANDROID_JNI_DIR/$ABI/.opencl_enabled" ]; then
          echo -e "${GREEN}  - $ABI: OpenCL enabled${NC}"
        else
          echo -e "${YELLOW}  - $ABI: OpenCL not enabled${NC}"
        fi
      done
    else
      echo -e "${YELLOW}⚠ OpenCL was requested but libraries are not available${NC}"
      echo -e "${YELLOW}  Run build_android_gpu_backend.sh first to build OpenCL libraries${NC}"
    fi
  fi
  
  if [ "$BUILD_VULKAN" = true ]; then
    if [ "$VULKAN_AVAILABLE" = true ]; then
      echo -e "${GREEN}✓ Vulkan support is enabled and headers are available${NC}"
      # List ABIs with Vulkan enabled
      for ABI in "${ABIS[@]}"; do
        if [ -f "$ANDROID_JNI_DIR/$ABI/.vulkan_enabled" ]; then
          echo -e "${GREEN}  - $ABI: Vulkan enabled${NC}"
        else
          echo -e "${YELLOW}  - $ABI: Vulkan not enabled${NC}"
        fi
      done
      
      if [ -n "$GLSLC_PATH" ]; then
        echo -e "${GREEN}✓ Vulkan shader compiler (glslc) is available: $GLSLC_PATH${NC}"
      else
        echo -e "${YELLOW}⚠ Vulkan shader compiler (glslc) was not found${NC}"
        echo -e "${YELLOW}  This may cause shader compilation issues${NC}"
      fi
    else
      echo -e "${YELLOW}⚠ Vulkan was requested but headers are not available${NC}"
      echo -e "${YELLOW}  Run build_android_gpu_backend.sh first to get Vulkan headers${NC}"
    fi
  fi
fi


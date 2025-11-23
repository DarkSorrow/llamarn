#!/bin/bash
set -e

# Script to build GGML GPU backend libraries (libggml-opencl.so and libggml-vulkan.so)
# This script is used in CI/build pipeline to build GPU backends separately
# from the main llama.cpp libraries. The backends are built as dynamic libraries
# that will be loaded at runtime if GPU support is available on the device.

# Get the absolute path of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get project root directory (one level up from script dir)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Source the version information
. "$SCRIPT_DIR/used_version.sh"

# Use ANDROID_MIN_SDK from used_version.sh for consistent API level
if [ -z "$ANDROID_MIN_SDK" ]; then
  ANDROID_MIN_SDK=33  # Default from used_version.sh
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Print usage information
print_usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --help                 Print this help message"
  echo "  --abi=[all|arm64-v8a|x86_64|armeabi-v7a|x86]  Specify which ABI to build for (default: all)"
  echo "  --no-opencl            Disable OpenCL GPU backend"
  echo "  --no-vulkan            Disable Vulkan GPU backend"
  echo "  --debug                Build in debug mode"
  echo "  --clean                Clean previous builds before building"
  echo "  --ndk-path=[path]      Specify a custom path to the Android NDK"
}

# Default values
BUILD_ABI="all"
BUILD_OPENCL=true
BUILD_VULKAN=true
BUILD_TYPE="Release"
CLEAN_BUILD=false
CUSTOM_NDK_PATH=""

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
    --debug)
      BUILD_TYPE="Debug"
      ;;
    --clean)
      CLEAN_BUILD=true
      ;;
    --ndk-path=*)
      CUSTOM_NDK_PATH="${arg#*=}"
      ;;
    *)
      echo -e "${RED}Unknown option: $arg${NC}"
      print_usage
      exit 1
      ;;
  esac
done

# Set up paths
LLAMA_CPP_DIR="$PROJECT_ROOT/cpp/llama.cpp"
PREBUILT_DIR="$PROJECT_ROOT/prebuilt"
PREBUILT_BUILD_DIR="$PREBUILT_DIR/build"
PREBUILT_GPU_DIR="$PREBUILT_DIR/gpu"
PREBUILT_EXTERNAL_DIR="$PREBUILT_DIR/libs/external"
OPENCL_INCLUDE_DIR="$PREBUILT_EXTERNAL_DIR/opencl/include"
VULKAN_INCLUDE_DIR="$PREBUILT_EXTERNAL_DIR/vulkan/include"

# Try to use the user-provided NDK path first
if [ -n "$CUSTOM_NDK_PATH" ]; then
  NDK_PATH="$CUSTOM_NDK_PATH"
  echo -e "${GREEN}Using custom NDK path: $NDK_PATH${NC}"
  
  if [ ! -d "$NDK_PATH" ]; then
    echo -e "${RED}Custom NDK path not found at $NDK_PATH${NC}"
    exit 1
  fi
else
  # Try to find NDK from ANDROID_HOME
  if [ -z "$ANDROID_HOME" ]; then
    echo -e "${RED}ANDROID_HOME is not set${NC}"
    exit 1
  fi
  
  if [ -d "$ANDROID_HOME/ndk" ]; then
    NEWEST_NDK_VERSION=$(ls -1 "$ANDROID_HOME/ndk" | sort -rV | head -n 1)
    if [ -n "$NEWEST_NDK_VERSION" ]; then
      NDK_PATH="$ANDROID_HOME/ndk/$NEWEST_NDK_VERSION"
      echo -e "${GREEN}Found NDK version $NEWEST_NDK_VERSION${NC}"
    else
      NDK_PATH="$ANDROID_HOME/ndk/$NDK_VERSION"
    fi
  else
    NDK_PATH="$ANDROID_HOME/ndk/$NDK_VERSION"
  fi
  
  if [ ! -d "$NDK_PATH" ]; then
    echo -e "${RED}NDK not found at $NDK_PATH${NC}"
    exit 1
  fi
fi

# Detect host platform
HOST_OS=$(uname -s | tr '[:upper:]' '[:lower:]')
HOST_ARCH=$(uname -m)
if [ "$HOST_ARCH" = "x86_64" ]; then
  HOST_TAG="${HOST_OS}-x86_64"
else
  HOST_TAG="${HOST_OS}-${HOST_ARCH}"
fi

HOST_PLATFORM_DIR="$NDK_PATH/toolchains/llvm/prebuilt/$HOST_TAG"

# Extract Android platform version
# Use ANDROID_MIN_SDK from used_version.sh (API 33) for consistent builds and better Vulkan support
# vkGetPhysicalDeviceFeatures2 requires Vulkan 1.1+ (API 24+), but higher API levels have better support
ANDROID_PLATFORM="android-$ANDROID_MIN_SDK"

# Verify the platform exists in NDK
if [ -d "$NDK_PATH/platforms" ] && [ ! -d "$NDK_PATH/platforms/$ANDROID_PLATFORM" ]; then
  # Fallback to highest available if our target doesn't exist
  echo -e "${YELLOW}Platform $ANDROID_PLATFORM not found, checking available platforms...${NC}"
  HIGHEST_PLATFORM=$(ls -1 "$NDK_PATH/platforms" | sort -V | tail -n 1)
  if [ -n "$HIGHEST_PLATFORM" ]; then
    ANDROID_PLATFORM="$HIGHEST_PLATFORM"
    ANDROID_MIN_SDK=${ANDROID_PLATFORM#android-}
    echo -e "${YELLOW}Using highest available platform: $ANDROID_PLATFORM${NC}"
  fi
fi

# Ensure minimum API 24 for Vulkan 1.1+ features
if [ "$ANDROID_MIN_SDK" -lt 24 ]; then
  ANDROID_MIN_SDK=24
  ANDROID_PLATFORM="android-$ANDROID_MIN_SDK"
  echo -e "${YELLOW}Warning: Minimum API level 24 required for Vulkan, using $ANDROID_PLATFORM${NC}"
fi

echo -e "${GREEN}Using Android platform: $ANDROID_PLATFORM (API level $ANDROID_MIN_SDK)${NC}"

# Check if llama.cpp exists
if [ ! -d "$LLAMA_CPP_DIR" ]; then
  echo -e "${RED}llama.cpp repository not found at: $LLAMA_CPP_DIR${NC}"
  exit 1
fi

# Check for OpenCL headers
OPENCL_AVAILABLE=false
if [ "$BUILD_OPENCL" = true ]; then
  if [ -d "$OPENCL_INCLUDE_DIR/CL" ] && [ "$(ls -A $OPENCL_INCLUDE_DIR/CL 2>/dev/null)" ]; then
    OPENCL_AVAILABLE=true
    echo -e "${GREEN}OpenCL headers found${NC}"
  else
    echo -e "${YELLOW}OpenCL headers not found at $OPENCL_INCLUDE_DIR/CL${NC}"
    echo -e "${YELLOW}Run build_android_gpu_backend.sh first to prepare headers${NC}"
  fi
fi

# Check for Vulkan headers
VULKAN_AVAILABLE=false
if [ "$BUILD_VULKAN" = true ]; then
  if [ -d "$VULKAN_INCLUDE_DIR/vulkan" ] && [ "$(ls -A $VULKAN_INCLUDE_DIR/vulkan 2>/dev/null)" ]; then
    VULKAN_AVAILABLE=true
    echo -e "${GREEN}Vulkan headers found${NC}"
  else
    # Check NDK sysroot for Vulkan headers
    if [ -d "$HOST_PLATFORM_DIR/sysroot/usr/include/vulkan" ]; then
      VULKAN_AVAILABLE=true
      echo -e "${GREEN}Vulkan headers found in NDK sysroot${NC}"
    else
      echo -e "${YELLOW}Vulkan headers not found${NC}"
      echo -e "${YELLOW}Run build_android_gpu_backend.sh first to prepare headers${NC}"
    fi
  fi
fi

# Define ABIs to build
if [ "$BUILD_ABI" = "all" ]; then
  ABIS=("arm64-v8a" "x86_64" "armeabi-v7a" "x86")
elif [ "$BUILD_ABI" = "arm64-v8a" ] || [ "$BUILD_ABI" = "x86_64" ] || [ "$BUILD_ABI" = "armeabi-v7a" ] || [ "$BUILD_ABI" = "x86" ]; then
  ABIS=("$BUILD_ABI")
else
  echo -e "${RED}Invalid ABI: $BUILD_ABI${NC}"
  exit 1
fi

# Build function for each ABI
build_for_abi() {
  local ABI=$1
  echo -e "${GREEN}Building GGML GPU backends for $ABI${NC}"
  
  # Set ABI-specific flags
  if [ "$ABI" = "arm64-v8a" ]; then
    ANDROID_ABI="arm64-v8a"
  elif [ "$ABI" = "x86_64" ]; then
    ANDROID_ABI="x86_64"
  elif [ "$ABI" = "armeabi-v7a" ]; then
    ANDROID_ABI="armeabi-v7a"
  elif [ "$ABI" = "x86" ]; then
    ANDROID_ABI="x86"
  fi
  
  # Skip GPU backend build for 32-bit ABIs (no reliable OpenCL/Vulkan support)
  if [ "$ANDROID_ABI" = "armeabi-v7a" ] || [ "$ANDROID_ABI" = "x86" ]; then
    echo -e "${YELLOW}Skipping GPU backend build for 32-bit ABI $ANDROID_ABI${NC}"
    mkdir -p "$PREBUILT_GPU_DIR/$ABI"
    touch "$PREBUILT_GPU_DIR/$ABI/.gpu_skipped"
    return 0
  fi
  
  # Create build directory
  BUILD_DIR="$PREBUILT_BUILD_DIR/gpu-backends-$ABI"
  if [ "$CLEAN_BUILD" = true ]; then
    rm -rf "$BUILD_DIR"
  fi
  mkdir -p "$BUILD_DIR"
  
  pushd "$BUILD_DIR"
  
  # Configure CMake for GPU backends only
  CMAKE_ARGS=(
    -DCMAKE_TOOLCHAIN_FILE="$NDK_PATH/build/cmake/android.toolchain.cmake"
    -DANDROID_ABI="$ANDROID_ABI"
    -DANDROID_PLATFORM="$ANDROID_PLATFORM"
    -DANDROID_STL=c++_shared
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/install"
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$BUILD_DIR/lib"
    -DBUILD_SHARED_LIBS=ON
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_EXAMPLES=OFF
    -DLLAMA_BUILD_SERVER=OFF
    -DLLAMA_CURL=OFF
    -DGGML_BACKEND_DL=ON          # Enable dynamic loading - backends as separate .so files
    -DGGML_CPU=OFF                # Don't build CPU backend here (built in main libs)
    -DGGML_METAL=OFF
    -DGGML_CUDA=OFF
    -DGGML_HIP=OFF
    -DGGML_OPENCL=OFF             # Will be enabled conditionally
    -DGGML_VULKAN=OFF              # Will be enabled conditionally
    # Help CMake find libraries in NDK sysroot
    -DCMAKE_FIND_ROOT_PATH="$HOST_PLATFORM_DIR/sysroot"
    -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=BOTH
    -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH
  )
  
  # Add OpenCL if available and enabled
  GPU_CMAKE_FLAGS=()
  if [ "$BUILD_OPENCL" = true ] && [ "$OPENCL_AVAILABLE" = true ]; then
    CMAKE_ARGS+=(-DGGML_OPENCL=ON)
    GPU_CMAKE_FLAGS+=(-DCL_INCLUDE_DIR="$OPENCL_INCLUDE_DIR")
    
    # Help CMake find OpenCL library in NDK sysroot (installed by build_android_gpu_backend.sh)
    # Set architecture-specific library path
    if [ "$ABI" = "arm64-v8a" ]; then
      ARCH="aarch64"
    elif [ "$ABI" = "x86_64" ]; then
      ARCH="x86_64"
    elif [ "$ABI" = "armeabi-v7a" ]; then
      ARCH="arm"
    elif [ "$ABI" = "x86" ]; then
      ARCH="i686"
    fi
    
    # Try multiple possible locations for libOpenCL.so
    OPENCL_LIB_PATH=""
    
    # First try: Direct in arch directory (older NDK structure)
    TRY_PATH="$HOST_PLATFORM_DIR/sysroot/usr/lib/$ARCH-linux-android/libOpenCL.so"
    if [ -f "$TRY_PATH" ]; then
      OPENCL_LIB_PATH="$TRY_PATH"
      echo -e "${GREEN}Found OpenCL library at: $OPENCL_LIB_PATH${NC}"
    fi
    
    # Second try: API level subdirectory (newer NDK structure)
    if [ -z "$OPENCL_LIB_PATH" ]; then
      TRY_PATH="$HOST_PLATFORM_DIR/sysroot/usr/lib/$ARCH-linux-android/$ANDROID_MIN_SDK/libOpenCL.so"
      if [ -f "$TRY_PATH" ]; then
        OPENCL_LIB_PATH="$TRY_PATH"
        echo -e "${GREEN}Found OpenCL library at: $OPENCL_LIB_PATH${NC}"
      fi
    fi
    
    # Third try: Search for any libOpenCL.so in the arch directory
    if [ -z "$OPENCL_LIB_PATH" ]; then
      FOUND_LIB=$(find "$HOST_PLATFORM_DIR/sysroot/usr/lib/$ARCH-linux-android" -name "libOpenCL.so" 2>/dev/null | head -1)
      if [ -n "$FOUND_LIB" ] && [ -f "$FOUND_LIB" ]; then
        OPENCL_LIB_PATH="$FOUND_LIB"
        echo -e "${GREEN}Found OpenCL library at: $OPENCL_LIB_PATH${NC}"
      fi
    fi
    
    if [ -n "$OPENCL_LIB_PATH" ] && [ -f "$OPENCL_LIB_PATH" ]; then
      # Set all possible CMake variables that FindOpenCL might look for
      # FindOpenCL.cmake looks for OpenCL_LIBRARY specifically
      GPU_CMAKE_FLAGS+=(-DOpenCL_LIBRARY="$OPENCL_LIB_PATH")
      GPU_CMAKE_FLAGS+=(-DCL_LIBRARY="$OPENCL_LIB_PATH")
      GPU_CMAKE_FLAGS+=(-DOpenCL_INCLUDE_DIR="$OPENCL_INCLUDE_DIR")
      GPU_CMAKE_FLAGS+=(-DCL_INCLUDE_DIR="$OPENCL_INCLUDE_DIR")
      # Also set as cache variable to help FindOpenCL
      GPU_CMAKE_FLAGS+=(-DOpenCL_LIBRARY:FILEPATH="$OPENCL_LIB_PATH")
      echo -e "${GREEN}Configured OpenCL library: $OPENCL_LIB_PATH${NC}"
      echo -e "${GREEN}Configured OpenCL headers: $OPENCL_INCLUDE_DIR${NC}"
    else
      echo -e "${RED}ERROR: OpenCL library not found${NC}"
      echo -e "${YELLOW}Make sure build_android_gpu_backend.sh was run first to build the ICD loader${NC}"
      echo -e "${YELLOW}Searched locations:${NC}"
      echo -e "${YELLOW}  1. $HOST_PLATFORM_DIR/sysroot/usr/lib/$ARCH-linux-android/libOpenCL.so${NC}"
      echo -e "${YELLOW}  2. $HOST_PLATFORM_DIR/sysroot/usr/lib/$ARCH-linux-android/$ANDROID_MIN_SDK/libOpenCL.so${NC}"
      # List what's actually in the directory
      if [ -d "$HOST_PLATFORM_DIR/sysroot/usr/lib/$ARCH-linux-android" ]; then
        echo -e "${YELLOW}Files in arch directory:${NC}"
        ls -la "$HOST_PLATFORM_DIR/sysroot/usr/lib/$ARCH-linux-android/" | head -10
      fi
      return 1
    fi
    
    echo -e "${GREEN}OpenCL backend enabled for $ABI${NC}"
  fi
  
  # Add Vulkan if available and enabled
  if [ "$BUILD_VULKAN" = true ] && [ "$VULKAN_AVAILABLE" = true ]; then
    CMAKE_ARGS+=(-DGGML_VULKAN=ON)
    CMAKE_ARGS+=(-DGGML_VULKAN_CHECK_RESULTS=OFF)
    CMAKE_ARGS+=(-DGGML_VULKAN_DEBUG=OFF)
    CMAKE_ARGS+=(-DVK_USE_PLATFORM_ANDROID_KHR=ON)
    CMAKE_ARGS+=(-DGGML_VULKAN_DISABLE_FLASHATTN=ON)
    
    # Use Vulkan headers from prebuilt or NDK
    if [ -d "$VULKAN_INCLUDE_DIR/vulkan" ]; then
      GPU_CMAKE_FLAGS+=(-DVulkan_INCLUDE_DIR="$VULKAN_INCLUDE_DIR")
    elif [ -d "$HOST_PLATFORM_DIR/sysroot/usr/include" ]; then
      GPU_CMAKE_FLAGS+=(-DVulkan_INCLUDE_DIR="$HOST_PLATFORM_DIR/sysroot/usr/include")
    fi
    
    # Explicitly set Vulkan library path to use the correct API level
    # Use the highest available API level for Vulkan (vkGetPhysicalDeviceFeatures2 requires Vulkan 1.1+)
    VULKAN_LIB_PATH="$HOST_PLATFORM_DIR/sysroot/usr/lib/$ARCH-linux-android/$ANDROID_MIN_SDK/libvulkan.so"
    if [ -f "$VULKAN_LIB_PATH" ]; then
      GPU_CMAKE_FLAGS+=(-DVulkan_LIBRARY="$VULKAN_LIB_PATH")
      echo -e "${GREEN}Using Vulkan library from NDK sysroot: $VULKAN_LIB_PATH${NC}"
    else
      # Fallback: try to find any libvulkan.so in the sysroot
      VULKAN_LIB_FALLBACK=$(find "$HOST_PLATFORM_DIR/sysroot/usr/lib/$ARCH-linux-android" -name "libvulkan.so" 2>/dev/null | sort -V | tail -1)
      if [ -n "$VULKAN_LIB_FALLBACK" ] && [ -f "$VULKAN_LIB_FALLBACK" ]; then
        GPU_CMAKE_FLAGS+=(-DVulkan_LIBRARY="$VULKAN_LIB_FALLBACK")
        echo -e "${GREEN}Using Vulkan library (fallback): $VULKAN_LIB_FALLBACK${NC}"
      else
        echo -e "${YELLOW}Warning: Vulkan library not found, CMake will try to find it${NC}"
        echo -e "${YELLOW}  Searched: $VULKAN_LIB_PATH${NC}"
      fi
    fi
    
    # Find glslc
    GLSLC_PATH="$NDK_PATH/shader-tools/$HOST_TAG/glslc"
    if [ -f "$GLSLC_PATH" ] && [ -x "$GLSLC_PATH" ]; then
      GPU_CMAKE_FLAGS+=(-DVulkan_GLSLC_EXECUTABLE="$GLSLC_PATH")
    fi
    
    echo -e "${GREEN}Vulkan backend enabled for $ABI${NC}"
  fi
  
  # Configure
  echo -e "${YELLOW}Configuring CMake for $ABI...${NC}"
  cmake "$LLAMA_CPP_DIR" "${CMAKE_ARGS[@]}" "${GPU_CMAKE_FLAGS[@]}" || {
    echo -e "${RED}CMake configuration failed for $ABI${NC}"
    popd
    return 1
  }
  
  # Build only the GPU backend targets
  echo -e "${YELLOW}Building GPU backends for $ABI...${NC}"
  TARGETS=()
  if [ "$BUILD_OPENCL" = true ] && [ "$OPENCL_AVAILABLE" = true ]; then
    TARGETS+=(ggml-opencl)
  fi
  if [ "$BUILD_VULKAN" = true ] && [ "$VULKAN_AVAILABLE" = true ]; then
    TARGETS+=(ggml-vulkan)
  fi
  
  if [ ${#TARGETS[@]} -eq 0 ]; then
    echo -e "${YELLOW}No GPU backends to build for $ABI${NC}"
    popd
    return 0
  fi
  
  cmake --build . --target "${TARGETS[@]}" --config "$BUILD_TYPE" -j$(nproc) || {
    echo -e "${RED}Build failed for $ABI${NC}"
    popd
    return 1
  }
  
  # Copy built libraries to prebuilt/gpu directory
  mkdir -p "$PREBUILT_GPU_DIR/$ABI"
  
  for backend_lib in libggml-opencl.so libggml-vulkan.so; do
    if [ -f "$BUILD_DIR/bin/$backend_lib" ]; then
      cp "$BUILD_DIR/bin/$backend_lib" "$PREBUILT_GPU_DIR/$ABI/"
      echo -e "${GREEN}Copied $backend_lib to prebuilt/gpu/$ABI/${NC}"
      # Create flag files
      if [[ "$backend_lib" == "libggml-opencl.so" ]]; then
        touch "$PREBUILT_GPU_DIR/$ABI/.opencl_enabled"
      elif [[ "$backend_lib" == "libggml-vulkan.so" ]]; then
        touch "$PREBUILT_GPU_DIR/$ABI/.vulkan_enabled"
      fi
    elif [ -f "$BUILD_DIR/lib/$backend_lib" ]; then
      cp "$BUILD_DIR/lib/$backend_lib" "$PREBUILT_GPU_DIR/$ABI/"
      echo -e "${GREEN}Copied $backend_lib to prebuilt/gpu/$ABI/${NC}"
      # Create flag files
      if [[ "$backend_lib" == "libggml-opencl.so" ]]; then
        touch "$PREBUILT_GPU_DIR/$ABI/.opencl_enabled"
      elif [[ "$backend_lib" == "libggml-vulkan.so" ]]; then
        touch "$PREBUILT_GPU_DIR/$ABI/.vulkan_enabled"
      fi
    fi
  done
  
  popd
  
  echo -e "${GREEN}Successfully built GPU backends for $ABI${NC}"
  return 0
}

# Build for each ABI
BUILD_SUCCESS=true
FAILED_ABIS=()

for ABI in "${ABIS[@]}"; do
  build_for_abi "$ABI"
  if [ $? -ne 0 ]; then
    BUILD_SUCCESS=false
    FAILED_ABIS+=("$ABI")
  fi
done

# Final status
if [ "$BUILD_SUCCESS" = true ]; then
  echo -e "${GREEN}=== GGML GPU backend build complete ===${NC}"
  echo -e "${GREEN}Libraries are available in $PREBUILT_GPU_DIR${NC}"
  
  # List built libraries
  for ABI in "${ABIS[@]}"; do
    if [ -d "$PREBUILT_GPU_DIR/$ABI" ]; then
      echo -e "${GREEN}$ABI libraries:${NC}"
      ls -la "$PREBUILT_GPU_DIR/$ABI/" | grep -E "libggml-.*\.so" || echo "  (none)"
    fi
  done
  
  exit 0
else
  echo -e "${RED}Build failed for ABIs: ${FAILED_ABIS[*]}${NC}"
  exit 1
fi


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

# Print usage information
print_usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --help                 Print this help message"
  echo "  --abi=[all|arm64-v8a|x86_64]  Specify which ABI to build for (default: all)"
  echo "  --no-opencl            Disable OpenCL GPU acceleration"
  echo "  --no-vulkan            Disable Vulkan GPU acceleration"
  echo "  --debug                Build in debug mode"
  echo "  --clean                Clean previous builds before building"
  echo "  --ndk-path=[path]      Specify a custom path to the Android NDK"
  echo "  --cmake-flags=[flags]  Additional CMake flags to pass to the build"
}

# Default values
BUILD_ABI="all"
BUILD_OPENCL=true
BUILD_VULKAN=true
BUILD_TYPE="Release"
CLEAN_BUILD=false
CUSTOM_NDK_PATH=""
CUSTOM_CMAKE_FLAGS=""

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

# Define directories
ANDROID_DIR="$PROJECT_ROOT/android"
ANDROID_JNI_DIR="$ANDROID_DIR/src/main/jniLibs"
CPP_DIR="$PROJECT_ROOT/cpp"
LLAMA_CPP_DIR="$CPP_DIR/llama.cpp"
BUILD_DIR="$PROJECT_ROOT/build-android"

# Clean up if requested
if [ "$CLEAN_BUILD" = true ]; then
  echo -e "${YELLOW}Cleaning build artifacts...${NC}"
  rm -rf "$BUILD_DIR"
  rm -rf "$ANDROID_JNI_DIR"
fi

# Create necessary directories
mkdir -p "$BUILD_DIR"
mkdir -p "$ANDROID_JNI_DIR"

# Detect OS and cores
if [[ "$OSTYPE" == "darwin"* ]]; then
  N_CORES=$(sysctl -n hw.ncpu)
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
  N_CORES=$(nproc)
else
  N_CORES=4
fi

echo -e "${YELLOW}Using $N_CORES cores for building${NC}"

# Setup NDK path
if [ -n "$CUSTOM_NDK_PATH" ]; then
  NDK_PATH="$CUSTOM_NDK_PATH"
else
  if [ -z "$ANDROID_HOME" ]; then
    if [ -n "$ANDROID_SDK_ROOT" ]; then
      ANDROID_HOME="$ANDROID_SDK_ROOT"
    elif [ -d "$HOME/Android/Sdk" ]; then
      ANDROID_HOME="$HOME/Android/Sdk"
    elif [ -d "$HOME/Library/Android/sdk" ]; then
      ANDROID_HOME="$HOME/Library/Android/sdk"
    else
      echo -e "${RED}Android SDK not found. Please set ANDROID_HOME or use --ndk-path${NC}"
      exit 1
    fi
  fi
  
  # Find NDK
  if [ -d "$ANDROID_HOME/ndk" ]; then
    NEWEST_NDK_VERSION=$(ls -1 "$ANDROID_HOME/ndk" | sort -rV | head -n 1)
    NDK_PATH="$ANDROID_HOME/ndk/$NEWEST_NDK_VERSION"
  elif [ -d "$ANDROID_HOME/ndk-bundle" ]; then
    NDK_PATH="$ANDROID_HOME/ndk-bundle"
  else
    echo -e "${RED}NDK not found. Please install Android NDK${NC}"
    exit 1
  fi
fi

if [ ! -d "$NDK_PATH" ]; then
  echo -e "${RED}NDK path not found: $NDK_PATH${NC}"
  exit 1
fi

echo -e "${GREEN}Using NDK: $NDK_PATH${NC}"

# Detect Android platform
HOST_TAG_DIR=$(ls -1 "$NDK_PATH/toolchains/llvm/prebuilt/" | head -n 1)
HOST_PLATFORM_DIR="$NDK_PATH/toolchains/llvm/prebuilt/$HOST_TAG_DIR"

if [ -d "$HOST_PLATFORM_DIR/sysroot/usr/lib/aarch64-linux-android" ]; then
  API_LEVELS=$(find "$HOST_PLATFORM_DIR/sysroot/usr/lib/aarch64-linux-android" -maxdepth 1 -type d -name "[0-9]*" 2>/dev/null | sort -V)
  if [ -n "$API_LEVELS" ]; then
    HIGHEST_API=$(basename $(echo "$API_LEVELS" | tail -n 1))
    ANDROID_MIN_SDK=$HIGHEST_API
  else
    ANDROID_MIN_SDK=24
  fi
else
  ANDROID_MIN_SDK=24
fi

ANDROID_PLATFORM="android-$ANDROID_MIN_SDK"
echo -e "${GREEN}Using Android platform: $ANDROID_PLATFORM${NC}"

# Check llama.cpp
if [ ! -d "$LLAMA_CPP_DIR" ] || [ ! -f "$LLAMA_CPP_DIR/CMakeLists.txt" ]; then
  echo -e "${RED}llama.cpp not found at: $LLAMA_CPP_DIR${NC}"
  echo -e "${YELLOW}Please run setupLlamaCpp.sh init first${NC}"
  exit 1
fi

# Check for glslc (for Vulkan)
GLSLC_PATH=""
if [ "$BUILD_VULKAN" = true ]; then
  NDK_GLSLC="$NDK_PATH/shader-tools/$HOST_TAG_DIR/glslc"
  if [ -f "$NDK_GLSLC" ]; then
    GLSLC_PATH="$NDK_GLSLC"
    echo -e "${GREEN}Found glslc compiler: $GLSLC_PATH${NC}"
  else
    SYS_GLSLC=$(which glslc 2>/dev/null || echo "")
    if [ -n "$SYS_GLSLC" ]; then
      GLSLC_PATH="$SYS_GLSLC"
      echo -e "${GREEN}Found glslc in system PATH: $GLSLC_PATH${NC}"
    else
      echo -e "${YELLOW}Warning: glslc not found, Vulkan shaders may not compile${NC}"
    fi
  fi
fi

# Define ABIs to build
if [ "$BUILD_ABI" = "all" ]; then
  ABIS=("arm64-v8a" "x86_64")
elif [ "$BUILD_ABI" = "arm64-v8a" ] || [ "$BUILD_ABI" = "x86_64" ]; then
  ABIS=("$BUILD_ABI")
else
  echo -e "${RED}Invalid ABI: $BUILD_ABI. Supported: all, arm64-v8a, x86_64${NC}"
  exit 1
fi

# Build function for each ABI
build_for_abi() {
  local ABI=$1
  echo -e "${GREEN}Building for $ABI...${NC}"
  
  local ABI_BUILD_DIR="$BUILD_DIR/$ABI"
  mkdir -p "$ABI_BUILD_DIR"
  mkdir -p "$ANDROID_JNI_DIR/$ABI"
  
  # Base CMake arguments
  local CMAKE_ARGS=(
    -DCMAKE_TOOLCHAIN_FILE="$NDK_PATH/build/cmake/android.toolchain.cmake"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DANDROID_ABI="$ABI"
    -DANDROID_PLATFORM="$ANDROID_PLATFORM"
    -DBUILD_SHARED_LIBS=ON
    -DLLAMA_BUILD_SERVER=OFF
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_EXAMPLES=OFF
    -DGGML_OPENMP=OFF
    -DGGML_NATIVE=OFF
    -DGGML_USE_K_QUANTS=ON
    -DLLAMA_CURL=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations"
    # Enable dynamic backend loading
    -DGGML_BACKEND_DL=ON
    # Enable CPU backend (always available)
    -DGGML_CPU=ON
    # Disable other backends by default
    -DGGML_METAL=OFF
    -DGGML_CUDA=OFF
    -DGGML_HIP=OFF
    # CPU optimizations (disable x86 specific ones for ARM)
    -DGGML_SSE42=OFF
    -DGGML_AVX=OFF
    -DGGML_AVX2=OFF
    -DGGML_FMA=OFF
    -DGGML_F16C=OFF
  )
  
  # Add GPU backends if enabled
  if [ "$BUILD_OPENCL" = true ]; then
    CMAKE_ARGS+=(-DGGML_OPENCL=ON)
    echo -e "${GREEN}Enabling OpenCL for $ABI${NC}"
  else
    CMAKE_ARGS+=(-DGGML_OPENCL=OFF)
  fi
  
  if [ "$BUILD_VULKAN" = true ]; then
    CMAKE_ARGS+=(
      -DGGML_VULKAN=ON
      -DGGML_VULKAN_CHECK_RESULTS=OFF
      -DGGML_VULKAN_DEBUG=OFF
      -DGGML_VULKAN_MEMORY_DEBUG=OFF
      -DGGML_VULKAN_VALIDATE=OFF
      -DGGML_VULKAN_RUN_TESTS=OFF
      -DVK_USE_PLATFORM_ANDROID_KHR=ON
      -DGGML_VULKAN_DISABLE_FLASHATTN=ON
    )
    
    # Add glslc if available
    if [ -n "$GLSLC_PATH" ]; then
      CMAKE_ARGS+=(-DVulkan_GLSLC_EXECUTABLE="$GLSLC_PATH")
    fi
    
    # Set Vulkan library path
    if [ "$ABI" = "arm64-v8a" ]; then
      VULKAN_LIB_PATH="$HOST_PLATFORM_DIR/sysroot/usr/lib/aarch64-linux-android/$ANDROID_MIN_SDK/libvulkan.so"
    else
      VULKAN_LIB_PATH="$HOST_PLATFORM_DIR/sysroot/usr/lib/x86_64-linux-android/$ANDROID_MIN_SDK/libvulkan.so"
    fi
    
    if [ -f "$VULKAN_LIB_PATH" ]; then
      CMAKE_ARGS+=(-DVulkan_LIBRARY="$VULKAN_LIB_PATH")
    fi
    
    # Set Vulkan include path - prefer system headers if available (for vulkan.hpp)
    if [ -f "/usr/include/vulkan/vulkan.hpp" ]; then
      CMAKE_ARGS+=(-DVulkan_INCLUDE_DIR="/usr/include")
      echo -e "${GREEN}Using system Vulkan headers for $ABI${NC}"
    else
      CMAKE_ARGS+=(-DVulkan_INCLUDE_DIR="$HOST_PLATFORM_DIR/sysroot/usr/include")
      echo -e "${YELLOW}Using NDK Vulkan headers for $ABI (may lack vulkan.hpp)${NC}"
    fi
    
    echo -e "${GREEN}Enabling Vulkan for $ABI${NC}"
  else
    CMAKE_ARGS+=(-DGGML_VULKAN=OFF)
  fi
  
  # Add custom flags if provided
  if [ -n "$CUSTOM_CMAKE_FLAGS" ]; then
    CMAKE_ARGS+=($CUSTOM_CMAKE_FLAGS)
  fi
  
  # Configure
  pushd "$ABI_BUILD_DIR"
  echo -e "${YELLOW}Configuring build for $ABI...${NC}"
  cmake "$LLAMA_CPP_DIR" "${CMAKE_ARGS[@]}" || {
    echo -e "${RED}CMake configuration failed for $ABI${NC}"
    popd
    return 1
  }
  
  # Build
  echo -e "${YELLOW}Building libraries for $ABI...${NC}"
  cmake --build . --config "$BUILD_TYPE" -j$N_CORES || {
    echo -e "${RED}Build failed for $ABI${NC}"
    popd
    return 1
  }
  
  popd
  
  # Copy libraries to jniLibs
  echo -e "${YELLOW}Copying libraries for $ABI...${NC}"
  
  # Main libraries (required)
  for lib in libllama.so libggml.so libggml-base.so libggml-cpu.so; do
    if [ -f "$ABI_BUILD_DIR/bin/$lib" ]; then
      cp "$ABI_BUILD_DIR/bin/$lib" "$ANDROID_JNI_DIR/$ABI/"
      echo -e "${GREEN}Copied $lib for $ABI${NC}"
    elif [ -f "$ABI_BUILD_DIR/$lib" ]; then
      cp "$ABI_BUILD_DIR/$lib" "$ANDROID_JNI_DIR/$ABI/"
      echo -e "${GREEN}Copied $lib for $ABI${NC}"
    else
      echo -e "${YELLOW}Warning: $lib not found for $ABI${NC}"
    fi
  done
  
  # GPU backend libraries (optional)
  if [ "$BUILD_OPENCL" = true ]; then
    for lib in libggml-opencl.so libOpenCL.so; do
      if [ -f "$ABI_BUILD_DIR/bin/$lib" ]; then
        cp "$ABI_BUILD_DIR/bin/$lib" "$ANDROID_JNI_DIR/$ABI/"
        echo -e "${GREEN}Copied OpenCL $lib for $ABI${NC}"
      elif [ -f "$ABI_BUILD_DIR/$lib" ]; then
        cp "$ABI_BUILD_DIR/$lib" "$ANDROID_JNI_DIR/$ABI/"
        echo -e "${GREEN}Copied OpenCL $lib for $ABI${NC}"
      fi
    done
  fi
  
  if [ "$BUILD_VULKAN" = true ]; then
    if [ -f "$ABI_BUILD_DIR/bin/libggml-vulkan.so" ]; then
      cp "$ABI_BUILD_DIR/bin/libggml-vulkan.so" "$ANDROID_JNI_DIR/$ABI/"
      echo -e "${GREEN}Copied Vulkan library for $ABI${NC}"
    elif [ -f "$ABI_BUILD_DIR/libggml-vulkan.so" ]; then
      cp "$ABI_BUILD_DIR/libggml-vulkan.so" "$ANDROID_JNI_DIR/$ABI/"
      echo -e "${GREEN}Copied Vulkan library for $ABI${NC}"
    else
      echo -e "${YELLOW}Vulkan library not found for $ABI${NC}"
    fi
  fi
  
  echo -e "${GREEN}Build completed for $ABI${NC}"
  return 0
}

# Build for all ABIs
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
  echo -e "${GREEN}=== BUILD COMPLETED SUCCESSFULLY ===${NC}"
  echo -e "${GREEN}All libraries built and copied to: $ANDROID_JNI_DIR${NC}"
  
  # Show what was built
  echo -e "${YELLOW}=== BUILT LIBRARIES ===${NC}"
  for ABI in "${ABIS[@]}"; do
    echo -e "${GREEN}$ABI:${NC}"
    ls -la "$ANDROID_JNI_DIR/$ABI/"
    
    # Show GPU support status based on library existence
    if [ -f "$ANDROID_JNI_DIR/$ABI/libggml-opencl.so" ]; then
      echo -e "${GREEN}  ✓ OpenCL support available${NC}"
    fi
    if [ -f "$ANDROID_JNI_DIR/$ABI/libggml-vulkan.so" ]; then
      echo -e "${GREEN}  ✓ Vulkan support available${NC}"
    fi
    echo ""
  done
  
  echo -e "${GREEN}Libraries are ready for Android CMake build!${NC}"
else
  echo -e "${RED}Some builds failed: ${FAILED_ABIS[*]}${NC}"
  exit 1
fi


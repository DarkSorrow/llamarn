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
      echo -e "${RED}Unknown argument: $arg${NC}"
      print_usage
      exit 1
      ;;
  esac
done

# Define prebuilt directory for all intermediary files
PREBUILT_DIR="$PROJECT_ROOT/prebuilt"
PREBUILT_GPU_DIR="$PREBUILT_DIR/gpu"
PREBUILT_EXTERNAL_DIR="$PREBUILT_DIR/libs/external"

# Define directories
THIRD_PARTY_DIR="$PREBUILT_DIR/third_party"
OPENCL_HEADERS_DIR="$THIRD_PARTY_DIR/OpenCL-Headers"
OPENCL_LOADER_DIR="$THIRD_PARTY_DIR/OpenCL-ICD-Loader"
OPENCL_INCLUDE_DIR="$PREBUILT_EXTERNAL_DIR/opencl/include"
OPENCL_LIB_DIR="$PREBUILT_EXTERNAL_DIR/opencl/lib"
VULKAN_HEADERS_DIR="$THIRD_PARTY_DIR/Vulkan-Headers"
VULKAN_INCLUDE_DIR="$PREBUILT_EXTERNAL_DIR/vulkan/include"

# Clean up if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo -e "${YELLOW}Cleaning previous build artifacts...${NC}"
    rm -rf "$PREBUILT_GPU_DIR"
    rm -rf "$OPENCL_HEADERS_DIR"
    rm -rf "$OPENCL_LOADER_DIR"
    rm -rf "$VULKAN_HEADERS_DIR"
    rm -rf "$OPENCL_INCLUDE_DIR"
    rm -rf "$OPENCL_LIB_DIR"
    rm -rf "$VULKAN_INCLUDE_DIR"
fi

# Create necessary directories
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p "$PREBUILT_DIR"
mkdir -p "$PREBUILT_GPU_DIR"
mkdir -p "$PREBUILT_GPU_DIR/arm64-v8a"
mkdir -p "$PREBUILT_GPU_DIR/x86_64"
mkdir -p "$PREBUILT_EXTERNAL_DIR"
mkdir -p "$THIRD_PARTY_DIR"
mkdir -p "$OPENCL_INCLUDE_DIR"
mkdir -p "$OPENCL_LIB_DIR"
mkdir -p "$VULKAN_INCLUDE_DIR"

# Determine platform and setup environment
if [[ "$OSTYPE" == "darwin"* ]]; then
  echo -e "${YELLOW}Building on macOS${NC}"
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
  # Try to detect from the LLVM toolchain
  HOST_TAG=$(ls "$NDK_PATH/toolchains/llvm/prebuilt/")
  PLATFORMS_DIR="$NDK_PATH/toolchains/llvm/prebuilt/$HOST_TAG/sysroot/usr/lib/aarch64-linux-android"
  
  if [ -d "$PLATFORMS_DIR" ]; then
    # Get the highest API level available
    ANDROID_MIN_SDK=$(ls -1 "$PLATFORMS_DIR" | grep -E '^[0-9]+$' | sort -rn | head -n 1)
    ANDROID_PLATFORM="android-$ANDROID_MIN_SDK"
    echo -e "${GREEN}Using Android platform: $ANDROID_PLATFORM (API level $ANDROID_MIN_SDK)${NC}"
  else
    # Default to API level 24 (Android 7.0) if we can't detect
    ANDROID_MIN_SDK=24
    ANDROID_PLATFORM="android-$ANDROID_MIN_SDK"
    echo -e "${YELLOW}Could not detect Android platform, defaulting to $ANDROID_PLATFORM (API level $ANDROID_MIN_SDK)${NC}"
  fi
else
  # Default to API level 24 (Android 7.0)
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

# Define the host platform dir
HOST_PLATFORM_DIR="$NDK_PATH/toolchains/llvm/prebuilt/$HOST_TAG"

# Define ABIS to build
ABIS=()
if [ "$BUILD_ABI" = "all" ]; then
  ABIS=("arm64-v8a" "x86_64")
elif [ "$BUILD_ABI" = "arm64-v8a" ] || [ "$BUILD_ABI" = "x86_64" ]; then
  ABIS=("$BUILD_ABI")
else
  echo -e "${RED}Invalid ABI: $BUILD_ABI. Supported ABIs are: all, arm64-v8a, x86_64${NC}"
  exit 1
fi

# Build OpenCL if enabled
if [ "$BUILD_OPENCL" = true ]; then
  echo -e "${GREEN}=== Building OpenCL libraries ===${NC}"
  
  # Get OpenCL Headers
  if [ ! -d "$OPENCL_HEADERS_DIR" ]; then
    echo -e "${YELLOW}Cloning OpenCL-Headers...${NC}"
    git clone https://github.com/KhronosGroup/OpenCL-Headers "$OPENCL_HEADERS_DIR"
    # Copy headers to NDK include directory
    echo -e "${YELLOW}Installing OpenCL headers to NDK...${NC}"
    mkdir -p "$HOST_PLATFORM_DIR/sysroot/usr/include/CL"
    cp -r "$OPENCL_HEADERS_DIR/CL/"* "$HOST_PLATFORM_DIR/sysroot/usr/include/CL/"
  else
    echo -e "${YELLOW}OpenCL-Headers already cloned, using existing copy${NC}"
  fi
  
  # Install OpenCL headers to our prebuilt dir
  mkdir -p "$OPENCL_INCLUDE_DIR/CL"
  cp -r "$OPENCL_HEADERS_DIR/CL/"* "$OPENCL_INCLUDE_DIR/CL/"
  
  # Build OpenCL ICD Loader for each ABI
  if [ ! -d "$OPENCL_LOADER_DIR" ]; then
    echo -e "${YELLOW}Cloning OpenCL-ICD-Loader...${NC}"
    git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader "$OPENCL_LOADER_DIR"
  else
    echo -e "${YELLOW}OpenCL-ICD-Loader already cloned, using existing copy${NC}"
  fi
  
  for ABI in "${ABIS[@]}"; do
    echo -e "${GREEN}Building OpenCL ICD Loader for $ABI${NC}"
    
    # Set architecture-specific variables
    if [ "$ABI" = "arm64-v8a" ]; then
      ANDROID_ABI="arm64-v8a"
      ARCH="aarch64"
    elif [ "$ABI" = "x86_64" ]; then
      ANDROID_ABI="x86_64"
      ARCH="x86_64"
    fi
    
    # Create build directory
    OPENCL_BUILD_DIR="$OPENCL_LOADER_DIR/build-$ABI"
    mkdir -p "$OPENCL_BUILD_DIR"
    
    pushd "$OPENCL_BUILD_DIR"
    
    # Configure and build
    cmake .. -G Ninja \
      -DCMAKE_TOOLCHAIN_FILE="$NDK_PATH/build/cmake/android.toolchain.cmake" \
      -DANDROID_ABI="$ANDROID_ABI" \
      -DANDROID_PLATFORM="$ANDROID_PLATFORM" \
      -DANDROID_STL=c++_shared \
      -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
      -DOPENCL_ICD_LOADER_HEADERS_DIR="$HOST_PLATFORM_DIR/sysroot/usr/include" \
      -DBUILD_SHARED_LIBS=ON
    
    ninja
    
    # Create destination directory and copy the library
    DEST_DIR="$OPENCL_LIB_DIR/$ABI"
    mkdir -p "$DEST_DIR"
    cp "libOpenCL.so" "$DEST_DIR/"
    
    # Copy to the prebuilt/gpu directory for CI
    cp "libOpenCL.so" "$PREBUILT_GPU_DIR/$ABI/"
    
    # Create a flag file to indicate OpenCL is available
    touch "$PREBUILT_GPU_DIR/$ABI/.opencl_enabled"
    
    # Install to NDK sysroot for other components to link against
    cp "libOpenCL.so" "$HOST_PLATFORM_DIR/sysroot/usr/lib/$ARCH-linux-android/"
    
    popd
    
    echo -e "${GREEN}Successfully built OpenCL ICD Loader for $ABI${NC}"
  done
fi

# Build Vulkan if enabled
if [ "$BUILD_VULKAN" = true ]; then
  echo -e "${GREEN}=== Building Vulkan libraries ===${NC}"
  
  # Get Vulkan Headers
  if [ ! -d "$VULKAN_HEADERS_DIR" ]; then
    echo -e "${YELLOW}Cloning Vulkan-Headers...${NC}"
    git clone https://github.com/KhronosGroup/Vulkan-Headers "$VULKAN_HEADERS_DIR"
  else
    echo -e "${YELLOW}Vulkan-Headers already cloned, using existing copy${NC}"
  fi
  
  # Install Vulkan headers to our prebuilt dir
  mkdir -p "$VULKAN_INCLUDE_DIR"
  cp -r "$VULKAN_HEADERS_DIR/include/"* "$VULKAN_INCLUDE_DIR/"
  
  # The NDK already has Vulkan libraries, so we verify they're available
  # and print information about them
  for ABI in "${ABIS[@]}"; do
    if [ "$ABI" = "arm64-v8a" ]; then
      ARCH="aarch64"
    elif [ "$ABI" = "x86_64" ]; then
      ARCH="x86_64"
    fi
    
    # Check if Vulkan library exists in NDK
    NDK_VULKAN_LIB="$HOST_PLATFORM_DIR/sysroot/usr/lib/$ARCH-linux-android/$ANDROID_MIN_SDK/libvulkan.so"
    
    if [ -f "$NDK_VULKAN_LIB" ]; then
      echo -e "${GREEN}Found Vulkan library in NDK for $ABI: $NDK_VULKAN_LIB${NC}"
    else
      echo -e "${YELLOW}Warning: Vulkan library not found in NDK for $ABI${NC}"
      echo -e "${YELLOW}Expected at: $NDK_VULKAN_LIB${NC}"
      echo -e "${YELLOW}Android app will need to include Vulkan library from device${NC}"
    fi
  done
  
  # Install glslc compiler to our PATH if it's available
  GLSLC_PATH=""
  
  # First check for glslc in NDK
  NDK_GLSLC="$NDK_PATH/shader-tools/$HOST_TAG/glslc"
  if [ -f "$NDK_GLSLC" ]; then
    GLSLC_PATH="$NDK_GLSLC"
    echo -e "${GREEN}Found glslc compiler in NDK: $GLSLC_PATH${NC}"
  else
    # Try to find glslc in system PATH
    SYS_GLSLC=$(which glslc 2>/dev/null || echo "")
    if [ -n "$SYS_GLSLC" ]; then
      GLSLC_PATH="$SYS_GLSLC"
      echo -e "${GREEN}Found glslc compiler in system PATH: $GLSLC_PATH${NC}"
    else
      echo -e "${YELLOW}Warning: glslc compiler not found in NDK or system PATH${NC}"
      echo -e "${YELLOW}This might cause issues when building apps that use Vulkan shaders${NC}"
    fi
  fi
  
  # If we found glslc, test if it supports the extensions we need
  if [ -n "$GLSLC_PATH" ]; then
    # Create a temporary shader file to test capabilities
    TMP_SHADER_FILE=$(mktemp)
    echo -e "#version 450\n#extension GL_KHR_cooperative_matrix : enable\nvoid main() {}" > "$TMP_SHADER_FILE"
    
    # Test KHR_cooperative_matrix support
    $GLSLC_PATH -o /dev/null -fshader-stage=compute --target-env=vulkan1.3 "$TMP_SHADER_FILE" 2>/dev/null
    if [ $? -eq 0 ]; then
      echo -e "${GREEN}glslc supports GL_KHR_cooperative_matrix extension${NC}"
      
      # Create Vulkan flag files for each ABI
      for ABI in "${ABIS[@]}"; do
        touch "$PREBUILT_GPU_DIR/$ABI/.vulkan_enabled"
        
        # Copy glslc to the GPU prebuilt directory for reference
        if [ "$HOST_TAG" = "linux-x86_64" ]; then
          # Only copy for Linux as it will be used by the CI workflow
          mkdir -p "$PREBUILT_GPU_DIR/tools"
          cp "$GLSLC_PATH" "$PREBUILT_GPU_DIR/tools/glslc"
        fi
      done
    else
      echo -e "${YELLOW}Warning: glslc does not support GL_KHR_cooperative_matrix extension${NC}"
    fi
    
    # Test NV_cooperative_matrix2 support
    echo -e "#version 450\n#extension GL_NV_cooperative_matrix2 : enable\nvoid main() {}" > "$TMP_SHADER_FILE"
    $GLSLC_PATH -o /dev/null -fshader-stage=compute --target-env=vulkan1.3 "$TMP_SHADER_FILE" 2>/dev/null
    if [ $? -eq 0 ]; then
      echo -e "${GREEN}glslc supports GL_NV_cooperative_matrix2 extension${NC}"
    else
      echo -e "${YELLOW}Warning: glslc does not support GL_NV_cooperative_matrix2 extension${NC}"
    fi
    
    # Clean up
    rm "$TMP_SHADER_FILE"
  fi
else
  echo -e "${YELLOW}Vulkan support is disabled${NC}"
  
  # Clean up any existing Vulkan-related files to avoid issues with CMake
  echo -e "${YELLOW}Cleaning up any existing Vulkan-related build files...${NC}"
  
  # Check if any previous build directories exist
  if [ -d "$PREBUILT_DIR/build-android" ]; then
    for ABI in "${ABIS[@]}"; do
      BUILD_DIR="$PREBUILT_DIR/build-android/$ABI"
      
      # Remove any Vulkan-related CMake files that might cause issues
      if [ -f "$BUILD_DIR/host-toolchain.cmake" ]; then
        echo -e "${YELLOW}Removing existing host-toolchain.cmake for $ABI${NC}"
        rm -f "$BUILD_DIR/host-toolchain.cmake"
      fi
      
      if [ -f "$BUILD_DIR/android-custom.toolchain.cmake" ]; then
        echo -e "${YELLOW}Removing existing android-custom.toolchain.cmake for $ABI${NC}"
        rm -f "$BUILD_DIR/android-custom.toolchain.cmake"
      fi
      
      # Remove any Vulkan flag files
      if [ -f "$PREBUILT_GPU_DIR/$ABI/.vulkan_enabled" ]; then
        echo -e "${YELLOW}Removing Vulkan enabled flag for $ABI${NC}"
        rm -f "$PREBUILT_GPU_DIR/$ABI/.vulkan_enabled"
      fi
    done
  fi
fi

echo -e "${GREEN}=== GPU backends build complete ===${NC}"
echo -e "${GREEN}OpenCL headers: $OPENCL_INCLUDE_DIR${NC}"
echo -e "${GREEN}OpenCL libraries: $OPENCL_LIB_DIR${NC}"
echo -e "${GREEN}Vulkan headers: $VULKAN_INCLUDE_DIR${NC}"

if [ -n "$GLSLC_PATH" ]; then
  echo -e "${GREEN}Vulkan shader compiler (glslc): $GLSLC_PATH${NC}"
fi

# Verify the build output for CI usage
for ABI in "${ABIS[@]}"; do
  echo -e "${YELLOW}Verifying build artifacts for $ABI...${NC}"
  
  # Check OpenCL libraries
  if [ -f "$PREBUILT_GPU_DIR/$ABI/libOpenCL.so" ] && [ -f "$PREBUILT_GPU_DIR/$ABI/.opencl_enabled" ]; then
    echo -e "${GREEN}✓ OpenCL library for $ABI is available at $PREBUILT_GPU_DIR/$ABI/libOpenCL.so${NC}"
    ls -la "$PREBUILT_GPU_DIR/$ABI/libOpenCL.so"
  else
    echo -e "${RED}✗ OpenCL library for $ABI is missing from $PREBUILT_GPU_DIR/$ABI/${NC}"
    # Try to copy it again if it's in the source location but not in the destination
    if [ -f "$OPENCL_LIB_DIR/$ABI/libOpenCL.so" ]; then
      mkdir -p "$PREBUILT_GPU_DIR/$ABI"
      cp "$OPENCL_LIB_DIR/$ABI/libOpenCL.so" "$PREBUILT_GPU_DIR/$ABI/"
      touch "$PREBUILT_GPU_DIR/$ABI/.opencl_enabled"
      echo -e "${GREEN}✓ OpenCL library for $ABI was copied to $PREBUILT_GPU_DIR/$ABI/libOpenCL.so${NC}"
    fi
  fi
  
  # Check Vulkan flag file
  if [ -f "$PREBUILT_GPU_DIR/$ABI/.vulkan_enabled" ]; then
    echo -e "${GREEN}✓ Vulkan support for $ABI is enabled${NC}"
  else
    # Create it if we have the headers
    if [ -d "$VULKAN_INCLUDE_DIR" ] && [ -n "$GLSLC_PATH" ]; then
      touch "$PREBUILT_GPU_DIR/$ABI/.vulkan_enabled"
      echo -e "${GREEN}✓ Vulkan support flag for $ABI was created${NC}"
    else
      echo -e "${RED}✗ Vulkan support for $ABI is not enabled${NC}"
    fi
  fi
done

echo -e "${GREEN}Add the following to your build_android_external.sh command:${NC}"
echo -e "${YELLOW}-DVulkan_INCLUDE_DIR=$VULKAN_INCLUDE_DIR${NC}"

if [ -n "$GLSLC_PATH" ]; then
  echo -e "${YELLOW}-DVulkan_GLSLC_EXECUTABLE=$GLSLC_PATH${NC}"
fi

echo -e "${GREEN}Done!${NC}"

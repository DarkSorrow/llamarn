#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Get the absolute path of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get project root directory (one level up from script dir)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# CI Usage Note:
# This script is called directly in the CI workflow to download and set up
# the iOS framework. It handles all the necessary setup without requiring
# the setupLlamaCpp.sh script first.

# Define paths
CPP_DIR="$PROJECT_ROOT/cpp"
PREBUILT_DIR="$PROJECT_ROOT/ios/libs"
INCLUDE_DIR="$PROJECT_ROOT/ios/include"
TEMP_DIR="$PROJECT_ROOT/.llamacpp-temp"

# Import version from used_version.sh if it exists
. "$SCRIPT_DIR/used_version.sh"
# Check for requirements
check_required_tool() {
    local tool=$1
    local install_message=$2

    if ! command -v $tool &> /dev/null; then
        echo -e "${RED}Error: $tool is required but not found.${NC}"
        echo "$install_message"
        exit 1
    fi
}

echo "Checking for required tools..."
check_required_tool "curl" "Please install curl (brew install curl)"
check_required_tool "unzip" "Please install unzip (brew install unzip)"

# Create necessary directories
mkdir -p "$PREBUILT_DIR"
mkdir -p "$INCLUDE_DIR"
mkdir -p "$TEMP_DIR"

# Function to download a file
download_file() {
  local url="$1"
  local output_path="$2"
  
  echo -e "${YELLOW}Downloading from: $url${NC}"
  echo -e "${YELLOW}Saving to: $output_path${NC}"
  
  curl -L -o "$output_path" "$url"
  return $?
}

# Copy all necessary header files from llama.cpp to ios/include
copy_header_files() {
  echo -e "${YELLOW}Copying header files to $INCLUDE_DIR...${NC}"
  
  # Main headers from llama.cpp repository
  cp -f "$CPP_DIR/llama.cpp/include/llama.h" "$INCLUDE_DIR/"
  cp -f "$CPP_DIR/llama.cpp/include/llama-cpp.h" "$INCLUDE_DIR/"
  
  # Common headers
  cp -f "$CPP_DIR/llama.cpp/common/common.h" "$INCLUDE_DIR/"
  cp -f "$CPP_DIR/llama.cpp/common/log.h" "$INCLUDE_DIR/"
  cp -f "$CPP_DIR/llama.cpp/common/sampling.h" "$INCLUDE_DIR/"
  cp -f "$CPP_DIR/llama.cpp/common/chat.h" "$INCLUDE_DIR/"
  cp -f "$CPP_DIR/llama.cpp/common/ngram-cache.h" "$INCLUDE_DIR/"
  cp -f "$CPP_DIR/llama.cpp/common/json-schema-to-grammar.h" "$INCLUDE_DIR/"
  cp -f "$CPP_DIR/llama.cpp/common/speculative.h" "$INCLUDE_DIR/"
  
  # Create minja subdirectory
  mkdir -p "$INCLUDE_DIR/common/minja"
  
  # Minja headers
  cp -f "$CPP_DIR/llama.cpp/common/minja/minja.hpp" "$INCLUDE_DIR/common/minja/"
  cp -f "$CPP_DIR/llama.cpp/common/minja/chat-template.hpp" "$INCLUDE_DIR/common/minja/"
  
  # JSON headers
  cp -f "$CPP_DIR/llama.cpp/common/json.hpp" "$INCLUDE_DIR/common/"
  cp -f "$CPP_DIR/llama.cpp/common/base64.hpp" "$INCLUDE_DIR/common/"
  
  echo -e "${GREEN}Header files copied successfully${NC}"
  return 0
}

# Download and setup iOS framework
download_ios_framework() {
  echo -e "${YELLOW}Downloading iOS framework (version: $LLAMA_CPP_TAG)...${NC}"
  
  # Check if framework already exists and is valid
  if [ -d "$PREBUILT_DIR/llama.xcframework" ] && 
     { [ -d "$PREBUILT_DIR/llama.xcframework/ios-arm64/llama.framework" ] || 
       [ -d "$PREBUILT_DIR/llama.xcframework/ios-arm64_x86_64-simulator/llama.framework" ]; }; then
    echo -e "${GREEN}iOS framework already exists at: $PREBUILT_DIR/llama.xcframework${NC}"
    
    # Skip if not forced
    if [ "$1" != "--force" ]; then
      echo -e "${YELLOW}Use '--force' to redownload or 'clean' to remove it${NC}"
      # Still copy header files even if framework exists
      copy_header_files
      return 0
    else
      echo -e "${YELLOW}Force flag detected, redownloading...${NC}"
    fi
  fi
  
  # Check if we have local llama.cpp source available
  # If it exists, we'll copy headers but still download the prebuilt framework
  if [ -d "$CPP_DIR/llama.cpp" ] && [ -f "$CPP_DIR/llama.cpp/include/llama.h" ]; then
    echo -e "${GREEN}Found local llama.cpp source at: $CPP_DIR/llama.cpp${NC}"
    echo -e "${YELLOW}Will use local source for headers but still download prebuilt framework${NC}"
    FOUND_LOCAL_SOURCE=1
  else
    echo -e "${YELLOW}No local llama.cpp source found, downloading everything...${NC}"
    FOUND_LOCAL_SOURCE=0
  fi
  
  # Download URL - use the same format as in llama_cpp_ios.sh
  local url="https://github.com/ggerganov/llama.cpp/releases/download/$LLAMA_CPP_TAG/llama-$LLAMA_CPP_TAG-xcframework.zip"
  local temp_zip="$TEMP_DIR/ios_framework.zip"
  
  echo -e "${YELLOW}Using URL: $url${NC}"
  
  # Download the framework
  if ! download_file "$url" "$temp_zip"; then
    echo -e "${RED}Failed to download iOS framework${NC}"
    return 1
  fi
  
  # Remove any existing framework
  rm -rf "$PREBUILT_DIR/llama.xcframework"
  
  # Extract the framework
  echo -e "${YELLOW}Extracting framework...${NC}"
  unzip -q -o "$temp_zip" -d "$TEMP_DIR"
  
  # Find the extracted framework
  local xcframework_path=$(find "$TEMP_DIR" -name "*.xcframework" -type d | head -n 1)
  
  if [ -z "$xcframework_path" ]; then
    echo -e "${RED}Error: No .xcframework found in downloaded archive${NC}"
    rm -f "$temp_zip"
    return 1
  fi
  
  # Move it to the prebuilt location
  echo -e "${YELLOW}Moving framework to: $PREBUILT_DIR/llama.xcframework${NC}"
  mv "$xcframework_path" "$PREBUILT_DIR/llama.xcframework"
  
  # Verify the framework has the necessary slices
  if [[ ! -d "$PREBUILT_DIR/llama.xcframework/ios-arm64/llama.framework" && 
        ! -d "$PREBUILT_DIR/llama.xcframework/ios-arm64_x86_64-simulator/llama.framework" ]]; then
    echo -e "${RED}Error: iOS framework slices not found in downloaded archive${NC}"
    find "$PREBUILT_DIR/llama.xcframework" -type d | sort
    rm -f "$temp_zip"
    return 1
  fi
  
  # Copy header files
  copy_header_files
  
  # Clean up temporary files
  rm -f "$temp_zip"
  
  echo -e "${GREEN}iOS framework downloaded and installed successfully to:${NC}"
  echo -e "${GREEN}$PREBUILT_DIR/llama.xcframework${NC}"
  
  # List framework architectures
  echo -e "${YELLOW}Framework slices available:${NC}"
  find "$PREBUILT_DIR/llama.xcframework" -maxdepth 1 -type d -not -path "$PREBUILT_DIR/llama.xcframework" | sort
  
  return 0
}

# Clean up iOS framework
clean_ios_framework() {
  echo -e "${YELLOW}Cleaning iOS framework...${NC}"
  
  rm -rf "$PREBUILT_DIR/llama.xcframework"
  rm -rf "$INCLUDE_DIR"
  rm -rf "$TEMP_DIR"
  
  echo -e "${GREEN}iOS framework cleaned successfully${NC}"
  return 0
}

# Main function
main() {
  if [ $# -eq 0 ]; then
    echo "Usage: $0 {init|clean|version} [--force]"
    echo "  init     - Download and setup iOS framework to ios/libs directory"
    echo "  clean    - Remove iOS framework"
    echo "  version  - Show the llama.cpp version used"
    echo ""
    echo "Options:"
    echo "  --force  - Force redownload even if framework exists"
    exit 1
  fi
  
  local command="$1"
  shift
  
  case "$command" in
    "init")
      download_ios_framework "$1"
      ;;
    "clean")
      clean_ios_framework
      ;;
    "version")
      echo "Using llama.cpp version: $LLAMA_CPP_TAG"
      ;;
    *)
      echo "Unknown command: $command"
      echo "Usage: $0 {init|clean|version} [--force]"
      exit 1
      ;;
  esac
}

# Run main with all arguments
main "$@"

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

# Create necessary directories
mkdir -p "$PREBUILT_DIR"
mkdir -p "$INCLUDE_DIR"
mkdir -p "$TEMP_DIR"

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
  
  # JSON headers — nlohmann is vendored, resolved via HEADER_SEARCH_PATHS at build time
  mkdir -p "$INCLUDE_DIR/nlohmann"
  cp -f "$CPP_DIR/llama.cpp/vendor/nlohmann/json.hpp" "$INCLUDE_DIR/nlohmann/"
  cp -f "$CPP_DIR/llama.cpp/common/base64.hpp" "$INCLUDE_DIR/"
  # Note: minja/ vendor was removed upstream; jinja template engine is now
  # in common/jinja/ and compiled directly into the bridge (no copy needed).
  
  echo -e "${GREEN}Header files copied successfully${NC}"
  return 0
}

# Build the iOS xcframework from the pinned llama.cpp source.
#
# Why we build instead of downloading the release zip:
# As of llama.cpp b10648 (commit 805984d67, PR #27252 — "release.yml builds only
# macos and ios-device to cut the build time"), the published
# llama-<tag>-xcframework.zip contains ONLY macos + ios-device. It no longer has
# an iOS *simulator* slice. Since this library ships its xcframework inside the
# npm tarball (package.json `files` includes "ios"), a simulator-less framework
# would break `pod install` and simulator builds for every consumer of the
# package. There is no alternative published artifact — the upstream workflow
# that builds all slices (build-apple.yml) uploads a GitHub Actions artifact with
# retention-days: 1, which is neither stable nor publicly downloadable.
#
# build-xcframework.sh accepts an explicit build list, so we ask for exactly the
# two iOS slices. xcodebuild -create-xcframework writes a correct Info.plist for
# whatever slices it was given, so no hand-maintained plist template is needed.
build_ios_framework() {
  local force="$1"
  local llama_src="$CPP_DIR/llama.cpp"
  local device_slice="$PREBUILT_DIR/llama.xcframework/ios-arm64/llama.framework"
  local sim_slice="$PREBUILT_DIR/llama.xcframework/ios-arm64_x86_64-simulator/llama.framework"

  echo -e "${YELLOW}Setting up iOS framework (version: $LLAMA_CPP_TAG)...${NC}"

  # Require BOTH slices before considering the framework usable. The old check
  # used OR, so a framework missing the simulator slice looked "already valid"
  # and every re-run silently skipped the rebuild.
  if [ -d "$device_slice" ] && [ -d "$sim_slice" ]; then
    echo -e "${GREEN}iOS framework already present with device + simulator slices${NC}"
    if [ "$force" != "--force" ]; then
      echo -e "${YELLOW}Use '--force' to rebuild or 'clean' to remove it${NC}"
      copy_header_files
      return 0
    fi
    echo -e "${YELLOW}Force flag detected, rebuilding...${NC}"
  fi

  if [ ! -f "$llama_src/build-xcframework.sh" ]; then
    echo -e "${RED}Error: llama.cpp source not found at $llama_src${NC}"
    echo -e "${YELLOW}Run 'npm run setup-llama-cpp' first to init the submodule.${NC}"
    return 1
  fi

  check_required_tool "xcodebuild" "Install Xcode and run: xcode-select --install"
  check_required_tool "cmake" "Please install cmake (brew install cmake)"

  # Warn if the checkout drifted from the pinned commit — the shipped binary must
  # match the headers we compile the bridge against.
  local head_commit
  head_commit="$(git -C "$llama_src" rev-parse HEAD 2>/dev/null)"
  if [ -n "$LLAMA_CPP_COMMIT" ] && [ -n "$head_commit" ] && [ "$head_commit" != "$LLAMA_CPP_COMMIT" ]; then
    echo -e "${YELLOW}Warning: cpp/llama.cpp is at $head_commit${NC}"
    echo -e "${YELLOW}         but used_version.sh pins $LLAMA_CPP_COMMIT${NC}"
  fi

  echo -e "${YELLOW}Building iOS device + simulator slices from source...${NC}"
  echo -e "${YELLOW}(this compiles llama.cpp twice and can take a while)${NC}"
  if ! ( cd "$llama_src" && ./build-xcframework.sh ios-sim ios-device ); then
    echo -e "${RED}Error: build-xcframework.sh failed${NC}"
    return 1
  fi

  local built="$llama_src/build-apple/llama.xcframework"
  if [ ! -d "$built" ]; then
    echo -e "${RED}Error: expected xcframework not found at $built${NC}"
    return 1
  fi

  echo -e "${YELLOW}Installing framework to: $PREBUILT_DIR/llama.xcframework${NC}"
  rm -rf "$PREBUILT_DIR/llama.xcframework"
  mkdir -p "$PREBUILT_DIR"
  cp -R "$built" "$PREBUILT_DIR/llama.xcframework"

  # Both slices are mandatory. Previously a missing simulator slice was only a
  # warning, which is how a device-only framework reached CocoaPods and produced
  # the misleading "contains both static and dynamic frameworks" error.
  if [ ! -d "$device_slice" ]; then
    echo -e "${RED}Error: iOS device slice (ios-arm64) missing after build${NC}"
    find "$PREBUILT_DIR/llama.xcframework" -maxdepth 1 -type d | sort
    return 1
  fi
  if [ ! -d "$sim_slice" ]; then
    echo -e "${RED}Error: iOS simulator slice (ios-arm64_x86_64-simulator) missing after build${NC}"
    find "$PREBUILT_DIR/llama.xcframework" -maxdepth 1 -type d | sort
    return 1
  fi

  copy_header_files

  echo -e "${GREEN}iOS framework built and installed successfully to:${NC}"
  echo -e "${GREEN}$PREBUILT_DIR/llama.xcframework${NC}"
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
    echo "  init     - Build and install the iOS framework into ios/libs"
    echo "  clean    - Remove iOS framework"
    echo "  version  - Show the llama.cpp version used"
    echo ""
    echo "Options:"
    echo "  --force  - Force rebuild even if framework exists"
    exit 1
  fi
  
  local command="$1"
  shift
  
  case "$command" in
    "init")
      build_ios_framework "$1"
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

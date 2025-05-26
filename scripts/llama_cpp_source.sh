#!/bin/bash
# This script manages the llama.cpp submodule version
# It ensures we use a specific, tested version of llama.cpp

set -e

# Path to the llama.cpp submodule
LLAMA_CPP_DIR="cpp/llama.cpp"

# Temp directory for caching
TEMP_DIR=".llamacpp-temp"
mkdir -p $TEMP_DIR

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to initialize the submodule
init_submodule() {
  # Check if the directory already exists with the right commit
  if [ -d "$LLAMA_CPP_DIR/.git" ]; then
    cd $LLAMA_CPP_DIR
    CURRENT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "none")
    cd - > /dev/null
    
    # If the commit is already correct, we can skip
    if [ "$CURRENT_COMMIT" = "$LLAMA_CPP_COMMIT" ]; then
      echo -e "${GREEN}llama.cpp is already at the correct commit: ${LLAMA_CPP_COMMIT}${NC}"
      return 0
    else
      echo -e "${YELLOW}llama.cpp exists but is at commit $CURRENT_COMMIT, updating to $LLAMA_CPP_COMMIT${NC}"
    fi
  else
    echo -e "${YELLOW}Initializing llama.cpp repository...${NC}"
  fi
  
  # Check if git is available
  if ! command -v git &> /dev/null; then
    echo -e "${RED}Error: git is not installed${NC}"
    exit 1
  fi
  
  # Create the directory structure if it doesn't exist
  mkdir -p $LLAMA_CPP_DIR
  
  # Check if we have a cached clone that we can copy
  if [ -d "$TEMP_DIR/llama.cpp/.git" ]; then
    echo -e "${YELLOW}Using cached repository${NC}"
    
    # Check if the cached repo has the commit we need
    cd "$TEMP_DIR/llama.cpp"
    git fetch origin -q
    if git cat-file -e $LLAMA_CPP_COMMIT^{commit} 2>/dev/null; then
      echo -e "${GREEN}Cached repository has the required commit${NC}"
      # Copy the cached repo to the destination
      cd - > /dev/null
      rsync -a --exclude .git "$TEMP_DIR/llama.cpp/" "$LLAMA_CPP_DIR/"
      
      # Copy just the git directory separately (better than cp -r)
      rsync -a "$TEMP_DIR/llama.cpp/.git" "$LLAMA_CPP_DIR/"
    else
      echo -e "${YELLOW}Cached repository doesn't have the required commit, updating${NC}"
      git checkout master -q
      git pull origin master -q
      cd - > /dev/null
      rsync -a --exclude .git "$TEMP_DIR/llama.cpp/" "$LLAMA_CPP_DIR/"
      rsync -a "$TEMP_DIR/llama.cpp/.git" "$LLAMA_CPP_DIR/"
    fi
  else
    # If not a git repo or no cache, clone the repository
    if [ ! -d "$LLAMA_CPP_DIR/.git" ]; then
      echo -e "${YELLOW}Cloning fresh repository${NC}"
      rm -rf $LLAMA_CPP_DIR
      git clone https://github.com/ggml-org/llama.cpp.git $LLAMA_CPP_DIR
      
      # Also clone to cache
      rm -rf "$TEMP_DIR/llama.cpp"
      git clone https://github.com/ggml-org/llama.cpp.git "$TEMP_DIR/llama.cpp"
    fi
  fi
  
  # Go to the submodule directory
  cd $LLAMA_CPP_DIR
  
  # Checkout the specific commit (and update cache)
  git checkout $LLAMA_CPP_COMMIT -q
  cd - > /dev/null
  
  if [ -d "$TEMP_DIR/llama.cpp/.git" ]; then
    cd "$TEMP_DIR/llama.cpp"
    git checkout $LLAMA_CPP_COMMIT -q 2>/dev/null || echo "Couldn't update cache, continuing anyway"
    cd - > /dev/null
  fi
  
  echo -e "${GREEN}Successfully initialized llama.cpp to commit: ${LLAMA_CPP_COMMIT}${NC}"
}

# Function to display current submodule status
show_status() {
  echo -e "${YELLOW}Current llama.cpp submodule status:${NC}"
  
  if [ -d "$LLAMA_CPP_DIR/.git" ]; then
    cd $LLAMA_CPP_DIR
    CURRENT_COMMIT=$(git rev-parse HEAD)
    CURRENT_BRANCH=$(git branch --show-current || echo "detached HEAD")
    CURRENT_TAG=$(git describe --tags --exact-match 2>/dev/null || echo "no tag")
    
    echo -e "Branch: ${GREEN}$CURRENT_BRANCH${NC}"
    echo -e "Tag: ${GREEN}$CURRENT_TAG${NC}"
    echo -e "Commit: ${GREEN}$CURRENT_COMMIT${NC}"
    
    # Check if we're on the desired version
    if [ "$CURRENT_COMMIT" = "$LLAMA_CPP_COMMIT" ]; then
      echo -e "${GREEN}✓ llama.cpp is at the correct commit${NC}"
    else
      echo -e "${RED}✗ llama.cpp is NOT at the expected commit${NC}"
      echo -e "${YELLOW}Expected commit: $LLAMA_CPP_COMMIT${NC}"
      echo -e "${YELLOW}Current commit:  $CURRENT_COMMIT${NC}"
    fi
    cd - > /dev/null
  else
    echo -e "${RED}Repository not initialized${NC}"
  fi
}

# Function to list available tags
list_tags() {
  echo -e "${YELLOW}Available llama.cpp release tags:${NC}"
  
  # Fetch tags directly from the remote
  echo -e "${YELLOW}Fetching tags from remote repository...${NC}"
  git ls-remote --tags https://github.com/ggml-org/llama.cpp.git | grep -o 'refs/tags/b[0-9]*' | sed 's/refs\/tags\///' | sort -rV | head -10
  
  echo -e "\n${YELLOW}To update llama.cpp version, edit LLAMA_CPP_COMMIT in this script.${NC}"
}

# Main function
main() {
  case "$1" in
    init)
      init_submodule
      ;;
    status)
      show_status
      ;;
    list-tags)
      list_tags
      ;;
    *)
      echo "Usage: $0 {init|status|list-tags}"
      echo "  init       - Initialize and set the llama.cpp repository to the correct version"
      echo "  status     - Show current llama.cpp repository status"
      echo "  list-tags  - List available llama.cpp release tags"
      ;;
  esac
}

# Run the main function with the provided arguments
main "$@" 
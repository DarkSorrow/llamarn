# Makefile for @novastera-oss/llamarn
# Provides convenient targets for cleaning, updating, and preparing the project

.PHONY: clean clean-all update help

# Default target
all: help

# Help target
help:
	@echo "Available targets:"
	@echo "  clean      - Clean all build artifacts and temporary files"
	@echo "  clean-all  - Clean everything including llama.cpp setup"
	@echo "  update     - Setup llama.cpp and build Android for macOS"
	@echo "  prepare    - Build the library using react-native-builder-bob"
	@echo "  all        - Show this help message"

# Clean target - runs all clean scripts from package.json
clean:
	@echo "🧹 Cleaning build artifacts..."
	npm run clean
	@echo "🧹 Cleaning Android build files..."
	npm run clean-android
	@echo "🧹 Cleaning prebuilt files..."
	npm run clean-prebuilt
	@echo "✅ Clean completed"

# Clean all target - includes llama.cpp cleanup
clean-all: clean
	@echo "🧹 Cleaning llama.cpp setup..."
	npm run clean-llama
	@echo "✅ Clean all completed"

# Update target - setup llama.cpp and build Android for macOS
update:
	@echo "🔄 Setting up llama.cpp..."
	npm run setup-llama-cpp
	@echo "🔨 Building Android for macOS..."
	npm run build-android-macos
	@echo "✅ Update completed"
	@echo "📦 Preparing library build..."
	npm run prepare
	@echo "✅ Prepare completed"

# Full workflow target - clean, update, and prepare
full: clean-all update
	@echo "🎉 Full workflow completed!" 
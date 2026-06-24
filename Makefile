# Makefile for @novastera-oss/llamarn
# Provides convenient targets for installing, cleaning, updating, and preparing the project

.PHONY: all help install clean clean-all update prepare pods test full

# Default target
all: help

# Help target
help:
	@echo "Available targets:"
	@echo "  install    - Install npm dependencies (root + example workspace)"
	@echo "  clean      - Clean all build artifacts and temporary files"
	@echo "  clean-all  - Clean everything including llama.cpp setup"
	@echo "  update     - Setup llama.cpp, build Android for macOS, and prepare the library"
	@echo "  prepare    - Build the library using react-native-builder-bob (codegen + lib/)"
	@echo "  pods       - Install/update CocoaPods for the example iOS app"
	@echo "  test       - Run typecheck and the JS test suite"
	@echo "  full       - clean-all + update (full rebuild workflow)"
	@echo "  all        - Show this help message"

# Install target - npm workspaces (root + example) in one shot
install:
	@echo "📦 Installing npm dependencies..."
	npm install
	@echo "✅ Install completed"

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
	$(MAKE) prepare

# Prepare target - regenerate codegen and the lib/ build output
prepare:
	@echo "📦 Preparing library build..."
	npm run prepare
	@echo "✅ Prepare completed"

# Pods target - (re)install CocoaPods for the example app.
# Required after any package.json change (root or example) that affects
# native dependencies, or after a TS spec change that needs codegen rerun.
pods:
	@echo "🍫 Installing CocoaPods for the example app..."
	cd example/ios && pod install
	@echo "✅ Pods installed"

# Test target - typecheck + JS test suite
test:
	@echo "🔎 Typechecking..."
	npm run typecheck
	@echo "🧪 Running tests..."
	npm test
	@echo "✅ Test completed"

# Full workflow target - clean, update, and prepare
full: clean-all update
	@echo "🎉 Full workflow completed!"

#!/bin/bash
# Build and run the host-side native bridge tests (cpp/tests).
# Usage: scripts/test_native_host.sh [test-name-filter]
# Set RN_TEST_MODEL=/abs/path/model.gguf to run the model-backed tests.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${RN_NATIVE_TEST_BUILD_DIR:-$ROOT_DIR/build/native-tests}"

EXTRA_ARGS=()
if [ "$(uname)" = "Darwin" ]; then
    # Use the Xcode toolchain + SDK consistently; a stale CommandLineTools SDK can break the link step.
    EXTRA_ARGS+=(
        "-DCMAKE_C_COMPILER=$(xcrun --find clang)"
        "-DCMAKE_CXX_COMPILER=$(xcrun --find clang++)"
        "-DCMAKE_OSX_SYSROOT=$(xcrun --sdk macosx --show-sdk-path 2>/dev/null || true)"
    )
    XCODE_SDK="$(xcode-select -p)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
    if [ -d "$XCODE_SDK" ]; then
        EXTRA_ARGS[2]="-DCMAKE_OSX_SYSROOT=$XCODE_SDK"
    fi
fi

cmake -S "$ROOT_DIR/cpp/tests" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DRN_TESTS_WITH_BRIDGE=ON \
    "${EXTRA_ARGS[@]}" ${RN_NATIVE_TEST_CMAKE_ARGS:-}
cmake --build "$BUILD_DIR" --target rn_bridge_tests -j "$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
"$BUILD_DIR/rn_bridge_tests" "$@"

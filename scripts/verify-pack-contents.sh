#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Running npm pack dry run..."
npm pack --dry-run >/dev/null

echo "Creating temporary tarball for content verification..."
TMP_TARBALL_DIR="$(mktemp -d)"
npm pack --pack-destination "$TMP_TARBALL_DIR" >/dev/null

TMP_LIST_FILE="$(mktemp)"
cleanup() {
  rm -f "$TMP_LIST_FILE"
  rm -rf "$TMP_TARBALL_DIR"
}
trap cleanup EXIT

shopt -s nullglob
PACKED_TARBALLS=("$TMP_TARBALL_DIR"/*.tgz)
shopt -u nullglob

if [ "${#PACKED_TARBALLS[@]}" -eq 0 ]; then
  echo "ERROR: npm pack did not produce a tarball."
  exit 1
fi

TARBALL_PATH="${PACKED_TARBALLS[0]}"
tar -tf "$TARBALL_PATH" >"$TMP_LIST_FILE"

# cpp/rn-common-sources.cmake uses CMake file(GLOB ...) — those patterns are not tarball paths.
# Mirror the same rules: tools/mtmd/*.cpp minus the excluded files, tools/mtmd/models/*.cpp,
# and the chat template parsers in common/parsers/.
collect_mtmd_source_paths() {
  local mtmd_dir="${ROOT_DIR}/cpp/llama.cpp/tools/mtmd"
  shopt -s nullglob
  local f
  for f in "${mtmd_dir}"/*.cpp; do
    case "$(basename "$f")" in
      mtmd-cli.cpp|mtmd-helper-gen.cpp|deprecation-warning.cpp) continue ;;
    esac
    printf '%s\n' "${f#"${ROOT_DIR}/"}"
  done
  for f in "${mtmd_dir}/models"/*.cpp "${ROOT_DIR}/cpp/llama.cpp/common/parsers"/*.cpp; do
    printf '%s\n' "${f#"${ROOT_DIR}/"}"
  done
  shopt -u nullglob
}

if grep -Eq '\.dSYM/|/dSYMs/|llama-dSYMs' "$TMP_LIST_FILE"; then
  echo "ERROR: debug symbols (dSYMs) must not be shipped in the npm package:"
  grep -E '\.dSYM/|/dSYMs/|llama-dSYMs' "$TMP_LIST_FILE" | head -5
  exit 1
fi

MTMD_PATHS=()
while IFS= read -r mtmd_path; do
  [[ -n "$mtmd_path" ]] && MTMD_PATHS+=("$mtmd_path")
done < <(collect_mtmd_source_paths | sort -u)

if [ "${#MTMD_PATHS[@]}" -eq 0 ]; then
  echo "ERROR: Could not collect MTMD / parser source paths."
  exit 1
fi

REQUIRED_PATHS=(
  "android/CMakeLists.txt"
  "cpp/rn-common-sources.cmake"
  "cpp/llama.cpp/common/parsers/sources.cmake"
  "cpp/llama.cpp/common/json-schema.cpp"
  "${MTMD_PATHS[@]}"
)
MISSING_PATHS=()

for path in "${REQUIRED_PATHS[@]}"; do
  if ! grep -Fq "package/${path}" "$TMP_LIST_FILE"; then
    MISSING_PATHS+=("$path")
  fi
done

if [ "${#MISSING_PATHS[@]}" -gt 0 ]; then
  echo "ERROR: Required files are missing from npm package tarball:"
  for path in "${MISSING_PATHS[@]}"; do
    echo "  - $path"
  done
  exit 1
fi

echo "Pack verification passed."
echo "Verified ${#REQUIRED_PATHS[@]} required paths, including MTMD sources."

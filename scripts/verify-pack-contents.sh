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

MTMD_PATHS=()
while IFS= read -r mtmd_path; do
  MTMD_PATHS+=("$mtmd_path")
done < <(
  grep -o '\${MTMD_DIR}/[^ )]*' android/CMakeLists.txt \
    | sed 's#\${MTMD_DIR}/#cpp/llama.cpp/tools/mtmd/#' \
    | sort -u
)

if [ "${#MTMD_PATHS[@]}" -eq 0 ]; then
  echo "ERROR: Could not extract MTMD source paths from android/CMakeLists.txt."
  exit 1
fi

REQUIRED_PATHS=("android/CMakeLists.txt" "${MTMD_PATHS[@]}")
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

# llama.cpp common + mtmd sources compiled into the mobile bridge.
# Shared by android/CMakeLists.txt and cpp/tests (the host tests link exactly this list).
# Keep RNLlamaCpp.podspec source_files in sync.
#
# Requires LLAMA_CPP_DIR. Defines RN_COMMON_SOURCES, RN_MTMD_SOURCES.
#
# Left out on purpose (CLI / server / speculative only, nothing in the bridge calls them):
#   arg.cpp (pulls preset, download, hf-cache), console.cpp, download.cpp, hf-cache.cpp,
#   preset.cpp, subproc.cpp, file-lock.cpp, imatrix-loader.cpp,
#   tools/mtmd/mtmd-helper-gen.cpp (TTS generation), tools/mtmd/mtmd-cli.cpp,
#   tools/mtmd/deprecation-warning.cpp (defines main).

set(_RN_COMMON_DIR ${LLAMA_CPP_DIR}/common)

include(${_RN_COMMON_DIR}/parsers/sources.cmake)

set(RN_COMMON_SOURCES
    ${LLAMA_CHAT_PARSERS_SOURCES}
    ${_RN_COMMON_DIR}/log.cpp
    ${_RN_COMMON_DIR}/common.cpp
    ${_RN_COMMON_DIR}/fit.cpp
    ${_RN_COMMON_DIR}/sampling.cpp
    ${_RN_COMMON_DIR}/chat.cpp
    ${_RN_COMMON_DIR}/chat-auto-parser-generator.cpp
    ${_RN_COMMON_DIR}/chat-auto-parser-helpers.cpp
    ${_RN_COMMON_DIR}/chat-diff-analyzer.cpp
    ${_RN_COMMON_DIR}/chat-peg-parser.cpp
    ${_RN_COMMON_DIR}/reasoning-budget.cpp
    ${_RN_COMMON_DIR}/debug.cpp
    ${_RN_COMMON_DIR}/jinja/lexer.cpp
    ${_RN_COMMON_DIR}/jinja/parser.cpp
    ${_RN_COMMON_DIR}/jinja/runtime.cpp
    ${_RN_COMMON_DIR}/jinja/value.cpp
    ${_RN_COMMON_DIR}/jinja/string.cpp
    ${_RN_COMMON_DIR}/jinja/caps.cpp
    ${_RN_COMMON_DIR}/peg-parser.cpp
    ${_RN_COMMON_DIR}/unicode.cpp
    ${_RN_COMMON_DIR}/json.cpp
    ${_RN_COMMON_DIR}/json-schema.cpp
    ${_RN_COMMON_DIR}/json-schema-to-grammar.cpp
    ${_RN_COMMON_DIR}/trie.cpp
    ${_RN_COMMON_DIR}/llguidance.cpp
    # common_fit_apply (common.cpp) calls common_base_params_to_speculative
    ${_RN_COMMON_DIR}/speculative.cpp
    ${_RN_COMMON_DIR}/ngram-cache.cpp
    ${_RN_COMMON_DIR}/ngram-map.cpp
    ${_RN_COMMON_DIR}/ngram-mod.cpp
)

set(_RN_MTMD_DIR ${LLAMA_CPP_DIR}/tools/mtmd)
file(GLOB _RN_MTMD_ROOT_SOURCES  CONFIGURE_DEPENDS "${_RN_MTMD_DIR}/*.cpp")
file(GLOB _RN_MTMD_MODEL_SOURCES CONFIGURE_DEPENDS "${_RN_MTMD_DIR}/models/*.cpp")
list(REMOVE_ITEM _RN_MTMD_ROOT_SOURCES
    "${_RN_MTMD_DIR}/mtmd-cli.cpp"
    "${_RN_MTMD_DIR}/mtmd-helper-gen.cpp"
    "${_RN_MTMD_DIR}/deprecation-warning.cpp"
)

# vendor/hash: mtmd-helper.cpp derives bitmap IDs via hash_sha256_hex(); only sha256 is reachable.
set(RN_MTMD_SOURCES
    ${_RN_MTMD_ROOT_SOURCES}
    ${_RN_MTMD_MODEL_SOURCES}
    ${LLAMA_CPP_DIR}/vendor/hash/hash.cpp
    ${LLAMA_CPP_DIR}/vendor/hash/sha256/sha256.c
)

# Migration notes: llama.cpp b12177 → b12651

Status: done for the bridge. Source lists include `chat-tts-norm.cpp`, and the iOS podspec compiles the split mtmd sources plus `vendor/hash`. Chat replies are a `common_chat_input` appended per sampled token and truncated on stop or end-of-generation, then passed to `session.finish`. App init leaves `fit_params` at the upstream default. No JavaScript context field changed meaning. Host tests: `npm run test:native` — 10 passed, 12 skipped (`RN_TEST_MODEL` unset, including `image_number_png_tokenizes`). Prebuilt `jniLibs` and `llama.xcframework` rebuild is tracked separately and is not done by the host-test run.

| | From | To |
| --- | --- | --- |
| Tag | `b12177` | `b12651` |
| Commit | `6c01da6278ab6c69ed67d9124806b50d2b299ba7` | `f5e6915c28f10330885fcb9243a93ae98d3115ab` |
| Date | 2026-11-06 | 2026-11-23 |
| Commit count | | 474 |
| llama.cpp version | 0.9.0 | 1.0.0 |
| ggml version | 0.29.0 | 1.0.0 |

`scripts/used_version.sh` and the `cpp/llama.cpp` checkout are on `b12651`. The bridge under `cpp/` (everything except `cpp/llama.cpp/`) accumulates the reply the way `tools/server` does. Prebuilt `android/src/main/jniLibs` and `ios/libs/llama.xcframework` still have to be rebuilt from this commit before a device binary matches the checkout. Do not edit files under `cpp/llama.cpp/`.

Scope of this library: mobile only (Android OpenCL / Vulkan / optional Hexagon / dynamic CPU backends, iOS Metal). The completion path is one context, one sequence, no HTTP server. Server, realtime, TTS generation, speculative decoding, and the web UI stay out of the bridge.

Release notes checked: every tag from `b12177` to `b12651` (the GitHub release body for each tag is that build's commit subject). The items below are the ones that change this library's compile or its completion / chat / mtmd behavior. Kernel and model additions that ride along inside a rebuilt `.so` are listed in section 3 so they are not treated as bridge work.

---

## 1. Compile breaks (must fix before the library builds)

### 1.1 `common/chat-tts-norm.cpp` is now required

Added in b12302 (`chat: allow TTS model to provide custom text normalizer via jinja`, #32053). `common/chat.cpp` includes `chat-tts-norm.h` and calls `common_chat_tts_norm_add_builtins`. Upstream lists the file in `common/CMakeLists.txt`. Our lists do not:

- `cpp/rn-common-sources.cmake` (`RN_COMMON_SOURCES`) — Android and `cpp/tests`
- `RNLlamaCpp.podspec` `s.source_files`

Without the `.cpp`, chat template init fails at link time on both platforms. Add `common/chat-tts-norm.cpp` (and the header on the podspec). This is the Jinja builtin table for TTS templates. It does not turn on audio generation. Leave `tools/mtmd/mtmd-helper-gen.cpp` out, same as b12177.

### 1.2 iOS source list still names the removed `clip` files

b12211 (`mtmd: refactoring: remove clip.h, consolidate code into mtmd`, #31865) deleted:

- `tools/mtmd/clip.cpp`, `clip.h`, `clip-impl.h`, `clip-model.h`, `clip-graph.h`

and split that code into:

- `tools/mtmd/mtmd-graph.cpp` / `mtmd-graph.h`
- `tools/mtmd/mtmd-model.cpp` / `mtmd-model.h`
- `tools/mtmd/mtmd-subcontext.cpp` / `mtmd-subcontext.h`
- `tools/mtmd/mtmd-impl.h`

`RNLlamaCpp.podspec` still lists the five `clip*` files and does not list the four replacements. CocoaPods will fail the iOS compile on the missing paths, and `mtmd.cpp` will fail to link the symbols that moved.

`cpp/rn-common-sources.cmake` globs `tools/mtmd/*.cpp` and `tools/mtmd/models/*.cpp`, so the Android / host-test build picks the new files up and drops the deleted ones. Keep that glob. Keep the existing removals: `mtmd-cli.cpp`, `mtmd-helper-gen.cpp`, `deprecation-warning.cpp`.

`tools/mtmd/mtmd-helper-video-ffmpeg.cpp` is inside the glob. The whole file is under `#ifdef MTMD_VIDEO`. Leave `MTMD_VIDEO` unset. Upstream's own CMake turns it off unless `LLAMA_SUBPROCESS` is on, and the built-in decoder shells out to `ffmpeg` / `ffprobe`. An empty translation unit is fine. A defined `MTMD_VIDEO` pulls `vendor/sheredom/subprocess.h` and is a desktop feature.

The podspec also still omits `vendor/hash/hash.cpp` and `vendor/hash/sha256/sha256.c`. `mtmd-helper.cpp` calls `hash_sha256_hex` (already true at b12177). `cpp/rn-common-sources.cmake` already compiles them. Add the same two files to the podspec in this source-list edit.

`common/parsers/inkling.cpp` (b12455, #32381, grammar follow-up b12650) is already on both lists: Android via `include(parsers/sources.cmake)`, iOS via the `parsers/*.{h,cpp}` glob. No extra line if those stay.

New `tools/mtmd/models/*.cpp` files the glob and the podspec `models/*` glob will compile. A failure in any of them fails the app build:

`fish-s2-gen.cpp`, `fish-s2-spkenc.cpp`, `inkling2v.cpp`, `kokoro.cpp`, `magevl.cpp`, `moshi.cpp`, `music3-gen.cpp`, `snake-vae.cpp`, `tiny-voca.cpp`, `vibevoice-asr.cpp`, `voxcpm2-gen.cpp`, `voxcpm2-spkenc.cpp`, `yue2-gen.cpp`, `zonos3-gen.cpp`, `zonos3-spkenc.cpp`

They are weight graphs for models we do not drive. We still do not call `mtmd_helper_gen_audio_*`. Confirm each file builds with our include paths. Leave them in the glob so the next mtmd bump does not need a hand-maintained allow-list.

### 1.3 Headers copied to `ios/include` must be refreshed

`scripts/build_apple_external.sh` `copy_header_files` copies `include/llama.h` and `common/chat.h` (among others) into `ios/include`. The podspec searches `ios/include` before `cpp/llama.cpp`. A stale copy wins over the submodule. Re-run the copy as part of the xcframework rebuild. `llama.h` at this commit marks `llama_decode` / `llama_batch_init` deprecated; `chat.h` has the token-aware session fields. `chat-tts-norm.h` is included as `"chat-tts-norm.h"` and resolves through `HEADER_SEARCH_PATHS` to `cpp/llama.cpp/common`. It does not need a copy.

---

## 2. Behavior changes that can compile and still be wrong

### 2.1 Chat parsing now requires token ids

b12638 (`chat : implement token aware parsing`, #29307). Documented in `cpp/llama.cpp/docs/development/parsing.md`.

Special-token matchers are `p.token("<|end|>")` and the like. `common_peg_parser_builder::token` registers the piece against the vocab. At parse time `common_peg_token_parser` succeeds only when `tokens[pos]` is that id. The same characters with `LLAMA_TOKEN_NULL` do not match. Pieces that are not in the vocab still fall back to a literal.

`common_chat_input(std::string)` fills every `tokens` entry with `LLAMA_TOKEN_NULL`.

Our chat path in `cpp/rn-completion.cpp` does:

```cpp
parsed_msg = session.finish(common_chat_input(result.content));
```

The server does not. `tools/server/server-context.cpp` `process_token` appends each sampled piece with its id:

```cpp
slot.generated.append(token_str, result.tok);
```

and `server-task.cpp` passes that `common_chat_input` to `feed` / `finish`.

Parsers that call `p.token` on registered control tokens will miss them on our string-only input and return the markers as plain content. Confirmed call sites include `common/parsers/gemma4.cpp`, `gpt-oss.cpp`, `qwen3-coder.cpp`, `inkling.cpp`, `muse-glimmer.cpp` (`<|channel|>`, `<|end|>`, `<tool_call>`, `</think>`, and the inkling header tokens). A template whose markers are plain literals still parses.

The session constructor already tokenizes the generation prompt when a vocab is passed (`common_chat_input_tokenize` in `common_chat_session::common_chat_session`). We pass `rn_ctx->vocab`, so that half is fine. Only the generated chunk is wrong.

`common_chat_input_tokenize` on the finished string is the wrong fix. It re-tokenizes, and if the pieces do not round-trip it logs and returns a token-less input (`common/chat.cpp`). Build the input the way the server does: `append(piece, token_id)` for each sampled token inside the generation loop, then `session.finish(accumulated)`. One `finish` at the end is enough for the final message. Call `session.feed` on the same accumulated input only if partial tool-call deltas are required during streaming.

The prompt side (`session.prompt()`, `rn_tokenize_chat_prompt`) is separate. KV prefix reuse still has to tokenize the rendered prompt parts. Do not reuse the generated-token vector for that.

### 2.2 `llama_decode` / `llama_batch_init` are deprecated, and embeddings stay on them

b12445 (`llama : migrate to llama_batch_ext`, #32356) and b12447 (`llama : deprecate old batch API`, #32306).

`llama_batch_init`, `llama_batch_free`, `llama_batch_get_one`, `llama_encode`, and `llama_decode` are still in `include/llama.h`, wrapped in `DEPRECATED(...)`. The hint is `llama_process` with `llama_batch_ext`. `llama.h` still says `get_embeddings()` / `get_logits()` for `llama_batch_ext` are TODO.

`cpp/LlamaCppModel.cpp` embedding path calls `llama_batch_init` / `llama_decode` / `llama_get_embeddings_ith` / `llama_get_embeddings_seq` and says the extended-batch getters are not upstream yet. That comment is still true. Leave this path on `llama_batch`. The deprecation is a warning. Our Android and podspec flags do not promote it to an error.

Text ingest and the generation step already use `common_batch` + `llama_process` (`cpp/rn-completion.cpp`). `common_batch` gained `decision_group` and `state_row` (`common/common.h`). Both default so that existing `add` / `get()` call sites keep working. No change on that path.

`llama_process` return codes are unchanged: `0` success, `1` no KV slot, `2` aborted, `-1` invalid batch, `< -1` fatal. Our loops already treat non-zero as failure and rewind KV after an abort. Keep that.

### 2.3 `fit_params` still defaults to true

`common_params::fit_params` is still `true`. `common_init_from_params` still calls `common_fit_apply` unless it is set false. `common_fit_params` now takes `const std::vector<std::string> & paths` instead of one `const char *` (split GGUF). `common_init_from_params` loads through new `common_model_load`: one path calls `llama_model_load_from_file`, more than one calls `llama_model_load_from_splits`. We set a single `model.path` and leave `shards` empty, so load stays a single file.

`llama_context_params` gained `moe_cache_cpu_share` (default 0, all MoE cache misses uploaded). `common_context_params_to_llama` copies it. We leave `moe_cache_size` at 0. `defrag_thold` was removed from `llama_context_params`. We never set it.

`n_gpu_layers` can still be rewritten by the fitter when the device-memory probe disagrees with the JS value. Log `n_gpu_layers` and `n_ctx` after `common_init_from_params` on a device build and keep the existing decision (fit off on mobile, or fit on with the probe checked). That decision was already open at b12177. This bump does not close it.

### 2.4 Multimodal eval signatures we call are unchanged

Checked against `tools/mtmd/mtmd.h` and `mtmd-helper.h`:

- `mtmd_tokenize`, `mtmd_encode_chunk`, `mtmd_batch_init` / `add_chunk` / `encode` / `get_output_embd` / `free`
- `mtmd_helper_eval_chunk_single`, `mtmd_helper_decode_image_chunk`, `mtmd_helper_eval_chunks`
- `mtmd_helper_bitmap_init_from_file` / `_from_buf`, `mtmd_helper_init_opt_default`
- `mtmd_init_from_file`, `mtmd_context_params_default`

`mtmd_encode_chunk` stores its output on the context and must not be called from two threads at once. `mtmd_batch_encode` stores its output on the batch and is marked thread-safe. Our `mtmd_eval_chunks_batched` is single-threaded. Keep it that way.

`mtmd_gen_inp` grew fields (`speed`, `n_steps`, `cfg_scale`, `backbone_state`, ...). We do not construct that struct. New `mtmd_gen_audio_type` enumerators are append-only.

### 2.5 `reasoning_effort = none` is a server concern

b12651's range includes `chat : pass reasoning_effort = none if supported` (#32763). The server, in `tools/server/server-common.cpp`, maps the OAI field: `"none"` is erased from the kwargs when the template's `reasoning_effort_levels` does not list it, and a level in `reasoning_effort_rejected` becomes a 400. We do not parse an OAI `reasoning_effort` field. We set `enable_thinking` from `reasoning_budget` in `PureCppImpl.cpp`. Leave that mapping. Template caps are available as `common_chat_templates_get_caps` (JSON, already the case at b12177) if a later change wants the level list in JS.

---

## 3. Build scripts, versions, and what the rebuilt binaries pick up

`LLAMA_VERSION_MAJOR` moved from 0 to 1 and `LLAMA_VERSION_MINOR` from 9 to 0 (b12352, #32190). `GGML_VERSION_MAJOR` moved from 0 to 1 and `GGML_VERSION_MINOR` from 29 to 0 (same train, ggml/1690). `cpp/build-info.cpp` supplies our `llama_build_number` / `llama_commit`. Keep that as the module's build identity. Do not also compile `common/build-info.cpp`.

`ggml_ftype` is deprecated and kept until ggml 2 (`ggml.h`, b12590 / #32647). We do not use it.

Rebuild from `f5e6915c28f10330885fcb9243a93ae98d3115ab`:

- Android: `scripts/build_android_gpu_backend.sh`, then `scripts/build_android_ggml_gpu_backends.sh`, then `scripts/build_android_external.sh`, so `android/src/main/jniLibs/<abi>/` matches this commit.
- iOS: `scripts/build_apple_external.sh` (`build-xcframework.sh ios-sim ios-device`). Do not download the GitHub release zip. The published zip still has no iOS simulator slice, and `package.json` ships `ios/` to npm consumers.

Flags that still exist and should stay:

- `GGML_BACKEND_DL`, `GGML_CPU`, `GGML_CPU_ALL_VARIANTS`, `GGML_CPU_KLEIDIAI`
- `GGML_OPENCL`, `GGML_OPENCL_USE_ADRENO_KERNELS` (default ON upstream)
- `GGML_VULKAN` and the debug/validate toggles we already set OFF
- `GGML_HEXAGON`
- `GGML_METAL` for the iOS xcframework only (`GGML_METAL=OFF` on Android)
- `GGML_NATIVE=OFF` for the Android cross build

b12399 (`vulkan: Explicitly include SPIRV-Headers`, #31416) made `ggml-vulkan` `find_package(SPIRV-Headers CONFIG REQUIRED)` and added cache var `GGML_VULKAN_SPIRV_INCLUDE_DIR`. `scripts/build_android_ggml_gpu_backends.sh` and `scripts/build_android_external.sh` already stage the headers and pass `SPIRV-Headers_DIR`. Keep that. Vulkan stays disabled in `android/CMakeLists.txt` because of emulator crashes. That is our policy.

The rebuilt GPU libraries include, with no bridge change:

- OpenCL: Adreno coopmat GEMM (Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K, b12600 area), decomposed flash attention, A7 binary kernels. `GGML_OPENCL_USE_ADRENO_BIN_KERNELS` defaults OFF. Leave it off. We do not ship the binary kernel library.
- Vulkan: subgroup arithmetic disabled on Qualcomm's driver (b12558, #30826), plus the other Adreno/Qualcomm shader fixes in this range. They matter only if a later build turns `GGML_VULKAN` on.
- Hexagon: `libhtp` loads without the legacy env vars (b12197, #31781), plus HVX kernels (TIMESTEP_EMBEDDING, SSM_SCAN, IQ2_XXS, NVFP4, Q5_0/Q5_1). Our `dlopen` of `libggml-hexagon.so` and `ggml_backend_init` is unchanged. `ggml_backend_load_all()` still scans the filesystem and is still wrong inside an APK. Keep `load_android_backends()`.
- KV cache type `GGML_TYPE_F8_E4M3` is accepted by the CLI (`--cache-type-k/v`, #28392, b12274). `common_params::cache_type_k/v` still default to `GGML_TYPE_F16`. We do not expose the CLI flag. Leave the default.

`ggml_backend_init` and `ggml_backend_score` are still the exported symbols `PureCppImpl.cpp` loads with `dlsym`.

---

## 4. What stayed compatible (do not rewrite)

Checked against the symbols `cpp/*.h` and `cpp/*.cpp` actually call (excluding `cpp/llama.cpp/`):

- `llama_model_load_from_file`, `llama_model_free`, `llama_free`, `llama_model_default_params`, the `llama_model_params` fields we set (`n_gpu_layers`, `load_mode`)
- `llama_model_n_ctx_train`, `llama_model_n_embd`, `llama_model_n_embd_out`, `llama_model_n_layer`, `llama_model_n_params`, `llama_model_size`, `llama_model_desc`, `llama_model_get_vocab`, `llama_model_meta_val_str`, `llama_model_meta_key_str`
- `llama_backend_init`, `llama_supports_gpu_offload`, `llama_set_n_threads`, `llama_set_embeddings`, `llama_set_abort_callback`
- `llama_tokenize`, `llama_vocab_eos`, `llama_vocab_is_eog`, `llama_vocab_n_tokens`, `llama_vocab_get_add_bos`
- `llama_n_ctx`, `llama_n_batch`, `llama_get_memory`, `llama_memory_clear`, `llama_memory_seq_rm`, `llama_memory_seq_add`, `llama_memory_can_shift`
- `llama_get_embeddings_ith`, `llama_get_embeddings_seq`, `llama_pooling_type`, `llama_perf_context`, `llama_perf_context_reset`
- `llama_process`, `common_batch`
- `common_init_from_params`, `common_params_sampling` fields we assign (`temp`, `top_k`, `top_p`, `min_p`, penalties, `seed`, `grammar`, `grammar_lazy`, `grammar_triggers`, `preserved_tokens`, `reasoning_budget_*`)
- `common_sampler_init` / `sample` / `accept` / `reset` / `free`
- `common_tokenize`, `common_token_to_piece`, `common_embd_normalize`
- `common_chat_templates_init`, `common_chat_templates_apply`, `common_chat_session` (`apply_sampling`, `prompt`, `additional_stops`, `thinking_start_tag`, `thinking_end_tags`, `feed`, `finish`)
- `common_chat_msgs_parse_oaicompat`, `common_chat_tools_parse_oaicompat`, `common_chat_tool_choice_parse_oaicompat`
- `common_reasoning_format_from_name`, `common_chat_format_example`
- `common_grammar`
- mtmd entry points listed in section 2.4
- Android `dlsym("ggml_backend_init")` and `dlsym("ggml_backend_score")`

Removed in this range, and not called by the bridge: `llama_load_model_from_file`, `llama_free_model`, `llama_new_context_with_model`, `llama_n_ctx_train` / `llama_n_embd` / `llama_n_layer` / `llama_n_vocab` (the old names), `llama_set_warmup`, `llama_get_state_size` and the other pre-`llama_state_*` session helpers, `llama_token_get_text` / `llama_token_is_eog` / `llama_token_bos` and the rest of that alias set, `llama_sampler_init_grammar_lazy`, `llama_add_bos_token`. `ggml_diag_mask_inf` and `ggml_diag_mask_zero` are gone from `ggml.h`. We do not call them.

`common_chat_session_params` gained an optional `tokenize` callback used when there is no vocab (tests). We pass a vocab. Leave the callback unset.

---

## 5. Release items that stay out of the bridge

Present in the b12177..b12651 notes, and intentionally not wired:

- Speculative variants `COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH2` and `DRAFT_LILICORR`, NextN embedding APIs (`llama_set_embeddings_nextn`, `llama_set_ctx_state_src`, `llama_batch_ext_add_embd_row`). We do not create a draft context.
- TTS / ASR / music pipelines: Kokoro, VoxCPM2, Zonos3, Fish S2, MiniMax Music 3, YuE2, Tiny-Voca, Pocket TTS resampler, `common_chat_get_tts_prompt` / `common_chat_get_asr_prompt` signature extensions, `mtmd_helper_resampler`, `mtmd_gen_audio_g2p`. Model graphs compile (section 1.2). Nothing in `cpp/rn-*.cpp` calls them.
- Server-only: router / MCP proxy, realtime full-duplex (Hibiki), `/v1/audio/speech`, grammar+`ignore_eos` 400, offset pagination. Our completion path is not the HTTP server.
- Web UI (`LLAMA_BUILD_UI` description change, `npx @ggml-org/llama-chat`). We do not build the UI.
- `llama_model_load_from_splits` / `common_params_model::shards`. One GGUF path from JS. Split loading can wait until the public API grows a shard list.

---

## 6. Suggested order for the implementation plan

1. Add `common/chat-tts-norm.cpp` to `cpp/rn-common-sources.cmake` and `RNLlamaCpp.podspec`.
2. On the podspec, replace the `clip*` lines with `mtmd-graph`, `mtmd-model`, `mtmd-subcontext`, and `mtmd-impl.h`. Add `vendor/hash/hash.cpp` and `vendor/hash/sha256/sha256.c`. Leave `MTMD_VIDEO` unset and leave `mtmd-helper-gen.cpp` out.
3. In `run_chat_completion`, accumulate `common_chat_input` with `append(piece, token_id)` and pass that to `session.finish`. Keep the string-only constructor off the parse path.
4. Leave the embedding path on `llama_batch_init` + `llama_decode`.
5. Rebuild Android prebuilts and the iOS xcframework from `f5e6915c28f10330885fcb9243a93ae98d3115ab`, and re-run `copy_header_files`. Confirm `fit_params` still does what the b12177 plan decided for `n_gpu_layers`.
6. Run the checks below before calling the bump done.

## 7. Checks the plan should require

- Android arm64-v8a and iOS (device + simulator slices both present in the xcframework) configure and link. The iOS compile must not reference `clip.cpp`. The link must resolve `common_chat_tts_norm_add_builtins` and `hash_sha256_hex`.
- Text completion: short prompt, prompt longer than `n_batch`, context shift on a transformer, abort mid-generation then another request.
- Tool call on a `p.token` parser (Gemma 4, GPT-OSS, or Qwen3-Coder) and on a template whose markers are literals. The special-token case must return a tool call, not the marker text inside `content`.
- Thinking model whose end tag is a registered special token (`</think>` or `<|end|>`), with budget `0`, a positive budget, and the JS value used for unlimited.
- Embeddings: one sequence, pooling and non-pooling if both are supported by the public API. This is the path that still calls deprecated `llama_decode`.
- One image and two images (the custom `mtmd_batch` path).
- KV prefix reuse across two chat turns with stable message ids, and a third turn that edits an earlier message.
- Android: CPU backend variant loads, OpenCL loads when `libggml-opencl.so` is in the APK, Vulkan remains off in the app CMake, Hexagon still optional.
- `ios/include/llama.h` matches `cpp/llama.cpp/include/llama.h` after the header copy (`llama_decode` is the deprecated declaration).

# Migration notes: llama.cpp b12651 → b12780

Status: the version pin and the `cpp/llama.cpp` checkout are already on `b12780`. The bridge does not compile against that checkout. The arm64 CPU libraries in `android/src/main/jniLibs/arm64-v8a/` were written on 2026-11-26 and contain the string `1.1.0-dev`. `libggml-opencl.so`, `libggml-hexagon.so`, and `libggml-htp-v*.so` are absent from that directory and from `prebuilt/gpu/`. A phone build from this tree is CPU-only until the GPU scripts run. Host tests were not re-run for this note.

| | From | To |
| --- | --- | --- |
| Tag | `b12651` | `b12780` |
| Commit | `f5e6915c28f10330885fcb9243a93ae98d3115ab` | `09c1410642fa0395ca234571980093bc5c7f2ef5` |
| Date | 2026-11-23 | 2026-11-26 |
| Commit count | | 129 |
| llama.cpp version | 1.0.0 | 1.1.0 (`LLAMA_BUILD_IS_DEV` default ON, so the binary string is `1.1.0-dev`) |
| ggml version | 1.0.0 | 1.1.1 |

`scripts/used_version.sh` already sets `LLAMA_CPP_TAG=b12780` and `LLAMA_CPP_COMMIT=09c1410642fa0395ca234571980093bc5c7f2ef5`. `git describe` on the submodule is `b12780` (`v1.1.0-6-g09c141064`). ggml's own `CMakeLists.txt` is `1.1.1` (bump `2a59da5bf`, after the `1.1.0` bump `312504361`). Do not edit files under `cpp/llama.cpp/`.

Scope stays the mobile bridge: one context, one sequence, Android OpenCL / optional Hexagon / dynamic CPU, iOS Metal. Server, the web UI, speculative decoding, and TTS generation stay out.

Release notes checked: every commit subject from `b12651` to `b12780`, plus the public diffs of `include/llama.h`, `common/chat.h`, `common/common.h`, `ggml/include/ggml.h`, `ggml/include/ggml-backend.h`, the OpenCL and Hexagon CMake files, and `docs/backend/snapdragon/README.md` at this commit. The items below are the ones that change this library's compile, its chat path, or whether a Snapdragon phone can actually run the GPU and NPU libraries we ship.

---

## 1. Compile break: named `tool_choice`

b12774 area, commit `5bd90fa0e` (`common : implement named tool choice`, #31564).

`common_chat_tool_choice` is no longer an enum. It is a struct with a mode and an `allowed` name list. The old enumerator names still exist, on `enum common_chat_tool_choice_mode`:

- `COMMON_CHAT_TOOL_CHOICE_AUTO`
- `COMMON_CHAT_TOOL_CHOICE_REQUIRED`
- `COMMON_CHAT_TOOL_CHOICE_NONE`

The struct has a converting constructor from that enum and `operator==` / `operator!=` against the enum. This comparison in `cpp/rn-completion.cpp` still type-checks:

```cpp
template_inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE
```

The parser does not. The signature in `common/chat.h` is now:

```cpp
common_chat_tool_choice common_chat_tool_choice_parse_oaicompat(
    const common_json & tool_choice,
    const std::vector<common_chat_tool> & tools);
```

`common_json` accepts a `std::string` (it becomes a JSON string). The second argument has no default. These two calls are a hard compile error (one argument, the old signature):

- `cpp/rn-completion.cpp` (`common_chat_tool_choice_parse_oaicompat(options.tool_choice)`)
- `cpp/tests/test-completion.cpp` (`common_chat_tool_choice_parse_oaicompat("auto")`)

The server call in `tools/server/server-common.cpp` is the pattern to follow. Parse tools first, then pass the raw JSON value and the parsed tool vector:

```cpp
inputs.tools       = common_chat_tools_parse_oaicompat(tools);
inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(tool_choice, inputs.tools);
```

`common_chat_tool_choice_parse_oaicompat` accepts:

- `"auto"`, `"none"`, `"required"`
- `{ "type": "function", "function": { "name": "<one supplied tool>" } }` (mode `required`, `allowed` is that name)
- `{ "type": "allowed_tools", "allowed_tools": { "mode": "auto"|"required", "tools": [ { "type": "function", "function": { "name": "..." } } ] } }`

A name that is empty or that does not match exactly one entry in `tools` throws `std::invalid_argument`.

Our JS and C++ currently throw that information away. `src/NativeRNLlamaCpp.ts` types `tool_choice` as `string | 'auto' | 'none'`. `CompletionOptions::tool_choice` in `cpp/rn-utils.h` is a `std::string` defaulting to `"auto"`. `LlamaCppModel::parseCompletionOptions` (`cpp/LlamaCppModel.cpp`) stores a string, and when the value is an object it writes `"required"` and drops `function.name`. That comment says per-function enforcement does not exist. It does now. Passing the object through is the change.

Keep `options.tools` as JSON. Store `tool_choice` as JSON too (`common_json` or the same `json` type `tools` already uses), default `"auto"`. In `run_chat_completion`, call the parser only after `template_inputs.tools` is filled, and pass that vector as the second argument. An empty `tool_choice` stays the default `"auto"`. The KV render identity in `rn-completion.cpp` concatenates `options.tool_choice` as a string. Dump the JSON so a named choice and a plain `"required"` do not share a prefix.

Widen the TypeScript field so a caller can pass the OpenAI object as well as `"auto" | "none" | "required"`. Update the host test that calls the one-argument parser.

`common_chat_foreach_tool` is new and lives in `common/chat.cpp`, which both source lists already compile. No new source file for this change.

---

## 2. Token and chat handling that is already on the server path

Token-aware parsing landed at b12638 (#29307), inside the previous bump, and it is still the contract at b12780. `docs/development/parsing.md` in this checkout still says a special-token matcher succeeds only when `tokens[pos]` is that id. `common_chat_input(std::string)` still fills every id with `LLAMA_TOKEN_NULL`.

`cpp/rn-completion.cpp` already does the server thing:

```cpp
state.generated.append(token_text, token_id);
// ...
parsed_msg = session.finish(result.generated);
```

`common/chat.h` did not change `common_chat_input`, `append`, or `common_chat_session::finish` in this range. Leave this path. Do not go back to `session.finish(common_chat_input(result.content))`.

`rn_tokenize_chat_prompt` in `cpp/rn-utils.h` still tokenizes each prompt part with `common_tokenize`, and parses specials only on template parts (`!part.is_input`). That matches the session prompt. Leave it.

`common_chat_session`'s constructor, `common_chat_templates_init`, `common_chat_templates_apply`, `common_chat_msgs_parse_oaicompat`, and `common_chat_tools_parse_oaicompat` kept their signatures (`common/chat.h` diff in this range is the tool-choice struct, the foreach helper, and the parser signature above).

---

## 3. Tokenizer behavior inside libllama (no bridge rewrite)

Two vocab commits change which ids `common_tokenize` returns for some models. The function signature in `common/common.h` is unchanged. Our call sites stay source-compatible. The rebuilt `libllama.so` / xcframework is what changes the ids.

- `98f870c22` (`llama : honor metaspace_prepend_scheme for SPM`, #31986)
- `532d67426` (`vocab : implement Metaspace prepend_scheme always for BPE`, #33038)

BPE with `prepend_scheme = always` now starts each non-newline chunk with an escaped space. SPM follows the GGUF metaspace scheme instead of always prepending. `2a487782f` speeds up BPE (`vocab : speed up BPE tokenization`, #32859) without a signature change. `src/unicode.cpp` grew the regex splitter used by the BPE pre-tokenizer (`5a09582d8` / Raon path and the metaspace work). `common/unicode.cpp` is already on both source lists.

A chat prompt tokenized under b12651 and the same prompt tokenized under b12780 can differ for those vocabs. KV prefix reuse compares token ids, so the first request after the upgrade re-encodes instead of reusing a prefix built by the old binary. That is the correct outcome. There is no on-disk token cache in the bridge (`llama_state_*` is only read for its size in `cpp/tests/test-completion.cpp`).

`llama_tokenize`, `llama_vocab_eos`, `llama_vocab_is_eog`, `common_token_to_piece` are unchanged.

---

## 4. ggml 1.1.0 / 1.1.1, and what the bridge must not chase

ggml 1.1.0 (`312504361`) and 1.1.1 (`2a59da5bf`) are inside this checkout. The public `ggml.h` diff that matters:

- `ggml_gated_delta_net` is deprecated. `ggml_gated_delta_net_ext` adds pending-token replay (`d35dc4d9e`, #31669). The bridge does not call either. Model graphs inside llama.cpp do. OpenCL (`12d47ba2f`) and Hexagon (`e0f884d40`) in this range implement the new op. A rebuilt backend `.so` picks that up.
- `ggml_flash_attn_ext_add_indexed_bias` / `ggml_flash_attn_ext_has_indexed_bias` (`239242322`, `6d5f64872`). Used by Inkling 2 graphs. The bridge does not call them.
- `ggml_backend_dev_caps` gained `fast_async_host_input` (`62d6c8cb3`). We never construct that struct. Rebuilding ggml and llama together keeps the layout consistent.
- `ggml_ftype` stays deprecated until ggml 2. We still do not use it.
- `if (MATH_LIBRARY)` replaced `if (DEFINED MATH_LIBRARY)` in `ggml/src/CMakeLists.txt` (`7d9b931df`). `MATH_LIBRARY=NOTFOUND` no longer gets linked. Leave that upstream change alone.
- `llama_model_quantize_params` grew `target_bpw`, `target_size`, `state_file`, an imatrix pointer, and an abort callback (`30ec35ae2`). We do not call the quantize API.

`cpp/build-info.cpp` remains the module's `llama_build_number` / `llama_commit`. Do not also compile `common/build-info.cpp`.

iOS Metal: `1f3ea67fb` raised the iPhone command-buffer count, and `8d35ba4f7` reverted it in this same range. No podspec edit. The xcframework still has to be rebuilt so the other Metal kernels in the range (NVFP4, short mat-vec, top-k scratch) are in the binary.

New mtmd graph `tools/mtmd/models/hy-v4.cpp` (`c94333133`) is picked up by the Android glob in `cpp/rn-common-sources.cmake` and by the podspec `models/*.{h,cpp}` glob. It uses the existing mtmd graph helpers. Leave it in the glob. We still do not call `mtmd_helper_gen_audio_*` or `mtmd-helper-gen.cpp`.

---

## 5. Android: the phone GPU and the Snapdragon NPU

Upstream reference at this commit: `cpp/llama.cpp/docs/backend/snapdragon/README.md`. Snapdragon has three backends: CPU, Adreno via OpenCL (`GPUOpenCL`), and Hexagon NPU (`HTP0`, which llama.cpp treats as a GPU for `-ngl`). The README's Android preset is NDK r28, `HEXAGON_SDK_ROOT=/opt/hexagon/6.7.0.0`, `GGML_OPENCL=ON`, `GGML_HEXAGON=ON`. `scripts/used_version.sh` already pins NDK `28.2.13676358` and Hexagon SDK `6.7.0.0`. Keep those.

`cpp/llama.cpp/examples/llama.android/app/src/main/AndroidManifest.xml` sets `android:extractNativeLibs="true"`. Our example app does not. `android/build.gradle` uses Android Gradle Plugin 8.7.2, whose default `packaging.jniLibs.useLegacyPackaging` is false, so native libraries stay inside the APK.

### 5.1 What is on disk now

`android/src/main/jniLibs/arm64-v8a/` has `libllama.so`, `libggml.so`, `libggml-base.so`, and seven CPU variants (`armv8.0_1`, `armv8.2_1`, `armv8.2_2`, `armv8.6_1`, `armv9.0_1`, `armv9.2_1`, `armv9.2_2`). It does not have `libggml-opencl.so`, `libggml-vulkan.so`, `libggml-hexagon.so`, or any `libggml-htp-v*.so`. `prebuilt/gpu/` is empty. `android/CMakeLists.txt` copies OpenCL and Hexagon into the app output only when those files already exist in `jniLibs`. An Expo or example build from this tree never loads a GPU.

Rebuild order is unchanged: `scripts/build_android_gpu_backend.sh`, then `scripts/build_android_ggml_gpu_backends.sh`, then `scripts/build_android_external.sh`. Hexagon stays opt-in (`--hexagon` plus `HEXAGON_SDK_ROOT`). CI can keep skipping Hexagon. The phone package is where those `.so` files have to land.

### 5.2 Adreno OpenCL (the phone GPU)

Runtime load is `dlopen("libggml-opencl.so")` then `dlsym(..., "ggml_backend_init")` in `load_android_backends()` (`cpp/PureCppImpl.cpp`). The Android linker finds that soname in the APK. The ICD it then needs is the vendor library `libOpenCL.so`.

`android/src/main/AndroidManifest.xml` already declares `libOpenCL.so` with `android:required="false"`. That entry merges into an Expo app and is what puts the system OpenCL driver in the app linker namespace. Do not ship `libOpenCL.so` in the APK. Same rule as b12651 for `libvulkan.so`.

`GGML_OPENCL_USE_ADRENO_KERNELS` still defaults ON (`ggml/CMakeLists.txt`). Those are the source kernels compiled into `libggml-opencl.so`, including the new MoE coopmat kernels (`e6443bd7e`) and the Q6_K c8 kernel (`af5c58347`). Leave that default ON. That is the GPU implementation a phone should run.

`GGML_OPENCL_USE_ADRENO_BIN_KERNELS` is a new option and it defaults ON (`fd24ee26a`, #31759). The fetch of `adreno-opencl-kernels.dll` runs only for Windows ARM64. On Android the same define makes `ggml-opencl.cpp` `dlopen("libadreno-opencl-kernels.so")`. Qualcomm does not publish that library for Android. A failed `dlopen` falls back to the builtin kernels and logs it. Set both of these on the Android OpenCL configure, in `scripts/build_android_ggml_gpu_backends.sh` and `scripts/build_android_external.sh`, so the phone binary does not depend on a library we do not ship:

```text
-DGGML_OPENCL_USE_ADRENO_BIN_KERNELS=OFF
-DGGML_OPENCL_ADRENO_BIN_KERNELS_FETCH=OFF
```

`d9dae0344` turns Adreno E17 compiler workarounds off unless the driver version matches. That lives inside the rebuilt `libggml-opencl.so`. No bridge flag.

Vulkan stays out of `android/CMakeLists.txt` (the copy step is still commented out because of emulator crashes). A phone that has only Vulkan and no OpenCL ICD stays on CPU. That is the existing policy.

### 5.3 Hexagon NPU (provided, not exercised on CI)

`e0f884d40` (`ggml-hexagon 2.0`, #32582) is inside the backend we build, not a new public C API. `ggml_backend_init` is still the symbol `PureCppImpl.cpp` loads. Skel sonames are unchanged: `libggml-htp-v73.so`, `v75`, `v79`, `v81`, `v85`. `ggml/src/ggml-hexagon/CMakeLists.txt` still calls `build_htp_skel` for those five. The new host file `htp-lower.cpp` is in `ggml_add_backend_library`. The new DSP files (`dma-proxy.c`, `htp-tensor.c`, the split HMX kernels) are in `ggml/src/ggml-hexagon/htp/CMakeLists.txt`. Building the submodule with the existing `--hexagon` flags picks them up. Our copy loop already globs `libggml-htp-*.so`.

Defaults changed inside the backend, with no JS surface:

| | b12651 | b12780 |
| --- | --- | --- |
| `opt_mm_select` | 2 (FP16 HMX) | 3 (hybrid W8A8, FP16 on post-GLU). Multi-session v73 still forces 2 when the env var is unset. |
| `opt_oplower` | absent | 1 (internal FP32 activations lowered to FP16) |
| `opt_oppoll` | 0 (blocking) | 50 ms hybrid poll when the FastRPC driver supports non-blocking read, else 0 |

The README's accuracy ladder is the env vars `GGML_HEXAGON_MM_SELECT` and `GGML_HEXAGON_OPLOWER`. They have to be set before `ggml_backend_init`. We do not add a JS field for them in this bump. A phone that shows bad samples can set the env in a debug build. `GGML_HEXAGON_POWER_MODE` still defaults to `burst`.

How the skel is found (`ggml/src/ggml-hexagon/htp-drv.cpp`, `htp_lib_dir`, and `ggml-hexagon.cpp` `open_session`):

1. `GGML_HEXAGON_LIBHTP_DIR` if set.
2. Otherwise the directory of `libggml-hexagon.so` from `dladdr`, and that directory must contain a `libggml-htp-*.so` file (`std::filesystem::directory_iterator`).
3. FastRPC then opens `file:///<that dir>/libggml-htp-vNN.so`.

`load_android_backends()` also sets `ADSP_LIBRARY_PATH` from `dladdr` of `libggml.so`, and only when that path contains a slash and is a real directory. With `extractNativeLibs=false`, `dladdr` often returns a path inside `base.apk`. That is not a directory, the walk finds nothing, and the NPU session fails. OpenCL can still `dlopen` by soname in that mode. Hexagon cannot.

The example app (`example/android/app/build.gradle`, `example/android/app/src/main/AndroidManifest.xml`) never sets extraction. For the bare example, set `packagingOptions { jniLibs { useLegacyPackaging true } }` and `android:extractNativeLibs="true"` on `<application>`, matching `examples/llama.android`. For an Expo app (this library is the one Expo SDK 58 / React Native 0.88 apps consume; the example itself is a bare React Native app), the same switch is `expo-build-properties`:

```js
["expo-build-properties", { android: { useLegacyPackaging: true } }]
```

NDK r28 already aligns the Android ELF libraries (`libggml-opencl.so`, `libggml-hexagon.so`, `libllama.so`) to 16 KB, which is what Play requires when they are loaded by the Android linker. The HTP skels are Hexagon ELF. FastRPC reads them as files. After a phone install, confirm the five `libggml-htp-v*.so` files exist next to `libggml-hexagon.so` under the app's native library directory, and that `libcdsprpc.so` resolved. `libcdsprpc.so` is already in the manifest as `required="false"`. Keep that. Do not ship `libcdsprpc.so`.

`android/src/main/AndroidManifest.xml` also lists `libggml-opencl.so`, `libggml-vulkan.so`, `libggml-hexagon.so`, and the five HTP sonames as `<uses-native-library>`. That element is only for vendor libraries under `/vendor` or `/system`. Qualcomm does not ship our ggml builds there. `required="false"` lets the app install, and it does not publish the copies from `jniLibs`. The entries that do real work are `libOpenCL.so`, `libvulkan.so`, and `libcdsprpc.so`. Remove the `libggml-*` entries so the manifest stops describing our own libraries as system libraries.

Registration order in `load_android_backends()` is HTP, then OpenCL, then Vulkan, then the best CPU variant. Both HTP and OpenCL report `GGML_BACKEND_DEVICE_TYPE_GPU` (`ggml-hexagon.cpp`, `ggml-opencl.cpp`). We leave `common_params::split_mode` at `LLAMA_SPLIT_MODE_LAYER`. `src/llama.cpp` then puts every GPU device on the model and splits layers by free memory. On a phone where both backends initialize, one model is split across `HTP0` and Adreno. The Snapdragon README selects a single target with `--devices HTP0` or the OpenCL device. We have no device argument in the JS API. After the libs are packaged, read the load log (`offloaded N/M layers` and the device names) on a Snapdragon phone before deciding whether the bridge should pin one device. Do not add that pin in the same change as the parser fix.

`ggml_backend_load_all()` still scans the filesystem and still does nothing useful inside an APK. Keep the manual `dlopen` list.

When `--hexagon` is on, `scripts/build_android_external.sh` passes `CMAKE_C_FLAGS` / `CMAKE_CXX_FLAGS` as a replacement string (`-march=armv8.7a+fp16 ...`). That configure also builds OpenCL and the CPU variants. On the next arm64+Hexagon configure, confirm the NDK toolchain flags are still present and that `libggml-opencl.so` is an Android ELF. If the replacement drops `-fPIC` or the sysroot, append to the toolchain flags instead of replacing them.

### 5.4 CPU variants the loader never opens

`load_android_cpu_backends()` tries four sonames, best first: `armv8.6_1`, `armv8.2_2`, `armv8.2_1`, `armv8.0_1`. The arm64 directory from today's build also contains `libggml-cpu-android_armv9.2_2.so`, `armv9.2_1`, and `armv9.0_1`. Those three are never `dlopen`'d, so a Snapdragon 8 Elite (armv9) stays on the v8.6 library even though we built a newer one. `ggml_backend_score` returns 0 on a CPU that lacks the variant's features, and the loader already closes those handles. Add the three v9 sonames above `armv8.6_1`. The copy loop in `build_android_external.sh` already copies every `libggml-cpu*.so`. The comment a few lines above that loop still says the build produces only the four armv8 names. Update the comment.

`f20c5907c` adds Q4_0/Q8_0 repack kernels for ARM targets without dotprod. That is inside the rebuilt `armv8.0_1` library. No loader change for that commit.

---

## 6. What stayed compatible (do not rewrite)

Checked against the symbols `cpp/*.h` and `cpp/*.cpp` call, excluding `cpp/llama.cpp/`:

- `llama_model_load_from_file`, `llama_model_free`, `llama_free`, `llama_model_default_params`, the `llama_model_params` fields we set (`n_gpu_layers`, `load_mode`)
- `llama_model_n_ctx_train`, `llama_model_n_embd`, `llama_model_n_embd_out`, `llama_model_n_layer`, `llama_model_n_params`, `llama_model_size`, `llama_model_desc`, `llama_model_get_vocab`, `llama_model_meta_val_str`, `llama_model_meta_key_str`
- `llama_backend_init`, `llama_supports_gpu_offload`, `llama_set_n_threads`, `llama_set_embeddings`, `llama_set_abort_callback`
- `llama_tokenize`, `llama_vocab_eos`, `llama_vocab_is_eog`, `llama_vocab_n_tokens`, `llama_vocab_get_add_bos`
- `llama_n_ctx`, `llama_n_batch`, `llama_get_memory`, `llama_memory_clear`, `llama_memory_seq_rm`, `llama_memory_seq_add`, `llama_memory_can_shift`
- `llama_get_embeddings_ith`, `llama_get_embeddings_seq`, `llama_pooling_type`, `llama_perf_context`, `llama_perf_context_reset`
- `llama_process`, `common_batch` (`decision_group` and `state_row` still default)
- `common_init_from_params`, `common_params_sampling` fields we assign
- `common_sampler_init` / `sample` / `accept` / `reset` / `free`
- `common_tokenize`, `common_token_to_piece`, `common_embd_normalize`
- `common_chat_templates_init`, `common_chat_templates_apply`, `common_chat_session`
- `common_chat_msgs_parse_oaicompat`, `common_chat_tools_parse_oaicompat`
- `common_reasoning_format_from_name`, `common_chat_format_example`
- mtmd entry points the bridge calls (`mtmd_tokenize`, `mtmd_encode_chunk`, `mtmd_helper_eval_chunks`, bitmap init). `tools/mtmd/mtmd.h` did not change signature in this range.
- `dlsym("ggml_backend_init")` and `dlsym("ggml_backend_score")`

`common/common.h` gained `ui_agent`. We do not read it.

Embeddings in `cpp/LlamaCppModel.cpp` still call `llama_batch_init`, `llama_decode`, `llama_get_embeddings_ith`, and `llama_get_embeddings_seq`. `include/llama.h` still marks the old batch API deprecated and still says `get_embeddings()` / `get_logits()` for `llama_batch_ext` are TODO (line 1107 at this commit). Leave the embedding path on `llama_batch`. The deprecation stays a warning.

`fit_params` still defaults to true. `log_context_fit` in `cpp/PureCppImpl.cpp` still records when `n_ctx` moves. This bump does not change that policy.

`llama_process` return codes are unchanged: `0` success, `1` no KV slot, `2` aborted, `-1` invalid batch, `< -1` fatal.

---

## 7. Release items that stay out of the bridge

Present in b12651..b12780, and intentionally not wired:

- Inkling 2, Hy4 flash, RAON-1-mini, Domyn, Zaya1, DSpark. Their graphs compile inside libllama and, for Hy4, inside the mtmd glob. Nothing in `cpp/rn-*.cpp` selects them.
- Speculative decoding, NextN, DFlash. We do not create a draft context.
- Server slot scheduler, prompt-cache restore (`679d8677b` ON_DEVICE state fix), MCP image/audio tool results, named `tool_choice` on the HTTP server beyond the parser we already have to call.
- Web UI and `npx llama-chat`.
- CUDA / HIP / SYCL / MUSA BF16 `MUL_MAT` (`09c141064` and the other GPU-vendor commits). Android and iOS do not build those backends.
- Windows-only Adreno binary kernel DLL.
- `llama_model_load_from_splits`. One GGUF path from JS.

---

## 8. Order of work

1. Fix `common_chat_tool_choice_parse_oaicompat` at both call sites. Pass `common_json` plus the parsed tools. Store the JS `tool_choice` value as JSON, including `{ type: "function", function: { name } }` and `allowed_tools`, and put that JSON in the KV identity string. Widen `LlamaCompletionParams.tool_choice`.
2. Leave token accumulation as `generated.append(piece, token_id)` and `session.finish(result.generated)`. Leave embeddings on `llama_batch_init` / `llama_decode`.
3. Android OpenCL configure: `-DGGML_OPENCL_USE_ADRENO_BIN_KERNELS=OFF` and `-DGGML_OPENCL_ADRENO_BIN_KERNELS_FETCH=OFF`. Leave `GGML_OPENCL_USE_ADRENO_KERNELS` at its default ON.
4. Add the three `libggml-cpu-android_armv9*.so` names to `load_android_cpu_backends()`, above `armv8.6_1`.
5. Example app: `useLegacyPackaging true` and `android:extractNativeLibs="true"`. Document the Expo `expo-build-properties` `android.useLegacyPackaging: true` switch for apps that ship Hexagon. Drop the `<uses-native-library>` entries whose names are libraries we ship. Keep `libOpenCL.so`, `libvulkan.so`, and `libcdsprpc.so` at `required="false"`.
6. Rebuild Android prebuilts so `jniLibs/arm64-v8a` contains `libggml-opencl.so` and, when the SDK is present, `libggml-hexagon.so` plus the five HTP skels. Rebuild the iOS xcframework from `09c1410642fa0395ca234571980093bc5c7f2ef5` and re-run the header copy. Vulkan stays out of the app CMake copy step.

---

## 9. Checks

- Host `cpp/tests` configure and link. `test-completion.cpp` calls the two-argument parser. A named `tool_choice` whose `function.name` is one of the supplied tools restricts the grammar to that tool. A name that matches nothing throws.
- `"auto"`, `"none"`, and `"required"` still parse. An object is not collapsed to the string `"required"` before it reaches the parser.
- Text completion still finishes with `session.finish` on the `common_chat_input` that was appended per sampled token. A Gemma 4 / GPT-OSS / Qwen3-Coder tool call returns a tool call. A literal-marker template still parses.
- Embeddings still run. That path still calls deprecated `llama_decode`.
- Android arm64 `jniLibs` contains `libggml-opencl.so` and does not contain `libOpenCL.so`, `libvulkan.so`, or `libadreno-opencl-kernels.so`. `libggml-opencl.so` was built with `GGML_OPENCL_USE_ADRENO_BIN_KERNELS` off.
- On a Snapdragon phone (not the emulator): log shows `libOpenCL.so` resolved and at least one Adreno device, `n_gpu_layers > 0` offloads layers, and a missing OpenCL ICD falls back to CPU without aborting.
- On that same phone, with Hexagon packaged: `libggml-htp-vNN.so` for the device arch is a real file next to `libggml-hexagon.so`, `libcdsprpc.so` resolves, and the log shows `HTP0`. If both HTP and OpenCL initialize, record which devices received layers before changing `split_mode`.
- CPU: a v9 device scores an `armv9` variant; an older arm64 device scores `armv8.6_1` or lower and does not crash on the v9 `dlopen` (score 0, handle closed).
- `ios/include/llama.h` matches `cpp/llama.cpp/include/llama.h` after the header copy. `llama_decode` is still the deprecated declaration.

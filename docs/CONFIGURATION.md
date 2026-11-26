# Configuring models: text, thinking, vision, audio, and devices

This guide is for app developers who load GGUF models with `@novastera-oss/llamarn`. It covers what each option actually does in the native bridge (llama.cpp `b12780`), recipes for each model type, and per-device settings.

Every default below comes from the bridge source (`cpp/PureCppImpl.cpp`, `cpp/rn-completion.cpp`, `cpp/LlamaCppModel.cpp`). Where the TypeScript types allow more than native reads, this guide says so.

- [1. Load flow](#1-load-flow)
- [2. `initLlama` parameters](#2-initllama-parameters)
- [3. Recipes](#3-recipes)
- [4. Device presets](#4-device-presets)
- [5. Prompt pacing and thermal control](#5-prompt-pacing-and-thermal-control)
- [6. Completion parameters](#6-completion-parameters)
- [7. Streaming](#7-streaming)
- [8. Troubleshooting](#8-troubleshooting)

---

## 1. Load flow

1. Call `loadLlamaModelInfo(modelPath, mmprojPath?)` first. It reads the GGUF on the CPU, frees it, and returns sizing hints.
2. Build the `initLlama` options from those hints, the model card, and the device.
3. Keep the returned context for the session. Call `release()` when you are done. An unreleased model keeps its RAM/VRAM until the process dies.

```ts
import { loadLlamaModelInfo, initLlama } from '@novastera-oss/llamarn';

const info = await loadLlamaModelInfo(modelPath);

const ctx = await initLlama({
  model: modelPath,
  n_ctx: 8192,
  n_gpu_layers: info.optimalGpuLayers,
  chunk_size: info.suggestedChunkSize,
  is_cpu_only: info.isCpuOnly,
  use_jinja: true,
});
```

What `loadLlamaModelInfo` returns that matters for configuration:

| Field | Meaning |
| --- | --- |
| `optimalGpuLayers` | Suggested `n_gpu_layers`. 0 when no GPU backend is available (always 0 on the iOS Simulator). Otherwise it fits a share of device RAM (20 % at ≥ 8 GB, 18 % at ≥ 6 GB, 15 % below), uses 80 % of that, subtracts the mmproj size when you pass `mmprojPath`, and is capped at 75 % of the model's layers. |
| `suggestedChunkSize` | 32 when `optimalGpuLayers` is 0, else 128. Pass it as `chunk_size`. |
| `isCpuOnly` | `optimalGpuLayers == 0`. Pass it as `is_cpu_only`. |
| `n_context` | The model's training context. `n_ctx` above this needs RoPE scaling the model was built for. |
| `kvBytesPerToken` | f16 K+V bytes for one token (absent for hybrid / MLA models). `n_ctx × kvBytesPerToken` is the KV cache size. `initLlama` does not clamp `n_ctx` for you. |
| `totalMemoryBytes`, `availableMemoryMB` | Device memory, for your own `n_ctx` ceiling. |
| `samplingDefaults` | Sampling values the model author stored in the GGUF (`general.sampling.*`), when present. Native uses them automatically when a completion omits that field (see [section 6](#6-completion-parameters)). |
| `architecture` | e.g. `qwen3`, `qwen35`, `gemma3`. Useful for per-family presets. |

---

## 2. `initLlama` parameters

### 2.1 Model, context, and memory

| Key | Default | What it does |
| --- | --- | --- |
| `model` | required | Path to the `.gguf`. `file://` prefixes are stripped. |
| `n_ctx` | `2048` | Context window in tokens (prompt plus reply). llama.cpp rounds it **up** to a multiple of 256 (ask for 3000, get 3072), and the KV cache is allocated for the rounded size. With `fit_params` (on, the upstream default) it may also be lowered to fit memory. A change is logged as `fit changed n_ctx from A to B`. |
| `n_batch` | `512` | Logical batch for prompt processing. One prompt chunk never exceeds it. |
| `n_ubatch` | `512` | Physical micro-batch. Lower it (128–256) on low-memory GPUs. |
| `n_threads` | cores − 2 (cores − 1 below 4, min 1) | CPU threads. Can be changed at runtime with `setNThreads`. |
| `n_gpu_layers` | `0` | Layers offloaded to the GPU. Ignored when no GPU backend is present. `> 0`: if GPU init fails, the bridge retries on CPU. `-1`: the llama.cpp default, where device fit picks the count. That path has no CPU retry. |
| `use_mmap` / `use_mlock` | unset | Leave both unset for `LLAMA_LOAD_MODE_AUTO`, which falls back on its own where the backend cannot mmap (Android OpenCL and Hexagon always). Setting either selects an explicit mode. |
| `n_keep` | `0` | Init-time keep. Prefer the per-completion `n_keep` ([section 6](#6-completion-parameters)). |
| `embedding` | `false` | Embedding mode ([recipe 3.6](#36-embeddings)). |
| `seed` | random | Default RNG seed. A completion can override it. |
| `lora_adapters` | none | `[{ path, scale? }]`, applied at load. |
| `verbose` | `0` | llama.cpp log level. |

### 2.2 RoPE and YaRN: leave them unset

| Key | Default | Meaning of the default |
| --- | --- | --- |
| `rope_freq_base` | `0` | Use the GGUF's trained base |
| `rope_freq_scale` | `0` | Use the GGUF's scaling |
| `yarn_ext_factor`, `yarn_attn_factor`, `yarn_beta_fast`, `yarn_beta_slow` | `-1` | Use the GGUF values |

> **Changed in this release.** Earlier builds always sent `rope_freq_base = 10000` and `yarn_ext_factor = 1`, which overrode every model's trained values. Qwen2.5 / Qwen3 train with a base of 1,000,000 and Llama 3 with 500,000. With 10000 they produce plausible text for a few hundred tokens, then drift, repeat, or forget the prompt, and thinking models get it worst. If your app copied those numbers into its own config, delete them. Set these keys only to extend context on a model whose card tells you exact values.

### 2.3 Chat template and thinking

| Key | Default | What it does |
| --- | --- | --- |
| `use_jinja` | `false` | Render the model's Jinja chat template. **Required for tool calls and for the reasoning parser.** Turn it on for every modern chat model. |
| `chat_template` | GGUF template | Override with Jinja template source (the full template text). If the template throws, the bridge falls back to plain-content rendering, then to the non-Jinja path. |
| `reasoning_budget` | `0` | `0` (or omitted) renders the template with `enable_thinking = false`. `-1` turns thinking on with no limit. `> 0` turns it on and caps thinking tokens with the reasoning-budget sampler. Thinking can only be turned on at init. |
| `reasoning_format` | `'auto'` | `'auto'` (same as `'deepseek'`): thinking goes to `message.reasoning_content` and `content` holds only the answer. `'deepseek-legacy'`: the same split in the final message. It differs only in how llama-server streams, so use `'auto'`. `'none'`: no split, so the raw `…</think>` ends up in `content`. **Changed in this release**: the default used to be `'none'`. Any other string makes `initLlama` throw. |
| `parallel_tool_calls` | template | Allow more than one tool call per reply. Omitted = what the chat template supports (llama-server's default). `false` limits the grammar to one call. |

### 2.4 Multimodal

| Key | Default | What it does |
| --- | --- | --- |
| `mmproj` | none | Path to the projector `.gguf` that matches the text model. A load failure does not fail `initLlama`: check `isMultimodalEnabled()`. It runs on the GPU when `n_gpu_layers != 0`. |
| `image_marker` | `<__media__>` | Media placeholder used inside the rendered prompt. Change it only if your prompts contain that literal. |
| `capabilities` | `[]` | Gates the dedicated methods: `image-encode` → `embedImage`, `audio-transcribe` → `transcribeAudio`, `vision-reasoning` → `visionReasoning`, and the matching `runOnFrame` capability. `completion` with `image_url` / `audio_url` parts only needs a loaded `mmproj`. `vision-chat` is accepted and gates nothing today. |

### 2.5 Prompt pacing

`chunk_size`, `is_cpu_only`, `prompt_chunk_gap_ms`: see [section 5](#5-prompt-pacing-and-thermal-control).

### 2.6 Grammar

`grammar` (GBNF) is a per-completion option, not an init option ([section 6](#6-completion-parameters)).

---

## 3. Recipes

### 3.1 Text chat (non-thinking model)

```ts
const ctx = await initLlama({
  model,
  n_ctx: 4096,
  n_gpu_layers: info.optimalGpuLayers,
  chunk_size: info.suggestedChunkSize,
  is_cpu_only: info.isCpuOnly,
  use_jinja: true,
});

const r = await ctx.completion({
  messages: [
    { id: 'sys', role: 'system', content: 'You are concise.' },
    { id: 'u1', role: 'user', content: 'Hi' },
  ],
  n_predict: 512,
});
const answer = r.choices?.[0]?.message.content;
```

Give each message a stable `id` so the next turn reuses the KV cache ([section 6](#6-completion-parameters)).

### 3.2 Thinking models (Qwen3, Qwen3.5, QwQ, DeepSeek-R1 distills)

```ts
const ctx = await initLlama({
  model,
  n_ctx: 8192,              // thinking replies are long: 8k minimum, 16k+ if memory allows
  n_gpu_layers: info.optimalGpuLayers,
  chunk_size: info.suggestedChunkSize,
  is_cpu_only: info.isCpuOnly,
  use_jinja: true,          // required: the reasoning parser comes from the template
  reasoning_budget: -1,     // thinking on; or e.g. 1024 to cap it
  reasoning_format: 'auto', // the default; shown for clarity
  // no rope_* / yarn_* keys
});

const r = await ctx.completion({
  messages,
  // Qwen3 model card, thinking mode. Do not use greedy decoding with thinking on.
  temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0,
  n_predict: 4096,
});
const { content, reasoning_content } = r.choices![0].message;
```

Notes:

- **Hybrid models** (Qwen3, Qwen3.5) switch thinking on or off through the template's `enable_thinking`, which comes from `reasoning_budget` at init. For a thinking-off session, use `reasoning_budget: 0` and the card's non-thinking sampling (Qwen3: `0.7 / 0.8 / 20 / 0`). Switching modes means a new `initLlama`.
- **Thinking-only models** (e.g. Qwen3-*-Thinking-2507, R1 distills) open `<think>` in the template whatever `enable_thinking` says. Use `reasoning_budget: -1` or a cap, never `0`. With `0` there is no budget sampler and nothing to stop a long think.
- **Forced-open thinking.** The template's generation prompt already ends with `<think>`, so the model's output begins with the reasoning text and contains only `</think>`. The parser accounts for this. Do not look for an opening `<think>` in the stream. Use the `reasoning_content` stream field ([section 7](#7-streaming)).
- **Budget per call.** With thinking on at init, a completion may pass `reasoning_budget: N` (> 0) to replace the cap for that call. `0` or `-1` on a completion cannot turn thinking off or on. `reasoning_budget_active` in the result says whether a cap was installed.
- **Repetition.** If a quantized model loops while thinking, the Qwen card suggests `presence_penalty` between 0 and 2 (try 1.0–1.5). Raise `n_ctx` before raising penalties.
- **History.** When you send an assistant turn back, send its `content`. You may also send `reasoning_content`, and the template decides whether to render it. The cache compares tokens, so a template that renders an earlier turn differently is handled: reuse stops where the tokens differ.

### 3.3 Vision chat

```ts
const info = await loadLlamaModelInfo(model, mmprojPath); // VRAM hint includes the projector
const ctx = await initLlama({
  model,
  mmproj: mmprojPath,
  n_ctx: 4096,              // one image is often 256–1500 tokens
  n_gpu_layers: info.optimalGpuLayers,
  chunk_size: info.suggestedChunkSize,
  is_cpu_only: info.isCpuOnly,
  use_jinja: true,
  capabilities: ['vision-chat'],
});

const r = await ctx.completion({
  messages: [{
    role: 'user',
    content: [
      { type: 'text', text: 'What is in this picture?' },
      { type: 'image_url', image_url: { url: 'file:///…/photo.jpg' } },
    ],
  }],
});
```

- A turn with media clears the KV cache (`kv_cleared_reason: 'media'`), and the context window does not shift on that turn. Budget `n_ctx` for the image tokens plus the reply.
- `mmproj` must come from the same model release as the text weights. A mismatched projector fails to load, and `isMultimodalEnabled()` returns `false`.
- Vision with thinking (e.g. Qwen3.5 VL thinking): combine this recipe with 3.2.

### 3.4 Audio

```ts
const ctx = await initLlama({
  model, mmproj: audioMmproj,
  n_ctx: 4096,
  use_jinja: true,
  capabilities: ['audio-transcribe'],
});
const { audio, audioSampleRate } = await ctx.getSupportedModalities();
const t = await ctx.transcribeAudio('/path/clip.wav'); // t.text
```

- You can also send `{ type: 'audio_url', audio_url: { url } }` parts to `completion` for chat about audio.
- `transcribeAudio` caps output at 2048 tokens and does not read `options.language` yet.
- **Fixed in this release:** `transcribeAudio` and `visionReasoning` used to stop after the first streamed token.

### 3.5 Multimodal (image and audio) on one model

Declare every capability you will call, and give the projector VRAM headroom:

```ts
const info = await loadLlamaModelInfo(model, mmprojPath);
const ctx = await initLlama({
  model, mmproj: mmprojPath,
  n_ctx: 8192,
  n_gpu_layers: info.optimalGpuLayers, // already reduced by the mmproj size
  use_jinja: true,
  capabilities: ['vision-chat', 'image-encode', 'audio-transcribe', 'vision-reasoning'],
});
```

### 3.6 Embeddings

```ts
const ctx = await initLlama({ model: embedModel, embedding: true, n_ctx: 2048, n_batch: 2048, n_ubatch: 2048 });
const e = await ctx.embedding({ input: ['first text', 'second text'] });
```

Make `n_batch` / `n_ubatch` at least as large as your longest input. Most embedding models need the whole input in one batch.

---

## 4. Device presets

These are starting points. Measure on your target devices.

| Device class | Settings |
| --- | --- |
| **iOS (Metal), A15 and newer / M-series** | `n_gpu_layers: info.optimalGpuLayers`, `chunk_size: 128`. Up to 4B Q4 at `n_ctx` 8192 is comfortable on 8 GB devices. |
| **iOS Simulator** | Always CPU (`optimalGpuLayers` is 0). Use a small model and `n_ctx` ≤ 2048. |
| **Android Snapdragon (Adreno, OpenCL)** | Needs `libggml-opencl.so` in the package. The system OpenCL driver is found through the manifest's `libOpenCL.so` entry. `n_gpu_layers: info.optimalGpuLayers`, `chunk_size: 128`, leave `use_mmap` unset (OpenCL cannot mmap; AUTO handles it). Q4_0 has the best Adreno kernels. |
| **Android Snapdragon with Hexagon NPU** | Opt-in build (`--hexagon`). The HTP skel files must be extracted to disk, see below. Llama.cpp treats the NPU as a GPU device (`HTP0`). If both HTP and OpenCL initialize, layers are split across them by free memory. Check the log (`offloaded N/M layers` and device names) before relying on the speed. |
| **Android Mali / PowerVR / other (no OpenCL ICD)** | Runs on CPU. Vulkan is not packaged (emulator crashes). `n_gpu_layers` has no effect. Use `chunk_size: 32`, `is_cpu_only: true`, and small models (≤ 2B Q4). |
| **Low RAM (< 6 GB)** | Keep `n_ctx` ≤ 4096, prefer 0.5–2B models, `n_ubatch: 256`. The 15 % VRAM tier in `optimalGpuLayers` already applies. |
| **Android emulator** | CPU, baseline `armv8.0` / x86_64 library. Use for functional tests only. |

**CPU variants (Android arm64):** the loader scores `armv9.2_2`, `armv9.2_1`, `armv9.0_1`, `armv8.6_1`, `armv8.2_2`, `armv8.2_1`, and `armv8.0_1`, and keeps the best one the CPU supports. Arm v9 phones (e.g. Snapdragon 8 Elite) now get the v9 builds. Older phones score them 0 and stay on v8.

### Hexagon packaging

FastRPC opens `libggml-htp-vNN.so` by file path, in the directory that holds `libggml-hexagon.so`. Modern Android Gradle Plugin keeps native libraries inside the APK, so that path does not exist and the NPU session fails. OpenCL is unaffected because it loads by soname. Turn extraction on in the app:

Bare React Native (`android/app/build.gradle` and the app manifest):

```groovy
android {
  packagingOptions { jniLibs { useLegacyPackaging true } }
}
```

```xml
<application android:extractNativeLibs="true" …>
```

Expo (`app.json` / `app.config.js`):

```js
plugins: [
  ['expo-build-properties', { android: { useLegacyPackaging: true } }],
]
```

The library manifest declares `libOpenCL.so`, `libvulkan.so`, and `libcdsprpc.so` as optional system libraries. The library never ships those files. Hexagon accuracy and speed knobs are environment variables read by the backend before init: `GGML_HEXAGON_MM_SELECT`, `GGML_HEXAGON_OPLOWER`, and `GGML_HEXAGON_POWER_MODE` (default `burst`). Set them in a debug build only if output looks wrong. See `cpp/llama.cpp/docs/backend/snapdragon/README.md`.

---

## 5. Prompt pacing and thermal control

The prompt is encoded in chunks of `chunk_size` tokens. This pacing was added after app crashes on real Android devices (GPU fence timeouts), and it stays as it is until it can be re-validated on hardware. Here is exactly what it does:

| Key | Default | Behavior |
| --- | --- | --- |
| `chunk_size` | `128` | Tokens per `llama_process` call while encoding the prompt. Clamped to 8–512 and to `n_batch`. Smaller chunks mean shorter GPU submissions, and that is the setting that matters for fence timeouts. Results do not depend on it. |
| `is_cpu_only` | `false` | Your declaration, not a detection. When `true`, a fixed **2 ms sleep follows every chunk** and the GPU rule below is skipped. It does not change `n_gpu_layers`. |
| `prompt_chunk_gap_ms` | `5` | GPU path only. A minimum time per chunk, measured from the chunk's start, plus 1 ms for every 2048 prompt tokens. The loop sleeps only for the part of the minimum the chunk did not already use. A chunk that takes longer gets no sleep. |

Pacing applies only to prompt encoding, not to token generation. None of these settings lowers sustained heat. To respond to heat, use:

- `setNThreads(n)`: non-blocking, applies before the next decode. Drive it from `ProcessInfo.thermalState` (iOS) or `PowerManager.getThermalStatus()` (Android).
- Fewer `n_gpu_layers`, a smaller model, or a smaller `n_ctx`.

`optimalGpuLayers` caps offload at 75 % of the layers, to leave CPU yield points. If you pass your own `n_gpu_layers`, test full offload on the target device before shipping it.

---

## 6. Completion parameters

| Key | Default | Notes |
| --- | --- | --- |
| `messages` | — | OpenAI format. Content can be a string or parts (`text`, `image_url`, `audio_url`). `prompt` is the raw-text alternative (no template). |
| `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `repeat_last_n`, `presence_penalty`, `frequency_penalty` | the model's | Omitted = the GGUF's `samplingDefaults` if present, else llama.cpp's (`0.8 / 0.95 / 40 / 0.05`, repeat 1.0 over 64, presence 0, frequency 0). Sampling is not part of the cache key. |
| `n_predict` / `max_tokens` | `-1` | `-1` = until EOS or a stop string. Capped to the free context. |
| `stop` | template stops | Added to the template's own stop strings. |
| `seed` | init seed | |
| `grammar` | none | GBNF. Cannot be combined with `tools` unless `tool_choice` is `'none'` (that combination is an invalid-parameter error). |
| `tools` | none | OpenAI function tools. Needs `use_jinja: true` at init. |
| `tool_choice` | `'auto'` | `'auto'`, `'none'`, `'required'`, `{ type: 'function', function: { name } }`, or `{ type: 'allowed_tools', allowed_tools: { mode: 'auto' \| 'required', tools: [{ type: 'function', function: { name } }] } }`. A name that does not match exactly one tool fails the call with an invalid-parameter error and leaves the cache untouched. Required mode is always enforced. Whether the grammar also excludes the other tools depends on the model's template handler upstream (at b12780 Qwen3.5 still allows every listed tool), so pass only the tools you want when it matters. |
| `reasoning_budget` | init cap | See [recipe 3.2](#32-thinking-models-qwen3-qwen35-qwq-deepseek-r1-distills). |
| `n_keep` | `0` | Context shift: `0` keeps the system message when it is first and its render is a token prefix of the prompt, else half the context. `> 0` keeps that many tokens. `-1` stops instead of shifting. |
| `prompt_id` | `''` | A change from the previous non-empty value clears the KV cache. |
| `reset_kv_cache` | `false` | Force a full clear. |
| `ignore_eos` | `false` | Ignore end-of-generation tokens. |

`timings.prompt_n` is llama.cpp's own counter of tokens evaluated into KV this call. It can read one token high, because upstream counts per sync and the first generated token can land in it. Treat a hit as "small `prompt_n` next to a large `usage.prompt_tokens`", not an exact difference. Outputs can also differ slightly between `chunk_size` values and between a cache hit and a full encode: llama.cpp does not promise identical logits across batch sizes (`tools/server/README.md`, `cache_prompt`).

KV reuse is by tokens, as llama-server's `cache_prompt`: the cache keeps the longest common prefix of what it holds (the previous prompt and its reply) and the new prompt's tokens, and encodes the rest. There is no message key to manage. Hybrid / recurrent models (Qwen3.5) can drop the previous reply but cannot roll back into an earlier prompt: an edited earlier message re-encodes from the start (`seq_rm`).

The result's `kv_cleared_reason` says why a call re-encoded from the start: `none`, `reset`, `media`, `prompt_id`, `render_identity` (Jinja, reasoning format, template, tools, `tool_choice`, `parallel_tool_calls`, grammar, or `enable_thinking` changed), `no_prefix`, or `seq_rm`.

---

## 7. Streaming

Pass a callback as the second argument to `completion` / `completionSync`. `completion` batches chunks to about one call every 33 ms, with back-pressure toward the JS thread. `completionSync` calls back for each chunk:

```ts
let answer = '', thinking = '';
const r = await ctx.completion({ messages }, (chunk) => {
  // chunk.token: raw text (always present)
  if (chunk.reasoning_content) thinking += chunk.reasoning_content;
  if (chunk.content) answer += chunk.content;
  render(thinking, answer);
});
// r.choices[0].message.{content, reasoning_content} is authoritative
```

- `token` is the raw text, including `</think>` and tool-call markup, exactly as before.
- `content` and `reasoning_content` (chat calls only) are the parser's deltas, computed the same way llama-server streams them. They can trail `token` by a few tokens. By the time the promise resolves, their concatenation equals the final message.
- Tool calls are not streamed as deltas. Read them from the result.
- If the parser fails mid-stream, the deltas stop. `token` keeps flowing, and the final result is still parsed in full.

---

## 8. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `</think>` or thinking text inside `content` | `reasoning_format: 'none'` (the old default), or `use_jinja` off. Use `use_jinja: true` with the default `'auto'`. |
| Fine for a few hundred tokens, then repetition, drift, or forgotten instructions (Qwen, Llama 3) | RoPE overridden. Remove every `rope_*` / `yarn_*` key from your config, and update to this release (the old default sent 10000). |
| Thinking never ends | Thinking-only model with `reasoning_budget: 0`, or `n_ctx` too small. Set a positive budget (e.g. 2048), raise `n_ctx`, use the card's sampling (no greedy), and try `presence_penalty` ~1.5. |
| Thinking does not appear | Hybrid model with `reasoning_budget: 0` (thinking is off by design). Set `-1` or a cap at init. |
| Tool calls come back as text | `use_jinja` is off, or the template has no tool support. Check `finish_reason`: `tool_call_parse_error` carries the parser message. |
| `Invalid tool_choice: …` | The named function is not in `tools`, or the name is empty. |
| `initLlama` throws on `reasoning_format` | The value must be `'none'`, `'auto'`, `'deepseek'`, or `'deepseek-legacy'`. |
| Android: GPU never used | `libggml-opencl.so` is missing from `jniLibs/arm64-v8a` (run the GPU build scripts), or the device has no OpenCL ICD (Mali). The log shows which backends registered. |
| Android: Hexagon session fails | Native libraries are not extracted ([Hexagon packaging](#hexagon-packaging)), or `libcdsprpc.so` is not available to the app. |
| Long prompts crash or freeze the display on Android GPU | Lower `chunk_size` (64 or 32). Keep `prompt_chunk_gap_ms` at 5 or higher. |

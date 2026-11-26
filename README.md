# LlamaRN

A thin, reliable React Native Turbo Module wrapping [llama.cpp](https://github.com/ggml-org/llama.cpp) for on-device LLM inference on iOS and Android.

## Features

- Model loading, text completion, and chat completion
- Metal GPU on iOS; OpenCL / Vulkan / Hexagon NPU on Android
- Automatic CPU/GPU detection and optimal GPU layer estimation
- Chat completion with Jinja template support
- Multi-turn KV cache prefix reuse (pass a stable `id` per message)
- Embeddings generation
- Function / tool calling
- Thinking and reasoning model support (`reasoning_budget`, `reasoning_format`)
- **Multimodal / Vision** — image-in-prompt chat, CLIP-style embeddings, Whisper-style transcription, open-ended vision reasoning, and zero-copy camera frame pipeline

## Breaking changes

See **[BREAKING.md](./BREAKING.md)** for the current migration notes: KV prefix reuse, thinking models and `reasoning_content`, sampling defaults, and llama.cpp `b12651`.

## Installation

```sh
npm install @novastera-oss/llamarn
```

## Developer Setup

### Prerequisites

1. Clone the repository
2. React Native development environment for your target platform(s)

### Initial Setup

```sh
npm install
npm run setup-llama-cpp        # init llama.cpp submodule
```

### Android

```sh
./scripts/build_android_external.sh   # build native libraries
cd example && npm run android
```

### iOS

```sh
cd example/ios && bundle exec pod install
cd .. && npm run ios
```

**Troubleshooting:**
- Android: `cd android && ./gradlew clean`
- iOS: `cd example/ios && rm -rf build Podfile.lock && pod install`

---

## Basic Usage

### Simple Completion

```js
import { initLlama } from '@novastera-oss/llamarn';

const context = await initLlama({
  model: '/path/to/model.gguf',
  n_ctx: 2048,
  n_batch: 512,
});

const result = await context.completion({
  prompt: 'What is artificial intelligence?',
  temperature: 0.7,
  top_p: 0.95,
});

console.log(result.text);
```

### Chat Completion

```js
const context = await initLlama({
  model: '/path/to/model.gguf',
  n_ctx: 4096,
  use_jinja: true,
});

const result = await context.completion({
  messages: [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user',   content: 'Tell me about quantum computing.' },
  ],
  temperature: 0.7,
});

console.log(result.text);
// OpenAI-compatible: result.choices[0].message.content
```

### Multi-Turn Chat with KV Cache Reuse

Assign a stable `id` to each message. The native layer re-uses KV cache entries for messages whose ID matches the previous turn — only new tokens are encoded.

```js
const history = [
  { role: 'system',    content: 'You are a helpful assistant.', id: 'sys-1'    },
  { role: 'user',      content: 'Hi!',                          id: 'turn-1-u' },
];

const r1 = await context.completion({ messages: history });

history.push({ role: 'assistant', content: r1.text,         id: 'turn-1-a' });
history.push({ role: 'user',      content: 'Tell me more.', id: 'turn-2-u' });

// Only [turn-2-u] is encoded; everything before hits the cache
const r2 = await context.completion({ messages: history });
```

**Rules:**
- Messages without an `id` are always fully re-encoded.
- If you edit a message's content, change its `id` so the cache is invalidated.
- `reset_kv_cache: true` forces a full clear.

```js
await context.completion({ messages: history, reset_kv_cache: true });
```

### Completion parameter naming

Completion request keys are strict snake_case to match the native layer (`top_p`, `top_k`, `min_p`, `repeat_penalty`, `frequency_penalty`, `presence_penalty`, `reset_kv_cache`, etc.). CamelCase aliases are not parsed by the native bridge.

### Thinking and Reasoning Models

```js
const context = await initLlama({
  model: '/path/to/reasoning-model.gguf',
  n_ctx: 8192,
  use_jinja: true,
  reasoning_budget: -1,       // omitted or 0 = thinking off, -1 = unlimited, >0 = token limit
  // reasoning_format defaults to 'auto': thinking goes to message.reasoning_content
  // do not set rope_* / yarn_*: the model's own values are used
});

const result = await context.completion({
  messages: [{ role: 'user', content: 'Solve: what is 15% of 240?' }],
  temperature: 0.6, top_p: 0.95, top_k: 20, min_p: 0, // Qwen3 card, thinking mode
}, (chunk) => {
  // chunk.reasoning_content / chunk.content are parsed deltas; chunk.token is raw text
});

const { content, reasoning_content } = result.choices![0].message;
```

Full guide for thinking, vision, audio, and per-device settings: **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)**.

See **[BREAKING.md](./BREAKING.md)** for multi-turn thinking, KV cache, and tool-calling pitfalls.

### Tool Calling

```js
const context = await initLlama({
  model: '/path/to/model.gguf',
  n_ctx: 2048,
  use_jinja: true,          // parse_tool_calls enabled automatically
  parallel_tool_calls: false,
});

const response = await context.completion({
  messages: [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user',   content: "What's the weather in Paris?" },
  ],
  tools: [{
    type: 'function',
    function: {
      name: 'get_weather',
      description: 'Get the current weather in a location',
      parameters: {
        type: 'object',
        properties: {
          location: { type: 'string' },
          unit:     { type: 'string', enum: ['celsius', 'fahrenheit'] },
        },
        required: ['location'],
      },
    },
  }],
  tool_choice: 'auto',
});

if (response.choices?.[0]?.finish_reason === 'tool_calls') {
  const call = response.choices[0].message.tool_calls[0];
  console.log(call.function.name, call.function.arguments);
}
```

### Embeddings

```js
const context = await initLlama({
  model: '/path/to/embedding-model.gguf',
  embedding: true,
  n_ctx: 2048,
});

const { data } = await context.embedding({ input: 'Text to embed' });
console.log(data[0].embedding); // Float32 array
```

---

## Multimodal / Vision

llamarn supports image and audio input for compatible models (LLaVA, Qwen-VL, Whisper) via a separate **multimodal projection model** (`mmproj` `.gguf`).

### Init with capabilities

```typescript
const model = await initLlama({
  model:        '/path/to/model.gguf',
  mmproj:       '/path/to/mmproj.gguf',
  capabilities: ['vision-chat'],  // 'image-encode' | 'audio-transcribe' | 'vision-reasoning'
  n_ctx: 4096,
  use_jinja: true,
});

const mods = await model.getSupportedModalities();
// { vision: true, audio: false }

const enabled = await model.isMultimodalEnabled();
// true
```

**Supported capabilities:**

| Value | Description |
|-------|-------------|
| `vision-chat` | Image-in-prompt chat completions (LLaVA, Qwen-VL) |
| `image-encode` | CLIP-style image embeddings |
| `audio-transcribe` | Whisper-style audio → text |
| `vision-reasoning` | Open-ended image analysis returning raw model text |

### Vision Chat

```typescript
const result = await model.completion({
  messages: [{
    role: 'user',
    content: [
      { type: 'text',      text: 'What is in this image?' },
      { type: 'image_url', image_url: { url: 'file:///path/to/image.jpg' } },
    ],
  }],
  n_predict: 256,
});

console.log(result.text);
```

Multiple images per message are supported. `file://` URIs and `data:image/...;base64,...` strings are both accepted. KV cache prefix reuse is automatically disabled when messages contain images.

### Image Embeddings

```typescript
const { embedding, n_tokens, n_embd } = await model.embedImage(
  'file:///path/to/image.jpg',
  { normalize: true },
);
// embedding: flat Float32 array, length = n_tokens * n_embd
```

### Audio Transcription

```typescript
const { text } = await model.transcribeAudio('file:///path/to/audio.wav');
```

### Open-ended Vision Reasoning

```typescript
const { raw_text } = await model.visionReasoning(
  'file:///path/to/image.jpg',
  { prompt: 'List all objects you see.' },
);
```

### Camera Frame (zero-copy)

Works with [VisionCamera](https://github.com/mrousavy/react-native-vision-camera) NativeBuffer frames — no disk I/O required.

```typescript
import { useFrameProcessor, runAsync } from 'react-native-vision-camera';

const frameProcessor = useFrameProcessor((frame) => {
  'worklet';
  runAsync(frame, async () => {
    // maxSize: 336 downsamples to ~330 KB during the pixel copy (recommended for LLaVA/CLIP)
    // Omit maxSize (or 0) for full-resolution analysis (OCR, detailed scenes)
    const result = await model.runOnFrame(
      frame, frame.width, frame.height, 'image-encode', { maxSize: 336 });

    if (result) {
      // result.embedding — Float32 array (n_tokens × n_embd)
      console.log('Tokens:', result.n_tokens);
    }
    // null means the encoder was busy with the previous frame — dropped automatically
  });
}, [model]);
```

Frame dropping is handled automatically in C++ via an atomic flag — no JS-side throttle needed.

**Supported model families:**

| Model | Capability |
|-------|-----------|
| LLaVA 1.5 / 1.6 | vision-chat |
| Qwen2-VL / Qwen3-VL | vision-chat |
| LLaMA 4 Scout | vision-chat |
| MiniCPM-V | vision-chat |
| Whisper | audio-transcribe |
| CLIP | image-encode |

---

## Model Info and Optimal Initialization

`loadLlamaModelInfo` queries a model's properties without fully loading it. The returned values are designed to be passed directly to `initLlama` — this is the **recommended initialization pattern**.

### Text-only model

```typescript
import { loadLlamaModelInfo, initLlama } from '@novastera-oss/llamarn';

const info = await loadLlamaModelInfo('/path/to/model.gguf');

const model = await initLlama({
  model:          '/path/to/model.gguf',
  n_ctx:          4096,
  n_gpu_layers:   info.optimalGpuLayers,   // device-safe GPU layer count
  chunk_size:     info.suggestedChunkSize,  // cooperative ingestion chunk size
  is_cpu_only:    info.isCpuOnly,
  prompt_chunk_gap_ms: 5,                  // GPU minimum inter-chunk gap (thermal pacing)
  use_jinja:      true,
});
```

### Vision model with mmproj

Pass `mmprojPath` so the VRAM budget is split between the LLM and the projection model before computing `optimalGpuLayers`:

```typescript
const info = await loadLlamaModelInfo('/path/to/model.gguf', '/path/to/mmproj.gguf');

const model = await initLlama({
  model:          '/path/to/model.gguf',
  mmproj:         '/path/to/mmproj.gguf',
  capabilities:   ['vision-chat'],
  n_ctx:          4096,
  n_gpu_layers:   info.optimalGpuLayers,   // already accounts for mmproj VRAM
  chunk_size:     info.suggestedChunkSize,
  is_cpu_only:    info.isCpuOnly,
  prompt_chunk_gap_ms: 5,
  use_jinja:      true,
});

console.log(info.mmprojSizeMB);      // MB reserved for the projection model
```

### All returned fields

```typescript
console.log(info.description);         // human-readable model name + quant
console.log(info.n_params);            // parameter count
console.log(info.n_layers);            // total layer count
console.log(info.model_size_bytes);    // quantized on-disk size in bytes
console.log(info.optimalGpuLayers);    // recommended n_gpu_layers for this device
console.log(info.availableMemoryMB);   // current free device RAM in MB
console.log(info.estimatedVramMB);     // estimated VRAM needed for optimalGpuLayers
console.log(info.architecture);        // e.g. "llama", "qwen2", "mistral"
console.log(info.gpuSupported);        // true if GPU offload is available
console.log(info.suggestedChunkSize);  // 32 (CPU-only) or 128 (GPU) — pass as chunk_size
console.log(info.isCpuOnly);           // true when optimalGpuLayers == 0
console.log(info.totalMemoryBytes);    // physical RAM. Size the context from this, not from free memory.
console.log(info.kvBytesPerToken);     // f16 K+V per token for a plain attention cache. Absent for hybrid and MLA models.
console.log(info.mmprojSizeMB);        // present only when mmprojPath was supplied
console.log(info.samplingDefaults);    // GGUF-embedded sampling params (see below)
```

### Sampling defaults from the model

`loadLlamaModelInfo` also reads any sampling parameters the model author embedded in the GGUF file and returns them as `samplingDefaults`. Only fields the model actually specifies are present — if a model doesn't embed a value, the key is absent.

```typescript
const info = await loadLlamaModelInfo('/path/to/model.gguf');
const sd = info.samplingDefaults ?? {};

// Merge: your explicit values > GGUF defaults > nothing
const result = await model.completion({
  messages,
  temperature:    sd.temperature    ?? 0.8,
  top_p:          sd.top_p          ?? 0.9,
  top_k:          sd.top_k          ?? 40,
  min_p:          sd.min_p          ?? 0.05,
  repeat_penalty: sd.repeat_penalty ?? 1.1,
});
```

**Important:** When you omit a sampling parameter from a `completion()` call entirely, the native layer now falls back to the GGUF-embedded defaults (loaded at `initLlama` time) rather than llama.cpp hardcoded values. This means **omitting a field is safe and intentional** — you only need to pass a value when you want to override what the model recommends. See [BREAKING.md](./BREAKING.md) for the full priority chain.

---

## Advanced Configuration

### GPU and Memory Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_gpu_layers` | `0` | Layers to offload to GPU — use `optimalGpuLayers` from `loadLlamaModelInfo` |
| `n_ctx` | `2048` | Context window size |
| `n_batch` | `512` | Prompt processing batch allocation size |
| `use_mmap` | _(unset)_ | Force memory-mapped loading on/off. Omit it — see note below |
| `use_mlock` | _(unset)_ | Lock model in RAM (prevents swapping) |

> **Load mode:** as of llama.cpp b10648 these two booleans map onto a single
> `llama_load_mode`. Omitting **both** selects `AUTO`, which memory-maps the model
> except on backends that report no mmap support (Android OpenCL and Hexagon always,
> Vulkan on integrated GPUs) — there it falls back on its own. Setting either flag
> opts out of `AUTO`: `use_mmap` alone gives `MMAP`/`NONE`, `use_mlock` alone gives
> `MLOCK` (which does **not** mmap), and both `true` gives `MMAP_MLOCK`. Prefer
> leaving them unset unless you have a measured reason not to.

`loadLlamaModelInfo` applies automatic device-tier VRAM budgeting when computing `optimalGpuLayers`:
- **≥ 8 GB RAM** (flagship — A17 Pro, M-series, Snapdragon 8 Gen 3): 20% of total RAM
- **≥ 6 GB RAM** (upper-mid — A15/A16, Snapdragon 8 Gen 2): 18% of total RAM
- **< 6 GB RAM** (budget / older devices): 15% of total RAM (conservative, avoids GPU fence timeouts)

Pass `n_gpu_layers: info.optimalGpuLayers` to `initLlama` and the right budget is applied automatically.

**Android arm64-v8a:** CPU inference uses [KleidiAI](https://github.com/ARM-software/kleidiai) v1.24.0 microkernels (SVE-256, SME/SME2, optimized GEMM for Q4_0/Q8_0/Q4_K/Q5_K). The runtime selects the best available kernel based on CPU features — no configuration needed. Older devices fall back to baseline kernels.

### Cooperative Ingestion Parameters

These control how the prompt is encoded into the KV cache — smaller chunks with OS yields prevent display fence timeouts on Android and UI starvation on CPU-only devices. Use the values from `loadLlamaModelInfo` directly.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | `128` | Tokens per `llama_decode` call during prompt encoding (8–512, independent of `n_batch`) |
| `is_cpu_only` | `false` | `true` → fixed 2 ms sleep after every chunk |
| `prompt_chunk_gap_ms` | `5` | GPU path: per-chunk time floor from the chunk start (+1 ms per 2048 prompt tokens). Chunks longer than the floor get no sleep. Not a thermal control: use `setNThreads` for that |

### Completion cache keys

Reuse is by message `id`, not by a hashed completion config.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prompt_id` | `""` | Optional. A change from the previous non-empty value clears KV. Empty does not clear. |
| `id` on a message | none | Prefix key. A leading run of equal non-empty ids is reused. The first empty id ends the run. |

- Sampling fields are not part of the cache key. Omitting `temperature` keeps the GGUF default. Sending a new temperature does not clear KV.
- Tools, grammar, `tool_choice`, the chat template, Jinja, the reasoning format, and `enable_thinking` are already in the render identity. Do not hash them into another key.
- `reset_kv_cache: true` forces a full clear.
- A chat result includes `timings`, `context_shifted`, and `kv_cleared_reason`. `timings.prompt_n` is the tokens evaluated into KV this call. A hit is a small `prompt_n` beside a large `usage.prompt_tokens`. `kv_cleared_reason` is `none` on a hit, or `reset`, `media`, `prompt_id`, `render_identity`, `no_prefix`, or `seq_rm`.
- `n_keep: 0` keeps the first message when that boundary fits, otherwise half the context. `n_keep: -1` stops instead of shifting. A shift keeps the leading ids that are still inside that head, and sets `context_shifted` on the call that slid.

```ts
const result = await model.completion({
  messages, // each message that should be reused has a stable id
  tools,
  prompt_id: 'support-bot-v3',
});
```

When tool-call parsing fails, `finish_reason` can be `tool_call_parse_error` and `tool_call_parse_error` contains the parser error message.

### Thinking and Reasoning Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reasoning_budget` | `0` | Omitted or `0` turns thinking off at load. `-1` is unlimited. `>0` is the sampler cap. A completion may pass a positive cap when init turned thinking on. `0` or `-1` on a completion does not change the template. `reasoning_budget_active` is true only when that cap was installed. |
| `reasoning_format` | `'auto'` | `'none'` \| `'auto'` \| `'deepseek'` \| `'deepseek-legacy'`. `'auto'` puts thinking in `reasoning_content`; `'none'` leaves it in `content` |
| `thinking_forced_open` | — | Deprecated, no effect (the template decides) |
| `parse_tool_calls` | — | Deprecated, no effect (`tools` + `use_jinja` parse tool calls) |
| `parallel_tool_calls` | template | Allow multiple tool calls per response. Omitted = what the chat template supports, as llama-server does |
| `rope_freq_base`, `rope_freq_scale`, `yarn_*` | from the model | Leave unset. Setting them overrides the GGUF's trained values |

---

## Model Path Handling

### iOS

- Bundle path (Xcode resource): `${RNFS.MainBundlePath}/model.gguf`
- Absolute path: `/var/mobile/…/model.gguf`

### Android

- Cache directory: `${RNFS.CachesDirectoryPath}/model.gguf`
- App assets (copied to cache first): `RNFS.copyFileAssets('model.gguf', dest)`

---

## Documentation

- [Configuration Guide](docs/CONFIGURATION.md) — What each option does; text, thinking, vision, audio, and multimodal recipes; device presets
- [Interface Documentation](INTERFACE.md) — Detailed API interfaces
- [Breaking Changes](BREAKING.md) — Migration notes
- [Example App](example/) — Simulator and emulator screens. They force `n_gpu_layers: 0` because those targets have no usable GPU. A device app still passes `optimalGpuLayers` from `loadLlamaModelInfo`.
- [Contributing Guide](CONTRIBUTING.md)

---

## About Novastera

**LlamaRN** is part of the **Novastera** open-source ecosystem. This library powers on-device LLM inference in [Novastera's](https://novastera.com) mobile applications — no data leaves the user's device.

Learn more: [https://novastera.com/resources](https://novastera.com/resources)

## License

Apache 2.0

## Acknowledgments

- **[mybigday/llama.rn](https://github.com/mybigday/llama.rn)** — foundational React Native binding for llama.cpp
- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** — the core C++ inference library
- **[react-native-pure-cpp-turbo-module-library](https://github.com/Zach-Dean-Attractions-io/react-native-pure-cpp-turbo-module-library)** — reference for the Android C++ Turbo Module pattern

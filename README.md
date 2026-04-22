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

See **[BREAKING.md](./BREAKING.md)** for migration notes (thinking models / `reasoning_content`, KV cache behavior, GGUF + Jinja + tools pitfalls) and the **v0.7.0** release summary.

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
  n_ctx: 4096,
  use_jinja: true,
  reasoning_budget: -1,          // -1 = unlimited, 0 = disabled, >0 = token limit
  reasoning_format: 'deepseek',  // 'none' | 'auto' | 'deepseek' | 'deepseek-legacy'
  thinking_forced_open: true,
});

const result = await context.completion({
  messages: [{ role: 'user', content: 'Solve: what is 15% of 240?' }],
});

// result.text may include <think>…</think> depending on the model
```

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
console.log(info.mmprojSizeMB);        // present only when mmprojPath was supplied
```

---

## Advanced Configuration

### GPU and Memory Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_gpu_layers` | `0` | Layers to offload to GPU — use `optimalGpuLayers` from `loadLlamaModelInfo` |
| `n_ctx` | `2048` | Context window size |
| `n_batch` | `512` | Prompt processing batch allocation size |
| `use_mmap` | `true` | Memory-mapped loading (faster startup) |
| `use_mlock` | `false` | Lock model in RAM (prevents swapping) |

### Cooperative Ingestion Parameters

These control how the prompt is encoded into the KV cache — smaller chunks with OS yields prevent display fence timeouts on Android and UI starvation on CPU-only devices. Use the values from `loadLlamaModelInfo` directly.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | `128` | Tokens per `llama_decode` call during prompt encoding (8–512, independent of `n_batch`) |
| `is_cpu_only` | `false` | `true` → 2 ms sleep after each chunk |
| `prompt_chunk_gap_ms` | `5` | GPU minimum inter-chunk gap; increase on thermally constrained devices |

### Completion pacing and cache keys

`completion()` now supports runtime pacing and cache-key controls:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `token_rate_cap` | `30` | Max generated tokens/sec (`0` = uncapped) |
| `token_buffer_size` | `4` | Stream callback flush cadence in tokens |
| `prompt_id` | `""` | Cache key for prompt/template/tool identity |
| `config_id` | `""` | Cache key for effective completion config identity (include tools + main system prompt identity) |

Use `prompt_id` and `config_id` together. If either is missing, config-cache reuse is skipped.
Changes to completion config are expected to take effect when `config_id` changes.

```ts
const promptSignature = {
  systemPromptVersion: 'support-bot-v3',
  toolNames: tools.map(t => t.function.name).sort(),
  template: 'chatml',
};

const configSignature = {
  model: 'qwen3-8b-q4_k_m',
  temperature: 0.6,
  top_p: 0.95,
  top_k: 40,
  min_p: 0.05,
  repeat_penalty: 1.05,
  repeat_last_n: 64,
  frequency_penalty: 0.0,
  presence_penalty: 0.0,
  tool_choice: 'auto',
  response_format: 'text',
};

// Use any stable JSON + hash utility available in your app.
const stableJson = (value: object) =>
  JSON.stringify(Object.keys(value).sort().reduce((acc, key) => {
    acc[key] = (value as Record<string, unknown>)[key];
    return acc;
  }, {} as Record<string, unknown>));

const prompt_id = `prompt-${sha256Hex(stableJson(promptSignature)).slice(0, 16)}`;
const config_id = `config-${sha256Hex(stableJson(configSignature)).slice(0, 16)}`;

const result = await model.completion({
  messages,
  tools,
  prompt_id,
  config_id,
  token_rate_cap: 30,
  token_buffer_size: 4,
});
```

When tool-call parsing fails, `finish_reason` can now be `tool_call_parse_error` and `tool_call_parse_error` will contain the parser error message.

### Thinking and Reasoning Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reasoning_budget` | `-1` | Token budget for thinking: `-1` unlimited, `0` disabled, `>0` limited |
| `reasoning_format` | `'auto'` | `'none'` \| `'auto'` \| `'deepseek'` \| `'deepseek-legacy'` |
| `thinking_forced_open` | `false` | Force thinking tags in every response |
| `parse_tool_calls` | `true` | Parse tool call JSON from output (auto-enabled with `use_jinja`) |
| `parallel_tool_calls` | `false` | Allow multiple tool calls per response |

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

- [Interface Documentation](INTERFACE.md) — Detailed API interfaces
- [Breaking Changes](BREAKING.md) — Migration notes
- [Example App](example/) — Working example with text chat and vision demo
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

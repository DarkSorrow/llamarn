# Breaking changes

This file lists **breaking changes and migration notes** so they are easy to find outside the main README.

---

## Streaming API simplification

### Removed: `token_rate_cap` and `token_buffer_size`

Both fields are **removed** from `LlamaCompletionParams`. Passing them is now a TypeScript type error.

- **`token_rate_cap`** — the sleep-based rate cap did not reduce GPU thermal load (the GPU finishes its compute before the sleep fires). It only delayed the display, not the heat. Removed entirely. Use `setNThreads()` for actual thermal control.
- **`token_buffer_size`** — display cadence is now controlled internally by a 33ms time-based flush (~30fps). No caller configuration needed or accepted.

**Migration**: remove any `token_rate_cap` or `token_buffer_size` from your `completion()` call sites.

### New: `model.setNThreads(n: number)`

Reduces inference thread count at runtime — the correct lever for thermal management.

```typescript
// On iOS thermal state change (NSProcessInfo.processInfo.thermalState):
// On Android thermal event (PowerManager.getThermalStatus()):
const factors = { nominal: 1.0, fair: 0.75, serious: 0.5, critical: 0.25 };
model.setNThreads(Math.max(1, Math.round(baseThreads * factors[thermalState])));
```

**Non-blocking**: `setNThreads` returns immediately — the JS/UI thread is never stalled. The new thread count takes effect on the next `llama_decode` call inside the inference thread. Safe to call during active generation.

### Streaming correctness fix

Partial stop-word characters (e.g. the `<` in `<|im_end|>`) are now correctly held until the full stop word is confirmed before deciding to discard or display them. Apps that filtered control-token fragments from streaming output can remove that workaround.

---

## Recommended model loading pattern

**Always call `loadLlamaModelInfo` before `initLlama`.** The function returns three fields specifically designed to be passed directly to `initLlama`:

| Field | Pass as | What it does |
|-------|---------|--------------|
| `optimalGpuLayers` | `n_gpu_layers` | GPU layer count computed from 15% of device RAM + 75% layer cap — prevents Android display fence timeouts |
| `suggestedChunkSize` | `chunk_size` | Prompt-ingestion chunk size: 32 for CPU-only devices, 128 for GPU devices |
| `isCpuOnly` | `is_cpu_only` | Enables CPU pacing path during prompt encoding (`true` = 2 ms sleep/chunk) |

```typescript
// Text-only model
const info = await loadLlamaModelInfo(modelPath);

const model = await initLlama({
  model:        modelPath,
  n_ctx:        4096,
  n_gpu_layers: info.optimalGpuLayers,
  chunk_size:   info.suggestedChunkSize,
  is_cpu_only:  info.isCpuOnly,
  prompt_chunk_gap_ms: 5, // new: deterministic GPU inter-chunk gap
  use_jinja:    true,
});
```

```typescript
// Vision model — pass mmprojPath so VRAM is split between LLM layers and projection model
const info = await loadLlamaModelInfo(modelPath, mmprojPath);

const model = await initLlama({
  model:        modelPath,
  mmproj:       mmprojPath,
  capabilities: ['vision-chat'],
  n_ctx:        4096,
  n_gpu_layers: info.optimalGpuLayers,  // already accounts for mmproj VRAM reservation
  chunk_size:   info.suggestedChunkSize,
  is_cpu_only:  info.isCpuOnly,
  prompt_chunk_gap_ms: 5,
  use_jinja:    true,
});
// info.mmprojSizeMB tells you how many MB were reserved for the projection model
```

**All `initLlama` parameters remain fully overridable.** The values from `loadLlamaModelInfo` are starting points; explicit values in `initLlama` always win:

```typescript
// Override chunk_size for a specific use case, keep the rest from info
const model = await initLlama({
  ...baseParams,
  chunk_size:   64,          // override suggested 128
  n_gpu_layers: info.optimalGpuLayers,
  is_cpu_only:  info.isCpuOnly,
});
```

**Why `chunk_size` and `n_batch` are different:**
- `n_batch` is the maximum batch allocation size passed to `llama_batch_init` — it controls the internal buffer.
- `chunk_size` is the number of tokens actually sent to `llama_decode` per call during **prompt ingestion only** (generation is unaffected). Smaller chunks let the OS scheduler run between GPU submits, preventing SurfaceFlinger fence timeouts on Android (`mLastRetireFence not released during 40ms`) and keeping the UI runloop alive on CPU-only devices.

## Completion cache keys and migration checklist

If you rely on chat/template/tool caching, pass both `prompt_id` and `config_id` on each `completion()` call.

### Completion request naming (snake_case only)

Completion parameters are now documented and consumed as snake_case keys. If your app sends camelCase aliases, migrate to canonical names.

- Use `reset_kv_cache` (not `resetKvCache`).
- Use canonical sampling keys like `top_p`, `top_k`, `min_p`, `repeat_penalty`, `frequency_penalty`, `presence_penalty`.

### What app teams must add

- Add `prompt_id` and `config_id` to your completion request payload.
- Recompute `prompt_id` when system prompt, template, or tools change.
- Recompute `config_id` when sampling/grammar/response-format changes.
- Treat `config_id` as the effective completion-config identity (include tools + main system prompt identity).
- Completion config changes are expected to take effect when `config_id` changes.
- Update finish-reason handling to include `tool_call_parse_error`.

### `config_id` example recipe

Build a stable object from the knobs that change model behavior, then hash it:

```ts
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

const stableJson = (value: object) =>
  JSON.stringify(Object.keys(value).sort().reduce((acc, key) => {
    acc[key] = (value as Record<string, unknown>)[key];
    return acc;
  }, {} as Record<string, unknown>));

const config_id = `config-${sha256Hex(stableJson(configSignature)).slice(0, 16)}`;
```

### Behavioral updates in this release

- Cache-hit path now renders with full tool/template inputs (tools are no longer stripped).
- Tool-call parse failures are surfaced as:
  - `finish_reason: "tool_call_parse_error"`
  - `tool_call_parse_error: "<parser message>"`
- Prompt ingestion pacing now uses deterministic sleeps:
  - CPU: fixed 2 ms per chunk
  - GPU: `prompt_chunk_gap_ms` minimum inter-chunk gap

### Compatibility note

If your client only accepts `finish_reason` in `stop | length | tool_calls`, add support for `tool_call_parse_error` before rolling out this version.

---

---

## Multi-turn conversations with thinking models

**Breaking change**: The bridge returns the model's raw output in `content`, which for thinking models (Qwen3, DeepSeek-R1, etc.) includes `<think>…</think>` blocks. You **must** strip these and store the thinking separately in `reasoning_content` before feeding the message back as history. If you pass the raw content back unchanged, the chat template receives malformed input and the app crashes on the second turn.

**Native behavior (KV cache):** The same `llama_context` is reused across calls; each completion run clears the KV cache at the start of `run_completion` (`llama_memory_clear(llama_get_memory(ctx), false)` in `cpp/rn-completion.cpp`). That way every call processes the **full** prompt from position 0. Without clearing, stale KV entries from the previous turn could cause assertion failures or incorrect decoding in `llama_decode` on the second and later turns.

**Example app:** `example/src/ModelChatTestScreen.tsx` shows the full pattern: `reasoning_content` on the `Message` type, an `extractThinking()` helper, `messageToApiPayload` forwarding `reasoning_content` to native, and assistant messages built with clean `content` + optional `reasoning_content` for normal replies, tool-call turns, and the final reply after tools.

```js
/**
 * Strip <think>…</think> from the start of model output.
 * Returns clean content for history and the raw thinking text separately.
 */
function extractThinking(content) {
  const match = content.match(/^<think>([\s\S]*?)<\/think>\s*/);
  if (!match) return { thinking: null, content };
  return { thinking: (match[1] ?? '').trim(), content: content.slice(match[0].length) };
}

// After receiving a completion:
const raw = result.choices[0].message.content;
const { thinking, content } = extractThinking(raw);

// Add the assistant message to history with clean content:
messages.push({
  role: 'assistant',
  content,                                          // response text only, no <think> tags
  ...(thinking ? { reasoning_content: thinking } : {}), // thinking in its own field
});

// On the next turn, send the full messages array as-is.
// The native bridge passes reasoning_content through to the chat template,
// which renders it correctly (e.g. Qwen3's jinja wraps it back in <think> tags
// during prompt construction without corrupting the conversation structure).
const nextResult = await context.completion({ messages, temperature: 0.6 });
```

---

## GGUF chat templates vs llama.cpp Jinja (tools / early Qwen3)

**What’s going on:** This is often a **version mismatch**, not a random React Native or Android bug.

* **Current LlamaRN** ships with a **recent llama.cpp**, which runs the model’s embedded chat template via **llama.cpp’s C++ Jinja-lite engine** (not full Python Jinja).
* Some **early Qwen3 (and similar) GGUFs** ship with templates that assume **Python-like Jinja**, e.g. calling `lstrip` as a **function** in places where the C++ engine only exposes **`lstrip` as a string method** (or similar).
* When that happens, the native runtime can throw (e.g. `Callee is not a function … (hint: 'lstrip')`) and **abort during template application** (often with **tools + `use_jinja: true`**), **before** your JS payload is the real issue.

**Why newer llama.cpp can “make it worse”:** Older stacks sometimes **didn’t execute** templates the same way; newer llama.cpp **parses and runs** them more strictly, so **old template + new parser/runtime** can surface **native crashes** that didn’t show up before.

**Fixes (ranked):**

1. **Prefer a newer GGUF** — Re-download / use a **recently re-exported** model for your family (Qwen3, etc.). Templates are often fixed or aligned with the runtime.
2. **Override or bypass the embedded template** — If your API supports it, supply a **custom chat template** or **disable** the embedded one and **build the prompt yourself** (system / user / history / tools as plain text). This removes dependence on broken metadata.
3. **Patch the GGUF metadata** (advanced) — Inspect the template (e.g. via `llama-cli --dump-metadata` or equivalent), replace unsupported patterns (e.g. problematic `lstrip` usage) with simpler equivalents, repack.
4. **Workarounds** — Where the binding allows, **turning off Jinja / autoparser-style paths** for that flow can avoid executing the broken template (often at the cost of **losing** automatic tool grammar / template integration).

**What usually does *not* help alone:** Tweaking only `messages` in JS (e.g. `content: null` vs `''`) when the crash is already in **Jinja execution** of the embedded template.

**Quick check:** If a **plain `prompt`** completion works but **chat + tools + Jinja** crashes, treat it as **template/runtime compatibility**, not app logic.

**Production note:** For mobile apps, **owning your prompt format** (assemble system, history, tools, and user text in one place) is often more **predictable** than relying on every vendor GGUF template across versions.

---

## Version 0.7.0

Use this section as the basis for release notes when publishing **v0.7.0**.

### Breaking / migration

- **Thinking models (multi-turn):** Same as [Multi-turn conversations with thinking models](#multi-turn-conversations-with-thinking-models) above: assistant `content` may include embedded thinking tags; strip them and pass **`reasoning_content`** when updating history, or the chat template can break and the app may crash on later turns.
- **KV cache:** Completions clear the context KV cache at the start of each run so the full prompt is evaluated from position 0; integrators should not assume incremental KV reuse across `completion` calls for the same session in the current API shape.

### Compatibility / operational

- **Old GGUF chat templates + new llama.cpp Jinja:** Some early Qwen3 (and similar) GGUFs embed templates that assume Python Jinja behavior; llama.cpp’s C++ Jinja runtime may **native-crash** on patterns such as `lstrip` used as a global call. Prefer updated GGUFs, override the template, or avoid Jinja for that path—see **GGUF chat templates vs llama.cpp Jinja** above.

### Docs / example

- **`BREAKING.md`** added so breaking changes are discoverable without reading the full README.
- Example app (`example/src/ModelChatTestScreen.tsx`) demonstrates `extractThinking`, `reasoning_content`, and `messageToApiPayload` for thinking + tools flows.

---

## Sampling defaults now come from the model, not llama.cpp hardcodes

**This is a behavioral breaking change.** `initLlama` now reads GGUF-embedded sampling parameters at load time and uses them as the default for every `completion()` call on that context.

**Before:** Omitting `temperature` (or any sampling param) from `completion()` fell back to llama.cpp hardcoded values (e.g. `temperature: 0.8`).

**After:** Omitting a sampling param falls back to whatever the model author embedded in the GGUF. Qwen3 thinking models embed `temperature: 0.6`, `top_k: 20` — those are now active by default without you passing them.

**If you see different output after upgrading**, either pass your preferred values explicitly or use `samplingDefaults` from `loadLlamaModelInfo` as your baseline:

```typescript
const info = await loadLlamaModelInfo(modelPath);
const sd = info.samplingDefaults ?? {};

const result = await model.completion({
  messages,
  temperature:    userTemp ?? sd.temperature    ?? 0.8,
  top_p:          sd.top_p          ?? 0.9,
  top_k:          sd.top_k          ?? 40,
  min_p:          sd.min_p          ?? 0.05,
  repeat_penalty: sd.repeat_penalty ?? 1.1,
});
```

### Sampling override priority chain

1. **JS explicitly sends a value** → used as-is
2. **JS omits the field** → GGUF-embedded defaults (loaded at `initLlama` time) apply
3. **GGUF has no metadata** → llama.cpp hardcoded defaults apply

Omitting a field is now safe and intentional — you only need to pass a value when overriding the model's recommendation.

### What's in `samplingDefaults`

Only fields the model actually specifies are present (never `null` or `0` as a placeholder):

| Key | Type | Description |
|-----|------|-------------|
| `temperature` | `number` | Sampling temperature |
| `top_p` | `number` | Top-p (nucleus) sampling |
| `top_k` | `number` | Top-k sampling |
| `min_p` | `number` | Min-p filtering |
| `repeat_penalty` | `number` | Repetition penalty |
| `repeat_last_n` | `number` | Window size for repeat penalty |
| `mirostat` | `number` | Mirostat mode (0 = off) |
| `mirostat_tau` | `number` | Mirostat target entropy |
| `mirostat_eta` | `number` | Mirostat learning rate |

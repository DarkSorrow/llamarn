# Sampling Defaults — Full Change Overview

## What Was Changed and Why

### The Problem

Small models (0.8B–3B) on Android were producing repetitive output. The root cause was a cascade of three issues:

1. `CompletionOptions` had hardcoded float defaults (`repeat_penalty = 1.0`, `temperature = 0.8`, etc.) that were unconditionally written into `sampling_params` on every request — overwriting whatever the model's GGUF metadata or `initLlama` had configured.
2. The JS test screen was sending `presence_penalty: 1.5` but no `repeat_penalty`, so the primary anti-repetition mechanism (`penalty_repeat`) was always `1.0` = disabled.
3. There was no way for the JS layer to know what sampling parameters the model author recommended — that information was buried in the GGUF file and never surfaced.

---

## Files Read to Understand the Problem

### Reference implementations

| File | What it showed |
|------|---------------|
| `cpp/llama.cpp/examples/llama.android/lib/src/main/cpp/ai_chat.cpp` | Android demo uses `common_sampler_init` with only `temp` set — all other params come from `common_params_sampling` defaults. No hardcoded overrides. |
| `cpp/llama.cpp/examples/llama.swiftui/llama.cpp.swift/LibLlama.swift` | iOS/Swift demo uses raw `llama_sampler_chain` with only `temp=0.4` and `dist`. No `common_sampler` at all — repetition penalties completely absent. |
| `cpp/llama.cpp/tools/server/server-context.cpp` | Server uses `common_sampler_init(model, task.params.sampling)` — sampling params come from the request, with model metadata already merged in by `common_params_sampling_init_from_model`. |
| `cpp/llama.cpp/tools/server/server-task.cpp` | Shows how the server parses per-request sampling overrides from JSON and merges them into `common_params_sampling`. |

### llama.cpp internals

| File | What it showed |
|------|---------------|
| `cpp/llama.cpp/common/common.h` | `common_params_sampling` struct — all sampling fields with their defaults. `penalty_repeat = 1.00f` (disabled), `penalty_last_n = 64`, `temp = 0.80f`, etc. |
| `cpp/llama.cpp/common/common.cpp` | `common_params_sampling_init_from_model()` — reads GGUF metadata keys (`LLAMA_MODEL_META_KEY_SAMPLING_TEMP`, `LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT`, etc.) and populates `common_params_sampling` only when the model has those keys. |
| `cpp/llama.cpp/common/sampling.cpp` | `common_sampler_init()`, `common_sampler_reset()`, `common_sampler_accept()` — confirmed that `penalty_repeat = 1.0` means the penalty sampler is a no-op. |
| `cpp/llama.cpp/include/llama.h` | `enum llama_model_meta_key` — full list of GGUF sampling metadata keys: `SAMPLING_TEMP`, `SAMPLING_TOP_K`, `SAMPLING_TOP_P`, `SAMPLING_MIN_P`, `SAMPLING_PENALTY_REPEAT`, `SAMPLING_PENALTY_LAST_N`, `SAMPLING_MIROSTAT`, `SAMPLING_MIROSTAT_TAU`, `SAMPLING_MIROSTAT_ETA`, `SAMPLING_SEQUENCE`. Also `llama_model_meta_val_str()` and `llama_model_meta_key_str()` APIs used to read them. |

### Our mobile library files

| File | What it showed |
|------|---------------|
| `cpp/rn-utils.h` | `CompletionOptions` struct — had hardcoded float defaults that were always applied, no sentinel mechanism. |
| `cpp/rn-completion.cpp` | `run_completion()` — unconditionally wrote all `CompletionOptions` fields into `sampling_params`, clobbering model defaults. |
| `cpp/PureCppImpl.cpp` | `loadLlamaModelInfo()` — loaded the model but never read GGUF sampling metadata before freeing it. |
| `cpp/LlamaCppModel.cpp` | `parseCompletionOptions()` — correctly only sets fields when JS provides them, but the struct defaults meant unset fields still had values. |
| `example/src/ModelChatTestScreen.tsx` | JS test screen — was sending `presence_penalty: 1.5` with no `repeat_penalty`, `min_p: 0` (disabled), and no use of model-recommended defaults. |

---

## Changes Made

### 1. `cpp/rn-utils.h` — Sentinel values in `CompletionOptions`

**Before:** All sampling fields had hardcoded defaults that were always applied.
```cpp
float temperature = 0.8f;
float repeat_penalty = 1.0f;  // 1.0 = disabled — always overwrote model defaults
int   repeat_last_n = 64;
```

**After:** Sentinel values — `NaN` for floats, `-1` for ints — mean "not set by caller".
```cpp
float temperature    = std::numeric_limits<float>::quiet_NaN();
float repeat_penalty = std::numeric_limits<float>::quiet_NaN();
int   repeat_last_n  = -1;
```

When a field is `NaN`/`-1`, `run_completion` leaves `sampling_params` untouched for that field, so the model's `initLlama` defaults (which themselves come from GGUF metadata) are preserved.

### 2. `cpp/rn-completion.cpp` — Conditional sampling override

**Before:** Unconditional assignment.
```cpp
sampling_params.penalty_repeat = options.repeat_penalty;  // always 1.0 if JS didn't send it
sampling_params.temp           = options.temperature;      // always 0.8 if JS didn't send it
```

**After:** Only applies when the caller explicitly set a value.
```cpp
auto applyF = [](float& dst, float src) { if (!std::isnan(src)) dst = src; };
applyF(sampling_params.penalty_repeat, options.repeat_penalty);
applyF(sampling_params.temp,           options.temperature);
if (options.repeat_last_n >= 0) sampling_params.penalty_last_n = options.repeat_last_n;
```

### 3. `cpp/PureCppImpl.cpp` — Read GGUF sampling metadata in `loadLlamaModelInfo`

Added a read pass before `llama_model_free()` that reads all available GGUF sampling keys using `llama_model_meta_val_str` + `llama_model_meta_key_str`. Only keys that are actually present in the GGUF are populated (sentinel `-1` otherwise).

The result is returned as `samplingDefaults` in the JS object:
```json
{
  "samplingDefaults": {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "repeat_penalty": 1.0
  }
}
```

Fields absent from the GGUF are simply not present in the object — never `null` or `0`.

### 4. `example/src/ModelChatTestScreen.tsx` — Use model defaults

**Before:** Hardcoded sampling params, `presence_penalty: 1.5` as the only anti-repetition mechanism, `min_p: 0` (disabled).

**After:** Reads `info.samplingDefaults` from `loadLlamaModelInfo` and uses those as the baseline. JS-supplied values still override them. Falls back to reasonable hardcoded values only when the GGUF has nothing.

```typescript
const sd = modelState.samplingDefaults;
const baseSampling = {
  temperature:    sd.temperature    ?? 0.8,
  repeat_penalty: sd.repeat_penalty ?? 1.1,
  min_p:          sd.min_p          ?? 0.05,
  // ...
};
```

---

## Priority Chain (Final State)

```
JS explicitly sends a value
        ↓ wins
GGUF model metadata (via samplingDefaults from loadLlamaModelInfo)
        ↓ wins
initLlama defaults (common_params_sampling, itself populated from GGUF by common_params_sampling_init_from_model)
        ↓ wins
llama.cpp hardcoded defaults (penalty_repeat=1.0, temp=0.8, etc.)
```

---

## What the Reference Implementations Do (Comparison)

| | Android (ai_chat.cpp) | iOS/Swift (LibLlama.swift) | Server | **Our library (after fix)** |
|---|---|---|---|---|
| Sampling source | `common_params_sampling` defaults only | Raw sampler chain, temp only | Per-request JSON merged with model metadata | GGUF metadata → initLlama → per-request JS override |
| `repeat_penalty` | Default `1.0` (off) | Not present | From request or model metadata | From GGUF or JS, never silently disabled |
| GGUF metadata used | Via `common_sampler_init` indirectly | No | Yes, via `common_params_sampling_init_from_model` | Yes, explicitly read and surfaced to JS |
| JS can override | N/A (JNI, not JS) | N/A (Swift) | Yes | Yes, always wins |
| Sentinel mechanism | N/A | N/A | JSON field presence check | NaN/−1 in `CompletionOptions` |

---

## Key Insight

`penalty_repeat = 1.0` in llama.cpp means "multiply token logit by 1.0" = no effect. It is not a small penalty — it is literally disabled. The Android and iOS reference demos both have this disabled by default. The server avoids the problem by reading GGUF metadata first. We now do the same.

Most GGUF models from Hugging Face (Qwen, Mistral, Llama, Gemma) embed their recommended sampling parameters. Models that don't embed them (older or custom GGUFs) fall back to llama.cpp defaults, which is the same behavior as before this change.

# Breaking changes

This file lists **breaking changes and migration notes** so they are easy to find outside the main README.

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

# Handoff: llamarn changes for Nola

Status: library plan. No llamarn code in this repo. Nola's app work stays in `llm-configuration-todo.md`.

Base the branch on the published **0.11.2** tarball (`@novastera-oss/llamarn`, 7 Nov 2026). That is what Nola runs, and it is npm `latest`. GitHub's default branch still says `0.8.11` in `package.json`. Its older notes (including "every completion clears the KV cache") describe a different tree. Do not implement from that branch.

Nola is the consumer this pass optimizes for: one loaded model, a long chat, a phone CPU or a small GPU, a context window that is also the KV allocation. The library already has the right cache. The work is to make that cache observable, to stop documenting keys it ignores, and to stop a context shift from throwing away the prefix it just kept.

## What already works and stays

`run_chat_completion` in `cpp/rn-completion.cpp` reuses KV when all of these hold:

1. `reset_kv_cache` is false.
2. The turn has no image or audio. Media always clears.
3. `prompt_id`, when both this call and the previous one set it, is unchanged. An empty `prompt_id` does not clear.
4. `build_kv_render_identity` is unchanged: Jinja, reasoning format, chat template, `tool_choice`, grammar, and the full tools JSON.
5. A leading run of messages has non-empty `id` strings equal to the ids stored from the previous successful call. The first empty id ends the run.

On a match, native drops KV after the last matched message's `token_end` and encodes only the tail (`kv_hint_pos`). `token_end` is the chat rendered with `add_generation_prompt = false`. The live prompt includes the assistant header, so a few header tokens are re-encoded every turn. That assignment is the contract. The comment directly above it still says to use `true`. Delete that comment. Leave the assignment.

`llama_memory_seq_rm` returns false on some recurrent models and some GPU backends. Native then full-clears. That path stays. It needs a reason code, below, so a full encode is explainable.

Sampling priority stays: an explicit completion field, else the GGUF value loaded at `initLlama`, else the llama.cpp hardcoded default. A missing field is `NaN` and means "use the model". A temperature change does not belong in the render identity. Changing temperature must not clear KV.

`use_mmap` and `use_mlock` both unset select `LLAMA_LOAD_MODE_AUTO`. Nola will stop passing them. The library mapping in `do_init_llama` is already right.

`getOptimalGpuLayers` budgets **weights**: 15 / 18 / 20 percent of RAM, then 80 percent of that, capped at 75 percent of the layer count. A budget that rounds to 0 still returns 1 layer when a GPU exists. The simulator returns 0. Leave that function alone in this pass, including the one-layer floor. Nola passes `optimalGpuLayers` through and does not raise it.

`suggestedChunkSize` (32 when that layer count is 0, else 128), the 2 ms CPU sleep, and `prompt_chunk_gap_ms` (default 5) stay. They are what keeps Android's display fences and a CPU-only UI alive. `setNThreads` / `requested_n_threads` stay a hook. No thermal policy in the library.

`loadSession` / `saveSession` stay as they are. Nola will not use them for chat. The conversation cache is the in-memory id index, not a file.

Message text is not hashed in C++. The caller owns ids. An id is reused only when role, content, `reasoning_content`, and the tool-call payload are unchanged. Nola will send a content hash for the system message and the stored row id for immutable chat rows. The library trusts the string.

## What Nola will keep doing in the app

These are product choices. They do not become llamarn defaults.

- Reasoning off unless the user turns it on. Budget 128 when on, hard cap 512, never `-1`. `-1` stays legal in the library for other callers. The app is the one that refuses it.
- `reset_kv_cache: true` on the first completion of a chat or page session, then false while that conversation appends.
- `prompt_id` set to the prompt id, so switching prompts clears. The system message id is a hash of the bytes actually sent. The same prompt id with different system text must still miss, which is why the message id exists. `prompt_id` is not a hash of the system text. The comment on `last_prompt_id` in `rn-llama.h` says a change means the system prompt or the tools changed. Tools are in the render identity. Fix the comment.
- Device context cap applied at load, not written back into the synced prompt. The library returns the bytes-per-token number. The app decides the ceiling.
- `n_batch` chosen by the app (256 Android, 128 iOS). The native default of 512 stays the default for other callers.

## 1. The chat result hides the only proof of a cache hit

`llama_perf_context_reset` runs at the start of `run_completion`. `timings.prompt_n` is then `llama_perf_context().n_p_eval`: tokens actually evaluated this call. `result.n_prompt_tokens` is `state.prompt_tokens.size()`, the full rendered prompt, hit or miss.

`LlamaCppModel::completionResultToJsi` returns early when `result.chat_response` is set. That object is the OpenAI JSON from the bottom of `run_chat_completion`. It has `usage.prompt_tokens` (the full length) and it does not have `timings` or `context_shifted`. The non-chat return is the only one that copies `timings.prompt_n` and `contextShifted`. Nola's chats take the early return, so a device cannot see a hit.

`usage.prompt_tokens` stays the full prompt length. OpenAI clients read it that way. Add a sibling on the chat object:

```json
"timings": { "prompt_n": 0, "prompt_ms": 0, "predicted_n": 0, "predicted_ms": 0, "total_ms": 0 },
"context_shifted": false
```

`prompt_n` is `n_p_eval`. The TypeScript comment on `LlamaCompletionResult.timings.prompt_n` currently says "Number of tokens in the prompt". That describes `usage.prompt_tokens`. Change the comment to say it is tokens evaluated into KV this call, and that a hit is a small `prompt_n` beside a large `usage.prompt_tokens`.

Also add `kv_cleared_reason` on the same object, a string, one of:

| Value | When |
| --- | --- |
| `none` | A prefix matched and `seq_rm` succeeded. `prompt_n` is the tail. |
| `reset` | `reset_kv_cache` was true. |
| `media` | The turn had image or audio. |
| `prompt_id` | Both calls set `prompt_id` and they differed. |
| `render_identity` | Jinja, reasoning format, template, tool choice, grammar, or tools JSON changed. |
| `no_prefix` | No leading id matched, including a missing id on the first message. |
| `seq_rm` | A prefix matched and `llama_memory_seq_rm` returned false, so native full-cleared. |
| `context_shift` | Set on the **next** call if the previous call shifted and the index was dropped. Omit it if section 3 keeps the prefix index, and keep `context_shifted` on the call that shifted. |

`seq_rm` is the one Nola cannot distinguish from a bad id today. Both produce a full `prompt_n`.

Acceptance: a chat completion JSON contains `timings.prompt_n`. A second turn that matches ids reports `prompt_n` near the new tail and `kv_cleared_reason: "none"`. A call with no ids reports `no_prefix` and `prompt_n` equal to the encoded length. A test can force `seq_rm` false with a stub or a recurrent model and expects `seq_rm`.

## 2. Delete the cache API the bridge does not run

`config_id` is parsed in `LlamaCppModel.cpp` onto `CompletionOptions` and never read. Reuse does not consult it. README still says to hash it, and that reuse is skipped when either `prompt_id` or `config_id` is missing. `BREAKING.md` repeats that checklist. `NativeRNLlamaCpp.ts` documents the field. Remove the field from the parser, the struct, the type, the README section "Completion pacing and cache keys", and that breaking checklist.

Replace that section with the five rules at the top of this file. Say explicitly:

- Sampling is not part of the cache key. Omitting temperature is how a caller keeps the GGUF default. Sending a new temperature does not clear KV.
- `prompt_id` is optional. Empty does not clear. A change clears.
- Tools are already in the render identity. A caller does not hash them again.
- The message `id` is the prefix key. Document it on `LlamaMessage` as an optional string. The C++ already reads `msg["id"]`. The TypeScript interface does not mention it, so callers cannot see the contract.

`token_rate_cap` and `token_buffer_size` are already removed from the type. `BREAKING.md` says so, and explains that a sleep after the GPU work does not cool the phone. README still lists both, with defaults 30 and 4. Delete those rows. Do not bring the rate cap back.

`LlamaCompletionParams.system_prompt` is not parsed. The system text belongs in `messages`, with an id. Remove the field from the type so there is one way in.

`LlamaCompletionParams.n_keep` is commented "reserved/no-op in generation path". It is the context-shift keep count in `run_completion`. Fix the comment in the same change as section 3.

`enable_thinking` is not in `build_kv_render_identity`. It is safe only because `do_init_llama` sets it once, from `reasoning_budget != 0`, and nothing changes it per call. Add the kwargs value to the identity string now. On one context it never changes, so hit rate is unchanged. A later per-request flag cannot reuse KV that was rendered with the other flag.

Acceptance: grepping the repo for `config_id` and `token_rate_cap` finds nothing outside a one-line migration note that says both are gone. A completion with a different temperature and the same ids still hits. A completion with a different tools JSON misses, with `kv_cleared_reason: "render_identity"`.

## 3. A context shift must keep the prefix it preserved

When `n_past + 1 >= n_ctx`, the loop shifts. The comment says `n_keep == 0` keeps 25 percent of the context. The code keeps `n_ctx / 2`:

```cpp
int n_keep = options.n_keep > 0 ? options.n_keep : (state.n_ctx / 2);
```

`n_keep < 0` stops instead of shifting. That part of the comment is right.

After the shift, `context_shifted` is set and `run_chat_completion` clears `kv_messages`. The tokens that were kept, including the system message, are still in the cache, and the next turn re-encodes them because the index is gone. That is the expensive case on a phone: the window filled, the shift paid for a memmove, and the following message prefills the whole transcript anyway.

Two changes.

**Auto keep is the first message boundary.** When `n_keep == 0` and `kv_messages` is non-empty, use `kv_messages[0].token_end` as the keep count, provided it is in `(0, n_ctx - 4)`. That is the system message when the caller put it first and gave it an id. When there is no index, keep today's `n_ctx / 2` so raw completions do not change. When the chosen keep leaves `n_discard <= 0`, stop generation the way an oversized explicit `n_keep` already does. Explicit `n_keep > 0` and `n_keep == -1` are unchanged.

Nola then does not pass a token count. Passing one would mean tokenizing in JavaScript, and a BPE split does not match the native boundary.

**The index keeps the leading run that is still a true prefix.** After a successful shift, walk `kv_messages` in order and keep the leading entries whose `token_end <= n_keep`. Stop at the first message that is not fully inside that head. Leave those `token_end` values alone: those positions were not moved. Drop every later entry, including messages that were shifted down rather than deleted. The next request still contains the discarded middle, so a tail id would match the wrong bytes. If the leading run is empty, clear the index as today.

`context_shifted` stays true on the call that shifted, so the app can tell the user the window slid. The following call can still hit on the system id.

Fix both comments (the 25 percent one, and the `n_keep` "no-op" one in `NativeRNLlamaCpp.ts`).

Acceptance: fill a context whose first message id is stable, force one shift, then send one new user message with the same leading id. `kv_cleared_reason` is `none` and `prompt_n` is the tail after the first message, not the whole transcript. `n_keep: -1` still stops with no shift. A completion with no message ids still shifts at `n_ctx / 2`.

## 4. Tell the app how big the KV cache is

`loadLlamaModelInfo` returns `n_context` (the trained window), `optimalGpuLayers`, and `estimatedVramMB` for **weights**. The KV buffer is allocated for the full `n_ctx` the caller passes to `initLlama`, and it is not in that budget. A typed context of tens of thousands can jetsam the app after the weight check has passed. Nola currently invents a RAM table (2048 / 4096 / 8192) because the library does not say how wide a context this phone can hold for this model.

Add two numbers to the model-info object. Do not clamp `n_ctx` inside `initLlama`. A caller who asked for 8192 gets 8192. The app applies the ceiling and can tell the user it did.

- `totalMemoryBytes`, from `getTotalPhysicalMemory`. `availableMemoryMB` is free memory at info-load time and is the wrong input for a ceiling.
- `kvBytesPerToken`, K and V, for the cache type the context will actually allocate. Today that is f16. Use the real head geometry: `llama_model_n_embd / llama_model_n_head` as the head dim, times `llama_model_n_head_kv`, times `llama_model_n_layer`, times 2 (K and V), times the element size. Sizing from full `n_embd` is several times too high on a GQA model (a 7B with 4 KV heads looks like it needs roughly a gigabyte at 4k when it needs a few hundred megabytes). If a model does not use that head layout, say so by leaving `kvBytesPerToken` off the object rather than returning a guess.

Verify the formula against one loaded context before publishing it. `llama_state_get_size` includes more than KV, so it is a ceiling check, not the number to return.

Do not add `flash_attn`, `cache_type_k`, or `cache_type_v` in this pass. They are not read anywhere in the 0.11.2 bridge. A field that is accepted and ignored is how a dead option gets stored and synced. When a later version honors a KV type, `kvBytesPerToken` changes in the same change. The RAM table in the app gets recomputed from the new number. The type is not a user toggle.

Acceptance: model info for a known GGUF returns `kvBytesPerToken` within a small factor of the KV the context allocates at `n_ctx = 2048`. `initLlama({ n_ctx: 8192 })` still builds 8192. `optimalGpuLayers` is unchanged for the same device and model.

## 5. Say when the reasoning cap is real

`reasoning_budget` is load-time. `do_init_llama` sets `enable_thinking` from `budget != 0`, and the sampler cap is copied from `rn_ctx->params.reasoning_budget` only when `session.thinking_end_tags()` is non-empty. A template that does not advertise an end tag ignores the budget. Thinking then runs until `n_predict`, EOS, or the context fills. The README default of `-1` is unlimited, and the pre-reserve comment in the generate loop expects 2000–5000 reasoning tokens in that case.

Leave the default. Add `reasoning_budget_active` on the chat result: true only when the sampler was installed for this call. Nola can then tell the user the cap is not enforced, instead of believing 128 tokens were applied.

A per-call budget that only changes the sampler, while `enable_thinking` stays as it was at init, is safe for the cache and is worth doing after section 1. The rules:

- Init budget `0` means the template is rendered with thinking off. A completion cannot turn thinking on.
- Init budget non-zero means thinking is on. A completion may pass a positive `reasoning_budget` and that value is the sampler cap for this call only.
- A completion value of `0` or `-1` does not flip the template. Ignore it and keep the init cap. Flipping the template is a new `initLlama`. The render-identity change from section 2 makes a flip visible if it ever happens on the same context.

Nola's switch stays "applies on next load". The number, once this exists, applies on the next message. Until it exists, Nola keeps sending the budget at init only.

Acceptance: a template with end tags and init budget 128 reports `reasoning_budget_active: true`, and a completion override of 32 stops the think at 32 without a reload and without a KV miss. A template with no end tags reports `reasoning_budget_active: false`. Init budget 0 stays off for every completion on that context.

## Order

Each step is its own change and is shippable. Nola does not bump until 1 and 2 are in a release, because the app's device check reads `timings.prompt_n` and stops sending `config_id`.

1. Chat result carries `timings`, `context_shifted`, and `kv_cleared_reason`. Comment fix on `prompt_n`. No cache behavior change.
2. Remove `config_id`, the README rate-cap rows, and the unread `system_prompt` param. Document message `id`. Add `enable_thinking` to the render identity.
3. Context shift keeps the first-message boundary and the leading id run.
4. `kvBytesPerToken` and `totalMemoryBytes` on model info.
5. `reasoning_budget_active`, then the per-call sampler cap that cannot flip `enable_thinking`.

## Out of scope

- Hashing message text in the library, or a JavaScript tokenizer for boundaries.
- Clamping `n_ctx` inside `initLlama`, or a built-in 2048/4096/8192 table.
- Changing `getOptimalGpuLayers`, chunk size, the inter-chunk sleep, or adding a thermal thread policy.
- Flash attention and KV quant as accepted-but-ignored options.
- Putting the chat transcript in `loadSession` / `saveSession`.
- A per-call `enable_thinking`. Thinking on or off stays a load.
- Nola's form, its default of reasoning off, its 512 hard cap, and its decision to omit `use_mmap` and `use_mlock`. Those land in the app against whatever this library releases.

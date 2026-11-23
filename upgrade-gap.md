# llama.cpp upgrade gap: b9777 → b10648

> Historical audit. The checkout is llama.cpp **b12651**
> (`f5e6915c28f10330885fcb9243a93ae98d3115ab`). Live notes for the current
> bump are in `migration-b12651.md`. The body below is the b9777 → b10648
> record and is unchanged.

Audit of upgrading the vendored `cpp/llama.cpp` submodule from tag **b9777**
(`88636e178ff2972e1002cf2024cb39008eda1192`) to tag **b10648**
(`deae5ee133a3c4c56fbd46c17c8c2103af3bd643`) — **871 commits**
(~2026-06-24 → 2026-08-27).

This file is the gap analysis only: what breaks, what stays the same, and what
must change in **our** bridge before the bump is mergeable. Per `CLAUDE.md`,
`cpp/llama.cpp/` stays read-only; all fixes go in `cpp/`, `android/`, `ios/`,
scripts, and the podspec.

**Current workspace state when this audit was written:**

- `scripts/used_version.sh` already points at `LLAMA_CPP_TAG="b10648"` /
  commit `deae5ee…`
- Submodule `cpp/llama.cpp` is **still checked out at b9777** (`88636e1…`) —
  `git submodule status` confirms `88636e178…`
- Bridge code has **not** yet been adapted for the new APIs

Companion doc for the previous bump: `upgrade.md` (b9587 → b9777).

**Revision 2 note:** this version replaces line-number guesses and paraphrased
upstream behavior with direct evidence. The `b10648` tag object
(`deae5ee133…`) was already present in the submodule's local object store
(`git -C cpp/llama.cpp cat-file -e deae5ee133… → OBJECT PRESENT`) even though
the working tree is still on b9777, so every upstream claim below was checked
with `git -C cpp/llama.cpp diff/show 88636e178..deae5ee133 -- <path>` — **no
checkout, no working-tree change to the submodule** (respects the read-only
rule; the submodule pointer bump itself is still left for the follow-up step).
Every "our call site" claim was checked by reading the current `cpp/*.{h,cpp}`
files directly. Two claims from revision 1 turned out to be wrong and are
corrected in place (see §7 and the ngram note below).

---

## TL;DR

- **Not a drop-in bump.** Core inference architecture is unchanged, but there
  are **multiple compile-breaking API changes** that hit our bridge directly,
  plus one **mobile-GPU-specific behavior change** that deserves a product
  decision, not just a compile fix.
- **Architecture stays valid**: Turbo Module → `PureCppImpl` / `LlamaCppModel`
  → common + llama C API → mtmd. Same lifecycle (init → tokenize/chat →
  sample → decode → multimodal eval).
- **Must adapt before it builds**: load-mode remap, thinking/reasoning-budget
  field renames, mtmd helper/`text_len` updates, `common_json` migration at 3
  exact call sites, and common-source lists in CMake/podspec (smaller than
  originally scoped — see §7).
- **New, verified in this revision**: upstream's `LLAMA_LOAD_MODE_AUTO`
  default silently disables mmap on exactly the GPU backends this repo ships
  for Android — OpenCL always, Vulkan on integrated GPUs (i.e. nearly every
  mobile SoC), Hexagon always. `CLAUDE.md` calls out "maximum device
  compatibility" as an Android priority, so this needs an explicit mapping
  decision (§1), not silent inheritance of upstream's default.
- After the compile-fixes, expect **behavioral deltas** (AUTO mmap, suppress
  tokens, multi end-tags, sampling defaults) — smoke-test, not redesign.

---

## Scope of the upstream delta

| Area | Approx. commits touching it | Notes |
|---|---|---|
| `include/llama.h` | 16 | `load_mode`, sampler signature tweaks, additive APIs |
| `common/` | 114 | `common_json`, sampling, chat, params layout |
| `tools/mtmd/` | 47 | bitmap helpers, `text_len`, video/TTS/audio-gen (experimental) |
| `ggml/include/` | 17 | mostly additive |
| `src/` | 120 | model load path, backends |

Verified file-level diff stats (`git diff --stat 88636e178..deae5ee133`):
`common/` 55 files changed (+5944/−1943), `include/`+`ggml/include/` 9 files
(+207/−43), `tools/mtmd/` 41 files (+8367/−1230).

Notable upstream commits/PRs that drive our gaps:

| Commit / PR theme | Impact on us |
|---|---|
| `e6dd0e29a` / `#20834` — mmap/mlock/dio → `load_mode` | **Compile break** in `PureCppImpl` |
| `153d324bc` / `#26081` — `LOAD_MODE_AUTO` skips mmap on devices without mmap support | **Mobile GPU behavior change** — verified device matrix in §1 |
| `910196f6b` / `#25544` — multiple reasoning-budget end sequences | **Compile break** in `rn-completion` |
| `d9f918d2d` / `#27511` — `common/json.h` abstraction | **Compile break** for chat parse / schema, 3 exact call sites |
| `935cad649` / `#26520` — penalties take `n_vocab`, throw on invalid values | Handled inside `common_sampler_init`; already caught by our existing `catch (const std::exception&)` — see Behavior changes |
| mtmd helper opt + `text_len` | **Compile / runtime break** in multimodal path |
| bitmap ID FNV → SHA-256 | Behavior if anything keyed on bitmap id (we don't) |

---

## What stays the same (compatible)

Grepped symbols our bridge uses (`cpp/*.{h,cpp}` excluding the submodule)
against the b10648 tip. These remain present with compatible call shapes:

| Symbol group | Used in | Status |
|---|---|---|
| `llama_model_n_layer`, `n_embd`, `n_embd_out`, `n_ctx_train`, `n_params`, `size`, `desc`, meta getters | `SystemUtils`, `PureCppImpl`, `LlamaCppModel` | OK |
| `llama_model_get_vocab`, `llama_vocab_n_tokens`, `get_add_bos`, `is_eog`, `eos` | `PureCppImpl`, `rn-completion` | OK |
| `llama_batch_init` / `free`, `llama_n_ctx`, `llama_decode`, `llama_get_embeddings*` | `rn-completion`, embeddings paths | OK |
| `llama_get_memory`, `llama_memory_clear`, `llama_memory_seq_rm` / `seq_add`, `llama_memory_can_shift` | KV / context-shift paths | OK |
| `common_init_from_params`, `common_params` / `common_params_sampling` (struct still exists; **fields changed** — see below) | `PureCppImpl` | Init entrypoint OK; field access needs remap |
| `common_sampler_init` / `sample` / `accept` / `reset` / `free` | `rn-completion` | Signature unchanged (whitespace-only diff); internals changed |
| `common_chat_templates_init` / `apply`, `common_chat_format_example`, `common_chat_parse`, parser params | `PureCppImpl`, `rn-completion` | OK |
| `common_tokenize`, `common_token_to_piece`, `common_embd_normalize`, `common_batch_*` | utils / completion | OK |
| `mtmd_init_from_file`, `mtmd_tokenize`, `mtmd_encode_chunk`, `mtmd_get_output_embd`, `mtmd_helper_eval_chunks`, `mtmd_batch_*` | `PureCppImpl`, `rn-multimodal`, `rn-completion` | Core OK; helper/input structs changed |
| `llama_context_params` additive fields (`n_outputs_max_per_seq`), `llama_model_params` additive fields (`load_mtp`) | n/a | **Confirmed safe**: our bridge never constructs `llama_context_params` or raw `llama_model_params` directly for the main init path — it goes through `common_init_from_params`/`common_context_params_to_llama`, which fill new fields internally. `loadLlamaModelInfo` does build `llama_model_params` directly but starts from `llama_model_default_params()`, so new fields get safe defaults automatically. |
| `llama_sampler_init_penalties`, `llama_sampler_init_dry` (raw C API) | n/a | Signatures changed (`n_vocab` added, `n_ctx_train` removed) but **verified**: nothing in `cpp/*.{h,cpp}` calls these directly — only `common_sampler_init` does, internally. No action needed. |

**Verdict:** the main element of the system (load → chat/tokenize → sample →
decode → optional mtmd) remains. The gap is adaptation, not a rewrite.

---

## Hard breaks (will not compile / will mis-run)

### 1. `use_mmap` / `use_mlock` / `use_direct_io` → `load_mode`

Upstream removed the three booleans from **both** `llama_model_params` and
`common_params` (verified: `git diff include/llama.h` shows all three
deleted from `llama_model_params`; `git diff common/common.h` shows the same
three deleted from `common_params`), replacing them with:

```c
enum llama_load_mode {
    LLAMA_LOAD_MODE_AUTO       = -1, // auto-detect based on device capabilities
    LLAMA_LOAD_MODE_NONE       =  0, // no special loading mode
    LLAMA_LOAD_MODE_MMAP       =  1, // memory map the model
    LLAMA_LOAD_MODE_MLOCK      =  2, // force system to keep model in RAM
    LLAMA_LOAD_MODE_MMAP_MLOCK =  3, // mmap + force system to keep model in RAM
    LLAMA_LOAD_MODE_DIRECT_IO  =  4, // use direct I/O if available
};
LLAMA_API const char *          llama_load_mode_name(enum llama_load_mode);
LLAMA_API enum llama_load_mode  llama_load_mode_from_str(const char * str);
```

`common_model_params_to_llama` (`common/common.cpp`) now does a straight
1:1 copy — `mparams.load_mode = params.load_mode;` — nothing more, so the
copy side of the fix is trivial. The work is entirely in **deciding what to
set `load_mode` to**.

**Our call sites today** (all still valid at b9777, confirmed by reading the
current files):

- `cpp/PureCppImpl.cpp:301-302` — `loadLlamaModelInfo` builds a fresh
  `llama_model_params params = llama_model_default_params();` then sets
  `params.use_direct_io = false; // Disable direct I/O for compatibility`.
  `use_direct_io` no longer exists on this struct — **this line alone stops
  the file from compiling**.
- `cpp/PureCppImpl.cpp:586-587` — `do_init_llama` copies
  `params.use_mmap = p.use_mmap; params.use_mlock = p.use_mlock;` onto a
  `rn_common_params params` (which wraps `common_params`) — same problem,
  `common_params` no longer has these fields.
- `cpp/PureCppImpl.cpp:706-707,735-736,852-853` — the JSI option parser
  declares `bool use_mmap = true; bool use_mlock = false;`, reads them via
  `SystemUtils::setIfExists(runtime, options, "use_mmap"/"use_mlock", ...)`,
  and copies them into `InitLlamaParams`. **Important**: this default of
  `true` is baked in at the C++ layer — `setIfExists` only overwrites when
  the JS caller actually passed the key, so today there is no way to tell
  "caller didn't pass `use_mmap`" apart from "caller passed `use_mmap: true`
  explicitly" once the value reaches `do_init_llama`. This matters for the
  decision below.
- `src/NativeRNLlamaCpp.ts:50-51` — still declares `use_mmap?: boolean` /
  `use_mlock?: boolean` on the JS spec. **No JS-facing change needed** —
  only the native mapping.

**Verified upstream mapping** — read directly from `common/arg.cpp`'s CLI
flag handlers (the canonical "how do the old booleans become `load_mode`"
reference, since the CLI keeps the old flags as deprecated aliases):

```cpp
// --mlock  → always MLOCK, never MMAP_MLOCK
params.load_mode = LLAMA_LOAD_MODE_MLOCK;
// --mmap / --no-mmap
params.load_mode = value ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
// -dio/--direct-io / -ndio/--no-direct-io
params.load_mode = value ? LLAMA_LOAD_MODE_DIRECT_IO : LLAMA_LOAD_MODE_NONE;
// -lm/--load-mode <auto|none|mmap|mlock|mmap+mlock|dio>
```

Note the CLI treats these as **mutually overriding** (last flag on the
command line wins — there's a `LOG_WRN` in `arg.cpp` if you combine
`--load-mode` with any of the legacy flags). That works for a CLI where flags
are sequential, but our JS API exposes `use_mmap` and `use_mlock` as two
**independent, simultaneously-settable** booleans, so we can't just port the
"last one wins" logic — we need to actually combine them. Recommended
mapping for our two booleans (uses `MMAP_MLOCK` for the one combination the
CLI mapping never produces, since it's exactly what that enum value is for):

| `use_mmap` | `use_mlock` | `load_mode` |
|---|---|---|
| unset / not passed | unset / not passed | `LLAMA_LOAD_MODE_AUTO` (needs an `std::optional<bool>` at the parse layer — see decision below) |
| `true` | `false` | `LLAMA_LOAD_MODE_MMAP` |
| `false` | `false` | `LLAMA_LOAD_MODE_NONE` |
| `true` | `true` | `LLAMA_LOAD_MODE_MMAP_MLOCK` |
| `false` | `true` | `LLAMA_LOAD_MODE_MLOCK` |

**Decision point — mobile GPU behavior, not just a compile fix.** Upstream's
default is now `AUTO`, and `AUTO` is not a no-op. Verified in
`src/llama-model.cpp:1384-1391`:

```cpp
if (ml.use_mmap && params.load_mode == LLAMA_LOAD_MODE_AUTO) {
    for (const auto & dev : devices) {
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev.dev, &props);
        if (!props.caps.mmap_support) { ml.use_mmap = false; break; }
    }
}
```

i.e. under `AUTO`, if **any** backend device involved in the load reports
`mmap_support = false`, mmap is silently turned off for the whole load. I
grepped every backend's `mmap_support` value at b10648
(`ggml/src/ggml-*/ggml-*.cpp`) since this repo ships exactly the backends
that matter here:

| Backend | `mmap_support` | Relevance to us |
|---|---|---|
| CPU | `true` | fine |
| Metal (iOS) | `true` | fine — iOS is unaffected |
| OpenCL (Android) | **`false`** (always) | every OpenCL-offloaded Android load loses mmap under AUTO |
| Vulkan (Android, currently disabled per CLAUDE.md §4.2) | **`!ctx->is_integrated_gpu`**, i.e. `false` on integrated GPUs | nearly all Android SoCs are integrated GPUs — if/when Vulkan is re-enabled, this hits almost every device |
| Hexagon (Android, optional) | **`false`** (always) | every Hexagon-offloaded load loses mmap under AUTO |

Combined with the note above that our JSI layer defaults `use_mmap = true`
unconditionally (not "unset"), the two paths diverge:

- **If we keep today's literal default** (JS omits `use_mmap` → C++ sees
  `true` → maps to explicit `LLAMA_LOAD_MODE_MMAP`): we get today's exact
  behavior (mmap always forced on) but we **opt out of** the new
  device-capability safety check entirely, because that check only fires
  `if (params.load_mode == LLAMA_LOAD_MODE_AUTO)` — explicit `MMAP` bypasses
  it (`llama-model-loader.cpp:554`: `use_mmap = load_mode == MMAP || ... ||
  load_mode == AUTO`, forced unconditionally for explicit `MMAP`). Whatever
  correctness/perf problem upstream added this check to prevent on
  iGPU/OpenCL/Hexagon paths, we'd still be exposed to it.
- **If we adopt upstream's default** (JS omits `use_mmap`/`use_mlock` → C++
  passes `LLAMA_LOAD_MODE_AUTO`, only overriding when the app explicitly sets
  one of the two booleans): matches upstream's own defaults and gets the new
  device-aware fallback for OpenCL/Hexagon/integrated-Vulkan, but is a
  **behavior change** from today (mmap may now silently turn off on those
  backends when it didn't before) and requires re-plumbing the option parser
  from `bool use_mmap = true` to `std::optional<bool>` so "not passed" is
  distinguishable from "passed as true".

Given `CLAUDE.md`'s explicit "maximum device compatibility" priority for
Android and the fact that OpenCL/Hexagon are enabled backends here, the
second option is the one that matches this repo's stated priorities — but
it's a product decision (silent perf/behavior change on existing installs),
not something to pick unilaterally while doing a mechanical bump. Flag this
to whoever signs off on the bump before merging.

`use_direct_io` in `loadLlamaModelInfo` (`PureCppImpl.cpp:302`): this call
loads with `n_gpu_layers = 0` (CPU only) purely to read metadata, and the
original intent ("disable direct I/O for compatibility") is already the
default behavior of `AUTO` on the CPU backend (`mmap_support = true` for
CPU, and direct I/O is never auto-selected — `AUTO` only chooses between
`NONE` and `MMAP` per the code above). Simplest correct fix: delete the line
entirely and let the field default (`llama_model_default_params()` already
sets `load_mode = LLAMA_LOAD_MODE_AUTO`, confirmed at
`src/llama-model.cpp:2633`).

---

### 2. `thinking_end_tag` → `thinking_end_tags`

`common_chat_params` (`common/chat.h`) changed, verified directly:

```diff
  std::string                         thinking_start_tag;  // e.g., "<think>"
- std::string                         thinking_end_tag;    // e.g., "</think>"
+ std::vector<std::string>            thinking_end_tags;   // e.g., "</think>"
```

Only the end tag became plural; `thinking_start_tag` stays a single string —
templates apparently only ever need one start marker but can register
several end markers (e.g. `</think>` plus a tool-call opener).

**Our call site:** `cpp/rn-completion.cpp:1122-1126`:

```cpp
if (chat_params.supports_thinking && !chat_params.thinking_end_tag.empty()) {
    cmpl_options.reasoning_budget_tokens    = rn_ctx->params.reasoning_budget;
    cmpl_options.reasoning_budget_start_tag = chat_params.thinking_start_tag;
    cmpl_options.reasoning_budget_end_tag   = chat_params.thinking_end_tag;
}
```

`cmpl_options` is a `CompletionOptions` (`cpp/rn-utils.h:104`), which itself
declares `std::string reasoning_budget_end_tag;` — singular, all the way
through to `rn-completion.cpp:215-226` where it's tokenized into
`sampling_params.reasoning_budget_end` (see §3).

**Recommended fix — promote to plural end-to-end, don't just truncate to
`.front()`.** Upstream's own server code
(`tools/server/server-schema.cpp:394-412`) treats `reasoning_budget_end_tags`
as the primary, user-facing field and only keeps the singular
`reasoning_budget_end_tag` as a backward-compat alias that's converted to a
one-element list. Since this is our own internal struct (not JS-facing —
`NativeRNLlamaCpp.ts` doesn't expose per-tag control today), the clean fix
is:

- `cpp/rn-utils.h:104`: `std::string reasoning_budget_end_tag;` →
  `std::vector<std::string> reasoning_budget_end_tags;`
- `cpp/rn-completion.cpp:1122-1126`: guard on
  `!chat_params.thinking_end_tags.empty()` and assign the whole vector
  (`cmpl_options.reasoning_budget_end_tags = chat_params.thinking_end_tags;`)
  instead of taking the first tag.
- §3 below shows the corresponding change in the tokenizing call site.

Taking only `.front()` here (the doc's original suggestion) would compile
and work for single-end-tag templates, but silently under-constrains any
template that registers a tool-call-opener as a second end sequence — the
budget sampler would then only recognize `</think>` and let the model run
past budget if it emits the other end sequence first. Since we already own
both ends of this internal struct, there's no reason to take the lossy
shortcut.

---

### 3. `reasoning_budget_end` type change (and how the forced-message tokenization also comes into play)

`common_params_sampling` (`common/common.h`), verified directly:

```diff
  std::vector<llama_token>  reasoning_budget_start;           // start tag token sequence
- std::vector<llama_token>  reasoning_budget_end;             // end tag token sequence
+ std::vector<llama_tokens> reasoning_budget_end;             // end tag token sequences; the first tag is used as the forcing sequence
  std::vector<llama_token>  reasoning_budget_forced;          // forced sequence (message + first end tag)
```

Only `reasoning_budget_end` changes shape (to a vector of token-sequences,
one per end tag); `reasoning_budget_start` and `reasoning_budget_forced`
stay flat `std::vector<llama_token>`. The lower-level
`common_reasoning_budget_init(vocab, start_seqs, end_seqs, forced_tokens,
budget, ...)` (`common/reasoning-budget.h`) also takes multiple *start*
sequences, but that's internal to `common_sampler_init` — we never call it
directly, so no action needed there.

**Our call site:** `cpp/rn-completion.cpp:215-226`:

```cpp
if (!options.reasoning_budget_start_tag.empty() && !options.reasoning_budget_end_tag.empty()) {
    sampling_params.reasoning_budget_tokens  = options.reasoning_budget_tokens;
    sampling_params.reasoning_budget_start   = common_tokenize(rn_ctx->vocab, options.reasoning_budget_start_tag, false, true);
    sampling_params.reasoning_budget_end     = common_tokenize(rn_ctx->vocab, options.reasoning_budget_end_tag,   false, true);
    if (!options.reasoning_budget_message.empty()) {
        sampling_params.reasoning_budget_forced = common_tokenize(rn_ctx->vocab,
            options.reasoning_budget_message + options.reasoning_budget_end_tag, false, true);
        sampling_params.reasoning_budget_message = options.reasoning_budget_message;
    } else {
        sampling_params.reasoning_budget_forced = sampling_params.reasoning_budget_end;
    }
}
```

Two things need to change here, both type errors today:

1. `reasoning_budget_end` assignment needs to become a loop over
   `options.reasoning_budget_end_tags` (per §2), pushing one
   `common_tokenize(...)` result per tag.
2. `reasoning_budget_forced = sampling_params.reasoning_budget_end;` no
   longer type-checks (`vector<llama_token>` = `vector<llama_tokens>`) —
   needs `.front()`.

Upstream's exact pattern (`tools/server/server-schema.cpp:394-425`,
read directly):

```cpp
ctx.params.sampling.reasoning_budget_end.clear();
for (const auto & t : data.at("reasoning_budget_end_tags")) {
    std::string tag = t.get<std::string>();
    if (!tag.empty()) {
        ctx.params.sampling.reasoning_budget_end.push_back(common_tokenize(ctx.vocab, tag, false, true));
    }
}
// ... later, when a forced message is set:
if (!ctx.params.sampling.reasoning_budget_end.empty()) {
    llama_tokens end_tag = ctx.params.sampling.reasoning_budget_end.front();
    llama_tokens message_tokens = common_tokenize(ctx.vocab, message, false, true);
    end_tag.insert(end_tag.begin(), message_tokens.begin(), message_tokens.end());
    ctx.params.sampling.reasoning_budget_forced = std::move(end_tag);
}
```

**Worth noting while touching this code (not a hard break, but a real
behavioral difference from upstream):** our current code tokenizes the
**concatenated string** `message + end_tag` as one unit
(`common_tokenize(vocab, options.reasoning_budget_message +
options.reasoning_budget_end_tag, ...)`), whereas upstream tokenizes the
message and the end tag **separately** and concatenates the *token* vectors.
BPE tokenization is not generally compositional across a concatenation
boundary (a token can span what would otherwise be a word/space boundary
between the two strings), so these two approaches can produce different
token sequences for the same message+tag pair. This is pre-existing in our
code (not introduced by this bump), but since the line has to be touched
anyway for the type fix, it's a good opportunity to align with upstream's
token-level concatenation for parity — flagging as a suggested improvement,
not a required one.

---

### 4. MTMD bitmap helpers gained an `opt` parameter

Verified in `tools/mtmd/mtmd-helper.h`:

```c
// b9777
mtmd_helper_bitmap_init_from_file(ctx, fname, placeholder);
mtmd_helper_bitmap_init_from_buf(ctx, buf, len, placeholder);

// b10648 — opt is passed BY VALUE, after placeholder
struct mtmd_helper_init_opt { struct mtmd_helper_video_init_params video_params; };
MTMD_API struct mtmd_helper_init_opt mtmd_helper_init_opt_default(void);

mtmd_helper_bitmap_init_from_file(ctx, fname, placeholder, opt);
mtmd_helper_bitmap_init_from_buf(ctx, buf, len, placeholder, opt);
```

`opt` only carries video decode settings (fps / ffmpeg dir / timestamp
interval) — irrelevant for still images and audio. Pass
`mtmd_helper_init_opt_default()`.

**Our call sites** (`cpp/rn-multimodal.cpp:73-91`, `load_bitmap_from_uri`,
all three branches):

```cpp
wrapper = mtmd_helper_bitmap_init_from_file(ctx, url.substr(7).c_str(), /* placeholder */ false);   // line 76
wrapper = mtmd_helper_bitmap_init_from_buf(ctx, bytes.data(), bytes.size(), /* placeholder */ false); // line 82
wrapper = mtmd_helper_bitmap_init_from_file(ctx, url.c_str(), /* placeholder */ false);              // line 84
```

Add `, mtmd_helper_init_opt_default()` to each of the three calls.

**Side note (confirmed, harmless for us):** bitmap IDs switched from FNV to
a SHA-256 hex string (`mtmd-helper.h` comment diff). We don't cache or key
anything on `mtmd_bitmap` id, so no action needed.

---

### 5. `mtmd_input_text.text_len` (runtime hazard, not just a compile error)

```c
struct mtmd_input_text {
    const char * text;
    size_t text_len;   // NEW — required, no default
    bool add_special;
    bool parse_special;
};
```

This is a plain C struct with no constructor, so **omitting `text_len` does
not fail to compile** — it leaves the field value-initialized to whatever
garbage/zero is on the stack, and upstream's tokenize path does
`input_text.assign(text->text, text->text_len)`. A stray `0` here silently
truncates the multimodal prompt to an empty string (not a crash, just wrong
output) — this is the one item in this doc that will pass a build and still
misbehave at runtime if missed, so it deserves its own smoke test beyond
"does it compile."

**Our call sites** (both set `.text` but never `.text_len`):

- `cpp/rn-completion.cpp:1152-1155`:
  ```cpp
  mtmd_input_text input_text;
  input_text.text          = cmpl_options.prompt.c_str();
  input_text.add_special   = true;
  input_text.parse_special = true;
  ```
  Add `input_text.text_len = cmpl_options.prompt.size();`
- `cpp/rn-multimodal.cpp:112-115`:
  ```cpp
  mtmd_input_text input_text;
  input_text.text          = mtmd_get_marker(mtmd_ctx);
  input_text.add_special   = false;
  input_text.parse_special = true;
  ```
  Add `input_text.text_len = strlen(mtmd_get_marker(mtmd_ctx));`

---

### 6. Chat / schema APIs now take `common_json`, not `nlohmann::ordered_json`

Upstream introduced `common/json.h` + `common/json.cpp` (`#27511`): a
pimpl'd, ordered-JSON wrapper over `nlohmann::json` with a deliberately
narrow surface (verified by reading `common/json.h` in full — no `.hpp`
value-header include and no implicit conversion to/from `nlohmann::json` in
either direction; the *only* interop points are `common_json::parse(const
std::string&)` and `.dump(int indent = -1)`). `common/chat.h` dropped its
`using json = nlohmann::ordered_json;` alias entirely in favor of
`common_json`.

Affected signatures we call, verified via `git diff common/chat.h`:

```cpp
std::vector<common_chat_msg>  common_chat_msgs_parse_oaicompat(const common_json &);
std::vector<common_chat_tool> common_chat_tools_parse_oaicompat(const common_json &);
common_json                   common_chat_msg::to_json_oaicompat(bool concat_typed_text = false) const;
```

(`common_chat_continuation_parse` and `common_chat_msg_delimiters_parse`
also moved to `common_json`, but we don't call either today — no action
needed unless we later add `continue_final_message` or
`message_delimiters` support.)

**Our bridge** still builds everything as `nlohmann::ordered_json` (aliased
`json` in `cpp/rn-utils.h:28`). Exactly **three** call sites cross the
boundary and need an explicit round-trip through `dump()`/`parse()` — there
is no shortcut, since neither type can construct from the other directly:

- `cpp/rn-completion.cpp:844`:
  ```cpp
  chat_msgs = common_chat_msgs_parse_oaicompat(effective_messages); // effective_messages is `const json&` (nlohmann)
  ```
  → `common_chat_msgs_parse_oaicompat(common_json::parse(effective_messages.dump()));`
- `cpp/rn-completion.cpp:978`:
  ```cpp
  template_inputs.tools = common_chat_tools_parse_oaicompat(options.tools); // options.tools is nlohmann json
  ```
  → `common_chat_tools_parse_oaicompat(common_json::parse(options.tools.dump()));`
- `cpp/rn-completion.cpp:1302`:
  ```cpp
  choice["message"] = parsed_msg.to_json_oaicompat(); // choice is nlohmann json; RHS is now common_json
  ```
  → `choice["message"] = json::parse(parsed_msg.to_json_oaicompat().dump());`
  (`json` here is our own `nlohmann::ordered_json` alias, not `common_json`.)

**Decision already made for us by upstream's API shape**: the doc's original
"pick one: migrate everything to `common_json`, or convert at the boundary"
resolves to the boundary-conversion approach by construction — `common_json`
doesn't expose enough of its internals (no iteration into raw values without
its own accessor API, no `nlohmann` escape hatch) to make a full migration
of our JSI-facing code (`LlamaCppModel.cpp`'s `jsiValueToJson` and friends)
a small change. Converting at these 3 call sites is not just the minimal
first patch, it's the only patch that doesn't also require rewriting the
JSI↔JSON layer.

---

### 7. Common sources: CMake / podspec drift — **smaller than originally scoped**

Verified precisely with `git diff --name-status` restricted to `common/`
between b9777 and b10648 (not stat, actual add/remove list):

```
removed:  common/json-partial.{h,cpp}
removed:  common/regex-partial.{h,cpp}
added:    common/json.{h,cpp}
added:    common/subproc.{h,cpp}
added:    common/trie.{h,cpp}
```

Our `android/CMakeLists.txt:64-90` and `RNLlamaCpp.podspec:31-52` both
currently list `regex-partial.{h,cpp}` and `json-partial.{h,cpp}`, and list
neither `json.cpp`, `subproc.cpp`, nor `trie.cpp`. **Corrections from
revision 1 of this doc, checked by grepping who actually `#include`s each
new/removed header across the whole b10648 tree:**

| File | Verified action | Evidence |
|---|---|---|
| `common/json.{h,cpp}` | **Add** to both CMakeLists and podspec | `#include "json.h"` reached from `common/chat.cpp`, `common/arg.cpp`, `common/jinja/caps.cpp`, `common/jinja/value.cpp`, `common/peg-parser.h`, `common/json-schema-to-grammar.h`, `common/chat-auto-parser.h` — **all files we already compile**. Won't link without it. |
| `common/trie.{h,cpp}` | **Add** to both CMakeLists and podspec | `#include "trie.h"` reached from `common/peg-parser.cpp` and `common/reasoning-budget.cpp` — **both already compiled by us**. Won't link without it. |
| `common/json-partial.{h,cpp}`, `common/regex-partial.{h,cpp}` | **Remove** from both — confirmed zero-risk | At b9777, `git grep '"regex-partial.h"\|"json-partial.h"' -- common/` shows **only their own `.cpp` files** including these headers — nothing else in `common/` (in particular, not `chat.cpp` or `peg-parser.cpp`) ever used them. They were already dead weight in our static lib; deleting them has no functional impact, no replacement code needed anywhere. |
| `common/subproc.{h,cpp}` | **Do not add** — revision 1's "only if something we compile pulls it" is answered: nothing does | `#include "subproc.h"` is reached only from `tools/server/server-mcp.cpp`, `server-models.cpp`, `server-tools.cpp` — all server-only files, none of which we compile. |
| `common/ngram-map.{h,cpp}`, `common/ngram-mod.{h,cpp}` | **Must add to both — this row was wrong twice; corrected against a real linker failure** | Revision 1 said "add if speculative stays"; revision 2 wrongly "corrected" that to "no change needed", claiming `ngram-mod.cpp` never existed. **Both claims about existence were wrong**: `ngram-map.{h,cpp}` and `ngram-mod.{h,cpp}` exist at *both* b9777 and b10648 (`git ls-tree 88636e178 common/ \| grep ngram`), and upstream's `common/CMakeLists.txt` lists them right next to `speculative.cpp`. Revision 2's error was checking *file existence* instead of *symbol references*. The actual iOS link failed with `Undefined symbols: common_ngram_map_begin(...), common_ngram_mod::add(int const*), common_ngram_simple_draft(...)` — all referenced from `speculative.o`, which is pulled into the link because `arg.cpp` (which we compile) references `common_speculative`. Fixed by adding both files to `android/CMakeLists.txt` and `RNLlamaCpp.podspec`. **Lesson: a source-list audit must resolve symbols, not just check that files exist.** |

Net effect: **two files added, two files removed**, not the four-plus-maybes
originally scoped.

Also refresh copied public headers:

- Android: `android/src/main/cpp/include/llama.h` (and related) via the
  Android build scripts
- iOS: `scripts/build_apple_external.sh` / xcframework + `ios/include`

Prebuilts (`jniLibs/*.so`, `ios/libs/llama.xcframework`) must be rebuilt at
`deae5ee` / `b10648` — version skew between bridge headers and shipped
libs is undefined behavior.

---

### 8. Upstream dropped the iOS **simulator** slice from the release xcframework

**Found while building, not during the source audit — this section was added
after the fact.** It is not a compile break in our bridge; it breaks
`pod install` and ships a broken package to consumers.

Upstream narrowed which platforms the *release* workflow builds:

```diff
# .github/workflows/release.yml
- ./build-xcframework.sh
+ ./build-xcframework.sh macos ios-device
```

Commit `805984d67` (PR #27252, "ci : reduce builds in build-xcframework.sh"),
whose message states plainly: *"release.yml builds only macos and ios-device to
cut the build time."* It is a CI cost reduction, **not** a deprecation of
simulator support — `build-xcframework.sh` still builds `ios-sim` and gained an
explicit build-list CLI in the same commit.

**Consequence:** `llama-b10648-xcframework.zip` — the only Apple asset on the
release (verified against the GitHub releases API) — contains only
`macos-arm64_x86_64` and `ios-arm64`. **No `ios-arm64_x86_64-simulator`.**

**How it surfaced:** our `scripts/build_apple_external.sh` downloaded that zip,
only *warned* when the simulator slice was absent, and then copied a
hand-maintained `info.plist.change` over the framework's `Info.plist` — a
template that hardcodes **both** slice declarations. The result was an
xcframework whose Info.plist advertised a slice that did not exist on disk, and
CocoaPods reported it as:

```
[!] Unable to install vendored xcframework `llama` for Pod `RNLlamaCpp`,
    because it contains both static and dynamic frameworks.
```

which is misleading — nothing was static. Two further traps made this sticky:

- The "already installed" check used **OR** across the two slices
  (`[ -d ios-arm64 ] || [ -d ...simulator ]`), so a device-only framework looked
  valid and every plain re-run silently skipped the rebuild. Only `--force`
  could get past it.
- CI cached the framework under key `ios-framework-${runner.os}-v2` with
  `restore-keys: ios-framework-` — **no llama.cpp version in the key**, so a
  framework built for an older tag would be restored for a new one.

**Why this is a shipping blocker, not a local annoyance:** this package vendors
the xcframework into the npm tarball. `package.json`'s `files` includes `"ios"`,
which whitelists `ios/libs/llama.xcframework/**` even though it is gitignored
(confirmed with `npm pack --dry-run`). Consumers never compile llama.cpp — they
get whatever slices the maintainer had locally at publish time. Publishing with
a device-only framework therefore breaks the iOS Simulator for **every** app
developer using this library.

**Adaptation applied:** build both iOS slices from the pinned submodule instead
of downloading the release zip —
`(cd cpp/llama.cpp && ./build-xcframework.sh ios-sim ios-device)` — then install
the result into `ios/libs/`. `xcodebuild -create-xcframework` writes a correct
`Info.plist` for exactly the slices it was given, so the hand-maintained
`info.plist.change` template is obsolete (and was the proximate cause of the
misleading error; it should be deleted). Missing-slice checks became hard
errors, the skip check now requires **both** slices (AND, not OR), and the CI
cache key is now derived from `hashFiles('scripts/used_version.sh')`.

There is no alternative upstream artifact to consume: the workflow that does
build every slice (`build-apple.yml`) uploads a GitHub Actions artifact with
`retention-days: 1`, which is neither durable nor publicly downloadable, and
`docs/xcframework.md` still documents consuming the same release zip.

**Unrelated observation, worth a follow-up:** the framework ships ~74 MB of
dSYMs against a 7.7 MB binary, and they go into the npm tarball (129.9 MB
packed / 440.2 MB unpacked). Stripping debug symbols from the published package
would cut it by roughly half.

---

### 9. `cpuparams_batch` never normalized → SIGSEGV on every `initLlama`

**Found by an actual crash on the iOS simulator, not by the source audit.**

`common_cpu_params::n_threads` defaults to **-1** (`common/common.h:69`). Our bridge
set `params.cpuparams.n_threads` but never `params.cpuparams_batch`, because
upstream normally fixes this inside `common_params_parse()` — which we skip, since
we build `common_params` programmatically rather than from argv.

Harmless at b9777. At b10648 a new `common_threadpools` (0 occurrences at b9777,
3 at b10648) builds a **batch** threadpool from `cpuparams_batch`, and
`ggml_threadpool_new_impl()` (`ggml-cpu.c:3306`) computes:

```c
const size_t workers_size = sizeof(struct ggml_compute_state) * tpp->n_threads;
struct ggml_compute_state * workers = ggml_aligned_malloc(workers_size);
memset(workers, 0, workers_size);   // SIGSEGV: workers == NULL
```

With `n_threads = -1` the multiply underflows to a huge `size_t`, the allocation
returns null, and `memset` writes to address 0. The crash report confirms it
exactly: `x0 = 0` (destination) and `x2 = 0xfffffffffffffde0` = -544, i.e.
`sizeof(ggml_compute_state) == 544` times -1.

**Adaptation applied** — mirror `arg.cpp:891-892` in `do_init_llama()`:

```cpp
postprocess_cpu_params(params.cpuparams,       nullptr);
postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);
```

The second call makes `cpuparams_batch` inherit `cpuparams` (role_model); the
first auto-detects via `common_cpu_get_num_math()` when `n_threads <= 0`. This was
platform-independent — it would have crashed identically on Android and on real
hardware.

---

### 10. `vendor/hash` (new) must be compiled into mtmd — Android link break

**Found by the Android link, after iOS had already built and run successfully.**

§4's note that the bitmap ID change (FNV → SHA-256) was "harmless unless we cache
by bitmap id" was right about runtime behavior and **wrong about packaging**:
`vendor/hash/` is an entirely new directory at b10648 (absent at b9777), and
`mtmd-helper.cpp:386` now calls `hash_sha256_hex()` from it. Upstream links it via
`target_link_libraries(mtmd PRIVATE ... vendor::hash ...)`
(`tools/mtmd/CMakeLists.txt:83`).

```
ld.lld: error: undefined symbol: hash_sha256_hex(void const*, unsigned long)
>>> referenced by mtmd-helper.cpp:386
>>>               mtmd-helper.cpp.o:(mtmd_helper_bitmap_init_from_buf)
```

**Why iOS passed and Android failed** — the two prebuilts differ:

| Prebuilt | exports `hash_sha256_hex` |
|---|---|
| iOS `llama.framework` (built from source, §8) | **yes** — `T __Z15hash_sha256_hexPKvm` |
| Android `libllama.so` (built by `build_android_external.sh`) | **no** — 0 symbols |

So iOS resolved the symbol from the framework by luck of how that binary is built,
while Android had nothing to resolve against. A green iOS build is not evidence
for Android whenever a symbol can come from a prebuilt.

**Adaptation applied** (`android/CMakeLists.txt`): compile
`vendor/hash/hash.cpp` + `vendor/hash/sha256/sha256.c` into the `mtmd` target, add
`vendor/hash` to its include dirs (`sha256.c` includes `"rotate-bits/rotate-bits.h"`),
and set `-w` on the vendored C file as upstream does. `hash.h` declares only
`hash_sha256_hex` and `hash.cpp` references only sha256, so sha1/xxhash are
deliberately excluded. The podspec is intentionally left unchanged — iOS links
today, and adding a static definition beside the framework's exported one risks a
duplicate-symbol clash for no present benefit.

---

## Behavior changes (after it compiles)

These do not necessarily break the build, but change observable behavior.

### Sampling / grammar

- `common_sampler_init` now **throws** `std::invalid_argument` if
  `penalty_repeat` ≤ 0 or non-finite, or if `penalty_freq` / `penalty_present`
  are non-finite (`src/llama-sampler.cpp:2743`, `llama_sampler_init_penalties`,
  now reached with a `[[noreturn]]`-style throw path added upstream). **We
  are exposed to this today**: `cpp/rn-completion.cpp:167-175`'s `applyF`
  lambda copies whatever float the JS caller passes for `repeat_penalty`
  straight into `sampling_params.penalty_repeat` with only a NaN guard — a
  caller passing `repeat_penalty: 0` (plausible if someone assumes 0 means
  "disabled", by analogy with other penalty fields) now throws where it
  previously didn't. **Verified safe**: `common_sampler_init` is called at
  `rn-completion.cpp:229`, inside the same function body as the
  `catch (const std::exception& e)` handlers at `rn-completion.cpp:685`
  (`run_completion`) and `rn-completion.cpp:1331` (`run_chat_completion`) —
  `std::invalid_argument` derives from `std::exception`, so this surfaces as
  a normal inference error, not a crash. No code change required; just add
  the "bad penalty value → clean rejection, not silent old behavior" case to
  the smoke-test list (already present in §5 checklist below).
- Model **suppress tokens** (`tokenizer.ggml.suppress_tokens`, exposed via
  the new `llama_vocab_get_suppress_tokens()`) are automatically merged into
  logit bias at `-INFINITY` inside sampler init. Previously only user
  `logit_bias` applied. No call-site change needed (we don't call the vocab
  getter directly), just a possible generation-quality delta on models that
  declare this metadata key.
- `penalty_last_n` / `dry_penalty_last_n`: comment no longer documents
  `-1 = context size`. Default `dry_penalty_last_n` changed **-1 → 64**.
- `llama_sampler_init_penalties` / `llama_sampler_init_dry` C API signatures
  changed (`n_vocab` added to penalties; `n_ctx_train` removed from dry); we
  go through `common_sampler_init` exclusively (verified — no direct call
  site in our bridge), so we inherit the new behavior without any signature
  change on our side.

### Chat / tools

- Large `common/chat.cpp` churn (1477 lines changed): more PEG formats (e.g.
  `COMMON_CHAT_FORMAT_PEG_MINIMAX_M3`), multi end-tags, and adjusted
  `grammar_lazy` / `include_grammar` rules for tools + response_format.
  Tool-calling apps should regression-test templates they care about (Qwen,
  Llama, Gemma, etc.).

### Load path

- Default `load_mode = AUTO`: may turn off mmap when a backend device lacks
  mmap support — **verified device matrix in §1**: this hits OpenCL always,
  Vulkan on integrated GPUs (i.e. most Android SoCs), and Hexagon always.
  This is the one item in this doc that's a genuine product decision, not
  just an adaptation — see the decision point in §1.
- `common_params_vocoder` removed; TTS-related fields moved. We do not use
  vocoder today — no action unless we add TTS later.

### Multimodal

- `mtmd_context_params.device` added (`nullptr` default is fine; we already
  set `use_gpu` from `n_gpu_layers`).
- Substantial new **experimental** surface: video decode helpers expanded,
  plus an entirely new audio-generation helper API
  (`mtmd_helper_gen_audio_*`, `mtmd_helper_gen_audio_inp`, TTS model files
  under `tools/mtmd/models/*tts*.cpp` and `pockettts-*.cpp`). None of this
  is called anywhere in our bridge today — out of scope unless we productize
  video/TTS (see "Explicitly out of scope" below).
- Bitmap ID algorithm change (FNV → SHA-256) — only matters for id-based
  caches; we don't have one.

### Persistence

- `LLAMA_SESSION_VERSION` 9 → 10, `LLAMA_STATE_SEQ_VERSION` 2 → 3 (verified
  directly in `include/llama.h`). Native session / seq-state files from
  older builds will not load. We do not currently expose a stable
  session-file API to JS; flag for awareness.

### Upstream helpers we do **not** use (awareness only)

- Free functions `common_context_seq_{rm,add,cp}` became `static`; public
  API is now `common_memory::{seq_rm,seq_add,seq_cp}`. We call
  `llama_memory_*` directly — unaffected.
- New small additive APIs we don't currently need but could adopt later for
  richer `loadLlamaModelInfo` output: `llama_version()`,
  `llama_ftype_name()` / `llama_model_ftype()`, `llama_load_mode_name()` /
  `llama_load_mode_from_str()` (the latter could replace hand-rolled string
  parsing if we ever expose `load_mode` as a string option instead of two
  booleans).
- Server-only CORS / MCP / router / download surfaces — out of mobile
  package scope (`library-release.md`).

---

## Suggested adaptation order

1. **Bump submodule** to `deae5ee` and align `scripts/used_version.sh`
   (already pointed at b10648 in this workspace).
2. **Get a product decision on `load_mode` defaults** (§1) before writing
   the mapping code — this determines whether the JSI option parser needs
   to change from `bool use_mmap = true` to `std::optional<bool>`, which is
   a slightly bigger change than a pure compile fix.
3. **Fix packaging**: add `json.cpp` / `trie.cpp`; remove `json-partial` /
   `regex-partial` from `android/CMakeLists.txt` + `RNLlamaCpp.podspec`. Do
   **not** add `subproc.cpp` or `ngram-map.cpp` (verified not needed — §7).
4. **Fix compile breaks in bridge** (order roughly by dependency):
   - `load_mode` mapping in `PureCppImpl` (per the decision in step 2)
   - `thinking_end_tags` (promote our `CompletionOptions` field to a vector
     — §2) + `reasoning_budget_end` vector-of-sequences (§3)
   - mtmd `opt` + `text_len` (§4, §5)
   - `common_json` boundary at the 3 exact call sites in `rn-completion.cpp`
     (§6)
5. **Rebuild prebuilts** (Android jniLibs all ABIs + GPU backends; Apple
   xcframework) and refresh include copies.
6. **Smoke tests** (device / simulator with real GGUF):
   - [ ] `initLlama` with default options (AUTO load, or today's forced-mmap
     default — whichever step 2 decided)
   - [ ] `initLlama` with `use_mmap` / `use_mlock` combinations, including
     both `true` (→ `MMAP_MLOCK`)
   - [ ] `initLlama` with GPU offload on OpenCL / Hexagon (if built) —
     confirm load still succeeds and note whether mmap silently degrades
     under AUTO as expected
   - [ ] Text completion + chat completion
   - [ ] Tool calls + grammar / JSON schema
   - [ ] Reasoning-budget / thinking models, specifically one template with
     a **single** end tag and one with **multiple** end tags if available
     (validates the §2 plural fix actually does something, not just compiles)
   - [ ] Single- and multi-image chat (`mtmd_helper_eval_chunks` + batched
     path) — specifically confirm image content survives (validates §5's
     `text_len` fix; a regression here is silent, not a crash)
   - [ ] `embedImage` / audio URI load
   - [ ] Malformed grammar / bad penalty values (e.g. `repeat_penalty: 0`) →
     clean error, no crash
   - [ ] `onModelLoadProgress` still fires for model + mmproj

---

## Explicitly out of scope this bump (unless product asks)

Same mobile filter as `upgrade.md`, plus new surface added in this delta:

- EAGLE3 / DFlash / DSpark speculative decoding (extra model + RAM)
- Server router / download / WebUI / MCP
- TTS / gen-audio mtmd pipelines — **new in this delta**:
  `mtmd_helper_gen_audio_*` API and `tools/mtmd/models/*tts*.cpp`,
  `pockettts-*.cpp`, `parakeet.cpp`, `mimo-audio.cpp` model files
- Video mtmd helpers beyond what we already use (ffmpeg-backed decode,
  timestamp chunking)
- Migrating the entire bridge off `nlohmann` to `common_json` everywhere
  (boundary conversion at 3 call sites is enough for the first land — see §6
  for why a full migration also isn't a small follow-up)

---

## Summary

| Question | Answer |
|---|---|
| Is the main system element the same? | **Yes** — same bridge architecture and core llama/common/mtmd call graph. |
| Is it a drop-in submodule bump? | **No** — 7 hard API breaks found by audit, plus three more (§8 xcframework packaging, §9 threadpool SIGSEGV, §10 vendor/hash link) that only surfaced when actually building and running. |
| Biggest risk | Not a compile break at all: `load_mode = AUTO` silently disables mmap on OpenCL/Hexagon/integrated-Vulkan-GPU loads (verified device-capability matrix, §1) — needs an explicit product decision on our default mapping, not just a mechanical fix. |
| Second risk | `mtmd_input_text.text_len` (§5) compiles fine either way and fails silently (empty/truncated multimodal prompt) if missed — needs a real device smoke test, not just a green build. |
| Found only by running | §9 (`cpuparams_batch` = -1 → SIGSEGV in the new `common_threadpools`) and §10 (`vendor/hash` missing → Android link error). Neither was visible to a source-level audit; §10 was additionally masked on iOS because the framework happens to export the symbol. |
| Late addition | §8: upstream's release xcframework no longer contains an iOS simulator slice. Not a source break, but it breaks `pod install` and would ship a simulator-less binary to every npm consumer. Fixed by building both iOS slices from the pinned submodule. |
| Corrected from earlier draft | `subproc.cpp` is not needed (§7); `json-partial`/`regex-partial` removal is risk-free with no replacement code (§7). **Revision 2's ngram claim was itself wrong** — `ngram-map.cpp` and `ngram-mod.cpp` must both be added, proven by a real linker failure (§7). |
| Effort shape | Focused bridge + packaging patch (smaller than originally scoped for §7, same order of magnitude overall as `upgrade.md`), one product decision (§1), then full native rebuild + smoke tests — not a from-scratch rewrite. |

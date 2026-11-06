# CLAUDE.md – llamarn (React Native Turbo Module for llama.cpp)

This file gives the LLM assistant the context and rules needed to work effectively on this repo: a **React Native Turbo Module** written in **C++** that wraps the **llama.cpp** C library (GGUF LLM loading and inference). Best practices here are aligned with our **Android** and **iOS** implementations and with the **scripts** and **CI** used to build and ship the module.

**Official references (New Architecture):**
- [Guides – reactwg/react-native-new-architecture](https://github.com/reactwg/react-native-new-architecture#guides) (Turbo Modules, Fabric, Codegen, backwards compatibility).
- [Turbo Native Modules](https://github.com/reactwg/react-native-new-architecture/blob/main/docs/turbo-modules.md): Spec naming, Codegen config, native implementation, event emitters.
- [Supporting custom C++ types](https://github.com/reactwg/react-native-new-architecture/blob/main/docs/cxx-custom-types.md): Bridging for custom types (`Bridging<T>`, `fromJs`/`toJs`), struct generators from Codegen.

When suggesting changes, follow these guides where they apply; this file adds repo-specific constraints (prebuilt libs, Android ABIs/GPU, llama.cpp wrapper).

---

## 1. Project overview

- **Goal**: On-device LLM inference in React Native using **llama.cpp** (third-party C library, implemented in C++) to load and run **GGUF** models.
- **Architecture**: New Architecture (Fabric) **Turbo Module** with a **TypeScript spec** and a **C++ implementation** that talks to React Native via **JSI** and to llama.cpp via its C API and common code.
- **llama.cpp source access**: Full upstream source is available locally under `cpp/llama.cpp/` and can be read to understand behavior, API contracts, and edge cases.
- **Reference implementation**: For API behavior and request/response conventions, use `cpp/llama.cpp/tools/server/` as the primary upstream inspiration when it maps well to mobile constraints.
- **Performance model**: This library is a high-performance inference runtime, not app orchestration. Keep native overhead minimal; avoid adding app-level state management (e.g., conversation/history orchestration) inside the library when it can live in the React Native app layer.
- **Platforms**:
  - **iOS**: Single **Metal** backend (unified for Apple devices); prebuilt **llama.xcframework**.
  - **Android**: Multiple **ABIs** and **GPU backends** (OpenCL, Vulkan, optional Hexagon); prebuilt `.so` libraries in `android/src/main/jniLibs/`; **maximum device compatibility** is a priority.

When suggesting changes, preserve compatibility with the existing Android spec, iOS spec, and C++ bridge; do not assume a single architecture or GPU stack.

---

## 2. Turbo Module layout and conventions

### 2.1 Spec and codegen

- **TypeScript spec**: `src/NativeRNLlamaCpp.ts`.
  - Naming follows the New Architecture convention: file **must** be named `Native<MODULE_NAME>` (here `NativeRNLlamaCpp`) so Codegen can find it; the interface **must** be named `Spec` for Turbo Native Modules.
  - Defines the **Turbo Module API** (e.g. `initLlama`, `loadLlamaModelInfo`).
  - Uses `TurboModuleRegistry.getEnforcing<Spec>('RNLlamaCpp')`.
  - **Module name** must stay `'RNLlamaCpp'` to match native registration.
- **Codegen**:
  - `package.json` → `codegenConfig`: `name`: `RNLlamaCppSpec`, `type`: `modules`, `jsSrcsDir`: `src`, output dirs: `ios/generated`, `android/generated`; Android `javaPackageName` as in package.json.
  - **Do not** hand-edit files under `ios/generated/` or `android/generated/`; they are produced from the spec. Apps generate (or consume) this output at build time.
- **C++ path**: This module uses the **cross-platform C++** Turbo Module path (not the Obj-C / Java native module path). The C++ impl extends the codegen-generated C++ spec base (e.g. `NativeRNLlamaCppCxxSpec<PureCppImpl>`); iOS registers it via `registerCxxModuleToGlobalModuleMap`, Android via the same C++ binary.
- **React Native config**: `react-native.config.js` points the Android C++ module to:
  - `cxxModuleCMakeListsPath: 'CMakeLists.txt'`
  - `cxxModuleHeaderName: 'PureCppImpl'`

Any new Turbo Module method must be added in **both** the TypeScript spec and the C++ implementation, and the C++ method signatures must match what Codegen generates from the spec.

### 2.2 C++ implementation (our code)

- **Entry / registration**:
  - **iOS**: `ios/OnLoad.mm` registers the C++ module with `registerCxxModuleToGlobalModuleMap` using `PureCppImpl::kModuleName` and a factory that returns `std::make_shared<PureCppImpl>(jsInvoker)`.
  - **Android**: Same C++ code; React Native’s codegen and CMake wire the library; the module name is still `RNLlamaCpp`.
- **Core C++ files** (under `cpp/`):
  - `PureCppImpl.h` / `PureCppImpl.cpp`: Turbo Module implementation; extends the codegen base spec (e.g. `NativeRNLlamaCppCxxSpec<PureCppImpl>`); implements `initLlama`, `loadLlamaModelInfo`; owns `rn_llama_context` and creates the JSI HostObject for the model.
  - `LlamaCppModel.h` / `LlamaCppModel.cpp`: JSI HostObject that exposes completion, tokenize, detokenize, embedding, etc., and parses options/results (uses llama.cpp common code where applicable).
  - `rn-llama.h`: Declares `rn_llama_context`, `rn_common_params`, and our completion entry points (`run_completion`, `run_chat_completion`); includes llama.cpp and common headers behind pragmas to silence unused-function warnings.
  - `rn-utils.h` (and related): `CompletionOptions`, `CompletionResult`, and helpers; JSON via `nlohmann/json`; must stay compatible with `common/sampling.h`, `chat.h`, etc.
  - `rn-completion.cpp`: Implements the completion logic that uses `rn_llama_context` and llama.cpp APIs.
  - `SystemUtils.h` / `SystemUtils.cpp`: Platform utilities (e.g. optimal GPU layers, model info).
  - `build-info.cpp`: Build/version info used by the native module.

The **single integration point** with llama.cpp is this C++ bridge (and the headers it includes). Skills or docs that add “best practices” for the **third-party C library** (e.g. in `cpp/llama.cpp/`) should be clearly separated from **our** bridge; our rules in this file take precedence for `cpp/*.{h,cpp}`, `ios/`, and `android/` integration.

### 2.3 Third-party library (llama.cpp)

- **Location**: `cpp/llama.cpp/` (submodule).
- **Immutability rule**: **DO NOT MODIFY** files under `cpp/llama.cpp/` for regular feature or bugfix work. This directory mirrors the **official upstream llama.cpp repository** and must remain unchanged in this project unless the user explicitly asks for an upstream sync/version bump workflow.
- **Usage**: We use its **C API** (`llama.h`, etc.) and **common** code (e.g. `common/`, `ggml/`). We do **not** modify upstream source for our feature set; we adapt in our own `cpp/` layer.
- **Server parity guidance**: When implementing completions/chat/tool-calling behavior, align with patterns from `cpp/llama.cpp/tools/server/` where practical, then adapt for mobile runtime constraints (threading, memory pressure, startup latency, JNI/JSI boundaries).
- **Version**: Pinned in `scripts/used_version.sh` (`LLAMA_CPP_COMMIT`, `LLAMA_CPP_TAG`). CI and build scripts use this; when updating, run full Android and iOS builds and the test suite.
- **Headers**: Android uses `android/src/main/cpp/include/` copies where needed; iOS uses `ios/include/` and podspec `HEADER_SEARCH_PATHS`; our C++ code includes from `cpp/llama.cpp/` or from the copied headers as configured. Do not add duplicate or conflicting declarations.

When adding skills or best-practice docs that reference **llama.cpp** (e.g. C++ style, API usage), state explicitly that they must remain **compatible with our implementation** in this repo (Android spec, iOS spec, and the C++ bridge above). Do not assume we control the third-party API surface.

---

## 3. C++ and JSI best practices (this repo)

- **Language**: C++20 (see `pod_target_xcconfig` in `RNLlamaCpp.podspec` and Android `CMAKE_CXX_STANDARD` in `android/CMakeLists.txt`).
- **Namespace**: Put all our Turbo Module and bridge code in `namespace facebook::react`.
- **Types**: Use JSI types (`jsi::Runtime&`, `jsi::Object`, `jsi::Value`, `jsi::String`, etc.) for the Turbo Module and HostObject APIs; convert to/from C++ and JSON in the bridge (e.g. `LlamaCppModel::parseCompletionOptions`, `completionResultToJsi`).

### 3.1 Custom C++ types and bridging

When adding **new types** that cross the JS ↔ C++ boundary in the **Turbo Module spec** (e.g. a new argument or return type), prefer React Native’s **bridging** so Codegen and the C++ spec stay type-safe:

- **Bridging header**: Provide a specialization `Bridging<YourType>` with `fromJs` and `toJs` in a header included by the C++ spec impl. See [Supporting custom C++ types](https://github.com/reactwg/react-native-new-architecture/blob/main/docs/cxx-custom-types.md). Use `#include <react/bridging/Bridging.h>` and keep conversions in one place.
- **Structs**: For object-like types already defined in the JS spec, Codegen can generate a struct + bridging base (e.g. `NativeRNLlamaCppCustomTypeBridging`); then alias the type and specialize `Bridging<YourType>` to inherit from that generated bridging. Member **order** in the C++ struct must match the spec.
- **Current pattern**: We do manual JSI ↔ C++ conversion in `LlamaCppModel` (e.g. `parseCompletionOptions`, `completionResultToJsi`) for the **HostObject** API; that’s fine. When adding new **spec-level** methods with custom structs or non-primitive types, consider adding a bridging header so the C++ signature can use the custom type directly and stay consistent with the spec.

### 3.2 Event emitters (optional)

Turbo Native Modules can expose **type-safe event emitters** (e.g. `EventEmitter<Payload>`) in the spec so native code can push events to JS. See the [Turbo Native Modules – Add Event Emitting Capabilities](https://github.com/reactwg/react-native-new-architecture/blob/main/docs/turbo-modules.md#add-event-emitting-capabilities) section. If we add streaming or progress events from native to JS, use this pattern (define in spec, implement `emit...` in C++/Obj-C/Java) and document it here.

### 3.3 General C++ and JSI

- **Ownership and lifetime**:
  - Prefer `std::unique_ptr` / `std::shared_ptr`; avoid raw `new`/`delete` for our types.
  - `rn_llama_context` is owned by the Turbo Module (e.g. inside `PureCppImpl`); the HostObject holds a non-owning pointer. Ensure release paths (e.g. `release()`) clear references and free resources in a well-defined order.
- **Thread safety**: Use `std::mutex` (or equivalent) when touching shared state (e.g. `rn_ctx_`, completion flags). Prefer running heavy work (e.g. inference) on a background path and invoking JS callbacks via `CallInvoker` so the JS thread is not blocked.
- **Async work**: Use `jsInvoker_->invokeAsync(...)` (or the pattern used in the repo) for callbacks into JS; do not call into JSI from arbitrary threads without dispatching to the JS runtime.
- **Errors**: Use the existing error conventions (e.g. `rn_error_type`, throwing or returning errors in a way the JS layer can interpret); keep behavior consistent with the current API (e.g. Promise rejections or returned error objects).
- **Headers**: Include only what you need; for llama.cpp and common, use the same `#pragma GCC diagnostic` approach as in `rn-llama.h` / `rn-utils.h` to avoid pulling in unused or noisy symbols. Do not introduce new global macros that conflict with llama.cpp or React Native.
- **Library boundary discipline**: Keep the native layer focused on model lifecycle, tokenization, inference, and low-level runtime operations. Treat chat array composition, long-lived conversation state, and app UX policy as app-layer concerns unless there is a clear native performance reason.

When adding spec methods or types, keep the spec and C++ in sync and prefer the official [Turbo Modules](https://github.com/reactwg/react-native-new-architecture/blob/main/docs/turbo-modules.md) and [custom C++ types](https://github.com/reactwg/react-native-new-architecture/blob/main/docs/cxx-custom-types.md) patterns so the module stays maintainable and type-safe.

---

## 4. Android: build, ABIs, and GPU backends

### 4.1 Build pipeline (order matters)

1. **Submodule and versions**: Ensure `cpp/llama.cpp` is present and at the commit in `scripts/used_version.sh`. CI runs `scripts/setupLlamaCpp.sh init` when needed.
2. **GPU backend headers** (no system loaders shipped): `scripts/build_android_gpu_backend.sh` prepares OpenCL/Vulkan headers and environment (e.g. `.vulkan_env`); it does **not** build system loaders. Used in CI with `--abi=all --clean` and env vars like `GGML_VK_DISABLE_COOPMAT=1`.
3. **GGML GPU backend libs**: `scripts/build_android_ggml_gpu_backends.sh` builds `libggml-opencl.so` and `libggml-vulkan.so` per ABI; output goes to `prebuilt/gpu/<abi>/`.
4. **Main native libs**: `scripts/build_android_external.sh` builds the core llama/ggml libraries per ABI (and optionally Hexagon for arm64-v8a). It **copies** GPU backends from `prebuilt/gpu/<abi>/` into `android/src/main/jniLibs/<abi>/`. Do not assume GPU libs are built inside the same script as the main libs.
5. **Gradle/CMake**: The Android app builds the **Turbo Module** (`RNLlamaCpp`) via CMake; `android/CMakeLists.txt` expects prebuilt `.so` files in `android/src/main/jniLibs/<abi>/` and links to them (IMPORTED targets). It does **not** build llama.cpp from source; it builds only our C++ bridge and the “common” static lib, and links to the prebuilts.

When changing build steps or adding ABIs/backends, keep this order and the contract (jniLibs populated before the app build). CI in `.github/workflows/ci.yml` encodes this (e.g. “Prepare GPU Backend Headers” → “Build GGML GPU Backend Libraries” → “Build Final Android Libraries” and copy into jniLibs).

### 4.2 ABIs and compatibility

- **Supported ABIs**: arm64-v8a, x86_64, armeabi-v7a, x86. Default for the library is 64-bit (arm64-v8a, x86_64); 32-bit get CPU-only builds. `reactNativeArchitectures` in `android/build.gradle` can override.
- **CPU backends**: On Android we use **dynamic backend loading** (`GGML_BACKEND_DL=1`). For arm64-v8a, multiple CPU variants (e.g. `libggml-cpu-android_armv8*.so`) are built and the runtime selects the best; for other ABIs a single `libggml-cpu.so` is used. Our C++ code in `PureCppImpl.cpp` implements `load_android_cpu_backends()` / `load_android_backends()` that `dlopen` these libs and register backends so they work from inside the APK (no filesystem iteration).
- **GPU backends**: OpenCL and Vulkan are built as separate `.so` files and loaded at runtime. We **do not ship** system loaders (`libOpenCL.so`, `libvulkan.so`); we ship only our backend libs (e.g. `libggml-opencl.so`, `libggml-vulkan.so`). Vulkan is currently **disabled** on Android in our CMake/copy steps to avoid emulator crashes; OpenCL is enabled when the lib is present. Hexagon (Snapdragon) is optional and built only when Hexagon SDK is available (CI may skip it).
- **Manifest**: `AndroidManifest` declares optional native libraries so PackageManager can load them when present; all GPU/system libs are `required="false"` for maximum install compatibility.

When adding or changing ABIs or backends, update the CI verification steps that check `jniLibs/<abi>/` for required and optional libs, and ensure we never ship system loader libs.

### 4.3 CMake and jniLibs

- **Prebuilts**: All llama/ggml (and optional GPU/Hexagon) libs are **IMPORTED** in `android/CMakeLists.txt` from `android/src/main/jniLibs/${ANDROID_ABI}/`. Our RNLlamaCpp target links to them and to `react_codegen_RNLlamaCppSpec`, and has POST_BUILD steps to copy those prebuilts (and CPU variant / GPU / Hexagon libs) into the build output so they are packaged in the APK.
- **Defines**: `GGML_BACKEND_DL=1`, `GGML_CPU=1`; do not set `GGML_USE_CPU=1` when using dynamic CPU backends. OpenCL/Vulkan are enabled only if the corresponding `.so` exists in jniLibs for that ABI (see the `EXISTS` checks in CMakeLists.txt).
- **Include paths**: Our C++ code and the “common” lib get include dirs for `cpp/`, `cpp/llama.cpp` (include, common, ggml/include, vendor), and the codegen dirs (`android/generated/jni`, etc.).

Do not change the CMake logic to build llama.cpp from source for the Android app; the design is “prebuild via scripts, then link and copy in CMake.”

---

## 5. iOS: Metal and prebuilt framework

- **Backend**: Metal only; unified across Apple devices. The podspec sets `-DLLAMA_METAL` and links Metal/MetalKit/Accelerate.
- **Prebuilt**: The app does not build llama.cpp from source. We use a prebuilt **llama.xcframework** (see `ios/libs/llama.xcframework`). Setup is done by `scripts/build_apple_external.sh init`, which downloads and unpacks the framework; CI caches it.
- **Podspec**: `RNLlamaCpp.podspec` lists our source files (iOS and C++ bridge under `cpp/`), `vendored_frameworks = "ios/libs/llama.xcframework"`, and `preserve_paths` for headers and cpp. **HEADER_SEARCH_PATHS** include `ios/include`, `cpp`, `ios/generated`, and paths under `cpp/llama.cpp` (include, common, vendor). When adding new C++ or Objective-C files, add them to `source_files` and ensure include paths stay consistent.
- **Codegen**: iOS codegen output lives under `ios/generated/`; the podspec and Xcode use it for the Turbo Module spec. Do not edit generated files by hand.

Skills or docs that mention “Metal” or “Apple GPU” should state that our iOS implementation is Metal-only and that we rely on the prebuilt xcframework produced by the upstream build (or our script), not on building llama.cpp inside the pod.

---

## 6. Scripts and CI

- **Version and env**: `scripts/used_version.sh` is the single source for llama.cpp commit/tag, NDK version, Android API levels, and optional Vulkan/OpenCL tags. Other scripts source it; CI and local builds should use the same versions.
- **Scripts** (all under `scripts/`):
  - `setupLlamaCpp.sh` – submodule init/update.
  - `build_apple_external.sh` – iOS: download/setup xcframework, copy headers to `ios/include`.
  - `build_android_gpu_backend.sh` – Android: GPU backend **headers** and env (OpenCL/Vulkan).
  - `build_android_ggml_gpu_backends.sh` – Android: build GGML OpenCL/Vulkan backend `.so` files.
  - `build_android_external.sh` – Android: build main llama/ggml libs per ABI, optional Hexagon, and copy GPU/CPU libs into `android/src/main/jniLibs/`.
- **CI**: `.github/workflows/ci.yml` runs setup, then Android build (with GPU header prep, GGML GPU backends, then final libs and jniLibs copy), then iOS build (framework setup), then tests and optional publish. Artifacts (e.g. jniLibs, xcframework) are passed between jobs. When changing scripts or build order, keep CI in sync and ensure the “Verify” steps still match the actual layout (e.g. required libs per ABI, no system libs in the package).

---

## 7. Skills and external best practices

- **Skills** that refer to the **third-party C library** (llama.cpp) or to “C++ best practices” in the abstract should be applied in a way that stays **compatible with this repo’s implementation**: same namespaces, same JSI/spec contract, same build and runtime layout (jniLibs, xcframework, codegen).
- **New skills** that add rules for “turbo modules” or “C++ in React Native” should explicitly reference this file and the fact that we use a **prebuilt** native stack on both platforms and a **single C++ bridge** (PureCppImpl, LlamaCppModel, rn-llama, rn-utils, rn-completion) as the only place that talks to llama.cpp.
- When in doubt, prefer **minimum change**: keep the current ABI set, backend set (OpenCL/Vulkan/Hexagon/CPU), and build sequence; document new options (e.g. new flags or backends) in this file or in `GPU.md` and update CI if they affect artifacts.

---

## 8. Summary for the LLM

- **Turbo Module**: Spec in `src/NativeRNLlamaCpp.ts` (name `NativeRNLlamaCpp`, interface `Spec`); C++ impl in `cpp/PureCppImpl.*` and `cpp/LlamaCppModel.*`; bridge to llama.cpp in `cpp/rn-llama.h`, `cpp/rn-utils.h`, `cpp/rn-completion.cpp`. Keep the spec and C++ API in sync; do not edit codegen output. Follow the [New Architecture Turbo Modules](https://github.com/reactwg/react-native-new-architecture/blob/main/docs/turbo-modules.md) and [custom C++ types](https://github.com/reactwg/react-native-new-architecture/blob/main/docs/cxx-custom-types.md) guides for new methods and types (bridging, event emitters).
- **Third-party**: llama.cpp lives in `cpp/llama.cpp/`; we wrap it, we don’t change it for feature work; version is in `scripts/used_version.sh`.
- **Hard rule**: Treat `cpp/llama.cpp/` as read-only upstream code. Implement changes in our bridge (`cpp/`, `ios/`, `android/`, scripts, podspec), not inside upstream sources.
- **Android**: Prebuild via scripts (GPU headers → GGML GPU libs → main libs); jniLibs populated for each ABI; CMake only links and copies. Multiple ABIs and GPU backends; dynamic loading; do not ship system loaders. CI encodes the full sequence.
- **iOS**: Metal only; prebuilt xcframework; podspec and scripts as above.
- **C++**: C++20, `facebook::react`, JSI types, smart pointers, mutex for shared state, CallInvoker for async. For new spec-level types use React’s `Bridging<T>` where appropriate. Our rules override generic C++ rules when they conflict with this layout.
- **Skills**: Any “best practice” or skill for the C library or for Turbo Modules must remain compatible with the Android spec, iOS spec, and this C++ bridge; this file is the source of truth for how the module is built and maintained.

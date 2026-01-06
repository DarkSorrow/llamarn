## Android GPU Overview

We now ship the exact GPU backends that `ggml-org/llama.cpp` expects, and they live side‑by‑side with the CPU libraries inside `android/src/main/jniLibs`:

- `libggml-opencl.so` (≈2.3 MB) – GGML OpenCL backend.
- `libggml-vulkan.so` (≈36 MB) – GGML Vulkan backend with every shader baked in.
- `libggml-cpu.so`, `libggml-base.so`, `libggml.so`, `libllama.so` – CPU + core runtime.

We build those once per ABI (`arm64-v8a`, `x86_64`). 32‑bit ABIs (`armeabi-v7a`, `x86`) remain CPU-only, so they only get the CPU `.so` files. The CI scripts copy the finished binaries directly into `android/src/main/jniLibs/<abi>/`, so consumers install the package and immediately have GPU backends available without any post-step.

### Build Pipeline (high level)

1. `scripts/build_android_gpu_backend.sh`
   - Clones the pinned Khronos headers.
   - Builds the OpenCL ICD loader only for linking and stages it under `prebuilt/gpu/<abi>/`.
   - Emits `.vulkan_env` files describing the NDK loader + include path.
2. `scripts/build_android_ggml_gpu_backends.sh`
   - Invokes the upstream ggml CMake targets (`ggml-opencl`, `ggml-vulkan`) for `arm64-v8a` and `x86_64`.
   - Drops the resulting `.so` files in `prebuilt/gpu/<abi>/`.
3. `scripts/build_android_external.sh`
   - Builds the CPU/llama libraries for every ABI.
   - Copies GPU backends from `prebuilt/gpu/<abi>/` into `android/src/main/jniLibs/<abi>/`.
   - Marks `.vulkan_enabled` / `.opencl_enabled` only when the ABI actually ships that backend.

This mirrors the approach in [`llama.rn`](https://github.com/mybigday/llama.rn) but keeps the artifacts in-repo so npm consumers don’t need to rebuild.

## Runtime Behaviour

We rely on the same “state of the art” detection path that the upstream Android sample and `llama.rn` use:

1. **Dynamic loading (`GGML_BACKEND_DL=1`)** – `libllama.so` loads whatever GPU backends exist in `jniLibs` at runtime.
2. **Backend priority** – we try OpenCL first, then Vulkan, then CPU. GGML handles that ordering internally by probing each backend and watching for successful device registration.
3. **Capability checks** – the React Native TurboModule (`cpp/PureCppImpl.cpp`) still calls `llama_supports_gpu_offload()` and uses `SystemUtils::getOptimalGpuLayers()` so we only offload layers when the device advertises enough VRAM.
4. **Manifest hints** – `AndroidManifest.xml` declares:
   ```xml
   <uses-native-library android:name="libggml-opencl.so" android:required="false"/>
   <uses-native-library android:name="libggml-vulkan.so" android:required="false"/>
   <uses-native-library android:name="libOpenCL.so" android:required="false"/>
   <uses-native-library android:name="libvulkan.so" android:required="false"/>
   ```
   Setting `required="false"` keeps installs working on CPU-only devices but nudges PackageManager to load the libraries whenever they exist.

## Device Classes

| Device type            | Behaviour                                                                                      |
|------------------------|-------------------------------------------------------------------------------------------------|
| `arm64-v8a` phone w/ GPU | Both backends shipped; GGML loads OpenCL first, falls back to Vulkan, then CPU as needed.      |
| `arm64-v8a` phone (CPU-only) | Manifest still allows install; `llama_supports_gpu_offload()` returns false, so we stay on CPU. |
| `x86_64` emulator / desktop | Same as arm64, useful for QA.                                                               |
| `armeabi-v7a` / `x86`  | CPU libraries only. No GPU backends are copied, and flag files are removed during the build.    |

## Observability & QA Notes

- The CI workflow verifies that no `libOpenCL.so` or `libvulkan.so` sneak into the package (system loaders only).
- `prebuilt/gpu/<abi>` contains a `.gpu_skipped` sentinel for 32‑bit ABIs so you can tell at a glance why a backend is missing.
- You can inspect `android/src/main/jniLibs/<abi>/` in the published tarball to confirm the right binaries were bundled (see sizes above).

## How To Use In-App

Nothing special is required in app code: loading `@novastera-oss/llamarn` gives you the prebuilt `jniLibs`. As long as you call the TurboModule normally, `llama.cpp` will handle:

1. Loading CPU + GPU `.so`s from `jniLibs`.
2. Detecting the best available backend at runtime.
3. Offloading the requested number of layers based on `SystemUtils::getOptimalGpuLayers` or user-provided overrides.

If you need to force CPU-only operation (e.g., on low-end devices), set `n_gpu_layers=0` when calling the module—`llama.cpp` will keep everything on the CPU even though the GPU binaries exist.
# GPU Backends: OpenCL and Vulkan (Android)

This document explains the requirements, architecture, and subtleties of GPU backend support for **Android** in the llamarn library. On Android, only two GPU backends are available: **OpenCL** and **Vulkan**, with **CPU** as the always-available fallback.

## Overview

The library supports GPU acceleration through two backends with automatic fallback:
1. **OpenCL** - Typically works best on Qualcomm Adreno GPUs (Snapdragon devices)
2. **Vulkan** - Broader device support but currently disabled by default due to emulator stability issues
3. **CPU** (always available) - Used for hybrid mode and final fallback

Backend selection is automatic: GGML registers all available backends, probes for compatible devices, and uses whichever backend successfully initializes. On Android, OpenCL is typically preferred for Qualcomm devices, while Vulkan offers broader compatibility but requires testing on real hardware.

## Architecture

### What We Build and Ship

We bundle the GGML GPU backends inside the APK:

- **`libggml-opencl.so`** – GGML OpenCL backend implementation (built only for 64-bit ABIs: `arm64-v8a`, `x86_64`).
- **`libggml-vulkan.so`** – GGML Vulkan backend implementation (built only for 64-bit ABIs; currently disabled by default on Android builds but shipped so QA teams can sideload it).

We compile the Khronos **OpenCL ICD loader** (`libOpenCL.so`) and stash it under `prebuilt/gpu/<abi>/` for **build-time linking only**. We do **not** ship it in the APK - the system will provide `libOpenCL.so` at runtime if the device supports OpenCL. This ensures we don't ship unnecessary libraries and lets the system handle GPU drivers.

> **Note on 32-bit ABIs:** Android 32-bit ABIs (`armeabi-v7a`, `x86`) rarely expose functional OpenCL/Vulkan stacks on modern devices. To keep CI deterministic and avoid shipping unused artifacts, GPU backends are not built for these ABIs. The scripts create placeholder markers (`.gpu_skipped`) so downstream packaging knows GPU acceleration is intentionally unavailable on 32-bit builds.

### What the System Provides

The platform is ultimately responsible for the GPU drivers:

- **`libOpenCL.so`** – Installable Client Driver loader. Preferred source is the device system image.
- **`libvulkan.so`** – Vulkan loader (standard with Android 7.0+).

### Why We Prefer System Libraries

1. **OpenCL ICD Loader (`libOpenCL.so`)**:
   - Built and staged under `prebuilt/gpu/<abi>` for **build-time linking only** (to link `libggml-opencl.so`).
   - **NOT shipped in APK** - the system will provide `libOpenCL.so` at runtime if the device supports OpenCL.
   - This ensures the loader stays in sync with OEM GPU drivers and we don't ship unnecessary libraries.

2. **Vulkan Loader (`libvulkan.so`)**:
   - Vulkan loaders are part of Android NDK since API 24 and should be supplied by the OS.
   - We do **not** ship our own Vulkan loader; the manifest entry simply requests the system one and allows install to continue when it is absent.

## Backend Detection and Loading

### Build-Time Configuration (Android)

On Android, only three backends are relevant: **OpenCL**, **Vulkan**, and **CPU**.

- `-DGGML_CPU=1` and `-DGGML_BACKEND_DL=1` are always on so Android CPU execution is guaranteed and GPU backends (OpenCL/Vulkan) stay dynamically loaded.
- `-DGGML_OPENCL=1` is set only when the corresponding `.so` files exist for the target ABI (arm64-v8a, armeabi-v7a, etc.), keeping unused symbols out of smaller builds.
- `-DGGML_VULKAN=1` is currently **disabled by default** in `android/CMakeLists.txt` to avoid known emulator crashes. Flip the commented block on when testing real Vulkan-capable Android hardware.

**Note**: Other backends (CUDA, Metal, HIP, etc.) are not applicable to Android and are not built or registered.

This setup mirrors upstream `ggml-org/llama.cpp`, but adds Android-specific gating so we never link against a backend that we can't package safely for Android devices.

### Runtime Detection (Android)

On Android, the backend loading system has multiple layers of safety checks for OpenCL and Vulkan:

1. **Library Loading** (`dlopen()`):
   - Attempts to load `libggml-opencl.so` or `libggml-vulkan.so` from the APK
   - If library cannot be loaded, that backend is skipped
   - Falls back to CPU if both GPU backends fail

2. **Backend Device Probing**:
   - **Note**: OpenCL and Vulkan backends in upstream llama.cpp do **not** implement `ggml_backend_score()` 
   - Instead, device selection happens during backend registration via device probing
   - OpenCL: `ggml_opencl_probe_devices()` scans for available OpenCL platforms and devices
   - Vulkan: Similar device enumeration happens during Vulkan backend initialization
   - If no devices are found during probing, the backend is registered but provides no devices (effectively disabled)

3. **Backend Initialization**:
   - Calls `ggml_backend_init()` which can return `nullptr` on failure
   - **OpenCL**: Probes for available devices via `clGetPlatformIDs()` (works best on Qualcomm Adreno GPUs)
   - **Vulkan**: Checks for Vulkan 1.2+ support and available devices (broader Android support)
   - If initialization fails, backend is skipped gracefully and next backend is tried

4. **Device Probing (Android)**:
   - **OpenCL**: Scans for available OpenCL platforms and devices (typically Qualcomm Snapdragon)
   - **Vulkan**: Enumerates Vulkan devices (most modern Android devices)
   - Unsupported devices are filtered out
   - If no GPU devices found, CPU backend is used automatically

5. **API Version Check**:
   - Verifies backend API version matches `GGML_BACKEND_API_VERSION`
   - Incompatible backends are rejected

### Backend Priority Order (Android-Specific)

On Android, only two GPU backends are available: **OpenCL** and **Vulkan**. GGML registers both during initialization, then selects GPU devices based on backend scoring and availability:

1. **GPU Device Selection** (llama.cpp lines 200-252):
   - Collects GPU devices from registered backends (OpenCL and/or Vulkan on Android)
   - **Note**: The OpenCL and Vulkan backends in upstream llama.cpp do **not** implement `ggml_backend_score()` - device selection is based on successful device probing during backend initialization
   - During `ggml_backend_opencl_reg()` / `ggml_backend_vulkan_reg()`, the backend probes for available devices
   - If devices are found, the backend is registered and devices are added to the model's device list
   - If no GPU devices found, falls back to CPU

2. **Backend Registration** (ggml-backend-reg.cpp):
   - Both OpenCL and Vulkan backends are registered if their `.so` files are present and can be loaded via `dlopen()`
   - The backend registration functions (`ggml_backend_opencl_reg()`, `ggml_backend_vulkan_reg()`) probe for devices during initialization
   - If device probing fails (no compatible devices found), the backend registration may return fewer or no devices, but the backend itself is still registered
   - Selection is based on which backend successfully finds and initializes compatible devices on the Android hardware

3. **OpenCL vs Vulkan on Android**:
   - **OpenCL**: Typically preferred for Qualcomm Adreno GPUs (Snapdragon devices) due to better driver support
   - **Vulkan**: Broader device support but currently disabled by default in our builds due to emulator stability issues
   - The actual backend used depends on which one successfully initializes and finds compatible devices on the specific Android hardware

4. **CPU** (always available on Android):
   - Always present as final fallback
   - Used for hybrid mode: some layers on GPU (OpenCL or Vulkan), rest on CPU
   - CPU backend is Android-specific and optimized for ARM architectures (arm64-v8a, armeabi-v7a)

### Hybrid Mode

When `n_gpu_layers > 0`:
- That many layers run on GPU (OpenCL or Vulkan)
- Remaining layers run on CPU
- CPU backend is always available for this hybrid operation

## Android Implementation Plan

### Build & Packaging Guarantees

1. **Pinned toolchain prep**: `scripts/build_android_gpu_backend.sh` fetches the exact Khronos OpenCL headers/ICD loader and Vulkan-Headers tag that CI uses, keeps them under `prebuilt/third_party` / `prebuilt/gpu`, and emits `.opencl_enabled` / `.vulkan_env` markers per ABI. The downstream builds read those files to find both the staged ICD loader and the `vulkan/vulkan.hpp` tree without ever mutating the system SDK, so local builds match CI bit-for-bit.
2. **CMake knobs**: Android builds always set `GGML_BACKEND_DL=1` and `GGML_CPU=1` so CPU code is statically available while GPU backends stay opt-in and dynamically loaded. OpenCL is compiled in only when `libggml-opencl.so` plus `libOpenCL.so` exist for the ABI, and Vulkan remains behind a toggle until emulator stability improves.

```115:146:android/CMakeLists.txt
target_compile_definitions(common PRIVATE 
    -DGGML_BACKEND_DL=1
    -DGGML_CPU=1
)
if(OPENCL_BACKEND_AVAILABLE)
    target_compile_definitions(RNLlamaCpp PRIVATE -DGGML_OPENCL=1)
endif()
```

3. **Android GPU backends**: The upstream `ggml/src/CMakeLists.txt` registers multiple backends, but on Android only OpenCL and Vulkan are relevant. Both are registered if their shared objects are present, which keeps us aligned with the `ggml-org/llama.cpp` packaging story.

```420:424:cpp/llama.cpp/ggml/src/CMakeLists.txt
ggml_add_backend(Vulkan)  # Android: Available
ggml_add_backend(OpenCL)  # Android: Available
# Other backends (CUDA, Metal, etc.) are not applicable to Android
```

### Runtime Capability Detection Flow

1. **Model scan (JSI `getModelInfo`)**: Before initialization we already load the GGUF with `n_gpu_layers = 0`. During this lightweight scan we call `llama_supports_gpu_offload()` and, if true, estimate `optimalGpuLayers` via `SystemUtils::getOptimalGpuLayers(model)` so the JS layer knows the safe upper bound.
   - **Performance note**: `llama_supports_gpu_offload()` only checks already-loaded backends. If `ggml_backend_load_all()` hasn't been called yet, GPU backends won't be detected. Backend loading (including device probing) happens lazily when first accessed or can be triggered explicitly.

2. **Launch-time decision (`initLlama`)**: The Turbo C++ module only forwards `n_gpu_layers` from JS when `gpuSupported` is still true at runtime (`llama_supports_gpu_offload()` is re-queried). Otherwise we silently keep `n_gpu_layers = 0`, guaranteeing CPU fallback even if JS forgot to guard.
   - **Performance note**: Device probing (OpenCL `clGetPlatformIDs()`, Vulkan device enumeration) happens during backend registration when `libggml-opencl.so` / `libggml-vulkan.so` are first loaded via `dlopen()`. This is typically fast (<100ms) but happens synchronously during model loading if backends aren't pre-loaded.

```102:138:cpp/PureCppImpl.cpp
bool gpuSupported = llama_supports_gpu_offload();
int optimalGpuLayers = gpuSupported ? SystemUtils::getOptimalGpuLayers(model) : 0;
...
if (options.hasProperty(runtime, "n_gpu_layers") && gpuSupported) {
    n_gpu_layers = options.getProperty(runtime, "n_gpu_layers").asNumber();
}
```

3. **GGML backend device selection (Android)**: During model loading, llama.cpp collects available GPU devices from registered backends. On Android, only OpenCL and Vulkan backends are considered. **Note**: These backends don't use `ggml_backend_score()` - instead, device availability is determined during backend registration when `ggml_backend_opencl_reg()` and `ggml_backend_vulkan_reg()` probe for devices. Devices are added to the model's device list if they successfully initialize. If no GPU devices are found (or both OpenCL and Vulkan fail to find devices), the system automatically falls back to the Android CPU backend—no React Native code needs to know which backend (if any) succeeded.
   
   **Performance impact**: Device probing happens once when backends are first loaded (via `dlopen()` of `libggml-opencl.so` / `libggml-vulkan.so`). This typically takes <100ms per backend and happens synchronously during model loading. The probing is fast because it only queries device availability (OpenCL `clGetPlatformIDs()`, Vulkan device enumeration) without initializing full GPU contexts. If you want to avoid this delay during model loading, you can pre-load backends by calling `ggml_backend_load_all()` early (e.g., during app startup), but this is optional - the current lazy loading approach is fast enough for most use cases.

### React Native Turbo Module Contract

| Stage | Native behavior | JS responsibility |
| --- | --- | --- |
| Model discovery (`getModelInfo`) | Returns `{ gpuSupported, optimalGpuLayers }` plus CPU-only stats. | Cache these flags per model/device and surface them in UI settings. |
| Session creation (`initLlama`) | Applies requested `n_gpu_layers` *only* when the native side confirms GPU availability. | Pass `Math.min(userRequestedLayers, optimalGpuLayers)` and default to `0` when the device is not GPU ready. |
| Inference loop | GGML dynamically loads OpenCL/Vulkan/CPU and reports errors; failures trigger CPU retry. | On GPU init error, prompt users that the run will continue on CPU and optionally persist a “GPU disabled” flag for that session. |

This mirrors the gating strategy described in [`mybigday/llama.rn` PR #210](https://github.com/mybigday/llama.rn/pull/210), ensuring every GPU path is opt-in and reversible without crashing non-GPU devices.

### Backend Priority & Safeguards

- **Library presence**: On Android we package `libggml-opencl.so`, the ICD loader, and (optionally) `libggml-vulkan.so`. If any of these files are missing for the running ABI, GGML simply skips that backend.
- **Driver probing (Android)**: `llama_supports_gpu_offload()` internally checks `ggml_backend_dev_by_type(GPU)` for OpenCL and Vulkan devices on Android. Failures such as broken emulator Vulkan stacks are filtered out before reaching JS. If no GPU devices are found, it returns false and CPU is used.
- **Device collection (Android)**: llama.cpp collects GPU devices from registered backends during model initialization (see `llama.cpp` lines 200-252). On Android, only OpenCL and Vulkan backends are available. The backend that successfully probes and scores devices gets used. OpenCL typically works better for Qualcomm Adreno GPUs (Snapdragon devices), while Vulkan has broader device support but is currently disabled by default due to emulator crashes. If neither GPU backend works, CPU is used automatically.
- **Resource budget**: `SystemUtils::getOptimalGpuLayers()` limits VRAM usage to ~80% of a conservative RAM slice (20% on Android). JS should never request more layers than this number unless the user explicitly overrides it.

#### Manifest vs. Runtime Checks

`ggml_backend_dev_by_type()` (often surfaced as `backend_ggml` checks in upstream discussions) only runs *after* the process starts. Android’s PackageManager, however, can strip optional native libs during install if the manifest never mentions them. Declaring `uses-native-library` with `android:required="false"` tells PackageManager to keep those shared objects around when available, while the runtime checks still decide whether we actually offload or fall back to CPU. We need both layers for reliable behavior across OEM builds.

### Device Classes & UX

1. **GPU-optimal devices** (Snapdragon 8 Gen 3 / X Elite): expose UI sliders up to `optimalGpuLayers`, default to that value, and surface an info banner confirming OpenCL usage.
2. **GPU-possible but unstable devices** (emulators, Mali GPUs without OpenCL): default to CPU, show “experimental” toggles, and log backend failures for telemetry.
3. **CPU-only devices**: hide GPU toggles entirely and keep packaging lightweight—no runtime penalty because dynamic loading will never find the optional libs.

Following the staged rollout from [`mybigday/llama.rn`](https://github.com/mybigday/llama.rn) keeps parity with devices where GPU already works today while preventing regressions elsewhere.

### Observability & Testing

- **Logging**: Add a verbose log when backend selection finishes (OpenCL/Vulkan/CPU) and another when we fall back due to driver errors. This mirrors how upstream llama.cpp traces backend picks.
- **Device matrix**: Run the OpenCL detection test on real Snapdragon hardware, Vulkan on Pixel 8/9 (API 34+), and CPU-only on emulators to ensure the fallback path never crashes.
- **Feature flagging**: Guard Vulkan behind a JS-accessible remote flag so we can enable it per device class without rebuilding, similar to how [`margelo/react-native-filament`](https://github.com/margelo/react-native-filament) gates advanced rendering features.

## Android Manifest Configuration

The `AndroidManifest.xml` declares the libraries we ship (not system libraries):

```xml
<!-- GGML OpenCL backend - built by us, shipped in APK -->
<uses-native-library 
    android:name="libggml-opencl.so" 
    android:required="false" />

<!-- GGML Vulkan backend - built by us, shipped in APK -->
<uses-native-library 
    android:name="libggml-vulkan.so" 
    android:required="false" />

<!-- System OpenCL loader - only mapped if present on the device -->
<uses-native-library 
    android:name="libOpenCL.so" 
    android:required="false" />

<!-- System Vulkan loader - only mapped if present on the device -->
<uses-native-library 
    android:name="libvulkan.so" 
    android:required="false" />
```

- `android:required="false"` keeps installs working on CPU-only devices but still nudges Android to link the loader when it exists.
- Declaring the system loaders explicitly prevents modern Android builds from optimizing them away even though the app mainly uses dynamic loading.

## Build Requirements

### OpenCL

1. **Headers**: OpenCL headers must be available during build (for compilation)
2. **ICD Loader for Build**: `libOpenCL.so` is built and staged under `prebuilt/gpu/<abi>` for build-time linking (to link `libggml-opencl.so`)
3. **Runtime**: System `libOpenCL.so` must be available on device if OpenCL is to work (we do NOT ship it)

### Vulkan

1. **Headers**: We install (clone) the pinned Khronos `Vulkan-Headers` tag (`$VULKAN_HEADERS_TAG`) under `prebuilt/third_party/Vulkan-Headers` so `vulkan/vulkan.hpp` is always available to ggml-vulkan regardless of which SDK the host machine had cached previously.
2. **Runtime**: System `libvulkan.so` is provided by Android on devices with Vulkan support (we only link against the loader that ships with the NDK/API level we pin in `used_version.sh`).

## Device Requirements

### OpenCL Support

- **Not standard**: OpenCL is NOT standard with Android NDK
- **Device-specific**: Only available on devices with OpenCL support (e.g., Qualcomm Snapdragon with Adreno GPU)
- **Verified devices** (from llama.cpp docs):
  - Snapdragon 8 Gen 3 (Adreno 750)
  - Snapdragon 8 Elite (Adreno 830)
  - Snapdragon X Elite (Adreno X85)

### Vulkan Support

- **Standard**: Vulkan is standard with Android NDK (since Android 7.0 / API 24)
- **Widely available**: Most modern Android devices support Vulkan
- **Emulator warning**: Vulkan may crash on Android emulators - runtime detection handles this

## Error Handling

The system handles failures gracefully:

1. **Missing System Libraries**: If `libOpenCL.so` or `libvulkan.so` are not available, the respective backend simply won't work (no crash)

2. **Backend Initialization Failure**: If backend initialization fails, it returns `nullptr` and is skipped

3. **Device Unavailable**: If no suitable GPU devices are found, backend is skipped

4. **Runtime Fallback**: If GPU initialization fails during model loading, the system automatically retries with CPU-only mode

## Testing Requirements

### Required Tests

1. **OpenCL Detection Test**:
   - Verify OpenCL backend is detected on devices with OpenCL support
   - Verify graceful failure on devices without OpenCL support
   - Test on real devices (Snapdragon devices)

2. **Vulkan Detection Test**:
   - Verify Vulkan backend is detected on devices with Vulkan support
   - Verify graceful failure on devices without Vulkan support
   - Test on real devices (avoid emulators for Vulkan)

3. **Backend Priority Test**:
   - Verify OpenCL is tried before Vulkan when both are available
   - Verify Vulkan is used when OpenCL is unavailable
   - Verify CPU fallback when both GPU backends fail

4. **Hybrid Mode Test**:
   - Verify hybrid mode works (some layers on GPU, rest on CPU)
   - Test with various `n_gpu_layers` values

5. **Error Recovery Test**:
   - Verify graceful fallback to CPU when GPU initialization fails
   - Verify no crashes when system libraries are missing

### Test Devices

- **OpenCL**: Test on Qualcomm Snapdragon devices (Snapdragon 8 Gen 3, 8 Elite, X Elite)
- **Vulkan**: Test on modern Android devices (Android 7.0+)
- **CPU Fallback**: Test on devices without GPU support

### Emulator Considerations

- **Vulkan**: May crash on Android emulators - runtime detection should skip it
- **OpenCL**: May not be available on emulators
- **Recommendation**: Always test on real devices for GPU backends

## Troubleshooting

### OpenCL Not Working

1. **Check device support**: Verify device has OpenCL support (Snapdragon with Adreno GPU)
2. **Check system library**: Verify `libOpenCL.so` exists on device (system library, not ours)
3. **Check logs**: Look for OpenCL initialization errors in logs
4. **Verify backend loaded**: Check if `libggml-opencl.so` is loaded successfully

### Vulkan Not Working

1. **Check Android version**: Requires Android 7.0+ (API 24+)
2. **Check device support**: Verify device has Vulkan support
3. **Emulator issues**: Vulkan may not work on emulators - test on real device
4. **Check logs**: Look for Vulkan initialization errors in logs

### Both Backends Failing

1. **CPU fallback**: System should automatically fall back to CPU
2. **Check logs**: Review initialization logs for specific error messages
3. **Verify libraries**: Ensure `libggml-opencl.so` and `libggml-vulkan.so` are in APK

## References

- [llama.cpp OpenCL Documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENCL.md)
- [llama.cpp Android Documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md)
- [Android Vulkan Documentation](https://developer.android.com/ndk/guides/graphics)

## Summary

- **We build and ship**: `libggml-opencl.so`, `libggml-vulkan.so`
- **We build but DON'T ship**: `libOpenCL.so` (built for build-time linking only, staged under `prebuilt/gpu/<abi>`)
- **System provides at runtime**: `libOpenCL.so` and `libvulkan.so` (if device supports them)
- **Fallback policy**: Prefer system loaders; treat bundled OpenCL loader as optional safety net
- **Detection**: Automatic with graceful fallback - GGML registers all backends and selects devices based on successful initialization
- **Backend selection**: Device-based (not explicit priority) - whichever backend successfully probes and scores devices gets used
- **Android preference**: OpenCL typically preferred for Qualcomm Adreno GPUs; Vulkan disabled by default due to emulator issues
- **Testing**: Required on real devices with GPU support (especially Snapdragon for OpenCL)




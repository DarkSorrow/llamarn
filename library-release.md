# Library Release — How Native Binaries Flow Into Target Apps

This document explains exactly what happens at build time when an app installs
`@novastera-oss/llamarn` from npm. Nothing in this library requires target apps
to run our build scripts or compile llama.cpp themselves.

---

## What is in the npm package

The `files` field in `package.json` publishes these directories verbatim:

```
src/          TypeScript source + generated JS bundle (lib/)
lib/          Compiled JS + type declarations
android/      Android build files + jniLibs/ (prebuilt .so files)
ios/          Podspec assets + libs/llama.xcframework (prebuilt)
cpp/          C++ bridge source + llama.cpp common/ source
              (llama.cpp examples, tests, docs are excluded)
RNLlamaCpp.podspec
react-native.config.js
```

The prebuilt native binaries ship **inside** the package:
- **Android:** `android/src/main/jniLibs/<abi>/*.so` — the llama/ggml shared libraries, included in the tarball at publish time.
- **iOS:** `ios/libs/llama.xcframework` — the prebuilt Apple framework, included in the tarball at publish time.

A target app never needs an internet connection to llama.cpp's GitHub at build time, and never compiles llama.cpp from source.

---

## Android — full build flow

### 1. npm install

The package lands at `node_modules/@novastera-oss/llamarn/`. The prebuilt `.so` files are already present at:

```
node_modules/@novastera-oss/llamarn/android/src/main/jniLibs/
  arm64-v8a/
    libllama.so
    libggml.so
    libggml-base.so
    libggml-cpu-android_armv8.0_1.so   ← baseline (emulator + old devices)
    libggml-cpu-android_armv8.2_1.so   ← DOTPROD
    libggml-cpu-android_armv8.2_2.so   ← DOTPROD + FP16
    libggml-cpu-android_armv8.6_1.so   ← DOTPROD + FP16 + MATMUL_INT8 (KleidiAI)
    libggml-opencl.so                  ← Adreno GPU (optional)
    libggml-hexagon.so                 ← Snapdragon NPU (optional)
    libggml-htp-v73.so … v81.so        ← HTP libraries for Hexagon
  x86_64/
    libllama.so  libggml.so  libggml-base.so  libggml-cpu.so
  armeabi-v7a/
    libllama.so  libggml.so  libggml-base.so  libggml-cpu.so
  x86/
    libllama.so  libggml.so  libggml-base.so  libggml-cpu.so
```

### 2. Gradle sync

`react-native.config.js` tells the React Native CLI that this is a C++ Turbo Module:
```js
cxxModuleCMakeListsPath: 'CMakeLists.txt'
cxxModuleHeaderName:     'PureCppImpl'
```

The CLI auto-links the library: it adds `android/CMakeLists.txt` from the package to the app's Gradle build.

`android/build.gradle` declares the jniLibs source directory:
```groovy
jniLibs.srcDirs = ['src/main/jniLibs']
```

This tells Gradle to package every `.so` in `jniLibs/` into the APK's `lib/<abi>/` folder.

### 3. CMake build (compile-only step)

CMake builds **only our bridge code** from source — specifically these C++ files:

| Compiled from source | Role |
|---------------------|------|
| `cpp/PureCppImpl.cpp` | Turbo Module entry point, `initLlama`, `loadLlamaModelInfo` |
| `cpp/LlamaCppModel.cpp` | JSI HostObject — `completion`, `tokenize`, `embedding`, etc. |
| `cpp/rn-completion.cpp` | Inference loop, context shift, sampler wiring |
| `cpp/SystemUtils.cpp` | Device memory / GPU layer estimation |
| `cpp/rn-multimodal.cpp` | Multimodal (vision/audio) helpers |
| `cpp/build-info.cpp` | Build metadata |
| `cpp/llama.cpp/common/*.cpp` | Chat template, sampling, Jinja, grammar, reasoning budget |

The prebuilt `.so` files are declared as **IMPORTED** targets in `android/CMakeLists.txt` — CMake reads their headers and links against them, but does not recompile them:

```cmake
add_library(llama SHARED IMPORTED)
set_target_properties(llama PROPERTIES
    IMPORTED_LOCATION ${JNI_LIBS_DIR}/${ANDROID_ABI}/libllama.so)
```

This produces a single output library: **`libRNLlamaCpp.so`**.

### 4. POST_BUILD copy

After `libRNLlamaCpp.so` is built, CMake copies the prebuilt `.so` files from `jniLibs/` into the same build output directory:

```cmake
add_custom_command(TARGET RNLlamaCpp POST_BUILD
    COMMAND cmake -E copy_if_different
        ${JNI_LIBS_DIR}/${ANDROID_ABI}/libllama.so
        $<TARGET_FILE_DIR:RNLlamaCpp>/libllama.so
    ...
)
```

This ensures the APK packager finds all dependent `.so` files in one place and includes them in `lib/<abi>/`.

### 5. APK packaging

Gradle packages into the APK:
```
lib/arm64-v8a/
  libRNLlamaCpp.so          ← compiled from source (bridge)
  libllama.so               ← prebuilt (copied from jniLibs)
  libggml.so                ← prebuilt
  libggml-base.so           ← prebuilt
  libggml-cpu-*.so          ← prebuilt CPU variants (arm64 only)
  libggml-opencl.so         ← prebuilt (if device-appropriate)
  libggml-hexagon.so        ← prebuilt (Snapdragon, if present)
  libggml-htp-*.so          ← prebuilt (Snapdragon, if present)
```

### 6. Runtime dynamic loading

At app startup, `PureCppImpl.cpp::load_android_backends()` is called. It uses `dlopen()` with just the library filename (no path) — Android's linker resolves them from the APK's extracted native lib directory:

```cpp
void* opencl_handle = dlopen("libggml-opencl.so", RTLD_LAZY | RTLD_LOCAL);
```

For CPU variants, `load_android_cpu_backends()` scores each variant and registers only the best one compatible with the device CPU:

```cpp
static constexpr std::array<const char*, 4> cpu_variants = {{
    "libggml-cpu-android_armv8.6_1.so",  // best — DOTPROD + FP16 + KleidiAI MATMUL
    "libggml-cpu-android_armv8.2_2.so",
    "libggml-cpu-android_armv8.2_1.so",
    "libggml-cpu-android_armv8.0_1.so",  // baseline fallback
}};
```

**No filesystem iteration.** `dlopen()` by filename is how APK-packaged libraries must be loaded — `ggml_backend_load_best()` uses filesystem iteration which doesn't work from inside an APK.

### 7. AndroidManifest.xml

The library's `AndroidManifest.xml` declares all GPU and Hexagon libraries as `required="false"` so the APK installs on any device regardless of which libraries are present:

```xml
<uses-native-library android:name="libggml-opencl.so"  android:required="false" />
<uses-native-library android:name="libggml-hexagon.so" android:required="false" />
```

System loaders (`libOpenCL.so`, `libvulkan.so`) are declared but **never shipped** — they are provided by the device OS.

---

## iOS — full build flow

### 1. npm install

Same package, same `node_modules/` path. The xcframework is at:
```
node_modules/@novastera-oss/llamarn/ios/libs/llama.xcframework/
  ios-arm64/llama.framework/          ← real device slice
  ios-arm64_x86_64-simulator/         ← simulator slice (CPU only)
```

The simulator slice has no GPU (Metal) kernel support — the xcframework was built without Metal compute shaders for the simulator slice, which is why `getOptimalGpuLayers` returns 0 for `TARGET_OS_SIMULATOR`.

### 2. pod install

CocoaPods reads `RNLlamaCpp.podspec`:

```ruby
s.vendored_frameworks = "ios/libs/llama.xcframework"
s.source_files = "ios/**/*.{h,m,mm}",
                 "cpp/*.{h,cpp}",
                 "cpp/llama.cpp/common/*.{h,cpp}",
                 "cpp/llama.cpp/common/jinja/*.cpp",
                 ...
```

**Compiled from source** (same C++ files as Android, same bridge code):
- All files in `cpp/` listed in `source_files`
- All `cpp/llama.cpp/common/*.cpp` files (chat, sampling, Jinja, grammar, reasoning budget)
- MTMD multimodal sources under `cpp/llama.cpp/tools/mtmd/`

**Linked as prebuilt framework** (not recompiled):
- `ios/libs/llama.xcframework` — contains `libllama` + `libggml` + `libggml-base` + `libggml-metal` etc.

The podspec disables header maps (`USE_HEADERMAP = NO`) and implicit module maps (`-fno-implicit-module-maps`) to prevent ggml.h from being resolved through two different physical paths (xcframework Headers vs source tree), which would cause `#pragma once` to fail and trigger `redefinition of ggml_status` errors.

### 3. Xcode build

Xcode compiles the bridge source files → links against the xcframework → produces the app binary. The xcframework is embedded in `YourApp.app/Frameworks/llama.framework`.

At runtime, `dlopen` is not used on iOS — the framework is loaded by dyld at app launch.

---

## What target apps must NOT do

| Don't | Reason |
|-------|--------|
| Run `scripts/build_android_external.sh` | Those are publisher scripts — they build the `.so` files that ship inside the package. Target apps should never run them. |
| Run `scripts/build_apple_external.sh` | Same — the xcframework is already in the package. |
| Try to build llama.cpp from source | The CMake build in the package does NOT build llama.cpp from source. It compiles only the bridge and links to prebuilts. |
| Add `cpp/llama.cpp` to their own CMakeLists | It is not a standalone CMake project to be consumed by the app. |

---

## What target apps must do

**Android** — nothing beyond `npm install`. Auto-link handles CMake, Gradle handles packaging.

Optional: restrict ABIs in `gradle.properties` to reduce APK size:
```properties
reactNativeArchitectures=arm64-v8a,x86_64
```

**iOS** — run `pod install` in the app's `ios/` directory. The podspec handles everything.

If updating from a previous version: run `pod install --repo-update` to ensure the podspec changes take effect.

---

## Release checklist (for library maintainers)

Before publishing a new version, CI must have:

1. **Built and committed Android `.so` files** to `android/src/main/jniLibs/` for all ABIs:
   - `arm64-v8a`: `libllama.so`, `libggml.so`, `libggml-base.so`, CPU variant libs, optional GPU/Hexagon libs
   - `x86_64`, `armeabi-v7a`, `x86`: `libllama.so`, `libggml.so`, `libggml-base.so`, `libggml-cpu.so`

2. **Downloaded the iOS xcframework** to `ios/libs/llama.xcframework/` matching `LLAMA_CPP_TAG` in `scripts/used_version.sh`.

3. **Ran `scripts/verify-pack-contents.sh`** — this dry-packs the tarball and verifies that `android/CMakeLists.txt` and all MTMD source files referenced in it are present in the package.

4. **Verified the JS bundle** by running `npm run prepare` (uses `react-native-builder-bob`), confirming `lib/` is populated.

5. **Published** with `npm publish --provenance --access public`.

If any of steps 1–2 are skipped, target apps will get a package with missing `.so` files or an empty xcframework, and the build will fail with "library not found" or similar.

---

## Verifying a published package

After publishing, download the tarball and inspect:

```bash
npm pack --dry-run    # list files that would be included
```

Or after publication:

```bash
npx npm-package-contents @novastera-oss/llamarn@<version>
```

Key things to verify:
- `android/src/main/jniLibs/arm64-v8a/libllama.so` — present and > 0 bytes
- `android/src/main/jniLibs/arm64-v8a/libggml-cpu-android_armv8.0_1.so` — baseline CPU variant present
- `ios/libs/llama.xcframework/ios-arm64/llama.framework/llama` — real device binary present
- `cpp/llama.cpp/common/chat.cpp` — common source present (compiled by both platforms)
- `cpp/llama.cpp/tools/mtmd/mtmd.cpp` — MTMD source present (required for multimodal)

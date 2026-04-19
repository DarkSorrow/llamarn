# Vision Phase — llamarn Multimodal Implementation

## Status: In Progress

## Phase 1 — Foundation (Build systems + capability system)
- [ ] vision-phase.md created
- [ ] android/CMakeLists.txt — add mtmd STATIC library
- [ ] RNLlamaCpp.podspec — add mtmd source files
- [ ] cpp/rn-llama.h — add mtmd_context + capabilities
- [ ] cpp/rn-utils.h — add mtmd_encoded_n_past + MediaItem helpers
- [ ] cpp/rn-multimodal.h/cpp — bitmap loading + capability types
- [ ] cpp/PureCppImpl.cpp — parse mmproj + capabilities, init mtmd
- [ ] src/NativeRNLlamaCpp.ts — capability types + new method signatures

## Phase 2 — Vision Chat (image in messages)
- [ ] cpp/rn-completion.cpp — multimodal encode path (mtmd_tokenize + eval_chunks)
- [ ] cpp/LlamaCppModel.cpp — isMultimodalEnabled + getSupportedModalities
- [ ] Example: VisionChatScreen

## Phase 3 — Encoder Functions
- [ ] embedImage() — CLIP-style image embeddings
- [ ] transcribeAudio() — Whisper-style audio transcription
- [ ] visionReasoning() — open-ended image analysis (returns text)
- [ ] Example: ImageEmbedScreen, AudioTranscribeScreen

## Phase 4 — Camera Pipeline
- [ ] runOnFrame() — NativeBuffer (HardwareBuffer/CVPixelBuffer) → C++ with downsampling
- [ ] Android: HardwareBuffer → stride-aware RGB → mtmd_bitmap
- [ ] iOS: CVPixelBuffer → stride-aware RGB → mtmd_bitmap

## Phase 5 — Multimodal Responses
- [ ] Parse model output for embedded image/audio data
- [ ] Route image outputs to file or return base64

## Phase 6 — Example + Docs
- [ ] All example screens
- [ ] README multimodal section

## Supported Model Families (reference)
| Model | Capability | Notes |
|-------|-----------|-------|
| LLaVA 1.5 / 1.6 | vision-chat | Classic multimodal chat |
| Qwen2-VL / Qwen3-VL | vision-chat | Dynamic resolution |
| LLaMA 4 Scout | vision-chat | Requires mmproj |
| MiniCPM-V | vision-chat | Mobile-optimised |
| Whisper | audio-transcribe | As standalone or mmproj sidecar |
| CLIP | image-encode | Embeddings only |

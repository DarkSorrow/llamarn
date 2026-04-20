# Vision Phase — llamarn Multimodal Implementation

## Status: Complete (Phases 1–4 + Docs)

## Phase 1 — Foundation (Build systems + capability system)
- [x] vision-phase.md created
- [x] android/CMakeLists.txt — add mtmd STATIC library
- [x] RNLlamaCpp.podspec — add mtmd source files
- [x] cpp/rn-llama.h — add mtmd_context + capabilities
- [x] cpp/rn-utils.h — add mtmd_encoded_n_past + MediaItem helpers
- [x] cpp/rn-multimodal.h/cpp — bitmap loading + capability types
- [x] cpp/PureCppImpl.cpp — parse mmproj + capabilities, init mtmd
- [x] src/NativeRNLlamaCpp.ts — capability types + new method signatures

## Phase 2 — Vision Chat (image in messages)
- [x] cpp/rn-completion.cpp — multimodal encode path (mtmd_tokenize + eval_chunks)
- [x] cpp/LlamaCppModel.cpp — isMultimodalEnabled + getSupportedModalities
- [x] Example: VisionChatScreen

## Phase 3 — Encoder Functions
- [x] embedImage() — CLIP-style image embeddings
- [x] transcribeAudio() — Whisper-style audio transcription
- [x] visionReasoning() — open-ended image analysis (returns text)
- [ ] Example: ImageEmbedScreen, AudioTranscribeScreen (VisionChatScreen covers embedImage; separate screens deferred)

## Phase 4 — Camera Pipeline
- [x] runOnFrame() — NativeBuffer (HardwareBuffer/CVPixelBuffer) → C++ with downsampling
- [x] Android: HardwareBuffer → stride-aware RGB → mtmd_bitmap (pixelStride==4 guard)
- [x] iOS: CVPixelBuffer → stride-aware RGB → mtmd_bitmap
- [x] FrameGuard RAII — unconditionally resets is_processing_frame_ on thread exit
- [x] Capability validation before spawning thread

## Phase 5 — Multimodal Responses
- [ ] Parse model output for embedded image/audio data (deferred — text output sufficient for current model families)
- [ ] Route image outputs to file or return base64

## Phase 6 — Example + Docs
- [x] VisionChatScreen (Describe + Embed, lazy model load, test image, iOS + Android)
- [x] README multimodal section
- [ ] AudioTranscribeScreen / ImageEmbedScreen (deferred)

## Critical Fixes Applied
- [x] F1a — is_released_ guard in every invokeAsync lambda
- [x] F2  — inference_mutex_ in transcribeAudio background thread
- [x] F3  — FrameGuard RAII struct in runOnFrame thread lambda
- [x] F4  — bm2 wraps raw_bm as first statement in runOnFrame thread
- [x] F5  — Android pixelStride != 4 guard
- [x] F6  — Capability validation in runOnFrameJsi before spawning thread

## PureCppImpl Refactor (Prerequisite)
- [x] R1 — safe_invoke helper
- [x] R2 — try_init_with_gpu_fallback static function
- [x] R3 — init_chat_templates_safe static function
- [x] R4 — ModelInitResult struct + do_init_llama (bug 1: kwargs order fix)
- [x] R5 — 3-phase initLlama thread lambda (mutex scope fix, exception flow fix)
- [x] R6 — loadLlamaModelInfo: safe_invoke + reject-not-resolve fix

## Cleanup Tasks Applied
- [x] C1 — multiply test stub (already removed)
- [x] C2 — LlamaMessage.content allows null + content-part arrays
- [x] C3 — finish_reason returns 'length' when context full or budget exhausted
- [x] C4 — getAvailableMemoryBytes() + availableMemoryMB + estimatedVramMB in loadLlamaModelInfo

## Supported Model Families (reference)
| Model | Capability | Notes |
|-------|-----------|-------|
| LLaVA 1.5 / 1.6 | vision-chat | Classic multimodal chat |
| Qwen2-VL / Qwen3-VL | vision-chat | Dynamic resolution |
| LLaMA 4 Scout | vision-chat | Requires mmproj |
| MiniCPM-V | vision-chat | Mobile-optimised |
| Whisper | audio-transcribe | As standalone or mmproj sidecar |
| CLIP | image-encode | Embeddings only |

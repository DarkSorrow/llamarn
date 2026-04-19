#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#include "mtmd.h"
#include "mtmd-helper.h"
#pragma GCC diagnostic pop

#include "llama.h"

#define JSON_ASSERT GGML_ASSERT
#include "nlohmann/json.hpp"
using json = nlohmann::ordered_json;

#include <string>
#include <vector>

namespace facebook::react {

// ---- Declared model capabilities -----------------------------------------
// The calling app declares which capabilities are active at initLlama time.
// This lets us allocate exactly what is needed and gate JSI methods cleanly.

enum class ModelCapability : uint32_t {
    VisionChat       = 1 << 0,  // multimodal text generation (LLaVA, Qwen-VL)
    ImageEncode      = 1 << 1,  // CLIP-style image embeddings
    AudioTranscribe  = 1 << 2,  // Whisper-style audio → text
    VisionReasoning  = 1 << 3,  // open-ended image analysis / captioning (returns text, not bboxes)
};

inline uint32_t capabilities_from_strings(const std::vector<std::string>& names) {
    uint32_t flags = 0;
    for (const auto& n : names) {
        if (n == "vision-chat")           flags |= static_cast<uint32_t>(ModelCapability::VisionChat);
        else if (n == "image-encode")     flags |= static_cast<uint32_t>(ModelCapability::ImageEncode);
        else if (n == "audio-transcribe") flags |= static_cast<uint32_t>(ModelCapability::AudioTranscribe);
        else if (n == "vision-reasoning") flags |= static_cast<uint32_t>(ModelCapability::VisionReasoning);
    }
    return flags;
}

inline bool has_capability(uint32_t flags, ModelCapability cap) {
    return (flags & static_cast<uint32_t>(cap)) != 0;
}

// ---- Media item (image/audio URL for multimodal chat) --------------------
enum class MediaType { Image, Audio };

struct MediaItem {
    std::string url;  // file:///... or data:mime;base64,...
    MediaType   type = MediaType::Image;
};

// ---- Base64 decode -------------------------------------------------------
std::vector<unsigned char> base64_decode_bytes(const std::string& encoded);

// ---- Media extraction from OpenAI-format message array -------------------
// Replaces image_url/audio_url content parts with <__media__> markers in place.
// Returns the list of media items in order of appearance.
std::vector<MediaItem> extract_media_from_messages(
    json& messages_json,
    const std::string& marker = "<__media__>");

// ---- Bitmap loaders ------------------------------------------------------
// Load a bitmap from file:// URI, data: URI, or plain file path.
// Returns nullptr on failure. Caller must call mtmd_bitmap_free().
mtmd_bitmap* load_bitmap_from_uri(mtmd_context* ctx, const std::string& url);

// Load a bitmap directly from raw RGB bytes (nx * ny * 3).
// Returns nullptr on failure. Caller must call mtmd_bitmap_free().
mtmd_bitmap* load_bitmap_from_rgb(uint32_t nx, uint32_t ny,
                                   const unsigned char* rgb_data);

// ---- Embedding result ----------------------------------------------------
struct EmbedResult {
    std::vector<float> embedding;  // flat array: n_tokens * n_embd floats
    bool success = false;
    std::string error_msg;
};

// Run the vision encoder on a single image bitmap.
// Returns raw embeddings (n_tokens * n_embd floats).
EmbedResult encode_image_to_embeddings(
    mtmd_context* mtmd_ctx,
    llama_context* ctx,
    mtmd_bitmap* bitmap,
    int n_batch);

// ---- Transcript result ---------------------------------------------------
struct TranscriptSegment {
    float start_s;
    float end_s;
    std::string text;
};

struct TranscriptResult {
    std::string text;
    std::vector<TranscriptSegment> segments;
    bool success = false;
    std::string error_msg;
};

// ---- Detection result ----------------------------------------------------
struct DetectedObject {
    std::string label;
    float confidence;
    struct { float x, y, w, h; } bbox; // normalised [0,1]
};

struct DetectionResult {
    std::vector<DetectedObject> objects;
    bool success = false;
    std::string error_msg;
};

// ---- Platform camera frame conversion ------------------------------------
// Performs a synchronous pixel copy (RGBA/BGRA → RGB) with optional nearest-
// neighbor downsampling during the copy. Hardware buffers include alignment
// padding — always use stride values, not width*bpp, to avoid diagonal skewing.
// max_dimension=0 preserves original resolution (full quality for OCR).
// max_dimension=336 shrinks ~8 MB 1080p → ~330 KB (LLaVA/CLIP recommended).
// Returns nullptr on failure. Caller must call mtmd_bitmap_free().
mtmd_bitmap* bitmap_from_native_frame(
    void* nativeHandle,
    uint32_t width,
    uint32_t height,
    bool isAndroid,
    float max_dimension = 0.0f);

} // namespace facebook::react

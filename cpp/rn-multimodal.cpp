#include "rn-multimodal.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__ANDROID__)
#include <android/hardware_buffer.h>
#endif

#if defined(__APPLE__)
#include <CoreVideo/CoreVideo.h>
#endif

namespace facebook::react {

std::vector<unsigned char> base64_decode_bytes(const std::string& encoded) {
    static const std::string chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<unsigned char> out;
    int val = 0, valb = -8;
    for (unsigned char c : encoded) {
        if (c == '=') break;
        auto pos = chars.find(static_cast<char>(c));
        if (pos == std::string::npos) continue;
        val = (val << 6) + static_cast<int>(pos);
        valb += 6;
        if (valb >= 0) {
            out.push_back(static_cast<unsigned char>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

std::vector<MediaItem> extract_media_from_messages(
    json& messages_json, const std::string& marker)
{
    std::vector<MediaItem> items;
    if (!messages_json.is_array()) return items;
    for (auto& msg : messages_json) {
        if (!msg.is_object() || !msg.contains("content")) continue;
        auto& content = msg["content"];
        if (!content.is_array()) continue;
        std::string text_content;
        for (const auto& part : content) {
            if (!part.is_object()) continue;
            std::string type = part.value("type", "");
            if (type == "text") {
                text_content += part.value("text", "");
            } else if (type == "image_url") {
                std::string url;
                if (part.contains("image_url") && part["image_url"].is_object()) {
                    url = part["image_url"].value("url", "");
                }
                text_content += marker;
                items.push_back({url, MediaType::Image});
            } else if (type == "audio_url") {
                // Explicit audio routing — avoids fragile image_url tricks
                std::string url;
                if (part.contains("audio_url") && part["audio_url"].is_object()) {
                    url = part["audio_url"].value("url", "");
                }
                text_content += marker;
                items.push_back({url, MediaType::Audio});
            }
        }
        content = text_content;
    }
    return items;
}

mtmd_bitmap* load_bitmap_from_uri(mtmd_context* ctx, const std::string& url) {
    if (url.size() >= 7 && url.substr(0, 7) == "file://") {
        return mtmd_helper_bitmap_init_from_file(ctx, url.substr(7).c_str());
    }
    if (url.size() >= 5 && url.substr(0, 5) == "data:") {
        auto comma = url.find(',');
        if (comma == std::string::npos) return nullptr;
        auto bytes = base64_decode_bytes(url.substr(comma + 1));
        if (bytes.empty()) return nullptr;
        return mtmd_helper_bitmap_init_from_buf(ctx, bytes.data(), bytes.size());
    }
    return mtmd_helper_bitmap_init_from_file(ctx, url.c_str());
}

mtmd_bitmap* load_bitmap_from_rgb(uint32_t nx, uint32_t ny,
                                   const unsigned char* rgb_data) {
    return mtmd_bitmap_init(nx, ny, rgb_data);
}

EmbedResult encode_image_to_embeddings(
    mtmd_context* mtmd_ctx,
    llama_context* llama_ctx,
    mtmd_bitmap* bitmap,
    int /*n_batch*/)
{
    EmbedResult result;
    if (!mtmd_ctx || !llama_ctx || !bitmap) {
        result.error_msg = "null context or bitmap";
        return result;
    }

    // Build a single-image input with a minimal text placeholder
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    mtmd_input_text input_text;
    input_text.text          = "<__media__>";
    input_text.add_special   = false;
    input_text.parse_special = true;
    const mtmd_bitmap* bm_arr[] = { bitmap };
    if (mtmd_tokenize(mtmd_ctx, chunks, &input_text, bm_arr, 1) != 0) {
        mtmd_input_chunks_free(chunks);
        result.error_msg = "mtmd_tokenize failed";
        return result;
    }

    // Find the image/audio chunk and encode it
    size_t n_chunks = mtmd_input_chunks_size(chunks);
    for (size_t i = 0; i < n_chunks; ++i) {
        const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks, i);
        enum mtmd_input_chunk_type ctype = mtmd_input_chunk_get_type(chunk);
        if (ctype != MTMD_INPUT_CHUNK_TYPE_IMAGE && ctype != MTMD_INPUT_CHUNK_TYPE_AUDIO) continue;
        if (mtmd_encode_chunk(mtmd_ctx, chunk) != 0) {
            mtmd_input_chunks_free(chunks);
            result.error_msg = "mtmd_encode_chunk failed";
            return result;
        }
        float* embd_ptr = mtmd_get_output_embd(mtmd_ctx);
        size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
        size_t n_embd   = static_cast<size_t>(llama_model_n_embd(llama_get_model(llama_ctx)));
        result.embedding.assign(embd_ptr, embd_ptr + n_tokens * n_embd);
        result.success = true;
        break;
    }
    mtmd_input_chunks_free(chunks);
    if (!result.success) result.error_msg = "no image/audio chunk found";
    return result;
}

// Performs a synchronous pixel copy (RGBA/BGRA → RGB) with optional nearest-
// neighbor downsampling during the copy. Hardware buffers include alignment
// padding — always use stride values, not width*bpp, to avoid diagonal skewing.
mtmd_bitmap* bitmap_from_native_frame(void* nativeHandle, uint32_t width,
                                       uint32_t height, bool /*isAndroid*/,
                                       float max_dimension) {
    float scale = 1.0f;
    if (max_dimension > 0.0f &&
        (width  > static_cast<uint32_t>(max_dimension) ||
         height > static_cast<uint32_t>(max_dimension))) {
        scale = max_dimension / static_cast<float>(std::max(width, height));
    }
    const uint32_t target_w = static_cast<uint32_t>(static_cast<float>(width)  * scale);
    const uint32_t target_h = static_cast<uint32_t>(static_cast<float>(height) * scale);
    std::vector<uint8_t> rgb(target_w * target_h * 3);

#if defined(__ANDROID__)
    AHardwareBuffer* hbuf = static_cast<AHardwareBuffer*>(nativeHandle);
    AHardwareBuffer_Planes planes{};
    if (AHardwareBuffer_lockPlanes(hbuf, AHARDWAREBUFFER_USAGE_CPU_READ_RARELY,
                                   -1, nullptr, &planes) != 0) return nullptr;
    if (planes.planes[0].pixelStride != 4) {
        AHardwareBuffer_unlock(hbuf, nullptr);
        return nullptr; // unsupported pixel format — only RGBA8888 (stride=4) supported
    }
    const uint8_t* src        = static_cast<const uint8_t*>(planes.planes[0].data);
    const uint32_t row_stride = static_cast<uint32_t>(planes.planes[0].rowStride);
    const uint32_t px_stride  = static_cast<uint32_t>(planes.planes[0].pixelStride); // 4 for RGBA8888
    for (uint32_t row = 0; row < target_h; ++row) {
        const uint32_t src_row = static_cast<uint32_t>(static_cast<float>(row) / scale);
        const uint8_t* rowp   = src + src_row * row_stride;
        for (uint32_t col = 0; col < target_w; ++col) {
            const uint32_t src_col = static_cast<uint32_t>(static_cast<float>(col) / scale);
            const uint8_t* px      = rowp + src_col * px_stride;
            const uint32_t out_idx = (row * target_w + col) * 3;
            rgb[out_idx + 0] = px[0]; // R
            rgb[out_idx + 1] = px[1]; // G
            rgb[out_idx + 2] = px[2]; // B
        }
    }
    AHardwareBuffer_unlock(hbuf, nullptr);

#elif defined(__APPLE__)
    CVPixelBufferRef pbuf = static_cast<CVPixelBufferRef>(nativeHandle);
    CVPixelBufferLockBaseAddress(pbuf, kCVPixelBufferLock_ReadOnly);
    const uint8_t* src = static_cast<const uint8_t*>(CVPixelBufferGetBaseAddress(pbuf));
    const size_t bpr   = CVPixelBufferGetBytesPerRow(pbuf); // includes hardware padding
    for (uint32_t row = 0; row < target_h; ++row) {
        const uint32_t src_row = static_cast<uint32_t>(static_cast<float>(row) / scale);
        const uint8_t* rowp   = src + static_cast<size_t>(src_row) * bpr;
        for (uint32_t col = 0; col < target_w; ++col) {
            const uint32_t src_col = static_cast<uint32_t>(static_cast<float>(col) / scale);
            const uint8_t* px      = rowp + static_cast<size_t>(src_col) * 4; // kCVPixelFormatType_32BGRA
            const uint32_t out_idx = (row * target_w + col) * 3;
            rgb[out_idx + 0] = px[2]; // R from BGRA
            rgb[out_idx + 1] = px[1]; // G
            rgb[out_idx + 2] = px[0]; // B
        }
    }
    CVPixelBufferUnlockBaseAddress(pbuf, kCVPixelBufferLock_ReadOnly);

#else
    (void)nativeHandle;
    return nullptr;
#endif

    return mtmd_bitmap_init(target_w, target_h, rgb.data());
}

} // namespace facebook::react

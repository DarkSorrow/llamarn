#include "fixtures.h"
#include "rn-multimodal.h"
#include "rn-test.h"

#include <string>
#include <vector>

using namespace facebook::react;

static int count_image_chunks(mtmd_context * ctx, const std::string & path, int copies) {
    const char * marker = mtmd_get_marker(ctx);
    std::string prompt;
    std::vector<mtmd_bitmap *> owned;
    for (int i = 0; i < copies; i++) {
        prompt += marker;
        mtmd_bitmap * bitmap = load_bitmap_from_uri(ctx, path);
        if (!bitmap) {
            for (mtmd_bitmap * b : owned) {
                mtmd_bitmap_free(b);
            }
            return -1;
        }
        owned.push_back(bitmap);
    }

    std::vector<const mtmd_bitmap *> ptrs(owned.begin(), owned.end());
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    mtmd_input_text text;
    text.text          = prompt.c_str();
    text.text_len      = prompt.size();
    text.add_special   = false;
    text.parse_special = false;
    const int32_t tok = mtmd_tokenize(ctx, chunks, &text, ptrs.data(), ptrs.size());

    int images = 0;
    if (tok == 0) {
        const size_t n = mtmd_input_chunks_size(chunks);
        for (size_t i = 0; i < n; i++) {
            if (mtmd_input_chunk_get_type(mtmd_input_chunks_get(chunks, i)) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                images++;
            }
        }
    }
    mtmd_input_chunks_free(chunks);
    for (mtmd_bitmap * b : owned) {
        mtmd_bitmap_free(b);
    }
    return tok == 0 ? images : -1;
}

RN_TEST(image_number_png_tokenizes) {
    auto m = rn_test_load_vision();
    if (!m) {
        RN_SKIP("RN_TEST_MODEL and RN_TEST_MMPROJ unset");
    }
    const std::string path = std::string(RN_TEST_ROOT) + "/../../example/assets/number-7.png";
    RN_CHECK_EQ(count_image_chunks(m->rn->mtmd_ctx, path, 1), 1);
    RN_CHECK_EQ(count_image_chunks(m->rn->mtmd_ctx, path, 2), 2);
}

#include "fixtures.h"
#include "rn-test.h"

RN_TEST(vocab_loads) {
    RN_CHECK(rn_test_vocab("qwen35") != nullptr);
}

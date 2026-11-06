#include "rn-test.h"

#include "llama.h"

#include <cstdio>
#include <cstring>
#include <exception>

int main(int argc, char ** argv) {
    const char * filter = argc > 1 ? argv[1] : nullptr;

    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fputs(text, stderr);
        }
    }, nullptr);
    llama_backend_init();

    int n_pass = 0, n_fail = 0, n_skip = 0;
    for (const auto & tc : rn_test_registry()) {
        if (filter && !strstr(tc.name, filter)) {
            continue;
        }
        try {
            tc.fn();
            printf("PASS %s\n", tc.name);
            n_pass++;
        } catch (const rn_test_skip & e) {
            printf("SKIP %s (%s)\n", tc.name, e.what());
            n_skip++;
        } catch (const std::exception & e) {
            printf("FAIL %s\n     %s\n", tc.name, e.what());
            n_fail++;
        }
        fflush(stdout);
    }
    printf("\n%d passed, %d failed, %d skipped\n", n_pass, n_fail, n_skip);

    llama_backend_free();
    return n_fail == 0 ? 0 : 1;
}

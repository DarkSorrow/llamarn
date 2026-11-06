#pragma once

#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct rn_test_failure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct rn_test_skip : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct rn_test_case {
    const char *          name;
    std::function<void()> fn;
};

inline std::vector<rn_test_case> & rn_test_registry() {
    static std::vector<rn_test_case> tests;
    return tests;
}

struct rn_test_register {
    rn_test_register(const char * name, std::function<void()> fn) {
        rn_test_registry().push_back({ name, std::move(fn) });
    }
};

#define RN_TEST(name)                                                  \
    static void rn_test_##name();                                      \
    static rn_test_register rn_test_reg_##name(#name, rn_test_##name); \
    static void rn_test_##name()

#define RN_CHECK(cond)                                                         \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::ostringstream rn_os_;                                         \
            rn_os_ << __FILE__ << ":" << __LINE__ << ": check failed: " #cond; \
            throw rn_test_failure(rn_os_.str());                               \
        }                                                                      \
    } while (0)

#define RN_CHECK_EQ(a, b)                                                         \
    do {                                                                          \
        const auto & rn_a_ = (a);                                                 \
        const auto & rn_b_ = (b);                                                 \
        if (!(rn_a_ == rn_b_)) {                                                  \
            std::ostringstream rn_os_;                                            \
            rn_os_ << __FILE__ << ":" << __LINE__ << ": " #a " == " #b " failed"; \
            throw rn_test_failure(rn_os_.str());                                  \
        }                                                                         \
    } while (0)

#define RN_SKIP(reason) throw rn_test_skip(reason)

cmake_minimum_required(VERSION 3.20)
project(rn_bridge_tests CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(RN_CPP_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(RN_LLAMA_DIR "${RN_CPP_DIR}/llama.cpp")

set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD     ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP      OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL        OFF CACHE BOOL "" FORCE)
set(GGML_METAL           OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE          ON  CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)

add_subdirectory(${RN_LLAMA_DIR} llama.cpp-build EXCLUDE_FROM_ALL)

option(RN_TESTS_WITH_BRIDGE "compile rn-completion.cpp into the tests" ON)

set(RN_TEST_SOURCES
    main.cpp
    fixtures.cpp
    test-smoke.cpp
    test-tokenize.cpp
)

if (RN_TESTS_WITH_BRIDGE)
    list(APPEND RN_TEST_SOURCES
        test-chat-session.cpp
        test-completion.cpp
        ${RN_CPP_DIR}/rn-completion.cpp
        ${RN_CPP_DIR}/rn-multimodal.cpp
    )
endif()

add_executable(rn_bridge_tests ${RN_TEST_SOURCES})
target_include_directories(rn_bridge_tests PRIVATE
    ${RN_CPP_DIR}
    ${RN_LLAMA_DIR}/common
    ${RN_LLAMA_DIR}/vendor
    ${RN_LLAMA_DIR}/tools/mtmd
)
target_compile_definitions(rn_bridge_tests PRIVATE
    RN_TEST_ROOT="${RN_LLAMA_DIR}"
    $<$<BOOL:${RN_TESTS_WITH_BRIDGE}>:RN_TESTS_WITH_BRIDGE=1>
)
target_link_libraries(rn_bridge_tests PRIVATE llama-common mtmd llama)
if (APPLE)
    target_link_libraries(rn_bridge_tests PRIVATE "-framework CoreVideo" "-framework CoreFoundation")
endif()

enable_testing()
add_test(NAME rn_bridge_tests COMMAND rn_bridge_tests)

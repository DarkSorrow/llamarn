// Wrapper to prevent Xcode's -DDEBUG=1 macro from breaking mtmd-audio.cpp's
// `constexpr bool DEBUG = false;` declaration (DEBUG substituted → syntax error).
#pragma push_macro("DEBUG")
#undef DEBUG
#include "llama.cpp/tools/mtmd/mtmd-audio.cpp"
#pragma pop_macro("DEBUG")

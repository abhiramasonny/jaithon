// Same program as str_search.jai. See tests/bench/README.md for why the C++ and
// Java rows are here and what they are not. The search is written out by hand
// on purpose: `find` would measure the standard library rather than the
// per-character indexing this benchmark is about.
#include <cstdint>
#include <cstdio>
#include <string>

#include <cstdlib>
#include <cstring>

// BENCH_LEVEL picks how much work this benchmark does: easy a sixteenth of it,
// medium a quarter, hard (the default, and anything unrecognised) all of it.
static long bench_scale() {
    const char *l = std::getenv("BENCH_LEVEL");
    if (l && std::strcmp(l, "easy") == 0) return 16;
    if (l && std::strcmp(l, "medium") == 0) return 4;
    return 1;
}
static const long SCALE = bench_scale();
// A repetition count around a fixed text: never below one pass.
static const long REPS = (5 / SCALE) < 1 ? 1 : (5 / SCALE);

static int64_t count_occurrences(const std::string &text, const std::string &needle) {
    int64_t n = (int64_t)text.size();
    int64_t m = (int64_t)needle.size();
    char first = needle[0];
    int64_t hits = 0;
    int64_t i = 0;
    while (i + m <= n) {
        if (text[(size_t)i] == first) {
            int64_t k = 1;
            while (k < m && text[(size_t)(i + k)] == needle[(size_t)k]) k += 1;
            if (k == m) hits += 1;
        }
        i += 1;
    }
    return hits;
}

int main() {
    const char *chunks[8] = {"abcdbadc", "bcadcbda", "cdabacbd", "dacbdabc",
                             "abdcadbc", "bdacbcad", "cabdbdca", "dbcaabcd"};
    std::string text;
    int64_t seed = 7;
    for (int64_t i = 0; i < 250000; i++) {
        seed = (seed * 1103515245 + 12345) % 2147483648;
        text += chunks[(seed / 65536) % 8];
    }

    int64_t hits = 0;
    for (long rep = 0; rep < REPS; rep++) hits = count_occurrences(text, "abcd");
    std::printf("%lld\n", (long long)text.size());
    std::printf("%lld\n", (long long)hits);
    return 0;
}

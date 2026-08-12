// Same program as list_ops.jai. See tests/bench/README.md for why the C++ and
// Java rows are here and what they are not.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

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
static const int64_t N = 10000000 / SCALE;
static const int64_t PROBE = 999999 / SCALE;

int main() {
    std::vector<int64_t> xs;
    for (int64_t i = 0; i < N; i++) {
        xs.push_back(i * 3);
    }

    int64_t total = 0;
    for (int64_t x : xs) {
        total += x % 11;
    }

    std::vector<int64_t> doubled(xs.size());
    std::transform(xs.begin(), xs.end(), doubled.begin(), [](int64_t x) { return x * 2; });

    std::printf("%lld\n", (long long)total);
    std::printf("%lld\n", (long long)doubled[(size_t)PROBE]);
    return 0;
}

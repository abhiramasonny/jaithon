// Same program as sieve.jai. See tests/bench/README.md for why the C++ and
// Java rows are here and what they are not.
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

int main() {
    const int64_t n = 10000000 / SCALE;
    std::vector<bool> flags;
    for (int64_t i = 0; i <= n; i++) flags.push_back(true);

    int64_t count = 0;
    int64_t total = 0;
    for (int64_t p = 2; p * p <= n; p++) {
        if (flags[(size_t)p]) {
            for (int64_t q = p * p; q <= n; q += p) {
                flags[(size_t)q] = false;
            }
        }
    }

    for (int64_t i = 2; i <= n; i++) {
        if (flags[(size_t)i]) {
            count += 1;
            total = (total + i) % 1000000007;
        }
    }
    std::printf("%lld\n", (long long)count);
    std::printf("%lld\n", (long long)total);
    return 0;
}

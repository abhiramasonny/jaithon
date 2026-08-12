// Same program as closure_calls.jai. See tests/bench/README.md for why the C++
// and Java rows are here and what they are not. std::function keeps the call
// indirect, which is the point of the benchmark.
#include <cstdint>
#include <cstdio>
#include <functional>

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
static const int64_t ROUNDS = 1000 / SCALE;

static std::function<int64_t(int64_t)> adder(int64_t step) {
    return [step](int64_t x) { return x + step; };
}

static int64_t apply_n(const std::function<int64_t(int64_t)> &f, int64_t start, int64_t times) {
    int64_t acc = start;
    for (int64_t i = 0; i < times; i++) {
        acc = f(acc);
    }
    return acc;
}

int main() {
    int64_t total = 0;
    for (int64_t k = 1; k <= ROUNDS; k++) {
        auto bump = adder(k);
        total = (total + apply_n(bump, 0, 100000)) % 1000000007;
    }
    std::printf("%lld\n", (long long)total);
    return 0;
}

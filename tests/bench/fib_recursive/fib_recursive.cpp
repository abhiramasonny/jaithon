// Same program as fib_recursive.jai.
#include <cstdint>
#include <cstdio>

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
// Work is exponential in n, so shorten it by subtracting.
static const int64_t DEPTH_DROP = SCALE == 16 ? 4 : (SCALE == 4 ? 2 : 0);

static int64_t fib(int64_t n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }

int main() {
    std::printf("%lld\n", (long long)fib(38 - DEPTH_DROP));
    return 0;
}

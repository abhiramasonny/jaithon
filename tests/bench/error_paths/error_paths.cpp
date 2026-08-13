// Same program as error_paths.jai. See tests/bench/README.md for why the C++
// and Java rows are here and what they are not.
//
// C++ exceptions are table-driven: a `try` that never fires is free at run
// time, and a throw costs an unwinder walk. Read the column with that in mind —
// the two phases of this benchmark are priced very differently here than in a
// bytecode VM that pushes a handler record.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

static long bench_scale() {
    const char *l = std::getenv("BENCH_LEVEL");
    if (l && std::strcmp(l, "easy") == 0) return 16;
    if (l && std::strcmp(l, "medium") == 0) return 4;
    return 1;
}
static const long SCALE = bench_scale();

static int64_t checked(int64_t x) {
    if (x % 8 == 3) {
        throw std::invalid_argument("bad value " + std::to_string(x));
    }
    return x + 1;
}

static int64_t calm(int64_t n) {
    int64_t total = 0;
    int64_t i = 0;
    while (i < n) {
        try {
            total += i % 13;
        } catch (const std::invalid_argument &) {
            total -= 1;
        }
        i += 1;
    }
    return total;
}

static int64_t storm(int64_t n) {
    int64_t ok = 0;
    int64_t bad = 0;
    int64_t i = 0;
    while (i < n) {
        try {
            ok += checked(i);
        } catch (const std::invalid_argument &) {
            bad += 1;
        }
        i += 1;
    }
    return ok - bad;
}

int main() {
    int64_t c = 8000000 / SCALE; if (c < 1) c = 1;
    int64_t s = 160000 / SCALE;  if (s < 1) s = 1;
    std::printf("%lld\n", (long long)calm(c));
    std::printf("%lld\n", (long long)storm(s));
    return 0;
}

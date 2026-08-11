// Same program as queens.jai. See tests/bench/README.md for why the C++ and
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
// A repetition count around a fixed board: never below one round.
static const long REPS = (3 / SCALE) < 1 ? 1 : (3 / SCALE);

static bool safe(const std::vector<int64_t> &cols, int64_t row, int64_t col) {
    for (int64_t r = 0; r < row; r++) {
        int64_t c = cols[(size_t)r];
        if (c == col) return false;
        if (c - r == col - row) return false;
        if (c + r == col + row) return false;
    }
    return true;
}

static int64_t place(std::vector<int64_t> &cols, int64_t row, int64_t n) {
    if (row == n) return 1;
    int64_t found = 0;
    for (int64_t col = 0; col < n; col++) {
        if (safe(cols, row, col)) {
            cols[(size_t)row] = col;
            found += place(cols, row + 1, n);
        }
    }
    return found;
}

int main() {
    const int64_t n = 11;
    std::vector<int64_t> cols((size_t)n, 0);
    int64_t total = 0;
    for (long rep = 0; rep < REPS; rep++) total = place(cols, 0, n);
    std::printf("%lld\n", (long long)n);
    std::printf("%lld\n", (long long)total);
    return 0;
}

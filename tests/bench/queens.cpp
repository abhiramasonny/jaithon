// Same program as queens.jai. See tests/bench/README.md for why the C++ and
// Java rows are here and what they are not.
#include <cstdint>
#include <cstdio>
#include <vector>

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
    const int64_t n = 9;
    std::vector<int64_t> cols((size_t)n, 0);
    int64_t total = 0;
    for (int rep = 0; rep < 3; rep++) total = place(cols, 0, n);
    std::printf("%lld\n", (long long)n);
    std::printf("%lld\n", (long long)total);
    return 0;
}

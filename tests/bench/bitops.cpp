// Same program as bitops.jai. See tests/bench/README.md for why the C++ and
// Java rows are here and what they are not. int64_t throughout, because
// jaithon's `int` is 64-bit and a narrower type here would be a different
// program.
#include <cstdint>
#include <cstdio>

static int64_t popcount(int64_t value) {
    int64_t v = value;
    int64_t bits = 0;
    while (v != 0) {
        v = v & (v - 1);
        bits += 1;
    }
    return bits;
}

int main() {
    const int64_t mask = 4294967295;
    int64_t seed = 7;
    int64_t ones = 0;
    int64_t checksum = 0;
    for (int64_t i = 0; i < 300000; i++) {
        seed = (seed * 1103515245 + 12345) % 2147483648;
        int64_t v = seed ^ checksum;
        v = v ^ (v >> 7);
        v = (v ^ (v << 3)) & mask;
        v = v ^ (v >> 11);
        int64_t bits = popcount(v);
        ones += bits;
        checksum = (checksum + v + bits) & mask;
        checksum = ((checksum << 1) | (checksum >> 31)) & mask;
    }
    std::printf("%lld\n", (long long)ones);
    std::printf("%lld\n", (long long)checksum);
    return 0;
}

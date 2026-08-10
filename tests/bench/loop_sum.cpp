// Same program as loop_sum.jai. See tests/bench/README.md for why the C++ and
// Java rows are here and what they are not.
#include <cstdint>
#include <cstdio>

int main() {
    int64_t total = 0;
    for (int64_t i = 0; i < 50000000; i++) {
        total += i % 7;
    }
    std::printf("%lld\n", (long long)total);
    return 0;
}

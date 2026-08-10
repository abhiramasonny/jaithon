// Same program as sieve.jai. See tests/bench/README.md for why the C++ and
// Java rows are here and what they are not.
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    const int64_t n = 1000000;
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

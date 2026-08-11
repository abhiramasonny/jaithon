// Same program as list_ops.jai. See tests/bench/README.md for why the C++ and
// Java rows are here and what they are not.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    std::vector<int64_t> xs;
    for (int64_t i = 0; i < 10000000; i++) {
        xs.push_back(i * 3);
    }

    int64_t total = 0;
    for (int64_t x : xs) {
        total += x % 11;
    }

    std::vector<int64_t> doubled(xs.size());
    std::transform(xs.begin(), xs.end(), doubled.begin(), [](int64_t x) { return x * 2; });

    std::printf("%lld\n", (long long)total);
    std::printf("%lld\n", (long long)doubled[999999]);
    return 0;
}

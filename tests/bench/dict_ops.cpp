// Same program as dict_ops.jai. See tests/bench/README.md for why the C++ and
// Java rows are here and what they are not.
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>

int main() {
    std::unordered_map<std::string, int64_t> counts;
    for (int64_t i = 0; i < 500000; i++) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "k%lld", (long long)(i % 10000));
        std::string key(buf);
        auto found = counts.find(key);
        int64_t previous = found == counts.end() ? 0 : found->second;
        counts[key] = previous + 1;
    }

    int64_t total = 0;
    for (const auto &entry : counts) {
        total += entry.second;
    }
    std::printf("%lld\n", (long long)counts.size());
    std::printf("%lld\n", (long long)total);
    return 0;
}

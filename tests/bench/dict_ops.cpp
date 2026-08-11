// Same program as dict_ops.jai. See tests/bench/README.md for why the C++ and
// Java rows are here and what they are not.
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>

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
static const int64_t ITERS = 30000000 / SCALE;

int main() {
    std::unordered_map<std::string, int64_t> counts;
    for (int64_t i = 0; i < ITERS; i++) {
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

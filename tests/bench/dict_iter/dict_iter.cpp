// Peer port of dict_iter.jai. Must print byte-identical output.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>

int main() {
    const char *env = std::getenv("BENCH_LEVEL");
    std::string level = env ? env : "hard";
    int scale = level == "easy" ? 20 : level == "medium" ? 5 : 1;
    const int KEYS = 20000;
    int reps = 120 / scale;
    if (reps < 1) reps = 1;

    std::unordered_map<std::string, long long> d;
    d.reserve(KEYS * 2);
    for (int i = 0; i < KEYS; i++) d["k" + std::to_string(i)] = i;

    long long total = 0;
    long long seen = 0;
    for (int r = 0; r < reps; r++) {
        total = 0;
        for (const auto &kv : d) total += kv.second;
        seen = 0;
        for (const auto &kv : d) {
            if (kv.first.size() > 4 && kv.second % 2 == 0) seen++;
        }
    }
    std::printf("%lld\n%lld\n", total, seen);
    return 0;
}

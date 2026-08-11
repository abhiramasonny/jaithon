// Same program as word_freq.jai. See tests/bench/README.md for why the C++ and
// Java rows are here and what they are not.
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>

int main() {
    std::string text;
    int64_t seed = 7;
    for (int64_t i = 0; i < 200000; i++) {
        seed = (seed * 1103515245 + 12345) % 2147483648;
        char buf[32];
        std::snprintf(buf, sizeof buf, "w%lld ", (long long)(seed % 500));
        text += buf;
    }

    std::unordered_map<std::string, int64_t> counts;
    size_t start = 0;
    size_t at = 0;
    size_t n = text.size();
    while (at < n) {
        if (text[at] == ' ') {
            if (at > start) {
                std::string word = text.substr(start, at - start);
                auto found = counts.find(word);
                int64_t previous = found == counts.end() ? 0 : found->second;
                counts[word] = previous + 1;
            }
            start = at + 1;
        }
        at++;
    }

    int64_t total = 0;
    for (const auto &entry : counts) total += entry.second;
    std::printf("%lld\n", (long long)counts.size());
    std::printf("%lld\n", (long long)total);
    return 0;
}

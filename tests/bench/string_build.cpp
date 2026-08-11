// Same program as string_build.jai. See tests/bench/README.md for why the C++
// and Java rows are here and what they are not. The split materialises its
// pieces rather than just counting separators, which is the work being timed.
#include <cstdint>
#include <cstdio>
#include <string>
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
static const int64_t N = 2000000 / SCALE;

static std::string join(const std::string &sep, const std::vector<std::string> &parts) {
    std::string out;
    for (size_t i = 0; i < parts.size(); i++) {
        if (i > 0) out += sep;
        out += parts[i];
    }
    return out;
}

static std::vector<std::string> split(const std::string &s, const std::string &sep) {
    std::vector<std::string> out;
    size_t start = 0;
    while (true) {
        size_t at = s.find(sep, start);
        if (at == std::string::npos) {
            out.push_back(s.substr(start));
            break;
        }
        out.push_back(s.substr(start, at - start));
        start = at + sep.size();
    }
    return out;
}

int main() {
    std::vector<std::string> parts;
    for (int64_t i = 0; i < N; i++) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "item-%lld", (long long)i);
        parts.push_back(std::string(buf));
    }
    std::string joined = join(",", parts);
    std::printf("%lld\n", (long long)joined.size());
    std::printf("%lld\n", (long long)split(joined, ",").size());
    return 0;
}

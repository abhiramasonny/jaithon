// Same program as string_build.jai. See tests/bench/README.md for why the C++
// and Java rows are here and what they are not. The split materialises its
// pieces rather than just counting separators, which is the work being timed.
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

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
    for (int64_t i = 0; i < 200000; i++) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "item-%lld", (long long)i);
        parts.push_back(std::string(buf));
    }
    std::string joined = join(",", parts);
    std::printf("%lld\n", (long long)joined.size());
    std::printf("%lld\n", (long long)split(joined, ",").size());
    return 0;
}

// Same program as sort_merge.jai. See tests/bench/README.md for why the C++ and
// Java rows are here and what they are not. Hand-written merge sort, not
// std::sort, for the same reason the original avoids `list.sort()`.
#include <cstdint>
#include <cstdio>
#include <vector>

static std::vector<int64_t> merge(const std::vector<int64_t> &left,
                                  const std::vector<int64_t> &right) {
    std::vector<int64_t> out;
    size_t i = 0;
    size_t j = 0;
    while (i < left.size() && j < right.size()) {
        if (left[i] <= right[j]) {
            out.push_back(left[i]);
            i += 1;
        } else {
            out.push_back(right[j]);
            j += 1;
        }
    }
    while (i < left.size()) {
        out.push_back(left[i]);
        i += 1;
    }
    while (j < right.size()) {
        out.push_back(right[j]);
        j += 1;
    }
    return out;
}

static std::vector<int64_t> sort(const std::vector<int64_t> &values) {
    if (values.size() <= 1) return values;
    size_t mid = values.size() / 2;
    std::vector<int64_t> left(values.begin(), values.begin() + mid);
    std::vector<int64_t> right(values.begin() + mid, values.end());
    return merge(sort(left), sort(right));
}

int main() {
    std::vector<int64_t> data;
    int64_t seed = 12345;
    for (int64_t i = 0; i < 1000000; i++) {
        seed = (seed * 1103515245 + 12345) % 2147483648;
        data.push_back(seed % 100000);
    }
    std::vector<int64_t> sorted = sort(data);
    int64_t checksum = 0;
    for (size_t at = 0; at < sorted.size(); at++) {
        checksum = (checksum + sorted[at] * ((int64_t)(at % 7) + 1)) % 1000000007;
    }
    std::printf("%lld\n", (long long)sorted.size());
    std::printf("%lld\n", (long long)checksum);
    return 0;
}

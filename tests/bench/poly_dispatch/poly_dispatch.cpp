// Same program as poly_dispatch.jai. See tests/bench/README.md for why the C++
// and Java rows are here and what they are not.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

static long bench_scale() {
    const char *l = std::getenv("BENCH_LEVEL");
    if (l && std::strcmp(l, "easy") == 0) return 16;
    if (l && std::strcmp(l, "medium") == 0) return 4;
    return 1;
}
static const long SCALE = bench_scale();
static const int OPS = 512;
static const int64_t M = 1000003;

struct Op {
    int64_t k;
    explicit Op(int64_t k_) : k(k_) {}
    virtual ~Op() {}
    virtual int64_t apply(int64_t x) const = 0;
};

struct AddK : Op {
    explicit AddK(int64_t k) : Op(k) {}
    int64_t apply(int64_t x) const override { return (x + k) % M; }
};
struct MulK : Op {
    explicit MulK(int64_t k) : Op(k) {}
    int64_t apply(int64_t x) const override { return (x * k) % M; }
};
struct SquareK : Op {
    explicit SquareK(int64_t k) : Op(k) {}
    int64_t apply(int64_t x) const override { return (x * x + k) % M; }
};
struct DoubleK : Op {
    explicit DoubleK(int64_t k) : Op(k) {}
    int64_t apply(int64_t x) const override { return (x * 2 + k) % M; }
};
struct DivK : Op {
    explicit DivK(int64_t k) : Op(k) {}
    int64_t apply(int64_t x) const override { return (x + x / k) % M; }
};
struct ModK : Op {
    explicit ModK(int64_t k) : Op(k) {}
    int64_t apply(int64_t x) const override { return (x + x % k) % M; }
};
struct FlipK : Op {
    explicit FlipK(int64_t k) : Op(k) {}
    int64_t apply(int64_t x) const override { return (M - 1 - x + k) % M; }
};
struct MixK : Op {
    explicit MixK(int64_t k) : Op(k) {}
    int64_t apply(int64_t x) const override { return (x + x / 3 + k) % M; }
};

static std::vector<std::unique_ptr<Op>> build(int n) {
    std::vector<std::unique_ptr<Op>> ops;
    int64_t s = 7;
    for (int i = 0; i < n; i++) {
        s = (s * 1103515245LL + 12345LL) % 2147483648LL;
        int64_t which = (s / 65536) % 8;
        int64_t k = s % 97 + 2;
        switch ((int)which) {
            case 0: ops.push_back(std::make_unique<AddK>(k)); break;
            case 1: ops.push_back(std::make_unique<MulK>(k)); break;
            case 2: ops.push_back(std::make_unique<SquareK>(k)); break;
            case 3: ops.push_back(std::make_unique<DoubleK>(k)); break;
            case 4: ops.push_back(std::make_unique<DivK>(k)); break;
            case 5: ops.push_back(std::make_unique<ModK>(k)); break;
            case 6: ops.push_back(std::make_unique<FlipK>(k)); break;
            default: ops.push_back(std::make_unique<MixK>(k)); break;
        }
    }
    return ops;
}

int main() {
    const int64_t reps = (8000 / SCALE) < 1 ? 1 : (8000 / SCALE);
    std::vector<std::unique_ptr<Op>> ops = build(OPS);
    int64_t acc = 1;
    int64_t check = 0;
    for (int64_t r = 0; r < reps; r++) {
        acc = (acc + r) % M;
        for (const std::unique_ptr<Op> &op : ops) acc = op->apply(acc);
        check = (check + acc) % M;
    }
    std::printf("%d\n", (int)ops.size());
    std::printf("%lld\n", (long long)acc);
    std::printf("%lld\n", (long long)check);
    return 0;
}

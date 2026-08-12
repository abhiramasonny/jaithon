// Same program as alloc_churn.jai. Heap-allocated so the allocator is exercised
// rather than optimised away into registers.
#include <cstdint>
#include <cstdio>
#include <memory>

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
static const int64_t ITERS = 4000000 / SCALE;

struct Point {
    int64_t x, y;
    Point(int64_t x, int64_t y) : x(x), y(y) {}
    int64_t dot(const Point &o) const { return x * o.x + y * o.y; }
};

int main() {
    int64_t total = 0;
    for (int64_t i = 0; i < ITERS; i++) {
        auto a = std::make_unique<Point>(i % 100, i % 37);
        auto b = std::make_unique<Point>(i % 53, i % 11);
        total = (total + a->dot(*b)) % 1000000007;
    }
    std::printf("%lld\n", (long long)total);
    return 0;
}

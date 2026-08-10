// Same program as object_dispatch.jai. See tests/bench/README.md for why the
// C++ and Java rows are here and what they are not. Vec2 is heap-allocated
// because `add` allocates in the original, and so the loop survives -O2.
#include <cstdio>
#include <memory>

struct Vec2 {
    double x, y;
    Vec2(double x, double y) : x(x), y(y) {}
    std::unique_ptr<Vec2> add(const Vec2 &o) const {
        return std::make_unique<Vec2>(x + o.x, y + o.y);
    }
    double dot(const Vec2 &o) const { return x * o.x + y * o.y; }
};

int main() {
    auto acc = std::make_unique<Vec2>(0.0, 0.0);
    auto step = std::make_unique<Vec2>(1.0, 2.0);
    double total = 0.0;
    for (int i = 0; i < 1000000; i++) {
        acc = acc->add(*step);
        total += acc->dot(*step);
    }
    std::printf("%.1f\n", total);
    return 0;
}

// Same program as mandelbrot.jai. See tests/bench/README.md for why the C++ and
// Java rows are here and what they are not. Each step of the recurrence lands in
// its own variable, so no statement holds a multiply feeding an add and there is
// nothing for the compiler to fuse — every port rounds the same way.
#include <cstdint>
#include <cstdio>

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
// The work is width*height, so halve each side per level.
static const int DIM_DIV = SCALE == 16 ? 4 : (SCALE == 4 ? 2 : 1);

int main() {
    const int width = 3200 / DIM_DIV;
    const int height = 240 / DIM_DIV;
    const int limit = 100;

    int64_t inside = 0;
    for (int py = 0; py < height; py++) {
        double y0 = (double)py * 2.0 / (double)height - 1.0;
        for (int px = 0; px < width; px++) {
            double x0 = (double)px * 3.0 / (double)width - 2.0;
            double x = 0.0;
            double y = 0.0;
            int i = 0;
            while (i < limit) {
                double x2 = x * x;
                double y2 = y * y;
                if (x2 + y2 > 4.0) break;
                double xy = x * y;
                y = 2.0 * xy + y0;
                x = x2 - y2 + x0;
                i++;
            }
            if (i == limit) inside++;
        }
    }
    std::printf("%d\n", limit);
    std::printf("%lld\n", (long long)inside);
    return 0;
}

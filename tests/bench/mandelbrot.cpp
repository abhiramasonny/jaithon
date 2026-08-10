// Same program as mandelbrot.jai. See tests/bench/README.md for why the C++ and
// Java rows are here and what they are not. Each step of the recurrence lands in
// its own variable, so no statement holds a multiply feeding an add and there is
// nothing for the compiler to fuse — every port rounds the same way.
#include <cstdint>
#include <cstdio>

int main() {
    const int width = 320;
    const int height = 240;
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

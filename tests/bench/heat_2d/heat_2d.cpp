// Same program as heat_2d.jai. See tests/bench/README.md for why the C++ and
// Java rows are here and what they are not.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static long bench_shrink() {
    const char *l = std::getenv("BENCH_LEVEL");
    if (l && std::strcmp(l, "easy") == 0) return 4;
    if (l && std::strcmp(l, "medium") == 0) return 2;
    return 1;
}
static const long SHRINK = bench_shrink();
static const int SIDE = (int)(500 / SHRINK);
static const int STEPS = (60 / SHRINK) < 1 ? 1 : (int)(60 / SHRINK);

static std::vector<std::vector<double>> plate(int n) {
    std::vector<std::vector<double>> g;
    for (int i = 0; i < n; i++) {
        std::vector<double> row;
        for (int j = 0; j < n; j++) {
            row.push_back(i == 0 ? 100.0 : 0.0);
        }
        g.push_back(row);
    }
    return g;
}

static void relax(std::vector<std::vector<double>> &src,
                  std::vector<std::vector<double>> &dst, int n) {
    for (int i = 1; i < n - 1; i++) {
        const std::vector<double> &up = src[(size_t)(i - 1)];
        const std::vector<double> &mid = src[(size_t)i];
        const std::vector<double> &down = src[(size_t)(i + 1)];
        std::vector<double> &out = dst[(size_t)i];
        for (int j = 1; j < n - 1; j++) {
            out[(size_t)j] = 0.25 * (up[(size_t)j] + down[(size_t)j] +
                                     mid[(size_t)(j - 1)] + mid[(size_t)(j + 1)]);
        }
    }
}

int main() {
    const int n = SIDE;
    std::vector<std::vector<double>> a = plate(n);
    std::vector<std::vector<double>> b = plate(n);

    int s = 0;
    while (s < STEPS) {
        relax(a, b, n);
        relax(b, a, n);
        s += 2;
    }

    double total = 0.0;
    for (int i = 0; i < n; i++) {
        const std::vector<double> &row = a[(size_t)i];
        for (int j = 0; j < n; j++) total += row[(size_t)j];
    }
    std::printf("%d\n", n);
    std::printf("%.6f\n", total);
    return 0;
}

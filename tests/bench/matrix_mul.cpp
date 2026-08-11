// Same program as matrix_mul.jai. See tests/bench/README.md for why the C++ and
// Java rows are here and what they are not.
#include <cstdint>
#include <cstdio>
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
// The work is n^3, so halve n per level rather than dividing by SCALE.
static const int DIM_DIV = SCALE == 16 ? 4 : (SCALE == 4 ? 2 : 1);

static std::vector<std::vector<double>> make(int n, int64_t seed) {
    std::vector<std::vector<double>> m;
    int64_t s = seed;
    for (int i = 0; i < n; i++) {
        std::vector<double> row;
        for (int j = 0; j < n; j++) {
            s = (s * 1103515245 + 12345) % 2147483648;
            row.push_back((double)(s % 1000) / 1000.0);
        }
        m.push_back(row);
    }
    return m;
}

int main() {
    const int n = 320 / DIM_DIV;
    std::vector<std::vector<double>> a = make(n, 12345);
    std::vector<std::vector<double>> b = make(n, 67890);

    std::vector<std::vector<double>> c;
    for (int i = 0; i < n; i++) {
        std::vector<double> row;
        const std::vector<double> &ai = a[i];
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += ai[k] * b[k][j];
            }
            row.push_back(sum);
        }
        c.push_back(row);
    }

    double trace = 0.0;
    for (int i = 0; i < n; i++) trace += c[i][i];
    std::printf("%d\n", n);
    std::printf("%.6f\n", trace);
    return 0;
}

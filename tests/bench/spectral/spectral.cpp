// Same program as spectral.jai. See tests/bench/README.md for why the C++ and
// Java rows are here and what they are not.
#include <cmath>
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
// A repetition count around a fixed matrix: never below one round.
static const long ROUNDS = (6 / SCALE) < 1 ? 1 : (6 / SCALE);

static double evalA(int64_t i, int64_t j) {
    int64_t s = i + j;
    return 1.0 / (double)(s * (s + 1) / 2 + i + 1);
}

static std::vector<double> timesA(const std::vector<double> &v, int n) {
    std::vector<double> out;
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) sum += evalA(i, j) * v[(size_t)j];
        out.push_back(sum);
    }
    return out;
}

static std::vector<double> timesAt(const std::vector<double> &v, int n) {
    std::vector<double> out;
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) sum += evalA(j, i) * v[(size_t)j];
        out.push_back(sum);
    }
    return out;
}

int main() {
    const int n = 550;
    std::vector<double> u((size_t)n, 1.0);
    std::vector<double> v;
    for (long r = 0; r < ROUNDS; r++) {
        v = timesAt(timesA(u, n), n);
        u = timesAt(timesA(v, n), n);
    }

    double vBv = 0.0;
    double vv = 0.0;
    for (int i = 0; i < n; i++) {
        vBv += u[(size_t)i] * v[(size_t)i];
        vv += v[(size_t)i] * v[(size_t)i];
    }
    std::printf("%d\n", n);
    std::printf("%.9f\n", std::pow(vBv / vv, 0.5));
    return 0;
}

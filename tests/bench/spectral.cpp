// Same program as spectral.jai. See tests/bench/README.md for why the C++ and
// Java rows are here and what they are not.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

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
    const int n = 150;
    std::vector<double> u((size_t)n, 1.0);
    std::vector<double> v;
    for (int r = 0; r < 6; r++) {
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

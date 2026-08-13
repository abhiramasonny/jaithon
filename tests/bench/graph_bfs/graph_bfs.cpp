// Same program as graph_bfs.jai. See tests/bench/README.md for why the C++ and
// Java rows are here and what they are not.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static long bench_scale() {
    const char *l = std::getenv("BENCH_LEVEL");
    if (l && std::strcmp(l, "easy") == 0) return 16;
    if (l && std::strcmp(l, "medium") == 0) return 4;
    return 1;
}
static const long SCALE = bench_scale();
static const int64_t NODES = 80000;
static const int64_t DEGREE = 6;

static std::vector<std::vector<int64_t>> build(int64_t n, int64_t deg) {
    std::vector<std::vector<int64_t>> g;
    int64_t s = 7;
    for (int64_t i = 0; i < n; i++) {
        std::vector<int64_t> adj;
        for (int64_t d = 0; d < deg; d++) {
            s = (s * 1103515245LL + 12345LL) % 2147483648LL;
            adj.push_back(s % n);
        }
        g.push_back(adj);
    }
    return g;
}

static int64_t bfs(const std::vector<std::vector<int64_t>> &g, int64_t n,
                   int64_t start) {
    std::vector<int64_t> dist;
    for (int64_t i = 0; i < n; i++) dist.push_back(-1);
    std::vector<int64_t> queue;
    queue.push_back(start);
    dist[(size_t)start] = 0;
    size_t head = 0;
    int64_t total = 0;
    while (head < queue.size()) {
        int64_t node = queue[head];
        head += 1;
        int64_t d = dist[(size_t)node];
        total += d;
        for (int64_t nb : g[(size_t)node]) {
            if (dist[(size_t)nb] < 0) {
                dist[(size_t)nb] = d + 1;
                queue.push_back(nb);
            }
        }
    }
    return total;
}

int main() {
    const int64_t reps = (6 / SCALE) < 1 ? 1 : (6 / SCALE);
    const int64_t n = NODES;
    std::vector<std::vector<int64_t>> g = build(n, DEGREE);
    int64_t total = 0;
    for (int64_t r = 0; r < reps; r++) total = bfs(g, n, 0);
    std::printf("%lld\n", (long long)n);
    std::printf("%lld\n", (long long)total);
    return 0;
}

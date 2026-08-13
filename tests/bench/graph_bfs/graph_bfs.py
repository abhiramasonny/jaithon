import os
import sys

_LEVEL = os.environ.get("BENCH_LEVEL", "hard")
SCALE = 16 if _LEVEL == "easy" else 4 if _LEVEL == "medium" else 1
NODES = 80_000
DEGREE = 6
REPS = max(6 // SCALE, 1)


def build(n, deg):
    g = []
    s = 7
    for _i in range(n):
        adj = []
        for _d in range(deg):
            s = (s * 1103515245 + 12345) % 2147483648
            adj.append(s % n)
        g.append(adj)
    return g


def bfs(g, n, start):
    dist = []
    for _i in range(n):
        dist.append(-1)
    queue = []
    queue.append(start)
    dist[start] = 0
    head = 0
    total = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        d = dist[node]
        total += d
        for nb in g[node]:
            if dist[nb] < 0:
                dist[nb] = d + 1
                queue.append(nb)
    return total


def main():
    n = NODES
    g = build(n, DEGREE)
    total = 0
    for _r in range(REPS):
        total = bfs(g, n, 0)
    print(n)
    print(total)
    return 0


if __name__ == "__main__":
    sys.setrecursionlimit(100000)
    main()

import os

_LEVEL = os.environ.get("BENCH_LEVEL", "hard")
SHRINK = 4 if _LEVEL == "easy" else 2 if _LEVEL == "medium" else 1
SIDE = 300 // SHRINK
STEPS = max(40 // SHRINK, 1)


def plate(n):
    g = []
    for i in range(n):
        row = []
        for _j in range(n):
            if i == 0:
                row.append(100.0)
            else:
                row.append(0.0)
        g.append(row)
    return g


def relax(src, dst, n):
    for i in range(1, n - 1):
        up = src[i - 1]
        mid = src[i]
        down = src[i + 1]
        out = dst[i]
        for j in range(1, n - 1):
            out[j] = 0.25 * (up[j] + down[j] + mid[j - 1] + mid[j + 1])


def main():
    n = SIDE
    a = plate(n)
    b = plate(n)

    s = 0
    while s < STEPS:
        relax(a, b, n)
        relax(b, a, n)
        s += 2

    total = 0.0
    for i in range(n):
        row = a[i]
        for j in range(n):
            total += row[j]
    print(n)
    print(f"{total:.6f}")
    return 0


main()

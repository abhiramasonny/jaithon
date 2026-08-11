import os

_LEVEL = os.environ.get("BENCH_LEVEL", "hard")
SCALE = 16 if _LEVEL == "easy" else 4 if _LEVEL == "medium" else 1
# The matrix size stays; the repetitions around it are what the level shortens.
REPS = max(1, 6 // SCALE)


def evalA(i, j):
    s = i + j
    return 1.0 / float(s * (s + 1) // 2 + i + 1)


def timesA(v, n):
    out = []
    for i in range(n):
        total = 0.0
        for j in range(n):
            total += evalA(i, j) * v[j]
        out.append(total)
    return out


def timesAt(v, n):
    out = []
    for i in range(n):
        total = 0.0
        for j in range(n):
            total += evalA(j, i) * v[j]
        out.append(total)
    return out


n = 550
u = [1.0] * n
v = []
for _r in range(REPS):
    v = timesAt(timesA(u, n), n)
    u = timesAt(timesA(v, n), n)

vBv = 0.0
vv = 0.0
for i in range(n):
    vBv += u[i] * v[i]
    vv += v[i] * v[i]
print(n)
print(f"{(vBv / vv) ** 0.5:.9f}")

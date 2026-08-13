import os

_LEVEL = os.environ.get("BENCH_LEVEL", "hard")
SCALE = 16 if _LEVEL == "easy" else 4 if _LEVEL == "medium" else 1
CALM = max(8_000_000 // SCALE, 1)
STORM = max(160_000 // SCALE, 1)


def checked(x):
    if x % 8 == 3:
        raise ValueError(f"bad value {x}")
    return x + 1


def calm(n):
    total = 0
    i = 0
    while i < n:
        try:
            total += i % 13
        except ValueError:
            total -= 1
        i += 1
    return total


def storm(n):
    ok = 0
    bad = 0
    i = 0
    while i < n:
        try:
            ok += checked(i)
        except ValueError:
            bad += 1
        i += 1
    return ok - bad


print(calm(CALM))
print(storm(STORM))

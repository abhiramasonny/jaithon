import os

LEVEL = os.environ.get("BENCH_LEVEL", "hard")
SCALE = 20 if LEVEL == "easy" else 5 if LEVEL == "medium" else 1
KEYS = 20000
REPS = max(120 // SCALE, 1)


def build(n):
    return {f"k{i}": i for i in range(n)}


def sum_values(d):
    total = 0
    for _k, v in d.items():
        total += v
    return total


def count_long_keys(d):
    seen = 0
    for k, v in d.items():
        if len(k) > 4 and v % 2 == 0:
            seen += 1
    return seen


def main():
    d = build(KEYS)
    total = 0
    seen = 0
    for _ in range(REPS):
        total = sum_values(d)
        seen = count_long_keys(d)
    print(total)
    print(seen)


main()

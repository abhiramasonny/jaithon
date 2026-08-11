import os

_LEVEL = os.environ.get("BENCH_LEVEL", "hard")
SCALE = 16 if _LEVEL == "easy" else 4 if _LEVEL == "medium" else 1
ITERS = 50_000_000 // SCALE


def main():
    total = 0
    i = 0
    while i < ITERS:
        total += i % 7
        i += 1
    print(total)

main()

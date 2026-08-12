import os

_LEVEL = os.environ.get("BENCH_LEVEL", "hard")
# Work is exponential in n, so the level subtracts from it instead of dividing.
DEPTH_DROP = 4 if _LEVEL == "easy" else 2 if _LEVEL == "medium" else 0
N = 38 - DEPTH_DROP


def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

print(fib(N))

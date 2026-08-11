import os

_LEVEL = os.environ.get("BENCH_LEVEL", "hard")
SCALE = 16 if _LEVEL == "easy" else 4 if _LEVEL == "medium" else 1
ITERS = 30_000_000 // SCALE

counts = {}
for i in range(ITERS):
    key = f"k{i % 10_000}"
    counts[key] = counts.get(key, 0) + 1

total = 0
for _key, value in counts.items():
    total += value
print(len(counts))
print(total)

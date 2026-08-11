import os

_LEVEL = os.environ.get("BENCH_LEVEL", "hard")
SCALE = 16 if _LEVEL == "easy" else 4 if _LEVEL == "medium" else 1
N = 2_000_000 // SCALE

parts = []
for i in range(N):
    parts.append(f"item-{i}")
joined = ",".join(parts)
print(len(joined))
print(len(joined.split(",")))

import os

_LEVEL = os.environ.get("BENCH_LEVEL", "hard")
SCALE = 16 if _LEVEL == "easy" else 4 if _LEVEL == "medium" else 1
N = 10_000_000 // SCALE
INDEX = 999_999 // SCALE

xs = []
for i in range(N):
    xs.append(i * 3)

total = 0
for x in xs:
    total += x % 11

doubled = [x * 2 for x in xs]
print(total)
print(doubled[INDEX])

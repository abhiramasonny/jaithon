counts = {}
for i in range(30_000_000):
    key = f"k{i % 10_000}"
    counts[key] = counts.get(key, 0) + 1

total = 0
for _key, value in counts.items():
    total += value
print(len(counts))
print(total)

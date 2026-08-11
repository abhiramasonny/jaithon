parts = []
for i in range(2_000_000):
    parts.append(f"item-{i}")
joined = ",".join(parts)
print(len(joined))
print(len(joined.split(",")))

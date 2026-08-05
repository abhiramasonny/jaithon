parts = []
for i in range(200_000):
    parts.append(f"item-{i}")
joined = ",".join(parts)
print(len(joined))
print(len(joined.split(",")))

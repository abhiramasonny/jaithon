xs = []
for i in range(10_000_000):
    xs.append(i * 3)

total = 0
for x in xs:
    total += x % 11

doubled = [x * 2 for x in xs]
print(total)
print(doubled[999_999])

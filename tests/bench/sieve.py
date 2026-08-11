n = 10_000_000
flags = []
for _i in range(n + 1):
    flags.append(True)

count = 0
total = 0
p = 2
while p * p <= n:
    if flags[p]:
        q = p * p
        while q <= n:
            flags[q] = False
            q += p
    p += 1

i = 2
while i <= n:
    if flags[i]:
        count += 1
        total = (total + i) % 1000000007
    i += 1
print(count)
print(total)

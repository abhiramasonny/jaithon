def make(n, seed):
    m = []
    s = seed
    for _i in range(n):
        row = []
        for _j in range(n):
            s = (s * 1103515245 + 12345) % 2147483648
            row.append(float(s % 1000) / 1000.0)
        m.append(row)
    return m


n = 320
a = make(n, 12345)
b = make(n, 67890)

c = []
for i in range(n):
    row = []
    ai = a[i]
    for j in range(n):
        total = 0.0
        for k in range(n):
            total += ai[k] * b[k][j]
        row.append(total)
    c.append(row)

trace = 0.0
for i in range(n):
    trace += c[i][i]
print(n)
print(f"{trace:.6f}")

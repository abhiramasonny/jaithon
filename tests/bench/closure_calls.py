def adder(step):
    return lambda x: x + step


def apply_n(f, start, times):
    acc = start
    i = 0
    while i < times:
        acc = f(acc)
        i += 1
    return acc


total = 0
k = 1
while k <= 1000:
    bump = adder(k)
    total = (total + apply_n(bump, 0, 100_000)) % 1000000007
    k += 1
print(total)

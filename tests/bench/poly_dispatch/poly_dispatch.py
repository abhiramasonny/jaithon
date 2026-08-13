import os

_LEVEL = os.environ.get("BENCH_LEVEL", "hard")
SCALE = 16 if _LEVEL == "easy" else 4 if _LEVEL == "medium" else 1
OPS = 512
REPS = max(8000 // SCALE, 1)
M = 1000003


class AddK:
    def __init__(self, k):
        self.k = k

    def apply(self, x):
        return (x + self.k) % M


class MulK:
    def __init__(self, k):
        self.k = k

    def apply(self, x):
        return (x * self.k) % M


class SquareK:
    def __init__(self, k):
        self.k = k

    def apply(self, x):
        return (x * x + self.k) % M


class DoubleK:
    def __init__(self, k):
        self.k = k

    def apply(self, x):
        return (x * 2 + self.k) % M


class DivK:
    def __init__(self, k):
        self.k = k

    def apply(self, x):
        return (x + x // self.k) % M


class ModK:
    def __init__(self, k):
        self.k = k

    def apply(self, x):
        return (x + x % self.k) % M


class FlipK:
    def __init__(self, k):
        self.k = k

    def apply(self, x):
        return (M - 1 - x + self.k) % M


class MixK:
    def __init__(self, k):
        self.k = k

    def apply(self, x):
        return (x + x // 3 + self.k) % M


def build(n):
    ops = []
    s = 7
    for _i in range(n):
        s = (s * 1103515245 + 12345) % 2147483648
        which = (s // 65536) % 8
        k = s % 97 + 2
        if which == 0:
            ops.append(AddK(k))
        elif which == 1:
            ops.append(MulK(k))
        elif which == 2:
            ops.append(SquareK(k))
        elif which == 3:
            ops.append(DoubleK(k))
        elif which == 4:
            ops.append(DivK(k))
        elif which == 5:
            ops.append(ModK(k))
        elif which == 6:
            ops.append(FlipK(k))
        else:
            ops.append(MixK(k))
    return ops


def main():
    ops = build(OPS)
    acc = 1
    check = 0
    for r in range(REPS):
        acc = (acc + r) % M
        for op in ops:
            acc = op.apply(acc)
        check = (check + acc) % M
    print(len(ops))
    print(acc)
    print(check)
    return 0


main()

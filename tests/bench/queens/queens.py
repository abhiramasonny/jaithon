import sys
import os

_LEVEL = os.environ.get("BENCH_LEVEL", "hard")
SCALE = 16 if _LEVEL == "easy" else 4 if _LEVEL == "medium" else 1
# The board size stays; the repetitions around it are what the level shortens.
REPS = max(1, 3 // SCALE)


def safe(cols, row, col):
    r = 0
    while r < row:
        c = cols[r]
        if c == col:
            return False
        if c - r == col - row:
            return False
        if c + r == col + row:
            return False
        r += 1
    return True


def place(cols, row, n):
    if row == n:
        return 1
    found = 0
    col = 0
    while col < n:
        if safe(cols, row, col):
            cols[row] = col
            found += place(cols, row + 1, n)
        col += 1
    return found


sys.setrecursionlimit(10000)
n = 11
cols = [0] * n
total = 0
for _rep in range(REPS):
    total = place(cols, 0, n)
print(n)
print(total)

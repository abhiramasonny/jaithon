import os

_LEVEL = os.environ.get("BENCH_LEVEL", "hard")
SCALE = 16 if _LEVEL == "easy" else 4 if _LEVEL == "medium" else 1
ITERS = 1_000_000 // SCALE


def popcount(value):
    v = value
    bits = 0
    while v != 0:
        v = v & (v - 1)
        bits += 1
    return bits


mask = 4294967295
seed = 7
ones = 0
checksum = 0
i = 0
while i < ITERS:
    seed = (seed * 1103515245 + 12345) % 2147483648
    v = seed ^ checksum
    v = v ^ (v >> 7)
    v = (v ^ (v << 3)) & mask
    v = v ^ (v >> 11)
    bits = popcount(v)
    ones += bits
    checksum = (checksum + v + bits) & mask
    checksum = ((checksum << 1) | (checksum >> 31)) & mask
    i += 1
print(ones)
print(checksum)

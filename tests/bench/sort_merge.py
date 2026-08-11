import sys

sys.setrecursionlimit(100000)


def merge(left, right):
    out = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    while i < len(left):
        out.append(left[i])
        i += 1
    while j < len(right):
        out.append(right[j])
        j += 1
    return out


def sort(values):
    if len(values) <= 1:
        return values
    mid = len(values) // 2
    return merge(sort(values[0:mid]), sort(values[mid:len(values)]))


data = []
seed = 12345
for _i in range(1_000_000):
    seed = (seed * 1103515245 + 12345) % 2147483648
    data.append(seed % 100000)
ordered = sort(data)
checksum = 0
at = 0
while at < len(ordered):
    checksum = (checksum + ordered[at] * (at % 7 + 1)) % 1000000007
    at += 1
print(len(ordered))
print(checksum)

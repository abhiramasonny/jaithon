import os

_LEVEL = os.environ.get("BENCH_LEVEL", "hard")
SCALE = 16 if _LEVEL == "easy" else 4 if _LEVEL == "medium" else 1
# The text stays the same size; the repetitions over it are what shortens.
REPS = max(1, 5 // SCALE)


def count_occurrences(text, needle):
    n = len(text)
    m = len(needle)
    first = needle[0]
    hits = 0
    i = 0
    while i + m <= n:
        if text[i] == first:
            k = 1
            while k < m and text[i + k] == needle[k]:
                k += 1
            if k == m:
                hits += 1
        i += 1
    return hits


chunks = ["abcdbadc", "bcadcbda", "cdabacbd", "dacbdabc",
          "abdcadbc", "bdacbcad", "cabdbdca", "dbcaabcd"]
parts = []
seed = 7
for _i in range(250_000):
    seed = (seed * 1103515245 + 12345) % 2147483648
    parts.append(chunks[(seed // 65536) % 8])
text = "".join(parts)

hits = 0
for _rep in range(REPS):
    hits = count_occurrences(text, "abcd")
print(len(text))
print(hits)

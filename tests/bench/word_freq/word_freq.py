import os

_LEVEL = os.environ.get("BENCH_LEVEL", "hard")
SCALE = 16 if _LEVEL == "easy" else 4 if _LEVEL == "medium" else 1
WORDS = 200_000 // SCALE

text = ""
seed = 7
for _i in range(WORDS):
    seed = (seed * 1103515245 + 12345) % 2147483648
    text = text + f"w{seed % 500} "

counts = {}
start = 0
at = 0
n = len(text)
while at < n:
    if text[at] == " ":
        if at > start:
            word = text[start:at]
            counts[word] = counts.get(word, 0) + 1
        start = at + 1
    at += 1

total = 0
for _key, value in counts.items():
    total += value
print(len(counts))
print(total)

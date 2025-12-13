sum = 0
i = 0
while i < 5000000:
    sum = sum + (i % 7)
    i = i + 1
print(sum)
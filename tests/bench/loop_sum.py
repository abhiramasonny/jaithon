def main():
    total = 0
    i = 0
    while i < 50_000_000:
        total += i % 7
        i += 1
    print(total)

main()

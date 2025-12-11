# Test: Recursion
# Expected output: 55, 8, 6

func fib(n)
    if n <= 1
        return n
    end
    return fib(n - 1) + fib(n - 2)
end

print fib(10)

func sumDigits(n)
    if n < 10
        return n
    end
    return n % 10 + sumDigits(floor(n / 10))
end

print sumDigits(35)

func countSteps(n)
    if n <= 0
        return 0
    end
    return 1 + countSteps(n - 1)
end

print countSteps(6)

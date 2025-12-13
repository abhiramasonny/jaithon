# Test: Closures and higher-order functions
# Expected output: 15, 9, 48

var multiplier = 5

func times(x)
    return x * multiplier
end

print times(3)

func applyTwice(f, x)
    return f(f(x))
end

func addOne(n)
    return n + 1
end

func timesTwo(n)
    return n * 2
end

print applyTwice(addOne, 7)
print applyTwice(timesTwo, 12)

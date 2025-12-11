# Test: Functions
# Expected output: 8, 120, hello world, 25, result=15

func add(a, b)
    return a + b
end

print add(3, 5)

func factorial(n)
    if n <= 1
        return 1
    end
    return n * factorial(n - 1)
end

print factorial(5)

func greet(name)
    return "hello " + name
end

print greet("world")

func square(x)
    var result = x * x
    return result
end

print square(5)

func combined(a, b)
    var sum = add(a, b)
    return sum
end

print "result=" + str(combined(7, 8))

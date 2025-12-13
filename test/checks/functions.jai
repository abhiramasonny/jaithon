# Test: Functions with typed Java-style declarations
# Expected output: 8, 120, hello world, 25, result=15

public static int add in Math(int a, int b)
    return a + b
end

print Math.add(3, 5)

public int factorial(int n)
    if n <= 1
        return 1
    end
    return n * factorial(n - 1)
end

print factorial(5)

private string greet(string name)
    return "hello " + name
end

print greet("world")

public var square(var x)
    var result = x * x
    return result
end

print square(5)

public int combined(int a, int b)
    return Math.add(a, b)
end

print "result=" + str(combined(7, 8))

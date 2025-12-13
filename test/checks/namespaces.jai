# Test: Namespaces + Java-style declarations
# Expected output: 78.5398, 31.4159, 200, 300

public static double PI in Circle = 3.14159

public double area in Circle(double r)
    return PI * r * r
end

public double circumference in Circle(double r)
    return 2 * PI * r
end

print Circle.area(5)
print Circle.circumference(5)

public int timesTwo in Utils(int x)
    return x * 2
end

public int triple in Utils(int x)
    return x * 3
end

print Utils.timesTwo(100)
print Utils.triple(100)

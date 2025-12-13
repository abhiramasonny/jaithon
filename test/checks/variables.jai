# Test: Variable declaration, assignment, and operations (Java-style + var)
# Expected output lines: 10, 15, hello, 3.14, 0.5, dyn, 2

public int counter = 10
print counter

counter = counter + 5
print counter

public string name = "hello"
print name

public static double pi = 3.14
print pi

float ratio
ratio = 0.5
print ratio

public var dynamic in metrics = "dyn"
public int metricCount in metrics = 2
print metrics.dynamic
print metrics.metricCount

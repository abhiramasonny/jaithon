# Test: Type conversions with typed vars
# Expected output: 42, 3.14, 123, number, string, array

# String to number
string s = "42"
print num(s)

double n = 3.14
print str(n)

int x = 123
print str(x)

print type(42)
print type("hello")
print type([1, 2, 3])

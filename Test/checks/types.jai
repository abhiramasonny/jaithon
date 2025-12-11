# Test: Type conversions
# Expected output: 42, 3.14, 123, number, string, array

# String to number
var s = "42"
print num(s)

# Number to string
var n = 3.14
print str(n)

# Concatenation with str
var x = 123
print str(x)

# Type checking
print type(42)
print type("hello")
print type([1, 2, 3])

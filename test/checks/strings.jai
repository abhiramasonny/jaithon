# Test: String operations
# Expected output: 5, helloworld, H, 2, HELLO, hello

var s = "hello"
print len(s)

var s2 = concat(s, "world")
print s2

print charAt("Hello", 0)

print indexOf("hello", "l")

print upper("hello")
print lower("HELLO")

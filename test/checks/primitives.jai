# Test: Primitive types and boxed helpers
# Expected output:
# 5, 12345678901, 1.5, 2.25, 12, 7, Z, 1, 42, 42, 42

public int a = 5
public long l = 12345678901
public float f = 1.5
public double d = 2.25
public short s = 12
public byte b = 7
public char c = "Z"
public bool flag = true

print a
print l
print f
print d
print s
print b
print c
print flag

print Integer.toString(Integer.parseInt("42"))
print Long.toString(Long.parseLong("42"))
print String.valueOf(42)

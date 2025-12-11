# Test: Advanced string helpers
# Expected output: hahaha, 1, 1, 1, hi, baz bar baz, cba

print repeat("ha", 3)
print startsWith("hello", "he")
print endsWith("hello", "lo")
print contains_str("abracadabra", "cada")
print trim("  hi  ")
print replace("foo bar foo", "foo", "baz")
print reverse_str("abc")

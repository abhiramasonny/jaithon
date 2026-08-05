# expect: E0005
# `1-9223372036854775808` is a subtraction — `1` ends an operand, so the `-` is
# binary — and its right side is an out-of-range literal.

print(1-9223372036854775808)

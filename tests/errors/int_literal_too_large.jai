# expect: E0005
# 9223372036854775808 is one past the largest `int`; only `-9223372036854775808`
# spells it, and there the `-` is part of the literal.

print(9223372036854775808)

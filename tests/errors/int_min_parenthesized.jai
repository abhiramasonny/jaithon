# expect: E0005
# The `-` must sit directly in front of the literal: a `(` in between leaves an
# out-of-range literal of its own.

print(-(9223372036854775808))

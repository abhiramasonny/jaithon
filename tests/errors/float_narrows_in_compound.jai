# expect: E0411
# A compound assignment writes back into its target, so an int target cannot
# take the float result even though the same operands widen fine in `a + b`.

var count = 0
count += 1.5
print(count)

# expect: E0005
# The extra magnitude is admitted for exactly one prefix `-`. A second one asks
# for `-INT_MIN`, which is not an `int`, so the literal is rejected.

print(--9223372036854775808)

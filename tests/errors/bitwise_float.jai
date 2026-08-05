# expect: E0406
# Widening stops at the bitwise operators: they work on the 64 bits of an int
# and a float has no bits to work on.

let mask = 0b1010
let scale = 2.0
print(mask & scale)

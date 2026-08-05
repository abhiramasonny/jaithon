# expect: E0801
# Circular imports are detected and reported with the cycle.

from .circular_import_b import b_value

pub let a_value = 1
print(b_value)

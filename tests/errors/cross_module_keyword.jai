# expect: E0603
# Keyword-argument names are checked across a module boundary too, so a typo in
# a parameter name is not deferred to a run-time TypeError.
from .cross_module.shapes import scaled

print(scaled(1, factor: 2, lable: "n"))

# expect: E0400
# A parameter's declared type crosses the boundary with its name: `label` is a
# `str` in the module that declares `scaled`, and an `int` is not one here.
from .cross_module.shapes import scaled

print(scaled(1, 2, 3))

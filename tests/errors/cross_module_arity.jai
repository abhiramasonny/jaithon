# expect: E0602
# Arity is checked across a module boundary exactly as it is inside one: the
# checker reads the imported declaration's parameter list out of that module's
# source, whether or not it has a warm `.jaic`.
from .cross_module.shapes import Box, scaled

let box = Box(3)
box.resize()
print(scaled(1, 2))

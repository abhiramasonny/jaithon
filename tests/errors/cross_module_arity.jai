# expect: E0602
from .cross_module.shapes import Box, scaled

let box = Box(3)
box.resize()
print(scaled(1, 2))

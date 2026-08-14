#: Imports, aliases, selective import, visibility, and single execution
#: of a module body (spec §8).
import .modules_demo.geometry
from .modules_demo.geometry import circle_area, UNIT
import .modules_demo.counter as tally

print(f"{geometry.circle_area(1.0):.5f}")
print(f"{circle_area(2.0):.5f}")
print(geometry.rect_area(3.0, 4.0))
print(UNIT)
print(geometry.scaled(2.5))

import .modules_demo.counter
print(tally.bump())
print(tally.bump())
print(tally.current())
print(counter.current())  # same module object, same state

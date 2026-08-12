#: Re-export the names that every Jaithon program has without writing an import.
#:
#: Loaded before the entry module on every run, so it holds only imports and
#: re-exports — no classes, no runtime cost.
from std.core import Iterable, Iterator, Comparable, Hashable, Printable
from std.core import Cloneable, Default, Container, Ordering, cmp

export { Iterable, Iterator, Comparable, Hashable, Printable, Cloneable }
export { Default, Container, Ordering, cmp }

# The exception hierarchy (spec §5.4) is registered by the runtime directly in
# builtins, so it must not be re-exported here — `export` only names declarations
# of *this* module.

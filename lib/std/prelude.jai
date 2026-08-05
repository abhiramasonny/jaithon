#: Re-export the names that every Jaithon program has without writing an import.
#:
#: The runtime loads this module before the entry module runs, so its cost is
#: paid on every start. It therefore contains imports and re-exports and nothing
#: else: no constants to build, no classes to define, no code to run.
from std.core import Iterable, Iterator, Comparable, Hashable, Printable
from std.core import Cloneable, Default, Container, Ordering, cmp

export { Iterable, Iterator, Comparable, Hashable, Printable, Cloneable }
export { Default, Container, Ordering, cmp }

# The exception hierarchy of spec §5.4 is registered directly in the builtins
# module by the runtime, so it is already visible everywhere and must not be
# re-exported here — `export` names declarations of *this* module.
#
# Unqualified names available in a fresh module are therefore: the traits above,
# the builtin functions of spec §9's std.core, and the exception classes.

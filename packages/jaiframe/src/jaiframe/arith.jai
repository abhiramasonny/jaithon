#: Elementwise arithmetic, comparison, logic — and the one door out to the host.
#:
#: Everything here is column-at-a-time. Two Metal entry points do all the work:
#: `frame_binary`, parameterised by an op code, and `frame_unary`. Both read
#: the validity lanes of their inputs and write the validity lane of their
#: result, so a null propagates inside the dispatch and the host never branches
#: on one; both reconstruct the two-lane `Int64`/`Datetime` representation in
#: `long` so integer arithmetic stays exact to 48 bits.
#:
#: One kernel with a switch rather than one kernel per operation: these are
#: memory-bound — a thread reads two floats and writes one — so the branch
#: costs nothing next to the traffic, while sixteen pipelines would be sixteen
#: compiles and sixteen cache entries.
#:
#: The dtype rules are pandas': a `Float64` anywhere makes the result
#: `Float64`, `Bool` counts as a number and promotes to `Int64`, division and
#: exponentiation always produce `Float64`, and a comparison always produces
#: `Bool`. Nulls propagate through comparisons rather than reading as false,
#: which is what pandas' nullable dtypes do and what a validity lane is for.
#:
#: ```
#: let high = Column.from_floats([38.0, 41.0, NAN])
#: let low = Column.from_floats([21.0, 25.0, 19.0])
#: let swing = subtract(high, low)       # [17.0, 16.0, null]
#: let hot = greater(high, 40.0)         # [false, true, null]
#: ```
#:
#: `map`, `apply`, `applymap`, `pipe` and `combine` are the escape hatch. They
#: live here because a user callback is the thing arithmetic cannot vectorise,
#: and every one of them costs a round trip and a host loop over the rows.
from .arith.callbacks import apply, applymap, combine, map, pipe
from .arith.kernels import cached_kernel
from .arith.operators import absolute, add, and_, ceiling, clip, divide, equal, exp
from .arith.operators import floor, floor_divide, greater, greater_equal, less
from .arith.operators import less_equal, log, log10, log2, modulo, multiply, negate
from .arith.operators import not_, not_equal, or_, power, reciprocal, round, sign
from .arith.operators import sqrt, square, subtract, truncate, xor
from .arith.rules import promote

export { cached_kernel, promote }
export { add, subtract, multiply, divide, floor_divide, modulo, power }
export { equal, not_equal, less, less_equal, greater, greater_equal }
export { and_, or_, xor, not_ }
export { negate, absolute, sign, round, floor, ceiling, truncate }
export { exp, log, log2, log10, sqrt, square, reciprocal, clip }
export { map, apply, applymap, pipe, combine }

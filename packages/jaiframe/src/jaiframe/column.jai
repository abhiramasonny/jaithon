#: The typed nullable column store every other jaiframe module is built on.
#:
#: Storage is one contiguous `std.gpu.Buffer` of float32 slots per lane,
#: exactly as jaicv's `Mat` is, so a whole column is one allocation and every
#: operation over it is one dispatch.
#:
#: A float32 slot is exact only to 2^24, and a pandas clone has to carry exact
#: int64 and exact millisecond timestamps. So a column is up to three lanes:
#:
#: - `values` — the payload for `Float64` and `Bool`, and the dictionary code
#:   for `Str` and `Category`.
#: - `wide` — present only for `Int64` and `Datetime`, where the value is
#:   `wide * 2^24 + values`. Both halves are integral and below 2^24, so both
#:   are exact in float32 and the pair is exact to 48 bits. Metal reconstructs
#:   it in `long`, never in `float`.
#: - `valid` — 1.0 where a row holds a value, 0.0 where it is missing. `null`
#:   means the column has no missing values at all, which is what lets a kernel
#:   skip the mask instead of reading a lane of ones.
#:
#: `Datetime` is milliseconds since the Unix epoch — datetime64[ms]. 48 bits of
#: milliseconds is +/-8900 years; 48 bits of nanoseconds would be +/-3 days,
#: which is not a calendar. `Float64` is a single float32 lane and is named for
#: source compatibility, the same compromise jaicv makes with `CV_64F`.
#:
#: `Str` and `Category` share one representation: codes in `values` and a
#: `levels` table on the side. That is the only encoding that keeps a string
#: column contiguous, and it is what makes string equality an integer compare
#: and therefore hashable by the same kernel that hashes numbers.
#:
#: Go through the accessors here. Nothing outside this file should read a lane
#: directly, because which lanes exist depends on the dtype and on whether the
#: column has ever held a null.
#:
#: ```
#: let ages = Column.from_ints([31, 47, 29])
#: defer { ages.free() }
#: ages.get(1).to_int()          # 47
#: ages.null_count()             # 0
#: ```
from .column.lanes import MAX_ROWS, MISSING_POSITION, WIDE_LIMIT, WIDE_SCALE
from .column.lanes import cached_kernel, f32_bits, iota, positions_buffer
from .column.lanes import prefix_sum, wide_pack, wide_unpack
from .column.types import Agg, Dtype, NA, Value
from .column.store import Column
from .column.combine import concat_columns, unify_levels
from .column.frame import DataFrame, Index, Series

export { MAX_ROWS, MISSING_POSITION, WIDE_LIMIT, WIDE_SCALE }
export { cached_kernel, f32_bits, iota, positions_buffer, prefix_sum }
export { wide_pack, wide_unpack }
export { Dtype, Value, NA, Agg }
export { Column, concat_columns, unify_levels }
export { Index, Series, DataFrame }

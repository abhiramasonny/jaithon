#: Columnar data frames for Jaithon — this language's answer to pandas.
#:
#: The names, argument order and defaults follow pandas, so a program reads the
#: same in both. Everything is re-exported here, which is the
#: `import jaiframe as pd` entry point; the submodules are importable directly
#: when only one is wanted.
#:
#: A column is one contiguous float32 device buffer per lane, the way jaicv's
#: `Mat` is, so a whole column is one allocation and an operation over it is
#: one dispatch. `Int64` and `Datetime` carry a second lane and are exact to 48
#: bits; text is dictionary-encoded, which is what makes a string comparison an
#: integer compare and a string key hashable. `jaiframe` needs a Metal device.
#:
#: ```
#: import jaiframe as pd
#:
#: let frame = pd.DataFrame(
#:     ["city", "high"],
#:     [pd.Column.from_strs(["Austin", "Dallas"]), pd.Column.from_floats([38.0, 41.0])]
#: )
#: defer { frame.free() }
#: frame.shape()                 # (2, 2)
#: frame.column("high").get(1)   # 41.0
#: ```
import std.gpu as gpu

from .column import Agg, Column, DataFrame, Dtype, Index, Series, Value
from .column import NA, MAX_ROWS, MISSING_POSITION, WIDE_LIMIT, WIDE_SCALE
from .column import cached_kernel, concat_columns, f32_bits, iota, positions_buffer
from .column import prefix_sum, unify_levels, wide_pack, wide_unpack

export { Dtype, Value, NA, Agg }
export { Column, Index, Series, DataFrame }
export { concat_columns, unify_levels, iota, positions_buffer, prefix_sum }
export { cached_kernel, f32_bits, wide_pack, wide_unpack }
export { MAX_ROWS, MISSING_POSITION, WIDE_SCALE, WIDE_LIMIT }

#: The package version, which the manifest also carries.
pub let VERSION = "0.1.0"

#: Whether the machine has a GPU backend that jaiframe can use.
pub fn is_available() -> bool { return gpu.is_available() }

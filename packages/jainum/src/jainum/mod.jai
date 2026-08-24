#: N-dimensional arrays on the GPU — this language's answer to numpy.
#:
#: An `NDArray` is a strided window onto one float32 device buffer, so a slice,
#: a transpose and a flip alias their parent instead of copying it. The dtype
#: is a semantic tag over that storage — it governs rounding, saturation and
#: result type, not element width — which is jaicv's depth model applied to
#: n dimensions.
#:
#: ```
#: import jainum as nn
#:
#: let a = nn.NDArray([2, 3])
#: defer { a.free() }
#: a.upload([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
#: print(a[0, 0..2])
#: print((a * 2.0).to_list())
#: ```
#:
#: Indexing takes an int, a `Slice`, a `range`, `NEWAXIS` or `ELLIPSIS` —
#: never `a[1:3]`, which the VM reserves for lists, strings and tuples. See
#: `array.jai` for why.
#:
#: The modules layer strictly: `array` is the root, everything else imports
#: downward from it, and each one adds its exports below as it lands.
import std.gpu as gpu

from .array import BOOL, COMPLEX64, FLOAT32, FLOAT64, INT32, INT64
from .array import CAST_BOOL, CAST_NONE, CAST_ROUND, GROUP, MAX_EXACT_INT
from .array import OP_ADD, OP_DIV, OP_MUL, OP_SUB
from .array import ELLIPSIS, NEWAXIS, IndexToken, NDArray, PRELUDE, Slice, sl
from .array import binary_arith, broadcast_pair, broadcast_shapes, broadcast_strides
from .array import contiguous_strides, copy_into, diagonal_view, dtype_is_bool
from .array import dtype_is_complex, dtype_is_integer, dtype_lanes, dtype_max
from .array import dtype_min, dtype_name, element_count, exclusive_scan
from .array import f32_bits, f32_from_bits, fill, gather_flat, is_contiguous
from .array import kernel_of, layout_buffer, normalize_axes, normalize_axis
from .array import promote, require_real, scatter_flat

export { NDArray, Slice, sl, NEWAXIS, ELLIPSIS, IndexToken }
export { BOOL, INT32, INT64, FLOAT32, FLOAT64, COMPLEX64, MAX_EXACT_INT }
export { dtype_name, dtype_lanes, dtype_is_bool, dtype_is_complex, dtype_is_integer }
export { dtype_min, dtype_max, promote, require_real }
export { element_count, normalize_axis, normalize_axes, contiguous_strides, is_contiguous }
export { broadcast_shapes, broadcast_strides, broadcast_pair, diagonal_view }
export { copy_into, fill, gather_flat, scatter_flat, exclusive_scan, binary_arith }
export { PRELUDE, GROUP, kernel_of, layout_buffer, f32_bits, f32_from_bits }
export { OP_ADD, OP_SUB, OP_MUL, OP_DIV, CAST_NONE, CAST_ROUND, CAST_BOOL }

#: The package version, which `jaithon.package.json` also carries.
pub let VERSION = "0.1.0"

#: Whether the machine has a GPU backend that jainum can use.
pub fn is_available() -> bool { return gpu.is_available() }

#: Name of the active GPU, or `"none"` when none exists.
pub fn device_name() -> str { return gpu.device_name() }

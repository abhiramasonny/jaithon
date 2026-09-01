#: `NDArray` — the strided n-dimensional array every other jainum module reads
#: and writes, and the Metal prelude they all compile against.
#:
#: Storage is one float32 device buffer per allocation. An `NDArray` is a window
#: onto it: an element offset and one stride per axis, so a slice, a transpose,
#: a flip and a diagonal all alias their parent and writing through one is
#: visible in the other. Strides are counted in float slots and may be negative,
#: which is what lets `a[::-1]` be a view rather than a copy.
#:
#: The declared dtype (`BOOL`, `INT32`, `COMPLEX64`, ...) governs rounding,
#: saturation and result type, not the width of the storage — jaicv's `Mat`
#: depth model, for the same reason. Every integer up to 2^24 is exact in
#: float32, `FLOAT64` is a source-compatible alias for float32 rather than a
#: wider element, and `COMPLEX64` is interleaved re/im with two float slots per
#: element, the way jaicv spells `CV_32FC2`.
#:
#: Indexing does not use slice syntax, and this will surprise everyone who
#: comes from numpy. `a[1:3]` reaches `OP_GET_SLICE` in the VM, which throws
#: for anything that is not a list, str or tuple; the checker's "a user type
#: needs `__getitem__`" is not backed by the runtime. So an index is an int, a
#: `Slice`, a first-class `range`, `NEWAXIS`, `ELLIPSIS`, or several of those
#: separated by commas — which the VM hands over as one tuple.
#:
#: ```
#: let a = NDArray([4, 5])
#: a.upload([float(index) for index in 0..20])
#: a[1]                              #: a view of shape [5]
#: a[1, 0..3]                        #: row 1, columns 0 1 2
#: a[ELLIPSIS, sl(null, null, -1)]   #: the last axis reversed, still a view
#: ```
#:
#: A `range` is the nicest spelling where the bounds are known and positive;
#: `sl` covers the open ends and the negative steps a `range` cannot write.
#:
#: This module is the root of jainum's import DAG: it may not import any other
#: jainum module, and everything else imports it.
from .array.bits import f32_bits, f32_from_bits
from .array.dtype import BOOL, COMPLEX64, FLOAT32, FLOAT64, INT32, INT64
from .array.dtype import CAST_BOOL, CAST_NONE, CAST_ROUND, GROUP, MAX_EXACT_INT
from .array.dtype import OP_ADD, OP_DIV, OP_MUL, OP_SUB
from .array.dtype import dtype_is_bool, dtype_is_complex, dtype_is_integer
from .array.dtype import dtype_lanes, dtype_max, dtype_min, dtype_name, promote
from .array.metal import PRELUDE, kernel_of
from .array.ndarray import NDArray, binary_arith, broadcast_pair, copy_into
from .array.ndarray import diagonal_view, exclusive_scan, fill, gather_flat
from .array.ndarray import is_trailing, layout_buffer, require_real, scatter_flat
from .array.shape import broadcast_shapes, broadcast_strides, contiguous_strides
from .array.shape import element_count, is_contiguous, normalize_axes, normalize_axis
from .array.slicing import ELLIPSIS, IndexToken, NEWAXIS, Slice, sl

export { BOOL, COMPLEX64, FLOAT32, FLOAT64, INT32, INT64 }
export { CAST_BOOL, CAST_NONE, CAST_ROUND, GROUP, MAX_EXACT_INT }
export { OP_ADD, OP_DIV, OP_MUL, OP_SUB }
export { ELLIPSIS, IndexToken, NDArray, NEWAXIS, PRELUDE, Slice, sl }
export { binary_arith, broadcast_pair, broadcast_shapes, broadcast_strides }
export { contiguous_strides, copy_into, diagonal_view, dtype_is_bool }
export { dtype_is_complex, dtype_is_integer, dtype_lanes, dtype_max }
export { dtype_min, dtype_name, element_count, exclusive_scan }
export { f32_bits, f32_from_bits, fill, gather_flat, is_contiguous, is_trailing }
export { kernel_of, layout_buffer, normalize_axes, normalize_axis }
export { promote, require_real, scatter_flat }

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
#: let grid = nn.reshape(nn.arange(12.0), [3, 4])
#: defer { grid.free() }
#:
#: let block = grid[0..2, 1..3]                      #: a view, not a copy
#: let weights = nn.array([1.0, 10.0, 100.0, 1000.0])
#: let weighted = nn.multiply(grid, weights)         #: broadcast along the last axis
#: let totals = nn.sum(weighted, axis: 1)            #: one value per row
#: print(totals.to_list())
#:
#: let a = nn.array([[3.0, 1.0], [1.0, 2.0]])
#: let b = nn.array([9.0, 8.0])
#: let x = nn.linalg.solve(a, b)
#: print(x.to_list())                                #: [2.0, 3.0]
#: ```
#:
#: Indexing takes an int, a `Slice`, a `range`, `NEWAXIS` or `ELLIPSIS` —
#: never `a[1:3]`, which the VM reserves for lists, strings and tuples. See
#: `array.jai` for why.
#:
#: `linalg`, `random` and `fft` stay namespaces, exactly as numpy keeps them:
#: `nn.linalg.solve`, `nn.random.default_rng`, `nn.fft.rfft`. Those three names
#: belong to their fields rather than to the array library, and `fft` is also
#: the name of a function inside its own module, which a flat surface has no
#: room for. Everything else is flat.
#:
#: Every module stays independently importable — `from jainum.reduce import
#: argmax` keeps working — so this facade is for callers who want the package
#: rather than one file. It is also the contract each module owes the package:
#: the names below are what a caller reaches through `import jainum`, and a
#: module that renames one of them breaks it.
#:
#: Nothing here has a body longer than a delegation. The modules layer
#: strictly: `array` is the root, everything else imports downward from it, and
#: an import cycle is a compile error.
import std.gpu as gpu

import .fft as fft
import .linalg as linalg
import .random as random

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

from .create import arange, array, asarray, diag, diagflat, empty, empty_like, eye
from .create import from_bytes, from_list, frombuffer, fromfunction, full, full_like
from .create import geomspace, identity, indices, linspace, linspace_step, logspace
from .create import meshgrid, ones, ones_like, tri, tril, triu, zeros, zeros_like

from .ufunc import FLOAT32_MAX, KIND_BOOL, KIND_FLOAT, KIND_INT, KIND_KEEP
from .ufunc import absolute, add, angle, arccos, arccosh, arcsin, arcsinh, arctan
from .ufunc import arctan2, arctanh, bitwise_and, bitwise_invert, bitwise_or
from .ufunc import bitwise_xor, cbrt, ceil, clip, copysign, cos, cosh, deg2rad
from .ufunc import degrees, divide, equal, exp, exp2, expm1, float_power, floor
from .ufunc import floor_divide, fmax, fmin, fmod, greater, greater_equal, hypot
from .ufunc import invert, isclose, isfinite, isinf, isnan, left_shift, less
from .ufunc import less_equal, log, log10, log1p, log2, logaddexp, logical_and
from .ufunc import logical_not, logical_or, logical_xor, maximum, minimum, mod
from .ufunc import multiply, nan_to_num, negative, not_equal, positive, power
from .ufunc import rad2deg, radians, reciprocal, remainder, right_shift, rint
from .ufunc import round, sign, sin, sinh, sqrt, square, subtract, tan, tanh
from .ufunc import true_divide, trunc, where

from .reduce import all, allclose, amax, amin, any, argmax, argmin, array_equal
from .reduce import count_nonzero, cummax, cummin, cumprod, cumsum, max, mean, min
from .reduce import nanargmax, nanargmin, nanmax, nanmean, nanmin, nanstd, nansum
from .reduce import nanvar, prod, ptp, std, sum, trace, variance

from .manipulate import PAD_EDGE, PAD_REFLECT, PAD_WRAP
from .manipulate import array_split, atleast_1d, atleast_2d, atleast_3d, diff
from .manipulate import broadcast_arrays, broadcast_to, column_stack, concatenate
from .manipulate import diagonal, dsplit, dstack, expand_dims, flatten, flip, fliplr
from .manipulate import flipud, hsplit, hstack, moveaxis, pad, permute_dims, ravel
from .manipulate import repeat, reshape, resize, roll, rollaxis, rot90, split
from .manipulate import squeeze, stack, swapaxes, tile, transpose, vsplit, vstack

from .sorting import HOST_SORT_LIMIT, UniqueResult, argpartition, argsort, in1d
from .sorting import intersect1d, isin, lexsort, partition, searchsorted, setdiff1d
from .sorting import setxor1d, sort, union1d, unique

from .index import MODE_CLIP, MODE_RAISE, MODE_WRAP, argwhere, choose, compress
from .index import extract, fancy_index, fill_diagonal, flatnonzero, index_exp, ix_
from .index import mask_select, nonzero, place, put, put_along_axis, putmask
from .index import repeat_by_counts, select, take, take_along_axis

from .matmul import cross, dot, einsum, inner, kron, matmul, matrix_power, outer
from .matmul import tensordot, vdot

from .fft import conj, conjugate

export { NDArray, Slice, sl, NEWAXIS, ELLIPSIS, IndexToken }
export { BOOL, INT32, INT64, FLOAT32, FLOAT64, COMPLEX64, MAX_EXACT_INT }
export { dtype_name, dtype_lanes, dtype_is_bool, dtype_is_complex, dtype_is_integer }
export { dtype_min, dtype_max, promote, require_real }
export { element_count, normalize_axis, normalize_axes, contiguous_strides, is_contiguous }
export { broadcast_shapes, broadcast_strides, broadcast_pair, diagonal_view }
export { copy_into, fill, gather_flat, scatter_flat, exclusive_scan, binary_arith }
export { PRELUDE, GROUP, kernel_of, layout_buffer, f32_bits, f32_from_bits }
export { OP_ADD, OP_SUB, OP_MUL, OP_DIV, CAST_NONE, CAST_ROUND, CAST_BOOL }

export { empty, zeros, ones, full, empty_like, zeros_like, ones_like, full_like }
export { arange, linspace, linspace_step, logspace, geomspace }
export { eye, identity, tri, tril, triu, diag, diagflat }
export { array, asarray, from_list, from_bytes, frombuffer }
export { meshgrid, indices, fromfunction }

export { add, subtract, multiply, divide, true_divide, negative, positive }
export { absolute, sign, reciprocal, sqrt, cbrt, square }
export { exp, exp2, expm1, log, log2, log10, log1p }
export { sin, cos, tan, arcsin, arccos, arctan, arctan2 }
export { sinh, cosh, tanh, arcsinh, arccosh, arctanh }
export { deg2rad, rad2deg, radians, degrees, hypot, logaddexp, copysign }
export { floor, ceil, trunc, rint, round, clip, where }
export { isnan, isinf, isfinite, isclose, nan_to_num, angle, FLOAT32_MAX }
export { power, float_power, remainder, mod, fmod, floor_divide }
export { maximum, minimum, fmax, fmin }
export { greater, greater_equal, less, less_equal, equal, not_equal }
export { logical_and, logical_or, logical_xor, logical_not }
export { bitwise_and, bitwise_or, bitwise_xor, bitwise_invert, invert }
export { left_shift, right_shift }
export { KIND_KEEP, KIND_FLOAT, KIND_BOOL, KIND_INT }

export { sum, prod, amin, amax, min, max, any, all, count_nonzero, ptp }
export { mean, variance, std, trace }
export { nansum, nanmin, nanmax, nanmean, nanvar, nanstd }
export { argmin, argmax, nanargmin, nanargmax }
export { cumsum, cumprod, cummax, cummin }
export { allclose, array_equal }

export { reshape, ravel, flatten, resize, squeeze, expand_dims }
export { transpose, permute_dims, swapaxes, moveaxis, rollaxis }
export { atleast_1d, atleast_2d, atleast_3d, broadcast_to, broadcast_arrays, diff }
export { diagonal, concatenate, stack, vstack, hstack, dstack, column_stack }
export { split, array_split, vsplit, hsplit, dsplit }
export { tile, repeat, flip, fliplr, flipud, roll, rot90 }
export { pad, PAD_EDGE, PAD_WRAP, PAD_REFLECT }

export { sort, argsort, partition, argpartition, searchsorted, lexsort }
export { unique, UniqueResult, isin, in1d, HOST_SORT_LIMIT }
export { union1d, intersect1d, setdiff1d, setxor1d }

export { take, take_along_axis, put, put_along_axis, place, putmask, mask_select }
export { nonzero, flatnonzero, argwhere, extract, compress, choose, select }
export { repeat_by_counts, fill_diagonal, index_exp, ix_, fancy_index }
export { MODE_RAISE, MODE_WRAP, MODE_CLIP }

export { matmul, dot, vdot, inner, outer, tensordot, kron, cross }
export { einsum, matrix_power }

export { conj, conjugate }

export { fft, linalg, random }

#: The package version, which `jaithon.package.json` also carries.
pub let VERSION = "0.1.0"

#: Whether the machine has a GPU backend that jainum can use.
pub fn is_available() -> bool { return gpu.is_available() }

#: Name of the active GPU, or `"none"` when none exists.
pub fn device_name() -> str { return gpu.device_name() }

#: Number of Metal devices.
pub fn device_count() -> int { return gpu.device_count() }

#: Select the Metal device later allocations live on. Must be called before
#: the first array exists, and raises once one does.
pub fn set_device(index: int) -> void { gpu.set_device(index) }

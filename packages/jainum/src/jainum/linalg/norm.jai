#: Vector and matrix norms, the one entry point here that stays on the device.
import std.math
from ..array import NDArray, element_count, normalize_axis, normalize_axes, require_real
from .common import _emit, _lead, _slice_matrix
from .reduction import RED_COUNT, RED_MAX_ABS, RED_MIN_ABS, RED_SUM_ABS, RED_SUM_POW, RED_SUM_SQ
from .reduction import _kept_shape, _move_axes_last, _reduce_axes
from .svd import _svd_host

pub fn _as_number(value: any, what: str) -> float {
    if value is float { return value }
    if value is int { return float(value) }
    throw TypeError(f"{what} expects a number, got {type_of(value)}")
}

#: The two axes a matrix norm spans, as a normalised ordered pair.
fn _axis_pair(axis: any, rank: int) -> (int, int) {
    var items: list[any] = []
    if axis is tuple { items = [item for item in axis] } elif axis is list { items = axis }
    if items.len() != 2 {
        throw ValueError("a matrix norm takes exactly two axes")
    }
    let pair = normalize_axes(
        [int(_as_number(items[0], "axis")), int(_as_number(items[1], "axis"))],
        rank
    )
    return (pair[0], pair[1])
}

fn _vector_norm(a: NDArray, p: float, axes: list[int], kept: list[int]) -> NDArray {
    if p == 2.0 { return _reduce_axes(a, axes, RED_SUM_SQ, 0.0, kept) }
    if p == 1.0 { return _reduce_axes(a, axes, RED_SUM_ABS, 0.0, kept) }
    if p == math.INF { return _reduce_axes(a, axes, RED_MAX_ABS, 0.0, kept) }
    if p == -math.INF { return _reduce_axes(a, axes, RED_MIN_ABS, 0.0, kept) }
    if p == 0.0 { return _reduce_axes(a, axes, RED_COUNT, 0.0, kept) }
    return _reduce_axes(a, axes, RED_SUM_POW, p, kept)
}

#: The 1, -1, inf and -inf matrix norms, which are a sum along one axis and
#: then an extreme along the other.
#:
#: Two dispatch pairs rather than a kernel of their own: the intermediate is
#: one number per row or column, which is small enough that a second pass over
#: it costs less than a second pipeline would.
fn _extreme_matrix_norm(a: NDArray, p: float, r: int, c: int, kept: list[int]) -> NDArray {
    let over_columns = p == 1.0 or p == -1.0
    let summed = if over_columns { r } else { c }
    let folded = if over_columns { c } else { r }
    let middle = _kept_shape(a.shape, [summed], false)
    let sums = _reduce_axes(a, [summed], RED_SUM_ABS, 0.0, middle)
    defer { sums.free() }
    let shifted = if folded > summed { folded - 1 } else { folded }
    let take = if p > 0.0 { RED_MAX_ABS } else { RED_MIN_ABS }
    return _reduce_axes(sums, [shifted], take, 0.0, kept)
}

#: Vector and matrix norms, as numpy spells them.
#:
#: `axis` selects between the two: `null` folds everything (a 2-norm when
#: `ord` is `null`, otherwise the matrix reading for a 2-d array and the
#: vector reading for a 1-d one), an int is a vector norm along that axis, and
#: a pair of ints is a matrix norm over that plane.
#:
#: Vector orders are `1`, `2`, `math.INF`, `-math.INF`, `0` — which counts
#: non-zeros — and any other `p`, taken as `(sum |x|^p)^(1/p)`. Matrix orders
#: are `"fro"`, `"nuc"`, `1`, `-1`, `2`, `-2`, `math.INF` and `-math.INF`.
#: `"nuc"` and `+/-2` go through the singular values and therefore through the
#: host; every other order is two dispatches.
pub fn norm(a: NDArray, ord: any = null, axis: any = null, keepdims: bool = false) -> NDArray {
    require_real(a, "norm")
    let rank = a.ndim()
    if axis is null {
        if ord is null {
            let every = [k for k in 0..rank]
            return _vector_norm(a, 2.0, every, _kept_shape(a.shape, every, keepdims))
        }
        if rank == 1 { return norm(a, ord, 0, keepdims) }
        if rank == 2 { return norm(a, ord, (0, 1), keepdims) }
        throw ValueError(f"norm needs an axis to read {ord} off {rank} dimensions")
    }
    if axis is int {
        let axes = [normalize_axis(axis, rank)]
        let p = if ord is null { 2.0 } else { _as_number(ord, "a vector order") }
        return _vector_norm(a, p, axes, _kept_shape(a.shape, axes, keepdims))
    }

    let (r, c) = _axis_pair(axis, rank)
    let kept = _kept_shape(a.shape, [r, c], keepdims)
    if ord is null { return _reduce_axes(a, [r, c], RED_SUM_SQ, 0.0, kept) }
    if ord is str {
        if ord == "fro" { return _reduce_axes(a, [r, c], RED_SUM_SQ, 0.0, kept) }
        if ord == "nuc" { return _singular_norm(a, r, c, 2, kept) }
        throw ValueError(f"norm does not know the matrix order {ord}")
    }
    let p = _as_number(ord, "a matrix order")
    if p == 2.0 { return _singular_norm(a, r, c, 0, kept) }
    if p == -2.0 { return _singular_norm(a, r, c, 1, kept) }
    if p == 1.0 or p == -1.0 or p == math.INF or p == -math.INF {
        return _extreme_matrix_norm(a, p, r, c, kept)
    }
    throw ValueError(f"norm does not know the matrix order {ord}")
}

#: The matrix norms that go through the singular values: `mode` 0 is the
#: largest, 1 the smallest, 2 their sum.
fn _singular_norm(a: NDArray, r: int, c: int, mode: int, kept: list[int]) -> NDArray {
    let view = _move_axes_last(a, [r, c])
    let rank = view.ndim()
    let m = view.shape[rank - 2]
    let n = view.shape[rank - 1]
    let k = min(m, n)
    let values = view.to_list()
    let batch = _lead(view.shape, 2)
    var out: list[float] = []
    for index in 0..element_count(batch) {
        let matrix = _slice_matrix(values, index * m * n, m * n)
        let (_u, s, _vt) = _svd_host(matrix, m, n, false, false)
        if mode == 2 {
            var total = 0.0
            for value in s { total += value }
            out.push(total)
        } elif mode == 1 {
            out.push(if k > 0 { s[k - 1] } else { 0.0 })
        } else {
            out.push(if k > 0 { s[0] } else { 0.0 })
        }
    }
    return _emit(out, kept)
}

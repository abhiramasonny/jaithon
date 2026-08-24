#: Factorizations, solvers and norms — the decompositions everything else in
#: numerical work is written on top of.
#:
#: ```
#: let (q, r) = qr(a)
#: let x = solve(a, b)
#: let (u, s, vh) = svd(a)
#: print(norm(a, "fro").item())
#: ```
#:
#: **Everything but `norm` runs on downloaded host values.** One `to_list` on
#: the way in, one `upload` on the way out, whatever the batch holds. That is
#: the same call `jaicv/core/linalg.jai` makes, and for the same reason: it is
#: handed three-by-threes, nine-by-nines and Jacobians a few dozen wide, sizes
#: where a kernel launch per panel costs more than the arithmetic inside the
#: panel. A Householder step on a 64-wide matrix is four thousand multiplies,
#: which an M2 Max finishes in less time than it takes to encode the dispatch
#: that would have done it. The reasoning holds up to a few hundred on a side.
#:
#: **Past that the trailing-matrix update is the whole cost**, and it is a
#: GEMM: an `n` by `n` LU is `n^3/3` operations of which all but `O(n^2)` live
#: in that one update. `blocked_panel` is the hook that switches to it — see
#: its note for the crossover and what still has to be wired.
#:
#: **`norm` is the exception and is a kernel**, because it is the one entry
#: point here that walks the whole array rather than one small matrix at a
#: time: a Frobenius norm of a ten-million-element array must not become ten
#: million host iterations, and a batched row-sum is a reduction like any
#: other. Two dispatches, partials then a fold.
#:
#: Stacked inputs loop over the leading axes: an `(..., m, n)` array is a
#: batch of `m` by `n` matrices and every entry point here takes one. The
#: batch axes of a solver's two arguments broadcast against each other.
#:
#: `eig` is real only. A real matrix can have complex conjugate eigenvalues,
#: and there is no `COMPLEX64` answer this can give without carrying complex
#: arithmetic through the whole Schur iteration, so it raises with the pair
#: named rather than returning the real part and calling it an eigenvalue.
#: `eigh` has no such case — a symmetric matrix's spectrum is real.
#:
#: A singular matrix raises `ValueError`, as `jaicv`'s `inverse` does; numpy's
#: `LinAlgError` has no counterpart in this language's error hierarchy.
import std.math
from std.gpu import Buffer
from .array import FLOAT32, GROUP, INT32, NDArray, PRELUDE
from .array import broadcast_shapes, element_count, f32_bits, fill, is_contiguous
from .array import kernel_of, layout_buffer, normalize_axis, normalize_axes, require_real

#: Columns a blocked factorization takes per panel.
#:
#: 64 fits a panel's `L` and its pivots in a few hundred kilobytes, which is
#: what keeps the panel factorization itself in cache while the trailing
#: update streams past it.
pub const BLOCKED_PANEL = 64

#: Side length at which a factorization stops sweeping and starts panelling.
#:
#: 256. Below it the unblocked sweep wins outright: a 128-wide LU is two
#: hundred thousand multiply-adds, and cutting it into two panels buys back
#: less locality than the extra triangular solve and the extra pass over the
#: trailing block cost. At 256 the trailing update is already 90% of the work
#: and grows as the cube, so that is where the panel starts paying.
pub const BLOCKED_FROM = 256

#: The panel width a factorization of an `n`-wide matrix should take, or 0 for
#: the unblocked sweep.
#:
#: This is the hook and the only place the decision is made. The blocked LU it
#: selects is written — `_lu_factor_host` panels, solves `L11 U12 = A12`, and
#: subtracts `L21 U12` from the trailing block — but that subtraction is still
#: `_gemm_subtract`, a host triple loop. Replacing that one call with
#: `matmul`'s GEMM is what moves the cube-order work onto the device; nothing
#: else about the factorization changes, which is the point of writing the
#: split now rather than later.
#:
#: QR wants the same treatment through the compact WY representation, at the
#: same crossover. It is unblocked today and its trailing update is the same
#: shape of GEMM.
pub fn blocked_panel(n: int) -> int {
    if n < BLOCKED_FROM { return 0 }
    return BLOCKED_PANEL
}

#: Reduction selector: running total of `|x|`.
const RED_SUM_ABS = 0
#: Reduction selector: running total of `x*x`.
const RED_SUM_SQ = 1
#: Reduction selector: largest `|x|`.
const RED_MAX_ABS = 2
#: Reduction selector: smallest `|x|`.
const RED_MIN_ABS = 3
#: Reduction selector: running total of `|x|^p`.
const RED_SUM_POW = 4
#: Reduction selector: how many elements are non-zero.
const RED_COUNT = 5

#: Chunks one output row's reduction is cut into at most.
#:
#: A single group per row leaves 256 threads on a whole-array norm, which is a
#: sixteenth of this device. 64 chunks puts 16384 threads on it and still
#: leaves the fold pass a run of 64 floats per row to walk.
const NORM_CHUNKS = 64

let SOURCE = PRELUDE + """
/* Norm reductions. `op` is one of the RED_* selectors and every kernel below
 * carries it rather than being compiled six times: the branch is uniform
 * across the grid, so it costs one predicted jump per thread and saves five
 * pipelines that would otherwise be built on first use. */
inline float nl_seed(uint op) {
    if (op == 3u) return INFINITY;
    return 0.0f;
}

inline float nl_join(float a, float b, uint op) {
    if (op == 2u) return max(a, b);
    if (op == 3u) return min(a, b);
    return a + b;
}

inline float nl_take(float acc, float value, uint op, float power) {
    float m = fabs(value);
    if (op == 0u) return acc + m;
    if (op == 1u) return acc + value * value;
    if (op == 2u) return max(acc, m);
    if (op == 3u) return min(acc, m);
    if (op == 4u) return acc + pow(m, power);
    return acc + (value != 0.0f ? 1.0f : 0.0f);
}

/* Four at a time. The adds are paired rather than chained so the two halves
 * issue independently; the accumulator is still serial across iterations,
 * which is where the dependency that matters lives. */
inline float nl_take4(float acc, float4 value, uint op, float power) {
    float4 m = fabs(value);
    if (op == 0u) return acc + ((m.x + m.y) + (m.z + m.w));
    if (op == 1u) return acc + dot(value, value);
    if (op == 2u) return max(acc, max(max(m.x, m.y), max(m.z, m.w)));
    if (op == 3u) return min(acc, min(min(m.x, m.y), min(m.z, m.w)));
    if (op == 4u) {
        float4 raised = pow(m, float4(power));
        return acc + ((raised.x + raised.y) + (raised.z + raised.w));
    }
    float4 hit = select(float4(0.0f), float4(1.0f), value != 0.0f);
    return acc + ((hit.x + hit.y) + (hit.z + hit.w));
}

/* Fold `value` across a whole threadgroup under `op`. `scratch` holds one
 * float per simdgroup; barriered at both ends so it can be reused. */
inline float nl_group_join(
    float value,
    threadgroup float* scratch,
    uint lane,
    uint warp,
    uint op
) {
    float folded = op == 2u ? simd_max(value) : (op == 3u ? simd_min(value) : simd_sum(value));
    if (lane == 0u) scratch[warp] = folded;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = scratch[0];
    for (uint j = 1u; j < ND_WARPS; j++) total = nl_join(total, scratch[j], op);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

/* Pass one: `chunks` groups per output row, each folding its stride of the
 * reduced run into one partial.
 *
 * Three read paths, narrowest first. `quads` non-zero means the run is dense
 * and both its base and its length are multiples of four, so a float4 load is
 * aligned; `packed` alone means dense but not divisible; neither means the
 * general strided walk through `nd_offset`, which is what a reduction over a
 * transposed view or a middle axis takes. */
kernel void nl_reduce_partial(device float* partials [[buffer(0)]],
                              device const float* src [[buffer(1)]],
                              device const float* meta [[buffer(2)]],
                              constant uint& inner [[buffer(3)]],
                              constant uint& chunks [[buffer(4)]],
                              constant uint& op [[buffer(5)]],
                              constant uint& power_bits [[buffer(6)]],
                              constant uint& base [[buffer(7)]],
                              constant uint& packed [[buffer(8)]],
                              constant uint& quads [[buffer(9)]],
                              uint group [[threadgroup_position_in_grid]],
                              uint lid [[thread_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[ND_WARPS];
    /* One divide and a subtract, not a divide and a modulo: the device has no
     * integer divide and pays the whole reciprocal sequence for each. */
    uint row = group / chunks;
    uint chunk = group - row * chunks;
    float power = as_type<float>(power_bits);
    float acc = nl_seed(op);
    uint start = chunk * ND_GROUP + lid;
    uint step = chunks * ND_GROUP;
    if (quads != 0u) {
        device const float4* run = (device const float4*)(src + base + row * inner);
        for (uint at = start; at < quads; at += step) acc = nl_take4(acc, run[at], op, power);
    } else if (packed != 0u) {
        uint from = base + row * inner;
        for (uint at = start; at < inner; at += step) {
            acc = nl_take(acc, src[from + at], op, power);
        }
    } else {
        for (uint at = start; at < inner; at += step) {
            acc = nl_take(acc, src[nd_offset(meta, 0u, row * inner + at)], op, power);
        }
    }
    float total = nl_group_join(acc, scratch, lane, warp, op);
    if (lid == 0u) partials[group] = total;
}

/* Pass two: one thread per output, folding that row's `chunks` partials and
 * applying whatever root the norm ends in. `finish` is 1 for a square root
 * and 2 for the general `1/p` power. */
kernel void nl_reduce_fold(device float* dst [[buffer(0)]],
                           device const float* partials [[buffer(1)]],
                           constant uint& outer [[buffer(2)]],
                           constant uint& chunks [[buffer(3)]],
                           constant uint& op [[buffer(4)]],
                           constant uint& finish [[buffer(5)]],
                           constant uint& root_bits [[buffer(6)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= outer) return;
    uint from = id * chunks;
    float acc = partials[from];
    for (uint k = 1u; k < chunks; k++) acc = nl_join(acc, partials[from + k], op);
    if (finish == 1u) acc = sqrt(acc);
    if (finish == 2u) acc = pow(acc, as_type<float>(root_bits));
    dst[id] = acc;
}
"""

#: A view with `axes` moved to the end, keeping the others in their order.
#:
#: A pure stride permutation, so nothing is copied: the reduction below wants
#: the reduced run last so a linear index splits into a row and a position
#: with one division.
fn _move_axes_last(a: NDArray, axes: list[int]) -> NDArray {
    var taken = [false for _axis in a.shape]
    for axis in axes { taken[axis] = true }
    var shape: list[int] = []
    var strides: list[int] = []
    for axis in 0..a.ndim() {
        if taken[axis] { continue }
        shape.push(a.shape[axis])
        strides.push(a.strides[axis])
    }
    for axis in axes {
        shape.push(a.shape[axis])
        strides.push(a.strides[axis])
    }
    return a.view(shape, strides, a.offset)
}

#: `shape` with `axes` dropped, or set to one when `keepdims`.
fn _kept_shape(shape: list[int], axes: list[int], keepdims: bool) -> list[int] {
    var taken = [false for _axis in shape]
    for axis in axes { taken[axis] = true }
    var out: list[int] = []
    for axis in 0..shape.len() {
        if not taken[axis] {
            out.push(shape[axis])
        } elif keepdims {
            out.push(1)
        }
    }
    return out
}

#: The root a reduction ends in: none, a square root, or `1/p`.
fn _finish_of(op: int) -> int {
    if op == RED_SUM_SQ { return 1 }
    if op == RED_SUM_POW { return 2 }
    return 0
}

#: Fold `axes` of `a` under `op`, into a fresh array shaped `out_shape`.
#:
#: `out_shape` is passed rather than derived because a matrix norm composes
#: two of these and only the second one knows where the `keepdims` axes go.
fn _reduce_axes(
    a: NDArray,
    axes: list[int],
    op: int,
    power: float,
    out_shape: list[int]
) -> NDArray {
    let view = _move_axes_last(a, axes)
    var inner = 1
    for axis in axes { inner *= a.shape[axis] }
    let out = NDArray(out_shape, FLOAT32)
    let outer = out.size()
    if outer == 0 { return out }
    if inner == 0 {
        fill(out, 0.0)
        return out
    }

    let packed = is_contiguous(view.shape, view.strides, 1)
    #: A float4 load must be sixteen-byte aligned. The run's length divides by
    #: four so every row starts a multiple of four along, and the base has to
    #: as well — a view sliced to an odd column would otherwise fault.
    let vectored = packed and inner % 4 == 0 and view.offset % 4 == 0
    let quads = if vectored { inner // 4 } else { 0 }
    let chunks = max(1, min(NORM_CHUNKS, (inner + GROUP * 4 - 1) // (GROUP * 4)))

    let partials = Buffer(outer * chunks)
    defer { partials.free() }
    let meta = layout_buffer([view])
    defer { meta.free() }
    kernel_of(SOURCE, "nl_reduce_partial").dispatch_async([partials, view.storage, meta], outer * chunks * GROUP, [inner, chunks, op, f32_bits(power), view.offset, if packed {
        1
    } else {
        0
    }, quads], GROUP)
    let root = if power != 0.0 { 1.0 / power } else { 0.0 }
    kernel_of(SOURCE, "nl_reduce_fold").dispatch_async(
        [out.storage, partials],
        outer,
        [outer, chunks, op, _finish_of(op), f32_bits(root)]
    )
    return out
}

fn _as_number(value: any, what: str) -> float {
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

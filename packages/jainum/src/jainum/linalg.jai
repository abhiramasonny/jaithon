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

#: Rounding unit of the float32 the storage actually holds.
#:
#: Every default tolerance here is built from this rather than from float64's
#: `math.EPSILON`. The host arithmetic runs in double, but the values reached
#: it through a float32 buffer, so a singular value below
#: `max(m, n) * FLOAT32_EPS * s_max` is indistinguishable from the rounding of
#: the ones above it however carefully the double arithmetic is done.
pub const FLOAT32_EPS = 1.1920928955078125e-7

#: How small a pivot has to be, against the largest one in the same `U`,
#: before a solve calls the matrix singular.
#:
#: An exactly singular float32 matrix does not leave a clean zero: eliminating
#: in double leaves the dependent pivot around 1e-16 of the largest. 1e-12
#: sits well above that noise and well below anything a merely ill-conditioned
#: float32 matrix produces.
const SINGULAR_RATIO = 1e-12

#: Off-diagonal weight, relative to the diagonal, at which Jacobi stops.
const JACOBI_TOL = 1e-15

#: Sweeps `eigh` will spend before it gives up.
#:
#: Cyclic Jacobi is quadratically convergent once the off-diagonal is small,
#: and 30 sweeps is far past where any conditioned matrix stops moving; the
#: cap is there so a NaN cannot spin forever.
const JACOBI_SWEEPS = 30

#: Iterations a QR sweep spends per column before it gives up.
const QR_ITERATIONS = 60

#: A stack of matrices, flat and row-major, with the batch shape they sit in.
fn _stacked(a: NDArray, what: str) -> (list[float], list[int], int, int) {
    require_real(a, what)
    let rank = a.ndim()
    if rank < 2 {
        throw ValueError(f"{what} needs a matrix or a stack of them, got {rank} dimensions")
    }
    var lead: list[int] = []
    for axis in 0..rank - 2 { lead.push(a.shape[axis]) }
    return (a.to_list(), lead, a.shape[rank - 2], a.shape[rank - 1])
}

#: A fresh array holding `values`, which must already be in element order.
fn _emit(values: list[float], shape: list[int], dtype: int = FLOAT32) -> NDArray {
    let out = NDArray(shape, dtype)
    out.upload(values)
    return out
}

fn _require_square(m: int, n: int, what: str) -> void {
    if m != n { throw ValueError(f"{what} needs a square matrix, got {m} by {n}") }
}

#: The `rows` by `columns` leading identity, row-major.
fn _identity_block(rows: int, columns: int) -> list[float] {
    var out = [0.0 for _slot in 0..rows * columns]
    for index in 0..min(rows, columns) { out[index * columns + index] = 1.0 }
    return out
}

#: `a` read the other way round, `rows` by `columns` in.
fn _transpose_host(a: list[float], rows: int, columns: int) -> list[float] {
    var out = [0.0 for _slot in 0..rows * columns]
    for row in 0..rows {
        for column in 0..columns { out[column * rows + row] = a[row * columns + column] }
    }
    return out
}

#: `count` values of `values` starting at `at`, as a matrix of its own.
fn _slice_matrix(values: list[float], at: int, count: int) -> list[float] {
    var out = [0.0 for _slot in 0..count]
    for index in 0..count { out[index] = values[at + index] }
    return out
}

#: Append every value of `src` to `dst`.
fn _extend(dst: list[float], src: list[float]) -> void {
    for value in src { dst.push(value) }
}

#: One panel of columns factored in place with partial pivoting.
#:
#: A swap moves the whole row — the finished `L` on its left and the untouched
#: trailing columns on its right — which is what LAPACK's separate `laswp`
#: pass does after the fact. It is the same memory traffic and it keeps the
#: pivoting in one place. The eliminations stop at the panel's right edge; the
#: trailing block belongs to the blocked driver, and for the unblocked sweep
#: the panel is the whole matrix and the two coincide.
#:
#: Deliberately call-free. This is the cube-order loop of the unblocked path,
#: and the JIT declines a whole body that contains any call — one `math.fabs`
#: here would cost the factorization a factor of two.
fn _lu_panel(work: list[float], n: int, piv: list[int], first: int, width: int) -> int {
    var swaps = 0
    let stop = first + width
    for column in first..stop {
        var pivot = column
        var best = 0.0
        for row in column..n {
            let value = work[row * n + column]
            let size = if value < 0.0 { -value } else { value }
            if size > best {
                best = size
                pivot = row
            }
        }
        piv[column] = pivot
        if pivot != column {
            swaps += 1
            for index in 0..n {
                let held = work[column * n + index]
                work[column * n + index] = work[pivot * n + index]
                work[pivot * n + index] = held
            }
        }
        let diagonal = work[column * n + column]
        if diagonal == 0.0 { continue }
        for row in column + 1..n {
            let factor = work[row * n + column] / diagonal
            work[row * n + column] = factor
            if factor == 0.0 { continue }
            for index in column + 1..stop {
                work[row * n + index] -= factor * work[column * n + index]
            }
        }
    }
    return swaps
}

#: `A12 <- L11^-1 A12`: the panel's eliminations, caught up with on the block
#: to its right.
fn _panel_forward(work: list[float], n: int, first: int, width: int, start: int) -> void {
    for row in 1..width {
        for step in 0..row {
            let factor = work[(first + row) * n + first + step]
            if factor == 0.0 { continue }
            for column in start..n {
                work[(first + row) * n + column] -= factor * work[(first + step) * n + column]
            }
        }
    }
}

#: `A22 <- A22 - L21 U12`, the trailing update, as a host triple loop.
#:
#: **This is the call the blocked path exists to hand to the device.** It is
#: the only cube-order work left outside the panel, it is a plain GEMM of
#: `rows` by `width` against `width` by `cols`, and `matmul.jai`'s `matmul` is
#: what belongs here: three strided views onto a device copy of `work` and one
#: dispatch replace this entire function, and nothing else about the
#: factorization changes. That is the whole reason the split is written now
#: rather than later — and it is also why it is not wired yet, since the
#: factorization holds `work` as a host list and a panel-by-panel round trip
#: would cost more than the triple loop at every size the unblocked path
#: still covers.
fn _gemm_subtract(work: list[float], n: int, first: int, width: int, rows: int, cols: int) -> void {
    let top = first + width
    for row in 0..rows {
        let target = (top + row) * n
        for step in 0..width {
            let factor = work[target + first + step]
            if factor == 0.0 { continue }
            let source = (first + step) * n + top
            for column in 0..cols {
                work[target + top + column] -= factor * work[source + column]
            }
        }
    }
}

#: Right-looking blocked LU: panel, triangular solve, GEMM, repeat.
fn _lu_blocked(work: list[float], n: int, piv: list[int], panel: int) -> int {
    var swaps = 0
    var first = 0
    while first < n {
        let width = min(panel, n - first)
        swaps += _lu_panel(work, n, piv, first, width)
        let rest = n - first - width
        if rest > 0 {
            _panel_forward(work, n, first, width, first + width)
            _gemm_subtract(work, n, first, width, rest, rest)
        }
        first += width
    }
    return swaps
}

#: `PA = LU` by partial pivoting: the packed factors, the row each step
#: swapped with, and the sign of the permutation.
fn _lu_factor_host(values: list[float], n: int) -> (list[float], list[int], int) {
    var work = [value for value in values]
    var piv = [index for index in 0..n]
    let panel = blocked_panel(n)
    let swaps = if panel == 0 {
        _lu_panel(work, n, piv, 0, n)
    } else {
        _lu_blocked(work, n, piv, panel)
    }
    return (work, piv, if swaps % 2 == 0 { 1 } else { -1 })
}

#: The largest `|U[i,i]|`, which the singularity test is relative to.
fn _pivot_scale(lu: list[float], n: int) -> float {
    var scale = 0.0
    for index in 0..n {
        let value = lu[index * n + index]
        let size = if value < 0.0 { -value } else { value }
        if size > scale { scale = size }
    }
    return scale
}

fn _require_nonsingular(lu: list[float], n: int, what: str) -> void {
    let floor = _pivot_scale(lu, n) * SINGULAR_RATIO
    for index in 0..n {
        let value = lu[index * n + index]
        let size = if value < 0.0 { -value } else { value }
        if size <= floor {
            throw ValueError(f"{what} was handed a singular matrix: pivot {index} of {n} vanished")
        }
    }
}

#: Apply a factorization's row swaps to a right-hand side.
fn _permute_rows(x: list[float], piv: list[int], n: int, wide: int) -> void {
    for row in 0..n {
        let other = piv[row]
        if other == row { continue }
        for column in 0..wide {
            let held = x[row * wide + column]
            x[row * wide + column] = x[other * wide + column]
            x[other * wide + column] = held
        }
    }
}

#: `x <- U^-1 L^-1 P b`, with `b` held as `n` by `wide` row-major.
fn _lu_solve_host(
    lu: list[float],
    piv: list[int],
    n: int,
    b: list[float],
    wide: int,
    what: str
) -> list[float] {
    _require_nonsingular(lu, n, what)
    var x = [value for value in b]
    _permute_rows(x, piv, n, wide)
    #: The packed factors are two triangles in one array, so the solve is the
    #: general triangular pass twice — down the unit lower one, then back up
    #: the upper one — and not a substitution written out again here.
    _triangular_host(lu, n, x, wide, true, true, false)
    _triangular_host(lu, n, x, wide, false, false, false)
    return x
}

#: One triangular solve, in place on `x`.
#:
#: `forward` says which end to start from — a lower triangle read normally and
#: an upper one read transposed both start at the top — and `transposed` is
#: whether the coefficient comes from `a[step, row]` instead of `a[row, step]`,
#: which is how `a^T x = b` is solved without forming the transpose.
fn _triangular_host(
    a: list[float],
    n: int,
    x: list[float],
    wide: int,
    forward: bool,
    unit: bool,
    transposed: bool
) -> void {
    for count in 0..n {
        let row = if forward { count } else { n - 1 - count }
        let first = if forward { 0 } else { row + 1 }
        let last = if forward { row } else { n }
        for step in first..last {
            let factor = if transposed { a[step * n + row] } else { a[row * n + step] }
            if factor == 0.0 { continue }
            for column in 0..wide {
                x[row * wide + column] -= factor * x[step * wide + column]
            }
        }
        if not unit {
            let diagonal = a[row * n + row]
            for column in 0..wide { x[row * wide + column] /= diagonal }
        }
    }
}

#: The dot product of the finished parts of two rows of `L`.
#:
#: Split out of `_cholesky_host` because that body carries a `math.sqrt` and a
#: `throw` and therefore never compiles; this is the cube-order half of the
#: factorization and it is call-free.
fn _cholesky_dot(out: list[float], n: int, row: int, column: int) -> float {
    var total = 0.0
    for step in 0..column { total += out[row * n + step] * out[column * n + step] }
    return total
}

#: `A = L L^T` for a symmetric positive definite `A`, lower triangle out.
#:
#: Only the lower triangle of the input is read, as LAPACK's `potrf` with
#: `uplo='L'` does: a matrix that is not quite symmetric factors as though its
#: lower triangle were the truth rather than raising.
fn _cholesky_host(values: list[float], n: int) -> list[float] {
    var out = [0.0 for _slot in 0..n * n]
    for row in 0..n {
        for column in 0..=row {
            let total = values[row * n + column] - _cholesky_dot(out, n, row, column)
            if column < row {
                out[row * n + column] = total / out[column * n + column]
            } elif total > 0.0 {
                out[row * n + row] = math.sqrt(total)
            } else {
                throw ValueError(
                    f"this matrix is not positive definite: pivot {row} came out {total}"
                )
            }
        }
    }
    return out
}

#: `target <- (I - 2 v v^T / v^T v) target` over rows `step..rows` and columns
#: `first..cols`, with `v` living in `vs[base + step .. base + rows]`.
#:
#: Call-free, and the cube-order loop of every Householder pass in the file —
#: QR, the bidiagonalization and the Hessenberg reduction all spend their time
#: here rather than in the reflector that feeds it.
fn _reflect_from(
    target: list[float],
    rows: int,
    cols: int,
    vs: list[float],
    base: int,
    step: int,
    vnorm: float,
    first: int
) -> void {
    if vnorm == 0.0 { return }
    for column in first..cols {
        var dot = 0.0
        for row in step..rows { dot += vs[base + row] * target[row * cols + column] }
        let factor = 2.0 * dot / vnorm
        for row in step..rows { target[row * cols + column] -= factor * vs[base + row] }
    }
}

#: The same reflector applied from the right: `target <- target H`, over rows
#: `first..rows` and columns `step..cols`, with `v` indexed by column.
fn _reflect_across(
    target: list[float],
    rows: int,
    cols: int,
    vs: list[float],
    base: int,
    step: int,
    vnorm: float,
    first: int
) -> void {
    if vnorm == 0.0 { return }
    for row in first..rows {
        var dot = 0.0
        for column in step..cols { dot += vs[base + column] * target[row * cols + column] }
        let factor = 2.0 * dot / vnorm
        for column in step..cols { target[row * cols + column] -= factor * vs[base + column] }
    }
}

#: `sum x^2` over `count` values `stride` apart, starting at `at`.
fn _segment_sigma(work: list[float], at: int, stride: int, count: int) -> float {
    var sigma = 0.0
    for index in 0..count {
        let value = work[at + index * stride]
        sigma += value * value
    }
    return sigma
}

#: Write `x - alpha e1` for that segment into `vs`, returning `v^T v`.
fn _segment_vector(
    work: list[float],
    at: int,
    stride: int,
    count: int,
    alpha: float,
    vs: list[float],
    vbase: int
) -> float {
    var vnorm = 0.0
    for index in 0..count {
        let value = if index == 0 { work[at] - alpha } else { work[at + index * stride] }
        vs[vbase + index] = value
        vnorm += value * value
    }
    return vnorm
}

#: The reflector that collapses `work[step.., column]` onto `alpha e1`.
#:
#: Returns `(alpha, v^T v)` and writes `v` into `vs[base + step ..]`. The sign
#: of `alpha` is taken away from the leading entry so that the subtraction
#: that forms `v` adds magnitudes instead of cancelling them — the one place
#: in a Householder pass where a naive sign loses digits.
#:
#: The two passes over the segment live in the helpers above rather than here
#: because of the `sqrt` between them: it is a call, the JIT declines any body
#: holding one, and these two loops are the quadratic term of every
#: factorization in the file.
fn _column_reflector(
    work: list[float],
    m: int,
    n: int,
    column: int,
    step: int,
    vs: list[float],
    base: int
) -> (float, float) {
    let at = step * n + column
    let count = m - step
    let sigma = _segment_sigma(work, at, n, count)
    if sigma == 0.0 { return (0.0, 0.0) }
    let alpha = if work[at] > 0.0 { -math.sqrt(sigma) } else { math.sqrt(sigma) }
    return (alpha, _segment_vector(work, at, n, count, alpha, vs, base + step))
}

#: The same for a row segment: collapses `work[row, step..]` onto `beta e1`.
fn _row_reflector(
    work: list[float],
    n: int,
    row: int,
    step: int,
    vs: list[float],
    base: int
) -> (float, float) {
    let at = row * n + step
    let count = n - step
    let sigma = _segment_sigma(work, at, 1, count)
    if sigma == 0.0 { return (0.0, 0.0) }
    let beta = if work[at] > 0.0 { -math.sqrt(sigma) } else { math.sqrt(sigma) }
    return (beta, _segment_vector(work, at, 1, count, beta, vs, base + step))
}

#: `A = QR` by Householder: `work` becomes `R` and the reflectors stay in the
#: returned `(vs, norms)`, one length-`m` vector per step.
fn _householder_qr(work: list[float], m: int, n: int, k: int) -> (list[float], list[float]) {
    var vs = [0.0 for _slot in 0..k * m]
    var norms = [0.0 for _step in 0..k]
    for step in 0..k {
        let (alpha, vnorm) = _column_reflector(work, m, n, step, step, vs, step * m)
        norms[step] = vnorm
        work[step * n + step] = alpha
        for row in step + 1..m { work[row * n + step] = 0.0 }
        _reflect_from(work, m, n, vs, step * m, step, vnorm, step + 1)
    }
    return (vs, norms)
}

#: `Q` as `m` by `columns`, from the reflectors of a Householder pass.
#:
#: Backwards, because `Q = H_0 H_1 ... H_k` and each `H` is applied from the
#: left of what the ones after it have already built.
fn _accumulate_reflectors(
    vs: list[float],
    norms: list[float],
    m: int,
    k: int,
    columns: int
) -> list[float] {
    var q = _identity_block(m, columns)
    var step = k - 1
    while step >= 0 {
        _reflect_from(q, m, columns, vs, step * m, step, norms[step], 0)
        step -= 1
    }
    return q
}

#: `R` cut out of a finished Householder pass, `rows` by `n`.
fn _upper_triangle(work: list[float], n: int, rows: int) -> list[float] {
    var out = [0.0 for _slot in 0..rows * n]
    for row in 0..rows {
        for column in row..n { out[row * n + column] = work[row * n + column] }
    }
    return out
}

#: `(c, s, r)` with `c a + s b = r` and `c b - s a = 0`.
fn _givens(a: float, b: float) -> (float, float, float) {
    if b == 0.0 { return (1.0, 0.0, a) }
    if a == 0.0 { return (0.0, 1.0, b) }
    let r = math.hypot(a, b)
    return (a / r, b / r, r)
}

#: Mix columns `p` and `q`: this is `A <- A G`, and `rows` of zero is the
#: no-op the value-only paths pass to skip accumulating a basis they will not
#: return.
fn _rotate_columns(
    a: list[float],
    rows: int,
    cols: int,
    p: int,
    q: int,
    c: float,
    s: float
) -> void {
    for row in 0..rows {
        let base = row * cols
        let x = a[base + p]
        let y = a[base + q]
        a[base + p] = c * x + s * y
        a[base + q] = c * y - s * x
    }
}

#: Mix rows `p` and `q` the same way, which is `A <- G^T A`.
fn _rotate_rows(a: list[float], cols: int, p: int, q: int, c: float, s: float) -> void {
    let top = p * cols
    let bottom = q * cols
    for column in 0..cols {
        let x = a[top + column]
        let y = a[bottom + column]
        a[top + column] = c * x + s * y
        a[bottom + column] = c * y - s * x
    }
}

fn _swap_columns(a: list[float], rows: int, cols: int, p: int, q: int) -> void {
    for row in 0..rows {
        let base = row * cols
        let held = a[base + p]
        a[base + p] = a[base + q]
        a[base + q] = held
    }
}

#: Householder bidiagonalization: `d` and `e` are the diagonal and the
#: superdiagonal of `B`, with `A = U B V^T`.
#:
#: `ucols` and `vcols` of zero skip accumulating that basis, which is most of
#: the work when only the singular values are wanted.
fn _bidiagonalize(
    work: list[float],
    m: int,
    n: int,
    ucols: int,
    vcols: int
) -> (list[float], list[float], list[float], list[float]) {
    var d = [0.0 for _index in 0..n]
    var e = [0.0 for _index in 0..n]
    var left = [0.0 for _slot in 0..n * m]
    var lnorms = [0.0 for _index in 0..n]
    var right = [0.0 for _slot in 0..n * n]
    var rnorms = [0.0 for _index in 0..n]

    for step in 0..n {
        let (alpha, vnorm) = _column_reflector(work, m, n, step, step, left, step * m)
        lnorms[step] = vnorm
        d[step] = alpha
        _reflect_from(work, m, n, left, step * m, step, vnorm, step + 1)
        if step + 2 < n {
            let (beta, wnorm) = _row_reflector(work, n, step, step + 1, right, step * n)
            rnorms[step] = wnorm
            e[step + 1] = beta
            _reflect_across(work, m, n, right, step * n, step + 1, wnorm, step + 1)
        } elif step + 1 < n {
            e[step + 1] = work[step * n + step + 1]
        }
    }

    var u = _identity_block(m, ucols)
    var step = n - 1
    while step >= 0 {
        _reflect_from(u, m, ucols, left, step * m, step, lnorms[step], 0)
        step -= 1
    }
    var v = _identity_block(n, vcols)
    step = n - 1
    while step >= 0 {
        _reflect_from(v, n, vcols, right, step * n, step + 1, rnorms[step], 0)
        step -= 1
    }
    return (d, e, u, v)
}

#: One implicit-shift QR sweep across the bidiagonal block `low..=high`.
#:
#: The shift is Wilkinson's on the trailing two by two of `B^T B`, and `B^T B`
#: is never formed: the first rotation is chosen from the shifted first column
#: alone, and the bulge it puts below the diagonal is chased down by
#: alternating a column rotation and a row rotation, each accumulated into `v`
#: and `u`. Squaring the entries once, in the shift, is the whole of the
#: accuracy difference between this and an eigendecomposition of `A^T A`.
fn _svd_sweep(
    d: list[float],
    e: list[float],
    u: list[float],
    v: list[float],
    urows: int,
    ucols: int,
    vrows: int,
    n: int,
    low: int,
    high: int
) -> void {
    let trailing = d[high] * d[high] + e[high] * e[high]
    let above = if high - 1 > low { e[high - 1] * e[high - 1] } else { 0.0 }
    let neighbour = d[high - 1] * d[high - 1] + above
    let corner = d[high - 1] * e[high]
    let half = (neighbour - trailing) / 2.0
    var mu = trailing
    if corner != 0.0 {
        let root = math.hypot(half, corner)
        let denom = if half >= 0.0 { half + root } else { half - root }
        if denom != 0.0 { mu = trailing - corner * corner / denom }
    }

    var y = d[low] * d[low] - mu
    var z = d[low] * e[low + 1]
    for k in low..high {
        let (c, s, r) = _givens(y, z)
        if k > low { e[k] = r }
        let dk = d[k]
        let ek = e[k + 1]
        d[k] = c * dk + s * ek
        e[k + 1] = c * ek - s * dk
        let bulge = s * d[k + 1]
        d[k + 1] = c * d[k + 1]
        _rotate_columns(v, vrows, n, k, k + 1, c, s)

        let (c2, s2, r2) = _givens(d[k], bulge)
        d[k] = r2
        let ek1 = e[k + 1]
        let dk1 = d[k + 1]
        e[k + 1] = c2 * ek1 + s2 * dk1
        d[k + 1] = c2 * dk1 - s2 * ek1
        if k + 2 <= high {
            y = e[k + 1]
            z = s2 * e[k + 2]
            e[k + 2] = c2 * e[k + 2]
        }
        _rotate_columns(u, urows, ucols, k, k + 1, c2, s2)
    }
}

#: Rotate away the row a vanished `d[at]` leaves hanging.
#:
#: A zero on the diagonal splits the block in two, but only once the
#: superdiagonal beside it is gone; leaving it there is what makes a
#: rank-deficient matrix stall instead of converging.
fn _svd_cancel(
    d: list[float],
    e: list[float],
    u: list[float],
    urows: int,
    ucols: int,
    at: int,
    high: int
) -> void {
    var extra = e[at + 1]
    e[at + 1] = 0.0
    for index in at + 1..=high {
        let (c, s, r) = _givens(d[index], extra)
        d[index] = r
        if index < high {
            extra = -s * e[index + 1]
            e[index + 1] = c * e[index + 1]
        }
        _rotate_columns(u, urows, ucols, index, at, c, s)
    }
}

#: Sweep the bidiagonal to diagonal, deflating from the bottom up.
fn _svd_iterate(
    d: list[float],
    e: list[float],
    u: list[float],
    v: list[float],
    urows: int,
    ucols: int,
    vrows: int,
    n: int
) -> void {
    var high = n - 1
    var budget = QR_ITERATIONS * n + 100
    while high > 0 {
        var index = high
        while index > 0 {
            let over = d[index - 1]
            let under = d[index]
            let scale = (if over < 0.0 {
                -over
            } else {
                over
            }) + (if under < 0.0 {
                -under
            } else {
                under
            })
            let value = e[index]
            let size = if value < 0.0 { -value } else { value }
            if size <= FLOAT32_EPS * scale { e[index] = 0.0 }
            index -= 1
        }
        if e[high] == 0.0 {
            high -= 1
            continue
        }
        var low = high
        while low > 0 and e[low] != 0.0 { low -= 1 }

        var scale = 0.0
        for k in low..=high {
            let value = d[k]
            let size = if value < 0.0 { -value } else { value }
            if size > scale { scale = size }
        }
        var vanished = -1
        for k in low..high {
            let value = d[k]
            let size = if value < 0.0 { -value } else { value }
            if size <= FLOAT32_EPS * scale { vanished = k }
        }
        if vanished >= 0 {
            d[vanished] = 0.0
            _svd_cancel(d, e, u, urows, ucols, vanished, high)
            continue
        }

        budget -= 1
        if budget <= 0 {
            throw RuntimeError(f"svd did not converge in {QR_ITERATIONS} sweeps a column")
        }
        _svd_sweep(d, e, u, v, urows, ucols, vrows, n, low, high)
    }
}

#: Make every singular value non-negative and put them in descending order,
#: carrying the two bases with them.
fn _svd_finish(
    d: list[float],
    u: list[float],
    v: list[float],
    urows: int,
    ucols: int,
    vrows: int,
    n: int
) -> void {
    for index in 0..n {
        if d[index] >= 0.0 { continue }
        d[index] = -d[index]
        for row in 0..vrows { v[row * n + index] = -v[row * n + index] }
    }
    for index in 0..n {
        var best = index
        for other in index + 1..n {
            if d[other] > d[best] { best = other }
        }
        if best == index { continue }
        let held = d[index]
        d[index] = d[best]
        d[best] = held
        _swap_columns(u, urows, ucols, index, best)
        _swap_columns(v, vrows, n, index, best)
    }
}

#: `A = U S V^T`, singular values descending.
#:
#: `full` decides whether `U` carries the `m - n` further columns that span
#: the orthogonal complement of the range; `vectors` of false skips both bases
#: and leaves the returned lists empty, which is what every entry point that
#: only reads the values takes.
#:
#: A wide matrix is turned round rather than special-cased: the transpose's
#: factorization is this one's with the two bases exchanged.
fn _svd_host(
    values: list[float],
    m: int,
    n: int,
    full: bool,
    vectors: bool
) -> (list[float], list[float], list[float]) {
    if m < n {
        let (ut, s, vt) = _svd_host(_transpose_host(values, m, n), n, m, full, vectors)
        if not vectors { return ([], s, []) }
        let wide = if full { n } else { m }
        return (_transpose_host(vt, m, m), s, _transpose_host(ut, n, wide))
    }
    var work = [value for value in values]
    let ucols = if not vectors { 0 } elif full { m } else { n }
    let vcols = if vectors { n } else { 0 }
    let urows = if vectors { m } else { 0 }
    let vrows = if vectors { n } else { 0 }
    let (d, e, u, v) = _bidiagonalize(work, m, n, ucols, vcols)
    _svd_iterate(d, e, u, v, urows, ucols, vrows, n)
    _svd_finish(d, u, v, urows, ucols, vrows, n)
    if not vectors { return ([], d, []) }
    return (u, d, _transpose_host(v, n, n))
}

#: Put eigenvalues in ascending order, carrying their vectors with them.
fn _order_ascending(values: list[float], basis: list[float], n: int, rows: int) -> void {
    for index in 0..n {
        var best = index
        for other in index + 1..n {
            if values[other] < values[best] { best = other }
        }
        if best == index { continue }
        let held = values[index]
        values[index] = values[best]
        values[best] = held
        _swap_columns(basis, rows, n, index, best)
    }
}

#: The weight sitting off the diagonal, against the weight on it.
fn _jacobi_off(a: list[float], n: int) -> (float, float) {
    var off = 0.0
    var on = 0.0
    for row in 0..n {
        for column in row + 1..n {
            let value = a[row * n + column]
            off += value * value
        }
        let diagonal = a[row * n + row]
        on += diagonal * diagonal
    }
    return (off, on)
}

#: `A = Q L Q^T` for a symmetric `A`, by cyclic Jacobi, eigenvalues ascending.
#:
#: Jacobi and not a tridiagonal QL: at these sizes it is a small constant more
#: arithmetic, it converges for any symmetric input without a shift strategy,
#: and it reaches the small eigenvalues to full relative accuracy — which is
#: what `pinv` and `matrix_rank` are actually asking about when they call it.
fn _jacobi_eigh(values: list[float], n: int, vectors: bool) -> (list[float], list[float]) {
    var a = [value for value in values]
    let rows = if vectors { n } else { 0 }
    var basis = _identity_block(rows, n)
    for _sweep in 0..JACOBI_SWEEPS {
        let (off, on) = _jacobi_off(a, n)
        if off <= JACOBI_TOL * (on + off) { break }
        for p in 0..n {
            for r in p + 1..n {
                let apr = a[p * n + r]
                if apr == 0.0 { continue }
                let theta = (a[p * n + p] - a[r * n + r]) / (2.0 * apr)
                let root = math.hypot(theta, 1.0)
                let t = if theta >= 0.0 { 1.0 / (theta + root) } else { -1.0 / (root - theta) }
                let c = 1.0 / math.sqrt(1.0 + t * t)
                let s = t * c
                _rotate_columns(a, n, n, p, r, c, s)
                _rotate_rows(a, n, p, r, c, s)
                a[p * n + r] = 0.0
                a[r * n + p] = 0.0
                _rotate_columns(basis, rows, n, p, r, c, s)
            }
        }
    }
    var spectrum = [0.0 for _index in 0..n]
    for index in 0..n { spectrum[index] = a[index * n + index] }
    _order_ascending(spectrum, basis, n, rows)
    return (spectrum, basis)
}

#: The triangle `uplo` names, mirrored into a full symmetric matrix.
#:
#: LAPACK reads one triangle and so does this: a matrix that is not quite
#: symmetric is factored as though the named triangle were the truth, rather
#: than being rejected or quietly averaged.
fn _symmetrize(values: list[float], at: int, n: int, lower: bool) -> list[float] {
    var out = [0.0 for _slot in 0..n * n]
    for row in 0..n {
        for column in 0..n {
            let take = if lower == (row >= column) { row * n + column } else { column * n + row }
            out[row * n + column] = values[at + take]
        }
    }
    return out
}

#: Reduce to upper Hessenberg by Householder, returning the basis `Z` with
#: `H = Z^T A Z`.
fn _hessenberg(work: list[float], n: int, vectors: bool) -> list[float] {
    var vs = [0.0 for _slot in 0..n * n]
    var norms = [0.0 for _index in 0..n]
    var step = 0
    while step < n - 2 {
        let (alpha, vnorm) = _column_reflector(work, n, n, step, step + 1, vs, step * n)
        norms[step] = vnorm
        work[(step + 1) * n + step] = alpha
        for row in step + 2..n { work[row * n + step] = 0.0 }
        _reflect_from(work, n, n, vs, step * n, step + 1, vnorm, step + 1)
        _reflect_across(work, n, n, vs, step * n, step + 1, vnorm, 0)
        step += 1
    }
    let rows = if vectors { n } else { 0 }
    var basis = _identity_block(rows, n)
    step = n - 3
    while step >= 0 {
        _reflect_from(basis, rows, n, vs, step * n, step + 1, norms[step], 0)
        step -= 1
    }
    return basis
}

#: One shifted QR sweep across the Hessenberg block `low..=high`.
#:
#: The shift is subtracted only from the block's diagonal and the rotations
#: only mix indices inside it, but each is applied across the whole row and
#: the whole column: that is what keeps the coupling to the already-converged
#: parts correct while leaving them otherwise alone.
fn _hessenberg_sweep(
    h: list[float],
    z: list[float],
    n: int,
    zrows: int,
    low: int,
    high: int,
    mu: float
) -> void {
    for k in low..=high { h[k * n + k] -= mu }
    var cosines = [0.0 for _index in low..high]
    var sines = [0.0 for _index in low..high]
    for k in low..high {
        let (c, s, _r) = _givens(h[k * n + k], h[(k + 1) * n + k])
        cosines[k - low] = c
        sines[k - low] = s
        _rotate_rows(h, n, k, k + 1, c, s)
        h[(k + 1) * n + k] = 0.0
    }
    for k in low..high {
        _rotate_columns(h, n, n, k, k + 1, cosines[k - low], sines[k - low])
        _rotate_columns(z, zrows, n, k, k + 1, cosines[k - low], sines[k - low])
    }
    for k in low..=high { h[k * n + k] += mu }
}

#: Triangularise a two by two block that has decoupled, or refuse it.
#:
#: A converged two by two with a negative discriminant is a genuine complex
#: conjugate pair — not the transient one an unconverged block can show — and
#: it is the one thing this cannot carry: there is no `COMPLEX64` answer to
#: give without running the whole Schur iteration in complex arithmetic. So it
#: names the pair and stops, rather than handing back a real part and calling
#: it an eigenvalue.
fn _split_pair(h: list[float], z: list[float], n: int, zrows: int, high: int) -> void {
    let top = h[(high - 1) * n + high - 1]
    let corner = h[high * n + high - 1]
    let bottom = h[high * n + high]
    let half = (top - bottom) / 2.0
    let disc = half * half + h[(high - 1) * n + high] * corner
    if disc < 0.0 {
        let real = (top + bottom) / 2.0
        let imaginary = math.sqrt(-disc)
        throw ValueError(
            f"eig found the complex pair {real} +/- {imaginary}i in rows {high - 1}..{high}; " +
                f"this factorization is real only, and eigh covers the symmetric case"
        )
    }
    if corner == 0.0 { return }
    let root = math.sqrt(disc)
    let near = if half >= 0.0 { half + root } else { half - root }
    let (c, s, _r) = _givens(near, corner)
    _rotate_columns(h, n, n, high - 1, high, c, s)
    _rotate_rows(h, n, high - 1, high, c, s)
    _rotate_columns(z, zrows, n, high - 1, high, c, s)
    h[high * n + high - 1] = 0.0
}

#: Shifted QR on an upper Hessenberg matrix, deflating from the bottom.
#:
#: A single real Wilkinson shift with Givens rotations, not Francis's double
#: shift: the one thing the double shift buys is staying in real arithmetic
#: through a complex pair, and a complex pair is exactly the case this raises
#: on. Every tenth stalled iteration takes an ad-hoc shift instead, which is
#: what breaks the cycles a repeated eigenvalue can otherwise sit in.
fn _hessenberg_qr(h: list[float], z: list[float], n: int, zrows: int) -> void {
    var high = n - 1
    var budget = QR_ITERATIONS * n + 100
    var stale = 0
    while high > 0 {
        var low = high
        while low > 0 {
            let sub = h[low * n + low - 1]
            let size = if sub < 0.0 { -sub } else { sub }
            let over = h[(low - 1) * n + low - 1]
            let under = h[low * n + low]
            let scale = (if over < 0.0 {
                -over
            } else {
                over
            }) + (if under < 0.0 {
                -under
            } else {
                under
            })
            if size <= FLOAT32_EPS * scale {
                h[low * n + low - 1] = 0.0
                break
            }
            low -= 1
        }
        if low == high {
            high -= 1
            stale = 0
            continue
        }
        if low == high - 1 {
            _split_pair(h, z, n, zrows, high)
            high -= 2
            stale = 0
            continue
        }

        let top = h[(high - 1) * n + high - 1]
        let bottom = h[high * n + high]
        let half = (top - bottom) / 2.0
        let disc = half * half + h[(high - 1) * n + high] * h[high * n + high - 1]
        var mu = bottom + half
        if disc >= 0.0 {
            let root = math.sqrt(disc)
            mu = bottom + half - (if half >= 0.0 { root } else { -root })
        }
        stale += 1
        if stale % 10 == 0 {
            let one = h[high * n + high - 1]
            let two = h[(high - 1) * n + high - 2]
            mu = (if one < 0.0 { -one } else { one }) + (if two < 0.0 { -two } else { two })
        }
        budget -= 1
        if budget <= 0 {
            throw RuntimeError(f"eig did not converge in {QR_ITERATIONS} sweeps an eigenvalue")
        }
        _hessenberg_sweep(h, z, n, zrows, low, high, mu)
    }
}

#: `x = Z y` for the leading `upto + 1` entries of `y`, returning `x^T x`.
fn _apply_basis(
    z: list[float],
    y: list[float],
    out: list[float],
    n: int,
    column: int,
    upto: int
) -> float {
    var norm = 0.0
    for row in 0..n {
        var total = 0.0
        for k in 0..=upto { total += z[row * n + k] * y[k] }
        out[row * n + column] = total
        norm += total * total
    }
    return norm
}

#: Eigenvectors of a triangular Schur form, mapped back through `z`.
#:
#: Back substitution on `(T - lambda I) y = 0` with the eigenvalue's own entry
#: pinned to one. A repeated eigenvalue makes a diagonal difference vanish;
#: the substitution floors it at the rounding of the matrix rather than
#: dividing by zero, which yields one usable vector for the pair instead of a
#: NaN — the same compromise LAPACK's `trevc` makes, and the reason a
#: defective matrix gets a basis here that is not of full rank.
fn _schur_vectors(t: list[float], z: list[float], n: int) -> list[float] {
    var out = [0.0 for _slot in 0..n * n]
    var scale = 0.0
    for index in 0..n * n {
        let value = t[index]
        let size = if value < 0.0 { -value } else { value }
        if size > scale { scale = size }
    }
    let floor = if scale > 0.0 { scale * FLOAT32_EPS } else { FLOAT32_EPS }
    for column in 0..n {
        let lambda = t[column * n + column]
        var y = [0.0 for _index in 0..n]
        y[column] = 1.0
        var row = column - 1
        while row >= 0 {
            var total = 0.0
            for k in row + 1..=column { total += t[row * n + k] * y[k] }
            var denom = t[row * n + row] - lambda
            if denom < floor and denom > -floor { denom = floor }
            y[row] = -total / denom
            row -= 1
        }
        let norm = _apply_basis(z, y, out, n, column, column)
        if norm <= 0.0 { continue }
        let inverse = 1.0 / math.sqrt(norm)
        for index in 0..n { out[index * n + column] *= inverse }
    }
    return out
}

#: `A = Z T Z^T`: the real Schur form, and the eigenvectors read off it.
fn _eig_host(values: list[float], n: int, vectors: bool) -> (list[float], list[float]) {
    var h = [value for value in values]
    let rows = if vectors { n } else { 0 }
    let basis = _hessenberg(h, n, vectors)
    _hessenberg_qr(h, basis, n, rows)
    var spectrum = [0.0 for _index in 0..n]
    for index in 0..n { spectrum[index] = h[index * n + index] }
    if not vectors { return (spectrum, []) }
    return (spectrum, _schur_vectors(h, basis, n))
}

#: `shape` with its last `drop` axes taken off — the batch a stack sits in.
fn _lead(shape: list[int], drop: int) -> list[int] {
    var out: list[int] = []
    for axis in 0..shape.len() - drop { out.push(shape[axis]) }
    return out
}

#: The two stacks a solver was handed, batch axes broadcast against each
#: other and matrices laid out row-major.
#:
#: `b` is a stack of vectors when it is one-dimensional and a stack of
#: matrices otherwise. That is numpy 2's rule and not numpy 1's: the old
#: `b.ndim == a.ndim - 1` reading cannot tell a `(3, 3)` right-hand side apart
#: as three vectors or one matrix, and picked differently depending on what
#: `a` happened to be.
fn _solver_pair(
    a: NDArray,
    b: NDArray,
    what: str
) -> (list[float], list[float], list[int], int, int, bool) {
    require_real(a, what)
    require_real(b, what)
    let arank = a.ndim()
    if arank < 2 {
        throw ValueError(f"{what} needs a matrix or a stack of them, got {arank} dimensions")
    }
    let n = a.shape[arank - 1]
    _require_square(a.shape[arank - 2], n, what)
    let brank = b.ndim()
    if brank < 1 { throw ValueError(f"{what} needs a right-hand side with an axis to solve along") }
    let vector = brank == 1
    let wide = if vector { 1 } else { b.shape[brank - 1] }
    let rows = if vector { b.shape[brank - 1] } else { b.shape[brank - 2] }
    if rows != n {
        throw ValueError(f"{what} cannot apply a {n} by {n} matrix to {rows} rows")
    }
    let batch = broadcast_shapes(_lead(a.shape, 2), _lead(b.shape, if vector { 1 } else { 2 }))
    let tail = if vector { [n] } else { [n, wide] }
    let left = a.broadcast_to(batch + [n, n])
    let right = b.broadcast_to(batch + tail)
    return (left.to_list(), right.to_list(), batch, n, wide, vector)
}

#: The shape a right-hand side keeps on the way out.
fn _rhs_shape(batch: list[int], n: int, wide: int, vector: bool) -> list[int] {
    if vector { return batch + [n] }
    return batch + [n, wide]
}

#: Solve `a x = b` for a square `a`, by LU with partial pivoting.
#:
#: A one-dimensional `b` is a single right-hand side and anything else is a
#: stack of matrices, which is numpy 2's rule. The batch axes of the two
#: broadcast against each other; a matrix stretched over a batch is factored
#: once per pair rather than once, which is the cost of writing the loop over
#: the broadcast result instead of over the two operands.
pub fn solve(a: NDArray, b: NDArray) -> NDArray {
    let (av, bv, batch, n, wide, vector) = _solver_pair(a, b, "solve")
    var out: list[float] = []
    for index in 0..element_count(batch) {
        let (lu, piv, _sign) = _lu_factor_host(_slice_matrix(av, index * n * n, n * n), n)
        let rhs = _slice_matrix(bv, index * n * wide, n * wide)
        _extend(out, _lu_solve_host(lu, piv, n, rhs, wide, "solve"))
    }
    return _emit(out, _rhs_shape(batch, n, wide, vector))
}

#: `PA = LU`: the packed factors, and the row each step swapped with.
#:
#: scipy hands the pair straight back to `lu_solve` as a tuple; here they are
#: two arrays and `lu_solve` takes two parameters, which is the spelling that
#: survives a stacked input without a tuple of stacks.
pub fn lu_factor(a: NDArray) -> (NDArray, NDArray) {
    let (values, batch, m, n) = _stacked(a, "lu_factor")
    _require_square(m, n, "lu_factor")
    var factors: list[float] = []
    var pivots: list[float] = []
    for index in 0..element_count(batch) {
        let (lu, piv, _sign) = _lu_factor_host(_slice_matrix(values, index * n * n, n * n), n)
        _extend(factors, lu)
        for row in piv { pivots.push(float(row)) }
    }
    return (_emit(factors, batch + [n, n]), _emit(pivots, batch + [n], INT32))
}

#: Solve `a x = b` from a factorization `lu_factor` already paid for.
pub fn lu_solve(lu: NDArray, piv: NDArray, b: NDArray) -> NDArray {
    let (av, bv, batch, n, wide, vector) = _solver_pair(lu, b, "lu_solve")
    require_real(piv, "lu_solve")
    let pv = piv.broadcast_to(batch + [n]).to_list()
    var out: list[float] = []
    for index in 0..element_count(batch) {
        var rows = [0 for _index in 0..n]
        for step in 0..n { rows[step] = int(pv[index * n + step] + 0.5) }
        let factors = _slice_matrix(av, index * n * n, n * n)
        let rhs = _slice_matrix(bv, index * n * wide, n * wide)
        _extend(out, _lu_solve_host(factors, rows, n, rhs, wide, "lu_solve"))
    }
    return _emit(out, _rhs_shape(batch, n, wide, vector))
}

#: Solve a triangular system without factoring anything.
#:
#: `lower` picks the triangle that is read; the other one is ignored rather
#: than checked, so passing a full matrix solves against its triangle.
#: `unit_diagonal` takes the diagonal as ones without reading it, and
#: `transpose` solves `a^T x = b` by reading the coefficients the other way
#: round instead of forming the transpose.
pub fn triangular_solve(
    a: NDArray,
    b: NDArray,
    lower: bool = false,
    unit_diagonal: bool = false,
    transpose: bool = false
) -> NDArray {
    let (av, bv, batch, n, wide, vector) = _solver_pair(a, b, "triangular_solve")
    let forward = lower != transpose
    var out: list[float] = []
    for index in 0..element_count(batch) {
        let matrix = _slice_matrix(av, index * n * n, n * n)
        if not unit_diagonal {
            for step in 0..n {
                if matrix[step * n + step] == 0.0 {
                    throw ValueError(
                        f"triangular_solve was handed a zero on the diagonal at {step}"
                    )
                }
            }
        }
        var x = _slice_matrix(bv, index * n * wide, n * wide)
        _triangular_host(matrix, n, x, wide, forward, unit_diagonal, transpose)
        _extend(out, x)
    }
    return _emit(out, _rhs_shape(batch, n, wide, vector))
}

#: `A = L L^T` for a symmetric positive definite `A`.
#:
#: Lower by default, as numpy's `cholesky` is; `upper` transposes the result
#: rather than factoring differently. Only the lower triangle of the input is
#: read either way.
pub fn cholesky(a: NDArray, upper: bool = false) -> NDArray {
    let (values, batch, m, n) = _stacked(a, "cholesky")
    _require_square(m, n, "cholesky")
    var out: list[float] = []
    for index in 0..element_count(batch) {
        let factor = _cholesky_host(_slice_matrix(values, index * n * n, n * n), n)
        _extend(out, if upper { _transpose_host(factor, n, n) } else { factor })
    }
    return _emit(out, batch + [n, n])
}

#: The inverse, by LU with partial pivoting against the identity.
#:
#: One factorization and `n` back substitutions, not `n` solves: the whole
#: point of keeping `L` and `U` around.
pub fn inv(a: NDArray) -> NDArray {
    let (values, batch, m, n) = _stacked(a, "inv")
    _require_square(m, n, "inv")
    var out: list[float] = []
    for index in 0..element_count(batch) {
        let (lu, piv, _sign) = _lu_factor_host(_slice_matrix(values, index * n * n, n * n), n)
        _extend(out, _lu_solve_host(lu, piv, n, _identity_block(n, n), n, "inv"))
    }
    return _emit(out, batch + [n, n])
}

#: The determinant: the permutation's sign times the diagonal of `U`.
pub fn det(a: NDArray) -> NDArray {
    let (values, batch, m, n) = _stacked(a, "det")
    _require_square(m, n, "det")
    var out: list[float] = []
    for index in 0..element_count(batch) {
        let (lu, _piv, sign) = _lu_factor_host(_slice_matrix(values, index * n * n, n * n), n)
        var total = float(sign)
        for step in 0..n { total *= lu[step * n + step] }
        out.push(total)
    }
    return _emit(out, batch)
}

#: The determinant split into a sign and the log of its magnitude.
#:
#: What `det` is for when the matrix is large enough that the product of the
#: pivots overflows or flushes to zero: a hundred pivots of 100 is 1e200,
#: which float32 storage cannot hold and the sum of logs can.
pub fn slogdet(a: NDArray) -> (NDArray, NDArray) {
    let (values, batch, m, n) = _stacked(a, "slogdet")
    _require_square(m, n, "slogdet")
    var signs: list[float] = []
    var logs: list[float] = []
    for index in 0..element_count(batch) {
        let (lu, _piv, sign) = _lu_factor_host(_slice_matrix(values, index * n * n, n * n), n)
        var total = 0.0
        var mark = float(sign)
        for step in 0..n {
            let pivot = lu[step * n + step]
            if pivot == 0.0 {
                mark = 0.0
                total = -math.INF
                break
            }
            if pivot < 0.0 { mark = -mark }
            total += math.ln(if pivot < 0.0 { -pivot } else { pivot })
        }
        signs.push(mark)
        logs.push(total)
    }
    return (_emit(signs, batch), _emit(logs, batch))
}

#: `A = QR` with orthonormal `Q`, by Householder reflections.
#:
#: `"reduced"` gives `Q` as `m` by `k` and `R` as `k` by `n` for
#: `k = min(m, n)`; `"complete"` widens `Q` to `m` by `m` and `R` to `m` by
#: `n`. numpy's `"r"` and `"raw"` have no spelling here — a tuple return
#: cannot change its arity — so take the second element of `"reduced"`.
pub fn qr(a: NDArray, mode: str = "reduced") -> (NDArray, NDArray) {
    let (values, batch, m, n) = _stacked(a, "qr")
    if mode != "reduced" and mode != "complete" {
        throw ValueError(f"qr does not know the mode {mode}; it takes \"reduced\" or \"complete\"")
    }
    let k = min(m, n)
    let wide = if mode == "complete" { m } else { k }
    var bases: list[float] = []
    var triangles: list[float] = []
    for index in 0..element_count(batch) {
        var work = _slice_matrix(values, index * m * n, m * n)
        let (vs, norms) = _householder_qr(work, m, n, k)
        _extend(bases, _accumulate_reflectors(vs, norms, m, k, wide))
        _extend(triangles, _upper_triangle(work, n, wide))
    }
    return (_emit(bases, batch + [m, wide]), _emit(triangles, batch + [wide, n]))
}

#: `A = U S V^T`, singular values descending.
#:
#: `full_matrices` widens `U` and `V^T` to the square bases that span the two
#: whole spaces rather than just the range and the row space. numpy's
#: `compute_uv` is `svdvals` here, for the same reason `qr` has no `"r"`.
pub fn svd(a: NDArray, full_matrices: bool = true) -> (NDArray, NDArray, NDArray) {
    let (values, batch, m, n) = _stacked(a, "svd")
    let k = min(m, n)
    let wide = if full_matrices { m } else { k }
    let tall = if full_matrices { n } else { k }
    var lefts: list[float] = []
    var spectra: list[float] = []
    var rights: list[float] = []
    for index in 0..element_count(batch) {
        let matrix = _slice_matrix(values, index * m * n, m * n)
        let (u, s, vt) = _svd_host(matrix, m, n, full_matrices, true)
        _extend(lefts, u)
        _extend(spectra, s)
        _extend(rights, vt)
    }
    return (
        _emit(lefts, batch + [m, wide]),
        _emit(spectra, batch + [k]),
        _emit(rights, batch + [tall, n])
    )
}

#: The singular values alone, descending.
#:
#: Worth its own entry point rather than dropping two of `svd`'s three: the
#: bases are the quadratic-by-linear part of the work, and every caller here
#: that only needs the values — `matrix_rank`, `cond`, the 2 and nuclear
#: norms — skips accumulating them.
pub fn svdvals(a: NDArray) -> NDArray {
    let (values, batch, m, n) = _stacked(a, "svdvals")
    var out: list[float] = []
    for index in 0..element_count(batch) {
        let matrix = _slice_matrix(values, index * m * n, m * n)
        let (_u, s, _vt) = _svd_host(matrix, m, n, false, false)
        _extend(out, s)
    }
    return _emit(out, batch + [min(m, n)])
}

fn _uplo_lower(uplo: str) -> bool {
    if uplo == "L" or uplo == "l" { return true }
    if uplo == "U" or uplo == "u" { return false }
    throw ValueError(f"uplo is \"L\" or \"U\", not {uplo}")
}

#: Eigenvalues and eigenvectors of a symmetric matrix, ascending.
#:
#: Only the triangle `uplo` names is read. The eigenvectors are the columns of
#: the second result, so `a @ v[:, i] == w[i] * v[:, i]`.
pub fn eigh(a: NDArray, uplo: str = "L") -> (NDArray, NDArray) {
    let (values, batch, m, n) = _stacked(a, "eigh")
    _require_square(m, n, "eigh")
    let lower = _uplo_lower(uplo)
    var spectra: list[float] = []
    var bases: list[float] = []
    for index in 0..element_count(batch) {
        let matrix = _symmetrize(values, index * n * n, n, lower)
        let (w, q) = _jacobi_eigh(matrix, n, true)
        _extend(spectra, w)
        _extend(bases, q)
    }
    return (_emit(spectra, batch + [n]), _emit(bases, batch + [n, n]))
}

#: The eigenvalues of a symmetric matrix, ascending, without their vectors.
pub fn eigvalsh(a: NDArray, uplo: str = "L") -> NDArray {
    let (values, batch, m, n) = _stacked(a, "eigvalsh")
    _require_square(m, n, "eigvalsh")
    let lower = _uplo_lower(uplo)
    var out: list[float] = []
    for index in 0..element_count(batch) {
        let matrix = _symmetrize(values, index * n * n, n, lower)
        let (w, _q) = _jacobi_eigh(matrix, n, false)
        _extend(out, w)
    }
    return _emit(out, batch + [n])
}

#: Eigenvalues and right eigenvectors of a general matrix — real ones only.
#:
#: Hessenberg reduction, then shifted QR to the real Schur form, then back
#: substitution for the vectors. A real matrix can have complex conjugate
#: eigenvalues and this raises `ValueError` naming the pair when it meets one:
#: there is no `COMPLEX64` answer it could give without carrying complex
#: arithmetic through the whole iteration, and returning the real part would
#: be a wrong answer rather than a missing one. `eigh` has no such case.
pub fn eig(a: NDArray) -> (NDArray, NDArray) {
    let (values, batch, m, n) = _stacked(a, "eig")
    _require_square(m, n, "eig")
    var spectra: list[float] = []
    var bases: list[float] = []
    for index in 0..element_count(batch) {
        let (w, z) = _eig_host(_slice_matrix(values, index * n * n, n * n), n, true)
        _extend(spectra, w)
        _extend(bases, z)
    }
    return (_emit(spectra, batch + [n]), _emit(bases, batch + [n, n]))
}

#: The eigenvalues of a general matrix, without their vectors. Real only, as
#: `eig` is.
pub fn eigvals(a: NDArray) -> NDArray {
    let (values, batch, m, n) = _stacked(a, "eigvals")
    _require_square(m, n, "eigvals")
    var out: list[float] = []
    for index in 0..element_count(batch) {
        let (w, _z) = _eig_host(_slice_matrix(values, index * n * n, n * n), n, false)
        _extend(out, w)
    }
    return _emit(out, batch + [n])
}

#: The cutoff below which a singular value is taken as noise.
#:
#: A negative `rcond` asks for the machine default, which is where this
#: departs from numpy: numpy's `1e-15` is a float64 rounding unit, and against
#: values that arrived through a float32 buffer it keeps eight orders of
#: magnitude of pure noise. `max(m, n) * FLOAT32_EPS` is the honest figure for
#: this storage, and passing a positive `rcond` still overrides it.
fn _rank_cutoff(s: list[float], m: int, n: int, rcond: float) -> float {
    if s.len() == 0 { return 0.0 }
    if rcond >= 0.0 { return rcond * s[0] }
    return float(max(m, n)) * FLOAT32_EPS * s[0]
}

#: `V S^+ U^T` accumulated into `out`, which is `n` by `m`.
fn _pinv_accumulate(
    out: list[float],
    u: list[float],
    s: list[float],
    vt: list[float],
    m: int,
    n: int,
    k: int,
    cutoff: float
) -> void {
    for step in 0..k {
        if s[step] <= cutoff { continue }
        let inverse = 1.0 / s[step]
        for row in 0..n {
            let weight = vt[step * n + row] * inverse
            if weight == 0.0 { continue }
            for column in 0..m { out[row * m + column] += weight * u[column * k + step] }
        }
    }
}

#: The Moore-Penrose pseudo-inverse, through the singular values.
#:
#: See `_rank_cutoff` for why the default `rcond` is not numpy's.
pub fn pinv(a: NDArray, rcond: float = -1.0) -> NDArray {
    let (values, batch, m, n) = _stacked(a, "pinv")
    let k = min(m, n)
    var out: list[float] = []
    for index in 0..element_count(batch) {
        let matrix = _slice_matrix(values, index * m * n, m * n)
        let (u, s, vt) = _svd_host(matrix, m, n, false, true)
        var block = [0.0 for _slot in 0..n * m]
        _pinv_accumulate(block, u, s, vt, m, n, k, _rank_cutoff(s, m, n, rcond))
        _extend(out, block)
    }
    return _emit(out, batch + [n, m])
}

#: How many singular values stand above the noise floor.
#:
#: `tol` is absolute when it is given and `max(m, n) * FLOAT32_EPS * s_max`
#: when it is negative, which is numpy's rule with this storage's rounding
#: unit in place of float64's.
pub fn matrix_rank(a: NDArray, tol: float = -1.0) -> NDArray {
    let (values, batch, m, n) = _stacked(a, "matrix_rank")
    var out: list[float] = []
    for index in 0..element_count(batch) {
        let matrix = _slice_matrix(values, index * m * n, m * n)
        let (_u, s, _vt) = _svd_host(matrix, m, n, false, false)
        let floor = if tol >= 0.0 { tol } else { _rank_cutoff(s, m, n, -1.0) }
        var found = 0
        for value in s {
            if value > floor { found += 1 }
        }
        out.push(float(found))
    }
    return _emit(out, batch, INT32)
}

#: The condition number: how far a solve can amplify a perturbation.
#:
#: `null` and `2` are `s_max / s_min` and go through the singular values;
#: `-2` inverts that ratio; every other order is `norm(a, p) * norm(inv, p)`
#: and therefore needs a square, invertible matrix. A singular matrix has an
#: infinite condition number and gets one rather than an error.
pub fn cond(a: NDArray, p: any = null) -> NDArray {
    if p is not null and not p is str {
        let order = _as_number(p, "a matrix order")
        if order != 2.0 and order != -2.0 { return _cond_by_norm(a, p) }
    }
    if p is str { return _cond_by_norm(a, p) }
    let smallest = p is not null and _as_number(p, "a matrix order") == -2.0
    let (values, batch, m, n) = _stacked(a, "cond")
    let k = min(m, n)
    var out: list[float] = []
    for index in 0..element_count(batch) {
        let matrix = _slice_matrix(values, index * m * n, m * n)
        let (_u, s, _vt) = _svd_host(matrix, m, n, false, false)
        let low = if k > 0 { s[k - 1] } else { 0.0 }
        let high = if k > 0 { s[0] } else { 0.0 }
        if smallest {
            out.push(if high > 0.0 { low / high } else { 0.0 })
        } else {
            out.push(if low > 0.0 { high / low } else { math.INF })
        }
    }
    return _emit(out, batch)
}

#: `norm(a, p) * norm(inv(a), p)`, the orders the singular values cannot give.
fn _cond_by_norm(a: NDArray, p: any) -> NDArray {
    let inverse = inv(a)
    defer { inverse.free() }
    let left = norm(a, p, (-2, -1))
    defer { left.free() }
    let right = norm(inverse, p, (-2, -1))
    defer { right.free() }
    return left * right
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

#: `U^T b`, then `V S^+` applied to it — the two halves of a least squares
#: solve, split out so the cubic work sits in a body with no call in it.
fn _lstsq_solve(
    x: list[float],
    projected: list[float],
    u: list[float],
    s: list[float],
    vt: list[float],
    rhs: list[float],
    m: int,
    n: int,
    k: int,
    wide: int,
    cutoff: float
) -> void {
    for step in 0..k {
        for column in 0..wide {
            var total = 0.0
            for row in 0..m { total += u[row * k + step] * rhs[row * wide + column] }
            projected[step * wide + column] = total
        }
    }
    for step in 0..k {
        if s[step] <= cutoff { continue }
        let inverse = 1.0 / s[step]
        for row in 0..n {
            let weight = vt[step * n + row] * inverse
            if weight == 0.0 { continue }
            for column in 0..wide {
                x[row * wide + column] += weight * projected[step * wide + column]
            }
        }
    }
}

#: The least squares fit: the solution, the residual, the rank, and the
#: singular values it was decided on.
#:
#: Through the SVD and not the normal equations: `A^T A` squares the condition
#: number, which for a design matrix that is nearly rank-deficient is the
#: difference between an answer and noise. A rank-deficient fit gets the
#: minimum-norm solution rather than an error.
#:
#: The residual is read off the projection instead of by forming `b - Ax`:
#: `||b||^2 - ||U^T b||^2` is exactly the part of `b` the range cannot reach,
#: and the solve has already paid for `U^T b`. It is reported only for the
#: overdetermined full-rank case, as numpy reports it, and is otherwise an
#: empty array.
#:
#: One matrix, not a stack — numpy's `lstsq` is 2-d only and the four results
#: do not all have a batch axis to grow.
pub fn lstsq(a: NDArray, b: NDArray, rcond: float = -1.0) -> (NDArray, NDArray, NDArray, NDArray) {
    require_real(a, "lstsq")
    require_real(b, "lstsq")
    if a.ndim() != 2 {
        throw ValueError(f"lstsq takes one matrix, not {a.ndim()} dimensions")
    }
    if b.ndim() != 1 and b.ndim() != 2 {
        throw ValueError(
            f"lstsq takes one right-hand side or a matrix of them, not {b.ndim()} dimensions"
        )
    }
    let m = a.shape[0]
    let n = a.shape[1]
    let vector = b.ndim() == 1
    let wide = if vector { 1 } else { b.shape[1] }
    if b.shape[0] != m {
        throw ValueError(f"lstsq cannot fit {m} rows against {b.shape[0]}")
    }

    let (u, s, vt) = _svd_host(a.to_list(), m, n, false, true)
    let rhs = b.to_list()
    let k = min(m, n)
    let cutoff = _rank_cutoff(s, m, n, rcond)
    var rank = 0
    for value in s {
        if value > cutoff { rank += 1 }
    }
    var x = [0.0 for _slot in 0..n * wide]
    var projected = [0.0 for _slot in 0..k * wide]
    _lstsq_solve(x, projected, u, s, vt, rhs, m, n, k, wide, cutoff)

    var residuals: list[float] = []
    if rank == n and m > n {
        for column in 0..wide {
            var total = 0.0
            for row in 0..m {
                let value = rhs[row * wide + column]
                total += value * value
            }
            for step in 0..k {
                let value = projected[step * wide + column]
                total -= value * value
            }
            residuals.push(if total > 0.0 { total } else { 0.0 })
        }
    }
    return (_emit(x, if vector {
        [n]
    } else {
        [n, wide]
    }), _emit(residuals, [residuals.len()]), _emit([float(rank)], [], INT32), _emit(s, [k]))
}

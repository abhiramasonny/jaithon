#: The shared floor: the two policies, the calling convention, the kernel
#: idiom, and the handful of numeric primitives every other jaisci module
#: reaches for.
#:
#: **Host arithmetic is f64, device arithmetic is f32.** A Jaithon `float` —
#: and so a `list[float]` — is a double; a `std.gpu.Buffer` holds singles. A
#: number that goes to the device and comes back has been rounded to about
#: seven digits, which is fine for a distance matrix and ruinous for a pivot.
#: That is why the `dense_*` factorisations below take and return `list[float]`
#: and never touch a buffer: small dense work of a few hundred rows stays on
#: the host, exactly where jaicv's `core/linalg.jai` already keeps it, and for
#: the same reason — at that size a kernel launch costs more than the
#: arithmetic and the extra digits are the whole point.
#:
#: **Only O(n^2) and up becomes a kernel.** Distance matrices, convolutions,
#: sparse products, Jacobian assembly, a distribution evaluated over a whole
#: array: those dispatch. A Newton step on twelve parameters does not. Between
#: the two sits the corollary that catches the common mistake — *a host loop
#: over more than a few thousand elements is a bug in this repository, not a
#: slow path*. It costs seconds. Every module states, in its own doc, which of
#: its loops are bounded by construction and which of its passes dispatch; the
#: bounded ones here are the `dense_*` family, whose bound is the size of the
#: system, and `approx_fprime`, whose bound is the length of the parameter
#: vector.
#:
#: **The calling convention, stated once for the whole package.** Bulk data —
#: a signal, a sample, a design matrix, a point set — is a `jainum.NDArray`,
#: because it lives on the device and is the thing that could be large. A
#: parameter vector, a set of filter coefficients and a small result are
#: `list[float]`, because they are short, they are read one element at a time
#: by host arithmetic, and rounding them to float32 would throw away the digits
#: a solver depends on. A scalar objective is `fn(list[float]) -> float`. A
#: model evaluated over data is `fn(NDArray, list[float]) -> NDArray`: the data
#: first because it is the operand, the parameters second because they are the
#: thing being fitted. `as_array`, `host`, `device_vector` and `host_matrix`
#: below are the four crossings between the two halves, and a module that
#: writes its own crossing has almost certainly got the precision wrong.
#:
#: **The kernel idiom.** A module writes
#:
#: ```
#: let SOURCE = jainum.PRELUDE + """
#: kernel void sci_something(...) { ... }
#: """
#: kernel_of(SOURCE, "sci_something").dispatch_async(buffers, threads, scalars)
#: ```
#:
#: `kernel_of` caches on the ENTRY NAME ALONE — it never looks at the source,
#: because hashing a thirty-kilobyte shader costs more than the dispatch it
#: precedes. Two modules that both name an entry `reduce` therefore get one
#: pipeline and wrong answers with no error anywhere. So every entry point in
#: jaisci carries its module's prefix, and the prefixes are reserved here:
#:
#: `sci_` core, `sla_` linalg, `ssp_` sparse, `ssk_` sparse_solve, `sop_`
#: minimize, `sro_` roots, `sfi_` fitting, `slp_` linprog, `squ_` integrate,
#: `sod_` ode, `sit_` interpolate, `snd_` interpnd, `scv_` convolve, `ssg_`
#: spectral, `sfl_` filters, `sdi_` distributions, `sst_` stats, `ssa_`
#: spatial, `sgo_` geometry. None of them collides with jainum's `nd_`, `nf_`,
#: `nr_`, `nc_`, `nds_`, `nm_` or `nl_`.
#:
#: **What is deliberately not here.** No array type, no creation helpers, no
#: reductions, no sorts, no FFT: `jainum` owns all of those and jaisci imports
#: them. The banded solve and the band layout are `linalg.jai`'s. `OdeResult`
#: is `ode.jai`'s, `TestResult`, `LinregressResult` and `DescribeResult` are
#: `stats.jai`'s, `LinprogResult` is `linprog.jai`'s — a record belongs in the
#: file that fills it unless more than one module returns it, which is true
#: only of the two below.
#:
#: ```
#: let x = jainum.linspace(0.0, 1.0, 4096)
#: defer { x.free() }
#: let y = arr_exp(x)
#: defer { y.free() }
#: let gradient = approx_fprime([1.0, 2.0], rosenbrock)
#: ```
import std.gpu as gpu
import std.math
from std.core import Printable

from jainum.array import FLOAT32, NDArray, PRELUDE, f32_bits, kernel_of, require_real
from jainum.create import array as as_nd, from_list, zeros

#: The package version, which `jaithon.package.json` also carries.
pub let VERSION = "0.1.0"

#: Side of the output block one `sci_matmul` threadgroup owns.
#:
#: 16 squared is 256 threads, one output each, and 2 KB of staged operands —
#: small enough that occupancy is never the limit and large enough that each
#: staged value is read sixteen times. jainum's `mm_tile` goes further with a
#: 4x4 accumulator per thread; this one is deliberately the simple version,
#: because the products jaisci takes are normal equations and covariances of a
#: few hundred rows rather than the training GEMMs that tuning pays for.
pub const TILE = 16

#: Whether the machine has a GPU backend that jaisci can use.
#:
#: The `dense_*` factorisations, `approx_fprime` and every scalar solver run
#: without one. Anything taking or returning an `NDArray` does not.
pub fn is_available() -> bool { return gpu.is_available() }

#: Raise unless there is a device, naming what the caller was trying to do.
#:
#: Every module's device-only entry point opens with this rather than letting
#: the failure surface as jainum's "needs a GPU to allocate an NDArray", which
#: says nothing about which jaisci call the user made.
pub fn require_gpu(what: str) -> void {
    if not gpu.is_available() { throw RuntimeError(f"jaisci needs a GPU to {what}") }
}

let SOURCE = PRELUDE + """
constexpr constant uint SCI_TILE = 16u;

/* op: 0 exp, 1 log, 2 sqrt, 3 absolute, 4 raise to `first`, 5 reciprocal,
 * 6 sign, 7 clamp between `first` and `second`, 8 lower-bound at `first`,
 * 9 upper-bound at `first`.
 *
 * One entry point with a selector rather than ten. The branch is uniform
 * across the grid and disappears next to the load and the store, while ten
 * entry points would be ten pipelines to compile the first time any of them
 * is touched. The two float parameters arrive as bits because
 * `Kernel.dispatch` carries only `list[int]`.
 *
 * Both bounds are explicit ops instead of a clamp against an infinity: an
 * infinity has to survive `f32_bits` and the caller's arithmetic to get here,
 * and a one-sided bound is the common case anyway. */
kernel void sci_unary(device float* out [[buffer(0)]],
                      device const float* src [[buffer(1)]],
                      constant uint& count [[buffer(2)]],
                      constant uint& op [[buffer(3)]],
                      constant uint& first [[buffer(4)]],
                      constant uint& second [[buffer(5)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float x = src[id];
    float p = as_type<float>(first);
    float q = as_type<float>(second);
    float r = exp(x);
    if (op == 1u) r = log(x);
    else if (op == 2u) r = sqrt(x);
    else if (op == 3u) r = fabs(x);
    else if (op == 4u) r = pow(x, p);
    else if (op == 5u) r = 1.0f / x;
    else if (op == 6u) r = sign(x);
    else if (op == 7u) r = clamp(x, p, q);
    else if (op == 8u) r = fmax(x, p);
    else if (op == 9u) r = fmin(x, p);
    out[id] = r;
}

/* op: 0 maximum, 1 minimum, 2 `a` raised to `b`. Shapes match exactly --
 * nothing here broadcasts, because the caller that wants broadcasting wants
 * jainum's `binary_arith`. */
kernel void sci_binary(device float* out [[buffer(0)]],
                       device const float* a [[buffer(1)]],
                       device const float* b [[buffer(2)]],
                       constant uint& count [[buffer(3)]],
                       constant uint& op [[buffer(4)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float x = a[id];
    float y = b[id];
    float r = fmax(x, y);
    if (op == 1u) r = fmin(x, y);
    else if (op == 2u) r = pow(x, y);
    out[id] = r;
}

/* Tiled GEMM: `out` is rows x columns, `a` is rows x inner, `b` is inner x
 * columns, all dense and row-major, all indexed from slot zero.
 *
 * The grid is one-dimensional because `Kernel.dispatch` is, so the group
 * index is unpacked against `across` -- the number of tiles along a row of
 * the output -- and the thread index against SCI_TILE. Every thread walks the
 * whole k loop whether or not its output is in range, which is what keeps the
 * two barriers uniform; only the final store is guarded.
 *
 * Staging is what makes this worth writing at all: each step reads
 * 2 * SCI_TILE values per thread from device memory and does SCI_TILE
 * multiplies with them, so the staged tile is read sixteen times for each
 * time it is written. */
kernel void sci_matmul(device float* out [[buffer(0)]],
                       device const float* a [[buffer(1)]],
                       device const float* b [[buffer(2)]],
                       constant uint& rows [[buffer(3)]],
                       constant uint& inner [[buffer(4)]],
                       constant uint& columns [[buffer(5)]],
                       constant uint& across [[buffer(6)]],
                       uint wid [[threadgroup_position_in_grid]],
                       uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float a_tile[SCI_TILE * SCI_TILE];
    threadgroup float b_tile[SCI_TILE * SCI_TILE];

    uint tile_row = wid / across;
    uint tile_column = wid - tile_row * across;
    uint local_row = lid / SCI_TILE;
    uint local_column = lid - local_row * SCI_TILE;
    uint row = tile_row * SCI_TILE + local_row;
    uint column = tile_column * SCI_TILE + local_column;

    float total = 0.0f;
    for (uint step = 0u; step < inner; step += SCI_TILE) {
        uint from_a = step + local_column;
        uint from_b = step + local_row;
        a_tile[lid] = (row < rows && from_a < inner) ? a[row * inner + from_a] : 0.0f;
        b_tile[lid] = (from_b < inner && column < columns)
            ? b[from_b * columns + column]
            : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0u; k < SCI_TILE; k++) {
            total = fma(
                a_tile[local_row * SCI_TILE + k],
                b_tile[k * SCI_TILE + local_column],
                total
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < rows && column < columns) out[row * columns + column] = total;
}
"""

#: Whatever the caller passed, as an `NDArray`.
#:
#: An `NDArray` comes back unchanged, a `list[float]` becomes a 1-d device
#: array and a `list[list[float]]` a 2-d one. The asymmetry is the trap: the
#: result aliases the argument in the first case and is a fresh allocation the
#: caller owns in the other two, so `defer { as_array(value).free() }` frees
#: somebody else's array half the time. Ask `value is NDArray` first, or keep
#: the flag `jainum.asarray` callers keep.
pub fn as_array(value: any) -> NDArray {
    if value is NDArray { return value }
    if value is list {
        require_gpu("read a list as an array")
        return as_nd(value, FLOAT32)
    }
    throw TypeError(f"as_array takes an NDArray or a list of floats, not {type_of(value)}")
}

#: The values as host doubles, in element order.
#:
#: A `list[float]` passes straight through, so a function that accepts either
#: half of the calling convention can end with one `host` and stop caring.
#: Downloading an `NDArray` is a device synchronisation and a copy — do it
#: once at the boundary, never inside a loop.
pub fn host(a: any) -> list[float] {
    if a is NDArray { return a.to_list() }
    if a is list {
        var out: list[float] = []
        for value in a { out.push(float(value)) }
        return out
    }
    throw TypeError(f"host takes an NDArray or a list of floats, not {type_of(a)}")
}

#: A matrix as row-major host doubles, with its two extents.
#:
#: Rank two exactly. A vector has two readings here — one row or one column —
#: and every caller that guessed would guess differently, so `reshape` it and
#: say which.
pub fn host_matrix(a: NDArray) -> (list[float], int, int) {
    if a.ndim() != 2 { throw ValueError(f"host_matrix needs a matrix, got shape {a.shape}") }
    return (a.to_list(), a.shape[0], a.shape[1])
}

#: A `list[float]` uploaded as a 1-d array the caller owns.
pub fn device_vector(values: list[float]) -> NDArray {
    require_gpu("upload a vector")
    return from_list(values, [values.len()])
}

#: A row-major `list[float]` uploaded as a `rows x cols` array the caller owns.
pub fn device_matrix(values: list[float], rows: int, cols: int) -> NDArray {
    require_gpu("upload a matrix")
    if values.len() != rows * cols {
        throw ValueError(f"{values.len()} values are not a {rows}x{cols} matrix")
    }
    return from_list(values, [rows, cols])
}

#: The array itself when it is already dense from slot zero, a copy otherwise,
#: with a flag saying which so the caller knows whether to free it.
fn dense_of(a: NDArray) -> (NDArray, bool) {
    let packed = a.contiguous()
    return (packed, not packed == a)
}

fn unary(a: NDArray, op: int, first: float, second: float, what: str) -> NDArray {
    require_real(a, what)
    let (src, owned) = dense_of(a)
    let out = NDArray(a.shape, FLOAT32)
    let count = out.size()
    if count > 0 {
        kernel_of(SOURCE, "sci_unary").dispatch_async(
            [out.buffer(), src.buffer()],
            count,
            [count, op, f32_bits(first), f32_bits(second)]
        )
    }
    if owned { src.free() }
    return out
}

fn binary(a: NDArray, b: NDArray, op: int, what: str) -> NDArray {
    require_real(a, what)
    require_real(b, what)
    if a.shape != b.shape {
        throw ValueError(f"{what} needs matching shapes, got {a.shape} and {b.shape}")
    }
    let (left, owns_left) = dense_of(a)
    let (right, owns_right) = dense_of(b)
    let out = NDArray(a.shape, FLOAT32)
    let count = out.size()
    if count > 0 {
        kernel_of(SOURCE, "sci_binary").dispatch_async(
            [out.buffer(), left.buffer(), right.buffer()],
            count,
            [count, op]
        )
    }
    if owns_left { left.free() }
    if owns_right { right.free() }
    return out
}

#: Elementwise `e ** x`.
#:
#: FOLLOWUP: delete when jainum's `ufunc` lands `exp`, and re-export that.
pub fn arr_exp(a: NDArray) -> NDArray { return unary(a, 0, 0.0, 0.0, "arr_exp") }

#: Elementwise natural logarithm. A non-positive element gives -inf or NaN, as
#: float32 does.
#:
#: FOLLOWUP: delete when jainum's `ufunc` lands `log`, and re-export that.
pub fn arr_log(a: NDArray) -> NDArray { return unary(a, 1, 0.0, 0.0, "arr_log") }

#: Elementwise square root.
#:
#: FOLLOWUP: delete when jainum's `ufunc` lands `sqrt`, and re-export that.
pub fn arr_sqrt(a: NDArray) -> NDArray { return unary(a, 2, 0.0, 0.0, "arr_sqrt") }

#: Elementwise absolute value.
#:
#: FOLLOWUP: delete when jainum's `ufunc` lands `absolute`, and re-export that.
pub fn arr_abs(a: NDArray) -> NDArray { return unary(a, 3, 0.0, 0.0, "arr_abs") }

#: Every element raised to `exponent`, which is a scalar or an array of the
#: same shape.
#:
#: A negative base with a fractional exponent is NaN, because Metal's `pow` is
#: the real-valued one. Squaring is `multiply`, not this.
#:
#: FOLLOWUP: delete when jainum's `ufunc` lands `power`, and re-export that.
pub fn arr_pow(a: NDArray, exponent: any) -> NDArray {
    if exponent is NDArray { return binary(a, exponent, 2, "arr_pow") }
    return unary(a, 4, float(exponent), 0.0, "arr_pow")
}

#: Elementwise `1 / x`. A zero gives an infinity.
#:
#: FOLLOWUP: delete when jainum's `ufunc` lands `reciprocal`, and re-export it.
pub fn arr_recip(a: NDArray) -> NDArray { return unary(a, 5, 0.0, 0.0, "arr_recip") }

#: Elementwise -1, 0 or 1. A NaN gives zero, which is Metal's reading.
#:
#: FOLLOWUP: delete when jainum's `ufunc` lands `sign`, and re-export that.
pub fn arr_sign(a: NDArray) -> NDArray { return unary(a, 6, 0.0, 0.0, "arr_sign") }

#: Every element held between the bounds given. Either bound may be null, and
#: at least one must not be.
#:
#: FOLLOWUP: delete when jainum's `ufunc` lands `clip`, and re-export that.
pub fn arr_clip(a: NDArray, low: float? = null, high: float? = null) -> NDArray {
    if low is not null and high is not null {
        if low > high { throw ValueError(f"arr_clip bounds cross: {low} is above {high}") }
        return unary(a, 7, low, high, "arr_clip")
    }
    if low is not null { return unary(a, 8, low, 0.0, "arr_clip") }
    if high is not null { return unary(a, 9, high, 0.0, "arr_clip") }
    throw ValueError("arr_clip needs at least one bound")
}

#: Elementwise larger of an array and either a scalar or a matching array.
#:
#: FOLLOWUP: delete when jainum's `ufunc` lands `maximum`, and re-export that.
pub fn arr_maximum(a: NDArray, other: any) -> NDArray {
    if other is NDArray { return binary(a, other, 0, "arr_maximum") }
    return unary(a, 8, float(other), 0.0, "arr_maximum")
}

#: Elementwise smaller of an array and either a scalar or a matching array.
#:
#: FOLLOWUP: delete when jainum's `ufunc` lands `minimum`, and re-export that.
pub fn arr_minimum(a: NDArray, other: any) -> NDArray {
    if other is NDArray { return binary(a, other, 1, "arr_minimum") }
    return unary(a, 9, float(other), 0.0, "arr_minimum")
}

#: `out = a b`, with every operand read as dense row-major from slot zero.
#:
#: Metal Performance Shaders takes the product when every physical row is a
#: multiple of sixteen bytes — the columns of `a` and of the result each
#: divisible by four — and the result is big enough to pay for the encode.
#: An `MPSNDArray` pads its innermost dimension out to a 16-byte row and reads
#: the wrong elements when it does not divide, which is why the test is on the
#: strides rather than on the size alone. Otherwise the tiled kernel runs.
fn gemm(a: NDArray, b: NDArray, out: NDArray, rows: int, inner: int, columns: int) -> void {
    if rows * columns == 0 or inner == 0 { return }
    let (left, owns_left) = dense_of(a)
    let (right, owns_right) = dense_of(b)
    if inner % 4 == 0 and columns % 4 == 0 and rows * columns >= gpu.MIN_GPU_ELEMENTS {
        gpu.matmul_async(left.buffer(), right.buffer(), out.buffer(), rows, inner, columns)
    } else {
        let across = (columns + TILE - 1) // TILE
        let down = (rows + TILE - 1) // TILE
        kernel_of(SOURCE, "sci_matmul").dispatch_async(
            [out.buffer(), left.buffer(), right.buffer()],
            down * across * TILE * TILE,
            [rows, inner, columns, across],
            group_size: TILE * TILE
        )
    }
    if owns_left { left.free() }
    if owns_right { right.free() }
}

#: Matrix product of two matrices.
#:
#: The only O(n^3) primitive in the package, and the one linalg, fitting, stats
#: and spatial all reach for: normal equations are `matmul(transpose(x), x)`, a
#: covariance is that scaled, a Gram matrix is that again, and a projection is
#: one more product. Everything else in jaisci that looks like a matrix product
#: — a banded solve, a sparse product, a Householder sweep — is deliberately
#: not one, because the zeros outnumber the values.
pub fn matmul(a: NDArray, b: NDArray) -> NDArray {
    require_real(a, "matmul")
    require_real(b, "matmul")
    if a.ndim() != 2 or b.ndim() != 2 {
        throw ValueError(f"matmul needs two matrices, got {a.shape} and {b.shape}")
    }
    let rows = a.shape[0]
    let inner = a.shape[1]
    let columns = b.shape[1]
    if b.shape[0] != inner { throw ValueError(f"cannot multiply {a.shape} by {b.shape}") }
    let out = zeros([rows, columns])
    gemm(a, b, out, rows, inner, columns)
    return out
}

#: `a x` for a matrix and a vector, as a 1-d array.
#:
#: `x` is an `NDArray` or a `list[float]`; a list is uploaded and freed here,
#: because a right-hand side short enough to be written as a list is short
#: enough that the upload is the cheap part of the call.
pub fn matvec(a: NDArray, x: any) -> NDArray {
    require_real(a, "matvec")
    if a.ndim() != 2 { throw ValueError(f"matvec needs a matrix, got shape {a.shape}") }
    let vector = as_array(x)
    let owned = not vector == x
    defer { if owned { vector.free() } }
    if vector.ndim() != 1 { throw ValueError(f"matvec needs a vector, got shape {vector.shape}") }
    let rows = a.shape[0]
    let inner = a.shape[1]
    if vector.shape[0] != inner {
        throw ValueError(f"cannot apply {a.shape} to a length-{vector.shape[0]} vector")
    }
    let out = zeros([rows])
    gemm(a, vector, out, rows, inner, 1)
    return out
}

#: Inner product of two vectors, as a host double.
#:
#: Two lists fold on the host in f64, which is both the accurate answer and the
#: only one available: a list is already host memory, so there is nothing to
#: dispatch over. Anything involving an `NDArray` goes through the GEMM as a
#: 1 x n by n x 1 product and syncs to read the one value back.
#:
#: That product leaves a single thread walking the inner axis, which is fine
#: for the gradients, search directions and residuals that reach it and wrong
#: for a vector of a million elements. Sum a `multiply` with `jainum.sum` if
#: the vector is that long; a dot product is O(n) and this file's rule is that
#: O(n) is not what a kernel is for.
pub fn dot(a: any, b: any) -> float {
    if a is list and b is list {
        if a.len() != b.len() {
            throw ValueError(f"dot needs matching lengths, got {a.len()} and {b.len()}")
        }
        var total = 0.0
        for (index, value) in a.enumerate() { total += float(value) * float(b[index]) }
        return total
    }
    let x = as_array(a)
    let owns_x = not x == a
    defer { if owns_x { x.free() } }
    let y = as_array(b)
    let owns_y = not y == b
    defer { if owns_y { y.free() } }
    if x.ndim() != 1 or y.ndim() != 1 {
        throw ValueError(f"dot needs two vectors, got {x.shape} and {y.shape}")
    }
    if x.shape[0] != y.shape[0] {
        throw ValueError(f"dot needs matching lengths, got {x.shape[0]} and {y.shape[0]}")
    }
    let out = zeros([1])
    defer { out.free() }
    gemm(x, y, out, 1, x.shape[0], 1)
    return out.item()
}

#: LU with partial pivoting, in float64 on the host.
#:
#: The whole `dense_*` family below takes a row-major `list[float]` and its
#: dimensions rather than an `NDArray`, and that is the precision policy
#: showing through: a pivot divided in float32 loses half its digits, and these
#: run at sizes — a spline system, a Jacobian, a covariance — where the device
#: would lose the launch overhead race anyway. jaicv's `core/linalg.jai` made
#: the same call for the same reason.
#:
#: Returns the combined factors (`L` below the diagonal with its implied unit
#: diagonal, `U` on and above), the row permutation, and the sign of that
#: permutation — `1` or `-1`, and `0` when a column was exactly zero and the
#: matrix is singular. `dense_lu_solve` consumes all three.
pub fn dense_lu(a: list[float], n: int) -> tuple[list[float], list[int], int] {
    if a.len() != n * n { throw ValueError(f"{a.len()} values are not an {n}x{n} matrix") }
    var lu = [value for value in a]
    var pivots: list[int] = []
    for index in 0..n { pivots.push(index) }
    var sign = 1
    for column in 0..n {
        var best = column
        var largest = math.fabs(lu[column * n + column])
        let after = column + 1
        for row in after..n {
            let size = math.fabs(lu[row * n + column])
            if size > largest {
                largest = size
                best = row
            }
        }
        if largest == 0.0 {
            sign = 0
            continue
        }
        if best != column {
            for index in 0..n {
                let held = lu[column * n + index]
                lu[column * n + index] = lu[best * n + index]
                lu[best * n + index] = held
            }
            let held = pivots[column]
            pivots[column] = pivots[best]
            pivots[best] = held
            sign = -sign
        }
        let diagonal = lu[column * n + column]
        for row in after..n {
            let factor = lu[row * n + column] / diagonal
            lu[row * n + column] = factor
            for index in after..n { lu[row * n + index] -= factor * lu[column * n + index] }
        }
    }
    return (lu, pivots, sign)
}

#: Solve with the factors `dense_lu` produced. `b` is row-major, `nrhs` wide.
pub fn dense_lu_solve(
    lu: list[float],
    pivots: list[int],
    b: list[float],
    n: int,
    nrhs: int = 1
) -> list[float] {
    if lu.len() != n * n { throw ValueError(f"{lu.len()} values are not an {n}x{n} factor") }
    if b.len() != n * nrhs { throw ValueError(f"{b.len()} values are not {n}x{nrhs}") }
    var x: list[float] = []
    for row in 0..n {
        for column in 0..nrhs { x.push(b[pivots[row] * nrhs + column]) }
    }
    for row in 0..n {
        for column in 0..nrhs {
            var total = x[row * nrhs + column]
            for k in 0..row { total -= lu[row * n + k] * x[k * nrhs + column] }
            x[row * nrhs + column] = total
        }
    }
    var row = n - 1
    while row >= 0 {
        let diagonal = lu[row * n + row]
        if diagonal == 0.0 { throw ValueError("the matrix is singular to working precision") }
        let after = row + 1
        for column in 0..nrhs {
            var total = x[row * nrhs + column]
            for k in after..n { total -= lu[row * n + k] * x[k * nrhs + column] }
            x[row * nrhs + column] = total / diagonal
        }
        row -= 1
    }
    return x
}

#: Cholesky factor `L`, lower triangular, with `L L^T = a`.
#:
#: Raises when a pivot is not positive, which is the cheapest test there is for
#: "this matrix is not positive definite" and the one every caller wants: a
#: covariance that has gone indefinite, a Hessian that is not a minimum.
pub fn dense_cholesky(a: list[float], n: int) -> list[float] {
    if a.len() != n * n { throw ValueError(f"{a.len()} values are not an {n}x{n} matrix") }
    var lower: list[float] = []
    for _index in 0..n * n { lower.push(0.0) }
    for row in 0..n {
        let through = row + 1
        for column in 0..through {
            var total = a[row * n + column]
            for k in 0..column { total -= lower[row * n + k] * lower[column * n + k] }
            if row != column {
                lower[row * n + column] = total / lower[column * n + column]
                continue
            }
            if total <= 0.0 {
                throw ValueError(
                    f"dense_cholesky needs a positive definite matrix, pivot {row} is {total}"
                )
            }
            lower[row * n + column] = math.sqrt(total)
        }
    }
    return lower
}

#: Thin QR by Householder reflections: `q` is `rows x cols`, `r` is `cols x cols`.
#:
#: Householder rather than Gram-Schmidt because the columns a least-squares fit
#: hands over are usually close to dependent — a polynomial design matrix, a
#: Jacobian near a plateau — and modified Gram-Schmidt loses orthogonality
#: exactly there.
pub fn dense_qr(a: list[float], rows: int, cols: int) -> tuple[list[float], list[float]] {
    if a.len() != rows * cols { throw ValueError(f"{a.len()} values are not {rows}x{cols}") }
    if rows < cols { throw ValueError(f"dense_qr needs rows >= cols, got {rows}x{cols}") }
    var upper = [value for value in a]
    var q: list[float] = []
    for row in 0..rows {
        for column in 0..rows { q.push(if row == column { 1.0 } else { 0.0 }) }
    }
    var reflector: list[float] = []
    for _index in 0..rows { reflector.push(0.0) }
    for column in 0..cols {
        var norm = 0.0
        for row in column..rows {
            let value = upper[row * cols + column]
            norm += value * value
        }
        norm = math.sqrt(norm)
        if norm == 0.0 { continue }
        let head = upper[column * cols + column]
        #: Reflect away from the head rather than towards it: subtracting two
        #: nearly equal numbers here is the classic way to lose the column.
        let alpha = if head >= 0.0 { -norm } else { norm }
        for row in 0..rows { reflector[row] = 0.0 }
        reflector[column] = head - alpha
        let below = column + 1
        for row in below..rows { reflector[row] = upper[row * cols + column] }
        var scale_ = 0.0
        for row in column..rows { scale_ += reflector[row] * reflector[row] }
        if scale_ == 0.0 { continue }
        for index in column..cols {
            var projection = 0.0
            for row in column..rows { projection += reflector[row] * upper[row * cols + index] }
            let factor = 2.0 * projection / scale_
            for row in column..rows { upper[row * cols + index] -= factor * reflector[row] }
        }
        for row in 0..rows {
            var projection = 0.0
            for k in column..rows { projection += q[row * rows + k] * reflector[k] }
            let factor = 2.0 * projection / scale_
            for k in column..rows { q[row * rows + k] -= factor * reflector[k] }
        }
    }
    var thin: list[float] = []
    for row in 0..rows {
        for column in 0..cols { thin.push(q[row * rows + column]) }
    }
    var triangle: list[float] = []
    for row in 0..cols {
        for column in 0..cols {
            triangle.push(if row <= column { upper[row * cols + column] } else { 0.0 })
        }
    }
    return (thin, triangle)
}

#: Solve a triangular system by substitution.
#:
#: `unit` reads the diagonal as ones without looking at it, which is how the
#: `L` inside `dense_lu`'s combined factor is stored.
pub fn dense_triangular(
    a: list[float],
    b: list[float],
    n: int,
    lower: bool,
    unit: bool = false
) -> list[float] {
    if a.len() != n * n { throw ValueError(f"{a.len()} values are not an {n}x{n} matrix") }
    if b.len() != n { throw ValueError(f"{b.len()} values are not a length-{n} vector") }
    var x = [value for value in b]
    if lower {
        for row in 0..n {
            var total = x[row]
            for k in 0..row { total -= a[row * n + k] * x[k] }
            x[row] = if unit { total } else { total / a[row * n + row] }
        }
        return x
    }
    var row = n - 1
    while row >= 0 {
        var total = x[row]
        let after = row + 1
        for k in after..n { total -= a[row * n + k] * x[k] }
        x[row] = if unit { total } else { total / a[row * n + row] }
        row -= 1
    }
    return x
}

#: Thomas's algorithm: an `n x n` tridiagonal solve in linear time.
#:
#: `lower` and `upper` are the off-diagonals, `n - 1` long each. No pivoting —
#: the systems that reach here are the diagonally dominant ones a spline or a
#: heat equation produces, where pivoting never triggers, and a caller who
#: cannot promise that should assemble a band and call `linalg.solve_banded`.
pub fn dense_tridiagonal(
    lower: list[float],
    diagonal: list[float],
    upper: list[float],
    rhs: list[float]
) -> list[float] {
    let n = diagonal.len()
    if n == 0 { return [] }
    if lower.len() != n - 1 or upper.len() != n - 1 {
        throw ValueError(f"a {n}-row tridiagonal needs {n - 1} off-diagonal values on each side")
    }
    if rhs.len() != n { throw ValueError(f"{rhs.len()} values are not a length-{n} vector") }
    var sweep = [value for value in upper]
    var solution = [value for value in rhs]
    var pivot = diagonal[0]
    if pivot == 0.0 { throw ValueError("dense_tridiagonal met a zero pivot") }
    if n > 1 { sweep[0] = upper[0] / pivot }
    solution[0] = rhs[0] / pivot
    for row in 1..n {
        pivot = diagonal[row] - lower[row - 1] * sweep[row - 1]
        if pivot == 0.0 { throw ValueError(f"dense_tridiagonal met a zero pivot at row {row}") }
        if row < n - 1 { sweep[row] = upper[row] / pivot }
        solution[row] = (rhs[row] - lower[row - 1] * solution[row - 1]) / pivot
    }
    var row = n - 2
    while row >= 0 {
        solution[row] -= sweep[row] * solution[row + 1]
        row -= 1
    }
    return solution
}

#: The Givens rotation that zeros `b` against `a`, as `(cosine, sine, radius)`.
#:
#: `hypot` rather than `sqrt(a*a + b*b)`: the QR updates that use this run on
#: residuals that can be tiny, and squaring them underflows long before the
#: rotation is meaningless.
pub fn givens(a: float, b: float) -> tuple[float, float, float] {
    if b == 0.0 { return (1.0, 0.0, a) }
    if a == 0.0 { return (0.0, 1.0, b) }
    let magnitude = math.hypot(a, b)
    let radius = if math.fabs(a) > math.fabs(b) {
        math.copysign(magnitude, a)
    } else {
        math.copysign(magnitude, b)
    }
    return (a / radius, b / radius, radius)
}

#: Forward-difference gradient of `f` at `x`.
#:
#: The step is `eps` scaled by the size of the coordinate rather than scipy's
#: flat absolute step: a parameter of order a thousand and one of order a
#: thousandth turn up in the same vector all the time — a rate beside an
#: amplitude — and one absolute step cannot serve both. The default `eps` is
#: the square root of float64 epsilon, which is where truncation error and
#: cancellation error meet for a first difference.
#:
#: `n + 1` evaluations. A method that can afford `2n` should use central
#: differences instead; this is the one every optimiser falls back to.
pub fn approx_fprime(
    x: list[float],
    f: fn(list[float]) -> float,
    eps: float = 1.4901161193847656e-08
) -> list[float] {
    let base = f(x)
    var out: list[float] = []
    for index in 0..x.len() {
        var shifted = [value for value in x]
        let size = math.fabs(x[index])
        let step = eps * (if size > 1.0 { size } else { 1.0 })
        shifted[index] = x[index] + step
        out.push((f(shifted) - base) / step)
    }
    return out
}

#: Forward-difference Jacobian of a vector residual, row-major and `m x n`.
#:
#: Row `i` is the gradient of residual `i`, so the layout is the one
#: `dense_qr` and the normal equations both want and the one
#: `least_squares` hands to its trust region without a transpose. The step
#: rule is `approx_fprime`'s, for the same reason.
#:
#: `n + 1` evaluations of `residual`, each returning `m` values. A residual
#: that is itself a device pass should evaluate all of `m` in one dispatch and
#: come back as a list — `n + 1` dispatches is the budget, not `m (n + 1)`.
pub fn approx_jacobian(
    x: list[float],
    residual: fn(list[float]) -> list[float],
    eps: float = 1.4901161193847656e-08
) -> list[float] {
    let base = residual(x)
    let m = base.len()
    let n = x.len()
    var columns: list[list[float]] = []
    for index in 0..n {
        var shifted = [value for value in x]
        let size = math.fabs(x[index])
        let step = eps * (if size > 1.0 { size } else { 1.0 })
        shifted[index] = x[index] + step
        let moved = residual(shifted)
        if moved.len() != m {
            throw ValueError(f"the residual returned {moved.len()} values, then {m}")
        }
        var column: list[float] = []
        for row in 0..m { column.push((moved[row] - base[row]) / step) }
        columns.push(column)
    }
    var out: list[float] = []
    for row in 0..m {
        for index in 0..n { out.push(columns[index][row]) }
    }
    return out
}

#: What every minimiser in this package returns, named as scipy names it.
#:
#: `status` is `0` for a converged run and an algorithm-specific code
#: otherwise; `message` says the same thing in words. `jac` is empty when the
#: method never formed a gradient.
pub class OptimizeResult: Printable {
    #: The parameter vector at the point the search stopped.
    pub var x: list[float]
    #: The objective there.
    pub var fun: float
    #: The gradient there, or empty when none was formed.
    pub var jac: list[float]
    #: Whether the stopping test was convergence rather than a limit.
    pub var success: bool
    #: Zero on success, an algorithm-specific code otherwise.
    pub var status: int
    #: The same verdict in words.
    pub var message: str
    #: Iterations taken.
    pub var nit: int
    #: Objective evaluations taken.
    pub var nfev: int

    pub fn init(
        self,
        x: list[float],
        fun: float,
        success: bool = true,
        status: int = 0,
        message: str = "",
        nit: int = 0,
        nfev: int = 0,
        jac: list[float] = []
    ) {
        self.x = x
        self.fun = fun
        self.jac = jac
        self.success = success
        self.status = status
        self.message = message
        self.nit = nit
        self.nfev = nfev
    }

    pub fn to_str(self) -> str {
        return f"OptimizeResult(fun={self.fun}, nit={self.nit}, success={self.success})"
    }

    fn __str__(self) -> str { return self.to_str() }
}

#: What a scalar root finder returns.
pub class RootResult: Printable {
    #: The root, or the best estimate when the search gave up.
    pub var root: float
    #: Iterations taken.
    pub var iterations: int
    #: Function evaluations taken.
    pub var function_calls: int
    #: Whether the bracket or the step closed to tolerance.
    pub var converged: bool
    #: A short reason: "converged", "max iterations", "no sign change".
    pub var flag: str

    pub fn init(
        self,
        root: float,
        iterations: int = 0,
        function_calls: int = 0,
        converged: bool = true,
        flag: str = "converged"
    ) {
        self.root = root
        self.iterations = iterations
        self.function_calls = function_calls
        self.converged = converged
        self.flag = flag
    }

    pub fn to_str(self) -> str {
        return f"RootResult(root={self.root}, converged={self.converged})"
    }

    fn __str__(self) -> str { return self.to_str() }
}

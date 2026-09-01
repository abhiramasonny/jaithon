#: The tolerances every factorization here is tuned against, and the small
#: host helpers they all reach for.
from ..array import FLOAT32, NDArray, require_real

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
pub const SINGULAR_RATIO = 1e-12

#: Off-diagonal weight, relative to the diagonal, at which Jacobi stops.
pub const JACOBI_TOL = 1e-15

#: Sweeps `eigh` will spend before it gives up.
#:
#: Cyclic Jacobi is quadratically convergent once the off-diagonal is small,
#: and 30 sweeps is far past where any conditioned matrix stops moving; the
#: cap is there so a NaN cannot spin forever.
pub const JACOBI_SWEEPS = 30

#: Iterations a QR sweep spends per column before it gives up.
pub const QR_ITERATIONS = 60

#: A stack of matrices, flat and row-major, with the batch shape they sit in.
pub fn _stacked(a: NDArray, what: str) -> (list[float], list[int], int, int) {
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
pub fn _emit(values: list[float], shape: list[int], dtype: int = FLOAT32) -> NDArray {
    let out = NDArray(shape, dtype)
    out.upload(values)
    return out
}

pub fn _require_square(m: int, n: int, what: str) -> void {
    if m != n { throw ValueError(f"{what} needs a square matrix, got {m} by {n}") }
}

#: The `rows` by `columns` leading identity, row-major.
pub fn _identity_block(rows: int, columns: int) -> list[float] {
    var out = [0.0 for _slot in 0..rows * columns]
    for index in 0..min(rows, columns) { out[index * columns + index] = 1.0 }
    return out
}

#: `a` read the other way round, `rows` by `columns` in.
pub fn _transpose_host(a: list[float], rows: int, columns: int) -> list[float] {
    var out = [0.0 for _slot in 0..rows * columns]
    for row in 0..rows {
        for column in 0..columns { out[column * rows + row] = a[row * columns + column] }
    }
    return out
}

#: `count` values of `values` starting at `at`, as a matrix of its own.
pub fn _slice_matrix(values: list[float], at: int, count: int) -> list[float] {
    var out = [0.0 for _slot in 0..count]
    for index in 0..count { out[index] = values[at + index] }
    return out
}

#: Append every value of `src` to `dst`.
pub fn _extend(dst: list[float], src: list[float]) -> void {
    for value in src { dst.push(value) }
}

#: `shape` with its last `drop` axes taken off — the batch a stack sits in.
pub fn _lead(shape: list[int], drop: int) -> list[int] {
    var out: list[int] = []
    for axis in 0..shape.len() - drop { out.push(shape[axis]) }
    return out
}

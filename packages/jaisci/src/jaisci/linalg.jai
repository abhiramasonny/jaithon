#: Dense linear algebra under scipy's names — `scipy.linalg`.
#:
#: Every routine here is a thin crossing over the factorisations in `core.jai`:
#: an `NDArray` comes in, `host_matrix` hands its values to a `dense_*` kernel
#: that works in host f64, and `device_matrix` puts the answer back. The
#: factorisations are the interesting code and they live there; this file is
#: the shape scipy callers expect, and the argument order and result tuples
#: follow scipy so a program translates line for line.
#:
#: Host f64 and not the device is deliberate for these. A dense LU is a
#: sequential elimination with a pivot search per column, which is latency
#: bound rather than bandwidth bound; the matrices that reach it are small
#: enough that a round trip would cost more than the arithmetic. The device is
#: for `matmul`, which is in `core.jai` and which these do not need.
import std.math

from .core import as_array, dense_cholesky, dense_lu, dense_lu_solve, dense_qr
from .core import dense_triangular, device_matrix, device_vector, host_matrix
from .core import require_gpu

from jainum import NDArray

#: A matrix and its extents, refusing a non-square one by name.
fn square_of(a: NDArray, what: str) -> (list[float], int) {
    let (values, rows, cols) = host_matrix(a)
    if rows != cols { throw ValueError(f"{what} needs a square matrix, got {rows}x{cols}") }
    return (values, rows)
}

#: The right-hand side as row-major values and its width.
#:
#: scipy takes either a vector or a matrix here and gives back what it was
#: given, so the width is carried rather than assumed and `solve` reshapes on
#: the way out.
fn rhs_of(b: any, n: int, what: str) -> (list[float], int, bool) {
    let array = as_array(b)
    let owned = not array == b
    defer { if owned { array.free() } }
    if array.ndim() == 1 {
        if array.shape[0] != n {
            throw ValueError(f"{what} got a length-{array.shape[0]} vector for an {n}x{n} matrix")
        }
        return (array.to_list(), 1, true)
    }
    if array.ndim() != 2 { throw ValueError(f"{what} needs a vector or a matrix on the right") }
    if array.shape[0] != n {
        throw ValueError(f"{what} got {array.shape[0]} rows for an {n}x{n} matrix")
    }
    return (array.to_list(), array.shape[1], false)
}

fn shaped(values: list[float], n: int, width: int, flat: bool) -> NDArray {
    if flat { return device_vector(values) }
    return device_matrix(values, n, width)
}

#: `LU` with its pivots, as `scipy.linalg.lu_factor` gives them.
#:
#: The combined factor is what `dense_lu` returns: `L` below the diagonal with
#: an implied unit diagonal, `U` on and above.
pub fn lu_factor(a: NDArray) -> (NDArray, list[int]) {
    require_gpu("factor a matrix")
    let (values, n) = square_of(a, "lu_factor")
    let (lu, pivots, sign) = dense_lu(values, n)
    if sign == 0 { throw ValueError("lu_factor was given a singular matrix") }
    return (device_matrix(lu, n, n), pivots)
}

#: Solve with the factors `lu_factor` produced.
pub fn lu_solve(factored: (NDArray, list[int]), b: any) -> NDArray {
    require_gpu("solve with a factor")
    let (lu_array, pivots) = factored
    let (lu, n) = square_of(lu_array, "lu_solve")
    let (rhs, width, flat) = rhs_of(b, n, "lu_solve")
    return shaped(dense_lu_solve(lu, pivots, rhs, n, width), n, width, flat)
}

#: `a x = b`, by LU with partial pivoting.
pub fn solve(a: NDArray, b: any) -> NDArray {
    require_gpu("solve a system")
    let (values, n) = square_of(a, "solve")
    let (lu, pivots, sign) = dense_lu(values, n)
    if sign == 0 { throw ValueError("solve was given a singular matrix") }
    let (rhs, width, flat) = rhs_of(b, n, "solve")
    return shaped(dense_lu_solve(lu, pivots, rhs, n, width), n, width, flat)
}

#: `a x = b` where `a` is triangular, without factoring anything.
pub fn solve_triangular(
    a: NDArray,
    b: any,
    lower: bool = false,
    unit_diagonal: bool = false
) -> NDArray {
    require_gpu("solve a triangular system")
    let (values, n) = square_of(a, "solve_triangular")
    let (rhs, width, flat) = rhs_of(b, n, "solve_triangular")
    if width != 1 {
        throw ValueError("solve_triangular takes one right-hand side at a time")
    }
    return shaped(dense_triangular(values, rhs, n, lower, unit_diagonal), n, 1, flat)
}

#: The Cholesky factor. `lower` picks which triangle carries it, as scipy does.
pub fn cholesky(a: NDArray, lower: bool = true) -> NDArray {
    require_gpu("factor a matrix")
    let (values, n) = square_of(a, "cholesky")
    let low = dense_cholesky(values, n)
    if lower { return device_matrix(low, n, n) }
    var upper: list[float] = []
    for row in 0..n {
        for column in 0..n { upper.push(low[column * n + row]) }
    }
    return device_matrix(upper, n, n)
}

#: `scipy.linalg.cho_factor`: the factor and which triangle holds it.
pub fn cho_factor(a: NDArray, lower: bool = true) -> (NDArray, bool) {
    return (cholesky(a, lower), lower)
}

#: Solve with the factor `cho_factor` produced.
#:
#: Two triangular solves, `L y = b` then `L^T x = y`, which is the whole point
#: of having the factor.
pub fn cho_solve(factored: (NDArray, bool), b: any) -> NDArray {
    require_gpu("solve with a factor")
    let (factor, lower) = factored
    let (values, n) = square_of(factor, "cho_solve")
    let (rhs, width, flat) = rhs_of(b, n, "cho_solve")
    if width != 1 { throw ValueError("cho_solve takes one right-hand side at a time") }
    var low = values
    if not lower {
        var flipped: list[float] = []
        for row in 0..n {
            for column in 0..n { flipped.push(values[column * n + row]) }
        }
        low = flipped
    }
    var upper: list[float] = []
    for row in 0..n {
        for column in 0..n { upper.push(low[column * n + row]) }
    }
    let forward = dense_triangular(low, rhs, n, true, false)
    return shaped(dense_triangular(upper, forward, n, false, false), n, 1, flat)
}

#: `Q` and `R` for a tall or square matrix — scipy's `qr(a, mode="economic")`.
#:
#: Thin, not full: `Q` is `rows x cols` and `R` is `cols x cols`, because
#: `dense_qr` never forms the columns of `Q` that span the orthogonal
#: complement. Those columns cost `rows x (rows - cols)` to store and nothing
#: here reads them; scipy's default `mode="full"` would return them and is
#: deliberately not what this is.
pub fn qr(a: NDArray) -> (NDArray, NDArray) {
    require_gpu("factor a matrix")
    let (values, rows, cols) = host_matrix(a)
    let (q, r) = dense_qr(values, rows, cols)
    return (device_matrix(q, rows, cols), device_matrix(r, cols, cols))
}

#: The determinant, from the LU factor's diagonal and the permutation's sign.
pub fn det(a: NDArray) -> float {
    let (values, n) = square_of(a, "det")
    let (lu, _pivots, sign) = dense_lu(values, n)
    if sign == 0 { return 0.0 }
    var total = float(sign)
    for index in 0..n { total *= lu[index * n + index] }
    return total
}

#: The inverse, by solving against the identity one column at a time.
#:
#: scipy has `inv` and so this does, but a caller who wants `inv(a) @ b` wants
#: `solve(a, b)` instead: one factorisation and one back-substitution rather
#: than `n` of them and a product.
pub fn inv(a: NDArray) -> NDArray {
    require_gpu("invert a matrix")
    let (values, n) = square_of(a, "inv")
    let (lu, pivots, sign) = dense_lu(values, n)
    if sign == 0 { throw ValueError("inv was given a singular matrix") }
    var identity: list[float] = []
    for row in 0..n {
        for column in 0..n { identity.push(if row == column { 1.0 } else { 0.0 }) }
    }
    return device_matrix(dense_lu_solve(lu, pivots, identity, n, n), n, n)
}

#: Least squares by QR: solve `R x = Q^T b`.
#:
#: `dense_qr` is thin, so `Q^T b` is `cols` values and the residual cannot be
#: read off the rows below them — those rows of `Q` were never formed. It comes
#: from Pythagoras instead: `b` splits into the part `Q` spans and the part it
#: does not, so the residual sum of squares is `|b|^2 - |Q^T b|^2`. Clamped at
#: zero because that subtraction of two nearly equal sums can land just below
#: it when the fit is exact.
pub fn lstsq(a: NDArray, b: any) -> (NDArray, float) {
    require_gpu("solve a least-squares system")
    let (values, rows, cols) = host_matrix(a)
    let (rhs, width, _flat) = rhs_of(b, rows, "lstsq")
    if width != 1 { throw ValueError("lstsq takes one right-hand side at a time") }
    let (q, r) = dense_qr(values, rows, cols)

    var projected: list[float] = []
    for column in 0..cols {
        var total = 0.0
        for row in 0..rows { total += q[row * cols + column] * rhs[row] }
        projected.push(total)
    }
    var squared = 0.0
    for value in rhs { squared += value * value }
    for value in projected { squared -= value * value }
    let residual = if squared > 0.0 { squared } else { 0.0 }
    return (device_vector(dense_triangular(r, projected, cols, false, false)), residual)
}

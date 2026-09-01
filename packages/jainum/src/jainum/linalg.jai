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
from .array import INT32, NDArray
from .array import element_count, require_real
from .linalg.common import BLOCKED_FROM, BLOCKED_PANEL, FLOAT32_EPS, blocked_panel
from .linalg.common import _emit, _extend, _require_square
from .linalg.common import _slice_matrix, _stacked
from .linalg.eigen import _eig_host, _jacobi_eigh, _symmetrize
from .linalg.norm import _as_number, norm
from .linalg.qr import _accumulate_reflectors, _householder_qr, _upper_triangle
from .linalg.solve import cholesky, det, inv, lu_factor, lu_solve, slogdet, solve, triangular_solve
from .linalg.svd import _svd_host

export { BLOCKED_FROM, BLOCKED_PANEL, FLOAT32_EPS, blocked_panel, norm }
export { cholesky, det, inv, lu_factor, lu_solve, slogdet, solve, triangular_solve }

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

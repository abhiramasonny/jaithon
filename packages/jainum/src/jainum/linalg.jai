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
from .array import FLOAT32, INT32, NDArray
from .array import broadcast_shapes, element_count, normalize_axis, normalize_axes, require_real
from .linalg.common import BLOCKED_FROM, BLOCKED_PANEL, FLOAT32_EPS, blocked_panel
from .linalg.common import _emit, _extend, _identity_block, _lead, _require_square
from .linalg.common import _slice_matrix, _stacked, _transpose_host
from .linalg.eigen import _eig_host, _jacobi_eigh, _symmetrize
from .linalg.lu import _cholesky_host, _lu_factor_host, _lu_solve_host, _triangular_host
from .linalg.qr import _accumulate_reflectors, _householder_qr, _upper_triangle
from .linalg.reduction import RED_COUNT, RED_MAX_ABS, RED_MIN_ABS, RED_SUM_ABS, RED_SUM_POW, RED_SUM_SQ
from .linalg.reduction import _kept_shape, _move_axes_last, _reduce_axes
from .linalg.svd import _svd_host

export { BLOCKED_FROM, BLOCKED_PANEL, FLOAT32_EPS, blocked_panel }

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

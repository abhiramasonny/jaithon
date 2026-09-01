#: The entry points that go through a triangular factorization: solves,
#: inverses and determinants.
import std.math
from ..array import INT32, NDArray, broadcast_shapes, element_count, require_real
from .common import _emit, _extend, _identity_block, _lead, _require_square
from .common import _slice_matrix, _stacked, _transpose_host
from .lu import _cholesky_host, _lu_factor_host, _lu_solve_host, _triangular_host

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

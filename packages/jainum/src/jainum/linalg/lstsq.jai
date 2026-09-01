#: What the singular values are for once they are in hand: pseudo-inverse,
#: rank, condition number and least squares.
import std.math
from ..array import INT32, NDArray, element_count, require_real
from .common import FLOAT32_EPS, _emit, _extend, _slice_matrix, _stacked
from .norm import _as_number, norm
from .solve import inv
from .svd import _svd_host

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

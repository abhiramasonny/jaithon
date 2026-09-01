#: The triangular factorizations: `PA = LU`, its solves, and Cholesky.
import std.math
from .common import SINGULAR_RATIO, blocked_panel

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
pub fn _lu_factor_host(values: list[float], n: int) -> (list[float], list[int], int) {
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
pub fn _lu_solve_host(
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
pub fn _triangular_host(
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
pub fn _cholesky_host(values: list[float], n: int) -> list[float] {
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

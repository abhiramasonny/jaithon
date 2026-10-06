#: `A = U S V^T`: bidiagonalize, then chase the superdiagonal to zero.
import std.math
from std.os import env_or
from .common import FLOAT32_EPS, QR_ITERATIONS, _identity_block, _transpose_host
from .qr import _column_reflector, _reflect_across, _reflect_from, _row_reflector
from .rotate import _givens, _rotate_columns, _swap_columns

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

#: `JAINUM_TALL_SVD=0` takes a tall matrix through the full bidiagonalization
#: like any other, instead of factoring `A = QR` first and the small `R`.
let _TALL_SVD = env_or("JAINUM_TALL_SVD", "1") != "0"

#: How much taller than wide a matrix must be before `_svd_host` takes the QR
#: first. LAPACK's crossover for the same choice sits near 1.6; two keeps the
#: square-ish cases on the path every existing test was written against.
const TALL_RATIO: int = 2

#: `sum a[ai..] * b[bi..]` over `count` contiguous values.
#: Four running totals, because one is a chain of dependent adds that runs at
#: the adder's latency rather than its throughput.
fn _dot_run(a: list[float], ai: int, b: list[float], bi: int, count: int) -> float {
    var t0 = 0.0
    var t1 = 0.0
    var t2 = 0.0
    var t3 = 0.0
    let whole = count - count % 4
    var index = 0
    while index < whole {
        t0 += a[ai + index] * b[bi + index]
        t1 += a[ai + index + 1] * b[bi + index + 1]
        t2 += a[ai + index + 2] * b[bi + index + 2]
        t3 += a[ai + index + 3] * b[bi + index + 3]
        index += 4
    }
    while index < count {
        t0 += a[ai + index] * b[bi + index]
        index += 1
    }
    return (t0 + t1) + (t2 + t3)
}

#: `b[bi..] -= factor * a[ai..]` over `count` contiguous values.
fn _axpy_run(a: list[float], ai: int, b: list[float], bi: int, count: int, factor: float) -> void {
    for index in 0..count { b[bi + index] -= factor * a[ai + index] }
}

#: Apply the reflector `vs[step * m ..]` (rows `step..m`) to every column of
#: the column-major `target`, `columns` wide, from column `first`.
fn _reflect_columns(
    target: list[float],
    m: int,
    columns: int,
    vs: list[float],
    step: int,
    vnorm: float,
    first: int
) -> void {
    if vnorm == 0.0 { return }
    let count = m - step
    let vbase = step * m + step
    for column in first..columns {
        let at = column * m + step
        let dot = _dot_run(vs, vbase, target, at, count)
        _axpy_run(vs, vbase, target, at, count, 2.0 * dot / vnorm)
    }
}

#: Householder QR of a column-major `m x n` matrix, in place: the top `n x n`
#: of `cols` becomes `R` (read it as `cols[j * m + i]`, `i <= j`) and the
#: reflectors are returned beside their `v^T v`. Every pass over a column is
#: a contiguous run, which is the whole reason for the transpose: the
#: row-major `_householder_qr` strides a full row between neighbours, and
#: at a hundred thousand rows that is a cache line per multiply.
fn _qr_columns(cols: list[float], m: int, n: int) -> (list[float], list[float]) {
    var vs = [0.0] * (n * m)
    var norms = [0.0] * n
    for step in 0..n {
        let at = step * m + step
        let count = m - step
        let sigma = _dot_run(cols, at, cols, at, count)
        if sigma == 0.0 { continue }
        let alpha = if cols[at] > 0.0 { -math.sqrt(sigma) } else { math.sqrt(sigma) }
        let vbase = step * m + step
        _copy_run(cols, at, vs, vbase, count)
        vs[vbase] -= alpha
        let vnorm = _dot_run(vs, vbase, vs, vbase, count)
        norms[step] = vnorm
        _reflect_columns(cols, m, n, vs, step, vnorm, step + 1)
        _zero_run(cols, at, count)
        cols[at] = alpha
    }
    return (vs, norms)
}

#: The thin SVD of a tall matrix through `A = QR`: `R = Ur S V^T` is a
#: square problem of the width, and `U = Q [Ur; 0]` is the reflectors
#: applied to `Ur`. The cubic-in-the-width work moves onto an `n x n` matrix
#: and what is left in the height is two passes of reflections, where the
#: bidiagonalization accumulated `U` through every bulge-chasing rotation.
#: The same factorization as before, so the same singular values to rounding.
fn _svd_tall(values: list[float], m: int, n: int, vectors: bool) -> (list[float], list[float], list[float]) {
    var cols = [0.0] * (m * n)
    _gather_columns(values, cols, m, n, m)
    let (vs, norms) = _qr_columns(cols, m, n)
    var r = [0.0] * (n * n)
    _upper_rows(cols, r, m, n)
    let (ur, s, vt) = _svd_host(r, n, n, false, vectors)
    if not vectors { return ([], s, []) }
    var basis = [0.0] * (m * n)
    _gather_columns(ur, basis, n, n, m)
    var step = n - 1
    while step >= 0 {
        _reflect_columns(basis, m, n, vs, step, norms[step], 0)
        step -= 1
    }
    var u = [0.0] * (m * n)
    _scatter_rows(basis, u, m, n)
    return (u, s, vt)
}

#: Whether `_svd_host` takes a matrix this shape through the QR first; a
#: least squares solve asks so that it can take `_tall_projection` instead.
pub fn _takes_qr_first(m: int, n: int) -> bool {
    return _TALL_SVD and n > 0 and m >= TALL_RATIO * n
}

#: What a least squares solve needs of a tall `A` without ever forming `U`:
#: `(U^T b, S, V^T)`, with `U^T b` as `n x wide`. `Q^T b` is the reflectors
#: applied to `b`; its top `n` rows are `[Ur^T 0] Q^T b`'s input, and the
#: height is walked twice -- the factorization and the right-hand side --
#: rather than four times.
pub fn _tall_projection(
    values: list[float],
    m: int,
    n: int,
    rhs: list[float],
    wide: int
) -> (list[float], list[float], list[float]) {
    var cols = [0.0] * (m * n)
    _gather_columns(values, cols, m, n, m)
    let (vs, norms) = _qr_columns(cols, m, n)
    var r = [0.0] * (n * n)
    _upper_rows(cols, r, m, n)
    var targets = [0.0] * (m * wide)
    _gather_columns(rhs, targets, m, wide, m)
    for step in 0..n { _reflect_columns(targets, m, wide, vs, step, norms[step], 0) }
    let (ur, s, vt) = _svd_host(r, n, n, false, true)
    var projected = [0.0] * (n * wide)
    _project_top(ur, targets, projected, m, n, wide)
    return (projected, s, vt)
}

#: `Ur^T (Q^T b)[0..n]`, with `Q^T b` column-major and `m` tall.
fn _project_top(ur: list[float], targets: list[float], out: list[float], m: int, n: int, wide: int) -> void {
    for step in 0..n {
        for column in 0..wide {
            var total = 0.0
            for row in 0..n { total += ur[row * n + step] * targets[column * m + row] }
            out[step * wide + column] = total
        }
    }
}

#: Rows `0..rows` of the row-major `values` (`n` wide) into the column-major
#: `cols`, whose columns are `height` long. Each of these copies is its own
#: function because a body with several loops and calls runs out of room
#: for the JIT's loop forms, and stays interpreted.
#: Taken in blocks of rows, so the strided side of the copy stays in cache
#: across the columns instead of missing once per element.
fn _gather_columns(values: list[float], cols: list[float], rows: int, n: int, height: int) -> void {
    var start = 0
    while start < rows {
        let stop = min(start + TRANSPOSE_BLOCK, rows)
        for column in 0..n {
            let base = column * height
            for row in start..stop { cols[base + row] = values[row * n + column] }
        }
        start = stop
    }
}

#: The column-major `cols` (`m` tall, `n` wide) back to row-major `out`.
fn _scatter_rows(cols: list[float], out: list[float], m: int, n: int) -> void {
    var start = 0
    while start < m {
        let stop = min(start + TRANSPOSE_BLOCK, m)
        for column in 0..n {
            let base = column * m
            for row in start..stop { out[row * n + column] = cols[base + row] }
        }
        start = stop
    }
}

const TRANSPOSE_BLOCK: int = 32

#: The upper triangle of the column-major `cols` as a row-major `n x n`.
fn _upper_rows(cols: list[float], r: list[float], m: int, n: int) -> void {
    for row in 0..n {
        for column in row..n { r[row * n + column] = cols[column * m + row] }
    }
}

fn _copy_run(source: list[float], si: int, target: list[float], ti: int, count: int) -> void {
    for index in 0..count { target[ti + index] = source[si + index] }
}

fn _zero_run(target: list[float], at: int, count: int) -> void {
    for index in 0..count { target[at + index] = 0.0 }
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
pub fn _svd_host(
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
    if _TALL_SVD and n > 0 and m >= TALL_RATIO * n and not (vectors and full) {
        return _svd_tall(values, m, n, vectors)
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

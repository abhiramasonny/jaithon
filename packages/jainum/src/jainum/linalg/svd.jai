#: `A = U S V^T`: bidiagonalize, then chase the superdiagonal to zero.
import std.math
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

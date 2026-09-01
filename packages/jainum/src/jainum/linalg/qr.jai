#: Householder reflectors: the QR pass, and the reflection every other
#: factorization here is built out of.
import std.math
from .common import _identity_block

#: `target <- (I - 2 v v^T / v^T v) target` over rows `step..rows` and columns
#: `first..cols`, with `v` living in `vs[base + step .. base + rows]`.
#:
#: Call-free, and the cube-order loop of every Householder pass in the file —
#: QR, the bidiagonalization and the Hessenberg reduction all spend their time
#: here rather than in the reflector that feeds it.
pub fn _reflect_from(
    target: list[float],
    rows: int,
    cols: int,
    vs: list[float],
    base: int,
    step: int,
    vnorm: float,
    first: int
) -> void {
    if vnorm == 0.0 { return }
    for column in first..cols {
        var dot = 0.0
        for row in step..rows { dot += vs[base + row] * target[row * cols + column] }
        let factor = 2.0 * dot / vnorm
        for row in step..rows { target[row * cols + column] -= factor * vs[base + row] }
    }
}

#: The same reflector applied from the right: `target <- target H`, over rows
#: `first..rows` and columns `step..cols`, with `v` indexed by column.
pub fn _reflect_across(
    target: list[float],
    rows: int,
    cols: int,
    vs: list[float],
    base: int,
    step: int,
    vnorm: float,
    first: int
) -> void {
    if vnorm == 0.0 { return }
    for row in first..rows {
        var dot = 0.0
        for column in step..cols { dot += vs[base + column] * target[row * cols + column] }
        let factor = 2.0 * dot / vnorm
        for column in step..cols { target[row * cols + column] -= factor * vs[base + column] }
    }
}

#: `sum x^2` over `count` values `stride` apart, starting at `at`.
fn _segment_sigma(work: list[float], at: int, stride: int, count: int) -> float {
    var sigma = 0.0
    for index in 0..count {
        let value = work[at + index * stride]
        sigma += value * value
    }
    return sigma
}

#: Write `x - alpha e1` for that segment into `vs`, returning `v^T v`.
fn _segment_vector(
    work: list[float],
    at: int,
    stride: int,
    count: int,
    alpha: float,
    vs: list[float],
    vbase: int
) -> float {
    var vnorm = 0.0
    for index in 0..count {
        let value = if index == 0 { work[at] - alpha } else { work[at + index * stride] }
        vs[vbase + index] = value
        vnorm += value * value
    }
    return vnorm
}

#: The reflector that collapses `work[step.., column]` onto `alpha e1`.
#:
#: Returns `(alpha, v^T v)` and writes `v` into `vs[base + step ..]`. The sign
#: of `alpha` is taken away from the leading entry so that the subtraction
#: that forms `v` adds magnitudes instead of cancelling them — the one place
#: in a Householder pass where a naive sign loses digits.
#:
#: The two passes over the segment live in the helpers above rather than here
#: because of the `sqrt` between them: it is a call, the JIT declines any body
#: holding one, and these two loops are the quadratic term of every
#: factorization in the file.
pub fn _column_reflector(
    work: list[float],
    m: int,
    n: int,
    column: int,
    step: int,
    vs: list[float],
    base: int
) -> (float, float) {
    let at = step * n + column
    let count = m - step
    let sigma = _segment_sigma(work, at, n, count)
    if sigma == 0.0 { return (0.0, 0.0) }
    let alpha = if work[at] > 0.0 { -math.sqrt(sigma) } else { math.sqrt(sigma) }
    return (alpha, _segment_vector(work, at, n, count, alpha, vs, base + step))
}

#: The same for a row segment: collapses `work[row, step..]` onto `beta e1`.
pub fn _row_reflector(
    work: list[float],
    n: int,
    row: int,
    step: int,
    vs: list[float],
    base: int
) -> (float, float) {
    let at = row * n + step
    let count = n - step
    let sigma = _segment_sigma(work, at, 1, count)
    if sigma == 0.0 { return (0.0, 0.0) }
    let beta = if work[at] > 0.0 { -math.sqrt(sigma) } else { math.sqrt(sigma) }
    return (beta, _segment_vector(work, at, 1, count, beta, vs, base + step))
}

#: `A = QR` by Householder: `work` becomes `R` and the reflectors stay in the
#: returned `(vs, norms)`, one length-`m` vector per step.
pub fn _householder_qr(work: list[float], m: int, n: int, k: int) -> (list[float], list[float]) {
    var vs = [0.0 for _slot in 0..k * m]
    var norms = [0.0 for _step in 0..k]
    for step in 0..k {
        let (alpha, vnorm) = _column_reflector(work, m, n, step, step, vs, step * m)
        norms[step] = vnorm
        work[step * n + step] = alpha
        for row in step + 1..m { work[row * n + step] = 0.0 }
        _reflect_from(work, m, n, vs, step * m, step, vnorm, step + 1)
    }
    return (vs, norms)
}

#: `Q` as `m` by `columns`, from the reflectors of a Householder pass.
#:
#: Backwards, because `Q = H_0 H_1 ... H_k` and each `H` is applied from the
#: left of what the ones after it have already built.
pub fn _accumulate_reflectors(
    vs: list[float],
    norms: list[float],
    m: int,
    k: int,
    columns: int
) -> list[float] {
    var q = _identity_block(m, columns)
    var step = k - 1
    while step >= 0 {
        _reflect_from(q, m, columns, vs, step * m, step, norms[step], 0)
        step -= 1
    }
    return q
}

#: `R` cut out of a finished Householder pass, `rows` by `n`.
pub fn _upper_triangle(work: list[float], n: int, rows: int) -> list[float] {
    var out = [0.0 for _slot in 0..rows * n]
    for row in 0..rows {
        for column in row..n { out[row * n + column] = work[row * n + column] }
    }
    return out
}

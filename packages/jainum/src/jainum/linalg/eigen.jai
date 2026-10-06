#: Eigenvalues: a tridiagonal QL (or cyclic Jacobi) for the symmetric case,
#: Hessenberg and a Schur sweep for the general one.
import std.math
from std.os import env_or
from .common import FLOAT32_EPS, JACOBI_SWEEPS, JACOBI_TOL, QR_ITERATIONS, _identity_block
from .qr import _column_reflector, _reflect_across, _reflect_from
from .rotate import _givens, _order_ascending, _rotate_columns, _rotate_rows

#: The weight sitting off the diagonal, against the weight on it.
fn _jacobi_off(a: list[float], n: int) -> (float, float) {
    var off = 0.0
    var on = 0.0
    for row in 0..n {
        for column in row + 1..n {
            let value = a[row * n + column]
            off += value * value
        }
        let diagonal = a[row * n + row]
        on += diagonal * diagonal
    }
    return (off, on)
}

#: The rotation `(c, s)` that zeroes `a[p][r]`, Rutishauser's stable form.
#:
#: Its own function because the square roots are calls on the `math` module,
#: and a module call anywhere in the sweep's body keeps the whole sweep out of
#: the JIT: with them here the rotation loop compiles and eigh 50x50 is about
#: twice as fast. `hypot(theta, 1)` is `sqrt(theta^2 + 1)` with the overflow
#: taken out, which past `1e150` is `|theta|` to every bit a double holds.
fn _jacobi_angle(app: float, arr: float, apr: float) -> (float, float) {
    let theta = (app - arr) / (2.0 * apr)
    let size = if theta >= 0.0 { theta } else { -theta }
    let root = if size > 1.0e150 { size } else { math.sqrt(theta * theta + 1.0) }
    let t = if theta >= 0.0 { 1.0 / (theta + root) } else { -1.0 / (root - theta) }
    let c = 1.0 / math.sqrt(1.0 + t * t)
    return (c, t * c)
}

#: `JAINUM_TRIDIAGONAL_EIGH=0` sends every symmetric problem through the
#: cyclic Jacobi sweep instead of the tridiagonal reduction and implicit QL.
let _TRIDIAGONAL = env_or("JAINUM_TRIDIAGONAL_EIGH", "1") != "0"

#: `A = Q L Q^T` for a symmetric `A`, eigenvalues ascending: what `eigh` and
#: `eigvalsh` run.
#:
#: The tridiagonal QL below, as LAPACK's `syev` does it, and cyclic Jacobi
#: when that cannot answer (an entry that is not finite, or a value that will
#: not converge) or when the switch turns it off.
pub fn _symmetric_eigh(values: list[float], n: int, vectors: bool) -> (list[float], list[float]) {
    if _TRIDIAGONAL {
        let (answered, spectrum, basis) = _tridiagonal_eigh(values, n, vectors)
        if answered { return (spectrum, basis) }
    }
    return _jacobi_eigh(values, n, vectors)
}

#: `sqrt(a^2 + b^2)` without the overflow, scaled by the larger side.
#:
#: Written out with `** 0.5`, which the JIT emits as one `fsqrt`, rather than
#: as `math.hypot`: the result of a builtin has no kind the JIT knows, so the
#: arithmetic it feeds refused to compile and the sweep ran interpreted.
fn _hypot(a: float, b: float) -> float {
    let x = if a < 0.0 { -a } else { a }
    let y = if b < 0.0 { -b } else { b }
    if x > y {
        let ratio = y / x
        return x * (1.0 + ratio * ratio) ** 0.5
    }
    if y == 0.0 { return 0.0 }
    let ratio = x / y
    return y * (1.0 + ratio * ratio) ** 0.5
}

#: The largest entry's magnitude, or -1 when an entry is not finite: the QL
#: iteration needs finite input to stop, and an ordered comparison on a NaN
#: raises.
fn _largest_entry(values: list[float]) -> float {
    var largest = 0.0
    var finite = true
    for value in values {
        if value != value or value - value != 0.0 {
            finite = false
        } else {
            let size = if value < 0.0 { -value } else { value }
            if size > largest { largest = size }
        }
    }
    return if finite { largest } else { -1.0 }
}

#: Outside this range of magnitudes the matrix is scaled to a largest entry
#: of one first, as LAPACK's `syev` does, so no product in the iteration
#: overflows or underflows; inside it the arithmetic is untouched.
const _SCALE_ABOVE: float = 1.0e100
const _SCALE_BELOW: float = 1.0e-100

#: `A = Q L Q^T` for a symmetric `A`: Householder down to tridiagonal, then
#: the implicit QL iteration with Wilkinson's shift (EISPACK's `tred2` and
#: `tql2`), eigenvalues ascending. The flag is false when it cannot answer.
#:
#: The reduction is about `4/3 n^3` and the iteration about `3 n^3` with the
#: vectors, where one Jacobi sweep is `2 n^3` and it takes six to ten of them.
#: Like `syev`, and numpy's `eigh`, the values are accurate to the rounding of
#: the largest one rather than each to its own relative precision; Jacobi,
#: behind the switch, is the relatively accurate one.
pub fn _tridiagonal_eigh(values: list[float], n: int, vectors: bool) -> (bool, list[float], list[float]) {
    var v = [value for value in values]
    var d = [0.0 for _index in 0..n]
    if n == 0 { return (true, d, v) }
    let largest = _largest_entry(values)
    if largest < 0.0 { return (false, d, v) }
    let scaled = largest > _SCALE_ABOVE or (largest > 0.0 and largest < _SCALE_BELOW)
    if scaled {
        for index in 0..n * n { v[index] /= largest }
    }
    var e = [0.0 for _index in 0..n]
    _tridiagonalize(v, d, e, n, vectors)
    let rows = if vectors { n } else { 0 }
    if not _implicit_ql(v, d, e, n, rows) { return (false, d, v) }
    if scaled {
        for index in 0..n { d[index] *= largest }
    }
    _order_ascending(d, v, n, rows)
    if not vectors { return (true, d, [0.0 for _slot in 0..0]) }
    return (true, d, v)
}


#: Householder reduction of the symmetric `v` (row-major, lower triangle
#: read) to tridiagonal form: the diagonal into `d`, the subdiagonal into
#: `e[1..]`, and with `vectors` the orthogonal basis accumulated into `v`.
#:
#: Each step is its own function, and so is the iteration's sweep below: a
#: body this size reuses its local slots across scopes with different kinds,
#: which keeps its loops from compiling.
fn _tridiagonalize(v: list[float], d: list[float], e: list[float], n: int, vectors: bool) -> void {
    for j in 0..n { d[j] = v[(n - 1) * n + j] }
    var i = n - 1
    while i > 0 {
        d[i] = _householder_row(v, d, e, n, i)
        i -= 1
    }
    e[0] = 0.0
    if not vectors {
        for j in 0..n { d[j] = v[j * n + j] }
        return
    }
    for step in 0..n - 1 {
        v[(n - 1) * n + step] = v[step * n + step]
        v[step * n + step] = 1.0
        let h = d[step + 1]
        if h != 0.0 { _apply_reflection(v, d, n, step, h) }
        for k in 0..=step { v[k * n + step + 1] = 0.0 }
    }
    for j in 0..n {
        d[j] = v[(n - 1) * n + j]
        v[(n - 1) * n + j] = 0.0
    }
    v[(n - 1) * n + n - 1] = 1.0
}

#: Zero row `i` left of the diagonal by one reflection, applied to the
#: leading `i x i` block; the reflector stays in row and column `i` of `v`.
#: Returns its `h`, which the accumulation divides by.
fn _householder_row(v: list[float], d: list[float], e: list[float], n: int, i: int) -> float {
    var scale = 0.0
    for k in 0..i {
        let value = d[k]
        scale += if value < 0.0 { -value } else { value }
    }
    if scale == 0.0 {
        e[i] = d[i - 1]
        for j in 0..i {
            d[j] = v[(i - 1) * n + j]
            v[i * n + j] = 0.0
            v[j * n + i] = 0.0
        }
        return 0.0
    }
    var h = 0.0
    for k in 0..i {
        d[k] /= scale
        h += d[k] * d[k]
    }
    let top = d[i - 1]
    var g = h ** 0.5
    if top > 0.0 { g = -g }
    e[i] = scale * g
    h -= top * g
    d[i - 1] = top - g
    for j in 0..i { e[j] = 0.0 }
    _reflect_lower(v, d, e, n, i)
    var f = 0.0
    for j in 0..i {
        e[j] /= h
        f += e[j] * d[j]
    }
    let hh = f / (h + h)
    for j in 0..i { e[j] -= hh * d[j] }
    _update_lower(v, d, e, n, i)
    return h
}

#: `e = A u` over the lower triangle of the leading `i x i` block.
fn _reflect_lower(v: list[float], d: list[float], e: list[float], n: int, i: int) -> void {
    for j in 0..i {
        let f = d[j]
        v[j * n + i] = f
        var g = e[j] + v[j * n + j] * f
        for k in j + 1..i {
            let below = v[k * n + j]
            g += below * d[k]
            e[k] += below * f
        }
        e[j] = g
    }
}

#: The rank-two update `A -= u w^T + w u^T` on the lower triangle, and the
#: next row into `d`.
fn _update_lower(v: list[float], d: list[float], e: list[float], n: int, i: int) -> void {
    for j in 0..i {
        let f = d[j]
        let g = e[j]
        for k in j..i { v[k * n + j] -= f * e[k] + g * d[k] }
        d[j] = v[(i - 1) * n + j]
        v[i * n + j] = 0.0
    }
}

#: Fold the reflector stored in column `step + 1` into the basis so far.
fn _apply_reflection(v: list[float], d: list[float], n: int, step: int, h: float) -> void {
    let next = step + 1
    for k in 0..=step { d[k] = v[k * n + next] / h }
    for j in 0..=step {
        var g = 0.0
        for k in 0..=step { g += v[k * n + next] * v[k * n + j] }
        for k in 0..=step { v[k * n + j] -= g * d[k] }
    }
}

fn _magnitude(value: float) -> float { return if value < 0.0 { -value } else { value } }

#: The implicit QL iteration on the tridiagonal `(d, e)`, rotating the first
#: `rows` rows of `v` along; false when a value will not converge.
fn _implicit_ql(v: list[float], d: list[float], e: list[float], n: int, rows: int) -> bool {
    for i in 1..n { e[i - 1] = e[i] }
    e[n - 1] = 0.0
    var shift = 0.0
    var tst1 = 0.0
    var converged = true
    let eps = 2.220446049250313e-16
    for l in 0..n {
        let size = _magnitude(d[l]) + _magnitude(e[l])
        if size > tst1 { tst1 = size }
        var m = l
        while m < n - 1 and _magnitude(e[m]) > eps * tst1 { m += 1 }
        if m > l {
            var sweeps = 0
            while sweeps <= QR_ITERATIONS and _magnitude(e[l]) > eps * tst1 {
                shift += _ql_sweep(v, d, e, n, rows, l, m)
                sweeps += 1
            }
            if _magnitude(e[l]) > eps * tst1 {
                converged = false
                break
            }
        }
        d[l] += shift
        e[l] = 0.0
    }
    return converged
}

#: One implicitly shifted QL sweep over rows `l..=m` of the tridiagonal,
#: returning the shift it took out of `d`.
fn _ql_sweep(v: list[float], d: list[float], e: list[float], n: int, rows: int, l: int, m: int) -> float {
    let g0 = d[l]
    let p0 = (d[l + 1] - g0) / (2.0 * e[l])
    var r0 = _hypot(p0, 1.0)
    if p0 < 0.0 { r0 = -r0 }
    d[l] = e[l] / (p0 + r0)
    d[l + 1] = e[l] * (p0 + r0)
    let dl1 = d[l + 1]
    let h0 = g0 - d[l]
    for t in l + 2..n { d[t] -= h0 }
    var p = d[m]
    var c = 1.0
    var c2 = 1.0
    var c3 = 1.0
    let el1 = e[l + 1]
    var s = 0.0
    var s2 = 0.0
    var i = m - 1
    while i >= l {
        c3 = c2
        c2 = c
        s2 = s
        let g = c * e[i]
        let h = c * p
        let r = _hypot(p, e[i])
        e[i + 1] = s * r
        s = e[i] / r
        c = p / r
        p = c * d[i] - s * g
        d[i + 1] = h + s * (c * g + s * d[i])
        for k in 0..rows {
            let at = k * n + i
            let held = v[at + 1]
            v[at + 1] = s * v[at] + c * held
            v[at] = c * v[at] - s * held
        }
        i -= 1
    }
    let q = -s * s2 * c3 * el1 * e[l] / dl1
    e[l] = s * q
    d[l] = c * q
    return h0
}


#: `A = Q L Q^T` for a symmetric `A`, by cyclic Jacobi, eigenvalues ascending.
#:
#: About eight times the tridiagonal QL's time at 50x50, but it converges for
#: any symmetric input without a shift strategy and reaches the small
#: eigenvalues to full relative accuracy.
pub fn _jacobi_eigh(values: list[float], n: int, vectors: bool) -> (list[float], list[float]) {
    var a = [value for value in values]
    let rows = if vectors { n } else { 0 }
    var basis = _identity_block(rows, n)
    for _sweep in 0..JACOBI_SWEEPS {
        let (off, on) = _jacobi_off(a, n)
        if off <= JACOBI_TOL * (on + off) { break }
        for p in 0..n {
            for r in p + 1..n {
                let apr = a[p * n + r]
                if apr == 0.0 { continue }
                let (c, s) = _jacobi_angle(a[p * n + p], a[r * n + r], apr)
                _rotate_columns(a, n, n, p, r, c, s)
                _rotate_rows(a, n, p, r, c, s)
                a[p * n + r] = 0.0
                a[r * n + p] = 0.0
                _rotate_columns(basis, rows, n, p, r, c, s)
            }
        }
    }
    var spectrum = [0.0 for _index in 0..n]
    for index in 0..n { spectrum[index] = a[index * n + index] }
    _order_ascending(spectrum, basis, n, rows)
    return (spectrum, basis)
}

#: The triangle `uplo` names, mirrored into a full symmetric matrix.
#:
#: LAPACK reads one triangle and so does this: a matrix that is not quite
#: symmetric is factored as though the named triangle were the truth, rather
#: than being rejected or quietly averaged.
pub fn _symmetrize(values: list[float], at: int, n: int, lower: bool) -> list[float] {
    var out = [0.0 for _slot in 0..n * n]
    for row in 0..n {
        for column in 0..n {
            let take = if lower == (row >= column) { row * n + column } else { column * n + row }
            out[row * n + column] = values[at + take]
        }
    }
    return out
}

#: Reduce to upper Hessenberg by Householder, returning the basis `Z` with
#: `H = Z^T A Z`.
fn _hessenberg(work: list[float], n: int, vectors: bool) -> list[float] {
    var vs = [0.0 for _slot in 0..n * n]
    var norms = [0.0 for _index in 0..n]
    var step = 0
    while step < n - 2 {
        let (alpha, vnorm) = _column_reflector(work, n, n, step, step + 1, vs, step * n)
        norms[step] = vnorm
        work[(step + 1) * n + step] = alpha
        for row in step + 2..n { work[row * n + step] = 0.0 }
        _reflect_from(work, n, n, vs, step * n, step + 1, vnorm, step + 1)
        _reflect_across(work, n, n, vs, step * n, step + 1, vnorm, 0)
        step += 1
    }
    let rows = if vectors { n } else { 0 }
    var basis = _identity_block(rows, n)
    step = n - 3
    while step >= 0 {
        _reflect_from(basis, rows, n, vs, step * n, step + 1, norms[step], 0)
        step -= 1
    }
    return basis
}

#: One shifted QR sweep across the Hessenberg block `low..=high`.
#:
#: The shift is subtracted only from the block's diagonal and the rotations
#: only mix indices inside it, but each is applied across the whole row and
#: the whole column: that is what keeps the coupling to the already-converged
#: parts correct while leaving them otherwise alone.
fn _hessenberg_sweep(
    h: list[float],
    z: list[float],
    n: int,
    zrows: int,
    low: int,
    high: int,
    mu: float
) -> void {
    for k in low..=high { h[k * n + k] -= mu }
    var cosines = [0.0 for _index in low..high]
    var sines = [0.0 for _index in low..high]
    for k in low..high {
        let (c, s, _r) = _givens(h[k * n + k], h[(k + 1) * n + k])
        cosines[k - low] = c
        sines[k - low] = s
        _rotate_rows(h, n, k, k + 1, c, s)
        h[(k + 1) * n + k] = 0.0
    }
    for k in low..high {
        _rotate_columns(h, n, n, k, k + 1, cosines[k - low], sines[k - low])
        _rotate_columns(z, zrows, n, k, k + 1, cosines[k - low], sines[k - low])
    }
    for k in low..=high { h[k * n + k] += mu }
}

#: Triangularise a two by two block that has decoupled, or refuse it.
#:
#: A converged two by two with a negative discriminant is a genuine complex
#: conjugate pair — not the transient one an unconverged block can show — and
#: it is the one thing this cannot carry: there is no `COMPLEX64` answer to
#: give without running the whole Schur iteration in complex arithmetic. So it
#: names the pair and stops, rather than handing back a real part and calling
#: it an eigenvalue.
fn _split_pair(h: list[float], z: list[float], n: int, zrows: int, high: int) -> void {
    let top = h[(high - 1) * n + high - 1]
    let corner = h[high * n + high - 1]
    let bottom = h[high * n + high]
    let half = (top - bottom) / 2.0
    let disc = half * half + h[(high - 1) * n + high] * corner
    if disc < 0.0 {
        let real = (top + bottom) / 2.0
        let imaginary = math.sqrt(-disc)
        throw ValueError(
            f"eig found the complex pair {real} +/- {imaginary}i in rows {high - 1}..{high}; " +
                f"this factorization is real only, and eigh covers the symmetric case"
        )
    }
    if corner == 0.0 { return }
    let root = math.sqrt(disc)
    let near = if half >= 0.0 { half + root } else { half - root }
    let (c, s, _r) = _givens(near, corner)
    _rotate_columns(h, n, n, high - 1, high, c, s)
    _rotate_rows(h, n, high - 1, high, c, s)
    _rotate_columns(z, zrows, n, high - 1, high, c, s)
    h[high * n + high - 1] = 0.0
}

#: Shifted QR on an upper Hessenberg matrix, deflating from the bottom.
#:
#: A single real Wilkinson shift with Givens rotations, not Francis's double
#: shift: the one thing the double shift buys is staying in real arithmetic
#: through a complex pair, and a complex pair is exactly the case this raises
#: on. Every tenth stalled iteration takes an ad-hoc shift instead, which is
#: what breaks the cycles a repeated eigenvalue can otherwise sit in.
fn _hessenberg_qr(h: list[float], z: list[float], n: int, zrows: int) -> void {
    var high = n - 1
    var budget = QR_ITERATIONS * n + 100
    var stale = 0
    while high > 0 {
        var low = high
        while low > 0 {
            let sub = h[low * n + low - 1]
            let size = if sub < 0.0 { -sub } else { sub }
            let over = h[(low - 1) * n + low - 1]
            let under = h[low * n + low]
            let scale = (if over < 0.0 {
                -over
            } else {
                over
            }) + (if under < 0.0 {
                -under
            } else {
                under
            })
            if size <= FLOAT32_EPS * scale {
                h[low * n + low - 1] = 0.0
                break
            }
            low -= 1
        }
        if low == high {
            high -= 1
            stale = 0
            continue
        }
        if low == high - 1 {
            _split_pair(h, z, n, zrows, high)
            high -= 2
            stale = 0
            continue
        }

        let top = h[(high - 1) * n + high - 1]
        let bottom = h[high * n + high]
        let half = (top - bottom) / 2.0
        let disc = half * half + h[(high - 1) * n + high] * h[high * n + high - 1]
        var mu = bottom + half
        if disc >= 0.0 {
            let root = math.sqrt(disc)
            mu = bottom + half - (if half >= 0.0 { root } else { -root })
        }
        stale += 1
        if stale % 10 == 0 {
            let one = h[high * n + high - 1]
            let two = h[(high - 1) * n + high - 2]
            mu = (if one < 0.0 { -one } else { one }) + (if two < 0.0 { -two } else { two })
        }
        budget -= 1
        if budget <= 0 {
            throw RuntimeError(f"eig did not converge in {QR_ITERATIONS} sweeps an eigenvalue")
        }
        _hessenberg_sweep(h, z, n, zrows, low, high, mu)
    }
}

#: `x = Z y` for the leading `upto + 1` entries of `y`, returning `x^T x`.
fn _apply_basis(
    z: list[float],
    y: list[float],
    out: list[float],
    n: int,
    column: int,
    upto: int
) -> float {
    var norm = 0.0
    for row in 0..n {
        var total = 0.0
        for k in 0..=upto { total += z[row * n + k] * y[k] }
        out[row * n + column] = total
        norm += total * total
    }
    return norm
}

#: Eigenvectors of a triangular Schur form, mapped back through `z`.
#:
#: Back substitution on `(T - lambda I) y = 0` with the eigenvalue's own entry
#: pinned to one. A repeated eigenvalue makes a diagonal difference vanish;
#: the substitution floors it at the rounding of the matrix rather than
#: dividing by zero, which yields one usable vector for the pair instead of a
#: NaN — the same compromise LAPACK's `trevc` makes, and the reason a
#: defective matrix gets a basis here that is not of full rank.
fn _schur_vectors(t: list[float], z: list[float], n: int) -> list[float] {
    var out = [0.0 for _slot in 0..n * n]
    var scale = 0.0
    for index in 0..n * n {
        let value = t[index]
        let size = if value < 0.0 { -value } else { value }
        if size > scale { scale = size }
    }
    let floor = if scale > 0.0 { scale * FLOAT32_EPS } else { FLOAT32_EPS }
    for column in 0..n {
        let lambda = t[column * n + column]
        var y = [0.0 for _index in 0..n]
        y[column] = 1.0
        var row = column - 1
        while row >= 0 {
            var total = 0.0
            for k in row + 1..=column { total += t[row * n + k] * y[k] }
            var denom = t[row * n + row] - lambda
            if denom < floor and denom > -floor { denom = floor }
            y[row] = -total / denom
            row -= 1
        }
        let norm = _apply_basis(z, y, out, n, column, column)
        if norm <= 0.0 { continue }
        let inverse = 1.0 / math.sqrt(norm)
        for index in 0..n { out[index * n + column] *= inverse }
    }
    return out
}

#: `A = Z T Z^T`: the real Schur form, and the eigenvectors read off it.
pub fn _eig_host(values: list[float], n: int, vectors: bool) -> (list[float], list[float]) {
    var h = [value for value in values]
    let rows = if vectors { n } else { 0 }
    let basis = _hessenberg(h, n, vectors)
    _hessenberg_qr(h, basis, n, rows)
    var spectrum = [0.0 for _index in 0..n]
    for index in 0..n { spectrum[index] = h[index * n + index] }
    if not vectors { return (spectrum, []) }
    return (spectrum, _schur_vectors(h, basis, n))
}

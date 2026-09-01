#: Eigenvalues: cyclic Jacobi for the symmetric case, Hessenberg and a Schur
#: sweep for the general one.
import std.math
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

#: `A = Q L Q^T` for a symmetric `A`, by cyclic Jacobi, eigenvalues ascending.
#:
#: Jacobi and not a tridiagonal QL: at these sizes it is a small constant more
#: arithmetic, it converges for any symmetric input without a shift strategy,
#: and it reaches the small eigenvalues to full relative accuracy — which is
#: what `pinv` and `matrix_rank` are actually asking about when they call it.
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
                let theta = (a[p * n + p] - a[r * n + r]) / (2.0 * apr)
                let root = math.hypot(theta, 1.0)
                let t = if theta >= 0.0 { 1.0 / (theta + root) } else { -1.0 / (root - theta) }
                let c = 1.0 / math.sqrt(1.0 + t * t)
                let s = t * c
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

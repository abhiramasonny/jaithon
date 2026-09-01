#: Givens rotations, and the column moves the iterative factorizations share.
import std.math

#: `(c, s, r)` with `c a + s b = r` and `c b - s a = 0`.
pub fn _givens(a: float, b: float) -> (float, float, float) {
    if b == 0.0 { return (1.0, 0.0, a) }
    if a == 0.0 { return (0.0, 1.0, b) }
    let r = math.hypot(a, b)
    return (a / r, b / r, r)
}

#: Mix columns `p` and `q`: this is `A <- A G`, and `rows` of zero is the
#: no-op the value-only paths pass to skip accumulating a basis they will not
#: return.
pub fn _rotate_columns(
    a: list[float],
    rows: int,
    cols: int,
    p: int,
    q: int,
    c: float,
    s: float
) -> void {
    for row in 0..rows {
        let base = row * cols
        let x = a[base + p]
        let y = a[base + q]
        a[base + p] = c * x + s * y
        a[base + q] = c * y - s * x
    }
}

#: Mix rows `p` and `q` the same way, which is `A <- G^T A`.
pub fn _rotate_rows(a: list[float], cols: int, p: int, q: int, c: float, s: float) -> void {
    let top = p * cols
    let bottom = q * cols
    for column in 0..cols {
        let x = a[top + column]
        let y = a[bottom + column]
        a[top + column] = c * x + s * y
        a[bottom + column] = c * y - s * x
    }
}

pub fn _swap_columns(a: list[float], rows: int, cols: int, p: int, q: int) -> void {
    for row in 0..rows {
        let base = row * cols
        let held = a[base + p]
        a[base + p] = a[base + q]
        a[base + q] = held
    }
}

#: Put eigenvalues in ascending order, carrying their vectors with them.
pub fn _order_ascending(values: list[float], basis: list[float], n: int, rows: int) -> void {
    for index in 0..n {
        var best = index
        for other in index + 1..n {
            if values[other] < values[best] { best = other }
        }
        if best == index { continue }
        let held = values[index]
        values[index] = values[best]
        values[best] = held
        _swap_columns(basis, rows, n, index, best)
    }
}

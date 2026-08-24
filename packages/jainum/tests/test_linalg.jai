#: Tests for jainum's linalg: the factorizations, the solvers and the norms.
#:
#: Nothing here compares against a number copied out of a previous run. A
#: factorization is checked by multiplying it back out, a solve by applying
#: the matrix to its answer, a spectrum by building the matrix from a spectrum
#: that was chosen first. The one piece of arithmetic the tests trust is
#: `product` below, which is written here rather than imported so that a
#: round-trip is not the module agreeing with itself.
#:
#: Everything that allocates returns early without a GPU, as jaicv's suite
#: does.
import std.gpu as gpu
import std.math
from std.test import assert_eq, assert_near, assert_throws, assert_true
from jainum.array import FLOAT32, NDArray, sl
from jainum.linalg import BLOCKED_FROM, BLOCKED_PANEL, blocked_panel
from jainum.linalg import cholesky, cond, det, eig, eigh, eigvals, eigvalsh
from jainum.linalg import inv, lstsq, lu_factor, lu_solve, matrix_rank, norm
from jainum.linalg import pinv, qr, slogdet, solve, svd, svdvals, triangular_solve

fn have_gpu() -> bool { return gpu.is_available() }

#: The matrix product, computed here and not by the package: every round-trip
#: below is only as independent as this function is.
fn product(a: list[float], m: int, k: int, b: list[float], n: int) -> list[float] {
    var out = [0.0 for _slot in 0..m * n]
    for row in 0..m {
        for step in 0..k {
            let factor = a[row * k + step]
            if factor == 0.0 { continue }
            for column in 0..n { out[row * n + column] += factor * b[step * n + column] }
        }
    }
    return out
}

fn transposed(a: list[float], rows: int, cols: int) -> list[float] {
    var out = [0.0 for _slot in 0..rows * cols]
    for row in 0..rows {
        for column in 0..cols { out[column * rows + row] = a[row * cols + column] }
    }
    return out
}

#: A deterministic spread over `(-1, 1)`, so a failure repeats exactly.
fn spread(count: int, seed: int) -> list[float] {
    var out: list[float] = []
    var state = seed * 2 + 1
    for _index in 0..count {
        state = (state * 1103515245 + 12345) % 2147483648
        out.push(float(state) / 1073741824.0 - 1.0)
    }
    return out
}

fn from_host(values: list[float], shape: list[int]) -> NDArray {
    let out = NDArray(shape, FLOAT32)
    out.upload(values)
    return out
}

fn assert_close(
    actual: list[float],
    expected: list[float],
    tolerance: float,
    message: str
) -> void {
    assert_eq(actual.len(), expected.len(), message)
    for index in 0..actual.len() {
        assert_near(actual[index], expected[index], tolerance, message)
    }
}

fn assert_identity(values: list[float], n: int, tolerance: float, message: str) -> void {
    for row in 0..n {
        for column in 0..n {
            let wanted = if row == column { 1.0 } else { 0.0 }
            assert_near(values[row * n + column], wanted, tolerance, message)
        }
    }
}

#: An orthogonal matrix from plane rotations, so a test can build a matrix
#: whose spectrum it already knows and whose inverse is its transpose.
fn rotation(n: int, seed: int) -> list[float] {
    var q = [0.0 for _slot in 0..n * n]
    for index in 0..n { q[index * n + index] = 1.0 }
    let angles = spread(n, seed)
    for p in 0..n - 1 {
        let c = math.cos(angles[p] * 3.0)
        let s = math.sin(angles[p] * 3.0)
        for row in 0..n {
            let x = q[row * n + p]
            let y = q[row * n + p + 1]
            q[row * n + p] = c * x + s * y
            q[row * n + p + 1] = c * y - s * x
        }
    }
    return q
}

#: `q diag(values) q^T`, which has `values` for its eigenvalues by
#: construction.
fn conjugated(q: list[float], middle: list[float], n: int) -> list[float] {
    let right = product(middle, n, n, transposed(q, n, n), n)
    return product(q, n, n, right, n)
}

fn diagonal_matrix(values: list[float]) -> list[float] {
    let n = values.len()
    var out = [0.0 for _slot in 0..n * n]
    for index in 0..n { out[index * n + index] = values[index] }
    return out
}

fn test_the_blocked_hook_names_one_crossover() -> void {
    assert_eq(blocked_panel(BLOCKED_FROM - 1), 0, "below the crossover the sweep wins outright")
    assert_eq(blocked_panel(BLOCKED_FROM), BLOCKED_PANEL, "at it the panel starts paying")
    assert_eq(blocked_panel(4096), BLOCKED_PANEL, "the panel width does not grow with the matrix")
    assert_true(
        BLOCKED_FROM > BLOCKED_PANEL,
        "a panel narrower than the crossover, or it never splits"
    )
}

fn test_lu_factors_the_matrix_it_was_handed() -> void {
    if not have_gpu() { return }
    let n = 6
    let values = spread(n * n, 11)
    let a = from_host(values, [n, n])
    defer { a.free() }

    let (lu, piv) = lu_factor(a)
    defer { lu.free() }
    defer { piv.free() }
    assert_eq(lu.shape, [n, n])
    assert_eq(piv.shape, [n])

    #: `P A` is the row swaps replayed in the order the factorization made
    #: them, and `L U` has to come back to exactly that.
    var permuted = [value for value in values]
    let rows = piv.to_list()
    for step in 0..n {
        let other = int(rows[step] + 0.5)
        if other == step { continue }
        for column in 0..n {
            let held = permuted[step * n + column]
            permuted[step * n + column] = permuted[other * n + column]
            permuted[other * n + column] = held
        }
    }

    let packed = lu.to_list()
    var lower = [0.0 for _slot in 0..n * n]
    var upper = [0.0 for _slot in 0..n * n]
    for row in 0..n {
        lower[row * n + row] = 1.0
        for column in 0..row { lower[row * n + column] = packed[row * n + column] }
        for column in row..n { upper[row * n + column] = packed[row * n + column] }
    }
    assert_close(product(lower, n, n, upper, n), permuted, 1e-5, "P A = L U")
}

fn test_solve_puts_the_right_hand_side_back() -> void {
    if not have_gpu() { return }
    let n = 7
    let values = spread(n * n, 3)
    let rhs = spread(n, 29)
    let a = from_host(values, [n, n])
    defer { a.free() }
    let b = from_host(rhs, [n])
    defer { b.free() }

    let x = solve(a, b)
    defer { x.free() }
    assert_eq(x.shape, [n], "a stack of vectors one deep is still a vector")
    assert_close(product(values, n, n, x.to_list(), 1), rhs, 1e-5, "a x = b")

    #: The same matrix against several right-hand sides at once, which shares
    #: the one factorization.
    let wide = from_host(spread(n * 3, 5), [n, 3])
    defer { wide.free() }
    let many = solve(a, wide)
    defer { many.free() }
    assert_eq(many.shape, [n, 3])
    assert_close(
        product(values, n, n, many.to_list(), 3),
        wide.to_list(),
        1e-5,
        "a x = b, three columns at a time"
    )

    assert_throws(
        ValueError,
        || solve(from_host(spread(6, 1), [2, 3]), b),
        "a solve needs a square"
    )
}

fn test_solve_broadcasts_the_batch_axes() -> void {
    if not have_gpu() { return }
    let n = 3
    let first = spread(n * n, 7)
    let second = spread(n * n, 13)
    let stack = from_host(first + second, [2, n, n])
    defer { stack.free() }
    let rhs = spread(n, 17)
    let b = from_host(rhs, [n])
    defer { b.free() }

    let x = solve(stack, b)
    defer { x.free() }
    assert_eq(x.shape, [2, n], "one right-hand side broadcast over two matrices")
    let answers = x.to_list()
    var left = [0.0 for _index in 0..n]
    var right = [0.0 for _index in 0..n]
    for index in 0..n {
        left[index] = answers[index]
        right[index] = answers[n + index]
    }
    assert_close(product(first, n, n, left, 1), rhs, 1e-5, "the first matrix solved its own")
    assert_close(product(second, n, n, right, 1), rhs, 1e-5, "and the second one its own")
}

fn test_lu_solve_reuses_what_lu_factor_paid_for() -> void {
    if not have_gpu() { return }
    let n = 5
    let values = spread(n * n, 23)
    let a = from_host(values, [n, n])
    defer { a.free() }
    let b = from_host(spread(n * 2, 31), [n, 2])
    defer { b.free() }

    let (lu, piv) = lu_factor(a)
    defer { lu.free() }
    defer { piv.free() }
    let x = lu_solve(lu, piv, b)
    defer { x.free() }
    assert_close(
        product(values, n, n, x.to_list(), 2),
        b.to_list(),
        1e-5,
        "a x = b from the factors"
    )

    let direct = solve(a, b)
    defer { direct.free() }
    assert_close(x.to_list(), direct.to_list(), 1e-5, "and the same answer solve reaches")
}

fn test_a_singular_matrix_is_refused_rather_than_guessed() -> void {
    if not have_gpu() { return }
    let a = from_host([1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 1.0, 1.0], [3, 3])
    defer { a.free() }
    let b = from_host([1.0, 2.0, 3.0], [3])
    defer { b.free() }
    assert_throws(ValueError, || solve(a, b), "the second row is twice the first")
    assert_throws(ValueError, || inv(a))
    let value = det(a)
    defer { value.free() }
    assert_near(value.item(), 0.0, 1e-12, "det says so without raising")
}

fn test_triangular_solve_reads_the_triangle_it_is_told_to() -> void {
    if not have_gpu() { return }
    let n = 5
    var lower = spread(n * n, 41)
    for row in 0..n {
        lower[row * n + row] = 2.0 + float(row)
        for column in row + 1..n { lower[row * n + column] = 0.0 }
    }
    let a = from_host(lower, [n, n])
    defer { a.free() }
    let b = from_host(spread(n * 2, 43), [n, 2])
    defer { b.free() }

    let x = triangular_solve(a, b, true)
    defer { x.free() }
    assert_close(product(lower, n, n, x.to_list(), 2), b.to_list(), 1e-5, "l x = b")

    #: The transposed solve reads the same storage the other way round, so it
    #: has to agree with an explicit transpose.
    let y = triangular_solve(a, b, true, false, true)
    defer { y.free() }
    assert_close(
        product(transposed(lower, n, n), n, n, y.to_list(), 2),
        b.to_list(),
        1e-5,
        "l^T x = b without forming l^T"
    )

    #: A unit diagonal ignores what is actually stored there.
    var unit = [value for value in lower]
    for row in 0..n { unit[row * n + row] = 1.0 }
    let z = triangular_solve(a, b, true, true)
    defer { z.free() }
    assert_close(
        product(unit, n, n, z.to_list(), 2),
        b.to_list(),
        1e-5,
        "the diagonal was not read"
    )
}

fn test_cholesky_rebuilds_the_matrix() -> void {
    if not have_gpu() { return }
    let n = 5
    let seed = spread(n * n, 53)
    var spd = product(seed, n, n, transposed(seed, n, n), n)
    for index in 0..n { spd[index * n + index] += float(n) }
    let a = from_host(spd, [n, n])
    defer { a.free() }

    let l = cholesky(a)
    defer { l.free() }
    let values = l.to_list()
    for row in 0..n {
        for column in row + 1..n {
            assert_eq(values[row * n + column], 0.0, "the upper triangle is empty")
        }
    }
    assert_close(product(values, n, n, transposed(values, n, n), n), spd, 1e-5, "a = l l^T")

    let upper = cholesky(a, true)
    defer { upper.free() }
    assert_close(upper.to_list(), transposed(values, n, n), 1e-12, "upper is the transpose")

    let indefinite = from_host([1.0, 2.0, 2.0, 1.0], [2, 2])
    defer { indefinite.free() }
    assert_throws(ValueError, || cholesky(indefinite), "this one has a negative eigenvalue")
}

fn test_qr_is_orthogonal_and_upper_triangular() -> void {
    if not have_gpu() { return }
    let m = 6
    let n = 4
    let values = spread(m * n, 61)
    let a = from_host(values, [m, n])
    defer { a.free() }

    let (q, r) = qr(a)
    defer { q.free() }
    defer { r.free() }
    assert_eq(q.shape, [m, n], "reduced q is m by k")
    assert_eq(r.shape, [n, n])
    let qv = q.to_list()
    let rv = r.to_list()
    assert_identity(product(transposed(qv, m, n), n, m, qv, n), n, 1e-5, "q^T q = I")
    for row in 0..n {
        for column in 0..row { assert_eq(rv[row * n + column], 0.0, "r is upper triangular") }
    }
    assert_close(product(qv, m, n, rv, n), values, 1e-5, "a = q r")

    let (full, wide) = qr(a, "complete")
    defer { full.free() }
    defer { wide.free() }
    assert_eq(full.shape, [m, m], "complete q spans the whole space")
    assert_eq(wide.shape, [m, n])
    let fv = full.to_list()
    assert_identity(
        product(transposed(fv, m, m), m, m, fv, m),
        m,
        1e-5,
        "the wide q is orthogonal too"
    )
    assert_close(product(fv, m, m, wide.to_list(), n), values, 1e-5, "a = q r, complete")

    #: A wide matrix factors the same way with `k = m`.
    let short = from_host(spread(3 * 5, 67), [3, 5])
    defer { short.free() }
    let (sq, sr) = qr(short)
    defer { sq.free() }
    defer { sr.free() }
    assert_eq(sq.shape, [3, 3])
    assert_eq(sr.shape, [3, 5])
    assert_close(
        product(sq.to_list(), 3, 3, sr.to_list(), 5),
        short.to_list(),
        1e-5,
        "a = q r for a wide matrix"
    )
    assert_throws(ValueError, || qr(a, "raw"), "the modes a tuple return cannot carry are named")
}

fn test_svd_rebuilds_the_matrix_it_took_apart() -> void {
    if not have_gpu() { return }
    let m = 6
    let n = 4
    let values = spread(m * n, 71)
    let a = from_host(values, [m, n])
    defer { a.free() }

    let (u, s, vt) = svd(a, false)
    defer { u.free() }
    defer { s.free() }
    defer { vt.free() }
    assert_eq(u.shape, [m, n])
    assert_eq(s.shape, [n])
    assert_eq(vt.shape, [n, n])

    let uv = u.to_list()
    let sv = s.to_list()
    let vv = vt.to_list()
    assert_identity(
        product(transposed(uv, m, n), n, m, uv, n),
        n,
        1e-5,
        "u has orthonormal columns"
    )
    assert_identity(product(vv, n, n, transposed(vv, n, n), n), n, 1e-5, "v is orthogonal")
    for index in 1..n {
        assert_true(sv[index] <= sv[index - 1] + 1e-12, "the values come out descending")
        assert_true(sv[index] >= 0.0, "and non-negative")
    }
    let scaled = product(diagonal_matrix(sv), n, n, vv, n)
    assert_close(product(uv, m, n, scaled, n), values, 1e-5, "a = u s v^T")

    #: The full bases carry the two extra columns that span what the range
    #: leaves over, and still rebuild the matrix.
    let (fu, fs, fvt) = svd(a)
    defer { fu.free() }
    defer { fs.free() }
    defer { fvt.free() }
    assert_eq(fu.shape, [m, m])
    assert_eq(fvt.shape, [n, n])
    let fuv = fu.to_list()
    assert_identity(
        product(transposed(fuv, m, m), m, m, fuv, m),
        m,
        1e-5,
        "the full u is orthogonal"
    )
    var padded = [0.0 for _slot in 0..m * n]
    for index in 0..n { padded[index * n + index] = fs.to_list()[index] }
    let widened = product(padded, m, n, fvt.to_list(), n)
    assert_close(product(fuv, m, m, widened, n), values, 1e-5, "a = u s v^T, full")
}

fn test_svd_finds_a_spectrum_that_was_chosen_first() -> void {
    if not have_gpu() { return }
    let n = 5
    let wanted = [9.0, 4.0, 2.5, 0.75, 0.125]
    let left = rotation(n, 83)
    let right = rotation(n, 97)
    let built = product(product(left, n, n, diagonal_matrix(wanted), n), n, n, right, n)
    let a = from_host(built, [n, n])
    defer { a.free() }

    let s = svdvals(a)
    defer { s.free() }
    assert_close(s.to_list(), wanted, 1e-5, "the values it was built from come back")

    #: A wide matrix and its transpose have the same singular values, which is
    #: the one identity the `m < n` path is written on.
    let wide = from_host(spread(3 * 7, 101), [3, 7])
    defer { wide.free() }
    let tall = from_host(transposed(wide.to_list(), 3, 7), [7, 3])
    defer { tall.free() }
    let one = svdvals(wide)
    defer { one.free() }
    let two = svdvals(tall)
    defer { two.free() }
    assert_close(one.to_list(), two.to_list(), 1e-5, "turning it round changes nothing")

    let (u, sv, vt) = svd(wide, false)
    defer { u.free() }
    defer { sv.free() }
    defer { vt.free() }
    assert_eq(u.shape, [3, 3])
    assert_eq(vt.shape, [3, 7])
    let scaled = product(diagonal_matrix(sv.to_list()), 3, 3, vt.to_list(), 7)
    assert_close(product(u.to_list(), 3, 3, scaled, 7), wide.to_list(), 1e-5, "a = u s v^T, wide")
}

fn test_eigh_diagonalises_a_symmetric_matrix() -> void {
    if not have_gpu() { return }
    let n = 5
    let wanted = [-3.0, -0.5, 0.25, 2.0, 6.0]
    let basis = rotation(n, 103)
    let built = conjugated(basis, diagonal_matrix(wanted), n)
    let a = from_host(built, [n, n])
    defer { a.free() }

    let (w, v) = eigh(a)
    defer { w.free() }
    defer { v.free() }
    assert_close(w.to_list(), wanted, 1e-5, "ascending, and the ones it was built from")

    let vv = v.to_list()
    assert_identity(
        product(transposed(vv, n, n), n, n, vv, n),
        n,
        1e-5,
        "the vectors are orthonormal"
    )
    #: `A V = V diag(w)` is the definition, and it does not care about the
    #: sign each column came out with.
    assert_close(
        product(built, n, n, vv, n),
        product(vv, n, n, diagonal_matrix(w.to_list()), n),
        1e-5,
        "a v = v diag(w)"
    )

    let alone = eigvalsh(a)
    defer { alone.free() }
    assert_close(alone.to_list(), w.to_list(), 1e-12, "the values-only path agrees")
}

fn test_eigh_reads_only_the_triangle_it_is_given() -> void {
    if not have_gpu() { return }
    #: The upper triangle here is a different matrix from the lower one, so a
    #: `uplo` that was ignored would show up as the same spectrum twice.
    let a = from_host([2.0, 9.0, 9.0, 0.0, 3.0, 9.0, 0.0, 1.0, 4.0], [3, 3])
    defer { a.free() }
    let lower = eigvalsh(a)
    defer { lower.free() }
    let upper = eigvalsh(a, "U")
    defer { upper.free() }
    let mirrored = from_host([2.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 1.0, 4.0], [3, 3])
    defer { mirrored.free() }
    let direct = eigvalsh(mirrored)
    defer { direct.free() }
    assert_close(lower.to_list(), direct.to_list(), 1e-5, "the lower triangle is the one read")
    assert_true(
        math.fabs(upper.to_list()[0] - direct.to_list()[0]) > 1.0,
        "and the upper triangle is a different matrix"
    )
    assert_throws(ValueError, || eigvalsh(a, "lower"), "uplo is one letter")
}

fn test_eig_finds_a_real_spectrum_through_a_similarity() -> void {
    if not have_gpu() { return }
    let n = 4
    let wanted = [5.0, -2.0, 1.5, 0.5]
    var upper = spread(n * n, 107)
    for row in 0..n {
        upper[row * n + row] = wanted[row]
        for column in 0..row { upper[row * n + column] = 0.0 }
    }
    #: A triangular matrix's eigenvalues are its diagonal, and a similarity
    #: does not move them, so this matrix has a spectrum known in advance
    #: while looking nothing like a triangle.
    let basis = rotation(n, 109)
    let built = conjugated(basis, upper, n)
    let a = from_host(built, [n, n])
    defer { a.free() }

    let (w, v) = eig(a)
    defer { w.free() }
    defer { v.free() }
    var found = [value for value in w.to_list()]
    for index in 0..n {
        var best = index
        for other in index + 1..n {
            if found[other] > found[best] { best = other }
        }
        let held = found[index]
        found[index] = found[best]
        found[best] = held
    }
    assert_close(found, [5.0, 1.5, 0.5, -2.0], 1e-5, "the diagonal it was built from")

    let vv = v.to_list()
    let left = product(built, n, n, vv, n)
    let right = product(vv, n, n, diagonal_matrix(w.to_list()), n)
    assert_close(left, right, 1e-5, "a v = v diag(w)")
    for column in 0..n {
        var total = 0.0
        for row in 0..n {
            let value = vv[row * n + column]
            total += value * value
        }
        assert_near(total, 1.0, 1e-5, "every eigenvector comes back normalised")
    }

    let alone = eigvals(a)
    defer { alone.free() }
    assert_close(alone.to_list(), w.to_list(), 1e-5, "the values-only path agrees")
}

fn test_eig_refuses_a_complex_pair_rather_than_halving_it() -> void {
    if not have_gpu() { return }
    #: A quarter turn has eigenvalues +i and -i and no real eigenvector at
    #: all; the real part alone would be a wrong answer, not a partial one.
    let turn = from_host([0.0, -1.0, 1.0, 0.0], [2, 2])
    defer { turn.free() }
    let error = assert_throws(ValueError, || eig(turn))
    assert_true(error.message.contains("complex"), "the message names what it found")
    assert_throws(ValueError, || eigvals(turn))
}

fn test_det_and_slogdet_agree_with_the_cofactor_rule() -> void {
    if not have_gpu() { return }
    let values = [2.0, -1.0, 3.0, 0.5, 4.0, -2.0, 1.0, 0.0, 5.0]
    let a = from_host(values, [3, 3])
    defer { a.free() }
    #: Sarrus's rule, which shares no code with an LU.
    let expected = values[0] * (values[4] * values[8] - values[5] * values[7]) - values[1] * (values[3] * values[8] - values[5] * values[6]) +
        values[2] * (values[3] * values[7] - values[4] * values[6])
    let value = det(a)
    defer { value.free() }
    assert_near(value.item(), expected, 1e-5, "the cofactor expansion agrees")

    let (sign, logarithm) = slogdet(a)
    defer { sign.free() }
    defer { logarithm.free() }
    assert_near(sign.item() * math.exp(logarithm.item()), expected, 1e-5, "sign times exp is det")

    #: A stack folds to one value per matrix, and a swap of two rows flips the
    #: sign — the property the permutation count is there to track.
    var swapped: list[float] = []
    for index in 0..9 { swapped.push(values[index]) }
    for column in 0..3 {
        let held = swapped[column]
        swapped[column] = swapped[3 + column]
        swapped[3 + column] = held
    }
    let stack = from_host(values + swapped, [2, 3, 3])
    defer { stack.free() }
    let both = det(stack)
    defer { both.free() }
    assert_eq(both.shape, [2], "one determinant per matrix")
    assert_near(both.to_list()[0], expected, 1e-9)
    assert_near(both.to_list()[1], -expected, 1e-5, "a row swap flips the sign")
}

fn test_inv_undoes_the_matrix() -> void {
    if not have_gpu() { return }
    let n = 6
    let values = spread(n * n, 113)
    let a = from_host(values, [n, n])
    defer { a.free() }
    let back = inv(a)
    defer { back.free() }
    assert_identity(product(values, n, n, back.to_list(), n), n, 1e-5, "a inv(a) = I")
    assert_identity(product(back.to_list(), n, n, values, n), n, 1e-5, "and the other way round")
}

fn test_pinv_satisfies_the_penrose_conditions() -> void {
    if not have_gpu() { return }
    let m = 5
    let n = 3
    #: Rank two by construction: the third column is the sum of the first two.
    var values = spread(m * n, 127)
    for row in 0..m {
        values[row * n + 2] = values[row * n] + values[row * n + 1]
    }
    let a = from_host(values, [m, n])
    defer { a.free() }

    let plus = pinv(a)
    defer { plus.free() }
    assert_eq(plus.shape, [n, m], "the pseudo-inverse transposes the shape")
    let pv = plus.to_list()
    let apa = product(product(values, m, n, pv, m), m, m, values, n)
    assert_close(apa, values, 1e-5, "a a+ a = a")
    let pap = product(product(pv, n, m, values, n), n, n, pv, m)
    assert_close(pap, pv, 1e-5, "a+ a a+ = a+")

    let rank = matrix_rank(a)
    defer { rank.free() }
    assert_eq(rank.item(), 2.0, "the dependent column does not count")

    #: For a square invertible matrix the pseudo-inverse is the inverse.
    let square = from_host(spread(16, 131), [4, 4])
    defer { square.free() }
    let one = pinv(square)
    defer { one.free() }
    let two = inv(square)
    defer { two.free() }
    assert_close(one.to_list(), two.to_list(), 1e-5, "no cutoff bites on a full-rank square")
}

fn test_matrix_rank_counts_what_was_built_into_the_matrix() -> void {
    if not have_gpu() { return }
    let m = 6
    let n = 5
    let left = spread(m * 2, 137)
    let right = spread(2 * n, 139)
    #: Two outer products summed: rank two, whatever the values look like.
    let built = product(left, m, 2, right, n)
    let a = from_host(built, [m, n])
    defer { a.free() }
    let rank = matrix_rank(a)
    defer { rank.free() }
    assert_eq(rank.item(), 2.0)

    let empty = from_host([0.0 for _slot in 0..m * n], [m, n])
    defer { empty.free() }
    let none = matrix_rank(empty)
    defer { none.free() }
    assert_eq(none.item(), 0.0, "a zero matrix has no rank at all")
}

fn test_cond_is_the_ratio_of_the_extremes() -> void {
    if not have_gpu() { return }
    let scale = [8.0, 2.0, 0.5]
    let basis = rotation(3, 149)
    let built = conjugated(basis, diagonal_matrix(scale), 3)
    let a = from_host(built, [3, 3])
    defer { a.free() }

    let two = cond(a)
    defer { two.free() }
    assert_near(two.item(), 16.0, 1e-5, "s_max over s_min, and a rotation changes neither")

    #: The one-norm reading has to agree with the definition it is built from,
    #: which the test can spell out for itself.
    let one = cond(a, 1)
    defer { one.free() }
    let back = inv(a)
    defer { back.free() }
    let left = norm(a, 1, (0, 1))
    defer { left.free() }
    let right = norm(back, 1, (0, 1))
    defer { right.free() }
    assert_near(one.item(), left.item() * right.item(), 1e-4, "norm(a) times norm(inv a)")

    let singular = from_host([1.0, 1.0, 1.0, 1.0], [2, 2])
    defer { singular.free() }
    let endless = cond(singular)
    defer { endless.free() }
    assert_true(endless.item() > 1e30, "a singular matrix has an infinite condition number")
}

fn test_lstsq_solves_the_normal_equations() -> void {
    if not have_gpu() { return }
    let m = 8
    let n = 3
    let design = spread(m * n, 151)
    let target = spread(m, 157)
    let a = from_host(design, [m, n])
    defer { a.free() }
    let b = from_host(target, [m])
    defer { b.free() }

    let (x, residual, rank, values) = lstsq(a, b)
    defer { x.free() }
    defer { residual.free() }
    defer { rank.free() }
    defer { values.free() }
    assert_eq(x.shape, [n])
    assert_eq(rank.item(), 3.0, "full column rank")
    assert_eq(values.shape, [n])

    #: The fit is the one whose residual is orthogonal to every column of the
    #: design — `A^T (A x - b) = 0` — which is the definition and not this
    #: module's route to it.
    let fitted = product(design, m, n, x.to_list(), 1)
    var gap = [0.0 for _index in 0..m]
    var squared = 0.0
    for row in 0..m {
        gap[row] = fitted[row] - target[row]
        squared += gap[row] * gap[row]
    }
    let projected = product(transposed(design, m, n), n, m, gap, 1)
    for index in 0..n {
        assert_near(projected[index], 0.0, 1e-5, "the residual is orthogonal to the design")
    }
    assert_eq(residual.shape, [1], "one residual for one right-hand side")
    assert_near(residual.to_list()[0], squared, 1e-5, "and it is the squared length of the gap")

    #: A rank-deficient fit has no unique answer, so numpy reports no residual
    #: and this follows it.
    var flat = spread(m * n, 163)
    for row in 0..m { flat[row * n + 2] = flat[row * n] }
    let short = from_host(flat, [m, n])
    defer { short.free() }
    let (_x2, empty, deficient, _s2) = lstsq(short, b)
    defer { _x2.free() }
    defer { empty.free() }
    defer { deficient.free() }
    defer { _s2.free() }
    assert_eq(deficient.item(), 2.0, "one column repeated")
    assert_eq(empty.shape, [0], "and no residual to report")
}

fn test_vector_norms_match_their_definitions() -> void {
    if not have_gpu() { return }
    let values = [3.0, -4.0, 0.0, 12.0]
    let a = from_host(values, [4])
    defer { a.free() }

    var sum_abs = 0.0
    var sum_sq = 0.0
    var largest = 0.0
    var smallest = math.INF
    var cubed = 0.0
    var counted = 0.0
    for value in values {
        let size = math.fabs(value)
        sum_abs += size
        sum_sq += value * value
        if size > largest { largest = size }
        if size < smallest { smallest = size }
        cubed += size * size * size
        if value != 0.0 { counted += 1.0 }
    }

    let two = norm(a)
    defer { two.free() }
    assert_near(two.item(), math.sqrt(sum_sq), 1e-5, "the default is the 2-norm")
    let one = norm(a, 1)
    defer { one.free() }
    assert_near(one.item(), sum_abs, 1e-5)
    let big = norm(a, math.INF)
    defer { big.free() }
    assert_near(big.item(), largest, 1e-5)
    let small = norm(a, -math.INF)
    defer { small.free() }
    assert_near(small.item(), smallest, 1e-5)
    let three = norm(a, 3.0)
    defer { three.free() }
    assert_near(three.item(), cubed ** (1.0 / 3.0), 1e-4, "a general p is the sum of powers")
    let zero = norm(a, 0)
    defer { zero.free() }
    assert_near(zero.item(), counted, 1e-5, "order zero counts what is not zero")
}

fn test_vector_norms_fold_one_axis_at_a_time() -> void {
    if not have_gpu() { return }
    let rows = 3
    let cols = 4
    let values = spread(rows * cols, 167)
    let a = from_host(values, [rows, cols])
    defer { a.free() }

    let along = norm(a, null, 1)
    defer { along.free() }
    assert_eq(along.shape, [rows], "one number per row")
    let folded = along.to_list()
    for row in 0..rows {
        var total = 0.0
        for column in 0..cols {
            let value = values[row * cols + column]
            total += value * value
        }
        assert_near(folded[row], math.sqrt(total), 1e-5, "the row's own length")
    }

    let kept = norm(a, null, 0, true)
    defer { kept.free() }
    assert_eq(kept.shape, [1, cols], "keepdims leaves the axis behind as a one")
}

fn test_matrix_norms_match_their_definitions() -> void {
    if not have_gpu() { return }
    let rows = 3
    let cols = 4
    let values = spread(rows * cols, 173)
    let a = from_host(values, [rows, cols])
    defer { a.free() }

    var squares = 0.0
    for value in values { squares += value * value }
    let frobenius = norm(a, "fro")
    defer { frobenius.free() }
    assert_near(frobenius.item(), math.sqrt(squares), 1e-5, "fro is the whole array's length")

    var widest_column = 0.0
    for column in 0..cols {
        var total = 0.0
        for row in 0..rows { total += math.fabs(values[row * cols + column]) }
        if total > widest_column { widest_column = total }
    }
    let one = norm(a, 1)
    defer { one.free() }
    assert_near(one.item(), widest_column, 1e-5, "the 1-norm is the widest column sum")

    var widest_row = 0.0
    for row in 0..rows {
        var total = 0.0
        for column in 0..cols { total += math.fabs(values[row * cols + column]) }
        if total > widest_row { widest_row = total }
    }
    let endless = norm(a, math.INF)
    defer { endless.free() }
    assert_near(endless.item(), widest_row, 1e-5, "and the inf-norm the widest row sum")

    #: The two orders that go through the singular values, against a matrix
    #: whose singular values are on its diagonal.
    let scaled = from_host([2.0, 0.0, 0.0, 0.0, -5.0, 0.0], [2, 3])
    defer { scaled.free() }
    let spectral = norm(scaled, 2)
    defer { spectral.free() }
    assert_near(spectral.item(), 5.0, 1e-5, "the 2-norm is the largest singular value")
    let nuclear = norm(scaled, "nuc")
    defer { nuclear.free() }
    assert_near(nuclear.item(), 7.0, 1e-5, "and the nuclear norm their sum")
    assert_throws(ValueError, || norm(a, "spectral"), "an order it does not know is named")
}

fn test_norms_reduce_a_stack_without_a_host_loop() -> void {
    if not have_gpu() { return }
    #: The kernel path is the one entry point here that walks the whole array,
    #: so it gets a size no host loop would be given.
    let count = 40000
    let values = spread(count, 179)
    let a = from_host(values, [10, 4000])
    defer { a.free() }

    var squares = 0.0
    for value in values { squares += value * value }
    let whole = norm(a)
    defer { whole.free() }
    assert_near(whole.item(), math.sqrt(squares), 1e-3, "one number for forty thousand")

    let rows = norm(a, null, 1)
    defer { rows.free() }
    assert_eq(rows.shape, [10])
    let folded = rows.to_list()
    var rebuilt = 0.0
    for value in folded { rebuilt += value * value }
    assert_near(math.sqrt(rebuilt), whole.item(), 1e-3, "the rows recombine into the whole")

    #: A strided view has to reduce over what it actually points at, not over
    #: the parent's dense run.
    let window = a[0..10, 0..8]
    var corner = 0.0
    for row in 0..10 {
        for column in 0..8 {
            let value = values[row * 4000 + column]
            corner += value * value
        }
    }
    let cut = norm(window)
    defer { cut.free() }
    assert_near(cut.item(), math.sqrt(corner), 1e-5, "the view's own values")
}

fn test_the_blocked_path_reaches_the_same_answer_as_the_sweep() -> void {
    if not have_gpu() { return }
    #: Four columns over the crossover, so this one solve panels and takes the
    #: trailing GEMM while every other test in this file went through the
    #: unblocked sweep. Diagonally dominant, so the comparison is about the
    #: two code paths and not about conditioning.
    let n = BLOCKED_FROM + 4
    var values = spread(n * n, 181)
    for index in 0..n { values[index * n + index] += float(n) }
    let a = from_host(values, [n, n])
    defer { a.free() }
    let rhs = spread(n, 191)
    let b = from_host(rhs, [n])
    defer { b.free() }

    let x = solve(a, b)
    defer { x.free() }
    assert_close(product(values, n, n, x.to_list(), 1), rhs, 1e-5, "a x = b, panelled")

    #: The same matrix one column narrower is under the crossover and takes
    #: the sweep; both routes have to solve their own system just as well.
    let small = BLOCKED_FROM - 1
    var narrow = spread(small * small, 181)
    for index in 0..small { narrow[index * small + index] += float(small) }
    let c = from_host(narrow, [small, small])
    defer { c.free() }
    let d = from_host(spread(small, 191), [small])
    defer { d.free() }
    let y = solve(c, d)
    defer { y.free() }
    assert_close(product(narrow, small, small, y.to_list(), 1), d.to_list(), 1e-5, "a x = b, swept")
}

fn test_svd_survives_a_rank_deficient_matrix() -> void {
    if not have_gpu() { return }
    let m = 8
    let n = 5
    #: Two dependent columns, so two singular values are noise and the
    #: iteration has to rotate the vanished diagonals out rather than stall
    #: dividing by them.
    var values = spread(m * n, 193)
    for row in 0..m {
        values[row * n + 3] = values[row * n]
        values[row * n + 4] = values[row * n + 1] * 2.0
    }
    let a = from_host(values, [m, n])
    defer { a.free() }

    let (u, s, vt) = svd(a, false)
    defer { u.free() }
    defer { s.free() }
    defer { vt.free() }
    let scaled = product(diagonal_matrix(s.to_list()), n, n, vt.to_list(), n)
    assert_close(product(u.to_list(), m, n, scaled, n), values, 1e-5, "a = u s v^T all the same")
    let sv = s.to_list()
    assert_true(sv[3] < sv[0] * 1e-5, "the fourth value is rounding")
    assert_true(sv[4] < sv[0] * 1e-5, "and so is the fifth")

    let rank = matrix_rank(a)
    defer { rank.free() }
    assert_eq(rank.item(), 3.0, "three independent columns")

    #: A matrix with nothing in it at all, which every loop here has to leave
    #: alone rather than divide by.
    let empty = from_host([0.0 for _slot in 0..12], [4, 3])
    defer { empty.free() }
    let none = svdvals(empty)
    defer { none.free() }
    assert_close(none.to_list(), [0.0, 0.0, 0.0], 1e-12, "no values to find")
}

fn test_svd_handles_repeated_and_degenerate_shapes() -> void {
    if not have_gpu() { return }
    #: Every singular value the same: the shift has nothing to separate and
    #: the sweep has to deflate on the first pass instead of spinning.
    var doubled = [0.0 for _slot in 0..36]
    for index in 0..6 { doubled[index * 6 + index] = 3.0 }
    let a = from_host(doubled, [6, 6])
    defer { a.free() }
    let s = svdvals(a)
    defer { s.free() }
    assert_close(s.to_list(), [3.0, 3.0, 3.0, 3.0, 3.0, 3.0], 1e-6, "six times the same value")

    let column = from_host([1.0, 2.0, 2.0], [3, 1])
    defer { column.free() }
    let one = svdvals(column)
    defer { one.free() }
    assert_eq(one.shape, [1])
    assert_near(one.to_list()[0], 3.0, 1e-6, "a column's only singular value is its length")

    let row = from_host([1.0, 2.0, 2.0], [1, 3])
    defer { row.free() }
    let other = svdvals(row)
    defer { other.free() }
    assert_near(other.to_list()[0], 3.0, 1e-6, "and its transpose's is the same")
}

fn test_the_factorizations_loop_over_the_leading_axes() -> void {
    if not have_gpu() { return }
    let m = 4
    let n = 3
    let first = spread(m * n, 197)
    let second = spread(m * n, 199)
    let stack = from_host(first + second, [2, m, n])
    defer { stack.free() }

    let (q, r) = qr(stack)
    defer { q.free() }
    defer { r.free() }
    assert_eq(q.shape, [2, m, n], "one q per matrix")
    assert_eq(r.shape, [2, n, n])
    let qv = q.to_list()
    let rv = r.to_list()
    var top: list[float] = []
    var bottom: list[float] = []
    for index in 0..m * n { top.push(qv[index]) }
    for index in 0..m * n { bottom.push(qv[m * n + index]) }
    var topr: list[float] = []
    var bottomr: list[float] = []
    for index in 0..n * n { topr.push(rv[index]) }
    for index in 0..n * n { bottomr.push(rv[n * n + index]) }
    assert_close(product(top, m, n, topr, n), first, 1e-5, "the first matrix rebuilt")
    assert_close(product(bottom, m, n, bottomr, n), second, 1e-5, "and the second one")

    let (u, s, vt) = svd(stack, false)
    defer { u.free() }
    defer { s.free() }
    defer { vt.free() }
    assert_eq(u.shape, [2, m, n])
    assert_eq(s.shape, [2, n], "one spectrum per matrix")
    assert_eq(vt.shape, [2, n, n])

    #: A batch of symmetric matrices, which `eigh` has to take one at a time.
    let sym = product(first, m, n, transposed(first, m, n), m)
    let symmetric = from_host(sym + sym, [2, m, m])
    defer { symmetric.free() }
    let (w, v) = eigh(symmetric)
    defer { w.free() }
    defer { v.free() }
    assert_eq(w.shape, [2, m])
    assert_eq(v.shape, [2, m, m])
    let spectrum = w.to_list()
    for index in 0..m {
        assert_near(spectrum[index], spectrum[m + index], 1e-6, "the same matrix twice")
    }
}

fn test_pinv_takes_the_cutoff_it_is_given() -> void {
    if not have_gpu() { return }
    #: One tiny singular value: keeping it makes the pseudo-inverse enormous,
    #: cutting it leaves a well behaved rank-one answer.
    let basis = rotation(3, 211)
    let built = conjugated(basis, diagonal_matrix([4.0, 1e-6, 1e-6]), 3)
    let a = from_host(built, [3, 3])
    defer { a.free() }

    let cut = pinv(a, 1e-3)
    defer { cut.free() }
    var largest = 0.0
    for value in cut.to_list() {
        let size = math.fabs(value)
        if size > largest { largest = size }
    }
    assert_true(largest < 1.0, "the small values were dropped")

    let kept = pinv(a, 1e-12)
    defer { kept.free() }
    var biggest = 0.0
    for value in kept.to_list() {
        let size = math.fabs(value)
        if size > biggest { biggest = size }
    }
    assert_true(biggest > 1e4, "and kept when the cutoff is below them")

    let counted = matrix_rank(a, 1e-3)
    defer { counted.free() }
    assert_eq(counted.item(), 1.0, "an absolute tolerance counts what stands above it")
}

fn test_norms_read_a_view_and_a_plane_of_a_stack() -> void {
    if not have_gpu() { return }
    let values = spread(24, 223)
    let a = from_host(values, [2, 3, 4])
    defer { a.free() }

    #: A matrix norm over a plane that is not the last two axes: the reduction
    #: has to follow the strides rather than the storage order.
    let planes = norm(a, "fro", (0, 2))
    defer { planes.free() }
    assert_eq(planes.shape, [3], "the axis left over")
    let folded = planes.to_list()
    for middle in 0..3 {
        var total = 0.0
        for outer in 0..2 {
            for inner in 0..4 {
                let value = values[outer * 12 + middle * 4 + inner]
                total += value * value
            }
        }
        assert_near(folded[middle], math.sqrt(total), 1e-5, "the plane's own values")
    }

    #: A reversed view aliases the same storage backwards, and a norm that
    #: read the parent's run instead would not notice.
    let flipped = a[0, sl(null, null, -1)]
    let whole = norm(flipped)
    defer { whole.free() }
    var head = 0.0
    for index in 0..12 { head += values[index] * values[index] }
    assert_near(whole.item(), math.sqrt(head), 1e-5, "the same values, read backwards")

    let stacked = norm(a, 1, (1, 2), true)
    defer { stacked.free() }
    assert_eq(stacked.shape, [2, 1, 1], "keepdims puts both axes back as ones")
}

fn test_eig_does_not_mistake_a_passing_pair_for_a_complex_one() -> void {
    if not have_gpu() { return }
    #: The companion matrix of `(x-1)(x-2)(x-3)(x-4)(x-5)`. Every root is real,
    #: but the iteration passes through two by two blocks that are not, so a
    #: refusal that tested a block before it had converged would fire here
    #: instead of returning five integers.
    let n = 5
    var companion = [0.0 for _slot in 0..n * n]
    let coefficients = [-120.0, 274.0, -225.0, 85.0, -15.0]
    for row in 1..n { companion[row * n + row - 1] = 1.0 }
    for row in 0..n { companion[row * n + n - 1] = -coefficients[row] }
    let a = from_host(companion, [n, n])
    defer { a.free() }

    let w = eigvals(a)
    defer { w.free() }
    var found = [value for value in w.to_list()]
    for index in 0..n {
        var best = index
        for other in index + 1..n {
            if found[other] < found[best] { best = other }
        }
        let held = found[index]
        found[index] = found[best]
        found[best] = held
    }
    assert_close(found, [1.0, 2.0, 3.0, 4.0, 5.0], 1e-4, "the roots the polynomial was built from")
}

fn test_a_zero_row_and_a_zero_column_leave_every_loop_alone() -> void {
    if not have_gpu() { return }
    #: An exactly vanished reflector at two different steps, which every
    #: Householder pass has to step over rather than divide by.
    let n = 5
    var holed = spread(n * n, 227)
    for column in 0..n { holed[2 * n + column] = 0.0 }
    for row in 0..n { holed[row * n + 3] = 0.0 }
    let a = from_host(holed, [n, n])
    defer { a.free() }

    let s = svdvals(a)
    defer { s.free() }
    assert_near(s.to_list()[n - 1], 0.0, 1e-6, "the empty column leaves nothing behind")
    let rank = matrix_rank(a)
    defer { rank.free() }
    assert_eq(rank.item(), 4.0)

    let (q, r) = qr(a)
    defer { q.free() }
    defer { r.free() }
    assert_close(
        product(q.to_list(), n, n, r.to_list(), n),
        holed,
        1e-5,
        "a = q r with a hole in it"
    )
    let qv = q.to_list()
    assert_identity(
        product(transposed(qv, n, n), n, n, qv, n),
        n,
        1e-5,
        "and q is still orthogonal"
    )

    let value = det(a)
    defer { value.free() }
    assert_near(value.item(), 0.0, 1e-6, "a matrix with an empty column has no determinant")
}

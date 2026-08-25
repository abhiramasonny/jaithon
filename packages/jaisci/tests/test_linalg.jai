#: `jaisci.linalg` against scipy, which is where every expected value came from.
#:
#: The module is a crossing rather than an algorithm — the factorisations live
#: in `core.jai` and have their own tests — so what is checked here is the
#: shape of the scipy surface over them: argument order, what comes back, which
#: triangle, and the two places the thin QR makes this differ from scipy's
#: default.
import std.gpu as gpu
from std.test import assert_eq, assert_near, assert_throws
from jainum.create import from_list
from jainum import NDArray
from jaisci.linalg import cho_factor, cho_solve, cholesky, det, inv, lstsq
from jaisci.linalg import lu_factor, lu_solve, qr, solve, solve_triangular

fn have_gpu() -> bool { return gpu.is_available() }

fn matrix(values: list[float], rows: int, cols: int) -> NDArray {
    return from_list(values, [rows, cols])
}

fn vector(values: list[float]) -> NDArray { return from_list(values, [values.len()]) }

fn assert_close(got: NDArray, expected: list[float], what: str) -> void {
    let values = got.to_list()
    assert_eq(values.len(), expected.len(), what)
    for (index, value) in values.enumerate() { assert_near(value, expected[index], 1e-6, what) }
}

fn test_solve_and_its_factored_form_agree_with_scipy() -> void {
    if not have_gpu() { return }
    let a = matrix([4.0, 3.0, 6.0, 3.0], 2, 2)
    defer { a.free() }
    let b = vector([10.0, 12.0])
    defer { b.free() }

    let direct = solve(a, b)
    defer { direct.free() }
    assert_close(direct, [1.0, 2.0], "solve")

    #: The same answer through the factor, which is the point of keeping one.
    let (lu, pivots) = lu_factor(a)
    defer { lu.free() }
    let staged = lu_solve((lu, pivots), b)
    defer { staged.free() }
    assert_close(staged, [1.0, 2.0], "lu_solve")

    #: A vector in, a vector out; a matrix in, a matrix out.
    let wide = matrix([10.0, 4.0, 12.0, 6.0], 2, 2)
    defer { wide.free() }
    let several = solve(a, wide)
    defer { several.free() }
    assert_eq(several.shape, [2, 2])
}

fn test_det_and_inv_agree_with_scipy() -> void {
    if not have_gpu() { return }
    let a = matrix([4.0, 3.0, 6.0, 3.0], 2, 2)
    defer { a.free() }
    assert_near(det(a), -6.0, 1e-6, "det")

    let inverted = inv(a)
    defer { inverted.free() }
    assert_close(inverted, [-0.5, 0.5, 1.0, -2.0 / 3.0], "inv")

    let singular = matrix([1.0, 2.0, 2.0, 4.0], 2, 2)
    defer { singular.free() }
    assert_near(det(singular), 0.0, 1e-9, "det of a singular matrix")
    assert_throws(ValueError, || inv(singular))
    assert_throws(ValueError, || solve(singular, vector([1.0, 1.0])))
}

fn test_cholesky_picks_the_triangle_it_is_asked_for() -> void {
    if not have_gpu() { return }
    let spd = matrix([4.0, 2.0, 2.0, 3.0], 2, 2)
    defer { spd.free() }

    let low = cholesky(spd, true)
    defer { low.free() }
    assert_close(low, [2.0, 0.0, 1.0, 1.4142135], "cholesky lower")

    let up = cholesky(spd, false)
    defer { up.free() }
    assert_close(up, [2.0, 1.0, 0.0, 1.4142135], "cholesky upper")

    #: Both triangles solve the same system, which is what `cho_factor` is for.
    let rhs = vector([1.0, 2.0])
    defer { rhs.free() }
    let from_low = cho_solve(cho_factor(spd, true), rhs)
    defer { from_low.free() }
    assert_close(from_low, [-0.125, 0.75], "cho_solve from the lower factor")
    let from_up = cho_solve(cho_factor(spd, false), rhs)
    defer { from_up.free() }
    assert_close(from_up, [-0.125, 0.75], "cho_solve from the upper factor")
}

fn test_solve_triangular_reads_the_triangle_it_is_told_to() -> void {
    if not have_gpu() { return }
    let low = matrix([2.0, 0.0, 1.0, 3.0], 2, 2)
    defer { low.free() }
    let answered = solve_triangular(low, vector([4.0, 9.0]), true)
    defer { answered.free() }
    assert_close(answered, [2.0, 7.0 / 3.0], "solve_triangular lower")
}

#: The thin QR, which is scipy's `mode="economic"` and not its default.
fn test_qr_is_thin_and_lstsq_rides_on_it() -> void {
    if not have_gpu() { return }
    let tall = matrix([1.0, 1.0, 1.0, 2.0, 1.0, 3.0], 3, 2)
    defer { tall.free() }

    let (q, r) = qr(tall)
    defer { q.free() }
    defer { r.free() }
    assert_eq(q.shape, [3, 2], "Q is rows by cols, not rows by rows")
    assert_eq(r.shape, [2, 2], "R is cols by cols")
    assert_close(r, [-1.7320508, -3.4641016, 0.0, -1.4142136], "qr r")

    let (coefficients, residual) = lstsq(tall, vector([2.0, 3.0, 5.0]))
    defer { coefficients.free() }
    assert_close(coefficients, [1.0 / 3.0, 1.5], "lstsq")
    assert_near(residual, 1.0 / 6.0, 1e-6, "lstsq residual")

    #: An exact fit leaves no residual, and the subtraction that computes it
    #: must not land below zero.
    let square = matrix([1.0, 0.0, 0.0, 1.0], 2, 2)
    defer { square.free() }
    let (exact, none) = lstsq(square, vector([3.0, 4.0]))
    defer { exact.free() }
    assert_close(exact, [3.0, 4.0], "lstsq on an exact fit")
    assert_near(none, 0.0, 1e-9, "an exact fit has no residual")
}

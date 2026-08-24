#: The shared floor: the bridge between host lists and device arrays, the
#: elementwise stopgaps, the GEMM, and the host dense factorisations.
#:
#: The device tests return early without a GPU, because an `NDArray` is a
#: device buffer and there is nothing to check without one. The `dense_*`,
#: `givens` and `approx_*` tests always run — pure host arithmetic in f64 is
#: the point of them, and the precision policy is what they are guarding.
import std.gpu as gpu
import std.math
from std.test import assert_eq, assert_near, assert_throws, assert_true
from jainum.create import arange, from_list, zeros
from jaisci.core import OptimizeResult, RootResult, TILE, VERSION
from jaisci.core import approx_fprime, approx_jacobian, arr_abs, arr_clip, arr_exp, arr_log
from jaisci.core import arr_maximum, arr_minimum, arr_pow, arr_recip, arr_sign, arr_sqrt
from jaisci.core import as_array, dense_cholesky, dense_lu, dense_lu_solve, dense_qr
from jaisci.core import dense_triangular, dense_tridiagonal, device_matrix, device_vector, dot
from jaisci.core import givens, host, host_matrix, is_available, matmul, matvec, require_gpu

fn assert_values(actual: list[float], expected: list[float], tolerance: float = 1e-5) -> void {
    assert_eq(actual.len(), expected.len())
    for (index, value) in actual.enumerate() { assert_near(value, expected[index], tolerance) }
}

#: A deterministic matrix with no structure a solver could exploit by accident.
#:
#: The diagonal boost is not decoration. `sin(ar + bc)` expands into
#: `sin(ar)cos(bc) + cos(ar)sin(bc)`, so the sine alone spans three columns —
#: two rank-one terms and the constant — and a square one of it is singular at
#: every size past three. Boosting the diagonal is what makes `dense_lu` face
#: an invertible matrix while leaving the off-diagonal structure alone.
fn woven(rows: int, cols: int) -> list[float] {
    var values: list[float] = []
    for row in 0..rows {
        for column in 0..cols {
            let mixed = math.sin(float(row) * 1.7 + float(column) * 0.9) + 0.25
            values.push(if row == column { mixed + 4.0 } else { mixed })
        }
    }
    return values
}

fn test_version_matches_the_manifest() -> void {
    assert_eq(VERSION, "0.1.0")
    assert_eq(is_available(), gpu.is_available())
    assert_eq(TILE * TILE, 256)
}

fn test_require_gpu_names_the_call() -> void {
    if gpu.is_available() { return }
    assert_throws(RuntimeError, || require_gpu("take a matrix product"))
}

fn test_as_array_aliases_an_array_and_uploads_a_list() -> void {
    if not gpu.is_available() { return }
    let already = zeros([3])
    defer { already.free() }
    assert_true(as_array(already) == already)

    let vector = as_array([1.0, 2.0, 3.0])
    defer { vector.free() }
    assert_eq(vector.shape, [3])
    assert_values(vector.to_list(), [1.0, 2.0, 3.0])

    let grid = as_array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    defer { grid.free() }
    assert_eq(grid.shape, [3, 2])
    assert_values(grid.to_list(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    assert_throws(TypeError, || as_array("not an array"))
}

fn test_host_and_device_round_trip() -> void {
    assert_values(host([2.5, -1.0]), [2.5, -1.0])
    if not gpu.is_available() { return }
    let vector = device_vector([4.0, 5.0, 6.0])
    defer { vector.free() }
    assert_values(host(vector), [4.0, 5.0, 6.0])

    let grid = device_matrix([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
    defer { grid.free() }
    let (values, rows, cols) = host_matrix(grid)
    assert_eq(rows, 2)
    assert_eq(cols, 3)
    assert_values(values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    assert_throws(ValueError, || device_matrix([1.0, 2.0, 3.0], 2, 2))
    assert_throws(ValueError, || host_matrix(vector))
}

fn test_the_bridge_reads_a_view_without_copying_the_parent() -> void {
    if not gpu.is_available() { return }
    let grid = device_matrix([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2)
    defer { grid.free() }
    let column = grid[0..3, 1]
    assert_values(host(column), [2.0, 4.0, 6.0])
    #: A strided window is not dense, so every device pass has to materialise
    #: it first; the answer must not depend on that having happened.
    let doubled = arr_pow(column, 2.0)
    defer { doubled.free() }
    assert_values(host(doubled), [4.0, 16.0, 36.0])
    assert_values(host(grid), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
}

fn test_the_unary_selector_covers_every_op() -> void {
    if not gpu.is_available() { return }
    let x = from_list([-2.0, -0.5, 0.5, 4.0], [4])
    defer { x.free() }

    let magnitude = arr_abs(x)
    defer { magnitude.free() }
    assert_values(host(magnitude), [2.0, 0.5, 0.5, 4.0])

    let marks = arr_sign(x)
    defer { marks.free() }
    assert_values(host(marks), [-1.0, -1.0, 1.0, 1.0])

    let inverted = arr_recip(x)
    defer { inverted.free() }
    assert_values(host(inverted), [-0.5, -2.0, 2.0, 0.25])

    let positive = from_list([1.0, 4.0, 9.0, 16.0], [4])
    defer { positive.free() }
    let roots = arr_sqrt(positive)
    defer { roots.free() }
    assert_values(host(roots), [1.0, 2.0, 3.0, 4.0])

    let logs = arr_log(positive)
    defer { logs.free() }
    assert_values(host(logs), [0.0, 1.3862944, 2.1972246, 2.7725887], 1e-4)

    let grown = arr_exp(from_list([0.0, 1.0], [2]))
    defer { grown.free() }
    assert_values(host(grown), [1.0, 2.7182817], 1e-5)

    let cubed = arr_pow(positive, 0.5)
    defer { cubed.free() }
    assert_values(host(cubed), [1.0, 2.0, 3.0, 4.0])
}

fn test_clip_and_the_two_one_sided_bounds() -> void {
    if not gpu.is_available() { return }
    let x = from_list([-3.0, -1.0, 1.0, 3.0], [4])
    defer { x.free() }

    let held = arr_clip(x, -2.0, 2.0)
    defer { held.free() }
    assert_values(host(held), [-2.0, -1.0, 1.0, 2.0])

    let floored = arr_clip(x, low: 0.0)
    defer { floored.free() }
    assert_values(host(floored), [0.0, 0.0, 1.0, 3.0])

    let capped = arr_clip(x, high: 0.0)
    defer { capped.free() }
    assert_values(host(capped), [-3.0, -1.0, 0.0, 0.0])

    assert_throws(ValueError, || arr_clip(x))
    assert_throws(ValueError, || arr_clip(x, 2.0, -2.0))
}

fn test_maximum_and_minimum_take_a_scalar_or_an_array() -> void {
    if not gpu.is_available() { return }
    let x = from_list([-3.0, -1.0, 1.0, 3.0], [4])
    defer { x.free() }
    let y = from_list([0.0, 0.0, 0.0, 0.0], [4])
    defer { y.free() }

    let paired = arr_maximum(x, y)
    defer { paired.free() }
    assert_values(host(paired), [0.0, 0.0, 1.0, 3.0])

    let scalar = arr_minimum(x, 0.0)
    defer { scalar.free() }
    assert_values(host(scalar), [-3.0, -1.0, 0.0, 0.0])

    let elementwise = arr_pow(from_list([2.0, 3.0], [2]), from_list([3.0, 2.0], [2]))
    defer { elementwise.free() }
    assert_values(host(elementwise), [8.0, 9.0], 1e-4)

    let wrong = from_list([1.0, 2.0], [2])
    defer { wrong.free() }
    assert_throws(ValueError, || arr_maximum(x, wrong))
}

fn test_matmul_agrees_with_the_definition() -> void {
    if not gpu.is_available() { return }
    let a = device_matrix([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
    defer { a.free() }
    let b = device_matrix([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2)
    defer { b.free() }
    let product = matmul(a, b)
    defer { product.free() }
    assert_eq(product.shape, [2, 2])
    assert_values(host(product), [58.0, 64.0, 139.0, 154.0], 1e-3)

    assert_throws(ValueError, || matmul(a, a))
}

#: The kernel stages 16x16 tiles, so a size that is neither a multiple of the
#: tile nor of four exercises both the ragged edge and the fallback path that
#: `gpu.matmul_async`'s 16-byte row rule sends it down.
fn test_matmul_on_a_ragged_size() -> void {
    if not gpu.is_available() { return }
    let rows = 19
    let inner = 23
    let cols = 17
    let left = woven(rows, inner)
    let right = woven(inner, cols)
    let a = device_matrix(left, rows, inner)
    defer { a.free() }
    let b = device_matrix(right, inner, cols)
    defer { b.free() }
    let product = matmul(a, b)
    defer { product.free() }
    let got = host(product)
    for row in 0..rows {
        for column in 0..cols {
            var total = 0.0
            for k in 0..inner { total += left[row * inner + k] * right[k * cols + column] }
            assert_near(got[row * cols + column], total, 1e-3)
        }
    }
}

#: Past `gpu.MIN_GPU_ELEMENTS` with every extent divisible by four, the product
#: goes to Metal Performance Shaders instead of the tiled kernel. Both paths
#: have to give the same answer, and only a size like this reaches the first.
fn test_matmul_takes_the_mps_path_when_the_strides_allow_it() -> void {
    if not gpu.is_available() { return }
    let n = 96
    let ramp = arange(0.0, float(n * n))
    defer { ramp.free() }
    let a = device_matrix(host(ramp), n, n)
    defer { a.free() }
    let identity = zeros([n, n])
    defer { identity.free() }
    for index in 0..n { identity[index, index] = 1.0 }
    let product = matmul(a, identity)
    defer { product.free() }
    let got = host(product)
    assert_eq(got.len(), n * n)
    assert_near(got[0], 0.0, 1e-3)
    assert_near(got[n * n - 1], float(n * n - 1), 1e-1)
    assert_near(got[n + 5], float(n + 5), 1e-2)
}

#: `matmul(transpose(x), x)` is the normal-equation product every fitting and
#: stats caller writes, and a jainum transpose is a *view* with swapped strides
#: rather than a copy. The GEMM addresses its operands from slot zero, so it has
#: to materialise that view first; if it ever stopped doing so this test would
#: read the untransposed matrix and still look plausible.
fn test_matmul_materialises_a_transposed_view() -> void {
    if not gpu.is_available() { return }
    let x = device_matrix([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2)
    defer { x.free() }
    let transposed = x.view([2, 3], [x.strides[1], x.strides[0]], x.offset)
    assert_values(host(transposed), [1.0, 3.0, 5.0, 2.0, 4.0, 6.0])
    let normal_equations = matmul(transposed, x)
    defer { normal_equations.free() }
    assert_eq(normal_equations.shape, [2, 2])
    assert_values(host(normal_equations), [35.0, 44.0, 44.0, 56.0], 1e-3)
}

fn test_matvec_and_dot() -> void {
    assert_near(dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), 32.0, 1e-12)
    assert_throws(ValueError, || dot([1.0], [1.0, 2.0]))
    if not gpu.is_available() { return }
    let a = device_matrix([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
    defer { a.free() }
    let applied = matvec(a, [1.0, 1.0, 1.0])
    defer { applied.free() }
    assert_eq(applied.shape, [2])
    assert_values(host(applied), [6.0, 15.0], 1e-4)

    let x = device_vector([1.0, 2.0, 3.0])
    defer { x.free() }
    assert_near(dot(x, x), 14.0, 1e-3)
    assert_near(dot(x, [1.0, 1.0, 1.0]), 6.0, 1e-3)
    assert_throws(ValueError, || matvec(a, [1.0, 1.0]))
}

fn test_dense_lu_solves_and_reports_singularity() -> void {
    let n = 5
    let a = woven(n, n)
    let (factors, pivots, sign) = dense_lu(a, n)
    assert_true(sign == 1 or sign == -1)
    var b: list[float] = []
    for row in 0..n { b.push(float(row) + 0.5) }
    let x = dense_lu_solve(factors, pivots, b, n)
    for row in 0..n {
        var total = 0.0
        for column in 0..n { total += a[row * n + column] * x[column] }
        assert_near(total, b[row], 1e-9)
    }
    let (_singular, _pivots, degenerate) = dense_lu([0.0, 0.0, 0.0, 0.0], 2)
    assert_eq(degenerate, 0)
    assert_throws(ValueError, || dense_lu([1.0, 2.0, 3.0], 2))
}

fn test_dense_lu_handles_several_right_hand_sides() -> void {
    let n = 3
    let a = [4.0, 1.0, 0.0, 1.0, 5.0, 2.0, 0.0, 2.0, 6.0]
    let (factors, pivots, _sign) = dense_lu(a, n)
    let b = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    let x = dense_lu_solve(factors, pivots, b, n, 2)
    assert_eq(x.len(), 6)
    for row in 0..n {
        for rhs in 0..2 {
            var total = 0.0
            for column in 0..n { total += a[row * n + column] * x[column * 2 + rhs] }
            assert_near(total, b[row * 2 + rhs], 1e-10)
        }
    }
}

fn test_dense_cholesky_factors_and_refuses_an_indefinite_matrix() -> void {
    let n = 4
    let raw = woven(n, n)
    #: A^T A + nI is positive definite for any A, which is the cheapest way to
    #: get a test matrix that is not also trivially diagonal.
    var spd: list[float] = []
    for row in 0..n {
        for column in 0..n {
            var total = if row == column { float(n) } else { 0.0 }
            for k in 0..n { total += raw[k * n + row] * raw[k * n + column] }
            spd.push(total)
        }
    }
    let lower = dense_cholesky(spd, n)
    for row in 0..n {
        for column in 0..n {
            var total = 0.0
            for k in 0..n { total += lower[row * n + k] * lower[column * n + k] }
            assert_near(total, spd[row * n + column], 1e-9)
        }
    }
    assert_throws(ValueError, || dense_cholesky([1.0, 2.0, 2.0, 1.0], 2))
}

fn test_dense_qr_is_orthogonal_and_triangular() -> void {
    let rows = 6
    let cols = 3
    let a = woven(rows, cols)
    let (q, r) = dense_qr(a, rows, cols)
    for row in 0..cols {
        for column in 0..cols {
            var total = 0.0
            for k in 0..rows { total += q[k * cols + row] * q[k * cols + column] }
            assert_near(total, if row == column { 1.0 } else { 0.0 }, 1e-10)
        }
    }
    for row in 0..cols {
        for column in 0..row { assert_near(r[row * cols + column], 0.0, 1e-14) }
    }
    for row in 0..rows {
        for column in 0..cols {
            var total = 0.0
            for k in 0..cols { total += q[row * cols + k] * r[k * cols + column] }
            assert_near(total, a[row * cols + column], 1e-10)
        }
    }
    assert_throws(ValueError, || dense_qr(woven(2, 3), 2, 3))
}

fn test_dense_triangular_reads_a_unit_diagonal() -> void {
    let lower = [2.0, 0.0, 0.0, 1.0, 3.0, 0.0, -1.0, 2.0, 4.0]
    let x = dense_triangular(lower, [2.0, 7.0, 9.0], 3, true)
    assert_values(x, [1.0, 2.0, 1.5], 1e-12)

    let upper = [2.0, 1.0, -1.0, 0.0, 3.0, 2.0, 0.0, 0.0, 4.0]
    let y = dense_triangular(upper, [3.0, 8.0, 4.0], 3, false)
    assert_values(y, [1.0, 2.0, 1.0], 1e-12)

    let unit = dense_triangular(lower, [1.0, 2.0, 3.0], 3, true, true)
    assert_values(unit, [1.0, 1.0, 2.0], 1e-12)
}

fn test_dense_tridiagonal_matches_a_dense_solve() -> void {
    let n = 6
    var lower: list[float] = []
    var diagonal: list[float] = []
    var upper: list[float] = []
    var rhs: list[float] = []
    for row in 0..n {
        diagonal.push(4.0 + float(row) * 0.1)
        rhs.push(float(row) - 2.0)
        if row < n - 1 {
            lower.push(-1.0)
            upper.push(-1.0 - float(row) * 0.05)
        }
    }
    let x = dense_tridiagonal(lower, diagonal, upper, rhs)
    for row in 0..n {
        var total = diagonal[row] * x[row]
        if row > 0 { total += lower[row - 1] * x[row - 1] }
        if row < n - 1 { total += upper[row] * x[row + 1] }
        assert_near(total, rhs[row], 1e-11)
    }
    assert_eq(dense_tridiagonal([], [], [], []).len(), 0)
    assert_throws(ValueError, || dense_tridiagonal([1.0], [1.0, 1.0], [], [0.0, 0.0]))
}

fn test_givens_zeros_the_second_component() -> void {
    let (cosine, sine, radius) = givens(3.0, 4.0)
    assert_near(cosine * 3.0 + sine * 4.0, radius, 1e-12)
    assert_near(-sine * 3.0 + cosine * 4.0, 0.0, 1e-12)
    assert_near(math.fabs(radius), 5.0, 1e-12)

    let (one, zero, kept) = givens(2.0, 0.0)
    assert_near(one, 1.0, 1e-15)
    assert_near(zero, 0.0, 1e-15)
    assert_near(kept, 2.0, 1e-15)

    #: A tiny pair still rotates: squaring these would underflow to zero and
    #: hand back a rotation of nothing, which is what `hypot` is here to avoid.
    let (small_c, small_s, small_r) = givens(3.0e-200, 4.0e-200)
    assert_near(small_c, 0.6, 1e-9)
    assert_near(small_s, 0.8, 1e-9)
    assert_true(small_r > 0.0)
}

fn quadratic(v: list[float]) -> float {
    return v[0] * v[0] * 2.0 + v[1] * v[1] + v[0] * v[1]
}

fn residual(v: list[float]) -> list[float] {
    return [v[0] + 2.0 * v[1] - 3.0, v[0] * v[0] - v[1], 4.0 * v[1]]
}

fn test_approx_fprime_scales_its_step() -> void {
    let gradient = approx_fprime([2.0, 5.0], quadratic)
    assert_near(gradient[0], 13.0, 1e-5)
    assert_near(gradient[1], 12.0, 1e-5)

    #: A coordinate of order a thousand needs a step of order a thousand times
    #: epsilon, which is the whole reason the step is relative.
    let far = approx_fprime([1000.0, 0.001], quadratic)
    assert_near(far[0], 4000.001, 1e-2)
    assert_near(far[1], 1000.002, 1e-2)
}

fn test_approx_jacobian_is_row_major() -> void {
    let jacobian = approx_jacobian([1.0, 2.0], residual)
    assert_eq(jacobian.len(), 6)
    assert_near(jacobian[0], 1.0, 1e-6)
    assert_near(jacobian[1], 2.0, 1e-6)
    assert_near(jacobian[2], 2.0, 1e-5)
    assert_near(jacobian[3], -1.0, 1e-6)
    assert_near(jacobian[4], 0.0, 1e-6)
    assert_near(jacobian[5], 4.0, 1e-6)
}

fn test_the_two_shared_records() -> void {
    let fit = OptimizeResult([1.0, 2.0], 0.5, nit: 7, nfev: 21, jac: [0.0, 0.0])
    assert_eq(fit.success, true)
    assert_eq(fit.status, 0)
    assert_eq(fit.nit, 7)
    assert_eq(fit.nfev, 21)
    assert_eq(fit.jac.len(), 2)
    assert_true(fit.to_str().contains("OptimizeResult"))

    let stopped = OptimizeResult([0.0], 1.0, false, 2, "max iterations")
    assert_eq(stopped.success, false)
    assert_eq(stopped.message, "max iterations")
    assert_eq(stopped.jac.len(), 0)

    let root = RootResult(1.5, 9, 11)
    assert_near(root.root, 1.5, 1e-15)
    assert_eq(root.iterations, 9)
    assert_eq(root.function_calls, 11)
    assert_eq(root.converged, true)
    assert_eq(root.flag, "converged")
    assert_true(root.to_str().contains("RootResult"))
}

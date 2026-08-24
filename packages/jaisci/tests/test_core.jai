#: The numeric floor: `Array` and its ops, the sorts, and the host dense
#: factorisations.
#:
#: The device tests return early without a GPU, because `Array` is a device
#: buffer and there is nothing to check without one. The `dense_*` tests always
#: run — they are pure host arithmetic, which is the point of them.
import std.gpu as gpu
import std.math
from std.test import assert_eq, assert_near, assert_throws, assert_true
from jaisci.core import Array, KernelCache, LinregressResult, OdeResult, OptimizeResult
from jaisci.core import RootResult, TestResult
from jaisci.core import abs as abs_of, add, amax, amin, approx_fprime, argmax, argmin, argsort
from jaisci.core import clamp, concat, cumsum, dense_banded, dense_cholesky, dense_lu
from jaisci.core import dense_lu_solve, dense_qr, dense_triangular, dense_tridiagonal
from jaisci.core import divide, dot, element_count, exp as exp_of, f32_bits, givens, log as log_of
from jaisci.core import matmul, maximum, mean, minimum, multiply, neg, offset, power, rank_data
from jaisci.core import scale, searchsorted, sort_values, sqrt as sqrt_of, subtract, sum as sum_of
from jaisci.core import take, transpose

fn assert_values(actual: list[float], expected: list[float], tolerance: float = 1e-5) -> void {
    assert_eq(actual.len(), expected.len())
    for (index, value) in actual.enumerate() { assert_near(value, expected[index], tolerance) }
}

fn ramp(count: int) -> Array {
    var values: list[float] = []
    for index in 0..count { values.push(float(index % 97) * 0.5 - 11.0) }
    return Array.from_list(values)
}

fn test_element_count_rejects_nonsense() -> void {
    assert_eq(element_count([2, 3, 4]), 24)
    assert_eq(element_count([3, 0]), 0)
    assert_throws(ValueError, || element_count([]))
    assert_throws(ValueError, || element_count([2, -1]))
}

fn test_f32_bits_round_trips_through_a_kernel() -> void {
    assert_eq(f32_bits(0.0), 0)
    assert_eq(f32_bits(1.0), 0x3F800000)
    assert_eq(f32_bits(-2.0), 0xC0000000)
    if not gpu.is_available() { return }
    let filled = Array.full([4], -3.25)
    defer { filled.free() }
    assert_values(filled.values(), [-3.25, -3.25, -3.25, -3.25])
}

fn test_kernel_cache_hands_back_the_same_pipeline() -> void {
    if not gpu.is_available() { return }
    let cache = KernelCache("""#include <metal_stdlib>
using namespace metal;
kernel void twice(device const float* src [[buffer(0)]],
                  device float* dst [[buffer(1)]],
                  constant uint& count [[buffer(2)]],
                  uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    dst[id] = src[id] * 2.0f;
}
""")
    let first = cache.get("twice")
    assert_true(first is cache.get("twice"))
    let input = Array.from_list([1.0, 2.0, 3.0])
    defer { input.free() }
    let output = Array([3])
    defer { output.free() }
    first.dispatch([input.buffer(), output.buffer()], 3, scalars: [3])
    assert_values(output.values(), [2.0, 4.0, 6.0])
}

fn test_constructors() -> void {
    if not gpu.is_available() { return }
    let zeros = Array.zeros([2, 3])
    defer { zeros.free() }
    assert_eq(zeros.shape, [2, 3])
    assert_eq(zeros.size, 6)
    assert_eq(zeros.rank(), 2)
    assert_eq(zeros.rows(), 2)
    assert_eq(zeros.cols(), 3)
    assert_eq(zeros.dim(-1), 3)
    assert_values(zeros.values(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    let ones = Array.ones([4])
    defer { ones.free() }
    assert_values(ones.values(), [1.0, 1.0, 1.0, 1.0])
    assert_eq(ones.cols(), 1)

    let steps = Array.arange(1.0, 4.0)
    defer { steps.free() }
    assert_values(steps.values(), [1.0, 2.0, 3.0])

    let spread = Array.linspace(0.0, 1.0, 5)
    defer { spread.free() }
    assert_values(spread.values(), [0.0, 0.25, 0.5, 0.75, 1.0])

    let identity = Array.eye(3)
    defer { identity.free() }
    assert_values(identity.values(), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    let wide = Array.eye(2, cols: 3)
    defer { wide.free() }
    assert_values(wide.values(), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
}

fn test_views_alias_and_reshape_does_not_copy() -> void {
    if not gpu.is_available() { return }
    let source = Array.from_list([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    defer { source.free() }
    let window = source.view(3, [3])
    defer { window.free() }
    assert_values(window.values(), [4.0, 5.0, 6.0])
    window.upload([7.0, 8.0, 9.0])
    assert_values(source.values(), [1.0, 2.0, 3.0, 7.0, 8.0, 9.0])

    let flat = source.reshape([6])
    defer { flat.free() }
    assert_eq(flat.shape, [6])
    assert_values(flat.values(), [1.0, 2.0, 3.0, 7.0, 8.0, 9.0])
    assert_throws(ValueError, || source.reshape([4]))

    let clone = source.copy()
    defer { clone.free() }
    clone.upload([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert_values(source.values(), [1.0, 2.0, 3.0, 7.0, 8.0, 9.0])
}

fn test_freed_arrays_refuse_to_be_read() -> void {
    if not gpu.is_available() { return }
    let doomed = Array.zeros([2])
    doomed.free()
    assert_true(doomed.is_freed())
    doomed.free()
    assert_throws(RuntimeError, || doomed.values())
}

fn test_elementwise_ops() -> void {
    if not gpu.is_available() { return }
    let a = Array.from_list([1.0, -2.0, 4.0, 9.0])
    defer { a.free() }
    let b = Array.from_list([2.0, 3.0, -1.0, 3.0])
    defer { b.free() }

    let summed = add(a, b)
    defer { summed.free() }
    assert_values(summed.values(), [3.0, 1.0, 3.0, 12.0])

    let difference = subtract(a, b)
    defer { difference.free() }
    assert_values(difference.values(), [-1.0, -5.0, 5.0, 6.0])

    let product = multiply(a, b)
    defer { product.free() }
    assert_values(product.values(), [2.0, -6.0, -4.0, 27.0])

    let quotient = divide(a, b)
    defer { quotient.free() }
    assert_values(quotient.values(), [0.5, -2.0 / 3.0, -4.0, 3.0])

    let larger = maximum(a, b)
    defer { larger.free() }
    assert_values(larger.values(), [2.0, 3.0, 4.0, 9.0])

    let smaller = minimum(a, b)
    defer { smaller.free() }
    assert_values(smaller.values(), [1.0, -2.0, -1.0, 3.0])

    let scaled = scale(a, 3.0)
    defer { scaled.free() }
    assert_values(scaled.values(), [3.0, -6.0, 12.0, 27.0])

    let shifted = offset(a, 0.5)
    defer { shifted.free() }
    assert_values(shifted.values(), [1.5, -1.5, 4.5, 9.5])

    let negated = neg(a)
    defer { negated.free() }
    assert_values(negated.values(), [-1.0, 2.0, -4.0, -9.0])

    let sizes = abs_of(a)
    defer { sizes.free() }
    assert_values(sizes.values(), [1.0, 2.0, 4.0, 9.0])

    let roots = sqrt_of(sizes)
    defer { roots.free() }
    assert_values(roots.values(), [1.0, math.sqrt(2.0), 2.0, 3.0])

    let cubes = power(a, 3.0)
    defer { cubes.free() }
    #: `pow` on the device is a float32 exp of a float32 log, so 9^3 lands a
    #: few units in the last place off 729 rather than on it.
    assert_values(cubes.values(), [1.0, -8.0, 64.0, 729.0], 1e-3)

    let held = clamp(a, -1.0, 5.0)
    defer { held.free() }
    assert_values(held.values(), [1.0, -1.0, 4.0, 5.0])

    let raised = exp_of(Array.from_list([0.0, 1.0]))
    defer { raised.free() }
    assert_values(raised.values(), [1.0, math.E])

    let logs = log_of(Array.from_list([1.0, math.E]))
    defer { logs.free() }
    assert_values(logs.values(), [0.0, 1.0])

    assert_throws(ValueError, || add(a, Array.zeros([3])))
}

fn test_reductions_agree_with_the_host() -> void {
    if not gpu.is_available() { return }
    let values = ramp(20000)
    defer { values.free() }
    let host = values.values()
    var total = 0.0
    var low = host[0]
    var high = host[0]
    var low_at = 0
    var high_at = 0
    for (index, value) in host.enumerate() {
        total += value
        if value < low {
            low = value
            low_at = index
        }
        if value > high {
            high = value
            high_at = index
        }
    }
    assert_near(sum_of(values), total, 0.05)
    assert_near(mean(values), total / float(host.len()), 1e-4)
    assert_near(amin(values), low, 1e-6)
    assert_near(amax(values), high, 1e-6)
    assert_eq(argmin(values), low_at)
    assert_eq(argmax(values), high_at)
    assert_near(dot(values, values), sum_of(multiply(values, values)), 5.0)

    let small = Array.from_list([3.0, -1.0, 7.0, 7.0])
    defer { small.free() }
    assert_near(sum_of(small), 16.0, 1e-6)
    assert_eq(argmax(small), 2)
    assert_eq(argmin(small), 1)
    assert_throws(ValueError, || mean(Array([0])))
}

fn test_cumsum_matches_a_running_total() -> void {
    if not gpu.is_available() { return }
    let values = Array.from_list([1.0, 2.0, 3.0, 4.0, 5.0])
    defer { values.free() }
    let running = cumsum(values)
    defer { running.free() }
    assert_values(running.values(), [1.0, 3.0, 6.0, 10.0, 15.0])

    let wide = Array.ones([5000])
    defer { wide.free() }
    let counted = cumsum(wide)
    defer { counted.free() }
    let got = counted.values()
    assert_near(got[0], 1.0, 1e-6)
    assert_near(got[4999], 5000.0, 1e-3)
}

fn test_matmul_and_transpose() -> void {
    if not gpu.is_available() { return }
    let a = Array.from_list([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    defer { a.free() }
    let b = Array.from_list([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2])
    defer { b.free() }
    let product = matmul(a, b)
    defer { product.free() }
    assert_eq(product.shape, [2, 2])
    assert_values(product.values(), [58.0, 64.0, 139.0, 154.0])

    let flipped = transpose(a)
    defer { flipped.free() }
    assert_eq(flipped.shape, [3, 2])
    assert_values(flipped.values(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])

    assert_throws(ValueError, || matmul(a, a))
}

fn test_large_matmul_takes_the_mps_path() -> void {
    if not gpu.is_available() { return }
    let a = Array.ones([64, 64])
    defer { a.free() }
    let b = Array.full([64, 64], 0.5)
    defer { b.free() }
    let product = matmul(a, b)
    defer { product.free() }
    let got = product.values()
    assert_near(got[0], 32.0, 1e-3)
    assert_near(got[4095], 32.0, 1e-3)
}

fn test_concat_and_take() -> void {
    if not gpu.is_available() { return }
    let a = Array.from_list([1.0, 2.0, 3.0, 4.0], [2, 2])
    defer { a.free() }
    let b = Array.from_list([5.0, 6.0], [1, 2])
    defer { b.free() }
    let stacked = concat([a, b])
    defer { stacked.free() }
    assert_eq(stacked.shape, [3, 2])
    assert_values(stacked.values(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    let c = Array.from_list([9.0, 8.0], [2, 1])
    defer { c.free() }
    let beside = concat([a, c], axis: 1)
    defer { beside.free() }
    assert_eq(beside.shape, [2, 3])
    assert_values(beside.values(), [1.0, 2.0, 9.0, 3.0, 4.0, 8.0])

    let flat_indices = Array.from_list([3.0, 0.0, -1.0])
    defer { flat_indices.free() }
    let picked = take(a, flat_indices)
    defer { picked.free() }
    assert_values(picked.values(), [4.0, 1.0, 4.0])

    let rows = Array.from_list([1.0, 0.0])
    defer { rows.free() }
    let gathered = take(a, rows, axis: 0)
    defer { gathered.free() }
    assert_eq(gathered.shape, [2, 2])
    assert_values(gathered.values(), [3.0, 4.0, 1.0, 2.0])
}

fn test_sorting_agrees_on_both_sides_of_the_gpu_threshold() -> void {
    if not gpu.is_available() { return }
    for count in [64, 5000, 9000] {
        let values = ramp(count)
        defer { values.free() }
        let ordered = sort_values(values)
        defer { ordered.free() }
        let expected = values.values().sort()
        assert_values(ordered.values(), expected)

        let order = argsort(values)
        defer { order.free() }
        let host = values.values()
        let positions = order.values()
        assert_eq(positions.len(), count)
        for index in 1..count {
            assert_true(host[int(positions[index - 1])] <= host[int(positions[index])])
        }
    }
}

fn test_searchsorted_and_rank_data() -> void {
    if not gpu.is_available() { return }
    let sorted_values = Array.from_list([1.0, 2.0, 2.0, 5.0])
    defer { sorted_values.free() }
    let queries = Array.from_list([0.0, 2.0, 3.0, 9.0])
    defer { queries.free() }
    let left = searchsorted(sorted_values, queries)
    defer { left.free() }
    assert_values(left.values(), [0.0, 1.0, 3.0, 4.0])
    let right = searchsorted(sorted_values, queries, right: true)
    defer { right.free() }
    assert_values(right.values(), [0.0, 3.0, 3.0, 4.0])

    let sample = Array.from_list([10.0, 20.0, 20.0, 30.0])
    defer { sample.free() }
    let ranks = rank_data(sample)
    defer { ranks.free() }
    assert_values(ranks.values(), [1.0, 2.5, 2.5, 4.0])
}

fn test_dense_lu_solves_and_signs() -> void {
    let a = [4.0, 3.0, 6.0, 3.0]
    let (lu, pivots, sign) = dense_lu(a, 2)
    assert_eq(sign, -1)
    let x = dense_lu_solve(lu, pivots, [10.0, 12.0], 2)
    assert_values(x, [1.0, 2.0], 1e-9)
    var check: list[float] = []
    for row in 0..2 {
        var total = 0.0
        for column in 0..2 { total += a[row * 2 + column] * x[column] }
        check.push(total)
    }
    assert_values(check, [10.0, 12.0], 1e-9)

    let (_singular, _pivots, zero) = dense_lu([1.0, 2.0, 2.0, 4.0], 2)
    assert_eq(zero, 0)
    assert_throws(ValueError, || dense_lu([1.0, 2.0], 2))
}

fn test_dense_lu_handles_several_right_hand_sides() -> void {
    let a = [2.0, 1.0, 1.0, 3.0]
    let (lu, pivots, _sign) = dense_lu(a, 2)
    let x = dense_lu_solve(lu, pivots, [1.0, 0.0, 0.0, 1.0], 2, nrhs: 2)
    assert_values(x, [0.6, -0.2, -0.2, 0.4], 1e-9)
}

fn test_dense_cholesky() -> void {
    let a = [4.0, 2.0, 2.0, 5.0]
    let lower = dense_cholesky(a, 2)
    assert_values(lower, [2.0, 0.0, 1.0, 2.0], 1e-12)
    assert_throws(ValueError, || dense_cholesky([1.0, 2.0, 2.0, 1.0], 2))
}

fn test_dense_qr_reconstructs_the_matrix() -> void {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    let (q, r) = dense_qr(a, 3, 2)
    for row in 0..2 {
        for column in 0..2 {
            if row > column { assert_near(r[row * 2 + column], 0.0, 1e-12) }
        }
    }
    var rebuilt: list[float] = []
    for row in 0..3 {
        for column in 0..2 {
            var total = 0.0
            for k in 0..2 { total += q[row * 2 + k] * r[k * 2 + column] }
            rebuilt.push(total)
        }
    }
    assert_values(rebuilt, a, 1e-10)
    for column in 0..2 {
        var norm = 0.0
        for row in 0..3 { norm += q[row * 2 + column] * q[row * 2 + column] }
        assert_near(norm, 1.0, 1e-12)
    }
    assert_throws(ValueError, || dense_qr([1.0, 2.0], 1, 2))
}

fn test_dense_triangular() -> void {
    let lower = [2.0, 0.0, 1.0, 3.0]
    assert_values(dense_triangular(lower, [4.0, 11.0], 2, true), [2.0, 3.0], 1e-12)
    let upper = [2.0, 1.0, 0.0, 3.0]
    assert_values(dense_triangular(upper, [7.0, 9.0], 2, false), [2.0, 3.0], 1e-12)
    let unit = dense_triangular([1.0, 0.0, 5.0, 1.0], [2.0, 3.0], 2, true, unit: true)
    assert_values(unit, [2.0, -7.0], 1e-12)
}

fn test_dense_tridiagonal_solves_a_second_difference() -> void {
    let n = 6
    var lower: list[float] = []
    var diagonal: list[float] = []
    var upper: list[float] = []
    var rhs: list[float] = []
    for index in 0..n {
        diagonal.push(2.0)
        rhs.push(1.0)
        if index < n - 1 {
            lower.push(-1.0)
            upper.push(-1.0)
        }
    }
    let x = dense_tridiagonal(lower, diagonal, upper, rhs)
    for row in 0..n {
        var total = 2.0 * x[row]
        if row > 0 { total -= x[row - 1] }
        if row < n - 1 { total -= x[row + 1] }
        assert_near(total, 1.0, 1e-10)
    }
}

fn test_dense_banded_matches_a_dense_solve() -> void {
    let n = 5
    let lower = 1
    let upper = 1
    var dense: list[float] = []
    for row in 0..n {
        for column in 0..n {
            var value = 0.0
            if row == column {
                value = 4.0 + float(row)
            } elif row == column + 1 {
                value = -1.0
            } elif column == row + 1 {
                value = 2.0
            }
            dense.push(value)
        }
    }
    var ab: list[float] = []
    for _index in 0..(lower + upper + 1) * n { ab.push(0.0) }
    for row in 0..n {
        for column in 0..n {
            if dense[row * n + column] == 0.0 { continue }
            ab[(upper + row - column) * n + column] = dense[row * n + column]
        }
    }
    var rhs: list[float] = []
    for index in 0..n { rhs.push(float(index) + 1.0) }
    let banded = dense_banded(lower, upper, ab, n, rhs)
    let (lu, pivots, _sign) = dense_lu(dense, n)
    let reference = dense_lu_solve(lu, pivots, rhs, n)
    assert_values(banded, reference, 1e-10)
}

fn test_givens_zeros_the_second_component() -> void {
    let (cosine, sine, radius) = givens(3.0, 4.0)
    assert_near(cosine * 3.0 + sine * 4.0, radius, 1e-12)
    assert_near(-sine * 3.0 + cosine * 4.0, 0.0, 1e-12)
    assert_near(math.fabs(radius), 5.0, 1e-12)
    let (one, zero, kept) = givens(2.5, 0.0)
    assert_near(one, 1.0, 1e-12)
    assert_near(zero, 0.0, 1e-12)
    assert_near(kept, 2.5, 1e-12)
}

fn test_result_records_carry_their_fields() -> void {
    let fit = OptimizeResult([1.0, 2.0], 0.5, nit: 7, nfev: 21)
    assert_eq(fit.x, [1.0, 2.0])
    assert_near(fit.fun, 0.5, 1e-12)
    assert_true(fit.success)
    assert_eq(fit.nit, 7)
    assert_eq(fit.jac, [])
    assert_true(fit.to_str().contains("OptimizeResult"))

    let root = RootResult(1.5, iterations: 4, converged: false, flag: "max iterations")
    assert_near(root.root, 1.5, 1e-12)
    assert_true(not root.converged)

    let path = OdeResult([0.0, 0.1], [[1.0], [0.9]], nfev: 12)
    assert_eq(path.t.len(), 2)
    assert_eq(path.y[1], [0.9])

    let verdict = TestResult(2.5, 0.012)
    assert_near(verdict.pvalue, 0.012, 1e-12)

    let line = LinregressResult(2.0, 1.0, rvalue: 0.99)
    assert_near(line.slope, 2.0, 1e-12)
    assert_near(line.rvalue, 0.99, 1e-12)
}

fn test_approx_fprime_finds_a_known_gradient() -> void {
    let quadratic = |x| x[0] * x[0] * 3.0 + x[1] * x[0]
    let gradient = approx_fprime([2.0, 5.0], quadratic)
    assert_near(gradient[0], 17.0, 1e-4)
    assert_near(gradient[1], 2.0, 1e-4)
}

fn test_searchsorted_and_rank_data_on_the_device_path() -> void {
    if not gpu.is_available() { return }
    let count = 6000
    let descending = neg(Array.arange(0.0, float(count)))
    defer { descending.free() }
    let ranks = rank_data(descending)
    defer { ranks.free() }
    let got = ranks.values()
    assert_eq(got.len(), count)
    assert_near(got[0], float(count), 1e-3)
    assert_near(got[count - 1], 1.0, 1e-3)
    assert_near(got[3000], float(count - 3000), 1e-3)

    let sorted_values = Array.from_list([0.0, 10.0, 20.0])
    defer { sorted_values.free() }
    let queries = Array.arange(0.0, float(count))
    defer { queries.free() }
    let found = searchsorted(sorted_values, queries)
    defer { found.free() }
    let places = found.values()
    assert_near(places[0], 0.0, 1e-6)
    assert_near(places[5], 1.0, 1e-6)
    assert_near(places[15], 2.0, 1e-6)
    assert_near(places[25], 3.0, 1e-6)
}

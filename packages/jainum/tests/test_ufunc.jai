#: Tests for jainum's elementwise ufuncs.
#:
#: Every device answer is checked against the definition of the operation
#: computed a second time on the host — `std.math` for the transcendentals,
#: plain arithmetic for the rest — and never against a number this module once
#: printed. A golden number would pass whichever of the two dispatch paths
#: happened to produce it, and the two paths are the whole risk here: a
#: contiguous call and a strided one run different kernels over different
#: index arithmetic, and only one of them is exercised by a fresh array.
#: `test_both_dispatch_paths_answer_the_same` is the test that exists for
#: that, and it is the one to read first when a ufunc goes wrong.
#:
#: Tolerances are relative and per element. `exp` over [-8, 8] spans six
#: orders of magnitude, and a bound scaled to the largest entry would let
#: every small one through unchecked. float32 carries about seven digits, so
#: 2e-5 relative is a real bound for a single transcendental and 1e-6 is one
#: for an operation that only moves bits.
#:
#: NaN is asserted with `value != value` rather than `assert_near`, which
#: refuses a NaN on either side by design, and it is *built* with `math.NAN`
#: rather than with `0.0 / 0.0`: dividing by a float zero raises
#: `DivisionByZeroError` in Jaithon, exactly as it raises `ZeroDivisionError`
#: in Python. Only the array elements ever hold a NaN, never a host
#: expression that made one.
import std.gpu as gpu
import std.math as math
from std.test import assert_eq, assert_near, assert_throws, assert_true
from jainum.array import BOOL, COMPLEX64, FLOAT32, INT32, NDArray, element_count, sl
from jainum.ufunc import absolute, add, angle, arccos, arccosh, arcsin, arcsinh
from jainum.ufunc import arctan, arctan2, arctanh, bitwise_and, bitwise_invert
from jainum.ufunc import bitwise_or, bitwise_xor, cbrt, ceil, clip, copysign, cos
from jainum.ufunc import cosh, deg2rad, degrees, divide, equal, exp, exp2, expm1
from jainum.ufunc import floor, floor_divide, fmax, fmin, fmod, greater
from jainum.ufunc import greater_equal, hypot, invert, isclose, isfinite, isinf
from jainum.ufunc import isnan, left_shift, less, less_equal, log, log10, log1p
from jainum.ufunc import log2, logaddexp, logical_and, logical_not, logical_or
from jainum.ufunc import logical_xor, maximum, minimum, mod, multiply, nan_to_num
from jainum.ufunc import negative, not_equal, positive, power, rad2deg, radians
from jainum.ufunc import reciprocal, remainder, right_shift, rint, round, sign
from jainum.ufunc import sin, sinh, sqrt, square, subtract, tan, tanh, true_divide
from jainum.ufunc import trunc, where

fn have_gpu() -> bool { return gpu.is_available() }

fn is_nan(value: float) -> bool { return value != value }

fn magnitude(value: float) -> float { return if value < 0.0 { -value } else { value } }

#: `std.math` ships no `fmax`/`fmin`, and every operand these two are handed
#: here is finite, so the plain comparison is the whole of the IEEE rule.
fn larger(a: float, b: float) -> float { return if a > b { a } else { b } }

fn smaller(a: float, b: float) -> float { return if a < b { a } else { b } }

fn holding(shape: list[int], values: list[float], dtype: int = FLOAT32) -> NDArray {
    let a = NDArray(shape, dtype)
    a.upload(values)
    return a
}

#: A deterministic spread with no pattern a kernel could accidentally agree
#: with, scaled into `[-half, half]`.
#:
#: Every value is a multiple of 1/64, which float32 holds exactly, so the
#: uploaded copy and the host reference start from the same number and any
#: difference that follows belongs to the operation.
fn spread(count: int, half: float) -> list[float] {
    var out = [0.0 for _index in 0..count]
    var state = 7919
    for index in 0..count {
        state = (state * 1103515245 + 12345) % 2147483648
        out[index] = float(state % 2048 - 1024) / 1024.0 * half
    }
    return out
}

#: Fail unless every element agrees with the host reference.
#:
#: `atol` is a floor under the relative bound, not a second test: a true zero
#: has no relative neighbourhood, and `sin` near a multiple of pi answers a
#: number whose every digit is cancellation.
fn assert_each_near(
    actual: list[float],
    expected: list[float],
    rtol: float,
    atol: float,
    what: str
) -> void {
    assert_eq(actual.len(), expected.len(), f"{what}: length")
    for index in 0..actual.len() {
        let bound = atol + rtol * magnitude(expected[index])
        assert_near(actual[index], expected[index], bound, f"{what} at {index}")
    }
}

#: The unaries whose host reading is a `std.math` call of the same shape,
#: grouped by the domain each is defined on.
#:
#: A table rather than one test apiece for the same reason the module itself
#: is a table: a row is the whole cost of covering a new ufunc, and a suite
#: that costs more than that per operation ends up covering half of them.
fn general_unaries() -> list[tuple[str, fn(NDArray) -> NDArray, fn(float) -> float]] {
    return [
        ("negative", negative, |x| -x),
        ("absolute", absolute, math.fabs),
        ("sign", sign, math.sign),
        ("square", square, |x| x * x),
        ("cbrt", cbrt, math.cbrt),
        ("exp", exp, math.exp),
        ("exp2", exp2, math.exp2),
        ("expm1", expm1, math.expm1),
        ("sin", sin, math.sin),
        ("cos", cos, math.cos),
        ("tan", tan, math.tan),
        ("arctan", arctan, math.atan),
        ("sinh", sinh, math.sinh),
        ("cosh", cosh, math.cosh),
        ("tanh", tanh, math.tanh),
        ("arcsinh", arcsinh, math.asinh),
        ("deg2rad", deg2rad, math.radians),
        ("rad2deg", rad2deg, math.degrees),
        ("floor", floor, math.floor),
        ("ceil", ceil, math.ceil),
        ("trunc", trunc, math.trunc),
    ]
}

fn positive_unaries() -> list[tuple[str, fn(NDArray) -> NDArray, fn(float) -> float]] {
    return [
        ("reciprocal", reciprocal, |x| 1.0 / x),
        ("sqrt", sqrt, math.sqrt),
        ("log", log, math.ln),
        ("log2", log2, math.log2),
        ("log10", log10, math.log10),
        ("log1p", log1p, math.log1p),
    ]
}

fn unit_unaries() -> list[tuple[str, fn(NDArray) -> NDArray, fn(float) -> float]] {
    return [
        ("arcsin", arcsin, math.asin),
        ("arccos", arccos, math.acos),
        ("arctanh", arctanh, math.atanh),
    ]
}

fn check_unary_table(
    table: list[tuple[str, fn(NDArray) -> NDArray, fn(float) -> float]],
    values: list[float]
) -> void {
    let a = holding([values.len()], values)
    defer { a.free() }
    for (name, device, host) in table {
        let got = device(a)
        defer { got.free() }
        assert_each_near(got.to_list(), values.map(host), 2.0e-5, 1.0e-6, name)
    }
}

fn test_the_general_unaries_match_the_host_library() -> void {
    if not have_gpu() { return }
    check_unary_table(general_unaries(), spread(97, 3.0))
}

fn test_the_positive_unaries_match_the_host_library() -> void {
    if not have_gpu() { return }

    #: Away from zero on purpose: `log` near it is unbounded and the tolerance
    #: would have to be a rubber one. The behaviour at and below zero is a
    #: separate test, because there the answer is an infinity or a NaN and not
    #: a number to compare.
    let values = spread(97, 3.9).map(|x| x + 4.0)
    check_unary_table(positive_unaries(), values)
}

fn test_the_bounded_unaries_match_the_host_library() -> void {
    if not have_gpu() { return }
    check_unary_table(unit_unaries(), spread(97, 0.95))

    let above_one = spread(61, 3.0).map(|x| x + 4.0)
    let a = holding([above_one.len()], above_one)
    defer { a.free() }
    let got = arccosh(a)
    defer { got.free() }
    assert_each_near(got.to_list(), above_one.map(math.acosh), 2.0e-5, 1.0e-6, "arccosh")
}

#: The two ufuncs that exist because the naive spelling loses every digit.
#:
#: At x = 1e-7 the float32 sum 1 + x rounds back to 1, so `exp(x) - 1` and
#: `log(1 + x)` both answer exactly zero. The true answers are within one part
#: in 1e7 of x, which is what makes this a test the naive form cannot pass by
#: accident.
fn test_expm1_and_log1p_keep_the_digits_the_naive_form_drops() -> void {
    if not have_gpu() { return }
    let tiny = [1.0e-7, -1.0e-7, 3.0e-8, 1.0e-6, -5.0e-7]
    let a = holding([tiny.len()], tiny)
    defer { a.free() }

    let raised = expm1(a)
    defer { raised.free() }
    let logged = log1p(a)
    defer { logged.free() }

    for index in 0..tiny.len() {
        let x = tiny[index]
        assert_each_near([raised.to_list()[index]], [x + x * x / 2.0], 1.0e-4, 0.0, "expm1")
        assert_each_near([logged.to_list()[index]], [x - x * x / 2.0], 1.0e-4, 0.0, "log1p")
    }
}

fn test_rint_rounds_halves_to_even_and_trunc_rounds_towards_zero() -> void {
    if not have_gpu() { return }
    let values = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, -1.25, 1.75]
    let a = holding([values.len()], values)
    defer { a.free() }

    let nearest = rint(a)
    defer { nearest.free() }
    assert_eq(nearest.to_list(), [-2.0, -2.0, -0.0, 0.0, 2.0, 2.0, 4.0, -1.0, 2.0])

    let towards_zero = trunc(a)
    defer { towards_zero.free() }
    assert_eq(towards_zero.to_list(), values.map(math.trunc))

    #: floor and ceil bracket every element, and the bracket is exactly one
    #: wide unless the element is already whole.
    let below = floor(a)
    defer { below.free() }
    let above = ceil(a)
    defer { above.free() }
    for index in 0..values.len() {
        assert_true(below.to_list()[index] <= values[index], "floor is below")
        assert_true(above.to_list()[index] >= values[index], "ceil is above")
        assert_near(above.to_list()[index] - below.to_list()[index], 1.0, 1.0e-6)
    }
}

fn test_sign_and_absolute_agree_on_every_element() -> void {
    if not have_gpu() { return }
    let values = [-3.5, -1.0, -0.0, 0.0, 1.0, 2.5]
    let a = holding([values.len()], values)
    defer { a.free() }
    let signs = sign(a)
    defer { signs.free() }
    let sizes = absolute(a)
    defer { sizes.free() }

    #: sign times magnitude is the element back again, which is the definition
    #: rather than a table of six answers.
    for index in 0..values.len() {
        assert_near(signs.to_list()[index] * sizes.to_list()[index], values[index], 1.0e-6)
        assert_true(sizes.to_list()[index] >= 0.0, "a magnitude is not negative")
    }
    assert_eq(signs.to_list()[2], 0.0, "the sign of a zero is zero, whichever zero it is")
}

fn test_the_predicates_flag_nan_and_the_infinities() -> void {
    if not have_gpu() { return }
    let big = 3.0e38
    let values = [0.0, 1.0, -1.0, big * 10.0, -big * 10.0, math.NAN]
    let a = holding([values.len()], values)
    defer { a.free() }

    let nans = isnan(a)
    defer { nans.free() }
    let infinities = isinf(a)
    defer { infinities.free() }
    let finite = isfinite(a)
    defer { finite.free() }

    assert_eq(nans.dtype, BOOL, "a predicate answers BOOL")
    assert_eq(nans.to_list(), [0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert_eq(infinities.to_list(), [0.0, 0.0, 0.0, 1.0, 1.0, 0.0])

    #: The three partition the line: every element is finite, infinite or NaN,
    #: and never two of those.
    for index in 0..values.len() {
        let total = nans.to_list()[index] + infinities.to_list()[index] + finite.to_list()[index]
        assert_eq(total, 1.0, f"element {index} is in exactly one class")
    }
}

fn test_logical_not_reads_nan_as_true() -> void {
    if not have_gpu() { return }
    let a = holding([5], [0.0, 1.0, -2.0, math.NAN, 1.0e-30])
    defer { a.free() }
    let flipped = logical_not(a)
    defer { flipped.free() }
    assert_eq(flipped.to_list(), [1.0, 0.0, 0.0, 0.0, 0.0], "only an exact zero is false")
}

#: The structural test this file exists for.
#:
#: A transpose, a reversed slice and a broadcast all leave the contiguous fast
#: path and go through `layout_buffer` and the divmod walk. Their answers must
#: be the answers the fast path gives for the same values, element for
#: element and bit for bit — an approximate agreement would hide an index that
#: is off by a stride.
#: A transposed view, built from `view` rather than borrowed from
#: `manipulate`: this file may only lean on `array`, and a swapped pair of
#: strides is the whole of a transpose anyway.
fn transposed(a: NDArray) -> NDArray {
    return a.view([a.shape[1], a.shape[0]], [a.strides[1], a.strides[0]], a.offset)
}

fn test_both_dispatch_paths_answer_the_same() -> void {
    if not have_gpu() { return }
    let a = holding([4, 6], spread(24, 2.5))
    defer { a.free() }

    let views = [transposed(a), a[sl(null, null, -1)], a[1..3, sl(null, null, 2)]]
    for view in views {
        let dense = view.copy()
        defer { dense.free() }
        assert_true(
            not view.is_contiguous() or view.offset != 0,
            "the view must be off the fast path"
        )

        let strided_answer = tanh(view)
        defer { strided_answer.free() }
        let dense_answer = tanh(dense)
        defer { dense_answer.free() }
        assert_eq(strided_answer.to_list(), dense_answer.to_list(), "unary paths agree")

        let strided_pair = arctan2(view, view)
        defer { strided_pair.free() }
        let dense_pair = arctan2(dense, dense)
        defer { dense_pair.free() }
        assert_eq(strided_pair.to_list(), dense_pair.to_list(), "binary paths agree")

        let strided_flags = greater(view, 0.0)
        defer { strided_flags.free() }
        let dense_flags = greater(dense, 0.0)
        defer { dense_flags.free() }
        assert_eq(strided_flags.to_list(), dense_flags.to_list(), "comparison paths agree")

        let strided_held = clip(view, -1.0, 1.0)
        defer { strided_held.free() }
        let dense_held = clip(dense, -1.0, 1.0)
        defer { dense_held.free() }
        assert_eq(strided_held.to_list(), dense_held.to_list(), "ternary paths agree")
    }
}

#: A binary with one strided operand and one dense one still takes the general
#: path, and must land on the same answer as two dense operands.
fn test_a_mixed_pair_takes_the_general_path_to_the_same_answer() -> void {
    if not have_gpu() { return }
    let a = holding([4, 6], spread(24, 2.5))
    defer { a.free() }
    let view = transposed(a)
    let dense = view.copy()
    defer { dense.free() }

    let mixed = hypot(view, dense)
    defer { mixed.free() }
    let both_dense = hypot(dense, dense)
    defer { both_dense.free() }
    assert_eq(mixed.to_list(), both_dense.to_list(), "one strided operand changes nothing")
}

fn test_a_broadcast_binary_matches_the_expanded_pair() -> void {
    if not have_gpu() { return }
    let rows = spread(3, 2.0)
    let columns = spread(5, 2.0)
    let a = holding([3, 1], rows)
    defer { a.free() }
    let b = holding([1, 5], columns)
    defer { b.free() }

    let got = hypot(a, b)
    defer { got.free() }
    assert_eq(got.shape, [3, 5])

    var expected = [0.0 for _index in 0..15]
    for row in 0..3 {
        for column in 0..5 {
            expected[row * 5 + column] = math.hypot(rows[row], columns[column])
        }
    }
    assert_each_near(got.to_list(), expected, 2.0e-6, 1.0e-7, "broadcast hypot")
}

fn test_a_scalar_may_stand_on_either_side() -> void {
    if not have_gpu() { return }
    let values = spread(11, 2.0)
    let a = holding([values.len()], values)
    defer { a.free() }

    let left = arctan2(a, 2.0)
    defer { left.free() }
    let right = arctan2(2.0, a)
    defer { right.free() }
    assert_each_near(left.to_list(), values.map(|x| math.atan2(x, 2.0)), 2.0e-6, 1.0e-7, "y array")
    assert_each_near(right.to_list(), values.map(|x| math.atan2(2.0, x)), 2.0e-6, 1.0e-7, "x array")

    #: The VM has no reflected dunder, so `2.0 - a` cannot reach `__sub__`.
    #: These take both orders, which is the reason they exist beside the
    #: operators.
    let flipped = subtract(2.0, a)
    defer { flipped.free() }
    assert_each_near(flipped.to_list(), values.map(|x| 2.0 - x), 1.0e-6, 1.0e-7, "scalar minus")

    assert_throws(TypeError, || arctan2(1.0, 2.0), "two scalars have no array to shape")
}

fn test_the_arithmetic_re_exports_are_the_operators() -> void {
    if not have_gpu() { return }
    let a = holding([4], [1.0, 2.0, 3.0, 4.0])
    defer { a.free() }
    let b = holding([4], [5.0, 6.0, 7.0, 8.0])
    defer { b.free() }

    let summed = add(a, b)
    defer { summed.free() }
    let operator_sum = a + b
    defer { operator_sum.free() }
    assert_eq(summed.to_list(), operator_sum.to_list())

    let quotient = divide(a, b)
    defer { quotient.free() }
    let also = true_divide(a, b)
    defer { also.free() }
    assert_eq(quotient.to_list(), also.to_list())

    let product = multiply(a, b)
    defer { product.free() }
    assert_eq(product.to_list(), [5.0, 12.0, 21.0, 32.0])
}

fn test_power_carries_the_sign_of_a_negative_base() -> void {
    if not have_gpu() { return }
    let bases = holding([6], [-2.0, -2.0, -2.0, 2.0, 9.0, 0.0])
    defer { bases.free() }
    let exponents = holding([6], [3.0, 2.0, 0.5, 0.5, 0.5, 0.0])
    defer { exponents.free() }
    let got = power(bases, exponents)
    defer { got.free() }

    assert_near(got.to_list()[0], -8.0, 1.0e-5, "an odd whole power keeps the sign")
    assert_near(got.to_list()[1], 4.0, 1.0e-5, "an even one loses it")
    assert_true(is_nan(got.to_list()[2]), "a negative base to a fractional power is not real")
    assert_near(got.to_list()[3], math.sqrt(2.0), 1.0e-5)
    assert_near(got.to_list()[4], 3.0, 1.0e-5)
    assert_near(got.to_list()[5], 1.0, 1.0e-6, "anything to the zeroth is one")

    #: Against the definition over a whole spread: a positive base raised to
    #: an arbitrary exponent is exp(b * log(a)).
    let values = spread(31, 2.0).map(|x| x + 3.0)
    let a = holding([values.len()], values)
    defer { a.free() }
    let cubed = power(a, 1.7)
    defer { cubed.free() }
    let by_logs = values.map(|x| math.exp(1.7 * math.ln(x)))
    assert_each_near(cubed.to_list(), by_logs, 4.0e-5, 1.0e-6, "power against exp of log")
}

fn test_remainder_takes_the_divisor_sign_and_fmod_the_dividend() -> void {
    if not have_gpu() { return }
    let a = holding([4], [5.0, -5.0, 5.0, -5.0])
    defer { a.free() }
    let b = holding([4], [3.0, 3.0, -3.0, -3.0])
    defer { b.free() }

    let python_style = remainder(a, b)
    defer { python_style.free() }
    let c_style = fmod(a, b)
    defer { c_style.free() }

    assert_each_near(python_style.to_list(), [2.0, 1.0, -1.0, -2.0], 1.0e-6, 1.0e-7, "remainder")
    assert_each_near(c_style.to_list(), [2.0, -2.0, 2.0, -2.0], 1.0e-6, 1.0e-7, "fmod")

    let also = mod(a, b)
    defer { also.free() }
    assert_eq(also.to_list(), python_style.to_list(), "mod is remainder, not fmod")
}

#: The identity that ties the pair together: `floor_divide(a, b) * b +
#: remainder(a, b)` is `a`, for every sign of both operands.
fn test_floor_divide_and_remainder_reconstruct_the_dividend() -> void {
    if not have_gpu() { return }
    let numerators = spread(41, 20.0)
    let denominators = spread(41, 5.0).map(|x| if x < 0.0 { x - 1.0 } else { x + 1.0 })
    let a = holding([41], numerators)
    defer { a.free() }
    let b = holding([41], denominators)
    defer { b.free() }

    let whole = floor_divide(a, b)
    defer { whole.free() }
    let rest = remainder(a, b)
    defer { rest.free() }

    for index in 0..41 {
        let rebuilt = whole.to_list()[index] * denominators[index] + rest.to_list()[index]
        assert_near(rebuilt, numerators[index], 1.0e-4, f"reconstruction at {index}")

        #: And the remainder sits in the half-open interval the divisor spans,
        #: which is what "the sign of the divisor" means.
        let scaled = rest.to_list()[index] / denominators[index]
        assert_true(scaled >= 0.0 and scaled < 1.0, f"remainder in range at {index}")
    }
}

fn test_hypot_survives_a_pair_that_would_overflow_squared() -> void {
    if not have_gpu() { return }
    let a = holding([3], [3.0e30, 3.0, 0.0])
    defer { a.free() }
    let b = holding([3], [4.0e30, 4.0, 0.0])
    defer { b.free() }
    let got = hypot(a, b)
    defer { got.free() }

    #: 3e30 squared is an infinity in float32 and 5e30 is not, so the naive
    #: sqrt(a*a + b*b) answers inf here and this is the whole point of the
    #: scaled form.
    assert_near(got.to_list()[0] / 1.0e30, 5.0, 1.0e-4, "a large pair keeps its answer")
    assert_near(got.to_list()[1], 5.0, 1.0e-6)
    assert_eq(got.to_list()[2], 0.0, "two zeros are zero apart, not NaN")
}

#: An infinity on either side is infinitely far from anything, a NaN on the
#: other side included — IEEE-754 says so and numpy 2.5.2 agrees. The scaled
#: form cannot reach that on its own: the larger magnitude is the NaN, and
#: inf/NaN is a NaN. A NaN met by a finite number stays a NaN.
fn test_hypot_is_infinite_beside_an_infinity() -> void {
    if not have_gpu() { return }
    let inf = 3.0e38 * 10.0
    let nan = math.NAN
    let a = holding([5], [inf, nan, inf, nan, -inf])
    defer { a.free() }
    let b = holding([5], [nan, inf, 3.0, 3.0, nan])
    defer { b.free() }
    let got = hypot(a, b)
    defer { got.free() }

    #: Read through `isinf` rather than through `> 3.0e38`: a comparison
    #: against a NaN raises here, and a regression should fail this test
    #: rather than error out of it.
    let infinite = isinf(got)
    defer { infinite.free() }
    assert_eq(infinite.to_list(), [1.0, 1.0, 1.0, 0.0, 1.0], "an infinity beats a NaN")
    assert_true(is_nan(got.to_list()[3]), "a NaN and a finite number are NaN apart")
    assert_eq(got.to_list()[4], got.to_list()[0], "a magnitude has no sign to keep")
}

fn test_logaddexp_survives_arguments_that_would_overflow() -> void {
    if not have_gpu() { return }
    let a = holding([4], [800.0, -800.0, 0.0, 1.0])
    defer { a.free() }
    let b = holding([4], [800.0, -800.0, 0.0, 2.0])
    defer { b.free() }
    let got = logaddexp(a, b)
    defer { got.free() }

    #: exp(800) is an infinity, so the naive form answers inf; log(2e800) is
    #: 800 + log 2 and that is what the shifted form must find.
    assert_near(got.to_list()[0], 800.0 + math.LN_2, 1.0e-3)
    assert_near(got.to_list()[1], -800.0 + math.LN_2, 1.0e-3)
    assert_near(got.to_list()[2], math.LN_2, 1.0e-6)
    assert_near(got.to_list()[3], math.ln(math.exp(1.0) + math.exp(2.0)), 1.0e-5)
}

fn test_maximum_propagates_nan_and_fmax_steps_over_it() -> void {
    if not have_gpu() { return }
    let nan = math.NAN
    let a = holding([4], [nan, 1.0, 3.0, nan])
    defer { a.free() }
    let b = holding([4], [2.0, nan, 1.0, nan])
    defer { b.free() }

    let bigger = maximum(a, b)
    defer { bigger.free() }
    let stepping = fmax(a, b)
    defer { stepping.free() }
    let smaller = minimum(a, b)
    defer { smaller.free() }
    let stepping_low = fmin(a, b)
    defer { stepping_low.free() }

    assert_true(is_nan(bigger.to_list()[0]), "maximum propagates a NaN operand")
    assert_true(is_nan(bigger.to_list()[1]))
    assert_eq(bigger.to_list()[2], 3.0)
    assert_true(is_nan(smaller.to_list()[0]), "and so does minimum")

    assert_eq(stepping.to_list()[0], 2.0, "fmax steps over the NaN")
    assert_eq(stepping.to_list()[1], 1.0)
    assert_eq(stepping_low.to_list()[0], 2.0)
    assert_true(is_nan(stepping.to_list()[3]), "two NaNs leave nothing to step to")
}

fn test_arctan2_names_the_quadrant_arctan_cannot() -> void {
    if not have_gpu() { return }
    let y = holding([4], [1.0, 1.0, -1.0, -1.0])
    defer { y.free() }
    let x = holding([4], [1.0, -1.0, -1.0, 1.0])
    defer { x.free() }
    let got = arctan2(y, x)
    defer { got.free() }

    let quarter = math.PI / 4.0
    assert_each_near(
        got.to_list(),
        [quarter, 3.0 * quarter, -3.0 * quarter, -quarter],
        1.0e-6,
        1.0e-7,
        "arctan2"
    )
}

#: The rows `atan(y / x)` cannot answer, which is every pair holding a zero
#: and every pair of two infinities.
#:
#: Each expectation is `np.arctan2` on numpy 2.5.2, which is IEEE-754's. The
#: device's own `atan2` answered NaN at the origin and at inf/inf, reflected
#: `atan2(1, -0)` to -pi/2, and dropped the sign of y from `atan2(-0, -8)`.
#: Three of the four are nowhere near each other, so a test that only names
#: the origin lets the other two through.
fn test_arctan2_holds_the_zero_row_and_the_diagonals() -> void {
    if not have_gpu() { return }
    let inf = 3.0e38 * 10.0
    let y = holding([10], [0.0, -0.0, 0.0, -0.0, 1.0, -1.0, 1.0, inf, -inf, inf])
    defer { y.free() }
    let x = holding([10], [0.0, 0.0, -0.0, -8.0, -0.0, -0.0, 0.0, inf, inf, -inf])
    defer { x.free() }
    let got = arctan2(y, x)
    defer { got.free() }

    let half = math.PI / 2.0
    let quarter = math.PI / 4.0
    assert_each_near(
        got.to_list(),
        [0.0, 0.0, math.PI, -math.PI, half, -half, half, quarter, -quarter, 3.0 * quarter],
        1.0e-6,
        1.0e-7,
        "arctan2 at a zero or an infinity"
    )
}

#: The same rows through `angle`, which is `arctan2` over a complex pair and
#: therefore inherits every one of them: a real part of -0 is what a phase of
#: exactly +pi/2 is made of.
fn test_angle_reads_a_negative_zero_real_part() -> void {
    if not have_gpu() { return }
    let parts = [-0.0, 1.0, -0.0, -1.0, -1.0, 0.0, -1.0, -0.0]
    let z = holding([4], parts, COMPLEX64)
    defer { z.free() }
    let got = angle(z)
    defer { got.free() }

    let half = math.PI / 2.0
    assert_each_near(
        got.to_list(),
        [half, -half, math.PI, -math.PI],
        1.0e-6,
        1.0e-7,
        "angle at a negative zero"
    )
}

fn test_copysign_reads_a_negative_zero_as_a_sign() -> void {
    if not have_gpu() { return }
    let a = holding([4], [3.0, 3.0, -3.0, -3.0])
    defer { a.free() }
    let b = holding([4], [1.0, -1.0, 1.0, -0.0])
    defer { b.free() }
    let got = copysign(a, b)
    defer { got.free() }
    assert_eq(got.to_list(), [3.0, -3.0, 3.0, -3.0])
}

fn test_the_comparisons_come_back_bool() -> void {
    if not have_gpu() { return }
    let left = spread(64, 3.0)
    let right = spread(64, 3.0).map(|x| -x)
    let a = holding([64], left)
    defer { a.free() }
    let b = holding([64], right)
    defer { b.free() }

    let table = [
        ("greater", greater(a, b), left.map(|_x| 0.0)),
        ("greater_equal", greater_equal(a, b), left.map(|_x| 0.0)),
        ("less", less(a, b), left.map(|_x| 0.0)),
        ("less_equal", less_equal(a, b), left.map(|_x| 0.0)),
        ("equal", equal(a, b), left.map(|_x| 0.0)),
        ("not_equal", not_equal(a, b), left.map(|_x| 0.0)),
    ]
    for (name, got, _unused) in table {
        defer { got.free() }
        assert_eq(got.dtype, BOOL, f"{name} answers BOOL")
        for value in got.to_list() {
            assert_true(value == 0.0 or value == 1.0, f"{name} answers a flag")
        }
    }

    #: Against the definition, element by element, and against each other: the
    #: six partition every pair into exactly one of less, equal and greater.
    let above = greater(a, b)
    defer { above.free() }
    let below = less(a, b)
    defer { below.free() }
    let same = equal(a, b)
    defer { same.free() }
    let differs = not_equal(a, b)
    defer { differs.free() }
    for index in 0..64 {
        let expected = if left[index] > right[index] { 1.0 } else { 0.0 }
        assert_eq(above.to_list()[index], expected, f"greater at {index}")
        let total = above.to_list()[index] + below.to_list()[index] + same.to_list()[index]
        assert_eq(total, 1.0, f"exactly one of <, ==, > holds at {index}")
        assert_eq(same.to_list()[index] + differs.to_list()[index], 1.0, "== and != are opposites")
    }
}

fn test_equality_reads_nan_as_equal_to_nothing() -> void {
    if not have_gpu() { return }
    let nan = math.NAN
    let a = holding([2], [nan, 1.0])
    defer { a.free() }
    let same = equal(a, a)
    defer { same.free() }
    let differs = not_equal(a, a)
    defer { differs.free() }
    assert_eq(same.to_list(), [0.0, 1.0], "a NaN is not equal to itself")
    assert_eq(differs.to_list(), [1.0, 0.0])
}

fn test_the_logical_trio_matches_its_truth_table() -> void {
    if not have_gpu() { return }
    let a = holding([4], [0.0, 0.0, 2.0, -3.0])
    defer { a.free() }
    let b = holding([4], [0.0, 5.0, 0.0, 1.0])
    defer { b.free() }

    let both = logical_and(a, b)
    defer { both.free() }
    let either = logical_or(a, b)
    defer { either.free() }
    let one_of = logical_xor(a, b)
    defer { one_of.free() }

    assert_eq(both.to_list(), [0.0, 0.0, 0.0, 1.0])
    assert_eq(either.to_list(), [0.0, 1.0, 1.0, 1.0])
    assert_eq(one_of.to_list(), [0.0, 1.0, 1.0, 0.0])
    assert_eq(both.dtype, BOOL)

    #: De Morgan, which no truth table can pass by accident.
    let neither = logical_not(either)
    defer { neither.free() }
    let not_a = logical_not(a)
    defer { not_a.free() }
    let not_b = logical_not(b)
    defer { not_b.free() }
    let and_of_nots = logical_and(not_a, not_b)
    defer { and_of_nots.free() }
    assert_eq(neither.to_list(), and_of_nots.to_list(), "not (a or b) is (not a) and (not b)")
}

fn test_the_bitwise_family_matches_host_integer_arithmetic() -> void {
    if not have_gpu() { return }
    let left = [12.0, 255.0, 1024.0, -7.0, 0.0]
    let right = [10.0, 15.0, 3.0, 3.0, 9.0]
    let a = holding([5], left, INT32)
    defer { a.free() }
    let b = holding([5], right, INT32)
    defer { b.free() }

    let anded = bitwise_and(a, b)
    defer { anded.free() }
    let ored = bitwise_or(a, b)
    defer { ored.free() }
    let xored = bitwise_xor(a, b)
    defer { xored.free() }

    assert_eq(anded.dtype, INT32, "a bitwise result keeps the integer dtype")
    for index in 0..5 {
        let x = int(left[index])
        let y = int(right[index])
        assert_eq(anded.to_list()[index], float(x & y), f"and at {index}")
        assert_eq(ored.to_list()[index], float(x | y), f"or at {index}")
        assert_eq(xored.to_list()[index], float(x ^ y), f"xor at {index}")

        #: The identity that ties the three together, so a kernel that
        #: computed all three the same way would still be caught.
        assert_eq(anded.to_list()[index] + ored.to_list()[index], float(x + y), f"sum at {index}")
    }
}

fn test_the_shifts_are_multiplication_and_division_by_a_power_of_two() -> void {
    if not have_gpu() { return }
    let values = [1.0, 3.0, 100.0, -8.0, 65536.0]
    let counts = [4.0, 2.0, 3.0, 1.0, 6.0]
    let a = holding([5], values, INT32)
    defer { a.free() }
    let b = holding([5], counts, INT32)
    defer { b.free() }

    let up = left_shift(a, b)
    defer { up.free() }
    let down = right_shift(a, b)
    defer { down.free() }

    for index in 0..5 {
        var factor = 1.0
        for _step in 0..int(counts[index]) { factor *= 2.0 }
        assert_eq(up.to_list()[index], values[index] * factor, f"left shift at {index}")
        assert_eq(
            down.to_list()[index],
            math.floor(values[index] / factor),
            f"right shift is arithmetic at {index}"
        )
    }
}

fn test_bitwise_invert_is_the_negative_successor() -> void {
    if not have_gpu() { return }
    let values = [0.0, 1.0, -1.0, 42.0, -43.0]
    let a = holding([5], values, INT32)
    defer { a.free() }
    let flipped = bitwise_invert(a)
    defer { flipped.free() }
    let also = invert(a)
    defer { also.free() }

    #: ~x is -x - 1 in two's complement, which is a reading of the operation
    #: rather than a table of five answers.
    for index in 0..5 {
        assert_eq(flipped.to_list()[index], -values[index] - 1.0, f"invert at {index}")
    }
    assert_eq(also.to_list(), flipped.to_list())
}

fn test_the_bitwise_family_refuses_a_float_dtype() -> void {
    if not have_gpu() { return }
    let a = holding([3], [1.0, 2.0, 3.0])
    defer { a.free() }
    assert_throws(TypeError, || bitwise_and(a, a), "a float has no bits to and")
    assert_throws(TypeError, || left_shift(a, a))
    assert_throws(TypeError, || bitwise_invert(a))
}

fn test_clip_holds_every_element_between_the_bounds() -> void {
    if not have_gpu() { return }
    let values = spread(53, 4.0)
    let a = holding([values.len()], values)
    defer { a.free() }

    let held = clip(a, -1.5, 2.0)
    defer { held.free() }
    for index in 0..values.len() {
        let expected = math.clamp(values[index], -1.5, 2.0)
        assert_near(held.to_list()[index], expected, 1.0e-6, f"clip at {index}")
    }

    #: One open side leaves the other alone.
    let floored = clip(a, 0.0, null)
    defer { floored.free() }
    let ceiled = clip(a, null, 0.0)
    defer { ceiled.free() }
    for index in 0..values.len() {
        assert_near(floored.to_list()[index], larger(values[index], 0.0), 1.0e-6)
        assert_near(ceiled.to_list()[index], smaller(values[index], 0.0), 1.0e-6)
    }

    assert_throws(ValueError, || clip(a), "a clip with no bounds clips nothing")
}

fn test_clip_keeps_nan_and_prefers_the_upper_bound() -> void {
    if not have_gpu() { return }
    let a = holding([3], [math.NAN, 5.0, -5.0])
    defer { a.free() }

    let held = clip(a, -1.0, 1.0)
    defer { held.free() }
    assert_true(is_nan(held.to_list()[0]), "clamp would have pinned the NaN to the low bound")
    assert_eq(held.to_list()[1], 1.0)
    assert_eq(held.to_list()[2], -1.0)

    #: Bounds given the wrong way round yield the upper one, as numpy has it.
    let crossed = clip(a, 3.0, 1.0)
    defer { crossed.free() }
    assert_eq(crossed.to_list()[1], 1.0)
    assert_eq(crossed.to_list()[2], 1.0)
}

fn test_clip_takes_array_bounds_and_broadcasts_them() -> void {
    if not have_gpu() { return }
    let a = holding([2, 3], [0.0, 5.0, -5.0, 2.0, -2.0, 9.0])
    defer { a.free() }
    let low = holding([3], [0.0, 0.0, -1.0])
    defer { low.free() }
    let held = clip(a, low, 4.0)
    defer { held.free() }
    assert_eq(held.shape, [2, 3])
    assert_eq(held.to_list(), [0.0, 4.0, -1.0, 2.0, 0.0, 4.0])
}

fn test_where_selects_without_promoting_against_the_condition() -> void {
    if not have_gpu() { return }
    let condition = holding([5], [1.0, 0.0, 1.0, 0.0, 1.0], BOOL)
    defer { condition.free() }
    let x = holding([5], [10.0, 20.0, 30.0, 40.0, 50.0])
    defer { x.free() }
    let y = holding([5], [-1.0, -2.0, -3.0, -4.0, -5.0])
    defer { y.free() }

    let picked = where(condition, x, y)
    defer { picked.free() }
    assert_eq(picked.to_list(), [10.0, -2.0, 30.0, -4.0, 50.0])
    assert_eq(picked.dtype, FLOAT32, "a BOOL condition does not drag the result down to BOOL")

    #: Scalars on either arm, and a condition that broadcasts against them.
    let scalars = where(condition, 1.0, 0.0)
    defer { scalars.free() }
    assert_eq(scalars.to_list(), condition.to_list(), "where over one and zero is the condition")

    #: A non-boolean condition reads by truthiness, and the arms broadcast.
    let counts = holding([3, 1], [0.0, 2.0, -1.0])
    defer { counts.free() }
    let row = holding([4], [1.0, 2.0, 3.0, 4.0])
    defer { row.free() }
    let mixed = where(counts, row, 0.0)
    defer { mixed.free() }
    assert_eq(mixed.shape, [3, 4])
    assert_eq(mixed.to_list(), [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
}

fn test_isclose_matches_its_own_definition() -> void {
    if not have_gpu() { return }
    let left = spread(37, 2.0)
    let right = spread(37, 2.0).map(|x| x * 1.000002)
    let a = holding([37], left)
    defer { a.free() }
    let b = holding([37], right)
    defer { b.free() }

    let rtol = 1.0e-5
    let atol = 1.0e-8
    let got = isclose(a, b, rtol, atol)
    defer { got.free() }
    assert_eq(got.dtype, BOOL)
    for index in 0..37 {
        let within = magnitude(left[index] - right[index]) <= atol + rtol * magnitude(right[index])
        assert_eq(got.to_list()[index], if within { 1.0 } else { 0.0 }, f"isclose at {index}")
    }
}

fn test_isclose_matches_two_infinities_and_optionally_two_nans() -> void {
    if not have_gpu() { return }
    let big = 3.0e38 * 10.0
    let nan = math.NAN
    let a = holding([4], [big, big, nan, 1.0])
    defer { a.free() }
    let b = holding([4], [big, -big, nan, 1.0])
    defer { b.free() }

    let strict = isclose(a, b)
    defer { strict.free() }
    assert_eq(strict.to_list()[0], 1.0, "|inf - inf| is NaN, so only the exact test can match them")
    assert_eq(strict.to_list()[1], 0.0, "opposite infinities are not close")
    assert_eq(strict.to_list()[2], 0.0, "a NaN is far from everything by default")
    assert_eq(strict.to_list()[3], 1.0)

    let forgiving = isclose(a, b, 1.0e-5, 1.0e-8, true)
    defer { forgiving.free() }
    assert_eq(forgiving.to_list()[2], 1.0, "equal_nan matches two NaNs")
}

fn test_nan_to_num_replaces_only_what_is_not_a_number() -> void {
    if not have_gpu() { return }
    let big = 3.0e38 * 10.0
    let values = [math.NAN, big, -big, 1.5, -2.5, 0.0]
    let a = holding([values.len()], values)
    defer { a.free() }

    let named = nan_to_num(a, -1.0, 99.0, -99.0)
    defer { named.free() }
    assert_eq(named.to_list(), [-1.0, 99.0, -99.0, 1.5, -2.5, 0.0])

    #: The defaults leave a finite answer everywhere, which is the whole
    #: contract of the operation.
    let defaulted = nan_to_num(a)
    defer { defaulted.free() }
    let finite = isfinite(defaulted)
    defer { finite.free() }
    for value in finite.to_list() { assert_eq(value, 1.0, "nothing unfinite survives") }
    assert_eq(defaulted.to_list()[0], 0.0, "NaN defaults to zero")
    assert_true(defaulted.to_list()[1] > 3.0e38, "an infinity defaults to the largest finite float")
}

fn test_absolute_and_angle_read_a_complex_array() -> void {
    if not have_gpu() { return }
    let parts = [3.0, 4.0, -1.0, 0.0, 0.0, -2.0, -1.0, -1.0]
    let z = holding([4], parts, COMPLEX64)
    defer { z.free() }

    let sizes = absolute(z)
    defer { sizes.free() }
    assert_eq(sizes.dtype, FLOAT32, "a complex magnitude is a real number")
    assert_eq(sizes.shape, [4])
    for index in 0..4 {
        let expected = math.hypot(parts[index * 2], parts[index * 2 + 1])
        assert_near(sizes.to_list()[index], expected, 1.0e-5, f"modulus at {index}")
    }

    let phases = angle(z)
    defer { phases.free() }
    for index in 0..4 {
        let expected = math.atan2(parts[index * 2 + 1], parts[index * 2])
        assert_near(phases.to_list()[index], expected, 1.0e-5, f"phase at {index}")
    }

    #: Modulus and phase rebuild the number, which is the definition the two
    #: share and neither can satisfy alone.
    for index in 0..4 {
        let modulus = sizes.to_list()[index]
        assert_near(modulus * math.cos(phases.to_list()[index]), parts[index * 2], 1.0e-4)
        assert_near(modulus * math.sin(phases.to_list()[index]), parts[index * 2 + 1], 1.0e-4)
    }

    let degrees_of = angle(z, true)
    defer { degrees_of.free() }
    for index in 0..4 {
        assert_near(degrees_of.to_list()[index], math.degrees(phases.to_list()[index]), 1.0e-3)
    }
}

fn test_angle_of_a_real_array_is_zero_or_pi() -> void {
    if not have_gpu() { return }
    let a = holding([4], [2.0, -2.0, 0.0, -0.5])
    defer { a.free() }
    let phases = angle(a)
    defer { phases.free() }
    assert_each_near(phases.to_list(), [0.0, math.PI, 0.0, math.PI], 1.0e-6, 1.0e-6, "real phase")
}

fn test_a_strided_complex_array_takes_the_general_path() -> void {
    if not have_gpu() { return }
    let z = holding([6], [float(index) - 2.0 for index in 0..12], COMPLEX64)
    defer { z.free() }
    let every_other = z[sl(null, null, 2)]
    let dense = every_other.copy()
    defer { dense.free() }

    let strided_answer = absolute(every_other)
    defer { strided_answer.free() }
    let dense_answer = absolute(dense)
    defer { dense_answer.free() }
    assert_eq(strided_answer.to_list(), dense_answer.to_list(), "the two complex paths agree")
}

fn test_every_other_entry_point_refuses_a_complex_array() -> void {
    if not have_gpu() { return }
    let z = holding([2], [1.0, 2.0, 3.0, 4.0], COMPLEX64)
    defer { z.free() }
    let real = holding([2], [1.0, 2.0])
    defer { real.free() }

    assert_throws(TypeError, || sqrt(z), "a complex square root has two answers and no lane")
    assert_throws(TypeError, || sign(z))
    assert_throws(TypeError, || greater(z, real))
    assert_throws(TypeError, || greater(real, z))
    assert_throws(TypeError, || clip(z, 0.0, 1.0))
    assert_throws(TypeError, || where(z, real, real))
    assert_throws(TypeError, || isclose(z, z))
}

fn test_the_result_dtype_follows_the_kind_of_the_ufunc() -> void {
    if not have_gpu() { return }
    let ints = holding([4], [1.0, 4.0, 9.0, 16.0], INT32)
    defer { ints.free() }
    let flags = holding([4], [1.0, 0.0, 1.0, 0.0], BOOL)
    defer { flags.free() }

    #: A transcendental of an integer array is not an integer array.
    let roots = sqrt(ints)
    defer { roots.free() }
    assert_eq(roots.dtype, FLOAT32)
    assert_eq(roots.to_list(), [1.0, 2.0, 3.0, 4.0])

    #: But the shape-preserving ones keep it, rounding through the dtype on
    #: the way out.
    let flipped = negative(ints)
    defer { flipped.free() }
    assert_eq(flipped.dtype, INT32)
    assert_eq(flipped.to_list(), [-1.0, -4.0, -9.0, -16.0])

    #: `reciprocal` is the deliberate exception: numpy keeps the integer dtype
    #: and so answers zero for every element but one, which is a wart of its
    #: rules rather than a reading of the operation.
    let inverted = reciprocal(ints)
    defer { inverted.free() }
    assert_eq(inverted.dtype, FLOAT32)
    assert_near(inverted.to_list()[3], 1.0 / 16.0, 1.0e-6)

    let compared = greater(ints, 4.0)
    defer { compared.free() }
    assert_eq(compared.dtype, BOOL)

    let promoted = maximum(ints, flags)
    defer { promoted.free() }
    assert_eq(promoted.dtype, INT32, "the lattice lifts BOOL to INT32")
}

fn test_an_integer_result_rounds_and_saturates_on_the_way_out() -> void {
    if not have_gpu() { return }
    let a = holding([3], [3.0, 5.0, 2000000000.0], INT32)
    defer { a.free() }

    #: The exponent is `20` and not `20.0` because a bare number carries its
    #: own type into the promotion: an int32 array raised to a *float* scalar
    #: leaves the integer dtypes, which is numpy's answer too — under NEP 50
    #: `np.power(np.int32 array, 20.0).dtype` is float64, checked against
    #: numpy 2.5.2.
    #:
    #: 3 to the 20th is far past what int32 holds, so the destination clamps
    #: rather than wrapping — the same rule `astype` follows. numpy wraps here
    #: instead (`np.power(int32([3]), 20)` answers -808182895); every jainum
    #: value lives in float32 storage, where the wrapped bit pattern does not
    #: exist to be recovered, so an integer dtype saturates throughout this
    #: package and this test states that rule rather than numpy's.
    let raised = power(a, 20)
    defer { raised.free() }
    assert_eq(raised.dtype, INT32)
    assert_eq(raised.to_list()[0], 2147483520.0, "the largest float32 below 2^31")

    let doubled = add(a, a)
    defer { doubled.free() }
    assert_eq(doubled.to_list()[2], 2147483520.0, "and a sum saturates there too")
}

fn test_round_moves_the_decimal_point_and_keeps_halves_even() -> void {
    if not have_gpu() { return }
    let a = holding([6], [1.2345, -1.2345, 2.5, 3.5, 1234.0, -0.05])
    defer { a.free() }

    let whole = round(a)
    defer { whole.free() }
    assert_eq(whole.to_list()[2], 2.0, "half rounds to even, so 2.5 is 2")
    assert_eq(whole.to_list()[3], 4.0, "and 3.5 is 4")

    let two_places = round(a, 2)
    defer { two_places.free() }
    assert_near(two_places.to_list()[0], 1.23, 1.0e-5)
    assert_near(two_places.to_list()[1], -1.23, 1.0e-5)

    let hundreds = round(a, -2)
    defer { hundreds.free() }
    assert_near(hundreds.to_list()[4], 1200.0, 1.0e-2)

    #: Rounding to n places never moves an element further than half a step of
    #: that size, which is the definition and not a table.
    for index in 0..6 {
        let moved = magnitude(two_places.to_list()[index] - a.to_list()[index])
        assert_true(moved <= 0.005 + 1.0e-5, f"two places moved element {index} too far")
    }
}

fn test_positive_is_a_copy_and_not_a_view() -> void {
    if not have_gpu() { return }
    let a = holding([3], [1.0, 2.0, 3.0])
    defer { a.free() }
    let same = positive(a)
    defer { same.free() }
    assert_eq(same.to_list(), a.to_list())

    same[0] = 99.0
    assert_eq(a.to_list()[0], 1.0, "writing through the copy leaves the original alone")
}

fn test_the_aliases_reach_the_same_kernels() -> void {
    if not have_gpu() { return }
    let a = holding([4], [0.0, 30.0, 90.0, 180.0])
    defer { a.free() }

    let by_name = deg2rad(a)
    defer { by_name.free() }
    let by_alias = radians(a)
    defer { by_alias.free() }
    assert_eq(by_name.to_list(), by_alias.to_list())

    let back = degrees(by_name)
    defer { back.free() }
    let also_back = rad2deg(by_name)
    defer { also_back.free() }
    assert_eq(back.to_list(), also_back.to_list())
    assert_each_near(back.to_list(), a.to_list(), 1.0e-5, 1.0e-5, "degrees round trip")
}

fn test_an_empty_array_produces_an_empty_result() -> void {
    if not have_gpu() { return }
    let empty = NDArray([3, 0], FLOAT32)
    defer { empty.free() }

    let rooted = sqrt(empty)
    defer { rooted.free() }
    assert_eq(rooted.shape, [3, 0])
    assert_eq(rooted.to_list(), [])

    let compared = greater(empty, 0.0)
    defer { compared.free() }
    assert_eq(compared.shape, [3, 0])
    assert_eq(compared.dtype, BOOL)

    let held = clip(empty, 0.0, 1.0)
    defer { held.free() }
    assert_eq(held.to_list(), [])
}

fn test_a_zero_dimensional_array_survives_every_arity() -> void {
    if not have_gpu() { return }
    let a = holding([2, 3], [1.0, 4.0, 9.0, 16.0, 25.0, 36.0])
    defer { a.free() }
    let one = a[1, 1]
    assert_eq(one.shape, [])

    let rooted = sqrt(one)
    defer { rooted.free() }
    assert_eq(rooted.shape, [])
    assert_near(rooted.item(), 5.0, 1.0e-5)

    let paired = maximum(one, 30.0)
    defer { paired.free() }
    assert_near(paired.item(), 30.0, 1.0e-6)

    let held = clip(one, 0.0, 10.0)
    defer { held.free() }
    assert_near(held.item(), 10.0, 1.0e-6)
}

#: The fast path moves a `float4` per thread and falls back to a scalar tail
#: for the last one to three elements. A length that divides by four never
#: reaches that tail, so every length modulo four has to be checked or the
#: ragged case ships untested.
fn test_every_length_modulo_four_is_covered_by_the_vector_path() -> void {
    if not have_gpu() { return }
    for count in 1..18 {
        let values = spread(count, 2.0)
        let a = holding([count], values)
        defer { a.free() }
        let got = exp(a)
        defer { got.free() }
        assert_each_near(got.to_list(), values.map(math.exp), 2.0e-5, 1.0e-6, f"exp of {count}")
    }
}

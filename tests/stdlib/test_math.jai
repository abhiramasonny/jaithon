#: `std.math` — the parts with a right answer. Floating point is compared with
#: `assert_near`; the integer functions are compared exactly, because that is
#: the point of having them separate from the float ones.
import std.math as math

from std.test import assert_eq, assert_true, assert_false, assert_near, assert_throws

fn test_nan_and_infinity_classify_correctly() -> void {
    let nan = math.INF - math.INF
    assert_true(math.is_nan(nan))
    assert_false(math.is_nan(0.0))
    assert_true(math.is_inf(math.INF))
    assert_true(math.is_inf(-math.INF))
    assert_false(math.is_inf(nan))
    assert_true(math.is_finite(1.0))
    assert_false(math.is_finite(math.INF))
    assert_false(math.is_finite(nan))
}

fn test_nan_is_not_equal_to_itself() -> void {
    let nan = math.INF - math.INF
    assert_false(nan == nan)
    assert_false(math.approx_eq(nan, nan))
}

fn test_approx_eq_is_relative_not_absolute() -> void {
    assert_true(math.approx_eq(0.1 + 0.2, 0.3))
    assert_false(math.approx_eq(1.0, 1.1))
    # Absolute tolerance is what saves a comparison against zero.
    assert_true(math.approx_eq(1e-20, 0.0))
}

fn test_rounding_family_agrees_on_the_hard_cases() -> void {
    assert_near(math.floor(-1.5), -2.0)
    assert_near(math.ceil(-1.5), -1.0)
    assert_near(math.trunc(-1.5), -1.0)
    assert_near(math.round(-1.5), -2.0)  # away from zero, not to even
    assert_near(math.round(1.5), 2.0)
    assert_near(math.round(2.5), 3.0)
    assert_near(math.round_to(3.14159, 2), 3.14)
    assert_near(math.fract(-1.25), -0.25)
}

fn test_sign_and_copysign_carry_the_sign_of_zero() -> void {
    assert_near(math.sign(-3.0), -1.0)
    assert_near(math.sign(3.0), 1.0)
    assert_near(math.sign(0.0), 0.0)
    assert_near(math.copysign(3.0, -1.0), -3.0)
    assert_near(math.copysign(-3.0, 1.0), 3.0)
}

fn test_sqrt_and_roots_reject_what_they_cannot_answer() -> void {
    assert_near(math.sqrt(9.0), 3.0)
    assert_near(math.sqrt(0.0), 0.0)
    assert_throws(ValueError, || math.sqrt(-1.0))
    assert_near(math.cbrt(-8.0), -2.0)  # odd root of a negative is real
    assert_near(math.nth_root(16.0, 4), 2.0)
}

fn test_logarithms_reject_non_positive_inputs() -> void {
    assert_near(math.ln(math.E), 1.0)
    assert_near(math.log2(8.0), 3.0)
    assert_near(math.log10(1000.0), 3.0)
    assert_near(math.log_base(81.0, 3.0), 4.0)
    assert_throws(ValueError, || math.ln(0.0))
    assert_throws(ValueError, || math.log2(-1.0))
}

fn test_log1p_and_expm1_keep_precision_near_zero() -> void {
    let tiny = 1e-15
    assert_near(math.log1p(tiny), tiny, 1e-24)
    assert_near(math.expm1(tiny), tiny, 1e-24)
    # The naive spelling loses most of the value at this magnitude, which is
    # the whole reason log1p exists.
    assert_true(math.fabs(math.ln(1.0 + tiny) - tiny) > 1e-24)
}

fn test_trigonometry_at_the_named_angles() -> void {
    assert_near(math.sin(0.0), 0.0)
    assert_near(math.cos(0.0), 1.0)
    assert_near(math.sin(math.PI / 2.0), 1.0)
    assert_near(math.atan2(1.0, 1.0), math.PI / 4.0)
    assert_near(math.degrees(math.PI), 180.0)
    assert_near(math.radians(180.0), math.PI)
    assert_throws(ValueError, || math.asin(2.0))
    assert_throws(ValueError, || math.acos(-2.0))
}

fn test_hypot_does_not_overflow_where_the_naive_form_would() -> void {
    assert_near(math.hypot(3.0, 4.0), 5.0)
    assert_near(math.hypot(0.0, 0.0), 0.0)
}

fn test_clamp_lerp_and_remap() -> void {
    assert_near(math.clamp(5.0, 0.0, 1.0), 1.0)
    assert_near(math.clamp(-5.0, 0.0, 1.0), 0.0)
    assert_near(math.clamp(0.5, 0.0, 1.0), 0.5)
    assert_near(math.lerp(0.0, 10.0, 0.25), 2.5)
    assert_near(math.inverse_lerp(0.0, 10.0, 2.5), 0.25)
    assert_near(math.remap(5.0, 0.0, 10.0, 0.0, 100.0), 50.0)
}

fn test_gcd_and_lcm_handle_zero_and_negatives() -> void {
    assert_eq(math.gcd(12, 18), 6)
    assert_eq(math.gcd(-12, 18), 6)
    assert_eq(math.gcd(0, 5), 5)
    assert_eq(math.gcd(0, 0), 0)
    assert_eq(math.lcm(4, 6), 12)
    assert_eq(math.lcm(0, 5), 0)
}

fn test_isqrt_is_the_exact_floor_of_the_square_root() -> void {
    assert_eq(math.isqrt(0), 0)
    assert_eq(math.isqrt(15), 3)
    assert_eq(math.isqrt(16), 4)
    assert_eq(math.isqrt(17), 4)
    # A float sqrt rounds up here; an exact isqrt must not.
    assert_eq(math.isqrt(4503599761588224), 67108864)
    assert_throws(ValueError, || math.isqrt(-1))
}

fn test_factorial_and_binomials() -> void {
    assert_eq(math.factorial(0), 1)
    assert_eq(math.factorial(5), 120)
    assert_eq(math.combinations(5, 2), 10)
    assert_eq(math.combinations(5, 0), 1)
    assert_eq(math.combinations(5, 6), 0)
    assert_eq(math.permutations(5, 2), 20)
    assert_throws(ValueError, || math.factorial(-1))
}

fn test_bit_helpers() -> void {
    assert_eq(math.next_power_of_two(0), 1)
    assert_eq(math.next_power_of_two(1), 1)
    assert_eq(math.next_power_of_two(17), 32)
    assert_eq(math.bit_count(0), 0)
    assert_eq(math.bit_count(255), 8)
    assert_eq(math.bit_length(0), 0)
    assert_eq(math.bit_length(255), 8)
    assert_eq(math.bit_length(256), 9)
}

fn test_checked_arithmetic_reports_overflow_as_null() -> void {
    assert_eq(math.checked_add(1, 2), 3)
    assert_true(math.checked_add(math.INT_MAX, 1) == null)
    assert_true(math.checked_sub(math.INT_MIN, 1) == null)
    assert_true(math.checked_mul(math.INT_MAX, 2) == null)
    assert_eq(math.checked_mul(0, math.INT_MAX), 0)
}

fn test_abs_of_int_min_cannot_be_represented() -> void {
    assert_eq(math.abs(-5), 5)
    assert_eq(math.abs(0), 0)
    assert_throws(OverflowError, || math.abs(math.INT_MIN))
}

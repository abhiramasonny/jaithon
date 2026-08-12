#: Elementary mathematics: constants, the floating-point functions built on the
#: `f64` primitive surface, and the integer helpers that belong beside them.
#:
#: Non-primitive functions (`expm1`, `log1p`, `hypot`, `cbrt`, ...) are derived
#: in a form chosen to stay accurate where the naive algebra cancels.
#:
#: Domain errors raise `ValueError` rather than return NaN — a NaN return is a
#: silent wrong answer that surfaces untraceably far from its cause. NaN
#: *propagation* is still preserved: a NaN in still yields a NaN out.

#: The ratio of a circle's circumference to its diameter.
pub const PI: float = 3.141592653589793

#: The base of the natural logarithm.
pub const E: float = 2.718281828459045

#: One full turn in radians, `2 * PI`.
pub const TAU: float = 6.283185307179586

#: The golden ratio, `(1 + sqrt(5)) / 2`.
pub const PHI: float = 1.618033988749895

#: The square root of two.
pub const SQRT_2: float = 1.4142135623730951

#: The natural logarithm of two.
pub const LN_2: float = 0.6931471805599453

#: The natural logarithm of ten.
pub const LN_10: float = 2.302585092994046

#: The difference between 1.0 and the next larger float, `2 ** -52`.
#:
#: Resolution near 1.0 only — near 1e9 the float gap is already 1.2e-7. Use
#: `approx_eq`, not a bare `EPSILON`, as a general tolerance.
pub const EPSILON: float = 2.220446049250313e-16

#: The largest representable `int`, `2 ** 63 - 1`.
pub const INT_MAX: int = 9223372036854775807

#: The smallest representable `int`, `-(2 ** 63)`.
#:
#: `-9223372036854775808` is a single literal (spec §1.6): the magnitude only
#: parses behind `-`, so negating this constant raises `OverflowError`.
pub const INT_MIN: int = (-9223372036854775808)

#: Positive infinity.
#:
#: No literal exists for it; IEEE-754 defines overflow of a finite product as
#: exact infinity, not a large finite stand-in.
pub let INF: float = 1e308 * 10.0

#: A quiet not-a-number.
#:
#: `NAN != NAN` is true, so test for it with `is_nan`, never with `==`.
pub let NAN: float = INF - INF

# --- Classification and comparison -------------------------------------------

#: Report whether `x` is a not-a-number value.
pub fn is_nan(x: float) -> bool { return x != x }

#: Report whether `x` is positive or negative infinity.
pub fn is_inf(x: float) -> bool { return x == INF or x == -INF }

#: Report whether `x` is neither infinite nor NaN.
pub fn is_finite(x: float) -> bool { return not is_nan(x) and not is_inf(x) }

#: Report whether `a` and `b` agree to within an absolute or a relative
#: tolerance.
#:
#: Floating rounding error makes `==` unreliable (`0.1 + 0.2 == 0.3` is false).
#: Absolute tolerance covers values near zero, where relative tolerance vanishes;
#: relative tolerance covers large values, where one ulp exceeds any fixed bound.
#: Close when *either* holds; infinities are close only to themselves, NaN to nothing.
#:
#: ```
#: approx_eq(0.1 + 0.2, 0.3)                              # true
#: approx_eq(1.0, 1.5)                                    # false
#: approx_eq(1e9, 1e9 + 1.0)                              # true
#: approx_eq(1e9, 1e9 + 1.0, relative_tolerance: 1e-12)   # false
#: ```
pub fn approx_eq(
    a: float,
    b: float,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-9
) -> bool {
    if absolute_tolerance < 0.0 {
        throw ValueError(f"absolute_tolerance must be non-negative, got {absolute_tolerance}")
    }
    if relative_tolerance < 0.0 {
        throw ValueError(f"relative_tolerance must be non-negative, got {relative_tolerance}")
    }
    if a == b { return true }
    if is_nan(a) or is_nan(b) { return false }
    if is_inf(a) or is_inf(b) { return false }
    let difference = fabs(a - b)
    if difference <= absolute_tolerance { return true }
    return difference <= relative_tolerance * max(fabs(a), fabs(b))
}

# --- Sign and magnitude ------------------------------------------------------

#: Return the absolute value of the float `x`.
#:
#: Total: `fabs(-0.0)` is `0.0` and `fabs(NAN)` is NaN.
pub fn fabs(x: float) -> float {
    if x < 0.0 { return -x }
    if x == 0.0 { return 0.0 }
    return x
}

#: Return the absolute value of the int `n`.
#:
#: Raises `OverflowError` for `INT_MIN`, whose magnitude `2 ** 63` is one past
#: the largest representable int. Use `fabs` for floats.
pub fn abs(n: int) -> int {
    if n == INT_MIN { throw OverflowError("abs(INT_MIN) is 2**63, too large for an int") }
    if n < 0 { return -n }
    return n
}

#: Return -1.0, 0.0 or 1.0 according to the sign of `x`.
#:
#: A zero keeps its sign; a NaN maps to NaN rather than an arbitrary direction.
#:
#: ```
#: sign(-4.5)   # -1.0
#: sign(0.0)    # 0.0
#: ```
pub fn sign(x: float) -> float {
    if is_nan(x) { return NAN }
    if x > 0.0 { return 1.0 }
    if x < 0.0 { return -1.0 }
    return x
}

#: Return `magnitude` carrying the sign of `sign_source`.
#:
#: The sign of a zero is respected — the reason this exists, since `-0.0 < 0.0`
#: is false and no comparison can recover it. A NaN source counts as positive.
#:
#: ```
#: copysign(3.0, -0.0)   # -3.0
#: copysign(-3.0, 2.0)   # 3.0
#: ```
pub fn copysign(magnitude: float, sign_source: float) -> float {
    if _signbit(sign_source) { return -fabs(magnitude) }
    return fabs(magnitude)
}

# --- Rounding ----------------------------------------------------------------

#: Return the largest integral float not greater than `x`.
#:
#: Result is a float — `floor(1e30)` has no int representation. Write
#: `int(floor(x))` when it's known to fit.
pub fn floor(x: float) -> float { return __prim__.f64_floor(x) }

#: Return the smallest integral float not less than `x`.
pub fn ceil(x: float) -> float { return __prim__.f64_ceil(x) }

#: Return `x` with its fractional part discarded, rounding towards zero.
pub fn trunc(x: float) -> float { return __prim__.f64_trunc(x) }

#: Return `x` rounded to the nearest integral float, ties away from zero.
#:
#: ```
#: round(2.5)    # 3.0
#: round(-2.5)   # -3.0
#: ```
pub fn round(x: float) -> float { return __prim__.f64_round(x) }

#: Return `x` rounded to `digits` places after the decimal point.
#:
#: Negative `digits` rounds left of the point. Follows the value actually
#: stored, not the literal typed — `1.005` is held as `1.00499999999999989`.
#:
#: ```
#: round_to(3.14159, 2)   # 3.14
#: round_to(1234.0, -2)   # 1200.0
#: round_to(1.005, 2)     # 1.0
#: ```
pub fn round_to(x: float, digits: int) -> float {
    if digits < -300 or digits > 300 {
        throw ValueError(f"round_to expects digits in [-300, 300], got {digits}")
    }
    if not is_finite(x) { return x }
    if digits == 0 { return round(x) }
    let scale = 10.0 ** digits
    let scaled = x * scale
    # Out-of-range scaling can't change the answer: `x` already holds fewer
    # significant digits than the request asks to keep.
    if not is_finite(scaled) { return x }
    return round(scaled) / scale
}

#: Return the fractional part of `x` with its sign, `x - trunc(x)`.
#:
#: Signed fraction, not the periodic one: `fract(-1.25)` is -0.25, unlike the
#: graphics convention `x - floor(x)` (0.75). Infinity yields a signed zero, matching `modf`.
#:
#: ```
#: fract(2.75)    # 0.75
#: fract(-1.25)   # -0.25
#: ```
pub fn fract(x: float) -> float {
    if is_inf(x) { return copysign(0.0, x) }
    return x - trunc(x)
}

# --- Powers, roots and exponentials ------------------------------------------

#: Return the square root of `x`, raising `ValueError` when `x` is negative.
pub fn sqrt(x: float) -> float {
    if x < 0.0 { throw ValueError(f"sqrt is undefined for a negative argument, got {x}") }
    return __prim__.f64_sqrt(x)
}

#: Return the real cube root of `x`, defined for negative arguments too.
#:
#: `x ** (1.0 / 3.0)` alone is wrong before it starts, because 1/3 is not
#: representable; one Newton step on `r ** 3 - x` recovers the lost bits.
#:
#: ```
#: approx_eq(cbrt(27.0), 3.0)    # true
#: approx_eq(cbrt(-8.0), -2.0)   # true
#: ```
pub fn cbrt(x: float) -> float {
    if x == 0.0 or not is_finite(x) { return x }
    let magnitude = fabs(x)
    let estimate = magnitude ** (1.0 / 3.0)
    let correction = (estimate * estimate * estimate - magnitude) / (3.0 * estimate * estimate)
    return copysign(estimate - correction, x)
}

#: Return the real `n`-th root of `x`.
#:
#: Raises `ValueError` when `n` is zero, when `x` is negative and `n` even (no
#: real root exists), and when `x` is zero and `n` negative (a pole).
#:
#: ```
#: approx_eq(nth_root(81.0, 4), 3.0)     # true
#: approx_eq(nth_root(-32.0, 5), -2.0)   # true
#: ```
pub fn nth_root(x: float, n: int) -> float {
    if n == 0 { throw ValueError("nth_root expects a non-zero n, got 0") }
    if x < 0.0 {
        if n % 2 == 0 {
            throw ValueError(f"nth_root({x}, {n}) has no real value: a negative x needs odd n")
        }
        return -nth_root(-x, n)
    }
    if x == 0.0 {
        if n < 0 { throw ValueError(f"nth_root(0.0, {n}) is undefined for negative n") }
        return 0.0
    }
    if n == 3 { return cbrt(x) }
    return x ** (1.0 / n)
}

#: Return `E ** x`.
#:
#: Overflows to `INF` rather than raising: an exponential too large to store is
#: not a domain error, it is a magnitude the format cannot hold.
pub fn exp(x: float) -> float { return __prim__.f64_exp(x) }

#: Return `2.0 ** x`.
#:
#: Taken as a power of two rather than `exp(x * LN_2)`, which rounds twice and
#: misses integral answers such as `exp2(10.0) == 1024.0`.
pub fn exp2(x: float) -> float { return 2.0 ** x }

#: Return `exp(x) - 1.0` without losing precision for small `x`.
#:
#: Near zero, `exp(x) - 1.0` cancels almost every significant bit. Kahan's
#: identity `(u - 1) * (x / ln(u))` restores them. Grouping matters: `(u - 1) * x`
#: overflows above `x == 700` where the answer is still finite.
#:
#: ```
#: approx_eq(expm1(1e-12), 1e-12)   # true
#: ```
pub fn expm1(x: float) -> float {
    let u = exp(x)
    if is_inf(u) { return u }
    if u == 1.0 { return x }
    # exp underflowed to zero; the answer is -1.0 to every available bit.
    if u - 1.0 == -1.0 { return -1.0 }
    return (u - 1.0) * (x / ln(u))
}

#: Return the Euclidean length `sqrt(x*x + y*y)` without intermediate overflow.
#:
#: Squaring first overflows above `x*x ~ 1.3e154` and goes subnormal below
#: `1.5e-154`, even where the length itself is representable. Factoring out the larger magnitude avoids both.
#:
#: ```
#: hypot(3.0, 4.0)                         # 5.0
#: approx_eq(hypot(3e200, 4e200), 5e200)   # true
#: ```
pub fn hypot(x: float, y: float) -> float {
    let a = fabs(x)
    let b = fabs(y)
    # IEEE-754: an infinite side wins even when the other side is NaN.
    if is_inf(a) or is_inf(b) { return INF }
    if is_nan(a) or is_nan(b) { return NAN }
    let larger = max(a, b)
    let smaller = min(a, b)
    if larger == 0.0 { return 0.0 }
    let ratio = smaller / larger
    return larger * sqrt(1.0 + ratio * ratio)
}

# --- Logarithms --------------------------------------------------------------

#: Return the natural logarithm of `x`.
#:
#: Raises `ValueError` for `x <= 0.0`; zero is a pole, not a value.
pub fn ln(x: float) -> float {
    if x < 0.0 { throw ValueError(f"ln is undefined for a negative argument, got {x}") }
    if x == 0.0 { throw ValueError("ln(0.0) is undefined; the limit is -inf") }
    return __prim__.f64_log(x)
}

#: Return the natural logarithm of `x`; the C spelling of `ln`.
pub fn log(x: float) -> float { return ln(x) }

#: Return the base-2 logarithm of `x`.
#:
#: Read off the exponent `frexp` already holds, so powers of two come out exact
#: instead of as the 9.999999999999998 that `ln(x) / LN_2` produces.
#:
#: ```
#: log2(1024.0)   # 10.0
#: ```
pub fn log2(x: float) -> float {
    if is_nan(x) { return x }
    if x < 0.0 { throw ValueError(f"log2 is undefined for a negative argument, got {x}") }
    if x == 0.0 { throw ValueError("log2(0.0) is undefined; the limit is -inf") }
    if is_inf(x) { return x }
    # Near 1.0 the exponent path cancels catastrophically (frexp's mantissa is
    # just under 1.0). On [0.5, 2.0], x - 1.0 is exact (Sterbenz), so log1p keeps every bit.
    if x > 0.5 and x < 2.0 { return log1p(x - 1.0) / LN_2 }
    let (mantissa, exponent) = frexp(x)
    return exponent + ln(mantissa) / LN_2
}

#: Return the base-10 logarithm of `x`.
#:
#: ```
#: log10(1000.0)   # 3.0
#: ```
pub fn log10(x: float) -> float {
    if is_nan(x) { return x }
    if x < 0.0 { throw ValueError(f"log10 is undefined for a negative argument, got {x}") }
    if x == 0.0 { throw ValueError("log10(0.0) is undefined; the limit is -inf") }
    if is_inf(x) { return x }
    let approximate = ln(x) / LN_10
    # Two roundings leave exact powers of ten just short of their integer;
    # snap back, but only when the input really is that power of ten.
    let nearest = round(approximate)
    if fabs(approximate - nearest) < 1e-10 and _is_power_of_ten(x, nearest) {
        return nearest
    }
    return approximate
}

#: Return `ln(1.0 + x)` without losing precision for small `x`.
#:
#: `1.0 + x` discards bits of `x` below the 53rd, so the naive form returns 0.0
#: under 1e-16. `ln(u) * x / (u - 1)` computes the ratio where the bits still exist.
#:
#: Raises `ValueError` for `x <= -1.0`.
pub fn log1p(x: float) -> float {
    if x < -1.0 { throw ValueError(f"log1p is undefined below -1.0, got {x}") }
    if x == -1.0 { throw ValueError("log1p(-1.0) is undefined; the limit is -inf") }
    if is_inf(x) { return x }
    let u = 1.0 + x
    if u == 1.0 { return x }
    return ln(u) * x / (u - 1.0)
}

#: Return the logarithm of `x` in the given `base`.
#:
#: Raises `ValueError` for a base that is not positive, for base 1.0 (all of
#: whose powers are 1.0), and for `x <= 0.0`.
#:
#: ```
#: approx_eq(log_base(81.0, 3.0), 4.0)   # true
#: log_base(1000.0, 10.0)                # 3.0
#: ```
pub fn log_base(x: float, base: float) -> float {
    if base <= 0.0 { throw ValueError(f"log_base expects a positive base, got {base}") }
    if base == 1.0 {
        throw ValueError("log_base is undefined for base 1.0; every power of 1 is 1")
    }
    # The dedicated routines are exact on their own powers; the ratio is not.
    if base == 2.0 { return log2(x) }
    if base == 10.0 { return log10(x) }
    return ln(x) / ln(base)
}

# --- Trigonometry ------------------------------------------------------------

#: Return the sine of `x` radians.
pub fn sin(x: float) -> float {
    _require_finite(x, "sin")
    return __prim__.f64_sin(x)
}

#: Return the cosine of `x` radians.
pub fn cos(x: float) -> float {
    _require_finite(x, "cos")
    return __prim__.f64_cos(x)
}

#: Return the tangent of `x` radians.
#:
#: No argument is exactly an odd multiple of pi/2 (`PI` is only the nearest
#: float to pi), so the poles are unreachable; near them the result is merely very large.
pub fn tan(x: float) -> float {
    _require_finite(x, "tan")
    return __prim__.f64_tan(x)
}

#: Return the arcsine of `x` in radians, in `[-PI/2, PI/2]`.
#:
#: Raises `ValueError` outside `[-1.0, 1.0]`.
pub fn asin(x: float) -> float {
    if x < -1.0 or x > 1.0 {
        throw ValueError(f"asin expects an argument in [-1.0, 1.0], got {x}")
    }
    return __prim__.f64_asin(x)
}

#: Return the arccosine of `x` in radians, in `[0, PI]`.
#:
#: Raises `ValueError` outside `[-1.0, 1.0]`.
pub fn acos(x: float) -> float {
    if x < -1.0 or x > 1.0 {
        throw ValueError(f"acos expects an argument in [-1.0, 1.0], got {x}")
    }
    return __prim__.f64_acos(x)
}

#: Return the arctangent of `x` in radians, in `(-PI/2, PI/2)`.
pub fn atan(x: float) -> float { return __prim__.f64_atan(x) }

#: Return the angle in radians from the positive x axis to the point `(x, y)`.
#:
#: Unlike `atan(y / x)` this keeps the quadrant, because it sees both signs, and
#: it is defined when `x` is zero. The result lies in `(-PI, PI]`.
#:
#: ```
#: approx_eq(atan2(1.0, 1.0), PI / 4.0)          # true
#: approx_eq(atan2(1.0, -1.0), 3.0 * PI / 4.0)   # true
#: ```
pub fn atan2(y: float, x: float) -> float { return __prim__.f64_atan2(y, x) }

#: Convert `angle` from radians to degrees.
#:
#: ```
#: degrees(PI)   # 180.0
#: ```
pub fn degrees(angle: float) -> float { return angle * (180.0 / PI) }

#: Convert `angle` from degrees to radians.
pub fn radians(angle: float) -> float { return angle * (PI / 180.0) }

# --- Hyperbolic functions ----------------------------------------------------

#: Return the hyperbolic sine of `x`.
#:
#: Built on `expm1`, so small arguments keep full precision where
#: `(exp(x) - exp(-x)) / 2.0` would cancel away every bit of them.
pub fn sinh(x: float) -> float {
    if x == 0.0 or not is_finite(x) { return x }
    let magnitude = fabs(x)
    if magnitude > 709.0 { return copysign(_exp_over_two(magnitude), x) }
    let u = expm1(magnitude)
    return copysign(0.5 * (u + u / (u + 1.0)), x)
}

#: Return the hyperbolic cosine of `x`, which is never less than 1.0.
pub fn cosh(x: float) -> float {
    if is_nan(x) { return x }
    let magnitude = fabs(x)
    if magnitude > 709.0 { return _exp_over_two(magnitude) }
    let scaled = exp(magnitude)
    return 0.5 * (scaled + 1.0 / scaled)
}

#: Return the hyperbolic tangent of `x`, in `(-1.0, 1.0)`.
pub fn tanh(x: float) -> float {
    if x == 0.0 or is_nan(x) { return x }
    let magnitude = fabs(x)
    # Past 20 the distance from 1.0 is below one ulp; the exponential would only underflow.
    if magnitude > 20.0 { return copysign(1.0, x) }
    let u = expm1(-2.0 * magnitude)
    return copysign(-u / (u + 2.0), x)
}

#: Return the inverse hyperbolic sine of `x`, defined for every real argument.
pub fn asinh(x: float) -> float {
    if x == 0.0 or not is_finite(x) { return x }
    let magnitude = fabs(x)
    # Beyond this the square would overflow; asinh has already collapsed into ln(2x) within a rounding.
    if magnitude > 1e150 { return copysign(ln(magnitude) + LN_2, x) }
    let square = magnitude * magnitude
    if magnitude < 1.0 {
        return copysign(log1p(magnitude + square / (1.0 + sqrt(1.0 + square))), x)
    }
    return copysign(ln(magnitude + sqrt(square + 1.0)), x)
}

#: Return the inverse hyperbolic cosine of `x`.
#:
#: Raises `ValueError` for `x < 1.0`, where there is no real value.
pub fn acosh(x: float) -> float {
    if is_nan(x) { return x }
    if x < 1.0 { throw ValueError(f"acosh expects an argument >= 1.0, got {x}") }
    if is_inf(x) { return x }
    if x > 1e150 { return ln(x) + LN_2 }
    let offset = x - 1.0
    # Near x == 1 the log argument approaches 1 and cancels; written via the offset, the expression keeps its bits.
    if offset < 0.5 { return log1p(offset + sqrt(offset * offset + 2.0 * offset)) }
    return ln(x + sqrt(x * x - 1.0))
}

#: Return the inverse hyperbolic tangent of `x`.
#:
#: Raises `ValueError` outside the open interval `(-1.0, 1.0)`; both endpoints
#: are poles.
pub fn atanh(x: float) -> float {
    if is_nan(x) { return x }
    if x <= -1.0 or x >= 1.0 {
        throw ValueError(f"atanh expects an argument in (-1.0, 1.0), got {x}")
    }
    let magnitude = fabs(x)
    return copysign(0.5 * log1p(2.0 * magnitude / (1.0 - magnitude)), x)
}

# --- Interpolation -----------------------------------------------------------

#: Return `value` confined to `[low, high]`.
#:
#: Raises `ValueError` when `low > high`, which has no answer. A NaN `value`
#: passes through: it is neither below nor above anything.
#:
#: ```
#: clamp(5.0, 0.0, 1.0)    # 1.0
#: clamp(-0.5, 0.0, 1.0)   # 0.0
#: ```
pub fn clamp(value: float, low: float, high: float) -> float {
    if low > high {
        throw ValueError(f"clamp expects low <= high, got low={low}, high={high}")
    }
    if value < low { return low }
    if value > high { return high }
    return value
}

#: Return the point `t` of the way from `a` to `b`.
#:
#: Endpoints are exact: `a + t * (b - a)` at `t == 1.0` can round short of `b`,
#: breaking animations that must land exactly. `t` outside `[0.0, 1.0]` extrapolates.
#:
#: ```
#: lerp(0.0, 10.0, 0.25)   # 2.5
#: lerp(2.0, 4.0, 1.0)     # 4.0
#: ```
pub fn lerp(a: float, b: float, t: float) -> float {
    if t == 0.0 { return a }
    if t == 1.0 { return b }
    return a + t * (b - a)
}

#: Return the position of `value` between `a` and `b`, inverting `lerp`.
#:
#: Raises `ValueError` when `a == b`, where every `t` maps to one point and no
#: inverse exists.
#:
#: ```
#: inverse_lerp(10.0, 20.0, 12.5)   # 0.25
#: ```
pub fn inverse_lerp(a: float, b: float, value: float) -> float {
    if a == b { throw ValueError(f"inverse_lerp expects distinct endpoints, both are {a}") }
    return (value - a) / (b - a)
}

#: Map `value` from the range `[in_low, in_high]` onto `[out_low, out_high]`.
#:
#: ```
#: remap(5.0, 0.0, 10.0, 0.0, 100.0)   # 50.0
#: remap(0.0, -1.0, 1.0, 0.0, 255.0)   # 127.5
#: ```
pub fn remap(
    value: float,
    in_low: float,
    in_high: float,
    out_low: float,
    out_high: float
) -> float {
    return lerp(out_low, out_high, inverse_lerp(in_low, in_high, value))
}

# --- Floating-point decomposition --------------------------------------------

#: Return the remainder of `x / y` with the sign of `x`.
#:
#: Exactly `x - trunc(x / y) * y`, with no rounding error of its own. Raises
#: `ValueError` for a zero divisor.
#:
#: ```
#: fmod(7.5, 2.0)    # 1.5
#: fmod(-7.5, 2.0)   # -1.5
#: ```
pub fn fmod(x: float, y: float) -> float {
    if y == 0.0 { throw ValueError("fmod expects a non-zero divisor, got 0.0") }
    return __prim__.f64_fmod(x, y)
}

#: Split `x` into its fractional and integral parts, both signed like `x`.
#:
#: ```
#: modf(-2.75)   # (-0.75, -2.0)
#: ```
pub fn modf(x: float) -> tuple[float, float] { return (fract(x), trunc(x)) }

#: Split `x` into a mantissa in `[0.5, 1.0)` and a power of two.
#:
#: `x == mantissa * 2.0 ** exponent` exactly, with no rounding: this is how the
#: bits are already stored. `frexp(0.0)` is `(0.0, 0)`.
#:
#: ```
#: frexp(12.0)   # (0.75, 4)
#: ```
pub fn frexp(x: float) -> tuple[float, int] { return __prim__.f64_frexp(x) }

#: Return `mantissa * 2.0 ** exponent`, scaling by a power of two exactly.
#:
#: ```
#: ldexp(0.75, 4)   # 12.0
#: ```
pub fn ldexp(mantissa: float, exponent: int) -> float {
    return __prim__.f64_ldexp(mantissa, exponent)
}

# --- Integer mathematics -----------------------------------------------------

#: Return the greatest common divisor of `a` and `b`, always non-negative.
#:
#: `gcd(0, 0)` is 0. Raises `OverflowError` only where the answer is `2 ** 63`,
#: which is `gcd(INT_MIN, 0)` and `gcd(INT_MIN, INT_MIN)`.
#:
#: ```
#: gcd(12, 18)    # 6
#: gcd(-12, 18)   # 6
#: ```
pub fn gcd(a: int, b: int) -> int {
    # Settled first so the loop never evaluates INT_MIN % -1, whose true value
    # is 0 but which traps on two's-complement division hardware.
    if a == 1 or a == -1 or b == 1 or b == -1 { return 1 }
    if a == 0 or b == 0 or a == INT_MIN and b == INT_MIN {
        let survivor = if a == 0 { b } else { a }
        if survivor == INT_MIN {
            throw OverflowError(f"gcd({a}, {b}) is 2**63, which does not fit in an int")
        }
        return abs(survivor)
    }
    var x = a
    var y = b
    while y != 0 {
        let remainder = x % y
        x = y
        y = remainder
    }
    return abs(x)
}

#: Return the least common multiple of `a` and `b`, always non-negative.
#:
#: `lcm(a, 0)` is 0. Raises `OverflowError` when the multiple does not fit in an
#: int, rather than wrapping to a number that is not a multiple of anything.
#:
#: ```
#: lcm(4, 6)    # 12
#: lcm(-4, 6)   # 12
#: ```
pub fn lcm(a: int, b: int) -> int {
    if a == 0 or b == 0 { return 0 }
    # Divide before multiplying: a * b overflows for operands whose lcm fits.
    let quotient = a // gcd(a, b)
    if quotient == INT_MIN or b == INT_MIN {
        throw OverflowError(f"lcm({a}, {b}) does not fit in an int")
    }
    let x = abs(quotient)
    let y = abs(b)
    if x > INT_MAX // y { throw OverflowError(f"lcm({a}, {b}) does not fit in an int") }
    return x * y
}

#: Return the largest int `r` with `r * r <= n`.
#:
#: Exact for every non-negative int, including above 2**53 where the float
#: estimate `int(sqrt(float(n)))` is off by one. Raises `ValueError` for negative `n`.
#:
#: ```
#: isqrt(17)   # 4
#: isqrt(16)   # 4
#: ```
pub fn isqrt(n: int) -> int {
    if n < 0 { throw ValueError(f"isqrt expects a non-negative int, got {n}") }
    if n == 0 { return 0 }
    var root = int(floor(sqrt(float(n))))
    if root < 1 { root = 1 }
    # Divides rather than squares: root * root overflows near INT_MAX, and the float estimate can start on either side.
    while root > n // root { root -= 1 }
    while root + 1 <= n // (root + 1) { root += 1 }
    return root
}

#: Return `n` factorial.
#:
#: Raises `ValueError` for a negative `n` and `OverflowError` above 20, since
#: `21!` exceeds `INT_MAX`.
#:
#: ```
#: factorial(0)   # 1
#: factorial(5)   # 120
#: ```
pub fn factorial(n: int) -> int {
    if n < 0 { throw ValueError(f"factorial is undefined for a negative n, got {n}") }
    if n > 20 {
        throw OverflowError(f"factorial({n}) overflows an int; 20 is the largest argument")
    }
    var result = 1
    for i in 2..=n { result *= i }
    return result
}

#: Return the number of ordered arrangements of `k` items drawn from `n`.
#:
#: Zero when `k > n`. Raises `ValueError` for negative arguments and
#: `OverflowError` when the count exceeds `INT_MAX`.
#:
#: ```
#: permutations(5, 2)   # 20
#: ```
pub fn permutations(n: int, k: int) -> int {
    if n < 0 or k < 0 {
        throw ValueError(f"permutations expects non-negative arguments, got ({n}, {k})")
    }
    if k > n { return 0 }
    var result = 1
    for i in 0..k {
        let factor = n - i
        if result > INT_MAX // factor {
            throw OverflowError(f"permutations({n}, {k}) does not fit in an int")
        }
        result *= factor
    }
    return result
}

#: Return the number of unordered `k`-subsets of `n` items, the binomial
#: coefficient.
#:
#: Every intermediate is itself a binomial coefficient no larger than the
#: answer, unlike the textbook `n! / (k! * (n-k)!)`, which overflows from
#: `n == 21` up. Zero when `k > n`.
#:
#: ```
#: combinations(5, 2)     # 10
#: combinations(62, 31)   # 465428353255261088
#: ```
pub fn combinations(n: int, k: int) -> int {
    if n < 0 or k < 0 {
        throw ValueError(f"combinations expects non-negative arguments, got ({n}, {k})")
    }
    if k > n { return 0 }
    let steps = min(k, n - k)
    var result = 1
    for i in 0..steps {
        let divisor = i + 1
        # result * (n - i) divides evenly by divisor, but result alone need not;
        # cancel the shared factor first to keep the step exact.
        let common = gcd(result, divisor)
        let reduced_result = result // common
        let factor = (n - i) // (divisor // common)
        if reduced_result > INT_MAX // factor {
            throw OverflowError(f"combinations({n}, {k}) does not fit in an int")
        }
        result = reduced_result * factor
    }
    return result
}

# --- Bit-level integer helpers -----------------------------------------------

#: Return the smallest power of two not less than `n`.
#:
#: `next_power_of_two(0)` is 1. Raises `ValueError` for a negative `n` and
#: `OverflowError` above `2 ** 62`, the largest power of two that fits.
#:
#: ```
#: next_power_of_two(17)   # 32
#: next_power_of_two(32)   # 32
#: ```
pub fn next_power_of_two(n: int) -> int {
    if n < 0 { throw ValueError(f"next_power_of_two expects a non-negative int, got {n}") }
    if n <= 1 { return 1 }
    let bits = bit_length(n - 1)
    if bits >= 63 {
        throw OverflowError(f"next_power_of_two({n}) is 2**{bits}, too large for an int")
    }
    return 1 << bits
}

#: Return the number of set bits in the two's-complement representation of `n`.
#:
#: All 64 bits are counted, so `bit_count(-1)` is 64.
#:
#: ```
#: bit_count(255)   # 8
#: bit_count(-1)    # 64
#: ```
pub fn bit_count(n: int) -> int {
    # Sign bit counted separately: left in, the first subtraction below would overflow for INT_MIN.
    let sign_bit = if n < 0 { 1 } else { 0 }
    var x = n & INT_MAX
    x = x - (x >> 1 & 0x5555555555555555)
    x = (x & 0x3333333333333333) + (x >> 2 & 0x3333333333333333)
    x = x + (x >> 4) & 0x0F0F0F0F0F0F0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    x = x + (x >> 32)
    return (x & 0x7F) + sign_bit
}

#: Return the number of bits needed to hold the magnitude of `n`.
#:
#: `bit_length(0)` is 0 and the sign is ignored, so this is the position of the
#: highest set bit of `abs(n)`. `INT_MIN` needs all 64.
#:
#: ```
#: bit_length(255)   # 8
#: bit_length(256)   # 9
#: ```
pub fn bit_length(n: int) -> int {
    if n == INT_MIN { return 64 }
    var x = abs(n)
    var length = 0
    if x >= 0x100000000 {
        x >>= 32
        length += 32
    }
    if x >= 0x10000 {
        x >>= 16
        length += 16
    }
    if x >= 0x100 {
        x >>= 8
        length += 8
    }
    if x >= 0x10 {
        x >>= 4
        length += 4
    }
    if x >= 0x4 {
        x >>= 2
        length += 2
    }
    if x >= 0x2 {
        x >>= 1
        length += 1
    }
    return length + x
}

# --- Checked integer arithmetic ----------------------------------------------

#: Return `a + b`, or null when the sum does not fit in an int.
#:
#: Plain `+` raises `OverflowError`; this is the form for when overflow is an
#: outcome to branch on rather than a bug.
#:
#: ```
#: checked_add(2, 3)         # 5
#: checked_add(INT_MAX, 1)   # null
#: ```
pub fn checked_add(a: int, b: int) -> int? {
    if b > 0 and a > INT_MAX - b { return null }
    if b < 0 and a < INT_MIN - b { return null }
    return a + b
}

#: Return `a - b`, or null when the difference does not fit in an int.
#:
#: ```
#: checked_sub(INT_MIN, 1)   # null
#: ```
pub fn checked_sub(a: int, b: int) -> int? {
    if b < 0 and a > INT_MAX + b { return null }
    if b > 0 and a < INT_MIN + b { return null }
    return a - b
}

#: Return `a * b`, or null when the product does not fit in an int.
#:
#: ```
#: checked_mul(3, 4)          # 12
#: checked_mul(INT_MAX, 2)    # null
#: checked_mul(INT_MIN, -1)   # null
#: ```
pub fn checked_mul(a: int, b: int) -> int? {
    if a == 0 or b == 0 { return 0 }
    if a == INT_MIN or b == INT_MIN {
        # 2**63 has no positive counterpart; the only surviving product is INT_MIN itself.
        if a == 1 or b == 1 { return INT_MIN }
        return null
    }
    let negative = (a < 0) != (b < 0)
    let x = abs(a)
    let y = abs(b)
    let quotient = INT_MAX // y
    if x <= quotient {
        let magnitude = x * y
        if negative { return -magnitude }
        return magnitude
    }
    # One magnitude overshoots INT_MAX yet is representable: exactly 2**63, i.e. INT_MIN when signs disagree.
    if negative and x == quotient + 1 and INT_MAX % y == y - 1 { return INT_MIN }
    return null
}

# --- Statistics --------------------------------------------------------------

#: Return the arithmetic mean of `values`.
#:
#: Summed with Neumaier compensation, which recovers the low-order bits a plain
#: running total loses — `[1e16, 1.0, 1.0]` totals 1e16 naively,
#: 1.0000000000000002e16 here.
#:
#: Raises `ValueError` for an empty list, which has no mean.
#:
#: ```
#: mean([1.0, 2.0, 3.0])   # 2.0
#: ```
pub fn mean(values: list[float]) -> float {
    if values.len() == 0 { throw ValueError("mean of an empty list is undefined") }
    return _compensated_sum(values) / values.len()
}

# --- Private helpers ---------------------------------------------------------

#: Return the sign bit of `x`, distinguishing -0.0 from 0.0.
fn _signbit(x: float) -> bool {
    if x < 0.0 { return true }
    if x > 0.0 { return false }
    if is_nan(x) { return false }
    # Zeros compare equal; IEEE-754 fixes atan2(0.0,-0.0) at PI, atan2(0.0,0.0) at 0.0, so it can distinguish them.
    return atan2(0.0, x) > 0.0
}

#: Return `exp(magnitude) / 2.0` for a magnitude past the range of `exp`.
#:
#: `exp` overflows at 709.78 but sinh/cosh stay finite to 710.48; squaring
#: `exp(m / 2)` keeps every intermediate in range (overflowing to infinity
#: past 710.48, which is correct there).
fn _exp_over_two(magnitude: float) -> float {
    let half = exp(magnitude * 0.5)
    return 0.5 * half * half
}

#: Raise `ValueError` when `x` is infinite, naming the caller.
fn _require_finite(x: float, function_name: str) -> void {
    if is_inf(x) {
        throw ValueError(f"{function_name} is undefined for an infinite argument, got {x}")
    }
}

#: Report whether `x` is exactly ten raised to the integral float `exponent`.
fn _is_power_of_ten(x: float, exponent: float) -> bool {
    # Past 1e22, powers of ten are no longer exactly representable, so no comparison against one can be trusted.
    if fabs(exponent) > 22.0 { return false }
    return 10.0 ** exponent == x
}

#: Return the sum of `values` with Neumaier compensation.
fn _compensated_sum(values: list[float]) -> float {
    var total = 0.0
    var compensation = 0.0
    for value in values {
        let partial = total + value
        # Larger operand keeps its bits; the smaller is where rounding went, so the lost part is recovered from it.
        if fabs(total) >= fabs(value) {
            compensation += total - partial + value
        } else {
            compensation += value - partial + total
        }
        total = partial
    }
    return total + compensation
}

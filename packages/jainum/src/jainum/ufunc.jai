#: Elementwise ufuncs: the unary maths, the broadcasting binaries, the
#: comparisons, and the three-operand selectors.
#:
#: Every kernel here is generated from a table of `(name, Metal expression,
#: result kind)`, the way jaitensor's `elementwise_kernels` builds its own.
#: Adding a ufunc is a table row: the generator folds the expression into two
#: entry points and the host side looks its result rule up by name. The
#: alternative — one kernel switching on an opcode — was measured in
#: jaitensor and lost badly, because a shared body is register-allocated for
#: the worst branch in it and the occupancy of every other branch collapses.
#:
#: Two dispatch paths per op, never one:
#:
#: - a contiguous same-shape fast path, which takes a flat `constant uint&
#:   count` and no metadata at all. It moves a `float4` per thread and falls
#:   back to a scalar tail for the last one to three elements, so a run that
#:   does not divide by four costs one ragged thread rather than a second
#:   kernel. This is what almost every call takes.
#: - the general path through `layout_buffer`, which walks arbitrary strides
#:   and therefore pays a division per axis. A slice, a transpose, a flip and
#:   a broadcast all arrive here.
#:
#: The split exists because the walk is the whole cost. The device has no
#: integer divide; `nf_walk2` and friends take one quotient per axis and
#: recover the index by subtraction rather than issuing a division and a
#: modulo apiece, and they walk every operand in a single pass so a binary
#: dispatch divides once per axis instead of three times. Even so the fast
#: path is worth having: it issues no divisions and reads four floats a
#: thread.
#:
#: There is no threadgroup tiling anywhere in this file, and there should not
#: be: an elementwise op reads each input slot exactly once, so neighbouring
#: threads share nothing and a staging copy through threadgroup memory would
#: be pure overhead. The tiling rule applies to windowed kernels, not these.
#:
#: Complex policy: this module is complex-aware for `absolute` and `angle`
#: only, plus the four arithmetic ops, which are re-exported from `NDArray`'s
#: operators rather than compiled a second time. Every other entry point
#: calls `require_real` first — a lane count of two would double every
#: generated kernel to serve a handful of operations that have no complex
#: reading anyway.
#:
#: Reductions live in `reduce` — `allclose` and `array_equal` included, since
#: both are folds over a comparison rather than comparisons themselves — and
#: `nonzero` and the search form of `where` live in `index`.
from .array import BOOL, COMPLEX64, FLOAT32, INT32
from .array import CAST_BOOL, CAST_NONE, CAST_ROUND
from .array import OP_ADD, OP_DIV, OP_MUL, OP_SUB
from .array import NDArray, PRELUDE, binary_arith, broadcast_shapes
from .array import dtype_is_complex, dtype_is_integer, dtype_max, dtype_min, dtype_name
from .array import f32_bits, kernel_of, layout_buffer, promote, require_real

#: An infinity, which is what an open `clip` bound is.
let _INF = 1.0e308 * 10.0

#: Result rule: the promotion of the operands, unchanged.
pub const KIND_KEEP = 0
#: Result rule: the promotion, lifted out of the integer dtypes. Every
#: transcendental takes this — `sqrt` of an int array is not an int array.
pub const KIND_FLOAT = 1
#: Result rule: `BOOL`, whatever went in. The comparisons and the predicates.
pub const KIND_BOOL = 2
#: Result rule: the promotion, and a float operand is refused. Only the
#: bitwise family, which has no meaning off the integers.
pub const KIND_INT = 3

#: Every unary ufunc as the Metal expression it applies to `value`, plus the
#: rule for its result dtype.
#:
#: `value` is a `float` in the scalar tail and a `float4` in the vector body;
#: the same text is folded into both, which is why an expression may only use
#: operators and the overloaded helpers below. A bare `atan2(0.0f, value)`
#: would not compile in the vector body — MSL will not mix a scalar and a
#: vector inside a maths call, only inside an operator — so anything shaped
#: like that becomes a helper.
#:
#: `p0`, `p1` and `p2` are the float scalars a row may ask for, decoded from
#: the trailing `constant uint&` bits. A row that never spells them gets no
#: declaration.
let UNARY_FORMS: list[tuple[str, str, int]] = [
    ("negative", "-value", KIND_KEEP),
    ("absolute", "fabs(value)", KIND_KEEP),
    ("sign", "nf_sign(value)", KIND_KEEP),
    ("reciprocal", "1.0f / value", KIND_FLOAT),
    ("sqrt", "sqrt(value)", KIND_FLOAT),
    ("cbrt", "nf_cbrt(value)", KIND_FLOAT),
    ("square", "value * value", KIND_KEEP),
    ("exp", "exp(value)", KIND_FLOAT),
    ("exp2", "exp2(value)", KIND_FLOAT),
    ("expm1", "nf_expm1(value)", KIND_FLOAT),
    ("log", "log(value)", KIND_FLOAT),
    ("log2", "log2(value)", KIND_FLOAT),
    ("log10", "log10(value)", KIND_FLOAT),
    ("log1p", "nf_log1p(value)", KIND_FLOAT),
    ("sin", "sin(value)", KIND_FLOAT),
    ("cos", "cos(value)", KIND_FLOAT),
    ("tan", "tan(value)", KIND_FLOAT),
    ("arcsin", "asin(value)", KIND_FLOAT),
    ("arccos", "acos(value)", KIND_FLOAT),
    ("arctan", "atan(value)", KIND_FLOAT),
    ("sinh", "sinh(value)", KIND_FLOAT),
    ("cosh", "cosh(value)", KIND_FLOAT),
    ("tanh", "tanh(value)", KIND_FLOAT),
    ("arcsinh", "asinh(value)", KIND_FLOAT),
    ("arccosh", "acosh(value)", KIND_FLOAT),
    ("arctanh", "atanh(value)", KIND_FLOAT),
    (
        "deg2rad",
        "value * 0.01745329251994329577f",
        KIND_FLOAT
    ),
    (
        "rad2deg",
        "value * 57.29577951308232088f",
        KIND_FLOAT
    ),
    ("floor", "floor(value)", KIND_KEEP),
    ("ceil", "ceil(value)", KIND_KEEP),
    ("trunc", "trunc(value)", KIND_KEEP),
    ("rint", "rint(value)", KIND_KEEP),
    ("isnan", "nf_isnan(value)", KIND_BOOL),
    ("isinf", "nf_isinf(value)", KIND_BOOL),
    ("isfinite", "nf_isfinite(value)", KIND_BOOL),
    ("logical_not", "nf_not(value)", KIND_BOOL),
    ("bitwise_invert", "nf_invert(value)", KIND_INT),
    (
        "nan_to_num",
        "nf_nan_to_num(value, p0, p1, p2)",
        KIND_KEEP
    ),
    ("angle", "nf_angle(value, p0)", KIND_FLOAT),
]

#: The same for the binaries, over `a` and `b`.
#:
#: `add`, `subtract`, `multiply` and `divide` are absent on purpose: they are
#: `NDArray`'s operators, which already handle the complex lanes this table
#: cannot, and re-exporting them costs nothing where compiling them twice
#: would cost a second set of kernels that disagree at the edges.
let BINARY_FORMS: list[tuple[str, str, int]] = [
    ("power", "nf_power(a, b)", KIND_KEEP),
    ("remainder", "nf_remainder(a, b)", KIND_KEEP),
    ("fmod", "fmod(a, b)", KIND_KEEP),
    ("floor_divide", "floor(a / b)", KIND_KEEP),
    ("hypot", "nf_hypot(a, b)", KIND_FLOAT),
    ("arctan2", "nf_atan2(a, b)", KIND_FLOAT),
    ("logaddexp", "nf_logaddexp(a, b)", KIND_FLOAT),
    ("maximum", "nf_maximum(a, b)", KIND_KEEP),
    ("minimum", "nf_minimum(a, b)", KIND_KEEP),
    ("fmax", "nf_fmax(a, b)", KIND_KEEP),
    ("fmin", "nf_fmin(a, b)", KIND_KEEP),
    ("copysign", "copysign(a, b)", KIND_FLOAT),
    ("greater", "nf_gt(a, b)", KIND_BOOL),
    ("greater_equal", "nf_ge(a, b)", KIND_BOOL),
    ("less", "nf_lt(a, b)", KIND_BOOL),
    ("less_equal", "nf_le(a, b)", KIND_BOOL),
    ("equal", "nf_eq(a, b)", KIND_BOOL),
    ("not_equal", "nf_ne(a, b)", KIND_BOOL),
    ("logical_and", "nf_and(a, b)", KIND_BOOL),
    ("logical_or", "nf_or(a, b)", KIND_BOOL),
    ("logical_xor", "nf_xor(a, b)", KIND_BOOL),
    ("bitwise_and", "nf_bit_and(a, b)", KIND_INT),
    ("bitwise_or", "nf_bit_or(a, b)", KIND_INT),
    ("bitwise_xor", "nf_bit_xor(a, b)", KIND_INT),
    ("left_shift", "nf_shl(a, b)", KIND_INT),
    ("right_shift", "nf_shr(a, b)", KIND_INT),
    (
        "isclose",
        "nf_isclose(a, b, p0, p1, p2)",
        KIND_BOOL
    ),
]

#: The three-operand forms, over `a`, `b` and `c`. Both could be written as
#: two binary passes; one kernel reads the operands once instead of writing
#: and re-reading an intermediate the size of the output.
let TERNARY_FORMS: list[tuple[str, str, int]] = [
    ("clip", "nf_clip(a, b, c)", KIND_KEEP),
    ("where", "nf_where(a, b, c)", KIND_KEEP),
]

#: The two complex-aware unaries, over the interleaved pair `re` and `im`.
#: Both collapse two lanes to one, which is why they cannot ride the table
#: above.
let COMPLEX_FORMS: list[tuple[str, str, int]] = [
    ("cabs", "nf_hypot(re, im)", KIND_FLOAT),
    ("cangle", "nf_atan2(im, re) * p0", KIND_FLOAT),
]

#: The hand-written half of the shader: the overloaded helpers the tables
#: call, the two loads the fast path uses, and the strided walks.
let HELPERS = """

/* Every helper below is a template so one body serves both the scalar tail
 * and the float4 body of the same generated kernel, and every branch inside
 * one is written with `select` rather than `?:`.
 *
 * That is not style. `?:` wants a scalar condition, `&&` and `||` are scalar
 * operators, and a comparison on a float4 yields a bool4 -- so the obvious
 * spelling of any of these compiles in the tail and fails in the body. The
 * price is that both arms are evaluated; none of them can trap, and a NaN or
 * an infinity produced by the arm that loses is discarded by the select. */
inline float nf_inf() { return as_type<float>(0x7f800000u); }
inline float nf_nan() { return as_type<float>(0x7fc00000u); }

/* Every NaN and infinity test below reads the bits. None of them compares a
 * value with itself and none of them compares against an infinity.
 *
 * The runtime compiles these with the default `MTLCompileOptions`, which
 * leaves fast math on, and fast math folds `v != v` to a constant false:
 * measured on an M2, `isnan` over an array holding a NaN answered all zeros,
 * `maximum` dropped the NaN it is supposed to propagate, and `nan_to_num`
 * replaced nothing. `reduce.jai` hit the same wall and took MSL's `isnan`,
 * which survives; this file takes the bit test instead because it also needs
 * `isinf` and `isfinite`, and the same flag carries `ninf` -- one family that
 * cannot be folded beats three spellings with three answers. The cost is an
 * AND and a compare on the integer pipe.
 *
 * Overloads rather than one template: `as_type` needs the unsigned type that
 * matches the lane count and MSL has no trait to name it. The templates
 * below call these and instantiate against whichever one fits. */
inline bool nf_nan_mask(float v) { return (as_type<uint>(v) & 0x7fffffffu) > 0x7f800000u; }
inline bool4 nf_nan_mask(float4 v) { return (as_type<uint4>(v) & 0x7fffffffu) > 0x7f800000u; }
inline bool nf_inf_mask(float v) { return (as_type<uint>(v) & 0x7fffffffu) == 0x7f800000u; }
inline bool4 nf_inf_mask(float4 v) { return (as_type<uint4>(v) & 0x7fffffffu) == 0x7f800000u; }
inline bool nf_finite_mask(float v) { return (as_type<uint>(v) & 0x7fffffffu) < 0x7f800000u; }
inline bool4 nf_finite_mask(float4 v) { return (as_type<uint4>(v) & 0x7fffffffu) < 0x7f800000u; }

/* Truth as numpy reads it: anything but zero, NaN included. */
template <typename T> inline T nf_bool(T v) { return select(T(1.0f), T(0.0f), v == T(0.0f)); }
template <typename T> inline T nf_not(T v) { return select(T(0.0f), T(1.0f), v == T(0.0f)); }

/* The logical trio through arithmetic on flags rather than through bool
 * vectors, whose operators MSL does not define. */
template <typename T> inline T nf_and(T a, T b) { return nf_bool(a) * nf_bool(b); }
template <typename T> inline T nf_or(T a, T b) { return nf_bool(nf_bool(a) + nf_bool(b)); }
template <typename T> inline T nf_xor(T a, T b) { return nf_bool(nf_bool(a) - nf_bool(b)); }

template <typename T> inline T nf_isnan(T v) { return select(T(0.0f), T(1.0f), nf_nan_mask(v)); }
template <typename T> inline T nf_isinf(T v) { return select(T(0.0f), T(1.0f), nf_inf_mask(v)); }
template <typename T> inline T nf_isfinite(T v) {
    return select(T(0.0f), T(1.0f), nf_finite_mask(v));
}

template <typename T> inline T nf_gt(T a, T b) { return select(T(0.0f), T(1.0f), a > b); }
template <typename T> inline T nf_ge(T a, T b) { return select(T(0.0f), T(1.0f), a >= b); }
template <typename T> inline T nf_lt(T a, T b) { return select(T(0.0f), T(1.0f), a < b); }
template <typename T> inline T nf_le(T a, T b) { return select(T(0.0f), T(1.0f), a <= b); }
template <typename T> inline T nf_eq(T a, T b) { return select(T(0.0f), T(1.0f), a == b); }
template <typename T> inline T nf_ne(T a, T b) { return select(T(0.0f), T(1.0f), a != b); }

template <typename T> inline T nf_where(T cond, T x, T y) { return select(y, x, cond != T(0.0f)); }

/* Metal's `sign` answers 0 for a NaN and numpy's answers NaN. */
template <typename T> inline T nf_sign(T v) { return select(sign(v), v, nf_nan_mask(v)); }

/* numpy's `maximum` propagates a NaN and its `fmax` drops it. Neither is
 * Metal's `fmax` here: under fast math that one is free to compile to a bare
 * comparison, which answers the second operand for `fmax(1.0f, NAN)` -- so
 * both readings are built from the ordered compare and an explicit NaN mask
 * rather than from a library call whose NaN behaviour is not guaranteed. */
template <typename T> inline T nf_maximum(T a, T b) {
    T ordered = select(b, a, a > b);
    return select(select(ordered, a, nf_nan_mask(a)), b, nf_nan_mask(b));
}
template <typename T> inline T nf_minimum(T a, T b) {
    T ordered = select(b, a, a < b);
    return select(select(ordered, a, nf_nan_mask(a)), b, nf_nan_mask(b));
}

/* The stepping pair: a NaN on one side leaves the other, and two NaNs leave
 * a NaN. */
template <typename T> inline T nf_fmax(T a, T b) {
    T ordered = select(b, a, a > b);
    return select(select(ordered, b, nf_nan_mask(a)), a, nf_nan_mask(b));
}
template <typename T> inline T nf_fmin(T a, T b) {
    T ordered = select(b, a, a < b);
    return select(select(ordered, b, nf_nan_mask(a)), a, nf_nan_mask(b));
}

/* `clamp` would be one instruction, but it inherits `fmax`'s NaN answer and
 * would clip a NaN to the lower bound. Bounds crossed over still yield the
 * upper one, as numpy's clip does. */
template <typename T> inline T nf_clip(T v, T low, T high) {
    return nf_minimum(nf_maximum(v, low), high);
}

/* A real cube root, which `pow` alone is not: it answers NaN for every
 * negative base. */
template <typename T> inline T nf_cbrt(T v) {
    return copysign(pow(fabs(v), T(1.0f / 3.0f)), v);
}

/* MSL has neither log1p nor expm1, and the naive `log(1 + x)` loses every
 * significant digit near zero: at x = 1e-7 the float32 sum rounds back to 1
 * and the answer comes back 0.
 *
 * Kahan's correction -- scaling log(1+x) by x / ((1+x) - 1), the exact factor
 * the rounding of the sum introduced -- is the usual repair, and it does not
 * work on this device. It needs a `log` that is faithful to the last bit just
 * above 1, and Metal's is not: measured on an M2, the corrected form answers
 * 1.65e-7 for log1p(1e-7) and 0 for log1p(3e-8). A series is what survives.
 *
 * Seven terms carry float32 through |x| < 1/8 -- the first dropped term is
 * x^7/8, which is 6e-8 of the answer at the threshold and falls off as the
 * seventh power below it. Past the threshold nothing cancels and the direct
 * form is already right. */
template <typename T> inline T nf_log1p(T x) {
    T series = x * (T(1.0f) + x * (T(-1.0f / 2.0f) + x * (T(1.0f / 3.0f)
             + x * (T(-1.0f / 4.0f) + x * (T(1.0f / 5.0f) + x * (T(-1.0f / 6.0f)
             + x * T(1.0f / 7.0f)))))));
    return select(log(T(1.0f) + x), series, fabs(x) < T(0.125f));
}

/* The same split for exp(x) - 1, where the cancellation is in the subtraction
 * rather than the sum. The exponential series converges far faster than the
 * logarithmic one, so six terms are already past float32 at the same
 * threshold. */
template <typename T> inline T nf_expm1(T x) {
    T series = x * (T(1.0f) + x * (T(1.0f / 2.0f) + x * (T(1.0f / 6.0f)
             + x * (T(1.0f / 24.0f) + x * (T(1.0f / 120.0f) + x * T(1.0f / 720.0f))))));
    return select(exp(x) - T(1.0f), series, fabs(x) < T(0.125f));
}

/* sqrt(a^2 + b^2) without the overflow: 1e30 squared is an infinity in
 * float32 and the true answer is not. Scaling by the larger magnitude keeps
 * the ratio inside [0, 1].
 *
 * An infinity on either side answers infinity whatever the other side holds,
 * including a NaN -- IEEE-754 says so and numpy 2.5.2 agrees, and it does
 * not fall out of the scaling: `nf_maximum` propagates the NaN it is
 * supposed to, the ratio then reads inf/NaN, and hypot(inf, NaN) came back
 * NaN. The test is the cheap one and it goes last, so it also covers
 * hypot(inf, inf), whose ratio is inf/inf. A NaN met by a finite number
 * stays a NaN, which is the scaling's own answer. */
template <typename T> inline T nf_hypot(T a, T b) {
    T big = nf_maximum(fabs(a), fabs(b));
    T small = nf_fmin(fabs(a), fabs(b));
    T ratio = small / big;
    T scaled = select(big * sqrt(T(1.0f) + ratio * ratio), T(0.0f), big == T(0.0f));
    T either_infinite = nf_isinf(a) + nf_isinf(b);
    return select(scaled, T(nf_inf()), either_infinite != T(0.0f));
}

/* log(exp(a) + exp(b)) for arguments that would overflow both exponentials.
 * The shifted form only holds where the larger one is finite; two negative
 * infinities would otherwise meet as inf - inf. */
template <typename T> inline T nf_logaddexp(T a, T b) {
    T big = nf_maximum(a, b);
    return select(big, big + nf_log1p(exp(-fabs(a - b))), nf_finite_mask(big));
}

/* A negative base raised to a whole power is real, and `pow` still answers
 * NaN for it, so the sign is carried by hand. A negative base raised to a
 * fractional power genuinely is not real, and that NaN is kept. */
template <typename T> inline T nf_power(T base, T exponent) {
    T rounded = rint(exponent);
    T magnitude = pow(fabs(base), exponent);
    T signed_magnitude = select(magnitude, -magnitude,
                                fmod(fabs(rounded), T(2.0f)) == T(1.0f));
    T reflected = select(T(nf_nan()), signed_magnitude, rounded == exponent);
    return select(reflected, pow(base, exponent), base >= T(0.0f));
}

/* numpy's `remainder` takes the sign of the divisor and C's `fmod` takes the
 * sign of the dividend, so the two disagree on exactly the mixed-sign cases.
 * The signs are compared through `copysign` rather than through `r * b < 0`:
 * that product underflows to zero for small operands and would then answer
 * with the wrong sign. */
template <typename T> inline T nf_remainder(T a, T b) {
    T r = fmod(a, b);
    T shift = select(T(0.0f), b, copysign(T(1.0f), r) != copysign(T(1.0f), b));
    return select(r + shift, r, r == T(0.0f));
}

/* Three ways to be close, or-ed as flags: inside the tolerance, exactly equal
 * -- which is what lets two infinities match, since |inf - inf| is NaN and
 * fails every tolerance -- and both NaN when the caller asked for it.
 *
 * The tolerance only speaks for a pair that is finite on both sides, which is
 * how numpy has it: it evaluates `|a - b| <= atol + rtol * |b|` on the finite
 * elements alone and falls back to exact equality on the rest. Ungated, the
 * bound is infinite wherever `b` is, so `isclose(1.0, inf)` and
 * `isclose(inf, -inf)` both come out true -- numpy 2.5.2 answers False for
 * both, and did so for every case checked here. */
template <typename T> inline T nf_isclose(T a, T b, float rtol, float atol, float equal_nan) {
    T finite = nf_isfinite(a) * nf_isfinite(b);
    T within = select(T(0.0f), T(1.0f), fabs(a - b) <= atol + rtol * fabs(b)) * finite;
    T exact = select(T(0.0f), T(1.0f), a == b);
    return nf_bool(within + exact + nf_isnan(a) * nf_isnan(b) * T(equal_nan));
}

/* The two infinities are told apart by their sign bit rather than by
 * `v == nf_inf()`, for the reason the mask helpers exist: a comparison
 * against an infinity is another thing fast math is entitled to fold. */
template <typename T> inline T nf_nan_to_num(T v, float filler, float positive, float negative) {
    T infinite = select(T(positive), T(negative), signbit(v));
    T out = select(v, infinite, nf_inf_mask(v));
    return select(out, T(filler), nf_nan_mask(v));
}

/* Metal's `atan2` reaches for `atan(y / x)`, and a zero in either argument
 * is a zero the quotient cannot carry back out. Measured on an M2 against
 * numpy 2.5.2, which agrees with IEEE-754 on all of them:
 *
 * - atan2(+-0, +-0) came back NaN, where the answer is +-0 for a positive x
 *   and +-pi for a negative one. 0/0 is NaN and there is nothing to recover.
 * - atan2(1, -0) came back -pi/2 and atan2(-1, -0) came back +pi/2, the
 *   answers reflected. 1 / -0 is -inf, so `atan` reads the sign of the
 *   divisor as if it belonged to y.
 * - atan2(-0, -8) came back +pi where the answer is -pi. -0 / -8 is +0, so
 *   the sign of y is gone before the quadrant is chosen -- and this one is
 *   nowhere near the origin, which is why the whole zero *row* is patched
 *   rather than the two axes crossing at it.
 * - atan2(+-inf, +-inf) came back NaN, where the answer is the diagonal of
 *   the quadrant: +-pi/4 or +-3pi/4. inf/inf is the same indeterminate form
 *   0/0 is, reached from the other end.
 *
 * Everywhere else the quotient loses nothing and the answer is Metal's own.
 * A NaN in either argument is excluded, because `x == 0` can hold while y is
 * NaN and the answer there is NaN, which `atan2` already gives.
 *
 * One expression covers the whole y == 0 row because `signbit` reads -0 and
 * every negative x alike, and `copysign` puts the sign of y back on top. The
 * three patched sets do not overlap: an argument is zero, or infinite, or
 * neither. */
template <typename T> inline T nf_atan2(T y, T x) {
    T flat = copysign(select(T(0.0f), T(M_PI_F), signbit(x)), y);
    T quarter = copysign(T(M_PI_2_F), y);
    T diagonal = copysign(select(T(M_PI_4_F), T(3.0f * M_PI_4_F), signbit(x)), y);
    T off_axis = atan2(y, x);

    T at_a_zero = nf_eq(x, T(0.0f)) + nf_eq(y, T(0.0f));
    T both_infinite = nf_isinf(x) * nf_isinf(y);
    T known = select(select(quarter, flat, y == T(0.0f)), diagonal,
                     both_infinite != T(0.0f));
    T reachable = T(1.0f) - nf_isnan(x);
    T patched = (at_a_zero + both_infinite) * reachable * (T(1.0f) - nf_isnan(y));
    return select(off_axis, known, patched != T(0.0f));
}

/* The phase of a real number is 0 or pi. Written as a helper because
 * `atan2(0.0f, value)` will not compile against a float4. */
template <typename T> inline T nf_angle(T v, float scale) {
    return nf_atan2(T(0.0f), v) * scale;
}

/* What an integer or boolean destination dtype means on float32 storage, in
 * the shape the generated kernels need. Mirrors the prelude's `nd_convert`,
 * which is scalar only. */
template <typename T> inline T nf_convert(T value, uint mode, float low, float high) {
    if (mode == 1u) return clamp(rint(value), T(low), T(high));
    if (mode == 2u) return nf_bool(value);
    return value;
}

/* The bitwise family cannot be a template: `int(v)` and `int4(v)` are the
 * same idea under two spellings and MSL has no trait to pick between them.
 * `rint` first, because an integer dtype is a promise about rounding and a
 * value that drifted to 2.9999998 must not truncate to 2. */
inline float nf_bit_and(float a, float b) { return float(int(rint(a)) & int(rint(b))); }
inline float4 nf_bit_and(float4 a, float4 b) { return float4(int4(rint(a)) & int4(rint(b))); }
inline float nf_bit_or(float a, float b) { return float(int(rint(a)) | int(rint(b))); }
inline float4 nf_bit_or(float4 a, float4 b) { return float4(int4(rint(a)) | int4(rint(b))); }
inline float nf_bit_xor(float a, float b) { return float(int(rint(a)) ^ int(rint(b))); }
inline float4 nf_bit_xor(float4 a, float4 b) { return float4(int4(rint(a)) ^ int4(rint(b))); }
inline float nf_shl(float a, float b) { return float(int(rint(a)) << int(rint(b))); }
inline float4 nf_shl(float4 a, float4 b) { return float4(int4(rint(a)) << int4(rint(b))); }
inline float nf_shr(float a, float b) { return float(int(rint(a)) >> int(rint(b))); }
inline float4 nf_shr(float4 a, float4 b) { return float4(int4(rint(a)) >> int4(rint(b))); }
inline float nf_invert(float a) { return float(~int(rint(a))); }
inline float4 nf_invert(float4 a) { return float4(~int4(rint(a))); }

/* An operand of the fast path is either as long as the output or one element
 * standing in for all of them -- a scalar the host wrapped, or a size-one
 * array broadcasting. `step` is 1 or 0 and is uniform across the dispatch, so
 * the branch is free and the pointer cast only happens on the arm taken. */
/* `step` names one of three layouts rather than a stride: 0 is one value
 * broadcast over everything, 1 is dense, and anything larger is an operand
 * that REPEATS every `step` elements -- a row added to every row of a matrix,
 * which is what a bias and a normalisation both are. That case used to miss
 * the dense path entirely and go through the strided walk one element at a
 * time; [4096, 4096] + [4096] cost 63.5 ms against 16.7 for the same bytes
 * with both operands full size.
 *
 * The wrap is `at - (at / step) * step`, one divide per four elements rather
 * than the walk's per-element index arithmetic. The host only names a wrap
 * whose length divides by four, so a float4 load can never straddle the seam
 * and stays sixteen-byte aligned. */
inline float4 nf_load4(device const float* p, uint at, uint step) {
    if (step == 0u) return float4(p[0]);
    if (step == 1u) return *((device const float4*)(p + at));
    return *((device const float4*)(p + at - (at / step) * step));
}
inline float nf_load1(device const float* p, uint at, uint step) {
    if (step == 0u) return p[0];
    if (step == 1u) return p[at];
    return p[at - (at / step) * step];
}

/* Every operand's float slot for element `linear`, in one walk.
 *
 * The prelude's `nd_offset` costs a division and a modulo per axis and is
 * called once per operand; the device has neither in hardware and the pair
 * costs twice what one does. These take the quotient once, recover the index
 * by subtraction, and accumulate into all the operands at the same time, so a
 * ternary dispatch over a rank-4 array issues four divisions where the
 * obvious spelling issues thirty-two. */
inline void nf_walk2(device const float* meta, uint linear,
                     thread int& first, thread int& second) {
    int rank = int(meta[0]);
    int base = 1 + rank;
    int step = rank + 1;
    first = int(meta[base]);
    second = int(meta[base + step]);
    uint rest = linear;
    for (int axis = rank - 1; axis >= 0; axis--) {
        uint extent = uint(meta[1 + axis]);
        uint next = rest / extent;
        int index = int(rest - next * extent);
        first += index * int(meta[base + 1 + axis]);
        second += index * int(meta[base + step + 1 + axis]);
        rest = next;
    }
}

inline void nf_walk3(device const float* meta, uint linear,
                     thread int& first, thread int& second, thread int& third) {
    int rank = int(meta[0]);
    int base = 1 + rank;
    int step = rank + 1;
    first = int(meta[base]);
    second = int(meta[base + step]);
    third = int(meta[base + 2 * step]);
    uint rest = linear;
    for (int axis = rank - 1; axis >= 0; axis--) {
        uint extent = uint(meta[1 + axis]);
        uint next = rest / extent;
        int index = int(rest - next * extent);
        first += index * int(meta[base + 1 + axis]);
        second += index * int(meta[base + step + 1 + axis]);
        third += index * int(meta[base + 2 * step + 1 + axis]);
        rest = next;
    }
}

inline void nf_walk4(device const float* meta, uint linear, thread int& first,
                     thread int& second, thread int& third, thread int& fourth) {
    int rank = int(meta[0]);
    int base = 1 + rank;
    int step = rank + 1;
    first = int(meta[base]);
    second = int(meta[base + step]);
    third = int(meta[base + 2 * step]);
    fourth = int(meta[base + 3 * step]);
    uint rest = linear;
    for (int axis = rank - 1; axis >= 0; axis--) {
        uint extent = uint(meta[1 + axis]);
        uint next = rest / extent;
        int index = int(rest - next * extent);
        first += index * int(meta[base + 1 + axis]);
        second += index * int(meta[base + step + 1 + axis]);
        third += index * int(meta[base + 2 * step + 1 + axis]);
        fourth += index * int(meta[base + 3 * step + 1 + axis]);
        rest = next;
    }
}
"""

let _UNARY_TEMPLATE = """
kernel void nfu_@NAME@(device float* dst [[buffer(0)]],
                       device const float* src [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       constant uint& mode [[buffer(3)]],
                       constant uint& low [[buffer(4)]],
                       constant uint& high [[buffer(5)]],
                       constant uint& b0 [[buffer(6)]],
                       constant uint& b1 [[buffer(7)]],
                       constant uint& b2 [[buffer(8)]],
                       uint id [[thread_position_in_grid]]) {
    float lo = as_type<float>(low);
    float hi = as_type<float>(high);
@PARAMS@    uint at = id * 4u;
    if (at + 4u <= count) {
        float4 value = *((device const float4*)(src + at));
        *((device float4*)(dst + at)) = nf_convert(@EXPR@, mode, lo, hi);
        return;
    }
    for (uint k = at; k < count; k++) {
        float value = src[k];
        dst[k] = nf_convert(@EXPR@, mode, lo, hi);
    }
}

kernel void nfu_@NAME@_strided(device float* dst [[buffer(0)]],
                               device const float* src [[buffer(1)]],
                               device const float* meta [[buffer(2)]],
                               constant uint& count [[buffer(3)]],
                               constant uint& mode [[buffer(4)]],
                               constant uint& low [[buffer(5)]],
                               constant uint& high [[buffer(6)]],
                               constant uint& b0 [[buffer(7)]],
                               constant uint& b1 [[buffer(8)]],
                               constant uint& b2 [[buffer(9)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float lo = as_type<float>(low);
    float hi = as_type<float>(high);
@PARAMS@    int to = 0;
    int from = 0;
    nf_walk2(meta, id, to, from);
    float value = src[from];
    dst[to] = nf_convert(@EXPR@, mode, lo, hi);
}
"""

let _BINARY_TEMPLATE = """
kernel void nfb_@NAME@(device float* dst [[buffer(0)]],
                       device const float* lhs [[buffer(1)]],
                       device const float* rhs [[buffer(2)]],
                       constant uint& count [[buffer(3)]],
                       constant uint& a_step [[buffer(4)]],
                       constant uint& b_step [[buffer(5)]],
                       constant uint& mode [[buffer(6)]],
                       constant uint& low [[buffer(7)]],
                       constant uint& high [[buffer(8)]],
                       constant uint& b0 [[buffer(9)]],
                       constant uint& b1 [[buffer(10)]],
                       constant uint& b2 [[buffer(11)]],
                       uint id [[thread_position_in_grid]]) {
    float lo = as_type<float>(low);
    float hi = as_type<float>(high);
@PARAMS@    uint at = id * 4u;
    if (at + 4u <= count) {
        float4 a = nf_load4(lhs, at, a_step);
        float4 b = nf_load4(rhs, at, b_step);
        *((device float4*)(dst + at)) = nf_convert(@EXPR@, mode, lo, hi);
        return;
    }
    for (uint k = at; k < count; k++) {
        float a = nf_load1(lhs, k, a_step);
        float b = nf_load1(rhs, k, b_step);
        dst[k] = nf_convert(@EXPR@, mode, lo, hi);
    }
}

kernel void nfb_@NAME@_strided(device float* dst [[buffer(0)]],
                               device const float* lhs [[buffer(1)]],
                               device const float* rhs [[buffer(2)]],
                               device const float* meta [[buffer(3)]],
                               constant uint& count [[buffer(4)]],
                               constant uint& mode [[buffer(5)]],
                               constant uint& low [[buffer(6)]],
                               constant uint& high [[buffer(7)]],
                               constant uint& b0 [[buffer(8)]],
                               constant uint& b1 [[buffer(9)]],
                               constant uint& b2 [[buffer(10)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float lo = as_type<float>(low);
    float hi = as_type<float>(high);
@PARAMS@    int to = 0;
    int left = 0;
    int right = 0;
    nf_walk3(meta, id, to, left, right);
    float a = lhs[left];
    float b = rhs[right];
    dst[to] = nf_convert(@EXPR@, mode, lo, hi);
}
"""

let _TERNARY_TEMPLATE = """
kernel void nft_@NAME@(device float* dst [[buffer(0)]],
                       device const float* first [[buffer(1)]],
                       device const float* second [[buffer(2)]],
                       device const float* third [[buffer(3)]],
                       constant uint& count [[buffer(4)]],
                       constant uint& a_step [[buffer(5)]],
                       constant uint& b_step [[buffer(6)]],
                       constant uint& c_step [[buffer(7)]],
                       constant uint& mode [[buffer(8)]],
                       constant uint& low [[buffer(9)]],
                       constant uint& high [[buffer(10)]],
                       uint id [[thread_position_in_grid]]) {
    float lo = as_type<float>(low);
    float hi = as_type<float>(high);
    uint at = id * 4u;
    if (at + 4u <= count) {
        float4 a = nf_load4(first, at, a_step);
        float4 b = nf_load4(second, at, b_step);
        float4 c = nf_load4(third, at, c_step);
        *((device float4*)(dst + at)) = nf_convert(@EXPR@, mode, lo, hi);
        return;
    }
    for (uint k = at; k < count; k++) {
        float a = nf_load1(first, k, a_step);
        float b = nf_load1(second, k, b_step);
        float c = nf_load1(third, k, c_step);
        dst[k] = nf_convert(@EXPR@, mode, lo, hi);
    }
}

kernel void nft_@NAME@_strided(device float* dst [[buffer(0)]],
                               device const float* first [[buffer(1)]],
                               device const float* second [[buffer(2)]],
                               device const float* third [[buffer(3)]],
                               device const float* meta [[buffer(4)]],
                               constant uint& count [[buffer(5)]],
                               constant uint& mode [[buffer(6)]],
                               constant uint& low [[buffer(7)]],
                               constant uint& high [[buffer(8)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float lo = as_type<float>(low);
    float hi = as_type<float>(high);
    int to = 0;
    int one = 0;
    int two = 0;
    int three = 0;
    nf_walk4(meta, id, to, one, two, three);
    float a = first[one];
    float b = second[two];
    float c = third[three];
    dst[to] = nf_convert(@EXPR@, mode, lo, hi);
}
"""

let _COMPLEX_TEMPLATE = """
kernel void nfc_@NAME@(device float* dst [[buffer(0)]],
                       device const float* src [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       constant uint& b0 [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
@PARAMS@    float re = src[id * 2u];
    float im = src[id * 2u + 1u];
    dst[id] = @EXPR@;
}

kernel void nfc_@NAME@_strided(device float* dst [[buffer(0)]],
                               device const float* src [[buffer(1)]],
                               device const float* meta [[buffer(2)]],
                               constant uint& count [[buffer(3)]],
                               constant uint& b0 [[buffer(4)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
@PARAMS@    int to = 0;
    int from = 0;
    nf_walk2(meta, id, to, from);
    float re = src[from];
    float im = src[from + 1];
    dst[to] = @EXPR@;
}
"""

#: The scalar declarations a row's expression actually reaches for.
#:
#: Emitted only where the text mentions them: an unread `as_type<float>` of a
#: constant argument is dead, but it warns, and a shader library that warns
#: forty times is one nobody reads the real diagnostics out of.
fn _param_lines(expression: str, most: int) -> str {
    var lines = ""
    for index in 0..most {
        let name = f"p{index}"
        if expression.contains(name) {
            lines += "    float " + name + " = as_type<float>(b" + f"{index}" + ");\n"
        }
    }
    return lines
}

fn _emit(template: str, name: str, expression: str, most: int) -> str {
    let with_params = template.replace("@PARAMS@", _param_lines(expression, most))
    let with_expression = with_params.replace("@EXPR@", expression)
    return with_expression.replace("@NAME@", name)
}

#: The generated half of the shader library: two entry points per table row.
fn ufunc_kernels() -> str {
    var parts: list[str] = []
    for (name, expression, _kind) in UNARY_FORMS {
        parts.push(_emit(_UNARY_TEMPLATE, name, expression, 3))
    }
    for (name, expression, _kind) in BINARY_FORMS {
        parts.push(_emit(_BINARY_TEMPLATE, name, expression, 3))
    }
    for (name, expression, _kind) in TERNARY_FORMS {
        parts.push(_emit(_TERNARY_TEMPLATE, name, expression, 0))
    }
    for (name, expression, _kind) in COMPLEX_FORMS {
        parts.push(_emit(_COMPLEX_TEMPLATE, name, expression, 1))
    }
    return "\n".join(parts)
}

fn _kinds(forms: list[tuple[str, str, int]], into: dict[str, int]) -> dict[str, int] {
    for (name, _expression, kind) in forms { into[name] = kind }
    return into
}

let SOURCE = PRELUDE + HELPERS + ufunc_kernels()

#: Result rule by entry name, so a wrapper names its ufunc once instead of
#: repeating a rule the table already carries. The ternaries are absent:
#: `where` must not promote against its condition, so both of them are
#: handed a dtype outright.
let KINDS = _kinds(BINARY_FORMS, _kinds(UNARY_FORMS, {}))

#: The dtype an entry produces from its operands' promotion.
fn _kind_dtype(left: int, right: int, entry: str) -> int {
    let kind = KINDS.get(entry, KIND_KEEP)
    if kind == KIND_BOOL { return BOOL }
    let out = promote(left, right)
    if kind == KIND_INT {
        if not dtype_is_integer(out) {
            throw TypeError(f"{entry} needs an integer dtype, not {dtype_name(out)}")
        }
        return out
    }
    if kind == KIND_FLOAT and dtype_is_integer(out) { return FLOAT32 }
    return out
}

fn _conversions() -> list[tuple[int, int, int]] {
    var out: list[tuple[int, int, int]] = []
    for dtype in 0..=COMPLEX64 {
        var mode = CAST_NONE
        if dtype == BOOL {
            mode = CAST_BOOL
        } elif dtype_is_integer(dtype) {
            mode = CAST_ROUND
        }
        out.push((mode, f32_bits(dtype_min(dtype)), f32_bits(dtype_max(dtype))))
    }
    return out
}

#: What each destination dtype asks of a generated kernel: the conversion
#: mode, and the bits of the two bounds it saturates against.
#:
#: Built once and indexed, rather than derived per dispatch. `f32_bits` walks
#: a `frexp` and a mantissa rounding that can carry into the exponent, and
#: running it twice in front of every elementwise launch to rediscover bounds
#: that six dtypes exhaust is host work standing between the caller and the
#: GPU. The three travel together because they are one rule: a kernel handed
#: `CAST_ROUND` with another dtype's bounds would round correctly into the
#: wrong range.
let CONVERSIONS = _conversions()

fn _as_float(value: any) -> float {
    if value is bool { return if value { 1.0 } else { 0.0 } }
    if value is int { return float(value) }
    if value is float { return value }
    throw TypeError(f"expected a number or an NDArray, got {type_of(value)}")
}

#: The dtype a bare Jaithon number contributes, from its own type and not its
#: value — the same rule `array` applies to the operators.
fn _scalar_dtype(value: any) -> int {
    if value is bool { return BOOL }
    if value is int { return INT32 }
    return FLOAT32
}

#: A scalar as a 0-d array.
#:
#: One host upload and no dispatch: `fill` would cost a kernel launch to write
#: a single float. The fast path reads it as a broadcast operand, so wrapping
#: a scalar this way does not push the call onto the strided path.
fn _scalar_array(value: any) -> NDArray {
    let out = NDArray([], _scalar_dtype(value))
    out.upload([_as_float(value)])
    return out
}

fn _operand(value: any) -> (NDArray, bool) {
    if value is NDArray { return (value, false) }
    return (_scalar_array(value), true)
}

#: How the fast path would read this operand: 1 for one element per thread, 0
#: for one element for all of them, -1 when it cannot be read that way at all.
#: Whether `sub` is exactly the trailing axes of `shape`.
#:
#: Exactly: no size-one axis standing in for a longer one. That is what makes
#: element `i` of the broadcast result element `i % size` of the operand, which
#: is the whole basis of the wrap below.
fn _is_trailing(sub: list[int], shape: list[int]) -> bool {
    if sub.len() == 0 or sub.len() >= shape.len() { return false }
    let lead = shape.len() - sub.len()
    for axis in 0..sub.len() {
        if sub[axis] != shape[lead + axis] { return false }
    }
    return true
}

#: 0 for one value over everything, 1 for dense, `n` for an operand that
#: repeats every `n`. See `nf_load4` for what the kernel does with it.
fn _dense_step(a: NDArray, shape: list[int]) -> int {
    if a.offset != 0 { return -1 }
    if a.size() == 1 { return 0 }
    if not a.is_contiguous() { return -1 }
    if a.shape == shape { return 1 }
    if a.size() % 4 == 0 and _is_trailing(a.shape, shape) { return a.size() }
    return -1
}

fn _unary(a: NDArray, entry: str, p0: float = 0.0, p1: float = 0.0, p2: float = 0.0) -> NDArray {
    require_real(a, entry)
    let dtype = _kind_dtype(a.dtype, a.dtype, entry)
    let out = NDArray(a.shape, dtype)
    let count = out.size()
    if count == 0 { return out }
    let (mode, low, high) = CONVERSIONS[dtype]
    if a.offset == 0 and a.is_contiguous() {
        kernel_of(SOURCE, f"nfu_{entry}").dispatch_async(
            [out.storage, a.storage],
            (count + 3) // 4,
            [count, mode, low, high, f32_bits(p0), f32_bits(p1), f32_bits(p2)]
        )
        return out
    }
    let meta = layout_buffer([out, a])
    defer { meta.free() }
    kernel_of(SOURCE, f"nfu_{entry}_strided").dispatch_async(
        [out.storage, a.storage, meta],
        count,
        [count, mode, low, high, f32_bits(p0), f32_bits(p1), f32_bits(p2)]
    )
    return out
}

fn _binary(
    left: any,
    right: any,
    entry: str,
    p0: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0
) -> NDArray {
    if not left is NDArray and not right is NDArray {
        throw TypeError(f"{entry} needs at least one array operand")
    }
    let (a, owns_a) = _operand(left)
    let (b, owns_b) = _operand(right)
    defer { if owns_a { a.free() } }
    defer { if owns_b { b.free() } }
    require_real(a, entry)
    require_real(b, entry)
    let dtype = _kind_dtype(a.dtype, b.dtype, entry)
    let shape = broadcast_shapes(a.shape, b.shape)
    let out = NDArray(shape, dtype)
    let count = out.size()
    if count == 0 { return out }
    let (mode, low, high) = CONVERSIONS[dtype]
    let a_step = _dense_step(a, shape)
    let b_step = _dense_step(b, shape)
    if a_step >= 0 and b_step >= 0 {
        kernel_of(SOURCE, f"nfb_{entry}").dispatch_async(
            [out.storage, a.storage, b.storage],
            (count + 3) // 4,
            [count, a_step, b_step, mode, low, high, f32_bits(p0), f32_bits(p1), f32_bits(p2)]
        )
        return out
    }
    let meta = layout_buffer([out, a.broadcast_to(shape), b.broadcast_to(shape)])
    defer { meta.free() }
    kernel_of(SOURCE, f"nfb_{entry}_strided").dispatch_async(
        [out.storage, a.storage, b.storage, meta],
        count,
        [count, mode, low, high, f32_bits(p0), f32_bits(p1), f32_bits(p2)]
    )
    return out
}

fn _ternary(first: NDArray, second: NDArray, third: NDArray, entry: str, dtype: int) -> NDArray {
    require_real(first, entry)
    require_real(second, entry)
    require_real(third, entry)
    let shape = broadcast_shapes(broadcast_shapes(first.shape, second.shape), third.shape)
    let out = NDArray(shape, dtype)
    let count = out.size()
    if count == 0 { return out }
    let (mode, low, high) = CONVERSIONS[dtype]
    let one = _dense_step(first, shape)
    let two = _dense_step(second, shape)
    let three = _dense_step(third, shape)
    if one >= 0 and two >= 0 and three >= 0 {
        kernel_of(SOURCE, f"nft_{entry}").dispatch_async(
            [out.storage, first.storage, second.storage, third.storage],
            (count + 3) // 4,
            [count, one, two, three, mode, low, high]
        )
        return out
    }
    let meta = layout_buffer(
        [
            out,
            first.broadcast_to(shape),
            second.broadcast_to(shape),
            third.broadcast_to(shape),
        ]
    )
    defer { meta.free() }
    kernel_of(SOURCE, f"nft_{entry}_strided").dispatch_async(
        [out.storage, first.storage, second.storage, third.storage, meta],
        count,
        [count, mode, low, high]
    )
    return out
}

fn _complex_unary(a: NDArray, entry: str, p0: float) -> NDArray {
    let out = NDArray(a.shape, FLOAT32)
    let count = out.size()
    if count == 0 { return out }
    if a.offset == 0 and a.is_contiguous() {
        kernel_of(SOURCE, f"nfc_{entry}").dispatch_async(
            [out.storage, a.storage],
            count,
            [count, f32_bits(p0)]
        )
        return out
    }
    let meta = layout_buffer([out, a])
    defer { meta.free() }
    kernel_of(SOURCE, f"nfc_{entry}_strided").dispatch_async(
        [out.storage, a.storage, meta],
        count,
        [count, f32_bits(p0)]
    )
    return out
}

#: A scalar operand on either side, for the four ops that are `NDArray`
#: methods. The VM has no reflected dunder, so `2.0 - a` cannot reach one.
fn _arith(left: any, right: any, op: int) -> NDArray {
    if left is NDArray { return binary_arith(left, right, op) }
    let scalar = _scalar_array(left)
    defer { scalar.free() }
    return binary_arith(scalar, right, op)
}

#: `a + b`, either side a scalar. The operator on `NDArray` under numpy's
#: name; complex lanes included, which is why it is not in the table.
pub fn add(a: any, b: any) -> NDArray { return _arith(a, b, OP_ADD) }

#: `a - b`, either side a scalar.
pub fn subtract(a: any, b: any) -> NDArray { return _arith(a, b, OP_SUB) }

#: `a * b`, either side a scalar.
pub fn multiply(a: any, b: any) -> NDArray { return _arith(a, b, OP_MUL) }

#: `a / b`, either side a scalar. True division: the result leaves the integer
#: dtypes even when both operands are in them.
pub fn divide(a: any, b: any) -> NDArray { return _arith(a, b, OP_DIV) }

#: `divide` under numpy's other name.
pub fn true_divide(a: any, b: any) -> NDArray { return _arith(a, b, OP_DIV) }

#: `-a`.
pub fn negative(a: NDArray) -> NDArray { return _unary(a, "negative") }

#: `+a`, which is a copy and not a view — numpy's `positive` is a ufunc too.
pub fn positive(a: NDArray) -> NDArray { return a.copy() }

#: Magnitude. The one real-and-complex entry point apart from `angle`: a
#: complex array collapses two lanes into a `FLOAT32` modulus.
pub fn absolute(a: NDArray) -> NDArray {
    if dtype_is_complex(a.dtype) { return _complex_unary(a, "cabs", 1.0) }
    return _unary(a, "absolute")
}

#: -1, 0 or 1 by the sign of each element. NaN stays NaN, as numpy's does and
#: Metal's `sign` does not.
pub fn sign(a: NDArray) -> NDArray { return _unary(a, "sign") }

#: `1 / a`, always in a float dtype.
#:
#: numpy keeps the integer dtype here and so answers 0 for every element but
#: 1 and -1. That is a wart of its type rules rather than a reading of the
#: operation, and reproducing it would only surprise people.
pub fn reciprocal(a: NDArray) -> NDArray { return _unary(a, "reciprocal") }

#: Square root. Negative elements come back NaN.
pub fn sqrt(a: NDArray) -> NDArray { return _unary(a, "sqrt") }

#: Real cube root, defined for negative elements as numpy's is.
pub fn cbrt(a: NDArray) -> NDArray { return _unary(a, "cbrt") }

#: `a * a`, in the input's dtype.
pub fn square(a: NDArray) -> NDArray { return _unary(a, "square") }

#: `e ** a`.
pub fn exp(a: NDArray) -> NDArray { return _unary(a, "exp") }

#: `2 ** a`.
pub fn exp2(a: NDArray) -> NDArray { return _unary(a, "exp2") }

#: `e ** a - 1`, accurate for small `a` where `exp(a) - 1` cancels away every
#: significant digit.
pub fn expm1(a: NDArray) -> NDArray { return _unary(a, "expm1") }

#: Natural logarithm.
pub fn log(a: NDArray) -> NDArray { return _unary(a, "log") }

#: Base-2 logarithm.
pub fn log2(a: NDArray) -> NDArray { return _unary(a, "log2") }

#: Base-10 logarithm.
pub fn log10(a: NDArray) -> NDArray { return _unary(a, "log10") }

#: `log(1 + a)`, accurate for small `a`.
pub fn log1p(a: NDArray) -> NDArray { return _unary(a, "log1p") }

#: Sine, in radians.
pub fn sin(a: NDArray) -> NDArray { return _unary(a, "sin") }

#: Cosine, in radians.
pub fn cos(a: NDArray) -> NDArray { return _unary(a, "cos") }

#: Tangent, in radians.
pub fn tan(a: NDArray) -> NDArray { return _unary(a, "tan") }

#: Inverse sine, in radians. Outside [-1, 1] the answer is NaN.
pub fn arcsin(a: NDArray) -> NDArray { return _unary(a, "arcsin") }

#: Inverse cosine, in radians.
pub fn arccos(a: NDArray) -> NDArray { return _unary(a, "arccos") }

#: Inverse tangent, in radians. `arctan2` is the two-argument form that knows
#: which quadrant it is in.
pub fn arctan(a: NDArray) -> NDArray { return _unary(a, "arctan") }

#: Hyperbolic sine.
pub fn sinh(a: NDArray) -> NDArray { return _unary(a, "sinh") }

#: Hyperbolic cosine.
pub fn cosh(a: NDArray) -> NDArray { return _unary(a, "cosh") }

#: Hyperbolic tangent.
pub fn tanh(a: NDArray) -> NDArray { return _unary(a, "tanh") }

#: Inverse hyperbolic sine.
pub fn arcsinh(a: NDArray) -> NDArray { return _unary(a, "arcsinh") }

#: Inverse hyperbolic cosine. Below 1 the answer is NaN.
pub fn arccosh(a: NDArray) -> NDArray { return _unary(a, "arccosh") }

#: Inverse hyperbolic tangent. Outside (-1, 1) the answer is NaN.
pub fn arctanh(a: NDArray) -> NDArray { return _unary(a, "arctanh") }

#: Degrees to radians.
pub fn deg2rad(a: NDArray) -> NDArray { return _unary(a, "deg2rad") }

#: Radians to degrees.
pub fn rad2deg(a: NDArray) -> NDArray { return _unary(a, "rad2deg") }

#: `deg2rad` under numpy's other name.
pub fn radians(a: NDArray) -> NDArray { return _unary(a, "deg2rad") }

#: `rad2deg` under numpy's other name.
pub fn degrees(a: NDArray) -> NDArray { return _unary(a, "rad2deg") }

#: Largest whole number no greater than each element.
pub fn floor(a: NDArray) -> NDArray { return _unary(a, "floor") }

#: Smallest whole number no less than each element.
pub fn ceil(a: NDArray) -> NDArray { return _unary(a, "ceil") }

#: Each element with its fractional part dropped, towards zero.
pub fn trunc(a: NDArray) -> NDArray { return _unary(a, "trunc") }

#: Nearest whole number, halves to even.
pub fn rint(a: NDArray) -> NDArray { return _unary(a, "rint") }

#: Nearest value with `decimals` places kept, halves to even.
#:
#: Halves to even and not away from zero, which is what numpy's `round` does
#: and what surprises everyone: `round(0.5)` is 0. The scaled form is three
#: dispatches rather than one, and is only reached when `decimals` is not
#: zero.
pub fn round(a: NDArray, decimals: int = 0) -> NDArray {
    if decimals == 0 { return _unary(a, "rint") }
    let places = if decimals < 0 { -decimals } else { decimals }
    var scale = 1.0
    for _step in 0..places { scale *= 10.0 }
    let factor = if decimals < 0 { 1.0 / scale } else { scale }
    let lifted = binary_arith(a, factor, OP_MUL)
    defer { lifted.free() }
    let rounded = _unary(lifted, "rint")
    defer { rounded.free() }
    return binary_arith(rounded, factor, OP_DIV)
}

#: Where each element is NaN.
pub fn isnan(a: NDArray) -> NDArray { return _unary(a, "isnan") }

#: Where each element is an infinity, of either sign.
pub fn isinf(a: NDArray) -> NDArray { return _unary(a, "isinf") }

#: Where each element is neither NaN nor an infinity.
pub fn isfinite(a: NDArray) -> NDArray { return _unary(a, "isfinite") }

#: Where each element is zero. NaN is truthy, so `logical_not` of NaN is
#: false.
pub fn logical_not(a: NDArray) -> NDArray { return _unary(a, "logical_not") }

#: Bitwise complement, `~a`. Integer dtypes only.
pub fn bitwise_invert(a: NDArray) -> NDArray { return _unary(a, "bitwise_invert") }

#: `bitwise_invert` under numpy's other name.
pub fn invert(a: NDArray) -> NDArray { return _unary(a, "bitwise_invert") }

#: Largest finite float32. numpy's `nan_to_num` replaces an infinity with the
#: dtype's largest finite value, and this is that value for the storage
#: underneath every jainum array.
pub const FLOAT32_MAX: float = 3.4028234663852886e38

#: NaN and the infinities replaced by finite numbers.
#:
#: The defaults are numpy's: zero for NaN, and the dtype's finite extremes for
#: the infinities.
pub fn nan_to_num(
    a: NDArray,
    nan: float = 0.0,
    posinf: float = FLOAT32_MAX,
    neginf: float = -FLOAT32_MAX
) -> NDArray {
    return _unary(a, "nan_to_num", nan, posinf, neginf)
}

#: The phase of each element, in radians unless `deg` asks for degrees.
#:
#: The second complex-aware entry point. A real array is not an error: its
#: phase is 0 where it is positive and pi where it is negative, which is what
#: `atan2(0, x)` answers.
pub fn angle(a: NDArray, deg: bool = false) -> NDArray {
    let scale = if deg { 57.29577951308232088 } else { 1.0 }
    if dtype_is_complex(a.dtype) { return _complex_unary(a, "cangle", scale) }
    return _unary(a, "angle", scale)
}

#: `a ** b`, broadcasting.
#:
#: A negative base raised to a whole power is real and comes back real; raised
#: to a fractional one it is not, and comes back NaN. numpy refuses an integer
#: array raised to a negative integer power outright; here the answer rounds
#: through the destination dtype like any other.
pub fn power(a: any, b: any) -> NDArray { return _binary(a, b, "power") }

#: `a ** b` under numpy's other name.
pub fn float_power(a: any, b: any) -> NDArray { return _binary(a, b, "power") }

#: Remainder with the sign of the divisor, as Python's `%` has it.
pub fn remainder(a: any, b: any) -> NDArray { return _binary(a, b, "remainder") }

#: `remainder` under numpy's other name. Not `fmod`, which differs on every
#: mixed-sign pair.
pub fn mod(a: any, b: any) -> NDArray { return _binary(a, b, "remainder") }

#: Remainder with the sign of the dividend, as C's `fmod` has it.
pub fn fmod(a: any, b: any) -> NDArray { return _binary(a, b, "fmod") }

#: `floor(a / b)`. Paired with `remainder`, so `floor_divide * b + remainder`
#: recovers `a`.
pub fn floor_divide(a: any, b: any) -> NDArray { return _binary(a, b, "floor_divide") }

#: `sqrt(a*a + b*b)`, computed so that a large pair does not overflow on the
#: way.
pub fn hypot(a: any, b: any) -> NDArray { return _binary(a, b, "hypot") }

#: The angle of the point `(b, a)`, in radians — `arctan(a / b)` told which
#: quadrant it is in. Arguments in numpy's order, y first.
pub fn arctan2(a: any, b: any) -> NDArray { return _binary(a, b, "arctan2") }

#: `log(exp(a) + exp(b))` for arguments large enough that both exponentials
#: would overflow on their own.
pub fn logaddexp(a: any, b: any) -> NDArray { return _binary(a, b, "logaddexp") }

#: Larger of the two, propagating NaN. `fmax` is the one that steps over it.
pub fn maximum(a: any, b: any) -> NDArray { return _binary(a, b, "maximum") }

#: Smaller of the two, propagating NaN.
pub fn minimum(a: any, b: any) -> NDArray { return _binary(a, b, "minimum") }

#: Larger of the two, ignoring a NaN operand.
pub fn fmax(a: any, b: any) -> NDArray { return _binary(a, b, "fmax") }

#: Smaller of the two, ignoring a NaN operand.
pub fn fmin(a: any, b: any) -> NDArray { return _binary(a, b, "fmin") }

#: The magnitude of `a` with the sign of `b`. A negative zero counts as a
#: sign, which is the whole point of the operation.
pub fn copysign(a: any, b: any) -> NDArray { return _binary(a, b, "copysign") }

#: `a > b`, as a `BOOL` array.
#:
#: This family is why `NDArray` has no `__eq__`: an elementwise `==` that
#: returns an array breaks every `assert_eq` and every `is not null` guard
#: that follows it. These are the spellings that work.
pub fn greater(a: any, b: any) -> NDArray { return _binary(a, b, "greater") }

#: `a >= b`, as a `BOOL` array.
pub fn greater_equal(a: any, b: any) -> NDArray { return _binary(a, b, "greater_equal") }

#: `a < b`, as a `BOOL` array.
pub fn less(a: any, b: any) -> NDArray { return _binary(a, b, "less") }

#: `a <= b`, as a `BOOL` array.
pub fn less_equal(a: any, b: any) -> NDArray { return _binary(a, b, "less_equal") }

#: `a == b`, as a `BOOL` array. NaN equals nothing, itself included.
pub fn equal(a: any, b: any) -> NDArray { return _binary(a, b, "equal") }

#: `a != b`, as a `BOOL` array.
pub fn not_equal(a: any, b: any) -> NDArray { return _binary(a, b, "not_equal") }

#: Both non-zero, as a `BOOL` array.
pub fn logical_and(a: any, b: any) -> NDArray { return _binary(a, b, "logical_and") }

#: Either non-zero.
pub fn logical_or(a: any, b: any) -> NDArray { return _binary(a, b, "logical_or") }

#: Exactly one non-zero.
pub fn logical_xor(a: any, b: any) -> NDArray { return _binary(a, b, "logical_xor") }

#: Bitwise and. Integer dtypes only, and exact while the operands stay under
#: 2^24 — past that the float32 storage no longer holds every whole number and
#: the bits being anded are not the bits the caller meant.
pub fn bitwise_and(a: any, b: any) -> NDArray { return _binary(a, b, "bitwise_and") }

#: Bitwise or, under the same exactness bound.
pub fn bitwise_or(a: any, b: any) -> NDArray { return _binary(a, b, "bitwise_or") }

#: Bitwise exclusive or, under the same exactness bound.
pub fn bitwise_xor(a: any, b: any) -> NDArray { return _binary(a, b, "bitwise_xor") }

#: `a << b`, under the same exactness bound.
pub fn left_shift(a: any, b: any) -> NDArray { return _binary(a, b, "left_shift") }

#: `a >> b`, arithmetic, so the sign is carried down.
pub fn right_shift(a: any, b: any) -> NDArray { return _binary(a, b, "right_shift") }

#: Where `a` and `b` agree to within `atol + rtol * |b|`, as a `BOOL` array.
#:
#: Asymmetric in `a` and `b`, as numpy's is. Two infinities of the same sign
#: are close — the tolerance test on them is `|inf - inf|`, which is NaN and
#: would otherwise call every pair of infinities far apart. `allclose`, which
#: folds this to one bool, lives in `reduce`.
pub fn isclose(
    a: any,
    b: any,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    equal_nan: bool = false
) -> NDArray {
    return _binary(a, b, "isclose", rtol, atol, if equal_nan { 1.0 } else { 0.0 })
}

#: A bound for `clip`: the caller's value, or an infinity carrying the array's
#: own dtype.
#:
#: The dtype matters even though the value is a sentinel: promoting against a
#: bare float would float an integer array whose caller only left one side
#: open, which is not what `clip(ints, 0, null)` means.
fn _bound(value: any, open: float, dtype: int) -> (NDArray, bool) {
    if value is null {
        let out = NDArray([], dtype)
        out.upload([open])
        return (out, true)
    }
    return _operand(value)
}

#: Each element held between `a_min` and `a_max`, either of which may be null,
#: a scalar or an array.
#:
#: One kernel and not a `maximum` followed by a `minimum`: the pair would
#: write and re-read an intermediate the size of the output for nothing. A NaN
#: element stays NaN, and bounds given the wrong way round yield `a_max`, both
#: as numpy has it.
pub fn clip(a: NDArray, a_min: any = null, a_max: any = null) -> NDArray {
    if a_min is null and a_max is null {
        throw ValueError("clip needs at least one of a_min and a_max")
    }
    let (low, owns_low) = _bound(a_min, -_INF, a.dtype)
    let (high, owns_high) = _bound(a_max, _INF, a.dtype)
    defer { if owns_low { low.free() } }
    defer { if owns_high { high.free() } }
    let dtype = promote(promote(a.dtype, low.dtype), high.dtype)
    return _ternary(a, low, high, "clip", dtype)
}

#: `x` where `condition` is non-zero and `y` where it is not.
#:
#: Only the three-argument form. numpy's one-argument `where` is a search
#: rather than a selection and lives in `index` beside `nonzero`. The
#: condition contributes nothing to the result dtype, which is why the
#: three-operand path takes its dtype from the caller instead of reading it
#: off the table.
pub fn where(condition: NDArray, x: any, y: any) -> NDArray {
    let (chosen, owns_chosen) = _operand(x)
    let (other, owns_other) = _operand(y)
    defer { if owns_chosen { chosen.free() } }
    defer { if owns_other { other.free() } }
    return _ternary(condition, chosen, other, "where", promote(chosen.dtype, other.dtype))
}

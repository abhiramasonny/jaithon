#: Elementwise arithmetic, comparison, logic — and the one door out to the host.
#:
#: Everything here is column-at-a-time. Two Metal entry points do all the work:
#: `frame_binary`, parameterised by an op code, and `frame_unary`. Both read
#: the validity lanes of their inputs and write the validity lane of their
#: result, so a null propagates inside the dispatch and the host never branches
#: on one; both reconstruct the two-lane `Int64`/`Datetime` representation in
#: `long` so integer arithmetic stays exact to 48 bits.
#:
#: One kernel with a switch rather than one kernel per operation: these are
#: memory-bound — a thread reads two floats and writes one — so the branch
#: costs nothing next to the traffic, while sixteen pipelines would be sixteen
#: compiles and sixteen cache entries.
#:
#: The dtype rules are pandas': a `Float64` anywhere makes the result
#: `Float64`, `Bool` counts as a number and promotes to `Int64`, division and
#: exponentiation always produce `Float64`, and a comparison always produces
#: `Bool`. Nulls propagate through comparisons rather than reading as false,
#: which is what pandas' nullable dtypes do and what a validity lane is for.
#:
#: ```
#: let high = Column.from_floats([38.0, 41.0, NAN])
#: let low = Column.from_floats([21.0, 25.0, 19.0])
#: let swing = subtract(high, low)       # [17.0, 16.0, null]
#: let hot = greater(high, 40.0)         # [false, true, null]
#: ```
#:
#: `map`, `apply`, `applymap`, `pipe` and `combine` are the escape hatch. They
#: live here because a user callback is the thing arithmetic cannot vectorise,
#: and every one of them costs a round trip and a host loop over the rows.
import std.gpu as gpu
from std.gpu import Buffer, Kernel
from std.math import NAN, is_nan
from std.algo.sort import natural_order, sort_by
from .column import Column, DataFrame, Dtype, Index, Series, Value
from .column import MISSING_POSITION
from .column import concat_columns, f32_bits, positions_buffer, unify_levels, wide_pack

#: Op codes for `frame_binary`. The Metal switch reads the same numbers, and
#: nothing but these constants may assign them.
const OP_ADD = 0
const OP_SUBTRACT = 1
const OP_MULTIPLY = 2
const OP_DIVIDE = 3
const OP_FLOOR_DIVIDE = 4
const OP_MODULO = 5
const OP_POWER = 6
const OP_EQUAL = 7
const OP_NOT_EQUAL = 8
const OP_LESS = 9
const OP_LESS_EQUAL = 10
const OP_GREATER = 11
const OP_GREATER_EQUAL = 12
const OP_AND = 13
const OP_OR = 14
const OP_XOR = 15

#: Op codes for `frame_unary`, likewise mirrored in the shader.
const UN_NEGATE = 0
const UN_ABSOLUTE = 1
const UN_SIGN = 2
const UN_ROUND = 3
const UN_FLOOR = 4
const UN_CEILING = 5
const UN_TRUNCATE = 6
const UN_EXP = 7
const UN_LOG = 8
const UN_LOG2 = 9
const UN_LOG10 = 10
const UN_SQRT = 11
const UN_SQUARE = 12
const UN_RECIPROCAL = 13
const UN_CLIP = 14
const UN_NOT = 15

let SOURCE = """#include <metal_stdlib>
using namespace metal;

constexpr constant long WIDE_SCALE = 16777216L;

/* The two lanes are only ever recombined in `long`. In `float` everything
 * below the top 24 bits would be thrown away, which is the whole reason the
 * second lane exists. */
inline long wide_value(float low, float high) {
    return (long)high * WIDE_SCALE + (long)low;
}

inline void wide_store(device float* low, device float* high, uint at, long value) {
    long hi = value >> 24;
    low[at] = (float)(value - (hi << 24));
    high[at] = (float)hi;
}

/* The bit pattern, because `x != x` does not survive the compile: the host
 * hands Metal the default `MTLCompileOptions`, whose fast math promises there
 * are no NaNs, and the front end folds a self-comparison to false. That is how
 * `0.0 / 0.0` and `log(-2)` were coming back as present values with a NaN in
 * them instead of as the nulls pandas reports. Exponent all ones with any
 * mantissa is a NaN; exactly `0x7F800000` is an infinity, which pandas keeps. */
inline bool is_nan_bits(float x) {
    return (as_type<uint>(x) & 0x7FFFFFFFu) > 0x7F800000u;
}

/* Python's `%`, whose sign follows the divisor. `fmod`'s follows the dividend,
 * so `-7 % 3` would come out -1 where a pandas user is owed 2. */
inline float float_mod(float a, float b) {
    float r = fmod(a, b);
    if (r != 0.0f && ((r < 0.0f) != (b < 0.0f))) r += b;
    return r;
}

/* Codes 3 and 6 -- divide and power -- never arrive here: the host promotes
 * both to Float64, because `pow` is a float function and an exact integer
 * power leaves the 48 bits this encoding holds at 6^18. */
inline long apply_binary_exact(uint op, long a, long b) {
    switch (op) {
        case 0: return a + b;
        case 1: return a - b;
        case 2: return a * b;
        case 4: {
            /* One division and a subtraction for the remainder. A `long`
             * divide is a library call on this hardware, and asking for the
             * quotient and the modulus separately costs two of them. */
            long q = a / b;
            long r = a - q * b;
            return (r != 0 && ((r < 0) != (b < 0))) ? q - 1 : q;
        }
        case 5: {
            long r = a - (a / b) * b;
            return (r != 0 && ((r < 0) != (b < 0))) ? r + b : r;
        }
        case 7: return (a == b) ? 1 : 0;
        case 8: return (a != b) ? 1 : 0;
        case 9: return (a < b) ? 1 : 0;
        case 10: return (a <= b) ? 1 : 0;
        case 11: return (a > b) ? 1 : 0;
        default: return (a >= b) ? 1 : 0;
    }
}

inline float apply_binary(uint op, float a, float b) {
    switch (op) {
        case 0: return a + b;
        case 1: return a - b;
        case 2: return a * b;
        case 3: return a / b;
        case 4: return floor(a / b);
        case 5: return float_mod(a, b);
        /* `precise::pow` rather than `pow`: under the fast-math `pow` the
         * device computes exp2(b * log2(a)), and 7 ** 2 came out 48.99998 --
         * four ULP low -- where pandas answers 49. The precise one is exact
         * on the small integer exponents that `** 2` and `** 3` are, and
         * within one ULP at 10 ** 3. These kernels are bound by their memory
         * traffic, so the extra instructions cost nothing measurable. */
        case 6: return precise::pow(a, b);
        case 7: return (a == b) ? 1.0f : 0.0f;
        case 8: return (a != b) ? 1.0f : 0.0f;
        case 9: return (a < b) ? 1.0f : 0.0f;
        case 10: return (a <= b) ? 1.0f : 0.0f;
        case 11: return (a > b) ? 1.0f : 0.0f;
        default: return (a >= b) ? 1.0f : 0.0f;
    }
}

/* Codes 7 to 11 and 13 -- the transcendentals and the reciprocal -- never
 * arrive here: their result is Float64 whatever went in. */
inline long apply_unary_exact(uint op, long v, long arg_a, long arg_b, uint has_a, uint has_b) {
    switch (op) {
        case 0: return -v;
        case 1: return (v < 0) ? -v : v;
        case 2: return (v > 0) ? 1L : ((v < 0) ? -1L : 0L);
        case 3: {
            /* `arg_a` is the multiple to snap to: 1 for a non-negative
             * `decimals`, 10^-decimals below it. Ties go to even, which is
             * what `rint` does on the float side and what numpy does. */
            long m = arg_a;
            if (m <= 1) return v;
            long q = v / m;
            long r = v - q * m;
            if (r < 0) { r += m; q -= 1; }
            long twice = r * 2;
            if (twice > m || (twice == m && (q & 1L) != 0)) q += 1;
            return q * m;
        }
        case 4:
        case 5:
        case 6: return v;
        case 12: return v * v;
        case 14: {
            long out = v;
            if (has_a != 0u && out < arg_a) out = arg_a;
            if (has_b != 0u && out > arg_b) out = arg_b;
            return out;
        }
        default: return (v == 0) ? 1L : 0L;
    }
}

inline float apply_unary(uint op, float v, float arg_a, float arg_b, uint has_a, uint has_b) {
    switch (op) {
        case 0: return -v;
        case 1: return fabs(v);
        case 2: return sign(v);
        case 3: return rint(v * arg_a) / arg_a;
        case 4: return floor(v);
        case 5: return ceil(v);
        case 6: return trunc(v);
        case 7: return exp(v);
        case 8: return log(v);
        case 9: return log2(v);
        case 10: return log10(v);
        case 11: return sqrt(v);
        case 12: return v * v;
        case 13: return 1.0f / v;
        case 14: {
            float out = v;
            if (has_a != 0u) out = fmax(out, arg_a);
            if (has_b != 0u) out = fmin(out, arg_b);
            return out;
        }
        default: return (v == 0.0f) ? 1.0f : 0.0f;
    }
}

/* One row of the binary kernel: the fill, the arithmetic and the validity the
 * pair implies. Both dispatch paths call this, so the op switch and the null
 * rules exist once.
 *
 * `fill_low`/`fill_high` are the fill value in whichever mode is running: the
 * float itself when `exact` is 0, the two halves of the integer when it is 1.
 * The host writes the pair for the mode it asked for and zeroes the other. */
inline void binary_row(uint op, uint exact,
                       float a_low, float a_high, uint a_is_wide, float a_present,
                       float b_low, float b_high, uint b_is_wide, float b_present,
                       uint has_fill, float fill_low, float fill_high,
                       thread float* value, thread long* value_exact,
                       thread float* present) {
    bool a_null = a_present == 0.0f;
    bool b_null = b_present == 0.0f;
    if (op >= 13u) {
        /* Kleene, as pandas' nullable booleans are: a present false settles an
         * `and` and a present true settles an `or` whatever the other side is,
         * so `false & NA` is false rather than unknown. `xor` has no such
         * short circuit, so either null makes it one. */
        bool a_true = a_low != 0.0f;
        bool b_true = b_low != 0.0f;
        if (op == 13u) {
            bool settled = (!a_null && !a_true) || (!b_null && !b_true);
            *value = (settled || a_null || b_null) ? 0.0f : 1.0f;
            *present = (settled || !(a_null || b_null)) ? 1.0f : 0.0f;
        } else if (op == 14u) {
            bool settled = (!a_null && a_true) || (!b_null && b_true);
            *value = settled ? 1.0f : 0.0f;
            *present = (settled || !(a_null || b_null)) ? 1.0f : 0.0f;
        } else {
            *value = (a_true != b_true) ? 1.0f : 0.0f;
            *present = (a_null || b_null) ? 0.0f : 1.0f;
        }
        *value_exact = 0L;
        return;
    }
    long ax = (a_is_wide != 0u) ? wide_value(a_low, a_high) : (long)a_low;
    long bx = (b_is_wide != 0u) ? wide_value(b_low, b_high) : (long)b_low;
    float av = (a_is_wide != 0u) ? (float)ax : a_low;
    float bv = (b_is_wide != 0u) ? (float)bx : b_low;
    float here = (a_null || b_null) ? 0.0f : 1.0f;
    if (has_fill != 0u) {
        /* pandas' rule: a fill stands in for one missing side, and a row
         * missing on both sides stays missing. */
        long fx = wide_value(fill_low, fill_high);
        if (a_null) { av = fill_low; ax = fx; }
        if (b_null) { bv = fill_low; bx = fx; }
        here = (a_null && b_null) ? 0.0f : 1.0f;
    }
    if (exact != 0u) {
        long divisor = bx;
        if ((op == 4u || op == 5u) && divisor == 0) {
            /* There is no integer answer, and an integer divide by zero is
             * undefined on the device rather than infinite, so the row is
             * missing and the divisor is made harmless. */
            divisor = 1;
            here = 0.0f;
        }
        long result = apply_binary_exact(op, ax, divisor);
        *value_exact = result;
        *value = (float)result;
    } else {
        float result = apply_binary(op, av, bv);
        /* NaN is how a float column spells missing, so an operation that made
         * one made a null: 0/0 and a negative under a log both land here. An
         * infinity does not -- pandas keeps 1/0 as inf. */
        if (is_nan_bits(result)) here = 0.0f;
        *value = result;
        *value_exact = 0L;
    }
    *present = here;
}

inline void unary_row(uint op, uint exact,
                      float low, float high, uint is_wide, float in_present,
                      uint has_a, float arg_a_low, float arg_a_high,
                      uint has_b, float arg_b_low, float arg_b_high,
                      thread float* value, thread long* value_exact,
                      thread float* present) {
    long vx = (is_wide != 0u) ? wide_value(low, high) : (long)low;
    float vf = (is_wide != 0u) ? (float)vx : low;
    float here = (in_present == 0.0f) ? 0.0f : 1.0f;
    if (exact != 0u) {
        long result = apply_unary_exact(op, vx,
                                        wide_value(arg_a_low, arg_a_high),
                                        wide_value(arg_b_low, arg_b_high),
                                        has_a, has_b);
        *value_exact = result;
        *value = (float)result;
    } else {
        float result = apply_unary(op, vf, arg_a_low, arg_b_low, has_a, has_b);
        if (is_nan_bits(result)) here = 0.0f;
        *value = result;
        *value_exact = 0L;
    }
    *present = here;
}

/* A stride of 0 is how a scalar operand arrives: a one-slot column read by
 * every row, rather than a full-length one the host had to fill first. */
kernel void frame_binary(device const float* a_values [[buffer(0)]],
                         device const float* a_wide [[buffer(1)]],
                         device const float* a_valid [[buffer(2)]],
                         device const float* b_values [[buffer(3)]],
                         device const float* b_wide [[buffer(4)]],
                         device const float* b_valid [[buffer(5)]],
                         device float* out_values [[buffer(6)]],
                         device float* out_wide [[buffer(7)]],
                         device float* out_valid [[buffer(8)]],
                         constant uint& rows [[buffer(9)]],
                         constant uint& op [[buffer(10)]],
                         constant uint& span [[buffer(11)]],
                         constant uint& a_is_wide [[buffer(12)]],
                         constant uint& b_is_wide [[buffer(13)]],
                         constant uint& out_is_wide [[buffer(14)]],
                         constant uint& a_has_valid [[buffer(15)]],
                         constant uint& b_has_valid [[buffer(16)]],
                         constant uint& out_has_valid [[buffer(17)]],
                         constant uint& a_stride [[buffer(18)]],
                         constant uint& b_stride [[buffer(19)]],
                         constant uint& exact [[buffer(20)]],
                         constant uint& has_fill [[buffer(21)]],
                         constant uint& fill_low_bits [[buffer(22)]],
                         constant uint& fill_high_bits [[buffer(23)]],
                         uint gid [[thread_position_in_grid]]) {
    uint at = gid * span;
    if (at >= rows) return;
    float fill_low = as_type<float>(fill_low_bits);
    float fill_high = as_type<float>(fill_high_bits);
    if (span == 4u) {
        /* Four rows a thread. Every lane is one float32 array of the same
         * length, so each becomes a single 16-byte load and a single store,
         * and a kernel that reads two floats per row for every one it writes
         * is bound by exactly that traffic. The arithmetic still runs a row at
         * a time: widening the op switch to float4 would double the shader to
         * save instructions that are not the bottleneck.
         *
         * The host only sets `span` to 4 when no lane is wide and the row
         * count divides by four, so there is no ragged tail and every quad
         * starts on a 16-byte boundary. */
        float4 a_quad = (a_stride != 0u)
            ? *((device const float4*)(a_values + at)) : float4(a_values[0]);
        float4 b_quad = (b_stride != 0u)
            ? *((device const float4*)(b_values + at)) : float4(b_values[0]);
        float4 a_mask = float4(1.0f);
        float4 b_mask = float4(1.0f);
        if (a_has_valid != 0u) {
            a_mask = (a_stride != 0u)
                ? *((device const float4*)(a_valid + at)) : float4(a_valid[0]);
        }
        if (b_has_valid != 0u) {
            b_mask = (b_stride != 0u)
                ? *((device const float4*)(b_valid + at)) : float4(b_valid[0]);
        }
        float a_in[4] = {a_quad.x, a_quad.y, a_quad.z, a_quad.w};
        float b_in[4] = {b_quad.x, b_quad.y, b_quad.z, b_quad.w};
        float a_ok[4] = {a_mask.x, a_mask.y, a_mask.z, a_mask.w};
        float b_ok[4] = {b_mask.x, b_mask.y, b_mask.z, b_mask.w};
        float made[4];
        float kept[4];
        for (uint k = 0u; k < 4u; k++) {
            float value;
            long unused;
            float present;
            binary_row(op, exact,
                       a_in[k], 0.0f, 0u, a_ok[k],
                       b_in[k], 0.0f, 0u, b_ok[k],
                       has_fill, fill_low, fill_high,
                       &value, &unused, &present);
            made[k] = value;
            kept[k] = present;
        }
        *((device float4*)(out_values + at)) = float4(made[0], made[1], made[2], made[3]);
        if (out_has_valid != 0u) {
            *((device float4*)(out_valid + at)) = float4(kept[0], kept[1], kept[2], kept[3]);
        }
        return;
    }
    uint ai = at * a_stride;
    uint bi = at * b_stride;
    float value;
    long value_exact;
    float present;
    binary_row(op, exact,
               a_values[ai],
               (a_is_wide != 0u) ? a_wide[ai] : 0.0f,
               a_is_wide,
               (a_has_valid != 0u) ? a_valid[ai] : 1.0f,
               b_values[bi],
               (b_is_wide != 0u) ? b_wide[bi] : 0.0f,
               b_is_wide,
               (b_has_valid != 0u) ? b_valid[bi] : 1.0f,
               has_fill, fill_low, fill_high,
               &value, &value_exact, &present);
    if (out_is_wide != 0u) wide_store(out_values, out_wide, at, value_exact);
    else out_values[at] = value;
    if (out_has_valid != 0u) out_valid[at] = present;
}

/* `exact` doubles as the store decision here: a unary operation writes two
 * lanes exactly when it computed in `long`, so a second flag could only ever
 * disagree with this one. */
kernel void frame_unary(device const float* in_values [[buffer(0)]],
                        device const float* in_wide [[buffer(1)]],
                        device const float* in_valid [[buffer(2)]],
                        device float* out_values [[buffer(3)]],
                        device float* out_wide [[buffer(4)]],
                        device float* out_valid [[buffer(5)]],
                        constant uint& rows [[buffer(6)]],
                        constant uint& op [[buffer(7)]],
                        constant uint& span [[buffer(8)]],
                        constant uint& in_is_wide [[buffer(9)]],
                        constant uint& in_has_valid [[buffer(10)]],
                        constant uint& out_has_valid [[buffer(11)]],
                        constant uint& exact [[buffer(12)]],
                        constant uint& has_arg_a [[buffer(13)]],
                        constant uint& arg_a_low_bits [[buffer(14)]],
                        constant uint& arg_a_high_bits [[buffer(15)]],
                        constant uint& has_arg_b [[buffer(16)]],
                        constant uint& arg_b_low_bits [[buffer(17)]],
                        constant uint& arg_b_high_bits [[buffer(18)]],
                        uint gid [[thread_position_in_grid]]) {
    uint at = gid * span;
    if (at >= rows) return;
    float arg_a_low = as_type<float>(arg_a_low_bits);
    float arg_a_high = as_type<float>(arg_a_high_bits);
    float arg_b_low = as_type<float>(arg_b_low_bits);
    float arg_b_high = as_type<float>(arg_b_high_bits);
    if (span == 4u) {
        float4 quad = *((device const float4*)(in_values + at));
        float4 mask = float4(1.0f);
        if (in_has_valid != 0u) mask = *((device const float4*)(in_valid + at));
        float in[4] = {quad.x, quad.y, quad.z, quad.w};
        float ok[4] = {mask.x, mask.y, mask.z, mask.w};
        float made[4];
        float kept[4];
        for (uint k = 0u; k < 4u; k++) {
            float value;
            long unused;
            float present;
            unary_row(op, 0u, in[k], 0.0f, 0u, ok[k],
                      has_arg_a, arg_a_low, arg_a_high,
                      has_arg_b, arg_b_low, arg_b_high,
                      &value, &unused, &present);
            made[k] = value;
            kept[k] = present;
        }
        *((device float4*)(out_values + at)) = float4(made[0], made[1], made[2], made[3]);
        if (out_has_valid != 0u) {
            *((device float4*)(out_valid + at)) = float4(kept[0], kept[1], kept[2], kept[3]);
        }
        return;
    }
    float value;
    long value_exact;
    float present;
    unary_row(op, exact,
              in_values[at],
              (in_is_wide != 0u) ? in_wide[at] : 0.0f,
              in_is_wide,
              (in_has_valid != 0u) ? in_valid[at] : 1.0f,
              has_arg_a, arg_a_low, arg_a_high,
              has_arg_b, arg_b_low, arg_b_high,
              &value, &value_exact, &present);
    if (exact != 0u) wide_store(out_values, out_wide, at, value_exact);
    else out_values[at] = value;
    if (out_has_valid != 0u) out_valid[at] = present;
}
"""

var _kernels: dict[str, Kernel] = {}

#: The compiled pipeline for one entry point in the arithmetic shader library.
pub fn cached_kernel(entry: str) -> Kernel {
    let existing = _kernels.get(entry, null)
    if existing is not null { return existing }
    let kernel = Kernel.compile(SOURCE, entry)
    _kernels[entry] = kernel
    return kernel
}

#: The dtype an operation between `a` and `b` computes in.
#:
#: pandas' ladder, and the reason it is a named function rather than a table
#: inside the dispatch: `astype`, a join's key check and a concatenation all
#: want to ask the same question.
#:
#: `Bool` promotes to `Int64` rather than staying `Bool`, so that `mask + mask`
#: counts the way `sum` over a mask counts; numpy would have made it a logical
#: or, which is what `or_` is for. Two dictionary columns share their dtype,
#: which is enough for a comparison and is refused for arithmetic further down.
#: `Datetime` only pairs with itself: a difference of timestamps is a duration,
#: which is the timeseries module's business.
pub fn promote(a: Dtype, b: Dtype) -> Dtype {
    if a.is_dictionary() and b.is_dictionary() { return a }
    if a.is_dictionary() or b.is_dictionary() {
        throw TypeError(f"{a.name()} and {b.name()} have no type in common")
    }
    if a == Dtype.Datetime or b == Dtype.Datetime {
        if a == b { return Dtype.Datetime }
        throw TypeError(f"{a.name()} and {b.name()} have no type in common")
    }
    if a == Dtype.Float64 or b == Dtype.Float64 { return Dtype.Float64 }
    return Dtype.Int64
}

#: Whether the op yields a `Bool` column whatever its operands were.
fn is_comparison(op: int) -> bool { return op >= OP_EQUAL and op <= OP_GREATER_EQUAL }

#: Whether the op orders its operands rather than only testing them equal.
fn is_ordering(op: int) -> bool { return op >= OP_LESS and op <= OP_GREATER_EQUAL }

fn is_logical(op: int) -> bool { return op >= OP_AND }

fn binary_name(op: int) -> str {
    let names = [
        "add",
        "subtract",
        "multiply",
        "divide",
        "floor_divide",
        "modulo",
        "power",
        "equal",
        "not_equal",
        "less",
        "less_equal",
        "greater",
        "greater_equal",
        "and_",
        "or_",
        "xor",
    ]
    return names[op]
}

fn unary_name(op: int) -> str {
    let names = [
        "negate",
        "absolute",
        "sign",
        "round",
        "floor",
        "ceiling",
        "truncate",
        "exp",
        "log",
        "log2",
        "log10",
        "sqrt",
        "square",
        "reciprocal",
        "clip",
        "not_",
    ]
    return names[op]
}

#: The dtype the result of `op` is stored as.
fn binary_result(op: int, common: Dtype) -> Dtype {
    if is_comparison(op) or is_logical(op) { return Dtype.Bool }
    if common.is_dictionary() {
        throw TypeError(f"{binary_name(op)} is not defined on {common.name()}")
    }
    if common == Dtype.Datetime {
        throw TypeError(f"{binary_name(op)} on two timestamps is a duration, which timeseries owns")
    }
    #: Division and exponentiation leave the integers whatever went in: 7 // 2
    #: is exact, but 7 / 2 is not an integer and 2 ** 40 is past the 48 bits
    #: two lanes hold.
    if op == OP_DIVIDE or op == OP_POWER { return Dtype.Float64 }
    return common
}

#: Whether the operation can turn present operands into a missing result, and
#: so needs a validity lane even when neither input had one.
fn binary_may_null(op: int, exact: bool) -> bool {
    if is_logical(op) { return false }
    if exact { return op == OP_FLOOR_DIVIDE or op == OP_MODULO }
    return op == OP_DIVIDE or op == OP_FLOOR_DIVIDE or op == OP_MODULO or op == OP_POWER
}

fn unary_may_null(op: int) -> bool {
    return op == UN_LOG or op == UN_LOG2 or op == UN_LOG10 or op == UN_SQRT
}

fn flag(value: bool) -> int { return if value { 1 } else { 0 } }

fn require_device(what: str) -> void {
    if not gpu.is_available() {
        throw RuntimeError(f"jaiframe needs a GPU to {what}")
    }
}

#: One operand of a binary dispatch: a column, and how far the kernel walks it.
#:
#: A stride of 0 is a scalar — a one-row column every thread reads slot zero
#: of. Materialising the scalar to full length instead would cost an allocation
#: and a fill the size of the data, which for a `frame + 1` is the whole frame.
class Side {
    #: The lanes the kernel reads.
    pub var column: Column
    #: 1 for a real column, 0 for a broadcast scalar.
    pub var stride: int
    #: Whether this side allocated its column and must release it.
    pub var owned: bool

    pub fn init(self, column: Column, stride: int, owned: bool) {
        self.column = column
        self.stride = stride
        self.owned = owned
    }

    #: Release the column when this side built it.
    pub fn release(self) -> void {
        if self.owned { self.column.free() }
    }
}

#: The column behind a `Series` or a `Column`, or null for a scalar.
fn column_of(value: any) -> Column? {
    if value is Column { return value }
    if value is Series { return value.values }
    return null
}

#: A host scalar as a `Value`, typed against the column it will meet.
fn scalar_value(value: any, against: Dtype) -> Value {
    if value is Value { return value }
    if value is bool { return Value.of_bool(value) }
    if value is int { return Value.of_int(value) }
    if value is float { return Value.of_float(value) }
    if value is str {
        return Value.of_str(value, if against.is_dictionary() { against } else { Dtype.Str })
    }
    throw TypeError("an operand must be a Series, a DataFrame, a Column or a scalar")
}

#: The dtype a one-row column for `cell` should carry.
#:
#: An integer written next to a `Datetime` column is milliseconds, not a count,
#: and a string next to a `Category` is a category. Typing the scalar against
#: its neighbour is what keeps `stamps > 1700000000000` from being a promotion
#: error.
fn scalar_dtype(cell: Value, against: Dtype) -> Dtype {
    if against == Dtype.Datetime and cell.dtype == Dtype.Int64 { return Dtype.Datetime }
    if against == Dtype.Category and cell.dtype == Dtype.Str { return Dtype.Category }
    return cell.dtype
}

fn side_of(value: any, against: Dtype) -> Side {
    let existing = column_of(value)
    if existing is not null { return Side(existing, 1, false) }
    let cell = scalar_value(value, against)
    if cell.is_na() { return Side(Column.nulls(against, 1), 0, true) }
    let dtype = scalar_dtype(cell, against)
    return Side(Column.full(dtype, 1, cell), 0, true)
}

#: The two scalars the kernel reads a fill or a bound through.
#:
#: In exact mode they are the halves of the 48-bit integer; in float mode the
#: first is the float and the second is zero, because `wide_value` would read
#: it either way and a stale half would be a silent wrong answer.
fn packed_bits(cell: Value?, exact: bool) -> (int, int) {
    if cell is null { return (0, 0) }
    if exact {
        let split = wide_pack(cell.to_int())
        return (f32_bits(split[0]), f32_bits(split[1]))
    }
    return (f32_bits(cell.to_float()), 0)
}

#: Whether four rows a thread is safe for this shape.
#:
#: The quad path reads and writes whole float4s, so no lane may be the high
#: half of a two-lane integer and the rows must divide by four. Everything else
#: — a scalar operand, a validity lane, an exact comparison of two `Bool`
#: columns — it handles.
fn quad_span(rows: int, wide_anywhere: bool) -> int {
    if wide_anywhere or rows == 0 or rows % 4 != 0 { return 1 }
    return 4
}

#: Run `op` over two operands that already agree on length.
fn dispatch_binary(a: Side, b: Side, rows: int, op: int, fill: Value?) -> Column {
    require_device("evaluate an expression")
    let common = promote(a.column.dtype, b.column.dtype)
    let out_dtype = binary_result(op, common)
    #: Exact means the arithmetic runs in `long`. Everything integral goes
    #: that way, including a `Bool` operand, whose 0 and 1 are exact in either
    #: mode. Dictionary codes stay in float: they are small non-negative
    #: integers and only ever compared.
    let numeric = common != Dtype.Float64 and not common.is_dictionary()
    let exact = numeric and out_dtype != Dtype.Float64
    let out_wide = out_dtype.is_wide()
    let inputs_null = a.column.has_validity() or b.column.has_validity()
    let with_validity = inputs_null or binary_may_null(op, exact)
    let out = Column.allocate(out_dtype, rows, with_validity: with_validity)
    if rows == 0 { return out }
    let fill_bits = packed_bits(fill, exact)
    let wide_anywhere = a.column.dtype.is_wide() or b.column.dtype.is_wide() or out_wide
    let span = quad_span(rows, wide_anywhere)
    cached_kernel("frame_binary").dispatch_async(
        [
            a.column.values,
            a.column.wide ?? a.column.values,
            a.column.valid ?? a.column.values,
            b.column.values,
            b.column.wide ?? b.column.values,
            b.column.valid ?? b.column.values,
            out.values,
            out.wide ?? out.values,
            out.valid ?? out.values,
        ],
        (rows + span - 1) // span,
        [
            rows,
            op,
            span,
            flag(a.column.dtype.is_wide()),
            flag(b.column.dtype.is_wide()),
            flag(out_wide),
            flag(a.column.has_validity()),
            flag(b.column.has_validity()),
            flag(with_validity),
            a.stride,
            b.stride,
            flag(exact),
            flag(fill is not null),
            fill_bits[0],
            fill_bits[1],
        ]
    )
    return out
}

#: The dtype a unary operation over `dtype` produces.
fn unary_result(op: int, dtype: Dtype) -> Dtype {
    if op == UN_NOT {
        if dtype != Dtype.Bool {
            throw TypeError(f"not_ is defined on bool, not on {dtype.name()}")
        }
        return Dtype.Bool
    }
    if dtype.is_dictionary() {
        throw TypeError(f"{unary_name(op)} is not defined on {dtype.name()}")
    }
    if dtype == Dtype.Datetime {
        #: Clipping a timestamp to a window is a date; negating one is not.
        if op != UN_CLIP {
            throw TypeError(f"{unary_name(op)} is not defined on {dtype.name()}")
        }
        return Dtype.Datetime
    }
    let transcendental = op == UN_EXP or op == UN_LOG or op == UN_LOG2 or op == UN_LOG10
    if transcendental or op == UN_SQRT or op == UN_RECIPROCAL { return Dtype.Float64 }
    if dtype == Dtype.Bool { return Dtype.Int64 }
    return dtype
}

fn dispatch_unary(source: Column, op: int, arg_a: Value?, arg_b: Value?) -> Column {
    require_device("evaluate an expression")
    let out_dtype = unary_result(op, source.dtype)
    let exact = out_dtype.is_wide()
    let with_validity = source.has_validity() or unary_may_null(op)
    let out = Column.allocate(out_dtype, source.len(), with_validity: with_validity)
    let rows = source.len()
    if rows == 0 { return out }
    let a_bits = packed_bits(arg_a, exact)
    let b_bits = packed_bits(arg_b, exact)
    let span = quad_span(rows, source.dtype.is_wide() or exact)
    cached_kernel("frame_unary").dispatch_async(
        [
            source.values,
            source.wide ?? source.values,
            source.valid ?? source.values,
            out.values,
            out.wide ?? out.values,
            out.valid ?? out.values,
        ],
        (rows + span - 1) // span,
        [
            rows,
            op,
            span,
            flag(source.dtype.is_wide()),
            flag(source.has_validity()),
            flag(with_validity),
            flag(exact),
            flag(arg_a is not null),
            a_bits[0],
            a_bits[1],
            flag(arg_b is not null),
            b_bits[0],
            b_bits[1],
        ]
    )
    return out
}

#: Every part's dictionary rebuilt onto one table whose codes are ranks.
#:
#: A text comparison is an integer compare on the codes, which only answers
#: `less` if the codes are in the strings' order. So the shared table is seeded
#: with the sorted union before `unify_levels` walks the parts, and the codes
#: it hands back are ranks. The sort is over the distinct values, not the rows:
#: a million-row city column with fifty cities sorts fifty strings.
fn on_one_ordered_table(parts: list[Column]) -> list[Column] {
    var names: list[str] = []
    var seen: dict[str, int] = {}
    for part in parts {
        for name in part.levels ?? [] {
            if seen.get(name, -1) >= 0 { continue }
            seen[name] = 1
            names.push(name)
        }
    }
    let seed = Column.from_strs([], parts[0].dtype)
    seed.set_levels(sort_by(names, natural_order))
    let unified = unify_levels([seed] + parts)
    seed.free()
    unified[0].free()
    return unified[1:unified.len()]
}

#: A one-row dictionary column whose code no level can ever match.
#:
#: `level_code` says a scalar the dictionary has never seen is in it, and this
#: is how that becomes an answer rather than a special case: comparing against
#: -1 gives false on every row in one dispatch, and the kernel carries the
#: source's nulls the way it always does. The alternative is remapping both
#: sides through `unify_levels` to discover the same thing.
fn absent_code_column(dtype: Dtype) -> Column {
    let column = Column.allocate(dtype, 1)
    column.values.fill(-1.0)
    return column
}

#: Whether `side` is a scalar string the other column's dictionary lacks.
fn scalar_is_unseen(side: Side, other: Column) -> bool {
    if side.stride != 0 or side.column.level_count() != 1 { return false }
    if not other.dtype.is_dictionary() or other.level_count() == 0 { return false }
    return other.level_code(side.column.level_at(0)) is null
}

#: Comparison between dictionary-encoded operands.
fn text_binary(a: Side, b: Side, rows: int, op: int) -> Column {
    if not is_comparison(op) {
        throw TypeError(f"{binary_name(op)} is not defined on text")
    }
    if not (a.column.dtype.is_dictionary() and b.column.dtype.is_dictionary()) {
        throw TypeError(f"cannot compare {a.column.dtype.name()} with {b.column.dtype.name()}")
    }
    if not is_ordering(op) {
        if scalar_is_unseen(a, b.column) {
            let sentinel = absent_code_column(a.column.dtype)
            let out = dispatch_binary(Side(sentinel, 0, false), b, rows, op, null)
            sentinel.free()
            return out
        }
        if scalar_is_unseen(b, a.column) {
            let sentinel = absent_code_column(b.column.dtype)
            let out = dispatch_binary(a, Side(sentinel, 0, false), rows, op, null)
            sentinel.free()
            return out
        }
        let unified = unify_levels([a.column, b.column])
        let out = dispatch_binary(
            Side(unified[0], a.stride, false),
            Side(unified[1], b.stride, false),
            rows,
            op,
            null
        )
        for part in unified { part.free() }
        return out
    }
    let ordered = on_one_ordered_table([a.column, b.column])
    let out = dispatch_binary(
        Side(ordered[0], a.stride, false),
        Side(ordered[1], b.stride, false),
        rows,
        op,
        null
    )
    for part in ordered { part.free() }
    return out
}

#: `op` over two columns, or a column and a scalar. No labels are consulted.
fn plain_binary(left: any, right: any, op: int, fill: Value?) -> Column {
    let a_column = column_of(left)
    let b_column = column_of(right)
    if a_column is null and b_column is null {
        throw TypeError(f"{binary_name(op)} needs a column on at least one side")
    }
    if a_column is not null and b_column is not null and a_column.len() != b_column.len() {
        throw ValueError(
            f"{a_column.len()} rows against {b_column.len()}: align them before {binary_name(op)}"
        )
    }
    let rows = if a_column is not null { a_column.len() } else { b_column.len() }
    let left_dtype = if a_column is not null { a_column.dtype } else { Dtype.Float64 }
    let right_dtype = if b_column is not null { b_column.dtype } else { Dtype.Float64 }
    let a = side_of(left, right_dtype)
    let b = side_of(right, left_dtype)
    let out = if a.column.dtype.is_dictionary() or b.column.dtype.is_dictionary() {
        text_binary(a, b, rows, op)
    } else {
        dispatch_binary(a, b, rows, op, fill)
    }
    a.release()
    b.release()
    return out
}

#: Every cell of a column, as host `Value`s.
#:
#: One download per lane, which is the whole cost of leaving the device. Every
#: callback in this module and the label matching in `align_indexes` go through
#: here so that there is exactly one place that pays it.
fn column_cells(column: Column) -> list[Value] {
    if column.len() == 0 { return [] }
    let dtype = column.dtype
    var out: list[Value] = []
    if dtype.is_dictionary() {
        for text in column.to_strs() {
            out.push(if text is null { Value.null_of(dtype) } else { Value.of_str(text, dtype) })
        }
        return out
    }
    let missing = column.null_mask()
    if dtype.is_wide() {
        for (row, value) in column.to_ints().enumerate() {
            out.push(if missing[row] { Value.null_of(dtype) } else { Value(dtype, integer: value) })
        }
        return out
    }
    for (row, value) in column.to_floats().enumerate() {
        if missing[row] or is_nan(value) {
            out.push(Value.null_of(dtype))
        } elif dtype == Dtype.Bool {
            out.push(Value.of_bool(value != 0.0))
        } else {
            out.push(Value.of_float(value))
        }
    }
    return out
}

#: The dtype a list of host cells should be stored as.
#:
#: A text cell anywhere makes the column text, because a dictionary can hold
#: the printed form of a number and no number can hold a word. Otherwise the
#: widest numeric kind present wins, which is the same ladder `promote` walks.
fn inferred_dtype(cells: list[Value], hint: Dtype?) -> Dtype {
    var saw_text = false
    var saw_float = false
    var saw_int = false
    var saw_time = false
    var saw_any = false
    for cell in cells {
        if cell.is_na() { continue }
        saw_any = true
        if cell.dtype.is_dictionary() {
            saw_text = true
        } elif cell.dtype == Dtype.Float64 {
            saw_float = true
        } elif cell.dtype == Dtype.Int64 {
            saw_int = true
        } elif cell.dtype == Dtype.Datetime {
            saw_time = true
        }
    }
    if not saw_any { return hint ?? Dtype.Float64 }
    if saw_text { return Dtype.Str }
    if saw_time and not saw_float and not saw_int { return Dtype.Datetime }
    if saw_float { return Dtype.Float64 }
    if saw_int or saw_time { return Dtype.Int64 }
    return Dtype.Bool
}

#: Host cells back into a column, in one upload.
fn column_of_cells(cells: list[Value], hint: Dtype? = null) -> Column {
    let dtype = inferred_dtype(cells, hint)
    var missing: list[bool] = []
    for cell in cells { missing.push(cell.is_na()) }
    if dtype.is_dictionary() {
        var texts: list[str?] = []
        for cell in cells { texts.push(if cell.is_na() { null } else { cell.to_str() }) }
        return Column.from_strs(texts, dtype)
    }
    if dtype.is_wide() {
        var numbers: list[int] = []
        for cell in cells { numbers.push(if cell.is_na() { 0 } else { cell.to_int() }) }
        return Column.from_ints(numbers, missing, dtype)
    }
    if dtype == Dtype.Bool {
        var flags: list[bool] = []
        for cell in cells { flags.push(not cell.is_na() and cell.to_bool()) }
        return Column.from_bools(flags, missing)
    }
    var numbers: list[float] = []
    for cell in cells { numbers.push(if cell.is_na() { NAN } else { cell.to_float() }) }
    return Column.from_floats(numbers, missing)
}

#: One collision-free host key per row of an index.
#:
#: Length-prefixed so that a two-level label of ("a", "bc") cannot read as
#: ("ab", "c"), and present-tagged so that a null level is not the string it
#: would print as.
fn index_keys(index: Index) -> list[str] {
    let levels = index.materialize()
    var keys: list[str] = []
    for _row in 0..index.len() { keys.push("") }
    for level in levels {
        for (row, cell) in column_cells(level).enumerate() {
            let part = if cell.is_na() { "!" } else { "=" + cell.to_str() }
            keys[row] = keys[row] + f"{part.len()}:{part}"
        }
    }
    for level in levels { level.free() }
    return keys
}

#: Where each side's rows land once two indexes are put on one set of labels.
class Alignment {
    #: The labels the result carries.
    pub var index: Index
    #: How many rows the result has.
    pub var length: int
    #: Source row per result row for the left side, `null` when it already
    #: lines up and nothing has to be gathered.
    pub var left: Buffer?
    #: The same for the right side.
    pub var right: Buffer?

    pub fn init(self, index: Index, length: int, left: Buffer?, right: Buffer?) {
        self.index = index
        self.length = length
        self.left = left
        self.right = right
    }

    #: Whether both sides already sit on the result's labels.
    pub fn is_identity(self) -> bool { return self.left is null and self.right is null }

    #: Release the position lanes. The index is handed to the result, so it
    #: outlives this and is not freed here.
    pub fn free_positions(self) -> void {
        let a = self.left
        if a is not null { a.free() }
        let b = self.right
        if b is not null { b.free() }
    }
}

#: Whether two indexes label the same rows without anything being compared.
#:
#: The same object is the common case — `frame["a"] + frame["b"]` share one —
#: and two default range indexes are the next. Anything else goes through the
#: label pass, which will discover an equality it could not have assumed.
fn indexes_agree(a: Index, b: Index) -> bool {
    if a == b { return true }
    if a.len() != b.len() { return false }
    if a.is_range() and b.is_range() { return a.start == b.start and a.step == b.step }
    return false
}

fn shared_names(a: Index, b: Index) -> list[str?] {
    var out: list[str?] = []
    for (at, name) in a.names.enumerate() {
        out.push(if at < b.names.len() and name == b.names[at] { name } else { null })
    }
    return out
}

#: Two indexes put on the union of their labels.
#:
#: The union is in first-appearance order — the left index, then the labels
#: only the right has — which is the order `unify_levels` and `concat_columns`
#: already use. pandas sorts it, and would then need the labels to be
#: orderable, which a multi-index of mixed dtypes is not.
#:
#: The matching is a host pass over the labels: one download per index level,
#: then a dictionary of keys. That is a pass over the rows, and it is why the
#: identity check above matters. The union index itself is built on the device,
#: by taking from the two label sets stacked end to end.
#:
#: This is the alignment the whole module runs on, and it belongs with the
#: index rather than with arithmetic; when `index.jai` exports `align`, this
#: function becomes a call into it.
fn align_indexes(a: Index, b: Index) -> Alignment {
    if indexes_agree(a, b) { return Alignment(a.copy(), a.len(), null, null) }
    if a.nlevels() != b.nlevels() {
        throw ValueError(f"an index of {a.nlevels()} levels cannot align with one of {b.nlevels()}")
    }
    let left_keys = index_keys(a)
    let right_keys = index_keys(b)
    var slot: dict[str, int] = {}
    var left_at: list[int] = []
    var right_at: list[int] = []
    for (row, key) in left_keys.enumerate() {
        if slot.get(key, -1) >= 0 {
            throw ValueError("cannot align on an index with duplicate labels")
        }
        slot[key] = left_at.len()
        left_at.push(row)
        right_at.push(MISSING_POSITION)
    }
    var placed: dict[str, int] = {}
    for (row, key) in right_keys.enumerate() {
        if placed.get(key, -1) >= 0 {
            throw ValueError("cannot align on an index with duplicate labels")
        }
        placed[key] = 1
        let at = slot.get(key, -1)
        if at >= 0 {
            right_at[at] = row
            continue
        }
        slot[key] = left_at.len()
        left_at.push(MISSING_POSITION)
        right_at.push(row)
    }
    let length = left_at.len()
    #: Same labels in the same order after all, which a stored index cannot be
    #: assumed into but is common enough to be worth not gathering for.
    if length == a.len() and length == b.len() and left_at == right_at {
        return Alignment(a.copy(), length, null, null)
    }
    let a_levels = a.materialize()
    let b_levels = b.materialize()
    var stacked: list[Column] = []
    for (at, level) in a_levels.enumerate() {
        stacked.push(concat_columns([level, b_levels[at]]))
    }
    var reach: list[int] = []
    for (at, from_left) in left_at.enumerate() {
        reach.push(if from_left >= 0 { from_left } else { a.len() + right_at[at] })
    }
    let lane = positions_buffer(reach)
    let index = Index.of(stacked.map(|level| level.take(lane, length)), shared_names(a, b))
    lane.free()
    for level in stacked { level.free() }
    for level in a_levels { level.free() }
    for level in b_levels { level.free() }
    return Alignment(index, length, positions_buffer(left_at), positions_buffer(right_at))
}

#: The column a side contributes to an aligned operation.
fn aligned_column(source: Column, positions: Buffer?, length: int) -> (Column, bool) {
    if positions is null { return (source, false) }
    return (source.take(positions, length, allow_missing: true), true)
}

#: pandas keeps a name only when both sides agree on it.
fn shared_name(a: str?, b: str?) -> str? { return if a == b { a } else { null } }

fn aligned_series(left: Series, right: Series, op: int, fill: Value?) -> Series {
    let plan = align_indexes(left.index, right.index)
    let a = aligned_column(left.values, plan.left, plan.length)
    let b = aligned_column(right.values, plan.right, plan.length)
    let out = plain_binary(a[0], b[0], op, fill)
    if a[1] { a[0].free() }
    if b[1] { b[0].free() }
    plan.free_positions()
    return Series(out, index: plan.index, name: shared_name(left.name, right.name))
}

fn series_binary(left: any, right: any, op: int, fill: Value?) -> Series {
    if left is Series and right is Series { return aligned_series(left, right, op, fill) }
    if left is Series {
        return Series(
            plain_binary(left.values, right, op, fill),
            index: left.index.copy(),
            name: left.name
        )
    }
    return Series(
        plain_binary(left, right.values, op, fill),
        index: right.index.copy(),
        name: right.name
    )
}

#: The union of two frames' column names, left order first.
fn union_names(a: list[str], b: list[str]) -> list[str] {
    var names: list[str] = []
    var seen: dict[str, int] = {}
    for name in a {
        seen[name] = 1
        names.push(name)
    }
    for name in b {
        if seen.get(name, -1) >= 0 { continue }
        seen[name] = 1
        names.push(name)
    }
    return names
}

fn aligned_frames(left: DataFrame, right: DataFrame, op: int, fill: Value?) -> DataFrame {
    let plan = align_indexes(left.index, right.index)
    let names = union_names(left.names, right.names)
    var columns: list[Column] = []
    for name in names {
        #: A column only one frame has meets a column of nulls, so a
        #: `fill_value` reaches it exactly as it reaches a missing row. That is
        #: what makes `a.add(b, fill_value: 0)` a union rather than an
        #: intersection.
        let dtype = if left.has(name) { left.column(name).dtype } else { right.column(name).dtype }
        let a = if left.has(name) {
            aligned_column(left.column(name), plan.left, plan.length)
        } else {
            (Column.nulls(dtype, plan.length), true)
        }
        let b = if right.has(name) {
            aligned_column(right.column(name), plan.right, plan.length)
        } else {
            (Column.nulls(dtype, plan.length), true)
        }
        columns.push(plain_binary(a[0], b[0], op, fill))
        if a[1] { a[0].free() }
        if b[1] { b[0].free() }
    }
    plan.free_positions()
    return DataFrame(names, columns, index: plan.index)
}

#: A frame against a series, which pandas matches to the column names.
#:
#: One scalar per column, so the series' values come down once and each column
#: meets a one-slot operand. A label the frame has no column for still becomes
#: a column, all null, the way pandas' union does.
fn frame_over_series(
    frame: DataFrame,
    series: Series,
    op: int,
    fill: Value?,
    frame_left: bool
) -> DataFrame {
    if series.index.nlevels() != 1 {
        throw ValueError("a frame lines up against a series of one label per column")
    }
    let levels = series.index.materialize()
    var labels: list[str] = []
    for cell in column_cells(levels[0]) { labels.push(cell.to_str()) }
    for level in levels { level.free() }
    let cells = column_cells(series.values)
    var lookup: dict[str, int] = {}
    for (at, label) in labels.enumerate() { lookup[label] = at }
    var names: list[str] = []
    var columns: list[Column] = []
    for (at, name) in frame.names.enumerate() {
        let found = lookup.get(name, -1)
        let scalar = if found >= 0 { cells[found] } else { Value.null_of(frame.columns[at].dtype) }
        names.push(name)
        columns.push(if frame_left {
            plain_binary(frame.columns[at], scalar, op, fill)
        } else {
            plain_binary(scalar, frame.columns[at], op, fill)
        })
    }
    for (at, label) in labels.enumerate() {
        if frame.has(label) { continue }
        let empty = Column.nulls(cells[at].dtype, frame.len())
        names.push(label)
        columns.push(if frame_left {
            plain_binary(empty, cells[at], op, fill)
        } else {
            plain_binary(cells[at], empty, op, fill)
        })
        empty.free()
    }
    return DataFrame(names, columns, index: frame.index.copy())
}

fn frame_over_scalar(
    frame: DataFrame,
    other: any,
    op: int,
    fill: Value?,
    frame_left: bool
) -> DataFrame {
    if column_of(other) is not null {
        throw TypeError(
            f"{binary_name(op)} between a DataFrame and a Column is ambiguous: " +
                "give the column an index and pass a Series"
        )
    }
    var columns: list[Column] = []
    for column in frame.columns {
        columns.push(if frame_left {
            plain_binary(column, other, op, fill)
        } else {
            plain_binary(other, column, op, fill)
        })
    }
    return frame._replace(columns: columns, index: frame.index.copy())
}

fn frame_binary(left: any, right: any, op: int, fill: Value?) -> DataFrame {
    if left is DataFrame and right is DataFrame { return aligned_frames(left, right, op, fill) }
    if left is DataFrame and right is Series {
        return frame_over_series(left, right, op, fill, true)
    }
    if right is DataFrame and left is Series {
        return frame_over_series(right, left, op, fill, false)
    }
    if left is DataFrame { return frame_over_scalar(left, right, op, fill, true) }
    return frame_over_scalar(right, left, op, fill, false)
}

#: The one entry every binary operation funnels through.
fn binary(left: any, right: any, op: int, fill: Value?) -> any {
    if left is DataFrame or right is DataFrame { return frame_binary(left, right, op, fill) }
    if left is Series or right is Series { return series_binary(left, right, op, fill) }
    return plain_binary(left, right, op, fill)
}

#: The one entry every unary operation funnels through.
fn unary(target: any, op: int, arg_a: Value?, arg_b: Value?) -> any {
    if target is DataFrame {
        var columns: list[Column] = []
        for column in target.columns { columns.push(dispatch_unary(column, op, arg_a, arg_b)) }
        return target._replace(columns: columns, index: target.index.copy())
    }
    if target is Series {
        return Series(
            dispatch_unary(target.values, op, arg_a, arg_b),
            index: target.index.copy(),
            name: target.name
        )
    }
    if target is Column { return dispatch_unary(target, op, arg_a, arg_b) }
    throw TypeError(f"{unary_name(op)} needs a Series, a DataFrame or a Column")
}

#: Elementwise sum. `fill_value` stands in for a missing operand on one side.
pub fn add(left: any, right: any, fill_value: Value? = null) -> any {
    return binary(left, right, OP_ADD, fill_value)
}

#: Elementwise difference.
pub fn subtract(left: any, right: any, fill_value: Value? = null) -> any {
    return binary(left, right, OP_SUBTRACT, fill_value)
}

#: Elementwise product.
pub fn multiply(left: any, right: any, fill_value: Value? = null) -> any {
    return binary(left, right, OP_MULTIPLY, fill_value)
}

#: Elementwise quotient, always `Float64`.
#:
#: Even over two `Int64` columns: 7 / 2 is not an integer, and pandas' `/` is
#: true division. `floor_divide` is the one that stays exact.
pub fn divide(left: any, right: any, fill_value: Value? = null) -> any {
    return binary(left, right, OP_DIVIDE, fill_value)
}

#: Elementwise `//`: the quotient rounded toward negative infinity.
#:
#: Exact over integers, and a zero divisor there gives a null rather than a
#: wrong number — integer division by zero has no answer, and the device does
#: not fault on one. Over floats it is `floor(a / b)`, so a zero divisor gives
#: the infinity a float division gives.
pub fn floor_divide(left: any, right: any, fill_value: Value? = null) -> any {
    return binary(left, right, OP_FLOOR_DIVIDE, fill_value)
}

#: Elementwise `%`, with the sign of the divisor as Python's has.
pub fn modulo(left: any, right: any, fill_value: Value? = null) -> any {
    return binary(left, right, OP_MODULO, fill_value)
}

#: Elementwise `**`, always `Float64`.
#:
#: `pow` is a float function on the device, and an exact integer power leaves
#: the 48 bits two lanes hold at 6 ** 18, so promoting is the honest answer
#: rather than one that is exact for a while.
pub fn power(left: any, right: any, fill_value: Value? = null) -> any {
    return binary(left, right, OP_POWER, fill_value)
}

#: Elementwise `==`, as a `Bool` column.
#:
#: A null on either side makes the answer null, not false — this module's
#: columns are nullable, so a comparison propagates the way pandas' nullable
#: dtypes do. `fillna(false)` over the result is the numpy-backed behaviour.
#: Text compares as codes on one dictionary, so it is an integer compare.
pub fn equal(left: any, right: any, fill_value: Value? = null) -> any {
    return binary(left, right, OP_EQUAL, fill_value)
}

#: Elementwise `!=`, as a `Bool` column.
pub fn not_equal(left: any, right: any, fill_value: Value? = null) -> any {
    return binary(left, right, OP_NOT_EQUAL, fill_value)
}

#: Elementwise `<`, as a `Bool` column.
#:
#: Over text this is the strings' order, reached by putting both dictionaries
#: on one table whose codes are ranks. The sort behind that is over the
#: distinct values, not the rows.
pub fn less(left: any, right: any, fill_value: Value? = null) -> any {
    return binary(left, right, OP_LESS, fill_value)
}

#: Elementwise `<=`, as a `Bool` column.
pub fn less_equal(left: any, right: any, fill_value: Value? = null) -> any {
    return binary(left, right, OP_LESS_EQUAL, fill_value)
}

#: Elementwise `>`, as a `Bool` column.
pub fn greater(left: any, right: any, fill_value: Value? = null) -> any {
    return binary(left, right, OP_GREATER, fill_value)
}

#: Elementwise `>=`, as a `Bool` column.
pub fn greater_equal(left: any, right: any, fill_value: Value? = null) -> any {
    return binary(left, right, OP_GREATER_EQUAL, fill_value)
}

#: Elementwise conjunction over `Bool` columns.
#:
#: Three-valued, as pandas' nullable booleans are: `false and null` is false,
#: because a present false settles the answer whatever the unknown side holds.
#: Only `null and null` and `true and null` are unknown.
pub fn and_(left: any, right: any) -> any { return binary(left, right, OP_AND, null) }

#: Elementwise disjunction, three-valued: `true or null` is true.
pub fn or_(left: any, right: any) -> any { return binary(left, right, OP_OR, null) }

#: Elementwise exclusive or. Unknown on either side makes it unknown — neither
#: value settles an inequality.
pub fn xor(left: any, right: any) -> any { return binary(left, right, OP_XOR, null) }

#: Elementwise negation of a `Bool` column.
pub fn not_(target: any) -> any { return unary(target, UN_NOT, null, null) }

#: Elementwise `-x`. `Bool` becomes `Int64`, as it does under `+`.
pub fn negate(target: any) -> any { return unary(target, UN_NEGATE, null, null) }

#: Elementwise magnitude.
pub fn absolute(target: any) -> any { return unary(target, UN_ABSOLUTE, null, null) }

#: -1, 0 or 1 by the sign of each value; a null stays null.
pub fn sign(target: any) -> any { return unary(target, UN_SIGN, null, null) }

#: 10 to the `n`, for `n` of either sign.
fn power_of_ten(n: int) -> float {
    var scale = 1.0
    let steps = if n < 0 { -n } else { n }
    for _step in 0..steps { scale *= 10.0 }
    return if n < 0 { 1.0 / scale } else { scale }
}

#: The integer 10 to the `n`, for a non-negative `n`.
fn integer_power_of_ten(n: int) -> int {
    var scale = 1
    for _step in 0..n { scale *= 10 }
    return scale
}

#: Round to `decimals` places, ties to even as numpy rounds.
#:
#: An integer column is already rounded at `decimals` 0 and above; below it the
#: value is snapped to a multiple of ten in `long`, so a `1234.round(-2)` stays
#: exact rather than going through a float.
pub fn round(target: any, decimals: int = 0) -> any {
    let numeric = Value.of_float(power_of_ten(decimals))
    let exact = Value.of_int(if decimals < 0 { integer_power_of_ten(-decimals) } else { 1 })
    if target is DataFrame or target is Series or target is Column {
        return round_over(target, numeric, exact)
    }
    throw TypeError("round needs a Series, a DataFrame or a Column")
}

#: `round` has to pick its argument by the column's dtype, which is per column
#: in a frame, so it dispatches a column at a time rather than through `unary`.
fn round_over(target: any, numeric: Value, exact: Value) -> any {
    if target is DataFrame {
        var columns: list[Column] = []
        for column in target.columns { columns.push(round_column(column, numeric, exact)) }
        return target._replace(columns: columns, index: target.index.copy())
    }
    if target is Series {
        return Series(
            round_column(target.values, numeric, exact),
            index: target.index.copy(),
            name: target.name
        )
    }
    return round_column(target, numeric, exact)
}

fn round_column(column: Column, numeric: Value, exact: Value) -> Column {
    let argument = if unary_result(UN_ROUND, column.dtype).is_wide() { exact } else { numeric }
    return dispatch_unary(column, UN_ROUND, argument, null)
}

#: Elementwise round toward negative infinity.
pub fn floor(target: any) -> any { return unary(target, UN_FLOOR, null, null) }

#: Elementwise round toward positive infinity.
pub fn ceiling(target: any) -> any { return unary(target, UN_CEILING, null, null) }

#: Elementwise round toward zero.
pub fn truncate(target: any) -> any { return unary(target, UN_TRUNCATE, null, null) }

#: Elementwise `e ** x`, always `Float64`.
pub fn exp(target: any) -> any { return unary(target, UN_EXP, null, null) }

#: Elementwise natural log. A non-positive value has none, so it becomes null.
pub fn log(target: any) -> any { return unary(target, UN_LOG, null, null) }

#: Elementwise base-2 log.
pub fn log2(target: any) -> any { return unary(target, UN_LOG2, null, null) }

#: Elementwise base-10 log.
pub fn log10(target: any) -> any { return unary(target, UN_LOG10, null, null) }

#: Elementwise square root. A negative value has none, so it becomes null.
pub fn sqrt(target: any) -> any { return unary(target, UN_SQRT, null, null) }

#: Elementwise `x * x`, keeping an integer column integral.
pub fn square(target: any) -> any { return unary(target, UN_SQUARE, null, null) }

#: Elementwise `1 / x`, always `Float64`. A zero gives an infinity, as pandas'
#: division does, rather than a null.
pub fn reciprocal(target: any) -> any { return unary(target, UN_RECIPROCAL, null, null) }

#: A bound as a `Value`, so a caller may write a bare number.
fn bound_value(bound: any) -> Value? {
    if bound is null { return null }
    if bound is Value { return if bound.is_na() { null } else { bound } }
    return scalar_value(bound, Dtype.Float64)
}

#: Every value held inside `[lower, upper]`. Either bound may be left out.
#:
#: A null row stays null: clipping is about the values that are there.
pub fn clip(target: any, lower: any = null, upper: any = null) -> any {
    return unary(target, UN_CLIP, bound_value(lower), bound_value(upper))
}

#: A `Str` index over a list of names, for a result labelled by column.
fn index_over_names(names: list[str]) -> Index {
    var labels: list[str?] = []
    for name in names { labels.push(name) }
    let level = Column.from_strs(labels)
    let unnamed: list[str?] = [null]
    return Index.of([level], unnamed)
}

#: `f` over every value of a series.
#:
#: A callback costs a host pass: the column comes down, the loop runs one
#: boxed `Value` per row, and the result goes back up. Reach for it when the
#: transform has no vectorised spelling; `add`, `multiply`, `clip`, `round`
#: and the rest of this module stay on the device and are the alternative.
#:
#: ```
#: map(series, |cell| if cell.is_na() { cell } else { Value.of_float(cell.to_float() * 2.0) })
#: ```
pub fn map(series: Series, f: fn(Value) -> Value) -> Series {
    var out: list[Value] = []
    for cell in column_cells(series.values) { out.push(f(cell)) }
    return Series(
        column_of_cells(out, series.dtype()),
        index: series.index.copy(),
        name: series.name
    )
}

#: `f` over each column of a frame, or each row when `axis` is 1.
#:
#: A callback costs a host pass: every column comes down once, the loop runs on
#: boxed `Value`s, and one column goes back up. The vectorised alternative for
#: a column-at-a-time summary is `reduce.jai`, and for an elementwise one it is
#: the arithmetic here.
#:
#: pandas hands the callback a `Series`; this hands it the cells. A `Series`
#: per row would be one device allocation per row, which for a frame of any
#: size is the slowest thing this package could do.
pub fn apply(frame: DataFrame, f: fn(list[Value]) -> Value, axis: int = 0) -> Series {
    if axis != 0 and axis != 1 { throw ValueError(f"axis must be 0 or 1, not {axis}") }
    var out: list[Value] = []
    if axis == 0 {
        for column in frame.columns { out.push(f(column_cells(column))) }
        return Series(column_of_cells(out), index: index_over_names(frame.names))
    }
    var downloaded: list[list[Value]] = []
    for column in frame.columns { downloaded.push(column_cells(column)) }
    for row in 0..frame.len() {
        var cells: list[Value] = []
        for column in downloaded { cells.push(column[row]) }
        out.push(f(cells))
    }
    return Series(column_of_cells(out), index: frame.index.copy())
}

#: `f` over every cell of a frame, keeping its shape.
#:
#: A callback costs a host pass, one per column: down, loop, up. An elementwise
#: transform that this module already spells — `absolute`, `clip`, `log`, a
#: comparison — stays on the device and is the alternative.
pub fn applymap(frame: DataFrame, f: fn(Value) -> Value) -> DataFrame {
    var columns: list[Column] = []
    for column in frame.columns {
        var cells: list[Value] = []
        for cell in column_cells(column) { cells.push(f(cell)) }
        columns.push(column_of_cells(cells, column.dtype))
    }
    return frame._replace(columns: columns, index: frame.index.copy())
}

#: `f(target)`, so a chain of whole-frame steps reads top to bottom.
#:
#: The one callback here that costs no host pass: nothing is downloaded, the
#: object is handed over whole, and whatever `f` does is what it costs. It is
#: in this file because it is the same escape hatch as the others — a place for
#: what the vectorised operations cannot say.
#:
#: ```
#: pipe(frame, |f| clip(f, 0.0))
#: ```
pub fn pipe(target: any, f: fn(any) -> any) -> any { return f(target) }

#: `f` over paired values of two series, aligned on their labels first.
#:
#: A callback costs a host pass: both columns come down, the loop runs on boxed
#: `Value`s, and the result goes back up. Any pairing this module already
#: spells — `add`, `power`, a comparison — is one dispatch instead, and a
#: two-argument callback that only picks a side is `where` in `missing.jai`.
pub fn combine(a: Series, b: Series, f: fn(Value, Value) -> Value) -> Series {
    let plan = align_indexes(a.index, b.index)
    let left = aligned_column(a.values, plan.left, plan.length)
    let right = aligned_column(b.values, plan.right, plan.length)
    let left_cells = column_cells(left[0])
    let right_cells = column_cells(right[0])
    var out: list[Value] = []
    for row in 0..plan.length { out.push(f(left_cells[row], right_cells[row])) }
    if left[1] { left[0].free() }
    if right[1] { right[0].free() }
    plan.free_positions()
    return Series(
        column_of_cells(out, a.dtype()),
        index: plan.index,
        name: shared_name(a.name, b.name)
    )
}

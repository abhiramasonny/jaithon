#: The shader library every dispatch runs, and the device check that guards it.
import std.gpu as gpu
from std.gpu import Kernel

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

pub fn flag(value: bool) -> int { return if value { 1 } else { 0 } }

pub fn require_device(what: str) -> void {
    if not gpu.is_available() {
        throw RuntimeError(f"jaiframe needs a GPU to {what}")
    }
}

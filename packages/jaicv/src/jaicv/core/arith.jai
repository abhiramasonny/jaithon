#: Element-by-element arithmetic, comparison, and bit operations on `Mat`.
#:
#: Every function here follows OpenCV: the result takes the shape of its inputs,
#: an integer destination saturates, and an optional `mask` limits which pixels
#: are written.
#:
#: ```
#: let sum = add(left, right)
#: let bright = add_weighted(left, 0.7, right, 0.3, 0.0)
#: let edges = compare(gradient, 40.0, CMP_GT)
#: ```
import std.gpu as gpu
import std.math
from std.gpu import Buffer, Kernel
from .mat import Mat, f32_bits, touch
from .types import CV_8U, CV_32F, CV_MAKETYPE, CV_MAT_CN, CV_MAT_DEPTH
from .types import CMP_EQ, CMP_GE, CMP_GT, CMP_LE, CMP_LT, CMP_NE, Scalar
from .types import depth_is_integer, depth_max, depth_min, type_to_string

#: Add.
pub const OP_ADD = 0
#: Subtract the second operand from the first.
pub const OP_SUBTRACT = 1
#: Multiply.
pub const OP_MULTIPLY = 2
#: Divide, yielding zero where the divisor is zero, as OpenCV does.
pub const OP_DIVIDE = 3
#: Absolute difference.
pub const OP_ABSDIFF = 4
#: Smaller of the two.
pub const OP_MIN = 5
#: Larger of the two.
pub const OP_MAX = 6
#: Bitwise and.
pub const OP_AND = 7
#: Bitwise or.
pub const OP_OR = 8
#: Bitwise exclusive or.
pub const OP_XOR = 9
#: Raise the first operand to the second.
pub const OP_POW = 10
#: Subtract the first operand from the second.
pub const OP_RSUBTRACT = 11

#: Absolute value.
pub const UN_ABS = 0
#: Square root.
pub const UN_SQRT = 1
#: Natural exponential.
pub const UN_EXP = 2
#: Natural logarithm.
pub const UN_LOG = 3
#: Bitwise complement.
pub const UN_NOT = 4
#: Negate.
pub const UN_NEGATE = 5
#: Square.
pub const UN_SQUARE = 6
#: Reciprocal, yielding zero at zero.
pub const UN_RECIPROCAL = 7

let SOURCE = """#include <metal_stdlib>
using namespace metal;

inline uint plane_index(uint id, uint cols, uint cn, uint off, uint step) {
    uint width = cols * cn;
    uint y = id / width;
    uint rest = id - y * width;
    return off + y * step + rest;
}

inline float saturate_to(float value, float low, float high, uint integral) {
    float v = value;
    if (integral != 0u) {
        v = rint(v);
        v = clamp(v, low, high);
    }
    return v;
}

inline float apply_binary(uint op, float a, float b) {
    if (op == 0u) return a + b;
    if (op == 1u) return a - b;
    if (op == 2u) return a * b;
    if (op == 3u) return b == 0.0f ? 0.0f : a / b;
    if (op == 4u) return fabs(a - b);
    if (op == 5u) return min(a, b);
    if (op == 6u) return max(a, b);
    if (op == 7u) return float(uint(a) & uint(b));
    if (op == 8u) return float(uint(a) | uint(b));
    if (op == 9u) return float(uint(a) ^ uint(b));
    if (op == 10u) {
        /* OpenCV raises to an integral power by repeated multiplication, which
         * `pow` does not reproduce exactly: pow(3, 2) comes back 8.999998. */
        float rounded = rint(b);
        if (rounded == b && fabs(rounded) <= 32.0f) {
            float acc = 1.0f;
            float base = rounded < 0.0f ? 1.0f / a : a;
            int times = int(fabs(rounded));
            for (int i = 0; i < times; ++i) acc *= base;
            return acc;
        }
        return pow(a, b);
    }
    if (op == 11u) return b - a;
    return a;
}

inline float apply_unary(uint op, float a) {
    if (op == 0u) return fabs(a);
    if (op == 1u) return a <= 0.0f ? 0.0f : sqrt(a);
    if (op == 2u) return exp(a);
    if (op == 3u) return a <= 0.0f ? -87.3365479f : log(a);
    if (op == 4u) return float(~uint(a) & 0xFFu);
    if (op == 5u) return -a;
    if (op == 6u) return a * a;
    if (op == 7u) return a == 0.0f ? 0.0f : 1.0f / a;
    return a;
}

inline bool apply_compare(uint op, float a, float b) {
    if (op == 0u) return a == b;
    if (op == 1u) return a > b;
    if (op == 2u) return a >= b;
    if (op == 3u) return a < b;
    if (op == 4u) return a <= b;
    return a != b;
}

kernel void arith_binary(device const float* a [[buffer(0)]],
                         device const float* b [[buffer(1)]],
                         device float* dst [[buffer(2)]],
                         device const float* mask [[buffer(3)]],
                         constant uint& count [[buffer(4)]],
                         constant uint& cols [[buffer(5)]],
                         constant uint& cn [[buffer(6)]],
                         constant uint& a_off [[buffer(7)]],
                         constant uint& a_step [[buffer(8)]],
                         constant uint& b_off [[buffer(9)]],
                         constant uint& b_step [[buffer(10)]],
                         constant uint& d_off [[buffer(11)]],
                         constant uint& d_step [[buffer(12)]],
                         constant uint& m_off [[buffer(13)]],
                         constant uint& m_step [[buffer(14)]],
                         constant uint& use_mask [[buffer(15)]],
                         constant uint& op [[buffer(16)]],
                         constant uint& scale_bits [[buffer(17)]],
                         constant uint& low_bits [[buffer(18)]],
                         constant uint& high_bits [[buffer(19)]],
                         constant uint& integral [[buffer(20)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    if (use_mask != 0u) {
        uint pixel = id / cn;
        if (mask[plane_index(pixel, cols, 1u, m_off, m_step)] == 0.0f) return;
    }
    float left = a[plane_index(id, cols, cn, a_off, a_step)];
    float right = b[plane_index(id, cols, cn, b_off, b_step)];
    float value = apply_binary(op, left, right) * as_type<float>(scale_bits);
    dst[plane_index(id, cols, cn, d_off, d_step)] =
        saturate_to(value, as_type<float>(low_bits), as_type<float>(high_bits), integral);
}

kernel void arith_scalar(device const float* a [[buffer(0)]],
                         device float* dst [[buffer(1)]],
                         device const float* mask [[buffer(2)]],
                         constant uint& count [[buffer(3)]],
                         constant uint& cols [[buffer(4)]],
                         constant uint& cn [[buffer(5)]],
                         constant uint& a_off [[buffer(6)]],
                         constant uint& a_step [[buffer(7)]],
                         constant uint& d_off [[buffer(8)]],
                         constant uint& d_step [[buffer(9)]],
                         constant uint& m_off [[buffer(10)]],
                         constant uint& m_step [[buffer(11)]],
                         constant uint& use_mask [[buffer(12)]],
                         constant uint& op [[buffer(13)]],
                         constant uint& c0 [[buffer(14)]],
                         constant uint& c1 [[buffer(15)]],
                         constant uint& c2 [[buffer(16)]],
                         constant uint& c3 [[buffer(17)]],
                         constant uint& low_bits [[buffer(18)]],
                         constant uint& high_bits [[buffer(19)]],
                         constant uint& integral [[buffer(20)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint channel = id % cn;
    if (use_mask != 0u) {
        uint pixel = id / cn;
        if (mask[plane_index(pixel, cols, 1u, m_off, m_step)] == 0.0f) return;
    }
    float operand = as_type<float>(c0);
    if (channel == 1u) operand = as_type<float>(c1);
    else if (channel == 2u) operand = as_type<float>(c2);
    else if (channel == 3u) operand = as_type<float>(c3);
    float left = a[plane_index(id, cols, cn, a_off, a_step)];
    dst[plane_index(id, cols, cn, d_off, d_step)] =
        saturate_to(apply_binary(op, left, operand),
                    as_type<float>(low_bits), as_type<float>(high_bits), integral);
}

kernel void arith_unary(device const float* a [[buffer(0)]],
                        device float* dst [[buffer(1)]],
                        constant uint& count [[buffer(2)]],
                        constant uint& cols [[buffer(3)]],
                        constant uint& cn [[buffer(4)]],
                        constant uint& a_off [[buffer(5)]],
                        constant uint& a_step [[buffer(6)]],
                        constant uint& d_off [[buffer(7)]],
                        constant uint& d_step [[buffer(8)]],
                        constant uint& op [[buffer(9)]],
                        constant uint& low_bits [[buffer(10)]],
                        constant uint& high_bits [[buffer(11)]],
                        constant uint& integral [[buffer(12)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float value = a[plane_index(id, cols, cn, a_off, a_step)];
    dst[plane_index(id, cols, cn, d_off, d_step)] =
        saturate_to(apply_unary(op, value),
                    as_type<float>(low_bits), as_type<float>(high_bits), integral);
}

kernel void arith_weighted(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device float* dst [[buffer(2)]],
                           constant uint& count [[buffer(3)]],
                           constant uint& cols [[buffer(4)]],
                           constant uint& cn [[buffer(5)]],
                           constant uint& a_off [[buffer(6)]],
                           constant uint& a_step [[buffer(7)]],
                           constant uint& b_off [[buffer(8)]],
                           constant uint& b_step [[buffer(9)]],
                           constant uint& d_off [[buffer(10)]],
                           constant uint& d_step [[buffer(11)]],
                           constant uint& alpha_bits [[buffer(12)]],
                           constant uint& beta_bits [[buffer(13)]],
                           constant uint& gamma_bits [[buffer(14)]],
                           constant uint& low_bits [[buffer(15)]],
                           constant uint& high_bits [[buffer(16)]],
                           constant uint& integral [[buffer(17)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float left = a[plane_index(id, cols, cn, a_off, a_step)];
    float right = b[plane_index(id, cols, cn, b_off, b_step)];
    float value = left * as_type<float>(alpha_bits) +
                  right * as_type<float>(beta_bits) +
                  as_type<float>(gamma_bits);
    dst[plane_index(id, cols, cn, d_off, d_step)] =
        saturate_to(value, as_type<float>(low_bits), as_type<float>(high_bits), integral);
}

kernel void arith_compare(device const float* a [[buffer(0)]],
                          device const float* b [[buffer(1)]],
                          device float* dst [[buffer(2)]],
                          constant uint& count [[buffer(3)]],
                          constant uint& cols [[buffer(4)]],
                          constant uint& cn [[buffer(5)]],
                          constant uint& a_off [[buffer(6)]],
                          constant uint& a_step [[buffer(7)]],
                          constant uint& b_off [[buffer(8)]],
                          constant uint& b_step [[buffer(9)]],
                          constant uint& d_off [[buffer(10)]],
                          constant uint& d_step [[buffer(11)]],
                          constant uint& op [[buffer(12)]],
                          constant uint& scalar_rhs [[buffer(13)]],
                          constant uint& value_bits [[buffer(14)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float left = a[plane_index(id, cols, cn, a_off, a_step)];
    float right = scalar_rhs != 0u
        ? as_type<float>(value_bits)
        : b[plane_index(id, cols, cn, b_off, b_step)];
    dst[plane_index(id, cols, cn, d_off, d_step)] =
        apply_compare(op, left, right) ? 255.0f : 0.0f;
}

kernel void arith_in_range(device const float* a [[buffer(0)]],
                           device float* dst [[buffer(1)]],
                           constant uint& count [[buffer(2)]],
                           constant uint& cols [[buffer(3)]],
                           constant uint& cn [[buffer(4)]],
                           constant uint& a_off [[buffer(5)]],
                           constant uint& a_step [[buffer(6)]],
                           constant uint& d_off [[buffer(7)]],
                           constant uint& d_step [[buffer(8)]],
                           constant uint& l0 [[buffer(9)]],
                           constant uint& l1 [[buffer(10)]],
                           constant uint& l2 [[buffer(11)]],
                           constant uint& l3 [[buffer(12)]],
                           constant uint& h0 [[buffer(13)]],
                           constant uint& h1 [[buffer(14)]],
                           constant uint& h2 [[buffer(15)]],
                           constant uint& h3 [[buffer(16)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;
    bool inside = true;
    for (uint channel = 0; channel < cn; ++channel) {
        float value = a[a_off + y * a_step + x * cn + channel];
        float low = as_type<float>(l0);
        float high = as_type<float>(h0);
        if (channel == 1u) { low = as_type<float>(l1); high = as_type<float>(h1); }
        else if (channel == 2u) { low = as_type<float>(l2); high = as_type<float>(h2); }
        else if (channel == 3u) { low = as_type<float>(l3); high = as_type<float>(h3); }
        if (value < low || value > high) inside = false;
    }
    dst[d_off + y * d_step + x] = inside ? 255.0f : 0.0f;
}

kernel void arith_lut(device const float* a [[buffer(0)]],
                      device const float* table [[buffer(1)]],
                      device float* dst [[buffer(2)]],
                      constant uint& count [[buffer(3)]],
                      constant uint& cols [[buffer(4)]],
                      constant uint& cn [[buffer(5)]],
                      constant uint& a_off [[buffer(6)]],
                      constant uint& a_step [[buffer(7)]],
                      constant uint& d_off [[buffer(8)]],
                      constant uint& d_step [[buffer(9)]],
                      constant uint& table_off [[buffer(10)]],
                      constant uint& entries [[buffer(11)]],
                      constant uint& table_cn [[buffer(12)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint channel = id % cn;
    float value = a[plane_index(id, cols, cn, a_off, a_step)];
    uint slot = uint(clamp(value, 0.0f, float(entries - 1u)));
    uint lane = table_cn == 1u ? 0u : channel;
    dst[plane_index(id, cols, cn, d_off, d_step)] = table[table_off + slot * table_cn + lane];
}

kernel void arith_polar(device const float* a [[buffer(0)]],
                        device const float* b [[buffer(1)]],
                        device float* magnitude [[buffer(2)]],
                        device float* angle [[buffer(3)]],
                        constant uint& count [[buffer(4)]],
                        constant uint& cols [[buffer(5)]],
                        constant uint& a_off [[buffer(6)]],
                        constant uint& a_step [[buffer(7)]],
                        constant uint& b_off [[buffer(8)]],
                        constant uint& b_step [[buffer(9)]],
                        constant uint& m_off [[buffer(10)]],
                        constant uint& m_step [[buffer(11)]],
                        constant uint& g_off [[buffer(12)]],
                        constant uint& g_step [[buffer(13)]],
                        constant uint& degrees [[buffer(14)]],
                        constant uint& inverse [[buffer(15)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;
    float left = a[a_off + y * a_step + x];
    float right = b[b_off + y * b_step + x];
    if (inverse != 0u) {
        float radians = right * (degrees != 0u ? 0.01745329251f : 1.0f);
        magnitude[m_off + y * m_step + x] = left * cos(radians);
        angle[g_off + y * g_step + x] = left * sin(radians);
        return;
    }
    magnitude[m_off + y * m_step + x] = sqrt(left * left + right * right);
    float theta = atan2(right, left);
    if (theta < 0.0f) theta += 6.28318530718f;
    angle[g_off + y * g_step + x] = theta * (degrees != 0u ? 57.2957795131f : 1.0f);
}
"""

var _kernels: dict[str, Kernel] = {}

fn kernel_for(entry: str) -> Kernel {
    let existing = _kernels.get(entry, null)
    if existing is not null { return existing }
    let compiled = Kernel.compile(SOURCE, entry)
    _kernels[entry] = compiled
    return compiled
}

fn saturation_of(type_code: int) -> tuple[int, int, int] {
    let depth = CV_MAT_DEPTH(type_code)
    let integral = if depth_is_integer(depth) { 1 } else { 0 }
    return (f32_bits(depth_min(depth)), f32_bits(depth_max(depth)), integral)
}

fn require_match(a: Mat, b: Mat, what: str) -> void {
    if a.rows != b.rows or a.cols != b.cols {
        throw ValueError(f"{what} needs matching shapes, got {a} and {b}")
    }
    if a.channels() != b.channels() {
        throw ValueError(f"{what} needs matching channel counts, got {a} and {b}")
    }
}

fn destination_for(src: Mat, dst: Mat?, type_code: int) -> Mat {
    let existing = dst
    if existing is null { return Mat(src.rows, src.cols, type_code) }
    if existing.rows != src.rows or existing.cols != src.cols {
        throw ValueError(f"destination {existing} does not match {src}")
    }
    return existing
}

#: Run one of the `OP_` codes over two matrices.
pub fn binary_op(
    a: Mat,
    b: Mat,
    op: int,
    dst: Mat? = null,
    mask: Mat? = null,
    dtype: int = -1,
    scale: float = 1.0
) -> Mat {
    require_match(a, b, "a binary operation")
    a.flush()
    b.flush()
    let type_code = if dtype >= 0 { CV_MAKETYPE(dtype, a.channels()) } else { a.flags }
    let out = destination_for(a, dst, type_code)
    let count = a.rows * a.cols * a.channels()
    if count == 0 { return out }
    let (low, high, integral) = saturation_of(out.flags)
    let use_mask = if mask is null { 0 } else { 1 }
    let mask_mat = if mask is null { out } else { mask }
    if mask is not null { mask.flush() }
    kernel_for("arith_binary").dispatch_async(
        [a.storage, b.storage, out.storage, mask_mat.storage],
        count,
        [
            count,
            a.cols,
            a.channels(),
            a.offset,
            a.step_,
            b.offset,
            b.step_,
            out.offset,
            out.step_,
            mask_mat.offset,
            mask_mat.step_,
            use_mask,
            op,
            f32_bits(scale),
            low,
            high,
            integral,
        ]
    )
    out.invalidate()
    return out
}

#: Run one of the `OP_` codes over a matrix and a per-channel constant.
pub fn scalar_op(
    a: Mat,
    value: Scalar,
    op: int,
    dst: Mat? = null,
    mask: Mat? = null,
    dtype: int = -1
) -> Mat {
    a.flush()
    let type_code = if dtype >= 0 { CV_MAKETYPE(dtype, a.channels()) } else { a.flags }
    let out = destination_for(a, dst, type_code)
    let count = a.rows * a.cols * a.channels()
    if count == 0 { return out }
    let (low, high, integral) = saturation_of(out.flags)
    let use_mask = if mask is null { 0 } else { 1 }
    let mask_mat = if mask is null { out } else { mask }
    if mask is not null { mask.flush() }
    kernel_for("arith_scalar").dispatch_async(
        [a.storage, out.storage, mask_mat.storage],
        count,
        [
            count,
            a.cols,
            a.channels(),
            a.offset,
            a.step_,
            out.offset,
            out.step_,
            mask_mat.offset,
            mask_mat.step_,
            use_mask,
            op,
            f32_bits(value[0]),
            f32_bits(value[1]),
            f32_bits(value[2]),
            f32_bits(value[3]),
            low,
            high,
            integral,
        ]
    )
    out.invalidate()
    return out
}

#: Run one of the `UN_` codes over a matrix.
pub fn unary_op(a: Mat, op: int, dst: Mat? = null, dtype: int = -1) -> Mat {
    a.flush()
    let type_code = if dtype >= 0 { CV_MAKETYPE(dtype, a.channels()) } else { a.flags }
    let out = destination_for(a, dst, type_code)
    let count = a.rows * a.cols * a.channels()
    if count == 0 { return out }
    let (low, high, integral) = saturation_of(out.flags)
    kernel_for("arith_unary").dispatch_async(
        [a.storage, out.storage],
        count,
        [
            count,
            a.cols,
            a.channels(),
            a.offset,
            a.step_,
            out.offset,
            out.step_,
            op,
            low,
            high,
            integral,
        ]
    )
    out.invalidate()
    return out
}

#: Sum of two matrices.
pub fn add(a: Mat, b: Mat, dst: Mat? = null, mask: Mat? = null, dtype: int = -1) -> Mat {
    return binary_op(a, b, OP_ADD, dst: dst, mask: mask, dtype: dtype)
}

#: Difference of two matrices.
pub fn subtract(a: Mat, b: Mat, dst: Mat? = null, mask: Mat? = null, dtype: int = -1) -> Mat {
    return binary_op(a, b, OP_SUBTRACT, dst: dst, mask: mask, dtype: dtype)
}

#: Per-element product, scaled by `scale`.
pub fn multiply(a: Mat, b: Mat, scale: float = 1.0, dst: Mat? = null, dtype: int = -1) -> Mat {
    return binary_op(a, b, OP_MULTIPLY, dst: dst, dtype: dtype, scale: scale)
}

#: Per-element quotient, scaled by `scale`. Dividing by zero yields zero.
pub fn divide(a: Mat, b: Mat, scale: float = 1.0, dst: Mat? = null, dtype: int = -1) -> Mat {
    return binary_op(a, b, OP_DIVIDE, dst: dst, dtype: dtype, scale: scale)
}

#: Absolute difference of two matrices.
pub fn absdiff(a: Mat, b: Mat, dst: Mat? = null) -> Mat {
    return binary_op(a, b, OP_ABSDIFF, dst: dst)
}

#: Per-element minimum.
pub fn min_mat(a: Mat, b: Mat, dst: Mat? = null) -> Mat {
    return binary_op(a, b, OP_MIN, dst: dst)
}

#: Per-element maximum.
pub fn max_mat(a: Mat, b: Mat, dst: Mat? = null) -> Mat {
    return binary_op(a, b, OP_MAX, dst: dst)
}

#: `a * alpha + b * beta + gamma`, saturated into the destination depth.
pub fn add_weighted(
    a: Mat,
    alpha: float,
    b: Mat,
    beta: float,
    gamma: float,
    dst: Mat? = null,
    dtype: int = -1
) -> Mat {
    require_match(a, b, "add_weighted")
    a.flush()
    b.flush()
    let type_code = if dtype >= 0 { CV_MAKETYPE(dtype, a.channels()) } else { a.flags }
    let out = destination_for(a, dst, type_code)
    let count = a.rows * a.cols * a.channels()
    if count == 0 { return out }
    let (low, high, integral) = saturation_of(out.flags)
    kernel_for("arith_weighted").dispatch_async(
        [a.storage, b.storage, out.storage],
        count,
        [
            count,
            a.cols,
            a.channels(),
            a.offset,
            a.step_,
            b.offset,
            b.step_,
            out.offset,
            out.step_,
            f32_bits(alpha),
            f32_bits(beta),
            f32_bits(gamma),
            low,
            high,
            integral,
        ]
    )
    out.invalidate()
    return out
}

#: `src * alpha + other`, the OpenCV `scaleAdd`.
pub fn scale_add(src: Mat, alpha: float, other: Mat, dst: Mat? = null) -> Mat {
    return add_weighted(src, alpha, other, 1.0, 0.0, dst: dst)
}

#: `src * alpha + beta`, saturated. `convertScaleAbs` also takes the modulus.
pub fn convert_scale_abs(src: Mat, alpha: float = 1.0, beta: float = 0.0) -> Mat {
    let scaled = src.convert_to(CV_MAKETYPE(CV_32F, src.channels()), alpha, beta)
    let magnitude = unary_op(scaled, UN_ABS)
    return magnitude.convert_to(CV_MAKETYPE(CV_8U, src.channels()))
}

#: Bitwise and of two matrices, or of a matrix and a constant.
pub fn bitwise_and(a: Mat, b: Mat, dst: Mat? = null, mask: Mat? = null) -> Mat {
    return binary_op(a, b, OP_AND, dst: dst, mask: mask)
}

#: Bitwise or.
pub fn bitwise_or(a: Mat, b: Mat, dst: Mat? = null, mask: Mat? = null) -> Mat {
    return binary_op(a, b, OP_OR, dst: dst, mask: mask)
}

#: Bitwise exclusive or.
pub fn bitwise_xor(a: Mat, b: Mat, dst: Mat? = null, mask: Mat? = null) -> Mat {
    return binary_op(a, b, OP_XOR, dst: dst, mask: mask)
}

#: Bitwise complement, over eight bits.
pub fn bitwise_not(a: Mat, dst: Mat? = null) -> Mat { return unary_op(a, UN_NOT, dst: dst) }

#: Absolute value.
pub fn abs_mat(a: Mat, dst: Mat? = null) -> Mat { return unary_op(a, UN_ABS, dst: dst) }

#: Element-wise square root.
pub fn sqrt_mat(a: Mat, dst: Mat? = null) -> Mat { return unary_op(a, UN_SQRT, dst: dst) }

#: Element-wise natural exponential.
pub fn exp_mat(a: Mat, dst: Mat? = null) -> Mat { return unary_op(a, UN_EXP, dst: dst) }

#: Element-wise natural logarithm.
pub fn log_mat(a: Mat, dst: Mat? = null) -> Mat { return unary_op(a, UN_LOG, dst: dst) }

#: Raise every element to `power`.
pub fn pow_mat(a: Mat, power: float, dst: Mat? = null) -> Mat {
    return scalar_op(a, Scalar.all(power), OP_POW, dst: dst)
}

#: Compare two matrices, or a matrix against a constant, with a `CMP_` code.
#: The result is `CV_8U`, 255 where the test holds.
pub fn compare(a: Mat, b: Mat, op: int, dst: Mat? = null) -> Mat {
    require_match(a, b, "compare")
    return _compare(a, b, 0.0, false, op, dst)
}

#: Compare a matrix against a constant.
pub fn compare_scalar(a: Mat, value: float, op: int, dst: Mat? = null) -> Mat {
    return _compare(a, a, value, true, op, dst)
}

fn _compare(a: Mat, b: Mat, value: float, scalar_rhs: bool, op: int, dst: Mat?) -> Mat {
    a.flush()
    b.flush()
    let out = destination_for(a, dst, CV_MAKETYPE(CV_8U, a.channels()))
    let count = a.rows * a.cols * a.channels()
    if count == 0 { return out }
    kernel_for("arith_compare").dispatch_async(
        [a.storage, b.storage, out.storage],
        count,
        [
            count,
            a.cols,
            a.channels(),
            a.offset,
            a.step_,
            b.offset,
            b.step_,
            out.offset,
            out.step_,
            op,
            if scalar_rhs { 1 } else { 0 },
            f32_bits(value),
        ]
    )
    out.invalidate()
    return out
}

#: A single-channel mask of the pixels whose every channel lies within bounds.
pub fn in_range(src: Mat, lower: Scalar, upper: Scalar, dst: Mat? = null) -> Mat {
    src.flush()
    let existing = dst
    let out = if existing is null {
        Mat(src.rows, src.cols, CV_MAKETYPE(CV_8U, 1))
    } else {
        existing
    }
    let count = src.rows * src.cols
    if count == 0 { return out }
    kernel_for("arith_in_range").dispatch_async(
        [src.storage, out.storage],
        count,
        [
            count,
            src.cols,
            src.channels(),
            src.offset,
            src.step_,
            out.offset,
            out.step_,
            f32_bits(lower[0]),
            f32_bits(lower[1]),
            f32_bits(lower[2]),
            f32_bits(lower[3]),
            f32_bits(upper[0]),
            f32_bits(upper[1]),
            f32_bits(upper[2]),
            f32_bits(upper[3]),
        ]
    )
    out.invalidate()
    return out
}

#: Map every element through a lookup table.
#:
#: The table is a 1 x n matrix, single-channel to map every channel the same
#: way or matching the source's channel count to map each one separately.
pub fn lut(src: Mat, table: Mat, dst: Mat? = null) -> Mat {
    src.flush()
    table.flush()
    let entries = table.rows * table.cols
    if entries <= 0 { throw ValueError("a lookup table needs at least one entry") }
    let table_cn = table.channels()
    if table_cn != 1 and table_cn != src.channels() {
        throw ValueError("a lookup table has one channel or as many as the source")
    }
    let out = destination_for(src, dst, CV_MAKETYPE(table.depth(), src.channels()))
    let count = src.rows * src.cols * src.channels()
    if count == 0 { return out }
    kernel_for("arith_lut").dispatch_async(
        [src.storage, table.storage, out.storage],
        count,
        [
            count,
            src.cols,
            src.channels(),
            src.offset,
            src.step_,
            out.offset,
            out.step_,
            table.offset,
            entries,
            table_cn,
        ]
    )
    out.invalidate()
    return out
}

#: Length of each `(x, y)` vector.
pub fn magnitude(x: Mat, y: Mat) -> Mat {
    let (mag, _angle) = cart_to_polar(x, y)
    return mag
}

#: Direction of each `(x, y)` vector.
pub fn phase(x: Mat, y: Mat, angle_in_degrees: bool = false) -> Mat {
    let (_mag, angle) = cart_to_polar(x, y, angle_in_degrees: angle_in_degrees)
    return angle
}

#: Length and direction of each `(x, y)` vector.
pub fn cart_to_polar(x: Mat, y: Mat, angle_in_degrees: bool = false) -> tuple[Mat, Mat] {
    return _polar(x, y, angle_in_degrees, false)
}

#: `(x, y)` from each length and direction.
pub fn polar_to_cart(
    magnitude_in: Mat,
    angle: Mat,
    angle_in_degrees: bool = false
) -> tuple[Mat, Mat] {
    return _polar(magnitude_in, angle, angle_in_degrees, true)
}

fn _polar(a: Mat, b: Mat, degrees: bool, inverse: bool) -> tuple[Mat, Mat] {
    require_match(a, b, "a polar conversion")
    if a.channels() != 1 { throw ValueError("a polar conversion needs single-channel input") }
    a.flush()
    b.flush()
    let first = Mat(a.rows, a.cols, CV_MAKETYPE(CV_32F, 1))
    let second = Mat(a.rows, a.cols, CV_MAKETYPE(CV_32F, 1))
    let count = a.rows * a.cols
    if count == 0 { return (first, second) }
    kernel_for("arith_polar").dispatch_async(
        [a.storage, b.storage, first.storage, second.storage],
        count,
        [
            count,
            a.cols,
            a.offset,
            a.step_,
            b.offset,
            b.step_,
            first.offset,
            first.step_,
            second.offset,
            second.step_,
            if degrees { 1 } else { 0 },
            if inverse { 1 } else { 0 },
        ]
    )
    first.invalidate()
    second.invalidate()
    return (first, second)
}

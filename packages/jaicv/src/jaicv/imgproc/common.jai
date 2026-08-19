#: Shader fragments every `imgproc` kernel shares: border handling, sampling,
#: and rounding into a destination depth.
#:
#: Each module pastes `PRELUDE` at the top of its own Metal source, so border
#: behaviour is defined once and every filter agrees with every warp.
from ..core.types import CV_MAT_DEPTH, depth_is_integer, depth_max, depth_min
from ..core.mat import Mat, f32_bits

#: Metal helpers shared by the filtering, warping, and morphology kernels.
pub let PRELUDE = """#include <metal_stdlib>
using namespace metal;

/* OpenCV's border codes, so a caller passes the same number to both. */
constant uint kBorderConstant = 0u;
constant uint kBorderReplicate = 1u;
constant uint kBorderReflect = 2u;
constant uint kBorderWrap = 3u;
constant uint kBorderReflect101 = 4u;

/* Fold an out-of-range coordinate back inside, or return -1 to mean "this
 * sample is the constant border value". */
inline int border_index(int position, int length, uint mode) {
    if (length <= 0) return -1;
    if (position >= 0 && position < length) return position;
    if (mode == kBorderConstant) return -1;
    if (mode == kBorderReplicate) return clamp(position, 0, length - 1);
    if (mode == kBorderWrap) {
        int wrapped = position % length;
        return wrapped < 0 ? wrapped + length : wrapped;
    }
    if (mode == kBorderReflect) {
        int p = position;
        while (p < 0 || p >= length) {
            if (p < 0) p = -p - 1;
            if (p >= length) p = 2 * length - p - 1;
        }
        return p;
    }
    /* BORDER_REFLECT_101 mirrors without repeating the edge pixel. */
    if (length == 1) return 0;
    int p = position;
    while (p < 0 || p >= length) {
        if (p < 0) p = -p;
        if (p >= length) p = 2 * length - p - 2;
    }
    return p;
}

inline float fetch(device const float* src, int y, int x, uint channel,
                   uint rows, uint cols, uint cn, uint off, uint step,
                   uint border, float constant_value) {
    int yy = border_index(y, int(rows), border);
    int xx = border_index(x, int(cols), border);
    if (yy < 0 || xx < 0) return constant_value;
    return src[off + uint(yy) * step + uint(xx) * cn + channel];
}

inline float saturate_to(float value, float low, float high, uint integral) {
    float v = value;
    if (integral != 0u) {
        v = rint(v);
        v = clamp(v, low, high);
    }
    return v;
}

inline void put(device float* dst, uint index, float value,
                float low, float high, uint integral) {
    dst[index] = saturate_to(value, low, high, integral);
}

/* Bilinear sample at a fractional position, with the same border rule as an
 * integer fetch. */
inline float sample_linear(device const float* src, float y, float x, uint channel,
                           uint rows, uint cols, uint cn, uint off, uint step,
                           uint border, float constant_value) {
    int y0 = int(floor(y));
    int x0 = int(floor(x));
    float fy = y - float(y0);
    float fx = x - float(x0);
    float v00 = fetch(src, y0, x0, channel, rows, cols, cn, off, step, border, constant_value);
    float v01 = fetch(src, y0, x0 + 1, channel, rows, cols, cn, off, step, border, constant_value);
    float v10 = fetch(src, y0 + 1, x0, channel, rows, cols, cn, off, step, border, constant_value);
    float v11 = fetch(src, y0 + 1, x0 + 1, channel, rows, cols, cn, off, step,
                      border, constant_value);
    float top = v00 + (v01 - v00) * fx;
    float bottom = v10 + (v11 - v10) * fx;
    return top + (bottom - top) * fy;
}

inline float sample_nearest(device const float* src, float y, float x, uint channel,
                            uint rows, uint cols, uint cn, uint off, uint step,
                            uint border, float constant_value) {
    return fetch(src, int(floor(y + 0.5f)), int(floor(x + 0.5f)), channel,
                 rows, cols, cn, off, step, border, constant_value);
}

/* The Catmull-Rom-like kernel OpenCV uses for INTER_CUBIC, with a = -0.75. */
inline float cubic_weight(float t) {
    const float a = -0.75f;
    float x = fabs(t);
    if (x <= 1.0f) return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
    if (x < 2.0f) return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
    return 0.0f;
}

inline float sample_cubic(device const float* src, float y, float x, uint channel,
                          uint rows, uint cols, uint cn, uint off, uint step,
                          uint border, float constant_value) {
    int y0 = int(floor(y));
    int x0 = int(floor(x));
    float fy = y - float(y0);
    float fx = x - float(x0);
    float total = 0.0f;
    for (int dy = -1; dy <= 2; ++dy) {
        float wy = cubic_weight(float(dy) - fy);
        if (wy == 0.0f) continue;
        for (int dx = -1; dx <= 2; ++dx) {
            float wx = cubic_weight(float(dx) - fx);
            if (wx == 0.0f) continue;
            total += wy * wx * fetch(src, y0 + dy, x0 + dx, channel, rows, cols, cn,
                                     off, step, border, constant_value);
        }
    }
    return total;
}

inline float sample_lanczos(device const float* src, float y, float x, uint channel,
                            uint rows, uint cols, uint cn, uint off, uint step,
                            uint border, float constant_value) {
    int y0 = int(floor(y));
    int x0 = int(floor(x));
    float fy = y - float(y0);
    float fx = x - float(x0);
    float total = 0.0f;
    float weight_total = 0.0f;
    for (int dy = -3; dy <= 4; ++dy) {
        float ty = float(dy) - fy;
        float wy = fabs(ty) < 1e-6f
            ? 1.0f
            : (fabs(ty) >= 4.0f ? 0.0f
               : 4.0f * sin(M_PI_F * ty) * sin(M_PI_F * ty / 4.0f) /
                 (M_PI_F * M_PI_F * ty * ty));
        if (wy == 0.0f) continue;
        for (int dx = -3; dx <= 4; ++dx) {
            float tx = float(dx) - fx;
            float wx = fabs(tx) < 1e-6f
                ? 1.0f
                : (fabs(tx) >= 4.0f ? 0.0f
                   : 4.0f * sin(M_PI_F * tx) * sin(M_PI_F * tx / 4.0f) /
                     (M_PI_F * M_PI_F * tx * tx));
            if (wx == 0.0f) continue;
            total += wy * wx * fetch(src, y0 + dy, x0 + dx, channel, rows, cols, cn,
                                     off, step, border, constant_value);
            weight_total += wy * wx;
        }
    }
    return weight_total == 0.0f ? 0.0f : total / weight_total;
}

inline float sample_at(device const float* src, float y, float x, uint channel,
                       uint rows, uint cols, uint cn, uint off, uint step,
                       uint border, float constant_value, uint interpolation) {
    if (interpolation == 0u) {
        return sample_nearest(src, y, x, channel, rows, cols, cn, off, step,
                              border, constant_value);
    }
    if (interpolation == 2u) {
        return sample_cubic(src, y, x, channel, rows, cols, cn, off, step,
                            border, constant_value);
    }
    if (interpolation == 4u) {
        return sample_lanczos(src, y, x, channel, rows, cols, cn, off, step,
                              border, constant_value);
    }
    return sample_linear(src, y, x, channel, rows, cols, cn, off, step,
                         border, constant_value);
}
"""

#: The three scalars every kernel needs to round into a destination depth:
#: lowest value, highest value, and whether rounding applies at all.
pub fn saturation_args(type_code: int) -> list[int] {
    let depth = CV_MAT_DEPTH(type_code)
    return [
        f32_bits(depth_min(depth)),
        f32_bits(depth_max(depth)),
        if depth_is_integer(depth) { 1 } else { 0 },
    ]
}

#: Append `extra` to `base`, the way each module builds its dispatch scalars.
pub fn joined(base: list[int], extra: list[int]) -> list[int] {
    var out = [value for value in base]
    for value in extra { out.push(value) }
    return out
}

#: The `[rows, cols, channels, offset, step]` a kernel needs to address `mat`.
pub fn layout_of(mat: Mat) -> list[int] {
    return [mat.rows, mat.cols, mat.channels(), mat.offset, mat.step_]
}

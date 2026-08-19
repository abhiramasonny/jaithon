#: `cvt_color` — the colour-space conversions, with OpenCV's code numbers.
#:
#: Every conversion assumes the source range its depth implies: `CV_8U` inputs
#: run 0..255, `CV_32F` inputs run 0..1. HSV hue is 0..179 for `CV_8U` and
#: 0..359 for `CV_32F`, exactly as OpenCV packs it.
#:
#: ```
#: let gray = cvt_color(frame, COLOR_BGR2GRAY)
#: let hsv = cvt_color(frame, COLOR_BGR2HSV)
#: ```
from std.gpu import Kernel
from ..core.mat import Mat, f32_bits
from ..core.types import CV_8U, CV_32F, CV_MAKETYPE, depth_is_integer, depth_max, depth_min

#: Blue-green-red to blue-green-red-alpha.
pub const COLOR_BGR2BGRA = 0
#: Red-green-blue to red-green-blue-alpha.
pub const COLOR_RGB2RGBA = 0
#: Drop the alpha channel.
pub const COLOR_BGRA2BGR = 1
#: Drop the alpha channel.
pub const COLOR_RGBA2RGB = 1
#: Swap the blue and red channels and add alpha.
pub const COLOR_BGR2RGBA = 2
#: Swap the blue and red channels and add alpha.
pub const COLOR_RGB2BGRA = 2
#: Drop alpha and swap blue with red.
pub const COLOR_RGBA2BGR = 3
#: Drop alpha and swap blue with red.
pub const COLOR_BGRA2RGB = 3
#: Swap the blue and red channels.
pub const COLOR_BGR2RGB = 4
#: Swap the blue and red channels.
pub const COLOR_RGB2BGR = 4
#: Swap the blue and red channels, keeping alpha.
pub const COLOR_BGRA2RGBA = 5
#: Swap the blue and red channels, keeping alpha.
pub const COLOR_RGBA2BGRA = 5
#: Blue-green-red to single-channel luma.
pub const COLOR_BGR2GRAY = 6
#: Red-green-blue to single-channel luma.
pub const COLOR_RGB2GRAY = 7
#: Luma to blue-green-red.
pub const COLOR_GRAY2BGR = 8
#: Luma to red-green-blue.
pub const COLOR_GRAY2RGB = 8
#: Luma to blue-green-red-alpha.
pub const COLOR_GRAY2BGRA = 9
#: Luma to red-green-blue-alpha.
pub const COLOR_GRAY2RGBA = 9
#: Blue-green-red-alpha to luma.
pub const COLOR_BGRA2GRAY = 10
#: Red-green-blue-alpha to luma.
pub const COLOR_RGBA2GRAY = 11
#: Blue-green-red to CIE XYZ.
pub const COLOR_BGR2XYZ = 32
#: Red-green-blue to CIE XYZ.
pub const COLOR_RGB2XYZ = 33
#: CIE XYZ to blue-green-red.
pub const COLOR_XYZ2BGR = 34
#: CIE XYZ to red-green-blue.
pub const COLOR_XYZ2RGB = 35
#: Blue-green-red to luma and chroma, the JPEG ordering.
pub const COLOR_BGR2YCrCb = 36
#: Red-green-blue to luma and chroma.
pub const COLOR_RGB2YCrCb = 37
#: Luma and chroma to blue-green-red.
pub const COLOR_YCrCb2BGR = 38
#: Luma and chroma to red-green-blue.
pub const COLOR_YCrCb2RGB = 39
#: Blue-green-red to hue, saturation, value.
pub const COLOR_BGR2HSV = 40
#: Red-green-blue to hue, saturation, value.
pub const COLOR_RGB2HSV = 41
#: Blue-green-red to CIE Lab.
pub const COLOR_BGR2Lab = 44
#: Red-green-blue to CIE Lab.
pub const COLOR_RGB2Lab = 45
#: Blue-green-red to hue, lightness, saturation.
pub const COLOR_BGR2HLS = 52
#: Red-green-blue to hue, lightness, saturation.
pub const COLOR_RGB2HLS = 53
#: Hue, saturation, value to blue-green-red.
pub const COLOR_HSV2BGR = 54
#: Hue, saturation, value to red-green-blue.
pub const COLOR_HSV2RGB = 55
#: CIE Lab to blue-green-red.
pub const COLOR_Lab2BGR = 56
#: CIE Lab to red-green-blue.
pub const COLOR_Lab2RGB = 57
#: Hue, lightness, saturation to blue-green-red.
pub const COLOR_HLS2BGR = 60
#: Hue, lightness, saturation to red-green-blue.
pub const COLOR_HLS2RGB = 61
#: Blue-green-red to luma and colour difference.
pub const COLOR_BGR2YUV = 82
#: Red-green-blue to luma and colour difference.
pub const COLOR_RGB2YUV = 83
#: Luma and colour difference to blue-green-red.
pub const COLOR_YUV2BGR = 84
#: Luma and colour difference to red-green-blue.
pub const COLOR_YUV2RGB = 85
#: Planar 4:2:0 with interleaved chroma to blue-green-red, the layout a camera
#: hands over.
pub const COLOR_YUV2BGR_NV12 = 91
#: Planar 4:2:0 with interleaved chroma to red-green-blue.
pub const COLOR_YUV2RGB_NV12 = 90
#: Packed 4:2:2 to blue-green-red.
pub const COLOR_YUV2BGR_YUY2 = 108
#: Packed 4:2:2 to red-green-blue.
pub const COLOR_YUV2RGB_YUY2 = 109

let SOURCE = """#include <metal_stdlib>
using namespace metal;

constant float kBlueWeight = 0.114f;
constant float kGreenWeight = 0.587f;
constant float kRedWeight = 0.299f;

inline float3 read_bgr(device const float* src, uint base, uint cn, uint swap_rb) {
    float b = src[base];
    float g = src[base + 1u];
    float r = cn >= 3u ? src[base + 2u] : src[base];
    if (swap_rb != 0u) { float t = b; b = r; r = t; }
    return float3(b, g, r);
}

/* Every store rounds and clamps into the destination depth, so an 8-bit result
 * comes back as whole numbers in 0..255 rather than 254.99998. */
constant float gLow = 0.0f;
constant float gHigh = 255.0f;

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

inline void write_bgr(device float* dst, uint base, uint cn, uint swap_rb, float3 v, float alpha,
                      float low, float high, uint integral) {
    float b = v.x;
    float g = v.y;
    float r = v.z;
    if (swap_rb != 0u) { float t = b; b = r; r = t; }
    put(dst, base, b, low, high, integral);
    if (cn >= 2u) put(dst, base + 1u, g, low, high, integral);
    if (cn >= 3u) put(dst, base + 2u, r, low, high, integral);
    if (cn >= 4u) put(dst, base + 3u, alpha, low, high, integral);
}

/* CIE Lab needs a linear-light D65 tristimulus, so the transfer curve comes off
 * before the matrix and goes back on after. */
inline float srgb_to_linear(float c) {
    return c <= 0.04045f ? c / 12.92f : pow((c + 0.055f) / 1.055f, 2.4f);
}

inline float linear_to_srgb(float c) {
    return c <= 0.0031308f ? c * 12.92f : 1.055f * pow(c, 1.0f / 2.4f) - 0.055f;
}

inline float lab_f(float t) {
    return t > 0.008856451679f ? pow(t, 1.0f / 3.0f) : (7.787037037f * t + 16.0f / 116.0f);
}

inline float lab_f_inv(float t) {
    float cube = t * t * t;
    return cube > 0.008856451679f ? cube : (t - 16.0f / 116.0f) / 7.787037037f;
}

inline float3 rgb_to_hsv(float3 bgr, float hue_scale) {
    float b = bgr.x, g = bgr.y, r = bgr.z;
    float high = max(r, max(g, b));
    float low = min(r, min(g, b));
    float span = high - low;
    float hue = 0.0f;
    if (span > 0.0f) {
        if (high == r) hue = 60.0f * fmod((g - b) / span, 6.0f);
        else if (high == g) hue = 60.0f * ((b - r) / span + 2.0f);
        else hue = 60.0f * ((r - g) / span + 4.0f);
    }
    if (hue < 0.0f) hue += 360.0f;
    if (hue >= 360.0f) hue -= 360.0f;
    float saturation = high <= 0.0f ? 0.0f : span / high;
    return float3(hue * hue_scale, saturation, high);
}

inline float3 hsv_to_rgb(float3 hsv, float hue_scale) {
    float hue = hsv.x / hue_scale;
    float saturation = hsv.y;
    float value = hsv.z;
    float chroma = value * saturation;
    float sector = fmod(hue / 60.0f, 6.0f);
    if (sector < 0.0f) sector += 6.0f;
    float second = chroma * (1.0f - fabs(fmod(sector, 2.0f) - 1.0f));
    float r = 0.0f, g = 0.0f, b = 0.0f;
    if (sector < 1.0f) { r = chroma; g = second; }
    else if (sector < 2.0f) { r = second; g = chroma; }
    else if (sector < 3.0f) { g = chroma; b = second; }
    else if (sector < 4.0f) { g = second; b = chroma; }
    else if (sector < 5.0f) { r = second; b = chroma; }
    else { r = chroma; b = second; }
    float lift = value - chroma;
    return float3(b + lift, g + lift, r + lift);
}

inline float3 rgb_to_hls(float3 bgr, float hue_scale) {
    float b = bgr.x, g = bgr.y, r = bgr.z;
    float high = max(r, max(g, b));
    float low = min(r, min(g, b));
    float span = high - low;
    float lightness = (high + low) * 0.5f;
    float hue = 0.0f;
    float saturation = 0.0f;
    if (span > 0.0f) {
        saturation = lightness < 0.5f ? span / (high + low) : span / (2.0f - high - low);
        if (high == r) hue = 60.0f * fmod((g - b) / span, 6.0f);
        else if (high == g) hue = 60.0f * ((b - r) / span + 2.0f);
        else hue = 60.0f * ((r - g) / span + 4.0f);
        if (hue < 0.0f) hue += 360.0f;
        if (hue >= 360.0f) hue -= 360.0f;
    }
    return float3(hue * hue_scale, lightness, saturation);
}

inline float hue_channel(float p, float q, float t) {
    float h = t;
    if (h < 0.0f) h += 1.0f;
    if (h > 1.0f) h -= 1.0f;
    if (h < 1.0f / 6.0f) return p + (q - p) * 6.0f * h;
    if (h < 1.0f / 2.0f) return q;
    if (h < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - h) * 6.0f;
    return p;
}

inline float3 hls_to_rgb(float3 hls, float hue_scale) {
    float hue = hls.x / hue_scale / 360.0f;
    float lightness = hls.y;
    float saturation = hls.z;
    if (saturation <= 0.0f) return float3(lightness, lightness, lightness);
    float q = lightness < 0.5f
        ? lightness * (1.0f + saturation)
        : lightness + saturation - lightness * saturation;
    float p = 2.0f * lightness - q;
    float r = hue_channel(p, q, hue + 1.0f / 3.0f);
    float g = hue_channel(p, q, hue);
    float b = hue_channel(p, q, hue - 1.0f / 3.0f);
    return float3(b, g, r);
}

kernel void cvt_color(device const float* src [[buffer(0)]],
                      device float* dst [[buffer(1)]],
                      constant uint& count [[buffer(2)]],
                      constant uint& cols [[buffer(3)]],
                      constant uint& src_cn [[buffer(4)]],
                      constant uint& dst_cn [[buffer(5)]],
                      constant uint& src_off [[buffer(6)]],
                      constant uint& src_step [[buffer(7)]],
                      constant uint& dst_off [[buffer(8)]],
                      constant uint& dst_step [[buffer(9)]],
                      constant uint& code [[buffer(10)]],
                      constant uint& swap_rb [[buffer(11)]],
                      constant uint& scale_bits [[buffer(12)]],
                      constant uint& hue_bits [[buffer(13)]],
                      constant uint& alpha_bits [[buffer(14)]],
                      constant uint& low_bits [[buffer(15)]],
                      constant uint& high_bits [[buffer(16)]],
                      constant uint& integral [[buffer(17)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;
    uint sbase = src_off + y * src_step + x * src_cn;
    uint dbase = dst_off + y * dst_step + x * dst_cn;
    float scale = as_type<float>(scale_bits);
    float hue_scale = as_type<float>(hue_bits);
    float alpha = as_type<float>(alpha_bits);
    float low = as_type<float>(low_bits);
    float high = as_type<float>(high_bits);

    if (code == 0u) {                            /* channel shuffles */
        float3 v = read_bgr(src, sbase, src_cn, swap_rb);
        float a = src_cn >= 4u ? src[sbase + 3u] : alpha;
        write_bgr(dst, dbase, dst_cn, 0u, v, a, low, high, integral);
        return;
    }
    if (code == 1u) {                            /* to grey */
        float3 v = read_bgr(src, sbase, src_cn, swap_rb);
        put(dst, dbase, kBlueWeight * v.x + kGreenWeight * v.y + kRedWeight * v.z, low, high, integral);
        return;
    }
    if (code == 2u) {                            /* from grey */
        float g = src[sbase];
        put(dst, dbase, g, low, high, integral);
        if (dst_cn >= 2u) put(dst, dbase + 1u, g, low, high, integral);
        if (dst_cn >= 3u) put(dst, dbase + 2u, g, low, high, integral);
        if (dst_cn >= 4u) put(dst, dbase + 3u, alpha, low, high, integral);
        return;
    }

    float3 v = read_bgr(src, sbase, src_cn, swap_rb);
    float b = v.x * scale, g = v.y * scale, r = v.z * scale;

    if (code == 3u) {                            /* to XYZ */
        float X = 0.412453f * r + 0.357580f * g + 0.180423f * b;
        float Y = 0.212671f * r + 0.715160f * g + 0.072169f * b;
        float Z = 0.019334f * r + 0.119193f * g + 0.950227f * b;
        put(dst, dbase, X / scale, low, high, integral);
        put(dst, dbase + 1u, Y / scale, low, high, integral);
        put(dst, dbase + 2u, Z / scale, low, high, integral);
        return;
    }
    if (code == 4u) {                            /* from XYZ */
        float X = src[sbase] * scale;
        float Y = src[sbase + 1u] * scale;
        float Z = src[sbase + 2u] * scale;
        float rr = 3.240479f * X - 1.537150f * Y - 0.498535f * Z;
        float gg = -0.969256f * X + 1.875991f * Y + 0.041556f * Z;
        float bb = 0.055648f * X - 0.204043f * Y + 1.057311f * Z;
        write_bgr(dst, dbase, dst_cn,
                  swap_rb, float3(bb / scale, gg / scale, rr / scale), alpha, low, high, integral);
        return;
    }
    if (code == 5u) {                            /* to YCrCb */
        float Y = kRedWeight * r + kGreenWeight * g + kBlueWeight * b;
        float delta = scale >= 1.0f ? 0.5f : 128.0f * scale;
        put(dst, dbase, Y / scale, low, high, integral);
        put(dst, dbase + 1u, ((r - Y) * 0.713f + delta) / scale, low, high, integral);
        put(dst, dbase + 2u, ((b - Y) * 0.564f + delta) / scale, low, high, integral);
        return;
    }
    if (code == 6u) {                            /* from YCrCb */
        float delta = scale >= 1.0f ? 0.5f : 128.0f * scale;
        float Y = src[sbase] * scale;
        float Cr = src[sbase + 1u] * scale - delta;
        float Cb = src[sbase + 2u] * scale - delta;
        float rr = Y + 1.403f * Cr;
        float gg = Y - 0.714f * Cr - 0.344f * Cb;
        float bb = Y + 1.773f * Cb;
        write_bgr(dst, dbase, dst_cn,
                  swap_rb, float3(bb / scale, gg / scale, rr / scale), alpha, low, high, integral);
        return;
    }
    if (code == 7u) {                            /* to HSV */
        float3 hsv = rgb_to_hsv(float3(b, g, r), hue_scale);
        /* A hue a hair under 360 rounds up to the top of the scale, which is
         * the same angle as zero; OpenCV stores zero. */
        float hue = hsv.x;
        if (integral != 0u && rint(hue) >= 360.0f * hue_scale) hue = 0.0f;
        put(dst, dbase, hue, low, high, integral);
        put(dst, dbase + 1u, hsv.y / scale, low, high, integral);
        put(dst, dbase + 2u, hsv.z / scale, low, high, integral);
        return;
    }
    if (code == 8u) {                            /* from HSV */
        float3 hsv = float3(src[sbase], src[sbase + 1u] * scale, src[sbase + 2u] * scale);
        float3 out = hsv_to_rgb(hsv, hue_scale);
        write_bgr(dst, dbase, dst_cn,
                  swap_rb, float3(out.x / scale, out.y / scale, out.z / scale), alpha, low, high, integral);
        return;
    }
    if (code == 9u) {                            /* to HLS */
        float3 hls = rgb_to_hls(float3(b, g, r), hue_scale);
        float hue = hls.x;
        if (integral != 0u && rint(hue) >= 360.0f * hue_scale) hue = 0.0f;
        put(dst, dbase, hue, low, high, integral);
        put(dst, dbase + 1u, hls.y / scale, low, high, integral);
        put(dst, dbase + 2u, hls.z / scale, low, high, integral);
        return;
    }
    if (code == 10u) {                           /* from HLS */
        float3 hls = float3(src[sbase], src[sbase + 1u] * scale, src[sbase + 2u] * scale);
        float3 out = hls_to_rgb(hls, hue_scale);
        write_bgr(dst, dbase, dst_cn,
                  swap_rb, float3(out.x / scale, out.y / scale, out.z / scale), alpha, low, high, integral);
        return;
    }
    if (code == 11u) {                           /* to Lab */
        float rl = srgb_to_linear(r);
        float gl = srgb_to_linear(g);
        float bl = srgb_to_linear(b);
        float X = (0.412453f * rl + 0.357580f * gl + 0.180423f * bl) / 0.950456f;
        float Y = 0.212671f * rl + 0.715160f * gl + 0.072169f * bl;
        float Z = (0.019334f * rl + 0.119193f * gl + 0.950227f * bl) / 1.088754f;
        float fx = lab_f(X), fy = lab_f(Y), fz = lab_f(Z);
        float L = 116.0f * fy - 16.0f;
        float A = 500.0f * (fx - fy);
        float B = 200.0f * (fy - fz);
        if (scale < 1.0f) {                      /* 8-bit packing */
            put(dst, dbase, L * 255.0f / 100.0f, low, high, integral);
            put(dst, dbase + 1u, A + 128.0f, low, high, integral);
            put(dst, dbase + 2u, B + 128.0f, low, high, integral);
        } else {
            put(dst, dbase, L, low, high, integral);
            put(dst, dbase + 1u, A, low, high, integral);
            put(dst, dbase + 2u, B, low, high, integral);
        }
        return;
    }
    if (code == 12u) {                           /* from Lab */
        float L = src[sbase];
        float A = src[sbase + 1u];
        float B = src[sbase + 2u];
        if (scale < 1.0f) {
            L = L * 100.0f / 255.0f;
            A = A - 128.0f;
            B = B - 128.0f;
        }
        float fy = (L + 16.0f) / 116.0f;
        float fx = fy + A / 500.0f;
        float fz = fy - B / 200.0f;
        float X = lab_f_inv(fx) * 0.950456f;
        float Y = lab_f_inv(fy);
        float Z = lab_f_inv(fz) * 1.088754f;
        float rl = 3.240479f * X - 1.537150f * Y - 0.498535f * Z;
        float gl = -0.969256f * X + 1.875991f * Y + 0.041556f * Z;
        float bl = 0.055648f * X - 0.204043f * Y + 1.057311f * Z;
        float rr = linear_to_srgb(clamp(rl, 0.0f, 1.0f));
        float gg = linear_to_srgb(clamp(gl, 0.0f, 1.0f));
        float bb = linear_to_srgb(clamp(bl, 0.0f, 1.0f));
        write_bgr(dst, dbase, dst_cn,
                  swap_rb, float3(bb / scale, gg / scale, rr / scale), alpha, low, high, integral);
        return;
    }
    if (code == 13u) {                           /* to YUV */
        float Y = kRedWeight * r + kGreenWeight * g + kBlueWeight * b;
        float delta = scale >= 1.0f ? 0.5f : 128.0f * scale;
        put(dst, dbase, Y / scale, low, high, integral);
        put(dst, dbase + 1u, (0.492f * (b - Y) + delta) / scale, low, high, integral);
        put(dst, dbase + 2u, (0.877f * (r - Y) + delta) / scale, low, high, integral);
        return;
    }
    if (code == 14u) {                           /* from YUV */
        float delta = scale >= 1.0f ? 0.5f : 128.0f * scale;
        float Y = src[sbase] * scale;
        float U = src[sbase + 1u] * scale - delta;
        float V = src[sbase + 2u] * scale - delta;
        float rr = Y + 1.140f * V;
        float gg = Y - 0.394f * U - 0.581f * V;
        float bb = Y + 2.032f * U;
        write_bgr(dst, dbase, dst_cn,
                  swap_rb, float3(bb / scale, gg / scale, rr / scale), alpha, low, high, integral);
        return;
    }
}

/* NV12 arrives as a full-resolution luma plane followed by half-resolution
 * interleaved chroma, which is what every camera on this platform produces. */
kernel void cvt_nv12(device const float* luma [[buffer(0)]],
                     device const float* chroma [[buffer(1)]],
                     device float* dst [[buffer(2)]],
                     constant uint& count [[buffer(3)]],
                     constant uint& cols [[buffer(4)]],
                     constant uint& luma_step [[buffer(5)]],
                     constant uint& chroma_step [[buffer(6)]],
                     constant uint& dst_cn [[buffer(7)]],
                     constant uint& dst_off [[buffer(8)]],
                     constant uint& dst_step [[buffer(9)]],
                     constant uint& swap_rb [[buffer(10)]],
                     constant uint& full_range [[buffer(11)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;
    float Y = luma[y * luma_step + x];
    uint cy = y / 2u;
    uint cx = (x / 2u) * 2u;
    float U = chroma[cy * chroma_step + cx] - 128.0f;
    float V = chroma[cy * chroma_step + cx + 1u] - 128.0f;
    if (full_range == 0u) Y = (Y - 16.0f) * 255.0f / 219.0f;
    float r = Y + 1.402f * V;
    float g = Y - 0.344136f * U - 0.714136f * V;
    float b = Y + 1.772f * U;
    write_bgr(dst, dst_off + y * dst_step + x * dst_cn, dst_cn,
              swap_rb, float3(b, g, r), 255.0f, gLow, gHigh, 1u);
}

/* YUY2 packs two pixels into four samples: Y0 U Y1 V. */
kernel void cvt_yuy2(device const float* packed [[buffer(0)]],
                     device float* dst [[buffer(1)]],
                     constant uint& count [[buffer(2)]],
                     constant uint& cols [[buffer(3)]],
                     constant uint& packed_step [[buffer(4)]],
                     constant uint& dst_cn [[buffer(5)]],
                     constant uint& dst_off [[buffer(6)]],
                     constant uint& dst_step [[buffer(7)]],
                     constant uint& swap_rb [[buffer(8)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;
    uint pair = (x / 2u) * 4u;
    float Y = packed[y * packed_step + pair + (x % 2u) * 2u];
    float U = packed[y * packed_step + pair + 1u] - 128.0f;
    float V = packed[y * packed_step + pair + 3u] - 128.0f;
    float r = Y + 1.402f * V;
    float g = Y - 0.344136f * U - 0.714136f * V;
    float b = Y + 1.772f * U;
    write_bgr(dst, dst_off + y * dst_step + x * dst_cn, dst_cn,
              swap_rb, float3(b, g, r), 255.0f, gLow, gHigh, 1u);
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

#: How one conversion code maps onto the shader: family, channel count out,
#: and whether red and blue are exchanged on the way through.
class Recipe {
    pub let family: int
    pub let out_channels: int
    pub let swap_rb: int

    fn init(self, family: int, out_channels: int, swap_rb: int) {
        self.family = family
        self.out_channels = out_channels
        self.swap_rb = swap_rb
    }
}

fn recipe_for(code: int, src_channels: int) -> Recipe {
    if code == COLOR_BGR2BGRA { return Recipe(0, 4, 0) }
    if code == COLOR_BGRA2BGR { return Recipe(0, 3, 0) }
    if code == COLOR_BGR2RGBA { return Recipe(0, 4, 1) }
    if code == COLOR_RGBA2BGR { return Recipe(0, 3, 1) }
    if code == COLOR_BGR2RGB { return Recipe(0, src_channels, 1) }
    if code == COLOR_BGRA2RGBA { return Recipe(0, 4, 1) }
    if code == COLOR_BGR2GRAY { return Recipe(1, 1, 0) }
    if code == COLOR_RGB2GRAY { return Recipe(1, 1, 1) }
    if code == COLOR_BGRA2GRAY { return Recipe(1, 1, 0) }
    if code == COLOR_RGBA2GRAY { return Recipe(1, 1, 1) }
    if code == COLOR_GRAY2BGR { return Recipe(2, 3, 0) }
    if code == COLOR_GRAY2BGRA { return Recipe(2, 4, 0) }
    if code == COLOR_BGR2XYZ { return Recipe(3, 3, 0) }
    if code == COLOR_RGB2XYZ { return Recipe(3, 3, 1) }
    if code == COLOR_XYZ2BGR { return Recipe(4, 3, 0) }
    if code == COLOR_XYZ2RGB { return Recipe(4, 3, 1) }
    if code == COLOR_BGR2YCrCb { return Recipe(5, 3, 0) }
    if code == COLOR_RGB2YCrCb { return Recipe(5, 3, 1) }
    if code == COLOR_YCrCb2BGR { return Recipe(6, 3, 0) }
    if code == COLOR_YCrCb2RGB { return Recipe(6, 3, 1) }
    if code == COLOR_BGR2HSV { return Recipe(7, 3, 0) }
    if code == COLOR_RGB2HSV { return Recipe(7, 3, 1) }
    if code == COLOR_HSV2BGR { return Recipe(8, 3, 0) }
    if code == COLOR_HSV2RGB { return Recipe(8, 3, 1) }
    if code == COLOR_BGR2HLS { return Recipe(9, 3, 0) }
    if code == COLOR_RGB2HLS { return Recipe(9, 3, 1) }
    if code == COLOR_HLS2BGR { return Recipe(10, 3, 0) }
    if code == COLOR_HLS2RGB { return Recipe(10, 3, 1) }
    if code == COLOR_BGR2Lab { return Recipe(11, 3, 0) }
    if code == COLOR_RGB2Lab { return Recipe(11, 3, 1) }
    if code == COLOR_Lab2BGR { return Recipe(12, 3, 0) }
    if code == COLOR_Lab2RGB { return Recipe(12, 3, 1) }
    if code == COLOR_BGR2YUV { return Recipe(13, 3, 0) }
    if code == COLOR_RGB2YUV { return Recipe(13, 3, 1) }
    if code == COLOR_YUV2BGR { return Recipe(14, 3, 0) }
    if code == COLOR_YUV2RGB { return Recipe(14, 3, 1) }
    throw ValueError(f"colour conversion code {code} is not supported")
}

#: Convert `src` between colour spaces.
#:
#: `dst_channels` overrides the channel count the code implies, which is how
#: OpenCV lets `COLOR_GRAY2BGR` produce four channels.
pub fn cvt_color(src: Mat, code: int, dst_channels: int = 0) -> Mat {
    src.flush()
    let recipe = recipe_for(code, src.channels())
    let out_cn = if dst_channels > 0 { dst_channels } else { recipe.out_channels }
    let depth = src.depth()
    let out = Mat(src.rows, src.cols, CV_MAKETYPE(depth, out_cn))
    let count = src.rows * src.cols
    if count == 0 { return out }
    #: An 8-bit image runs 0..255 and a float image runs 0..1, so the shader
    #: works in the float convention and scales on the way in and out.
    let scale = if depth == CV_32F { 1.0 } else { 1.0 / 255.0 }
    let hue_scale = if depth == CV_32F { 1.0 } else { 0.5 }
    let alpha = if depth == CV_32F { 1.0 } else { 255.0 }
    kernel_for("cvt_color").dispatch_async([src.storage, out.storage], count, [count, src.cols, src.channels(), out_cn, src.offset, src.step_, out.offset, out.step_, recipe.family, recipe.swap_rb, f32_bits(scale), f32_bits(hue_scale), f32_bits(alpha), f32_bits(depth_min(depth)), f32_bits(depth_max(depth)), if depth_is_integer(depth) {
        1
    } else {
        0
    }])
    out.invalidate()
    return out
}

#: Convert a camera's NV12 pair of planes into an interleaved colour image.
pub fn cvt_color_nv12(
    luma: Mat,
    chroma: Mat,
    code: int = COLOR_YUV2BGR_NV12,
    full_range: bool = true
) -> Mat {
    luma.flush()
    chroma.flush()
    let swap_rb = if code == COLOR_YUV2RGB_NV12 { 1 } else { 0 }
    let out = Mat(luma.rows, luma.cols, CV_MAKETYPE(CV_8U, 3))
    let count = luma.rows * luma.cols
    if count == 0 { return out }
    kernel_for("cvt_nv12").dispatch_async([luma.storage, chroma.storage, out.storage], count, [count, luma.cols, luma.step_, chroma.step_, 3, out.offset, out.step_, swap_rb, if full_range {
        1
    } else {
        0
    }])
    out.invalidate()
    return out
}

#: Convert packed 4:2:2 into an interleaved colour image.
pub fn cvt_color_yuy2(packed: Mat, code: int = COLOR_YUV2BGR_YUY2) -> Mat {
    packed.flush()
    let swap_rb = if code == COLOR_YUV2RGB_YUY2 { 1 } else { 0 }
    let cols = packed.cols * packed.channels() // 2
    let out = Mat(packed.rows, cols, CV_MAKETYPE(CV_8U, 3))
    let count = packed.rows * cols
    if count == 0 { return out }
    kernel_for("cvt_yuy2").dispatch_async(
        [packed.storage, out.storage],
        count,
        [count, cols, packed.step_, 3, out.offset, out.step_, swap_rb]
    )
    out.invalidate()
    return out
}

#: GPU tensor operations that keep their inputs and results on the device.
import std.gpu as gpu
import std.math
from std.gpu import Buffer, Kernel
import .gemm as gemm
from .tensor import GradNode, Tensor, element_count
from .trace import record_dense_forward, record_dense_backward

let SOURCE = """#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void binary_op(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device float* output [[buffer(2)]],
                      constant uint& count [[buffer(3)]],
                      constant uint& operation [[buffer(4)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    if (operation == 0) output[id] = a[id] + b[id];
    else if (operation == 1) output[id] = a[id] - b[id];
    else if (operation == 2) output[id] = a[id] * b[id];
    else output[id] = a[id] / b[id];
}

kernel void unary_op(device const float* input [[buffer(0)]],
                     device float* output [[buffer(1)]],
                     constant uint& count [[buffer(2)]],
                     constant uint& operation [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float value = input[id];
    if (operation == 0) output[id] = value;
    else if (operation == 1) output[id] = max(value, 0.0f);
    else if (operation == 2) output[id] = 1.0f / (1.0f + exp(-value));
    else if (operation == 3) output[id] = tanh(value);
    else if (operation == 4) output[id] = exp(value);
    else if (operation == 5) output[id] = log(value);
    else if (operation == 6) output[id] = fabs(value);
    else if (operation == 7) output[id] = -value;
    else if (operation == 8) output[id] = sqrt(value);
    else if (operation == 9) output[id] = rsqrt(value);
    else if (operation == 10) output[id] = value * value;
    else if (operation == 11) {
        float inner = 0.7978845608f * (value + 0.044715f * value * value * value);
        output[id] = 0.5f * value * (1.0f + tanh(inner));
    } else if (operation == 12) {
        output[id] = value / (1.0f + exp(-value));
    } else {
        output[id] = value > 0.0f ? value : 0.01f * value;
    }
}

kernel void scalar_mul(device const float* input [[buffer(0)]],
                       device const float* scalar [[buffer(1)]],
                       device float* output [[buffer(2)]],
                       constant uint& count [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    if (id < count) output[id] = input[id] * scalar[0];
}

kernel void matrix_transpose(device const float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             constant uint& rows [[buffer(2)]],
                             constant uint& columns [[buffer(3)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= rows * columns) return;
    uint row = id / columns;
    uint column = id % columns;
    output[column * rows + row] = input[id];
}

kernel void row_softmax(device const float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        constant uint& rows [[buffer(2)]],
                        constant uint& columns [[buffer(3)]],
                        uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    uint base = row * columns;
    float peak = input[base];
    for (uint column = 1; column < columns; ++column)
        peak = max(peak, input[base + column]);
    float total = 0.0f;
    for (uint column = 0; column < columns; ++column)
        total += exp(input[base + column] - peak);
    for (uint column = 0; column < columns; ++column)
        output[base + column] = exp(input[base + column] - peak) / total;
}

kernel void row_argmax(device const float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       constant uint& rows [[buffer(2)]],
                       constant uint& columns [[buffer(3)]],
                       uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    uint base = row * columns;
    float peak = input[base];
    uint best = 0;
    for (uint column = 1; column < columns; ++column) {
        float value = input[base + column];
        if (value > peak) { peak = value; best = column; }
    }
    output[row] = float(best);
}

kernel void one_hot_rows(device const float* indices [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         constant uint& rows [[buffer(2)]],
                         constant uint& depth [[buffer(3)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= rows * depth) return;
    uint row = id / depth;
    uint column = id % depth;
    output[id] = int(indices[row]) == int(column) ? 1.0f : 0.0f;
}

kernel void broadcast_add(device const float* a [[buffer(0)]],
                          device const float* b [[buffer(1)]],
                          device float* output [[buffer(2)]],
                          constant uint& inner [[buffer(3)]],
                          constant uint& total [[buffer(4)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= total) return;
    output[id] = a[id] + b[id % inner];
}

kernel void binary_backward_op(device const float* a [[buffer(0)]],
                               device const float* b [[buffer(1)]],
                               device const float* upstream [[buffer(2)]],
                               device float* grad_a [[buffer(3)]],
                               device float* grad_b [[buffer(4)]],
                               constant uint& count [[buffer(5)]],
                               constant uint& operation [[buffer(6)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float u = upstream[id];
    if (operation == 0) { grad_a[id] = u; grad_b[id] = u; }
    else if (operation == 1) { grad_a[id] = u; grad_b[id] = -u; }
    else if (operation == 2) { grad_a[id] = u * b[id]; grad_b[id] = u * a[id]; }
    else {
        float bv = b[id];
        grad_a[id] = u / bv;
        grad_b[id] = -u * a[id] / (bv * bv);
    }
}

kernel void reduce_leading(device const float* upstream [[buffer(0)]],
                           device float* grad_small [[buffer(1)]],
                           constant uint& outer [[buffer(2)]],
                           constant uint& inner [[buffer(3)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= inner) return;
    float total = 0.0f;
    for (uint row = 0; row < outer; ++row) total += upstream[row * inner + id];
    grad_small[id] = total;
}

kernel void reduce_leading4(device const float* upstream [[buffer(0)]],
                            device float* grad_small [[buffer(1)]],
                            constant uint& outer [[buffer(2)]],
                            constant uint& inner [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    uint col = id * 4;
    if (col >= inner) return;
    float4 total = float4(0.0f);
    for (uint row = 0; row < outer; ++row)
        total += *((device const float4*)(upstream + row * inner + col));
    *((device float4*)(grad_small + col)) = total;
}

kernel void reduce_leading_split(device const float* upstream [[buffer(0)]],
                                 device float* partials [[buffer(1)]],
                                 constant uint& outer [[buffer(2)]],
                                 constant uint& inner [[buffer(3)]],
                                 constant uint& chunks [[buffer(4)]],
                                 uint id [[thread_position_in_grid]]) {
    if (id >= inner * chunks) return;
    uint col = id % inner;
    uint chunk = id / inner;
    uint per = (outer + chunks - 1u) / chunks;
    uint start = chunk * per;
    uint stop = min(start + per, outer);
    float total = 0.0f;
    for (uint row = start; row < stop; ++row) total += upstream[row * inner + col];
    partials[chunk * inner + col] = total;
}

kernel void unary_backward_op(device const float* saved [[buffer(0)]],
                              device const float* upstream [[buffer(1)]],
                              device float* grad_input [[buffer(2)]],
                              constant uint& count [[buffer(3)]],
                              constant uint& operation [[buffer(4)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float s = saved[id];
    float u = upstream[id];
    if (operation == 0) grad_input[id] = u;
    else if (operation == 1) grad_input[id] = s > 0.0f ? u : 0.0f;
    else if (operation == 2) grad_input[id] = u * s * (1.0f - s);
    else if (operation == 3) grad_input[id] = u * (1.0f - s * s);
    else if (operation == 4) grad_input[id] = u * s;
    else if (operation == 5) grad_input[id] = u / s;
    else if (operation == 6) grad_input[id] = s >= 0.0f ? u : -u;
    else if (operation == 7) grad_input[id] = -u;
    else if (operation == 8) grad_input[id] = u * 0.5f / s;
    else if (operation == 9) {
        float y = s;
        grad_input[id] = u * (-0.5f) * y * y * y;
    } else if (operation == 10) grad_input[id] = u * 2.0f * s;
    else if (operation == 11) {
        float x = s;
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        float t = tanh(inner);
        float dt = 1.0f - t * t;
        float dinner = 0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x);
        grad_input[id] = u * (0.5f * (1.0f + t) + 0.5f * x * dt * dinner);
    } else if (operation == 12) {
        float x = s;
        float sig = 1.0f / (1.0f + exp(-x));
        grad_input[id] = u * (sig + x * sig * (1.0f - sig));
    } else {
        grad_input[id] = s > 0.0f ? u : 0.01f * u;
    }
}

kernel void unary_backward_relu4(device const float* saved [[buffer(0)]],
                                 device const float* upstream [[buffer(1)]],
                                 device float* grad_input [[buffer(2)]],
                                 constant uint& count [[buffer(3)]],
                                 uint id [[thread_position_in_grid]]) {
    uint i = id * 4;
    if (i >= count) return;
    float4 s = *((device const float4*)(saved + i));
    float4 u = *((device const float4*)(upstream + i));
    *((device float4*)(grad_input + i)) = select(float4(0.0f), u, s > 0.0f);
}

kernel void softmax_backward(device const float* output [[buffer(0)]],
                             device const float* upstream [[buffer(1)]],
                             device float* grad_input [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& columns [[buffer(4)]],
                             uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    uint base = row * columns;
    float dot = 0.0f;
    for (uint column = 0; column < columns; ++column) dot += upstream[base + column] * output[base + column];
    for (uint column = 0; column < columns; ++column)
        grad_input[base + column] = output[base + column] * (upstream[base + column] - dot);
}

kernel void copy_offset(device const float* source [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        constant uint& source_offset [[buffer(2)]],
                        constant uint& count [[buffer(3)]],
                        uint id [[thread_position_in_grid]]) {
    if (id < count) output[id] = source[source_offset + id];
}

kernel void col2im_nhwc(device const float* col [[buffer(0)]],
                        device float* grad [[buffer(1)]],
                        constant uint& N [[buffer(2)]],
                        constant uint& H [[buffer(3)]],
                        constant uint& W [[buffer(4)]],
                        constant uint& C [[buffer(5)]],
                        constant uint& KH [[buffer(6)]],
                        constant uint& KW [[buffer(7)]],
                        constant uint& SH [[buffer(8)]],
                        constant uint& SW [[buffer(9)]],
                        constant uint& PH [[buffer(10)]],
                        constant uint& PW [[buffer(11)]],
                        uint id [[thread_position_in_grid]]) {
    uint count = N * H * W * C;
    if (id >= count) return;
    uint outH = (H + 2 * PH - KH) / SH + 1;
    uint outW = (W + 2 * PW - KW) / SW + 1;
    uint K = KH * KW * C;
    uint c = id % C;
    uint iw = (id / C) % W;
    uint ih = (id / (C * W)) % H;
    uint n = id / (C * W * H);
    float total = 0.0f;
    for (uint kh = 0; kh < KH; ++kh) {
        int top = int(ih) + int(PH) - int(kh);
        if (top < 0 || uint(top) % SH != 0) continue;
        uint oh = uint(top) / SH;
        if (oh >= outH) continue;
        for (uint kw = 0; kw < KW; ++kw) {
            int left = int(iw) + int(PW) - int(kw);
            if (left < 0 || uint(left) % SW != 0) continue;
            uint ow = uint(left) / SW;
            if (ow >= outW) continue;
            uint row = (n * outH + oh) * outW + ow;
            total += col[row * K + (kh * KW + kw) * C + c];
        }
    }
    grad[id] = total;
}

kernel void max_pool2d_backward_nhwc(device const float* input [[buffer(0)]],
                                     device const float* upstream [[buffer(1)]],
                                     device float* grad [[buffer(2)]],
                                     constant uint& N [[buffer(3)]],
                                     constant uint& H [[buffer(4)]],
                                     constant uint& W [[buffer(5)]],
                                     constant uint& C [[buffer(6)]],
                                     constant uint& KH [[buffer(7)]],
                                     constant uint& KW [[buffer(8)]],
                                     constant uint& SH [[buffer(9)]],
                                     constant uint& SW [[buffer(10)]],
                                     constant uint& PH [[buffer(11)]],
                                     constant uint& PW [[buffer(12)]],
                                     constant uint& OH [[buffer(13)]],
                                     constant uint& OW [[buffer(14)]],
                                     uint id [[thread_position_in_grid]]) {
    uint count = N * H * W * C;
    if (id >= count) return;
    uint c = id % C;
    uint iw = (id / C) % W;
    uint ih = (id / (C * W)) % H;
    uint n = id / (C * W * H);
    float here = input[id];
    float total = 0.0f;
    for (uint kh = 0; kh < KH; ++kh) {
        int top = int(ih) + int(PH) - int(kh);
        if (top < 0 || uint(top) % SH != 0) continue;
        uint oh = uint(top) / SH;
        if (oh >= OH) continue;
        for (uint kw = 0; kw < KW; ++kw) {
            int left = int(iw) + int(PW) - int(kw);
            if (left < 0 || uint(left) % SW != 0) continue;
            uint ow = uint(left) / SW;
            if (ow >= OW) continue;
            float peak = -3.402823e+38f;
            uint best = 0xffffffffu;
            for (uint wh = 0; wh < KH; ++wh) {
                for (uint ww = 0; ww < KW; ++ww) {
                    int sh = int(oh * SH) - int(PH) + int(wh);
                    int sw = int(ow * SW) - int(PW) + int(ww);
                    if (sh < 0 || sw < 0 || sh >= int(H) || sw >= int(W)) continue;
                    uint slot = ((n * H + uint(sh)) * W + uint(sw)) * C + c;
                    float v = input[slot];
                    if (v > peak) { peak = v; best = slot; }
                }
            }
            if (best == id && peak == here)
                total += upstream[((n * OH + oh) * OW + ow) * C + c];
        }
    }
    grad[id] = total;
}

kernel void avg_pool2d_backward_nhwc(device const float* upstream [[buffer(0)]],
                                     device float* grad [[buffer(1)]],
                                     constant uint& N [[buffer(2)]],
                                     constant uint& H [[buffer(3)]],
                                     constant uint& W [[buffer(4)]],
                                     constant uint& C [[buffer(5)]],
                                     constant uint& KH [[buffer(6)]],
                                     constant uint& KW [[buffer(7)]],
                                     constant uint& SH [[buffer(8)]],
                                     constant uint& SW [[buffer(9)]],
                                     constant uint& PH [[buffer(10)]],
                                     constant uint& PW [[buffer(11)]],
                                     constant uint& OH [[buffer(12)]],
                                     constant uint& OW [[buffer(13)]],
                                     uint id [[thread_position_in_grid]]) {
    uint count = N * H * W * C;
    if (id >= count) return;
    uint c = id % C;
    uint iw = (id / C) % W;
    uint ih = (id / (C * W)) % H;
    uint n = id / (C * W * H);
    float total = 0.0f;
    for (uint kh = 0; kh < KH; ++kh) {
        int top = int(ih) + int(PH) - int(kh);
        if (top < 0 || uint(top) % SH != 0) continue;
        uint oh = uint(top) / SH;
        if (oh >= OH) continue;
        for (uint kw = 0; kw < KW; ++kw) {
            int left = int(iw) + int(PW) - int(kw);
            if (left < 0 || uint(left) % SW != 0) continue;
            uint ow = uint(left) / SW;
            if (ow >= OW) continue;
            uint seen = 0;
            for (uint wh = 0; wh < KH; ++wh) {
                for (uint ww = 0; ww < KW; ++ww) {
                    int sh = int(oh * SH) - int(PH) + int(wh);
                    int sw = int(ow * SW) - int(PW) + int(ww);
                    if (sh < 0 || sw < 0 || sh >= int(H) || sw >= int(W)) continue;
                    seen += 1;
                }
            }
            if (seen > 0)
                total += upstream[((n * OH + oh) * OW + ow) * C + c] / float(seen);
        }
    }
    grad[id] = total;
}

kernel void crop2d_nhwc(device const float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        constant uint& N [[buffer(2)]],
                        constant uint& H [[buffer(3)]],
                        constant uint& W [[buffer(4)]],
                        constant uint& C [[buffer(5)]],
                        constant uint& PT [[buffer(6)]],
                        constant uint& PL [[buffer(7)]],
                        constant uint& IH [[buffer(8)]],
                        constant uint& IW [[buffer(9)]],
                        uint id [[thread_position_in_grid]]) {
    uint count = N * H * W * C;
    if (id >= count) return;
    uint c = id % C;
    uint w = (id / C) % W;
    uint h = (id / (C * W)) % H;
    uint n = id / (C * W * H);
    output[id] = input[((n * IH + h + PT) * IW + w + PL) * C + c];
}

kernel void upsample_nearest2d_backward_nhwc(device const float* upstream [[buffer(0)]],
                                             device float* grad [[buffer(1)]],
                                             constant uint& N [[buffer(2)]],
                                             constant uint& H [[buffer(3)]],
                                             constant uint& W [[buffer(4)]],
                                             constant uint& C [[buffer(5)]],
                                             constant uint& SH [[buffer(6)]],
                                             constant uint& SW [[buffer(7)]],
                                             constant uint& OH [[buffer(8)]],
                                             constant uint& OW [[buffer(9)]],
                                             uint id [[thread_position_in_grid]]) {
    uint count = N * H * W * C;
    if (id >= count) return;
    uint c = id % C;
    uint w = (id / C) % W;
    uint h = (id / (C * W)) % H;
    uint n = id / (C * W * H);
    float total = 0.0f;
    for (uint dh = 0; dh < SH; ++dh) {
        uint oh = h * SH + dh;
        if (oh >= OH) continue;
        for (uint dw = 0; dw < SW; ++dw) {
            uint ow = w * SW + dw;
            if (ow >= OW) continue;
            total += upstream[((n * OH + oh) * OW + ow) * C + c];
        }
    }
    grad[id] = total;
}

kernel void repeat_last_backward(device const float* upstream [[buffer(0)]],
                                 device float* grad [[buffer(1)]],
                                 constant uint& rows [[buffer(2)]],
                                 constant uint& inner [[buffer(3)]],
                                 constant uint& times [[buffer(4)]],
                                 uint id [[thread_position_in_grid]]) {
    uint count = rows * inner;
    if (id >= count) return;
    uint col = id % inner;
    uint row = id / inner;
    float total = 0.0f;
    for (uint t = 0; t < times; ++t) total += upstream[row * inner * times + t * inner + col];
    grad[id] = total;
}

kernel void take_strided_rows(device const float* source [[buffer(0)]],
                              device float* dest [[buffer(1)]],
                              constant uint& rows [[buffer(2)]],
                              constant uint& dst_width [[buffer(3)]],
                              constant uint& src_width [[buffer(4)]],
                              constant uint& src_col [[buffer(5)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= rows * dst_width) return;
    uint r = id / dst_width;
    uint c = id % dst_width;
    dest[id] = source[r * src_width + src_col + c];
}

kernel void select_backward(device const float* a [[buffer(0)]],
                            device const float* b [[buffer(1)]],
                            device const float* upstream [[buffer(2)]],
                            device float* grad [[buffer(3)]],
                            constant uint& count [[buffer(4)]],
                            constant uint& want_max [[buffer(5)]],
                            constant uint& side [[buffer(6)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    bool a_wins = want_max != 0u ? a[id] >= b[id] : a[id] <= b[id];
    bool mine = side == 0u ? a_wins : !a_wins;
    grad[id] = mine ? upstream[id] : 0.0f;
}

kernel void layer_norm_backward(device const float* input [[buffer(0)]],
                                device const float* gamma [[buffer(1)]],
                                device const float* upstream [[buffer(2)]],
                                device float* grad [[buffer(3)]],
                                constant uint& rows [[buffer(4)]],
                                constant uint& width [[buffer(5)]],
                                uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    uint base = row * width;
    float total = 0.0f;
    for (uint i = 0; i < width; ++i) total += input[base + i];
    float m = total / float(width);
    float sq = 0.0f;
    for (uint i = 0; i < width; ++i) {
        float d = input[base + i] - m;
        sq += d * d;
    }
    float inv = rsqrt(sq / float(width) + 1e-5f);
    float sum_dy = 0.0f;
    float sum_dy_xhat = 0.0f;
    for (uint i = 0; i < width; ++i) {
        float dy = upstream[base + i] * gamma[i];
        sum_dy += dy;
        sum_dy_xhat += dy * (input[base + i] - m) * inv;
    }
    float scale = 1.0f / float(width);
    for (uint i = 0; i < width; ++i) {
        float dy = upstream[base + i] * gamma[i];
        float xhat = (input[base + i] - m) * inv;
        grad[base + i] = inv * (dy - scale * sum_dy - scale * xhat * sum_dy_xhat);
    }
}

kernel void layer_norm_params_backward(device const float* input [[buffer(0)]],
                                       device const float* upstream [[buffer(1)]],
                                       device float* grad_gamma [[buffer(2)]],
                                       device float* grad_beta [[buffer(3)]],
                                       constant uint& rows [[buffer(4)]],
                                       constant uint& width [[buffer(5)]],
                                       uint i [[thread_position_in_grid]]) {
    if (i >= width) return;
    float g = 0.0f;
    float b = 0.0f;
    for (uint row = 0; row < rows; ++row) {
        uint base = row * width;
        float total = 0.0f;
        for (uint j = 0; j < width; ++j) total += input[base + j];
        float m = total / float(width);
        float sq = 0.0f;
        for (uint j = 0; j < width; ++j) {
            float d = input[base + j] - m;
            sq += d * d;
        }
        float inv = rsqrt(sq / float(width) + 1e-5f);
        float dy = upstream[base + i];
        g += dy * (input[base + i] - m) * inv;
        b += dy;
    }
    grad_gamma[i] = g;
    grad_beta[i] = b;
}

kernel void norm_channel_backward(device const float* input [[buffer(0)]],
                                  device const float* mean [[buffer(1)]],
                                  device const float* variance [[buffer(2)]],
                                  device const float* gamma [[buffer(3)]],
                                  device const float* upstream [[buffer(4)]],
                                  device const float* sum_dy [[buffer(5)]],
                                  device const float* sum_dy_xhat [[buffer(6)]],
                                  device float* grad [[buffer(7)]],
                                  constant uint& count [[buffer(8)]],
                                  constant uint& channels [[buffer(9)]],
                                  constant uint& nspatial [[buffer(10)]],
                                  uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint c = id % channels;
    float inv = rsqrt(variance[c] + 1e-5f);
    float xhat = (input[id] - mean[c]) * inv;
    float dy = upstream[id] * gamma[c];
    float scale = 1.0f / float(nspatial);
    grad[id] = inv * (dy - scale * sum_dy[c] - scale * xhat * sum_dy_xhat[c]);
}

kernel void norm_channel_sums(device const float* input [[buffer(0)]],
                              device const float* mean [[buffer(1)]],
                              device const float* variance [[buffer(2)]],
                              device const float* gamma [[buffer(3)]],
                              device const float* upstream [[buffer(4)]],
                              device float* sum_dy [[buffer(5)]],
                              device float* sum_dy_xhat [[buffer(6)]],
                              device float* grad_gamma [[buffer(7)]],
                              device float* grad_beta [[buffer(8)]],
                              constant uint& nspatial [[buffer(9)]],
                              constant uint& channels [[buffer(10)]],
                              uint c [[thread_position_in_grid]]) {
    if (c >= channels) return;
    float inv = rsqrt(variance[c] + 1e-5f);
    float dy_total = 0.0f;
    float dy_xhat = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    for (uint i = 0; i < nspatial; ++i) {
        uint slot = i * channels + c;
        float xhat = (input[slot] - mean[c]) * inv;
        float up = upstream[slot];
        float dy = up * gamma[c];
        dy_total += dy;
        dy_xhat += dy * xhat;
        g += up * xhat;
        b += up;
    }
    sum_dy[c] = dy_total;
    sum_dy_xhat[c] = dy_xhat;
    grad_gamma[c] = g;
    grad_beta[c] = b;
}

kernel void group_norm_backward(device const float* input [[buffer(0)]],
                                device const float* mean [[buffer(1)]],
                                device const float* variance [[buffer(2)]],
                                device const float* gamma [[buffer(3)]],
                                device const float* upstream [[buffer(4)]],
                                device const float* sum_dy [[buffer(5)]],
                                device const float* sum_dy_xhat [[buffer(6)]],
                                device float* grad [[buffer(7)]],
                                constant uint& count [[buffer(8)]],
                                constant uint& spatial [[buffer(9)]],
                                constant uint& C [[buffer(10)]],
                                constant uint& groups [[buffer(11)]],
                                uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint c = id % C;
    uint n = id / (C * spatial);
    uint gs = C / groups;
    uint g = c / gs;
    uint slot = n * groups + g;
    float inv = rsqrt(variance[slot] + 1e-5f);
    float xhat = (input[id] - mean[slot]) * inv;
    float dy = upstream[id] * gamma[c];
    float scale = 1.0f / float(spatial * gs);
    grad[id] = inv * (dy - scale * sum_dy[slot] - scale * xhat * sum_dy_xhat[slot]);
}

kernel void group_norm_sums(device const float* input [[buffer(0)]],
                            device const float* mean [[buffer(1)]],
                            device const float* variance [[buffer(2)]],
                            device const float* gamma [[buffer(3)]],
                            device const float* upstream [[buffer(4)]],
                            device float* sum_dy [[buffer(5)]],
                            device float* sum_dy_xhat [[buffer(6)]],
                            constant uint& N [[buffer(7)]],
                            constant uint& spatial [[buffer(8)]],
                            constant uint& C [[buffer(9)]],
                            constant uint& groups [[buffer(10)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= N * groups) return;
    uint gs = C / groups;
    uint n = id / groups;
    uint g = id % groups;
    float inv = rsqrt(variance[id] + 1e-5f);
    float dy_total = 0.0f;
    float dy_xhat = 0.0f;
    for (uint hw = 0; hw < spatial; ++hw) {
        uint base = (n * spatial + hw) * C + g * gs;
        for (uint k = 0; k < gs; ++k) {
            float xhat = (input[base + k] - mean[id]) * inv;
            float dy = upstream[base + k] * gamma[g * gs + k];
            dy_total += dy;
            dy_xhat += dy * xhat;
        }
    }
    sum_dy[id] = dy_total;
    sum_dy_xhat[id] = dy_xhat;
}

kernel void channel_params_backward(device const float* input [[buffer(0)]],
                                    device const float* mean [[buffer(1)]],
                                    device const float* variance [[buffer(2)]],
                                    device const float* upstream [[buffer(3)]],
                                    device float* grad_gamma [[buffer(4)]],
                                    device float* grad_beta [[buffer(5)]],
                                    constant uint& nspatial [[buffer(6)]],
                                    constant uint& channels [[buffer(7)]],
                                    constant uint& spatial [[buffer(8)]],
                                    constant uint& groups [[buffer(9)]],
                                    uint c [[thread_position_in_grid]]) {
    if (c >= channels) return;
    uint gs = groups == 0u ? 0u : channels / groups;
    float g = 0.0f;
    float b = 0.0f;
    for (uint i = 0; i < nspatial; ++i) {
        uint slot = i * channels + c;
        uint pick = c;
        if (groups != 0u) {
            uint n = i / spatial;
            pick = n * groups + c / gs;
        }
        float inv = rsqrt(variance[pick] + 1e-5f);
        float xhat = (input[slot] - mean[pick]) * inv;
        g += upstream[slot] * xhat;
        b += upstream[slot];
    }
    grad_gamma[c] = g;
    grad_beta[c] = b;
}

kernel void add_scaled_into(device const float* value [[buffer(0)]],
                            device float* total [[buffer(1)]],
                            constant uint& weight [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
    if (id == 0) total[0] += value[0] * float(weight);
}

kernel void add_correct_into(device const float* logits [[buffer(0)]],
                             device const float* labels [[buffer(1)]],
                             device float* total [[buffer(2)]],
                             constant uint& batch [[buffer(3)]],
                             constant uint& classes [[buffer(4)]],
                             uint sample [[thread_position_in_grid]]) {
    if (sample >= batch) return;
    uint base = sample * classes;
    uint best = 0;
    float peak = logits[base];
    for (uint column = 1; column < classes; ++column) {
        float value = logits[base + column];
        if (value > peak) { peak = value; best = column; }
    }
    if (uint(labels[sample]) == best)
        atomic_fetch_add_explicit((device atomic<float>*)total, 1.0f, memory_order_relaxed);
}

kernel void im2col_nhwc(device const float* input [[buffer(0)]],
                        device float* col [[buffer(1)]],
                        constant uint& N [[buffer(2)]],
                        constant uint& H [[buffer(3)]],
                        constant uint& W [[buffer(4)]],
                        constant uint& C [[buffer(5)]],
                        constant uint& KH [[buffer(6)]],
                        constant uint& KW [[buffer(7)]],
                        constant uint& SH [[buffer(8)]],
                        constant uint& SW [[buffer(9)]],
                        constant uint& PH [[buffer(10)]],
                        constant uint& PW [[buffer(11)]],
                        uint id [[thread_position_in_grid]]) {
    uint outH = (H + 2 * PH - KH) / SH + 1;
    uint outW = (W + 2 * PW - KW) / SW + 1;
    uint K = KH * KW * C;
    uint rows = N * outH * outW;
    if (id >= rows * K) return;
    uint row = id / K;
    uint k = id % K;
    uint n = row / (outH * outW);
    uint rem = row % (outH * outW);
    uint oh = rem / outW;
    uint ow = rem % outW;
    uint kh = k / (KW * C);
    uint kwc = k % (KW * C);
    uint kw = kwc / C;
    uint c = kwc % C;
    int ih = int(oh * SH) - int(PH) + int(kh);
    int iw = int(ow * SW) - int(PW) + int(kw);
    float v = 0.0f;
    if (ih >= 0 && iw >= 0 && ih < int(H) && iw < int(W))
        v = input[((n * H + uint(ih)) * W + uint(iw)) * C + c];
    col[id] = v;
}

kernel void batch_norm_forward(device const float* input [[buffer(0)]],
                               device const float* mean [[buffer(1)]],
                               device const float* variance [[buffer(2)]],
                               device const float* gamma [[buffer(3)]],
                               device const float* beta [[buffer(4)]],
                               device float* output [[buffer(5)]],
                               constant uint& count [[buffer(6)]],
                               constant uint& channels [[buffer(7)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint c = id % channels;
    float inv = rsqrt(variance[c] + 1e-5f);
    output[id] = gamma[c] * (input[id] - mean[c]) * inv + beta[c];
}

kernel void moments_channel(device const float* input [[buffer(0)]],
                            device float* mean [[buffer(1)]],
                            device float* variance [[buffer(2)]],
                            constant uint& nspatial [[buffer(3)]],
                            constant uint& channels [[buffer(4)]],
                            uint c [[thread_position_in_grid]]) {
    if (c >= channels) return;
    float total = 0.0f;
    float sq = 0.0f;
    for (uint i = 0; i < nspatial; ++i) {
        float v = input[i * channels + c];
        total += v;
        sq += v * v;
    }
    float m = total / float(nspatial);
    mean[c] = m;
    variance[c] = sq / float(nspatial) - m * m;
}

kernel void layer_norm_forward(device const float* input [[buffer(0)]],
                               device const float* gamma [[buffer(1)]],
                               device const float* beta [[buffer(2)]],
                               device float* output [[buffer(3)]],
                               constant uint& rows [[buffer(4)]],
                               constant uint& width [[buffer(5)]],
                               uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    uint base = row * width;
    float total = 0.0f;
    for (uint i = 0; i < width; ++i) total += input[base + i];
    float m = total / float(width);
    float sq = 0.0f;
    for (uint i = 0; i < width; ++i) {
        float d = input[base + i] - m;
        sq += d * d;
    }
    float inv = rsqrt(sq / float(width) + 1e-5f);
    for (uint i = 0; i < width; ++i)
        output[base + i] = gamma[i] * (input[base + i] - m) * inv + beta[i];
}

kernel void gather_rows(device const float* src [[buffer(0)]],
                        device const float* indices [[buffer(1)]],
                        device float* dst [[buffer(2)]],
                        constant uint& rows [[buffer(3)]],
                        constant uint& width [[buffer(4)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= rows * width) return;
    uint r = id / width;
    uint c = id % width;
    uint src_row = uint(indices[r]);
    dst[id] = src[src_row * width + c];
}

kernel void scalar_mul_bits(device const float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant uint& count [[buffer(2)]],
                            constant uint& bits [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    if (id < count) output[id] = input[id] * as_type<float>(bits);
}

kernel void fill_bits(device float* output [[buffer(0)]],
                      constant uint& count [[buffer(1)]],
                      constant uint& bits [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    if (id < count) output[id] = as_type<float>(bits);
}

kernel void clamp_op(device const float* input [[buffer(0)]],
                     device float* output [[buffer(1)]],
                     constant uint& count [[buffer(2)]],
                     constant uint& lo_bits [[buffer(3)]],
                     constant uint& hi_bits [[buffer(4)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    output[id] = clamp(input[id], as_type<float>(lo_bits), as_type<float>(hi_bits));
}

kernel void clamp_backward(device const float* input [[buffer(0)]],
                           device const float* upstream [[buffer(1)]],
                           device float* grad [[buffer(2)]],
                           constant uint& count [[buffer(3)]],
                           constant uint& lo_bits [[buffer(4)]],
                           constant uint& hi_bits [[buffer(5)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float x = input[id];
    float lo = as_type<float>(lo_bits);
    float hi = as_type<float>(hi_bits);
    grad[id] = (x >= lo && x <= hi) ? upstream[id] : 0.0f;
}

kernel void maximum_op(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* output [[buffer(2)]],
                       constant uint& count [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    if (id < count) output[id] = max(a[id], b[id]);
}

kernel void minimum_op(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* output [[buffer(2)]],
                       constant uint& count [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    if (id < count) output[id] = min(a[id], b[id]);
}

kernel void reduce_sum_rows(device const float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant uint& rows [[buffer(2)]],
                            constant uint& columns [[buffer(3)]],
                            uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    uint base = row * columns;
    float total = 0.0f;
    for (uint c = 0; c < columns; ++c) total += input[base + c];
    output[row] = total;
}

kernel void reduce_sum_all(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant uint& count [[buffer(2)]],
                           uint gid [[thread_position_in_grid]],
                           uint lid [[thread_index_in_threadgroup]],
                           uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[256];
    scratch[lid] = (gid < count) ? input[gid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) output[wid] = scratch[0];
}

kernel void copy_range(device const float* source [[buffer(0)]],
                       device float* dest [[buffer(1)]],
                       constant uint& src_off [[buffer(2)]],
                       constant uint& dst_off [[buffer(3)]],
                       constant uint& count [[buffer(4)]],
                       uint id [[thread_position_in_grid]]) {
    if (id < count) dest[dst_off + id] = source[src_off + id];
}

kernel void copy_strided_rows(device const float* source [[buffer(0)]],
                              device float* dest [[buffer(1)]],
                              constant uint& rows [[buffer(2)]],
                              constant uint& src_width [[buffer(3)]],
                              constant uint& dst_width [[buffer(4)]],
                              constant uint& dst_col [[buffer(5)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= rows * src_width) return;
    uint r = id / src_width;
    uint c = id % src_width;
    dest[r * dst_width + dst_col + c] = source[r * src_width + c];
}

kernel void where_op(device const float* cond [[buffer(0)]],
                     device const float* a [[buffer(1)]],
                     device const float* b [[buffer(2)]],
                     device float* output [[buffer(3)]],
                     constant uint& count [[buffer(4)]],
                     uint id [[thread_position_in_grid]]) {
    if (id < count) output[id] = cond[id] != 0.0f ? a[id] : b[id];
}

kernel void pow_scalar(device const float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       constant uint& exp_bits [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    if (id < count) output[id] = pow(input[id], as_type<float>(exp_bits));
}

kernel void pad2d_nhwc(device const float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       constant uint& N [[buffer(2)]],
                       constant uint& H [[buffer(3)]],
                       constant uint& W [[buffer(4)]],
                       constant uint& C [[buffer(5)]],
                       constant uint& PT [[buffer(6)]],
                       constant uint& PB [[buffer(7)]],
                       constant uint& PL [[buffer(8)]],
                       constant uint& PR [[buffer(9)]],
                       uint id [[thread_position_in_grid]]) {
    uint oh = H + PT + PB;
    uint ow = W + PL + PR;
    uint nout = N * oh * ow * C;
    if (id >= nout) return;
    uint c = id % C;
    uint ow_i = (id / C) % ow;
    uint oh_i = (id / (C * ow)) % oh;
    uint n = id / (C * ow * oh);
    int ih = int(oh_i) - int(PT);
    int iw = int(ow_i) - int(PL);
    if (ih < 0 || iw < 0 || ih >= int(H) || iw >= int(W)) {
        output[id] = 0.0f;
        return;
    }
    output[id] = input[((n * H + uint(ih)) * W + uint(iw)) * C + c];
}

kernel void nhwc_to_nchw(device const float* input [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         constant uint& N [[buffer(2)]],
                         constant uint& H [[buffer(3)]],
                         constant uint& W [[buffer(4)]],
                         constant uint& C [[buffer(5)]],
                         uint id [[thread_position_in_grid]]) {
    uint count = N * H * W * C;
    if (id >= count) return;
    uint c = id % C;
    uint w = (id / C) % W;
    uint h = (id / (C * W)) % H;
    uint n = id / (C * W * H);
    output[((n * C + c) * H + h) * W + w] = input[id];
}

kernel void nchw_to_nhwc(device const float* input [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         constant uint& N [[buffer(2)]],
                         constant uint& C [[buffer(3)]],
                         constant uint& H [[buffer(4)]],
                         constant uint& W [[buffer(5)]],
                         uint id [[thread_position_in_grid]]) {
    uint count = N * C * H * W;
    if (id >= count) return;
    uint w = id % W;
    uint h = (id / W) % H;
    uint c = (id / (W * H)) % C;
    uint n = id / (W * H * C);
    output[((n * H + h) * W + w) * C + c] = input[id];
}

kernel void repeat_last(device const float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        constant uint& rows [[buffer(2)]],
                        constant uint& inner [[buffer(3)]],
                        constant uint& times [[buffer(4)]],
                        uint id [[thread_position_in_grid]]) {
    uint count = rows * inner * times;
    if (id >= count) return;
    uint col = id % (inner * times);
    uint row = id / (inner * times);
    output[id] = input[row * inner + (col % inner)];
}

kernel void bce_sample(device const float* pred [[buffer(0)]],
                       device const float* target [[buffer(1)]],
                       device float* out [[buffer(2)]],
                       constant uint& count [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float p = clamp(pred[id], 1e-7f, 1.0f - 1e-7f);
    float t = target[id];
    out[id] = -(t * log(p) + (1.0f - t) * log(1.0f - p));
}

kernel void max_pool2d_nhwc(device const float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant uint& N [[buffer(2)]],
                            constant uint& H [[buffer(3)]],
                            constant uint& W [[buffer(4)]],
                            constant uint& C [[buffer(5)]],
                            constant uint& KH [[buffer(6)]],
                            constant uint& KW [[buffer(7)]],
                            constant uint& SH [[buffer(8)]],
                            constant uint& SW [[buffer(9)]],
                            constant uint& PH [[buffer(10)]],
                            constant uint& PW [[buffer(11)]],
                            constant uint& OH [[buffer(12)]],
                            constant uint& OW [[buffer(13)]],
                            uint id [[thread_position_in_grid]]) {
    uint nout = N * OH * OW * C;
    if (id >= nout) return;
    uint c = id % C;
    uint ow = (id / C) % OW;
    uint oh = (id / (C * OW)) % OH;
    uint n = id / (C * OW * OH);
    float peak = -3.402823e+38f;
    for (uint kh = 0; kh < KH; ++kh) {
        for (uint kw = 0; kw < KW; ++kw) {
            int ih = int(oh * SH) - int(PH) + int(kh);
            int iw = int(ow * SW) - int(PW) + int(kw);
            if (ih < 0 || iw < 0 || ih >= int(H) || iw >= int(W)) continue;
            float v = input[((n * H + uint(ih)) * W + uint(iw)) * C + c];
            peak = max(peak, v);
        }
    }
    output[id] = peak;
}

kernel void avg_pool2d_nhwc(device const float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant uint& N [[buffer(2)]],
                            constant uint& H [[buffer(3)]],
                            constant uint& W [[buffer(4)]],
                            constant uint& C [[buffer(5)]],
                            constant uint& KH [[buffer(6)]],
                            constant uint& KW [[buffer(7)]],
                            constant uint& SH [[buffer(8)]],
                            constant uint& SW [[buffer(9)]],
                            constant uint& PH [[buffer(10)]],
                            constant uint& PW [[buffer(11)]],
                            constant uint& OH [[buffer(12)]],
                            constant uint& OW [[buffer(13)]],
                            uint id [[thread_position_in_grid]]) {
    uint nout = N * OH * OW * C;
    if (id >= nout) return;
    uint c = id % C;
    uint ow = (id / C) % OW;
    uint oh = (id / (C * OW)) % OH;
    uint n = id / (C * OW * OH);
    float total = 0.0f;
    uint seen = 0;
    for (uint kh = 0; kh < KH; ++kh) {
        for (uint kw = 0; kw < KW; ++kw) {
            int ih = int(oh * SH) - int(PH) + int(kh);
            int iw = int(ow * SW) - int(PW) + int(kw);
            if (ih < 0 || iw < 0 || ih >= int(H) || iw >= int(W)) continue;
            total += input[((n * H + uint(ih)) * W + uint(iw)) * C + c];
            seen += 1;
        }
    }
    output[id] = seen > 0 ? total / float(seen) : 0.0f;
}

kernel void dropout_apply(device const float* input [[buffer(0)]],
                          device const float* noise [[buffer(1)]],
                          device float* output [[buffer(2)]],
                          constant uint& count [[buffer(3)]],
                          constant uint& keep_bits [[buffer(4)]],
                          constant uint& scale_bits [[buffer(5)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float keep = as_type<float>(keep_bits);
    float scale = as_type<float>(scale_bits);
    output[id] = noise[id] < keep ? input[id] * scale : 0.0f;
}

kernel void mse_delta(device const float* prediction [[buffer(0)]],
                      device const float* target [[buffer(1)]],
                      device float* delta [[buffer(2)]],
                      constant uint& count [[buffer(3)]],
                      constant uint& scale_bits [[buffer(4)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float s = as_type<float>(scale_bits);
    delta[id] = s * (prediction[id] - target[id]);
}

kernel void eye_fill(device float* output [[buffer(0)]],
                     constant uint& n [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {
    uint count = n * n;
    if (id >= count) return;
    output[id] = (id / n == id % n) ? 1.0f : 0.0f;
}

kernel void flip_last(device const float* input [[buffer(0)]],
                      device float* output [[buffer(1)]],
                      constant uint& rows [[buffer(2)]],
                      constant uint& inner [[buffer(3)]],
                      uint id [[thread_position_in_grid]]) {
    uint count = rows * inner;
    if (id >= count) return;
    uint col = id % inner;
    uint row = id / inner;
    output[id] = input[row * inner + (inner - 1 - col)];
}

kernel void upsample_nearest2d_nhwc(device const float* input [[buffer(0)]],
                                    device float* output [[buffer(1)]],
                                    constant uint& N [[buffer(2)]],
                                    constant uint& H [[buffer(3)]],
                                    constant uint& W [[buffer(4)]],
                                    constant uint& C [[buffer(5)]],
                                    constant uint& SH [[buffer(6)]],
                                    constant uint& SW [[buffer(7)]],
                                    constant uint& OH [[buffer(8)]],
                                    constant uint& OW [[buffer(9)]],
                                    uint id [[thread_position_in_grid]]) {
    uint count = N * OH * OW * C;
    if (id >= count) return;
    uint c = id % C;
    uint ow = (id / C) % OW;
    uint oh = (id / (C * OW)) % OH;
    uint n = id / (C * OW * OH);
    uint ih = oh / SH;
    uint iw = ow / SW;
    output[id] = input[((n * H + ih) * W + iw) * C + c];
}

kernel void fill_from_scalar(device const float* scalar [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             constant uint& count [[buffer(2)]],
                             uint id [[thread_position_in_grid]]) {
    if (id < count) output[id] = scalar[0];
}

kernel void broadcast_last(device const float* reduced [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant uint& rows [[buffer(2)]],
                           constant uint& columns [[buffer(3)]],
                           uint id [[thread_position_in_grid]]) {
    uint count = rows * columns;
    if (id >= count) return;
    output[id] = reduced[id / columns];
}

kernel void scatter_add_rows(device const float* upstream [[buffer(0)]],
                             device const float* indices [[buffer(1)]],
                             device float* dest [[buffer(2)]],
                             constant uint& gathered [[buffer(3)]],
                             constant uint& width [[buffer(4)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= gathered * width) return;
    uint r = id / width;
    uint c = id % width;
    uint src_row = uint(indices[r]);
    atomic_fetch_add_explicit(
        (device atomic<float>*)(dest + src_row * width + c),
        upstream[id],
        memory_order_relaxed
    );
}

kernel void moments_group(device const float* input [[buffer(0)]],
                          device float* mean [[buffer(1)]],
                          device float* variance [[buffer(2)]],
                          constant uint& N [[buffer(3)]],
                          constant uint& spatial [[buffer(4)]],
                          constant uint& C [[buffer(5)]],
                          constant uint& groups [[buffer(6)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= N * groups) return;
    uint gs = C / groups;
    uint n = id / groups;
    uint g = id % groups;
    uint count = spatial * gs;
    float total = 0.0f;
    float sq = 0.0f;
    for (uint hw = 0; hw < spatial; ++hw) {
        uint base = (n * spatial + hw) * C + g * gs;
        for (uint c = 0; c < gs; ++c) {
            float v = input[base + c];
            total += v;
            sq += v * v;
        }
    }
    float m = total / float(count);
    mean[id] = m;
    variance[id] = sq / float(count) - m * m;
}

kernel void group_norm_forward(device const float* input [[buffer(0)]],
                               device const float* mean [[buffer(1)]],
                               device const float* variance [[buffer(2)]],
                               device const float* gamma [[buffer(3)]],
                               device const float* beta [[buffer(4)]],
                               device float* output [[buffer(5)]],
                               constant uint& count [[buffer(6)]],
                               constant uint& spatial [[buffer(7)]],
                               constant uint& C [[buffer(8)]],
                               constant uint& groups [[buffer(9)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint c = id % C;
    uint n = id / (C * spatial);
    uint gs = C / groups;
    uint g = c / gs;
    uint slot = n * groups + g;
    float inv = rsqrt(variance[slot] + 1e-5f);
    output[id] = gamma[c] * (input[id] - mean[slot]) * inv + beta[c];
}

kernel void bce_logits_sample(device const float* logits [[buffer(0)]],
                              device const float* target [[buffer(1)]],
                              device float* out [[buffer(2)]],
                              constant uint& count [[buffer(3)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float x = logits[id];
    float t = target[id];
    float ax = fabs(x);
    out[id] = max(x, 0.0f) - x * t + log(1.0f + exp(-ax));
}

kernel void bce_logits_backward(device const float* logits [[buffer(0)]],
                                device const float* target [[buffer(1)]],
                                device const float* upstream [[buffer(2)]],
                                device float* grad [[buffer(3)]],
                                constant uint& count [[buffer(4)]],
                                uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float x = logits[id];
    float t = target[id];
    float sig = 1.0f / (1.0f + exp(-x));
    grad[id] = (sig - t) * upstream[id];
}

kernel void bce_prob_backward(device const float* pred [[buffer(0)]],
                              device const float* target [[buffer(1)]],
                              device const float* upstream [[buffer(2)]],
                              device float* grad [[buffer(3)]],
                              constant uint& count [[buffer(4)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float p = clamp(pred[id], 1e-7f, 1.0f - 1e-7f);
    float t = target[id];
    grad[id] = (-t / p + (1.0f - t) / (1.0f - p)) * upstream[id];
}
"""

let ATTN_SOURCE = """#include <metal_stdlib>
using namespace metal;

kernel void attention_scale(device float* scores [[buffer(0)]],
                            constant uint& count [[buffer(1)]],
                            constant float& scale [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
    if (id < count) scores[id] *= scale;
}

kernel void split_heads(device const float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        constant uint& seq [[buffer(2)]],
                        constant uint& heads [[buffer(3)]],
                        constant uint& hd [[buffer(4)]],
                        uint id [[thread_position_in_grid]]) {
    uint count = heads * seq * hd;
    if (id >= count) return;
    uint d = id % hd;
    uint s = (id / hd) % seq;
    uint h = id / (hd * seq);
    uint dim = heads * hd;
    output[id] = input[s * dim + h * hd + d];
}

kernel void merge_heads(device const float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        constant uint& seq [[buffer(2)]],
                        constant uint& heads [[buffer(3)]],
                        constant uint& hd [[buffer(4)]],
                        uint id [[thread_position_in_grid]]) {
    uint count = heads * seq * hd;
    if (id >= count) return;
    uint d = id % hd;
    uint s = (id / hd) % seq;
    uint h = id / (hd * seq);
    uint dim = heads * hd;
    output[s * dim + h * hd + d] = input[id];
}

kernel void mha_scores(device const float* query [[buffer(0)]],
                       device const float* key [[buffer(1)]],
                       device float* scores [[buffer(2)]],
                       constant uint& seq [[buffer(3)]],
                       constant uint& heads [[buffer(4)]],
                       constant uint& hd [[buffer(5)]],
                       constant uint& scale_bits [[buffer(6)]],
                       uint id [[thread_position_in_grid]]) {
    uint count = heads * seq * seq;
    if (id >= count) return;
    uint j = id % seq;
    uint i = (id / seq) % seq;
    uint h = id / (seq * seq);
    uint dim = heads * hd;
    uint qbase = i * dim + h * hd;
    uint kbase = j * dim + h * hd;
    float acc = 0.0f;
    for (uint d = 0; d < hd; ++d) acc += query[qbase + d] * key[kbase + d];
    scores[id] = acc * as_type<float>(scale_bits);
}

kernel void mha_context(device const float* probs [[buffer(0)]],
                        device const float* value [[buffer(1)]],
                        device float* output [[buffer(2)]],
                        constant uint& seq [[buffer(3)]],
                        constant uint& heads [[buffer(4)]],
                        constant uint& hd [[buffer(5)]],
                        uint id [[thread_position_in_grid]]) {
    uint dim = heads * hd;
    uint count = seq * dim;
    if (id >= count) return;
    uint col = id % dim;
    uint i = id / dim;
    uint h = col / hd;
    uint d = col % hd;
    uint pbase = (h * seq + i) * seq;
    uint vcol = h * hd + d;
    float acc = 0.0f;
    for (uint j = 0; j < seq; ++j)
        acc += probs[pbase + j] * value[j * dim + vcol];
    output[id] = acc;
}

kernel void flash_mha(device const float* query [[buffer(0)]],
                      device const float* key [[buffer(1)]],
                      device const float* value [[buffer(2)]],
                      device float* output [[buffer(3)]],
                      constant uint& seq [[buffer(4)]],
                      constant uint& heads [[buffer(5)]],
                      constant uint& hd [[buffer(6)]],
                      constant uint& scale_bits [[buffer(7)]],
                      uint lid [[thread_index_in_threadgroup]],
                      uint wid [[threadgroup_position_in_grid]]) {
    constexpr uint BR = 16;
    constexpr uint BC = 16;
    constexpr uint HDMAX = 64;
    constexpr uint GROUP = 256;
    uint q_tiles = (seq + BR - 1) / BR;
    uint head = wid / q_tiles;
    uint qtile = wid % q_tiles;
    uint dim = heads * hd;
    uint q0 = qtile * BR;
    float scale = as_type<float>(scale_bits);
    threadgroup float Qs[BR * HDMAX];
    threadgroup float Ks[BC * HDMAX];
    threadgroup float S[BR * BC];
    threadgroup float m_i[BR];
    threadgroup float l_i[BR];
    threadgroup float acc[BR * HDMAX];
    for (uint w = lid; w < BR * HDMAX; w += GROUP) acc[w] = 0.0f;
    if (lid < BR) {
        m_i[lid] = -3.402823e+38f;
        l_i[lid] = 0.0f;
    }
    for (uint w = lid; w < BR * hd; w += GROUP) {
        uint r = w / hd;
        uint d = w % hd;
        uint row = q0 + r;
        Qs[r * HDMAX + d] = (row < seq) ? query[row * dim + head * hd + d] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint ti = lid / BC;
    uint tj = lid % BC;
    uint qrow = q0 + ti;
    for (uint k0 = 0; k0 < seq; k0 += BC) {
        for (uint w = lid; w < BC * hd; w += GROUP) {
            uint r = w / hd;
            uint d = w % hd;
            uint row = k0 + r;
            Ks[r * HDMAX + d] = (row < seq) ? key[row * dim + head * hd + d] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint krow = k0 + tj;
        float s = -3.402823e+38f;
        if (qrow < seq && krow < seq) {
            s = 0.0f;
            uint d = 0;
            for (; d + 4 <= hd; d += 4) {
                s += Qs[ti * HDMAX + d] * Ks[tj * HDMAX + d];
                s += Qs[ti * HDMAX + d + 1] * Ks[tj * HDMAX + d + 1];
                s += Qs[ti * HDMAX + d + 2] * Ks[tj * HDMAX + d + 2];
                s += Qs[ti * HDMAX + d + 3] * Ks[tj * HDMAX + d + 3];
            }
            for (; d < hd; ++d) s += Qs[ti * HDMAX + d] * Ks[tj * HDMAX + d];
            s *= scale;
        }
        S[ti * BC + tj] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tj == 0) {
            float row_max = S[ti * BC];
            for (uint j = 1; j < BC; ++j) row_max = max(row_max, S[ti * BC + j]);
            float m_new = max(m_i[ti], row_max);
            float alpha = (m_i[ti] < -1.0e30f) ? 0.0f : exp(m_i[ti] - m_new);
            float l_part = 0.0f;
            for (uint j = 0; j < BC; ++j) {
                float p = (k0 + j < seq) ? exp(S[ti * BC + j] - m_new) : 0.0f;
                S[ti * BC + j] = p;
                l_part += p;
            }
            if (qrow < seq) {
                for (uint d = 0; d < hd; ++d) acc[ti * HDMAX + d] *= alpha;
                l_i[ti] = l_i[ti] * alpha + l_part;
                m_i[ti] = m_new;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint w = lid; w < BC * hd; w += GROUP) {
            uint r = w / hd;
            uint d = w % hd;
            uint row = k0 + r;
            Ks[r * HDMAX + d] = (row < seq) ? value[row * dim + head * hd + d] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint w = lid; w < BR * hd; w += GROUP) {
            uint r = w / hd;
            uint d = w % hd;
            if (q0 + r < seq) {
                float add = 0.0f;
                for (uint j = 0; j < BC; ++j) add += S[r * BC + j] * Ks[j * HDMAX + d];
                acc[r * HDMAX + d] += add;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint w = lid; w < BR * hd; w += GROUP) {
        uint r = w / hd;
        uint d = w % hd;
        uint row = q0 + r;
        if (row < seq) {
            float inv = (l_i[r] > 0.0f) ? 1.0f / l_i[r] : 0.0f;
            output[row * dim + head * hd + d] = acc[r * HDMAX + d] * inv;
        }
    }
}

kernel void flash_mha_wide(device const float* query [[buffer(0)]],
                           device const float* key [[buffer(1)]],
                           device const float* value [[buffer(2)]],
                           device float* output [[buffer(3)]],
                           constant uint& seq [[buffer(4)]],
                           constant uint& heads [[buffer(5)]],
                           constant uint& hd [[buffer(6)]],
                           constant uint& scale_bits [[buffer(7)]],
                           uint lid [[thread_index_in_threadgroup]],
                           uint wid [[threadgroup_position_in_grid]]) {
    constexpr uint BR = 16;
    constexpr uint BC = 32;
    constexpr uint HDMAX = 64;
    constexpr uint GROUP = 256;
    uint q_tiles = (seq + BR - 1) / BR;
    uint head = wid / q_tiles;
    uint qtile = wid % q_tiles;
    uint dim = heads * hd;
    uint q0 = qtile * BR;
    float scale = as_type<float>(scale_bits);
    threadgroup half Qs[BR * HDMAX];
    threadgroup half Ks[BC * HDMAX];
    threadgroup float S[BR * BC];
    threadgroup float m_i[BR];
    threadgroup float l_i[BR];
    threadgroup float acc[BR * HDMAX];
    for (uint w = lid; w < BR * HDMAX; w += GROUP) acc[w] = 0.0f;
    if (lid < BR) {
        m_i[lid] = -3.402823e+38f;
        l_i[lid] = 0.0f;
    }
    for (uint w = lid; w < BR * hd; w += GROUP) {
        uint r = w / hd;
        uint d = w % hd;
        uint row = q0 + r;
        Qs[r * HDMAX + d] = half((row < seq) ? query[row * dim + head * hd + d] : 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k0 = 0; k0 < seq; k0 += BC) {
        for (uint w = lid; w < BC * hd; w += GROUP) {
            uint r = w / hd;
            uint d = w % hd;
            uint row = k0 + r;
            Ks[r * HDMAX + d] = half((row < seq) ? key[row * dim + head * hd + d] : 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint rep = 0; rep < 2; ++rep) {
            uint ti = (lid / BC) + rep * (GROUP / BC);
            uint tj = lid % BC;
            uint qrow = q0 + ti;
            uint krow = k0 + tj;
            float s = -3.402823e+38f;
            if (qrow < seq && krow < seq) {
                s = 0.0f;
                uint d = 0;
                for (; d + 8 <= hd; d += 8) {
                    s += float(Qs[ti * HDMAX + d]) * float(Ks[tj * HDMAX + d]);
                    s += float(Qs[ti * HDMAX + d + 1]) * float(Ks[tj * HDMAX + d + 1]);
                    s += float(Qs[ti * HDMAX + d + 2]) * float(Ks[tj * HDMAX + d + 2]);
                    s += float(Qs[ti * HDMAX + d + 3]) * float(Ks[tj * HDMAX + d + 3]);
                    s += float(Qs[ti * HDMAX + d + 4]) * float(Ks[tj * HDMAX + d + 4]);
                    s += float(Qs[ti * HDMAX + d + 5]) * float(Ks[tj * HDMAX + d + 5]);
                    s += float(Qs[ti * HDMAX + d + 6]) * float(Ks[tj * HDMAX + d + 6]);
                    s += float(Qs[ti * HDMAX + d + 7]) * float(Ks[tj * HDMAX + d + 7]);
                }
                for (; d < hd; ++d) s += float(Qs[ti * HDMAX + d]) * float(Ks[tj * HDMAX + d]);
                s *= scale;
            }
            S[ti * BC + tj] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < BR) {
            uint ti = lid;
            uint qrow = q0 + ti;
            float row_max = S[ti * BC];
            for (uint j = 1; j < BC; ++j) row_max = max(row_max, S[ti * BC + j]);
            float m_new = max(m_i[ti], row_max);
            float alpha = (m_i[ti] < -1.0e30f) ? 0.0f : exp(m_i[ti] - m_new);
            float l_part = 0.0f;
            for (uint j = 0; j < BC; ++j) {
                float p = (k0 + j < seq) ? exp(S[ti * BC + j] - m_new) : 0.0f;
                S[ti * BC + j] = p;
                l_part += p;
            }
            if (qrow < seq) {
                for (uint d = 0; d < hd; ++d) acc[ti * HDMAX + d] *= alpha;
                l_i[ti] = l_i[ti] * alpha + l_part;
                m_i[ti] = m_new;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint w = lid; w < BC * hd; w += GROUP) {
            uint r = w / hd;
            uint d = w % hd;
            uint row = k0 + r;
            Ks[r * HDMAX + d] = half((row < seq) ? value[row * dim + head * hd + d] : 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint w = lid; w < BR * hd; w += GROUP) {
            uint r = w / hd;
            uint d = w % hd;
            if (q0 + r < seq) {
                float add = 0.0f;
                uint j = 0;
                for (; j + 4 <= BC; j += 4) {
                    add += S[r * BC + j] * float(Ks[j * HDMAX + d]);
                    add += S[r * BC + j + 1] * float(Ks[(j + 1) * HDMAX + d]);
                    add += S[r * BC + j + 2] * float(Ks[(j + 2) * HDMAX + d]);
                    add += S[r * BC + j + 3] * float(Ks[(j + 3) * HDMAX + d]);
                }
                for (; j < BC; ++j) add += S[r * BC + j] * float(Ks[j * HDMAX + d]);
                acc[r * HDMAX + d] += add;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint w = lid; w < BR * hd; w += GROUP) {
        uint r = w / hd;
        uint d = w % hd;
        uint row = q0 + r;
        if (row < seq) {
            float inv = (l_i[r] > 0.0f) ? 1.0f / l_i[r] : 0.0f;
            output[row * dim + head * hd + d] = acc[r * HDMAX + d] * inv;
        }
    }
}

kernel void tril_square(device const float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        constant uint& rows [[buffer(2)]],
                        constant uint& cols [[buffer(3)]],
                        uint id [[thread_position_in_grid]]) {
    uint count = rows * cols;
    if (id >= count) return;
    uint r = id / cols;
    uint c = id % cols;
    output[id] = (c <= r) ? input[id] : 0.0f;
}
"""

var _kernel_cache: dict[str, Kernel] = {}

fn f32_bits(value: float) -> int {
    if value != value { return 0x7FC00000 }
    var mag = value
    var sign = 0
    if mag < 0.0 {
        sign = 1
        mag = -mag
    }
    if mag == 0.0 { return sign << 31 }
    if mag >= 3.4028234663852886e38 { return sign << 31 | 0x7F800000 }
    if mag <= 1.401298464324817e-45 { return sign << 31 }
    let split = __prim__.f64_frexp(mag)
    let biased = split[1] + 126
    if biased <= 0 {
        let frac = int(__prim__.f64_ldexp(mag, 149))
        return sign << 31 | frac & 0x7FFFFF
    }
    if biased >= 255 { return sign << 31 | 0x7F800000 }
    let frac = int((split[0] * 2.0 - 1.0) * 8388608.0 + 0.5)
    return sign << 31 | biased << 23 | frac & 0x7FFFFF
}

#: The compiled pipeline for one entry point in this module's shader library.
#:
#: Keyed by the entry name alone: there is exactly one `SOURCE`, and building a
#: key that included it meant concatenating and hashing 36 KB on every single
#: dispatch — 88 microseconds of host time per elementwise op, against 6 for
#: the dispatch itself.
pub fn cached_kernel(entry: str) -> Kernel {
    let existing = _kernel_cache.get(entry, null)
    if existing is not null { return existing }
    let kernel = Kernel.compile(SOURCE, entry)
    _kernel_cache[entry] = kernel
    return kernel
}

var _attn_cache: dict[str, Kernel] = {}

#: The same, for the attention shader library.
fn cached_attn_kernel(entry: str) -> Kernel {
    let existing = _attn_cache.get(entry, null)
    if existing is not null { return existing }
    let kernel = Kernel.compile(ATTN_SOURCE, entry)
    _attn_cache[entry] = kernel
    return kernel
}

fn binary(a: Tensor, b: Tensor, operation: int) -> Tensor {
    if not a.same_shape(b) { throw ValueError(f"tensor shapes differ: {a.shape} and {b.shape}") }
    let output = Tensor(a.shape)
    let kernel = cached_kernel("binary_op")
    kernel.dispatch_async([a.buffer(), b.buffer(), output.buffer()], a.size, [a.size, operation])
    return output
}

fn unary(input: Tensor, operation: int) -> Tensor {
    let output = Tensor(input.shape)
    let kernel = cached_kernel("unary_op")
    kernel.dispatch_async([input.buffer(), output.buffer()], input.size, [input.size, operation])
    return output
}

fn is_trailing_suffix(small: list[int], large: list[int]) -> bool {
    if small.len() > large.len() { return false }
    let offset = large.len() - small.len()
    for index in 0..small.len() {
        if small[index] != large[offset + index] { return false }
    }
    return true
}

fn broadcast_add(a: Tensor, b: Tensor) -> Tensor {
    let output = Tensor(a.shape)
    let kernel = cached_kernel("broadcast_add")
    kernel.dispatch_async([a.buffer(), b.buffer(), output.buffer()], a.size, [b.size, a.size])
    return output
}

fn binary_backward(a: Tensor, b: Tensor, upstream: Tensor, operation: int) -> list[Tensor] {
    let grad_a = Tensor(a.shape)
    let grad_b = Tensor(b.shape)
    let kernel = cached_kernel("binary_backward_op")
    kernel.dispatch_async(
        [a.buffer(), b.buffer(), upstream.buffer(), grad_a.buffer(), grad_b.buffer()],
        a.size,
        [a.size, operation]
    )
    return [grad_a, grad_b]
}

fn reduce_leading(upstream: Tensor, small_shape: list[int], out: Tensor? = null) -> Tensor {
    let inner = element_count(small_shape)
    let outer = upstream.size // inner
    let output = if out is not null {
        if out.size != inner {
            throw ValueError(f"reduce_leading out size {out.size} does not match {inner}")
        }
        out
    } else {
        Tensor(small_shape)
    }
    #: One thread a column reads every row of that column, which is fine while
    #: the columns outnumber the rows -- a dense layer's bias gradient is a
    #: hundred rows against a thousand columns. A convolution's is the other way
    #: round: thirty-two columns against a hundred and thirty thousand rows, so
    #: thirty-two threads did four million serial reads while the rest of the
    #: GPU sat idle, and it cost twelve times the whole rest of the layer.
    #: Splitting the rows first gives the machine something to do.
    let SPLIT_ROWS = 4096
    if outer > SPLIT_ROWS and inner * 64 < outer {
        let chunks = min(256, (outer + SPLIT_ROWS - 1) // SPLIT_ROWS)
        let partials = Tensor([chunks * inner])
        cached_kernel("reduce_leading_split").dispatch_async(
            [upstream.buffer(), partials.buffer()],
            inner * chunks,
            [outer, inner, chunks]
        )
        if inner % 4 == 0 {
            cached_kernel("reduce_leading4").dispatch_async(
                [partials.buffer(), output.buffer()],
                inner // 4,
                [chunks, inner]
            )
        } else {
            cached_kernel("reduce_leading").dispatch_async(
                [partials.buffer(), output.buffer()],
                inner,
                [chunks, inner]
            )
        }
        partials.free()
        return output
    }
    if inner % 4 == 0 {
        let kernel = cached_kernel("reduce_leading4")
        kernel.dispatch_async([upstream.buffer(), output.buffer()], inner // 4, [outer, inner])
    } else {
        let kernel = cached_kernel("reduce_leading")
        kernel.dispatch_async([upstream.buffer(), output.buffer()], inner, [outer, inner])
    }
    return output
}

#: Elementwise sum of equal-shaped tensors, or of `a` with a `b` whose shape
#: is a trailing suffix of `a`'s (for example `[units]` against `[batch,
#: units]`).
pub fn add(a: Tensor, b: Tensor) -> Tensor {
    let broadcasting = not a.same_shape(b)
    if broadcasting and not is_trailing_suffix(b.shape, a.shape) {
        throw ValueError(f"cannot broadcast {b.shape} against {a.shape}")
    }
    let output = if broadcasting { broadcast_add(a, b) } else { binary(a, b, 0) }
    if a.requires_grad or b.requires_grad {
        output.attach_creator(GradNode([a, b], |upstream| if broadcasting {
            [unary(upstream, 0), reduce_leading(upstream, b.shape)]
        } else {
            binary_backward(a, b, upstream, 0)
        }))
    }
    return output
}

#: Elementwise difference of equal-shaped tensors.
pub fn subtract(a: Tensor, b: Tensor) -> Tensor {
    let output = binary(a, b, 1)
    if a.requires_grad or b.requires_grad {
        output.attach_creator(GradNode([a, b], |upstream| binary_backward(a, b, upstream, 1)))
    }
    return output
}

#: Elementwise product of equal-shaped tensors.
pub fn multiply(a: Tensor, b: Tensor) -> Tensor {
    let output = binary(a, b, 2)
    if a.requires_grad or b.requires_grad {
        output.attach_creator(GradNode([a, b], |upstream| binary_backward(a, b, upstream, 2)))
    }
    return output
}

#: Elementwise quotient of equal-shaped tensors.
pub fn divide(a: Tensor, b: Tensor) -> Tensor {
    let output = binary(a, b, 3)
    if a.requires_grad or b.requires_grad {
        output.attach_creator(GradNode([a, b], |upstream| binary_backward(a, b, upstream, 3)))
    }
    return output
}

#: Multiply every element by one host scalar.
pub fn scale(input: Tensor, value: float) -> Tensor {
    let output = Tensor(input.shape)
    let kernel = cached_kernel("scalar_mul_bits")
    kernel.dispatch_async(
        [input.buffer(), output.buffer()],
        input.size,
        [input.size, f32_bits(value)]
    )
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [scale(upstream, value)]))
    }
    return output
}

#: Reinterpret a tensor as `shape` without copying device memory.
pub fn reshape(input: Tensor, shape: list[int]) -> Tensor {
    if element_count(shape) != input.size {
        throw ValueError(f"cannot reshape {input.shape} into {shape}")
    }
    let output = Tensor(shape, storage: input.buffer().view(0, input.size))
    if input.requires_grad {
        let original_shape = input.shape
        output.attach_creator(GradNode([input], |upstream| [reshape(upstream, original_shape)]))
    }
    return output
}

#: Report the kernel family selected for a matrix shape.
#:
#: Side-effect free: it reports the plan a shape already has, or the static
#: heuristic when it has never run, so profilers and benchmark tools can
#: explain a dispatch cliff without changing one.
pub fn matmul_backend(
    rows: int,
    inner: int,
    columns: int,
    trans_a: bool = false,
    trans_b: bool = false,
    epilogue: int = 0
) -> str {
    return gemm.planned_backend(rows, inner, columns, trans_a: trans_a, trans_b: trans_b)
}

fn raw_matmul(
    a: Tensor,
    b: Tensor,
    trans_a: bool = false,
    trans_b: bool = false,
    bias: Tensor? = null,
    epilogue: int = 0,
    out: Tensor? = null
) -> Tensor {
    let rows = if trans_a { a.shape[1] } else { a.shape[0] }
    let inner = if trans_a { a.shape[0] } else { a.shape[1] }
    let inner_b = if trans_b { b.shape[1] } else { b.shape[0] }
    let columns = if trans_b { b.shape[0] } else { b.shape[1] }
    if inner != inner_b {
        throw ValueError(f"matmul inner dimensions differ: {a.shape} and {b.shape}")
    }
    let output = if out is not null {
        if out.shape != [rows, columns] {
            throw ValueError(f"matmul out shape {out.shape} does not match [{rows}, {columns}]")
        }
        out
    } else {
        Tensor([rows, columns])
    }
    let bias_buffer = if bias is not null { bias.buffer() } else { output.buffer() }
    gemm.dispatch(
        a.buffer(),
        b.buffer(),
        output.buffer(),
        bias_buffer,
        rows,
        inner,
        columns,
        trans_a: trans_a,
        trans_b: trans_b,
        epilogue: epilogue
    )
    return output
}

fn raw_transpose(input: Tensor) -> Tensor {
    let rows = input.shape[0]
    let columns = input.shape[1]
    let output = Tensor([columns, rows])
    let kernel = cached_kernel("matrix_transpose")
    kernel.dispatch_async([input.buffer(), output.buffer()], input.size, [rows, columns])
    return output
}

#: Matrix product of two rank-two tensors.
#:
#: `trans_a` / `trans_b` read `a` or `b` as if transposed without copying.
#: Gradient graphs are only attached for the untransposed product.
pub fn matmul(
    a: Tensor,
    b: Tensor,
    trans_a: bool = false,
    trans_b: bool = false,
    out: Tensor? = null
) -> Tensor {
    if a.rank() != 2 or b.rank() != 2 {
        throw ValueError(f"matmul needs rank-two tensors, got {a.shape} and {b.shape}")
    }
    let a_inner = if trans_a { a.shape[0] } else { a.shape[1] }
    let b_inner = if trans_b { b.shape[1] } else { b.shape[0] }
    if a_inner != b_inner {
        throw ValueError(f"matmul inner dimensions differ: {a.shape} and {b.shape}")
    }
    let output = raw_matmul(a, b, trans_a: trans_a, trans_b: trans_b, out: out)
    if out is null and (a.requires_grad or b.requires_grad) {
        output.attach_creator(GradNode([a, b], fn(upstream: Tensor) -> list[Tensor] {
            if trans_a and trans_b {
                return [raw_matmul(b, upstream, trans_a: true, trans_b: true), raw_matmul(upstream, a, trans_a: true, trans_b: true)]
            }
            if trans_a {
                return [raw_matmul(b, upstream, trans_b: true), raw_matmul(a, upstream)]
            }
            if trans_b {
                return [raw_matmul(upstream, b), raw_matmul(upstream, a, trans_a: true)]
            }
            return [raw_matmul(upstream, b, trans_b: true), raw_matmul(a, upstream, trans_a: true)]
        }))
    }
    return output
}

#: Transpose a rank-two tensor.
pub fn transpose(input: Tensor) -> Tensor {
    if input.rank() != 2 { throw ValueError(f"transpose needs rank two, got {input.shape}") }
    let output = raw_transpose(input)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [raw_transpose(upstream)]))
    }
    return output
}

fn unary_backward(saved: Tensor, upstream: Tensor, operation: int) -> Tensor {
    let grad_input = Tensor(saved.shape)
    if operation == 1 and saved.size % 4 == 0 {
        let kernel = cached_kernel("unary_backward_relu4")
        kernel.dispatch_async(
            [saved.buffer(), upstream.buffer(), grad_input.buffer()],
            saved.size // 4,
            [saved.size]
        )
        return grad_input
    }
    let kernel = cached_kernel("unary_backward_op")
    kernel.dispatch_async(
        [saved.buffer(), upstream.buffer(), grad_input.buffer()],
        saved.size,
        [saved.size, operation]
    )
    return grad_input
}

#: Elementwise rectified linear activation.
pub fn relu(input: Tensor) -> Tensor {
    let output = unary(input, 1)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(output, upstream, 1)]))
    }
    return output
}

#: Elementwise logistic sigmoid activation.
pub fn sigmoid(input: Tensor) -> Tensor {
    let output = unary(input, 2)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(output, upstream, 2)]))
    }
    return output
}

#: Elementwise hyperbolic tangent activation.
pub fn tanh(input: Tensor) -> Tensor {
    let output = unary(input, 3)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(output, upstream, 3)]))
    }
    return output
}

#: Affine transform `x @ weights + bias` with an optional fused activation.
#:
#: `activation` matches `Dense.activation_code`: 0 linear, 1 ReLU, 2 sigmoid,
#: 3 tanh. Backward skips GEMMs for inputs that do not require a gradient,
#: so a first-layer feature batch never materializes `dX`.
pub fn dense_affine(x: Tensor, weights: Tensor, bias: Tensor, activation: int) -> Tensor {
    if activation < 0 or activation > 3 {
        throw ValueError(f"dense_affine activation must be 0..3, got {activation}")
    }
    record_dense_forward(x.shape, weights.shape)
    let output = raw_matmul(x, weights, bias: bias, epilogue: activation + 1)
    let need_x = x.requires_grad
    let need_w = weights.requires_grad
    let need_b = bias.requires_grad
    if need_x or need_w or need_b {
        var inputs: list[Tensor] = []
        if need_x { inputs.push(x) }
        if need_w { inputs.push(weights) }
        if need_b { inputs.push(bias) }
        output.attach_creator(GradNode(inputs, fn(upstream: Tensor) -> list[Tensor] {
            let affine_grad = if activation == 0 {
                upstream
            } else {
                unary_backward(output, upstream, activation)
            }
            var grads: list[Tensor] = []
            if need_x {
                grads.push(raw_matmul(affine_grad, weights, trans_b: true))
            }
            if need_w {
                grads.push(raw_matmul(x, affine_grad, trans_a: true, out: weights.grad))
            }
            if need_b {
                grads.push(reduce_leading(affine_grad, bias.shape, out: bias.grad))
            }
            if activation != 0 {
                affine_grad.free()
            }
            return grads
        }))
    }
    return output
}

#: Dense forward without autograd graph construction.
@trace
pub fn dense_static_forward(x: Tensor, weights: Tensor, bias: Tensor, activation: int) -> Tensor {
    if activation < 0 or activation > 3 {
        throw ValueError(f"dense_affine activation must be 0..3, got {activation}")
    }
    record_dense_forward(x.shape, weights.shape)
    return raw_matmul(x, weights, bias: bias, epilogue: activation + 1)
}

#: Dense backward into parameter `.grad` buffers; returns dInput (caller frees).
@trace
pub fn dense_static_backward(
    x: Tensor,
    weights: Tensor,
    bias: Tensor,
    output: Tensor,
    upstream: Tensor,
    activation: int
) -> Tensor {
    record_dense_backward(x.shape, weights.shape)
    let affine_grad = if activation == 0 {
        upstream
    } else {
        unary_backward(output, upstream, activation)
    }
    raw_matmul(x, affine_grad, trans_a: true, out: weights.grad)
    reduce_leading(affine_grad, bias.shape, out: bias.grad)
    let d_input = raw_matmul(affine_grad, weights, trans_b: true)
    if activation != 0 {
        affine_grad.free()
    }
    return d_input
}

var _warmed = false

#: Compile training kernels and run a tiny dispatch before the first timed epoch.
pub fn warmup_kernels() -> void {
    if _warmed { return }
    gemm.warmup()
    cached_kernel("unary_backward_op")
    cached_kernel("unary_backward_relu4")
    cached_kernel("reduce_leading")
    cached_kernel("reduce_leading4")
    cached_kernel("reduce_leading_split")
    cached_kernel("add_scaled_into")
    cached_kernel("add_correct_into")
    cached_kernel("im2col_nhwc")
    cached_kernel("batch_norm_forward")
    cached_kernel("moments_channel")
    cached_kernel("layer_norm_forward")
    cached_kernel("gather_rows")
    let wide = Tensor([64, 1024])
    let wide_w = Tensor([1024, 64])
    let wide_out = raw_matmul(wide, wide_w)
    let wide_k = Tensor([1024, 64])
    let wide_tn = raw_matmul(wide_k, wide_k, trans_a: true)
    let wide_nt = raw_matmul(wide, wide, trans_b: true)
    gpu.synchronize()
    wide_nt.free()
    wide_tn.free()
    wide_k.free()
    wide_out.free()
    wide.free()
    wide_w.free()
    let mid = Tensor([256, 256])
    let mid_out = raw_matmul(mid, mid, bias: mid, epilogue: 2)
    let mid_tn = raw_matmul(mid, mid, trans_a: true)
    let mid_nt = raw_matmul(mid, mid, trans_b: true)
    gpu.synchronize()
    mid_nt.free()
    mid_tn.free()
    mid_out.free()
    mid.free()
    let x = Tensor([512, 784])
    let weights = Tensor([784, 256])
    let bias = Tensor.zeros([256])
    let hidden = raw_matmul(x, weights, bias: bias, epilogue: 2)
    let grad_x = raw_matmul(hidden, weights, trans_b: true)
    let grad_w = raw_matmul(x, hidden, trans_a: true)
    let head_w = Tensor([256, 10])
    let head_b = Tensor.zeros([10])
    let logits = raw_matmul(hidden, head_w, bias: head_b, epilogue: 1)
    gpu.synchronize()
    logits.free()
    head_b.free()
    head_w.free()
    grad_w.free()
    grad_x.free()
    hidden.free()
    x.free()
    weights.free()
    bias.free()
    _warmed = true
}

#: Elementwise exponential.
pub fn exp(input: Tensor) -> Tensor {
    let output = unary(input, 4)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(output, upstream, 4)]))
    }
    return output
}

#: Elementwise natural logarithm.
pub fn log(input: Tensor) -> Tensor {
    let output = unary(input, 5)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(input, upstream, 5)]))
    }
    return output
}

#: Elementwise absolute value.
pub fn abs(input: Tensor) -> Tensor {
    let output = unary(input, 6)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(input, upstream, 6)]))
    }
    return output
}

#: Elementwise negation.
pub fn neg(input: Tensor) -> Tensor {
    let output = unary(input, 7)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(output, upstream, 7)]))
    }
    return output
}

#: Elementwise square root.
pub fn sqrt(input: Tensor) -> Tensor {
    let output = unary(input, 8)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(output, upstream, 8)]))
    }
    return output
}

#: Reciprocal square root.
pub fn rsqrt(input: Tensor) -> Tensor {
    let output = unary(input, 9)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(output, upstream, 9)]))
    }
    return output
}

#: Elementwise square.
pub fn square(input: Tensor) -> Tensor {
    let output = unary(input, 10)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(input, upstream, 10)]))
    }
    return output
}

#: Gaussian error linear unit (tanh approximation).
pub fn gelu(input: Tensor) -> Tensor {
    let output = unary(input, 11)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(input, upstream, 11)]))
    }
    return output
}

#: Sigmoid linear unit, `x * sigmoid(x)`.
pub fn silu(input: Tensor) -> Tensor {
    let output = unary(input, 12)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(input, upstream, 12)]))
    }
    return output
}

#: Leaky ReLU with negative slope 0.01.
pub fn leaky_relu(input: Tensor) -> Tensor {
    let output = unary(input, 13)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(input, upstream, 13)]))
    }
    return output
}

#: Clamp each value into `[lo, hi]`.
pub fn clamp(input: Tensor, lo: float, hi: float) -> Tensor {
    if lo > hi { throw ValueError(f"clamp lo {lo} is greater than hi {hi}") }
    let output = Tensor(input.shape)
    let kernel = cached_kernel("clamp_op")
    kernel.dispatch_async(
        [input.buffer(), output.buffer()],
        input.size,
        [input.size, f32_bits(lo), f32_bits(hi)]
    )
    if input.requires_grad {
        output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
            let g = Tensor(input.shape)
            let k = cached_kernel("clamp_backward")
            k.dispatch_async([input.buffer(), upstream.buffer(), g.buffer()], input.size, [input.size, f32_bits(lo), f32_bits(hi)])
            return [g]
        }))
    }
    return output
}

#: Route an upstream gradient to whichever side of a min/max comparison won.
fn select_grads(a: Tensor, b: Tensor, upstream: Tensor, want_max: int) -> list[Tensor] {
    let kernel = cached_kernel("select_backward")
    var grads: list[Tensor] = []
    for side in 0..2 {
        let grad = Tensor(a.shape)
        kernel.dispatch_async(
            [a.buffer(), b.buffer(), upstream.buffer(), grad.buffer()],
            a.size,
            [a.size, want_max, side]
        )
        grads.push(grad)
    }
    return grads
}

#: Elementwise maximum of equal-shaped tensors.
pub fn maximum(a: Tensor, b: Tensor) -> Tensor {
    if not a.same_shape(b) { throw ValueError(f"tensor shapes differ: {a.shape} and {b.shape}") }
    let output = Tensor(a.shape)
    let kernel = cached_kernel("maximum_op")
    kernel.dispatch_async([a.buffer(), b.buffer(), output.buffer()], a.size, [a.size])
    if a.requires_grad or b.requires_grad {
        output.attach_creator(GradNode([a, b], |upstream| select_grads(a, b, upstream, 1)))
    }
    return output
}

#: Elementwise minimum of equal-shaped tensors.
pub fn minimum(a: Tensor, b: Tensor) -> Tensor {
    if not a.same_shape(b) { throw ValueError(f"tensor shapes differ: {a.shape} and {b.shape}") }
    let output = Tensor(a.shape)
    let kernel = cached_kernel("minimum_op")
    kernel.dispatch_async([a.buffer(), b.buffer(), output.buffer()], a.size, [a.size])
    if a.requires_grad or b.requires_grad {
        output.attach_creator(GradNode([a, b], |upstream| select_grads(a, b, upstream, 0)))
    }
    return output
}

fn broadcast_scalar(source: Tensor, shape: list[int]) -> Tensor {
    let output = Tensor(shape)
    let kernel = cached_kernel("fill_from_scalar")
    kernel.dispatch_async([source.buffer(), output.buffer()], output.size, [output.size])
    return output
}

fn expand_last(reduced: Tensor, columns: int) -> Tensor {
    var shape: list[int] = []
    for dimension in reduced.shape { shape.push(dimension) }
    shape.push(columns)
    let output = Tensor(shape)
    let kernel = cached_kernel("broadcast_last")
    kernel.dispatch_async([reduced.buffer(), output.buffer()], output.size, [reduced.size, columns])
    return output
}

#: Sum of every element, or of the last axis when `axis` is `-1`.
pub fn sum(input: Tensor, axis: int? = null) -> Tensor {
    if axis is null {
        let groups = (input.size + 255) // 256
        let partials = Tensor([groups])
        let kernel = cached_kernel("reduce_sum_all")
        kernel.dispatch_async(
            [input.buffer(), partials.buffer()],
            groups * 256,
            [input.size],
            group_size: 256
        )
        let output = if groups == 1 {
            partials
        } else {
            let total = sum(partials)
            partials.free()
            total
        }
        if input.requires_grad {
            output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
                return [broadcast_scalar(upstream, input.shape)]
            }))
        }
        return output
    }
    let resolved = if axis < 0 { input.rank() + axis } else { axis }
    if resolved != input.rank() - 1 {
        throw ValueError("sum currently reduces all elements or the last axis")
    }
    let columns = input.shape[resolved]
    let rows = input.size // columns
    var out_shape: list[int] = []
    for i in 0..resolved { out_shape.push(input.shape[i]) }
    let output = Tensor(out_shape)
    let kernel = cached_kernel("reduce_sum_rows")
    kernel.dispatch_async([input.buffer(), output.buffer()], rows, [rows, columns])
    if input.requires_grad {
        output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
            return [expand_last(upstream, columns)]
        }))
    }
    return output
}

#: Mean of every element, or of the last axis.
pub fn mean(input: Tensor, axis: int? = null) -> Tensor {
    let reduced = sum(input, axis: axis)
    let count = if axis is null { float(input.size) } else { float(input.dim(-1)) }
    let scaled = scale(reduced, 1.0 / count)
    if not input.requires_grad { reduced.free() }
    return scaled
}

#: Concatenate tensors along `axis` (0 or the last axis).
pub fn cat(parts: list[Tensor], axis: int = 0) -> Tensor {
    if parts.len() == 0 { throw ValueError("cat needs at least one tensor") }
    let rank = parts[0].rank()
    let resolved = if axis < 0 { rank + axis } else { axis }
    if resolved != 0 and resolved != rank - 1 {
        throw ValueError("cat currently supports axis 0 or the last axis")
    }
    var along = 0
    for part in parts {
        if part.rank() != rank { throw ValueError("cat parts must have the same rank") }
        for i in 0..rank {
            if i == resolved { continue }
            if part.shape[i] != parts[0].shape[i] {
                throw ValueError(
                    f"cat shape mismatch at axis {i}: {part.shape} vs {parts[0].shape}"
                )
            }
        }
        along += part.shape[resolved]
    }
    var out_shape: list[int] = []
    for i in 0..rank {
        out_shape.push(if i == resolved { along } else { parts[0].shape[i] })
    }
    let output = Tensor(out_shape)
    var cursor = 0
    let kernel = cached_kernel("copy_range")
    var wants_grad = false
    for part in parts {
        if part.requires_grad { wants_grad = true }
    }
    if resolved == 0 {
        var offsets: list[int] = []
        for part in parts {
            offsets.push(cursor)
            kernel.dispatch_async(
                [part.buffer(), output.buffer()],
                part.size,
                [0, cursor, part.size]
            )
            cursor += part.size
        }
        if wants_grad {
            let shapes = [part.shape for part in parts]
            output.attach_creator(GradNode(parts, fn(upstream: Tensor) -> list[Tensor] {
                var grads: list[Tensor] = []
                let back = cached_kernel("copy_range")
                for (index, shape) in shapes.enumerate() {
                    let grad = Tensor(shape)
                    back.dispatch_async([upstream.buffer(), grad.buffer()], grad.size, [offsets[index], 0, grad.size])
                    grads.push(grad)
                }
                return grads
            }))
        }
        return output
    }
    let outer = parts[0].size // parts[0].shape[resolved]
    var prefix = 0
    var starts: list[int] = []
    let strided = cached_kernel("copy_strided_rows")
    for part in parts {
        let width = part.shape[resolved]
        starts.push(prefix)
        strided.dispatch_async(
            [part.buffer(), output.buffer()],
            outer * width,
            [outer, width, along, prefix]
        )
        prefix += width
    }
    if wants_grad {
        let shapes = [part.shape for part in parts]
        output.attach_creator(GradNode(parts, fn(upstream: Tensor) -> list[Tensor] {
            var grads: list[Tensor] = []
            let take = cached_kernel("take_strided_rows")
            for (index, shape) in shapes.enumerate() {
                let grad = Tensor(shape)
                let width = shape[resolved]
                take.dispatch_async([upstream.buffer(), grad.buffer()], outer * width, [outer, width, along, starts[index]])
                grads.push(grad)
            }
            return grads
        }))
    }
    return output
}

#: NHWC max pooling.
pub fn max_pool2d(input: Tensor, kernel_size: int, stride: int = 0, padding: int = 0) -> Tensor {
    return _pool2d(input, kernel_size, stride, padding, true)
}

#: NHWC average pooling (ignores padded holes).
pub fn avg_pool2d(input: Tensor, kernel_size: int, stride: int = 0, padding: int = 0) -> Tensor {
    return _pool2d(input, kernel_size, stride, padding, false)
}

fn _pool2d(input: Tensor, kernel_size: int, stride: int, padding: int, is_max: bool) -> Tensor {
    if input.rank() != 4 { throw ValueError(f"pool2d needs NHWC input, got {input.shape}") }
    if kernel_size <= 0 { throw ValueError("kernel_size must be positive") }
    let step = if stride <= 0 { kernel_size } else { stride }
    if padding < 0 { throw ValueError("padding must be non-negative") }
    let n = input.shape[0]
    let h = input.shape[1]
    let w = input.shape[2]
    let c = input.shape[3]
    let oh = conv2d_out_dim(h, kernel_size, step, padding)
    let ow = conv2d_out_dim(w, kernel_size, step, padding)
    if oh <= 0 or ow <= 0 { throw ValueError("pool2d produced an empty spatial size") }
    let output = Tensor([n, oh, ow, c])
    let entry = if is_max { "max_pool2d_nhwc" } else { "avg_pool2d_nhwc" }
    let kernel = cached_kernel(entry)
    kernel.dispatch_async(
        [input.buffer(), output.buffer()],
        n * oh * ow * c,
        [n, h, w, c, kernel_size, kernel_size, step, step, padding, padding, oh, ow]
    )
    if input.requires_grad {
        output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
            let grad = Tensor([n, h, w, c])
            let scalars = [n, h, w, c, kernel_size, kernel_size, step, step, padding, padding, oh, ow]
            if is_max {
                let back = cached_kernel("max_pool2d_backward_nhwc")
                back.dispatch_async([input.buffer(), upstream.buffer(), grad.buffer()], grad.size, scalars)
            } else {
                let back = cached_kernel("avg_pool2d_backward_nhwc")
                back.dispatch_async([upstream.buffer(), grad.buffer()], grad.size, scalars)
            }
            return [grad]
        }))
    }
    return output
}

#: Inverted dropout. `p` is the drop probability.
pub fn dropout(input: Tensor, p: float = 0.5, training: bool = true) -> Tensor {
    if p < 0.0 or p >= 1.0 { throw ValueError(f"dropout p must be in [0, 1), got {p}") }
    if not training or p == 0.0 {
        return reshape(input, input.shape)
    }
    let noise = Tensor(input.shape)
    noise.buffer().fill_uniform(0.0, 1.0, input.size + 1)
    let output = Tensor(input.shape)
    let keep = 1.0 - p
    let kernel = cached_kernel("dropout_apply")
    kernel.dispatch_async(
        [input.buffer(), noise.buffer(), output.buffer()],
        input.size,
        [input.size, f32_bits(keep), f32_bits(1.0 / keep)]
    )
    if input.requires_grad {
        output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
            let g = Tensor(input.shape)
            kernel.dispatch_async([upstream.buffer(), noise.buffer(), g.buffer()], input.size, [input.size, f32_bits(keep), f32_bits(1.0 / keep)])
            return [g]
        }, extras: [noise]))
    } else {
        noise.free()
    }
    return output
}

#: Mean squared error between equal-shaped tensors. Returns a scalar.
pub fn mse(prediction: Tensor, target: Tensor) -> Tensor {
    if not prediction.same_shape(target) {
        throw ValueError(f"mse shape mismatch: {prediction.shape} vs {target.shape}")
    }
    let diff = subtract(prediction, target)
    let sq = square(diff)
    let loss = mean(sq)
    if prediction.requires_grad or target.requires_grad { return loss }
    diff.free()
    sq.free()
    return loss
}

#: `cond != 0` picks `a`, otherwise `b`. All three must share a shape.
pub fn where(cond: Tensor, a: Tensor, b: Tensor) -> Tensor {
    if not cond.same_shape(a) or not a.same_shape(b) {
        throw ValueError(f"where needs matching shapes, got {cond.shape}, {a.shape}, {b.shape}")
    }
    let output = Tensor(a.shape)
    let kernel = cached_kernel("where_op")
    kernel.dispatch_async(
        [cond.buffer(), a.buffer(), b.buffer(), output.buffer()],
        a.size,
        [a.size]
    )
    return output
}

#: Raise each element to a host exponent.
pub fn pow(input: Tensor, exponent: float) -> Tensor {
    let output = Tensor(input.shape)
    let kernel = cached_kernel("pow_scalar")
    kernel.dispatch_async(
        [input.buffer(), output.buffer()],
        input.size,
        [input.size, f32_bits(exponent)]
    )
    if input.requires_grad {
        output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
            let lowered = Tensor(input.shape)
            let inner = cached_kernel("pow_scalar")
            inner.dispatch_async([input.buffer(), lowered.buffer()], input.size, [input.size, f32_bits(exponent - 1.0)])
            let chained = multiply(upstream, lowered)
            lowered.free()
            let grad = scale(chained, exponent)
            chained.free()
            return [grad]
        }))
    }
    return output
}

#: Insert a size-1 axis at `axis`.
pub fn unsqueeze(input: Tensor, axis: int) -> Tensor {
    let rank = input.rank()
    let resolved = if axis < 0 { rank + 1 + axis } else { axis }
    if resolved < 0 or resolved > rank {
        throw ValueError(f"unsqueeze axis {axis} is outside rank {rank}")
    }
    var shape: list[int] = []
    for i in 0..resolved { shape.push(input.shape[i]) }
    shape.push(1)
    for i in resolved..rank { shape.push(input.shape[i]) }
    return reshape(input, shape)
}

#: Drop size-1 axes. `axis` null drops every size-1 axis.
pub fn squeeze(input: Tensor, axis: int? = null) -> Tensor {
    var shape: list[int] = []
    if axis is null {
        for d in input.shape {
            if d != 1 { shape.push(d) }
        }
        if shape.len() == 0 { shape.push(1) }
        return reshape(input, shape)
    }
    let resolved = if axis < 0 { input.rank() + axis } else { axis }
    if resolved < 0 or resolved >= input.rank() {
        throw ValueError(f"squeeze axis {axis} is outside rank {input.rank()}")
    }
    if input.shape[resolved] != 1 {
        throw ValueError(f"cannot squeeze axis {axis} of size {input.shape[resolved]}")
    }
    for i in 0..input.rank() {
        if i != resolved { shape.push(input.shape[i]) }
    }
    if shape.len() == 0 { shape.push(1) }
    return reshape(input, shape)
}

#: Stack tensors by inserting a new leading axis, then concatenating.
pub fn stack(parts: list[Tensor]) -> Tensor {
    if parts.len() == 0 { throw ValueError("stack needs at least one tensor") }
    var expanded: list[Tensor] = []
    for part in parts { expanded.push(unsqueeze(part, 0)) }
    return cat(expanded, axis: 0)
}

#: Zero-pad an NHWC tensor. `pad` is `(top, bottom, left, right)`.
pub fn pad2d(input: Tensor, pad: list[int]) -> Tensor {
    if input.rank() != 4 { throw ValueError(f"pad2d needs NHWC input, got {input.shape}") }
    if pad.len() != 4 { throw ValueError("pad2d pad must be [top, bottom, left, right]") }
    for p in pad {
        if p < 0 { throw ValueError("pad2d pad values must be non-negative") }
    }
    let n = input.shape[0]
    let h = input.shape[1]
    let w = input.shape[2]
    let c = input.shape[3]
    let output = Tensor([n, h + pad[0] + pad[1], w + pad[2] + pad[3], c])
    let kernel = cached_kernel("pad2d_nhwc")
    kernel.dispatch_async(
        [input.buffer(), output.buffer()],
        output.size,
        [n, h, w, c, pad[0], pad[1], pad[2], pad[3]]
    )
    if input.requires_grad {
        let padded_h = h + pad[0] + pad[1]
        let padded_w = w + pad[2] + pad[3]
        let top = pad[0]
        let left = pad[2]
        output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
            let grad = Tensor([n, h, w, c])
            let crop = cached_kernel("crop2d_nhwc")
            crop.dispatch_async([upstream.buffer(), grad.buffer()], grad.size, [n, h, w, c, top, left, padded_h, padded_w])
            return [grad]
        }))
    }
    return output
}

#: NHWC `[N,H,W,C]` to NCHW `[N,C,H,W]`.
pub fn nchw(input: Tensor) -> Tensor {
    if input.rank() != 4 { throw ValueError(f"nchw needs rank four, got {input.shape}") }
    let n = input.shape[0]
    let h = input.shape[1]
    let w = input.shape[2]
    let c = input.shape[3]
    let output = Tensor([n, c, h, w])
    let kernel = cached_kernel("nhwc_to_nchw")
    kernel.dispatch_async([input.buffer(), output.buffer()], output.size, [n, h, w, c])
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [nhwc(upstream)]))
    }
    return output
}

#: NCHW `[N,C,H,W]` to NHWC `[N,H,W,C]`.
pub fn nhwc(input: Tensor) -> Tensor {
    if input.rank() != 4 { throw ValueError(f"nhwc needs rank four, got {input.shape}") }
    let n = input.shape[0]
    let c = input.shape[1]
    let h = input.shape[2]
    let w = input.shape[3]
    let output = Tensor([n, h, w, c])
    let kernel = cached_kernel("nchw_to_nhwc")
    kernel.dispatch_async([input.buffer(), output.buffer()], output.size, [n, c, h, w])
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [nchw(upstream)]))
    }
    return output
}

#: Permute axes. Rank two is transpose. Rank four supports NHWC<->NCHW layouts.
pub fn permute(input: Tensor, axes: list[int]) -> Tensor {
    if axes.len() != input.rank() {
        throw ValueError(f"permute axes length {axes.len()} does not match rank {input.rank()}")
    }
    if input.rank() == 2 {
        if axes[0] == 1 and axes[1] == 0 { return transpose(input) }
        if axes[0] == 0 and axes[1] == 1 { return reshape(input, input.shape) }
        throw ValueError(f"unsupported permute {axes}")
    }
    if input.rank() == 4 {
        if axes == [0, 3, 1, 2] { return nchw(input) }
        if axes == [0, 2, 3, 1] { return nhwc(input) }
        if axes == [0, 1, 2, 3] { return reshape(input, input.shape) }
        throw ValueError(f"unsupported permute {axes}")
    }
    throw ValueError("permute currently supports rank two and four")
}

#: Repeat the last axis `times` times.
pub fn repeat(input: Tensor, times: int, axis: int = -1) -> Tensor {
    if times <= 0 { throw ValueError(f"repeat times must be positive, got {times}") }
    let resolved = if axis < 0 { input.rank() + axis } else { axis }
    if resolved != input.rank() - 1 {
        throw ValueError("repeat currently supports the last axis")
    }
    let inner = input.shape[resolved]
    let rows = input.size // inner
    var shape: list[int] = []
    for i in 0..input.rank() {
        shape.push(if i == resolved { inner * times } else { input.shape[i] })
    }
    let output = Tensor(shape)
    let kernel = cached_kernel("repeat_last")
    kernel.dispatch_async([input.buffer(), output.buffer()], output.size, [rows, inner, times])
    if input.requires_grad {
        let original = input.shape
        output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
            let grad = Tensor(original)
            let back = cached_kernel("repeat_last_backward")
            back.dispatch_async([upstream.buffer(), grad.buffer()], grad.size, [rows, inner, times])
            return [grad]
        }))
    }
    return output
}

#: Mean binary cross-entropy between probabilities and 0/1 targets.
pub fn binary_cross_entropy(prediction: Tensor, target: Tensor) -> Tensor {
    if not prediction.same_shape(target) {
        throw ValueError(f"bce shape mismatch: {prediction.shape} vs {target.shape}")
    }
    let per = Tensor(prediction.shape)
    let kernel = cached_kernel("bce_sample")
    kernel.dispatch_async(
        [prediction.buffer(), target.buffer(), per.buffer()],
        prediction.size,
        [prediction.size]
    )
    if prediction.requires_grad {
        per.attach_creator(GradNode([prediction], fn(upstream: Tensor) -> list[Tensor] {
            let g = Tensor(prediction.shape)
            let backward_kernel = cached_kernel("bce_prob_backward")
            backward_kernel.dispatch_async([prediction.buffer(), target.buffer(), upstream.buffer(), g.buffer()], prediction.size, [prediction.size])
            return [g]
        }))
    }
    let loss = mean(per)
    if not prediction.requires_grad { per.free() }
    return loss
}

#: Mean binary cross-entropy from logits. Numerically stable; no sigmoid first.
pub fn binary_cross_entropy_with_logits(logits: Tensor, target: Tensor) -> Tensor {
    if not logits.same_shape(target) {
        throw ValueError(f"bce_with_logits shape mismatch: {logits.shape} vs {target.shape}")
    }
    let per = Tensor(logits.shape)
    let kernel = cached_kernel("bce_logits_sample")
    kernel.dispatch_async(
        [logits.buffer(), target.buffer(), per.buffer()],
        logits.size,
        [logits.size]
    )
    if logits.requires_grad {
        per.attach_creator(GradNode([logits], fn(upstream: Tensor) -> list[Tensor] {
            let g = Tensor(logits.shape)
            let backward_kernel = cached_kernel("bce_logits_backward")
            backward_kernel.dispatch_async([logits.buffer(), target.buffer(), upstream.buffer(), g.buffer()], logits.size, [logits.size])
            return [g]
        }))
    }
    let loss = mean(per)
    if not logits.requires_grad { per.free() }
    return loss
}

#: Keep the lower triangle of a rank-two tensor, including the diagonal.
pub fn tril(input: Tensor) -> Tensor {
    if input.rank() != 2 { throw ValueError(f"tril needs rank two, got {input.shape}") }
    let rows = input.shape[0]
    let cols = input.shape[1]
    let output = Tensor(input.shape)
    let kernel = cached_attn_kernel("tril_square")
    kernel.dispatch_async([input.buffer(), output.buffer()], output.size, [rows, cols])
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [tril(upstream)]))
    }
    return output
}

#: Identity matrix of size `n` on the GPU.
pub fn eye(n: int) -> Tensor {
    if n <= 0 { throw ValueError(f"eye needs a positive size, got {n}") }
    let output = Tensor([n, n])
    let kernel = cached_kernel("eye_fill")
    kernel.dispatch_async([output.buffer()], output.size, [n])
    return output
}

#: Reverse the last axis. Other axes are not yet supported.
pub fn flip(input: Tensor, axis: int = -1) -> Tensor {
    let rank = input.rank()
    let resolved = if axis < 0 { rank + axis } else { axis }
    if resolved != rank - 1 {
        throw ValueError("flip currently supports the last axis")
    }
    let inner = input.shape[resolved]
    let rows = input.size // inner
    let output = Tensor(input.shape)
    let kernel = cached_kernel("flip_last")
    kernel.dispatch_async([input.buffer(), output.buffer()], output.size, [rows, inner])
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [flip(upstream, axis: axis)]))
    }
    return output
}

#: Nearest-neighbor upsample of NHWC images by an integer `scale`.
pub fn upsample_nearest2d(input: Tensor, scale: int) -> Tensor {
    if input.rank() != 4 {
        throw ValueError(f"upsample_nearest2d needs NHWC input, got {input.shape}")
    }
    if scale <= 0 { throw ValueError(f"upsample scale must be positive, got {scale}") }
    let n = input.shape[0]
    let h = input.shape[1]
    let w = input.shape[2]
    let c = input.shape[3]
    let out_h = h * scale
    let out_w = w * scale
    let output = Tensor([n, out_h, out_w, c])
    let kernel = cached_kernel("upsample_nearest2d_nhwc")
    kernel.dispatch_async(
        [input.buffer(), output.buffer()],
        output.size,
        [n, h, w, c, scale, scale, out_h, out_w]
    )
    if input.requires_grad {
        output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
            let grad = Tensor([n, h, w, c])
            let back = cached_kernel("upsample_nearest2d_backward_nhwc")
            back.dispatch_async([upstream.buffer(), grad.buffer()], grad.size, [n, h, w, c, scale, scale, out_h, out_w])
            return [grad]
        }))
    }
    return output
}

#: Pack `[seq, dim]` into `[heads * seq, head_dim]` with each head's rows contiguous.
pub fn split_heads(input: Tensor, heads: int) -> Tensor {
    if input.rank() != 2 {
        throw ValueError(f"split_heads needs rank two, got {input.shape}")
    }
    if heads <= 0 { throw ValueError(f"split_heads needs a positive head count, got {heads}") }
    let seq = input.shape[0]
    let dim = input.shape[1]
    if dim % heads != 0 {
        throw ValueError(f"split_heads dim {dim} must divide heads {heads}")
    }
    let hd = dim // heads
    let output = Tensor([heads * seq, hd])
    let kernel = cached_attn_kernel("split_heads")
    kernel.dispatch_async([input.buffer(), output.buffer()], output.size, [seq, heads, hd])
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [merge_heads(upstream, heads, seq)]))
    }
    return output
}

#: Inverse of `split_heads`: `[heads * seq, head_dim]` back to `[seq, dim]`.
pub fn merge_heads(packed: Tensor, heads: int, seq: int) -> Tensor {
    if packed.rank() != 2 {
        throw ValueError(f"merge_heads needs rank two, got {packed.shape}")
    }
    if heads <= 0 or seq <= 0 {
        throw ValueError("merge_heads needs positive heads and sequence length")
    }
    if packed.shape[0] != heads * seq {
        throw ValueError(f"merge_heads packed rows {packed.shape[0]} != {heads} * {seq}")
    }
    let hd = packed.shape[1]
    let output = Tensor([seq, heads * hd])
    let kernel = cached_attn_kernel("merge_heads")
    kernel.dispatch_async([packed.buffer(), output.buffer()], packed.size, [seq, heads, hd])
    if packed.requires_grad {
        output.attach_creator(GradNode([packed], |upstream| [split_heads(upstream, heads)]))
    }
    return output
}

pub fn softmax(input: Tensor) -> Tensor {
    if input.rank() != 2 { throw ValueError(f"softmax needs rank two, got {input.shape}") }
    let rows = input.shape[0]
    let columns = input.shape[1]
    let output = Tensor(input.shape)
    let kernel = cached_kernel("row_softmax")
    kernel.dispatch_async([input.buffer(), output.buffer()], rows, [rows, columns])
    if input.requires_grad {
        output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
            let grad_input = Tensor(input.shape)
            let backward_kernel = cached_kernel("softmax_backward")
            backward_kernel.dispatch_async([output.buffer(), upstream.buffer(), grad_input.buffer()], rows, [rows, columns])
            return [grad_input]
        }))
    }
    return output
}

#: Index of the largest value in each row of a rank-two tensor.
#:
#: Indices use float storage because all Metal buffers in `std.gpu` are
#: float32. Downloaded values are exact for practical class counts.
pub fn argmax(input: Tensor) -> Tensor {
    if input.rank() != 2 { throw ValueError(f"argmax needs rank two, got {input.shape}") }
    let rows = input.shape[0]
    let columns = input.shape[1]
    let output = Tensor([rows])
    let kernel = cached_kernel("row_argmax")
    kernel.dispatch_async([input.buffer(), output.buffer()], rows, [rows, columns])
    return output
}

#: Expand a rank-one tensor of integer-valued indices into one-hot rows.
pub fn one_hot(indices: Tensor, depth: int) -> Tensor {
    if indices.rank() != 1 { throw ValueError(f"one_hot needs rank one, got {indices.shape}") }
    if depth <= 0 { throw ValueError(f"depth must be positive, got {depth}") }
    let rows = indices.shape[0]
    let output = Tensor([rows, depth])
    let kernel = cached_kernel("one_hot_rows")
    kernel.dispatch_async([indices.buffer(), output.buffer()], rows * depth, [rows, depth])
    return output
}

#: Copy a contiguous range of `source` as a view of the same device memory.
#:
#: `source_offset` is in elements. The result is a leaf: it does not attach a
#: gradient graph. Freeing the view does not free `source`.
pub fn slice_flat(source: Tensor, source_offset: int, shape: list[int]) -> Tensor {
    let n = element_count(shape)
    if source_offset < 0 or source_offset + n > source.size {
        throw ValueError(
            f"slice_flat offset {source_offset} and size {n} " +
                f"run past source size {source.size}"
        )
    }
    let output = source.view(source_offset, shape)
    if source.requires_grad {
        let whole = source.shape
        output.attach_creator(GradNode([source], fn(upstream: Tensor) -> list[Tensor] {
            let grad = Tensor.zeros(whole)
            let kernel = cached_kernel("copy_range")
            kernel.dispatch_async([upstream.buffer(), grad.buffer()], n, [0, source_offset, n])
            return [grad]
        }))
    }
    return output
}

#: Copy `source` into `dest`. Both must have the same number of elements.
pub fn copy_into(source: Tensor, dest: Tensor) -> void {
    if source.size != dest.size {
        throw ValueError(f"copy_into size mismatch: {source.size} vs {dest.size}")
    }
    let kernel = cached_kernel("copy_offset")
    kernel.dispatch_async([source.buffer(), dest.buffer()], dest.size, [0, dest.size])
}

#: Add `value[0] * weight` into a scalar accumulator tensor.
pub fn add_scaled_into(value: Tensor, total: Tensor, weight: int) -> void {
    if value.size != 1 or total.size != 1 {
        throw ValueError("add_scaled_into needs scalar tensors")
    }
    if weight < 0 { throw ValueError(f"weight must be non-negative, got {weight}") }
    let kernel = cached_kernel("add_scaled_into")
    kernel.dispatch_async([value.buffer(), total.buffer()], 1, [weight])
}

#: Add the number of correct argmax classes into a scalar accumulator.
pub fn add_correct_into(logits: Tensor, labels: Tensor, total: Tensor) -> void {
    if logits.rank() != 2 {
        throw ValueError(f"add_correct_into needs rank-two logits, got {logits.shape}")
    }
    if total.size != 1 { throw ValueError("add_correct_into needs a scalar accumulator") }
    let batch = logits.shape[0]
    let classes = logits.shape[1]
    if labels.size != batch {
        throw ValueError(f"add_correct_into needs {batch} labels, got {labels.size}")
    }
    let kernel = cached_kernel("add_correct_into")
    kernel.dispatch_async(
        [logits.buffer(), labels.buffer(), total.buffer()],
        batch,
        [batch, classes]
    )
}

#: Output extent of one spatial axis under a convolution or pooling window.
pub fn conv2d_out_dim(size: int, kernel: int, stride: int, padding: int) -> int {
    return (size + 2 * padding - kernel) // stride + 1
}

#: NHWC 2D convolution. Weights are HWIO `[kh, kw, cin, cout]`.
#:
#: `activation` fuses an elementwise epilogue into the convolution's own
#: dispatch instead of a separate pass over the output: 0 none, 1 ReLU. A
#: fused activation needs `bias` -- the epilogue that carries it always adds
#: one, so there is nowhere to route "activation but no bias" -- which
#: `Conv2d.forward` never hits since every convolution layer owns a bias.
@trace
pub fn conv2d(
    input: Tensor,
    weights: Tensor,
    bias: Tensor? = null,
    stride: int = 1,
    padding: int = 0,
    activation: int = 0
) -> Tensor {
    if input.rank() != 4 {
        throw ValueError(f"conv2d needs NHWC input, got {input.shape}")
    }
    if weights.rank() != 4 {
        throw ValueError(f"conv2d needs HWIO weights, got {weights.shape}")
    }
    let n = input.shape[0]
    let h = input.shape[1]
    let w = input.shape[2]
    let cin = input.shape[3]
    let kh = weights.shape[0]
    let kw = weights.shape[1]
    if weights.shape[2] != cin {
        throw ValueError(f"conv2d channel mismatch: input {cin} vs weights {weights.shape[2]}")
    }
    let cout = weights.shape[3]
    if stride <= 0 { throw ValueError(f"conv2d stride must be positive, got {stride}") }
    if padding < 0 { throw ValueError(f"conv2d padding must be non-negative, got {padding}") }
    if activation < 0 or activation > 1 {
        throw ValueError(f"conv2d activation must be 0 or 1, got {activation}")
    }
    if activation != 0 and bias is null {
        throw ValueError("conv2d needs a bias to fuse an activation")
    }
    let out_h = conv2d_out_dim(h, kh, stride, padding)
    let out_w = conv2d_out_dim(w, kw, stride, padding)
    if out_h <= 0 or out_w <= 0 {
        throw ValueError(f"conv2d output spatial size is empty: {out_h}x{out_w}")
    }
    if bias is not null and bias.size != cout {
        throw ValueError(f"conv2d bias has {bias.size} values, expected {cout}")
    }
    let output = Tensor([n, out_h, out_w, cout])
    let by_im2col = fn() -> void {
        let k = kh * kw * cin
        let rows = n * out_h * out_w
        let col = Tensor([rows, k])
        let kernel = cached_kernel("im2col_nhwc")
        kernel.dispatch_async([input.buffer(), col.buffer()], rows * k, [n, h, w, cin, kh, kw, stride, stride, padding, padding])
        let flat_w = Tensor([k, cout], storage: weights.buffer().view(0, k * cout))
        let flat_out = Tensor([rows, cout], storage: output.buffer())
        let epilogue = if bias is null {
            0
        } elif activation == 1 {
            2
        } else {
            1
        }
        raw_matmul(col, flat_w, bias: bias, epilogue: epilogue, out: flat_out)
        col.free()
    }
    try {
        gpu.conv2d_async(input.buffer(), weights.buffer(), output.buffer(), n, h, w, cin, cout, kh, kw, stride_h: stride, stride_w: stride, pad_h: padding, pad_w: padding, bias: if bias is null {
            null
        } else {
            bias.buffer()
        }, activation: activation)
    } catch _error: RuntimeError {
        by_im2col()
    }
    attach_conv_grad(input, weights, bias, output, stride, padding, activation)
    return output
}

#: NCHW 2D convolution. Weights are OIHW `[cout, cin, kh, kw]`.
#:
#: The layout every exported model file uses, run without being rearranged
#: into the one `conv2d` wants. MPS takes NCHW natively, so the transposes
#: that used to bracket each call -- two over the activations and one over the
#: weights -- were paying for nothing: on YOLOv8n they were most of the time
#: the convolution appeared to cost.
#:
#: Inference only, in the sense that nothing here attaches a gradient: this is
#: for running an imported graph forward. Train with `conv2d`.
#:
#: `activation` fuses an epilogue into the same dispatch: 0 none, 1 ReLU,
#: 2 SiLU, and as in `conv2d` a fused activation needs a bias.
@trace
pub fn conv2d_nchw(
    input: Tensor,
    weights: Tensor,
    bias: Tensor? = null,
    stride: int = 1,
    padding: int = 0,
    activation: int = 0
) -> Tensor {
    if input.rank() != 4 {
        throw ValueError(f"conv2d_nchw needs NCHW input, got {input.shape}")
    }
    if weights.rank() != 4 {
        throw ValueError(f"conv2d_nchw needs OIHW weights, got {weights.shape}")
    }
    let n = input.shape[0]
    let cin = input.shape[1]
    let h = input.shape[2]
    let w = input.shape[3]
    let cout = weights.shape[0]
    let kh = weights.shape[2]
    let kw = weights.shape[3]
    if weights.shape[1] != cin {
        throw ValueError(
            f"conv2d_nchw channel mismatch: input {cin} vs weights {weights.shape[1]}"
        )
    }
    if stride <= 0 { throw ValueError(f"conv2d_nchw stride must be positive, got {stride}") }
    if padding < 0 {
        throw ValueError(f"conv2d_nchw padding must be non-negative, got {padding}")
    }
    if activation < 0 or activation > 2 {
        throw ValueError(f"conv2d_nchw activation must be 0, 1 or 2, got {activation}")
    }
    if activation != 0 and bias is null {
        throw ValueError("conv2d_nchw needs a bias to fuse an activation")
    }
    if bias is not null and bias.size != cout {
        throw ValueError(f"conv2d_nchw bias has {bias.size} values, expected {cout}")
    }
    let out_h = conv2d_out_dim(h, kh, stride, padding)
    let out_w = conv2d_out_dim(w, kw, stride, padding)
    if out_h <= 0 or out_w <= 0 {
        throw ValueError(f"conv2d_nchw output spatial size is empty: {out_h}x{out_w}")
    }
    let output = Tensor([n, cout, out_h, out_w])
    try {
        gpu.conv2d_async(
            input.buffer(),
            weights.buffer(),
            output.buffer(),
            n,
            h,
            w,
            cin,
            cout,
            kh,
            kw,
            stride_h: stride,
            stride_w: stride,
            pad_h: padding,
            pad_w: padding,
            bias: if bias is null { null } else { bias.buffer() },
            activation: activation,
            layout: 1
        )
    } catch _error: RuntimeError {
        #: A shape MPS will not take. Rather than keep a second NCHW kernel
        #: alive for it, go the long way round through `conv2d`, which has its
        #: own im2col fallback -- correct, rare, and slow only when it fires.
        output.free()
        let nhwc_in = nhwc(input)
        #: OIHW to HWIO. `permute` has no kernel for `[2, 3, 1, 0]`, but the
        #: swap is two it does have: `nhwc` puts the input channel last,
        #: leaving the output channel as the only axis out of place, and a
        #: matrix transpose across the flattened rest moves it to the end.
        #:
        #: `reshape` shares its argument's storage, so only the two tensors
        #: that own a buffer -- the `nhwc` and the `transpose` -- are freed,
        #: and each of those only once the last view onto it is finished with.
        let ohwi = nhwc(weights)
        let columns = transpose(reshape(ohwi, [cout, kh * kw * cin]))
        ohwi.free()
        let hwio = reshape(columns, [kh, kw, cin, cout])
        let done = conv2d(
            nhwc_in,
            hwio,
            bias: bias,
            stride: stride,
            padding: padding,
            activation: if activation == 2 { 0 } else { activation }
        )
        nhwc_in.free()
        columns.free()
        let back = nchw(done)
        done.free()
        if activation != 2 { return back }
        let fused = silu(back)
        back.free()
        return fused
    }
    return output
}

#: Backward for `conv2d`, as im2col plus two products.
#:
#: The columns are rebuilt here rather than kept from the forward pass: a
#: `[batch * out_h * out_w, kh * kw * cin]` matrix is tens of megabytes for an
#: ordinary layer, and the forward may not have built one at all when the
#: native convolution took the work.
fn attach_conv_grad(
    input: Tensor,
    weights: Tensor,
    bias: Tensor?,
    output: Tensor,
    stride: int,
    padding: int,
    activation: int
) -> void {
    let wants_bias = bias is not null and bias.requires_grad
    if not (input.requires_grad or weights.requires_grad or wants_bias) { return }
    var inputs = [input, weights]
    if wants_bias { inputs.push(bias) }
    let n = input.shape[0]
    let h = input.shape[1]
    let w = input.shape[2]
    let cin = input.shape[3]
    let kh = weights.shape[0]
    let kw = weights.shape[1]
    let cout = weights.shape[3]
    let out_h = output.shape[1]
    let out_w = output.shape[2]
    output.attach_creator(GradNode(inputs, fn(raw_upstream: Tensor) -> list[Tensor] {
        #: `output` already carries the fused activation, so its own gradient
        #: is masked here before any of the convolution math below sees it --
        #: the same rewrite `dense_affine` does for its epilogue, and the
        #: reason `conv2d` only fuses an activation whose backward needs
        #: nothing but the (already-available) forward output.
        let upstream = if activation == 1 { unary_backward(output, raw_upstream, 1) } else { raw_upstream }
        let k = kh * kw * cin
        let rows = n * out_h * out_w
        let flat_up = Tensor([rows, cout], storage: upstream.buffer().view(0, rows * cout))
        let grad_weights = Tensor([kh, kw, cin, cout])
        let grad_input = Tensor([n, h, w, cin])

        #: The device's own convolution gradients when it has them. The im2col
        #: pair below computes the same two tensors, and does it by building a
        #: `[n * out_h * out_w, kh * kw * cin]` matrix twice per step: a
        #: hundred and fifty megabytes allocated, filled and dropped on every
        #: batch of a sixty-four-channel layer. Either may be declined on its
        #: own -- a shape MPS will not take, or no device at all -- so each is
        #: asked for separately and each has its own fallback.
        var have_weights_grad = false
        var have_input_grad = false
        try {
            have_weights_grad = gpu.conv2d_weights_grad_async(
                upstream.buffer(),
                input.buffer(),
                grad_weights.buffer(),
                n,
                h,
                w,
                cin,
                cout,
                kh,
                kw,
                stride_h: stride,
                stride_w: stride,
                pad_h: padding,
                pad_w: padding
            )
            have_input_grad = gpu.conv2d_data_grad_async(
                upstream.buffer(),
                weights.buffer(),
                grad_input.buffer(),
                n,
                h,
                w,
                cin,
                cout,
                kh,
                kw,
                stride_h: stride,
                stride_w: stride,
                pad_h: padding,
                pad_w: padding
            )
        } catch _error: RuntimeError {
            have_weights_grad = false
            have_input_grad = false
        }

        if not have_weights_grad {
            let col = Tensor([rows, k])
            let im2col = cached_kernel("im2col_nhwc")
            im2col.dispatch_async(
                [input.buffer(), col.buffer()],
                rows * k,
                [n, h, w, cin, kh, kw, stride, stride, padding, padding]
            )
            let flat_gw = Tensor([k, cout], storage: grad_weights.buffer())
            raw_matmul(col, flat_up, trans_a: true, out: flat_gw)
            col.free()
        }
        if not have_input_grad {
            let flat_w = Tensor([k, cout], storage: weights.buffer().view(0, k * cout))
            let grad_col = raw_matmul(flat_up, flat_w, trans_b: true)
            let col2im = cached_kernel("col2im_nhwc")
            col2im.dispatch_async(
                [grad_col.buffer(), grad_input.buffer()],
                grad_input.size,
                [n, h, w, cin, kh, kw, stride, stride, padding, padding]
            )
            grad_col.free()
        }
        var grads = [grad_input, grad_weights]
        if wants_bias { grads.push(reduce_leading(flat_up, [cout])) }
        if activation == 1 { upstream.free() }
        return grads
    }))
}

#: Channel-last batch normalization using per-channel mean and variance of `x`.
pub fn batch_norm(input: Tensor, gamma: Tensor, beta: Tensor) -> Tensor {
    if input.rank() != 4 {
        throw ValueError(f"batch_norm needs NHWC input, got {input.shape}")
    }
    let channels = input.shape[3]
    if gamma.size != channels or beta.size != channels {
        throw ValueError("batch_norm gamma/beta must match the channel count")
    }
    let nspatial = input.shape[0] * input.shape[1] * input.shape[2]
    let mean = Tensor([channels])
    let variance = Tensor([channels])
    let moments = cached_kernel("moments_channel")
    moments.dispatch_async(
        [input.buffer(), mean.buffer(), variance.buffer()],
        channels,
        [nspatial, channels]
    )
    let output = Tensor(input.shape)
    let kernel = cached_kernel("batch_norm_forward")
    kernel.dispatch_async(
        [
            input.buffer(),
            mean.buffer(),
            variance.buffer(),
            gamma.buffer(),
            beta.buffer(),
            output.buffer(),
        ],
        input.size,
        [input.size, channels]
    )
    if input.requires_grad or gamma.requires_grad or beta.requires_grad {
        output.attach_creator(GradNode([input, gamma, beta], fn(upstream: Tensor) -> list[Tensor] {
            let sum_dy = Tensor([channels])
            let sum_dy_xhat = Tensor([channels])
            let grad_gamma = Tensor([channels])
            let grad_beta = Tensor([channels])
            let sums = cached_kernel("norm_channel_sums")
            sums.dispatch_async([input.buffer(), mean.buffer(), variance.buffer(), gamma.buffer(), upstream.buffer(), sum_dy.buffer(), sum_dy_xhat.buffer(), grad_gamma.buffer(), grad_beta.buffer()], channels, [nspatial, channels])
            let grad_input = Tensor(input.shape)
            let back = cached_kernel("norm_channel_backward")
            back.dispatch_async([input.buffer(), mean.buffer(), variance.buffer(), gamma.buffer(), upstream.buffer(), sum_dy.buffer(), sum_dy_xhat.buffer(), grad_input.buffer()], input.size, [input.size, channels, nspatial])
            sum_dy.free()
            sum_dy_xhat.free()
            return [grad_input, grad_gamma, grad_beta]
        }, extras: [mean, variance]))
        return output
    }
    mean.free()
    variance.free()
    return output
}

#: Normalize the last axis of a rank-two tensor.
pub fn layer_norm(input: Tensor, gamma: Tensor, beta: Tensor) -> Tensor {
    if input.rank() != 2 {
        throw ValueError(f"layer_norm needs rank-two input, got {input.shape}")
    }
    let width = input.shape[1]
    if gamma.size != width or beta.size != width {
        throw ValueError("layer_norm gamma/beta must match the last dimension")
    }
    let output = Tensor(input.shape)
    let kernel = cached_kernel("layer_norm_forward")
    kernel.dispatch_async(
        [input.buffer(), gamma.buffer(), beta.buffer(), output.buffer()],
        input.shape[0],
        [input.shape[0], width]
    )
    if input.requires_grad or gamma.requires_grad or beta.requires_grad {
        let rows = input.shape[0]
        output.attach_creator(GradNode([input, gamma, beta], fn(upstream: Tensor) -> list[Tensor] {
            let grad_input = Tensor(input.shape)
            let back = cached_kernel("layer_norm_backward")
            back.dispatch_async([input.buffer(), gamma.buffer(), upstream.buffer(), grad_input.buffer()], rows, [rows, width])
            let grad_gamma = Tensor([width])
            let grad_beta = Tensor([width])
            let params = cached_kernel("layer_norm_params_backward")
            params.dispatch_async([input.buffer(), upstream.buffer(), grad_gamma.buffer(), grad_beta.buffer()], width, [rows, width])
            return [grad_input, grad_gamma, grad_beta]
        }))
    }
    return output
}

#: Scaled dot-product attention for rank-two `[seq, dim]` tensors.
pub fn scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor {
    if query.rank() != 2 or key.rank() != 2 or value.rank() != 2 {
        throw ValueError("scaled_dot_product_attention needs rank-two Q, K, V")
    }
    if query.shape[1] != key.shape[1] {
        throw ValueError("query/key depth mismatch")
    }
    if key.shape[0] != value.shape[0] {
        throw ValueError("key/value sequence mismatch")
    }
    let factor = 1.0 / math.sqrt(float(query.shape[1]))
    if query.requires_grad or key.requires_grad or value.requires_grad {
        return matmul(softmax(scale(matmul(query, key, trans_b: true), factor)), value)
    }
    let scores = raw_matmul(query, key, trans_b: true)
    let scaled = scale(scores, factor)
    scores.free()
    let probs = softmax(scaled)
    scaled.free()
    let output = raw_matmul(probs, value)
    probs.free()
    return output
}

#: Per-head scaled-dot-product attention on packed `[seq, dim]` Q/K/V.
#:
#: The fused device path is a Metal prefill kernel: packed `[seq, heads, hd]`
#: stays packed, Q/K/V stream through half SRAM, and QK/PV use simdgroup MMA.
#: MPSGraph SDPA is the fallback when the kernel cannot run. Last resort is
#: GEMM. Scale is `1 / sqrt(head_dim)`.
pub fn multi_head_attention(query: Tensor, key: Tensor, value: Tensor, heads: int) -> Tensor {
    if heads == 1 { return scaled_dot_product_attention(query, key, value) }
    if query.rank() != 2 or key.rank() != 2 or value.rank() != 2 {
        throw ValueError("multi_head_attention needs rank-two Q, K, V")
    }
    if heads <= 0 {
        throw ValueError(f"multi_head_attention needs a positive head count, got {heads}")
    }
    if query.shape != key.shape or query.shape != value.shape {
        throw ValueError(
            f"multi_head_attention Q/K/V shape mismatch: {query.shape} {key.shape} {value.shape}"
        )
    }
    let seq = query.shape[0]
    let dim = query.shape[1]
    if dim % heads != 0 {
        throw ValueError(f"multi_head_attention dim {dim} must divide heads {heads}")
    }
    let hd = dim // heads
    let factor = 1.0 / math.sqrt(float(hd))
    if query.requires_grad or key.requires_grad or value.requires_grad {
        return mha_by_ops(query, key, value, heads, seq, hd, factor)
    }
    let output = Tensor(query.shape)
    if gpu.mha_async(
        query.buffer(),
        key.buffer(),
        value.buffer(),
        output.buffer(),
        seq,
        heads,
        hd,
        factor
    ) {
        return output
    }
    if seq >= 16 and hd >= 32 and hd <= 64 {
        let q_tiles = (seq + 15) // 16
        let kernel = cached_attn_kernel("flash_mha_wide")
        kernel.dispatch_async(
            [query.buffer(), key.buffer(), value.buffer(), output.buffer()],
            q_tiles * heads * 256,
            [seq, heads, hd, f32_bits(factor)],
            group_size: 256
        )
        return output
    }
    if seq >= 16 and hd <= 64 {
        let q_tiles = (seq + 15) // 16
        let kernel = cached_attn_kernel("flash_mha")
        kernel.dispatch_async(
            [query.buffer(), key.buffer(), value.buffer(), output.buffer()],
            q_tiles * heads * 256,
            [seq, heads, hd, f32_bits(factor)],
            group_size: 256
        )
        return output
    }
    output.free()
    return mha_by_gemm(query, key, value, heads, seq, hd, factor)
}

#: Multi-head attention built from differentiable ops.
#:
#: The fused kernels above never materialise the attention probabilities, so
#: there is nothing for a backward pass to read; a graph that needs gradients
#: gets this composition instead. Nothing is freed eagerly here — every
#: intermediate belongs to the graph until `backward` collects it.
fn mha_by_ops(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    heads: int,
    seq: int,
    hd: int,
    factor: float
) -> Tensor {
    let qs = split_heads(query, heads)
    let ks = split_heads(key, heads)
    let vs = split_heads(value, heads)
    var parts: list[Tensor] = []
    for h in 0..heads {
        let qh = slice_flat(qs, h * seq * hd, [seq, hd])
        let kh = slice_flat(ks, h * seq * hd, [seq, hd])
        let vh = slice_flat(vs, h * seq * hd, [seq, hd])
        let probs = softmax(scale(matmul(qh, kh, trans_b: true), factor))
        parts.push(matmul(probs, vh))
    }
    return merge_heads(cat(parts), heads, seq)
}

fn mha_by_gemm(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    heads: int,
    seq: int,
    hd: int,
    factor: float
) -> Tensor {
    let qs = split_heads(query, heads)
    let ks = split_heads(key, heads)
    let vs = split_heads(value, heads)
    var parts: list[Tensor] = []
    for h in 0..heads {
        let qh = qs.view(h * seq * hd, [seq, hd])
        let kh = ks.view(h * seq * hd, [seq, hd])
        let vh = vs.view(h * seq * hd, [seq, hd])
        let scores = raw_matmul(qh, kh, trans_b: true)
        let scaled = scale(scores, factor)
        scores.free()
        let probs = softmax(scaled)
        scaled.free()
        parts.push(raw_matmul(probs, vh))
        qh.free()
        kh.free()
        vh.free()
        probs.free()
    }
    qs.free()
    ks.free()
    vs.free()
    let packed = cat(parts)
    for part in parts { part.free() }
    let output = merge_heads(packed, heads, seq)
    packed.free()
    return output
}

#: Gather `indices.len()` rows from a rank-two source.
pub fn gather_rows(source: Tensor, indices: Tensor) -> Tensor {
    if source.rank() != 2 {
        throw ValueError(f"gather_rows needs a rank-two source, got {source.shape}")
    }
    if indices.rank() != 1 {
        throw ValueError(f"gather_rows needs a rank-one index tensor, got {indices.shape}")
    }
    let width = source.shape[1]
    let output = Tensor([indices.size, width])
    let kernel = cached_kernel("gather_rows")
    kernel.dispatch_async(
        [source.buffer(), indices.buffer(), output.buffer()],
        output.size,
        [indices.size, width]
    )
    if source.requires_grad {
        output.attach_creator(GradNode([source], fn(upstream: Tensor) -> list[Tensor] {
            return [scatter_add_rows(upstream, indices, source.shape[0], width)]
        }))
    }
    return output
}

fn scatter_add_rows(upstream: Tensor, indices: Tensor, vocab: int, width: int) -> Tensor {
    let dest = Tensor.zeros([vocab, width])
    let kernel = cached_kernel("scatter_add_rows")
    kernel.dispatch_async(
        [upstream.buffer(), indices.buffer(), dest.buffer()],
        upstream.size,
        [indices.size, width]
    )
    return dest
}

#: NHWC group normalization. `groups` must divide the channel count.
pub fn group_norm(input: Tensor, gamma: Tensor, beta: Tensor, groups: int) -> Tensor {
    if input.rank() != 4 {
        throw ValueError(f"group_norm needs NHWC input, got {input.shape}")
    }
    if groups <= 0 { throw ValueError(f"group_norm groups must be positive, got {groups}") }
    let n = input.shape[0]
    let h = input.shape[1]
    let w = input.shape[2]
    let c = input.shape[3]
    if c % groups != 0 {
        throw ValueError(f"group_norm channels {c} must divide groups {groups}")
    }
    if gamma.size != c or beta.size != c {
        throw ValueError("group_norm gamma/beta must match the channel count")
    }
    let spatial = h * w
    let slots = n * groups
    let mean = Tensor([slots])
    let variance = Tensor([slots])
    let moments = cached_kernel("moments_group")
    moments.dispatch_async(
        [input.buffer(), mean.buffer(), variance.buffer()],
        slots,
        [n, spatial, c, groups]
    )
    let output = Tensor(input.shape)
    let kernel = cached_kernel("group_norm_forward")
    kernel.dispatch_async(
        [
            input.buffer(),
            mean.buffer(),
            variance.buffer(),
            gamma.buffer(),
            beta.buffer(),
            output.buffer(),
        ],
        output.size,
        [output.size, spatial, c, groups]
    )
    if input.requires_grad or gamma.requires_grad or beta.requires_grad {
        output.attach_creator(GradNode([input, gamma, beta], fn(upstream: Tensor) -> list[Tensor] {
            let sum_dy = Tensor([slots])
            let sum_dy_xhat = Tensor([slots])
            let sums = cached_kernel("group_norm_sums")
            sums.dispatch_async([input.buffer(), mean.buffer(), variance.buffer(), gamma.buffer(), upstream.buffer(), sum_dy.buffer(), sum_dy_xhat.buffer()], slots, [n, spatial, c, groups])
            let grad_input = Tensor(input.shape)
            let back = cached_kernel("group_norm_backward")
            back.dispatch_async([input.buffer(), mean.buffer(), variance.buffer(), gamma.buffer(), upstream.buffer(), sum_dy.buffer(), sum_dy_xhat.buffer(), grad_input.buffer()], input.size, [input.size, spatial, c, groups])
            let grad_gamma = Tensor([c])
            let grad_beta = Tensor([c])
            let params = cached_kernel("channel_params_backward")
            params.dispatch_async([input.buffer(), mean.buffer(), variance.buffer(), upstream.buffer(), grad_gamma.buffer(), grad_beta.buffer()], c, [n * spatial, c, spatial, groups])
            sum_dy.free()
            sum_dy_xhat.free()
            return [grad_input, grad_gamma, grad_beta]
        }, extras: [mean, variance]))
        return output
    }
    mean.free()
    variance.free()
    return output
}

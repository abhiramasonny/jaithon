#: GPU tensor operations that keep their inputs and results on the device.
import std.gpu as gpu
import std.math
from std.gpu import Buffer, Kernel
import .gemm as gemm
from .tensor import GradNode, Tensor, element_count
from .trace import record_dense_forward, record_dense_backward

let BASE_SOURCE = """#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

/* Total `value` over a whole threadgroup, handing every thread the answer.
 *
 * The statistics a normalisation needs are sums over an axis that is not the
 * one being written: a channel mean runs over the batch and every pixel. Giving
 * each of those outputs one thread leaves a GPU running a whole tensor's work
 * on as many threads as there are channels. A threadgroup strides the axis
 * instead and reduces through this.
 *
 * `lanes` must hold 32 floats, and the group must be 1024 threads or fewer, so
 * that one entry per simdgroup fits. Barriered at both ends, so the same
 * scratch can carry the next reduction. */
static inline float group_total(
    float value,
    threadgroup float *lanes,
    uint lid,
    uint lane,
    uint warp
) {
    if (lid < 32u) lanes[lid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    value = simd_sum(value);
    if (lane == 0u) lanes[warp] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint j = 0u; j < 32u; j++) total += lanes[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

/* The largest value a threadgroup holds, and the lowest index carrying it.
 *
 * The lowest, because a loop that keeps a running best with a strict `>` keeps
 * the first of any tie, and these reductions replaced such loops. A thread that
 * drew no elements passes `0xffffffff`, which never wins the minimum. */
static inline float group_peak(
    float value,
    uint index,
    threadgroup float *lanes,
    threadgroup float *shared_top,
    threadgroup atomic_uint *shared_index,
    uint lid,
    uint lane,
    uint warp,
    thread uint *out_index
) {
    const float floor_value = -3.402823e+38f;
    if (lid < 32u) lanes[lid] = floor_value;
    if (lid == 0u) atomic_store_explicit(shared_index, 0xffffffffu, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float top = simd_max(value);
    if (lane == 0u) lanes[warp] = top;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        float m = floor_value;
        for (uint j = 0u; j < 32u; j++) m = max(m, lanes[j]);
        *shared_top = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float best = *shared_top;
    if (value == best) atomic_fetch_min_explicit(shared_index, index, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    *out_index = atomic_load_explicit(shared_index, memory_order_relaxed);
    return best;
}

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

/* A transpose through a tile, so that both halves of it are coalesced.
 *
 * Read straight and written straight, one of the two has to stride: with a
 * thread per element, neighbouring threads read neighbouring floats and then
 * write `rows` apart, and every write in a simdgroup lands on its own cache
 * line. Staging a 32 by 32 square in threadgroup memory lets each thread read
 * the row it is nearest and write the column it is nearest, and neither side
 * strides.
 *
 * The tile is 33 wide on purpose. At 32 the elements of a column all fall in
 * one bank, and the write pass -- which walks a column of the tile -- would
 * serialise on it. */
kernel void matrix_transpose(device const float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             constant uint& rows [[buffer(2)]],
                             constant uint& columns [[buffer(3)]],
                             constant uint& tiles_across [[buffer(4)]],
                             uint group [[threadgroup_position_in_grid]],
                             uint lid [[thread_index_in_threadgroup]]) {
    const uint TILE = 32u;
    threadgroup float square[32][33];
    uint tile_x = group % tiles_across;
    uint tile_y = group / tiles_across;
    uint lx = lid % TILE;
    uint ly = lid / TILE;

    for (uint step = 0u; step < TILE; step += 8u) {
        uint r = tile_y * TILE + ly + step;
        uint c = tile_x * TILE + lx;
        if (r < rows && c < columns) square[ly + step][lx] = input[r * columns + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 0u; step < TILE; step += 8u) {
        /* A row of the output is a column of the input. */
        uint r = tile_x * TILE + ly + step;
        uint c = tile_y * TILE + lx;
        if (r < columns && c < rows) output[r * rows + c] = square[lx][ly + step];
    }
}

/* A threadgroup to the row, the same shape as the normalisation above and for
 * the same reason: a thread per row runs a classifier head over as many
 * threads as the batch has rows, whatever the row's width. */
kernel void row_softmax(device const float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        constant uint& rows [[buffer(2)]],
                        constant uint& columns [[buffer(3)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint lid [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint warp [[simdgroup_index_in_threadgroup]]) {
    if (row >= rows) return;
    const uint GROUP = 256u;
    const uint base = row * columns;
    const float floor_value = -3.402823e+38f;
    threadgroup float lanes[32];
    threadgroup float shared_peak;
    threadgroup float shared_total;

    float peak = floor_value;
    for (uint i = lid; i < columns; i += GROUP) peak = max(peak, input[base + i]);
    if (lid < 32u) lanes[lid] = floor_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    peak = simd_max(peak);
    if (lane == 0u) lanes[warp] = peak;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        float m = floor_value;
        for (uint j = 0u; j < 32u; j++) m = max(m, lanes[j]);
        shared_peak = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float m = shared_peak;

    float total = 0.0f;
    for (uint i = lid; i < columns; i += GROUP) total += exp(input[base + i] - m);
    if (lid < 32u) lanes[lid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = simd_sum(total);
    if (lane == 0u) lanes[warp] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        float s = 0.0f;
        for (uint j = 0u; j < 32u; j++) s += lanes[j];
        shared_total = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float denominator = shared_total;

    for (uint i = lid; i < columns; i += GROUP)
        output[base + i] = exp(input[base + i] - m) / denominator;
}

kernel void row_max(device const float* input [[buffer(0)]],
                    device float* output [[buffer(1)]],
                    constant uint& rows [[buffer(2)]],
                    constant uint& columns [[buffer(3)]],
                    uint row [[threadgroup_position_in_grid]],
                    uint lid [[thread_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint warp [[simdgroup_index_in_threadgroup]]) {
    if (row >= rows) return;
    const uint GROUP = 256u;
    threadgroup float lanes[32];
    threadgroup float shared_top;
    threadgroup atomic_uint shared_index;
    uint base = row * columns;
    float peak = -3.402823e+38f;
    uint best = 0xffffffffu;
    for (uint column = lid; column < columns; column += GROUP) {
        float value = input[base + column];
        if (value > peak) { peak = value; best = column; }
    }
    uint chosen = 0u;
    peak = group_peak(peak, best, lanes, &shared_top, &shared_index, lid, lane, warp, &chosen);
    if (lid == 0u) output[row] = peak;
}

kernel void row_argmax(device const float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       constant uint& rows [[buffer(2)]],
                       constant uint& columns [[buffer(3)]],
                       uint row [[threadgroup_position_in_grid]],
                       uint lid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint warp [[simdgroup_index_in_threadgroup]]) {
    if (row >= rows) return;
    const uint GROUP = 256u;
    threadgroup float lanes[32];
    threadgroup float shared_top;
    threadgroup atomic_uint shared_index;
    uint base = row * columns;
    float peak = -3.402823e+38f;
    uint best = 0xffffffffu;
    for (uint column = lid; column < columns; column += GROUP) {
        float value = input[base + column];
        if (value > peak) { peak = value; best = column; }
    }
    uint chosen = 0u;
    group_peak(peak, best, lanes, &shared_top, &shared_index, lid, lane, warp, &chosen);
    if (lid == 0u) output[row] = float(chosen);
}

kernel void row_peak(device const float* input [[buffer(0)]],
                     device float* values [[buffer(1)]],
                     device float* indices [[buffer(2)]],
                     constant uint& rows [[buffer(3)]],
                     constant uint& columns [[buffer(4)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint lid [[thread_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint warp [[simdgroup_index_in_threadgroup]]) {
    if (row >= rows) return;
    const uint GROUP = 256u;
    threadgroup float lanes[32];
    threadgroup float shared_top;
    threadgroup atomic_uint shared_index;
    uint base = row * columns;
    float peak = -3.402823e+38f;
    uint best = 0xffffffffu;
    for (uint column = lid; column < columns; column += GROUP) {
        float value = input[base + column];
        if (value > peak) { peak = value; best = column; }
    }
    uint chosen = 0u;
    peak = group_peak(peak, best, lanes, &shared_top, &shared_index, lid, lane, warp, &chosen);
    if (lid != 0u) return;
    values[row] = peak;
    indices[row] = float(chosen);
}

kernel void column_peak(device const float* input [[buffer(0)]],
                        device float* values [[buffer(1)]],
                        device float* indices [[buffer(2)]],
                        constant uint& rows [[buffer(3)]],
                        constant uint& columns [[buffer(4)]],
                        uint column [[thread_position_in_grid]]) {
    if (column >= columns) return;
    float peak = input[column];
    uint best = 0;
    for (uint row = 1; row < rows; ++row) {
        float value = input[row * columns + column];
        if (value > peak) { peak = value; best = row; }
    }
    values[column] = peak;
    indices[column] = float(best);
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
                             uint row [[threadgroup_position_in_grid]],
                             uint lid [[thread_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint warp [[simdgroup_index_in_threadgroup]]) {
    if (row >= rows) return;
    const uint GROUP = 256u;
    const uint base = row * columns;
    threadgroup float lanes[32];
    threadgroup float shared_dot;

    float dot = 0.0f;
    for (uint i = lid; i < columns; i += GROUP) dot += upstream[base + i] * output[base + i];
    if (lid < 32u) lanes[lid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dot = simd_sum(dot);
    if (lane == 0u) lanes[warp] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        float s = 0.0f;
        for (uint j = 0u; j < 32u; j++) s += lanes[j];
        shared_dot = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = shared_dot;

    for (uint i = lid; i < columns; i += GROUP)
        grad_input[base + i] = output[base + i] * (upstream[base + i] - total);
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

/* Max pooling's backward for windows that overlap, in two passes.
 *
 * One pass cannot do it: a thread holding an input element does not know
 * whether it won a window without scanning that window, and with overlapping
 * windows it is in several. The kernel this replaces did exactly that -- for
 * each of the windows covering it, rescan the whole window -- and so read k^4
 * elements to write one. A 3x3 stride-2 window over [64,56,56,64] took 2.90 ms
 * against torch's 1.07 for the same shape. These two passes take 0.43.
 *
 * So the scan happens once per window instead: the first pass writes, for each
 * output, which tap of its window won, and the second reads those back. The
 * taps are the same size as the output, a quarter of the input at stride two,
 * and the second pass reads each one only from the handful of inputs it can
 * speak for.
 *
 * The taps are floats rather than integers because that is the buffer the
 * tensor allocator hands out, and a tap index is small enough to be exact in
 * one. -1 means the window found no winner, which happens when every value in
 * it is NaN, or is smaller than the sentinel the scan starts from.
 *
 * The second pass walks the windows in descending order, which is the order
 * the single-pass form summed them in -- it counted taps up and so windows
 * down -- so a gradient that lands in more than one window adds up to the same
 * float it did before. */
kernel void max_pool2d_taps_nhwc(device const float* input [[buffer(0)]],
                                 device float* taps [[buffer(1)]],
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
    if (id >= N * OH * OW * C) return;
    uint rest = id / C;
    uint c = id - rest * C;
    uint row = rest / OW;
    uint ow = rest - row * OW;
    uint n = row / OH;
    uint oh = row - n * OH;
    float peak = -3.402823e+38f;
    float best = -1.0f;
    for (uint kh = 0u; kh < KH; ++kh) {
        int ih = int(oh * SH) - int(PH) + int(kh);
        if (ih < 0 || ih >= int(H)) continue;
        for (uint kw = 0u; kw < KW; ++kw) {
            int iw = int(ow * SW) - int(PW) + int(kw);
            if (iw < 0 || iw >= int(W)) continue;
            float v = input[((n * H + uint(ih)) * W + uint(iw)) * C + c];
            if (v > peak) { peak = v; best = float(kh * KW + kw); }
        }
    }
    taps[id] = best;
}

/* Four channels at a time; see the note on the forward pass.
 *
 * This is where the quads pay in this operation. Both passes decode a position
 * with three divisions and the device has no integer divide, and the two of
 * them together on [64,56,56,64] run at 0.44 ms with four channels a thread
 * against 1.20 scalar -- 2.7x, at 3x3 stride 2 and at 2x2 alike. The disjoint
 * arm gets nothing from the same change because it is already at the memory
 * ceiling; this arm reads the input, writes the taps, reads them back and
 * writes the gradient, and the arithmetic per byte is what a thread waits on.
 * A channel count that is not a multiple of four takes the scalar pair. */
kernel void max_pool2d_taps_nhwc4(device const float4* input [[buffer(0)]],
                                  device float4* taps [[buffer(1)]],
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
    uint quads = C >> 2u;
    if (id >= N * OH * OW * quads) return;
    uint rest = id / quads;
    uint q = id - rest * quads;
    uint row = rest / OW;
    uint ow = rest - row * OW;
    uint n = row / OH;
    uint oh = row - n * OH;
    float4 peak = float4(-3.402823e+38f);
    float4 best = float4(-1.0f);
    for (uint kh = 0u; kh < KH; ++kh) {
        int ih = int(oh * SH) - int(PH) + int(kh);
        if (ih < 0 || ih >= int(H)) continue;
        for (uint kw = 0u; kw < KW; ++kw) {
            int iw = int(ow * SW) - int(PW) + int(kw);
            if (iw < 0 || iw >= int(W)) continue;
            float4 v = input[((n * H + uint(ih)) * W + uint(iw)) * quads + q];
            bool4 took = v > peak;
            peak = select(peak, v, took);
            best = select(best, float4(float(kh * KW + kw)), took);
        }
    }
    taps[id] = best;
}

/* Which windows an input row can be in, given where it sits.
 *
 * Row `ih` is in window `oh` when `oh * S - P <= ih <= oh * S - P + K - 1`,
 * which bounds `oh` both ways. Working the bounds out costs two divisions,
 * where testing every tap for divisibility -- what the single-pass form did --
 * cost two per tap, and the device has no integer divide. */
static inline uint2 window_span(uint ih, uint K, uint S, uint P, uint O) {
    uint reach = ih + P;
    uint last = reach / S;
    if (last > O - 1u) last = O - 1u;
    int room = int(reach) - int(K) + 1;
    uint first = room <= 0 ? 0u : uint((room + int(S) - 1) / int(S));
    return uint2(first, last);
}

kernel void max_pool2d_backward_taps_nhwc(device const float* taps [[buffer(0)]],
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
    if (id >= N * H * W * C) return;
    uint rest = id / C;
    uint c = id - rest * C;
    uint row = rest / W;
    uint iw = rest - row * W;
    uint n = row / H;
    uint ih = row - n * H;
    uint2 down = window_span(ih, KH, SH, PH, OH);
    uint2 across = window_span(iw, KW, SW, PW, OW);
    float total = 0.0f;
    if (down.x <= down.y && across.x <= across.y) {
        for (uint oh = down.y + 1u; oh-- > down.x;) {
            uint kh = ih + PH - oh * SH;
            for (uint ow = across.y + 1u; ow-- > across.x;) {
                uint kw = iw + PW - ow * SW;
                uint out = ((n * OH + oh) * OW + ow) * C + c;
                if (taps[out] == float(kh * KW + kw)) total += upstream[out];
            }
        }
    }
    grad[id] = total;
}

kernel void max_pool2d_backward_taps_nhwc4(device const float4* taps [[buffer(0)]],
                                           device const float4* upstream [[buffer(1)]],
                                           device float4* grad [[buffer(2)]],
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
    uint quads = C >> 2u;
    if (id >= N * H * W * quads) return;
    uint rest = id / quads;
    uint q = id - rest * quads;
    uint row = rest / W;
    uint iw = rest - row * W;
    uint n = row / H;
    uint ih = row - n * H;
    uint2 down = window_span(ih, KH, SH, PH, OH);
    uint2 across = window_span(iw, KW, SW, PW, OW);
    float4 total = float4(0.0f);
    if (down.x <= down.y && across.x <= across.y) {
        for (uint oh = down.y + 1u; oh-- > down.x;) {
            uint kh = ih + PH - oh * SH;
            for (uint ow = across.y + 1u; ow-- > across.x;) {
                uint kw = iw + PW - ow * SW;
                uint out = ((n * OH + oh) * OW + ow) * quads + q;
                bool4 mine = taps[out] == float4(float(kh * KW + kw));
                total += select(float4(0.0f), upstream[out], mine);
            }
        }
    }
    grad[id] = total;
}

/* The same gradient, for windows that tile the input without overlapping.
 *
 * The two passes above are what an overlap costs: which tap won has to be
 * written down, because an input can win several windows and no one thread
 * sees them all. When the stride equals the window and the windows cover the
 * input exactly, neither is true -- each input is in one window and in one --
 * so a thread can take an output, find the peak over the k^2 inputs behind it,
 * and write that whole window at once: the upstream value to the winner, zero
 * to the rest. Every input is read once and written once, which is the least
 * this can move, and nothing is accumulated.
 *
 * 115 MB in 0.27 ms at [64,56,56,64] with a 2x2 window, against 1.52 for the
 * k^4 rescan this replaced and 0.96 for torch.
 *
 * The winner is the first of equal maxima in scan order, the one the two-pass
 * form picks as well, so the answer is the same element for element.
 *
 * The caller checks the tiling: a window that ran off the input, or one that
 * left a row uncovered, would read or leave rubbish. */
kernel void max_pool2d_backward_disjoint_nhwc(device const float* input [[buffer(0)]],
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
    if (id >= N * OH * OW * C) return;
    uint rest = id / C;
    uint c = id - rest * C;
    uint row = rest / OW;
    uint ow = rest - row * OW;
    uint n = row / OH;
    uint oh = row - n * OH;
    uint corner = (n * H + oh * KH) * W + ow * KW;
    float peak = -3.402823e+38f;
    uint best = 0xffffffffu;
    for (uint kh = 0u; kh < KH; ++kh) {
        for (uint kw = 0u; kw < KW; ++kw) {
            uint tap = kh * KW + kw;
            float v = input[(corner + kh * W + kw) * C + c];
            if (v > peak) { peak = v; best = tap; }
        }
    }
    float up = upstream[id];
    for (uint kh = 0u; kh < KH; ++kh) {
        for (uint kw = 0u; kw < KW; ++kw) {
            uint tap = kh * KW + kw;
            grad[(corner + kh * W + kw) * C + c] = tap == best ? up : 0.0f;
        }
    }
}

/* Four channels at a time, the way the forward pass takes them.
 *
 * The channels sit next to each other, so a thread holding four of them turns
 * four four-byte reads into one sixteen-byte read and works out where it is
 * once instead of four times -- which matters because the position costs three
 * divisions and the device has no integer divide.
 *
 * On [64,56,56,64] it buys nothing: 0.298 against the scalar form's 0.295,
 * because at 115 MB in 0.29 ms this arm is already moving bytes as fast as the
 * device will move them and no thread is waiting on arithmetic. What it is
 * worth is the shapes under that ceiling -- [8,56,56,64] 0.039 to 0.030,
 * [64,28,28,128] 0.112 to 0.096, [4,224,224,4] 0.026 to 0.022, all around
 * 1.15x to 1.3x. The large float4 win in this operation is on the two-pass arm
 * rather than here; see `max_pool2d_taps_nhwc4`.
 *
 * The peak is tracked per lane, since the four channels each have their own
 * winner, and `select` keeps that branchless. */
kernel void max_pool2d_backward_disjoint_nhwc4(device const float4* input [[buffer(0)]],
                                               device const float4* upstream [[buffer(1)]],
                                               device float4* grad [[buffer(2)]],
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
    uint quads = C >> 2u;
    if (id >= N * OH * OW * quads) return;
    uint rest = id / quads;
    uint q = id - rest * quads;
    uint row = rest / OW;
    uint ow = rest - row * OW;
    uint n = row / OH;
    uint oh = row - n * OH;
    uint corner = (n * H + oh * KH) * W + ow * KW;
    float4 peak = float4(-3.402823e+38f);
    uint4 best = uint4(0xffffffffu);
    for (uint kh = 0u; kh < KH; ++kh) {
        for (uint kw = 0u; kw < KW; ++kw) {
            uint4 tap = uint4(kh * KW + kw);
            float4 v = input[(corner + kh * W + kw) * quads + q];
            bool4 took = v > peak;
            peak = select(peak, v, took);
            best = select(best, tap, took);
        }
    }
    float4 up = upstream[id];
    for (uint kh = 0u; kh < KH; ++kh) {
        for (uint kw = 0u; kw < KW; ++kw) {
            uint4 tap = uint4(kh * KW + kw);
            grad[(corner + kh * W + kw) * quads + q] = select(float4(0.0f), up, best == tap);
        }
    }
}

/* An input pixel's gradient is the sum, over the windows covering it, of that
 * window's upstream value divided by the taps the window found inside the
 * image. Two loops used to do that and neither is needed.
 *
 * The covering windows are a contiguous run ending at (ih + PH) / SH, so one
 * division finds the last of them and the walk downwards stops as soon as a
 * window no longer reaches -- no modulo per offset, and nothing visited that
 * is then thrown away. The tap count is separable, rows(oh) * cols(ow), each
 * one window edge clamped against one image edge; counting it instead cost an
 * inner KH*KW loop per covering window, which is what max pooling's backward
 * does not have and why average pooling was the slower of the two here while
 * everywhere else it is the faster.
 *
 * oh and ow still descend, which is the order the offset loops visited taps
 * in, so the sum accumulates in the order it did before and comes out bit for
 * bit the same. On [64, 56, 56, 64] with a 2x2 window that took 1.43 ms to
 * 0.21, against torch's 0.66. */
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
    uint rest = id / C;
    uint c = id - rest * C;
    uint row = rest / W;
    uint iw = rest - row * W;
    uint n = row / H;
    uint ih = row - n * H;
    uint reach_h = ih + PH;
    uint reach_w = iw + PW;
    uint last_h = reach_h / SH;
    if (last_h >= OH) last_h = OH - 1u;
    uint last_w = reach_w / SW;
    if (last_w >= OW) last_w = OW - 1u;
    float total = 0.0f;
    for (int oh = int(last_h); oh >= 0; --oh) {
        if (reach_h - uint(oh) * SH >= KH) break;
        int top = int(uint(oh) * SH) - int(PH);
        uint rows = uint(min(top + int(KH), int(H)) - max(top, 0));
        for (int ow = int(last_w); ow >= 0; --ow) {
            if (reach_w - uint(ow) * SW >= KW) break;
            int left = int(uint(ow) * SW) - int(PW);
            uint cols = uint(min(left + int(KW), int(W)) - max(left, 0));
            total += upstream[((n * OH + uint(oh)) * OW + uint(ow)) * C + c] / float(rows * cols);
        }
    }
    grad[id] = total;
}

/* The same, four channels at a time, for the usual case of a channel count
 * that divides by four. It is the same arithmetic against a quarter as many
 * threads, and the index work above is per thread rather than per element, so
 * the quarter is most of what the shape below is worth. */
kernel void avg_pool2d_backward_nhwc4(device const float4* upstream [[buffer(0)]],
                                      device float4* grad [[buffer(1)]],
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
    uint quads = C >> 2u;
    if (id >= N * H * W * quads) return;
    uint rest = id / quads;
    uint q = id - rest * quads;
    uint row = rest / W;
    uint iw = rest - row * W;
    uint n = row / H;
    uint ih = row - n * H;
    uint reach_h = ih + PH;
    uint reach_w = iw + PW;
    uint last_h = reach_h / SH;
    if (last_h >= OH) last_h = OH - 1u;
    uint last_w = reach_w / SW;
    if (last_w >= OW) last_w = OW - 1u;
    float4 total = float4(0.0f);
    for (int oh = int(last_h); oh >= 0; --oh) {
        if (reach_h - uint(oh) * SH >= KH) break;
        int top = int(uint(oh) * SH) - int(PH);
        uint rows = uint(min(top + int(KH), int(H)) - max(top, 0));
        for (int ow = int(last_w); ow >= 0; --ow) {
            if (reach_w - uint(ow) * SW >= KW) break;
            int left = int(uint(ow) * SW) - int(PW);
            uint cols = uint(min(left + int(KW), int(W)) - max(left, 0));
            total += upstream[((n * OH + uint(oh)) * OW + uint(ow)) * quads + q] / float(rows * cols);
        }
    }
    grad[id] = total;
}

/* The gradient again, for windows that tile the input without overlapping.
 *
 * Then every input belongs to one window and every window is whole, so the
 * run of covering windows is a single one found by a division and the divisor
 * is the constant KH*KW -- no walk, no clamping against an edge that is never
 * reached. What is left is a gather with a scale, and it runs at the rate a
 * copy of the same bytes does: 0.21 ms to 0.167 on the shape above, 382 GB/s,
 * where upsample_nearest2d moves the same 64 MB in 0.166. There is nothing
 * further here to find. The caller checks the tiling; overlapping windows or
 * an uncovered row would read the wrong window. */
kernel void avg_pool2d_backward_tiled_nhwc(device const float* upstream [[buffer(0)]],
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
    if (id >= N * H * W * C) return;
    uint rest = id / C;
    uint c = id - rest * C;
    uint row = rest / W;
    uint iw = rest - row * W;
    uint n = row / H;
    uint ih = row - n * H;
    uint oh = ih / KH;
    uint ow = iw / KW;
    grad[id] = upstream[((n * OH + oh) * OW + ow) * C + c] / float(KH * KW);
}

kernel void avg_pool2d_backward_tiled_nhwc4(device const float4* upstream [[buffer(0)]],
                                            device float4* grad [[buffer(1)]],
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
    uint quads = C >> 2u;
    if (id >= N * H * W * quads) return;
    uint rest = id / quads;
    uint q = id - rest * quads;
    uint row = rest / W;
    uint iw = rest - row * W;
    uint n = row / H;
    uint ih = row - n * H;
    uint oh = ih / KH;
    uint ow = iw / KW;
    grad[id] = upstream[((n * OH + oh) * OW + ow) * quads + q] / float(KH * KW);
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

/* A threadgroup to the row, for the same reason the forward takes one: a
 * thread per row leaves a wide normalisation running on as many threads as
 * there are rows, which for a batch of images flattened to a few thousand
 * columns is a hundred-odd threads doing four passes each.
 *
 * The mean and the variance come out of one pass here rather than two --
 * sum(x) and sum(x*x) give both -- so the row is read three times, not four. */
kernel void layer_norm_backward(device const float* input [[buffer(0)]],
                                device const float* gamma [[buffer(1)]],
                                device const float* upstream [[buffer(2)]],
                                device float* grad [[buffer(3)]],
                                device float* row_mean [[buffer(4)]],
                                device float* row_inv [[buffer(5)]],
                                constant uint& rows [[buffer(6)]],
                                constant uint& width [[buffer(7)]],
                                uint row [[threadgroup_position_in_grid]],
                                uint lid [[thread_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint warp [[simdgroup_index_in_threadgroup]]) {
    if (row >= rows) return;
    const uint GROUP = 256u;
    const uint base = row * width;
    const float scale = 1.0f / float(width);
    threadgroup float lanes_a[32];
    threadgroup float lanes_b[32];
    threadgroup float shared_mean;
    threadgroup float shared_inv;
    threadgroup float shared_dy;
    threadgroup float shared_dy_xhat;

    /* The mean first, then the spread about it -- the same two passes the
     * forward makes, and for the same reason.
     *
     * Summing the squares alongside the values and taking `E[x^2] - E[x]^2`
     * reads the row once instead of twice, and was how this stood. Those two
     * terms are equal to seven digits once the mean is large beside the
     * spread, which is exactly the residual stream a normalisation is put
     * there to tame: at a mean of ten thousand and a spread of one they
     * cancel to nothing, `rsqrt` is handed a negative, and every one of the
     * row's gradients comes back NaN while the forward stays finite. A
     * second read of a row already in cache is the cheaper half of that
     * trade. */
    float total = 0.0f;
    if (lid < 32u) {
        lanes_a[lid] = 0.0f;
        lanes_b[lid] = 0.0f;
    }
    for (uint i = lid; i < width; i += GROUP) total += input[base + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = simd_sum(total);
    if (lane == 0u) lanes_a[warp] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        float sa = 0.0f;
        for (uint j = 0u; j < 32u; j++) sa += lanes_a[j];
        shared_mean = sa * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float square = 0.0f;
    for (uint i = lid; i < width; i += GROUP) {
        float d = input[base + i] - shared_mean;
        square += d * d;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    square = simd_sum(square);
    if (lane == 0u) lanes_b[warp] = square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        float sb = 0.0f;
        for (uint j = 0u; j < 32u; j++) sb += lanes_b[j];
        shared_inv = rsqrt(sb * scale + 1e-5f);
        /* Handed to the parameter pass, which is spread over columns and so
         * cannot reduce a row for itself. */
        row_mean[row] = shared_mean;
        row_inv[row] = shared_inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float m = shared_mean;
    const float inv = shared_inv;

    float sum_dy = 0.0f;
    float sum_dy_xhat = 0.0f;
    for (uint i = lid; i < width; i += GROUP) {
        float dy = upstream[base + i] * gamma[i];
        sum_dy += dy;
        sum_dy_xhat += dy * (input[base + i] - m) * inv;
    }
    if (lid < 32u) {
        lanes_a[lid] = 0.0f;
        lanes_b[lid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum_dy = simd_sum(sum_dy);
    sum_dy_xhat = simd_sum(sum_dy_xhat);
    if (lane == 0u) {
        lanes_a[warp] = sum_dy;
        lanes_b[warp] = sum_dy_xhat;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        float sa = 0.0f;
        float sb = 0.0f;
        for (uint j = 0u; j < 32u; j++) {
            sa += lanes_a[j];
            sb += lanes_b[j];
        }
        shared_dy = sa;
        shared_dy_xhat = sb;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean_dy = shared_dy * scale;
    const float mean_dy_xhat = shared_dy_xhat * scale;

    for (uint i = lid; i < width; i += GROUP) {
        float dy = upstream[base + i] * gamma[i];
        float xhat = (input[base + i] - m) * inv;
        grad[base + i] = inv * (dy - mean_dy - xhat * mean_dy_xhat);
    }
}

/* One thread to a column, reading the row statistics the input pass left
 * behind.
 *
 * Each thread used to reduce every row itself to recover that row's mean and
 * variance, which is the same work repeated once per column: for a row of four
 * thousand columns it turned a pass over the batch into billions of reads, and
 * a normalisation whose forward costs 0.03 ms had a backward costing 87 ms. */
kernel void layer_norm_params_backward(device const float* input [[buffer(0)]],
                                       device const float* upstream [[buffer(1)]],
                                       device float* grad_gamma [[buffer(2)]],
                                       device float* grad_beta [[buffer(3)]],
                                       device const float* row_mean [[buffer(4)]],
                                       device const float* row_inv [[buffer(5)]],
                                       constant uint& rows [[buffer(6)]],
                                       constant uint& width [[buffer(7)]],
                                       uint i [[thread_position_in_grid]]) {
    if (i >= width) return;
    float g = 0.0f;
    float b = 0.0f;
    for (uint row = 0; row < rows; ++row) {
        uint base = row * width;
        float dy = upstream[base + i];
        g += dy * (input[base + i] - row_mean[row]) * row_inv[row];
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
                              uint c [[threadgroup_position_in_grid]],
                              uint lid [[thread_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint warp [[simdgroup_index_in_threadgroup]]) {
    if (c >= channels) return;
    const uint GROUP = 256u;
    threadgroup float lanes[32];
    float inv = rsqrt(variance[c] + 1e-5f);
    float dy_total = 0.0f;
    float dy_xhat = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    for (uint i = lid; i < nspatial; i += GROUP) {
        uint slot = i * channels + c;
        float xhat = (input[slot] - mean[c]) * inv;
        float up = upstream[slot];
        float dy = up * gamma[c];
        dy_total += dy;
        dy_xhat += dy * xhat;
        g += up * xhat;
        b += up;
    }
    dy_total = group_total(dy_total, lanes, lid, lane, warp);
    dy_xhat = group_total(dy_xhat, lanes, lid, lane, warp);
    g = group_total(g, lanes, lid, lane, warp);
    b = group_total(b, lanes, lid, lane, warp);
    if (lid != 0u) return;
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
                            uint id [[threadgroup_position_in_grid]],
                            uint lid [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint warp [[simdgroup_index_in_threadgroup]]) {
    if (id >= N * groups) return;
    const uint GROUP = 256u;
    threadgroup float lanes[32];
    uint gs = C / groups;
    uint n = id / groups;
    uint g = id % groups;
    float inv = rsqrt(variance[id] + 1e-5f);
    float dy_total = 0.0f;
    float dy_xhat = 0.0f;
    for (uint hw = lid; hw < spatial; hw += GROUP) {
        uint base = (n * spatial + hw) * C + g * gs;
        for (uint k = 0; k < gs; ++k) {
            float xhat = (input[base + k] - mean[id]) * inv;
            float dy = upstream[base + k] * gamma[g * gs + k];
            dy_total += dy;
            dy_xhat += dy * xhat;
        }
    }
    dy_total = group_total(dy_total, lanes, lid, lane, warp);
    dy_xhat = group_total(dy_xhat, lanes, lid, lane, warp);
    if (lid != 0u) return;
    sum_dy[id] = dy_total;
    sum_dy_xhat[id] = dy_xhat;
}

/* The group backward's four sums in one coalesced pass.
 *
 * `group_norm_sums` and `channel_params_backward` between them read the whole
 * tensor twice, and neither read is coalesced: both walk one channel per
 * thread down the NHWC row stride, so a 256-thread group touches 256 separate
 * cache lines a step. Together they were 2.57 ms of the 2.96 ms backward on
 * [64,56,56,64] with eight groups -- 66 GB/s for the parameter pass and
 * 100 GB/s for the group pass, against the 396 GB/s the elementwise pass in
 * the same backward already reaches.
 *
 * This reads the tensor once, in order. A thread takes element `lid` of a
 * contiguous block and steps by 256, so the whole group reads one run of 256
 * floats a step; because the kernel only runs when the channel count divides
 * 256, the channel a thread lands on never changes and its slot, mean and
 * `rsqrt` are read once outside the loop. What comes out is the two sums per
 * (sample, channel) that every one of the four totals is built from:
 * `sum(upstream)` and `sum(upstream * xhat)`.
 *
 * The gamma weighting the group sums want is not applied here. Doing it in
 * the join costs C multiplies rather than one per element, and lets the same
 * partials serve the parameter gradients, which want the sums unweighted. */
kernel void group_norm_grad_partials(device const float* input [[buffer(0)]],
                                     device const float* mean [[buffer(1)]],
                                     device const float* variance [[buffer(2)]],
                                     device const float* upstream [[buffer(3)]],
                                     device float* part_dy [[buffer(4)]],
                                     device float* part_dy_xhat [[buffer(5)]],
                                     constant uint& spatial [[buffer(6)]],
                                     constant uint& C [[buffer(7)]],
                                     constant uint& groups [[buffer(8)]],
                                     constant uint& rows_each [[buffer(9)]],
                                     constant uint& per_sample [[buffer(10)]],
                                     uint chunk [[threadgroup_position_in_grid]],
                                     uint lid [[thread_index_in_threadgroup]]) {
    const uint GROUP = 256u;
    threadgroup float shared_dy[256];
    threadgroup float shared_xhat[256];
    /* A block never straddles two samples, so the slot -- and with it the
     * mean and the variance -- is fixed for the whole loop. */
    uint n = chunk / per_sample;
    uint within = chunk - n * per_sample;
    uint c = lid % C;
    uint slot = n * groups + c / (C / groups);
    float inv = rsqrt(variance[slot] + 1e-5f);
    float centre = mean[slot];
    uint base = n * spatial * C;
    uint first = within * rows_each;
    uint last = min(first + rows_each, spatial);
    float dy_total = 0.0f;
    float dy_xhat = 0.0f;
    for (uint f = base + first * C + lid; f < base + last * C; f += GROUP) {
        float up = upstream[f];
        dy_total += up;
        dy_xhat += up * (input[f] - centre) * inv;
    }
    shared_dy[lid] = dy_total;
    shared_xhat[lid] = dy_xhat;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid >= C) return;
    float a = 0.0f;
    float b = 0.0f;
    for (uint j = lid; j < GROUP; j += C) {
        a += shared_dy[j];
        b += shared_xhat[j];
    }
    part_dy[chunk * C + lid] = a;
    part_dy_xhat[chunk * C + lid] = b;
}

/* The group sums: over this sample's blocks, and over the channels of one
 * group, weighted by gamma. One thread a slot, of which there are N*groups. */
kernel void group_norm_slot_join(device const float* part_dy [[buffer(0)]],
                                 device const float* part_dy_xhat [[buffer(1)]],
                                 device const float* gamma [[buffer(2)]],
                                 device float* sum_dy [[buffer(3)]],
                                 device float* sum_dy_xhat [[buffer(4)]],
                                 constant uint& C [[buffer(5)]],
                                 constant uint& groups [[buffer(6)]],
                                 constant uint& per_sample [[buffer(7)]],
                                 constant uint& slots [[buffer(8)]],
                                 uint id [[thread_position_in_grid]]) {
    if (id >= slots) return;
    uint n = id / groups;
    uint g = id - n * groups;
    uint gs = C / groups;
    float dy = 0.0f;
    float dy_xhat = 0.0f;
    for (uint k = 0u; k < per_sample; ++k) {
        uint row = (n * per_sample + k) * C + g * gs;
        for (uint j = 0u; j < gs; ++j) {
            float w = gamma[g * gs + j];
            dy += w * part_dy[row + j];
            dy_xhat += w * part_dy_xhat[row + j];
        }
    }
    sum_dy[id] = dy;
    sum_dy_xhat[id] = dy_xhat;
}

/* The parameter gradients: the same partials summed down the other axis, over
 * every block of every sample. A threadgroup a channel rather than a thread,
 * because there are as many blocks as there are samples times the split and
 * that outgrows what one thread should walk. */
kernel void group_norm_param_join(device const float* part_dy [[buffer(0)]],
                                  device const float* part_dy_xhat [[buffer(1)]],
                                  device float* grad_gamma [[buffer(2)]],
                                  device float* grad_beta [[buffer(3)]],
                                  constant uint& chunks [[buffer(4)]],
                                  constant uint& C [[buffer(5)]],
                                  uint c [[threadgroup_position_in_grid]],
                                  uint lid [[thread_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint warp [[simdgroup_index_in_threadgroup]]) {
    if (c >= C) return;
    const uint GROUP = 256u;
    threadgroup float lanes[32];
    float dy = 0.0f;
    float dy_xhat = 0.0f;
    for (uint k = lid; k < chunks; k += GROUP) {
        dy += part_dy[k * C + c];
        dy_xhat += part_dy_xhat[k * C + c];
    }
    dy = group_total(dy, lanes, lid, lane, warp);
    dy_xhat = group_total(dy_xhat, lanes, lid, lane, warp);
    if (lid != 0u) return;
    grad_beta[c] = dy;
    grad_gamma[c] = dy_xhat;
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
                                    uint c [[threadgroup_position_in_grid]],
                                    uint lid [[thread_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint warp [[simdgroup_index_in_threadgroup]]) {
    if (c >= channels) return;
    const uint GROUP = 256u;
    threadgroup float lanes[32];
    uint gs = groups == 0u ? 0u : channels / groups;
    float g = 0.0f;
    float b = 0.0f;
    for (uint i = lid; i < nspatial; i += GROUP) {
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
    g = group_total(g, lanes, lid, lane, warp);
    b = group_total(b, lanes, lid, lane, warp);
    if (lid != 0u) return;
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

/* Channel statistics read the way the tensor is laid out.
 *
 * A channel's values are every C-th float of an NHWC tensor, so a threadgroup
 * that owns one channel reads with a stride and every lane of a simdgroup lands
 * on a different cache line. A threadgroup here owns a contiguous block of the
 * tensor and all its channels at once: lane `lid` always sees channel
 * `lid % channels`, because the group is a whole multiple of the channel count
 * and every block starts on a channel boundary. Consecutive lanes then read
 * consecutive floats.
 *
 * One block cannot cover a whole tensor and leave the GPU busy, so each writes
 * its own partial and a second pass adds them up. Only for channel counts that
 * divide the group; `moments_channel` still serves the rest.
 */
kernel void channel_partials(device const float* input [[buffer(0)]],
                             device float* part_total [[buffer(1)]],
                             device float* part_square [[buffer(2)]],
                             constant uint& nspatial [[buffer(3)]],
                             constant uint& channels [[buffer(4)]],
                             constant uint& rows_each [[buffer(5)]],
                             uint chunk [[threadgroup_position_in_grid]],
                             uint lid [[thread_index_in_threadgroup]]) {
    const uint GROUP = 256u;
    threadgroup float shared_total[256];
    threadgroup float shared_square[256];
    uint first = chunk * rows_each;
    uint last = min(first + rows_each, nspatial);
    /* Which channel this thread walks. The stride is GROUP and this kernel
     * only runs when the channel count divides it, so the channel a thread
     * touches never changes and the reference can be read once. */
    uint mine = lid % channels;
    float shift = input[mine];
    float total = 0.0f;
    float square = 0.0f;
    for (uint f = first * channels + lid; f < last * channels; f += GROUP) {
        float v = input[f];
        float d = v - shift;
        total += v;
        square += d * d;
    }
    shared_total[lid] = total;
    shared_square[lid] = square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid >= channels) return;
    float a = 0.0f;
    float b = 0.0f;
    for (uint j = lid; j < GROUP; j += channels) {
        a += shared_total[j];
        b += shared_square[j];
    }
    part_total[chunk * channels + lid] = a;
    part_square[chunk * channels + lid] = b;
}

/* The second pass: one thread to a channel, over however many blocks ran. */
/* The spread is taken about a value near the middle of the data, not about
 * zero.
 *
 * `E[x^2] - E[x]^2` is one pass and the obvious way to write it, and it
 * throws away every digit it has once the mean is large beside the spread:
 * an activation offset by a thousand, with a spread of one, makes those two
 * terms agree to seven figures, so the subtraction keeps none of them and
 * `rsqrt` is handed a negative. That is not a corner case -- batch norm on
 * such a block came back all NaN, in the FORWARD, not merely in the
 * gradient.
 *
 * Subtracting any fixed `k` first leaves the variance unchanged and makes
 * both terms O(spread^2) instead of O(mean^2). The first element of the
 * channel is a good enough `k` and costs one load: it is drawn from the same
 * distribution as everything being summed, which is all the trick needs. */
kernel void channel_moments_join(device const float* part_total [[buffer(0)]],
                                 device const float* part_square [[buffer(1)]],
                                 device float* mean [[buffer(2)]],
                                 device float* variance [[buffer(3)]],
                                 device const float* input [[buffer(4)]],
                                 constant uint& chunks [[buffer(5)]],
                                 constant uint& channels [[buffer(6)]],
                                 constant uint& nspatial [[buffer(7)]],
                                 uint c [[thread_position_in_grid]]) {
    if (c >= channels) return;
    float total = 0.0f;
    float square = 0.0f;
    for (uint k = 0; k < chunks; ++k) {
        total += part_total[k * channels + c];
        square += part_square[k * channels + c];
    }
    float m = total / float(nspatial);
    mean[c] = m;
    /* The partials hold the squares about `input[c]`, so the mean has to be
     * moved there too before the two are subtracted. */
    float offset = m - input[c];
    variance[c] = max(0.0f, square / float(nspatial) - offset * offset);
}

/* The backward's four sums, read the same coalesced way. */
kernel void channel_grad_partials(device const float* input [[buffer(0)]],
                                  device const float* mean [[buffer(1)]],
                                  device const float* variance [[buffer(2)]],
                                  device const float* gamma [[buffer(3)]],
                                  device const float* upstream [[buffer(4)]],
                                  device float* part_dy [[buffer(5)]],
                                  device float* part_dy_xhat [[buffer(6)]],
                                  constant uint& nspatial [[buffer(7)]],
                                  constant uint& channels [[buffer(8)]],
                                  constant uint& rows_each [[buffer(9)]],
                                  uint chunk [[threadgroup_position_in_grid]],
                                  uint lid [[thread_index_in_threadgroup]]) {
    const uint GROUP = 256u;
    threadgroup float shared_dy[256];
    threadgroup float shared_xhat[256];
    uint c = lid % channels;
    float inv = rsqrt(variance[c] + 1e-5f);
    float centre = mean[c];
    uint first = chunk * rows_each;
    uint last = min(first + rows_each, nspatial);
    float dy_total = 0.0f;
    float dy_xhat = 0.0f;
    for (uint f = first * channels + lid; f < last * channels; f += GROUP) {
        float up = upstream[f];
        dy_total += up;
        dy_xhat += up * (input[f] - centre) * inv;
    }
    shared_dy[lid] = dy_total;
    shared_xhat[lid] = dy_xhat;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid >= channels) return;
    float a = 0.0f;
    float b = 0.0f;
    for (uint j = lid; j < GROUP; j += channels) {
        a += shared_dy[j];
        b += shared_xhat[j];
    }
    part_dy[chunk * channels + lid] = a;
    part_dy_xhat[chunk * channels + lid] = b;
}

/* Joins the partials and turns them into what the two consumers want: the
 * parameter gradients are the raw sums, and the input pass wants them weighted
 * by gamma. */
kernel void channel_grad_join(device const float* part_dy [[buffer(0)]],
                              device const float* part_dy_xhat [[buffer(1)]],
                              device const float* gamma [[buffer(2)]],
                              device float* sum_dy [[buffer(3)]],
                              device float* sum_dy_xhat [[buffer(4)]],
                              device float* grad_gamma [[buffer(5)]],
                              device float* grad_beta [[buffer(6)]],
                              constant uint& chunks [[buffer(7)]],
                              constant uint& channels [[buffer(8)]],
                              uint c [[thread_position_in_grid]]) {
    if (c >= channels) return;
    float dy = 0.0f;
    float dy_xhat = 0.0f;
    for (uint k = 0; k < chunks; ++k) {
        dy += part_dy[k * channels + c];
        dy_xhat += part_dy_xhat[k * channels + c];
    }
    grad_beta[c] = dy;
    grad_gamma[c] = dy_xhat;
    sum_dy[c] = dy * gamma[c];
    sum_dy_xhat[c] = dy_xhat * gamma[c];
}

kernel void moments_channel(device const float* input [[buffer(0)]],
                            device float* mean [[buffer(1)]],
                            device float* variance [[buffer(2)]],
                            constant uint& nspatial [[buffer(3)]],
                            constant uint& channels [[buffer(4)]],
                            uint c [[threadgroup_position_in_grid]],
                            uint lid [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint warp [[simdgroup_index_in_threadgroup]]) {
    if (c >= channels) return;
    const uint GROUP = 256u;
    threadgroup float lanes[32];
    float shift = input[c];
    float total = 0.0f;
    float sq = 0.0f;
    for (uint i = lid; i < nspatial; i += GROUP) {
        float d = input[i * channels + c] - shift;
        total += d;
        sq += d * d;
    }
    total = group_total(total, lanes, lid, lane, warp);
    sq = group_total(sq, lanes, lid, lane, warp);
    if (lid != 0u) return;
    float centre = total / float(nspatial);
    mean[c] = shift + centre;
    variance[c] = max(0.0f, sq / float(nspatial) - centre * centre);
}

/* A threadgroup to a row, rather than a thread to a row.
 *
 * One thread a row leaves the device almost idle -- a 512 by 768 normalisation
 * is 512 threads where it wants tens of thousands -- and each of them walks
 * its own row three times over, which no neighbouring thread shares a cache
 * line with. It measured 1.7 ms against 0.18 for a matrix product over the
 * same numbers, and two of them sit in every transformer block.
 *
 * Here the group strides the row together, so consecutive threads take
 * consecutive elements, and the two sums come back through simd_sum. */
kernel void layer_norm_forward(device const float* input [[buffer(0)]],
                               device const float* gamma [[buffer(1)]],
                               device const float* beta [[buffer(2)]],
                               device float* output [[buffer(3)]],
                               constant uint& rows [[buffer(4)]],
                               constant uint& width [[buffer(5)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint lid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint warp [[simdgroup_index_in_threadgroup]]) {
    if (row >= rows) return;
    const uint GROUP = 256u;
    const uint base = row * width;
    threadgroup float lanes[32];
    threadgroup float shared_mean;
    threadgroup float shared_inv;

    float total = 0.0f;
    for (uint i = lid; i < width; i += GROUP) total += input[base + i];
    if (lid < 32u) lanes[lid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = simd_sum(total);
    if (lane == 0u) lanes[warp] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        float s = 0.0f;
        for (uint j = 0u; j < 32u; j++) s += lanes[j];
        shared_mean = s / float(width);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float m = shared_mean;

    float sq = 0.0f;
    for (uint i = lid; i < width; i += GROUP) {
        float d = input[base + i] - m;
        sq += d * d;
    }
    if (lid < 32u) lanes[lid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sq = simd_sum(sq);
    if (lane == 0u) lanes[warp] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        float s = 0.0f;
        for (uint j = 0u; j < 32u; j++) s += lanes[j];
        shared_inv = rsqrt(s / float(width) + 1e-5f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = shared_inv;

    for (uint i = lid; i < width; i += GROUP) {
        output[base + i] = gamma[i] * (input[base + i] - m) * inv + beta[i];
    }
}

kernel void gather_rows(device const float* src [[buffer(0)]],
                        device const float* indices [[buffer(1)]],
                        device float* dst [[buffer(2)]],
                        constant uint& rows [[buffer(3)]],
                        constant uint& width [[buffer(4)]],
                        constant uint& table [[buffer(5)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= rows * width) return;
    uint r = id / width;
    uint c = id % width;
    float asked = indices[r];
    /* An index off the end of the table, or a negative one -- which `uint`
     * turns into four billion rather than into a complaint. Reading there is
     * whatever happens to be next in the pool. A token id outside the
     * vocabulary is an ordinary mistake to make, so it reads as zero rather
     * than as somebody else's memory. */
    if (!(asked >= 0.0f) || uint(asked) >= table) {
        dst[id] = 0.0f;
        return;
    }
    dst[id] = src[uint(asked) * width + c];
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
                            uint row [[threadgroup_position_in_grid]],
                            uint lid [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint warp [[simdgroup_index_in_threadgroup]]) {
    if (row >= rows) return;
    const uint GROUP = 256u;
    threadgroup float lanes[32];
    uint base = row * columns;
    float total = 0.0f;
    for (uint c = lid; c < columns; c += GROUP) total += input[base + c];
    total = group_total(total, lanes, lid, lane, warp);
    if (lid == 0u) output[row] = total;
}

/* Each thread walks the array in strides of the whole grid and keeps a
 * running total in a register, so the expensive part -- combining the
 * threadgroup -- happens once per thousands of elements rather than once per
 * two hundred and fifty-six. The combine itself is a simdgroup reduction and
 * a sweep over the eight lane totals, which costs one barrier where the
 * shared-memory tree it replaces cost eight.
 *
 * A reduction reads its input and writes almost nothing, so it should run at
 * whatever the device can read. The version this replaces managed 46 GB/s on
 * four million floats where an elementwise pass over the same bytes managed
 * 432. */
kernel void reduce_sum_all(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant uint& count [[buffer(2)]],
                           constant uint& span [[buffer(3)]],
                           uint gid [[thread_position_in_grid]],
                           uint lid [[thread_index_in_threadgroup]],
                           uint wid [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint warp [[simdgroup_index_in_threadgroup]]) {
    float total = 0.0f;
    for (uint i = gid; i < count; i += span) total += input[i];

    threadgroup float lanes[32];
    if (lid < 32u) lanes[lid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = simd_sum(total);
    if (lane == 0u) lanes[warp] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        float sum = 0.0f;
        for (uint j = 0u; j < 32u; j++) sum += lanes[j];
        output[wid] = sum;
    }
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

/* The same, four channels at a time, for the usual case of a channel count
 * that divides by four.
 *
 * A thread per output element works out its own position, and the device has
 * no integer divide -- each one is a short sequence, and this kernel does
 * nothing else, so their count is the whole cost. Four channels at once
 * quarters them and moves sixteen bytes instead of four. The position is also
 * unpicked with one division a level and a subtraction for the remainder,
 * rather than a division and a modulo apiece. */
kernel void pad2d_nhwc4(device const float4* input [[buffer(0)]],
                        device float4* output [[buffer(1)]],
                        constant uint& N [[buffer(2)]],
                        constant uint& H [[buffer(3)]],
                        constant uint& W [[buffer(4)]],
                        constant uint& C [[buffer(5)]],
                        constant uint& PT [[buffer(6)]],
                        constant uint& PB [[buffer(7)]],
                        constant uint& PL [[buffer(8)]],
                        constant uint& PR [[buffer(9)]],
                        uint id [[thread_position_in_grid]]) {
    uint quads = C >> 2u;
    uint oh = H + PT + PB;
    uint ow = W + PL + PR;
    if (id >= N * oh * ow * quads) return;
    uint rest = id / quads;
    uint q = id - rest * quads;
    uint row = rest / ow;
    uint owi = rest - row * ow;
    uint n = row / oh;
    uint ohi = row - n * oh;
    int ih = int(ohi) - int(PT);
    int iw = int(owi) - int(PL);
    if (ih < 0 || iw < 0 || ih >= int(H) || iw >= int(W)) {
        output[id] = float4(0.0f);
        return;
    }
    output[id] = input[((n * H + uint(ih)) * W + uint(iw)) * quads + q];
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

/* Held to `TRANSPOSE_TILE` and `TRANSPOSE_GROUP` on the host. */
#define TR_TILE 32u
#define TR_ROWS 8u

/* `N` matrices of `rows` by `cols`, each handed back the other way round.
 *
 * Swapping two axes means one side of the copy is strided whichever way it is
 * written a thread at a time: laying out NCHW as NHWC wrote every fourth
 * kilobyte and took 2.0 ms on 64x64x56x56 where the peer took 0.9. Going via a
 * tile in threadgroup memory makes both sides sequential -- read a 32 by 32
 * square along its rows, write it back along its columns -- and the square is
 * held 33 wide so that consecutive rows of it land in different banks.
 *
 * Both directions are this one kernel: NHWC to NCHW transposes `H*W` by `C`,
 * and NCHW to NHWC transposes `C` by `H*W`. */
kernel void transpose_batched(device const float* input [[buffer(0)]],
                              device float* output [[buffer(1)]],
                              constant uint& rows [[buffer(2)]],
                              constant uint& cols [[buffer(3)]],
                              constant uint& across [[buffer(4)]],
                              constant uint& down [[buffer(5)]],
                              uint lid [[thread_index_in_threadgroup]],
                              uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float tile[TR_TILE][TR_TILE + 1u];
    uint per_image = across * down;
    uint n = wid / per_image;
    uint rest = wid - n * per_image;
    uint ty = rest / across;
    uint tx = rest - ty * across;
    uint lx = lid & (TR_TILE - 1u);
    uint ly = lid >> 5u;
    uint base = n * rows * cols;

    for (uint step = 0u; step < TR_TILE; step += TR_ROWS) {
        uint i = ty * TR_TILE + ly + step;
        uint j = tx * TR_TILE + lx;
        float value = 0.0f;
        if (i < rows && j < cols) value = input[base + i * cols + j];
        tile[ly + step][lx] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 0u; step < TR_TILE; step += TR_ROWS) {
        uint j = tx * TR_TILE + ly + step;
        uint i = ty * TR_TILE + lx;
        if (j < cols && i < rows) output[base + j * rows + i] = tile[lx][ly + step];
    }
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

/* The same, four at a time, when the row length divides by four -- which
 * keeps every copy of the row aligned to a float4 boundary. */
kernel void repeat_last4(device const float4* input [[buffer(0)]],
                         device float4* output [[buffer(1)]],
                         constant uint& rows [[buffer(2)]],
                         constant uint& inner [[buffer(3)]],
                         constant uint& times [[buffer(4)]],
                         uint id [[thread_position_in_grid]]) {
    uint quads = inner >> 2u;
    uint width = quads * times;
    if (id >= rows * width) return;
    uint row = id / width;
    uint col = id - row * width;
    uint copy = col / quads;
    output[id] = input[row * quads + (col - copy * quads)];
}

/* One division a level and a subtraction for the remainder, rather than a
 * division and a modulo apiece: the device has no integer divide, and a copy
 * kernel does nothing else. */
kernel void repeat_last(device const float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        constant uint& rows [[buffer(2)]],
                        constant uint& inner [[buffer(3)]],
                        constant uint& times [[buffer(4)]],
                        uint id [[thread_position_in_grid]]) {
    uint width = inner * times;
    if (id >= rows * width) return;
    uint row = id / width;
    uint col = id - row * width;
    uint copy = col / inner;
    output[id] = input[row * inner + (col - copy * inner)];
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

/* The pooling pair, four channels at a time, for the usual case of a channel
 * count that divides by four.
 *
 * A window is read once per channel and the channels sit next to each other,
 * so a thread that takes four of them does one sixteen-byte read where four
 * threads did four four-byte ones -- and works out its position in the output
 * once instead of four times, which matters because the device has no integer
 * divide. */
kernel void max_pool2d_nhwc4(device const float4* input [[buffer(0)]],
                             device float4* output [[buffer(1)]],
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
    uint quads = C >> 2u;
    if (id >= N * OH * OW * quads) return;
    uint rest = id / quads;
    uint q = id - rest * quads;
    uint row = rest / OW;
    uint ow = rest - row * OW;
    uint n = row / OH;
    uint oh = row - n * OH;
    float4 peak = float4(-3.402823e+38f);
    for (uint kh = 0u; kh < KH; ++kh) {
        int ih = int(oh * SH) - int(PH) + int(kh);
        if (ih < 0 || ih >= int(H)) continue;
        for (uint kw = 0u; kw < KW; ++kw) {
            int iw = int(ow * SW) - int(PW) + int(kw);
            if (iw < 0 || iw >= int(W)) continue;
            peak = max(peak, input[((n * H + uint(ih)) * W + uint(iw)) * quads + q]);
        }
    }
    output[id] = peak;
}

kernel void avg_pool2d_nhwc4(device const float4* input [[buffer(0)]],
                             device float4* output [[buffer(1)]],
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
    uint quads = C >> 2u;
    if (id >= N * OH * OW * quads) return;
    uint rest = id / quads;
    uint q = id - rest * quads;
    uint row = rest / OW;
    uint ow = rest - row * OW;
    uint n = row / OH;
    uint oh = row - n * OH;
    float4 total = float4(0.0f);
    uint seen = 0u;
    for (uint kh = 0u; kh < KH; ++kh) {
        int ih = int(oh * SH) - int(PH) + int(kh);
        if (ih < 0 || ih >= int(H)) continue;
        for (uint kw = 0u; kw < KW; ++kw) {
            int iw = int(ow * SW) - int(PW) + int(kw);
            if (iw < 0 || iw >= int(W)) continue;
            total += input[((n * H + uint(ih)) * W + uint(iw)) * quads + q];
            seen += 1u;
        }
    }
    output[id] = seen > 0u ? total / float(seen) : float4(0.0f);
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
    /* Unpicked with one division a level and a subtraction for the remainder,
     * rather than a division and a modulo apiece. The device has no integer
     * divide -- each one is a short sequence -- and this kernel does nothing
     * else, so the count of them is the whole cost. */
    uint rest = id / C;
    uint c = id - rest * C;
    uint row = rest / OW;
    uint ow = rest - row * OW;
    uint n = row / OH;
    uint oh = row - n * OH;
    /* Shifting instead of dividing when the scale is a power of two was tried
     * here and measured the same, so the branch is not carried. */
    uint ih = oh / SH;
    uint iw = ow / SW;
    output[id] = input[((n * H + ih) * W + iw) * C + c];
}

/* The same, four channels at a time, for the usual case of a channel count
 * that divides by four.
 *
 * A thread per output element unpicks its own position, which is four integer
 * divisions the device has no instruction for, and then moves four bytes.
 * Four channels at once quarters the divisions and turns the move into a
 * sixteen-byte one; the source address is unchanged across the group, so the
 * read is the same read. Worth 0.82 ms to 0.35 ms on 64x64x64x64. */
kernel void upsample_nearest2d_nhwc4(device const float4* input [[buffer(0)]],
                                     device float4* output [[buffer(1)]],
                                     constant uint& N [[buffer(2)]],
                                     constant uint& H [[buffer(3)]],
                                     constant uint& W [[buffer(4)]],
                                     constant uint& C [[buffer(5)]],
                                     constant uint& SH [[buffer(6)]],
                                     constant uint& SW [[buffer(7)]],
                                     constant uint& OH [[buffer(8)]],
                                     constant uint& OW [[buffer(9)]],
                                     uint id [[thread_position_in_grid]]) {
    uint quads = C >> 2u;
    if (id >= N * OH * OW * quads) return;
    uint rest = id / quads;
    uint q = id - rest * quads;
    uint row = rest / OW;
    uint ow = rest - row * OW;
    uint n = row / OH;
    uint oh = row - n * OH;
    uint ih = oh / SH;
    uint iw = ow / SW;
    output[id] = input[((n * H + ih) * W + iw) * quads + q];
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
                             constant uint& table [[buffer(5)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= gathered * width) return;
    uint r = id / width;
    uint c = id % width;
    float asked = indices[r];
    /* The same guard as the read, and it matters more here: this one is a
     * write, so an index past the end of the table does not read somebody
     * else's memory, it changes it. */
    if (!(asked >= 0.0f) || uint(asked) >= table) return;
    atomic_fetch_add_explicit(
        (device atomic<float>*)(dest + uint(asked) * width + c),
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
                          uint id [[threadgroup_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint warp [[simdgroup_index_in_threadgroup]]) {
    if (id >= N * groups) return;
    const uint GROUP = 256u;
    threadgroup float lanes[32];
    uint gs = C / groups;
    uint n = id / groups;
    uint g = id % groups;
    uint count = spatial * gs;
    float shift = input[(n * spatial) * C + g * gs];
    float total = 0.0f;
    float sq = 0.0f;
    for (uint hw = lid; hw < spatial; hw += GROUP) {
        uint base = (n * spatial + hw) * C + g * gs;
        for (uint c = 0; c < gs; ++c) {
            float d = input[base + c] - shift;
            total += d;
            sq += d * d;
        }
    }
    total = group_total(total, lanes, lid, lane, warp);
    sq = group_total(sq, lanes, lid, lane, warp);
    if (lid != 0u) return;
    float centre = total / float(count);
    mean[id] = shift + centre;
    variance[id] = max(0.0f, sq / float(count) - centre * centre);
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

#: Each unary operation as the expression it applies to `value`, in the order
#: their operation codes have always had.
#:
#: One kernel is generated per entry instead of the single `unary_op` that
#: switched on a code at runtime. The switch was not the cost -- the shared
#: body was: every branch's maths lives in one function, so the compiler sizes
#: its register use for the worst of them and the GPU runs far fewer threads at
#: once than it could. A specialised copy of SiLU measured 313 GB/s against the
#: shared kernel's 60 on the same buffers.
#:
#: Compiling more entry points is close to free: the Metal library is cached by
#: source text, so each of these costs a pipeline object and no recompilation.
let UNARY_FORMS = [
    ("copy", "value"),
    ("relu", "max(value, 0.0f)"),
    ("sigmoid", "1.0f / (1.0f + exp(-value))"),
    ("tanh", "tanh(value)"),
    ("exp", "exp(value)"),
    ("log", "log(value)"),
    ("abs", "fabs(value)"),
    ("neg", "-value"),
    ("sqrt", "sqrt(value)"),
    ("rsqrt", "rsqrt(value)"),
    ("square", "value * value"),
    (
        "gelu",
        "0.5f * value * (1.0f + tanh(0.7978845608f * (value + 0.044715f * value * value * " +
            "value)))"
    ),
    ("silu", "value / (1.0f + exp(-value))"),
    ("leaky_relu", "value > 0.0f ? value : 0.01f * value"),
]

#: The same, for the four elementwise binary operations.
let BINARY_FORMS = [("add", "+"), ("sub", "-"), ("mul", "*"), ("div", "/")]

#: The generated half of the shader library.
fn elementwise_kernels() -> str {
    var parts: list[str] = []
    for (name, expression) in UNARY_FORMS {
        parts.push(
            f"kernel void unary_{name}(device const float* input [[buffer(0)]],\n" +
                "                     device float* output [[buffer(1)]],\n" +
                "                     constant uint& count [[buffer(2)]],\n" +
                "                     uint id [[thread_position_in_grid]]) {\n" +
                "    if (id >= count) return;\n" +
                "    float value = input[id];\n" +
                f"    output[id] = {expression};\n" +
                "}\n"
        )
    }
    for (name, symbol) in BINARY_FORMS {
        parts.push(
            f"kernel void binary_{name}(device const float* a [[buffer(0)]],\n" +
                "                      device const float* b [[buffer(1)]],\n" +
                "                      device float* output [[buffer(2)]],\n" +
                "                      constant uint& count [[buffer(3)]],\n" +
                "                      uint id [[thread_position_in_grid]]) {\n" +
                "    if (id >= count) return;\n" +
                f"    output[id] = a[id] {symbol} b[id];\n" +
                "}\n"
        )
    }
    return "\n".join(parts)
}

let SOURCE = BASE_SOURCE + elementwise_kernels()

let ATTN_SOURCE = """#include <metal_stdlib>
#include <metal_simdgroup_matrix>
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

/* Row softmax of every head's scores at once, scaling as it reads.
 *
 * In place, and the aliasing is safe rather than lucky: a threadgroup owns
 * one row, the two reductions finish behind a barrier before the last loop
 * writes, and in that loop thread i writes only the element it just read.
 * Scaling here rather than in a separate `scale` costs nothing and saves two
 * passes over the score matrix, which at seq 1024 with 16 heads is 16.7M
 * elements read and written twice. The products are the same float32
 * multiplies the separate pass did, in the same order, so the answer is
 * bit-for-bit what `softmax(scale(scores, factor))` gave. */
kernel void mha_softmax_scaled(device float* scores [[buffer(0)]],
                               constant uint& rows [[buffer(1)]],
                               constant uint& columns [[buffer(2)]],
                               constant uint& scale_bits [[buffer(3)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint lid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint warp [[simdgroup_index_in_threadgroup]]) {
    if (row >= rows) return;
    const uint GROUP = 256u;
    const uint base = row * columns;
    const float factor = as_type<float>(scale_bits);
    const float floor_value = -3.402823e+38f;
    threadgroup float lanes[32];
    threadgroup float shared_peak;
    threadgroup float shared_total;

    float peak = floor_value;
    for (uint i = lid; i < columns; i += GROUP) peak = max(peak, scores[base + i] * factor);
    if (lid < 32u) lanes[lid] = floor_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    peak = simd_max(peak);
    if (lane == 0u) lanes[warp] = peak;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        float m = floor_value;
        for (uint j = 0u; j < 32u; j++) m = max(m, lanes[j]);
        shared_peak = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float m = shared_peak;

    float total = 0.0f;
    for (uint i = lid; i < columns; i += GROUP) total += exp(scores[base + i] * factor - m);
    if (lid < 32u) lanes[lid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = simd_sum(total);
    if (lane == 0u) lanes[warp] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        float s = 0.0f;
        for (uint j = 0u; j < 32u; j++) s += lanes[j];
        shared_total = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float denominator = shared_total;

    for (uint i = lid; i < columns; i += GROUP)
        scores[base + i] = exp(scores[base + i] * factor - m) / denominator;
}

/* The gradient of the above, also in place and for the same reason.
 *
 * `grad` arrives holding d(probs) and leaves holding d(scores); the reduction
 * has read every element it needs before the write loop starts. The factor
 * multiplies last, as it did when `scale` was the op after this one, so the
 * rounding is unchanged. */
kernel void mha_softmax_backward_scaled(device const float* probs [[buffer(0)]],
                                        device float* grad [[buffer(1)]],
                                        constant uint& rows [[buffer(2)]],
                                        constant uint& columns [[buffer(3)]],
                                        constant uint& scale_bits [[buffer(4)]],
                                        uint row [[threadgroup_position_in_grid]],
                                        uint lid [[thread_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint warp [[simdgroup_index_in_threadgroup]]) {
    if (row >= rows) return;
    const uint GROUP = 256u;
    const uint base = row * columns;
    const float factor = as_type<float>(scale_bits);
    threadgroup float lanes[32];
    threadgroup float shared_dot;

    float dot = 0.0f;
    for (uint i = lid; i < columns; i += GROUP) dot += grad[base + i] * probs[base + i];
    if (lid < 32u) lanes[lid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dot = simd_sum(dot);
    if (lane == 0u) lanes[warp] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        float s = 0.0f;
        for (uint j = 0u; j < 32u; j++) s += lanes[j];
        shared_dot = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = shared_dot;

    for (uint i = lid; i < columns; i += GROUP)
        grad[base + i] = factor * (probs[base + i] * (grad[base + i] - total));
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

/* Every head's Q K-transpose in one dispatch, on simdgroup matrices.
 *
 * The per-head loop this replaces issued one GEMM for each head, and the route
 * the GEMM router picks for [1024,64]x[64,1024] runs at 2.3 TFLOP/s where a
 * square 1024 product reaches 8.3 on the same device: the inner dimension is
 * 64, far too short for a tile to earn back the cost of loading its operands.
 * Sixteen of those measured 0.93 ms. Here one threadgroup owns a 64x64 patch
 * of one head's scores and every head is one grid, so the short inner
 * dimension is walked once per patch: 0.40 ms for the same sixteen.
 *
 * Two simdgroups split the patch by rows and each holds a 32x64 strip as
 * thirty-two 8x8 accumulators. Two rather than four because the accumulators
 * are the whole point: thirty-two of them issue thirty-two multiply-
 * accumulates against twelve fragment loads, where four simdgroups each
 * holding a 32x32 corner issued sixteen against eight, and the better ratio
 * measured 7% across the six dispatches.
 *
 * The unroll pragmas are load-bearing, not tidiness: indexed by a loop the
 * compiler leaves rolled, `acc` lives in thread-local memory instead of
 * registers and every accumulate becomes a spill and a reload. The 2x2 shape
 * measured 4.78 ms before the pragmas went in and 0.41 after -- a factor of
 * twelve on nothing but where sixteen matrices live.
 *
 * Q, K, V and the context stay in the caller's packed [seq, heads * hd]
 * layout: a fragment is loaded with the packed row stride and the head's
 * column offset, which costs an addition and removes the three `split_heads`
 * copies the composition needed in front and the three `merge_heads` copies
 * its backward needed behind -- 0.31 ms of pure transposition per call.
 *
 * `mha_mma_ok` guards all three of these on hd == 64 and seq a multiple of 64.
 * The fragments are 8x8 and nothing here masks a partial tile. */
kernel void mha_scores_mma(device const float* a [[buffer(0)]],
                           device const float* b [[buffer(1)]],
                           device float* out [[buffer(2)]],
                           constant uint& seq [[buffer(3)]],
                           constant uint& heads [[buffer(4)]],
                           constant uint& hd [[buffer(5)]],
                           constant uint& tiles [[buffer(6)]],
                           uint wid [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    uint dim = heads * hd;
    uint tile_m = wid % tiles;
    uint rest = wid / tiles;
    uint tile_n = rest % tiles;
    uint head = rest / tiles;
    uint m0 = tile_m * 64u + sg * 32u;
    uint n0 = tile_n * 64u;
    device const float* A = a + (ulong)head * hd + (ulong)m0 * dim;
    device const float* B = b + (ulong)head * hd + (ulong)n0 * dim;
    device float* C = out + (ulong)head * seq * seq + (ulong)m0 * seq + n0;
    simdgroup_float8x8 acc[4][8];
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 4u; ++i)
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 8u; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0u; k < hd; k += 8u) {
        simdgroup_float8x8 af[4];
        simdgroup_float8x8 bf[8];
#pragma clang loop unroll(full)
        for (uint i = 0u; i < 4u; ++i)
            simdgroup_load(af[i], A + (ulong)(i * 8u) * dim + k, dim);
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 8u; ++j)
            simdgroup_load(bf[j], B + (ulong)(j * 8u) * dim + k, dim, ulong2(0, 0), true);
#pragma clang loop unroll(full)
        for (uint i = 0u; i < 4u; ++i)
#pragma clang loop unroll(full)
            for (uint j = 0u; j < 8u; ++j)
                simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
    }
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 4u; ++i)
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 8u; ++j)
            simdgroup_store(acc[i][j], C + (ulong)(i * 8u) * seq + j * 8u, seq);
}

/* Every head's probabilities times its values, likewise in one dispatch.
 *
 * Same 64x64 patch and same pair of simdgroups as above; the patch spans the
 * whole of hd rather than a slice of it, because hd is 64. The backward pass
 * reaches this for dQ = dS K, the same product against a different pair of
 * operands. Sixteen heads: 0.76 ms as separate GEMMs, 0.36 here. */
kernel void mha_context_mma(device const float* p [[buffer(0)]],
                            device const float* v [[buffer(1)]],
                            device float* out [[buffer(2)]],
                            constant uint& seq [[buffer(3)]],
                            constant uint& heads [[buffer(4)]],
                            constant uint& hd [[buffer(5)]],
                            uint wid [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
    uint dim = heads * hd;
    uint tiles = seq / 64u;
    uint tile_m = wid % tiles;
    uint head = wid / tiles;
    uint m0 = tile_m * 64u + sg * 32u;
    device const float* A = p + (ulong)head * seq * seq + (ulong)m0 * seq;
    device const float* B = v + (ulong)head * hd;
    device float* C = out + (ulong)head * hd + (ulong)m0 * dim;
    simdgroup_float8x8 acc[4][8];
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 4u; ++i)
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 8u; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0u; k < seq; k += 8u) {
        simdgroup_float8x8 af[4];
        simdgroup_float8x8 bf[8];
#pragma clang loop unroll(full)
        for (uint i = 0u; i < 4u; ++i)
            simdgroup_load(af[i], A + (ulong)(i * 8u) * seq + k, seq);
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 8u; ++j)
            simdgroup_load(bf[j], B + (ulong)k * dim + j * 8u, dim);
#pragma clang loop unroll(full)
        for (uint i = 0u; i < 4u; ++i)
#pragma clang loop unroll(full)
            for (uint j = 0u; j < 8u; ++j)
                simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
    }
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 4u; ++i)
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 8u; ++j)
            simdgroup_store(acc[i][j], C + (ulong)(i * 8u) * dim + j * 8u, dim);
}

/* The score matrix transposed against a [seq, hd] operand, for dV and dK.
 *
 * Reading the transpose is free here -- `simdgroup_load` takes the flag and
 * the fragment arrives the right way round -- so neither gradient needs the
 * copy that a transposing GEMM would have made. Sixteen heads: 0.76 ms as
 * separate GEMMs, 0.37 here. */
kernel void mha_transposed_mma(device const float* p [[buffer(0)]],
                               device const float* u [[buffer(1)]],
                               device float* out [[buffer(2)]],
                               constant uint& seq [[buffer(3)]],
                               constant uint& heads [[buffer(4)]],
                               constant uint& hd [[buffer(5)]],
                               uint wid [[threadgroup_position_in_grid]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
    uint dim = heads * hd;
    uint tiles = seq / 64u;
    uint tile_m = wid % tiles;
    uint head = wid / tiles;
    uint m0 = tile_m * 64u + sg * 32u;
    device const float* A = p + (ulong)head * seq * seq + m0;
    device const float* B = u + (ulong)head * hd;
    device float* C = out + (ulong)head * hd + (ulong)m0 * dim;
    simdgroup_float8x8 acc[4][8];
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 4u; ++i)
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 8u; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0u; k < seq; k += 8u) {
        simdgroup_float8x8 af[4];
        simdgroup_float8x8 bf[8];
#pragma clang loop unroll(full)
        for (uint i = 0u; i < 4u; ++i)
            simdgroup_load(af[i], A + (ulong)k * seq + i * 8u, seq, ulong2(0, 0), true);
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 8u; ++j)
            simdgroup_load(bf[j], B + (ulong)k * dim + j * 8u, dim);
#pragma clang loop unroll(full)
        for (uint i = 0u; i < 4u; ++i)
#pragma clang loop unroll(full)
            for (uint j = 0u; j < 8u; ++j)
                simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
    }
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 4u; ++i)
#pragma clang loop unroll(full)
        for (uint j = 0u; j < 8u; ++j)
            simdgroup_store(acc[i][j], C + (ulong)(i * 8u) * dim + j * 8u, dim);
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
#: The threadgroup width every kernel in this module that reduces an axis
#: assumes it will get.
const KERNEL_GROUP = 256

#: Refuse a kernel the device will not run at the width its source assumes.
#:
#: The reducing kernels here stride their axis with `i += 256` and are asked
#: for with a threadgroup of 256, but the dispatch clamps that silently to
#: whatever the pipeline allows. A kernel whose registers or threadgroup
#: memory push its limit below 256 would then run, say, 128 threads while the
#: stride stayed 256 -- and elements 128 to 255 of every row would simply
#: never be read. A softmax missing half its denominator, a mean over half
#: its row, no error anywhere. It does not happen on this machine, where all
#: of them allow 1024, but it is a property of the driver and the device
#: rather than of this source, so it is checked rather than assumed.
fn checked(entry: str, kernel: Kernel) -> Kernel {
    let allowed = kernel.max_group_size()
    if allowed < KERNEL_GROUP {
        throw RuntimeError(
            f"kernel '{entry}' assumes a threadgroup of {KERNEL_GROUP} but this device allows {allowed}"
        )
    }
    return kernel
}

pub fn cached_kernel(entry: str) -> Kernel {
    let existing = _kernel_cache.get(entry, null)
    if existing is not null { return existing }
    let kernel = checked(entry, Kernel.compile(SOURCE, entry))
    _kernel_cache[entry] = kernel
    return kernel
}

var _attn_cache: dict[str, Kernel] = {}

#: The same, for the attention shader library.
fn cached_attn_kernel(entry: str) -> Kernel {
    let existing = _attn_cache.get(entry, null)
    if existing is not null { return existing }
    let kernel = checked(entry, Kernel.compile(ATTN_SOURCE, entry))
    _attn_cache[entry] = kernel
    return kernel
}

fn binary(a: Tensor, b: Tensor, operation: int) -> Tensor {
    if not a.same_shape(b) { throw ValueError(f"tensor shapes differ: {a.shape} and {b.shape}") }
    if operation < 0 or operation >= BINARY_FORMS.len() {
        throw ValueError(f"no binary operation has code {operation}")
    }
    let (name, _symbol) = BINARY_FORMS[operation]
    let output = Tensor(a.shape)
    let kernel = cached_kernel(f"binary_{name}")
    kernel.dispatch_async([a.buffer(), b.buffer(), output.buffer()], a.size, [a.size])
    return output
}

fn unary(input: Tensor, operation: int) -> Tensor {
    if operation < 0 or operation >= UNARY_FORMS.len() {
        throw ValueError(f"no unary operation has code {operation}")
    }
    let (name, _expression) = UNARY_FORMS[operation]
    let output = Tensor(input.shape)
    let kernel = cached_kernel(f"unary_{name}")
    kernel.dispatch_async([input.buffer(), output.buffer()], input.size, [input.size])
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
    #: hundred rows against a thousand columns. A convolution's is the other
    #: way round: thirty-two columns against a hundred and thirty thousand
    #: rows, so thirty-two threads did four million serial reads while the rest
    #: of the GPU sat idle.
    #:
    #: Splitting the rows fixes that, but only if it is split far enough. One
    #: pass with a couple of hundred chunks still left five hundred threads
    #: running on a machine that wants tens of thousands, and the reduction
    #: measured a millisecond and a third whatever the layer's size -- the
    #: signature of work that is waiting rather than reading. So it splits
    #: repeatedly, each pass turning `outer` rows into `chunks`, until what is
    #: left is small enough for one thread a column to finish quickly.
    #:
    #: The column index is what varies fastest across threads, which is what
    #: keeps neighbouring threads reading neighbouring addresses.
    let FINAL_ROWS = 64
    let TARGET_THREADS = 65536
    let MIN_ROWS_PER_CHUNK = 8
    var rows = outer
    var source = upstream
    var owned: Tensor? = null
    while rows > FINAL_ROWS {
        let chunks = min(max(1, TARGET_THREADS // inner), max(1, rows // MIN_ROWS_PER_CHUNK))
        if chunks >= rows { break }
        let partials = Tensor([chunks * inner])
        cached_kernel("reduce_leading_split").dispatch_async(
            [source.buffer(), partials.buffer()],
            inner * chunks,
            [rows, inner, chunks]
        )
        let previous = owned
        if previous is not null { previous.free() }
        owned = partials
        source = partials
        rows = chunks
    }

    if inner % 4 == 0 {
        cached_kernel("reduce_leading4").dispatch_async(
            [source.buffer(), output.buffer()],
            inner // 4,
            [rows, inner]
        )
    } else {
        cached_kernel("reduce_leading").dispatch_async(
            [source.buffer(), output.buffer()],
            inner,
            [rows, inner]
        )
    }
    let last = owned
    if last is not null { last.free() }
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

#: Multiply every element by one number that is itself on the device.
#:
#: `scale` takes a host float, which is no use to a backward pass: the number
#: it wants is the upstream gradient, and fetching that to the host to pass it
#: back down would mean waiting for the device mid-graph.
pub fn scale_by(input: Tensor, factor: Tensor) -> Tensor {
    if factor.size != 1 {
        throw ValueError(f"scale_by needs a single number, got {factor.shape}")
    }
    let output = Tensor(input.shape)
    cached_kernel("scalar_mul").dispatch_async(
        [input.buffer(), factor.buffer(), output.buffer()],
        input.size,
        [input.size]
    )
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
    let across = (columns + 31) // 32
    let down = (rows + 31) // 32
    kernel.dispatch_async(
        [input.buffer(), output.buffer()],
        across * down * 256,
        [rows, columns, across],
        group_size: 256
    )
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

#: `x @ weights + bias` with GELU applied on the way out.
#:
#: One kernel rather than two. The separate form writes the whole product to
#: memory and reads it straight back to shape it -- for a transformer's inner
#: layer that is a 256 by 3072 tensor written and read for nothing, and a
#: second dispatch behind it.
#:
#: Fused only when nothing here wants a gradient. GELU's derivative is a
#: function of what went *into* it, and a fused epilogue keeps only what came
#: out, so training takes the two-step path and keeps the intermediate it needs.
pub fn linear_gelu(x: Tensor, weights: Tensor, bias: Tensor) -> Tensor {
    if x.requires_grad or weights.requires_grad or bias.requires_grad {
        return gelu(dense_affine(x, weights, bias, 0))
    }
    return raw_matmul(x, weights, bias: bias, epilogue: 5)
}

#: `residual + x @ weights`, in one kernel.
#:
#: The shape a transformer block ends each half with. Written apart it is a
#: product into memory and then a pass that reads it, reads the residual and
#: writes a third tensor -- three times the traffic of the add for an operation
#: that is only an add, and a dispatch of its own behind the product.
#:
#: `residual` has to be the whole output's shape, not a row of it; the epilogue
#: that adds a vector is the one `dense_affine` uses. Fused only when nothing
#: wants a gradient, for the same reason `linear_gelu` is.
pub fn linear_residual(x: Tensor, weights: Tensor, residual: Tensor) -> Tensor {
    let rows = x.shape[0]
    let columns = weights.shape[weights.rank() - 1]
    if residual.rank() != 2 or residual.shape[0] != rows or residual.shape[1] != columns {
        throw ValueError(
            f"linear_residual needs a residual of [{rows}, {columns}], got {residual.shape}"
        )
    }
    if x.requires_grad or weights.requires_grad or residual.requires_grad {
        return add(residual, matmul(x, weights))
    }
    return raw_matmul(x, weights, bias: residual, epilogue: 6)
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
        #: Enough groups to fill the machine and no more. Each one sweeps its
        #: share of the input in a register, so giving every element its own
        #: thread would only add threadgroups to combine.
        let ELEMENTS_PER_GROUP = 2048
        let groups = min(1024, max(1, (input.size + ELEMENTS_PER_GROUP - 1) // ELEMENTS_PER_GROUP))
        let partials = Tensor([groups])
        let kernel = cached_kernel("reduce_sum_all")
        kernel.dispatch_async(
            [input.buffer(), partials.buffer()],
            groups * 256,
            [input.size, groups * 256],
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
    kernel.dispatch_async(
        [input.buffer(), output.buffer()],
        rows * 256,
        [rows, columns],
        group_size: 256
    )
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
    let wide = c % 4 == 0
    let narrow = if is_max { "max_pool2d_nhwc" } else { "avg_pool2d_nhwc" }
    let entry = if wide { narrow + "4" } else { narrow }
    let lanes = if wide { c // 4 } else { c }
    cached_kernel(entry).dispatch_async(
        [input.buffer(), output.buffer()],
        n * oh * ow * lanes,
        [n, h, w, c, kernel_size, kernel_size, step, step, padding, padding, oh, ow]
    )
    if input.requires_grad {
        output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
            let grad = Tensor([n, h, w, c])
            let scalars = [n, h, w, c, kernel_size, kernel_size, step, step, padding, padding, oh, ow]
            if is_max {
                #: Windows that tile the input exactly cover every element
                #: once and no element twice, so a thread can take a window,
                #: scan it once and write the whole of it -- no rescanning, and
                #: nothing to add up. That is the shape pooling almost always
                #: has, 2x2 stride 2 over an even picture, and it was worth
                #: 5.6x. Anything else goes the two-pass way.
                let tiles = padding == 0 and step == kernel_size
                let covers = oh * kernel_size == h and ow * kernel_size == w
                if tiles and covers {
                    let name = if wide { "max_pool2d_backward_disjoint_nhwc4" } else { "max_pool2d_backward_disjoint_nhwc" }
                    let back = cached_kernel(name)
                    back.dispatch_async([input.buffer(), upstream.buffer(), grad.buffer()], n * oh * ow * lanes, scalars)
                } else {
                    let taps = Tensor([n, oh, ow, c])
                    let mark = if wide { "max_pool2d_taps_nhwc4" } else { "max_pool2d_taps_nhwc" }
                    let spread = if wide { "max_pool2d_backward_taps_nhwc4" } else { "max_pool2d_backward_taps_nhwc" }
                    cached_kernel(mark).dispatch_async([input.buffer(), taps.buffer()], n * oh * ow * lanes, scalars)
                    cached_kernel(spread).dispatch_async([taps.buffer(), upstream.buffer(), grad.buffer()], n * h * w * lanes, scalars)
                    taps.free()
                }
            } else {
                let tiles = padding == 0 and step == kernel_size
                let covers = oh * kernel_size == h and ow * kernel_size == w
                let whole = tiles and covers
                let entry = if whole {
                    if wide { "avg_pool2d_backward_tiled_nhwc4" } else { "avg_pool2d_backward_tiled_nhwc" }
                } elif wide {
                    "avg_pool2d_backward_nhwc4"
                } else {
                    "avg_pool2d_backward_nhwc"
                }
                let threads = if wide { grad.size // 4 } else { grad.size }
                cached_kernel(entry).dispatch_async([upstream.buffer(), grad.buffer()], threads, scalars)
            }
            return [grad]
        }))
    }
    return output
}

#: How many masks have been drawn, so that the next one differs from the last.
var _dropout_draws = 0

#: Inverted dropout. `p` is the drop probability.
#:
#: `seed` fixes the mask, for a test that wants the same one twice. Left out,
#: each call draws a fresh one.
#:
#: The seed used to be the tensor's own size, which is the same number on
#: every step -- so the same units were dropped every step of every epoch,
#: and two layers of the same shape dropped the same ones as each other. That
#: is not regularisation, it is a fixed pruning mask chosen by the shape, and
#: being perfectly repeatable it never looked wrong.
pub fn dropout(input: Tensor, p: float = 0.5, training: bool = true, seed: int? = null) -> Tensor {
    if p < 0.0 or p >= 1.0 { throw ValueError(f"dropout p must be in [0, 1), got {p}") }
    if not training or p == 0.0 {
        return reshape(input, input.shape)
    }
    let noise = Tensor(input.shape)
    let asked = if seed is null {
        _dropout_draws += 1
        _dropout_draws * 2654435761 + input.size
    } else {
        seed
    }
    noise.buffer().fill_uniform(0.0, 1.0, asked)
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
    let wide = c % 4 == 0
    let entry = if wide { "pad2d_nhwc4" } else { "pad2d_nhwc" }
    let threads = if wide { output.size // 4 } else { output.size }
    cached_kernel(entry).dispatch_async(
        [input.buffer(), output.buffer()],
        threads,
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
#: Tiles a side, and the group that walks one. Matched by `TR_TILE` and
#: `TR_ROWS` in the kernel source.
const TRANSPOSE_TILE = 32
const TRANSPOSE_GROUP = 256

#: `count` matrices of `rows` by `cols`, each written back the other way up.
fn dispatch_transpose(input: Tensor, output: Tensor, count: int, rows: int, cols: int) -> void {
    let across = (cols + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE
    let down = (rows + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE
    cached_kernel("transpose_batched").dispatch_async(
        [input.buffer(), output.buffer()],
        count * across * down * TRANSPOSE_GROUP,
        [rows, cols, across, down],
        group_size: TRANSPOSE_GROUP
    )
}

pub fn nchw(input: Tensor) -> Tensor {
    if input.rank() != 4 { throw ValueError(f"nchw needs rank four, got {input.shape}") }
    let n = input.shape[0]
    let h = input.shape[1]
    let w = input.shape[2]
    let c = input.shape[3]
    let output = Tensor([n, c, h, w])
    dispatch_transpose(input, output, n, h * w, c)
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
    dispatch_transpose(input, output, n, c, h * w)
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
    let wide = inner % 4 == 0
    let entry = if wide { "repeat_last4" } else { "repeat_last" }
    let threads = if wide { output.size // 4 } else { output.size }
    cached_kernel(entry).dispatch_async(
        [input.buffer(), output.buffer()],
        threads,
        [rows, inner, times]
    )
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
    let wide = c % 4 == 0
    let entry = if wide { "upsample_nearest2d_nhwc4" } else { "upsample_nearest2d_nhwc" }
    let threads = if wide { output.size // 4 } else { output.size }
    cached_kernel(entry).dispatch_async(
        [input.buffer(), output.buffer()],
        threads,
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
    kernel.dispatch_async(
        [input.buffer(), output.buffer()],
        rows * 256,
        [rows, columns],
        group_size: 256
    )
    if input.requires_grad {
        output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
            let grad_input = Tensor(input.shape)
            let backward_kernel = cached_kernel("softmax_backward")
            backward_kernel.dispatch_async([output.buffer(), upstream.buffer(), grad_input.buffer()], rows * 256, [rows, columns], group_size: 256)
            return [grad_input]
        }))
    }
    return output
}

#: Index of the largest value in each row of a rank-two tensor.
#:
#: Indices use float storage because all Metal buffers in `std.gpu` are
#: float32. Downloaded values are exact for practical class counts.
#: The largest value along the last axis, or of the whole tensor.
#:
#: The companion to `argmax`, which gives where the largest is but not what it
#: is. A detector wants both -- the class and its score -- and computing the
#: value on the host afterwards means downloading everything the reduction was
#: supposed to save.
pub fn amax(input: Tensor, axis: int? = null) -> Tensor {
    if axis is null {
        let flat = reshape(input, [1, input.size])
        let whole = amax(flat, axis: 1)
        flat.free()
        return whole
    }
    let resolved = if axis < 0 { input.rank() + axis } else { axis }
    if resolved != input.rank() - 1 {
        throw ValueError("amax reduces all elements or the last axis")
    }
    let columns = input.shape[resolved]
    let rows = input.size // columns
    var shape: list[int] = []
    for index in 0..input.rank() - 1 { shape.push(input.shape[index]) }
    if shape.len() == 0 { shape.push(1) }
    let output = Tensor(shape)
    cached_kernel("row_max").dispatch_async(
        [input.buffer(), output.buffer()],
        rows * 256,
        [rows, columns],
        group_size: 256
    )
    return output
}

pub fn argmax(input: Tensor) -> Tensor {
    if input.rank() != 2 { throw ValueError(f"argmax needs rank two, got {input.shape}") }
    let rows = input.shape[0]
    let columns = input.shape[1]
    let output = Tensor([rows])
    let kernel = cached_kernel("row_argmax")
    kernel.dispatch_async(
        [input.buffer(), output.buffer()],
        rows * 256,
        [rows, columns],
        group_size: 256
    )
    return output
}

#: The largest value along an axis and where it sits, in one pass.
#:
#: `amax` and `argmax` each read the whole tensor to work out the same thing,
#: so asking for both costs twice what it needs to. Whoever wants only one of
#: them should still say so. On the 8400 by 80 class scores a detector produces
#: per frame the pair now costs 0.137 ms, against 0.65 ms when a row was one
#: thread's serial walk.
#:
#: Reducing down the columns rather than along the rows is a separate kernel
#: rather than a transpose and the other one. The transpose is a whole extra
#: pass over the tensor and an allocation the size of it -- 0.3 ms and 2.7 MB
#: for those same scores -- and reading a column is no slower: neighbouring
#: threads take neighbouring elements of each row in turn, which is the access
#: the device wants.
pub fn max_with_index(input: Tensor, axis: int = -1) -> tuple[Tensor, Tensor] {
    if input.rank() != 2 {
        throw ValueError(f"max_with_index needs rank two, got {input.shape}")
    }
    let resolved = if axis < 0 { input.rank() + axis } else { axis }
    if resolved != 0 and resolved != 1 {
        throw ValueError(f"max_with_index takes axis 0 or 1, got {axis}")
    }
    let rows = input.shape[0]
    let columns = input.shape[1]
    let kept = if resolved == 1 { rows } else { columns }
    let values = Tensor([kept])
    let indices = Tensor([kept])
    #: Down a row a threadgroup reduces together; across one, neighbouring
    #: threads already take neighbouring elements and a thread each is right.
    if resolved == 1 {
        cached_kernel("row_peak").dispatch_async(
            [input.buffer(), values.buffer(), indices.buffer()],
            kept * 256,
            [rows, columns],
            group_size: 256
        )
    } else {
        cached_kernel("column_peak").dispatch_async(
            [input.buffer(), values.buffer(), indices.buffer()],
            kept,
            [rows, columns]
        )
    }
    return (values, indices)
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
        throw ValueError(f"conv2d_nchw channel mismatch: input {cin} vs weights {weights.shape[1]}")
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
        gpu.conv2d_async(input.buffer(), weights.buffer(), output.buffer(), n, h, w, cin, cout, kh, kw, stride_h: stride, stride_w: stride, pad_h: padding, pad_w: padding, bias: if bias is null {
            null
        } else {
            bias.buffer()
        }, activation: activation, layout: 1)
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
        let done = conv2d(nhwc_in, hwio, bias: bias, stride: stride, padding: padding, activation: if activation == 2 {
            0
        } else {
            activation
        })
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
    #: Only the gradients something downstream will read.
    #:
    #: The first convolution of a network is fed the batch itself, which is a
    #: leaf and wants no gradient -- and the gradient with respect to the input
    #: costs a whole second convolution and an activation-sized allocation to
    #: produce. Asking for it anyway was about a sixth of what the backward
    #: pass of a two-layer network did.
    let wants_input = input.requires_grad
    let wants_weights = weights.requires_grad
    let wants_bias = bias is not null and bias.requires_grad
    if not (wants_input or wants_weights or wants_bias) { return }
    var inputs: list[Tensor] = []
    if wants_input { inputs.push(input) }
    if wants_weights { inputs.push(weights) }
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
        var grad_weights: Tensor? = null
        var grad_input: Tensor? = null
        if wants_weights { grad_weights = Tensor([kh, kw, cin, cout]) }
        if wants_input { grad_input = Tensor([n, h, w, cin]) }

        #: The device's own convolution gradients when it has them. The im2col
        #: pair below computes the same two tensors, and does it by building a
        #: `[n * out_h * out_w, kh * kw * cin]` matrix twice per step: a
        #: hundred and fifty megabytes allocated, filled and dropped on every
        #: batch of a sixty-four-channel layer. Either may be declined on its
        #: own -- a shape MPS will not take, or no device at all -- so each is
        #: asked for separately and each has its own fallback.
        #: Each gradient is asked for on its own: either may be declined for
        #: a shape MPS will not take, and either may not be wanted at all.
        var have_weights_grad = not wants_weights
        var have_input_grad = not wants_input
        try {
            if wants_weights {
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
            }
            if wants_input {
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
            }
        } catch _error: RuntimeError {
            have_weights_grad = not wants_weights
            have_input_grad = not wants_input
        }

        #: The im2col pair computes the same two tensors when the device
        #: declines. It builds an `[n * out_h * out_w, kh * kw * cin]` matrix
        #: to do it -- a hundred and fifty megabytes for a sixty-four-channel
        #: layer -- which is why it is the fallback and not the path.
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

        #: Same order as `inputs`, which listed only what wanted a gradient.
        var grads: list[Tensor] = []
        if wants_input { grads.push(grad_input) }
        if wants_weights { grads.push(grad_weights) }
        if wants_bias { grads.push(reduce_leading(flat_up, [cout])) }
        if activation == 1 { upstream.free() }
        return grads
    }))
}

#: Channel-last batch normalization using per-channel mean and variance of `x`.
#: Blocks for the coalesced channel reduction, or zero when it does not apply.
#:
#: Each block's threads stay on one channel apiece only if the group holds a
#: whole number of channels, and the split has to be worth making: a couple of
#: rows a block spends more on joining the partials than the first pass saves.
fn channel_chunks(nspatial: int, channels: int) -> int {
    if channels > 256 or 256 % channels != 0 { return 0 }
    let wanted = min(128, nspatial // 64)
    if wanted < 2 { return 0 }
    return wanted
}

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
    let chunks = channel_chunks(nspatial, channels)
    if chunks > 0 {
        let rows_each = (nspatial + chunks - 1) // chunks
        let part_total = Tensor([chunks * channels])
        let part_square = Tensor([chunks * channels])
        cached_kernel("channel_partials").dispatch_async(
            [input.buffer(), part_total.buffer(), part_square.buffer()],
            chunks * 256,
            [nspatial, channels, rows_each],
            group_size: 256
        )
        cached_kernel("channel_moments_join").dispatch_async(
            [
                part_total.buffer(),
                part_square.buffer(),
                mean.buffer(),
                variance.buffer(),
                input.buffer(),
            ],
            channels,
            [chunks, channels, nspatial]
        )
        part_total.free()
        part_square.free()
    } else {
        cached_kernel("moments_channel").dispatch_async(
            [input.buffer(), mean.buffer(), variance.buffer()],
            channels * 256,
            [nspatial, channels],
            group_size: 256
        )
    }
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
            let split = channel_chunks(nspatial, channels)
            if split > 0 {
                let rows_each = (nspatial + split - 1) // split
                let part_dy = Tensor([split * channels])
                let part_dy_xhat = Tensor([split * channels])
                cached_kernel("channel_grad_partials").dispatch_async([input.buffer(), mean.buffer(), variance.buffer(), gamma.buffer(), upstream.buffer(), part_dy.buffer(), part_dy_xhat.buffer()], split * 256, [nspatial, channels, rows_each], group_size: 256)
                cached_kernel("channel_grad_join").dispatch_async([part_dy.buffer(), part_dy_xhat.buffer(), gamma.buffer(), sum_dy.buffer(), sum_dy_xhat.buffer(), grad_gamma.buffer(), grad_beta.buffer()], channels, [split, channels])
                part_dy.free()
                part_dy_xhat.free()
            } else {
                cached_kernel("norm_channel_sums").dispatch_async([input.buffer(), mean.buffer(), variance.buffer(), gamma.buffer(), upstream.buffer(), sum_dy.buffer(), sum_dy_xhat.buffer(), grad_gamma.buffer(), grad_beta.buffer()], channels * 256, [nspatial, channels], group_size: 256)
            }
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
    #: A threadgroup of 256 to each row, so the dispatch is that many threads
    #: per row rather than one.
    kernel.dispatch_async(
        [input.buffer(), gamma.buffer(), beta.buffer(), output.buffer()],
        input.shape[0] * 256,
        [input.shape[0], width],
        group_size: 256
    )
    if input.requires_grad or gamma.requires_grad or beta.requires_grad {
        let rows = input.shape[0]
        output.attach_creator(GradNode([input, gamma, beta], fn(upstream: Tensor) -> list[Tensor] {
            let grad_input = Tensor(input.shape)
            #: Written by the input pass and read by the parameter pass, which
            #: runs after it on the same queue.
            let row_mean = Tensor([rows])
            let row_inv = Tensor([rows])
            let back = cached_kernel("layer_norm_backward")
            back.dispatch_async(
                [
                    input.buffer(),
                    gamma.buffer(),
                    upstream.buffer(),
                    grad_input.buffer(),
                    row_mean.buffer(),
                    row_inv.buffer(),
                ],
                rows * 256,
                [rows, width],
                group_size: 256
            )
            let grad_gamma = Tensor([width])
            let grad_beta = Tensor([width])
            let params = cached_kernel("layer_norm_params_backward")
            params.dispatch_async(
                [
                    input.buffer(),
                    upstream.buffer(),
                    grad_gamma.buffer(),
                    grad_beta.buffer(),
                    row_mean.buffer(),
                    row_inv.buffer(),
                ],
                width,
                [rows, width]
            )
            row_mean.free()
            row_inv.free()
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
        if mha_mma_ok(seq, hd) { return mha_by_mma(query, key, value, heads, seq, hd, factor) }
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

#: Whether the batched simdgroup-matrix attention kernels can take this shape.
#:
#: They tile in 8x8 fragments with no masking, so the sequence has to fill
#: whole 64-row patches, and the head dimension has to be exactly the 64 the
#: 2x2 of simdgroups covers. Everything else keeps `mha_by_ops`, which has no
#: shape restriction at all.
fn mha_mma_ok(seq: int, hd: int) -> bool {
    return hd == 64 and seq % 64 == 0
}

#: Multi-head attention as six batched dispatches, forward and backward.
#:
#: `mha_by_ops` below is the same arithmetic written as ops, and it is what
#: runs whenever `mha_mma_ok` says no. What this saves over it is not a
#: different algorithm but a different shape of work. That version issues one
#: GEMM per head, and the route the GEMM router picks for the two attention
#: shapes -- [seq, 64] against [64, seq], and [seq, seq] against [seq, 64] --
#: reaches 2.3 and 2.8 TFLOP/s where a square 1024 product reaches 8.3 on the
#: same device. At seq 1024, dim 1024 and 16 heads the whole forward and
#: backward took 8.39 ms that way, of which the two forward loops alone were
#: 0.93 and 0.76. The six dispatches here measure 2.25 together, dropping
#: `split_heads` and `merge_heads` -- the kernels read the packed layout
#: directly -- takes another 0.31 off, and the pass now measures 3.15 against
#: torch's 4.36 for the same shape, both timed with the graph rebuilt each
#: iteration because backward consumes it.
#:
#: The score matrix is still written out, unlike the fused inference kernel,
#: because the backward pass has to read it. Nothing else is materialised.
fn mha_by_mma(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    heads: int,
    seq: int,
    hd: int,
    factor: float
) -> Tensor {
    let tiles = seq // 64
    let probs = Tensor([heads * seq, seq])
    cached_attn_kernel("mha_scores_mma").dispatch_async(
        [query.buffer(), key.buffer(), probs.buffer()],
        tiles * tiles * heads * 64,
        [seq, heads, hd, tiles],
        group_size: 64
    )
    cached_attn_kernel("mha_softmax_scaled").dispatch_async(
        [probs.buffer()],
        heads * seq * 256,
        [heads * seq, seq, f32_bits(factor)],
        group_size: 256
    )
    let output = Tensor([seq, heads * hd])
    cached_attn_kernel("mha_context_mma").dispatch_async(
        [probs.buffer(), value.buffer(), output.buffer()],
        tiles * heads * 64,
        [seq, heads, hd],
        group_size: 64
    )
    if query.requires_grad or key.requires_grad or value.requires_grad {
        output.attach_creator(GradNode([query, key, value], fn(upstream: Tensor) -> list[Tensor] {
            return mha_mma_grads(upstream, query, key, value, probs, heads, seq, hd, factor)
        }, extras: [probs]))
    }
    return output
}

#: Gradients of `mha_by_mma`, in four dispatches around the softmax backward.
#:
#: The four products are the ones `matmul`'s backward would have chosen, in
#: the same operand order; only the batching and the layout differ. Each
#: gradient comes out packed like the input it belongs to, so nothing has to
#: be un-split afterwards. `grad_probs` carries d(probs) into the softmax
#: backward and comes back holding d(scores), which saves a second
#: sixty-seven-megabyte buffer at seq 1024.
fn mha_mma_grads(
    upstream: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    probs: Tensor,
    heads: int,
    seq: int,
    hd: int,
    factor: float
) -> list[Tensor] {
    let tiles = seq // 64
    let dim = heads * hd
    let grad_q = Tensor([seq, dim])
    let grad_k = Tensor([seq, dim])
    let grad_v = Tensor([seq, dim])
    let grad_probs = Tensor([heads * seq, seq])
    cached_attn_kernel("mha_scores_mma").dispatch_async(
        [upstream.buffer(), value.buffer(), grad_probs.buffer()],
        tiles * tiles * heads * 64,
        [seq, heads, hd, tiles],
        group_size: 64
    )
    cached_attn_kernel("mha_transposed_mma").dispatch_async(
        [probs.buffer(), upstream.buffer(), grad_v.buffer()],
        tiles * heads * 64,
        [seq, heads, hd],
        group_size: 64
    )
    cached_attn_kernel("mha_softmax_backward_scaled").dispatch_async(
        [probs.buffer(), grad_probs.buffer()],
        heads * seq * 256,
        [heads * seq, seq, f32_bits(factor)],
        group_size: 256
    )
    cached_attn_kernel("mha_context_mma").dispatch_async(
        [grad_probs.buffer(), key.buffer(), grad_q.buffer()],
        tiles * heads * 64,
        [seq, heads, hd],
        group_size: 64
    )
    cached_attn_kernel("mha_transposed_mma").dispatch_async(
        [grad_probs.buffer(), query.buffer(), grad_k.buffer()],
        tiles * heads * 64,
        [seq, heads, hd],
        group_size: 64
    )
    grad_probs.free()
    return [grad_q, grad_k, grad_v]
}

#: Multi-head attention built from differentiable ops.
#:
#: The fused kernels above never materialise the attention probabilities, so
#: there is nothing for a backward pass to read; a graph that needs gradients
#: gets this composition instead. Every head's scores live in one
#: `[heads * seq, seq]` tensor and every head's context in one
#: `[heads * seq, hd]` tensor, so the only per-head work is the four GEMMs
#: that must be per-head; the softmax, the scale and the packing are one
#: dispatch each across all of them.
#:
#: The composition this replaced sliced Q, K and V per head with `slice_flat`
#: and reassembled the result with `cat`. Forward that is nearly free -- a
#: slice is a view -- but `slice_flat`'s backward allocates a zero tensor the
#: size of the whole packed activation and copies one head into it, and the
#: three slices per head over sixteen heads then arrive at `qs`, `ks` and `vs`
#: as forty-eight full-size gradients to be summed. At seq 1024 that pattern
#: alone measured 1.59 ms: forty-eight four-megabyte zero fills and
#: forty-eight adds, every one of which adds a head to forty-seven zeros. The
#: GEMMs below write into a view of the single gradient buffer instead, so
#: each element is written exactly once. Adding zero is exact, so the answer
#: is unchanged bit for bit.
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
    let probs = Tensor([heads * seq, seq])
    for h in 0..heads {
        let qh = qs.view(h * seq * hd, [seq, hd])
        let kh = ks.view(h * seq * hd, [seq, hd])
        let sh = probs.view(h * seq * seq, [seq, seq])
        raw_matmul(qh, kh, trans_b: true, out: sh)
        qh.free()
        kh.free()
        sh.free()
    }
    cached_attn_kernel("mha_softmax_scaled").dispatch_async(
        [probs.buffer()],
        heads * seq * 256,
        [heads * seq, seq, f32_bits(factor)],
        group_size: 256
    )
    let context = Tensor([heads * seq, hd])
    for h in 0..heads {
        let ph = probs.view(h * seq * seq, [seq, seq])
        let vh = vs.view(h * seq * hd, [seq, hd])
        let oh = context.view(h * seq * hd, [seq, hd])
        raw_matmul(ph, vh, out: oh)
        ph.free()
        vh.free()
        oh.free()
    }
    if qs.requires_grad or ks.requires_grad or vs.requires_grad {
        context.attach_creator(GradNode([qs, ks, vs], fn(upstream: Tensor) -> list[Tensor] {
            return mha_grads(upstream, qs, ks, vs, probs, heads, seq, hd, factor)
        }, extras: [probs]))
    }
    return merge_heads(context, heads, seq)
}

#: Gradients of `mha_by_ops` with respect to the split-head Q, K and V.
#:
#: The four products per head are the ones `matmul`'s own backward would have
#: chosen for the same two products, in the same operand order, so the
#: arithmetic is identical; what differs is that each writes straight into its
#: slice of one gradient buffer. `grad_probs` carries d(probs) into the
#: softmax backward and comes back holding d(scores), which saves a second
#: sixty-seven-megabyte buffer at seq 1024.
fn mha_grads(
    upstream: Tensor,
    qs: Tensor,
    ks: Tensor,
    vs: Tensor,
    probs: Tensor,
    heads: int,
    seq: int,
    hd: int,
    factor: float
) -> list[Tensor] {
    let grad_q = Tensor([heads * seq, hd])
    let grad_k = Tensor([heads * seq, hd])
    let grad_v = Tensor([heads * seq, hd])
    let grad_probs = Tensor([heads * seq, seq])
    for h in 0..heads {
        let ph = probs.view(h * seq * seq, [seq, seq])
        let uh = upstream.view(h * seq * hd, [seq, hd])
        let vh = vs.view(h * seq * hd, [seq, hd])
        let gp = grad_probs.view(h * seq * seq, [seq, seq])
        let gv = grad_v.view(h * seq * hd, [seq, hd])
        raw_matmul(uh, vh, trans_b: true, out: gp)
        raw_matmul(ph, uh, trans_a: true, out: gv)
        ph.free()
        uh.free()
        vh.free()
        gp.free()
        gv.free()
    }
    cached_attn_kernel("mha_softmax_backward_scaled").dispatch_async(
        [probs.buffer(), grad_probs.buffer()],
        heads * seq * 256,
        [heads * seq, seq, f32_bits(factor)],
        group_size: 256
    )
    for h in 0..heads {
        let gs = grad_probs.view(h * seq * seq, [seq, seq])
        let qh = qs.view(h * seq * hd, [seq, hd])
        let kh = ks.view(h * seq * hd, [seq, hd])
        let gq = grad_q.view(h * seq * hd, [seq, hd])
        let gk = grad_k.view(h * seq * hd, [seq, hd])
        raw_matmul(gs, kh, out: gq)
        raw_matmul(gs, qh, trans_a: true, out: gk)
        gs.free()
        qh.free()
        kh.free()
        gq.free()
        gk.free()
    }
    grad_probs.free()
    return [grad_q, grad_k, grad_v]
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
#: Read whole rows of `source`, one per entry of `indices`.
#:
#: The gradient of this scatters back into the table with device atomics, and
#: a token that occurs twice in a batch has two threads adding to the same
#: slot. Float addition is not associative, so which order they arrive in
#: changes the last few bits: two runs from the same seed on the same data
#: give gradients that agree to about ten digits and not beyond. It is enough
#: to make a long training run diverge from its own repeat. Other libraries
#: are the same here for the same reason; making it exact means sorting the
#: indices and reducing each run of them in order, which costs more than the
#: reproducibility is usually worth.
pub fn gather_rows(source: Tensor, indices: Tensor, out_shape: list[int]? = null) -> Tensor {
    if source.rank() != 2 {
        throw ValueError(f"gather_rows needs a rank-two source, got {source.shape}")
    }
    #: The indices are read flat whatever shape they arrive in -- only how
    #: many there are matters, and a batch of index rows is the ordinary way
    #: an embedding is asked for. `out_shape` says what to call the result.
    if indices.rank() < 1 {
        throw ValueError(f"gather_rows needs an index tensor, got {indices.shape}")
    }
    let width = source.shape[1]
    var shape = [indices.size, width]
    if out_shape is not null {
        if element_count(out_shape) != indices.size * width {
            throw ValueError(
                f"gather_rows cannot reshape {indices.size}x{width} rows into {out_shape}"
            )
        }
        shape = out_shape
    }
    let output = Tensor(shape)
    let kernel = cached_kernel("gather_rows")
    kernel.dispatch_async(
        [source.buffer(), indices.buffer(), output.buffer()],
        output.size,
        [indices.size, width, source.shape[0]]
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
        [indices.size, width, vocab]
    )
    return dest
}

#: Blocks per sample for the coalesced group reduction, or zero when it does
#: not apply.
#:
#: A block's threads each stay on one channel for the whole loop only if the
#: group of 256 holds a whole number of channels; nothing else makes the read
#: coalesced, so a channel count that does not divide 256 keeps the old pair
#: of kernels.
#:
#: Roughly 128 blocks in total, because the first pass reaches its ceiling
#: there and the joins do not. Swept on [64,56,56,64] with eight groups: the
#: first pass is 0.30 ms at 64 blocks and then flat at 0.23 ms from 128 all
#: the way to 1024, while the joins cost about 0.07 microseconds a block and
#: so climb from 0.018 ms to 0.088 ms across the same range. `spatial // 64`
#: stops a short feature map being cut into blocks of a row or two.
fn group_blocks(samples: int, spatial: int, channels: int) -> int {
    if channels > 256 or 256 % channels != 0 { return 0 }
    let room = spatial // 64
    if room < 1 { return 0 }
    let wanted = max(1, 128 // samples)
    return min(wanted, room)
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
        slots * 256,
        [n, spatial, c, groups],
        group_size: 256
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
            let grad_gamma = Tensor([c])
            let grad_beta = Tensor([c])
            let per_sample = group_blocks(n, spatial, c)
            if per_sample > 0 {
                let blocks = n * per_sample
                let rows_each = (spatial + per_sample - 1) // per_sample
                let part_dy = Tensor([blocks * c])
                let part_dy_xhat = Tensor([blocks * c])
                cached_kernel("group_norm_grad_partials").dispatch_async([input.buffer(), mean.buffer(), variance.buffer(), upstream.buffer(), part_dy.buffer(), part_dy_xhat.buffer()], blocks * 256, [spatial, c, groups, rows_each, per_sample], group_size: 256)
                cached_kernel("group_norm_slot_join").dispatch_async([part_dy.buffer(), part_dy_xhat.buffer(), gamma.buffer(), sum_dy.buffer(), sum_dy_xhat.buffer()], slots, [c, groups, per_sample, slots])
                cached_kernel("group_norm_param_join").dispatch_async([part_dy.buffer(), part_dy_xhat.buffer(), grad_gamma.buffer(), grad_beta.buffer()], c * 256, [blocks, c], group_size: 256)
                part_dy.free()
                part_dy_xhat.free()
            } else {
                cached_kernel("group_norm_sums").dispatch_async([input.buffer(), mean.buffer(), variance.buffer(), gamma.buffer(), upstream.buffer(), sum_dy.buffer(), sum_dy_xhat.buffer()], slots * 256, [n, spatial, c, groups], group_size: 256)
                cached_kernel("channel_params_backward").dispatch_async([input.buffer(), mean.buffer(), variance.buffer(), upstream.buffer(), grad_gamma.buffer(), grad_beta.buffer()], c * 256, [n * spatial, c, spatial, groups], group_size: 256)
            }
            let grad_input = Tensor(input.shape)
            let back = cached_kernel("group_norm_backward")
            back.dispatch_async([input.buffer(), mean.buffer(), variance.buffer(), gamma.buffer(), upstream.buffer(), sum_dy.buffer(), sum_dy_xhat.buffer(), grad_input.buffer()], input.size, [input.size, spatial, c, groups])
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

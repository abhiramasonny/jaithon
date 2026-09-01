#: The Metal every graph operator dispatches into, and the conversions a
#: dispatch's scalar arguments need.
#:
#: One source string holds all of it, so the entry points share a compile and
#: `kernel_for` hands back the same `Kernel` on every later call.
import std.math
from std.gpu import Kernel

#: The deepest tensor the strided kernels index. Rank eight covers every
#: layout a vision network uses and keeps the per-axis shape and stride inside
#: Metal's thirty-one buffer bindings.
pub const MAX_RANK = 8

let SOURCE = """#include <metal_stdlib>
using namespace metal;

constexpr constant uint MAX_RANK = 8;

kernel void copy_flat(device const float* input [[buffer(0)]],
                      device float* output [[buffer(1)]],
                      constant uint& count [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    if (id < count) output[id] = input[id];
}

kernel void strided_gather(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant uint& count [[buffer(2)]],
                           constant uint& rank [[buffer(3)]],
                           constant uint& base [[buffer(4)]],
                           constant uint& d0 [[buffer(5)]],
                           constant uint& d1 [[buffer(6)]],
                           constant uint& d2 [[buffer(7)]],
                           constant uint& d3 [[buffer(8)]],
                           constant uint& d4 [[buffer(9)]],
                           constant uint& d5 [[buffer(10)]],
                           constant uint& d6 [[buffer(11)]],
                           constant uint& d7 [[buffer(12)]],
                           constant uint& t0 [[buffer(13)]],
                           constant uint& t1 [[buffer(14)]],
                           constant uint& t2 [[buffer(15)]],
                           constant uint& t3 [[buffer(16)]],
                           constant uint& t4 [[buffer(17)]],
                           constant uint& t5 [[buffer(18)]],
                           constant uint& t6 [[buffer(19)]],
                           constant uint& t7 [[buffer(20)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint dims[MAX_RANK] = {d0, d1, d2, d3, d4, d5, d6, d7};
    int steps[MAX_RANK] = {as_type<int>(t0), as_type<int>(t1), as_type<int>(t2),
                           as_type<int>(t3), as_type<int>(t4), as_type<int>(t5),
                           as_type<int>(t6), as_type<int>(t7)};
    int source = as_type<int>(base);
    uint rest = id;
    for (int axis = int(rank) - 1; axis >= 0; --axis) {
        uint coord = rest % dims[axis];
        rest /= dims[axis];
        source += int(coord) * steps[axis];
    }
    output[id] = input[source];
}

kernel void pad_nd(device const float* input [[buffer(0)]],
                   device float* output [[buffer(1)]],
                   constant uint& count [[buffer(2)]],
                   constant uint& rank [[buffer(3)]],
                   constant uint& mode [[buffer(4)]],
                   constant uint& valueBits [[buffer(5)]],
                   constant uint& o0 [[buffer(6)]],
                   constant uint& o1 [[buffer(7)]],
                   constant uint& o2 [[buffer(8)]],
                   constant uint& o3 [[buffer(9)]],
                   constant uint& o4 [[buffer(10)]],
                   constant uint& o5 [[buffer(11)]],
                   constant uint& o6 [[buffer(12)]],
                   constant uint& o7 [[buffer(13)]],
                   constant uint& i0 [[buffer(14)]],
                   constant uint& i1 [[buffer(15)]],
                   constant uint& i2 [[buffer(16)]],
                   constant uint& i3 [[buffer(17)]],
                   constant uint& i4 [[buffer(18)]],
                   constant uint& i5 [[buffer(19)]],
                   constant uint& i6 [[buffer(20)]],
                   constant uint& i7 [[buffer(21)]],
                   constant uint& b0 [[buffer(22)]],
                   constant uint& b1 [[buffer(23)]],
                   constant uint& b2 [[buffer(24)]],
                   constant uint& b3 [[buffer(25)]],
                   constant uint& b4 [[buffer(26)]],
                   constant uint& b5 [[buffer(27)]],
                   constant uint& b6 [[buffer(28)]],
                   constant uint& b7 [[buffer(29)]],
                   uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint outDims[MAX_RANK] = {o0, o1, o2, o3, o4, o5, o6, o7};
    uint inDims[MAX_RANK] = {i0, i1, i2, i3, i4, i5, i6, i7};
    int before[MAX_RANK] = {as_type<int>(b0), as_type<int>(b1), as_type<int>(b2),
                            as_type<int>(b3), as_type<int>(b4), as_type<int>(b5),
                            as_type<int>(b6), as_type<int>(b7)};
    int source = 0;
    int stride = 1;
    uint rest = id;
    for (int axis = int(rank) - 1; axis >= 0; --axis) {
        int limit = int(inDims[axis]);
        int coord = int(rest % outDims[axis]) - before[axis];
        rest /= outDims[axis];
        if (coord < 0 || coord >= limit) {
            if (mode == 0) {
                output[id] = as_type<float>(valueBits);
                return;
            }
            if (mode == 2 || limit == 1) {
                coord = coord < 0 ? 0 : limit - 1;
            } else {
                int period = 2 * (limit - 1);
                coord = coord % period;
                if (coord < 0) coord += period;
                if (coord >= limit) coord = period - coord;
            }
        }
        source += coord * stride;
        stride *= limit;
    }
    output[id] = input[source];
}

kernel void gather_axis(device const float* data [[buffer(0)]],
                        device const float* indices [[buffer(1)]],
                        device float* output [[buffer(2)]],
                        constant uint& count [[buffer(3)]],
                        constant uint& axisDim [[buffer(4)]],
                        constant uint& inner [[buffer(5)]],
                        constant uint& picks [[buffer(6)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint i = id % inner;
    uint k = (id / inner) % picks;
    uint outer = id / (inner * picks);
    int chosen = int(indices[k]);
    if (chosen < 0) chosen += int(axisDim);
    chosen = clamp(chosen, 0, int(axisDim) - 1);
    output[id] = data[(outer * axisDim + uint(chosen)) * inner + i];
}

kernel void concat_into(device const float* part [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        constant uint& count [[buffer(2)]],
                        constant uint& partDim [[buffer(3)]],
                        constant uint& inner [[buffer(4)]],
                        constant uint& outDim [[buffer(5)]],
                        constant uint& offset [[buffer(6)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint i = id % inner;
    uint a = (id / inner) % partDim;
    uint outer = id / (inner * partDim);
    output[(outer * outDim + offset + a) * inner + i] = part[id];
}

kernel void row_reduce(device const float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       constant uint& rows [[buffer(2)]],
                       constant uint& columns [[buffer(3)]],
                       constant uint& op [[buffer(4)]],
                       uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    uint base = row * columns;
    float best = input[base];
    uint bestIndex = 0;
    float total = 0.0f;
    for (uint i = 0; i < columns; ++i) {
        float v = input[base + i];
        total += v;
        if (op == 2 || op == 4 || op == 6) {
            if (v > best || (op == 6 && v == best)) { best = v; bestIndex = i; }
        } else if (op == 3 || op == 5 || op == 7) {
            if (v < best || (op == 7 && v == best)) { best = v; bestIndex = i; }
        }
    }
    if (op == 0) output[row] = total;
    else if (op == 1) output[row] = total / float(columns);
    else if (op == 2 || op == 3) output[row] = best;
    else output[row] = float(bestIndex);
}

kernel void row_softmax_ex(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant uint& rows [[buffer(2)]],
                           constant uint& columns [[buffer(3)]],
                           constant uint& wantLog [[buffer(4)]],
                           uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    uint base = row * columns;
    float peak = input[base];
    for (uint i = 1; i < columns; ++i) peak = max(peak, input[base + i]);
    float total = 0.0f;
    for (uint i = 0; i < columns; ++i) total += exp(input[base + i] - peak);
    float shift = log(total);
    for (uint i = 0; i < columns; ++i) {
        float centred = input[base + i] - peak;
        output[base + i] = wantLog != 0 ? centred - shift : exp(centred) / total;
    }
}

/* Metal has no erf(). This is Abramowitz and Stegun 7.1.26, whose error is
   below 1.5e-7 -- under a float32 ulp for most of the range and far under any
   tolerance a network is compared at. */
static float erf_approx(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float a = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * a);
    float poly = t * (0.254829592f +
                 t * (-0.284496736f +
                 t * (1.421413741f +
                 t * (-1.453152027f + t * 1.061405429f))));
    return sign * (1.0f - poly * exp(-a * a));
}

kernel void unary_ex(device const float* input [[buffer(0)]],
                     device float* output [[buffer(1)]],
                     constant uint& count [[buffer(2)]],
                     constant uint& op [[buffer(3)]],
                     constant uint& p0 [[buffer(4)]],
                     constant uint& p1 [[buffer(5)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float x = input[id];
    float a = as_type<float>(p0);
    float b = as_type<float>(p1);
    float y = x;
    if (op == 0) y = x >= 0.0f ? x : a * (exp(x) - 1.0f);
    else if (op == 1) y = clamp(a * x + b, 0.0f, 1.0f);
    else if (op == 2) y = log(1.0f + exp(-fabs(x))) + max(x, 0.0f);
    else if (op == 3) y = erf_approx(x);
    else if (op == 4) y = x != 0.0f ? 0.0f : 1.0f;
    else if (op == 5) y = x >= 0.0f ? x : a * x;
    else if (op == 6) y = trunc(x);
    else if (op == 7) y = x != 0.0f ? 1.0f : 0.0f;
    output[id] = y;
}

kernel void binary_ex(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device float* output [[buffer(2)]],
                      constant uint& count [[buffer(3)]],
                      constant uint& op [[buffer(4)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float x = a[id];
    float y = b[id];
    float r = 0.0f;
    if (op == 0) r = x == y ? 1.0f : 0.0f;
    else if (op == 1) r = x > y ? 1.0f : 0.0f;
    else if (op == 2) r = x < y ? 1.0f : 0.0f;
    else if (op == 3) r = x != 0.0f && y != 0.0f ? 1.0f : 0.0f;
    else if (op == 4) r = x != 0.0f || y != 0.0f ? 1.0f : 0.0f;
    else if (op == 5) {
        /* pow() is NaN for a negative base, but an integer exponent has a
           perfectly good answer and ONNX graphs do ask for one. */
        if (x < 0.0f && y == trunc(y)) {
            float magnitude = pow(-x, y);
            r = fmod(fabs(y), 2.0f) == 1.0f ? -magnitude : magnitude;
        } else {
            r = pow(x, y);
        }
    } else if (op == 7) {
        /* Sign follows the divisor, which is what ONNX's Mod means with
           `fmod` off and what Python's `%` does. */
        r = y == 0.0f ? NAN : x - floor(x / y) * y;
    } else if (op == 8) {
        /* Sign follows the dividend: C's fmod, which is Mod with `fmod` on. */
        r = y == 0.0f ? NAN : fmod(x, y);
    } else r = x < 0.0f ? x * y : x;
    output[id] = r;
}

/* One LSTM timestep's gates, cell and hidden state.
 *
 * `input_gates` holds `Xt W' + b` for every timestep at once and is read at
 * this step's offset; `hidden_gates` is `H(t-1) R'` for this step alone. Both
 * are summed here rather than beforehand, which saves the slice and the add
 * that would otherwise be two more dispatches on a path that runs once per
 * timestep and is limited by how many it takes.
 *
 * ONNX orders the gates i, o, f, c along the row, and that is the order here.
 */
kernel void lstm_gates(device const float* input_gates [[buffer(0)]],
                       device const float* hidden_gates [[buffer(1)]],
                       device const float* cell_in [[buffer(2)]],
                       device float* cell_out [[buffer(3)]],
                       device float* hidden_out [[buffer(4)]],
                       constant uint& count [[buffer(5)]],
                       constant uint& hidden [[buffer(6)]],
                       constant uint& clip_bits [[buffer(7)]],
                       constant uint& input_base [[buffer(8)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint row = id / hidden;
    uint col = id - row * hidden;
    uint base = row * 4u * hidden;
    uint from = input_base + base;
    float gi = input_gates[from + col] + hidden_gates[base + col];
    float go = input_gates[from + hidden + col] + hidden_gates[base + hidden + col];
    float gf = input_gates[from + 2u * hidden + col] + hidden_gates[base + 2u * hidden + col];
    float gc = input_gates[from + 3u * hidden + col] + hidden_gates[base + 3u * hidden + col];
    float clip = as_type<float>(clip_bits);
    if (clip > 0.0f) {
        gi = clamp(gi, -clip, clip);
        go = clamp(go, -clip, clip);
        gf = clamp(gf, -clip, clip);
        gc = clamp(gc, -clip, clip);
    }
    float i = 1.0f / (1.0f + exp(-gi));
    float o = 1.0f / (1.0f + exp(-go));
    float f = 1.0f / (1.0f + exp(-gf));
    float g = tanh(gc);
    float c = f * cell_in[id] + i * g;
    cell_out[id] = c;
    hidden_out[id] = o * tanh(c);
}

/* One GRU timestep with `linear_before_reset` set: the reset gate scales the
 * recurrent product after it is formed, so nothing else has to be computed
 * between the gates and the answer and this finishes the step on its own.
 *
 * ONNX orders the gates z, r, h. */
kernel void gru_step_linear(device const float* input_gates [[buffer(0)]],
                            device const float* hidden_gates [[buffer(1)]],
                            device const float* recurrent_bias [[buffer(2)]],
                            device const float* hidden_in [[buffer(3)]],
                            device float* hidden_out [[buffer(4)]],
                            constant uint& count [[buffer(5)]],
                            constant uint& hidden [[buffer(6)]],
                            constant uint& clip_bits [[buffer(7)]],
                            constant uint& input_base [[buffer(8)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint row = id / hidden;
    uint col = id - row * hidden;
    uint base = row * 3u * hidden;
    uint from = input_base + base;
    float clip = as_type<float>(clip_bits);
    float gz = input_gates[from + col] + hidden_gates[base + col] + recurrent_bias[col];
    float gr = input_gates[from + hidden + col] + hidden_gates[base + hidden + col] +
               recurrent_bias[hidden + col];
    if (clip > 0.0f) {
        gz = clamp(gz, -clip, clip);
        gr = clamp(gr, -clip, clip);
    }
    float z = 1.0f / (1.0f + exp(-gz));
    float r = 1.0f / (1.0f + exp(-gr));
    float gh = input_gates[from + 2u * hidden + col] +
               r * (hidden_gates[base + 2u * hidden + col] + recurrent_bias[2u * hidden + col]);
    if (clip > 0.0f) gh = clamp(gh, -clip, clip);
    float h = tanh(gh);
    hidden_out[id] = (1.0f - z) * h + z * hidden_in[id];
}

/* The first half of a GRU timestep without `linear_before_reset`: the reset
 * gate scales the hidden state BEFORE it meets the recurrent weights, so the
 * step has to stop here and take a second matrix product. `z` is kept for the
 * half that follows. */
kernel void gru_reset(device const float* input_gates [[buffer(0)]],
                      device const float* hidden_gates [[buffer(1)]],
                      device const float* recurrent_bias [[buffer(2)]],
                      device const float* hidden_in [[buffer(3)]],
                      device float* gate_z [[buffer(4)]],
                      device float* scaled [[buffer(5)]],
                      constant uint& count [[buffer(6)]],
                      constant uint& hidden [[buffer(7)]],
                      constant uint& clip_bits [[buffer(8)]],
                      constant uint& input_base [[buffer(9)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint row = id / hidden;
    uint col = id - row * hidden;
    uint base = row * 3u * hidden;
    uint from = input_base + base;
    float clip = as_type<float>(clip_bits);
    float gz = input_gates[from + col] + hidden_gates[base + col] + recurrent_bias[col];
    float gr = input_gates[from + hidden + col] + hidden_gates[base + hidden + col] +
               recurrent_bias[hidden + col];
    if (clip > 0.0f) {
        gz = clamp(gz, -clip, clip);
        gr = clamp(gr, -clip, clip);
    }
    gate_z[id] = 1.0f / (1.0f + exp(-gz));
    scaled[id] = (1.0f / (1.0f + exp(-gr))) * hidden_in[id];
}

/* The second half: the candidate from the reset-scaled product, and the
 * interpolation the update gate asks for. */
kernel void gru_finish(device const float* input_gates [[buffer(0)]],
                       device const float* reset_product [[buffer(1)]],
                       device const float* gate_z [[buffer(2)]],
                       device const float* hidden_in [[buffer(3)]],
                       device float* hidden_out [[buffer(4)]],
                       constant uint& count [[buffer(5)]],
                       constant uint& hidden [[buffer(6)]],
                       constant uint& clip_bits [[buffer(7)]],
                       constant uint& input_base [[buffer(8)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint row = id / hidden;
    uint col = id - row * hidden;
    float clip = as_type<float>(clip_bits);
    float gh = input_gates[input_base + row * 3u * hidden + 2u * hidden + col] +
               reset_product[id];
    if (clip > 0.0f) gh = clamp(gh, -clip, clip);
    float h = tanh(gh);
    float z = gate_z[id];
    hidden_out[id] = (1.0f - z) * h + z * hidden_in[id];
}

/* Layer normalisation over the trailing axes, one threadgroup a row.
 *
 * A transformer runs one of these before every attention and every feed
 * forward block, so it is on the critical path dozens of times a forward pass.
 * A thread a row would leave a couple of hundred threads doing three serial
 * sweeps each; a threadgroup a row puts the whole machine on it and reduces
 * with simdgroup sums.
 *
 * `mean` and `inv_std` are written unconditionally -- they are one float a row
 * and ONNX lets a graph ask for them. */
kernel void layer_norm_rows(device const float* input [[buffer(0)]],
                            device const float* gamma [[buffer(1)]],
                            device const float* beta [[buffer(2)]],
                            device float* output [[buffer(3)]],
                            device float* mean_out [[buffer(4)]],
                            device float* inv_out [[buffer(5)]],
                            constant uint& width [[buffer(6)]],
                            constant uint& eps_bits [[buffer(7)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint lid [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint warp [[simdgroup_index_in_threadgroup]]) {
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
        shared_inv = rsqrt(s / float(width) + as_type<float>(eps_bits));
        mean_out[row] = m;
        inv_out[row] = shared_inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = shared_inv;

    for (uint i = lid; i < width; i += GROUP) {
        output[base + i] = gamma[i] * (input[base + i] - m) * inv + beta[i];
    }
}

kernel void channel_moments_nchw(device const float* input [[buffer(0)]],
                                 device float* mean [[buffer(1)]],
                                 device float* variance [[buffer(2)]],
                                 constant uint& planes [[buffer(3)]],
                                 constant uint& spatial [[buffer(4)]],
                                 uint plane [[thread_position_in_grid]]) {
    if (plane >= planes) return;
    uint base = plane * spatial;
    float total = 0.0f;
    for (uint i = 0; i < spatial; ++i) total += input[base + i];
    float centre = total / float(spatial);
    float spread = 0.0f;
    for (uint i = 0; i < spatial; ++i) {
        float d = input[base + i] - centre;
        spread += d * d;
    }
    mean[plane] = centre;
    variance[plane] = spread / float(spatial);
}

kernel void affine_norm_nchw(device const float* input [[buffer(0)]],
                             device const float* mean [[buffer(1)]],
                             device const float* variance [[buffer(2)]],
                             device const float* scale [[buffer(3)]],
                             device const float* bias [[buffer(4)]],
                             device float* output [[buffer(5)]],
                             constant uint& count [[buffer(6)]],
                             constant uint& channels [[buffer(7)]],
                             constant uint& spatial [[buffer(8)]],
                             constant uint& epsBits [[buffer(9)]],
                             constant uint& perSample [[buffer(10)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint plane = id / spatial;
    uint c = plane % channels;
    uint stat = perSample != 0 ? plane : c;
    float inv = rsqrt(variance[stat] + as_type<float>(epsBits));
    output[id] = (input[id] - mean[stat]) * inv * scale[c] + bias[c];
}

kernel void lrn_nchw(device const float* input [[buffer(0)]],
                     device float* output [[buffer(1)]],
                     constant uint& count [[buffer(2)]],
                     constant uint& channels [[buffer(3)]],
                     constant uint& spatial [[buffer(4)]],
                     constant uint& size [[buffer(5)]],
                     constant uint& alphaBits [[buffer(6)]],
                     constant uint& betaBits [[buffer(7)]],
                     constant uint& biasBits [[buffer(8)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint i = id % spatial;
    uint plane = id / spatial;
    uint c = plane % channels;
    uint n = plane / channels;
    int back = int(size - 1) / 2;
    int first = max(0, int(c) - back);
    int last = min(int(channels) - 1, int(c) + int(size) - 1 - back);
    float square = 0.0f;
    for (int k = first; k <= last; ++k) {
        float v = input[(n * channels + uint(k)) * spatial + i];
        square += v * v;
    }
    float alpha = as_type<float>(alphaBits);
    float beta = as_type<float>(betaBits);
    float bias = as_type<float>(biasBits);
    output[id] = input[id] / pow(bias + alpha / float(size) * square, beta);
}

static float source_coord(uint out_index, uint out_size, uint in_size, uint coord) {
    float scale = float(out_size) / float(in_size);
    if (coord == 1) return float(out_index) / scale;
    if (coord == 2) {
        if (out_size == 1) return 0.0f;
        return float(out_index) * float(in_size - 1) / float(out_size - 1);
    }
    if (coord == 3 && out_size == 1) return -0.5f;
    return (float(out_index) + 0.5f) / scale - 0.5f;
}

static uint nearest_index(float position, uint limit, uint rule) {
    float picked;
    if (rule == 1) picked = floor(position + 0.5f);
    else if (rule == 2) picked = floor(position);
    else if (rule == 3) picked = ceil(position);
    else picked = ceil(position - 0.5f);
    return uint(clamp(int(picked), 0, int(limit) - 1));
}

kernel void resize_nchw(device const float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        constant uint& count [[buffer(2)]],
                        constant uint& inH [[buffer(3)]],
                        constant uint& inW [[buffer(4)]],
                        constant uint& outH [[buffer(5)]],
                        constant uint& outW [[buffer(6)]],
                        constant uint& mode [[buffer(7)]],
                        constant uint& coord [[buffer(8)]],
                        constant uint& rule [[buffer(9)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint ow = id % outW;
    uint oh = (id / outW) % outH;
    uint plane = id / (outW * outH);
    uint base = plane * inH * inW;
    float y = source_coord(oh, outH, inH, coord);
    float x = source_coord(ow, outW, inW, coord);
    if (mode == 0) {
        uint sy = nearest_index(y, inH, rule);
        uint sx = nearest_index(x, inW, rule);
        output[id] = input[base + sy * inW + sx];
        return;
    }
    float clampedY = clamp(y, 0.0f, float(inH - 1));
    float clampedX = clamp(x, 0.0f, float(inW - 1));
    uint y0 = uint(floor(clampedY));
    uint x0 = uint(floor(clampedX));
    uint y1 = min(y0 + 1, inH - 1);
    uint x1 = min(x0 + 1, inW - 1);
    float fy = clampedY - float(y0);
    float fx = clampedX - float(x0);
    float top = input[base + y0 * inW + x0] * (1.0f - fx) + input[base + y0 * inW + x1] * fx;
    float bottom = input[base + y1 * inW + x0] * (1.0f - fx) + input[base + y1 * inW + x1] * fx;
    output[id] = top * (1.0f - fy) + bottom * fy;
}

kernel void conv_transpose_nchw(device const float* input [[buffer(0)]],
                                device const float* weights [[buffer(1)]],
                                device const float* bias [[buffer(2)]],
                                device float* output [[buffer(3)]],
                                constant uint& count [[buffer(4)]],
                                constant uint& inChannels [[buffer(5)]],
                                constant uint& inH [[buffer(6)]],
                                constant uint& inW [[buffer(7)]],
                                constant uint& outChannels [[buffer(8)]],
                                constant uint& outH [[buffer(9)]],
                                constant uint& outW [[buffer(10)]],
                                constant uint& kh [[buffer(11)]],
                                constant uint& kw [[buffer(12)]],
                                constant uint& strideH [[buffer(13)]],
                                constant uint& strideW [[buffer(14)]],
                                constant uint& padH [[buffer(15)]],
                                constant uint& padW [[buffer(16)]],
                                constant uint& dilationH [[buffer(17)]],
                                constant uint& dilationW [[buffer(18)]],
                                constant uint& hasBias [[buffer(19)]],
                                uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint ow = id % outW;
    uint oh = (id / outW) % outH;
    uint co = (id / (outW * outH)) % outChannels;
    uint n = id / (outW * outH * outChannels);
    float total = hasBias != 0 ? bias[co] : 0.0f;
    for (uint ci = 0; ci < inChannels; ++ci) {
        for (uint r = 0; r < kh; ++r) {
            int reach = int(oh) + int(padH) - int(r * dilationH);
            if (reach < 0 || reach % int(strideH) != 0) continue;
            int ih = reach / int(strideH);
            if (ih >= int(inH)) continue;
            for (uint s = 0; s < kw; ++s) {
                int span = int(ow) + int(padW) - int(s * dilationW);
                if (span < 0 || span % int(strideW) != 0) continue;
                int iw = span / int(strideW);
                if (iw >= int(inW)) continue;
                float x = input[((n * inChannels + ci) * inH + uint(ih)) * inW + uint(iw)];
                float w = weights[((ci * outChannels + co) * kh + r) * kw + s];
                total += x * w;
            }
        }
    }
    output[id] = total;
}
"""

var _kernels: dict[str, Kernel] = {}

pub fn kernel_for(entry: str) -> Kernel {
    let existing = _kernels.get(entry, null)
    if existing is not null { return existing }
    let compiled = Kernel.compile(SOURCE, entry)
    _kernels[entry] = compiled
    return compiled
}

#: The float32 bit pattern, so a float can travel as one of a dispatch's
#: unsigned scalars. `std.gpu` keeps its own copy of this private.
pub fn f32_bits(value: float) -> int {
    if value != value { return 0x7FC00000 }
    var magnitude = value
    var sign = 0
    if magnitude < 0.0 {
        sign = 1
        magnitude = -magnitude
    }
    if magnitude == 0.0 { return sign << 31 }
    if magnitude >= 3.4028234663852886e38 { return sign << 31 | 0x7F800000 }
    if magnitude <= 1.401298464324817e-45 { return sign << 31 }
    let (mantissa, exponent) = math.frexp(magnitude)
    var biased = exponent + 126
    if biased <= 0 { return sign << 31 | int(math.ldexp(magnitude, 149)) & 0x7FFFFF }
    var fraction = int((mantissa * 2.0 - 1.0) * 8388608.0 + 0.5)
    #: Rounding the mantissa can carry it past the top of the field, and the
    #: carry belongs in the exponent -- dropping it instead halves the number.
    if fraction >= 8388608 {
        fraction = 0
        biased += 1
    }
    if biased >= 255 { return sign << 31 | 0x7F800000 }
    return sign << 31 | biased << 23 | fraction & 0x7FFFFF
}

#: A possibly-negative integer as the unsigned scalar a dispatch carries. The
#: kernels read these back with `as_type<int>`.
pub fn u32(value: int) -> int { return value & 0xFFFFFFFF }

#: The ends of the finite float32 range. The low one is what a max pool's
#: padding is filled with, so a padded window never wins; both are what an
#: absent `Clip` bound falls back to.
pub const LOWEST_FLOAT = -3.4028234663852886e38
pub const HIGHEST_FLOAT = 3.4028234663852886e38

#: The graph a network becomes once it has been imported, and the executor
#: that runs it.
#:
#: Every importer -- ONNX, Caffe, TensorFlow, Torch -- lands here, and they all
#: land in the same vocabulary: the `op` string on a `GraphNode` is an ONNX
#: operator name. There is one executor because there is one vocabulary, so a
#: Caffe `Convolution` and a TensorFlow `Conv2D` both arrive as `Conv` and
#: neither importer carries its own arithmetic.
#:
#: Tensors are jaitensor's, so they live on the GPU and hold float32 and
#: nothing else. Shapes, indices and booleans therefore ride in float32 too --
#: exact up to 2^24, which is every index a vision network ever produces. The
#: layout is ONNX's throughout: `NCHW` for images, `MCkHkW` for convolution
#: weights. jaitensor's convolution and pooling are channel-last, so this
#: module transposes around them rather than the other way round; the transpose
#: is a kernel and the convolution is not, which is the right way for that cost
#: to fall.
#:
#: Ownership is simple and worth stating because it is what stops a graph from
#: freeing a tensor it still needs. Every value a node produces is a fresh
#: tensor this module owns. Initializers and the caller's feeds are never
#: freed, and never handed back as an output without being copied first. An
#: intermediate is freed as soon as the last node that reads it has run.
#:
#: Two shapes ONNX has and jaitensor does not, which an importer has to know
#: about because it is the one holding the file:
#:
#: - There is no rank-zero tensor, so a scalar arrives and leaves as `[1]`.
#:   Broadcasting treats the two the same; what differs is that an operator
#:   which would have dropped an axis (a `Gather` by a scalar index) leaves a
#:   width of one behind instead.
#: - There is no empty tensor, so an optional input that a file skips must be
#:   passed as an empty *name* in `GraphNode.inputs` -- position is what
#:   identifies an argument -- and never as a zero-length tensor.
#:
#: ```
#: let graph = load_onnx("mobilenet.onnx")
#: let outputs = run_graph(graph, {"input": blob})
#: defer { for (_name, tensor) in outputs.items() { tensor.free() } }
#: ```
import std.math
import jaitensor as jt
from jaitensor import Tensor
from std.gpu import Kernel
import std.gpu as gpu
from std.os import env
from std.time import Instant

#: The deepest tensor the strided kernels index. Rank eight covers every
#: layout a vision network uses and keeps the per-axis shape and stride inside
#: Metal's thirty-one buffer bindings.
const MAX_RANK = 8

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
 * `gates` already holds `Xt W' + H(t-1) R' + b` for the whole batch, so all
 * that is left is the four activations and the two state updates -- as one
 * kernel rather than the dozen elementwise passes writing it out of jaitensor
 * pieces would take, which matters because this runs once per timestep.
 *
 * ONNX orders the gates i, o, f, c along the row, and that is the order here.
 */
kernel void lstm_gates(device const float* gates [[buffer(0)]],
                       device const float* cell_in [[buffer(1)]],
                       device float* cell_out [[buffer(2)]],
                       device float* hidden_out [[buffer(3)]],
                       constant uint& count [[buffer(4)]],
                       constant uint& hidden [[buffer(5)]],
                       constant uint& clip_bits [[buffer(6)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint row = id / hidden;
    uint col = id - row * hidden;
    uint base = row * 4u * hidden;
    float gi = gates[base + col];
    float go = gates[base + hidden + col];
    float gf = gates[base + 2u * hidden + col];
    float gc = gates[base + 3u * hidden + col];
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
                            device const float* hidden_in [[buffer(2)]],
                            device float* hidden_out [[buffer(3)]],
                            constant uint& count [[buffer(4)]],
                            constant uint& hidden [[buffer(5)]],
                            constant uint& clip_bits [[buffer(6)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint row = id / hidden;
    uint col = id - row * hidden;
    uint base = row * 3u * hidden;
    float clip = as_type<float>(clip_bits);
    float gz = input_gates[base + col] + hidden_gates[base + col];
    float gr = input_gates[base + hidden + col] + hidden_gates[base + hidden + col];
    if (clip > 0.0f) {
        gz = clamp(gz, -clip, clip);
        gr = clamp(gr, -clip, clip);
    }
    float z = 1.0f / (1.0f + exp(-gz));
    float r = 1.0f / (1.0f + exp(-gr));
    float gh = input_gates[base + 2u * hidden + col] +
               r * hidden_gates[base + 2u * hidden + col];
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
                      device const float* hidden_in [[buffer(2)]],
                      device float* gate_z [[buffer(3)]],
                      device float* scaled [[buffer(4)]],
                      constant uint& count [[buffer(5)]],
                      constant uint& hidden [[buffer(6)]],
                      constant uint& clip_bits [[buffer(7)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint row = id / hidden;
    uint col = id - row * hidden;
    uint base = row * 3u * hidden;
    float clip = as_type<float>(clip_bits);
    float gz = input_gates[base + col] + hidden_gates[base + col];
    float gr = input_gates[base + hidden + col] + hidden_gates[base + hidden + col];
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
                       uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint row = id / hidden;
    uint col = id - row * hidden;
    float clip = as_type<float>(clip_bits);
    float gh = input_gates[row * 3u * hidden + 2u * hidden + col] + reset_product[id];
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

fn kernel_for(entry: str) -> Kernel {
    let existing = _kernels.get(entry, null)
    if existing is not null { return existing }
    let compiled = Kernel.compile(SOURCE, entry)
    _kernels[entry] = compiled
    return compiled
}

#: The float32 bit pattern, so a float can travel as one of a dispatch's
#: unsigned scalars. `std.gpu` keeps its own copy of this private.
fn f32_bits(value: float) -> int {
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
fn u32(value: int) -> int { return value & 0xFFFFFFFF }

#: The ends of the finite float32 range. The low one is what a max pool's
#: padding is filled with, so a padded window never wins; both are what an
#: absent `Clip` bound falls back to.
const LOWEST_FLOAT = -3.4028234663852886e38
const HIGHEST_FLOAT = 3.4028234663852886e38

#: One attribute on a node, in whichever of ONNX's seven shapes it arrived.
#:
#: `kind` says which field to read. An importer that guesses wrong writes a
#: field the executor never looks at, so the readers below check `kind` and
#: name the node when it does not match.
pub class Attr {
    #: `"int"`, `"float"`, `"str"`, `"ints"`, `"floats"`, `"strs"` or `"tensor"`.
    pub var kind: str
    pub var i: int
    pub var f: float
    pub var s: str
    pub var ints: list[int]
    pub var floats: list[float]
    pub var strs: list[str]
    pub var tensor: Tensor?

    pub fn init(
        self,
        kind: str,
        i: int = 0,
        f: float = 0.0,
        s: str = "",
        ints: list[int] = [],
        floats: list[float] = [],
        strs: list[str] = [],
        tensor: Tensor? = null
    ) {
        self.kind = kind
        self.i = i
        self.f = f
        self.s = s
        self.ints = [value for value in ints]
        self.floats = [value for value in floats]
        self.strs = [value for value in strs]
        self.tensor = tensor
    }

    pub static fn of_int(value: int) -> Attr { return Attr("int", i: value) }

    pub static fn of_float(value: float) -> Attr { return Attr("float", f: value) }

    pub static fn of_str(value: str) -> Attr { return Attr("str", s: value) }

    pub static fn of_ints(values: list[int]) -> Attr { return Attr("ints", ints: values) }

    pub static fn of_floats(values: list[float]) -> Attr { return Attr("floats", floats: values) }

    pub static fn of_strs(values: list[str]) -> Attr { return Attr("strs", strs: values) }

    pub static fn of_tensor(value: Tensor) -> Attr { return Attr("tensor", tensor: value) }

    fn __str__(self) -> str { return f"Attr({self.kind})" }
}

#: One operator application. `op` is an ONNX operator name whatever the file
#: it came out of, and `inputs` and `outputs` are the names it reads and
#: writes. An empty input name is ONNX's way of skipping an optional input,
#: and is kept rather than dropped because position is what identifies an
#: argument.
pub class GraphNode {
    pub var op: str
    pub var name: str
    pub var inputs: list[str]
    pub var outputs: list[str]
    pub var attrs: dict[str, Attr]
    #: Work an operator can do once and reuse on every later run of the graph.
    #:
    #: A convolution's weights are an initializer, so the layout swap from
    #: ONNX's `[M, C/g, kH, kW]` to jaitensor's `[kH, kW, C/g, M]` produces the
    #: same tensor every frame -- redoing it measured 12.5 ms a frame on
    #: YOLOv8n, a tenth of the whole graph, for a result that never changed.
    #: `cached` owns its tensors and `free_cached` releases them. `cached_of`
    #: holds the input each entry was computed from, so a caller who feeds a
    #: different tensor next time gets a fresh result rather than a stale one;
    #: those are borrowed, never freed here.
    pub var cached: dict[str, Tensor]
    pub var cached_of: dict[str, Tensor]

    pub fn init(
        self,
        op: str,
        name: str = "",
        inputs: list[str] = [],
        outputs: list[str] = [],
        attrs: dict[str, Attr] = {}
    ) {
        self.op = op
        self.name = name
        self.inputs = [value for value in inputs]
        self.outputs = [value for value in outputs]
        self.attrs = {key: value for (key, value) in attrs.items()}
        self.cached = {}
        self.cached_of = {}
    }

    #: Drop anything an operator memoised on this node. Only `cached` is
    #: freed; `cached_of` refers to tensors the graph owns.
    pub fn free_cached(self) -> void {
        for (_key, tensor) in self.cached.items() { tensor.free() }
        self.cached = {}
        self.cached_of = {}
    }

    #: How this node is named in an error. Nodes are often anonymous in an
    #: exported file, so the first output stands in -- it is unique and it is
    #: what the rest of the graph calls this node's result.
    pub fn label(self) -> str {
        if self.name.len() > 0 { return f"operator {self.op} at node {self.name}" }
        if self.outputs.len() > 0 { return f"operator {self.op} at node {self.outputs[0]}" }
        return f"operator {self.op} at an unnamed node"
    }

    fn __str__(self) -> str { return self.label() }
}

#: A whole network: the nodes, the names fed in, the names read out, and the
#: weights that came with the file.
pub class Graph {
    pub var nodes: list[GraphNode]
    pub var inputs: list[str]
    pub var outputs: list[str]
    pub var initializers: dict[str, Tensor]
    var plan: list[GraphNode]?

    pub fn init(
        self,
        nodes: list[GraphNode] = [],
        inputs: list[str] = [],
        outputs: list[str] = [],
        initializers: dict[str, Tensor] = {}
    ) {
        self.nodes = [node for node in nodes]
        self.inputs = [name for name in inputs]
        self.outputs = [name for name in outputs]
        self.initializers = {key: value for (key, value) in initializers.items()}
        self.plan = null
    }

    #: The nodes in an order where every input is produced before it is read.
    #:
    #: Raises `ValueError` naming the node and the name when something is
    #: consumed that nothing produces, and naming every node left over when
    #: what remains is a cycle.
    pub fn topological(self) -> list[GraphNode] {
        var producer: dict[str, int] = {}
        for (index, node) in self.nodes.enumerate() {
            for name in node.outputs {
                if name.len() == 0 { continue }
                let earlier = producer.get(name, null)
                if earlier is not null {
                    throw ValueError(
                        f"{node.label()} and {self.nodes[earlier].label()} both produce {name}"
                    )
                }
                producer[name] = index
            }
        }

        var ready: dict[str, bool] = {}
        for name in self.inputs { ready[name] = true }
        for (name, _weight) in self.initializers.items() { ready[name] = true }

        #: How many inputs each node is still waiting for, and which nodes are
        #: waiting on each name. Indexing the second direction once is what
        #: keeps this linear in the edges rather than quadratic in the nodes.
        var pending: list[int] = []
        var waiting_on: dict[str, list[int]] = {}
        for (index, node) in self.nodes.enumerate() {
            var count = 0
            for name in node.inputs {
                if name.len() == 0 or ready.get(name, false) { continue }
                if producer.get(name, null) is null {
                    throw ValueError(
                        f"{node.label()} reads {name}, which nothing in the graph produces"
                    )
                }
                count += 1
                if waiting_on.get(name, null) is null { waiting_on[name] = [] }
                waiting_on[name].push(index)
            }
            pending.push(count)
        }

        var frontier: list[int] = []
        for (index, count) in pending.enumerate() {
            if count == 0 { frontier.push(index) }
        }
        var order: list[GraphNode] = []
        var emitted = [false for _node in self.nodes]
        var cursor = 0
        while cursor < frontier.len() {
            let index = frontier[cursor]
            cursor += 1
            emitted[index] = true
            let node = self.nodes[index]
            order.push(node)
            for name in node.outputs {
                if name.len() == 0 { continue }
                let blocked = waiting_on.get(name, null)
                if blocked is null { continue }
                for other in blocked {
                    pending[other] -= 1
                    if pending[other] == 0 { frontier.push(other) }
                }
            }
        }

        if order.len() != self.nodes.len() {
            var stuck: list[str] = []
            for (index, node) in self.nodes.enumerate() {
                if not emitted[index] { stuck.push(node.label()) }
            }
            throw ValueError(f"these nodes form a cycle: {'; '.join(stuck)}")
        }
        return order
    }

    #: The order `run_graph` actually walks: topologically sorted, then
    #: rewritten by `fuse_activations`.
    #:
    #: Held onto because neither step depends on the values flowing through --
    #: only on the shape of the graph -- and redoing them per frame cost more
    #: than some of the operators do. A caller who edits `nodes`, `inputs`,
    #: `outputs` or `initializers` afterwards has to call `invalidate`.
    pub fn schedule(self) -> list[GraphNode] {
        let ready = self.plan
        if ready is not null { return ready }
        let built = fuse_activations(self.topological(), self.outputs)
        self.plan = built
        return built
    }

    #: Forget the schedule and everything the operators memoised, so the next
    #: run rebuilds both. Needed after the graph is edited, and harmless
    #: otherwise.
    pub fn invalidate(self) -> void {
        self.plan = null
        for node in self.nodes { node.free_cached() }
    }

    fn __str__(self) -> str {
        return f"Graph({self.nodes.len()} nodes, in={self.inputs}, out={self.outputs})"
    }
}

#: Rewrite `Mul(x, Sigmoid(x))` to a single `Silu` node.
#:
#: Exporters have no choice but to spell SiLU that way -- ONNX has no operator
#: for it -- so a detector that uses it as its activation pays for two full
#: passes over every activation tensor and an intermediate the size of one.
#: On YOLOv8n that pair was 35% of the graph's warm time.
#:
#: The rewrite only fires when the Sigmoid's result goes nowhere else: another
#: reader, or the graph itself naming it as an output, means the value is
#: wanted on its own and folding it away would lose it. The Mul's operands may
#: be either way round, since multiplication does not care and exporters
#: disagree.
fn fuse_activations(order: list[GraphNode], outputs: list[str]) -> list[GraphNode] {
    let readers = count_readers(order, outputs)

    #: Which Sigmoid produced each name, so the Mul that follows can find it
    #: without a second sweep.
    var sigmoid_of: dict[str, GraphNode] = {}
    var folded: dict[str, bool] = {}
    var fused: list[GraphNode] = []
    for node in order {
        if node.op == "Sigmoid" and node.inputs.len() == 1 and node.outputs.len() == 1 {
            let produced = node.outputs[0]
            if produced.len() > 0 and readers.get(produced, 0) == 1 {
                sigmoid_of[produced] = node
            }
            fused.push(node)
            continue
        }
        if node.op == "Mul" and node.inputs.len() == 2 and node.outputs.len() == 1 {
            let left = node.inputs[0]
            let right = node.inputs[1]
            var source = sigmoid_of.get(right, null)
            var other = left
            if source is null {
                source = sigmoid_of.get(left, null)
                other = right
            }
            if source is not null and source.inputs[0] == other {
                folded[source.outputs[0]] = true
                fused.push(
                    GraphNode(
                        op: "Silu",
                        name: node.name,
                        inputs: [other],
                        outputs: [node.outputs[0]]
                    )
                )
                continue
            }
        }
        fused.push(node)
    }
    var kept: list[GraphNode] = []
    for node in fused {
        if node.op == "Sigmoid" and folded.get(node.outputs[0], false) { continue }
        kept.push(node)
    }
    return kept
}

#: How many places each name is consumed, counting the graph's own outputs as
#: a reader -- a value the caller asked for is wanted whatever else reads it.
fn count_readers(order: list[GraphNode], outputs: list[str]) -> dict[str, int] {
    var readers: dict[str, int] = {}
    for node in order {
        for name in node.inputs {
            if name.len() > 0 { readers[name] = readers.get(name, 0) + 1 }
        }
    }
    for name in outputs { readers[name] = readers.get(name, 0) + 1 }
    return readers
}


#: Number of elements a shape holds.
fn volume(shape: list[int]) -> int {
    var total = 1
    for dimension in shape { total *= dimension }
    return total
}

#: Row-major strides for a shape, in elements.
fn contiguous_strides(shape: list[int]) -> list[int] {
    var strides = [0 for _dimension in shape]
    var running = 1
    var axis = shape.len() - 1
    while axis >= 0 {
        strides[axis] = running
        running *= shape[axis]
        axis -= 1
    }
    return strides
}

#: ONNX has rank-zero tensors and jaitensor does not, so a scalar is carried
#: as `[1]`. It broadcasts the same way, which is the only thing that ever
#: looks at the difference.
fn shaped(dims: list[int]) -> list[int] {
    if dims.len() == 0 { return [1] }
    return dims
}

fn check_rank(rank: int, what: str) -> void {
    if rank > MAX_RANK {
        throw ValueError(f"{what} needs rank {rank}, and this executor indexes at most {MAX_RANK}")
    }
}

#: A fresh contiguous copy of `input`, reinterpreted as `shape`.
fn copy_of(input: Tensor, shape: list[int]) -> Tensor {
    let output = Tensor(shape)
    if output.size != input.size {
        throw ValueError(f"cannot view {input.shape} ({input.size} values) as {shape}")
    }
    kernel_for("copy_flat").dispatch_async(
        [input.buffer(), output.buffer()],
        input.size,
        [input.size]
    )
    return output
}

#: Read `input` through `base` and `strides` and write the result out dense.
#:
#: This one kernel is every layout change the executor makes: a transpose is a
#: permuted stride list, a slice is an offset base and a scaled stride, and a
#: broadcast is a stride of zero.
fn gather_strided(input: Tensor, shape: list[int], base: int, strides: list[int]) -> Tensor {
    let rank = shape.len()
    check_rank(rank, "a strided read")
    let output = Tensor(shape)
    var scalars = [output.size, rank, u32(base)]
    for axis in 0..MAX_RANK { scalars.push(if axis < rank { shape[axis] } else { 1 }) }
    for axis in 0..MAX_RANK { scalars.push(u32(if axis < rank { strides[axis] } else { 0 })) }
    kernel_for("strided_gather").dispatch_async(
        [input.buffer(), output.buffer()],
        output.size,
        scalars
    )
    return output
}

#: `input` with its axes reordered so axis `perm[k]` becomes axis `k`.
fn transpose_to(input: Tensor, perm: list[int]) -> Tensor {
    let source = contiguous_strides(input.shape)
    var shape: list[int] = []
    var strides: list[int] = []
    for axis in perm {
        shape.push(input.shape[axis])
        strides.push(source[axis])
    }
    return gather_strided(input, shape, 0, strides)
}

#: A contiguous window of `input`, `length` wide, starting at `start` on `axis`.
fn slice_axis(input: Tensor, axis: int, start: int, length: int) -> Tensor {
    let strides = contiguous_strides(input.shape)
    var shape = [dimension for dimension in input.shape]
    shape[axis] = length
    return gather_strided(input, shape, start * strides[axis], strides)
}

#: The shape numpy would broadcast `a` and `b` to.
fn broadcast_shape(a: list[int], b: list[int], what: str) -> list[int] {
    let rank = max(a.len(), b.len())
    let left_pad = rank - a.len()
    let right_pad = rank - b.len()
    var shape: list[int] = []
    for axis in 0..rank {
        let left = if axis < left_pad { 1 } else { a[axis - left_pad] }
        let right = if axis < right_pad { 1 } else { b[axis - right_pad] }
        if left == right or right == 1 {
            shape.push(left)
        } elif left == 1 {
            shape.push(right)
        } else {
            throw ValueError(f"{what} cannot broadcast {a} against {b}")
        }
    }
    return shape
}

#: `input` stretched to `shape` by numpy's rules, with a stride of zero on
#: every axis being repeated so nothing is actually copied more than once.
fn expand_to(input: Tensor, shape: list[int]) -> Tensor {
    let offset = shape.len() - input.rank()
    if offset < 0 {
        throw ValueError(f"cannot broadcast {input.shape} up to the shorter {shape}")
    }
    let source = contiguous_strides(input.shape)
    var strides = [0 for _dimension in shape]
    for axis in 0..input.rank() {
        let dimension = input.shape[axis]
        let target = shape[offset + axis]
        if dimension == target {
            strides[offset + axis] = source[axis]
        } elif dimension != 1 {
            throw ValueError(f"cannot broadcast {input.shape} up to {shape}")
        }
    }
    return gather_strided(input, shape, 0, strides)
}

#: Apply `combine` to `a` and `b` after stretching both to their common shape.
#:
#: Frees only what it stretched, so an operand that needed no stretching is
#: handed straight through and is still the caller's.
fn elementwise(a: Tensor, b: Tensor, what: str, combine: fn(Tensor, Tensor) -> Tensor) -> Tensor {
    let shape = broadcast_shape(a.shape, b.shape, what)
    let left = if a.shape == shape { a } else { expand_to(a, shape) }
    let right = if b.shape == shape { b } else { expand_to(b, shape) }
    let output = combine(left, right)
    if left is not a { left.free() }
    if right is not b { right.free() }
    return output
}

fn binary_ex_of(a: Tensor, b: Tensor, op: int) -> Tensor {
    let output = Tensor(a.shape)
    kernel_for("binary_ex").dispatch_async(
        [a.buffer(), b.buffer(), output.buffer()],
        a.size,
        [a.size, op]
    )
    return output
}

fn unary_ex_of(input: Tensor, op: int, first: float = 0.0, second: float = 0.0) -> Tensor {
    let output = Tensor(input.shape)
    kernel_for("unary_ex").dispatch_async(
        [input.buffer(), output.buffer()],
        input.size,
        [input.size, op, f32_bits(first), f32_bits(second)]
    )
    return output
}

#: Reduce the last axis of a `[rows, columns]` view. `op` picks sum, mean, max,
#: min, or the index of the first or last extreme.
fn row_reduce_of(
    input: Tensor,
    rows: int,
    columns: int,
    op: int,
    shape: list[int]? = null
) -> Tensor {
    let output = Tensor(shape ?? [rows])
    kernel_for("row_reduce").dispatch_async(
        [input.buffer(), output.buffer()],
        rows,
        [rows, columns, op]
    )
    return output
}

fn softmax_rows(input: Tensor, rows: int, columns: int, want_log: bool) -> Tensor {
    let output = Tensor(input.shape)
    let logging = if want_log { 1 } else { 0 }
    kernel_for("row_softmax_ex").dispatch_async(
        [input.buffer(), output.buffer()],
        rows,
        [rows, columns, logging]
    )
    return output
}

#: Grow (or, with negative counts, crop) every axis. `mode` is 0 constant,
#: 1 reflect, 2 edge.
fn pad_nd_of(
    input: Tensor,
    before: list[int],
    after: list[int],
    mode: int,
    value: float
) -> Tensor {
    let rank = input.rank()
    check_rank(rank, "a pad")
    var shape: list[int] = []
    for axis in 0..rank {
        let width = input.shape[axis] + before[axis] + after[axis]
        if width <= 0 {
            throw ValueError(f"padding {before}/{after} leaves {input.shape} empty on axis {axis}")
        }
        shape.push(width)
    }
    let output = Tensor(shape)
    var scalars = [output.size, rank, mode, f32_bits(value)]
    for axis in 0..MAX_RANK { scalars.push(if axis < rank { shape[axis] } else { 1 }) }
    for axis in 0..MAX_RANK { scalars.push(if axis < rank { input.shape[axis] } else { 1 }) }
    for axis in 0..MAX_RANK { scalars.push(u32(if axis < rank { before[axis] } else { 0 })) }
    kernel_for("pad_nd").dispatch_async([input.buffer(), output.buffer()], output.size, scalars)
    return output
}

#: Join tensors along `axis`. Every part must agree on every other axis.
fn concat_along(parts: list[Tensor], axis: int, what: str) -> Tensor {
    if parts.len() == 0 { throw ValueError(f"{what} needs at least one input") }
    let rank = parts[0].rank()
    var width = 0
    for part in parts {
        if part.rank() != rank {
            throw ValueError(f"{what} joins ranks {parts[0].shape} and {part.shape}")
        }
        for other in 0..rank {
            if other == axis or part.size == 0 { continue }
            if part.shape[other] != parts[0].shape[other] {
                throw ValueError(f"{what} cannot join {parts[0].shape} and {part.shape}")
            }
        }
        width += part.shape[axis]
    }
    var shape = [dimension for dimension in parts[0].shape]
    shape[axis] = width
    let output = Tensor(shape)
    if output.size == 0 { return output }
    var inner = 1
    for other in axis + 1..rank { inner *= shape[other] }
    var offset = 0
    let kernel = kernel_for("concat_into")
    for part in parts {
        #: An empty part contributes nothing and moves the offset nowhere.
        if part.size == 0 { continue }
        kernel.dispatch_async(
            [part.buffer(), output.buffer()],
            part.size,
            [part.size, part.shape[axis], inner, width, offset]
        )
        offset += part.shape[axis]
    }
    return output
}

#: The largest index this executor will read out of a tensor.
#:
#: An exported model writes `INT64_MAX` for "to the end of this axis", and
#: that number does not survive a float32 tensor -- it comes back a shade
#: above what an `int` holds and overflows the conversion. Every use these
#: integers have is a bound, a shape or an index, and saturating is the right
#: answer for all three.
const SATURATED_INDEX = 1 << 53

#: The integers a tensor of shape or index values holds.
fn host_ints(input: Tensor) -> list[int] {
    let ceiling = float(SATURATED_INDEX)
    var values: list[int] = []
    for value in input.download() {
        if value >= ceiling {
            values.push(SATURATED_INDEX)
        } elif value <= -ceiling {
            values.push(-SATURATED_INDEX)
        } else {
            values.push(int(value))
        }
    }
    return values
}

fn normalize_axis(axis: int, rank: int, what: str) -> int {
    let resolved = if axis < 0 { rank + axis } else { axis }
    if resolved < 0 or resolved >= rank {
        throw ValueError(f"{what} names axis {axis}, which is outside rank {rank}")
    }
    return resolved
}

fn attr_int(node: GraphNode, name: str, fallback: int) -> int {
    let found = node.attrs.get(name, null)
    if found is null { return fallback }
    if found.kind != "int" {
        throw ValueError(f"{node.label()} carries {name} as {found.kind}, expected int")
    }
    return found.i
}

fn attr_float(node: GraphNode, name: str, fallback: float) -> float {
    let found = node.attrs.get(name, null)
    if found is null { return fallback }
    if found.kind == "int" { return float(found.i) }
    if found.kind != "float" {
        throw ValueError(f"{node.label()} carries {name} as {found.kind}, expected float")
    }
    return found.f
}

fn attr_str(node: GraphNode, name: str, fallback: str) -> str {
    let found = node.attrs.get(name, null)
    if found is null { return fallback }
    if found.kind != "str" {
        throw ValueError(f"{node.label()} carries {name} as {found.kind}, expected str")
    }
    return found.s
}

fn attr_ints(node: GraphNode, name: str, fallback: list[int]) -> list[int] {
    let found = node.attrs.get(name, null)
    if found is null { return fallback }
    if found.kind != "ints" {
        throw ValueError(f"{node.label()} carries {name} as {found.kind}, expected ints")
    }
    return found.ints
}

fn attr_floats(node: GraphNode, name: str, fallback: list[float]) -> list[float] {
    let found = node.attrs.get(name, null)
    if found is null { return fallback }
    if found.kind != "floats" {
        throw ValueError(f"{node.label()} carries {name} as {found.kind}, expected floats")
    }
    return found.floats
}

fn maybe(ins: list[Tensor?], index: int) -> Tensor? {
    if index < 0 or index >= ins.len() { return null }
    return ins[index]
}

fn need(ins: list[Tensor?], index: int, node: GraphNode) -> Tensor {
    let value = maybe(ins, index)
    if value is null { throw ValueError(f"{node.label()} needs an input at position {index}") }
    return value
}

#: Integers from an input if the graph passes them as one, otherwise from the
#: attribute an older opset put them in.
fn ints_from(node: GraphNode, ins: list[Tensor?], index: int, name: str) -> list[int]? {
    let supplied = maybe(ins, index)
    if supplied is not null { return host_ints(supplied) }
    let found = node.attrs.get(name, null)
    if found is null { return null }
    if found.kind != "ints" {
        throw ValueError(f"{node.label()} carries {name} as {found.kind}, expected ints")
    }
    return found.ints
}

fn op_relu(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [jt.relu(need(ins, 0, node))]
}

fn op_sigmoid(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [jt.sigmoid(need(ins, 0, node))]
}

#: `x * sigmoid(x)`, which YOLOv8 and most modern detectors use everywhere.
#:
#: No exporter emits this: ONNX has no SiLU, so a file spells it as a Sigmoid
#: and a Mul. `fuse_activations` spots that pair and rewrites it to this, which
#: turns two passes over the tensor into one and drops the intermediate.
fn op_silu(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [jt.silu(need(ins, 0, node))]
}

fn op_tanh(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [jt.tanh(need(ins, 0, node))]
}

fn op_sqrt(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [jt.sqrt(need(ins, 0, node))]
}

fn op_exp(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [jt.exp(need(ins, 0, node))]
}

fn op_log(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [jt.log(need(ins, 0, node))]
}

fn op_abs(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [jt.abs(need(ins, 0, node))]
}

fn op_neg(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [jt.neg(need(ins, 0, node))]
}

fn op_elu(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [unary_ex_of(need(ins, 0, node), 0, first: attr_float(node, "alpha", 1.0))]
}

fn op_leaky_relu(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [unary_ex_of(need(ins, 0, node), 5, first: attr_float(node, "alpha", 0.01))]
}

fn op_hard_sigmoid(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let alpha = attr_float(node, "alpha", 0.2)
    let beta = attr_float(node, "beta", 0.5)
    return [unary_ex_of(need(ins, 0, node), 1, first: alpha, second: beta)]
}

fn op_softplus(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [unary_ex_of(need(ins, 0, node), 2)]
}

fn op_erf(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [unary_ex_of(need(ins, 0, node), 3)]
}

fn op_not(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [unary_ex_of(need(ins, 0, node), 4)]
}

fn op_identity(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let input = need(ins, 0, node)
    return [copy_of(input, input.shape)]
}

#: Inference-time dropout is the identity, and the mask it can be asked for is
#: all ones because nothing was dropped.
fn op_dropout(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let input = need(ins, 0, node)
    var produced = [copy_of(input, input.shape)]
    if node.outputs.len() > 1 { produced.push(Tensor.ones(input.shape)) }
    return produced
}

fn op_prelu(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let input = need(ins, 0, node)
    let slope = need(ins, 1, node)
    let stretched = if slope.shape == input.shape { slope } else { expand_to(slope, input.shape) }
    let output = binary_ex_of(input, stretched, 6)
    if stretched is not slope { stretched.free() }
    return [output]
}

fn op_clip(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let input = need(ins, 0, node)
    var low = attr_float(node, "min", LOWEST_FLOAT)
    var high = attr_float(node, "max", HIGHEST_FLOAT)
    let low_input = maybe(ins, 1)
    if low_input is not null {
        if low_input.size != 1 {
            throw ValueError(f"{node.label()} needs a single minimum, got {low_input.shape}")
        }
        low = low_input.download()[0]
    }
    let high_input = maybe(ins, 2)
    if high_input is not null {
        if high_input.size != 1 {
            throw ValueError(f"{node.label()} needs a single maximum, got {high_input.shape}")
        }
        high = high_input.download()[0]
    }
    return [jt.clamp(input, low, high)]
}

fn op_add(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), jt.add)]
}

fn op_sub(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), jt.subtract)]
}

fn op_mul(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), jt.multiply)]
}

fn op_div(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), jt.divide)]
}

fn op_pow(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let combine = |a, b| binary_ex_of(a, b, 5)
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), combine)]
}

#: ONNX `Mod`: the remainder, with the sign following whichever operand the
#: `fmod` attribute says.
#:
#: Transformer exports reach for this in the arithmetic that works out head
#: offsets from a shape, so a graph runner without it stops at the first
#: attention block.
fn op_mod(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let code = if attr_int(node, "fmod", 0) != 0 { 8 } else { 7 }
    let combine = |a, b| binary_ex_of(a, b, code)
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), combine)]
}

fn op_equal(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let combine = |a, b| binary_ex_of(a, b, 0)
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), combine)]
}

fn op_greater(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let combine = |a, b| binary_ex_of(a, b, 1)
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), combine)]
}

fn op_less(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let combine = |a, b| binary_ex_of(a, b, 2)
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), combine)]
}

fn op_and(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let combine = |a, b| binary_ex_of(a, b, 3)
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), combine)]
}

fn op_or(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let combine = |a, b| binary_ex_of(a, b, 4)
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), combine)]
}

#: Fold a variadic operator left to right, broadcasting at every step.
fn fold_inputs(
    node: GraphNode,
    ins: list[Tensor?],
    combine: fn(Tensor, Tensor) -> Tensor
) -> Tensor {
    if ins.len() == 0 { throw ValueError(f"{node.label()} needs at least one input") }
    var total = need(ins, 0, node)
    var mine = false
    for index in 1..ins.len() {
        let merged = elementwise(total, need(ins, index, node), node.label(), combine)
        if mine { total.free() }
        total = merged
        mine = true
    }
    if mine { return total }
    return copy_of(total, total.shape)
}

fn op_min(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [fold_inputs(node, ins, jt.minimum)]
}

fn op_max(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [fold_inputs(node, ins, jt.maximum)]
}

fn op_sum(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [fold_inputs(node, ins, jt.add)]
}

fn op_mean(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let total = fold_inputs(node, ins, jt.add)
    let averaged = jt.scale(total, 1.0 / float(ins.len()))
    total.free()
    return [averaged]
}

fn op_where(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let condition = need(ins, 0, node)
    let yes = need(ins, 1, node)
    let no = need(ins, 2, node)
    let shape = broadcast_shape(
        broadcast_shape(condition.shape, yes.shape, node.label()),
        no.shape,
        node.label()
    )
    let picked = if condition.shape == shape { condition } else { expand_to(condition, shape) }
    let left = if yes.shape == shape { yes } else { expand_to(yes, shape) }
    let right = if no.shape == shape { no } else { expand_to(no, shape) }
    let output = jt.where(picked, left, right)
    if picked is not condition { picked.free() }
    if left is not yes { left.free() }
    if right is not no { right.free() }
    return [output]
}

fn op_reshape(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let data = need(ins, 0, node)
    var dims = host_ints(need(ins, 1, node))
    let allowzero = attr_int(node, "allowzero", 0)
    var infer = -1
    var known = 1
    for (axis, dimension) in dims.enumerate() {
        if dimension == 0 and allowzero == 0 {
            if axis >= data.rank() {
                throw ValueError(f"{node.label()} copies axis {axis} of a rank-{data.rank()} input")
            }
            dims[axis] = data.shape[axis]
            known *= dims[axis]
        } elif dimension == -1 {
            if infer >= 0 { throw ValueError(f"{node.label()} leaves two dimensions to infer") }
            infer = axis
        } elif dimension < 0 {
            throw ValueError(f"{node.label()} asks for the negative dimension {dimension}")
        } else {
            known *= dimension
        }
    }
    if infer >= 0 {
        if known == 0 or data.size % known != 0 {
            throw ValueError(f"{node.label()} cannot fit {data.size} values into {dims}")
        }
        dims[infer] = data.size // known
    }
    return [copy_of(data, shaped(dims))]
}

fn op_flatten(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let data = need(ins, 0, node)
    let rank = data.rank()
    let axis = attr_int(node, "axis", 1)
    let split = if axis < 0 { rank + axis } else { axis }
    if split < 0 or split > rank {
        throw ValueError(f"{node.label()} flattens at axis {axis}, outside rank {rank}")
    }
    var outer = 1
    for other in 0..split { outer *= data.shape[other] }
    return [copy_of(data, [outer, data.size // outer])]
}

fn op_squeeze(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let data = need(ins, 0, node)
    let asked = ints_from(node, ins, 1, "axes")
    var dims: list[int] = []
    if asked is null {
        for dimension in data.shape {
            if dimension != 1 { dims.push(dimension) }
        }
    } else {
        var dropped: list[int] = []
        for axis in asked { dropped.push(normalize_axis(axis, data.rank(), node.label())) }
        for (axis, dimension) in data.shape.enumerate() {
            if axis in dropped {
                if dimension != 1 {
                    throw ValueError(
                        f"{node.label()} squeezes axis {axis}, which is {dimension} wide"
                    )
                }
                continue
            }
            dims.push(dimension)
        }
    }
    return [copy_of(data, shaped(dims))]
}

fn op_unsqueeze(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let data = need(ins, 0, node)
    let asked = ints_from(node, ins, 1, "axes")
    if asked is null { throw ValueError(f"{node.label()} needs the axes to insert") }
    let rank = data.rank() + asked.len()
    var inserted: list[int] = []
    for axis in asked { inserted.push(normalize_axis(axis, rank, node.label())) }
    var dims: list[int] = []
    var next = 0
    for axis in 0..rank {
        if axis in inserted {
            dims.push(1)
        } else {
            dims.push(data.shape[next])
            next += 1
        }
    }
    return [copy_of(data, dims)]
}

fn op_transpose(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let data = need(ins, 0, node)
    let rank = data.rank()
    var perm = attr_ints(node, "perm", [])
    if perm.len() == 0 {
        var reversed: list[int] = []
        var axis = rank - 1
        while axis >= 0 {
            reversed.push(axis)
            axis -= 1
        }
        perm = reversed
    }
    if perm.len() != rank {
        throw ValueError(f"{node.label()} permutes {perm.len()} axes of a rank-{rank} input")
    }
    var seen: list[int] = []
    for axis in perm {
        let resolved = normalize_axis(axis, rank, node.label())
        if resolved in seen { throw ValueError(f"{node.label()} names axis {resolved} twice") }
        seen.push(resolved)
    }
    return [transpose_to(data, seen)]
}

fn op_concat(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    if ins.len() == 0 { throw ValueError(f"{node.label()} needs at least one input") }
    var parts: list[Tensor] = []
    for index in 0..ins.len() { parts.push(need(ins, index, node)) }
    let axis = normalize_axis(attr_int(node, "axis", 0), parts[0].rank(), node.label())
    return [concat_along(parts, axis, node.label())]
}

fn op_split(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let data = need(ins, 0, node)
    let axis = normalize_axis(attr_int(node, "axis", 0), data.rank(), node.label())
    let width = data.shape[axis]
    let wanted = max(node.outputs.len(), 1)
    var widths = ints_from(node, ins, 1, "split") ?? []
    if widths.len() == 0 {
        let parts = attr_int(node, "num_outputs", wanted)
        if parts <= 0 or width % parts != 0 {
            throw ValueError(f"{node.label()} cannot cut {width} into {parts} even pieces")
        }
        widths = [width // parts for _piece in 0..parts]
    }
    var total = 0
    for piece in widths { total += piece }
    if total != width {
        throw ValueError(f"{node.label()} splits into {widths}, which is not axis {axis}'s {width}")
    }
    var produced: list[Tensor] = []
    var start = 0
    for piece in widths {
        produced.push(slice_axis(data, axis, start, piece))
        start += piece
    }
    return produced
}

fn op_slice(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let data = need(ins, 0, node)
    let rank = data.rank()
    let starts = ints_from(node, ins, 1, "starts")
    let ends = ints_from(node, ins, 2, "ends")
    if starts is null or ends is null {
        throw ValueError(f"{node.label()} needs both starts and ends")
    }
    if starts.len() != ends.len() {
        throw ValueError(f"{node.label()} has {starts.len()} starts and {ends.len()} ends")
    }
    let axes = ints_from(node, ins, 3, "axes") ?? [axis for axis in 0..starts.len()]
    let steps = ints_from(node, ins, 4, "steps") ?? [1 for _axis in starts]
    let source = contiguous_strides(data.shape)
    var shape = [dimension for dimension in data.shape]
    var strides = [stride for stride in source]
    var base = 0
    for (slot, axis) in axes.enumerate() {
        let resolved = normalize_axis(axis, rank, node.label())
        let width = data.shape[resolved]
        let step = steps[slot]
        if step == 0 {
            throw ValueError(f"{node.label()} slices axis {resolved} with a step of zero")
        }
        var first = starts[slot]
        var last = ends[slot]
        if first < 0 { first += width }
        if last < 0 { last += width }
        var span = 0
        if step > 0 {
            first = min(max(first, 0), width)
            last = min(max(last, 0), width)
            span = (last - first + step - 1) // step
        } else {
            first = min(max(first, 0), width - 1)
            last = min(max(last, -1), width - 1)
            #: `ceil((last - first) / step)` with both sides negative, written
            #: for a floor division. Getting the rounding backwards here only
            #: shows up once the step is steeper than one.
            span = (last - first + step + 1) // step
        }
        #: A range that comes up empty is a legitimate answer, not a mistake:
        #: exported graphs slice the tail off a shape and get nothing whenever
        #: the tail is not there, and then concatenate the nothing back in.
        shape[resolved] = max(span, 0)
        strides[resolved] = source[resolved] * step
        if shape[resolved] > 0 { base += first * source[resolved] }
    }
    for dimension in shape {
        if dimension == 0 { return [Tensor(shape)] }
    }
    return [gather_strided(data, shape, base, strides)]
}

fn op_gather(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let data = need(ins, 0, node)
    let indices = need(ins, 1, node)
    let axis = normalize_axis(attr_int(node, "axis", 0), data.rank(), node.label())
    var inner = 1
    for other in axis + 1..data.rank() { inner *= data.shape[other] }
    var dims: list[int] = []
    for other in 0..axis { dims.push(data.shape[other]) }
    #: A rank-zero index would drop this axis in ONNX, but jaitensor has no
    #: rank-zero tensor, so such an index arrives as `[1]` and the axis stays
    #: as a width of one. A graph that needs it gone follows with a Squeeze.
    for dimension in indices.shape { dims.push(dimension) }
    for other in axis + 1..data.rank() { dims.push(data.shape[other]) }
    let output = Tensor(dims)
    kernel_for("gather_axis").dispatch_async(
        [data.buffer(), indices.buffer(), output.buffer()],
        output.size,
        [output.size, data.shape[axis], inner, indices.size]
    )
    return [output]
}

fn op_pad(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let data = need(ins, 0, node)
    let rank = data.rank()
    let widths = ints_from(node, ins, 1, "pads")
    if widths is null { throw ValueError(f"{node.label()} needs the pad widths") }
    let name = attr_str(node, "mode", "constant")
    let mode = match name {
        "constant" => 0,
        "reflect" => 1,
        "edge" => 2,
        _ => throw ValueError(f"{node.label()} asks for pad mode {name}, which is not implemented"),
    }
    var value = attr_float(node, "value", 0.0)
    let supplied = maybe(ins, 2)
    if supplied is not null {
        if supplied.size != 1 {
            throw ValueError(f"{node.label()} needs a single pad value, got {supplied.shape}")
        }
        value = supplied.download()[0]
    }
    let axes = ints_from(node, ins, 3, "axes") ?? [axis for axis in 0..rank]
    if widths.len() != 2 * axes.len() {
        throw ValueError(f"{node.label()} has {widths.len()} pad widths for {axes.len()} axes")
    }
    var before = [0 for _axis in 0..rank]
    var after = [0 for _axis in 0..rank]
    for (slot, axis) in axes.enumerate() {
        let resolved = normalize_axis(axis, rank, node.label())
        before[resolved] = widths[slot]
        after[resolved] = widths[axes.len() + slot]
    }
    return [pad_nd_of(data, before, after, mode, value)]
}

fn op_shape(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let data = need(ins, 0, node)
    let rank = data.rank()
    var first = attr_int(node, "start", 0)
    var last = attr_int(node, "end", rank)
    if first < 0 { first += rank }
    if last < 0 { last += rank }
    first = min(max(first, 0), rank)
    last = min(max(last, 0), rank)
    if last <= first {
        throw ValueError(f"{node.label()} asks for axes [{first}, {last}) of a rank-{rank} input")
    }
    return [Tensor([last - first], [float(data.shape[axis]) for axis in first..last])]
}

#: ONNX's numbering for `Cast`'s `to`, which is TensorProto.DataType.
let INTEGER_TYPES = [2, 3, 4, 5, 6, 7, 12, 13]
let FLOAT_TYPES = [1, 10, 11, 16]

fn op_cast(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let data = need(ins, 0, node)
    let target = attr_int(node, "to", 1)
    if target in FLOAT_TYPES { return [copy_of(data, data.shape)] }
    if target in INTEGER_TYPES { return [unary_ex_of(data, 6)] }
    if target == 9 { return [unary_ex_of(data, 7)] }
    throw ValueError(f"{node.label()} casts to ONNX type {target}, which is not implemented")
}

fn op_constant(node: GraphNode, _ins: list[Tensor?]) -> list[Tensor] {
    let held = node.attrs.get("value", null)
    if held is not null {
        if held.kind != "tensor" {
            throw ValueError(f"{node.label()} carries value as {held.kind}, expected tensor")
        }
        let stored = held.tensor
        if stored is null { throw ValueError(f"{node.label()} carries an empty value") }
        return [copy_of(stored, stored.shape)]
    }
    let single = node.attrs.get("value_float", null)
    if single is not null { return [Tensor([1], [single.f])] }
    let whole = node.attrs.get("value_int", null)
    if whole is not null { return [Tensor([1], [float(whole.i)])] }
    let many = node.attrs.get("value_floats", null)
    if many is not null { return [Tensor([many.floats.len()], many.floats)] }
    let counts = node.attrs.get("value_ints", null)
    if counts is not null {
        return [Tensor([counts.ints.len()], [float(value) for value in counts.ints])]
    }
    throw ValueError(f"{node.label()} has no value attribute this executor understands")
}

fn op_constant_of_shape(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let dims = host_ints(need(ins, 0, node))
    var value = 0.0
    let held = node.attrs.get("value", null)
    if held is not null {
        let stored = held.tensor
        if stored is null or stored.size != 1 {
            throw ValueError(f"{node.label()} needs a one-element value attribute")
        }
        value = stored.download()[0]
    }
    return [Tensor.full(shaped(dims), value)]
}

#: The explicit padding a window operator wants, as ONNX's
#: `[begin..., end...]`, working out what `auto_pad` meant when that is how
#: the file said it.
fn resolve_pads(
    node: GraphNode,
    spatial: list[int],
    window: list[int],
    strides: list[int],
    dilations: list[int]
) -> list[int] {
    let axes = spatial.len()
    let auto_pad = attr_str(node, "auto_pad", "NOTSET")
    if auto_pad == "NOTSET" {
        let given = attr_ints(node, "pads", [])
        if given.len() == 0 { return [0 for _slot in 0..2 * axes] }
        if given.len() != 2 * axes {
            throw ValueError(f"{node.label()} has {given.len()} pads for {axes} spatial axes")
        }
        return given
    }
    if auto_pad == "VALID" { return [0 for _slot in 0..2 * axes] }
    if auto_pad != "SAME_UPPER" and auto_pad != "SAME_LOWER" {
        throw ValueError(f"{node.label()} asks for auto_pad {auto_pad}, which is not implemented")
    }
    var pads = [0 for _slot in 0..2 * axes]
    for axis in 0..axes {
        let reach = dilations[axis] * (window[axis] - 1) + 1
        let covered = (spatial[axis] + strides[axis] - 1) // strides[axis]
        let total = max(0, (covered - 1) * strides[axis] + reach - spatial[axis])
        let front = if auto_pad == "SAME_UPPER" { total // 2 } else { total - total // 2 }
        pads[axis] = front
        pads[axes + axis] = total - front
    }
    return pads
}

#: Everything here convolves and pools in two dimensions, so a per-axis
#: attribute that is not a pair belongs to a 1-D or 3-D operator and would
#: otherwise fail somewhere further in with a confusing index.
fn two_of(values: list[int], node: GraphNode, what: str) -> list[int] {
    if values.len() != 2 {
        throw ValueError(
            f"{node.label()} has {what} {values}, and this executor only does " +
                "two-dimensional windows"
        )
    }
    return values
}

fn only_square(values: list[int], node: GraphNode, what: str) -> int {
    for value in values {
        if value != values[0] {
            throw ValueError(
                f"{node.label()} has {what} {values}, and jaitensor's window operators " +
                    "take one value for both axes"
            )
        }
    }
    return values[0]
}

#: NCHW `[N, C, H, W]` grown by ONNX's `[top, left, bottom, right]`.
fn pad_spatial(input: Tensor, pads: list[int], value: float) -> Tensor {
    if pads[0] == 0 and pads[1] == 0 and pads[2] == 0 and pads[3] == 0 { return input }
    return pad_nd_of(input, [0, 0, pads[0], pads[1]], [0, 0, pads[2], pads[3]], 0, value)
}

fn op_conv(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let x = need(ins, 0, node)
    let w = need(ins, 1, node)
    let bias = maybe(ins, 2)
    if x.rank() != 4 {
        throw ValueError(f"{node.label()} needs a 4-D NCHW input, got {x.shape}")
    }
    if w.rank() != 4 {
        throw ValueError(f"{node.label()} needs 4-D weights, got {w.shape}")
    }
    let dilations = two_of(attr_ints(node, "dilations", [1, 1]), node, "dilations")
    for spread in dilations {
        if spread != 1 {
            throw ValueError(
                f"{node.label()} has dilations {dilations}, which jaitensor's convolution " +
                    "does not take"
            )
        }
    }
    let window = two_of(
        attr_ints(node, "kernel_shape", [w.shape[2], w.shape[3]]),
        node,
        "kernel_shape"
    )
    if window != [w.shape[2], w.shape[3]] {
        throw ValueError(f"{node.label()} declares kernel_shape {window} for weights {w.shape}")
    }
    let strides = two_of(attr_ints(node, "strides", [1, 1]), node, "strides")
    let step = only_square(strides, node, "strides")
    let pads = resolve_pads(node, [x.shape[2], x.shape[3]], window, strides, [1, 1])
    let groups = attr_int(node, "group", 1)
    let out_channels = w.shape[0]
    let in_per_group = w.shape[1]
    if groups <= 0 or x.shape[1] % groups != 0 or out_channels % groups != 0 {
        throw ValueError(f"{node.label()} splits {x.shape[1]} channels into {groups} groups")
    }
    if x.shape[1] // groups != in_per_group {
        throw ValueError(
            f"{node.label()} has weights {w.shape} for {x.shape[1]} input channels in " +
                f"{groups} groups"
        )
    }
    if bias is not null and bias.size != out_channels {
        throw ValueError(f"{node.label()} has {bias.size} biases for {out_channels} filters")
    }

    #: One padding value goes to jaitensor when the four agree, because its
    #: convolution already skips the holes; anything else is materialised
    #: first, which costs a copy but is the only way to say "one row on top".
    let even = pads[0] == pads[1] and pads[1] == pads[2] and pads[2] == pads[3]
    let padded = if even { x } else { pad_spatial(x, pads, 0.0) }
    let inner_pad = if even { pads[0] } else { 0 }

    #: Everything below stays in the layout the file already uses. ONNX is
    #: NCHW activations and OIHW weights, which is exactly what `conv2d_nchw`
    #: takes, so nothing is transposed on the way in or out -- the round trip
    #: this used to make was two thirds of what convolution appeared to cost.
    var pieces: list[Tensor] = []
    let out_per_group = out_channels // groups
    for group in 0..groups {
        let slice_in = if groups == 1 {
            padded
        } else {
            slice_axis(padded, 1, group * in_per_group, in_per_group)
        }
        #: A group's filters are a contiguous run of OIHW's first axis, and
        #: they are the same run on every frame, so the slice is kept on the
        #: node. Ungrouped convolutions -- nearly all of them -- slice nothing
        #: and hand the initializer straight through.
        let slice_w = if groups == 1 { w } else { group_weights(node, w, group, out_per_group) }
        var slice_bias: Tensor? = null
        if bias is not null {
            slice_bias = if groups == 1 {
                bias
            } else {
                slice_axis(bias, 0, group * out_per_group, out_per_group)
            }
        }
        pieces.push(
            jt.conv2d_nchw(
                slice_in,
                slice_w,
                bias: slice_bias,
                stride: step,
                padding: inner_pad
            )
        )
        if slice_in is not padded { slice_in.free() }
        if slice_bias is not null and slice_bias is not bias { slice_bias.free() }
    }
    if padded is not x { padded.free() }
    let output = if pieces.len() == 1 { pieces[0] } else { concat_along(pieces, 1, node.label()) }
    if output is not pieces[0] {
        for piece in pieces { piece.free() }
    }
    return [output]
}

#: One group's filters out of an OIHW weight tensor, memoised on the node.
#:
#: The slice depends only on the weights, which are an initializer, so it is
#: the same tensor every run -- but it is recomputed if the caller ever feeds
#: different weights under the same name.
fn group_weights(node: GraphNode, w: Tensor, group: int, out_per_group: int) -> Tensor {
    let key = f"w{group}"
    let kept = node.cached.get(key, null)
    if kept is not null and node.cached_of.get(key, null) is w { return kept }
    if kept is not null { kept.free() }
    let cut = slice_axis(w, 0, group * out_per_group, out_per_group)
    node.cached[key] = cut
    node.cached_of[key] = w
    return cut
}

fn op_conv_transpose(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let x = need(ins, 0, node)
    let w = need(ins, 1, node)
    let bias = maybe(ins, 2)
    if x.rank() != 4 { throw ValueError(f"{node.label()} needs a 4-D NCHW input, got {x.shape}") }
    if w.rank() != 4 { throw ValueError(f"{node.label()} needs 4-D weights, got {w.shape}") }
    let groups = attr_int(node, "group", 1)
    let in_channels = x.shape[1]
    let out_per_group = w.shape[1]
    if groups <= 0 or in_channels % groups != 0 or w.shape[0] != in_channels {
        throw ValueError(f"{node.label()} has weights {w.shape} for input {x.shape}")
    }
    let window = two_of(
        attr_ints(node, "kernel_shape", [w.shape[2], w.shape[3]]),
        node,
        "kernel_shape"
    )
    let strides = two_of(attr_ints(node, "strides", [1, 1]), node, "strides")
    let dilations = two_of(attr_ints(node, "dilations", [1, 1]), node, "dilations")
    let extra = two_of(attr_ints(node, "output_padding", [0, 0]), node, "output_padding")
    let pads = resolve_pads(node, [x.shape[2], x.shape[3]], window, strides, dilations)
    let out_channels = out_per_group * groups
    if bias is not null and bias.size != out_channels {
        throw ValueError(f"{node.label()} has {bias.size} biases for {out_channels} filters")
    }
    var spatial: list[int] = []
    for axis in 0..2 {
        let reach = dilations[axis] * (window[axis] - 1) + 1
        spatial.push(
            (x.shape[2 + axis] - 1) * strides[axis] - pads[axis] - pads[2 + axis] +
                reach +
                extra[axis]
        )
    }
    #: `output_shape` may name the spatial axes alone or the whole output, and
    #: officially it regenerates the padding rather than describing it. What
    #: is safe is to check it, and to say so when it disagrees, instead of
    #: quietly producing a differently shaped tensor.
    let declared = attr_ints(node, "output_shape", [])
    let named = if declared.len() == 4 { [declared[2], declared[3]] } else { declared }
    if named.len() > 0 and named != spatial {
        throw ValueError(
            f"{node.label()} declares output_shape {declared}, but its pads and strides " +
                f"give {spatial}"
        )
    }
    let in_per_group = in_channels // groups
    var pieces: list[Tensor] = []
    for group in 0..groups {
        let part_x = if groups == 1 {
            x
        } else {
            slice_axis(x, 1, group * in_per_group, in_per_group)
        }
        let part_w = if groups == 1 {
            w
        } else {
            slice_axis(w, 0, group * in_per_group, in_per_group)
        }
        var part_bias: Tensor? = null
        if bias is not null {
            part_bias = if groups == 1 {
                bias
            } else {
                slice_axis(bias, 0, group * out_per_group, out_per_group)
            }
        }
        let output = Tensor([x.shape[0], out_per_group, spatial[0], spatial[1]])
        let bias_buffer = if part_bias is null { output.buffer() } else { part_bias.buffer() }
        kernel_for("conv_transpose_nchw").dispatch_async([part_x.buffer(), part_w.buffer(), bias_buffer, output.buffer()], output.size, [output.size, in_per_group, x.shape[2], x.shape[3], out_per_group, spatial[0], spatial[1], window[0], window[1], strides[0], strides[1], pads[0], pads[1], dilations[0], dilations[1], if part_bias is null {
            0
        } else {
            1
        }])
        pieces.push(output)
        if part_x is not x { part_x.free() }
        if part_w is not w { part_w.free() }
        if part_bias is not null and part_bias is not bias { part_bias.free() }
    }
    if pieces.len() == 1 { return [pieces[0]] }
    let joined = concat_along(pieces, 1, node.label())
    for piece in pieces { piece.free() }
    return [joined]
}

fn op_pool(node: GraphNode, ins: list[Tensor?], is_max: bool) -> list[Tensor] {
    let x = need(ins, 0, node)
    if x.rank() != 4 { throw ValueError(f"{node.label()} needs a 4-D NCHW input, got {x.shape}") }
    if node.outputs.len() > 1 {
        throw ValueError(f"{node.label()} is asked for pooling indices, which is not implemented")
    }
    let window = two_of(attr_ints(node, "kernel_shape", []), node, "kernel_shape")
    let dilations = two_of(attr_ints(node, "dilations", [1, 1]), node, "dilations")
    for spread in dilations {
        if spread != 1 {
            throw ValueError(f"{node.label()} has dilations {dilations}, which is not implemented")
        }
    }
    let strides = two_of(attr_ints(node, "strides", [1, 1]), node, "strides")
    let size = only_square(window, node, "kernel_shape")
    let step = only_square(strides, node, "strides")
    var pads = resolve_pads(node, [x.shape[2], x.shape[3]], window, strides, [1, 1])

    #: `ceil_mode` keeps the window that hangs off the end, which is the same
    #: shape as padding the end far enough that the floor formula finds it.
    if attr_int(node, "ceil_mode", 0) != 0 {
        for axis in 0..2 {
            let extent = x.shape[2 + axis] + pads[axis] + pads[2 + axis]
            let covered = (extent - window[axis] + step - 1) // step + 1
            pads[2 + axis] += max(0, (covered - 1) * step + window[axis] - extent)
        }
    }
    let empty = pads[0] == 0 and pads[1] == 0 and pads[2] == 0 and pads[3] == 0
    let filler = if is_max { LOWEST_FLOAT } else { 0.0 }
    let padded = pad_spatial(x, pads, filler)
    let channel_last = jt.nhwc(padded)
    if padded is not x { padded.free() }
    var pooled = if is_max {
        jt.max_pool2d(channel_last, size, stride: step, padding: 0)
    } else {
        jt.avg_pool2d(channel_last, size, stride: step, padding: 0)
    }

    #: An average pool that does not count the padding is the padded sum over
    #: the count of real values, and pooling a mask of ones is exactly that
    #: count -- so the correction is one more pool and a divide.
    if not is_max and not empty and attr_int(node, "count_include_pad", 0) == 0 {
        let mask = Tensor.ones(x.shape)
        let spread = pad_spatial(mask, pads, 0.0)
        mask.free()
        let mask_last = jt.nhwc(spread)
        spread.free()
        let share = jt.avg_pool2d(mask_last, size, stride: step, padding: 0)
        mask_last.free()
        let corrected = jt.divide(pooled, share)
        pooled.free()
        share.free()
        pooled = corrected
    }
    channel_last.free()
    let output = jt.nchw(pooled)
    pooled.free()
    return [output]
}

fn op_max_pool(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_pool(node, ins, true)
}

fn op_average_pool(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_pool(node, ins, false)
}

fn op_global_pool(node: GraphNode, ins: list[Tensor?], op: int) -> list[Tensor] {
    let x = need(ins, 0, node)
    if x.rank() < 3 {
        throw ValueError(f"{node.label()} needs at least a 3-D NCHW input, got {x.shape}")
    }
    let planes = x.shape[0] * x.shape[1]
    var dims = [x.shape[0], x.shape[1]]
    for _axis in 2..x.rank() { dims.push(1) }
    return [row_reduce_of(x, planes, x.size // planes, op, shape: dims)]
}

fn op_global_average_pool(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_global_pool(node, ins, 1)
}

fn op_global_max_pool(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_global_pool(node, ins, 2)
}

fn op_batch_norm(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let x = need(ins, 0, node)
    let scale = need(ins, 1, node)
    let shift = need(ins, 2, node)
    let mean = need(ins, 3, node)
    let variance = need(ins, 4, node)
    if x.rank() < 2 {
        throw ValueError(f"{node.label()} needs at least a 2-D input, got {x.shape}")
    }
    if node.outputs.len() > 1 {
        throw ValueError(
            f"{node.label()} is asked for its running statistics, which only training " +
                "produces and this executor does not"
        )
    }
    let channels = x.shape[1]
    for parameter in [scale, shift, mean, variance] {
        if parameter.size != channels {
            throw ValueError(
                f"{node.label()} has a {parameter.size}-wide parameter for {channels} channels"
            )
        }
    }
    let output = Tensor(x.shape)
    kernel_for("affine_norm_nchw").dispatch_async(
        [
            x.buffer(),
            mean.buffer(),
            variance.buffer(),
            scale.buffer(),
            shift.buffer(),
            output.buffer(),
        ],
        x.size,
        [
            x.size,
            channels,
            x.size // (x.shape[0] * channels),
            f32_bits(attr_float(node, "epsilon", 1e-5)),
            0,
        ]
    )
    return [output]
}

#: ONNX `LayerNormalization`: normalise each row over the axes from `axis` on.
#:
#: Every transformer export -- ViT, CvT, and anything with an attention block
#: -- opens with one of these, so a graph runner without it cannot load them at
#: all. The axes from `axis` onward are flattened into one row, which is what
#: normalising over them means, and the scale and shift run along that row.
fn op_layer_norm(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let x = need(ins, 0, node)
    let scale = need(ins, 1, node)
    let shift = maybe(ins, 2)
    let rank = x.rank()
    let raw_axis = attr_int(node, "axis", -1)
    let axis = if raw_axis < 0 { rank + raw_axis } else { raw_axis }
    if axis < 0 or axis >= rank {
        throw ValueError(f"{node.label()} normalises axis {raw_axis} of a rank-{rank} input")
    }
    var width = 1
    for other in axis..rank { width *= x.shape[other] }
    if width <= 0 { throw ValueError(f"{node.label()} normalises an empty axis of {x.shape}") }
    let rows = x.size // width
    if scale.size != width {
        throw ValueError(f"{node.label()} scales {scale.size} of {width} normalised values")
    }
    if shift is not null and shift.size != width {
        throw ValueError(f"{node.label()} shifts {shift.size} of {width} normalised values")
    }
    #: The shift is optional in the operator and not in the kernel, so an
    #: absent one becomes zeros rather than a second code path.
    let bias = if shift is not null { shift } else { jt.Tensor.zeros([width]) }
    let output = jt.Tensor(x.shape)
    let mean = jt.Tensor([rows])
    let inv_std = jt.Tensor([rows])
    kernel_for("layer_norm_rows").dispatch_async(
        [
            x.buffer(),
            scale.buffer(),
            bias.buffer(),
            output.buffer(),
            mean.buffer(),
            inv_std.buffer(),
        ],
        rows * 256,
        [width, f32_bits(attr_float(node, "epsilon", 1.0e-5))],
        group_size: 256
    )
    if shift is null { bias.free() }
    #: `Mean` and `InvStdDev` are optional outputs, shaped like the input with
    #: the normalised axes collapsed to one. The ones a graph does not name are
    #: dropped by the caller.
    var moment_shape: list[int] = []
    for other in 0..rank { moment_shape.push(if other < axis { x.shape[other] } else { 1 }) }
    return [output, jt.reshape(mean, moment_shape), jt.reshape(inv_std, moment_shape)]
}

#: One direction of an LSTM, from `[seq, batch, input]` to `[seq, batch, H]`.
#:
#: `x_gates` already holds every timestep's `Xt W' + b`, because that half does
#: not depend on the recurrence and so is one matrix product for the whole
#: sequence rather than one per step. What is left per step is the recurrent
#: product and the gates.
fn lstm_direction(
    x_gates: Tensor,
    recurrent: Tensor,
    initial_h: Tensor,
    initial_c: Tensor,
    steps: int,
    batch: int,
    hidden: int,
    clip: float,
    reverse: bool
) -> tuple[Tensor, Tensor, Tensor] {
    #: `steps` owns every hidden state it produces, because they are also the
    #: output sequence. The cell state is not kept, so each new one replaces
    #: and frees the one before -- except the caller's, which the caller frees.
    var state_h = initial_h
    var state_c = initial_c
    var own_cell = false
    var produced: list[Tensor] = []
    for index in 0..steps {
        let step = if reverse { steps - 1 - index } else { index }
        let at = slice_axis(x_gates, 0, step * batch, batch)
        let recurred = jt.matmul(state_h, recurrent, trans_b: true)
        let gates = jt.add(at, recurred)
        at.free()
        recurred.free()

        let next_c = jt.Tensor([batch, hidden])
        let next_h = jt.Tensor([batch, hidden])
        kernel_for("lstm_gates").dispatch_async(
            [gates.buffer(), state_c.buffer(), next_c.buffer(), next_h.buffer()],
            batch * hidden,
            [batch * hidden, hidden, f32_bits(clip)]
        )
        gates.free()
        if own_cell { state_c.free() }
        state_c = next_c
        own_cell = true
        state_h = next_h
        produced.push(next_h)
    }

    #: A reversed pass produces its steps back to front, and the output is
    #: indexed by time whichever way it was walked.
    var ordered = produced
    if reverse {
        ordered = []
        for index in 0..produced.len() { ordered.push(produced[produced.len() - 1 - index]) }
    }
    var rows: list[Tensor] = []
    for part in ordered { rows.push(copy_of(part, [1, batch, hidden])) }
    let sequence = if rows.len() == 1 {
        rows[0]
    } else {
        concat_along(rows, 0, "an LSTM's output sequence")
    }
    if rows.len() > 1 {
        for row in rows { row.free() }
    }
    let final_h = copy_of(state_h, [batch, hidden])
    let final_c = copy_of(state_c, [batch, hidden])
    for part in produced { part.free() }
    if own_cell { state_c.free() }
    return (sequence, final_h, final_c)
}

#: ONNX `LSTM`.
#:
#: Weights arrive as `[directions, 4H, input]` and `[directions, 4H, H]` with
#: the gates ordered i, o, f, c, and the bias as `[directions, 8H]` holding the
#: input biases and then the recurrent ones. Outputs are the whole sequence
#: `[seq, directions, batch, H]` and the final hidden and cell states.
fn op_lstm(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let x = need(ins, 0, node)
    let w = need(ins, 1, node)
    let r = need(ins, 2, node)
    let bias = maybe(ins, 3)
    if maybe(ins, 4) is not null {
        throw ValueError(
            f"{node.label()} gives per-sequence lengths, which this executor does not read; " +
                "every sequence in the batch has to be the same length"
        )
    }
    if maybe(ins, 7) is not null {
        throw ValueError(f"{node.label()} has peephole weights, which this executor does not apply")
    }
    if node.attrs.get("activations", null) is not null {
        throw ValueError(
            f"{node.label()} names its own activations; this executor uses the sigmoid and " +
                "tanh the operator defaults to"
        )
    }
    if attr_int(node, "layout", 0) != 0 {
        throw ValueError(f"{node.label()} uses the batch-first layout, which this executor does not read")
    }
    if attr_int(node, "input_forget", 0) != 0 {
        throw ValueError(f"{node.label()} couples its input and forget gates, which this executor does not")
    }
    if x.rank() != 3 { throw ValueError(f"{node.label()} needs a 3-D input, got {x.shape}") }
    if w.rank() != 3 or r.rank() != 3 {
        throw ValueError(f"{node.label()} needs 3-D weights, got {w.shape} and {r.shape}")
    }
    let steps = x.shape[0]
    let batch = x.shape[1]
    let input_size = x.shape[2]
    let directions = w.shape[0]
    let hidden = attr_int(node, "hidden_size", r.shape[2])
    let facing = attr_str(node, "direction", "forward")
    let wanted = if facing == "bidirectional" { 2 } else { 1 }
    if directions != wanted or r.shape[0] != directions {
        throw ValueError(f"{node.label()} is {facing} but carries {directions} directions of weights")
    }
    if w.shape[1] != 4 * hidden or r.shape[1] != 4 * hidden or r.shape[2] != hidden {
        throw ValueError(f"{node.label()} has weights {w.shape} and {r.shape} for {hidden} hidden units")
    }
    if w.shape[2] != input_size {
        throw ValueError(f"{node.label()} has weights {w.shape} for {input_size} inputs")
    }
    let clip = attr_float(node, "clip", 0.0)

    let flat_x = copy_of(x, [steps * batch, input_size])
    var sequences: list[Tensor] = []
    var last_h: list[Tensor] = []
    var last_c: list[Tensor] = []
    for direction in 0..directions {
        let wd = taken(slice_axis(w, 0, direction, 1), [4 * hidden, input_size])
        let rd = taken(slice_axis(r, 0, direction, 1), [4 * hidden, hidden])
        #: Every timestep's input half at once: it does not depend on the
        #: recurrence, so it is one product for the sequence rather than one
        #: for each step of it.
        var x_gates = jt.matmul(flat_x, wd, trans_b: true)
        wd.free()
        if bias is not null {
            #: `[directions, 8H]` is the input biases followed by the recurrent
            #: ones, and both land on the same gates.
            let row = taken(slice_axis(bias, 0, direction, 1), [8 * hidden])
            let wb = taken(slice_axis(row, 0, 0, 4 * hidden), [4 * hidden])
            let rb = taken(slice_axis(row, 0, 4 * hidden, 4 * hidden), [4 * hidden])
            let both = jt.add(wb, rb)
            let shifted = jt.add(x_gates, both)
            x_gates.free()
            x_gates = shifted
            row.free()
            wb.free()
            rb.free()
            both.free()
        }
        let given_h = maybe(ins, 5)
        let given_c = maybe(ins, 6)
        let start_h = if given_h is null {
            jt.Tensor.zeros([batch, hidden])
        } else {
            taken(slice_axis(given_h, 0, direction, 1), [batch, hidden])
        }
        let start_c = if given_c is null {
            jt.Tensor.zeros([batch, hidden])
        } else {
            taken(slice_axis(given_c, 0, direction, 1), [batch, hidden])
        }
        let (sequence, final_h, final_c) = lstm_direction(
            x_gates,
            rd,
            start_h,
            start_c,
            steps,
            batch,
            hidden,
            clip,
            facing == "reverse" or direction == 1
        )
        x_gates.free()
        rd.free()
        start_h.free()
        start_c.free()
        sequences.push(copy_of(sequence, [steps, 1, batch, hidden]))
        sequence.free()
        last_h.push(copy_of(final_h, [1, batch, hidden]))
        last_c.push(copy_of(final_c, [1, batch, hidden]))
        final_h.free()
        final_c.free()
    }
    flat_x.free()

    let all_y = if sequences.len() == 1 { sequences[0] } else { concat_along(sequences, 1, node.label()) }
    let all_h = if last_h.len() == 1 { last_h[0] } else { concat_along(last_h, 0, node.label()) }
    let all_c = if last_c.len() == 1 { last_c[0] } else { concat_along(last_c, 0, node.label()) }
    if sequences.len() > 1 {
        for part in sequences { part.free() }
        for part in last_h { part.free() }
        for part in last_c { part.free() }
    }
    return [all_y, all_h, all_c]
}

#: One direction of a GRU, from `[seq, batch, input]` to `[seq, batch, H]`.
fn gru_direction(
    x_gates: Tensor,
    recurrent: Tensor,
    recurrent_h: Tensor,
    recurrent_bias: Tensor?,
    initial_h: Tensor,
    steps: int,
    batch: int,
    hidden: int,
    clip: float,
    reverse: bool,
    linear_before_reset: bool
) -> tuple[Tensor, Tensor] {
    var state_h = initial_h
    var produced: list[Tensor] = []
    for index in 0..steps {
        let step = if reverse { steps - 1 - index } else { index }
        let at = slice_axis(x_gates, 0, step * batch, batch)
        let product = jt.matmul(state_h, recurrent, trans_b: true)
        #: The recurrent bias belongs to this product rather than to the input
        #: half, because with `linear_before_reset` the reset gate scales it.
        let recurred = if recurrent_bias is null {
            product
        } else {
            let shifted = jt.add(product, recurrent_bias)
            product.free()
            shifted
        }
        let next_h = jt.Tensor([batch, hidden])
        if linear_before_reset {
            kernel_for("gru_step_linear").dispatch_async(
                [at.buffer(), recurred.buffer(), state_h.buffer(), next_h.buffer()],
                batch * hidden,
                [batch * hidden, hidden, f32_bits(clip)]
            )
        } else {
            #: The reset gate scales the hidden state before the weights see
            #: it, so this half stops to take a second product.
            let gate_z = jt.Tensor([batch, hidden])
            let scaled = jt.Tensor([batch, hidden])
            kernel_for("gru_reset").dispatch_async(
                [
                    at.buffer(),
                    recurred.buffer(),
                    state_h.buffer(),
                    gate_z.buffer(),
                    scaled.buffer(),
                ],
                batch * hidden,
                [batch * hidden, hidden, f32_bits(clip)]
            )
            let reset_product = jt.matmul(scaled, recurrent_h, trans_b: true)
            kernel_for("gru_finish").dispatch_async(
                [
                    at.buffer(),
                    reset_product.buffer(),
                    gate_z.buffer(),
                    state_h.buffer(),
                    next_h.buffer(),
                ],
                batch * hidden,
                [batch * hidden, hidden, f32_bits(clip)]
            )
            gate_z.free()
            scaled.free()
            reset_product.free()
        }
        at.free()
        recurred.free()
        state_h = next_h
        produced.push(next_h)
    }

    var ordered = produced
    if reverse {
        ordered = []
        for index in 0..produced.len() { ordered.push(produced[produced.len() - 1 - index]) }
    }
    var rows: list[Tensor] = []
    for part in ordered { rows.push(copy_of(part, [1, batch, hidden])) }
    let sequence = if rows.len() == 1 {
        rows[0]
    } else {
        concat_along(rows, 0, "a GRU's output sequence")
    }
    if rows.len() > 1 {
        for row in rows { row.free() }
    }
    let final_h = copy_of(state_h, [batch, hidden])
    for part in produced { part.free() }
    return (sequence, final_h)
}

#: ONNX `GRU`.
#:
#: Weights are `[directions, 3H, input]` and `[directions, 3H, H]` with the
#: gates ordered z, r, h, and the bias `[directions, 6H]` holds the input set
#: then the recurrent one. `linear_before_reset` decides whether the reset gate
#: scales the hidden state before or after the recurrent weights, which are
#: different answers -- and it decides where the recurrent bias for the
#: candidate gate belongs, since only one of the two puts it inside the reset.
fn op_gru(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let x = need(ins, 0, node)
    let w = need(ins, 1, node)
    let r = need(ins, 2, node)
    let bias = maybe(ins, 3)
    if maybe(ins, 4) is not null {
        throw ValueError(
            f"{node.label()} gives per-sequence lengths, which this executor does not read; " +
                "every sequence in the batch has to be the same length"
        )
    }
    if node.attrs.get("activations", null) is not null {
        throw ValueError(
            f"{node.label()} names its own activations; this executor uses the sigmoid and " +
                "tanh the operator defaults to"
        )
    }
    if attr_int(node, "layout", 0) != 0 {
        throw ValueError(f"{node.label()} uses the batch-first layout, which this executor does not read")
    }
    if x.rank() != 3 { throw ValueError(f"{node.label()} needs a 3-D input, got {x.shape}") }
    if w.rank() != 3 or r.rank() != 3 {
        throw ValueError(f"{node.label()} needs 3-D weights, got {w.shape} and {r.shape}")
    }
    let steps = x.shape[0]
    let batch = x.shape[1]
    let input_size = x.shape[2]
    let directions = w.shape[0]
    let hidden = attr_int(node, "hidden_size", r.shape[2])
    let facing = attr_str(node, "direction", "forward")
    let wanted = if facing == "bidirectional" { 2 } else { 1 }
    if directions != wanted or r.shape[0] != directions {
        throw ValueError(f"{node.label()} is {facing} but carries {directions} directions of weights")
    }
    if w.shape[1] != 3 * hidden or r.shape[1] != 3 * hidden or r.shape[2] != hidden {
        throw ValueError(f"{node.label()} has weights {w.shape} and {r.shape} for {hidden} hidden units")
    }
    if w.shape[2] != input_size {
        throw ValueError(f"{node.label()} has weights {w.shape} for {input_size} inputs")
    }
    let clip = attr_float(node, "clip", 0.0)
    let linear = attr_int(node, "linear_before_reset", 0) != 0

    let flat_x = copy_of(x, [steps * batch, input_size])
    var sequences: list[Tensor] = []
    var last_h: list[Tensor] = []
    for direction in 0..directions {
        let wd = taken(slice_axis(w, 0, direction, 1), [3 * hidden, input_size])
        let rd = taken(slice_axis(r, 0, direction, 1), [3 * hidden, hidden])
        let rh = taken(slice_axis(rd, 0, 2 * hidden, hidden), [hidden, hidden])
        var x_gates = jt.matmul(flat_x, wd, trans_b: true)
        wd.free()
        var recurrent_bias: Tensor? = null
        if bias is not null {
            let row = taken(slice_axis(bias, 0, direction, 1), [6 * hidden])
            let wb = taken(slice_axis(row, 0, 0, 3 * hidden), [3 * hidden])
            let rb = taken(slice_axis(row, 0, 3 * hidden, 3 * hidden), [3 * hidden])
            #: Without `linear_before_reset` the candidate's recurrent bias is
            #: outside the reset, so it is a constant and belongs with the
            #: input half; the recurrent half then carries only z and r.
            var input_bias = wb
            var carried = rb
            if not linear {
                let zeros = jt.Tensor.zeros([hidden])
                let head = taken(slice_axis(rb, 0, 0, 2 * hidden), [2 * hidden])
                let tail = taken(slice_axis(rb, 0, 2 * hidden, hidden), [hidden])
                carried = concat_along([head, zeros], 0, node.label())
                input_bias = jt.add(wb, concat_along([jt.Tensor.zeros([2 * hidden]), tail], 0, node.label()))
                head.free()
                tail.free()
                zeros.free()
            }
            let shifted = jt.add(x_gates, input_bias)
            x_gates.free()
            x_gates = shifted
            recurrent_bias = carried
            if input_bias is not wb { input_bias.free() }
            wb.free()
            if carried is not rb { rb.free() }
            row.free()
        }
        let given_h = maybe(ins, 5)
        let start_h = if given_h is null {
            jt.Tensor.zeros([batch, hidden])
        } else {
            taken(slice_axis(given_h, 0, direction, 1), [batch, hidden])
        }
        let (sequence, final_h) = gru_direction(
            x_gates,
            rd,
            rh,
            recurrent_bias,
            start_h,
            steps,
            batch,
            hidden,
            clip,
            facing == "reverse" or direction == 1,
            linear
        )
        x_gates.free()
        rd.free()
        rh.free()
        start_h.free()
        let held = recurrent_bias
        if held is not null { held.free() }
        sequences.push(copy_of(sequence, [steps, 1, batch, hidden]))
        sequence.free()
        last_h.push(copy_of(final_h, [1, batch, hidden]))
        final_h.free()
    }
    flat_x.free()

    let all_y = if sequences.len() == 1 { sequences[0] } else { concat_along(sequences, 1, node.label()) }
    let all_h = if last_h.len() == 1 { last_h[0] } else { concat_along(last_h, 0, node.label()) }
    if sequences.len() > 1 {
        for part in sequences { part.free() }
        for part in last_h { part.free() }
    }
    return [all_y, all_h]
}

#: `copy_of` on a freshly cut slice, freeing the slice. Slicing allocates, and
#: every use of it here wants the copy and not the cut.
fn taken(cut: Tensor, shape: list[int]) -> Tensor {
    let kept = copy_of(cut, shape)
    cut.free()
    return kept
}

fn op_instance_norm(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let x = need(ins, 0, node)
    let scale = need(ins, 1, node)
    let shift = need(ins, 2, node)
    if x.rank() < 3 {
        throw ValueError(f"{node.label()} needs at least a 3-D NCHW input, got {x.shape}")
    }
    let channels = x.shape[1]
    if scale.size != channels or shift.size != channels {
        throw ValueError(f"{node.label()} has parameters for {scale.size} of {channels} channels")
    }
    let planes = x.shape[0] * channels
    let spatial = x.size // planes
    let mean = Tensor([planes])
    let variance = Tensor([planes])
    kernel_for("channel_moments_nchw").dispatch_async(
        [x.buffer(), mean.buffer(), variance.buffer()],
        planes,
        [planes, spatial]
    )
    let output = Tensor(x.shape)
    kernel_for("affine_norm_nchw").dispatch_async(
        [
            x.buffer(),
            mean.buffer(),
            variance.buffer(),
            scale.buffer(),
            shift.buffer(),
            output.buffer(),
        ],
        x.size,
        [x.size, channels, spatial, f32_bits(attr_float(node, "epsilon", 1e-5)), 1]
    )
    mean.free()
    variance.free()
    return [output]
}

fn op_lrn(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let x = need(ins, 0, node)
    if x.rank() < 3 {
        throw ValueError(f"{node.label()} needs at least a 3-D NCHW input, got {x.shape}")
    }
    let size = attr_int(node, "size", 0)
    if size <= 0 { throw ValueError(f"{node.label()} needs a positive size, got {size}") }
    let channels = x.shape[1]
    let output = Tensor(x.shape)
    kernel_for("lrn_nchw").dispatch_async(
        [x.buffer(), output.buffer()],
        x.size,
        [
            x.size,
            channels,
            x.size // (x.shape[0] * channels),
            size,
            f32_bits(attr_float(node, "alpha", 0.0001)),
            f32_bits(attr_float(node, "beta", 0.75)),
            f32_bits(attr_float(node, "bias", 1.0)),
        ]
    )
    return [output]
}

fn op_gemm(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let a = need(ins, 0, node)
    let b = need(ins, 1, node)
    if a.rank() != 2 or b.rank() != 2 {
        throw ValueError(f"{node.label()} needs rank-two operands, got {a.shape} and {b.shape}")
    }
    let alpha = attr_float(node, "alpha", 1.0)
    let beta = attr_float(node, "beta", 1.0)
    let product = jt.matmul(
        a,
        b,
        trans_a: attr_int(node, "transA", 0) != 0,
        trans_b: attr_int(node, "transB", 0) != 0
    )
    var output = product
    if alpha != 1.0 {
        output = jt.scale(product, alpha)
        product.free()
    }
    let c = maybe(ins, 2)
    if c is null or beta == 0.0 { return [output] }
    let stretched = if c.shape == output.shape { c } else { expand_to(c, output.shape) }
    let weighted = if beta == 1.0 { stretched } else { jt.scale(stretched, beta) }
    let summed = jt.add(output, weighted)
    if weighted is not stretched { weighted.free() }
    if stretched is not c { stretched.free() }
    output.free()
    return [summed]
}

#: One `[rows, inner] x [inner, columns]` product per page of the broadcast
#: batch, written straight into a window of the output.
fn batched_matmul(
    left: Tensor,
    right: Tensor,
    batch: list[int],
    rows: int,
    inner: int,
    columns: int
) -> Tensor {
    var left_shape = [dimension for dimension in batch]
    left_shape.push(rows)
    left_shape.push(inner)
    var right_shape = [dimension for dimension in batch]
    right_shape.push(inner)
    right_shape.push(columns)
    var dims = [dimension for dimension in batch]
    dims.push(rows)
    dims.push(columns)
    let spread_left = if left.shape == left_shape { left } else { expand_to(left, left_shape) }
    let spread_right = if right.shape == right_shape {
        right
    } else {
        expand_to(right, right_shape)
    }
    let output = Tensor(dims)
    for index in 0..volume(batch) {
        let page_a = Tensor(
            [rows, inner],
            storage: spread_left.buffer().view(index * rows * inner, rows * inner)
        )
        let page_b = Tensor(
            [inner, columns],
            storage: spread_right.buffer().view(index * inner * columns, inner * columns)
        )
        let page_out = Tensor(
            [rows, columns],
            storage: output.buffer().view(index * rows * columns, rows * columns)
        )
        jt.matmul(page_a, page_b, out: page_out)
        page_a.free()
        page_b.free()
        page_out.free()
    }
    if spread_left is not left { spread_left.free() }
    if spread_right is not right { spread_right.free() }
    return output
}

fn op_matmul(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let a = need(ins, 0, node)
    let b = need(ins, 1, node)
    #: ONNX promotes a vector operand to a matrix for the product and drops
    #: the axis again afterwards, so both shapes are normalised here and the
    #: result is trimmed at the end.
    let left = if a.rank() == 1 { copy_of(a, [1, a.size]) } else { a }
    let right = if b.rank() == 1 { copy_of(b, [b.size, 1]) } else { b }
    let rows = left.shape[left.rank() - 2]
    let inner = left.shape[left.rank() - 1]
    let columns = right.shape[right.rank() - 1]
    if right.shape[right.rank() - 2] != inner {
        throw ValueError(f"{node.label()} cannot multiply {a.shape} by {b.shape}")
    }
    var batch: list[int] = []
    if left.rank() > 2 or right.rank() > 2 {
        batch = broadcast_shape(
            [left.shape[axis] for axis in 0..left.rank() - 2],
            [right.shape[axis] for axis in 0..right.rank() - 2],
            node.label()
        )
    }
    let product = if batch.len() == 0 {
        jt.matmul(left, right)
    } else {
        batched_matmul(left, right, batch, rows, inner, columns)
    }
    if left is not a { left.free() }
    if right is not b { right.free() }
    if a.rank() != 1 and b.rank() != 1 { return [product] }
    var trimmed = [dimension for dimension in batch]
    if a.rank() != 1 { trimmed.push(rows) }
    if b.rank() != 1 { trimmed.push(columns) }
    let squeezed = copy_of(product, shaped(trimmed))
    product.free()
    return [squeezed]
}

#: Reduce `axes` away, by moving them to the end and reducing the rows that
#: leaves. `op` is `row_reduce`'s: sum, mean, max, min, or an extreme's index.
fn reduce_over(x: Tensor, axes: list[int], keepdims: bool, op: int) -> Tensor {
    let rank = x.rank()
    var kept: list[int] = []
    for axis in 0..rank {
        if not axis in axes { kept.push(axis) }
    }
    var perm = [axis for axis in kept]
    for axis in axes { perm.push(axis) }
    var ordered = true
    for (slot, axis) in perm.enumerate() {
        if slot != axis { ordered = false }
    }
    let moved = if ordered { x } else { transpose_to(x, perm) }
    var inner = 1
    for axis in axes { inner *= x.shape[axis] }
    var dims: list[int] = []
    if keepdims {
        for axis in 0..rank { dims.push(if axis in axes { 1 } else { x.shape[axis] }) }
    } else {
        for axis in kept { dims.push(x.shape[axis]) }
    }
    let output = row_reduce_of(moved, x.size // inner, inner, op, shape: shaped(dims))
    if moved is not x { moved.free() }
    return output
}

fn sorted_axes(node: GraphNode, asked: list[int], rank: int) -> list[int] {
    var resolved: list[int] = []
    for axis in asked {
        let normalized = normalize_axis(axis, rank, node.label())
        if not normalized in resolved { resolved.push(normalized) }
    }
    resolved.sort()
    return resolved
}

fn op_reduce(node: GraphNode, ins: list[Tensor?], op: int) -> list[Tensor] {
    let x = need(ins, 0, node)
    let asked = ints_from(node, ins, 1, "axes") ?? []
    if asked.len() == 0 and attr_int(node, "noop_with_empty_axes", 0) != 0 {
        return [copy_of(x, x.shape)]
    }
    let wanted = if asked.len() == 0 { [axis for axis in 0..x.rank()] } else { asked }
    let axes = sorted_axes(node, wanted, x.rank())
    return [reduce_over(x, axes, attr_int(node, "keepdims", 1) != 0, op)]
}

fn op_reduce_sum(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_reduce(node, ins, 0)
}

fn op_reduce_mean(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_reduce(node, ins, 1)
}

fn op_reduce_max(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_reduce(node, ins, 2)
}

fn op_reduce_min(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_reduce(node, ins, 3)
}

fn op_arg_extreme(node: GraphNode, ins: list[Tensor?], want_max: bool) -> list[Tensor] {
    let x = need(ins, 0, node)
    let axis = normalize_axis(attr_int(node, "axis", 0), x.rank(), node.label())
    var op = if want_max { 4 } else { 5 }
    if attr_int(node, "select_last_index", 0) != 0 { op += 2 }
    return [reduce_over(x, [axis], attr_int(node, "keepdims", 1) != 0, op)]
}

fn op_argmax(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_arg_extreme(node, ins, true)
}

fn op_argmin(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_arg_extreme(node, ins, false)
}

fn softmax_like(node: GraphNode, ins: list[Tensor?], want_log: bool) -> list[Tensor] {
    let x = need(ins, 0, node)
    let rank = x.rank()
    let axis = normalize_axis(attr_int(node, "axis", -1), rank, node.label())
    let columns = x.shape[axis]
    if axis == rank - 1 { return [softmax_rows(x, x.size // columns, columns, want_log)] }
    var perm: list[int] = []
    for other in 0..rank {
        if other != axis { perm.push(other) }
    }
    perm.push(axis)
    let moved = transpose_to(x, perm)
    let normalized = softmax_rows(moved, x.size // columns, columns, want_log)
    moved.free()
    var back = [0 for _axis in 0..rank]
    for (slot, other) in perm.enumerate() { back[other] = slot }
    let output = transpose_to(normalized, back)
    normalized.free()
    return [output]
}

fn op_softmax(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return softmax_like(node, ins, false)
}

fn op_log_softmax(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return softmax_like(node, ins, true)
}

fn resample(node: GraphNode, x: Tensor, height: int, width: int, coord: int, rule: int) -> Tensor {
    if x.rank() != 4 { throw ValueError(f"{node.label()} needs a 4-D NCHW input, got {x.shape}") }
    let name = attr_str(node, "mode", "nearest")
    let mode = match name {
        "nearest" => 0,
        "linear" => 1,
        "bilinear" => 1,
        _ => throw ValueError(f"{node.label()} asks for {name} resampling, which is not implemented"),
    }
    if height <= 0 or width <= 0 {
        throw ValueError(f"{node.label()} resamples {x.shape} to {height}x{width}")
    }
    let output = Tensor([x.shape[0], x.shape[1], height, width])
    kernel_for("resize_nchw").dispatch_async(
        [x.buffer(), output.buffer()],
        output.size,
        [output.size, x.shape[2], x.shape[3], height, width, mode, coord, rule]
    )
    return output
}

fn scaled_extent(node: GraphNode, x: Tensor, scales: list[float]) -> tuple[int, int] {
    if scales.len() != 4 {
        throw ValueError(f"{node.label()} needs one scale an axis, got {scales.len()}")
    }
    if scales[0] != 1.0 or scales[1] != 1.0 {
        throw ValueError(
            f"{node.label()} scales batch and channels by {scales[0]} and {scales[1]}, and " +
                "this executor only resamples the spatial axes"
        )
    }
    return (int(float(x.shape[2]) * scales[2]), int(float(x.shape[3]) * scales[3]))
}

fn op_resize(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let x = need(ins, 0, node)
    let rounding = attr_str(node, "nearest_mode", "round_prefer_floor")
    let rule = match rounding {
        "round_prefer_floor" => 0,
        "round_prefer_ceil" => 1,
        "floor" => 2,
        "ceil" => 3,
        _ => throw ValueError(f"{node.label()} asks for nearest_mode {rounding}, which is not implemented"),
    }
    let transform = attr_str(node, "coordinate_transformation_mode", "half_pixel")
    let coord = match transform {
        "half_pixel" => 0,
        "asymmetric" => 1,
        "align_corners" => 2,
        "pytorch_half_pixel" => 3,
        _ => throw ValueError(f"{node.label()} asks for {transform} coordinates, which is not implemented"),
    }
    #: Resize-10 took only `(X, scales)`; every version since puts `roi`
    #: second, so a two-input node is the old shape and anything longer is
    #: the new one.
    let scales_at = if ins.len() == 2 { 1 } else { 2 }
    let sizes = maybe(ins, 3)
    if sizes is not null {
        let dims = host_ints(sizes)
        if dims.len() != 4 {
            throw ValueError(f"{node.label()} needs one size an axis, got {dims.len()}")
        }
        if dims[0] != x.shape[0] or dims[1] != x.shape[1] {
            throw ValueError(
                f"{node.label()} resizes {x.shape} to {dims}, changing batch or channels"
            )
        }
        return [resample(node, x, dims[2], dims[3], coord, rule)]
    }
    let scales = maybe(ins, scales_at)
    if scales is null { throw ValueError(f"{node.label()} needs either scales or sizes") }
    let (height, width) = scaled_extent(node, x, scales.download())
    return [resample(node, x, height, width, coord, rule)]
}

fn op_upsample(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let x = need(ins, 0, node)
    let supplied = maybe(ins, 1)
    let scales = if supplied is not null {
        supplied.download()
    } else {
        attr_floats(node, "scales", [])
    }
    let (height, width) = scaled_extent(node, x, scales)
    #: Upsample predates both `coordinate_transformation_mode` and
    #: `nearest_mode`: it always read the source coordinate as `out / scale`
    #: and took the floor of it.
    return [resample(node, x, height, width, 1, 2)]
}

var _handlers: dict[str, fn(GraphNode, list[Tensor?]) -> list[Tensor]] = {}

#: Every operator the executor knows, keyed by its ONNX name.
#:
#: Built once on first use rather than at module load, because a top-level
#: dict of function values would have to name each one before it is defined.
fn handlers() -> dict[str, fn(GraphNode, list[Tensor?]) -> list[Tensor]] {
    if _handlers.len() > 0 { return _handlers }
    _handlers["Conv"] = op_conv
    _handlers["ConvTranspose"] = op_conv_transpose
    _handlers["Gemm"] = op_gemm
    _handlers["MatMul"] = op_matmul
    _handlers["Relu"] = op_relu
    _handlers["LeakyRelu"] = op_leaky_relu
    _handlers["PRelu"] = op_prelu
    _handlers["Elu"] = op_elu
    _handlers["Sigmoid"] = op_sigmoid
    _handlers["Silu"] = op_silu
    _handlers["HardSigmoid"] = op_hard_sigmoid
    _handlers["Tanh"] = op_tanh
    _handlers["Softplus"] = op_softplus
    _handlers["Erf"] = op_erf
    _handlers["Softmax"] = op_softmax
    _handlers["LogSoftmax"] = op_log_softmax
    _handlers["Clip"] = op_clip
    _handlers["Add"] = op_add
    _handlers["Sub"] = op_sub
    _handlers["Mul"] = op_mul
    _handlers["Div"] = op_div
    _handlers["Mod"] = op_mod
    _handlers["Pow"] = op_pow
    _handlers["Sqrt"] = op_sqrt
    _handlers["Exp"] = op_exp
    _handlers["Log"] = op_log
    _handlers["Abs"] = op_abs
    _handlers["Neg"] = op_neg
    _handlers["Min"] = op_min
    _handlers["Max"] = op_max
    _handlers["Sum"] = op_sum
    _handlers["Mean"] = op_mean
    _handlers["Concat"] = op_concat
    _handlers["Split"] = op_split
    _handlers["Reshape"] = op_reshape
    _handlers["Flatten"] = op_flatten
    _handlers["Squeeze"] = op_squeeze
    _handlers["Unsqueeze"] = op_unsqueeze
    _handlers["Transpose"] = op_transpose
    _handlers["Slice"] = op_slice
    _handlers["Gather"] = op_gather
    _handlers["Pad"] = op_pad
    _handlers["MaxPool"] = op_max_pool
    _handlers["AveragePool"] = op_average_pool
    _handlers["GlobalAveragePool"] = op_global_average_pool
    _handlers["GlobalMaxPool"] = op_global_max_pool
    _handlers["BatchNormalization"] = op_batch_norm
    _handlers["InstanceNormalization"] = op_instance_norm
    _handlers["LayerNormalization"] = op_layer_norm
    _handlers["GRU"] = op_gru
    _handlers["LSTM"] = op_lstm
    _handlers["LRN"] = op_lrn
    _handlers["Dropout"] = op_dropout
    _handlers["Identity"] = op_identity
    _handlers["Constant"] = op_constant
    _handlers["ConstantOfShape"] = op_constant_of_shape
    _handlers["Shape"] = op_shape
    _handlers["Cast"] = op_cast
    _handlers["ReduceMean"] = op_reduce_mean
    _handlers["ReduceSum"] = op_reduce_sum
    _handlers["ReduceMax"] = op_reduce_max
    _handlers["ReduceMin"] = op_reduce_min
    _handlers["ArgMax"] = op_argmax
    _handlers["ArgMin"] = op_argmin
    _handlers["Resize"] = op_resize
    _handlers["Upsample"] = op_upsample
    _handlers["Where"] = op_where
    _handlers["Equal"] = op_equal
    _handlers["Greater"] = op_greater
    _handlers["Less"] = op_less
    _handlers["Not"] = op_not
    _handlers["And"] = op_and
    _handlers["Or"] = op_or
    return _handlers
}

#: Whether `run_graph` can execute this ONNX operator. An importer can ask
#: before it builds a node, and say what it could not translate up front
#: rather than letting the executor discover it halfway through a forward
#: pass.
pub fn is_implemented(op: str) -> bool { return op in handlers() }

#: Every operator name `run_graph` understands, in alphabetical order.
pub fn implemented_operators() -> list[str] {
    var names: list[str] = []
    for (name, _handler) in handlers().items() { names.push(name) }
    names.sort()
    return names
}

#: Run `graph` on `feeds` and return one tensor for each name in
#: `graph.outputs`.
#:
#: The caller owns every tensor that comes back and should free them. Nothing
#: passed in is freed or modified: an output that is only an alias of a feed or
#: an initializer is copied so that freeing the results cannot reach back into
#: the caller's weights.
#:
#: Raises `ValueError` naming the node for an operator that is not
#: implemented, for a missing feed, and for every shape the operators
#: themselves refuse.
pub fn run_graph(graph: Graph, feeds: dict[str, Tensor]) -> dict[str, Tensor] {
    if graph.outputs.len() == 0 {
        throw ValueError("this graph names no outputs, so there is nothing to run it for")
    }
    let order = graph.schedule()
    var values: dict[str, Tensor] = {}
    var mine: dict[str, bool] = {}
    for (name, weight) in graph.initializers.items() { values[name] = weight }
    for (name, fed) in feeds.items() { values[name] = fed }
    for name in graph.inputs {
        if values.get(name, null) is null {
            throw ValueError(f"the graph input {name} was not fed")
        }
    }

    #: The index of the last node that reads each name, which is when the
    #: value behind it stops being needed.
    var last_read: dict[str, int] = {}
    for (index, node) in order.enumerate() {
        for name in node.inputs {
            if name.len() > 0 { last_read[name] = index }
        }
    }

    let table = handlers()
    let profiling = env("JAICV_GRAPH_PROFILE") is not null
    var op_ms: dict[str, float] = {}
    var op_count: dict[str, int] = {}
    var produced_slot: list[Tensor] = []
    try {
        for (index, node) in order.enumerate() {
            var ins: list[Tensor?] = []
            for name in node.inputs {
                if name.len() == 0 {
                    ins.push(null)
                    continue
                }
                let found = values.get(name, null)
                if found is null {
                    throw ValueError(f"{node.label()} reads {name}, which has no value")
                }
                ins.push(found)
            }
            #: Whose the input was *before* this node ran: a node may write a
            #: name it also reads, and freeing on the state afterwards would
            #: release the caller's initializer or the fresh output instead of
            #: the value that has just been consumed.
            var owned_input: list[bool] = []
            for name in node.inputs {
                owned_input.push(name.len() > 0 and mine.get(name, false))
            }
            let handler = table.get(node.op, null)
            if handler is null { throw ValueError(f"{node.label()} is not implemented") }
            #: `JAICV_GRAPH_PROFILE=1` totals the wall clock per operator and
            #: prints the table when the graph finishes. Every op queues GPU
            #: work rather than waiting for it, so the figures only mean
            #: anything with a synchronise between them -- which is why the
            #: profile is off by default rather than merely unprinted.
            if profiling {
                gpu.synchronize()
                let started = Instant.now()
                let timed = handler(node, ins)
                gpu.synchronize()
                let spent_ms = started.elapsed().to_seconds() * 1000.0
                op_ms[node.op] = op_ms.get(node.op, 0.0) + spent_ms
                op_count[node.op] = op_count.get(node.op, 0) + 1
                produced_slot = timed
            } else {
                produced_slot = handler(node, ins)
            }
            let produced = produced_slot
            if produced.len() < node.outputs.len() {
                throw ValueError(
                    f"{node.label()} names {node.outputs.len()} outputs and produced " +
                        f"{produced.len()}"
                )
            }
            for (slot, tensor) in produced.enumerate() {
                let name = if slot < node.outputs.len() { node.outputs[slot] } else { "" }
                if name.len() == 0 {
                    tensor.free()
                    continue
                }
                values[name] = tensor
                mine[name] = true
            }
            for (slot, name) in node.inputs.enumerate() {
                if not owned_input[slot] or last_read.get(name, -1) != index { continue }
                if name in graph.outputs { continue }
                let spent = ins[slot]
                if spent is not null { spent.free() }
            }
        }
    } catch failure: Error {
        for (name, owned) in mine.items() {
            if not owned { continue }
            let stranded = values.get(name, null)
            if stranded is not null { stranded.free() }
        }
        throw failure
    }

    var results: dict[str, Tensor] = {}
    for name in graph.outputs {
        let value = values.get(name, null)
        if value is null { throw ValueError(f"the graph output {name} was never produced") }
        #: A feed or an initializer routed straight to an output is copied, so
        #: that freeing the results never frees the caller's own tensor.
        results[name] = if mine.get(name, false) { value } else { copy_of(value, value.shape) }
    }
    for (name, owned) in mine.items() {
        if owned and not name in graph.outputs {
            let leftover = values.get(name, null)
            if leftover is not null { leftover.free() }
        }
    }
    if profiling { report_profile(op_ms, op_count) }
    return results
}

#: The per-operator totals `JAICV_GRAPH_PROFILE` collected, slowest first.
fn report_profile(op_ms: dict[str, float], op_count: dict[str, int]) -> void {
    var names: list[str] = []
    for name in op_ms.keys() { names.push(name) }
    #: A plain insertion sort: this runs once at the end of a graph and there
    #: are a few dozen distinct operators at most.
    for outer in 1..names.len() {
        let held = names[outer]
        var inner = outer - 1
        while inner >= 0 and op_ms[names[inner]] < op_ms[held] {
            names[inner + 1] = names[inner]
            inner -= 1
        }
        names[inner + 1] = held
    }
    var total = 0.0
    for name in names { total += op_ms[name] }
    print(f"[graph] {total:.1f}ms over {names.len()} operator kinds")
    for name in names {
        let spent = op_ms[name]
        print(
            f"[graph]   {name:<22} {spent:8.2f}ms  {op_count[name]:4} calls  " +
                f"{spent / total * 100.0:5.1f}%"
        )
    }
}

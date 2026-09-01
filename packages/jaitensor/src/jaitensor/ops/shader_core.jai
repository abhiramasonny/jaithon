#: Metal prelude, the elementwise and row-reduction kernels, and their
#: backward passes.
pub let CORE_SOURCE = """#include <metal_stdlib>
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

"""

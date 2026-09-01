#: The attention shader library: split/merge heads, the MMA score and
#: context kernels, the flash prefill kernels, and `tril`.
pub let ATTN_SOURCE = """#include <metal_stdlib>
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

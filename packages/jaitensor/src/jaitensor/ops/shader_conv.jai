#: Metal kernels for im2col, the normalisation forward passes, gathers,
#: and the remaining elementwise and reduction entry points.
pub let CONV_SOURCE = """kernel void im2col_nhwc(device const float* input [[buffer(0)]],
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

"""

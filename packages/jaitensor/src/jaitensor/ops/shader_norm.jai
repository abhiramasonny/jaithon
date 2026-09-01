#: Metal kernels for the normalisation backward passes and the scalar
#: accumulators.
pub let NORM_SOURCE = """/* A threadgroup to the row, for the same reason the forward takes one: a
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

"""

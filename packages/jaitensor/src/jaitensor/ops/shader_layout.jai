#: Metal kernels for layout changes, the pooling forward passes, and the
#: loss entry points.
pub let LAYOUT_SOURCE = """/* Held to `TRANSPOSE_TILE` and `TRANSPOSE_GROUP` on the host. */
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

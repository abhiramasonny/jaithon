#: Metal kernels for the pooling, cropping and upsampling backward passes.
pub let POOL_SOURCE = """kernel void col2im_nhwc(device const float* col [[buffer(0)]],
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

"""

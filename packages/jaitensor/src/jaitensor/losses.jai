#: Loss functions.
import std.gpu as gpu
from std.gpu import Buffer, Kernel
from std.os import env_or
from .tensor import Tensor, GradNode
from .trace import record_sparse_ce
from .ops import mse as mse_op, binary_cross_entropy as bce_op
from .ops import binary_cross_entropy_with_logits as bce_logits_op
from .ops import scale_by

#: Something a `Sequential` can train against: turns a prediction and a
#: target into one scalar tensor, differentiable back into `prediction`.
pub trait Loss {
    #: Return the scalar loss for this batch, recording a `GradNode` on
    #: `prediction` when it requires a gradient. `loss_acc` receives
    #: `mean * acc_weight` when both are provided, so the training loop can
    #: skip a second scalar kernel.
    fn forward(
        self,
        prediction: Tensor,
        target: Tensor,
        loss_acc: Tensor? = null,
        acc_weight: int = 0
    ) -> Tensor

    #: Whether this loss expects the final layer to hand it raw logits
    #: instead of an already-normalized distribution (e.g. softmax output).
    #: `Sequential._prepare` checks this against the final layer's
    #: `wants_softmax_output()` to catch a model/loss mismatch before
    #: training starts.
    fn is_from_logits(self) -> bool { return false }

    #: Sparse softmax cross-entropy, which the fused two-layer SGD path can run.
    fn is_sparse_categorical(self) -> bool { return false }
}

let SOURCE = """#include <metal_stdlib>
using namespace metal;

kernel void sample_loss_and_delta(device const float* logits [[buffer(0)]],
                                  device const float* labels [[buffer(1)]],
                                  device float* delta [[buffer(2)]],
                                  device float* sample_loss [[buffer(3)]],
                                  device const float* inv_batch [[buffer(4)]],
                                  constant uint& batch [[buffer(5)]],
                                  constant uint& classes [[buffer(6)]],
                                  uint sample [[thread_position_in_grid]]) {
    if (sample >= batch) return;
    uint base = sample * classes;
    float peak = logits[base];
    for (uint digit = 1; digit < classes; ++digit) peak = max(peak, logits[base + digit]);
    float total = 0.0f;
    for (uint digit = 0; digit < classes; ++digit) total += exp(logits[base + digit] - peak);
    uint answer = uint(labels[sample]);
    float scale = inv_batch[0];
    for (uint digit = 0; digit < classes; ++digit) {
        float probability = exp(logits[base + digit] - peak) / total;
        float d = probability - (digit == answer ? 1.0f : 0.0f);
        delta[base + digit] = d * scale;
    }
    sample_loss[sample] = peak + log(total) - logits[base + answer];
}

// Sparse cross-entropy a row at a time across many threads.
//
// One simdgroup takes a row when it is narrow and a whole threadgroup of 256
// when it is wide. Neighbouring threads read neighbouring classes, so every
// read is coalesced, and each thread keeps an online maximum and the sum of
// exponentials below it in one pass; the partial pairs are then joined
// across the simdgroup and, for a wide row, across the threadgroup. The
// delta, when wanted, is a second pass over the same row. The gradient
// kernels repeat exactly the delta's arithmetic from the saved (max, sum),
// so the autograd gradient is the static path's delta bit for bit when the
// upstream is one.

constant constexpr float CE_LOW = -3.0e38f;
constant constexpr uint CE_WIDE = 256;

inline float2 ce_scan(device const float* row, uint classes, uint first, uint step) {
    float peak = CE_LOW;
    float total = 0.0f;
    for (uint digit = first; digit < classes; digit += step) {
        float v = row[digit];
        if (v > peak) {
            total = total * exp(peak - v) + 1.0f;
            peak = v;
        } else {
            total += exp(v - peak);
        }
    }
    return float2(peak, total);
}

inline float2 ce_join(float2 part) {
    float peak = simd_max(part.x);
    float total = simd_sum(part.y * exp(part.x - peak));
    return float2(peak, total);
}

inline float ce_delta(float v, float peak, float total, bool hit, float scale) {
    float probability = exp(v - peak) / total;
    float d = probability - (hit ? 1.0f : 0.0f);
    return d * scale;
}

inline void ce_finish(device const float* x,
                      device const float* labels,
                      device float* sample_loss,
                      device float* stats,
                      uint row,
                      float2 joined) {
    uint answer = uint(labels[row]);
    sample_loss[row] = joined.x + log(joined.y) - x[answer];
    stats[row * 3] = joined.x;
    stats[row * 3 + 1] = joined.y;
    stats[row * 3 + 2] = labels[row];
}

kernel void ce_rows_narrow(device const float* logits [[buffer(0)]],
                           device const float* labels [[buffer(1)]],
                           device float* delta [[buffer(2)]],
                           device float* sample_loss [[buffer(3)]],
                           device const float* inv_batch [[buffer(4)]],
                           device float* stats [[buffer(5)]],
                           constant uint& batch [[buffer(6)]],
                           constant uint& classes [[buffer(7)]],
                           constant uint& write_delta [[buffer(8)]],
                           uint gid [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    uint row = gid / 32;
    if (row >= batch) return;
    device const float* x = logits + ulong(row) * classes;
    float2 joined = ce_join(ce_scan(x, classes, lane, 32));
    if (lane == 0) ce_finish(x, labels, sample_loss, stats, row, joined);
    if (write_delta == 0) return;
    uint answer = uint(labels[row]);
    float scale = inv_batch[0];
    device float* d = delta + ulong(row) * classes;
    for (uint digit = lane; digit < classes; digit += 32) {
        d[digit] = ce_delta(x[digit], joined.x, joined.y, digit == answer, scale);
    }
}

kernel void ce_rows_wide(device const float* logits [[buffer(0)]],
                         device const float* labels [[buffer(1)]],
                         device float* delta [[buffer(2)]],
                         device float* sample_loss [[buffer(3)]],
                         device const float* inv_batch [[buffer(4)]],
                         device float* stats [[buffer(5)]],
                         constant uint& batch [[buffer(6)]],
                         constant uint& classes [[buffer(7)]],
                         constant uint& write_delta [[buffer(8)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint lid [[thread_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float peaks[CE_WIDE / 32];
    threadgroup float totals[CE_WIDE / 32];
    if (row >= batch) return;
    device const float* x = logits + ulong(row) * classes;
    float2 part = ce_join(ce_scan(x, classes, lid, CE_WIDE));
    if (lane == 0) {
        peaks[group] = part.x;
        totals[group] = part.y;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Every simdgroup joins the same eight pairs in the same order, so they
    // all hold the same answer and no second barrier is needed.
    float2 gathered = lane < CE_WIDE / 32 ? float2(peaks[lane], totals[lane]) : float2(CE_LOW, 0.0f);
    float2 joined = ce_join(gathered);
    if (lid == 0) ce_finish(x, labels, sample_loss, stats, row, joined);
    if (write_delta == 0) return;
    uint answer = uint(labels[row]);
    float scale = inv_batch[0];
    device float* d = delta + ulong(row) * classes;
    for (uint digit = lid; digit < classes; digit += CE_WIDE) {
        d[digit] = ce_delta(x[digit], joined.x, joined.y, digit == answer, scale);
    }
}

kernel void ce_grad_narrow(device const float* logits [[buffer(0)]],
                           device const float* stats [[buffer(1)]],
                           device const float* inv_batch [[buffer(2)]],
                           device const float* upstream [[buffer(3)]],
                           device float* grad [[buffer(4)]],
                           constant uint& batch [[buffer(5)]],
                           constant uint& classes [[buffer(6)]],
                           uint gid [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    uint row = gid / 32;
    if (row >= batch) return;
    float peak = stats[row * 3];
    float total = stats[row * 3 + 1];
    uint answer = uint(stats[row * 3 + 2]);
    float scale = inv_batch[0];
    float up = upstream[0];
    device const float* x = logits + ulong(row) * classes;
    device float* g = grad + ulong(row) * classes;
    for (uint digit = lane; digit < classes; digit += 32) {
        g[digit] = ce_delta(x[digit], peak, total, digit == answer, scale) * up;
    }
}

kernel void ce_grad_wide(device const float* logits [[buffer(0)]],
                         device const float* stats [[buffer(1)]],
                         device const float* inv_batch [[buffer(2)]],
                         device const float* upstream [[buffer(3)]],
                         device float* grad [[buffer(4)]],
                         constant uint& batch [[buffer(5)]],
                         constant uint& classes [[buffer(6)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint lid [[thread_index_in_threadgroup]]) {
    if (row >= batch) return;
    float peak = stats[row * 3];
    float total = stats[row * 3 + 1];
    uint answer = uint(stats[row * 3 + 2]);
    float scale = inv_batch[0];
    float up = upstream[0];
    device const float* x = logits + ulong(row) * classes;
    device float* g = grad + ulong(row) * classes;
    for (uint digit = lid; digit < classes; digit += CE_WIDE) {
        g[digit] = ce_delta(x[digit], peak, total, digit == answer, scale) * up;
    }
}

constexpr constant uint LOSS_GROUP = 256;

kernel void reduce_mean_partial(device const float* values [[buffer(0)]],
                                device float* partials [[buffer(1)]],
                                constant uint& n [[buffer(2)]],
                                uint gid [[thread_position_in_grid]],
                                uint lid [[thread_index_in_threadgroup]],
                                uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[LOSS_GROUP];
    scratch[lid] = (gid < n) ? values[gid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = LOSS_GROUP / 2; stride > 0; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) partials[wid] = scratch[0];
}

kernel void finalize_mean(device const float* partials [[buffer(0)]],
                          device float* mean [[buffer(1)]],
                          device float* epoch_total [[buffer(2)]],
                          constant uint& partial_count [[buffer(3)]],
                          constant uint& batch [[buffer(4)]],
                          constant uint& weight [[buffer(5)]],
                          uint id [[thread_position_in_grid]]) {
    if (id != 0) return;
    float total = 0.0f;
    for (uint i = 0; i < partial_count; ++i) total += partials[i];
    float value = total / float(batch);
    mean[0] = value;
    if (weight != 0) epoch_total[0] += value * float(weight);
}

"""

const _GROUP = 256

#: Whether sparse cross-entropy runs a row across a simdgroup or threadgroup
#: (`ce_rows_*`, coalesced, one pass for the statistics) or a row per thread
#: (`sample_loss_and_delta`, three serial passes, strided reads). On by
#: default; `JAITENSOR_CE_ROWS=0` takes the old kernel, for an A/B in one
#: binary. At 2048 rows the old one took 214 ms against torch's 14.6 at a
#: 50257-class vocabulary. Read once.
let _CE_ROWS = env_or("JAITENSOR_CE_ROWS", "1") != "0"

#: Rows up to this many classes go to one simdgroup each, wider ones to a
#: threadgroup of `_CE_WIDE`.
const _CE_NARROW_MAX = 1024
const _CE_WIDE = 256

var _kernel_cache: dict[str, Kernel] = {}
var _scale_batch = 0
var _scale_buffer: Buffer? = null

fn cached(entry: str) -> Kernel {
    let existing = _kernel_cache.get(entry, null)
    if existing is not null { return existing }
    let kernel = Kernel.compile(SOURCE, entry)
    _kernel_cache[entry] = kernel
    return kernel
}

fn scale_buffer_for(batch: int) -> Buffer {
    let existing = _scale_buffer
    if existing is not null {
        if _scale_batch == batch { return existing }
        existing.free()
    }
    let created = Buffer.of([1.0 / float(batch)])
    _scale_buffer = created
    _scale_batch = batch
    return created
}

#: Compile loss kernels before the first timed training step.
pub fn warmup_kernels() -> void {
    if _CE_ROWS {
        cached("ce_rows_narrow")
        cached("ce_rows_wide")
        cached("ce_grad_narrow")
        cached("ce_grad_wide")
    } else {
        cached("sample_loss_and_delta")
    }
    cached("reduce_mean_partial")
    cached("finalize_mean")
}

@trace
pub fn sparse_ce_core(
    prediction: Tensor,
    target: Tensor,
    loss_acc: Tensor,
    acc_weight: int
) -> tuple[Tensor, Tensor] {
    if prediction.rank() != 2 {
        throw ValueError(f"loss needs rank-two predictions, got {prediction.shape}")
    }
    let batch = prediction.shape[0]
    let classes = prediction.shape[1]
    if target.size != batch {
        throw ValueError(f"loss needs {batch} labels, got {target.size}")
    }
    record_sparse_ce(batch, classes)
    let delta = Tensor([batch, classes])
    let sample_loss = Tensor([batch])
    if _CE_ROWS {
        let stats = Tensor([batch, 3])
        dispatch_ce_rows(prediction, target, delta, sample_loss, stats, true)
        stats.free()
    } else {
        let sample_kernel = cached("sample_loss_and_delta")
        sample_kernel.dispatch_async(
            [
                prediction.buffer(),
                target.buffer(),
                delta.buffer(),
                sample_loss.buffer(),
                scale_buffer_for(batch),
            ],
            batch,
            [batch, classes]
        )
    }
    let mean = mean_of_samples(sample_loss, batch, loss_acc, acc_weight)
    sample_loss.free()
    return (mean, delta)
}

#: Launch `ce_rows_*`: per-sample loss, the row statistics a later gradient
#: needs (max, sum of exponentials, label), and the delta when `with_delta`.
fn dispatch_ce_rows(
    prediction: Tensor,
    target: Tensor,
    delta: Tensor?,
    sample_loss: Tensor,
    stats: Tensor,
    with_delta: bool
) -> void {
    let batch = prediction.shape[0]
    let classes = prediction.shape[1]
    #: Bound to something real when there is no delta; it is never written.
    let delta_buffer = if delta is not null { delta.buffer() } else { stats.buffer() }
    let write = if with_delta and delta is not null { 1 } else { 0 }
    let buffers = [
        prediction.buffer(),
        target.buffer(),
        delta_buffer,
        sample_loss.buffer(),
        scale_buffer_for(batch),
        stats.buffer(),
    ]
    if classes <= _CE_NARROW_MAX {
        cached("ce_rows_narrow").dispatch_async(buffers, batch * 32, [batch, classes, write], group_size: 256)
    } else {
        cached("ce_rows_wide").dispatch_async(buffers, batch * _CE_WIDE, [batch, classes, write], group_size: _CE_WIDE)
    }
}

#: The gradient of the mean loss with respect to the logits, times the
#: device scalar `upstream`, from the statistics `dispatch_ce_rows` saved.
fn ce_grad(prediction: Tensor, stats: Tensor, upstream: Tensor) -> Tensor {
    let batch = prediction.shape[0]
    let classes = prediction.shape[1]
    let grad = Tensor([batch, classes])
    let buffers = [
        prediction.buffer(),
        stats.buffer(),
        scale_buffer_for(batch),
        upstream.buffer(),
        grad.buffer(),
    ]
    if classes <= _CE_NARROW_MAX {
        cached("ce_grad_narrow").dispatch_async(buffers, batch * 32, [batch, classes], group_size: 256)
    } else {
        cached("ce_grad_wide").dispatch_async(buffers, batch * _CE_WIDE, [batch, classes], group_size: _CE_WIDE)
    }
    return grad
}

#: Mean sparse cross-entropy and the row statistics its gradient needs,
#: without the delta: one read of the logits instead of two reads and a write.
fn sparse_ce_stats(
    prediction: Tensor,
    target: Tensor,
    loss_acc: Tensor?,
    acc_weight: int
) -> tuple[Tensor, Tensor] {
    let batch = prediction.shape[0]
    let classes = prediction.shape[1]
    record_sparse_ce(batch, classes)
    let sample_loss = Tensor([batch])
    let stats = Tensor([batch, 3])
    dispatch_ce_rows(prediction, target, null, sample_loss, stats, false)
    let mean = mean_of_samples(sample_loss, batch, loss_acc, acc_weight)
    sample_loss.free()
    return (mean, stats)
}

#: The mean of `batch` per-sample losses, also added to `loss_acc` with
#: `acc_weight` when that is positive. Without an accumulator nothing is
#: added, and the kernel's unused slot is bound to the partials.
fn mean_of_samples(sample_loss: Tensor, batch: int, loss_acc: Tensor?, acc_weight: int) -> Tensor {
    let groups = (batch + _GROUP - 1) // _GROUP
    let partials = Tensor([groups])
    let reduce_kernel = cached("reduce_mean_partial")
    reduce_kernel.dispatch_async(
        [sample_loss.buffer(), partials.buffer()],
        groups * _GROUP,
        [batch],
        group_size: _GROUP
    )
    let mean = Tensor([1])
    let finalize_kernel = cached("finalize_mean")
    let weight = if acc_weight > 0 and loss_acc is not null { acc_weight } else { 0 }
    let total_buffer = if loss_acc is not null { loss_acc.buffer() } else { partials.buffer() }
    finalize_kernel.dispatch_async(
        [partials.buffer(), mean.buffer(), total_buffer],
        1,
        [groups, batch, weight]
    )
    partials.free()
    return mean
}

#: Mean sparse softmax cross-entropy and per-sample delta for static backward.
#:
#: `delta` is scaled by `1 / batch` and must be freed by the caller.
pub fn sparse_ce_with_delta(
    prediction: Tensor,
    target: Tensor,
    loss_acc: Tensor? = null,
    acc_weight: int = 0
) -> tuple[Tensor, Tensor] {
    if loss_acc is not null {
        return sparse_ce_core(prediction, target, loss_acc, acc_weight)
    }
    let dummy = Tensor.zeros([1])
    let result = sparse_ce_core(prediction, target, dummy, 0)
    dummy.free()
    return result
}

#: Softmax cross-entropy against one integer class per sample.
pub class SparseCategoricalCrossentropy: Loss {
    #: Whether the final layer returns raw logits instead of softmax values.
    pub let from_logits: bool

    #: Configure sparse classification from probabilities or raw logits.
    #:
    #: Use a final `Activation.Softmax` with the default. Set `from_logits` and
    #: use a final `Activation.Linear` when callers need direct logits from
    #: `Sequential.predict`.
    pub fn init(self, from_logits: bool = false) { self.from_logits = from_logits }

    #: Mean sparse softmax cross-entropy over the batch, differentiable
    #: back into `prediction`. `prediction` is treated as raw logits either
    #: way — the fused kernel computes softmax internally either way, for
    #: the same numerically-stable log-sum-exp reason today's kernel does.
    pub fn forward(
        self,
        prediction: Tensor,
        target: Tensor,
        loss_acc: Tensor? = null,
        acc_weight: int = 0
    ) -> Tensor {
        if prediction.rank() != 2 {
            throw ValueError(f"loss needs rank-two predictions, got {prediction.shape}")
        }
        let batch = prediction.shape[0]
        let classes = prediction.shape[1]
        if target.size != batch {
            throw ValueError(f"loss needs {batch} labels, got {target.size}")
        }
        if _CE_ROWS {
            #: The delta waits for the backward, which computes it straight
            #: from the logits and three numbers a row: one pass over the
            #: logits here, one read and one write there, and nothing the size
            #: of the logits held between the two.
            let (mean, stats) = sparse_ce_stats(prediction, target, loss_acc, acc_weight)
            if prediction.requires_grad {
                mean.attach_creator(
                    GradNode([prediction], |upstream| [ce_grad(prediction, stats, upstream)], extras: [stats])
                )
            } else {
                stats.free()
            }
            return mean
        }
        let (mean, delta) = sparse_ce_with_delta(prediction, target, loss_acc, acc_weight)
        if prediction.requires_grad {
            #: Multiplied by what came down from above, not handed back as
            #: it stands. Every other loss here composes out of ops that each
            #: carry the upstream along, so only this one -- whose backward
            #: is written out by hand around a fused kernel -- dropped it,
            #: and the effect is that weighting a loss changes the number and
            #: not the gradient. `scale(loss, 3)` moved the parameters by
            #: exactly as much as `loss` did.
            mean.attach_creator(
                GradNode([prediction], |upstream| [scale_by(delta, upstream)], extras: [delta])
            )
        } else {
            delta.free()
        }
        return mean
    }

    #: Whether this loss was configured for raw logits — see `from_logits`.
    pub fn is_from_logits(self) -> bool { return self.from_logits }

    #: Sparse softmax cross-entropy, which the fused two-layer SGD path can run.
    pub fn is_sparse_categorical(self) -> bool { return true }
}

#: Mean squared error between a prediction tensor and a same-shaped target.
pub class MeanSquaredError: Loss {
    pub fn init(self) { }

    pub fn forward(
        self,
        prediction: Tensor,
        target: Tensor,
        loss_acc: Tensor? = null,
        acc_weight: int = 0
    ) -> Tensor {
        let _ = loss_acc
        let _ = acc_weight
        return mse_op(prediction, target)
    }
}

#: Mean binary cross-entropy. Pass `from_logits: true` for a stable logits path.
pub class BinaryCrossentropy: Loss {
    pub let from_logits: bool

    pub fn init(self, from_logits: bool = false) { self.from_logits = from_logits }

    pub fn forward(
        self,
        prediction: Tensor,
        target: Tensor,
        loss_acc: Tensor? = null,
        acc_weight: int = 0
    ) -> Tensor {
        let _ = loss_acc
        let _ = acc_weight
        if self.from_logits { return bce_logits_op(prediction, target) }
        return bce_op(prediction, target)
    }

    pub fn is_from_logits(self) -> bool { return self.from_logits }
}

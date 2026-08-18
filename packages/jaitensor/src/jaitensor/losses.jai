#: Loss functions.
import std.gpu as gpu
from std.gpu import Buffer, Kernel
from .tensor import Tensor, GradNode
from .trace import record_sparse_ce
from .ops import mse as mse_op, binary_cross_entropy as bce_op
from .ops import binary_cross_entropy_with_logits as bce_logits_op

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
    cached("sample_loss_and_delta")
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
    let weight = if acc_weight > 0 { acc_weight } else { 0 }
    finalize_kernel.dispatch_async(
        [partials.buffer(), mean.buffer(), loss_acc.buffer()],
        1,
        [groups, batch, weight]
    )
    sample_loss.free()
    partials.free()
    return (mean, delta)
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
        let (mean, delta) = sparse_ce_with_delta(prediction, target, loss_acc, acc_weight)
        if prediction.requires_grad {
            mean.attach_creator(GradNode([prediction], |upstream| [delta], extras: [delta]))
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

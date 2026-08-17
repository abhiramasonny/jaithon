#: Loss functions.
import std.gpu as gpu
from std.gpu import Kernel
from .tensor import Tensor, GradNode

#: Something a `Sequential` can train against: turns a prediction and a
#: target into one scalar tensor, differentiable back into `prediction`.
pub trait Loss {
    #: Return the scalar loss for this batch, recording a `GradNode` on
    #: `prediction` when it requires a gradient.
    fn forward(self, prediction: Tensor, target: Tensor) -> Tensor

    #: Whether this loss expects the final layer to hand it raw logits
    #: instead of an already-normalized distribution (e.g. softmax output).
    #: `Sequential._prepare` checks this against the final layer's
    #: `wants_softmax_output()` to catch a model/loss mismatch before
    #: training starts.
    fn is_from_logits(self) -> bool { return false }
}

let SOURCE = """#include <metal_stdlib>
using namespace metal;

kernel void sample_loss_and_delta(device const float* logits [[buffer(0)]],
                                  device const float* labels [[buffer(1)]],
                                  device float* delta [[buffer(2)]],
                                  device float* sample_loss [[buffer(3)]],
                                  constant uint& batch [[buffer(4)]],
                                  constant uint& classes [[buffer(5)]],
                                  uint sample [[thread_position_in_grid]]) {
    if (sample >= batch) return;
    uint base = sample * classes;
    float peak = logits[base];
    for (uint digit = 1; digit < classes; ++digit) peak = max(peak, logits[base + digit]);
    float total = 0.0f;
    for (uint digit = 0; digit < classes; ++digit) total += exp(logits[base + digit] - peak);
    uint answer = uint(labels[sample]);
    for (uint digit = 0; digit < classes; ++digit) {
        float probability = exp(logits[base + digit] - peak) / total;
        delta[base + digit] = probability - (digit == answer ? 1.0f : 0.0f);
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
                          constant uint& partial_count [[buffer(2)]],
                          constant uint& batch [[buffer(3)]],
                          uint id [[thread_position_in_grid]]) {
    if (id != 0) return;
    float total = 0.0f;
    for (uint i = 0; i < partial_count; ++i) total += partials[i];
    mean[0] = total / float(batch);
}

kernel void scale_by_upstream(device const float* delta [[buffer(0)]],
                              device float* grad [[buffer(1)]],
                              device const float* factor [[buffer(2)]],
                              constant uint& count [[buffer(3)]],
                              uint id [[thread_position_in_grid]]) {
    if (id < count) grad[id] = delta[id] * factor[0];
}
"""

const _GROUP = 256

var _kernel_cache: dict[str, Kernel] = {}

fn cached(entry: str) -> Kernel {
    let existing = _kernel_cache.get(entry, null)
    if existing is not null { return existing }
    let kernel = Kernel.compile(SOURCE, entry)
    _kernel_cache[entry] = kernel
    return kernel
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
    pub fn forward(self, prediction: Tensor, target: Tensor) -> Tensor {
        if prediction.rank() != 2 {
            throw ValueError(f"loss needs rank-two predictions, got {prediction.shape}")
        }
        let batch = prediction.shape[0]
        let classes = prediction.shape[1]
        let delta = Tensor([batch, classes])
        let sample_loss = Tensor([batch])
        let sample_kernel = cached("sample_loss_and_delta")
        sample_kernel.dispatch_async(
            [prediction.buffer(), target.buffer(), delta.buffer(), sample_loss.buffer()],
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
        finalize_kernel.dispatch_async([partials.buffer(), mean.buffer()], 1, [groups, batch])
        sample_loss.free()
        partials.free()
        delta.free()
        if prediction.requires_grad {
            mean.attach_creator(GradNode([prediction], fn(upstream: Tensor) -> list[Tensor] {
                let recomputed_delta = Tensor([batch, classes])
                let discarded_loss = Tensor([batch])
                defer {
                    discarded_loss.free()
                }
                let recompute_kernel = cached("sample_loss_and_delta")
                recompute_kernel.dispatch_async([prediction.buffer(), target.buffer(), recomputed_delta.buffer(), discarded_loss.buffer()], batch, [batch, classes])
                let grad = Tensor(recomputed_delta.shape)
                let scale_kernel = cached("scale_by_upstream")
                let factor_buffer = gpu.Buffer.of([1.0 / float(batch)])
                defer {
                    factor_buffer.free()
                }
                scale_kernel.dispatch_async([recomputed_delta.buffer(), grad.buffer(), factor_buffer], recomputed_delta.size, [recomputed_delta.size])
                recomputed_delta.free()
                return [grad]
            }))
        }
        return mean
    }

    #: Whether this loss was configured for raw logits — see `from_logits`.
    pub fn is_from_logits(self) -> bool { return self.from_logits }
}

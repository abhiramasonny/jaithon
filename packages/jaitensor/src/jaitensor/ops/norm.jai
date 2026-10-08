#: Batch, layer and group normalisation, and dropout.
from ..tensor import GradNode, Tensor
from .elementwise import add, relu as relu_op
from .kernels import cached_kernel, f32_bits
from .shape import reshape
from std.os import env

#: How many masks have been drawn, so that the next one differs from the last.
var _dropout_draws = 0

#: Inverted dropout. `p` is the drop probability.
#:
#: `seed` fixes the mask, for a test that wants the same one twice. Left out,
#: each call draws a fresh one.
#:
#: The seed used to be the tensor's own size, which is the same number on
#: every step -- so the same units were dropped every step of every epoch,
#: and two layers of the same shape dropped the same ones as each other. That
#: is not regularisation, it is a fixed pruning mask chosen by the shape, and
#: being perfectly repeatable it never looked wrong.
pub fn dropout(input: Tensor, p: float = 0.5, training: bool = true, seed: int? = null) -> Tensor {
    if p < 0.0 or p >= 1.0 { throw ValueError(f"dropout p must be in [0, 1), got {p}") }
    if not training or p == 0.0 {
        return reshape(input, input.shape)
    }
    let noise = Tensor(input.shape)
    let asked = if seed is null {
        _dropout_draws += 1
        _dropout_draws * 2654435761 + input.size
    } else {
        seed
    }
    noise.buffer().fill_uniform(0.0, 1.0, asked)
    let output = Tensor(input.shape)
    let keep = 1.0 - p
    let kernel = cached_kernel("dropout_apply")
    kernel.dispatch_async(
        [input.buffer(), noise.buffer(), output.buffer()],
        input.size,
        [input.size, f32_bits(keep), f32_bits(1.0 / keep)]
    )
    if input.requires_grad {
        output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
            let g = Tensor(input.shape)
            kernel.dispatch_async([upstream.buffer(), noise.buffer(), g.buffer()], input.size, [input.size, f32_bits(keep), f32_bits(1.0 / keep)])
            return [g]
        }, extras: [noise]))
    } else {
        noise.free()
    }
    return output
}

#: Whether `batch_norm`'s residual add and ReLU run inside its own pass, and
#: its elementwise passes four channels to a thread. `JAITENSOR_BN_FUSED=0`
#: composes the add and the ReLU from `add` and `relu` instead, the way a
#: model wrote them before, and keeps the one-float kernels, for an A/B in
#: one binary. Read once.
var _bn_fused: int = -1

fn bn_fused_on() -> bool {
    if _bn_fused < 0 {
        let text = env("JAITENSOR_BN_FUSED")
        _bn_fused = if text is not null and text == "0" { 0 } else { 1 }
    }
    return _bn_fused == 1
}

#: Channel-last batch normalization using per-channel mean and variance of `x`.
#: Blocks for the coalesced channel reduction, or zero when it does not apply.
#:
#: Each block's threads stay on one channel apiece only if the group holds a
#: whole number of channels, and the split has to be worth making: a couple of
#: rows a block spends more on joining the partials than the first pass saves.
#:
#: Where the channels come in fours the blocks read a float4 a step
#: (`channel_partials4`, `batch_norm_fused_partials4`): a [128,32,32,32]
#: forward went from 104-112 us to 93-95, [128,8,8,128] from 52-60 to 47-49.
#:
#: 128 blocks spread the reduction over the long NHW axis well enough: over
#: ResNet-8's blocks, 256 and 512 measured the same or up to 60% slower (a
#: [128,32,32,32] forward 133 us at 128, 138 at 256, 176 at 512), the joins
#: growing with the count while the first pass was already at its ceiling.
fn channel_chunks(nspatial: int, channels: int) -> int {
    if channels > 256 or 256 % channels != 0 { return 0 }
    let wanted = min(128, nspatial // 64)
    if wanted < 2 { return 0 }
    return wanted
}

#: The per-channel mean and biased variance of an NHWC block, as two
#: `[channels]` tensors the caller frees.
fn channel_moments(input: Tensor, nspatial: int, channels: int) -> list[Tensor] {
    let mean = Tensor([channels])
    let variance = Tensor([channels])
    let chunks = channel_chunks(nspatial, channels)
    if chunks > 0 {
        let rows_each = (nspatial + chunks - 1) // chunks
        let part_total = Tensor([chunks * channels])
        let part_square = Tensor([chunks * channels])
        let quads = channels % 4 == 0 and bn_fused_on()
        cached_kernel(if quads {
            "channel_partials4"
        } else {
            "channel_partials"
        }).dispatch_async([input.buffer(), part_total.buffer(), part_square.buffer()], chunks * 256, [nspatial, channels, rows_each], group_size: 256)
        cached_kernel("channel_moments_join").dispatch_async(
            [
                part_total.buffer(),
                part_square.buffer(),
                mean.buffer(),
                variance.buffer(),
                input.buffer(),
            ],
            channels,
            [chunks, channels, nspatial]
        )
        part_total.free()
        part_square.free()
    } else {
        cached_kernel("moments_channel").dispatch_async(
            [input.buffer(), mean.buffer(), variance.buffer()],
            channels * 256,
            [nspatial, channels],
            group_size: 256
        )
    }
    return [mean, variance]
}

#: `relu(batch_norm(x) + residual)` written out of the separate operations, as
#: models wrote it before the fused pass existed. The reference the fused pass
#: is held to, and what `JAITENSOR_BN_FUSED=0` runs.
fn composed_batch_norm(
    input: Tensor,
    gamma: Tensor,
    beta: Tensor,
    residual: Tensor?,
    then_relu: bool
) -> Tensor {
    let normed = batch_norm(input, gamma, beta)
    let summed = if residual is not null { add(normed, residual) } else { normed }
    let result = if then_relu { relu_op(summed) } else { summed }
    #: Without a graph nothing will free the steps in between.
    if not result.requires_grad {
        if summed is not normed { normed.free() }
        if result is not summed { summed.free() }
    }
    return result
}

#: Channel-last (NHWC) batch normalization over the batch's own statistics.
#:
#: `residual`, when given, is added after the affine step and `relu` then
#: applied, so `batch_norm(x, g, b, residual: s, relu: true)` is
#: `relu(batch_norm(x, g, b) + s)` -- a residual block's tail -- in one pass
#: over the activations instead of three, and its backward in two instead of
#: five: the ReLU mask is applied as the upstream gradient is read, and the
#: skip's gradient, which is that masked gradient, is written by the pass that
#: reads it. The result and every gradient match the composed operations.
pub fn batch_norm(
    input: Tensor,
    gamma: Tensor,
    beta: Tensor,
    residual: Tensor? = null,
    relu: bool = false
) -> Tensor {
    if input.rank() != 4 {
        throw ValueError(f"batch_norm needs NHWC input, got {input.shape}")
    }
    let channels = input.shape[3]
    if gamma.size != channels or beta.size != channels {
        throw ValueError("batch_norm gamma/beta must match the channel count")
    }
    let skip = residual
    if skip is not null and skip.shape != input.shape {
        throw ValueError(f"batch_norm residual {skip.shape} must match the input {input.shape}")
    }
    let fused = skip is not null or relu
    if fused and not bn_fused_on() { return composed_batch_norm(input, gamma, beta, skip, relu) }
    let nspatial = input.shape[0] * input.shape[1] * input.shape[2]
    let moments = channel_moments(input, nspatial, channels)
    let mean = moments[0]
    let variance = moments[1]
    let output = Tensor(input.shape)
    let flags = (if skip is not null { 1 } else { 0 }) + (if relu { 2 } else { 0 })
    #: Four channels to a thread wherever they come in fours, fused or not:
    #: the plain affine step ran 133 us over a [128,32,32,32] block one float
    #: to a thread and 112 four to one, with the ReLU folded in as well.
    let quads = channels % 4 == 0 and bn_fused_on()
    if fused or quads {
        let skip_buffer = if skip is not null { skip.buffer() } else { input.buffer() }
        let entry = if quads { "batch_norm_apply4" } else { "batch_norm_apply" }
        let count = if quads { input.size // 4 } else { input.size }
        cached_kernel(entry).dispatch_async(
            [
                input.buffer(),
                mean.buffer(),
                variance.buffer(),
                gamma.buffer(),
                beta.buffer(),
                skip_buffer,
                output.buffer(),
            ],
            count,
            [count, channels, flags]
        )
    } else {
        cached_kernel("batch_norm_forward").dispatch_async(
            [
                input.buffer(),
                mean.buffer(),
                variance.buffer(),
                gamma.buffer(),
                beta.buffer(),
                output.buffer(),
            ],
            input.size,
            [input.size, channels]
        )
    }
    let skip_grad = skip is not null and skip.requires_grad
    if input.requires_grad or gamma.requires_grad or beta.requires_grad or skip_grad {
        var inputs = [input, gamma, beta]
        if skip is not null { inputs.push(skip) }
        output.attach_creator(GradNode(inputs, fn(upstream: Tensor) -> list[Tensor] {
            let sum_dy = Tensor([channels])
            let sum_dy_xhat = Tensor([channels])
            let grad_gamma = Tensor([channels])
            let grad_beta = Tensor([channels])
            let grad_skip: Tensor? = if skip is not null { Tensor(input.shape) } else { null }
            #: What the input pass reads as the upstream gradient: `upstream`
            #: itself, or the masked copy the generic path makes.
            var masked: Tensor? = null
            let split = channel_chunks(nspatial, channels)
            if split > 0 {
                let rows_each = (nspatial + split - 1) // split
                let part_dy = Tensor([split * channels])
                let part_dy_xhat = Tensor([split * channels])
                let skip_out = if grad_skip is not null { grad_skip.buffer() } else { part_dy.buffer() }
                if quads {
                    #: The fused pass with nothing to fuse is the plain sums, in
                    #: the order the fused one takes them.
                    let wanted = if fused { flags } else { 0 }
                    cached_kernel("batch_norm_fused_partials4").dispatch_async([input.buffer(), mean.buffer(), variance.buffer(), output.buffer(), upstream.buffer(), part_dy.buffer(), part_dy_xhat.buffer(), skip_out], split * 256, [nspatial, channels, rows_each, wanted], group_size: 256)
                } elif fused {
                    cached_kernel("batch_norm_fused_partials").dispatch_async([input.buffer(), mean.buffer(), variance.buffer(), output.buffer(), upstream.buffer(), part_dy.buffer(), part_dy_xhat.buffer(), skip_out], split * 256, [nspatial, channels, rows_each, flags], group_size: 256)
                } else {
                    cached_kernel("channel_grad_partials").dispatch_async([input.buffer(), mean.buffer(), variance.buffer(), gamma.buffer(), upstream.buffer(), part_dy.buffer(), part_dy_xhat.buffer()], split * 256, [nspatial, channels, rows_each], group_size: 256)
                }
                cached_kernel("channel_grad_join").dispatch_async([part_dy.buffer(), part_dy_xhat.buffer(), gamma.buffer(), sum_dy.buffer(), sum_dy_xhat.buffer(), grad_gamma.buffer(), grad_beta.buffer()], channels, [split, channels])
                part_dy.free()
                part_dy_xhat.free()
            } else {
                var source = upstream
                if fused {
                    let made = if grad_skip is not null { grad_skip } else { Tensor(input.shape) }
                    cached_kernel("batch_norm_relu_mask").dispatch_async([output.buffer(), upstream.buffer(), made.buffer()], input.size, [input.size, flags])
                    masked = made
                    source = made
                }
                cached_kernel("norm_channel_sums").dispatch_async([input.buffer(), mean.buffer(), variance.buffer(), gamma.buffer(), source.buffer(), sum_dy.buffer(), sum_dy_xhat.buffer(), grad_gamma.buffer(), grad_beta.buffer()], channels * 256, [nspatial, channels], group_size: 256)
            }
            let grad_input = Tensor(input.shape)
            let held = masked
            #: With a skip, its gradient already is the masked upstream one, so
            #: the input pass reads that instead of the output and the upstream
            #: gradient both: one read less of the block.
            let ready = if held is not null { held } elif grad_skip is not null { grad_skip } else { null }
            if quads {
                let source = if ready is not null { ready } else { upstream }
                let mask = if ready is null and relu { 2 } else { 0 }
                cached_kernel("batch_norm_backward4").dispatch_async([input.buffer(), mean.buffer(), variance.buffer(), gamma.buffer(), output.buffer(), source.buffer(), sum_dy.buffer(), sum_dy_xhat.buffer(), grad_input.buffer()], input.size // 4, [input.size // 4, channels, nspatial, mask])
                if held is not null and grad_skip is null { held.free() }
            } elif fused and ready is null {
                cached_kernel("batch_norm_fused_backward").dispatch_async([input.buffer(), mean.buffer(), variance.buffer(), gamma.buffer(), output.buffer(), upstream.buffer(), sum_dy.buffer(), sum_dy_xhat.buffer(), grad_input.buffer()], input.size, [input.size, channels, nspatial, flags])
            } elif ready is not null {
                cached_kernel("norm_channel_backward").dispatch_async([input.buffer(), mean.buffer(), variance.buffer(), gamma.buffer(), ready.buffer(), sum_dy.buffer(), sum_dy_xhat.buffer(), grad_input.buffer()], input.size, [input.size, channels, nspatial])
                if held is not null and grad_skip is null { held.free() }
            } else {
                cached_kernel("norm_channel_backward").dispatch_async([input.buffer(), mean.buffer(), variance.buffer(), gamma.buffer(), upstream.buffer(), sum_dy.buffer(), sum_dy_xhat.buffer(), grad_input.buffer()], input.size, [input.size, channels, nspatial])
            }
            sum_dy.free()
            sum_dy_xhat.free()
            if grad_skip is not null { return [grad_input, grad_gamma, grad_beta, grad_skip] }
            return [grad_input, grad_gamma, grad_beta]
        }, extras: [mean, variance]))
        return output
    }
    mean.free()
    variance.free()
    return output
}

#: Normalize the last axis of a rank-two tensor.
pub fn layer_norm(input: Tensor, gamma: Tensor, beta: Tensor) -> Tensor {
    if input.rank() != 2 {
        throw ValueError(f"layer_norm needs rank-two input, got {input.shape}")
    }
    let width = input.shape[1]
    if gamma.size != width or beta.size != width {
        throw ValueError("layer_norm gamma/beta must match the last dimension")
    }
    let output = Tensor(input.shape)
    let kernel = cached_kernel("layer_norm_forward")
    #: A threadgroup of 256 to each row, so the dispatch is that many threads
    #: per row rather than one.
    kernel.dispatch_async(
        [input.buffer(), gamma.buffer(), beta.buffer(), output.buffer()],
        input.shape[0] * 256,
        [input.shape[0], width],
        group_size: 256
    )
    if input.requires_grad or gamma.requires_grad or beta.requires_grad {
        let rows = input.shape[0]
        output.attach_creator(GradNode([input, gamma, beta], fn(upstream: Tensor) -> list[Tensor] {
            let grad_input = Tensor(input.shape)
            #: Written by the input pass and read by the parameter pass, which
            #: runs after it on the same queue.
            let row_mean = Tensor([rows])
            let row_inv = Tensor([rows])
            let back = cached_kernel("layer_norm_backward")
            back.dispatch_async(
                [
                    input.buffer(),
                    gamma.buffer(),
                    upstream.buffer(),
                    grad_input.buffer(),
                    row_mean.buffer(),
                    row_inv.buffer(),
                ],
                rows * 256,
                [rows, width],
                group_size: 256
            )
            let grad_gamma = Tensor([width])
            let grad_beta = Tensor([width])
            layer_norm_params(input, upstream, grad_gamma, grad_beta, row_mean, row_inv, rows, width)
            row_mean.free()
            row_inv.free()
            return [grad_input, grad_gamma, grad_beta]
        }))
    }
    return output
}

#: Whether the layer-norm parameter gradients take the strip kernels below.
#: `JAITENSOR_LN_PARAM_STRIPS=0` keeps the one-thread-per-column kernel they
#: replaced, for an A/B in one binary. Read once.
var _ln_param_strips: int = -1

fn ln_param_strips_on() -> bool {
    if _ln_param_strips < 0 {
        let text = env("JAITENSOR_LN_PARAM_STRIPS")
        _ln_param_strips = if text is not null and text == "0" { 0 } else { 1 }
    }
    return _ln_param_strips == 1
}

#: The gamma and beta gradients of `layer_norm`: column sums over every row.
#:
#: A threadgroup to each 32 columns and each strip of rows. The strips are
#: sized so the grid holds about 256 threadgroups -- enough to fill the device
#: at a 768-wide model and a thousand rows -- but never thinner than 64 rows,
#: below which the join's extra pass costs more than the strips save. One
#: strip writes the gradients directly and skips the join.
fn layer_norm_params(
    input: Tensor,
    upstream: Tensor,
    grad_gamma: Tensor,
    grad_beta: Tensor,
    row_mean: Tensor,
    row_inv: Tensor,
    rows: int,
    width: int
) -> void {
    if not ln_param_strips_on() {
        cached_kernel("layer_norm_params_backward").dispatch_async(
            [
                input.buffer(),
                upstream.buffer(),
                grad_gamma.buffer(),
                grad_beta.buffer(),
                row_mean.buffer(),
                row_inv.buffer(),
            ],
            width,
            [rows, width]
        )
        return
    }
    let blocks = (width + 31) // 32
    let wanted = max(1, 256 // blocks)
    let strips = max(1, min(wanted, rows // 64))
    let rows_each = (rows + strips - 1) // strips
    let partials = cached_kernel("layer_norm_params_partials")
    if strips == 1 {
        partials.dispatch_async(
            [
                input.buffer(),
                upstream.buffer(),
                grad_gamma.buffer(),
                grad_beta.buffer(),
                row_mean.buffer(),
                row_inv.buffer(),
            ],
            blocks * 256,
            [rows, width, blocks, rows_each],
            group_size: 256
        )
        return
    }
    let part_gamma = Tensor([strips * width])
    let part_beta = Tensor([strips * width])
    partials.dispatch_async(
        [
            input.buffer(),
            upstream.buffer(),
            part_gamma.buffer(),
            part_beta.buffer(),
            row_mean.buffer(),
            row_inv.buffer(),
        ],
        strips * blocks * 256,
        [rows, width, blocks, rows_each],
        group_size: 256
    )
    cached_kernel("layer_norm_params_join").dispatch_async(
        [part_gamma.buffer(), part_beta.buffer(), grad_gamma.buffer(), grad_beta.buffer()],
        width,
        [strips, width]
    )
    part_gamma.free()
    part_beta.free()
}

#: Blocks per sample for the coalesced group reduction, or zero when it does
#: not apply.
#:
#: A block's threads each stay on one channel for the whole loop only if the
#: group of 256 holds a whole number of channels; nothing else makes the read
#: coalesced, so a channel count that does not divide 256 keeps the old pair
#: of kernels.
#:
#: Roughly 128 blocks in total, because the first pass reaches its ceiling
#: there and the joins do not. Swept on [64,56,56,64] with eight groups: the
#: first pass is 0.30 ms at 64 blocks and then flat at 0.23 ms from 128 all
#: the way to 1024, while the joins cost about 0.07 microseconds a block and
#: so climb from 0.018 ms to 0.088 ms across the same range. `spatial // 64`
#: stops a short feature map being cut into blocks of a row or two.
fn group_blocks(samples: int, spatial: int, channels: int) -> int {
    if channels > 256 or 256 % channels != 0 { return 0 }
    let room = spatial // 64
    if room < 1 { return 0 }
    let wanted = max(1, 128 // samples)
    return min(wanted, room)
}

#: NHWC group normalization. `groups` must divide the channel count.
pub fn group_norm(input: Tensor, gamma: Tensor, beta: Tensor, groups: int) -> Tensor {
    if input.rank() != 4 {
        throw ValueError(f"group_norm needs NHWC input, got {input.shape}")
    }
    if groups <= 0 { throw ValueError(f"group_norm groups must be positive, got {groups}") }
    let n = input.shape[0]
    let h = input.shape[1]
    let w = input.shape[2]
    let c = input.shape[3]
    if c % groups != 0 {
        throw ValueError(f"group_norm channels {c} must divide groups {groups}")
    }
    if gamma.size != c or beta.size != c {
        throw ValueError("group_norm gamma/beta must match the channel count")
    }
    let spatial = h * w
    let slots = n * groups
    let mean = Tensor([slots])
    let variance = Tensor([slots])
    let moments = cached_kernel("moments_group")
    moments.dispatch_async(
        [input.buffer(), mean.buffer(), variance.buffer()],
        slots * 256,
        [n, spatial, c, groups],
        group_size: 256
    )
    let output = Tensor(input.shape)
    let kernel = cached_kernel("group_norm_forward")
    kernel.dispatch_async(
        [
            input.buffer(),
            mean.buffer(),
            variance.buffer(),
            gamma.buffer(),
            beta.buffer(),
            output.buffer(),
        ],
        output.size,
        [output.size, spatial, c, groups]
    )
    if input.requires_grad or gamma.requires_grad or beta.requires_grad {
        output.attach_creator(GradNode([input, gamma, beta], fn(upstream: Tensor) -> list[Tensor] {
            let sum_dy = Tensor([slots])
            let sum_dy_xhat = Tensor([slots])
            let grad_gamma = Tensor([c])
            let grad_beta = Tensor([c])
            let per_sample = group_blocks(n, spatial, c)
            if per_sample > 0 {
                let blocks = n * per_sample
                let rows_each = (spatial + per_sample - 1) // per_sample
                let part_dy = Tensor([blocks * c])
                let part_dy_xhat = Tensor([blocks * c])
                cached_kernel("group_norm_grad_partials").dispatch_async([input.buffer(), mean.buffer(), variance.buffer(), upstream.buffer(), part_dy.buffer(), part_dy_xhat.buffer()], blocks * 256, [spatial, c, groups, rows_each, per_sample], group_size: 256)
                cached_kernel("group_norm_slot_join").dispatch_async([part_dy.buffer(), part_dy_xhat.buffer(), gamma.buffer(), sum_dy.buffer(), sum_dy_xhat.buffer()], slots, [c, groups, per_sample, slots])
                cached_kernel("group_norm_param_join").dispatch_async([part_dy.buffer(), part_dy_xhat.buffer(), grad_gamma.buffer(), grad_beta.buffer()], c * 256, [blocks, c], group_size: 256)
                part_dy.free()
                part_dy_xhat.free()
            } else {
                cached_kernel("group_norm_sums").dispatch_async([input.buffer(), mean.buffer(), variance.buffer(), gamma.buffer(), upstream.buffer(), sum_dy.buffer(), sum_dy_xhat.buffer()], slots * 256, [n, spatial, c, groups], group_size: 256)
                cached_kernel("channel_params_backward").dispatch_async([input.buffer(), mean.buffer(), variance.buffer(), upstream.buffer(), grad_gamma.buffer(), grad_beta.buffer()], c * 256, [n * spatial, c, spatial, groups], group_size: 256)
            }
            let grad_input = Tensor(input.shape)
            let back = cached_kernel("group_norm_backward")
            back.dispatch_async([input.buffer(), mean.buffer(), variance.buffer(), gamma.buffer(), upstream.buffer(), sum_dy.buffer(), sum_dy_xhat.buffer(), grad_input.buffer()], input.size, [input.size, spatial, c, groups])
            sum_dy.free()
            sum_dy_xhat.free()
            return [grad_input, grad_gamma, grad_beta]
        }, extras: [mean, variance]))
        return output
    }
    mean.free()
    variance.free()
    return output
}

#: Batch, layer and group normalisation, and dropout.
from ..tensor import GradNode, Tensor
from .kernels import cached_kernel, f32_bits
from .shape import reshape

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

#: Channel-last batch normalization using per-channel mean and variance of `x`.
#: Blocks for the coalesced channel reduction, or zero when it does not apply.
#:
#: Each block's threads stay on one channel apiece only if the group holds a
#: whole number of channels, and the split has to be worth making: a couple of
#: rows a block spends more on joining the partials than the first pass saves.
fn channel_chunks(nspatial: int, channels: int) -> int {
    if channels > 256 or 256 % channels != 0 { return 0 }
    let wanted = min(128, nspatial // 64)
    if wanted < 2 { return 0 }
    return wanted
}

pub fn batch_norm(input: Tensor, gamma: Tensor, beta: Tensor) -> Tensor {
    if input.rank() != 4 {
        throw ValueError(f"batch_norm needs NHWC input, got {input.shape}")
    }
    let channels = input.shape[3]
    if gamma.size != channels or beta.size != channels {
        throw ValueError("batch_norm gamma/beta must match the channel count")
    }
    let nspatial = input.shape[0] * input.shape[1] * input.shape[2]
    let mean = Tensor([channels])
    let variance = Tensor([channels])
    let chunks = channel_chunks(nspatial, channels)
    if chunks > 0 {
        let rows_each = (nspatial + chunks - 1) // chunks
        let part_total = Tensor([chunks * channels])
        let part_square = Tensor([chunks * channels])
        cached_kernel("channel_partials").dispatch_async(
            [input.buffer(), part_total.buffer(), part_square.buffer()],
            chunks * 256,
            [nspatial, channels, rows_each],
            group_size: 256
        )
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
    let output = Tensor(input.shape)
    let kernel = cached_kernel("batch_norm_forward")
    kernel.dispatch_async(
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
    if input.requires_grad or gamma.requires_grad or beta.requires_grad {
        output.attach_creator(GradNode([input, gamma, beta], fn(upstream: Tensor) -> list[Tensor] {
            let sum_dy = Tensor([channels])
            let sum_dy_xhat = Tensor([channels])
            let grad_gamma = Tensor([channels])
            let grad_beta = Tensor([channels])
            let split = channel_chunks(nspatial, channels)
            if split > 0 {
                let rows_each = (nspatial + split - 1) // split
                let part_dy = Tensor([split * channels])
                let part_dy_xhat = Tensor([split * channels])
                cached_kernel("channel_grad_partials").dispatch_async([input.buffer(), mean.buffer(), variance.buffer(), gamma.buffer(), upstream.buffer(), part_dy.buffer(), part_dy_xhat.buffer()], split * 256, [nspatial, channels, rows_each], group_size: 256)
                cached_kernel("channel_grad_join").dispatch_async([part_dy.buffer(), part_dy_xhat.buffer(), gamma.buffer(), sum_dy.buffer(), sum_dy_xhat.buffer(), grad_gamma.buffer(), grad_beta.buffer()], channels, [split, channels])
                part_dy.free()
                part_dy_xhat.free()
            } else {
                cached_kernel("norm_channel_sums").dispatch_async([input.buffer(), mean.buffer(), variance.buffer(), gamma.buffer(), upstream.buffer(), sum_dy.buffer(), sum_dy_xhat.buffer(), grad_gamma.buffer(), grad_beta.buffer()], channels * 256, [nspatial, channels], group_size: 256)
            }
            let grad_input = Tensor(input.shape)
            let back = cached_kernel("norm_channel_backward")
            back.dispatch_async([input.buffer(), mean.buffer(), variance.buffer(), gamma.buffer(), upstream.buffer(), sum_dy.buffer(), sum_dy_xhat.buffer(), grad_input.buffer()], input.size, [input.size, channels, nspatial])
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
            let params = cached_kernel("layer_norm_params_backward")
            params.dispatch_async(
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
            row_mean.free()
            row_inv.free()
            return [grad_input, grad_gamma, grad_beta]
        }))
    }
    return output
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

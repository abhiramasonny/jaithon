#: GPU tensor operations that keep their inputs and results on the device.
import std.gpu as gpu
import std.math
import .gemm as gemm
from .tensor import GradNode, Tensor, element_count
from .trace import record_dense_forward, record_dense_backward
from .ops.shaders import BINARY_FORMS, UNARY_FORMS
from .ops.kernels import cached_attn_kernel, cached_kernel, f32_bits
from .ops.kernels import reduce_leading, unary_backward
from .ops.shape import cat, copy_into, eye, flip, gather_rows, merge_heads, nchw
from .ops.shape import nhwc, one_hot, pad2d, permute, repeat, reshape, slice_flat
from .ops.shape import split_heads, squeeze, stack, transpose, tril, unsqueeze
from .ops.shape import upsample_nearest2d
from .ops.elementwise import abs, add, clamp, divide, exp, gelu, leaky_relu, log
from .ops.elementwise import maximum, minimum, multiply, neg, pow, relu, rsqrt, scale
from .ops.elementwise import scale_by, sigmoid, silu, sqrt, square, subtract, tanh, where
from .ops.reduce import add_correct_into, add_scaled_into, amax, argmax
from .ops.reduce import binary_cross_entropy, binary_cross_entropy_with_logits
from .ops.reduce import max_with_index, mean, mse, softmax, sum
from .ops.matmul import dense_affine, dense_static_backward, dense_static_forward
from .ops.matmul import linear_gelu, linear_residual, matmul, matmul_backend
from .ops.matmul import raw_matmul, raw_matmul_batched, warmup_kernels

#: NHWC max pooling.
pub fn max_pool2d(input: Tensor, kernel_size: int, stride: int = 0, padding: int = 0) -> Tensor {
    return _pool2d(input, kernel_size, stride, padding, true)
}

#: NHWC average pooling (ignores padded holes).
pub fn avg_pool2d(input: Tensor, kernel_size: int, stride: int = 0, padding: int = 0) -> Tensor {
    return _pool2d(input, kernel_size, stride, padding, false)
}

fn _pool2d(input: Tensor, kernel_size: int, stride: int, padding: int, is_max: bool) -> Tensor {
    if input.rank() != 4 { throw ValueError(f"pool2d needs NHWC input, got {input.shape}") }
    if kernel_size <= 0 { throw ValueError("kernel_size must be positive") }
    let step = if stride <= 0 { kernel_size } else { stride }
    if padding < 0 { throw ValueError("padding must be non-negative") }
    let n = input.shape[0]
    let h = input.shape[1]
    let w = input.shape[2]
    let c = input.shape[3]
    let oh = conv2d_out_dim(h, kernel_size, step, padding)
    let ow = conv2d_out_dim(w, kernel_size, step, padding)
    if oh <= 0 or ow <= 0 { throw ValueError("pool2d produced an empty spatial size") }
    let output = Tensor([n, oh, ow, c])
    let wide = c % 4 == 0
    let narrow = if is_max { "max_pool2d_nhwc" } else { "avg_pool2d_nhwc" }
    let entry = if wide { narrow + "4" } else { narrow }
    let lanes = if wide { c // 4 } else { c }
    cached_kernel(entry).dispatch_async(
        [input.buffer(), output.buffer()],
        n * oh * ow * lanes,
        [n, h, w, c, kernel_size, kernel_size, step, step, padding, padding, oh, ow]
    )
    if input.requires_grad {
        output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
            let grad = Tensor([n, h, w, c])
            let scalars = [n, h, w, c, kernel_size, kernel_size, step, step, padding, padding, oh, ow]
            if is_max {
                #: Windows that tile the input exactly cover every element
                #: once and no element twice, so a thread can take a window,
                #: scan it once and write the whole of it -- no rescanning, and
                #: nothing to add up. That is the shape pooling almost always
                #: has, 2x2 stride 2 over an even picture, and it was worth
                #: 5.6x. Anything else goes the two-pass way.
                let tiles = padding == 0 and step == kernel_size
                let covers = oh * kernel_size == h and ow * kernel_size == w
                if tiles and covers {
                    let name = if wide { "max_pool2d_backward_disjoint_nhwc4" } else { "max_pool2d_backward_disjoint_nhwc" }
                    let back = cached_kernel(name)
                    back.dispatch_async([input.buffer(), upstream.buffer(), grad.buffer()], n * oh * ow * lanes, scalars)
                } else {
                    let taps = Tensor([n, oh, ow, c])
                    let mark = if wide { "max_pool2d_taps_nhwc4" } else { "max_pool2d_taps_nhwc" }
                    let spread = if wide { "max_pool2d_backward_taps_nhwc4" } else { "max_pool2d_backward_taps_nhwc" }
                    cached_kernel(mark).dispatch_async([input.buffer(), taps.buffer()], n * oh * ow * lanes, scalars)
                    cached_kernel(spread).dispatch_async([taps.buffer(), upstream.buffer(), grad.buffer()], n * h * w * lanes, scalars)
                    taps.free()
                }
            } else {
                let tiles = padding == 0 and step == kernel_size
                let covers = oh * kernel_size == h and ow * kernel_size == w
                let whole = tiles and covers
                let entry = if whole {
                    if wide { "avg_pool2d_backward_tiled_nhwc4" } else { "avg_pool2d_backward_tiled_nhwc" }
                } elif wide {
                    "avg_pool2d_backward_nhwc4"
                } else {
                    "avg_pool2d_backward_nhwc"
                }
                let threads = if wide { grad.size // 4 } else { grad.size }
                cached_kernel(entry).dispatch_async([upstream.buffer(), grad.buffer()], threads, scalars)
            }
            return [grad]
        }))
    }
    return output
}

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

#: Output extent of one spatial axis under a convolution or pooling window.
pub fn conv2d_out_dim(size: int, kernel: int, stride: int, padding: int) -> int {
    return (size + 2 * padding - kernel) // stride + 1
}

#: NHWC 2D convolution. Weights are HWIO `[kh, kw, cin, cout]`.
#:
#: `activation` fuses an elementwise epilogue into the convolution's own
#: dispatch instead of a separate pass over the output: 0 none, 1 ReLU. A
#: fused activation needs `bias` -- the epilogue that carries it always adds
#: one, so there is nowhere to route "activation but no bias" -- which
#: `Conv2d.forward` never hits since every convolution layer owns a bias.
@trace
pub fn conv2d(
    input: Tensor,
    weights: Tensor,
    bias: Tensor? = null,
    stride: int = 1,
    padding: int = 0,
    activation: int = 0
) -> Tensor {
    if input.rank() != 4 {
        throw ValueError(f"conv2d needs NHWC input, got {input.shape}")
    }
    if weights.rank() != 4 {
        throw ValueError(f"conv2d needs HWIO weights, got {weights.shape}")
    }
    let n = input.shape[0]
    let h = input.shape[1]
    let w = input.shape[2]
    let cin = input.shape[3]
    let kh = weights.shape[0]
    let kw = weights.shape[1]
    if weights.shape[2] != cin {
        throw ValueError(f"conv2d channel mismatch: input {cin} vs weights {weights.shape[2]}")
    }
    let cout = weights.shape[3]
    if stride <= 0 { throw ValueError(f"conv2d stride must be positive, got {stride}") }
    if padding < 0 { throw ValueError(f"conv2d padding must be non-negative, got {padding}") }
    if activation < 0 or activation > 1 {
        throw ValueError(f"conv2d activation must be 0 or 1, got {activation}")
    }
    if activation != 0 and bias is null {
        throw ValueError("conv2d needs a bias to fuse an activation")
    }
    let out_h = conv2d_out_dim(h, kh, stride, padding)
    let out_w = conv2d_out_dim(w, kw, stride, padding)
    if out_h <= 0 or out_w <= 0 {
        throw ValueError(f"conv2d output spatial size is empty: {out_h}x{out_w}")
    }
    if bias is not null and bias.size != cout {
        throw ValueError(f"conv2d bias has {bias.size} values, expected {cout}")
    }
    let output = Tensor([n, out_h, out_w, cout])
    let by_im2col = fn() -> void {
        let k = kh * kw * cin
        let rows = n * out_h * out_w
        let col = Tensor([rows, k])
        let kernel = cached_kernel("im2col_nhwc")
        kernel.dispatch_async([input.buffer(), col.buffer()], rows * k, [n, h, w, cin, kh, kw, stride, stride, padding, padding])
        let flat_w = Tensor([k, cout], storage: weights.buffer().view(0, k * cout))
        let flat_out = Tensor([rows, cout], storage: output.buffer())
        let epilogue = if bias is null {
            0
        } elif activation == 1 {
            2
        } else {
            1
        }
        raw_matmul(col, flat_w, bias: bias, epilogue: epilogue, out: flat_out)
        col.free()
    }
    try {
        gpu.conv2d_async(input.buffer(), weights.buffer(), output.buffer(), n, h, w, cin, cout, kh, kw, stride_h: stride, stride_w: stride, pad_h: padding, pad_w: padding, bias: if bias is null {
            null
        } else {
            bias.buffer()
        }, activation: activation)
    } catch _error: RuntimeError {
        by_im2col()
    }
    attach_conv_grad(input, weights, bias, output, stride, padding, activation)
    return output
}

#: NCHW 2D convolution. Weights are OIHW `[cout, cin, kh, kw]`.
#:
#: The layout every exported model file uses, run without being rearranged
#: into the one `conv2d` wants. MPS takes NCHW natively, so the transposes
#: that used to bracket each call -- two over the activations and one over the
#: weights -- were paying for nothing: on YOLOv8n they were most of the time
#: the convolution appeared to cost.
#:
#: Inference only, in the sense that nothing here attaches a gradient: this is
#: for running an imported graph forward. Train with `conv2d`.
#:
#: `activation` fuses an epilogue into the same dispatch: 0 none, 1 ReLU,
#: 2 SiLU, and as in `conv2d` a fused activation needs a bias.
@trace
pub fn conv2d_nchw(
    input: Tensor,
    weights: Tensor,
    bias: Tensor? = null,
    stride: int = 1,
    padding: int = 0,
    activation: int = 0
) -> Tensor {
    if input.rank() != 4 {
        throw ValueError(f"conv2d_nchw needs NCHW input, got {input.shape}")
    }
    if weights.rank() != 4 {
        throw ValueError(f"conv2d_nchw needs OIHW weights, got {weights.shape}")
    }
    let n = input.shape[0]
    let cin = input.shape[1]
    let h = input.shape[2]
    let w = input.shape[3]
    let cout = weights.shape[0]
    let kh = weights.shape[2]
    let kw = weights.shape[3]
    if weights.shape[1] != cin {
        throw ValueError(f"conv2d_nchw channel mismatch: input {cin} vs weights {weights.shape[1]}")
    }
    if stride <= 0 { throw ValueError(f"conv2d_nchw stride must be positive, got {stride}") }
    if padding < 0 {
        throw ValueError(f"conv2d_nchw padding must be non-negative, got {padding}")
    }
    if activation < 0 or activation > 2 {
        throw ValueError(f"conv2d_nchw activation must be 0, 1 or 2, got {activation}")
    }
    if activation != 0 and bias is null {
        throw ValueError("conv2d_nchw needs a bias to fuse an activation")
    }
    if bias is not null and bias.size != cout {
        throw ValueError(f"conv2d_nchw bias has {bias.size} values, expected {cout}")
    }
    let out_h = conv2d_out_dim(h, kh, stride, padding)
    let out_w = conv2d_out_dim(w, kw, stride, padding)
    if out_h <= 0 or out_w <= 0 {
        throw ValueError(f"conv2d_nchw output spatial size is empty: {out_h}x{out_w}")
    }
    let output = Tensor([n, cout, out_h, out_w])
    try {
        gpu.conv2d_async(input.buffer(), weights.buffer(), output.buffer(), n, h, w, cin, cout, kh, kw, stride_h: stride, stride_w: stride, pad_h: padding, pad_w: padding, bias: if bias is null {
            null
        } else {
            bias.buffer()
        }, activation: activation, layout: 1)
    } catch _error: RuntimeError {
        #: A shape MPS will not take. Rather than keep a second NCHW kernel
        #: alive for it, go the long way round through `conv2d`, which has its
        #: own im2col fallback -- correct, rare, and slow only when it fires.
        output.free()
        let nhwc_in = nhwc(input)
        #: OIHW to HWIO. `permute` has no kernel for `[2, 3, 1, 0]`, but the
        #: swap is two it does have: `nhwc` puts the input channel last,
        #: leaving the output channel as the only axis out of place, and a
        #: matrix transpose across the flattened rest moves it to the end.
        #:
        #: `reshape` shares its argument's storage, so only the two tensors
        #: that own a buffer -- the `nhwc` and the `transpose` -- are freed,
        #: and each of those only once the last view onto it is finished with.
        let ohwi = nhwc(weights)
        let columns = transpose(reshape(ohwi, [cout, kh * kw * cin]))
        ohwi.free()
        let hwio = reshape(columns, [kh, kw, cin, cout])
        let done = conv2d(nhwc_in, hwio, bias: bias, stride: stride, padding: padding, activation: if activation == 2 {
            0
        } else {
            activation
        })
        nhwc_in.free()
        columns.free()
        let back = nchw(done)
        done.free()
        if activation != 2 { return back }
        let fused = silu(back)
        back.free()
        return fused
    }
    return output
}

#: Backward for `conv2d`, as im2col plus two products.
#:
#: The columns are rebuilt here rather than kept from the forward pass: a
#: `[batch * out_h * out_w, kh * kw * cin]` matrix is tens of megabytes for an
#: ordinary layer, and the forward may not have built one at all when the
#: native convolution took the work.
fn attach_conv_grad(
    input: Tensor,
    weights: Tensor,
    bias: Tensor?,
    output: Tensor,
    stride: int,
    padding: int,
    activation: int
) -> void {
    #: Only the gradients something downstream will read.
    #:
    #: The first convolution of a network is fed the batch itself, which is a
    #: leaf and wants no gradient -- and the gradient with respect to the input
    #: costs a whole second convolution and an activation-sized allocation to
    #: produce. Asking for it anyway was about a sixth of what the backward
    #: pass of a two-layer network did.
    let wants_input = input.requires_grad
    let wants_weights = weights.requires_grad
    let wants_bias = bias is not null and bias.requires_grad
    if not (wants_input or wants_weights or wants_bias) { return }
    var inputs: list[Tensor] = []
    if wants_input { inputs.push(input) }
    if wants_weights { inputs.push(weights) }
    if wants_bias { inputs.push(bias) }
    let n = input.shape[0]
    let h = input.shape[1]
    let w = input.shape[2]
    let cin = input.shape[3]
    let kh = weights.shape[0]
    let kw = weights.shape[1]
    let cout = weights.shape[3]
    let out_h = output.shape[1]
    let out_w = output.shape[2]
    output.attach_creator(GradNode(inputs, fn(raw_upstream: Tensor) -> list[Tensor] {
        #: `output` already carries the fused activation, so its own gradient
        #: is masked here before any of the convolution math below sees it --
        #: the same rewrite `dense_affine` does for its epilogue, and the
        #: reason `conv2d` only fuses an activation whose backward needs
        #: nothing but the (already-available) forward output.
        let upstream = if activation == 1 { unary_backward(output, raw_upstream, 1) } else { raw_upstream }
        let k = kh * kw * cin
        let rows = n * out_h * out_w
        let flat_up = Tensor([rows, cout], storage: upstream.buffer().view(0, rows * cout))
        var grad_weights: Tensor? = null
        var grad_input: Tensor? = null
        if wants_weights { grad_weights = Tensor([kh, kw, cin, cout]) }
        if wants_input { grad_input = Tensor([n, h, w, cin]) }

        #: The device's own convolution gradients when it has them. The im2col
        #: pair below computes the same two tensors, and does it by building a
        #: `[n * out_h * out_w, kh * kw * cin]` matrix twice per step: a
        #: hundred and fifty megabytes allocated, filled and dropped on every
        #: batch of a sixty-four-channel layer. Either may be declined on its
        #: own -- a shape MPS will not take, or no device at all -- so each is
        #: asked for separately and each has its own fallback.
        #: Each gradient is asked for on its own: either may be declined for
        #: a shape MPS will not take, and either may not be wanted at all.
        var have_weights_grad = not wants_weights
        var have_input_grad = not wants_input
        try {
            if wants_weights {
                have_weights_grad = gpu.conv2d_weights_grad_async(
                    upstream.buffer(),
                    input.buffer(),
                    grad_weights.buffer(),
                    n,
                    h,
                    w,
                    cin,
                    cout,
                    kh,
                    kw,
                    stride_h: stride,
                    stride_w: stride,
                    pad_h: padding,
                    pad_w: padding
                )
            }
            if wants_input {
                have_input_grad = gpu.conv2d_data_grad_async(
                    upstream.buffer(),
                    weights.buffer(),
                    grad_input.buffer(),
                    n,
                    h,
                    w,
                    cin,
                    cout,
                    kh,
                    kw,
                    stride_h: stride,
                    stride_w: stride,
                    pad_h: padding,
                    pad_w: padding
                )
            }
        } catch _error: RuntimeError {
            have_weights_grad = not wants_weights
            have_input_grad = not wants_input
        }

        #: The im2col pair computes the same two tensors when the device
        #: declines. It builds an `[n * out_h * out_w, kh * kw * cin]` matrix
        #: to do it -- a hundred and fifty megabytes for a sixty-four-channel
        #: layer -- which is why it is the fallback and not the path.
        if not have_weights_grad {
            let col = Tensor([rows, k])
            let im2col = cached_kernel("im2col_nhwc")
            im2col.dispatch_async(
                [input.buffer(), col.buffer()],
                rows * k,
                [n, h, w, cin, kh, kw, stride, stride, padding, padding]
            )
            let flat_gw = Tensor([k, cout], storage: grad_weights.buffer())
            raw_matmul(col, flat_up, trans_a: true, out: flat_gw)
            col.free()
        }
        if not have_input_grad {
            let flat_w = Tensor([k, cout], storage: weights.buffer().view(0, k * cout))
            let grad_col = raw_matmul(flat_up, flat_w, trans_b: true)
            let col2im = cached_kernel("col2im_nhwc")
            col2im.dispatch_async(
                [grad_col.buffer(), grad_input.buffer()],
                grad_input.size,
                [n, h, w, cin, kh, kw, stride, stride, padding, padding]
            )
            grad_col.free()
        }

        #: Same order as `inputs`, which listed only what wanted a gradient.
        var grads: list[Tensor] = []
        if wants_input { grads.push(grad_input) }
        if wants_weights { grads.push(grad_weights) }
        if wants_bias { grads.push(reduce_leading(flat_up, [cout])) }
        if activation == 1 { upstream.free() }
        return grads
    }))
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

#: Scaled dot-product attention for rank-two `[seq, dim]` tensors.
pub fn scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor {
    if query.rank() != 2 or key.rank() != 2 or value.rank() != 2 {
        throw ValueError("scaled_dot_product_attention needs rank-two Q, K, V")
    }
    if query.shape[1] != key.shape[1] {
        throw ValueError("query/key depth mismatch")
    }
    if key.shape[0] != value.shape[0] {
        throw ValueError("key/value sequence mismatch")
    }
    let factor = 1.0 / math.sqrt(float(query.shape[1]))
    if query.requires_grad or key.requires_grad or value.requires_grad {
        return matmul(softmax(scale(matmul(query, key, trans_b: true), factor)), value)
    }
    let scores = raw_matmul(query, key, trans_b: true)
    let scaled = scale(scores, factor)
    scores.free()
    let probs = softmax(scaled)
    scaled.free()
    let output = raw_matmul(probs, value)
    probs.free()
    return output
}

#: Per-head scaled-dot-product attention on packed `[seq, dim]` Q/K/V.
#:
#: The fused device path is a Metal prefill kernel: packed `[seq, heads, hd]`
#: stays packed, Q/K/V stream through half SRAM, and QK/PV use simdgroup MMA.
#: MPSGraph SDPA is the fallback when the kernel cannot run. Last resort is
#: GEMM. Scale is `1 / sqrt(head_dim)`.
pub fn multi_head_attention(query: Tensor, key: Tensor, value: Tensor, heads: int) -> Tensor {
    if heads == 1 { return scaled_dot_product_attention(query, key, value) }
    if query.rank() != 2 or key.rank() != 2 or value.rank() != 2 {
        throw ValueError("multi_head_attention needs rank-two Q, K, V")
    }
    if heads <= 0 {
        throw ValueError(f"multi_head_attention needs a positive head count, got {heads}")
    }
    if query.shape != key.shape or query.shape != value.shape {
        throw ValueError(
            f"multi_head_attention Q/K/V shape mismatch: {query.shape} {key.shape} {value.shape}"
        )
    }
    let seq = query.shape[0]
    let dim = query.shape[1]
    if dim % heads != 0 {
        throw ValueError(f"multi_head_attention dim {dim} must divide heads {heads}")
    }
    let hd = dim // heads
    let factor = 1.0 / math.sqrt(float(hd))
    if query.requires_grad or key.requires_grad or value.requires_grad {
        if mha_mma_ok(seq, hd) { return mha_by_mma(query, key, value, heads, seq, hd, factor) }
        return mha_by_ops(query, key, value, heads, seq, hd, factor)
    }
    let output = Tensor(query.shape)
    if gpu.mha_async(
        query.buffer(),
        key.buffer(),
        value.buffer(),
        output.buffer(),
        seq,
        heads,
        hd,
        factor
    ) {
        return output
    }
    if seq >= 16 and hd >= 32 and hd <= 64 {
        let q_tiles = (seq + 15) // 16
        let kernel = cached_attn_kernel("flash_mha_wide")
        kernel.dispatch_async(
            [query.buffer(), key.buffer(), value.buffer(), output.buffer()],
            q_tiles * heads * 256,
            [seq, heads, hd, f32_bits(factor)],
            group_size: 256
        )
        return output
    }
    if seq >= 16 and hd <= 64 {
        let q_tiles = (seq + 15) // 16
        let kernel = cached_attn_kernel("flash_mha")
        kernel.dispatch_async(
            [query.buffer(), key.buffer(), value.buffer(), output.buffer()],
            q_tiles * heads * 256,
            [seq, heads, hd, f32_bits(factor)],
            group_size: 256
        )
        return output
    }
    output.free()
    return mha_by_gemm(query, key, value, heads, seq, hd, factor)
}

#: Whether the batched simdgroup-matrix attention kernels can take this shape.
#:
#: They tile in 8x8 fragments with no masking, so the sequence has to fill
#: whole 64-row patches, and the head dimension has to be exactly the 64 the
#: 2x2 of simdgroups covers. Everything else keeps `mha_by_ops`, which has no
#: shape restriction at all.
fn mha_mma_ok(seq: int, hd: int) -> bool {
    return hd == 64 and seq % 64 == 0
}

#: Multi-head attention as six batched dispatches, forward and backward.
#:
#: `mha_by_ops` below is the same arithmetic written as ops, and it is what
#: runs whenever `mha_mma_ok` says no. What this saves over it is not a
#: different algorithm but a different shape of work. That version issues one
#: GEMM per head, and the route the GEMM router picks for the two attention
#: shapes -- [seq, 64] against [64, seq], and [seq, seq] against [seq, 64] --
#: reaches 2.3 and 2.8 TFLOP/s where a square 1024 product reaches 8.3 on the
#: same device. At seq 1024, dim 1024 and 16 heads the whole forward and
#: backward took 8.39 ms that way, of which the two forward loops alone were
#: 0.93 and 0.76. The six dispatches here measure 2.25 together, dropping
#: `split_heads` and `merge_heads` -- the kernels read the packed layout
#: directly -- takes another 0.31 off, and the pass now measures 3.15 against
#: torch's 4.36 for the same shape, both timed with the graph rebuilt each
#: iteration because backward consumes it.
#:
#: The score matrix is still written out, unlike the fused inference kernel,
#: because the backward pass has to read it. Nothing else is materialised.
fn mha_by_mma(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    heads: int,
    seq: int,
    hd: int,
    factor: float
) -> Tensor {
    let tiles = seq // 64
    let probs = Tensor([heads * seq, seq])
    cached_attn_kernel("mha_scores_mma").dispatch_async(
        [query.buffer(), key.buffer(), probs.buffer()],
        tiles * tiles * heads * 64,
        [seq, heads, hd, tiles],
        group_size: 64
    )
    cached_attn_kernel("mha_softmax_scaled").dispatch_async(
        [probs.buffer()],
        heads * seq * 256,
        [heads * seq, seq, f32_bits(factor)],
        group_size: 256
    )
    let output = Tensor([seq, heads * hd])
    cached_attn_kernel("mha_context_mma").dispatch_async(
        [probs.buffer(), value.buffer(), output.buffer()],
        tiles * heads * 64,
        [seq, heads, hd],
        group_size: 64
    )
    if query.requires_grad or key.requires_grad or value.requires_grad {
        output.attach_creator(GradNode([query, key, value], fn(upstream: Tensor) -> list[Tensor] {
            return mha_mma_grads(upstream, query, key, value, probs, heads, seq, hd, factor)
        }, extras: [probs]))
    }
    return output
}

#: Gradients of `mha_by_mma`, in four dispatches around the softmax backward.
#:
#: The four products are the ones `matmul`'s backward would have chosen, in
#: the same operand order; only the batching and the layout differ. Each
#: gradient comes out packed like the input it belongs to, so nothing has to
#: be un-split afterwards. `grad_probs` carries d(probs) into the softmax
#: backward and comes back holding d(scores), which saves a second
#: sixty-seven-megabyte buffer at seq 1024.
fn mha_mma_grads(
    upstream: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    probs: Tensor,
    heads: int,
    seq: int,
    hd: int,
    factor: float
) -> list[Tensor] {
    let tiles = seq // 64
    let dim = heads * hd
    let grad_q = Tensor([seq, dim])
    let grad_k = Tensor([seq, dim])
    let grad_v = Tensor([seq, dim])
    let grad_probs = Tensor([heads * seq, seq])
    cached_attn_kernel("mha_scores_mma").dispatch_async(
        [upstream.buffer(), value.buffer(), grad_probs.buffer()],
        tiles * tiles * heads * 64,
        [seq, heads, hd, tiles],
        group_size: 64
    )
    cached_attn_kernel("mha_transposed_mma").dispatch_async(
        [probs.buffer(), upstream.buffer(), grad_v.buffer()],
        tiles * heads * 64,
        [seq, heads, hd],
        group_size: 64
    )
    cached_attn_kernel("mha_softmax_backward_scaled").dispatch_async(
        [probs.buffer(), grad_probs.buffer()],
        heads * seq * 256,
        [heads * seq, seq, f32_bits(factor)],
        group_size: 256
    )
    cached_attn_kernel("mha_context_mma").dispatch_async(
        [grad_probs.buffer(), key.buffer(), grad_q.buffer()],
        tiles * heads * 64,
        [seq, heads, hd],
        group_size: 64
    )
    cached_attn_kernel("mha_transposed_mma").dispatch_async(
        [grad_probs.buffer(), query.buffer(), grad_k.buffer()],
        tiles * heads * 64,
        [seq, heads, hd],
        group_size: 64
    )
    grad_probs.free()
    return [grad_q, grad_k, grad_v]
}

#: Multi-head attention built from differentiable ops.
#:
#: The fused kernels above never materialise the attention probabilities, so
#: there is nothing for a backward pass to read; a graph that needs gradients
#: gets this composition instead. Every head's scores live in one
#: `[heads * seq, seq]` tensor and every head's context in one
#: `[heads * seq, hd]` tensor, so the only per-head work is the four GEMMs
#: that must be per-head; the softmax, the scale and the packing are one
#: dispatch each across all of them.
#:
#: The composition this replaced sliced Q, K and V per head with `slice_flat`
#: and reassembled the result with `cat`. Forward that is nearly free -- a
#: slice is a view -- but `slice_flat`'s backward allocates a zero tensor the
#: size of the whole packed activation and copies one head into it, and the
#: three slices per head over sixteen heads then arrive at `qs`, `ks` and `vs`
#: as forty-eight full-size gradients to be summed. At seq 1024 that pattern
#: alone measured 1.59 ms: forty-eight four-megabyte zero fills and
#: forty-eight adds, every one of which adds a head to forty-seven zeros. The
#: GEMMs below write into a view of the single gradient buffer instead, so
#: each element is written exactly once. Adding zero is exact, so the answer
#: is unchanged bit for bit.
fn mha_by_ops(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    heads: int,
    seq: int,
    hd: int,
    factor: float
) -> Tensor {
    let qs = split_heads(query, heads)
    let ks = split_heads(key, heads)
    let vs = split_heads(value, heads)
    let probs = Tensor([heads * seq, seq])
    raw_matmul_batched(qs, ks, probs, heads, seq, hd, seq, seq * hd, seq * hd, trans_b: true)
    cached_attn_kernel("mha_softmax_scaled").dispatch_async(
        [probs.buffer()],
        heads * seq * 256,
        [heads * seq, seq, f32_bits(factor)],
        group_size: 256
    )
    let context = Tensor([heads * seq, hd])
    raw_matmul_batched(probs, vs, context, heads, seq, seq, hd, seq * seq, seq * hd)
    if qs.requires_grad or ks.requires_grad or vs.requires_grad {
        context.attach_creator(GradNode([qs, ks, vs], fn(upstream: Tensor) -> list[Tensor] {
            return mha_grads(upstream, qs, ks, vs, probs, heads, seq, hd, factor)
        }, extras: [probs]))
    }
    return merge_heads(context, heads, seq)
}

#: Gradients of `mha_by_ops` with respect to the split-head Q, K and V.
#:
#: The four products per head are the ones `matmul`'s own backward would have
#: chosen for the same two products, in the same operand order, so the
#: arithmetic is identical; what differs is that each writes straight into its
#: slice of one gradient buffer. `grad_probs` carries d(probs) into the
#: softmax backward and comes back holding d(scores), which saves a second
#: sixty-seven-megabyte buffer at seq 1024.
fn mha_grads(
    upstream: Tensor,
    qs: Tensor,
    ks: Tensor,
    vs: Tensor,
    probs: Tensor,
    heads: int,
    seq: int,
    hd: int,
    factor: float
) -> list[Tensor] {
    let grad_q = Tensor([heads * seq, hd])
    let grad_k = Tensor([heads * seq, hd])
    let grad_v = Tensor([heads * seq, hd])
    let grad_probs = Tensor([heads * seq, seq])
    raw_matmul_batched(
        upstream,
        vs,
        grad_probs,
        heads,
        seq,
        hd,
        seq,
        seq * hd,
        seq * hd,
        trans_b: true
    )
    raw_matmul_batched(
        probs,
        upstream,
        grad_v,
        heads,
        seq,
        seq,
        hd,
        seq * seq,
        seq * hd,
        trans_a: true
    )
    cached_attn_kernel("mha_softmax_backward_scaled").dispatch_async(
        [probs.buffer(), grad_probs.buffer()],
        heads * seq * 256,
        [heads * seq, seq, f32_bits(factor)],
        group_size: 256
    )
    raw_matmul_batched(grad_probs, ks, grad_q, heads, seq, seq, hd, seq * seq, seq * hd)
    raw_matmul_batched(
        grad_probs,
        qs,
        grad_k,
        heads,
        seq,
        seq,
        hd,
        seq * seq,
        seq * hd,
        trans_a: true
    )
    grad_probs.free()
    return [grad_q, grad_k, grad_v]
}

fn mha_by_gemm(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    heads: int,
    seq: int,
    hd: int,
    factor: float
) -> Tensor {
    let qs = split_heads(query, heads)
    let ks = split_heads(key, heads)
    let vs = split_heads(value, heads)
    var parts: list[Tensor] = []
    for h in 0..heads {
        let qh = qs.view(h * seq * hd, [seq, hd])
        let kh = ks.view(h * seq * hd, [seq, hd])
        let vh = vs.view(h * seq * hd, [seq, hd])
        let scores = raw_matmul(qh, kh, trans_b: true)
        let scaled = scale(scores, factor)
        scores.free()
        let probs = softmax(scaled)
        scaled.free()
        parts.push(raw_matmul(probs, vh))
        qh.free()
        kh.free()
        vh.free()
        probs.free()
    }
    qs.free()
    ks.free()
    vs.free()
    let packed = cat(parts)
    for part in parts { part.free() }
    let output = merge_heads(packed, heads, seq)
    packed.free()
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

export { cached_kernel }

export { cat, copy_into, eye, flip, gather_rows, merge_heads, nchw, nhwc, one_hot, pad2d, permute, repeat, reshape, slice_flat, split_heads, squeeze, stack, transpose, tril, unsqueeze, upsample_nearest2d }

export { abs, add, clamp, divide, exp, gelu, leaky_relu, log, maximum, minimum, multiply, neg, pow, relu, rsqrt, scale, scale_by, sigmoid, silu, sqrt, square, subtract, tanh, where }

export { add_correct_into, add_scaled_into, amax, argmax, binary_cross_entropy, binary_cross_entropy_with_logits, max_with_index, mean, mse, softmax, sum }

export { dense_affine, dense_static_backward, dense_static_forward, linear_gelu, linear_residual, matmul, matmul_backend, warmup_kernels }

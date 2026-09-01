#: Convolution and pooling, and the im2col fallbacks their gradients take
#: when the device declines a shape.
import std.gpu as gpu
from ..tensor import GradNode, Tensor
from .elementwise import silu
from .kernels import cached_kernel, reduce_leading, unary_backward
from .matmul import raw_matmul
from .shape import nchw, nhwc, reshape, transpose

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

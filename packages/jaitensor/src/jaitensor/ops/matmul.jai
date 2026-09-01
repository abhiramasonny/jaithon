#: Matrix products, and the dense and fused linear layers built on them.
import std.gpu as gpu
import ..gemm as gemm
from ..tensor import GradNode, Tensor
from ..trace import record_dense_forward, record_dense_backward
from .elementwise import add, gelu
from .kernels import cached_kernel, reduce_leading, unary_backward

#: Report the kernel family selected for a matrix shape.
#:
#: Side-effect free: it reports the plan a shape already has, or the static
#: heuristic when it has never run, so profilers and benchmark tools can
#: explain a dispatch cliff without changing one.
pub fn matmul_backend(
    rows: int,
    inner: int,
    columns: int,
    trans_a: bool = false,
    trans_b: bool = false,
    epilogue: int = 0
) -> str {
    return gemm.planned_backend(rows, inner, columns, trans_a: trans_a, trans_b: trans_b)
}

pub fn raw_matmul(
    a: Tensor,
    b: Tensor,
    trans_a: bool = false,
    trans_b: bool = false,
    bias: Tensor? = null,
    epilogue: int = 0,
    out: Tensor? = null
) -> Tensor {
    let rows = if trans_a { a.shape[1] } else { a.shape[0] }
    let inner = if trans_a { a.shape[0] } else { a.shape[1] }
    let inner_b = if trans_b { b.shape[1] } else { b.shape[0] }
    let columns = if trans_b { b.shape[0] } else { b.shape[1] }
    if inner != inner_b {
        throw ValueError(f"matmul inner dimensions differ: {a.shape} and {b.shape}")
    }
    let output = if out is not null {
        if out.shape != [rows, columns] {
            throw ValueError(f"matmul out shape {out.shape} does not match [{rows}, {columns}]")
        }
        out
    } else {
        Tensor([rows, columns])
    }
    let bias_buffer = if bias is not null { bias.buffer() } else { output.buffer() }
    gemm.dispatch(
        a.buffer(),
        b.buffer(),
        output.buffer(),
        bias_buffer,
        rows,
        inner,
        columns,
        trans_a: trans_a,
        trans_b: trans_b,
        epilogue: epilogue
    )
    return output
}

#: `batch` matmuls of one shape, stacked along the leading axis, in one dispatch.
#:
#: `a`, `b` and `out` each hold `batch` matrices back to back; the two operand
#: strides are given because an operand may be shared across the batch, which a
#: stride of zero says. `out` is always `batch` matrices of `rows * columns`.
pub fn raw_matmul_batched(
    a: Tensor,
    b: Tensor,
    out: Tensor,
    batch: int,
    rows: int,
    inner: int,
    columns: int,
    stride_a: int,
    stride_b: int,
    trans_a: bool = false,
    trans_b: bool = false
) -> void {
    if out.shape != [batch * rows, columns] {
        throw ValueError(
            f"batched matmul out shape {out.shape} does not match [{batch * rows}, {columns}]"
        )
    }
    gemm.dispatch_batched(
        a.buffer(),
        b.buffer(),
        out.buffer(),
        out.buffer(),
        rows,
        inner,
        columns,
        batch,
        stride_a,
        stride_b,
        trans_a: trans_a,
        trans_b: trans_b
    )
}

#: Matrix product of two rank-two tensors.
#:
#: `trans_a` / `trans_b` read `a` or `b` as if transposed without copying.
#: Gradient graphs are only attached for the untransposed product.
pub fn matmul(
    a: Tensor,
    b: Tensor,
    trans_a: bool = false,
    trans_b: bool = false,
    out: Tensor? = null
) -> Tensor {
    if a.rank() != 2 or b.rank() != 2 {
        throw ValueError(f"matmul needs rank-two tensors, got {a.shape} and {b.shape}")
    }
    let a_inner = if trans_a { a.shape[0] } else { a.shape[1] }
    let b_inner = if trans_b { b.shape[1] } else { b.shape[0] }
    if a_inner != b_inner {
        throw ValueError(f"matmul inner dimensions differ: {a.shape} and {b.shape}")
    }
    let output = raw_matmul(a, b, trans_a: trans_a, trans_b: trans_b, out: out)
    if out is null and (a.requires_grad or b.requires_grad) {
        output.attach_creator(GradNode([a, b], fn(upstream: Tensor) -> list[Tensor] {
            if trans_a and trans_b {
                return [raw_matmul(b, upstream, trans_a: true, trans_b: true), raw_matmul(upstream, a, trans_a: true, trans_b: true)]
            }
            if trans_a {
                return [raw_matmul(b, upstream, trans_b: true), raw_matmul(a, upstream)]
            }
            if trans_b {
                return [raw_matmul(upstream, b), raw_matmul(upstream, a, trans_a: true)]
            }
            return [raw_matmul(upstream, b, trans_b: true), raw_matmul(a, upstream, trans_a: true)]
        }))
    }
    return output
}

#: Affine transform `x @ weights + bias` with an optional fused activation.
#:
#: `activation` matches `Dense.activation_code`: 0 linear, 1 ReLU, 2 sigmoid,
#: 3 tanh. Backward skips GEMMs for inputs that do not require a gradient,
#: so a first-layer feature batch never materializes `dX`.
pub fn dense_affine(x: Tensor, weights: Tensor, bias: Tensor, activation: int) -> Tensor {
    if activation < 0 or activation > 3 {
        throw ValueError(f"dense_affine activation must be 0..3, got {activation}")
    }
    record_dense_forward(x.shape, weights.shape)
    let output = raw_matmul(x, weights, bias: bias, epilogue: activation + 1)
    let need_x = x.requires_grad
    let need_w = weights.requires_grad
    let need_b = bias.requires_grad
    if need_x or need_w or need_b {
        var inputs: list[Tensor] = []
        if need_x { inputs.push(x) }
        if need_w { inputs.push(weights) }
        if need_b { inputs.push(bias) }
        output.attach_creator(GradNode(inputs, fn(upstream: Tensor) -> list[Tensor] {
            let affine_grad = if activation == 0 {
                upstream
            } else {
                unary_backward(output, upstream, activation)
            }
            var grads: list[Tensor] = []
            if need_x {
                grads.push(raw_matmul(affine_grad, weights, trans_b: true))
            }
            if need_w {
                grads.push(raw_matmul(x, affine_grad, trans_a: true, out: weights.grad))
            }
            if need_b {
                grads.push(reduce_leading(affine_grad, bias.shape, out: bias.grad))
            }
            if activation != 0 {
                affine_grad.free()
            }
            return grads
        }))
    }
    return output
}

#: `x @ weights + bias` with GELU applied on the way out.
#:
#: One kernel rather than two. The separate form writes the whole product to
#: memory and reads it straight back to shape it -- for a transformer's inner
#: layer that is a 256 by 3072 tensor written and read for nothing, and a
#: second dispatch behind it.
#:
#: Fused only when nothing here wants a gradient. GELU's derivative is a
#: function of what went *into* it, and a fused epilogue keeps only what came
#: out, so training takes the two-step path and keeps the intermediate it needs.
pub fn linear_gelu(x: Tensor, weights: Tensor, bias: Tensor) -> Tensor {
    if x.requires_grad or weights.requires_grad or bias.requires_grad {
        return gelu(dense_affine(x, weights, bias, 0))
    }
    return raw_matmul(x, weights, bias: bias, epilogue: 5)
}

#: `residual + x @ weights`, in one kernel.
#:
#: The shape a transformer block ends each half with. Written apart it is a
#: product into memory and then a pass that reads it, reads the residual and
#: writes a third tensor -- three times the traffic of the add for an operation
#: that is only an add, and a dispatch of its own behind the product.
#:
#: `residual` has to be the whole output's shape, not a row of it; the epilogue
#: that adds a vector is the one `dense_affine` uses. Fused only when nothing
#: wants a gradient, for the same reason `linear_gelu` is.
pub fn linear_residual(x: Tensor, weights: Tensor, residual: Tensor) -> Tensor {
    let rows = x.shape[0]
    let columns = weights.shape[weights.rank() - 1]
    if residual.rank() != 2 or residual.shape[0] != rows or residual.shape[1] != columns {
        throw ValueError(
            f"linear_residual needs a residual of [{rows}, {columns}], got {residual.shape}"
        )
    }
    if x.requires_grad or weights.requires_grad or residual.requires_grad {
        return add(residual, matmul(x, weights))
    }
    return raw_matmul(x, weights, bias: residual, epilogue: 6)
}

#: Dense forward without autograd graph construction.
@trace
pub fn dense_static_forward(x: Tensor, weights: Tensor, bias: Tensor, activation: int) -> Tensor {
    if activation < 0 or activation > 3 {
        throw ValueError(f"dense_affine activation must be 0..3, got {activation}")
    }
    record_dense_forward(x.shape, weights.shape)
    return raw_matmul(x, weights, bias: bias, epilogue: activation + 1)
}

#: Dense backward into parameter `.grad` buffers; returns dInput (caller frees).
@trace
pub fn dense_static_backward(
    x: Tensor,
    weights: Tensor,
    bias: Tensor,
    output: Tensor,
    upstream: Tensor,
    activation: int
) -> Tensor {
    record_dense_backward(x.shape, weights.shape)
    let affine_grad = if activation == 0 {
        upstream
    } else {
        unary_backward(output, upstream, activation)
    }
    raw_matmul(x, affine_grad, trans_a: true, out: weights.grad)
    reduce_leading(affine_grad, bias.shape, out: bias.grad)
    let d_input = raw_matmul(affine_grad, weights, trans_b: true)
    if activation != 0 {
        affine_grad.free()
    }
    return d_input
}

var _warmed = false

#: Compile training kernels and run a tiny dispatch before the first timed epoch.
pub fn warmup_kernels() -> void {
    if _warmed { return }
    gemm.warmup()
    cached_kernel("unary_backward_op")
    cached_kernel("unary_backward_relu4")
    cached_kernel("reduce_leading")
    cached_kernel("reduce_leading4")
    cached_kernel("reduce_leading_split")
    cached_kernel("add_scaled_into")
    cached_kernel("add_correct_into")
    cached_kernel("im2col_nhwc")
    cached_kernel("batch_norm_forward")
    cached_kernel("moments_channel")
    cached_kernel("layer_norm_forward")
    cached_kernel("gather_rows")
    let wide = Tensor([64, 1024])
    let wide_w = Tensor([1024, 64])
    let wide_out = raw_matmul(wide, wide_w)
    let wide_k = Tensor([1024, 64])
    let wide_tn = raw_matmul(wide_k, wide_k, trans_a: true)
    let wide_nt = raw_matmul(wide, wide, trans_b: true)
    gpu.synchronize()
    wide_nt.free()
    wide_tn.free()
    wide_k.free()
    wide_out.free()
    wide.free()
    wide_w.free()
    let mid = Tensor([256, 256])
    let mid_out = raw_matmul(mid, mid, bias: mid, epilogue: 2)
    let mid_tn = raw_matmul(mid, mid, trans_a: true)
    let mid_nt = raw_matmul(mid, mid, trans_b: true)
    gpu.synchronize()
    mid_nt.free()
    mid_tn.free()
    mid_out.free()
    mid.free()
    let x = Tensor([512, 784])
    let weights = Tensor([784, 256])
    let bias = Tensor.zeros([256])
    let hidden = raw_matmul(x, weights, bias: bias, epilogue: 2)
    let grad_x = raw_matmul(hidden, weights, trans_b: true)
    let grad_w = raw_matmul(x, hidden, trans_a: true)
    let head_w = Tensor([256, 10])
    let head_b = Tensor.zeros([10])
    let logits = raw_matmul(hidden, head_w, bias: head_b, epilogue: 1)
    gpu.synchronize()
    logits.free()
    head_b.free()
    head_w.free()
    grad_w.free()
    grad_x.free()
    hidden.free()
    x.free()
    weights.free()
    bias.free()
    _warmed = true
}

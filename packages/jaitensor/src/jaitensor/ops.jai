#: GPU tensor operations that keep their inputs and results on the device.
import std.gpu as gpu
import std.math
import .gemm as gemm
from .tensor import GradNode, Tensor, element_count
from .trace import record_dense_forward, record_dense_backward
from .ops.shaders import BINARY_FORMS, UNARY_FORMS
from .ops.kernels import cached_attn_kernel, cached_kernel, f32_bits
from .ops.kernels import reduce_leading, unary_backward

fn binary(a: Tensor, b: Tensor, operation: int) -> Tensor {
    if not a.same_shape(b) { throw ValueError(f"tensor shapes differ: {a.shape} and {b.shape}") }
    if operation < 0 or operation >= BINARY_FORMS.len() {
        throw ValueError(f"no binary operation has code {operation}")
    }
    let (name, _symbol) = BINARY_FORMS[operation]
    let output = Tensor(a.shape)
    let kernel = cached_kernel(f"binary_{name}")
    kernel.dispatch_async([a.buffer(), b.buffer(), output.buffer()], a.size, [a.size])
    return output
}

fn unary(input: Tensor, operation: int) -> Tensor {
    if operation < 0 or operation >= UNARY_FORMS.len() {
        throw ValueError(f"no unary operation has code {operation}")
    }
    let (name, _expression) = UNARY_FORMS[operation]
    let output = Tensor(input.shape)
    let kernel = cached_kernel(f"unary_{name}")
    kernel.dispatch_async([input.buffer(), output.buffer()], input.size, [input.size])
    return output
}

fn is_trailing_suffix(small: list[int], large: list[int]) -> bool {
    if small.len() > large.len() { return false }
    let offset = large.len() - small.len()
    for index in 0..small.len() {
        if small[index] != large[offset + index] { return false }
    }
    return true
}

fn broadcast_add(a: Tensor, b: Tensor) -> Tensor {
    let output = Tensor(a.shape)
    let kernel = cached_kernel("broadcast_add")
    kernel.dispatch_async([a.buffer(), b.buffer(), output.buffer()], a.size, [b.size, a.size])
    return output
}

fn binary_backward(a: Tensor, b: Tensor, upstream: Tensor, operation: int) -> list[Tensor] {
    let grad_a = Tensor(a.shape)
    let grad_b = Tensor(b.shape)
    let kernel = cached_kernel("binary_backward_op")
    kernel.dispatch_async(
        [a.buffer(), b.buffer(), upstream.buffer(), grad_a.buffer(), grad_b.buffer()],
        a.size,
        [a.size, operation]
    )
    return [grad_a, grad_b]
}

#: Elementwise sum of equal-shaped tensors, or of `a` with a `b` whose shape
#: is a trailing suffix of `a`'s (for example `[units]` against `[batch,
#: units]`).
pub fn add(a: Tensor, b: Tensor) -> Tensor {
    let broadcasting = not a.same_shape(b)
    if broadcasting and not is_trailing_suffix(b.shape, a.shape) {
        throw ValueError(f"cannot broadcast {b.shape} against {a.shape}")
    }
    let output = if broadcasting { broadcast_add(a, b) } else { binary(a, b, 0) }
    if a.requires_grad or b.requires_grad {
        output.attach_creator(GradNode([a, b], |upstream| if broadcasting {
            [unary(upstream, 0), reduce_leading(upstream, b.shape)]
        } else {
            binary_backward(a, b, upstream, 0)
        }))
    }
    return output
}

#: Elementwise difference of equal-shaped tensors.
pub fn subtract(a: Tensor, b: Tensor) -> Tensor {
    let output = binary(a, b, 1)
    if a.requires_grad or b.requires_grad {
        output.attach_creator(GradNode([a, b], |upstream| binary_backward(a, b, upstream, 1)))
    }
    return output
}

#: Elementwise product of equal-shaped tensors.
pub fn multiply(a: Tensor, b: Tensor) -> Tensor {
    let output = binary(a, b, 2)
    if a.requires_grad or b.requires_grad {
        output.attach_creator(GradNode([a, b], |upstream| binary_backward(a, b, upstream, 2)))
    }
    return output
}

#: Elementwise quotient of equal-shaped tensors.
pub fn divide(a: Tensor, b: Tensor) -> Tensor {
    let output = binary(a, b, 3)
    if a.requires_grad or b.requires_grad {
        output.attach_creator(GradNode([a, b], |upstream| binary_backward(a, b, upstream, 3)))
    }
    return output
}

#: Multiply every element by one host scalar.
pub fn scale(input: Tensor, value: float) -> Tensor {
    let output = Tensor(input.shape)
    let kernel = cached_kernel("scalar_mul_bits")
    kernel.dispatch_async(
        [input.buffer(), output.buffer()],
        input.size,
        [input.size, f32_bits(value)]
    )
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [scale(upstream, value)]))
    }
    return output
}

#: Multiply every element by one number that is itself on the device.
#:
#: `scale` takes a host float, which is no use to a backward pass: the number
#: it wants is the upstream gradient, and fetching that to the host to pass it
#: back down would mean waiting for the device mid-graph.
pub fn scale_by(input: Tensor, factor: Tensor) -> Tensor {
    if factor.size != 1 {
        throw ValueError(f"scale_by needs a single number, got {factor.shape}")
    }
    let output = Tensor(input.shape)
    cached_kernel("scalar_mul").dispatch_async(
        [input.buffer(), factor.buffer(), output.buffer()],
        input.size,
        [input.size]
    )
    return output
}

#: Reinterpret a tensor as `shape` without copying device memory.
pub fn reshape(input: Tensor, shape: list[int]) -> Tensor {
    if element_count(shape) != input.size {
        throw ValueError(f"cannot reshape {input.shape} into {shape}")
    }
    let output = Tensor(shape, storage: input.buffer().view(0, input.size))
    if input.requires_grad {
        let original_shape = input.shape
        output.attach_creator(GradNode([input], |upstream| [reshape(upstream, original_shape)]))
    }
    return output
}

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

fn raw_matmul(
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
fn raw_matmul_batched(
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

fn raw_transpose(input: Tensor) -> Tensor {
    let rows = input.shape[0]
    let columns = input.shape[1]
    let output = Tensor([columns, rows])
    let kernel = cached_kernel("matrix_transpose")
    let across = (columns + 31) // 32
    let down = (rows + 31) // 32
    kernel.dispatch_async(
        [input.buffer(), output.buffer()],
        across * down * 256,
        [rows, columns, across],
        group_size: 256
    )
    return output
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

#: Transpose a rank-two tensor.
pub fn transpose(input: Tensor) -> Tensor {
    if input.rank() != 2 { throw ValueError(f"transpose needs rank two, got {input.shape}") }
    let output = raw_transpose(input)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [raw_transpose(upstream)]))
    }
    return output
}

#: Elementwise rectified linear activation.
pub fn relu(input: Tensor) -> Tensor {
    let output = unary(input, 1)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(output, upstream, 1)]))
    }
    return output
}

#: Elementwise logistic sigmoid activation.
pub fn sigmoid(input: Tensor) -> Tensor {
    let output = unary(input, 2)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(output, upstream, 2)]))
    }
    return output
}

#: Elementwise hyperbolic tangent activation.
pub fn tanh(input: Tensor) -> Tensor {
    let output = unary(input, 3)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(output, upstream, 3)]))
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

#: Elementwise exponential.
pub fn exp(input: Tensor) -> Tensor {
    let output = unary(input, 4)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(output, upstream, 4)]))
    }
    return output
}

#: Elementwise natural logarithm.
pub fn log(input: Tensor) -> Tensor {
    let output = unary(input, 5)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(input, upstream, 5)]))
    }
    return output
}

#: Elementwise absolute value.
pub fn abs(input: Tensor) -> Tensor {
    let output = unary(input, 6)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(input, upstream, 6)]))
    }
    return output
}

#: Elementwise negation.
pub fn neg(input: Tensor) -> Tensor {
    let output = unary(input, 7)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(output, upstream, 7)]))
    }
    return output
}

#: Elementwise square root.
pub fn sqrt(input: Tensor) -> Tensor {
    let output = unary(input, 8)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(output, upstream, 8)]))
    }
    return output
}

#: Reciprocal square root.
pub fn rsqrt(input: Tensor) -> Tensor {
    let output = unary(input, 9)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(output, upstream, 9)]))
    }
    return output
}

#: Elementwise square.
pub fn square(input: Tensor) -> Tensor {
    let output = unary(input, 10)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(input, upstream, 10)]))
    }
    return output
}

#: Gaussian error linear unit (tanh approximation).
pub fn gelu(input: Tensor) -> Tensor {
    let output = unary(input, 11)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(input, upstream, 11)]))
    }
    return output
}

#: Sigmoid linear unit, `x * sigmoid(x)`.
pub fn silu(input: Tensor) -> Tensor {
    let output = unary(input, 12)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(input, upstream, 12)]))
    }
    return output
}

#: Leaky ReLU with negative slope 0.01.
pub fn leaky_relu(input: Tensor) -> Tensor {
    let output = unary(input, 13)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(input, upstream, 13)]))
    }
    return output
}

#: Clamp each value into `[lo, hi]`.
pub fn clamp(input: Tensor, lo: float, hi: float) -> Tensor {
    if lo > hi { throw ValueError(f"clamp lo {lo} is greater than hi {hi}") }
    let output = Tensor(input.shape)
    let kernel = cached_kernel("clamp_op")
    kernel.dispatch_async(
        [input.buffer(), output.buffer()],
        input.size,
        [input.size, f32_bits(lo), f32_bits(hi)]
    )
    if input.requires_grad {
        output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
            let g = Tensor(input.shape)
            let k = cached_kernel("clamp_backward")
            k.dispatch_async([input.buffer(), upstream.buffer(), g.buffer()], input.size, [input.size, f32_bits(lo), f32_bits(hi)])
            return [g]
        }))
    }
    return output
}

#: Route an upstream gradient to whichever side of a min/max comparison won.
fn select_grads(a: Tensor, b: Tensor, upstream: Tensor, want_max: int) -> list[Tensor] {
    let kernel = cached_kernel("select_backward")
    var grads: list[Tensor] = []
    for side in 0..2 {
        let grad = Tensor(a.shape)
        kernel.dispatch_async(
            [a.buffer(), b.buffer(), upstream.buffer(), grad.buffer()],
            a.size,
            [a.size, want_max, side]
        )
        grads.push(grad)
    }
    return grads
}

#: Elementwise maximum of equal-shaped tensors.
pub fn maximum(a: Tensor, b: Tensor) -> Tensor {
    if not a.same_shape(b) { throw ValueError(f"tensor shapes differ: {a.shape} and {b.shape}") }
    let output = Tensor(a.shape)
    let kernel = cached_kernel("maximum_op")
    kernel.dispatch_async([a.buffer(), b.buffer(), output.buffer()], a.size, [a.size])
    if a.requires_grad or b.requires_grad {
        output.attach_creator(GradNode([a, b], |upstream| select_grads(a, b, upstream, 1)))
    }
    return output
}

#: Elementwise minimum of equal-shaped tensors.
pub fn minimum(a: Tensor, b: Tensor) -> Tensor {
    if not a.same_shape(b) { throw ValueError(f"tensor shapes differ: {a.shape} and {b.shape}") }
    let output = Tensor(a.shape)
    let kernel = cached_kernel("minimum_op")
    kernel.dispatch_async([a.buffer(), b.buffer(), output.buffer()], a.size, [a.size])
    if a.requires_grad or b.requires_grad {
        output.attach_creator(GradNode([a, b], |upstream| select_grads(a, b, upstream, 0)))
    }
    return output
}

fn broadcast_scalar(source: Tensor, shape: list[int]) -> Tensor {
    let output = Tensor(shape)
    let kernel = cached_kernel("fill_from_scalar")
    kernel.dispatch_async([source.buffer(), output.buffer()], output.size, [output.size])
    return output
}

fn expand_last(reduced: Tensor, columns: int) -> Tensor {
    var shape: list[int] = []
    for dimension in reduced.shape { shape.push(dimension) }
    shape.push(columns)
    let output = Tensor(shape)
    let kernel = cached_kernel("broadcast_last")
    kernel.dispatch_async([reduced.buffer(), output.buffer()], output.size, [reduced.size, columns])
    return output
}

#: Sum of every element, or of the last axis when `axis` is `-1`.
pub fn sum(input: Tensor, axis: int? = null) -> Tensor {
    if axis is null {
        #: Enough groups to fill the machine and no more. Each one sweeps its
        #: share of the input in a register, so giving every element its own
        #: thread would only add threadgroups to combine.
        let ELEMENTS_PER_GROUP = 2048
        let groups = min(1024, max(1, (input.size + ELEMENTS_PER_GROUP - 1) // ELEMENTS_PER_GROUP))
        let partials = Tensor([groups])
        let kernel = cached_kernel("reduce_sum_all")
        kernel.dispatch_async(
            [input.buffer(), partials.buffer()],
            groups * 256,
            [input.size, groups * 256],
            group_size: 256
        )
        let output = if groups == 1 {
            partials
        } else {
            let total = sum(partials)
            partials.free()
            total
        }
        if input.requires_grad {
            output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
                return [broadcast_scalar(upstream, input.shape)]
            }))
        }
        return output
    }
    let resolved = if axis < 0 { input.rank() + axis } else { axis }
    if resolved != input.rank() - 1 {
        throw ValueError("sum currently reduces all elements or the last axis")
    }
    let columns = input.shape[resolved]
    let rows = input.size // columns
    var out_shape: list[int] = []
    for i in 0..resolved { out_shape.push(input.shape[i]) }
    let output = Tensor(out_shape)
    let kernel = cached_kernel("reduce_sum_rows")
    kernel.dispatch_async(
        [input.buffer(), output.buffer()],
        rows * 256,
        [rows, columns],
        group_size: 256
    )
    if input.requires_grad {
        output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
            return [expand_last(upstream, columns)]
        }))
    }
    return output
}

#: Mean of every element, or of the last axis.
pub fn mean(input: Tensor, axis: int? = null) -> Tensor {
    let reduced = sum(input, axis: axis)
    let count = if axis is null { float(input.size) } else { float(input.dim(-1)) }
    let scaled = scale(reduced, 1.0 / count)
    if not input.requires_grad { reduced.free() }
    return scaled
}

#: Concatenate tensors along `axis` (0 or the last axis).
pub fn cat(parts: list[Tensor], axis: int = 0) -> Tensor {
    if parts.len() == 0 { throw ValueError("cat needs at least one tensor") }
    let rank = parts[0].rank()
    let resolved = if axis < 0 { rank + axis } else { axis }
    if resolved != 0 and resolved != rank - 1 {
        throw ValueError("cat currently supports axis 0 or the last axis")
    }
    var along = 0
    for part in parts {
        if part.rank() != rank { throw ValueError("cat parts must have the same rank") }
        for i in 0..rank {
            if i == resolved { continue }
            if part.shape[i] != parts[0].shape[i] {
                throw ValueError(
                    f"cat shape mismatch at axis {i}: {part.shape} vs {parts[0].shape}"
                )
            }
        }
        along += part.shape[resolved]
    }
    var out_shape: list[int] = []
    for i in 0..rank {
        out_shape.push(if i == resolved { along } else { parts[0].shape[i] })
    }
    let output = Tensor(out_shape)
    var cursor = 0
    let kernel = cached_kernel("copy_range")
    var wants_grad = false
    for part in parts {
        if part.requires_grad { wants_grad = true }
    }
    if resolved == 0 {
        var offsets: list[int] = []
        for part in parts {
            offsets.push(cursor)
            kernel.dispatch_async(
                [part.buffer(), output.buffer()],
                part.size,
                [0, cursor, part.size]
            )
            cursor += part.size
        }
        if wants_grad {
            let shapes = [part.shape for part in parts]
            output.attach_creator(GradNode(parts, fn(upstream: Tensor) -> list[Tensor] {
                var grads: list[Tensor] = []
                let back = cached_kernel("copy_range")
                for (index, shape) in shapes.enumerate() {
                    let grad = Tensor(shape)
                    back.dispatch_async([upstream.buffer(), grad.buffer()], grad.size, [offsets[index], 0, grad.size])
                    grads.push(grad)
                }
                return grads
            }))
        }
        return output
    }
    let outer = parts[0].size // parts[0].shape[resolved]
    var prefix = 0
    var starts: list[int] = []
    let strided = cached_kernel("copy_strided_rows")
    for part in parts {
        let width = part.shape[resolved]
        starts.push(prefix)
        strided.dispatch_async(
            [part.buffer(), output.buffer()],
            outer * width,
            [outer, width, along, prefix]
        )
        prefix += width
    }
    if wants_grad {
        let shapes = [part.shape for part in parts]
        output.attach_creator(GradNode(parts, fn(upstream: Tensor) -> list[Tensor] {
            var grads: list[Tensor] = []
            let take = cached_kernel("take_strided_rows")
            for (index, shape) in shapes.enumerate() {
                let grad = Tensor(shape)
                let width = shape[resolved]
                take.dispatch_async([upstream.buffer(), grad.buffer()], outer * width, [outer, width, along, starts[index]])
                grads.push(grad)
            }
            return grads
        }))
    }
    return output
}

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

#: Mean squared error between equal-shaped tensors. Returns a scalar.
pub fn mse(prediction: Tensor, target: Tensor) -> Tensor {
    if not prediction.same_shape(target) {
        throw ValueError(f"mse shape mismatch: {prediction.shape} vs {target.shape}")
    }
    let diff = subtract(prediction, target)
    let sq = square(diff)
    let loss = mean(sq)
    if prediction.requires_grad or target.requires_grad { return loss }
    diff.free()
    sq.free()
    return loss
}

#: `cond != 0` picks `a`, otherwise `b`. All three must share a shape.
pub fn where(cond: Tensor, a: Tensor, b: Tensor) -> Tensor {
    if not cond.same_shape(a) or not a.same_shape(b) {
        throw ValueError(f"where needs matching shapes, got {cond.shape}, {a.shape}, {b.shape}")
    }
    let output = Tensor(a.shape)
    let kernel = cached_kernel("where_op")
    kernel.dispatch_async(
        [cond.buffer(), a.buffer(), b.buffer(), output.buffer()],
        a.size,
        [a.size]
    )
    return output
}

#: Raise each element to a host exponent.
pub fn pow(input: Tensor, exponent: float) -> Tensor {
    let output = Tensor(input.shape)
    let kernel = cached_kernel("pow_scalar")
    kernel.dispatch_async(
        [input.buffer(), output.buffer()],
        input.size,
        [input.size, f32_bits(exponent)]
    )
    if input.requires_grad {
        output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
            let lowered = Tensor(input.shape)
            let inner = cached_kernel("pow_scalar")
            inner.dispatch_async([input.buffer(), lowered.buffer()], input.size, [input.size, f32_bits(exponent - 1.0)])
            let chained = multiply(upstream, lowered)
            lowered.free()
            let grad = scale(chained, exponent)
            chained.free()
            return [grad]
        }))
    }
    return output
}

#: Insert a size-1 axis at `axis`.
pub fn unsqueeze(input: Tensor, axis: int) -> Tensor {
    let rank = input.rank()
    let resolved = if axis < 0 { rank + 1 + axis } else { axis }
    if resolved < 0 or resolved > rank {
        throw ValueError(f"unsqueeze axis {axis} is outside rank {rank}")
    }
    var shape: list[int] = []
    for i in 0..resolved { shape.push(input.shape[i]) }
    shape.push(1)
    for i in resolved..rank { shape.push(input.shape[i]) }
    return reshape(input, shape)
}

#: Drop size-1 axes. `axis` null drops every size-1 axis.
pub fn squeeze(input: Tensor, axis: int? = null) -> Tensor {
    var shape: list[int] = []
    if axis is null {
        for d in input.shape {
            if d != 1 { shape.push(d) }
        }
        if shape.len() == 0 { shape.push(1) }
        return reshape(input, shape)
    }
    let resolved = if axis < 0 { input.rank() + axis } else { axis }
    if resolved < 0 or resolved >= input.rank() {
        throw ValueError(f"squeeze axis {axis} is outside rank {input.rank()}")
    }
    if input.shape[resolved] != 1 {
        throw ValueError(f"cannot squeeze axis {axis} of size {input.shape[resolved]}")
    }
    for i in 0..input.rank() {
        if i != resolved { shape.push(input.shape[i]) }
    }
    if shape.len() == 0 { shape.push(1) }
    return reshape(input, shape)
}

#: Stack tensors by inserting a new leading axis, then concatenating.
pub fn stack(parts: list[Tensor]) -> Tensor {
    if parts.len() == 0 { throw ValueError("stack needs at least one tensor") }
    var expanded: list[Tensor] = []
    for part in parts { expanded.push(unsqueeze(part, 0)) }
    return cat(expanded, axis: 0)
}

#: Zero-pad an NHWC tensor. `pad` is `(top, bottom, left, right)`.
pub fn pad2d(input: Tensor, pad: list[int]) -> Tensor {
    if input.rank() != 4 { throw ValueError(f"pad2d needs NHWC input, got {input.shape}") }
    if pad.len() != 4 { throw ValueError("pad2d pad must be [top, bottom, left, right]") }
    for p in pad {
        if p < 0 { throw ValueError("pad2d pad values must be non-negative") }
    }
    let n = input.shape[0]
    let h = input.shape[1]
    let w = input.shape[2]
    let c = input.shape[3]
    let output = Tensor([n, h + pad[0] + pad[1], w + pad[2] + pad[3], c])
    let wide = c % 4 == 0
    let entry = if wide { "pad2d_nhwc4" } else { "pad2d_nhwc" }
    let threads = if wide { output.size // 4 } else { output.size }
    cached_kernel(entry).dispatch_async(
        [input.buffer(), output.buffer()],
        threads,
        [n, h, w, c, pad[0], pad[1], pad[2], pad[3]]
    )
    if input.requires_grad {
        let padded_h = h + pad[0] + pad[1]
        let padded_w = w + pad[2] + pad[3]
        let top = pad[0]
        let left = pad[2]
        output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
            let grad = Tensor([n, h, w, c])
            let crop = cached_kernel("crop2d_nhwc")
            crop.dispatch_async([upstream.buffer(), grad.buffer()], grad.size, [n, h, w, c, top, left, padded_h, padded_w])
            return [grad]
        }))
    }
    return output
}

#: NHWC `[N,H,W,C]` to NCHW `[N,C,H,W]`.
#: Tiles a side, and the group that walks one. Matched by `TR_TILE` and
#: `TR_ROWS` in the kernel source.
const TRANSPOSE_TILE = 32
const TRANSPOSE_GROUP = 256

#: `count` matrices of `rows` by `cols`, each written back the other way up.
fn dispatch_transpose(input: Tensor, output: Tensor, count: int, rows: int, cols: int) -> void {
    let across = (cols + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE
    let down = (rows + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE
    cached_kernel("transpose_batched").dispatch_async(
        [input.buffer(), output.buffer()],
        count * across * down * TRANSPOSE_GROUP,
        [rows, cols, across, down],
        group_size: TRANSPOSE_GROUP
    )
}

pub fn nchw(input: Tensor) -> Tensor {
    if input.rank() != 4 { throw ValueError(f"nchw needs rank four, got {input.shape}") }
    let n = input.shape[0]
    let h = input.shape[1]
    let w = input.shape[2]
    let c = input.shape[3]
    let output = Tensor([n, c, h, w])
    dispatch_transpose(input, output, n, h * w, c)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [nhwc(upstream)]))
    }
    return output
}

#: NCHW `[N,C,H,W]` to NHWC `[N,H,W,C]`.
pub fn nhwc(input: Tensor) -> Tensor {
    if input.rank() != 4 { throw ValueError(f"nhwc needs rank four, got {input.shape}") }
    let n = input.shape[0]
    let c = input.shape[1]
    let h = input.shape[2]
    let w = input.shape[3]
    let output = Tensor([n, h, w, c])
    dispatch_transpose(input, output, n, c, h * w)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [nchw(upstream)]))
    }
    return output
}

#: Permute axes. Rank two is transpose. Rank four supports NHWC<->NCHW layouts.
pub fn permute(input: Tensor, axes: list[int]) -> Tensor {
    if axes.len() != input.rank() {
        throw ValueError(f"permute axes length {axes.len()} does not match rank {input.rank()}")
    }
    if input.rank() == 2 {
        if axes[0] == 1 and axes[1] == 0 { return transpose(input) }
        if axes[0] == 0 and axes[1] == 1 { return reshape(input, input.shape) }
        throw ValueError(f"unsupported permute {axes}")
    }
    if input.rank() == 4 {
        if axes == [0, 3, 1, 2] { return nchw(input) }
        if axes == [0, 2, 3, 1] { return nhwc(input) }
        if axes == [0, 1, 2, 3] { return reshape(input, input.shape) }
        throw ValueError(f"unsupported permute {axes}")
    }
    throw ValueError("permute currently supports rank two and four")
}

#: Repeat the last axis `times` times.
pub fn repeat(input: Tensor, times: int, axis: int = -1) -> Tensor {
    if times <= 0 { throw ValueError(f"repeat times must be positive, got {times}") }
    let resolved = if axis < 0 { input.rank() + axis } else { axis }
    if resolved != input.rank() - 1 {
        throw ValueError("repeat currently supports the last axis")
    }
    let inner = input.shape[resolved]
    let rows = input.size // inner
    var shape: list[int] = []
    for i in 0..input.rank() {
        shape.push(if i == resolved { inner * times } else { input.shape[i] })
    }
    let output = Tensor(shape)
    let wide = inner % 4 == 0
    let entry = if wide { "repeat_last4" } else { "repeat_last" }
    let threads = if wide { output.size // 4 } else { output.size }
    cached_kernel(entry).dispatch_async(
        [input.buffer(), output.buffer()],
        threads,
        [rows, inner, times]
    )
    if input.requires_grad {
        let original = input.shape
        output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
            let grad = Tensor(original)
            let back = cached_kernel("repeat_last_backward")
            back.dispatch_async([upstream.buffer(), grad.buffer()], grad.size, [rows, inner, times])
            return [grad]
        }))
    }
    return output
}

#: Mean binary cross-entropy between probabilities and 0/1 targets.
pub fn binary_cross_entropy(prediction: Tensor, target: Tensor) -> Tensor {
    if not prediction.same_shape(target) {
        throw ValueError(f"bce shape mismatch: {prediction.shape} vs {target.shape}")
    }
    let per = Tensor(prediction.shape)
    let kernel = cached_kernel("bce_sample")
    kernel.dispatch_async(
        [prediction.buffer(), target.buffer(), per.buffer()],
        prediction.size,
        [prediction.size]
    )
    if prediction.requires_grad {
        per.attach_creator(GradNode([prediction], fn(upstream: Tensor) -> list[Tensor] {
            let g = Tensor(prediction.shape)
            let backward_kernel = cached_kernel("bce_prob_backward")
            backward_kernel.dispatch_async([prediction.buffer(), target.buffer(), upstream.buffer(), g.buffer()], prediction.size, [prediction.size])
            return [g]
        }))
    }
    let loss = mean(per)
    if not prediction.requires_grad { per.free() }
    return loss
}

#: Mean binary cross-entropy from logits. Numerically stable; no sigmoid first.
pub fn binary_cross_entropy_with_logits(logits: Tensor, target: Tensor) -> Tensor {
    if not logits.same_shape(target) {
        throw ValueError(f"bce_with_logits shape mismatch: {logits.shape} vs {target.shape}")
    }
    let per = Tensor(logits.shape)
    let kernel = cached_kernel("bce_logits_sample")
    kernel.dispatch_async(
        [logits.buffer(), target.buffer(), per.buffer()],
        logits.size,
        [logits.size]
    )
    if logits.requires_grad {
        per.attach_creator(GradNode([logits], fn(upstream: Tensor) -> list[Tensor] {
            let g = Tensor(logits.shape)
            let backward_kernel = cached_kernel("bce_logits_backward")
            backward_kernel.dispatch_async([logits.buffer(), target.buffer(), upstream.buffer(), g.buffer()], logits.size, [logits.size])
            return [g]
        }))
    }
    let loss = mean(per)
    if not logits.requires_grad { per.free() }
    return loss
}

#: Keep the lower triangle of a rank-two tensor, including the diagonal.
pub fn tril(input: Tensor) -> Tensor {
    if input.rank() != 2 { throw ValueError(f"tril needs rank two, got {input.shape}") }
    let rows = input.shape[0]
    let cols = input.shape[1]
    let output = Tensor(input.shape)
    let kernel = cached_attn_kernel("tril_square")
    kernel.dispatch_async([input.buffer(), output.buffer()], output.size, [rows, cols])
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [tril(upstream)]))
    }
    return output
}

#: Identity matrix of size `n` on the GPU.
pub fn eye(n: int) -> Tensor {
    if n <= 0 { throw ValueError(f"eye needs a positive size, got {n}") }
    let output = Tensor([n, n])
    let kernel = cached_kernel("eye_fill")
    kernel.dispatch_async([output.buffer()], output.size, [n])
    return output
}

#: Reverse the last axis. Other axes are not yet supported.
pub fn flip(input: Tensor, axis: int = -1) -> Tensor {
    let rank = input.rank()
    let resolved = if axis < 0 { rank + axis } else { axis }
    if resolved != rank - 1 {
        throw ValueError("flip currently supports the last axis")
    }
    let inner = input.shape[resolved]
    let rows = input.size // inner
    let output = Tensor(input.shape)
    let kernel = cached_kernel("flip_last")
    kernel.dispatch_async([input.buffer(), output.buffer()], output.size, [rows, inner])
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [flip(upstream, axis: axis)]))
    }
    return output
}

#: Nearest-neighbor upsample of NHWC images by an integer `scale`.
pub fn upsample_nearest2d(input: Tensor, scale: int) -> Tensor {
    if input.rank() != 4 {
        throw ValueError(f"upsample_nearest2d needs NHWC input, got {input.shape}")
    }
    if scale <= 0 { throw ValueError(f"upsample scale must be positive, got {scale}") }
    let n = input.shape[0]
    let h = input.shape[1]
    let w = input.shape[2]
    let c = input.shape[3]
    let out_h = h * scale
    let out_w = w * scale
    let output = Tensor([n, out_h, out_w, c])
    let wide = c % 4 == 0
    let entry = if wide { "upsample_nearest2d_nhwc4" } else { "upsample_nearest2d_nhwc" }
    let threads = if wide { output.size // 4 } else { output.size }
    cached_kernel(entry).dispatch_async(
        [input.buffer(), output.buffer()],
        threads,
        [n, h, w, c, scale, scale, out_h, out_w]
    )
    if input.requires_grad {
        output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
            let grad = Tensor([n, h, w, c])
            let back = cached_kernel("upsample_nearest2d_backward_nhwc")
            back.dispatch_async([upstream.buffer(), grad.buffer()], grad.size, [n, h, w, c, scale, scale, out_h, out_w])
            return [grad]
        }))
    }
    return output
}

#: Pack `[seq, dim]` into `[heads * seq, head_dim]` with each head's rows contiguous.
pub fn split_heads(input: Tensor, heads: int) -> Tensor {
    if input.rank() != 2 {
        throw ValueError(f"split_heads needs rank two, got {input.shape}")
    }
    if heads <= 0 { throw ValueError(f"split_heads needs a positive head count, got {heads}") }
    let seq = input.shape[0]
    let dim = input.shape[1]
    if dim % heads != 0 {
        throw ValueError(f"split_heads dim {dim} must divide heads {heads}")
    }
    let hd = dim // heads
    let output = Tensor([heads * seq, hd])
    let kernel = cached_attn_kernel("split_heads")
    kernel.dispatch_async([input.buffer(), output.buffer()], output.size, [seq, heads, hd])
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [merge_heads(upstream, heads, seq)]))
    }
    return output
}

#: Inverse of `split_heads`: `[heads * seq, head_dim]` back to `[seq, dim]`.
pub fn merge_heads(packed: Tensor, heads: int, seq: int) -> Tensor {
    if packed.rank() != 2 {
        throw ValueError(f"merge_heads needs rank two, got {packed.shape}")
    }
    if heads <= 0 or seq <= 0 {
        throw ValueError("merge_heads needs positive heads and sequence length")
    }
    if packed.shape[0] != heads * seq {
        throw ValueError(f"merge_heads packed rows {packed.shape[0]} != {heads} * {seq}")
    }
    let hd = packed.shape[1]
    let output = Tensor([seq, heads * hd])
    let kernel = cached_attn_kernel("merge_heads")
    kernel.dispatch_async([packed.buffer(), output.buffer()], packed.size, [seq, heads, hd])
    if packed.requires_grad {
        output.attach_creator(GradNode([packed], |upstream| [split_heads(upstream, heads)]))
    }
    return output
}

pub fn softmax(input: Tensor) -> Tensor {
    if input.rank() != 2 { throw ValueError(f"softmax needs rank two, got {input.shape}") }
    let rows = input.shape[0]
    let columns = input.shape[1]
    let output = Tensor(input.shape)
    let kernel = cached_kernel("row_softmax")
    kernel.dispatch_async(
        [input.buffer(), output.buffer()],
        rows * 256,
        [rows, columns],
        group_size: 256
    )
    if input.requires_grad {
        output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
            let grad_input = Tensor(input.shape)
            let backward_kernel = cached_kernel("softmax_backward")
            backward_kernel.dispatch_async([output.buffer(), upstream.buffer(), grad_input.buffer()], rows * 256, [rows, columns], group_size: 256)
            return [grad_input]
        }))
    }
    return output
}

#: Index of the largest value in each row of a rank-two tensor.
#:
#: Indices use float storage because all Metal buffers in `std.gpu` are
#: float32. Downloaded values are exact for practical class counts.
#: The largest value along the last axis, or of the whole tensor.
#:
#: The companion to `argmax`, which gives where the largest is but not what it
#: is. A detector wants both -- the class and its score -- and computing the
#: value on the host afterwards means downloading everything the reduction was
#: supposed to save.
pub fn amax(input: Tensor, axis: int? = null) -> Tensor {
    if axis is null {
        let flat = reshape(input, [1, input.size])
        let whole = amax(flat, axis: 1)
        flat.free()
        return whole
    }
    let resolved = if axis < 0 { input.rank() + axis } else { axis }
    if resolved != input.rank() - 1 {
        throw ValueError("amax reduces all elements or the last axis")
    }
    let columns = input.shape[resolved]
    let rows = input.size // columns
    var shape: list[int] = []
    for index in 0..input.rank() - 1 { shape.push(input.shape[index]) }
    if shape.len() == 0 { shape.push(1) }
    let output = Tensor(shape)
    cached_kernel("row_max").dispatch_async(
        [input.buffer(), output.buffer()],
        rows * 256,
        [rows, columns],
        group_size: 256
    )
    return output
}

pub fn argmax(input: Tensor) -> Tensor {
    if input.rank() != 2 { throw ValueError(f"argmax needs rank two, got {input.shape}") }
    let rows = input.shape[0]
    let columns = input.shape[1]
    let output = Tensor([rows])
    let kernel = cached_kernel("row_argmax")
    kernel.dispatch_async(
        [input.buffer(), output.buffer()],
        rows * 256,
        [rows, columns],
        group_size: 256
    )
    return output
}

#: The largest value along an axis and where it sits, in one pass.
#:
#: `amax` and `argmax` each read the whole tensor to work out the same thing,
#: so asking for both costs twice what it needs to. Whoever wants only one of
#: them should still say so. On the 8400 by 80 class scores a detector produces
#: per frame the pair now costs 0.137 ms, against 0.65 ms when a row was one
#: thread's serial walk.
#:
#: Reducing down the columns rather than along the rows is a separate kernel
#: rather than a transpose and the other one. The transpose is a whole extra
#: pass over the tensor and an allocation the size of it -- 0.3 ms and 2.7 MB
#: for those same scores -- and reading a column is no slower: neighbouring
#: threads take neighbouring elements of each row in turn, which is the access
#: the device wants.
pub fn max_with_index(input: Tensor, axis: int = -1) -> tuple[Tensor, Tensor] {
    if input.rank() != 2 {
        throw ValueError(f"max_with_index needs rank two, got {input.shape}")
    }
    let resolved = if axis < 0 { input.rank() + axis } else { axis }
    if resolved != 0 and resolved != 1 {
        throw ValueError(f"max_with_index takes axis 0 or 1, got {axis}")
    }
    let rows = input.shape[0]
    let columns = input.shape[1]
    let kept = if resolved == 1 { rows } else { columns }
    let values = Tensor([kept])
    let indices = Tensor([kept])
    #: Down a row a threadgroup reduces together; across one, neighbouring
    #: threads already take neighbouring elements and a thread each is right.
    if resolved == 1 {
        cached_kernel("row_peak").dispatch_async(
            [input.buffer(), values.buffer(), indices.buffer()],
            kept * 256,
            [rows, columns],
            group_size: 256
        )
    } else {
        cached_kernel("column_peak").dispatch_async(
            [input.buffer(), values.buffer(), indices.buffer()],
            kept,
            [rows, columns]
        )
    }
    return (values, indices)
}

#: Expand a rank-one tensor of integer-valued indices into one-hot rows.
pub fn one_hot(indices: Tensor, depth: int) -> Tensor {
    if indices.rank() != 1 { throw ValueError(f"one_hot needs rank one, got {indices.shape}") }
    if depth <= 0 { throw ValueError(f"depth must be positive, got {depth}") }
    let rows = indices.shape[0]
    let output = Tensor([rows, depth])
    let kernel = cached_kernel("one_hot_rows")
    kernel.dispatch_async([indices.buffer(), output.buffer()], rows * depth, [rows, depth])
    return output
}

#: Copy a contiguous range of `source` as a view of the same device memory.
#:
#: `source_offset` is in elements. The result is a leaf: it does not attach a
#: gradient graph. Freeing the view does not free `source`.
pub fn slice_flat(source: Tensor, source_offset: int, shape: list[int]) -> Tensor {
    let n = element_count(shape)
    if source_offset < 0 or source_offset + n > source.size {
        throw ValueError(
            f"slice_flat offset {source_offset} and size {n} " +
                f"run past source size {source.size}"
        )
    }
    let output = source.view(source_offset, shape)
    if source.requires_grad {
        let whole = source.shape
        output.attach_creator(GradNode([source], fn(upstream: Tensor) -> list[Tensor] {
            let grad = Tensor.zeros(whole)
            let kernel = cached_kernel("copy_range")
            kernel.dispatch_async([upstream.buffer(), grad.buffer()], n, [0, source_offset, n])
            return [grad]
        }))
    }
    return output
}

#: Copy `source` into `dest`. Both must have the same number of elements.
pub fn copy_into(source: Tensor, dest: Tensor) -> void {
    if source.size != dest.size {
        throw ValueError(f"copy_into size mismatch: {source.size} vs {dest.size}")
    }
    let kernel = cached_kernel("copy_offset")
    kernel.dispatch_async([source.buffer(), dest.buffer()], dest.size, [0, dest.size])
}

#: Add `value[0] * weight` into a scalar accumulator tensor.
pub fn add_scaled_into(value: Tensor, total: Tensor, weight: int) -> void {
    if value.size != 1 or total.size != 1 {
        throw ValueError("add_scaled_into needs scalar tensors")
    }
    if weight < 0 { throw ValueError(f"weight must be non-negative, got {weight}") }
    let kernel = cached_kernel("add_scaled_into")
    kernel.dispatch_async([value.buffer(), total.buffer()], 1, [weight])
}

#: Add the number of correct argmax classes into a scalar accumulator.
pub fn add_correct_into(logits: Tensor, labels: Tensor, total: Tensor) -> void {
    if logits.rank() != 2 {
        throw ValueError(f"add_correct_into needs rank-two logits, got {logits.shape}")
    }
    if total.size != 1 { throw ValueError("add_correct_into needs a scalar accumulator") }
    let batch = logits.shape[0]
    let classes = logits.shape[1]
    if labels.size != batch {
        throw ValueError(f"add_correct_into needs {batch} labels, got {labels.size}")
    }
    let kernel = cached_kernel("add_correct_into")
    kernel.dispatch_async(
        [logits.buffer(), labels.buffer(), total.buffer()],
        batch,
        [batch, classes]
    )
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

#: Gather `indices.len()` rows from a rank-two source.
#: Read whole rows of `source`, one per entry of `indices`.
#:
#: The gradient of this scatters back into the table with device atomics, and
#: a token that occurs twice in a batch has two threads adding to the same
#: slot. Float addition is not associative, so which order they arrive in
#: changes the last few bits: two runs from the same seed on the same data
#: give gradients that agree to about ten digits and not beyond. It is enough
#: to make a long training run diverge from its own repeat. Other libraries
#: are the same here for the same reason; making it exact means sorting the
#: indices and reducing each run of them in order, which costs more than the
#: reproducibility is usually worth.
pub fn gather_rows(source: Tensor, indices: Tensor, out_shape: list[int]? = null) -> Tensor {
    if source.rank() != 2 {
        throw ValueError(f"gather_rows needs a rank-two source, got {source.shape}")
    }
    #: The indices are read flat whatever shape they arrive in -- only how
    #: many there are matters, and a batch of index rows is the ordinary way
    #: an embedding is asked for. `out_shape` says what to call the result.
    if indices.rank() < 1 {
        throw ValueError(f"gather_rows needs an index tensor, got {indices.shape}")
    }
    let width = source.shape[1]
    var shape = [indices.size, width]
    if out_shape is not null {
        if element_count(out_shape) != indices.size * width {
            throw ValueError(
                f"gather_rows cannot reshape {indices.size}x{width} rows into {out_shape}"
            )
        }
        shape = out_shape
    }
    let output = Tensor(shape)
    let kernel = cached_kernel("gather_rows")
    kernel.dispatch_async(
        [source.buffer(), indices.buffer(), output.buffer()],
        output.size,
        [indices.size, width, source.shape[0]]
    )
    if source.requires_grad {
        output.attach_creator(GradNode([source], fn(upstream: Tensor) -> list[Tensor] {
            return [scatter_add_rows(upstream, indices, source.shape[0], width)]
        }))
    }
    return output
}

fn scatter_add_rows(upstream: Tensor, indices: Tensor, vocab: int, width: int) -> Tensor {
    let dest = Tensor.zeros([vocab, width])
    let kernel = cached_kernel("scatter_add_rows")
    kernel.dispatch_async(
        [upstream.buffer(), indices.buffer(), dest.buffer()],
        upstream.size,
        [indices.size, width, vocab]
    )
    return dest
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

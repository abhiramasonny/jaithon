#: Elementwise unary and binary ops, their broadcasting, and the
#: activations built from them.
from ..tensor import GradNode, Tensor
from .kernels import cached_kernel, f32_bits, reduce_leading, unary_backward
from .shaders import BINARY_FORMS, UNARY_FORMS

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

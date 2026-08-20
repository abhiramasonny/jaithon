#: Every differentiable operation, against a central difference of itself.
#:
#: jaitensor has forty-eight operations that attach a `GradNode`, and a wrong
#: backward is the hardest kind of bug this library can have: the model still
#: trains, the loss still falls, and the answer is quietly worse than it should
#: be. Nothing but a numerical comparison finds one. Before this file a handful
#: of ops had such a comparison and the rest had none; the convolution's three
#: gradients were among the ones that did not, and two of them were rewritten
#: the week this was written.
#:
#: Each case perturbs one input by a thousandth and asks whether the analytic
#: gradient predicts the change. `1e-2` throughout, which is what a step that
#: size buys over float32 -- tighter would make the test about rounding rather
#: than about the derivative.
#:
#: Inputs are chosen to stay away from every kink. `relu`, `leaky_relu`,
#: `maximum`, `minimum`, `abs` and `clamp` are not differentiable where their
#: two branches meet, and a central difference across that point measures
#: neither side, so no case puts a value there.
import jaitensor as jt
import std.math
from jaitensor.ops import abs as abs_op, avg_pool2d, batch_norm, cat, clamp, conv2d
from jaitensor.ops import binary_cross_entropy, binary_cross_entropy_with_logits
from jaitensor.ops import dense_affine, divide, exp, flip, gather_rows, gelu, group_norm
from jaitensor.ops import layer_norm, leaky_relu, log, matmul, max_pool2d, maximum
from jaitensor.ops import merge_heads, minimum, nchw, neg, nhwc, pad2d, pow as pow_op
from jaitensor.ops import relu, repeat, reshape, rsqrt, scale, sigmoid, silu, slice_flat
from jaitensor.ops import softmax, split_heads, sqrt as sqrt_op, square, sum as sum_op
from jaitensor.ops import tanh as tanh_op, transpose, tril, upsample_nearest2d
from jaitensor.tensor import Tensor
from std.test import assert_eq, assert_near, assert_true

fn have_gpu() -> bool { return jt.is_available() }

fn numerical_grad(f: fn(list[float]) -> float, x: list[float], index: int) -> float {
    let eps = 1e-3
    var plus = [value for value in x]
    var minus = [value for value in x]
    plus[index] += eps
    minus[index] -= eps
    return (f(plus) - f(minus)) / (2.0 * eps)
}

#: One op, one input, checked at every element.
#:
#: `apply` takes the tensor under test and returns whatever the op produced;
#: the result is summed to a scalar so there is a single number to
#: differentiate, and `sum`'s own gradient is one everywhere, so what is being
#: measured is the op's.
fn check(apply: fn(Tensor) -> Tensor, shape: list[int], values: list[float]) -> void {
    if not have_gpu() { return }
    let loss_of = fn(x: list[float]) -> float {
        let input = Tensor(shape, x)
        let output = apply(input)
        let total = jt.sum(output)
        let result = total.download()[0]
        input.free()
        output.free()
        total.free()
        return result
    }
    let input = Tensor(shape, values, requires_grad: true)
    let output = apply(input)
    let total = jt.sum(output)
    let intermediates = total.backward()
    let gradient = input.grad
    if gradient is null { throw AssertionError("the operation set no gradient") }
    let analytic = gradient.download()
    assert_eq(analytic.len(), values.len())
    for index in 0..values.len() {
        let expected = numerical_grad(loss_of, values, index)
        #: Relative once the gradient is large. A central difference over a sum
        #: of a hundred float32 terms carries a few parts in a thousand of
        #: rounding, and a fixed absolute bound would make this a test of
        #: float32 rather than of the derivative. Half a percent is far tighter
        #: than any real backward error.
        assert_near(analytic[index], expected, max(1e-2, math.fabs(expected) * 5e-3))
    }
    for tensor in intermediates { tensor.free() }
    input.free()
    total.free()
}

#: Values away from zero on both sides and away from one, so nothing sits on a
#: kink and nothing makes a logarithm or a reciprocal blow up.
fn spread(count: int) -> list[float] {
    var out: list[float] = []
    for index in 0..count {
        let step = float(index % 7) * 0.31 + 0.24
        out.push(if index % 2 == 0 { step } else { -step })
    }
    return out
}

#: The same spread kept positive, for the ops that need it. Deliberately not
#: the magnitudes `spread` uses: `maximum` and `minimum` are not differentiable
#: where their two inputs meet, and a second tensor that ties with the first at
#: every other element measures the kink rather than either side of it.
fn positive(count: int) -> list[float] {
    var out: list[float] = []
    for index in 0..count { out.push(float(index % 5) * 0.27 + 0.11) }
    return out
}

#: Strictly inside zero and one, for the losses that take a probability.
fn probabilities(count: int) -> list[float] {
    var out: list[float] = []
    for index in 0..count { out.push(0.15 + float(index % 6) * 0.12) }
    return out
}

fn test_elementwise_gradients() -> void {
    if not have_gpu() { return }
    let flat = [6]
    check(relu, flat, spread(6))
    check(leaky_relu, flat, spread(6))
    check(sigmoid, flat, spread(6))
    check(tanh_op, flat, spread(6))
    check(gelu, flat, spread(6))
    check(silu, flat, spread(6))
    check(exp, flat, spread(6))
    check(neg, flat, spread(6))
    check(square, flat, spread(6))
    check(abs_op, flat, spread(6))
    check(log, flat, positive(6))
    check(sqrt_op, flat, positive(6))
    check(rsqrt, flat, positive(6))
}

fn test_scalar_parameter_gradients() -> void {
    if not have_gpu() { return }
    check(fn(x: Tensor) -> Tensor { return scale(x, 2.5) }, [6], spread(6))
    check(fn(x: Tensor) -> Tensor { return pow_op(x, 3.0) }, [6], positive(6))
    #: Away from both bounds, where clamp is the identity and differentiable.
    check(fn(x: Tensor) -> Tensor { return clamp(x, -4.0, 4.0) }, [6], spread(6))
}

fn test_binary_gradients() -> void {
    if not have_gpu() { return }
    let other = Tensor([6], positive(6))
    check(fn(x: Tensor) -> Tensor { return jt.add(x, other) }, [6], spread(6))
    check(fn(x: Tensor) -> Tensor { return jt.subtract(x, other) }, [6], spread(6))
    check(fn(x: Tensor) -> Tensor { return jt.multiply(x, other) }, [6], spread(6))
    check(fn(x: Tensor) -> Tensor { return divide(x, other) }, [6], spread(6))
    #: No element ties with `other`, so neither branch of the comparison sits
    #: on the point where it is not differentiable.
    check(fn(x: Tensor) -> Tensor { return maximum(x, other) }, [6], spread(6))
    check(fn(x: Tensor) -> Tensor { return minimum(x, other) }, [6], spread(6))
    other.free()

    #: A trailing shape broadcast against a leading one, which is the path a
    #: bias takes and a different arm from the equal-shape case above.
    let bias = Tensor([3], [0.4, -0.7, 1.1])
    check(fn(x: Tensor) -> Tensor { return jt.add(x, bias) }, [2, 3], spread(6))
    bias.free()
}

fn test_shape_gradients() -> void {
    if not have_gpu() { return }
    check(fn(x: Tensor) -> Tensor { return reshape(x, [3, 2]) }, [2, 3], spread(6))
    check(transpose, [2, 3], spread(6))
    check(fn(x: Tensor) -> Tensor { return flip(x, 1) }, [2, 3], spread(6))
    check(fn(x: Tensor) -> Tensor { return repeat(x, 3, 1) }, [2, 1], spread(2))
    check(fn(x: Tensor) -> Tensor { return slice_flat(x, 2, [4]) }, [8], spread(8))
    check(tril, [3, 3], spread(9))
    check(fn(x: Tensor) -> Tensor { return split_heads(x, 2) }, [4, 6], spread(24))
    check(fn(x: Tensor) -> Tensor { return merge_heads(x, 2, 2) }, [4, 3], spread(12))
    check(nchw, [1, 2, 2, 3], spread(12))
    check(nhwc, [1, 3, 2, 2], spread(12))

    #: Concatenation with a second part, so the gradient has to be split back
    #: along the axis rather than passed through whole.
    let tail = Tensor([2, 2], positive(4))
    check(fn(x: Tensor) -> Tensor { return cat([x, tail], 1) }, [2, 3], spread(6))
    check(fn(x: Tensor) -> Tensor { return cat([tail, x], 0) }, [2, 2], spread(4))
    tail.free()
}

fn test_reduction_gradients() -> void {
    if not have_gpu() { return }
    check(fn(x: Tensor) -> Tensor { return sum_op(x) }, [6], spread(6))
    #: Only the last axis and the whole tensor are reducible; `sum` says so.
    check(fn(x: Tensor) -> Tensor { return sum_op(x, 1) }, [3, 4], spread(12))
    check(fn(x: Tensor) -> Tensor { return sum_op(x, -1) }, [3, 4], spread(12))
    check(softmax, [3, 4], spread(12))
}

fn test_spatial_gradients() -> void {
    if not have_gpu() { return }
    check(fn(x: Tensor) -> Tensor { return avg_pool2d(x, 2) }, [1, 4, 4, 2], spread(32))
    #: Max pooling picks one element a window, so the window must have a clear
    #: winner or the derivative is not defined there.
    check(fn(x: Tensor) -> Tensor { return max_pool2d(x, 2) }, [1, 4, 4, 1], positive(16))
    check(fn(x: Tensor) -> Tensor { return pad2d(x, [1, 1, 1, 1]) }, [1, 3, 3, 2], spread(18))
    check(fn(x: Tensor) -> Tensor { return upsample_nearest2d(x, 2) }, [1, 2, 2, 2], spread(8))
}

fn test_normalisation_gradients() -> void {
    if not have_gpu() { return }
    let gamma4 = Tensor([2], [1.3, 0.7])
    let beta4 = Tensor([2], [-0.2, 0.5])
    check(fn(x: Tensor) -> Tensor { return batch_norm(x, gamma4, beta4) }, [2, 2, 2, 2], spread(16))
    check(fn(x: Tensor) -> Tensor {
        return group_norm(x, gamma4, beta4, 2)
    }, [2, 2, 2, 2], spread(16))
    gamma4.free()
    beta4.free()

    let gamma2 = Tensor([4], [1.1, 0.9, 1.4, 0.6])
    let beta2 = Tensor([4], [0.2, -0.3, 0.1, 0.4])
    check(fn(x: Tensor) -> Tensor { return layer_norm(x, gamma2, beta2) }, [3, 4], spread(12))
    gamma2.free()
    beta2.free()
}

fn test_matmul_and_dense_gradients() -> void {
    if not have_gpu() { return }
    let right = Tensor([3, 2], positive(6))
    check(fn(x: Tensor) -> Tensor { return matmul(x, right) }, [4, 3], spread(12))
    right.free()

    let left = Tensor([4, 3], spread(12))
    check(fn(x: Tensor) -> Tensor { return matmul(left, x) }, [3, 2], positive(6))
    left.free()

    #: Both transposed forms, which take different arms of the backward.
    let bt = Tensor([2, 3], positive(6))
    check(fn(x: Tensor) -> Tensor { return matmul(x, bt, trans_b: true) }, [4, 3], spread(12))
    bt.free()
    let at = Tensor([3, 4], spread(12))
    check(fn(x: Tensor) -> Tensor { return matmul(at, x, trans_a: true) }, [3, 2], positive(6))
    at.free()

    #: Every activation the fused dense layer can apply, since each has its own
    #: backward inside the same function.
    let weights = Tensor([3, 2], positive(6))
    let bias = Tensor([2], [0.3, -0.4])
    for activation in 0..4 {
        check(fn(x: Tensor) -> Tensor {
            return dense_affine(x, weights, bias, activation)
        }, [4, 3], spread(12))
    }
    weights.free()
    bias.free()
}

fn test_loss_gradients() -> void {
    if not have_gpu() { return }
    let target = Tensor([6], [1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    check(fn(x: Tensor) -> Tensor { return binary_cross_entropy(x, target) }, [6], probabilities(6))
    check(fn(x: Tensor) -> Tensor {
        return binary_cross_entropy_with_logits(x, target)
    }, [6], spread(6))
    target.free()
}

#: Small on purpose. The comparison is a difference of two sums, and every
#: extra output the convolution writes adds float32 rounding to both sides
#: while the derivative being measured stays the same size -- a five by five
#: input over three channels put the noise at half a percent, which is the
#: same order as a real error would be. One image, four by four, keeps the
#: sums short enough that the difference means something.
fn test_convolution_gradients() -> void {
    if not have_gpu() { return }
    let weights = Tensor([3, 3, 2, 2], spread(36))
    let bias = Tensor([2], [0.2, -0.1])
    check(fn(x: Tensor) -> Tensor {
        return conv2d(x, weights, bias: bias, padding: 1)
    }, [1, 4, 4, 2], spread(32))
    check(fn(x: Tensor) -> Tensor {
        return conv2d(x, weights, padding: 0)
    }, [1, 4, 4, 2], spread(32))
    check(fn(x: Tensor) -> Tensor {
        return conv2d(x, weights, stride: 2, padding: 1)
    }, [1, 5, 5, 2], spread(50))
    weights.free()
    bias.free()
}

#: `conv2d`'s fused ReLU epilogue (`activation: 1`), which routes through the
#: native conv+bias+activation graph on the forward pass and a masked
#: upstream on the backward one instead of the ops-level `relu` on top of a
#: separate `conv2d` call above -- its own path through both directions, so
#: `spread`'s sign-alternating values are wrong here: `positive` on every
#: input keeps every pre-activation sum away from ReLU's kink, which
#: `spread`'s cancellation cannot promise.
fn test_convolution_relu_gradients() -> void {
    if not have_gpu() { return }
    let weights = Tensor([3, 3, 2, 2], positive(36))
    let bias = Tensor([2], [0.2, 0.15])
    check(fn(x: Tensor) -> Tensor {
        return conv2d(x, weights, bias: bias, padding: 1, activation: 1)
    }, [1, 4, 4, 2], positive(32))
    weights.free()
    bias.free()

    let image = Tensor([1, 4, 4, 2], positive(32))
    let bias2 = Tensor([2], [0.2, 0.15])
    check(fn(w: Tensor) -> Tensor {
        return conv2d(image, w, bias: bias2, padding: 1, activation: 1)
    }, [3, 3, 2, 2], positive(36))
    let weights2 = Tensor([3, 3, 2, 2], positive(36))
    check(fn(b: Tensor) -> Tensor {
        return conv2d(image, weights2, bias: b, padding: 1, activation: 1)
    }, [2], [0.2, 0.15])
    image.free()
    bias2.free()
    weights2.free()
}

#: The weights' and bias's own gradients, which the input-side cases above do
#: not touch: `check` perturbs whatever tensor it is given, and a wrong weight
#: gradient with a right input gradient trains just as wrongly.
fn test_parameter_side_gradients() -> void {
    if not have_gpu() { return }
    let image = Tensor([1, 4, 4, 2], spread(32))
    check(fn(w: Tensor) -> Tensor { return conv2d(image, w, padding: 1) }, [3, 3, 2, 2], spread(36))
    let weights = Tensor([3, 3, 2, 2], spread(36))
    check(fn(b: Tensor) -> Tensor {
        return conv2d(image, weights, bias: b, padding: 1)
    }, [2], [0.2, -0.1])
    image.free()
    weights.free()

    let flat = Tensor([4, 3], spread(12))
    let bias = Tensor([2], [0.3, -0.4])
    check(fn(w: Tensor) -> Tensor { return dense_affine(flat, w, bias, 0) }, [3, 2], positive(6))
    let weights2 = Tensor([3, 2], positive(6))
    check(fn(b: Tensor) -> Tensor { return dense_affine(flat, weights2, b, 0) }, [2], [0.3, -0.4])
    flat.free()
    bias.free()
    weights2.free()

    let normalised = Tensor([3, 4], spread(12))
    let beta = Tensor([4], [0.2, -0.3, 0.1, 0.4])
    check(fn(g: Tensor) -> Tensor {
        return layer_norm(normalised, g, beta)
    }, [4], [1.1, 0.9, 1.4, 0.6])
    let gamma = Tensor([4], [1.1, 0.9, 1.4, 0.6])
    check(fn(b: Tensor) -> Tensor {
        return layer_norm(normalised, gamma, b)
    }, [4], [0.2, -0.3, 0.1, 0.4])
    normalised.free()
    beta.free()
    gamma.free()
}

#: `gather_rows` differentiates in its table, not in its indices. A row picked
#: twice must collect both gradients, which is the case a naive scatter drops.
fn test_gather_rows_accumulates_repeated_indices() -> void {
    if not have_gpu() { return }
    let indices = Tensor([4], [2.0, 0.0, 2.0, 1.0])
    check(fn(t: Tensor) -> Tensor { return gather_rows(t, indices) }, [3, 2], spread(6))
    indices.free()
}

#: Not trained through, but it must not corrupt what passes: with `training`
#: false it is the identity, and its gradient is one.
fn test_dropout_is_transparent_when_not_training() -> void {
    if not have_gpu() { return }
    check(fn(x: Tensor) -> Tensor { return jt.dropout(x, 0.5, false) }, [6], spread(6))
}

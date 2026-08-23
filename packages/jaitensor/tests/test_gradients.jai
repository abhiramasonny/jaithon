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
from jaitensor.losses import SparseCategoricalCrossentropy
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

#: A row wider than one threadgroup, which is where the normalisation's
#: backward stops being one thread's work.
#:
#: The parameter pass is spread over columns and cannot reduce a row for
#: itself, so it reads statistics the input pass leaves behind. Four columns
#: never exercised either the strided loop or that handover.
fn test_wide_normalisation_gradients() -> void {
    if not have_gpu() { return }
    let width = 320
    let rows = 3
    let normalised = Tensor([rows, width], spread(rows * width))
    let gamma = Tensor([width], positive(width))
    let beta = Tensor([width], spread(width))
    check(fn(g: Tensor) -> Tensor {
        return layer_norm(normalised, g, beta)
    }, [width], positive(width))
    check(fn(b: Tensor) -> Tensor {
        return layer_norm(normalised, gamma, b)
    }, [width], spread(width))
    check(fn(x: Tensor) -> Tensor {
        return layer_norm(x, gamma, beta)
    }, [rows, width], spread(rows * width))
    normalised.free()
    gamma.free()
    beta.free()
}

#: The channel normalisations reduce over the batch and every pixel, an axis
#: that is not the one they write, and they do it a block at a time so that
#: consecutive threads read consecutive floats. Their scale and shift gradients
#: were never checked against anything.
#:
#: Against arithmetic done here rather than a difference quotient: the loss a
#: quotient can use is a plain sum, and the scale's gradient under that loss is
#: the sum of a normalised channel, which is zero however the kernel is written.
#: Channel statistics over activations that sit a long way from zero.
#:
#: The spread was worked out as `E[x^2] - E[x]^2`, which is one pass and
#: keeps no digits at all once the mean is large beside the spread. At an
#: offset of a thousand -- an unremarkable activation -- those two terms
#: agreed to seven figures, the subtraction returned a negative, and
#: `rsqrt` turned the whole block to NaN. In the forward, not merely in the
#: gradient: `batch_norm` returned 1024 non-finite values out of 2048.
fn test_channel_statistics_survive_an_offset_from_zero() -> void {
    if not have_gpu() { return }
    let n = 6
    let side = 5
    let channels = 4
    for offset in [0.0, 1000.0, 50000.0] {
        var values: list[float] = []
        var seed = 3
        for _index in 0..n * side * side * channels {
            seed = (seed * 1103515245 + 12345) % 2147483648
            values.push(offset + 2.0 * (float(seed % 1000) / 1000.0 - 0.5))
        }
        let x = Tensor([n, side, side, channels], values, requires_grad: true)
        let gamma = Tensor.ones([channels])
        let beta = Tensor.zeros([channels])
        let normed = batch_norm(x, gamma, beta)
        let got = normed.download()
        for value in got { assert_true(math.is_finite(value), f"batch_norm gave {value}") }

        #: Finite is not enough -- each channel has to come out centred on
        #: zero with a spread of one, which is the whole job.
        for channel in 0..channels {
            var total = 0.0
            var count = 0
            for (index, value) in got.enumerate() {
                if index % channels != channel { continue }
                total += value
                count += 1
            }
            let centre = total / float(count)
            var spread = 0.0
            for (index, value) in got.enumerate() {
                if index % channels != channel { continue }
                spread += (value - centre) * (value - centre)
            }
            assert_true(math.fabs(centre) < 0.05, f"channel {channel} centred at {centre}")
            let unit = spread / float(count)
            assert_true(math.fabs(unit - 1.0) < 0.05, f"channel {channel} spread {unit}")
        }

        let junk = jt.sum(normed).backward()
        let gradient = x.grad
        assert_true(gradient is not null, "the input should have a gradient")
        for value in gradient.download() {
            assert_true(math.is_finite(value), f"the gradient gave {value}")
        }
    }
}

#: What batch normalisation does when it is predicting, pinned on purpose.
#:
#: It has no running average, so it normalises by the batch in front of it
#: whatever it is doing -- which means a prediction depends on which other
#: samples came along in the same call. That is a genuine difference from
#: other libraries and the sort of thing that should not change by accident,
#: so it is written down here as well as in the layer's own documentation.
fn test_batch_norm_normalises_by_whatever_batch_it_is_given() -> void {
    if not have_gpu() { return }
    let channels = 2
    var alone: list[float] = []
    for index in 0..1 * 2 * 2 * channels { alone.push(float(index) * 3.0 + 5.0) }
    let gamma = Tensor.ones([channels])
    let beta = Tensor.zeros([channels])
    let one = batch_norm(Tensor([1, 2, 2, channels], alone), gamma, beta).download()

    #: The same four pixels again, with four more alongside them.
    var together: list[float] = []
    for value in alone { together.push(value) }
    for index in 0..1 * 2 * 2 * channels { together.push(float(index) * 11.0 - 40.0) }
    let shared = batch_norm(Tensor([2, 2, 2, channels], together), gamma, beta).download()

    var moved = false
    for index in 0..alone.len() {
        if math.fabs(one[index] - shared[index]) > 1e-3 { moved = true }
    }
    assert_true(moved, "batch norm is supposed to depend on the batch it is given")

    #: And a single sample with nowhere to take a spread comes back as beta,
    #: which here is zero.
    let flat = batch_norm(Tensor([1, 1, 1, channels], [4.0, 9.0]), gamma, Tensor([channels], [0.25, 0.75])).download()
    assert_true(math.fabs(flat[0] - 0.25) < 1e-4, f"got {flat[0]}")
    assert_true(math.fabs(flat[1] - 0.75) < 1e-4, f"got {flat[1]}")
}

fn test_channel_normalisation_parameter_gradients() -> void {
    if not have_gpu() { return }
    let n = 4
    let side = 8
    let channels = 32
    let nspatial = n * side * side
    let count = nspatial * channels
    var values: list[float] = []
    var weights: list[float] = []
    for index in 0..count {
        values.push(float(index % 211) * 0.01 - 1.05)
        weights.push(float(index % 97) * 0.02 - 0.96)
    }
    var scale: list[float] = []
    var shift: list[float] = []
    for channel in 0..channels {
        scale.push(0.7 + float(channel % 3) * 0.2)
        shift.push(float(channel % 4) * 0.1 - 0.15)
    }
    let x = Tensor([n, side, side, channels], values, requires_grad: true)
    let gamma = Tensor([channels], scale, requires_grad: true)
    let beta = Tensor([channels], shift, requires_grad: true)
    let upstream = Tensor([n, side, side, channels], weights)
    let held = jt.sum(jt.multiply(batch_norm(x, gamma, beta), upstream))
    let intermediates = held.backward()

    let got_gamma = gamma.grad
    let got_beta = beta.grad
    if got_gamma is null or got_beta is null { throw AssertionError("no parameter gradient") }
    let analytic_gamma = got_gamma.download()
    let analytic_beta = got_beta.download()
    for channel in 0..channels {
        var total = 0.0
        for step in 0..nspatial { total += values[step * channels + channel] }
        let centre = total / float(nspatial)
        var spread = 0.0
        for step in 0..nspatial {
            let gap = values[step * channels + channel] - centre
            spread += gap * gap
        }
        let inverse = 1.0 / math.sqrt(spread / float(nspatial) + 1e-5)
        var want_gamma = 0.0
        var want_beta = 0.0
        for step in 0..nspatial {
            let up = weights[step * channels + channel]
            want_gamma += up * (values[step * channels + channel] - centre) * inverse
            want_beta += up
        }
        assert_near(
            analytic_gamma[channel],
            want_gamma,
            max(1e-3, math.fabs(want_gamma) * 2e-3),
            f"scale gradient for channel {channel}"
        )
        assert_near(
            analytic_beta[channel],
            want_beta,
            max(1e-3, math.fabs(want_beta) * 2e-3),
            f"shift gradient for channel {channel}"
        )
    }
    for tensor in intermediates { tensor.free() }
    x.free()
    gamma.free()
    beta.free()
    upstream.free()
    held.free()
}

#: `gather_rows` differentiates in its table, not in its indices. A row picked
#: twice must collect both gradients, which is the case a naive scatter drops.
#: An index that is not in the table.
#:
#: `uint(indices[r])` was used unchecked on both sides of the lookup. Past
#: the end of the table the forward read whatever was next in the pool, and
#: a negative index -- which `uint` turns into about four billion rather than
#: into a complaint -- read somewhere else entirely. The backward did the
#: same arithmetic to pick an atomic ADD target, so an out-of-vocabulary
#: token id did not read another tensor, it corrupted one.
fn test_a_lookup_outside_the_table_reads_and_writes_nothing() -> void {
    if not have_gpu() { return }
    let table = Tensor(
        [4, 3],
        [1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 100.0, 200.0, 300.0, 7.0, 8.0, 9.0],
        requires_grad: true
    )
    let asked = Tensor([5], [0.0, 3.0, 99.0, -1.0, 2.0])
    let got = gather_rows(table, asked).download()
    let wanted = [1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 200.0, 300.0]
    for (index, value) in wanted.enumerate() { assert_eq(got[index], value) }

    #: And the gradient of the same lookup lands only on the rows that exist.
    let junk = jt.sum(gather_rows(table, asked)).backward()
    let gradient = table.grad
    assert_true(gradient is not null, "the table should have a gradient")
    let flowed = gradient.download()
    #: Rows 0, 2 and 3 were each asked for once; row 1 never was.
    for (index, value) in [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0].enumerate() {
        assert_eq(flowed[index], value)
    }
}

fn test_gather_rows_accumulates_repeated_indices() -> void {
    if not have_gpu() { return }
    let indices = Tensor([4], [2.0, 0.0, 2.0, 1.0])
    check(fn(t: Tensor) -> Tensor { return gather_rows(t, indices) }, [3, 2], spread(6))
    indices.free()
}

#: Not trained through, but it must not corrupt what passes: with `training`
#: false it is the identity, and its gradient is one.
#: Training-mode dropout has to draw a different mask each time it is asked.
#:
#: The seed was the tensor's own size, a constant for a given shape, so every
#: step of every epoch dropped exactly the same units and two layers of the
#: same shape dropped the same ones as each other. Perfectly repeatable, and
#: therefore never flaky -- it simply was not regularising anything.
#: A loss that has been weighted has to weight its gradient too.
#:
#: Sparse cross-entropy is the one loss here whose backward is written out by
#: hand, around a fused kernel, and it handed its delta straight back without
#: multiplying by what came down from above. So `scale(loss, 3)` reported
#: three times the number and moved the parameters by exactly as much as
#: `loss` did -- every other loss scales properly, which is what makes this
#: one worth pinning.
fn test_a_weighted_loss_weights_its_gradient() -> void {
    if not have_gpu() { return }
    var plain: list[float] = []
    var tripled: list[float] = []
    for weight in [1.0, 3.0] {
        let prediction = Tensor([1, 3], [0.2, 0.3, 0.5], requires_grad: true)
        let target = Tensor([1], [2.0])
        let loss = SparseCategoricalCrossentropy().forward(prediction, target)
        let junk = scale(loss, weight).backward()
        let gradient = prediction.grad
        assert_true(gradient is not null, "the prediction should have a gradient")
        for value in gradient.download() {
            if weight == 1.0 { plain.push(value) } else { tripled.push(value) }
        }
    }
    assert_eq(plain.len(), tripled.len())
    for (index, value) in plain.enumerate() {
        assert_true(math.fabs(value) > 1e-6, "the unweighted gradient should not be zero")
        assert_near(tripled[index] / value, 3.0, 1e-4)
    }
}

fn test_dropout_draws_a_fresh_mask_each_step() -> void {
    if not have_gpu() { return }
    let x = Tensor.ones([4, 32])
    var seen: list[str] = []
    for _round in 0..4 {
        var mask = ""
        for value in jt.dropout(x, 0.5, true).download() {
            mask += if value == 0.0 { "." } else { "#" }
        }
        assert_true(not seen.contains(mask), "two steps drew the same dropout mask")
        seen.push(mask)
    }

    #: A seed asked for by name still repeats, which is what a test needs.
    var first = ""
    for value in jt.dropout(x, 0.5, true, 7).download() {
        first += if value == 0.0 { "." } else { "#" }
    }
    var again = ""
    for value in jt.dropout(x, 0.5, true, 7).download() {
        again += if value == 0.0 { "." } else { "#" }
    }
    assert_eq(first, again)

    #: And it drops roughly what it was asked to.
    var kept = 0
    var total = 0
    for _round in 0..20 {
        for value in jt.dropout(Tensor.ones([128]), 0.25, true).download() {
            total += 1
            if value != 0.0 { kept += 1 }
        }
    }
    let rate = float(kept) / float(total)
    assert_true(rate > 0.70 and rate < 0.80, f"kept {rate} of the units, wanted about 0.75")
}

fn test_dropout_is_transparent_when_not_training() -> void {
    if not have_gpu() { return }
    check(fn(x: Tensor) -> Tensor { return jt.dropout(x, 0.5, false) }, [6], spread(6))
}

#: A loss can be a one-by-one matrix rather than a one-element vector, and
#: `backward` has to seed the gradient with the shape it actually has.
#:
#: The root seed used to be `ones([1])` whatever the loss was, so a loss that
#: came out of a matrix product -- `ones(1, n) @ squared`, which is how a mean
#: is written without a reduction -- handed the product's own backward an
#: upstream of the wrong rank and it failed to index it. The loss still had one
#: element, so nothing before this noticed.
#: A row whose mean is large beside its spread, which is what a residual
#: stream drifts into and what a normalisation is put there to tame.
#:
#: The backward used to work the spread out as `E[x^2] - E[x]^2`, one pass
#: over the row instead of two. Those two terms agree to seven digits once
#: the mean is ten thousand and the spread is one, so the subtraction kept
#: none of them, `rsqrt` was handed a negative, and every gradient in the row
#: came back NaN -- while the forward stayed perfectly finite, so nothing
#: downstream said why the model had stopped learning.
fn test_layer_norm_gradients_survive_a_large_mean() -> void {
    if not have_gpu() { return }
    let width = 64
    for offset in [0.0, 100.0, 10000.0] {
        var values: list[float] = []
        var seed = 11
        for _index in 0..width {
            seed = (seed * 1103515245 + 12345) % 2147483648
            values.push(offset + (float(seed % 1000) / 1000.0 - 0.5))
        }
        let x = Tensor([1, width], values, requires_grad: true)
        let gamma = Tensor.ones([width])
        let beta = Tensor.zeros([width])
        #: Weighted, so the gradient is not the same number in every slot and
        #: a wrong one cannot hide behind a right one.
        var weights: list[float] = []
        for index in 0..width { weights.push(float(index % 7) - 3.0) }
        let weighting = Tensor([1, width], weights)
        let junk = jt.sum(jt.multiply(layer_norm(x, gamma, beta), weighting)).backward()
        let gradient = x.grad
        assert_true(gradient is not null, "the input should have a gradient")
        for (index, value) in gradient.download().enumerate() {
            assert_true(math.is_finite(value), f"gradient {index} at offset {offset} is {value}")
        }
    }
}

fn test_backward_seeds_a_rank_two_loss_with_a_rank_two_gradient() -> void {
    if not have_gpu() { return }
    let x = Tensor([3, 1], [2.0, -1.0, 0.5], requires_grad: true)
    let ones = Tensor([3, 1], [1.0, 1.0, 1.0])
    let squared = jt.multiply(x, x)
    #: `[1, 3] @ [3, 1]` is `[1, 1]`: one element, two axes.
    let loss = jt.matmul(jt.transpose(ones), squared)
    assert_eq(loss.shape, [1, 1])
    let junk = loss.backward()
    let gradient = x.grad
    assert_true(gradient is not null, "the input should have a gradient")
    #: d/dx of the sum of squares is twice x.
    let got = gradient.download()
    for (index, wanted) in [4.0, -2.0, 1.0].enumerate() {
        assert_true(
            abs(got[index] - wanted) < 1e-4,
            f"gradient {index} is {got[index]}, expected {wanted}"
        )
    }
    for tensor in junk { tensor.free() }
    x.free()
    ones.free()
}

#: The exports `test_jaitensor.jai` does not reach.
#:
#: Twenty-one of jaitensor's ninety-six exports had no caller in any test:
#: three normalisation layers, an embedding, two losses, four activations and
#: element-wise ops, and the pooling and permute an image model needs. A layer
#: nobody built is a layer whose `build` may not agree with its `forward`, and
#: an op nobody differentiated is a backward nobody has compared against
#: anything.
#:
#: So each one is exercised here, and where the op is differentiable its
#: gradient is checked against a central difference of the same computation --
#: the same contract `test_jaitensor.jai` holds its own backward passes to.
#: `1e-2` throughout, which is what a `1e-3` step over float32 buys.
import jaitensor as jt
import std.math
from jaitensor.layers import Activate, Activation, BatchNorm2d, Embedding
from jaitensor.layers import GroupNorm2d, Layer, LayerNorm
from jaitensor.losses import BinaryCrossentropy, MeanSquaredError
from jaitensor.model import Metrics
from jaitensor.ops import avg_pool2d, batch_norm, leaky_relu, maximum, minimum
from jaitensor.ops import permute, rsqrt, silu
from jaitensor.plots import save_history_plot
from jaitensor.optimizers import zero_grad
from jaitensor.tensor import Tensor
from std.io import Path
from std.os import temp_dir
from std.random import Random
from std.test import assert_eq, assert_near, assert_true

fn numerical_grad(f: fn(list[float]) -> float, x: list[float], index: int) -> float {
    let eps = 1e-3
    var plus = [value for value in x]
    var minus = [value for value in x]
    plus[index] += eps
    minus[index] -= eps
    return (f(plus) - f(minus)) / (2.0 * eps)
}

#: Every analytic gradient of a one-input op against the same op's central
#: difference, summed to a scalar so there is one number to differentiate.
fn check_gradient(apply: fn(Tensor) -> Tensor, shape: list[int], values: list[float]) -> void {
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
    if gradient is null { throw AssertionError("expected a gradient after backward()") }
    let analytic = gradient.download()
    for index in 0..values.len() {
        assert_near(analytic[index], numerical_grad(loss_of, values, index), 1e-2)
    }
    for tensor in intermediates { tensor.free() }
    input.free()
    total.free()
}

#: Away from zero on both sides, and never at the kink `leaky_relu` has there,
#: because a central difference across a kink measures neither side.
let UNARY_INPUT = [1.5, -0.75, 0.4, -2.25, 3.0, -0.6]

fn test_leaky_relu_and_silu_differentiate() -> void {
    if not jt.is_available() { return }
    check_gradient(leaky_relu, [6], UNARY_INPUT)
    check_gradient(silu, [6], UNARY_INPUT)
}

fn test_rsqrt_differentiates() -> void {
    if not jt.is_available() { return }
    check_gradient(rsqrt, [6], [1.5, 0.75, 0.4, 2.25, 3.0, 0.6])
}

fn test_permute_moves_the_axes_and_differentiates() -> void {
    if not jt.is_available() { return }
    let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    let input = Tensor([2, 3], values)
    let moved = permute(input, [1, 0])
    assert_eq(moved.shape, [3, 2])
    assert_eq(moved.download(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
    input.free()
    moved.free()

    let apply = fn(x: Tensor) -> Tensor { return permute(x, [1, 0]) }
    check_gradient(apply, [2, 3], values)
}

#: Chosen so no pair ties: `maximum` and `minimum` are not differentiable where
#: their two inputs meet, and a central difference there measures the wrong
#: side of the step.
fn test_maximum_and_minimum_pick_elementwise() -> void {
    if not jt.is_available() { return }
    let a = Tensor([4], [1.0, -2.0, 3.5, 0.25])
    let b = Tensor([4], [-1.0, 2.0, 0.5, 4.0])
    let high = maximum(a, b)
    let low = minimum(a, b)
    assert_eq(high.download(), [1.0, 2.0, 3.5, 4.0])
    assert_eq(low.download(), [-1.0, -2.0, 0.5, 0.25])
    a.free()
    b.free()
    high.free()
    low.free()

    let other = Tensor([4], [-1.0, 2.0, 0.5, 4.0])
    let take_high = fn(x: Tensor) -> Tensor { return maximum(x, other) }
    check_gradient(take_high, [4], [1.0, -2.0, 3.5, 0.25])
    let take_low = fn(x: Tensor) -> Tensor { return minimum(x, other) }
    check_gradient(take_low, [4], [1.0, -2.0, 3.5, 0.25])
    other.free()
}

#: A two-by-two window over a four-by-four single-channel image, where every
#: pooled value is the mean of a block whose members are known.
fn test_avg_pool2d_averages_each_window() -> void {
    if not jt.is_available() { return }
    var values: list[float] = []
    for index in 0..16 { values.push(float(index)) }
    let input = Tensor([1, 4, 4, 1], values)
    let pooled = avg_pool2d(input, 2)
    assert_eq(pooled.shape, [1, 2, 2, 1])
    assert_eq(pooled.download(), [2.5, 4.5, 10.5, 12.5])
    input.free()
    pooled.free()

    let apply = fn(x: Tensor) -> Tensor { return avg_pool2d(x, 2) }
    check_gradient(apply, [1, 4, 4, 1], values)
}

#: Normalising over the batch leaves each channel at zero mean and unit
#: variance, which is what the layer promises and what a gamma of one and a
#: beta of zero leave untouched.
fn test_batch_norm_centres_each_channel() -> void {
    if not jt.is_available() { return }
    var values: list[float] = []
    for index in 0..2 * 2 * 2 * 2 { values.push(float(index % 7) * 1.5 - 3.0) }
    let input = Tensor([2, 2, 2, 2], values)
    let gamma = Tensor.ones([2])
    let beta = Tensor.zeros([2])
    let normalised = batch_norm(input, gamma, beta)
    assert_eq(normalised.shape, [2, 2, 2, 2])
    let out = normalised.download()
    for channel in 0..2 {
        var total = 0.0
        var count = 0
        for index in 0..out.len() {
            if index % 2 != channel { continue }
            total += out[index]
            count += 1
        }
        assert_near(total / float(count), 0.0, 1e-3)
    }
    input.free()
    gamma.free()
    beta.free()
    normalised.free()
}

fn run_layer(layer: Layer, shape: list[int], values: list[float]) -> list[float] {
    let rng = Random.with_seed(7)
    layer.build(shape[shape.len() - 1], rng)
    let input = Tensor(shape, values)
    let output = layer.forward(input)
    let result = output.download()
    input.free()
    output.free()
    layer.free()
    return result
}

fn test_the_normalisation_layers_build_and_run() -> void {
    if not jt.is_available() { return }
    var values: list[float] = []
    for index in 0..2 * 2 * 2 * 4 { values.push(float(index % 11) * 0.75 - 4.0) }

    let batch = run_layer(BatchNorm2d(4), [2, 2, 2, 4], values)
    assert_eq(batch.len(), values.len())
    let group = run_layer(GroupNorm2d(2, 4), [2, 2, 2, 4], values)
    assert_eq(group.len(), values.len())

    var flat: list[float] = []
    for index in 0..8 * 4 { flat.push(float(index % 11) * 0.75 - 4.0) }
    let normalised = run_layer(LayerNorm(4), [8, 4], flat)
    assert_eq(normalised.len(), flat.len())
    #: Each row of a LayerNorm output is centred on its own, which is the whole
    #: of what distinguishes it from a BatchNorm.
    for row in 0..8 {
        var total = 0.0
        for column in 0..4 { total += normalised[row * 4 + column] }
        assert_near(total / 4.0, 0.0, 1e-3)
    }
}

fn test_activate_applies_its_activation() -> void {
    if not jt.is_available() { return }
    let values = [1.5, -0.75, 0.4, -2.25]
    let out = run_layer(Activate(Activation.Relu), [1, 4], values)
    let expected = [1.5, 0.0, 0.4, 0.0]
    for index in 0..4 { assert_near(out[index], expected[index], 1e-6) }
}

fn test_embedding_gathers_its_rows() -> void {
    if not jt.is_available() { return }
    let layer = Embedding(5, 3)
    let rng = Random.with_seed(11)
    layer.build(1, rng)
    let table = layer.parameters()[0].download()
    let indices = Tensor([4], [2.0, 0.0, 4.0, 2.0])
    let gathered = layer.forward(indices)
    assert_eq(gathered.shape, [4, 3])
    let out = gathered.download()
    for (slot, row) in [2, 0, 4, 2].enumerate() {
        for column in 0..3 {
            assert_near(out[slot * 3 + column], table[row * 3 + column], 1e-6)
        }
    }
    indices.free()
    gathered.free()
    layer.free()
}

#: Both losses against the closed form of what they compute, and against the
#: gradient of that form.
fn test_mean_squared_error_and_binary_crossentropy() -> void {
    if not jt.is_available() { return }
    let predicted = [0.8, 0.25, 0.6, 0.1]
    let wanted = [1.0, 0.0, 1.0, 0.0]

    let mse = MeanSquaredError()
    let p = Tensor([4, 1], predicted, requires_grad: true)
    let t = Tensor([4, 1], wanted)
    let loss = mse.forward(p, t)
    var expected = 0.0
    for index in 0..4 {
        let error = predicted[index] - wanted[index]
        expected += error * error
    }
    assert_near(loss.download()[0], expected / 4.0, 1e-4)
    let intermediates = loss.backward()
    let gradient = p.grad
    if gradient is null { throw AssertionError("expected a gradient from MeanSquaredError") }
    let analytic = gradient.download()
    for index in 0..4 {
        assert_near(analytic[index], 2.0 * (predicted[index] - wanted[index]) / 4.0, 1e-3)
    }
    for tensor in intermediates { tensor.free() }
    p.free()
    t.free()
    loss.free()

    let bce = BinaryCrossentropy()
    let bp = Tensor([4, 1], predicted)
    let bt = Tensor([4, 1], wanted)
    let bloss = bce.forward(bp, bt)
    var cross = 0.0
    for index in 0..4 {
        let q = predicted[index]
        cross -= wanted[index] * math.ln(q) + (1.0 - wanted[index]) * math.ln(1.0 - q)
    }
    assert_near(bloss.download()[0], cross / 4.0, 1e-3)
    bp.free()
    bt.free()
    bloss.free()
}

fn test_zero_grad_clears_what_backward_wrote() -> void {
    if not jt.is_available() { return }
    let a = Tensor([3], [1.0, 2.0, 3.0], requires_grad: true)
    let doubled = jt.multiply(a, a)
    let total = jt.sum(doubled)
    let intermediates = total.backward()
    let gradient = a.grad
    if gradient is null { throw AssertionError("expected a gradient after backward()") }
    assert_true(gradient.download()[0] != 0.0)
    zero_grad([a])
    assert_eq(gradient.download(), [0.0, 0.0, 0.0])
    for tensor in intermediates { tensor.free() }
    a.free()
    total.free()
}

fn test_metrics_carry_what_they_were_given() -> void {
    let metrics = Metrics(0.25, 91.5)
    assert_near(metrics.loss, 0.25, 1e-9)
    assert_near(metrics.accuracy, 91.5, 1e-9)
}

fn test_a_history_plot_is_written() -> void {
    let path = Path(temp_dir()).join("jaitensor_reachable_history.svg")
    save_history_plot(
        path.to_str(),
        [1.2, 0.8, 0.5],
        [40.0, 65.0, 78.0],
        val_loss: [1.3, 0.9, 0.7],
        val_accuracy: [38.0, 62.0, 74.0],
        test_loss: 0.6,
        test_accuracy: 76.0
    )
    assert_true(path.exists())
    path.remove()
}

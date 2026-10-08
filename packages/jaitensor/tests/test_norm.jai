#: Layer normalisation's parameter gradients at the sizes a transformer uses.
#:
#: test_gradients.jai checks `layer_norm` against a central difference, but at
#: three rows, where the column sums are one strip and the join never runs.
#: These sizes split the rows into strips, give a width that is not a whole
#: number of 32-column blocks, and are checked against the host in float64.
import jaitensor as jt
import std.math
from jaitensor.compiled import compile_step, refusal
from jaitensor.layers import Activation, AvgPool2d, BatchNorm2d, Conv2d, Dense, Flatten, Layer
from jaitensor.losses import SparseCategoricalCrossentropy
from jaitensor.optimizers import SGD, prepare_state
from jaitensor.tensor import Tensor
from std.os import env, set_env, unset_env
from std.random import Random
from std.test import assert_eq, assert_near, assert_throws, assert_true

fn wave(count: int, phase: float, amplitude: float, offset: float) -> list[float] {
    var out: list[float] = []
    for index in 0..count { out.push(offset + amplitude * math.sin(float(index) * 0.377 + phase)) }
    return out
}

fn check_parameter_gradients(rows: int, width: int) -> void {
    let xs = wave(rows * width, 0.3, 2.0, 0.5)
    let ws = wave(rows * width, 1.9, 1.0, 0.0)
    let gs = wave(width, 0.7, 0.5, 1.0)
    let x = Tensor([rows, width], xs, requires_grad: true)
    let gamma = Tensor([width], gs, requires_grad: true)
    let beta = Tensor([width], wave(width, 2.2, 0.3, 0.0), requires_grad: true)
    let w = Tensor([rows, width], ws)
    let y = jt.layer_norm(x, gamma, beta)
    let product = jt.multiply(y, w)
    let total = jt.sum(product)
    let held = total.backward()
    let gg = gamma.grad
    let gb = beta.grad
    assert_true(gg is not null and gb is not null)
    if gg is null or gb is null { return }
    let grad_gamma = gg.download()
    let grad_beta = gb.download()
    var want_gamma: list[float] = [0.0 for _ in 0..width]
    var want_beta: list[float] = [0.0 for _ in 0..width]
    for r in 0..rows {
        var mean = 0.0
        for c in 0..width { mean += xs[r * width + c] }
        mean /= float(width)
        var spread = 0.0
        for c in 0..width {
            let d = xs[r * width + c] - mean
            spread += d * d
        }
        let inv = 1.0 / math.sqrt(spread / float(width) + 1e-5)
        for c in 0..width {
            let dy = ws[r * width + c]
            want_gamma[c] += dy * (xs[r * width + c] - mean) * inv
            want_beta[c] += dy
        }
    }
    assert_eq(grad_gamma.len(), width)
    for c in 0..width {
        assert_near(grad_gamma[c], want_gamma[c], 1e-3 + 1e-4 * math.fabs(want_gamma[c]))
        assert_near(grad_beta[c], want_beta[c], 1e-3 + 1e-4 * math.fabs(want_beta[c]))
    }
    for tensor in held { tensor.free() }
    total.free()
    x.free()
    gamma.free()
    beta.free()
    w.free()
}

fn test_layer_norm_parameter_gradients_at_transformer_sizes() -> void {
    if not jt.is_available() { return }
    check_parameter_gradients(3, 8)
    check_parameter_gradients(130, 100)
    check_parameter_gradients(1024, 768)
    check_parameter_gradients(4096, 64)
}

fn grads_of(tensors: list[Tensor]) -> list[list[float]] {
    var out: list[list[float]] = []
    for tensor in tensors {
        let g = tensor.grad
        if g is null {
            out.push([])
        } else {
            out.push(g.download())
            g.free()
            tensor.grad = null
        }
    }
    return out
}

fn assert_all_near(got: list[float], want: list[float], label: str) -> void {
    assert_eq(got.len(), want.len(), label)
    for index in 0..want.len() {
        let tolerance = 1e-5 * (1.0 + math.fabs(want[index]))
        assert_true(
            math.fabs(got[index] - want[index]) <= tolerance,
            f"{label}[{index}]: fused {got[index]} composed {want[index]}"
        )
    }
}

#: `batch_norm(x, residual: s, relu: true)` against the operations it fuses,
#: `relu(add(batch_norm(x), s))`, forward and every gradient, at shapes that
#: take each reduction path: channel counts that divide 256 with enough rows
#: for the coalesced partials, too few rows for them, and channel counts that
#: neither divide 256 nor come in fours.
fn check_fused_against_composed(shape: list[int], with_skip: bool, with_relu: bool) -> void {
    let count = shape[0] * shape[1] * shape[2] * shape[3]
    let channels = shape[3]
    let x = Tensor(shape, wave(count, 0.3, 2.0, 0.5), requires_grad: true)
    let skip = Tensor(shape, wave(count, 2.1, 1.5, -0.2), requires_grad: true)
    let gamma = Tensor([channels], wave(channels, 0.7, 0.5, 1.0), requires_grad: true)
    let beta = Tensor([channels], wave(channels, 2.2, 0.3, 0.0), requires_grad: true)
    let w = Tensor(shape, wave(count, 1.9, 1.0, 0.0))
    let watched = [x, gamma, beta, skip]
    let label = f"{shape} skip={with_skip} relu={with_relu}"

    let fused = jt.batch_norm(x, gamma, beta, residual: if with_skip { skip } else { null }, relu: with_relu)
    let fused_values = fused.download()
    let fused_total = jt.sum(jt.multiply(fused, w))
    for held in fused_total.backward() { held.free() }
    fused_total.free()
    let fused_grads = grads_of(watched)

    var composed = jt.batch_norm(x, gamma, beta)
    if with_skip { composed = jt.add(composed, skip) }
    if with_relu { composed = jt.relu(composed) }
    let composed_values = composed.download()
    let composed_total = jt.sum(jt.multiply(composed, w))
    for held in composed_total.backward() { held.free() }
    composed_total.free()
    let composed_grads = grads_of(watched)

    assert_all_near(fused_values, composed_values, f"{label} forward")
    let names = ["x", "gamma", "beta", "skip"]
    for index in 0..watched.len() {
        if index == 3 and not with_skip { continue }
        assert_true(fused_grads[index].len() > 0, f"{label} {names[index]} has no gradient")
        assert_all_near(fused_grads[index], composed_grads[index], f"{label} grad {names[index]}")
    }
    for tensor in [x, skip, gamma, beta, w] { tensor.free() }
}

fn test_fused_batch_norm_matches_the_composed_operations() -> void {
    if not jt.is_available() { return }
    for shape in [[8, 8, 8, 32], [4, 8, 8, 64], [2, 2, 2, 4], [3, 4, 5, 6], [2, 8, 8, 3], [16, 4, 4, 128]] {
        check_fused_against_composed(shape, true, true)
        check_fused_against_composed(shape, false, true)
        check_fused_against_composed(shape, true, false)
    }
}

fn test_batch_norm_layer_relu_and_residual_match_the_composed_operations() -> void {
    if not jt.is_available() { return }
    let shape = [4, 8, 8, 32]
    let count = 4 * 8 * 8 * 32
    let rng = Random.with_seed(5)
    let x = Tensor(shape, wave(count, 0.3, 2.0, 0.5))
    let skip = Tensor(shape, wave(count, 2.1, 1.5, -0.2))
    let layer = jt.BatchNorm2d(relu: true)
    layer.build_from_shape([8, 8, 32], rng)
    let plain = jt.BatchNorm2d()
    plain.build_from_shape([8, 8, 32], rng)
    let normed = plain.forward(x)
    let summed = jt.add(normed, skip)
    let want_relu = jt.relu(normed).download()
    let want_res = jt.relu(summed).download()
    assert_all_near(layer.forward(x).download(), want_relu, "BatchNorm2d(relu: true)")
    assert_all_near(layer.forward_residual(x, skip).download(), want_res, "forward_residual")
    assert_true(layer.relu and not plain.relu)
    assert_throws(ValueError, fn() { layer.forward_residual(x, Tensor([4, 8, 8, 16])) })
}

fn bn_relu_layers(seed: int) -> list[Layer] {
    let rng = Random.with_seed(seed)
    let layers: list[Layer] = [
        Conv2d(8, 3, padding: 1, in_channels: 3),
        BatchNorm2d(relu: true),
        AvgPool2d(2),
        Flatten(),
        Dense(5, activation: Activation.Linear),
    ]
    var current = [8, 8, 3]
    for layer in layers {
        layer.build_from_shape(current, rng)
        current = layer.output_shape()
    }
    return layers
}

#: A `BatchNorm2d(relu: true)` trained eagerly, through the fused kernels, and
#: as one compiled graph, which has to apply the ReLU itself: eight steps of
#: each from the same start must land on the same parameters. Without the
#: ReLU in the graph the two part company on the first step.
fn test_a_relu_batch_norm_trains_alike_eagerly_and_compiled() -> void {
    if not jt.is_available() { return }
    let x = Tensor([16, 8, 8, 3])
    x.buffer().fill_uniform(-1.0, 1.0, 3)
    var labels: list[float] = []
    for index in 0..16 { labels.push(float(index % 5)) }
    let y = Tensor([16], labels)
    let previous = env("JAITENSOR_BN_GRAPH")
    set_env("JAITENSOR_BN_GRAPH", "2")
    defer {
        if previous is not null { set_env("JAITENSOR_BN_GRAPH", previous) } else { unset_env("JAITENSOR_BN_GRAPH") }
    }
    let step = compile_step(bn_relu_layers(5), [16, 8, 8, 3], 5, 0.05)
    assert_true(step is not null, f"would not compile: {refusal()}")
    if step is null { return }
    for _iteration in 0..8 { step.run(x, y) }
    step.settle()

    let eager = bn_relu_layers(5)
    let loss = SparseCategoricalCrossentropy(from_logits: true)
    let optimizer = SGD(learning_rate: 0.05)
    var params: list[Tensor] = []
    for layer in eager {
        for param in layer.parameters() { params.push(param) }
    }
    prepare_state(params, optimizer.settings())
    for _step in 0..8 {
        var carried = x
        for layer in eager { carried = layer.forward(carried) }
        for held in loss.forward(carried, y).backward() { held.free() }
        optimizer.step(params)
        for param in params {
            let g = param.grad
            if g is not null { g.free() }
            param.grad = null
        }
    }
    assert_eq(step.params.len(), params.len())
    for (index, param) in step.params.enumerate() {
        let got = param.download()
        let expected = params[index].download()
        for slot in 0..got.len() {
            assert_near(got[slot], expected[slot], 1e-4, f"parameter {index} slot {slot}")
        }
    }
    step.free()
    x.free()
    y.free()
}

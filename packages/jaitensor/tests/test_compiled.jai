#: A whole training step compiled to one graph, against running it an
#: operation at a time.
#:
#: The compiled step is only worth having if it trains the same model; these
#: hold it to the eager path's own numbers rather than to a remembered table.
import jaitensor as jt
import std.gpu as gpu
import std.math
from std.random import Random
from std.test import assert_eq, assert_near, assert_true
from jaitensor import Tensor
from jaitensor.compiled import compile_step, refusal
from jaitensor.layers import Activation, AvgPool2d, Conv2d, Dense, Embedding, Flatten, Layer
from jaitensor.layers import LayerNorm, MaxPool2d
from jaitensor.losses import SparseCategoricalCrossentropy
from jaitensor.optimizers import SGD, prepare_state

let BATCH = 16
let SIDE = 8
let CHANNELS = 3
let CLASSES = 5
let RATE = 0.05

fn convolutional(seed: int) -> list[Layer] {
    let rng = Random.with_seed(seed)
    let layers: list[Layer] = [
        Conv2d(
            6,
            3,
            padding: 1,
            in_channels: CHANNELS,
            activation: Activation.Relu
        ),
        MaxPool2d(2),
        Flatten(),
        Dense(CLASSES, activation: Activation.Linear),
    ]
    var current = [SIDE, SIDE, CHANNELS]
    for layer in layers {
        layer.build_from_shape(current, rng)
        current = layer.output_shape()
    }
    return layers
}

fn batch() -> tuple[Tensor, Tensor] {
    let x = Tensor([BATCH, SIDE, SIDE, CHANNELS])
    x.buffer().fill_uniform(-1.0, 1.0, 3)
    var labels: list[float] = []
    for index in 0..BATCH { labels.push(float(index % CLASSES)) }
    return (x, Tensor([BATCH], labels))
}

fn eager_steps(layers: list[Layer], x: Tensor, y: Tensor, steps: int) -> list[Tensor] {
    let loss = SparseCategoricalCrossentropy(from_logits: true)
    let optimizer = SGD(learning_rate: RATE)
    var params: list[Tensor] = []
    for layer in layers {
        for param in layer.parameters() { params.push(param) }
    }
    prepare_state(params, optimizer.settings())
    for _step in 0..steps {
        var carried = x
        for layer in layers { carried = layer.forward(carried) }
        loss.forward(carried, y).backward()
        optimizer.step(params)
        for param in params { param.grad = null }
    }
    return params
}

#: Twelve steps of each, from the same start, must land in the same place.
fn test_a_compiled_step_trains_what_the_eager_one_trains() -> void {
    if not gpu.is_available() { return }
    let (x, y) = batch()
    let compiled_layers = convolutional(5)
    let step = compile_step(compiled_layers, [BATCH, SIDE, SIDE, CHANNELS], CLASSES, RATE)
    assert_true(step is not null, f"would not compile: {refusal()}")
    for _iteration in 0..12 { step.run(x, y) }
    step.settle()

    let eager = eager_steps(convolutional(5), x, y, 12)
    assert_eq(step.params.len(), eager.len())
    for (index, param) in step.params.enumerate() {
        let got = param.download()
        let want = eager[index].download()
        assert_eq(got.len(), want.len(), f"parameter {index} came out the wrong size")
        for slot in 0..got.len() {
            assert_near(got[slot], want[slot], 1e-5, f"parameter {index} slot {slot}")
        }
    }
    step.free()
}

#: The loss the compiled step reports is the loss the eager path computes.
fn test_a_compiled_step_reports_the_same_loss() -> void {
    if not gpu.is_available() { return }
    let (x, y) = batch()
    let layers = convolutional(5)
    #: A rate of zero, so the step reports the loss without moving anything.
    let step = compile_step(layers, [BATCH, SIDE, SIDE, CHANNELS], CLASSES, 0.0)
    assert_true(step is not null, f"would not compile: {refusal()}")
    let reported = step.run(x, y)

    var carried = x
    for layer in convolutional(5) { carried = layer.forward(carried) }
    let want = SparseCategoricalCrossentropy(from_logits: true).forward(carried, y).download()[0]
    assert_near(reported, want, 1e-5)
    step.free()
}

#: A model with a layer the step cannot express keeps the ordinary path, and
#: says which one stopped it rather than failing silently.
fn test_a_model_it_cannot_express_is_refused_with_a_reason() -> void {
    if not gpu.is_available() { return }
    let rng = Random.with_seed(1)
    let layers: list[Layer] = [
        Flatten(),
        Embedding(4, 4),
        Dense(CLASSES, activation: Activation.Linear),
    ]
    var current = [SIDE, SIDE, CHANNELS]
    for layer in layers {
        layer.build_from_shape(current, rng)
        current = layer.output_shape()
    }
    let step = compile_step(layers, [BATCH, SIDE, SIDE, CHANNELS], CLASSES, RATE)
    assert_true(step is null, "a layer it cannot express should have been refused")
    let why = refusal()
    assert_true(why is not null, "a refusal should say why")
}

#: `Sequential.fit` takes the compiled path for a model that allows it, and one
#: step of it lands where a step done by hand lands.
#:
#: One step rather than many on purpose. The two orderings of float32 drift
#: apart over an epoch once a ReLU is deciding which side of zero a value falls
#: on, which is a property of the arithmetic and not a fault in either.
fn test_fit_takes_the_compiled_path_and_agrees_for_a_step() -> void {
    if not gpu.is_available() { return }
    let samples = 32
    var pixels: list[float] = []
    for index in 0..samples * SIDE * SIDE * CHANNELS {
        pixels.push(float(index * 37 % 211 - 105) * 0.01)
    }
    var labels: list[float] = []
    for index in 0..samples { labels.push(float(index % CLASSES)) }
    let features = Tensor([samples, SIDE, SIDE, CHANNELS], pixels)
    let targets = Tensor([samples], labels)

    let model = jt.Sequential(
        [
            Conv2d(6, 3, padding: 1, in_channels: CHANNELS, activation: Activation.Relu),
            MaxPool2d(2),
            Flatten(),
            Dense(CLASSES, activation: Activation.Linear),
        ]
    )
    model.compile(
        optimizer: SGD(learning_rate: RATE),
        loss: SparseCategoricalCrossentropy(from_logits: true)
    )
    #: One batch, so exactly one step.
    model.fit(
        jt.Dataset(features, targets),
        epochs: 1,
        batch_size: samples,
        verbose: false,
        plot_path: null
    )
    var got: list[float] = []
    for layer in model.layers {
        for param in layer.parameters() {
            for value in param.download() { got.push(value) }
        }
    }

    #: The same model built the same way, stepped by hand.
    let byhand = jt.Sequential(
        [
            Conv2d(
                6,
                3,
                padding: 1,
                in_channels: CHANNELS,
                activation: Activation.Relu
            ),
            MaxPool2d(2),
            Flatten(),
            Dense(CLASSES, activation: Activation.Linear),
        ]
    )
    #: Built the way `Sequential` builds, so the two start from the same
    #: weights, then stepped by hand rather than through `fit`.
    let rng = Random.with_seed(7)
    var current = [SIDE, SIDE, CHANNELS]
    for layer in byhand.layers {
        layer.build_from_shape(current, rng)
        current = layer.output_shape()
    }
    let eager = eager_steps(byhand.layers, features, targets, 1)
    var want: list[float] = []
    for param in eager {
        for value in param.download() { want.push(value) }
    }

    assert_eq(got.len(), want.len(), "the two paths trained different parameter counts")
    assert_true(got.len() > 0, "the model trained no parameters")
    for index in 0..got.len() { assert_near(got[index], want[index], 1e-5, f"weight {index}") }
}

#: Every activation the step claims to express gives what the layer gives.
#:
#: `activation_code` reports Gelu, Silu and LeakyRelu as linear, because the
#: layers apply those after the affine part rather than inside it. A step built
#: from that code alone would quietly train a different model, so this holds
#: each one against the layer's own forward.
fn test_every_expressible_activation_matches_the_layer() -> void {
    if not gpu.is_available() { return }
    for activation in [
        Activation.Linear,
        Activation.Relu,
        Activation.Sigmoid,
        Activation.Tanh,
        Activation.Silu,
        Activation.LeakyRelu,
        Activation.Gelu,
    ] {
        let rng = Random.with_seed(9)
        let layers: list[Layer] = [Flatten(), Dense(CLASSES, activation: activation)]
        var current = [SIDE, SIDE, CHANNELS]
        for layer in layers {
            layer.build_from_shape(current, rng)
            current = layer.output_shape()
        }
        let (x, y) = batch()
        #: A rate of zero, so this compares the forward and nothing else.
        let step = compile_step(layers, [BATCH, SIDE, SIDE, CHANNELS], CLASSES, 0.0)
        assert_true(step is not null, f"{activation} would not compile: {refusal()}")
        let scores = Tensor([BATCH, CLASSES])
        let held = Tensor([1])
        step.step_into(x, y, scores, held)
        let got = scores.download()

        var carried = x
        for layer in layers { carried = layer.forward(carried) }
        let want = carried.download()
        for index in 0..want.len() {
            assert_near(got[index], want[index], 1e-4, f"{activation} at {index}")
        }
        step.free()
    }
}

#: Momentum, and Nesterov momentum, carried through the graph rather than left
#: to the ordinary path.
#:
#: The velocity lives in the graph as another pair of tensors, fed in and
#: produced, so a step of it has to land where the optimizer's own kernel lands.
fn test_momentum_is_carried_through_the_graph() -> void {
    if not gpu.is_available() { return }
    for nesterov in [false, true] {
        let (x, y) = batch()
        let step = compile_step(
            convolutional(5),
            [BATCH, SIDE, SIDE, CHANNELS],
            CLASSES,
            RATE,
            0.9,
            nesterov
        )
        assert_true(step is not null, f"momentum would not compile: {refusal()}")
        #: More than one step, because a velocity that is not carried looks
        #: right on the first one and wrong on every one after it.
        for _iteration in 0..4 { step.run(x, y) }
        step.settle()

        let layers = convolutional(5)
        let loss = SparseCategoricalCrossentropy(from_logits: true)
        let optimizer = SGD(learning_rate: RATE, momentum: 0.9, nesterov: nesterov)
        var params: list[Tensor] = []
        for layer in layers {
            for param in layer.parameters() { params.push(param) }
        }
        prepare_state(params, optimizer.settings())
        for _iteration in 0..4 {
            var carried = x
            for layer in layers { carried = layer.forward(carried) }
            loss.forward(carried, y).backward()
            optimizer.step(params)
            for param in params { param.grad = null }
        }

        for (index, param) in step.params.enumerate() {
            let got = param.download()
            let want = params[index].download()
            for slot in 0..got.len() {
                assert_near(got[slot], want[slot], 1e-4, f"nesterov {nesterov} parameter {index}")
            }
        }
        step.free()
    }
}

#: Adam, with its two moments and its bias correction, carried through the
#: graph rather than left to the optimizer's own kernel.
#:
#: The correction divides by what each moment would have been had it not
#: started at rest, which depends on how many steps have been taken -- so the
#: count is carried too. Several steps, because a count that does not advance
#: looks right on the first and wrong after.
fn test_adam_is_carried_through_the_graph() -> void {
    if not gpu.is_available() { return }
    let (x, y) = batch()
    let rate = 0.01
    let step = compile_step(
        convolutional(5),
        [BATCH, SIDE, SIDE, CHANNELS],
        CLASSES,
        rate,
        adam: true
    )
    assert_true(step is not null, f"adam would not compile: {refusal()}")
    for _iteration in 0..5 { step.run(x, y) }
    step.settle()

    let layers = convolutional(5)
    let loss = SparseCategoricalCrossentropy(from_logits: true)
    let optimizer = jt.Adam(learning_rate: rate)
    var params: list[Tensor] = []
    for layer in layers {
        for param in layer.parameters() { params.push(param) }
    }
    prepare_state(params, optimizer.settings())
    for _iteration in 0..5 {
        var carried = x
        for layer in layers { carried = layer.forward(carried) }
        loss.forward(carried, y).backward()
        optimizer.step(params)
        for param in params { param.grad = null }
    }

    for (index, param) in step.params.enumerate() {
        let got = param.download()
        let want = params[index].download()
        for slot in 0..got.len() {
            assert_near(got[slot], want[slot], 1e-4, f"parameter {index} slot {slot}")
        }
    }
    step.free()
}

#: A batch cut out of a bigger tensor has to train on its own rows.
#:
#: The graph is handed a device buffer, and a slice is that buffer with an
#: origin. Binding it from the start instead trains on the first batch of the
#: epoch every time -- the reported loss still falls, because it really is
#: fitting the rows it keeps being shown, so nothing looks wrong until the
#: model is asked about the rest of the data.
fn test_a_compiled_step_reads_the_slice_it_was_given() -> void {
    if not gpu.is_available() { return }
    let width = 16
    let rng = Random.with_seed(4)
    var rows: list[float] = []
    for index in 0..2 * BATCH * width { rows.push(rng.next_float() * 2.0 - 1.0) }
    let pool = Tensor([2 * BATCH, width], rows)
    defer { pool.free() }
    var answers: list[float] = []
    for index in 0..BATCH { answers.push(float(index % CLASSES)) }
    let y = Tensor([BATCH], answers)
    defer { y.free() }

    #: The same rows twice: once as a window into `pool`, once on their own.
    let second = pool.view(BATCH * width, [BATCH, width])
    defer { second.free() }
    var copied: list[float] = []
    for slot in 0..BATCH * width { copied.push(rows[BATCH * width + slot]) }
    let alone = Tensor([BATCH, width], copied)
    defer { alone.free() }

    let through_view = losses_from(dense_only(6), second, y, [BATCH, width])
    let on_its_own = losses_from(dense_only(6), alone, y, [BATCH, width])
    assert_eq(through_view.len(), on_its_own.len())
    for index in 0..through_view.len() {
        assert_near(
            through_view[index],
            on_its_own[index],
            1e-5,
            f"step {index} read different rows through the view"
        )
    }
}

fn dense_only(seed: int) -> list[Layer] {
    let rng = Random.with_seed(seed)
    let layers: list[Layer] = [
        Dense(12, activation: Activation.Relu),
        Dense(CLASSES, activation: Activation.Linear),
    ]
    var current = [16]
    for layer in layers {
        layer.build_from_shape(current, rng)
        current = layer.output_shape()
    }
    return layers
}

fn losses_from(layers: list[Layer], x: Tensor, y: Tensor, shape: list[int]) -> list[float] {
    let step = compile_step(layers, shape, CLASSES, RATE)
    assert_true(step is not null, f"would not compile: {refusal()}")
    var seen: list[float] = []
    for _iteration in 0..4 { seen.push(step.run(x, y)) }
    step.free()
    return seen
}

#: Normalisation and average pooling belong on the compiled path too, and a
#: model built from them has to train where the eager one trains.
fn test_a_compiled_step_expresses_normalisation_and_average_pooling() -> void {
    if not gpu.is_available() { return }
    let (x, y) = batch()
    defer { x.free() }
    defer { y.free() }
    let step = compile_step(wider(5), [BATCH, SIDE, SIDE, CHANNELS], CLASSES, RATE)
    assert_true(step is not null, f"would not compile: {refusal()}")
    for _iteration in 0..8 { step.run(x, y) }
    step.settle()

    let eager = wider(5)
    let want = eager_steps(eager, x, y, 8)
    assert_eq(step.params.len(), want.len())
    for (index, param) in step.params.enumerate() {
        let got = param.download()
        let expected = want[index].download()
        for slot in 0..got.len() {
            assert_near(got[slot], expected[slot], 1e-4, f"parameter {index} slot {slot}")
        }
    }
    step.free()
}

#: A pool that pads is a different pool, and the step has to say so.
fn test_a_compiled_pool_keeps_its_padding() -> void {
    if not gpu.is_available() { return }
    let (x, y) = batch()
    defer { x.free() }
    defer { y.free() }
    let step = compile_step(padded(5), [BATCH, SIDE, SIDE, CHANNELS], CLASSES, RATE)
    assert_true(step is not null, f"would not compile: {refusal()}")
    for _iteration in 0..6 { step.run(x, y) }
    step.settle()

    let want = eager_steps(padded(5), x, y, 6)
    for (index, param) in step.params.enumerate() {
        let got = param.download()
        let expected = want[index].download()
        for slot in 0..got.len() {
            assert_near(got[slot], expected[slot], 1e-4, f"parameter {index} slot {slot}")
        }
    }
    step.free()
}

fn wider(seed: int) -> list[Layer] {
    let rng = Random.with_seed(seed)
    let layers: list[Layer] = [
        Conv2d(
            6,
            3,
            padding: 1,
            in_channels: CHANNELS,
            activation: Activation.Relu
        ),
        AvgPool2d(2),
        Flatten(),
        LayerNorm(),
        Dense(CLASSES, activation: Activation.Linear),
    ]
    var current = [SIDE, SIDE, CHANNELS]
    for layer in layers {
        layer.build_from_shape(current, rng)
        current = layer.output_shape()
    }
    return layers
}

fn padded(seed: int) -> list[Layer] {
    let rng = Random.with_seed(seed)
    let layers: list[Layer] = [
        Conv2d(
            6,
            3,
            padding: 1,
            in_channels: CHANNELS,
            activation: Activation.Relu
        ),
        MaxPool2d(3, stride: 2, padding: 1),
        Flatten(),
        Dense(CLASSES, activation: Activation.Linear),
    ]
    var current = [SIDE, SIDE, CHANNELS]
    for layer in layers {
        layer.build_from_shape(current, rng)
        current = layer.output_shape()
    }
    return layers
}

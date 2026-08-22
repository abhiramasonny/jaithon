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
from jaitensor.layers import Activation, Conv2d, Dense, Flatten, Layer, MaxPool2d
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
    let layers: list[Layer] = [Flatten(), Dense(CLASSES, activation: Activation.Sigmoid)]
    var current = [SIDE, SIDE, CHANNELS]
    for layer in layers {
        layer.build_from_shape(current, rng)
        current = layer.output_shape()
    }
    let step = compile_step(layers, [BATCH, SIDE, SIDE, CHANNELS], CLASSES, RATE)
    assert_true(step is null, "an activation it cannot express should have been refused")
    let why = refusal()
    assert_true(why is not null, "a refusal should say why")
}

#: The paths that skip autograd: the fused epoch kernels and the compiled static step.
import jaitensor as jt
from jaitensor import Dataset
from jaitensor.ops import dense_static_forward
from std.test import assert_eq
from std.trace import is_traced

fn separable_data(repeats: int = 32) -> Dataset {
    var features: list[float] = []
    var labels: list[int] = []
    for _repeat in 0..repeats {
        features.extend([-2.0, -1.0, -1.0, -2.0, 2.0, 1.0, 1.0, 2.0])
        labels.extend([0, 0, 1, 1])
    }
    return jt.Dataset.from_flat(features, labels, 2)
}

fn test_fused_two_layer_relu_sgd_reduces_loss() -> void {
    if not jt.is_available() { return }
    let data = separable_data(8)
    defer { data.free() }
    let model = jt.Sequential(
        [
            jt.Dense(512, activation: jt.Activation.Relu),
            jt.Dense(2, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(jt.SGD(learning_rate: 0.2), loss: jt.SparseCategoricalCrossentropy())
    let history = model.fit(data, epochs: 6, batch_size: 32, verbose: false, plot_path: null)
    assert history.loss[history.loss.len() - 1] < history.loss[0]
}

fn test_fused_mlp_step_with_momentum_reduces_loss() -> void {
    if not jt.is_available() { return }
    let data = separable_data(8)
    defer { data.free() }
    let model = jt.Sequential(
        [
            jt.Dense(512, activation: jt.Activation.Relu),
            jt.Dense(2, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(
        jt.SGD(learning_rate: 0.2, momentum: 0.5),
        loss: jt.SparseCategoricalCrossentropy()
    )
    let history = model.fit(data, epochs: 6, batch_size: 32, verbose: false, plot_path: null)
    assert history.loss[history.loss.len() - 1] < history.loss[0]
}

fn test_fused_three_layer_relu_sgd_reduces_loss() -> void {
    if not jt.is_available() { return }
    let data = separable_data(8)
    defer { data.free() }
    let model = jt.Sequential(
        [
            jt.Dense(256, activation: jt.Activation.Relu),
            jt.Dense(256, activation: jt.Activation.Relu),
            jt.Dense(256, activation: jt.Activation.Relu),
            jt.Dense(2, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(jt.SGD(learning_rate: 0.2), loss: jt.SparseCategoricalCrossentropy())
    let history = model.fit(data, epochs: 10, batch_size: 32, verbose: false, plot_path: null)
    assert history.loss[history.loss.len() - 1] < history.loss[0]
}

fn test_fused_mlp_step_adam_reduces_loss() -> void {
    if not jt.is_available() { return }
    let data = separable_data(8)
    defer { data.free() }
    let model = jt.Sequential(
        [
            jt.Dense(512, activation: jt.Activation.Relu),
            jt.Dense(2, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(jt.Adam(learning_rate: 0.05), loss: jt.SparseCategoricalCrossentropy())
    let history = model.fit(data, epochs: 6, batch_size: 32, verbose: false, plot_path: null)
    assert history.loss[history.loss.len() - 1] < history.loss[0]
}

fn test_execution_plan_built_on_compile() -> void {
    if not jt.is_available() { return }
    let model = jt.Sequential(
        [
            jt.Dense(16, activation: jt.Activation.Relu),
            jt.Dense(4, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(jt.SGD(), input_units: 2, batch_size: 32)
    let plan = model.execution_plan()
    assert plan is not null
    assert_eq(plan.input_units, 2)
    assert_eq(plan.batch_size, 32)
    assert_eq(plan.layer_widths, [16, 4])
    assert plan.static_step
}

fn test_static_step_reduces_loss() -> void {
    if not jt.is_available() { return }
    let data = separable_data(8)
    defer { data.free() }
    let model = jt.Sequential(
        [
            jt.Dense(32, activation: jt.Activation.Relu),
            jt.Dense(2, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(
        jt.SGD(learning_rate: 0.2),
        loss: jt.SparseCategoricalCrossentropy(),
        input_units: 2,
        batch_size: 32
    )
    let history = model.fit(data, epochs: 8, batch_size: 32, verbose: false, plot_path: null)
    assert history.loss[history.loss.len() - 1] < history.loss[0]
}

fn test_static_step_matches_autograd_weights() -> void {
    if not jt.is_available() { return }
    let data = separable_data(8)
    defer { data.free() }
    let static_model = jt.Sequential(
        [
            jt.Dense(16, activation: jt.Activation.Relu),
            jt.Dense(2, activation: jt.Activation.Softmax),
        ]
    )
    defer { static_model.free() }
    let autograd_model = jt.Sequential(
        [
            jt.Dense(16, activation: jt.Activation.Relu),
            jt.Dense(2, activation: jt.Activation.Softmax),
        ]
    )
    defer { autograd_model.free() }
    static_model.build(2)
    autograd_model.build(2)
    let (weights, bias) = static_model.layers[0].get_weights()
    let (head_weights, head_bias) = static_model.layers[1].get_weights()
    autograd_model.layers[0].set_weights(weights, bias)
    autograd_model.layers[1].set_weights(head_weights, head_bias)
    static_model.compile(
        jt.SGD(learning_rate: 0.1),
        loss: jt.SparseCategoricalCrossentropy(),
        input_units: 2,
        batch_size: 32
    )
    autograd_model.compile(jt.SGD(learning_rate: 0.1), loss: jt.SparseCategoricalCrossentropy())
    let static_history = static_model.fit(
        data,
        epochs: 3,
        batch_size: 32,
        verbose: false,
        plot_path: null
    )
    let autograd_history = autograd_model.fit(
        data,
        epochs: 3,
        batch_size: 32,
        verbose: false,
        plot_path: null
    )
    assert abs(static_history.loss[2] - autograd_history.loss[2]) < 0.05
}

fn test_static_kernels_are_traced() -> void {
    assert is_traced(dense_static_forward)
}

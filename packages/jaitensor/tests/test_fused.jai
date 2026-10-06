#: The paths that skip autograd: the fused epoch kernels and the compiled static step.
import jaitensor as jt
from jaitensor import Dataset
from jaitensor.compiled import set_flat_policy
from jaitensor.ops import dense_static_forward
from std.test import assert_eq, assert_near
import std.gpu as gpu
from std.gpu import Buffer
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
    #: Flat rows would take the compiled step now; this is about the static one.
    let policy = set_flat_policy(0)
    defer { set_flat_policy(policy) }
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
    #: Flat rows would take the compiled step now; this is about the static one.
    let policy = set_flat_policy(0)
    defer { set_flat_policy(policy) }
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

fn filled(size: int, seed: int) -> Buffer {
    let buffer = gpu.Buffer(size)
    if seed == 0 { buffer.fill_zero() } else { buffer.fill_uniform(-0.1, 0.1, seed) }
    return buffer
}

#: The device epoch counts the same loss and the same right answers, step for
#: step, as the per-step kernel it batches up.
#:
#: The epoch ping-pongs its loss and correct-count accumulators between the
#: caller's buffers and scratch, the way it ping-pongs the weights, instead of
#: copying them home after every step. So an odd number of steps ends with the
#: totals in scratch, a second epoch starts from whichever weight side the
#: first left, and a read straight after the call has to wait for the last
#: step rather than the first. Odd and even, one hidden layer and three, two
#: epochs each, read without a synchronize in between.
fn test_the_device_epoch_counts_what_the_steps_count() -> void {
    if not jt.is_available() { return }
    let batch = 32
    let inputs = 16
    let hidden = 256
    let classes = 4
    let rate = 0.05
    for deep in [false, true] {
        for steps in [5, 4] {
            let samples = steps * batch
            var rows: list[float] = []
            var answers: list[float] = []
            var seed = 11
            for _sample in 0..samples {
                seed = (seed * 1103515245 + 12345) % 2147483648
                let kind = seed % classes
                for slot in 0..inputs {
                    let signal = if slot % classes == kind { 1.0 } else { 0.0 }
                    rows.push(signal + float(seed % 7) * 0.05)
                }
                answers.push(float(kind))
            }
            let x = jt.Tensor([samples, inputs], rows)
            let y = jt.Tensor([samples], answers)
            var got: list[float] = []
            var want: list[float] = []
            for by_epoch in [true, false] {
                let w1 = filled(inputs * hidden, 1)
                let b1 = filled(hidden, 0)
                let w2 = filled(hidden * hidden, 2)
                let b2 = filled(hidden, 0)
                let w3 = filled(hidden * hidden, 3)
                let b3 = filled(hidden, 0)
                let w4 = filled(hidden * classes, 4)
                let b4 = filled(classes, 0)
                let loss = filled(1, 0)
                let correct = filled(1, 0)
                var seen: list[float] = []
                for _epoch in 0..2 {
                    if by_epoch {
                        if deep {
                            gpu.mlp3_sgd_epoch(
                                x.buffer(), w1, b1, w2, b2, w3, b3, w4, b4, y.buffer(), loss,
                                samples, batch, inputs, hidden, classes, rate,
                                flush_every: 2, correct_acc: correct
                            )
                        } else {
                            gpu.mlp_sgd_epoch(
                                x.buffer(), w1, b1, w4, b4, y.buffer(), loss,
                                samples, batch, inputs, hidden, classes, rate,
                                flush_every: 2, correct_acc: correct
                            )
                        }
                    } else {
                        for step in 0..steps {
                            let xs = x.view(step * batch * inputs, [batch, inputs])
                            let ys = y.view(step * batch, [batch])
                            if deep {
                                gpu.mlp3_sgd_step(
                                    xs.buffer(), w1, b1, w2, b2, w3, b3, w4, b4, ys.buffer(), loss,
                                    batch, inputs, hidden, classes, rate, correct_acc: correct
                                )
                            } else {
                                gpu.mlp_sgd_step(
                                    xs.buffer(), w1, b1, w4, b4, ys.buffer(), loss,
                                    batch, inputs, hidden, classes, rate, correct_acc: correct
                                )
                            }
                            xs.free()
                            ys.free()
                        }
                    }
                    #: Read at once, the way fit reads an epoch's totals.
                    seen.push(loss.download()[0])
                    seen.push(correct.download()[0])
                    loss.fill_zero()
                    correct.fill_zero()
                }
                gpu.synchronize()
                seen.extend(w4.download())
                for buffer in [w1, b1, w2, b2, w3, b3, w4, b4, loss, correct] { buffer.free() }
                if by_epoch { got = seen } else { want = seen }
            }
            x.free()
            y.free()
            assert_eq(got.len(), want.len())
            for index in 0..got.len() {
                assert_near(got[index], want[index], 1e-4, f"deep {deep} steps {steps} slot {index}")
            }
            assert got[1] > 0.0
        }
    }
}

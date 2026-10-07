#: The optimizers, one parameter at a time: momentum, moments, and the settings buffer.
import jaitensor as jt
import std.math
from std.test import assert_throws
from jaitensor import Dataset, Layer, Optimizer
from jaitensor.compiled import compiled_step_allowed, set_compiled_step

fn test_sgd_step_momentum_persists_and_affects_a_second_step() -> void {
    if not jt.is_available() { return }
    let start = 2.0
    let parameter = jt.Tensor([1], [start], requires_grad: true)
    let optimizer = jt.SGD(learning_rate: 0.1, momentum: 0.5)

    let first_grad = jt.Tensor([1], [1.0])
    parameter.grad = first_grad
    optimizer.step([parameter])
    let value_1 = parameter.download()[0]
    let velocity_1 = (start - value_1) / 0.1

    first_grad.free()
    let second_grad = jt.Tensor([1], [2.0])
    parameter.grad = second_grad
    optimizer.step([parameter])
    let velocity_2 = 0.5 * velocity_1 + 2.0
    let expected_value_2 = value_1 - 0.1 * velocity_2
    assert math.fabs(parameter.download()[0] - expected_value_2) < 1e-5

    second_grad.free()
}

fn test_sgd_step_matches_the_textbook_update_on_one_parameter() -> void {
    if not jt.is_available() { return }
    let parameter = jt.Tensor([2], [1.0, -1.0], requires_grad: true)
    parameter.grad = jt.Tensor([2], [0.5, 2.0])
    let optimizer = jt.SGD(learning_rate: 0.1)
    optimizer.step([parameter])
    let updated = parameter.download()
    assert math.fabs(updated[0] - (1.0 - 0.1 * 0.5)) < 1e-5
    assert math.fabs(updated[1] - (-1.0 - 0.1 * 2.0)) < 1e-5
    parameter.free()
}

#: A learning rate that changes between steps has to reach the device.
#:
#: The settings buffer was written when a parameter first saw an optimizer
#: and never again, so the card kept reading the opening rate: a schedule
#: decaying 1.0, 0.1, 0.01, 0.001 moved the parameter by a whole 1.0 four
#: times over. Nothing complains, because a loss that is falling too fast is
#: still falling.
fn test_a_changing_learning_rate_reaches_the_device() -> void {
    if not jt.is_available() { return }
    let parameter = jt.Tensor([1], [0.0], requires_grad: true)
    var expected = 0.0
    for rate in [1.0, 0.1, 0.01, 0.001] {
        let gradient = jt.Tensor([1], [1.0])
        parameter.grad = gradient
        jt.SGD(learning_rate: rate).step([parameter])
        expected -= rate
        assert math.fabs(parameter.download()[0] - expected) < 1e-6
        gradient.free()
    }
    parameter.free()
}

#: The same parameter handed to a different kind of optimizer.
#:
#: Plain descent keeps three numbers in its settings and Adam six, so a
#: buffer sized for the first is too short for the second and Adam would read
#: past the end of it.
fn test_switching_optimizer_kind_resizes_the_settings() -> void {
    if not jt.is_available() { return }
    let parameter = jt.Tensor([1], [0.0], requires_grad: true)
    let first = jt.Tensor([1], [1.0])
    parameter.grad = first
    jt.SGD(learning_rate: 0.1).step([parameter])
    first.free()
    let second = jt.Tensor([1], [1.0])
    parameter.grad = second
    jt.Adam(learning_rate: 0.1).step([parameter])
    let landed = parameter.download()[0]
    #: Adam's first step is the learning rate, near enough, whatever came
    #: before it.
    assert math.is_finite(landed)
    assert math.fabs(landed - (-0.1 - 0.1)) < 1e-3
    second.free()
    parameter.free()
}

fn test_adam_step_moments_persist_and_affect_a_second_step() -> void {
    if not jt.is_available() { return }
    let parameter = jt.Tensor([1], [0.0], requires_grad: true)
    let beta1 = 0.5
    let beta2 = 0.5
    let epsilon = 1e-7
    let optimizer = jt.Adam(learning_rate: 0.001, beta1: beta1, beta2: beta2, epsilon: epsilon)

    let warmup_grad = jt.Tensor([1], [1.0])
    parameter.grad = warmup_grad
    for _index in 0..40 { optimizer.step([parameter]) }
    warmup_grad.free()
    let value_before = parameter.download()[0]
    let converged_first_moment = 1.0
    let converged_second_moment = 1.0

    let next_grad = jt.Tensor([1], [3.0])
    parameter.grad = next_grad
    optimizer.step([parameter])
    let first_moment_next = beta1 * converged_first_moment + (1.0 - beta1) * 3.0
    let second_moment_next = beta2 * converged_second_moment + (1.0 - beta2) * 3.0 * 3.0
    let expected_value = value_before -
        0.001 * first_moment_next / (math.sqrt(second_moment_next) + epsilon)
    assert math.fabs(parameter.download()[0] - expected_value) < 1e-6

    next_grad.free()
}

fn test_adam_step_matches_the_textbook_first_step_update_on_one_parameter() -> void {
    if not jt.is_available() { return }
    let parameter = jt.Tensor([1], [0.0], requires_grad: true)
    parameter.grad = jt.Tensor([1], [1.0])
    let optimizer = jt.Adam(learning_rate: 0.001, beta1: 0.9, beta2: 0.999, epsilon: 1e-7)
    optimizer.step([parameter])
    let first_moment = 0.1
    let second_moment = 0.001
    let corrected_first = first_moment / (1.0 - 0.9)
    let corrected_second = second_moment / (1.0 - 0.999)
    let expected = 0.0 - 0.001 * corrected_first / (math.sqrt(corrected_second) + 1e-7)
    assert math.fabs(parameter.download()[0] - expected) < 1e-6
    parameter.free()
}

fn test_optimizer_step_requires_every_parameter_to_have_a_gradient() -> void {
    if not jt.is_available() { return }
    let parameter = jt.Tensor([1], [1.0], requires_grad: true)
    let optimizer = jt.SGD()
    let _error = assert_throws(RuntimeError, fn() -> any {
        optimizer.step([parameter])
        return null
    })
    parameter.free()
}

fn test_optimizer_step_state_does_not_leak_to_a_new_parameter_reusing_a_freed_handle() -> void {
    if not jt.is_available() { return }
    let optimizer = jt.SGD(learning_rate: 0.1, momentum: 0.9)

    let stale_parameter = jt.Tensor([1], [5.0], requires_grad: true)
    stale_parameter.grad = jt.Tensor([1], [2.0])
    optimizer.step([stale_parameter])
    stale_parameter.free()

    let fresh_parameter = jt.Tensor([1], [1.0], requires_grad: true)
    fresh_parameter.grad = jt.Tensor([1], [0.5])
    optimizer.step([fresh_parameter])
    let expected = 1.0 - 0.1 * 0.5
    assert math.fabs(fresh_parameter.download()[0] - expected) < 1e-5

    fresh_parameter.free()
}

fn test_optimizer_momentum_starts_at_zero_on_a_recycled_device_buffer() -> void {
    if not jt.is_available() { return }
    #: A freed buffer goes back to a pool for the next allocation of the same
    #: size, so a new parameter's momentum is very likely to land on memory a
    #: previous parameter's momentum used. Zeroing it is a host write, and the
    #: update that filled it may be encoded and not yet run -- so without a
    #: wait, the old update lands after the zeroing and the new parameter
    #: starts with someone else's velocity.
    #:
    #: The churn is what makes this deterministic: it leaves buffers of the
    #: right size in the pool before the parameter under test asks for one.
    for _round in 0..8 {
        let scratch = jt.Tensor([1], [3.0], requires_grad: true)
        scratch.grad = jt.Tensor([1], [1.0])
        jt.SGD(learning_rate: 0.1, momentum: 0.9).step([scratch])
        scratch.free()
    }

    let optimizer = jt.SGD(learning_rate: 0.1, momentum: 0.9)
    let parameter = jt.Tensor([1], [1.0], requires_grad: true)
    parameter.grad = jt.Tensor([1], [0.5])
    optimizer.step([parameter])
    #: One step from rest moves by the learning rate times the gradient, with
    #: no velocity term yet.
    assert math.fabs(parameter.download()[0] - (1.0 - 0.1 * 0.5)) < 1e-5
    parameter.free()
}

fn test_optimizer_configuration_rejects_invalid_values() -> void {
    let _rate = assert_throws(ValueError, || jt.SGD(learning_rate: 0.0))
    let _momentum = assert_throws(ValueError, || jt.SGD(momentum: 1.0))
    let _nesterov = assert_throws(ValueError, || jt.SGD(nesterov: true))
    let _beta = assert_throws(ValueError, || jt.Adam(beta1: 1.0))
    let _epsilon = assert_throws(ValueError, || jt.Adam(epsilon: 0.0))
}

fn rows_that_carry_their_class(samples: int, width: int, classes: int) -> Dataset {
    var features: list[float] = []
    var labels: list[float] = []
    var seed = 17
    for sample in 0..samples {
        let kind = sample % classes
        for slot in 0..width {
            seed = (seed * 1103515245 + 12345) % 2147483648
            let signal = if slot % classes == kind { 1.0 } else { 0.0 }
            features.push(signal + float(seed % 100) * 0.002)
        }
        labels.push(float(kind))
    }
    return Dataset(jt.Tensor([samples, width], features), jt.Tensor([samples], labels))
}

#: Fitting one epoch at a time trains the same model as fitting them together.
#:
#: Every `fit` prepares the optimiser's state, and preparing it asked for the
#: state of a plain descent -- which, for a model trained with Adam, is a state
#: of the wrong kind, so it was thrown away and made again from rest. Every
#: second `fit` started Adam over: moments zeroed, step count back to one, the
#: bias correction re-applied. A training loop that fits an epoch at a time,
#: to evaluate or save between them, trained a different model from one that
#: did not, and nothing said so. The static path, the fused forward-backward
#: path and the autograd path all prepare state this way.
fn test_fitting_epoch_by_epoch_keeps_the_optimiser_state() -> void {
    if not jt.is_available() { return }
    let was = compiled_step_allowed()
    set_compiled_step(false)
    defer { set_compiled_step(was) }
    #: Static (two narrow hidden layers), fused forward-backward (one wide
    #: one) and autograd (a normalisation), each with Adam and with momentum.
    for shape in ["static", "fused", "autograd"] {
        for name in ["adam", "momentum"] {
            var traces: list[list[float]] = []
            for epoch_at_a_time in [false, true] {
                let data = rows_that_carry_their_class(64, 16, 4)
                let layers: list[Layer] = if shape == "fused" {
                    [jt.Dense(256, activation: jt.Activation.Relu), jt.Dense(4, activation: jt.Activation.Softmax)]
                } elif shape == "autograd" {
                    [jt.Dense(16, activation: jt.Activation.Relu), jt.LayerNorm(), jt.Dense(4, activation: jt.Activation.Softmax)]
                } else {
                    [jt.Dense(16, activation: jt.Activation.Relu), jt.Dense(8, activation: jt.Activation.Relu), jt.Dense(4, activation: jt.Activation.Softmax)]
                }
                let model = jt.Sequential(layers)
                let optimizer: Optimizer = if name == "adam" {
                    jt.Adam(learning_rate: 0.01)
                } else {
                    jt.SGD(learning_rate: 0.05, momentum: 0.9)
                }
                model.compile(optimizer: optimizer, loss: jt.SparseCategoricalCrossentropy())
                var trace: list[float] = []
                if epoch_at_a_time {
                    for _epoch in 0..3 {
                        let history = model.fit(
                            data,
                            epochs: 1,
                            batch_size: 16,
                            verbose: false,
                            plot_path: null
                        )
                        trace.push(history.loss[0])
                    }
                } else {
                    let history = model.fit(
                        data,
                        epochs: 3,
                        batch_size: 16,
                        verbose: false,
                        plot_path: null
                    )
                    trace.extend(history.loss)
                }
                let want = if shape == "fused" { "fused-mlp" } else { shape }
                assert model.training_route() == want, f"{shape} took {model.training_route()}"
                traces.push(trace)
                model.free()
                data.free()
            }
            for index in 0..3 {
                assert math.fabs(traces[1][index] - traces[0][index]) < 1e-5, f"{shape} {name} epoch {index}: {traces[1]} against {traces[0]}"
            }
        }
    }
}

#: The optimizers, one parameter at a time: momentum, moments, and the settings buffer.
import jaitensor as jt
import std.gpu as gpu
import std.math
from std.test import assert_eq, assert_throws
from jaitensor import Dataset, Layer, Optimizer, Tensor
from jaitensor.compiled import compiled_step_allowed, set_compiled_step
from jaitensor.optimizers import float32_bits

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
    #: An ordered comparison with NaN throws a TypeError, so these are
    #: tested for NaN first.
    let _nan_decay = assert_throws(ValueError, || jt.SGD(weight_decay: math.NAN))
    let _nan_adam_decay = assert_throws(ValueError, || jt.Adam(weight_decay: math.NAN))
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

#: Three steps of an optimizer on one three-element parameter, from the same
#: gradients the PyTorch references below were computed with (float64).
fn three_steps(optimizer: Optimizer) -> list[float] {
    let parameter = jt.Tensor([3], [1.0, -2.0, 0.5], requires_grad: true)
    let gradients = [[0.5, -1.0, 2.0], [0.25, 0.75, -1.5], [-0.5, 0.1, 0.3]]
    for values in gradients {
        let gradient = jt.Tensor([3], values)
        parameter.grad = gradient
        optimizer.step([parameter])
        gradient.free()
    }
    parameter.grad = null
    let landed = parameter.download()
    parameter.free()
    return landed
}

fn assert_close(got: list[float], want: list[float], tolerance: float, what: str) -> void {
    assert got.len() == want.len(), f"{what}: {got.len()} values, expected {want.len()}"
    for (index, value) in want.enumerate() {
        assert math.fabs(got[index] - value) <= tolerance, f"{what}[{index}]: {got[index]} against torch {value}"
    }
}

#: Weight decay against `torch.optim.SGD(weight_decay=...)`: the L2 term joins
#: the gradient before the momentum, in plain, momentum and Nesterov descent.
fn test_sgd_weight_decay_matches_torch() -> void {
    if not jt.is_available() { return }
    assert_close(
        three_steps(jt.SGD(learning_rate: 0.1, weight_decay: 0.05)),
        [0.9606986249999999, -1.9557722499999999, 0.4137824375],
        1e-6,
        "sgd"
    )
    assert_close(
        three_steps(jt.SGD(learning_rate: 0.1, momentum: 0.9, weight_decay: 0.05)),
        [0.840143625, -1.82716225, 0.2020799375],
        1e-6,
        "momentum"
    )
    assert_close(
        three_steps(jt.SGD(learning_rate: 0.1, momentum: 0.9, nesterov: true, weight_decay: 5e-4)),
        [0.8549278414117676, -1.8775831203235354, 0.16157469395507135],
        1e-6,
        "nesterov"
    )
}

#: `Adam(weight_decay=...)` is PyTorch's coupled L2 form; `AdamW` decays the
#: weights directly. They differ, and each matches its torch counterpart.
fn test_adam_weight_decay_and_adamw_match_torch() -> void {
    if not jt.is_available() { return }
    let coupled = three_steps(jt.Adam(learning_rate: 0.01, epsilon: 1e-8, weight_decay: 0.1))
    assert_close(
        coupled,
        [0.9772928152861048, -1.9842420340514224, 0.48705850466617945],
        2e-6,
        "adam"
    )
    let decoupled = three_steps(jt.AdamW(learning_rate: 0.01, epsilon: 1e-8, weight_decay: 0.1))
    assert_close(
        decoupled,
        [0.9766026841558618, -1.9829551877290008, 0.4861818426362763],
        2e-6,
        "adamw"
    )
    assert math.fabs(coupled[0] - decoupled[0]) > 1e-4
}

#: Without decay, the decaying optimizers are the ones they extend -- the
#: same kind, so the same training paths, and the same numbers.
fn test_zero_weight_decay_is_the_undecayed_optimizer() -> void {
    if not jt.is_available() { return }
    assert jt.SGD(momentum: 0.9, weight_decay: 0.0).kind() == jt.SGD(momentum: 0.9).kind()
    assert jt.Adam(weight_decay: 0.0).kind() == jt.Adam().kind()
    assert jt.AdamW(weight_decay: 0.0).kind() == jt.Adam().kind()
    assert jt.SGD(weight_decay: 1e-4).kind() != jt.SGD().kind()
    assert jt.AdamW().kind() != jt.Adam(weight_decay: 0.01).kind()
    let plain = three_steps(jt.Adam(learning_rate: 0.01))
    let unweighted = three_steps(jt.AdamW(learning_rate: 0.01, weight_decay: 0.0))
    assert_close(unweighted, plain, 0.0, "adamw without decay")
    let _negative = assert_throws(ValueError, || jt.SGD(weight_decay: -1.0))
    let _adam = assert_throws(ValueError, || jt.AdamW(weight_decay: -0.1))
}

#: Assigning `learning_rate` between steps reaches the device, and the
#: momentum the optimizer has built up carries on through the change.
fn test_assigning_the_learning_rate_keeps_the_momentum() -> void {
    if not jt.is_available() { return }
    let parameter = jt.Tensor([1], [0.0], requires_grad: true)
    let optimizer = jt.SGD(learning_rate: 1.0, momentum: 0.5)
    var velocity = 0.0
    var expected = 0.0
    for rate in [1.0, 0.1, 0.01, 0.5] {
        optimizer.learning_rate = rate
        let gradient = jt.Tensor([1], [1.0])
        parameter.grad = gradient
        optimizer.step([parameter])
        velocity = 0.5 * velocity + 1.0
        expected -= rate * velocity
        let landed = parameter.download()[0]
        assert math.fabs(landed - expected) < 1e-6, f"rate {rate}: {landed} against {expected}"
        gradient.free()
    }
    parameter.free()
}

#: `clip_grad_norm` against `torch.nn.utils.clip_grad_norm_`: the norm is over
#: every gradient as one vector, the returned norm is the one before
#: clipping, and gradients already inside the bound are left alone.
fn test_clip_grad_norm_matches_torch() -> void {
    if not jt.is_available() { return }
    let a = jt.Tensor([3], [0.0, 0.0, 0.0], requires_grad: true)
    let b = jt.Tensor([2], [0.0, 0.0], requires_grad: true)
    let unused = jt.Tensor([2], [0.0, 0.0], requires_grad: true)
    let grad_a = jt.Tensor([3], [3.0, -4.0, 1.0])
    let grad_b = jt.Tensor([2], [2.0, 0.5])
    a.grad = grad_a
    b.grad = grad_b
    let norm = jt.clip_grad_norm([a, unused, b], 1.5)
    assert math.fabs(norm.download()[0] - 5.5) < 1e-5
    norm.free()
    assert_close(
        grad_a.download(),
        [0.8181816694215147, -1.0909088925620196, 0.2727272231405049],
        1e-6,
        "a"
    )
    assert_close(grad_b.download(), [0.5454544462810098, 0.13636361157025245], 1e-6, "b")
    grad_a.upload([0.3, -0.4, 0.1])
    grad_b.upload([0.2, 0.05])
    let inside = jt.clip_grad_norm([a, b], 1.5)
    assert math.fabs(inside.download()[0] - 0.55) < 1e-6
    inside.free()
    assert_close(grad_a.download(), [0.3, -0.4, 0.1], 1e-7, "a inside")
    assert_close(grad_b.download(), [0.2, 0.05], 1e-7, "b inside")
    let _bound = assert_throws(ValueError, || jt.clip_grad_norm([a], 0.0))
    let _nan_bound = assert_throws(ValueError, || jt.clip_grad_norm([a], math.NAN))
    a.free()
    b.free()
    unused.free()
}

#: An infinite `max_norm` is PyTorch's way of asking for the norm without
#: clipping (`clip_grad_norm_(params, float("inf"))` returns 5.0 here and
#: leaves the gradients alone). It used to throw converting the bound, after
#: three scratch buffers had been allocated, and leave them behind.
fn test_clip_grad_norm_with_an_infinite_bound_only_measures() -> void {
    if not jt.is_available() { return }
    let p = jt.Tensor([5], [0.0, 0.0, 0.0, 0.0, 0.0], requires_grad: true)
    let grad = jt.Tensor([5], [3.0, 4.0, 0.0, 0.0, 0.0])
    p.grad = grad
    let live = gpu.live_buffers()
    let norm = jt.clip_grad_norm([p], math.INF)
    assert math.fabs(norm.download()[0] - 5.0) < 1e-6
    norm.free()
    assert_eq(gpu.live_buffers(), live)
    assert_close(grad.download(), [3.0, 4.0, 0.0, 0.0, 0.0], 0.0, "unclipped")
    p.free()
}

#: A NaN anywhere makes the norm NaN, and PyTorch's clipping coefficient
#: `min(max_norm / (nan + 1e-6), 1)` is NaN too, so every gradient becomes
#: NaN. Metal's `min` returns the number for a NaN, which used to leave the
#: finite gradients as they were.
fn test_clip_grad_norm_with_a_nan_norm_poisons_every_gradient() -> void {
    if not jt.is_available() { return }
    let a = jt.Tensor([2], [0.0, 0.0], requires_grad: true)
    let b = jt.Tensor([3], [0.0, 0.0, 0.0], requires_grad: true)
    let grad_a = jt.Tensor([2], [math.NAN, 1.0])
    let grad_b = jt.Tensor([3], [2.0, -2.0, 0.5])
    a.grad = grad_a
    b.grad = grad_b
    let norm = jt.clip_grad_norm([a, b], 1.0)
    assert math.is_nan(norm.download()[0])
    norm.free()
    for value in grad_a.download() { assert math.is_nan(value), f"a: {value}" }
    for value in grad_b.download() { assert math.is_nan(value), f"b: {value}" }
    a.free()
    b.free()
}

#: Many threadgroups per gradient, and many gradients: the partial sums have
#: to land in their own slots and all of them be joined.
fn test_clip_grad_norm_sums_large_gradients() -> void {
    if not jt.is_available() { return }
    var parameters: list[Tensor] = []
    var total = 0.0
    for size in [1, 4095, 4097, 300000, 77] {
        let parameter = jt.Tensor([size], requires_grad: true)
        parameter.grad = jt.Tensor.full([size], 0.5)
        total += 0.25 * float(size)
        parameters.push(parameter)
    }
    let norm = jt.clip_grad_norm(parameters, 1e9)
    assert math.fabs(norm.download()[0] - math.sqrt(total)) < 1e-3 * math.sqrt(total)
    norm.free()
    let clipped = jt.clip_grad_norm(parameters, 2.0)
    clipped.free()
    let after = jt.clip_grad_norm(parameters, 1e9)
    assert math.fabs(after.download()[0] - 2.0) < 1e-4
    after.free()
    for parameter in parameters { parameter.free() }
}

#: `one_cycle_lr` against `torch.optim.lr_scheduler.OneCycleLR` (cosine, two
#: phases, its default divisors), read before each step.
fn test_one_cycle_lr_matches_torch() -> void {
    let torch_lrs = [
        0.008000000000000007,
        0.026334368540005065,
        0.07433436854000505,
        0.13366563145999494,
        0.18166563145999495,
        0.2,
        0.19749280124701749,
        0.19009692640269477,
        0.17818323551421,
        0.1623491307899526,
        0.14338860035826018,
        0.12225240438725786,
        0.1000004,
        0.07774839561274215,
        0.056612199641739844,
        0.0376516692100474,
        0.021817564485790017,
        0.009903873597305258,
        0.002507998752982511,
        8e-07,
    ]
    for (step, want) in torch_lrs.enumerate() {
        let got = jt.one_cycle_lr(step, 20, 0.2)
        assert math.fabs(got - want) < 1e-12, f"step {step}: {got} against torch {want}"
    }
    let steps = [0, 1, 1000, 2931, 2932, 2933, 5000, 11728, 11729]
    let long = [
        0.016000000000000014,
        0.016000110253183875,
        0.11609708476489422,
        0.3999999724367021,
        0.39999999681199155,
        0.39999997130792453,
        0.34787414387959736,
        1.6127520337131734e-06,
        1.6e-06,
    ]
    for (index, step) in steps.enumerate() {
        let got = jt.one_cycle_lr(step, 11730, 0.4, pct_start: 0.25)
        assert math.fabs(got - long[index]) < 1e-12, f"step {step}: {got} against torch {long[index]}"
    }
    let _past = assert_throws(ValueError, || jt.one_cycle_lr(21, 20, 0.1))
}

#: The float arguments go to the device as bits, and a number just under a
#: power of two must round up to it rather than come out as half of it.
fn test_float32_bits_rounds_like_a_float_buffer() -> void {
    let values = [
        1.0,
        0.99999999,
        1.9999999999,
        0.24999999999,
        0.1,
        -0.1,
        5e-4,
        3.4028234663852886e38,
        1e-40,
        0.0,
        math.INF,
        -math.INF,
    ]
    let wanted = [
        1065353216,
        1065353216,
        1073741824,
        1048576000,
        1036831949,
        3184315597,
        973279855,
        2139095039,
        71362,
        0,
        2139095040,
        4286578688,
    ]
    for (index, value) in values.enumerate() {
        let got = float32_bits(value)
        assert got == wanted[index], f"{value}: {got} against {wanted[index]}"
    }
}

#: A model whose optimizer decays its weights does not take a path that has
#: no decay term: the fused and compiled steps keep their own update and
#: would quietly train without it. It takes a path that calls `step`, and
#: the decay shows in the weights.
fn test_a_decaying_optimizer_takes_a_path_that_applies_it() -> void {
    if not jt.is_available() { return }
    let was = compiled_step_allowed()
    defer { set_compiled_step(was) }
    var norms: list[float] = []
    for decay in [0.0, 0.5] {
        #: The undecayed model is held to the fused path it would take with
        #: compiling off; the decayed one may compile, and must not.
        set_compiled_step(decay != 0.0)
        let data = rows_that_carry_their_class(64, 16, 4)
        let model = jt.Sequential(
            [
                jt.Dense(256, activation: jt.Activation.Relu),
                jt.Dense(4, activation: jt.Activation.Softmax),
            ]
        )
        model.compile(
            optimizer: jt.SGD(learning_rate: 0.05, momentum: 0.9, weight_decay: decay),
            loss: jt.SparseCategoricalCrossentropy()
        )
        let _history = model.fit(data, epochs: 3, batch_size: 16, verbose: false, plot_path: null)
        let route = model.training_route()
        if decay == 0.0 {
            assert route == "fused-mlp", f"undecayed took {route}"
        } else {
            assert route != "fused-mlp" and route != "fused-sgd" and route != "compiled", f"decayed took {route}"
        }
        var total = 0.0
        for parameter in model.layers[0].parameters() {
            for value in parameter.download() { total += value * value }
        }
        norms.push(total)
        model.free()
        data.free()
    }
    assert norms[1] < 0.95 * norms[0], f"decay left the weights at {norms[1]} against {norms[0]}"
}

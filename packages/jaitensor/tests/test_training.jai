#: `Sequential` end to end: fit, predict, evaluate, and what a run leaves behind.
import jaitensor as jt
import std.math
import std.os
from jaitensor import Dataset
from std.io import Path, read_file
from std.random import Random
from std.test import assert_eq, assert_near, assert_throws, assert_true

fn separable_data(repeats: int = 32) -> Dataset {
    var features: list[float] = []
    var labels: list[int] = []
    for _repeat in 0..repeats {
        features.extend([-2.0, -1.0, -1.0, -2.0, 2.0, 1.0, 1.0, 2.0])
        labels.extend([0, 0, 1, 1])
    }
    return jt.Dataset.from_flat(features, labels, 2)
}

fn assert_values_near(actual: list[float], expected: list[float], tolerance: float = 1e-5) -> void {
    assert_eq(actual.len(), expected.len())
    for (index, value) in actual.enumerate() {
        assert_near(value, expected[index], tolerance)
    }
}

fn test_dense_forward_matches_a_hand_computed_affine_plus_relu() -> void {
    if not jt.is_available() { return }
    let layer = jt.Dense(2, activation: jt.Activation.Relu)
    let rng = Random.with_seed(1)
    layer.build(3, rng)
    layer.set_weights([1.0, 0.0, 0.0, 1.0, -1.0, 1.0], [0.5, -0.5])
    let x = jt.Tensor([1, 3], [1.0, 2.0, 3.0])
    let y = layer.forward(x)
    assert_eq(y.download(), [0.0, 4.5])
    let intermediates = y.discard_graph()
    for tensor in intermediates { tensor.free() }
    x.free()
    y.free()
    layer.free()
}

fn test_a_tiny_sequential_models_loss_decreases_over_training_steps() -> void {
    if not jt.is_available() { return }
    let features = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    let labels = [0, 1, 1, 0]
    let data = jt.Dataset.from_flat(features, labels, 2)
    defer { data.free() }
    let model = jt.Sequential(
        [
            jt.Dense(8, activation: jt.Activation.Tanh),
            jt.Dense(2, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(jt.Adam(learning_rate: 0.05), loss: jt.SparseCategoricalCrossentropy())
    let history = model.fit(data, epochs: 30, batch_size: 4, verbose: false, plot_path: null)
    assert history.loss[history.loss.len() - 1] < history.loss[0]
}

fn test_fit_leaves_no_leftover_gradient_on_any_parameter_between_batches() -> void {
    if not jt.is_available() { return }
    let data = separable_data()
    defer { data.free() }
    let model = jt.Sequential([jt.Dense(2, activation: jt.Activation.Softmax)])
    defer { model.free() }
    model.compile(jt.SGD(learning_rate: 0.1))
    model.fit(data, epochs: 2, batch_size: 64, verbose: false, plot_path: null)
    for layer in model.layers {
        for parameter in layer.parameters() {
            assert_true(parameter.grad is null)
        }
    }
}

fn test_fit_rejects_a_label_outside_the_models_class_count() -> void {
    if not jt.is_available() { return }
    let data = jt.Dataset.from_flat([0.0, 0.0, 0.0, 1.0], [0, 5], 2)
    defer { data.free() }
    let model = jt.Sequential([jt.Dense(2, activation: jt.Activation.Softmax)])
    defer { model.free() }
    model.compile(jt.SGD())
    let _error = assert_throws(ValueError, fn() -> any {
        model.fit(data, epochs: 1, batch_size: 2, verbose: false, plot_path: null)
        return null
    })
}

fn test_sequential_sgd_predicts_and_reports_validation_metrics() -> void {
    if not jt.is_available() { return }
    let data = separable_data()
    let model = jt.Sequential([jt.Dense(2, activation: jt.Activation.Softmax)])
    defer { data.free() }
    defer { model.free() }
    let plot = Path(os.temp_dir()).join("jaithon-jaitensor-fit-test.svg")
    defer { if plot.exists() { plot.remove() } }
    model.compile(jt.SGD(learning_rate: 0.4, momentum: 0.2))
    let history = model.fit(
        data,
        epochs: 8,
        batch_size: 64,
        verbose: false,
        validation_data: data,
        test_data: data,
        plot_path: plot.to_str()
    )
    assert_eq(history.accuracy.len(), 8)
    assert_eq(history.val_accuracy.len(), 8)
    assert_true(history.test_accuracy is not null)
    assert_true((history.test_accuracy ?? 0.0) >= 99.0)
    assert_eq(history.plot_path, plot.to_str())
    assert_true(plot.is_file())
    assert_true(history.accuracy[7] >= 99.0, f"accuracy was {history.accuracy[7]}")
    let probabilities = model.predict(data.features, batch_size: 31)
    defer { probabilities.free() }
    assert_eq(probabilities.shape, [128, 2])
    assert_eq(model.predict_classes(data.features).len(), 128)
}

fn test_predict_returns_a_detached_tensor_with_no_creator() -> void {
    if not jt.is_available() { return }
    let data = separable_data()
    defer { data.free() }
    let model = jt.Sequential([jt.Dense(2, activation: jt.Activation.Softmax)])
    defer { model.free() }
    model.compile(jt.SGD())
    let probabilities = model.predict(data.features)
    defer { probabilities.free() }
    assert_true(not probabilities.requires_grad)
    assert_true(probabilities.creator is null)
}

fn test_adam_trains_sigmoid_and_tanh_layers() -> void {
    if not jt.is_available() { return }
    let data = separable_data()
    let model = jt.Sequential(
        [
            jt.Dense(4, activation: jt.Activation.Tanh),
            jt.Dense(3, activation: jt.Activation.Sigmoid),
            jt.Dense(2, activation: jt.Activation.Linear),
        ]
    )
    defer { data.free() }
    defer { model.free() }
    model.compile(
        jt.Adam(learning_rate: 0.03),
        loss: jt.SparseCategoricalCrossentropy(from_logits: true)
    )
    let history = model.fit(data, epochs: 10, batch_size: 64, verbose: false, plot_path: null)
    assert_true(history.accuracy[9] >= 99.0, f"Adam accuracy was {history.accuracy[9]}")
}

#: The same parameter, stepped by one kind of optimizer and then another.
#:
#: The optimizer's state is kept in a table keyed by the parameter, and a
#: parameter does not know what stepped it last -- so Adam picked up whatever
#: plain descent had left in `first_moment` and read it as its own first
#: moment. After five momentum steps at 0.9 that velocity is about 41, and
#: Adam's opening step moved the parameter by 1.93 where a clean one moves
#: 0.01. A hundred and ninety times, silently.
#: The accuracy `fit` reports has to be the accuracy the model has.
#:
#: The fused path that runs a whole run of epochs on the device pushed a flat
#: zero into the history, every epoch, because it never asked the kernel for
#: the tally it was perfectly willing to keep. `History.accuracy` is
#: documented as the percentage after each epoch and `plot()` draws it, so a
#: model learning perfectly well showed a flat line along the bottom.
#: A table whose row length is not a multiple of four.
#:
#: The fused kernels take each batch as a window into the dataset, and the
#: platform's array view pads its innermost dimension out to sixteen bytes --
#: so it refuses to build a window that starts part way in and whose rows are
#: five, six, seven or nine floats long. `fit` did not fall back, it threw:
#: "gpu_mlp_sgd_epoch(): the kernel failed", for most of the widths a table
#: of measurements actually has. The first batch starts at offset zero and is
#: exempt, so a run of fewer than two batches appeared to work while training
#: on the first batch alone.
#: A model built for a flat row, then fitted on pictures of the same size.
#:
#: Sixty-four values is sixty-four values either way, and the build guard
#: compared only the count -- so it returned happy with every convolution
#: still sized for a flat row of 64, and the graph handed to the platform
#: failed its own verification and took the process down with SIGABRT. Not a
#: throw: exit 134, no traceback, nothing a `try` could catch.
fn test_building_flat_then_fitting_pictures_is_refused() -> void {
    if not jt.is_available() { return }
    var pixels: list[float] = []
    var labels: list[float] = []
    for index in 0..32 {
        for _slot in 0..64 { pixels.push(float(index % 5) * 0.2) }
        labels.push(float(index % 3))
    }
    let data = jt.Dataset(jt.Tensor([32, 8, 8, 1], pixels), jt.Tensor([32], labels))
    defer { data.free() }
    let model = jt.Sequential(
        [
            jt.Conv2d(4, 3, padding: 1, in_channels: 1, activation: jt.Activation.Relu),
            jt.MaxPool2d(2),
            jt.Flatten(),
            jt.Dense(3, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(optimizer: jt.SGD(learning_rate: 0.01), loss: jt.SparseCategoricalCrossentropy())
    model.build(64)
    assert_throws(
        ValueError,
        || model.fit(data, epochs: 1, batch_size: 8, verbose: false, plot_path: null)
    )
}

fn test_a_model_trains_on_any_number_of_features() -> void {
    if not jt.is_available() { return }
    for width in [4, 5, 6, 7, 9, 16] {
        var features: list[float] = []
        var labels: list[float] = []
        var seed = 4
        for _index in 0..256 {
            seed = (seed * 1103515245 + 12345) % 2147483648
            let kind = seed % 4
            for _slot in 0..width { features.push(float(kind) + float(seed % 50) / 200.0) }
            labels.push(float(kind))
        }
        let data = jt.Dataset(jt.Tensor([256, width], features), jt.Tensor([256], labels))
        let model = jt.Sequential(
            [
                jt.Dense(256, activation: jt.Activation.Relu),
                jt.Dense(4, activation: jt.Activation.Softmax),
            ]
        )
        model.compile(
            optimizer: jt.SGD(learning_rate: 0.01),
            loss: jt.SparseCategoricalCrossentropy()
        )
        #: More than two batches, so a window at a non-zero offset is really
        #: asked for.
        let history = model.fit(data, epochs: 1, batch_size: 64, verbose: false, plot_path: null)
        assert_eq(history.loss.len(), 1)
        assert math.is_finite(history.loss[0])
        model.free()
        data.free()
    }
}

fn test_the_fused_path_reports_the_accuracy_it_reached() -> void {
    if not jt.is_available() { return }
    var features: list[float] = []
    var labels: list[float] = []
    var seed = 4
    for _index in 0..256 {
        seed = (seed * 1103515245 + 12345) % 2147483648
        let kind = seed % 4
        for _slot in 0..16 { features.push(float(kind) + float(seed % 50) / 200.0) }
        labels.push(float(kind))
    }
    let data = jt.Dataset(jt.Tensor([256, 16], features), jt.Tensor([256], labels))
    defer { data.free() }
    let model = jt.Sequential(
        [
            jt.Dense(512, activation: jt.Activation.Relu),
            jt.Dense(4, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(optimizer: jt.SGD(learning_rate: 0.05), loss: jt.SparseCategoricalCrossentropy())
    #: No validation data and not verbose, which is what selects the path
    #: that runs every epoch on the device without coming back between them.
    let history = model.fit(data, epochs: 3, batch_size: 64, verbose: false, plot_path: null)
    assert_eq(history.accuracy.len(), 3)
    for value in history.accuracy {
        assert math.is_finite(value)
        assert value > 0.0
        assert value <= 100.0
    }
    #: And it agrees with what the model actually scores, near enough that a
    #: number invented from nowhere would not pass.
    let measured = model.evaluate(data).accuracy
    assert math.fabs(history.accuracy[2] - measured) < 15.0
}

fn test_a_second_kind_of_optimizer_starts_from_nothing() -> void {
    if not jt.is_available() { return }
    let shared = jt.Tensor([1], [0.0], requires_grad: true)
    for _round in 0..5 {
        shared.grad = jt.Tensor([1], [10.0])
        jt.SGD(learning_rate: 1.0e-9, momentum: 0.9).step([shared])
    }
    let before = shared.download()[0]
    shared.grad = jt.Tensor([1], [1.0])
    jt.Adam(learning_rate: 0.01).step([shared])
    let moved = shared.download()[0] - before

    let clean = jt.Tensor([1], [0.0], requires_grad: true)
    clean.grad = jt.Tensor([1], [1.0])
    jt.Adam(learning_rate: 0.01).step([clean])
    let expected = clean.download()[0]

    assert math.fabs(moved - expected) < 1e-6
    #: And that step really is the learning rate, so the comparison is not
    #: two wrong numbers agreeing.
    assert math.fabs(expected + 0.01) < 1e-4
    shared.free()
    clean.free()
}

fn test_history_plot_uses_the_jaiplot_package_dependency() -> void {
    let history = jt.History()
    history.loss.extend([1.2, 0.7, 0.4])
    history.accuracy.extend([42.0, 71.0, 89.0])
    history.val_loss.extend([1.3, 0.8, 0.5])
    history.val_accuracy.extend([39.0, 68.0, 85.0])
    history.test_loss = 0.45
    history.test_accuracy = 87.0
    let path = Path(os.temp_dir()).join("jaithon-jaitensor-history-test.svg")
    defer { if path.exists() { path.remove() } }
    history.plot(path.to_str())
    let document = read_file(path.to_str())
    assert_true(document.starts_with("<svg"))
    assert_true(document.contains("validation"))
    assert_true(document.contains("test"))
    assert_eq(history.plot_path, path.to_str())
}

fn test_a_convolution_actually_trains() -> void {
    if not jt.is_available() { return }
    #: Conv2d recorded no gradient at all until its backward pass existed, and
    #: `Flatten` handed the head a view with no creator, so the graph stopped
    #: at the first dense layer. Both are invisible from the outside: the model
    #: runs, the loss moves a little as the head fits, and the filters never
    #: change. This asserts the filters move.
    let batch = 16
    var pixels: list[float] = []
    var labels: list[float] = []
    for sample in 0..batch {
        let kind = sample % 2
        for row in 0..8 {
            for column in 0..8 {
                pixels.push(if kind == 0 { float(row) * 0.1 } else { float(column) * 0.1 })
            }
        }
        labels.push(float(kind))
    }
    let data = jt.Dataset(jt.Tensor([batch, 8, 8, 1], pixels), jt.Tensor([batch], labels))
    defer { data.free() }
    let conv = jt.Conv2d(4, 3, padding: 1, in_channels: 1, activation: jt.Activation.Relu)
    let model = jt.Sequential(
        [
            conv,
            jt.MaxPool2d(2),
            jt.Flatten(),
            jt.Dense(2, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(jt.SGD(learning_rate: 0.2))
    model.fit(data, epochs: 1, batch_size: 8, verbose: false, plot_path: null)
    let before = conv.parameters()[0].download()
    let history = model.fit(data, epochs: 6, batch_size: 8, verbose: false, plot_path: null)
    let after = conv.parameters()[0].download()
    var moved = 0.0
    for index in 0..before.len() {
        let step = if after[index] > before[index] {
            after[index] - before[index]
        } else {
            before[index] - after[index]
        }
        if step > moved { moved = step }
    }
    assert_true(moved > 1e-4)
    assert_true(history.loss[5] < history.loss[0])
}

fn conv_drift_after(samples: int, batch_size: int) -> float {
    var pixels: list[float] = []
    var labels: list[float] = []
    for _sample in 0..samples {
        for index in 0..64 { pixels.push(float(index * 7 % 11) * 0.1 - 0.5) }
        labels.push(0.0)
    }
    let data = jt.Dataset(jt.Tensor([samples, 8, 8, 1], pixels), jt.Tensor([samples], labels))
    defer { data.free() }
    let conv = jt.Conv2d(4, 3, padding: 1, in_channels: 1, activation: jt.Activation.Relu)
    let model = jt.Sequential([conv, jt.Flatten(), jt.Dense(2, activation: jt.Activation.Softmax)])
    defer { model.free() }
    #: Small enough that the parameters barely move, so every batch sees the
    #: same gradient and the totals are comparable.
    model.compile(jt.SGD(learning_rate: 1e-6))
    #: One whole-batch pass first, so the layers are built and both variants
    #: start from the same weights.
    model.fit(data, epochs: 1, batch_size: samples, verbose: false, plot_path: null)
    let before = conv.parameters()[0].download()
    model.fit(data, epochs: 1, batch_size: batch_size, verbose: false, plot_path: null)
    let after = conv.parameters()[0].download()
    var total = 0.0
    for index in 0..before.len() {
        let step = after[index] - before[index]
        total += step * step
    }
    return math.sqrt(total)
}

fn test_gradients_do_not_carry_between_batches() -> void {
    if not jt.is_available() { return }
    #: `Optimizer.step` reads `.grad` and leaves it alone; the training loop is
    #: what resets it. Without that reset, an op whose backward allocates a
    #: fresh gradient summed its batches, so four identical batches moved the
    #: filters by one, two, three and four gradients instead of one each —
    #: ten steps of drift against four. The two dense ops that write their
    #: gradient in place were immune, so every existing model hid it.
    let one_batch = conv_drift_after(4, 4)
    let four_batches = conv_drift_after(4, 1)
    assert_true(one_batch > 0.0)
    let ratio = four_batches / one_batch
    assert_true(ratio > 3.5)
    assert_true(ratio < 4.5)
}

fn test_sequential_predict_skips_dropout() -> void {
    if not jt.is_available() { return }
    let data = separable_data(8)
    defer { data.free() }
    let model = jt.Sequential(
        [
            jt.Dense(4, activation: jt.Activation.Relu),
            jt.Dropout(0.9),
            jt.Dense(2, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(jt.SGD(learning_rate: 0.05))
    model.fit(data, epochs: 1, batch_size: 8, verbose: false, plot_path: null)
    let batch = data.features.view(0, [8, 2])
    defer { batch.free() }
    let first = model.predict(batch)
    defer { first.free() }
    let second = model.predict(batch)
    defer { second.free() }
    assert_values_near(first.download(), second.download(), 1e-5)
}

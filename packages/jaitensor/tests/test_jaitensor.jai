#: GPU tensor operations, generic datasets, optimizers, and model lifecycle.
import jaitensor as jt
import std.gpu as gpu
import std.math
import std.os
from jaitensor import Dataset
from jaitensor.gemm import default_plan, plan_from_text
from jaitensor.ops import dense_static_forward, matmul_backend
from jaitensor.tensor import GradNode, Tensor
from std.io import Path, read_file
from std.random import Random
from std.test import assert_eq, assert_near, assert_throws, assert_true
from std.trace import is_traced

fn assert_values_near(actual: list[float], expected: list[float], tolerance: float = 1e-5) -> void {
    assert_eq(actual.len(), expected.len())
    for (index, value) in actual.enumerate() {
        assert_near(value, expected[index], tolerance)
    }
}

fn separable_data(repeats: int = 32) -> Dataset {
    var features: list[float] = []
    var labels: list[int] = []
    for _repeat in 0..repeats {
        features.extend([-2.0, -1.0, -1.0, -2.0, 2.0, 1.0, 1.0, 2.0])
        labels.extend([0, 0, 1, 1])
    }
    return jt.Dataset.from_flat(features, labels, 2)
}

fn test_tensor_constructors_shape_helpers_and_round_trip() -> void {
    if not jt.is_available() { return }
    let tensor = jt.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])
    let constant = jt.constant([5.0, 6.0], [2])
    let zeroes = jt.zeros([2])
    let ones = jt.ones([3])
    let full = jt.fill([2], 2.5)
    defer { tensor.free() }
    defer { constant.free() }
    defer { zeroes.free() }
    defer { ones.free() }
    defer { full.free() }
    assert_eq(tensor.shape, [2, 2])
    assert_eq(tensor.rank(), 2)
    assert_eq(tensor.dim(0), 2)
    assert_eq(tensor.dim(-1), 2)
    assert_eq(tensor.download(), [1.0, 2.0, 3.0, 4.0])
    assert_eq(constant.download(), [5.0, 6.0])
    assert_eq(zeroes.download(), [0.0, 0.0])
    assert_eq(ones.download(), [1.0, 1.0, 1.0])
    assert_eq(full.download(), [2.5, 2.5])
    let _error = assert_throws(IndexError, || tensor.dim(2))
}

fn test_dataset_constructors_are_format_agnostic() -> void {
    if not jt.is_available() { return }
    let rows = jt.Dataset.from_tensor_slices([[1.0, 2.0], [3.0, 4.0]], [4, 7])
    let packed = jt.Dataset.from_u8(
        bytes([99, 2, 4, 6, 8]),
        bytes([99, 3, 5]),
        2,
        2,
        feature_offset: 1,
        label_offset: 1,
        feature_scale: 0.5
    )
    defer { rows.free() }
    defer { packed.free() }
    assert_eq(rows.samples, 2)
    assert_eq(rows.feature_count, 2)
    assert_eq(rows.batch_count(1), 2)
    assert_eq(rows.batch_count(3), 1)
    assert_eq(rows.labels.download(), [4.0, 7.0])
    assert_eq(packed.features.download(), [1.0, 2.0, 3.0, 4.0])
    assert_eq(packed.labels.download(), [3.0, 5.0])
    let _ragged = assert_throws(ValueError, || jt.Dataset.from_rows([[1.0], [2.0, 3.0]], [0, 1]))
}

fn test_elementwise_tensor_operations_run_on_device() -> void {
    if not jt.is_available() { return }
    let a = jt.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])
    let b = jt.Tensor([2, 2], [4.0, 3.0, 2.0, 1.0])
    let added = jt.add(a, b)
    let subtracted = jt.subtract(a, b)
    let multiplied = jt.multiply(a, b)
    let divided = jt.divide(a, b)
    let scaled = jt.scale(a, 0.5)
    let reshaped = jt.reshape(a, [4])
    defer { a.free() }
    defer { b.free() }
    defer { added.free() }
    defer { subtracted.free() }
    defer { multiplied.free() }
    defer { divided.free() }
    defer { scaled.free() }
    defer { reshaped.free() }
    assert_values_near(added.download(), [5.0, 5.0, 5.0, 5.0])
    assert_values_near(subtracted.download(), [-3.0, -1.0, 1.0, 3.0])
    assert_values_near(multiplied.download(), [4.0, 6.0, 6.0, 4.0])
    assert_values_near(divided.download(), [0.25, 2.0 / 3.0, 1.5, 4.0])
    assert_values_near(scaled.download(), [0.5, 1.0, 1.5, 2.0])
    assert_eq(reshaped.shape, [4])
}

fn test_matrix_activation_and_classification_operations() -> void {
    if not jt.is_available() { return }
    let a = jt.Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    let b = jt.Tensor([3, 2], [1.0, 2.0, 0.0, 1.0, 2.0, 0.0])
    let product = jt.matmul(a, b)
    let transposed = jt.transpose(a)
    let values = jt.Tensor([1, 3], [-1.0, 0.0, 1.0])
    let relu = jt.relu(values)
    let sigmoid = jt.sigmoid(values)
    let tanh = jt.tanh(values)
    let probabilities = jt.softmax(values)
    let classes = jt.argmax(probabilities)
    let hot = jt.one_hot(classes, 3)
    defer { a.free() }
    defer { b.free() }
    defer { product.free() }
    defer { transposed.free() }
    defer { values.free() }
    defer { relu.free() }
    defer { sigmoid.free() }
    defer { tanh.free() }
    defer { probabilities.free() }
    defer { classes.free() }
    defer { hot.free() }
    assert_values_near(product.download(), [7.0, 4.0, 16.0, 13.0])
    assert_values_near(transposed.download(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
    assert_values_near(relu.download(), [0.0, 0.0, 1.0])
    assert_values_near(sigmoid.download(), [0.268941, 0.5, 0.731059], 1e-5)
    assert_values_near(tanh.download(), [-0.761594, 0.0, 0.761594], 1e-5)
    let result = probabilities.download()
    assert_near(result[0] + result[1] + result[2], 1.0, 1e-5)
    assert_eq(classes.download(), [2.0])
    assert_eq(hot.download(), [0.0, 0.0, 1.0])
}

fn test_add_broadcasts_a_trailing_suffix_shape() -> void {
    if not jt.is_available() { return }
    let batched = jt.Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    let bias = jt.Tensor([3], [10.0, 20.0, 30.0])
    let result = jt.add(batched, bias)
    assert_eq(result.download(), [11.0, 22.0, 33.0, 14.0, 25.0, 36.0])
    batched.free()
    bias.free()
    result.free()
}

fn test_slice_flat_copies_a_device_range() -> void {
    if not jt.is_available() { return }
    let source = jt.Tensor([4, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    defer { source.free() }
    let sliced = jt.slice_flat(source, 3, [2, 3])
    defer { sliced.free() }
    assert_eq(sliced.shape, [2, 3])
    assert_eq(sliced.download(), [4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    assert_true(not sliced.requires_grad)
    assert_true(sliced.creator is null)
    let _error = assert_throws(ValueError, || jt.slice_flat(source, 10, [2, 3]))
}

fn test_tensor_view_reads_a_parent_range_without_copying() -> void {
    if not jt.is_available() { return }
    let source = jt.Tensor([4, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    defer { source.free() }
    let viewed = source.view(3, [2, 3])
    defer { viewed.free() }
    assert_eq(viewed.shape, [2, 3])
    assert_eq(viewed.download(), [4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    let _error = assert_throws(ValueError, || source.view(10, [2, 3]))
}

fn test_device_metric_kernels_accumulate_loss_and_correct_counts() -> void {
    if not jt.is_available() { return }
    let logits = jt.Tensor([2, 3], [0.0, 2.0, 1.0, 4.0, 0.0, 1.0])
    let labels = jt.Tensor([2], [1.0, 0.0])
    let correct = jt.zeros([1])
    let loss_total = jt.zeros([1])
    let mean = jt.Tensor([1], [0.5])
    defer { logits.free() }
    defer { labels.free() }
    defer { correct.free() }
    defer { loss_total.free() }
    defer { mean.free() }
    jt.add_correct_into(logits, labels, correct)
    jt.add_scaled_into(mean, loss_total, 4)
    assert_values_near(correct.download(), [2.0])
    assert_values_near(loss_total.download(), [2.0])
}

fn test_add_rejects_a_shape_that_is_not_a_trailing_suffix() -> void {
    if not jt.is_available() { return }
    let a = jt.Tensor([2, 3], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    let b = jt.Tensor([4], [0.0, 0.0, 0.0, 0.0])
    var raised = false
    try { jt.add(a, b) } catch e: ValueError { raised = true }
    assert raised
    a.free()
    b.free()
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

fn test_weights_round_trip_through_memory_and_json() -> void {
    if not jt.is_available() { return }
    let data = separable_data(2)
    let source = jt.Sequential([jt.Dense(2, activation: jt.Activation.Softmax)])
    let restored = jt.Sequential([jt.Dense(2, activation: jt.Activation.Softmax)])
    let path = Path(os.temp_dir()).join("jaithon-jaitensor-weights-test.json")
    defer { data.free() }
    defer { source.free() }
    defer { restored.free() }
    defer { if path.exists() { path.remove() } }
    source.compile(jt.SGD())
    source.build(2)
    let weights = source.get_weights()
    source.set_weights(weights)
    source.save_weights(path.to_str())
    restored.compile(jt.SGD())
    restored.load_weights(path.to_str())
    assert_eq(restored.get_weights(), weights)
    assert_eq(restored.predict_classes(data.features), source.predict_classes(data.features))
}

#: A model with a convolution in it, saved and loaded back.
#:
#: The file recorded how many numbers a sample holds and not what shape they
#: are in, and a convolution cannot be rebuilt from the product: the load
#: went down the flat path, every shape-dependent layer came out the wrong
#: size, and it threw on the first parameter that did not match. So no model
#: with a convolution in it could be reloaded at all -- which is most of the
#: reason to save one.
fn test_a_convolutional_model_survives_a_save_and_a_load() -> void {
    if not jt.is_available() { return }
    let path = Path(os.temp_dir()).join("jaithon-jaitensor-conv-weights.json")
    defer { if path.exists() { path.remove() } }
    var pixels: list[float] = []
    var labels: list[float] = []
    var seed = 9
    for _sample in 0..24 {
        seed = (seed * 1103515245 + 12345) % 2147483648
        for _slot in 0..8 * 8 { pixels.push(float(seed % 100) / 100.0) }
        labels.push(float(seed % 3))
    }
    let data = jt.Dataset(jt.Tensor([24, 8, 8, 1], pixels), jt.Tensor([24], labels))
    defer { data.free() }

    let source = jt.Sequential(
        [
            jt.Conv2d(4, 3, padding: 1, in_channels: 1, activation: jt.Activation.Relu),
            jt.MaxPool2d(2),
            jt.Flatten(),
            jt.Dense(3, activation: jt.Activation.Softmax),
        ]
    )
    defer { source.free() }
    source.compile(optimizer: jt.SGD(learning_rate: 0.05), loss: jt.SparseCategoricalCrossentropy())
    source.fit(data, epochs: 2, batch_size: 8, verbose: false, plot_path: null)
    let flat = jt.Tensor([24, 64], pixels)
    let before = source.predict(flat).download()
    source.save_weights(path.to_str())

    let restored = jt.Sequential(
        [
            jt.Conv2d(
                4,
                3,
                padding: 1,
                in_channels: 1,
                activation: jt.Activation.Relu
            ),
            jt.MaxPool2d(2),
            jt.Flatten(),
            jt.Dense(3, activation: jt.Activation.Softmax),
        ]
    )
    defer { restored.free() }
    restored.load_weights(path.to_str())
    let after = restored.predict(flat).download()
    assert_eq(after.len(), before.len())
    for (index, value) in before.enumerate() {
        assert math.fabs(value - after[index]) < 1e-6
    }
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

fn test_conv2d_matches_spatial_output_and_channel_count() -> void {
    if not jt.is_available() { return }
    let x = jt.Tensor(
        [1, 4, 4, 1],
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
        ]
    )
    let weights = jt.Tensor([2, 2, 1, 1], [1.0, 0.0, 0.0, 0.0])
    let y = jt.conv2d(x, weights, stride: 1, padding: 0)
    defer { x.free() }
    defer { weights.free() }
    defer { y.free() }
    assert_eq(y.shape, [1, 3, 3, 1])
    let values = y.download()
    assert_near(values[0], 1.0)
    assert_near(values[1], 2.0)
    assert_near(values[3], 5.0)
}

fn test_layer_norm_centers_each_row() -> void {
    if not jt.is_available() { return }
    let x = jt.Tensor([2, 4], [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0])
    let gamma = jt.ones([4])
    let beta = jt.zeros([4])
    let y = jt.layer_norm(x, gamma, beta)
    defer { x.free() }
    defer { gamma.free() }
    defer { beta.free() }
    defer { y.free() }
    let values = y.download()
    var mean = 0.0
    for index in 0..4 { mean += values[index] }
    assert_near(mean / 4.0, 0.0, 1e-4)
}

fn test_attention_keeps_sequence_shape() -> void {
    if not jt.is_available() { return }
    let q = jt.Tensor([2, 4], [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    let k = jt.Tensor([2, 4], [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    let v = jt.Tensor([2, 4], [2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0])
    let y = jt.scaled_dot_product_attention(q, k, v)
    defer { q.free() }
    defer { k.free() }
    defer { v.free() }
    defer { y.free() }
    assert_eq(y.shape, [2, 4])
    let same = jt.multi_head_attention(q, k, v, 1)
    defer { same.free() }
    assert_values_near(y.download(), same.download(), 1e-5)
}

#: Two epochs of the same run have to see the samples in different orders.
#:
#: Each of the three epoch bodies built its own loader with the default seed,
#: so every epoch of a run drew the same permutation -- the samples were
#: reordered once when training began and then frozen in that order, which is
#: most of what shuffling is for. Nothing observable went wrong; the model
#: simply saw the same batches over and over.
fn test_a_loader_per_epoch_orders_the_samples_differently() -> void {
    if not jt.is_available() { return }
    var features: list[float] = []
    var labels: list[int] = []
    for index in 0..24 {
        features.push(float(index))
        labels.push(index)
    }
    let data = jt.Dataset.from_flat(features, labels, 1)
    defer { data.free() }
    var seen: list[str] = []
    for epoch in 1..4 {
        let loader = jt.DataLoader(data, 6, shuffle: true, seed: epoch)
        let (batch_features, batch_labels) = loader.next_batch()
        var order = ""
        for value in batch_labels.download() { order += f"{int(value)}," }
        assert not seen.contains(order)
        seen.push(order)
        batch_features.free()
        batch_labels.free()
        #: And the order tensor goes back, which is the other half of what
        #: building one of these per epoch used to cost.
        loader.free()
        loader.free()
    }
}

#: An embedding fed a batch of index rows, which is how a model in this
#: library is trained.
#:
#: The layer only ever accepted a flat list of indices, so putting one in a
#: `Sequential` and calling `fit` threw on the first forward: a batch of four
#: contexts arrived as `[4, context]` and the gather refused anything but
#: rank one. There was no way to use the layer in a batched model at all.
fn test_an_embedding_takes_a_batch_of_index_rows() -> void {
    if not jt.is_available() { return }
    let vocab = 11
    let context = 3
    let width = 5
    var indices: list[float] = []
    var labels: list[int] = []
    var seed = 5
    for _sample in 0..32 {
        for _slot in 0..context {
            seed = (seed * 1103515245 + 12345) % 2147483648
            indices.push(float(seed % vocab))
        }
        labels.push(seed % vocab)
    }
    let data = jt.Dataset(
        jt.Tensor([32, context], indices),
        jt.Tensor([32], [float(value) for value in labels])
    )
    defer { data.free() }
    let embedding = jt.Embedding(vocab, width)
    let model = jt.Sequential(
        [
            embedding,
            jt.Dense(16, activation: jt.Activation.Relu),
            jt.Dense(vocab, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(optimizer: jt.Adam(learning_rate: 0.02), loss: jt.SparseCategoricalCrossentropy())
    let history = model.fit(data, epochs: 4, batch_size: 8, verbose: false, plot_path: null)
    #: One row of indices becomes `context` lookups laid end to end, so the
    #: layer after it sees that much width. The layer learns this when it is
    #: built, which is the first time the model is fitted.
    assert_eq(embedding.output_width(), context * width)
    assert_eq(history.loss.len(), 4)
    for value in history.loss { assert math.is_finite(value) }
    assert history.loss[3] < history.loss[0]
}

fn test_dataloader_shuffle_covers_every_sample() -> void {
    if not jt.is_available() { return }
    let data = jt.Dataset.from_flat([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0, 1, 2], 2)
    defer { data.free() }
    let loader = jt.DataLoader(data, 2, shuffle: true, seed: 3)
    var seen = 0
    while loader.has_next() {
        let (features, labels) = loader.next_batch()
        seen += labels.size
        features.free()
        labels.free()
    }
    assert_eq(seen, 3)
}

fn test_dataloader_shuffle_keeps_labels_with_their_rows() -> void {
    if not jt.is_available() { return }
    #: Row `i` carries its own answer in its first column, so a batch that
    #: pairs a row with somebody else's label cannot go unnoticed.
    let data = jt.Dataset.from_flat(
        [10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0],
        [1, 2, 3, 4],
        2
    )
    defer { data.free() }
    let loader = jt.DataLoader(data, 2, shuffle: true, seed: 3)
    var seen = 0
    while loader.has_next() {
        let (features, labels) = loader.next_batch()
        let rows = features.download()
        let answers = labels.download()
        for index in 0..answers.len() {
            assert_near(
                answers[index],
                rows[index * 2] / 10.0,
                1e-6,
                f"row {index} was handed the wrong label"
            )
            seen += 1
        }
        features.free()
        labels.free()
    }
    assert_eq(seen, 4)
}

fn test_dataloader_shuffle_keeps_images_shaped() -> void {
    if not jt.is_available() { return }
    var pixels: list[float] = []
    var answers: list[float] = []
    for sample in 0..4 {
        for _slot in 0..2 * 2 * 3 { pixels.push(float(sample)) }
        answers.push(float(sample))
    }
    let data = jt.Dataset(jt.Tensor([4, 2, 2, 3], pixels), jt.Tensor([4], answers))
    defer { data.free() }
    let loader = jt.DataLoader(data, 2, shuffle: true, seed: 7)
    while loader.has_next() {
        let (features, labels) = loader.next_batch()
        assert_eq(features.shape, [2, 2, 2, 3], "a shuffled image batch lost its shape")
        let rows = features.download()
        let picked = labels.download()
        for index in 0..picked.len() {
            assert_near(
                picked[index],
                rows[index * 12],
                1e-6,
                f"image {index} was handed the wrong label"
            )
        }
        features.free()
        labels.free()
    }
}

fn test_conv2d_layer_builds_from_image_shape() -> void {
    if not jt.is_available() { return }
    var pixels: list[float] = []
    var labels: list[int] = []
    for sample in 0..4 {
        for _pixel in 0..16 { pixels.push(float(sample)) }
        labels.push(sample % 2)
    }
    let features = jt.Tensor([4, 4, 4, 1], pixels)
    let data = jt.Dataset(features, jt.Tensor([4], [float(label) for label in labels]))
    defer { data.free() }
    let model = jt.Sequential(
        [
            jt.Conv2d(2, 3, padding: 1, in_channels: 1),
            jt.Flatten(),
            jt.Dense(2, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(jt.SGD(learning_rate: 0.05))
    let history = model.fit(data, epochs: 1, batch_size: 2, verbose: false, plot_path: null)
    assert_true(history.loss.len() == 1)
}

fn test_pooling_reports_its_pooled_shape_to_the_next_layer() -> void {
    if not jt.is_available() { return }
    #: `Sequential` sizes each layer from the shape the one before it reports.
    #: A pool that reported its input shape sized the Dense head as if nothing
    #: had been pooled, and the model failed on its first forward with a matmul
    #: shape error instead of at build time.
    var pixels: list[float] = []
    var labels: list[float] = []
    for sample in 0..4 {
        for _pixel in 0..64 { pixels.push(float(sample) * 0.1) }
        labels.push(float(sample % 2))
    }
    let features = jt.Tensor([4, 8, 8, 1], pixels)
    let data = jt.Dataset(features, jt.Tensor([4], labels))
    defer { data.free() }
    let model = jt.Sequential(
        [
            jt.Conv2d(2, 3, padding: 1, in_channels: 1),
            jt.MaxPool2d(2),
            jt.Flatten(),
            jt.Dense(2, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(jt.SGD(learning_rate: 0.05))
    let history = model.fit(data, epochs: 1, batch_size: 2, verbose: false, plot_path: null)
    assert_true(history.loss.len() == 1)
}

fn test_average_pooling_reports_its_pooled_shape_too() -> void {
    if not jt.is_available() { return }
    var pixels: list[float] = []
    var labels: list[float] = []
    for sample in 0..4 {
        for _pixel in 0..64 { pixels.push(float(sample) * 0.1) }
        labels.push(float(sample % 2))
    }
    let features = jt.Tensor([4, 8, 8, 1], pixels)
    let data = jt.Dataset(features, jt.Tensor([4], labels))
    defer { data.free() }
    let model = jt.Sequential(
        [
            jt.Conv2d(2, 3, padding: 1, in_channels: 1),
            jt.AvgPool2d(2),
            jt.Flatten(),
            jt.Dense(2, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(jt.SGD(learning_rate: 0.05))
    let history = model.fit(data, epochs: 1, batch_size: 2, verbose: false, plot_path: null)
    assert_true(history.loss.len() == 1)
}

fn test_conv2d_on_a_batch_slice_reads_that_slice() -> void {
    if not jt.is_available() { return }
    #: An `MPSNDArray` pads its innermost dimension to a sixteen-byte row, so a
    #: windowed feed misreads a packed buffer whenever that dimension is not a
    #: multiple of four floats. Three-channel NHWC images are exactly that
    #: shape, and every batch after the first convolved the wrong pixels — a
    #: model that trained on batch zero forever and looked merely untrainable.
    #: Two channels here so the row stays unaligned; the copy is the reference.
    #: Every operand has to clear the 512-byte floor the native path declines
    #: below, or the fallback runs and the window is never exercised.
    let batch = 16
    let side = 8
    let channels = 2
    let stride = side * side * channels
    var values: list[float] = []
    for index in 0..3 * batch * stride {
        values.push(float(index * 7 % 13) * 0.05 - 0.3)
    }
    let all = jt.Tensor([3 * batch, side, side, channels], values)
    defer { all.free() }
    var weight_values: list[float] = []
    for index in 0..3 * 3 * channels * 8 {
        weight_values.push(float(index * 5 % 9) * 0.05 - 0.2)
    }
    let weights = jt.Tensor([3, 3, channels, 8], weight_values)
    defer { weights.free() }
    for slice in 1..3 {
        let window = all.view(slice * batch * stride, [batch, side, side, channels])
        let copied = jt.Tensor([batch, side, side, channels], window.download())
        let from_window = jt.conv2d(window, weights, padding: 1).download()
        let from_copy = jt.conv2d(copied, weights, padding: 1).download()
        for index in 0..from_copy.len() {
            assert_near(from_window[index], from_copy[index], 1e-4)
        }
        window.free()
        copied.free()
    }
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

fn test_device_count_is_non_negative() -> void {
    if not jt.is_available() { return }
    assert_true(jt.device_count() >= 1)
}

fn shaped_identity(x: Tensor[float, 2, 3]) -> Tensor[float, 2, 3] {
    return x
}

fn test_shaped_tensor_annotation_accepts_matching_runtime_value() -> void {
    if not jt.is_available() { return }
    let tensor = jt.Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    defer { tensor.free() }
    let got = shaped_identity(tensor)
    assert_eq(got.size, 6)
}

fn test_unary_abs_neg_square_and_gelu() -> void {
    if not jt.is_available() { return }
    let x = jt.Tensor([4], [-2.0, -0.0, 0.5, 2.0])
    defer { x.free() }
    let a = jt.abs(x)
    let n = jt.neg(x)
    let s = jt.square(x)
    let g = jt.gelu(jt.Tensor([1], [0.0]))
    defer { a.free() }
    defer { n.free() }
    defer { s.free() }
    defer { g.free() }
    assert_values_near(a.download(), [2.0, 0.0, 0.5, 2.0], 1e-5)
    assert_values_near(n.download(), [2.0, 0.0, -0.5, -2.0], 1e-5)
    assert_values_near(s.download(), [4.0, 0.0, 0.25, 4.0], 1e-5)
    assert_values_near(g.download(), [0.0], 1e-5)
}

fn test_sum_mean_clamp_and_cat() -> void {
    if not jt.is_available() { return }
    let x = jt.Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    defer { x.free() }
    let all = jt.sum(x)
    let rows = jt.sum(x, axis: -1)
    let avg = jt.mean(x)
    let clipped = jt.clamp(x, 2.0, 5.0)
    let stacked = jt.cat([x, x], axis: 0)
    defer { all.free() }
    defer { rows.free() }
    defer { avg.free() }
    defer { clipped.free() }
    defer { stacked.free() }
    assert_values_near(all.download(), [21.0], 1e-4)
    assert_values_near(rows.download(), [6.0, 15.0], 1e-4)
    assert_values_near(avg.download(), [3.5], 1e-4)
    assert_values_near(clipped.download(), [2.0, 2.0, 3.0, 4.0, 5.0, 5.0], 1e-5)
    assert_eq(stacked.shape, [4, 3])
}

fn test_max_pool2d_and_mse() -> void {
    if not jt.is_available() { return }
    let images = jt.Tensor([1, 2, 2, 1], [1.0, 2.0, 3.0, 4.0])
    defer { images.free() }
    let pooled = jt.max_pool2d(images, 2)
    defer { pooled.free() }
    assert_eq(pooled.shape, [1, 1, 1, 1])
    assert_values_near(pooled.download(), [4.0], 1e-5)
    let pred = jt.Tensor([2], [1.0, 3.0])
    let target = jt.Tensor([2], [1.0, 1.0])
    defer { pred.free() }
    defer { target.free() }
    let loss = jt.mse(pred, target)
    defer { loss.free() }
    assert_values_near(loss.download(), [2.0], 1e-4)
}

fn test_dropout_eval_is_identity_and_reshape_is_a_view() -> void {
    if not jt.is_available() { return }
    let x = jt.Tensor([3], [1.0, 2.0, 3.0])
    defer { x.free() }
    let kept = jt.dropout(x, 0.5, training: false)
    defer { kept.free() }
    assert_eq(kept.download(), [1.0, 2.0, 3.0])
    let viewed = jt.reshape(x, [1, 3])
    defer { viewed.free() }
    assert_eq(viewed.shape, [1, 3])
    assert_eq(viewed.download(), [1.0, 2.0, 3.0])
}

fn test_cat_last_axis_and_pad2d() -> void {
    if not jt.is_available() { return }
    let a = jt.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])
    let b = jt.Tensor([2, 1], [5.0, 6.0])
    defer { a.free() }
    defer { b.free() }
    let joined = jt.cat([a, b], axis: 1)
    defer { joined.free() }
    assert_eq(joined.shape, [2, 3])
    assert_values_near(joined.download(), [1.0, 2.0, 5.0, 3.0, 4.0, 6.0], 1e-5)
    let image = jt.Tensor([1, 1, 1, 1], [9.0])
    defer { image.free() }
    let padded = jt.pad2d(image, [1, 0, 0, 1])
    defer { padded.free() }
    assert_eq(padded.shape, [1, 2, 2, 1])
    assert_values_near(padded.download(), [0.0, 0.0, 9.0, 0.0], 1e-5)
}

fn test_where_squeeze_stack_and_pow() -> void {
    if not jt.is_available() { return }
    let cond = jt.Tensor([3], [1.0, 0.0, 1.0])
    let left = jt.Tensor([3], [2.0, 3.0, 4.0])
    let right = jt.Tensor([3], [8.0, 9.0, 10.0])
    defer { cond.free() }
    defer { left.free() }
    defer { right.free() }
    let picked = jt.where(cond, left, right)
    defer { picked.free() }
    assert_values_near(picked.download(), [2.0, 9.0, 4.0], 1e-5)
    let unit = jt.unsqueeze(left, 0)
    defer { unit.free() }
    assert_eq(unit.shape, [1, 3])
    let back = jt.squeeze(unit, axis: 0)
    defer { back.free() }
    assert_eq(back.shape, [3])
    let stacked = jt.stack([left, right])
    defer { stacked.free() }
    assert_eq(stacked.shape, [2, 3])
    let squared = jt.pow(left, 2.0)
    defer { squared.free() }
    assert_values_near(squared.download(), [4.0, 9.0, 16.0], 1e-4)
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

fn test_ones_full_skip_host_lists_and_layouts() -> void {
    if not jt.is_available() { return }
    let ones = jt.ones([3])
    let filled = jt.fill([2], 7.0)
    defer { ones.free() }
    defer { filled.free() }
    assert_eq(ones.download(), [1.0, 1.0, 1.0])
    assert_eq(filled.download(), [7.0, 7.0])
    let nhwc = jt.Tensor([1, 2, 2, 1], [1.0, 2.0, 3.0, 4.0])
    defer { nhwc.free() }
    let nchw = jt.nchw(nhwc)
    defer { nchw.free() }
    assert_eq(nchw.shape, [1, 1, 2, 2])
    assert_values_near(nchw.download(), [1.0, 2.0, 3.0, 4.0], 1e-5)
    let back = jt.nhwc(nchw)
    defer { back.free() }
    assert_values_near(back.download(), [1.0, 2.0, 3.0, 4.0], 1e-5)
    let src = jt.Tensor([2, 1], [1.0, 2.0])
    defer { src.free() }
    let tiled = jt.repeat(src, 3)
    defer { tiled.free() }
    assert_eq(tiled.shape, [2, 3])
    assert_values_near(tiled.download(), [1.0, 1.0, 1.0, 2.0, 2.0, 2.0], 1e-5)
    let pred = jt.Tensor([2], [0.9, 0.1])
    let target = jt.Tensor([2], [1.0, 0.0])
    defer { pred.free() }
    defer { target.free() }
    let loss = jt.binary_cross_entropy(pred, target)
    defer { loss.free() }
    assert_true(loss.download()[0] > 0.0)
    assert_true(loss.download()[0] < 0.5)
}

#: Pooling and repeat both take a widened kernel when the run they walk
#: divides by four -- the channels for pooling, the row length for repeat.
#: Each case here is checked against the same shape one wider, which takes
#: the narrow arm.
fn test_pooling_and_repeat_on_both_arms() -> void {
    if not jt.is_available() { return }
    #: One image, two rows of four, four channels: the widened arm.
    let wide = jt.Tensor(
        [1, 2, 4, 4],
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            5.0,
            6.0,
            6.0,
            7.0,
            8.0,
            9.0,
            1.0,
            2.0,
            2.0,
            3.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            8.0,
            9.0,
            9.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            5.0,
            6.0,
            6.0,
        ]
    )
    defer { wide.free() }
    let peak = jt.max_pool2d(wide, 2)
    defer { peak.free() }
    assert_eq(peak.shape, [1, 1, 2, 4])
    assert_eq(peak.download(), [8.0, 8.0, 9.0, 9.0, 7.0, 8.0, 9.0, 6.0])
    let mild = jt.avg_pool2d(wide, 2)
    defer { mild.free() }
    assert_eq(mild.download(), [4.5, 5.0, 6.0, 6.5, 3.75, 4.25, 5.25, 3.5])

    #: Two channels, which takes the narrow arm.
    let narrow = jt.Tensor(
        [1, 2, 4, 2],
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
        ]
    )
    defer { narrow.free() }
    let narrow_peak = jt.max_pool2d(narrow, 2)
    defer { narrow_peak.free() }
    assert_eq(narrow_peak.shape, [1, 1, 2, 2])
    assert_eq(narrow_peak.download(), [9.0, 4.0, 7.0, 8.0])
    let narrow_mild = jt.avg_pool2d(narrow, 2)
    defer { narrow_mild.free() }
    assert_eq(narrow_mild.download(), [3.75, 2.5, 5.5, 6.5])

    #: Repeat widens when the row length divides by four.
    let row = jt.Tensor([2, 4], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    defer { row.free() }
    let grown = jt.repeat(row, 2)
    defer { grown.free() }
    assert_eq(grown.shape, [2, 8])
    assert_eq(
        grown.download(),
        [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0]
    )
    let odd = jt.Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    defer { odd.free() }
    let odd_grown = jt.repeat(odd, 2)
    defer { odd_grown.free() }
    assert_eq(odd_grown.download(), [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0])
}

#: Both layout swaps go through a tiled transpose, and padding takes a
#: widened kernel when the channels divide by four. The sizes here straddle
#: the tile so the ragged edge is walked as well as the whole squares.
fn test_layout_swaps_and_padding_at_the_tile_edge() -> void {
    if not jt.is_available() { return }
    for shape in [[1, 1, 1, 1], [2, 3, 5, 4], [1, 33, 31, 3], [1, 32, 32, 32]] {
        let source = jt.Tensor(shape)
        defer { source.free() }
        source.buffer().fill_uniform(-1.0, 1.0, shape[1] * 100 + shape[2] * 10 + shape[3])
        let want = source.buffer().download()
        let n = shape[0]
        let h = shape[1]
        let w = shape[2]
        let c = shape[3]

        let planar = jt.nchw(source)
        defer { planar.free() }
        assert_eq(planar.shape, [n, c, h, w])
        let planar_values = planar.buffer().download()
        for index in 0..source.size {
            let rest = index // c
            let channel = index - rest * c
            let row = rest // w
            let column = rest - row * w
            let image = row // h
            let line = row - image * h
            assert_eq(planar_values[((image * c + channel) * h + line) * w + column], want[index])
        }

        let back = jt.nhwc(planar)
        defer { back.free() }
        assert_eq(back.shape, shape)
        assert_eq(back.buffer().download(), want)

        let padded = jt.pad2d(source, [1, 2, 3, 0])
        defer { padded.free() }
        assert_eq(padded.shape, [n, h + 3, w + 3, c])
        let padded_values = padded.buffer().download()
        for index in 0..padded.size {
            let rest = index // c
            let channel = index - rest * c
            let row = rest // (w + 3)
            let column = rest - row * (w + 3)
            let image = row // (h + 3)
            let line = row - image * (h + 3)
            let inside = line >= 1 and line < h + 1 and column >= 3
            let expect = if inside {
                want[((image * h + line - 1) * w + column - 3) * c + channel]
            } else {
                0.0
            }
            assert_eq(padded_values[index], expect)
        }
    }
}

fn test_eye_flip_upsample_and_split_heads() -> void {
    if not jt.is_available() { return }
    let ident = jt.eye(3)
    defer { ident.free() }
    assert_eq(ident.download(), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    let src = jt.Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    defer { src.free() }
    let flipped = jt.flip(src)
    defer { flipped.free() }
    assert_eq(flipped.download(), [3.0, 2.0, 1.0, 6.0, 5.0, 4.0])
    let image = jt.Tensor([1, 1, 2, 1], [1.0, 2.0])
    defer { image.free() }
    let up = jt.upsample_nearest2d(image, 2)
    defer { up.free() }
    assert_eq(up.shape, [1, 2, 4, 1])
    assert_eq(up.download(), [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0])
    #: A channel count that divides by four takes the widened kernel, which
    #: moves a float4 a thread and works out its position once for the group
    #: rather than once a channel. The row above keeps the narrow one covered.
    let wide_in = jt.Tensor([1, 1, 2, 4], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    defer { wide_in.free() }
    let wide_up = jt.upsample_nearest2d(wide_in, 2)
    defer { wide_up.free() }
    assert_eq(wide_up.shape, [1, 2, 4, 4])
    let one_row = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0]
    assert_eq(wide_up.download(), one_row + one_row)
    let packed_in = jt.Tensor([2, 4], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    defer { packed_in.free() }
    let split = jt.split_heads(packed_in, 2)
    defer { split.free() }
    assert_eq(split.shape, [4, 2])
    assert_eq(split.download(), [1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0])
    let merged = jt.merge_heads(split, 2, 2)
    defer { merged.free() }
    assert_eq(merged.download(), packed_in.download())
}

fn test_multi_head_attention_matches_per_head_sdpa() -> void {
    if not jt.is_available() { return }
    let q = jt.Tensor([2, 4], [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    let k = jt.Tensor([2, 4], [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    let v = jt.Tensor([2, 4], [2.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 5.0])
    defer { q.free() }
    defer { k.free() }
    defer { v.free() }
    let fused = jt.multi_head_attention(q, k, v, 2)
    defer { fused.free() }
    let seq = 2
    let heads = 2
    let hd = 2
    let qs = jt.split_heads(q, heads)
    let ks = jt.split_heads(k, heads)
    let vs = jt.split_heads(v, heads)
    defer { qs.free() }
    defer { ks.free() }
    defer { vs.free() }
    var parts: list[Tensor] = []
    for h in 0..heads {
        let qh = qs.view(h * seq * hd, [seq, hd])
        let kh = ks.view(h * seq * hd, [seq, hd])
        let vh = vs.view(h * seq * hd, [seq, hd])
        parts.push(jt.scaled_dot_product_attention(qh, kh, vh))
        qh.free()
        kh.free()
        vh.free()
    }
    let packed = jt.cat(parts)
    for part in parts { part.free() }
    defer { packed.free() }
    let looped = jt.merge_heads(packed, heads, seq)
    defer { looped.free() }
    assert_values_near(fused.download(), looped.download(), 1e-5)
    let rng = Random.with_seed(1)
    let layer = jt.MultiHeadAttention(4, heads: 2)
    defer { layer.free() }
    layer.build(4, rng)
    let y = layer.forward(q)
    defer { y.free() }
    assert_eq(y.shape, [2, 4])
}

fn test_group_norm_centers_each_group() -> void {
    if not jt.is_available() { return }
    let x = jt.Tensor([1, 1, 2, 2], [1.0, 3.0, 3.0, 5.0])
    defer { x.free() }
    let gamma = jt.ones([2])
    let beta = jt.zeros([2])
    defer { gamma.free() }
    defer { beta.free() }
    let y = jt.group_norm(x, gamma, beta, 2)
    defer { y.free() }
    let values = y.download()
    assert_near((values[0] + values[2]) / 2.0, 0.0, 1e-4)
    assert_near((values[1] + values[3]) / 2.0, 0.0, 1e-4)
}

fn test_tril_and_bce_with_logits() -> void {
    if not jt.is_available() { return }
    let square = jt.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])
    defer { square.free() }
    let lower = jt.tril(square)
    defer { lower.free() }
    assert_eq(lower.download(), [1.0, 0.0, 3.0, 4.0])
    let logits = jt.Tensor([2], [0.0, 10.0], requires_grad: true)
    let target = jt.Tensor([2], [0.0, 1.0])
    defer { logits.free() }
    defer { target.free() }
    let loss = jt.binary_cross_entropy_with_logits(logits, target)
    defer { loss.free() }
    assert_near(loss.download()[0], math.log(2.0) / 2.0, 1e-4)
    let extras = loss.backward()
    for tensor in extras { tensor.free() }
    let g = logits.grad
    if g is null { throw AssertionError("expected logits.grad after bce_with_logits") }
    assert_near(g.download()[0], 0.25, 1e-4)
    assert_near(g.download()[1], 0.0, 1e-4)
}

fn test_flash_attention_matches_per_head_sdpa() -> void {
    if not jt.is_available() { return }
    let seq = 16
    let heads = 2
    let hd = 2
    var values: list[float] = []
    for i in 0..seq * heads * hd {
        values.push(float(i * 3 % 11) / 5.0 - 1.0)
    }
    let q = jt.Tensor([seq, heads * hd], values)
    let k = jt.Tensor([seq, heads * hd], values)
    var vvals: list[float] = []
    for i in 0..seq * heads * hd {
        vvals.push(float(i * 5 % 7) / 3.0)
    }
    let v = jt.Tensor([seq, heads * hd], vvals)
    defer { q.free() }
    defer { k.free() }
    defer { v.free() }
    let fused = jt.multi_head_attention(q, k, v, heads)
    defer { fused.free() }
    let qs = jt.split_heads(q, heads)
    let ks = jt.split_heads(k, heads)
    let vs = jt.split_heads(v, heads)
    defer { qs.free() }
    defer { ks.free() }
    defer { vs.free() }
    var parts: list[Tensor] = []
    for h in 0..heads {
        let qh = qs.view(h * seq * hd, [seq, hd])
        let kh = ks.view(h * seq * hd, [seq, hd])
        let vh = vs.view(h * seq * hd, [seq, hd])
        parts.push(jt.scaled_dot_product_attention(qh, kh, vh))
        qh.free()
        kh.free()
        vh.free()
    }
    let packed = jt.cat(parts)
    for part in parts { part.free() }
    defer { packed.free() }
    let looped = jt.merge_heads(packed, heads, seq)
    defer { looped.free() }
    assert_values_near(fused.download(), looped.download(), 2e-4)
}

fn test_wide_flash_attention_matches_per_head_sdpa() -> void {
    if not jt.is_available() { return }
    let seq = 96
    let heads = 2
    let hd = 32
    var values: list[float] = []
    for i in 0..seq * heads * hd {
        values.push(float(i * 3 % 11) / 5.0 - 1.0)
    }
    let q = jt.Tensor([seq, heads * hd], values)
    let k = jt.Tensor([seq, heads * hd], values)
    var vvals: list[float] = []
    for i in 0..seq * heads * hd {
        vvals.push(float(i * 5 % 7) / 3.0)
    }
    let v = jt.Tensor([seq, heads * hd], vvals)
    defer { q.free() }
    defer { k.free() }
    defer { v.free() }
    let fused = jt.multi_head_attention(q, k, v, heads)
    defer { fused.free() }
    let qs = jt.split_heads(q, heads)
    let ks = jt.split_heads(k, heads)
    let vs = jt.split_heads(v, heads)
    defer { qs.free() }
    defer { ks.free() }
    defer { vs.free() }
    var parts: list[Tensor] = []
    for h in 0..heads {
        let qh = qs.view(h * seq * hd, [seq, hd])
        let kh = ks.view(h * seq * hd, [seq, hd])
        let vh = vs.view(h * seq * hd, [seq, hd])
        parts.push(jt.scaled_dot_product_attention(qh, kh, vh))
        qh.free()
        kh.free()
        vh.free()
    }
    let packed = jt.cat(parts)
    for part in parts { part.free() }
    defer { packed.free() }
    let looped = jt.merge_heads(packed, heads, seq)
    defer { looped.free() }
    assert_values_near(fused.download(), looped.download(), 2e-3)
}

#: `amax` is the companion to `argmax`: where `argmax` says which element of a
#: row is largest, `amax` says what it is. A detector wants both, and working
#: the value out on the host afterwards means downloading everything the
#: reduction was there to avoid.
fn test_amax_reduces_the_last_axis_and_the_whole_tensor() -> void {
    if not jt.is_available() { return }
    let values = [3.0, -1.0, 2.5, 0.0, -4.0, -0.5, 7.25, 1.0, 6.0, 6.0, -2.0, 5.5]
    let x = Tensor([3, 4], values)
    let along = jt.amax(x, axis: -1).download()
    assert_eq(along.len(), 3)
    assert_near(along[0], 3.0, 1e-6)
    assert_near(along[1], 7.25, 1e-6)
    assert_near(along[2], 6.0, 1e-6)

    #: A tie keeps the first, the same rule `argmax` follows.
    let where = jt.argmax(x).download()
    assert_eq(int(where[2]), 0)

    let whole = jt.amax(x).download()
    assert_eq(whole.len(), 1)
    assert_near(whole[0], 7.25, 1e-6)

    #: A single column is still a reduction, of one element each.
    let column = Tensor([4, 1], [1.0, 2.0, 3.0, 4.0])
    let each = jt.amax(column, axis: -1).download()
    assert_eq(each, [1.0, 2.0, 3.0, 4.0])

    let _refused = assert_throws(ValueError, || jt.amax(x, axis: 0))
    x.free()
    column.free()
}

#: `max_with_index` agrees with asking for the two separately.
fn test_max_with_index_matches_amax_and_argmax() -> void {
    if not jt.is_available() { return }
    var values: list[float] = []
    for index in 0..64 * 40 {
        #: Every row has its peak somewhere different, and negatives so that a
        #: zero-initialised answer would show.
        values.push(float(index * 37 % 79 - 40) * 0.25)
    }
    let input = Tensor([64, 40], values)
    defer { input.free() }
    let (peaks, wheres) = jt.max_with_index(input)
    defer { peaks.free() }
    defer { wheres.free() }
    let apart_values = jt.amax(input, axis: -1)
    defer { apart_values.free() }
    let apart_indices = jt.argmax(input)
    defer { apart_indices.free() }
    assert_eq(peaks.download(), apart_values.download())
    assert_eq(wheres.download(), apart_indices.download())
}

fn test_max_with_index_refuses_anything_but_rank_two() -> void {
    if not jt.is_available() { return }
    let flat = Tensor([8])
    defer { flat.free() }
    assert_throws(ValueError, || jt.max_with_index(flat))
}

#: Reducing down the columns gives what transposing and reducing along the rows
#: gives, which is the pass it exists to avoid.
fn test_max_with_index_down_the_columns_matches_a_transpose() -> void {
    if not jt.is_available() { return }
    var values: list[float] = []
    for index in 0..24 * 57 { values.push(float(index * 41 % 91 - 45) * 0.125) }
    let input = Tensor([24, 57], values)
    defer { input.free() }
    let (peaks, wheres) = jt.max_with_index(input, axis: 0)
    defer { peaks.free() }
    defer { wheres.free() }
    let flipped = jt.transpose(input)
    defer { flipped.free() }
    let (want_peaks, want_wheres) = jt.max_with_index(flipped, axis: -1)
    defer { want_peaks.free() }
    defer { want_wheres.free() }
    assert_eq(peaks.shape, [57])
    assert_eq(peaks.download(), want_peaks.download())
    assert_eq(wheres.download(), want_wheres.download())
}

fn test_max_with_index_refuses_an_axis_it_has_no_kernel_for() -> void {
    if not jt.is_available() { return }
    let input = Tensor([4, 5])
    defer { input.free() }
    assert_throws(ValueError, || jt.max_with_index(input, axis: 2))
}

#: A row wider than the threadgroup, checked against arithmetic done here.
#:
#: The kernel spreads a row across 256 threads and brings the two sums back
#: through a simd reduction, so a row narrower than that never exercises the
#: striding or the second stage of the reduction. Widths either side of 256 and
#: of a warp are the ones that catch a reduction wired up wrongly.
fn test_layer_norm_matches_arithmetic_over_wide_rows() -> void {
    if not jt.is_available() { return }
    for width in [17, 32, 33, 256, 257, 768] {
        let rows = 3
        var values: list[float] = []
        for index in 0..rows * width {
            values.push(float(index * 31 % 97 - 48) * 0.125)
        }
        var scales: list[float] = []
        var shifts: list[float] = []
        for index in 0..width {
            scales.push(1.0 + float(index % 5) * 0.25)
            shifts.push(float(index % 7) * 0.5 - 1.5)
        }
        let x = jt.Tensor([rows, width], values)
        let gamma = jt.Tensor([width], scales)
        let beta = jt.Tensor([width], shifts)
        let y = jt.layer_norm(x, gamma, beta)
        let got = y.download()
        for row in 0..rows {
            var total = 0.0
            for column in 0..width { total += values[row * width + column] }
            let mean = total / float(width)
            var square = 0.0
            for column in 0..width {
                let gap = values[row * width + column] - mean
                square += gap * gap
            }
            let inverse = 1.0 / math.sqrt(square / float(width) + 1e-5)
            for column in 0..width {
                let centred = values[row * width + column] - mean
                let want = scales[column] * centred * inverse + shifts[column]
                assert_near(
                    got[row * width + column],
                    want,
                    1e-3,
                    f"width {width} at {row},{column}"
                )
            }
        }
        x.free()
        gamma.free()
        beta.free()
        y.free()
    }
}

#: The fused product-then-GELU gives what the two steps give.
fn test_linear_gelu_matches_the_two_steps() -> void {
    if not jt.is_available() { return }
    let rows = 40
    let inner = 64
    let columns = 96
    var xs: list[float] = []
    for index in 0..rows * inner { xs.push(float(index * 17 % 61 - 30) * 0.05) }
    var ws: list[float] = []
    for index in 0..inner * columns { ws.push(float(index * 23 % 47 - 23) * 0.03) }
    var bs: list[float] = []
    for index in 0..columns { bs.push(float(index % 9) * 0.1 - 0.4) }
    let x = Tensor([rows, inner], xs)
    let w = Tensor([inner, columns], ws)
    let b = Tensor([columns], bs)
    defer { x.free() }
    defer { w.free() }
    defer { b.free() }

    let fused = jt.linear_gelu(x, w, b)
    defer { fused.free() }
    let apart = jt.gelu(jt.add(jt.matmul(x, w), b))
    defer { apart.free() }
    let got = fused.download()
    let want = apart.download()
    assert_eq(got.len(), want.len())
    for index in 0..got.len() {
        assert_near(got[index], want[index], 1e-4, f"at {index}")
    }
}

#: Training takes the unfused path, because GELU's derivative needs what went
#: into it and the fused kernel keeps only what came out.
fn test_linear_gelu_still_differentiates() -> void {
    if not jt.is_available() { return }
    let x = Tensor([4, 6])
    let w = Tensor([6, 5])
    let b = Tensor.zeros([5])
    x.buffer().fill_uniform(-1.0, 1.0, 3)
    w.buffer().fill_uniform(-1.0, 1.0, 4)
    w.requires_grad = true
    let out = jt.linear_gelu(x, w, b)
    assert_eq(out.shape, [4, 5])
    jt.sum(out).backward()
    let grad = w.grad
    assert_true(grad is not null, "the weight got no gradient")
}

#: The fused residual-and-product gives what the two steps give.
fn test_linear_residual_matches_the_two_steps() -> void {
    if not jt.is_available() { return }
    let rows = 37
    let inner = 53
    let columns = 71
    var xs: list[float] = []
    for index in 0..rows * inner { xs.push(float(index * 19 % 71 - 35) * 0.04) }
    var ws: list[float] = []
    for index in 0..inner * columns { ws.push(float(index * 29 % 53 - 26) * 0.02) }
    var rs: list[float] = []
    for index in 0..rows * columns { rs.push(float(index * 13 % 37 - 18) * 0.07) }
    let x = Tensor([rows, inner], xs)
    let w = Tensor([inner, columns], ws)
    let residual = Tensor([rows, columns], rs)
    defer { x.free() }
    defer { w.free() }
    defer { residual.free() }

    let fused = jt.linear_residual(x, w, residual)
    defer { fused.free() }
    let apart = jt.add(residual, jt.matmul(x, w))
    defer { apart.free() }
    let got = fused.download()
    let want = apart.download()
    assert_eq(got.len(), want.len())
    for index in 0..got.len() { assert_near(got[index], want[index], 1e-4, f"at {index}") }
}

fn test_linear_residual_refuses_a_residual_of_the_wrong_shape() -> void {
    if not jt.is_available() { return }
    let x = Tensor([4, 6])
    let w = Tensor([6, 5])
    let wrong = Tensor([4, 6])
    defer { x.free() }
    defer { w.free() }
    defer { wrong.free() }
    assert_throws(ValueError, || jt.linear_residual(x, w, wrong))
}

#: A row wider than the threadgroup that reduces it.
#:
#: These reductions used to run one thread to a row, so a row of any width was
#: one thread's serial walk and the width never mattered. Now the group strides
#: the row, and a row of 1000 takes four passes with a partial per thread to
#: combine -- which is the case a row of four never reaches.
fn test_wide_rows_reduce_the_same_as_narrow_ones() -> void {
    if not jt.is_available() { return }
    let rows = 3
    let columns = 1000
    var values: list[float] = []
    for row in 0..rows {
        for column in 0..columns {
            values.push(float((row * 7 + column * 13) % 101) * 0.01)
        }
    }
    #: One unmistakable peak per row, past where a single group would stop.
    values[0 * columns + 700] = 9.5
    values[1 * columns + 999] = 9.5
    values[2 * columns + 0] = 9.5
    let x = jt.Tensor([rows, columns], values)
    defer { x.free() }

    let totals = jt.sum(x, axis: -1).download()
    let averages = jt.mean(x, axis: -1).download()
    let peaks = jt.amax(x, axis: -1).download()
    let where = jt.argmax(x).download()
    let (also_peaks, also_where) = jt.max_with_index(x)
    defer { also_peaks.free() }
    defer { also_where.free() }
    let peak_values = also_peaks.download()
    let peak_places = also_where.download()

    let wanted = [700.0, 999.0, 0.0]
    for row in 0..rows {
        var total = 0.0
        for column in 0..columns { total += values[row * columns + column] }
        assert_near(totals[row], total, total * 1e-5 + 1e-4, f"row {row} total")
        assert_near(averages[row], total / float(columns), 1e-5, f"row {row} mean")
        assert_near(peaks[row], 9.5, 1e-6, f"row {row} peak")
        assert_near(where[row], wanted[row], 1e-6, f"row {row} peak position")
        assert_near(peak_values[row], 9.5, 1e-6, f"row {row} peak, together")
        assert_near(peak_places[row], wanted[row], 1e-6, f"row {row} position, together")
    }
}

#: Every element equal: the first one wins, whichever thread happened to see it.
fn test_a_wide_row_of_ties_keeps_the_first() -> void {
    if not jt.is_available() { return }
    let columns = 700
    var values: list[float] = []
    for _column in 0..columns { values.push(2.5) }
    let x = jt.Tensor([1, columns], values)
    defer { x.free() }
    assert_near(jt.argmax(x).download()[0], 0.0, 1e-6, "a row of ties should keep the first")
    let (peak, place) = jt.max_with_index(x)
    defer { peak.free() }
    defer { place.free() }
    assert_near(peak.download()[0], 2.5, 1e-6)
    assert_near(place.download()[0], 0.0, 1e-6, "a row of ties should keep the first")
}

#: The four exports no test called directly.
#:
#: Three of them are reached through something else -- the activations through
#: `Dense`, the NCHW convolution through jaicv's imported graphs -- so the gap
#: was in this suite rather than in the code. `rsqrt` had no caller anywhere,
#: which is the one worth checking: an unused export is an unverified one.
fn test_the_exports_no_test_reached() -> void {
    if not jt.is_available() { return }
    let x = jt.Tensor([2, 3], [4.0, 1.0, 0.25, 16.0, 100.0, 0.0625])
    defer { x.free() }
    let roots = jt.rsqrt(x).download()
    let wanted = [0.5, 1.0, 2.0, 0.25, 0.1, 4.0]
    for index in 0..wanted.len() {
        assert_near(roots[index], wanted[index], 1e-5, f"rsqrt at {index}")
    }

    let mixed = jt.Tensor([2, 3], [-2.0, -0.5, 0.0, 0.5, 2.0, 3.0])
    defer { mixed.free() }
    let curved = jt.silu(mixed).download()
    for (index, value) in [-2.0, -0.5, 0.0, 0.5, 2.0, 3.0].enumerate() {
        #: x * sigmoid(x), which is what the layer applies.
        let want = value / (1.0 + math.exp(-value))
        assert_near(curved[index], want, 1e-5, f"silu at {index}")
    }
    #: A hundredth below zero, which is the slope the kernel is written with.
    let leaky = jt.leaky_relu(mixed).download()
    for (index, value) in [-2.0, -0.5, 0.0, 0.5, 2.0, 3.0].enumerate() {
        let want = if value > 0.0 { value } else { value * 0.01 }
        assert_near(leaky[index], want, 1e-5, f"leaky_relu at {index}")
    }

    #: One channel, one 2x2 kernel of ones over a 3x3 ramp: each output is the
    #: sum of the square it covers.
    let image = jt.Tensor([1, 1, 3, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    defer { image.free() }
    let weights = jt.Tensor([1, 1, 2, 2], [1.0, 1.0, 1.0, 1.0])
    defer { weights.free() }
    let convolved = jt.conv2d_nchw(image, weights)
    defer { convolved.free() }
    assert_eq(convolved.shape, [1, 1, 2, 2])
    let sums = convolved.download()
    for (index, want) in [12.0, 16.0, 24.0, 28.0].enumerate() {
        assert_near(sums[index], want, 1e-5, f"conv2d_nchw at {index}")
    }
}

#: Transpose at sizes that do not land on the tile it stages through.
#:
#: The kernel reads and writes 32 by 32 squares through threadgroup memory so
#: that neither pass strides. Every edge of that -- a matrix narrower than one
#: tile, one taller, one that ends part way through -- is a bounds test the
#: naive version never had.
fn test_transpose_handles_every_edge_of_its_tile() -> void {
    if not jt.is_available() { return }
    for shape in [[1, 1], [1, 37], [37, 1], [31, 33], [32, 32], [33, 31], [64, 65], [129, 257]] {
        let rows = shape[0]
        let cols = shape[1]
        var values: list[float] = []
        for index in 0..rows * cols { values.push(float(index) * 0.5 - 3.0) }
        let x = jt.Tensor([rows, cols], values)
        let flipped = jt.transpose(x)
        assert_eq(flipped.shape, [cols, rows], f"{rows}x{cols} came back the wrong shape")
        let got = flipped.download()
        for r in 0..rows {
            for c in 0..cols {
                assert_near(
                    got[c * rows + r],
                    values[r * cols + c],
                    1e-6,
                    f"{rows}x{cols} at {r},{c}"
                )
            }
        }
        x.free()
        flipped.free()
    }
}

#: Average pooling's backward computed the other way round.
#:
#: The kernel gathers: an input pixel works out which windows reach it and
#: what each of those windows divided by. This walks the windows instead and
#: hands each of its in-bounds taps a share, so the two agree only if the
#: window range and the tap count agree -- the two things the kernel now
#: arrives at arithmetically rather than by walking every offset.
fn avg_pool2d_backward_scattered(
    upstream: list[float],
    shape: list[int],
    window: int,
    stride: int,
    padding: int
) -> list[float] {
    let images = shape[0]
    let height = shape[1]
    let width = shape[2]
    let channels = shape[3]
    let rows = (height + 2 * padding - window) // stride + 1
    let cols = (width + 2 * padding - window) // stride + 1
    var grad: list[float] = []
    for _slot in 0..images * height * width * channels { grad.push(0.0) }
    for image in 0..images {
        for row in 0..rows {
            for col in 0..cols {
                var seen = 0
                for down in 0..window {
                    for across in 0..window {
                        let source_row = row * stride - padding + down
                        let source_col = col * stride - padding + across
                        if source_row < 0 or source_row >= height { continue }
                        if source_col < 0 or source_col >= width { continue }
                        seen += 1
                    }
                }
                if seen == 0 { continue }
                for down in 0..window {
                    for across in 0..window {
                        let source_row = row * stride - padding + down
                        let source_col = col * stride - padding + across
                        if source_row < 0 or source_row >= height { continue }
                        if source_col < 0 or source_col >= width { continue }
                        let window_base = ((image * rows + row) * cols + col) * channels
                        let pixel_row = image * height + source_row
                        let pixel_base = (pixel_row * width + source_col) * channels
                        for channel in 0..channels {
                            let share = upstream[window_base + channel] / float(seen)
                            grad[pixel_base + channel] += share
                        }
                    }
                }
            }
        }
    }
    return grad
}

#: The gathered backward against the scattered one, over the shapes that
#: separate the arms of it: channels that divide by four and channels that do
#: not, a ragged edge whose last row no window reaches, overlapping windows,
#: and padding, which is the only thing that makes the divisor vary within one
#: call. Where every divisor is a power of two the comparison is exact,
#: because there the arithmetic is.
#:
#: The last two strides are wider than the window, which no other case here
#: is. That leaves columns between the windows that nothing covers, and the
#: gather walks down from the last window that could reach an input and stops
#: at the first that does not -- so for those columns it must stop before it
#: has added anything, and the answer is a zero the loop never writes into.
fn test_avg_pool2d_backward_matches_a_scattered_reference() -> void {
    let cases = [
        [1, 4, 4, 4, 2, 2, 0],
        [2, 5, 7, 8, 2, 2, 0],
        [1, 5, 5, 3, 3, 1, 1],
        [2, 6, 6, 4, 3, 2, 1],
        [1, 7, 5, 1, 3, 1, 0],
        [1, 4, 4, 8, 2, 1, 1],
        [1, 6, 6, 2, 2, 2, 0],
        [1, 7, 7, 4, 2, 3, 0],
        [1, 8, 8, 3, 2, 3, 0],
    ]
    for probe in cases {
        let shape = [probe[0], probe[1], probe[2], probe[3]]
        let window = probe[4]
        let stride = probe[5]
        let padding = probe[6]
        let count = shape[0] * shape[1] * shape[2] * shape[3]
        var values: list[float] = []
        for index in 0..count { values.push(float(index % 13) * 0.25 - 1.5) }
        let input = jt.Tensor(shape, values, requires_grad: true)
        let pooled = jt.avg_pool2d(input, window, stride: stride, padding: padding)
        var seeds: list[float] = []
        for index in 0..pooled.size { seeds.push(float(index % 7) * 0.5 - 1.0) }
        let upstream = jt.Tensor(pooled.shape, seeds)
        let node = pooled.creator
        if node is null { throw RuntimeError(f"avg_pool2d recorded no backward for {shape}") }
        let grads = node.apply(upstream)
        assert_eq(grads.len(), 1, f"{shape} produced the wrong number of gradients")
        let expected = avg_pool2d_backward_scattered(seeds, shape, window, stride, padding)
        let got = grads[0].download()
        let exact = padding == 0 and window == 2
        if exact and stride == 2 {
            assert_eq(got, expected, f"{shape} window {window} stride {stride} is not exact")
        } else {
            assert_values_near(got, expected, 1e-6)
        }
        for produced in grads { produced.free() }
        upstream.free()
        pooled.free()
        input.free()
    }
}

#: Max pooling's backward, worked out the other way round.
#:
#: The kernels walk the input and ask which windows a pixel won; this walks
#: the windows and hands each one's upstream value to the pixel that won it,
#: which is the definition the kernels are an arrangement of. Ties go to the
#: first tap, as they do in the forward pass.
fn max_pool2d_backward_scattered(
    values: list[float],
    upstream: list[float],
    shape: list[int],
    window: int,
    stride: int,
    padding: int
) -> list[float] {
    let images = shape[0]
    let height = shape[1]
    let width = shape[2]
    let channels = shape[3]
    let rows = (height + 2 * padding - window) // stride + 1
    let cols = (width + 2 * padding - window) // stride + 1
    var grad: list[float] = []
    for _slot in 0..images * height * width * channels { grad.push(0.0) }
    for image in 0..images {
        for row in 0..rows {
            for col in 0..cols {
                for channel in 0..channels {
                    var peak = 0.0
                    var best = -1
                    for down in 0..window {
                        for across in 0..window {
                            let source_row = row * stride - padding + down
                            let source_col = col * stride - padding + across
                            if source_row < 0 or source_row >= height { continue }
                            if source_col < 0 or source_col >= width { continue }
                            let pixel_row = image * height + source_row
                            let pixel = (pixel_row * width + source_col) * channels + channel
                            let beaten = best >= 0 and values[pixel] <= peak
                            if beaten { continue }
                            peak = values[pixel]
                            best = pixel
                        }
                    }
                    if best < 0 { continue }
                    grad[best] += upstream[((image * rows + row) * cols + col) * channels + channel]
                }
            }
        }
    }
    return grad
}

#: The gathered backward against the scattered one, over the shapes that
#: separate the arms of it: windows that tile the input, which are written a
#: whole window at a time, and windows that overlap or leave a row uncovered,
#: which go through the tap buffer instead. Channels that divide by four take
#: a different kernel from channels that do not, so both appear.
#:
#: Every value is a multiple of a quarter and every window holds at most nine
#: of them, so the sums are exact whichever order they are added in, and the
#: comparison can be for equality rather than nearness.
fn test_max_pool2d_backward_matches_a_scattered_reference() -> void {
    if not jt.is_available() { return }
    let cases = [
        [1, 4, 4, 4, 2, 2, 0],
        [2, 6, 6, 3, 2, 2, 0],
        [1, 5, 5, 4, 2, 2, 0],
        [2, 7, 7, 8, 3, 2, 0],
        [1, 5, 5, 3, 3, 1, 0],
        [1, 6, 6, 4, 3, 2, 1],
        [1, 8, 8, 16, 2, 2, 0],
    ]
    for probe in cases {
        let shape = [probe[0], probe[1], probe[2], probe[3]]
        let window = probe[4]
        let stride = probe[5]
        let padding = probe[6]
        let count = shape[0] * shape[1] * shape[2] * shape[3]
        var values: list[float] = []
        for index in 0..count { values.push(float(index * 7 % 23) * 0.25 - 2.75) }
        let input = jt.Tensor(shape, values, requires_grad: true)
        let pooled = jt.max_pool2d(input, window, stride: stride, padding: padding)
        var seeds: list[float] = []
        for index in 0..pooled.size { seeds.push(float(index % 7) * 0.5 - 1.0) }
        let upstream = jt.Tensor(pooled.shape, seeds)
        let node = pooled.creator
        if node is null { throw RuntimeError(f"max_pool2d recorded no backward for {shape}") }
        let grads = node.apply(upstream)
        assert_eq(grads.len(), 1, f"{shape} produced the wrong number of gradients")
        let expected = max_pool2d_backward_scattered(values, seeds, shape, window, stride, padding)
        let what = f"{shape} window {window} stride {stride} pad {padding}"
        assert_eq(grads[0].download(), expected, what)
        for produced in grads { produced.free() }
        upstream.free()
        pooled.free()
        input.free()
    }
}

#: A window whose values are all the same has no winner on the values, so the
#: one it picks is decided by the order the scan runs in. Both sides give it
#: to the first tap; the whole window would be wrong, and so would none of it.
fn test_max_pool2d_backward_gives_a_tied_window_to_its_first_tap() -> void {
    if not jt.is_available() { return }
    let shape = [1, 4, 4, 2]
    var values: list[float] = []
    for _slot in 0..32 { values.push(1.5) }
    let input = jt.Tensor(shape, values, requires_grad: true)
    defer { input.free() }
    let pooled = jt.max_pool2d(input, 2)
    defer { pooled.free() }
    var seeds: list[float] = []
    for index in 0..pooled.size { seeds.push(float(index) + 1.0) }
    let upstream = jt.Tensor(pooled.shape, seeds)
    defer { upstream.free() }
    let node = pooled.creator
    if node is null { throw RuntimeError("max_pool2d recorded no backward") }
    let grads = node.apply(upstream)
    let expected = max_pool2d_backward_scattered(values, seeds, shape, 2, 2, 0)
    assert_eq(grads[0].download(), expected, "a tied window should go to its first tap")
    var given = 0.0
    for share in expected { given += share }
    var sent = 0.0
    for seed in seeds { sent += seed }
    assert_near(given, sent, 1e-9, "every window should hand on exactly what it was given")
    for produced in grads { produced.free() }
}

#: `multi_head_attention` against the same arithmetic written one head at a
#: time, at sizes the batched GEMM actually tiles.
#:
#: The tests above run at sequence two and head dimension two, where every
#: product is a single tile and split-K never engages, so they say nothing
#: about the batched dispatch that carries the per-head products. These sizes
#: do: 128 divides the tile shape exactly and takes the direct kernel, 100 does
#: not and takes the staged one, and both run enough heads that a mistake in
#: the per-matrix stride shows as one head holding another head's answer rather
#: than as rounding.
#:
#: The temporaries come back rather than being freed here because the caller
#: that asks for gradients needs them alive until backward has read them.
fn attention_by_head(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    heads: int,
    seq: int,
    hd: int
) -> (Tensor, list[Tensor]) {
    let factor = 1.0 / math.sqrt(float(hd))
    let qs = jt.split_heads(query, heads)
    let ks = jt.split_heads(key, heads)
    let vs = jt.split_heads(value, heads)
    var held: list[Tensor] = [qs, ks, vs]
    var parts: list[Tensor] = []
    for h in 0..heads {
        let qh = jt.slice_flat(qs, h * seq * hd, [seq, hd])
        let kh = jt.slice_flat(ks, h * seq * hd, [seq, hd])
        let vh = jt.slice_flat(vs, h * seq * hd, [seq, hd])
        let scores = jt.matmul(qh, kh, trans_b: true)
        let scaled = jt.scale(scores, factor)
        let probs = jt.softmax(scaled)
        parts.push(jt.matmul(probs, vh))
        for tensor in [qh, kh, vh, scores, scaled, probs] { held.push(tensor) }
    }
    let packed = jt.cat(parts)
    for part in parts { held.push(part) }
    held.push(packed)
    return (jt.merge_heads(packed, heads, seq), held)
}

#: Q, K and V carry gradients so that the differentiable composition runs.
#:
#: Without `requires_grad` this routes to the fused inference kernel, which
#: never materialises the scores and shares no code with the path under test:
#: the check then passes with the batched dispatch's per-matrix stride set to
#: zero. The gradient test below cannot stand in for this one either, because
#: the gradients of `sum(attention(q, k, v))` are recomputed from the
#: probabilities and never read the forward context product at all.
fn check_attention_against_per_head(seq: int, dim: int, heads: int) -> void {
    let hd = dim // heads
    let q = Tensor([seq, dim], requires_grad: true)
    let k = Tensor([seq, dim], requires_grad: true)
    let v = Tensor([seq, dim], requires_grad: true)
    defer { q.free() }
    defer { k.free() }
    defer { v.free() }
    q.buffer().fill_uniform(-1.0, 1.0, 11)
    k.buffer().fill_uniform(-1.0, 1.0, 22)
    v.buffer().fill_uniform(-1.0, 1.0, 33)
    let fused = jt.multi_head_attention(q, k, v, heads)
    defer { fused.free() }
    defer { for tensor in fused.discard_graph() { tensor.free() } }
    let (apart, held) = attention_by_head(q, k, v, heads, seq, hd)
    defer { apart.free() }
    defer { for tensor in held { if not tensor.is_freed() { tensor.free() } } }
    assert_eq(fused.shape, [seq, dim])
    assert_values_near(fused.download(), apart.download(), 1e-3)
}

fn test_multi_head_attention_tiled_shapes_match_per_head() -> void {
    if not jt.is_available() { return }
    check_attention_against_per_head(128, 192, 6)
    check_attention_against_per_head(100, 192, 6)
    check_attention_against_per_head(128, 384, 3)
}

#: The same comparison for the gradients, which is where the batched GEMM
#: carries both transposes and the split reduction.
fn test_multi_head_attention_tiled_gradients_match_per_head() -> void {
    if not jt.is_available() { return }
    let seq = 128
    let dim = 192
    let heads = 6
    let hd = dim // heads
    var collected: list[list[list[float]]] = []
    for fused_path in [true, false] {
        let q = Tensor([seq, dim], requires_grad: true)
        let k = Tensor([seq, dim], requires_grad: true)
        let v = Tensor([seq, dim], requires_grad: true)
        q.buffer().fill_uniform(-1.0, 1.0, 11)
        k.buffer().fill_uniform(-1.0, 1.0, 22)
        v.buffer().fill_uniform(-1.0, 1.0, 33)
        var held: list[Tensor] = []
        let y = if fused_path {
            jt.multi_head_attention(q, k, v, heads)
        } else {
            let (built, temporaries) = attention_by_head(q, k, v, heads, seq, hd)
            for tensor in temporaries {
                held.push(tensor)
            }
            built
        }
        let loss = jt.sum(y)
        for tensor in loss.backward() { held.push(tensor) }
        var gradients: list[list[float]] = []
        for parameter in [q, k, v] {
            let gradient = parameter.grad
            assert_true(gradient is not null)
            if gradient is not null { gradients.push(gradient.download()) }
        }
        collected.push(gradients)
        loss.free()
        y.free()
        for tensor in held {
            if not tensor.is_freed() { tensor.free() }
        }
        q.free()
        k.free()
        v.free()
    }
    assert_eq(collected.len(), 2)
    assert_eq(collected[0].len(), 3)
    for (index, values) in collected[0].enumerate() {
        assert_values_near(values, collected[1][index], 1e-3)
    }
}

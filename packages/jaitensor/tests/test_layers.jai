#: The layer objects: what shape each one builds itself for and hands on.
import jaitensor as jt
import std.math
from std.test import assert_eq, assert_near, assert_true

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

#: The layer objects: what shape each one builds itself for and hands on.
import jaitensor as jt
import std.gpu as gpu
import std.math
from std.random import Random
from std.test import assert_eq, assert_near, assert_throws, assert_true

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

#: `lookup` keeps the indices' shape, `[batch, seq]` to `[batch, seq, dim]`,
#: which is what a transformer needs; `forward` keeps the `[batch, seq * dim]`
#: block a `Sequential` sizes its next layer for. Both read the same rows and
#: send the gradient to the same table.
fn test_an_embedding_looks_up_rows_in_the_shape_of_its_indices() -> void {
    if not jt.is_available() { return }
    let vocab = 7
    let width = 3
    let embedding = jt.Embedding(vocab, width)
    defer { embedding.free() }
    embedding.build(2, Random.with_seed(3))
    let table = embedding.parameters()[0].download()
    let ids = [4.0, 0.0, 6.0, 4.0, 2.0, 1.0]
    let indices = jt.Tensor([2, 3], ids)
    defer { indices.free() }
    let rows = embedding.lookup(indices)
    assert_eq(rows.shape, [2, 3, width])
    let flat = embedding.forward(indices)
    assert_eq(flat.shape, [2, 3 * width])
    let got = rows.download()
    assert_eq(got, flat.download())
    for (slot, id) in ids.enumerate() {
        for c in 0..width { assert_eq(got[slot * width + c], table[int(id) * width + c]) }
    }
    let total = jt.sum(rows)
    for held in total.backward() { held.free() }
    total.free()
    let grad = embedding.parameters()[0].grad
    assert_true(grad is not null)
    if grad is not null {
        let g = grad.download()
        #: Row 4 is looked up twice, rows 3 and 5 never.
        assert_near(g[4 * width], 2.0, 1e-6)
        assert_near(g[3 * width], 0.0, 1e-6)
        assert_near(g[0], 1.0, 1e-6)
    }
    for t in [rows, flat] { t.free() }
    let single = jt.Tensor([3], [1.0, 2.0, 3.0])
    let one = embedding.lookup(single)
    assert_eq(one.shape, [3, width])
    for t in [single, one] { t.free() }
}

#: Positions `start..start + seq` of the table are added to every sequence of
#: a batch, and the table's gradient is the sum over the batch.
fn test_positional_embedding_adds_each_position_to_every_sequence() -> void {
    if not jt.is_available() { return }
    let max_seq = 10
    let width = 4
    let positions = jt.PositionalEmbedding(max_seq, width)
    defer { positions.free() }
    positions.build(width, Random.with_seed(5))
    let table = positions.parameters()[0].download()
    let batch = 3
    let seq = 6
    let xs = [float(i) * 0.25 for i in 0..batch * seq * width]
    let x = jt.Tensor([batch, seq, width], xs)
    defer { x.free() }
    for start in [0, 2, 4] {
        let y = positions.forward_at(x, start)
        assert_eq(y.shape, [batch, seq, width])
        let got = y.download()
        for b in 0..batch {
            for t in 0..seq {
                for c in 0..width {
                    let at = (b * seq + t) * width + c
                    assert_near(got[at], xs[at] + table[(start + t) * width + c], 1e-6)
                }
            }
        }
        let total = jt.sum(y)
        for held in total.backward() { held.free() }
        total.free()
        y.free()
        let grad = positions.parameters()[0].grad
        assert_true(grad is not null)
        if grad is not null {
            let g = grad.download()
            for row in 0..max_seq {
                let want = if row >= start and row < start + seq { float(batch) } else { 0.0 }
                assert_near(g[row * width], want, 1e-6)
            }
        }
        jt.zero_grad(positions.parameters())
    }
    #: One sequence, `[seq, dim]`, is the rank-two case.
    let one = jt.Tensor([seq, width], xs[0:seq * width])
    let y = positions.forward(one)
    assert_near(y.download()[width], xs[width] + table[width], 1e-6)
    for t in [one, y] { t.free() }
    assert_throws(ValueError, || positions.forward_at(x, 5))
}

#: Sequence-first `[seq, batch, dim]` gets position `t` on every row of step
#: `t`, the same sum a batch-first input gets, and the same table gradient.
fn test_positional_embedding_takes_a_sequence_first_batch() -> void {
    if not jt.is_available() { return }
    let width = 4
    let seq = 5
    let batch = 3
    let positions = jt.PositionalEmbedding(8, width, batch_first: false)
    defer { positions.free() }
    positions.build(width, Random.with_seed(5))
    let table = positions.parameters()[0].download()
    let xs = [float(i) * 0.5 for i in 0..seq * batch * width]
    let x = jt.Tensor([seq, batch, width], xs)
    defer { x.free() }
    let y = positions.forward_at(x, 2)
    assert_eq(y.shape, [seq, batch, width])
    let got = y.download()
    for t in 0..seq {
        for b in 0..batch {
            for c in 0..width {
                let at = (t * batch + b) * width + c
                assert_near(got[at], xs[at] + table[(2 + t) * width + c], 1e-6)
            }
        }
    }
    let total = jt.sum(y)
    for held in total.backward() { held.free() }
    total.free()
    y.free()
    let grad = positions.parameters()[0].grad
    assert_true(grad is not null)
    if grad is not null {
        let g = grad.download()
        assert_near(g[0], 0.0, 1e-6)
        assert_near(g[2 * width], float(batch), 1e-6)
        assert_near(g[6 * width], float(batch), 1e-6)
        assert_near(g[7 * width], 0.0, 1e-6)
    }
}

#: A table that does not train hands out its rows as a view: the same values
#: as the gather, and freeing it leaves the table alone.
fn test_frozen_positions_are_a_view_of_the_table() -> void {
    if not jt.is_available() { return }
    let positions = jt.PositionalEmbedding(6, 3)
    defer { positions.free() }
    positions.build(3, Random.with_seed(8))
    let table = positions.parameters()[0]
    let gathered = positions.positions(2, start: 3)
    let want = gathered.download()
    assert_true(gathered.creator is not null)
    gathered.free()
    table.requires_grad = false
    let viewed = positions.positions(2, start: 3)
    assert_eq(viewed.shape, [2, 3])
    assert_eq(viewed.download(), want)
    assert_true(viewed.creator is null)
    viewed.free()
    assert_eq(table.download()[9:15], want)
}

#: Several recorded forwards at different lengths and offsets, then one
#: backward. Each gather reads its index again in the backward, so asking for
#: another length must not free the index an earlier forward still needs: it
#: threw "this Tensor has already been freed". Both layouts and the rank-two
#: case, and the graph frees what it took, so nothing is left over.
fn test_positions_at_several_lengths_share_one_backward() -> void {
    if not jt.is_available() { return }
    let width = 4
    let max_seq = 10
    let batch = 2
    for batch_first in [true, false] {
        let positions = jt.PositionalEmbedding(max_seq, width, batch_first: batch_first)
        defer { positions.free() }
        positions.build(width, Random.with_seed(3))
        let table = positions.parameters()[0]
        let short = if batch_first { [batch, 3, width] } else { [3, batch, width] }
        let long = if batch_first { [batch, 6, width] } else { [6, batch, width] }
        var live = -1
        for round in 0..3 {
            let xa = jt.Tensor.zeros(short)
            let xb = jt.Tensor.zeros(long)
            let xc = jt.Tensor.zeros([5, width])
            let a = positions.forward_at(xa, 0)
            let b = positions.forward_at(xb, 4)
            let c = positions.forward(xc)
            let total = jt.add(jt.add(jt.sum(a), jt.sum(b)), jt.sum(c))
            for held in total.backward() { held.free() }
            for t in [total, xa, xb, xc] { t.free() }
            let grad = table.grad
            assert_true(grad is not null)
            if grad is not null {
                let g = grad.download()
                for row in 0..max_seq {
                    var want = 0.0
                    if row < 3 { want += float(batch) }
                    if row >= 4 { want += float(batch) }
                    if row < 5 { want += 1.0 }
                    for column in 0..width { assert_near(g[row * width + column], want, 1e-6) }
                }
            }
            jt.zero_grad(positions.parameters())
            if round == 1 { live = gpu.live_buffers() }
            if round == 2 { assert_eq(gpu.live_buffers(), live) }
        }
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

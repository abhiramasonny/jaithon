#: The layer objects: what shape each one builds itself for and hands on.
import jaitensor as jt
import std.gpu as gpu
import std.math
from std.gpu import Buffer
from std.random import Random
from std.test import assert_eq, assert_near, assert_throws, assert_true
from jaitensor import Layer, ResBlock, Tensor

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

fn formula(count: int, f: fn(int) -> float) -> list[float] {
    return [f(index) for index in 0..count]
}

#: A built block with every parameter set by formula: the formulas the
#: PyTorch reference values in `test_resblock_matches_torch` were made from.
fn residual_block_for(shape: list[int], channels: int, stride: int) -> ResBlock {
    let block = jt.ResBlock(channels, stride: stride)
    block.build_from_shape(shape, Random.with_seed(1))
    let inputs = shape[2]
    let parameters = block.parameters()
    parameters[0].upload(formula(9 * inputs * channels, |i| 0.2 * math.cos(0.53 * float(i))))
    parameters[3].upload(
        formula(9 * channels * channels, |i| 0.2 * math.sin(0.29 * float(i) + 0.4))
    )
    var norm_slots = [1, 4]
    if parameters.len() > 6 {
        parameters[6].upload(formula(inputs * channels, |i| 0.3 * math.cos(0.71 * float(i) + 0.2)))
        norm_slots.push(7)
    }
    for (which, slot) in norm_slots.enumerate() {
        parameters[slot].upload(formula(channels, |k| 1.0 + 0.1 * math.sin(float(k + which))))
        parameters[slot + 1].upload(formula(channels, |k| 0.05 * math.cos(float(k + which))))
    }
    return block
}

#: Forward the block, or the same thing written out op by op, and take
#: sum(out * r) back through it. Returns the output, then every gradient:
#: input first, then the parameters in `parameters()` order.
fn residual_gradients(
    block: ResBlock,
    shape: list[int],
    batch: int,
    composed: bool
) -> list[list[float]] {
    let full = [batch, shape[0], shape[1], shape[2]]
    var size = batch
    for d in shape { size *= d }
    let x = jt.Tensor(full, formula(size, |i| math.sin(0.37 * float(i) + 0.1)), requires_grad: true)
    let p = block.parameters()
    for parameter in p { parameter.grad = null }
    let out = if composed {
        let inner = jt.relu(jt.batch_norm(jt.conv2d(x, p[0], stride: block.stride, padding: 1), p[1], p[2]))
        let main = jt.batch_norm(jt.conv2d(inner, p[3], padding: 1), p[4], p[5])
        let skip = if block.projects {
            jt.batch_norm(jt.conv2d(x, p[6], stride: block.stride), p[7], p[8])
        } else {
            x
        }
        jt.relu(jt.add(main, skip))
    } else {
        block.forward(x)
    }
    let r = jt.Tensor(out.shape, formula(out.size, |i| math.cos(0.11 * float(i))))
    let total = jt.sum(jt.multiply(out, r))
    let held = total.backward()
    var results: list[list[float]] = [out.download(), [total.download()[0]]]
    let dx = x.grad
    if dx is null { throw AssertionError("no input gradient") }
    results.push(dx.download())
    for parameter in p {
        let gradient = parameter.grad
        if gradient is null { throw AssertionError("a parameter has no gradient") }
        results.push(gradient.download())
        gradient.free()
        parameter.grad = null
    }
    for tensor in held { tensor.free() }
    dx.free()
    total.free()
    out.free()
    r.free()
    x.free()
    return results
}

fn total_of(values: list[float]) -> float {
    var sum = 0.0
    for value in values { sum += value }
    return sum
}

#: The block's gradients are the composed ops' gradients, value for value, for
#: a projecting stride-2 block and an identity one -- the guard for any later
#: change to how the block computes, such as a fused norm, add and ReLU.
fn test_resblock_grads_equal_the_composed_ops() -> void {
    if not jt.is_available() { return }
    for (shape, channels, stride, batch) in [([6, 6, 3], 4, 2, 2), ([4, 4, 4], 4, 1, 3)] {
        let block = residual_block_for(shape, channels, stride)
        assert_eq(block.projects, stride != 1)
        let by_block = residual_gradients(block, shape, batch, false)
        let by_ops = residual_gradients(block, shape, batch, true)
        assert_eq(by_block.len(), by_ops.len())
        for which in 0..by_ops.len() {
            assert_eq(by_block[which].len(), by_ops[which].len())
            for index in 0..by_ops[which].len() {
                assert_near(by_block[which][index], by_ops[which][index], 1e-5)
            }
        }
        block.free()
    }
}

#: Against PyTorch (float64, scratch reference written alongside this test):
#: conv2d, batch_norm(training) with eps 1e-5, relu and the add, over the
#: same formula-filled input and weights.
fn test_resblock_matches_torch() -> void {
    if not jt.is_available() { return }
    let block = residual_block_for([6, 6, 3], 4, 2)
    let got = residual_gradients(block, [6, 6, 3], 2, false)
    assert_eq(block.output_shape(), [3, 3, 4])
    assert_near(got[1][0], 1.2264257557609548, 1e-3)
    let out = [0.33348134885730374, 0.8040403551418778, 1.451771534879697, 0.5598669091504183]
    let dx = [0.5771239286106222, -1.4197143825904082, 2.9803747163730216, 0.2690760413608884]
    for index in 0..4 {
        assert_near(got[0][index], out[index], 1e-4)
        assert_near(got[2][index], dx[index], 1e-3)
    }
    assert_near(total_of(got[2]), 0.20999611111815142, 1e-3)
    assert_near(total_of(got[3]), 123.2467647388892, 2e-2)
    assert_near(got[3][0], 2.8577647061970937, 1e-3)
    assert_near(total_of(got[6]), -47.43867306957737, 1e-2)
    let dg1 = [-1.468511138457086, 2.367015845132579, -1.3750966619987843, 0.29379540681799693]
    let db2 = [2.736544481956602, 1.0935854823795657, 1.3710093074555854, 1.7443494832848023]
    let dg3 = [2.4488736146419865, 4.35196263484087, 3.6309830799310783, 1.0248435118605923]
    for k in 0..4 {
        assert_near(got[4][k], dg1[k], 1e-3)
        assert_near(got[8][k], db2[k], 1e-3)
        assert_near(got[10][k], dg3[k], 1e-3)
    }
    assert_near(total_of(got[9]), 55.75301253636987, 1e-2)
    block.free()

    let identity = residual_block_for([4, 4, 4], 4, 1)
    let plain = residual_gradients(identity, [4, 4, 4], 3, false)
    assert_eq(identity.projects, false)
    assert_eq(plain.len(), 2 + 1 + 6)
    assert_near(plain[1][0], -0.5747420327786019, 1e-3)
    assert_near(total_of(plain[2]), -0.8178202911818033, 1e-3)
    assert_near(plain[2][0], 3.9630381104014174, 1e-3)
    assert_near(total_of(plain[3]), -563.9193173590868, 5e-2)
    assert_near(total_of(plain[6]), -149.54959283980114, 2e-2)
    identity.free()
}

#: Inside `Sequential`: the shape goes through, a projection appears exactly
#: where the shape changes, and the parameters are counted.
fn test_resblock_builds_inside_sequential() -> void {
    if not jt.is_available() { return }
    let model = jt.Sequential(
        [
            jt.Conv2d(8, 3, padding: 1),
            jt.BatchNorm2d(),
            jt.Activate(jt.Activation.Relu),
            jt.ResBlock(8),
            jt.ResBlock(16, stride: 2),
            jt.AvgPool2d(4),
            jt.Flatten(),
            jt.Dense(3, activation: jt.Activation.Softmax),
        ]
    )
    let images = jt.Tensor([4, 8, 8, 2], formula(4 * 8 * 8 * 2, |i| math.sin(float(i))))
    let labels = jt.Tensor([4], [0.0, 1.0, 2.0, 1.0])
    let data = jt.Dataset(images, labels)
    model.compile(optimizer: jt.SGD(learning_rate: 0.05, momentum: 0.9, weight_decay: 5e-4))
    let history = model.fit(data, epochs: 2, batch_size: 4, verbose: false, plot_path: null)
    assert_true(math.is_finite(history.loss[1]))
    assert_eq(model.training_route(), "autograd")
    let identity = model.layers[3]
    let projected = model.layers[4]
    assert_eq(identity.output_shape(), [8, 8, 8])
    assert_eq(projected.output_shape(), [4, 4, 16])
    #: 3x3x8x8 twice and four norms of 8; then 3x3x8x16, 3x3x16x16, 1x1x8x16
    #: and three norms of 16.
    var counted = 0
    for parameter in identity.parameters() { counted += parameter.size }
    assert_eq(counted, 2 * 576 + 4 * 8)
    counted = 0
    for parameter in projected.parameters() { counted += parameter.size }
    assert_eq(counted, 1152 + 2304 + 128 + 6 * 16)
    model.free()
    data.free()
}

#: The highest handle among the next 256 buffers. Handles are handed out
#: lowest free slot first, so this stays put while everything made in between
#: is freed, and climbs by one for every handle something kept below it. (One
#: probe is not enough: a kept handle can sit above a slot a later free opens.)
fn handle_frontier() -> int {
    var probes: list[Buffer] = []
    var top = 0
    for _index in 0..256 {
        let probe = Buffer(1)
        top = max(top, probe.handle_of())
        probes.push(probe)
    }
    for probe in probes { probe.free() }
    return top
}

#: A training step through residual blocks frees every handle it made.
#:
#: A convolution's backward took a flat view of its upstream gradient and
#: never freed it: one handle kept per convolution per step. Handles are found
#: by a scan from the bottom of the table, so the step grew slower the longer
#: a run went on -- ResNet-20's thirtieth epoch took three times its first,
#: all of it in that scan. `handle_frontier` climbs by one for each kept handle.
fn test_a_residual_training_step_keeps_no_handles() -> void {
    if not jt.is_available() { return }
    let layers: list[Layer] = [
        jt.Conv2d(4, 3, padding: 1, in_channels: 2),
        jt.ResBlock(4),
        jt.ResBlock(8, stride: 2),
        jt.AvgPool2d(2),
        jt.Flatten(),
        jt.Dense(3, activation: jt.Activation.Softmax),
    ]
    var shape = [4, 4, 2]
    let rng = Random.with_seed(2)
    for layer in layers {
        layer.build_from_shape(shape, rng)
        shape = layer.output_shape()
    }
    var parameters: list[Tensor] = []
    for layer in layers { parameters.extend(layer.parameters()) }
    let optimizer = jt.SGD(learning_rate: 0.01, momentum: 0.9, weight_decay: 1e-4)
    let loss_fn = jt.SparseCategoricalCrossentropy()
    let x = jt.Tensor([4, 4, 4, 2], formula(128, |i| math.sin(float(i))))
    let y = jt.Tensor([4], [0.0, 1.0, 2.0, 0.0])
    var before = 0
    for step in 0..6 {
        if step == 3 { before = handle_frontier() }
        var current = x
        for layer in layers { current = layer.forward(current) }
        let loss = loss_fn.forward(current, y)
        jt.zero_grad(parameters)
        let graph = loss.backward()
        optimizer.step(parameters)
        loss.free()
        for held in graph { held.free() }
    }
    assert_eq(handle_frontier(), before)
    for layer in layers { layer.free() }
    x.free()
    y.free()
}

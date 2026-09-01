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

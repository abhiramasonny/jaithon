#: The pooling backwards, each against the same sum walked the other way round.
import jaitensor as jt
from std.test import assert_eq, assert_near

fn assert_values_near(actual: list[float], expected: list[float], tolerance: float = 1e-5) -> void {
    assert_eq(actual.len(), expected.len())
    for (index, value) in actual.enumerate() {
        assert_near(value, expected[index], tolerance)
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

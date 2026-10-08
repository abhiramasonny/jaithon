#: `Dataset` and the loader: how a batch is built and what shuffling has to keep.
import jaitensor as jt
from std.test import assert_eq, assert_near, assert_throws

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

#: The shift (dx, dy) and flip that turn `source` image `n` into `output`
#: image `n`, as `[dx, dy, flip]`, or an empty list when none does. Pixels
#: brought in from outside the image are zero; the source holds no zeros.
fn crop_that_made(source: list[float], output: list[float], n: int, height: int, width: int, channels: int, padding: int) -> list[int] {
    let per = height * width * channels
    for flip in [0, 1] {
        for dy in -padding..=padding {
            for dx in -padding..=padding {
                var same = true
                for y in 0..height {
                    for x in 0..width {
                        let sx = (if flip == 1 { width - 1 - x } else { x }) + dx
                        let sy = y + dy
                        for c in 0..channels {
                            let want = if sx >= 0 and sx < width and sy >= 0 and sy < height {
                                source[n * per + (sy * width + sx) * channels + c]
                            } else {
                                0.0
                            }
                            if output[n * per + (y * width + x) * channels + c] != want { same = false }
                        }
                    }
                }
                if same { return [dx, dy, flip] }
            }
        }
    }
    return []
}

#: Every image comes out as itself shifted by at most `padding` each way,
#: zero-filled, and mirrored or not: a crop of the padded image. A seed gives
#: the same batch every time and another seed another batch, and across a
#: batch every kind of crop turns up.
fn test_random_crop_flip_is_a_seeded_padded_crop() -> void {
    if not jt.is_available() { return }
    let images = 64
    let height = 5
    let width = 6
    let channels = 2
    let padding = 2
    let size = images * height * width * channels
    let source = jt.Tensor([images, height, width, channels], [float(index + 1) for index in 0..size])
    let values = source.download()
    let first = jt.random_crop_flip(source, padding: padding, seed: 11)
    let again = jt.random_crop_flip(source, padding: padding, seed: 11)
    let other = jt.random_crop_flip(source, padding: padding, seed: 12)
    let a = first.download()
    assert_eq(a, again.download())
    assert a != other.download()
    var shifts: dict[str, bool] = {}
    var flips = 0
    for n in 0..images {
        let crop = crop_that_made(values, a, n, height, width, channels, padding)
        assert crop.len() == 3, f"image {n} is not a crop of its source"
        shifts[f"{crop[0]},{crop[1]}"] = true
        flips += crop[2]
    }
    assert shifts.len() >= 15, f"only {shifts.len()} of 25 shifts in 64 images"
    assert flips > 16 and flips < 48, f"{flips} of 64 flipped"

    #: Without flipping, every crop is an unflipped one; written in place.
    let held = jt.Tensor(source.shape)
    jt.random_crop_flip_into(source, held, padding: padding, seed: 11, flip: false)
    let unflipped = held.download()
    for n in 0..images {
        let crop = crop_that_made(values, unflipped, n, height, width, channels, padding)
        assert crop.len() == 3 and crop[2] == 0, f"image {n} flipped or not a crop"
    }
    #: No padding and no flip is a copy.
    jt.random_crop_flip_into(source, held, padding: 0, seed: 5, flip: false)
    assert_eq(held.download(), values)
    let _self = assert_throws(ValueError, || jt.random_crop_flip_into(source, source))
    let _flat = assert_throws(ValueError, || jt.random_crop_flip(jt.Tensor([4, 3])))
    for tensor in [source, first, again, other, held] { tensor.free() }
}

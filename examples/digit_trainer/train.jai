#: Train a digit classifier on glyphs jaicv draws, then read handwriting-ish
#: digits back with it.
#:
#:     jaithon run examples/digit_trainer/train.jai
#:
#: The point is that both halves are in this repository and neither needs a
#: download: jaicv RENDERS the training set with `put_text`, jaitensor TRAINS
#: on it, and jaicv draws the results back out. There is no MNIST here and no
#: weights to fetch -- the data is generated, so the whole thing runs from a
#: clean checkout.
#:
#: What makes the generated set worth training on is that every sample is
#: distorted differently: a rotation, a scale, a shift, a stroke width and a
#: little noise, drawn from a seeded generator. A network trained on ten
#: pristine glyphs would memorise ten bitmaps and learn nothing.
import jaicv as cv
from jaicv import Mat, Point, Point2f, Rect, Scalar, Size
from jaitensor import Activation, Adam, Dataset, Dense, Sequential
from jaitensor import SparseCategoricalCrossentropy, Tensor, is_available
from std.random import Random
from std.time import Instant

let SIDE = 28
let CLASSES = 10
let TRAIN_PER_CLASS = 400
let TEST_PER_CLASS = 40
let EPOCHS = 120

#: One distorted digit, as `SIDE * SIDE` values in 0..1.
fn render_digit(digit: int, rng: Random) -> list[float] {
    var canvas = Mat.zeros(SIDE, SIDE, cv.CV_8UC1)
    #: A scale that fills most of the box, varied enough that the network
    #: cannot key on stroke length alone.
    let scale = 0.85 + rng.next_float() * 0.45
    let thickness = if rng.next_float() < 0.4 { 1 } else { 2 }
    let face = if rng.next_float() < 0.5 { cv.FONT_HERSHEY_SIMPLEX } else { cv.FONT_HERSHEY_DUPLEX }
    let text = f"{digit}"
    let (size, baseline) = cv.get_text_size(text, face, scale, thickness)
    #: Centred, then shifted by a couple of pixels either way.
    let x = (SIDE - size.width) // 2 + int(rng.next_float() * 5.0) - 2
    let y = (SIDE + size.height - baseline) // 2 + int(rng.next_float() * 5.0) - 2
    cv.put_text(canvas, text, Point(x, y), face, scale, Scalar(255.0), thickness)

    #: A small rotation about the centre, which is the distortion a classifier
    #: most needs to see: a `1` leaning left and a `7` leaning right are
    #: otherwise the easiest pair in the set to confuse.
    let angle = (rng.next_float() - 0.5) * 34.0
    let rotation = cv.get_rotation_matrix_2d(
        cv.Point2f(float(SIDE) * 0.5, float(SIDE) * 0.5),
        angle,
        1.0
    )
    var warped = cv.warp_affine(canvas, rotation, Size(SIDE, SIDE))

    var pixels = warped.values()
    var out = __prim__.list_filled(SIDE * SIDE, 0.0)
    for index in 0..SIDE * SIDE {
        let noise = (rng.next_float() - 0.5) * 24.0
        var value = (pixels[index] + noise) / 255.0
        if value < 0.0 { value = 0.0 }
        if value > 1.0 { value = 1.0 }
        out[index] = value
    }
    return out
}

fn build_set(per_class: int, rng: Random) -> tuple[Tensor, Tensor] {
    let count = per_class * CLASSES
    var pixels = __prim__.list_filled(count * SIDE * SIDE, 0.0)
    var labels = __prim__.list_filled(count, 0.0)
    var at = 0
    for digit in 0..CLASSES {
        for _sample in 0..per_class {
            let image = render_digit(digit, rng)
            for index in 0..SIDE * SIDE { pixels[at * SIDE * SIDE + index] = image[index] }
            labels[at] = float(digit)
            at += 1
        }
    }
    return (Tensor([count, SIDE * SIDE], pixels), Tensor([count], labels))
}

#: A contact sheet of what the network got wrong, which says more about a
#: classifier than its accuracy does.
fn write_mistakes(images: Tensor, truth: list[float], predicted: list[int]) -> int {
    var wrong: list[int] = []
    for index in 0..truth.len() {
        if int(truth[index]) != predicted[index] { wrong.push(index) }
    }
    if wrong.len() == 0 { return 0 }

    let columns = 10
    let cell = SIDE * 3
    let rows = (min(wrong.len(), 40) + columns - 1) // columns
    var sheet = Mat.zeros(rows * cell, columns * cell, cv.CV_8UC3)
    let flat = images.download()
    for slot in 0..min(wrong.len(), 40) {
        let index = wrong[slot]
        var tile = __prim__.list_filled(SIDE * SIDE, 0.0)
        for pixel in 0..SIDE * SIDE {
            tile[pixel] = flat[index * SIDE * SIDE + pixel] * 255.0
        }
        var small = Mat.from_list(SIDE, SIDE, cv.CV_8UC1, tile)
        small.flush()
        let big = cv.cvt_color(cv.resize(small, Size(cell, cell)), cv.COLOR_GRAY2BGR)
        let top = slot // columns * cell
        let left = slot % columns * cell
        #: A view onto the cell, written through: `roi` shares the sheet's
        #: storage, so copying into it copies into the sheet.
        big.copy_to(sheet.roi(Rect(left, top, cell, cell)))
        cv.put_text(
            sheet,
            f"{predicted[index]}",
            Point(left + 4, top + cell - 6),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            Scalar(60.0, 60.0, 240.0),
            2
        )
        cv.put_text(
            sheet,
            f"{int(truth[index])}",
            Point(left + cell - 20, top + 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            Scalar(80.0, 220.0, 100.0),
            1
        )
    }
    cv.imwrite("digit_mistakes.png", sheet)
    return wrong.len()
}

fn main() -> int {
    if not is_available() {
        print("this needs a Metal device")
        return 1
    }
    var rng = Random()
    rng.reseed(20260820)

    var clock = Instant.now()
    let (train_x, train_y) = build_set(TRAIN_PER_CLASS, rng)
    let (test_x, test_y) = build_set(TEST_PER_CLASS, rng)
    let train = Dataset(train_x, train_y)
    let test = Dataset(test_x, test_y)
    print(
        f"rendered {TRAIN_PER_CLASS * CLASSES} training and {TEST_PER_CLASS * CLASSES} test digits in {clock.elapsed().to_seconds():.1f}s"
    )

    #: Deliberately a plain MLP rather than a convolutional net: on 28x28
    #: glyphs it reaches the high nineties, trains in seconds, and keeps the
    #: example about the two libraries meeting rather than about architecture.
    var model = Sequential(
        [
            Dense(128, activation: Activation.Relu),
            Dense(64, activation: Activation.Relu),
            Dense(CLASSES, activation: Activation.Softmax),
        ]
    )
    model.compile(optimizer: Adam(learning_rate: 0.002), loss: SparseCategoricalCrossentropy())

    clock = Instant.now()
    let history = model.fit(train, epochs: EPOCHS, batch_size: 64, test_data: test)
    print(f"trained {EPOCHS} epochs in {clock.elapsed().to_seconds():.1f}s")

    #: `predict_classes` hands back the winning index per sample already, so
    #: there is nothing to download from it.
    let guesses = model.predict_classes(test_x)
    let truth = test_y.download()
    var correct = 0
    for index in 0..truth.len() {
        if int(truth[index]) == guesses[index] { correct += 1 }
    }
    let accuracy = float(correct) / float(truth.len()) * 100.0
    print(f"test accuracy {accuracy:.1f}% ({correct} of {truth.len()})")

    let wrong = write_mistakes(test_x, truth, guesses)
    if wrong > 0 {
        print(
            f"{wrong} wrong; the first few are in digit_mistakes.png (guess bottom-left, truth top-right)"
        )
    } else {
        print("nothing wrong on the test set")
    }
    return 0
}

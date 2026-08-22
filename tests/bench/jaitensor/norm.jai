#: The same CIFAR-10 network as conv.jai, with a normalisation after each
#: convolution -- which is what a modern vision network actually looks like.
#:
#: Normalisation reduces over the batch and every pixel, an axis that is not
#: the one it writes, and none of the other benchmarks train through one. That
#: gap hid a backward pass costing eighty-seven milliseconds where its forward
#: cost a twentieth of one.
from jaitensor import Activation, BatchNorm2d, Conv2d, Dataset, Dense, Flatten, Layer, MaxPool2d
from jaitensor import Sequential
from jaitensor import SparseCategoricalCrossentropy, SGD, Tensor, nhwc
from jaitensor import is_available, set_mixed_precision
from std.io import Path, read_bytes
from std.os import env_or, run_checked
from std.time import Instant
import std.gpu as gpu
import std.math

const SIDE = 32
const CHANNELS = 3
const CLASSES = 10
const PIXELS = SIDE * SIDE * CHANNELS

fn ensure_data() -> void {
    run_checked(["python3", "tests/bench/jaitensor/prepare_data.py", "cifar10"])
}

#: CIFAR-10 arrives planar, so the batch is uploaded as NCHW and permuted once.
fn load_cifar(samples: int) -> Dataset {
    let root = Path("data/cifar-10")
    let images = read_bytes(root.join("train-images.bin").to_str())
    let labels = read_bytes(root.join("train-labels.bin").to_str())
    if images.len() // PIXELS < samples { throw ValueError("cifar-10 train split is too small") }
    let planar = Tensor([samples, CHANNELS, SIDE, SIDE])
    planar.buffer().upload_u8(images, 0, samples * PIXELS, 0, 1.0 / 255.0)
    let features = nhwc(planar)
    planar.free()
    var values: list[float] = []
    for index in 0..samples { values.push(float(labels[index])) }
    return Dataset(features, Tensor([samples], values))
}

fn make_layers(width: int) -> list[Layer] {
    return [
        Conv2d(width, 3, padding: 1, in_channels: CHANNELS, activation: Activation.Relu),
        BatchNorm2d(),
        MaxPool2d(2),
        Conv2d(width * 2, 3, padding: 1, activation: Activation.Relu),
        BatchNorm2d(),
        MaxPool2d(2),
        Flatten(),
        Dense(128, activation: Activation.Relu),
        Dense(CLASSES, activation: Activation.Softmax),
    ]
}

#: Multiply-accumulates per sample, forward only; the record scales by six for
#: the backward pass exactly as the dense benchmark does.
fn macs_per_sample(width: int) -> int {
    var total = 9 * CHANNELS * width * SIDE * SIDE
    let half = SIDE // 2
    total += 9 * width * (width * 2) * half * half
    let quarter = half // 2
    let flat = quarter * quarter * width * 2
    total += flat * 128
    return total + 128 * CLASSES
}

fn run(name: str, width: int, samples: int, batch_size: int, epochs: int, verbose: bool) -> void {
    let train = load_cifar(samples)
    defer { train.free() }
    let model = Sequential(make_layers(width))
    defer { model.free() }
    model.compile(
        optimizer: SGD(learning_rate: 0.05),
        loss: SparseCategoricalCrossentropy(),
        mixed_precision: true
    )
    model.fit(train, epochs: 1, batch_size: batch_size, verbose: false, plot_path: null)
    gpu.synchronize()
    let started = Instant.now()
    let history = model.fit(
        train,
        epochs: epochs,
        batch_size: batch_size,
        verbose: verbose,
        plot_path: null
    )
    gpu.synchronize()
    let seconds = started.elapsed().to_seconds()
    let processed = samples // batch_size * batch_size * epochs
    let flops = 6 * processed * macs_per_sample(width)
    var valid = history.loss.len() == epochs and seconds > 0.0
    for value in history.loss {
        if not math.is_finite(value) or value <= 0.0 { valid = false }
    }
    if history.loss.len() > 1 and history.loss[history.loss.len() - 1] > history.loss[0] * 1.05 {
        valid = false
    }
    let check = if valid { "ok" } else { "invalid" }
    let final_loss = if history.loss.len() == 0 {
        0.0
    } else {
        history.loss[history.loss.len() - 1]
    }
    print(f"jtb\t{name}\t{seconds:.9f}\t{flops}\t{processed}\t{check}\t{final_loss:.6f}")
}

fn main() -> int {
    if not is_available() { throw RuntimeError("bench needs a GPU") }
    set_mixed_precision(true)
    let level = env_or("BENCH_LEVEL", "hard")
    let verbose = env_or("JAITENSOR_VERBOSE", "0") != "0"
    ensure_data()
    let samples = if level == "easy" { 4096 } elif level == "medium" { 8192 } else { 16384 }
    let epochs = if level == "easy" { 1 } elif level == "medium" { 2 } else { 3 }
    run("norm-small", 32, samples, 128, epochs, verbose)
    run("norm-wide", 64, samples, 256, epochs, verbose)
    return 0
}

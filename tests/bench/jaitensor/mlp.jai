from jaitensor import Activation, Dataset, Dense, Layer, Sequential, SparseCategoricalCrossentropy, SGD
from jaitensor import device_name, is_available
from std.io import Path, read_bytes
from std.os import env_or, run_checked
from std.time import Instant
import std.gpu as gpu
import std.math

fn read_u32(data: bytes, offset: int) -> int {
    if offset + 4 > data.len() { throw ValueError("truncated IDX header") }
    return data[offset] << 24 | data[offset + 1] << 16 | data[offset + 2] << 8 | data[offset + 3]
}

fn join_ints(values: list[int]) -> str {
    var text = ""
    for (index, value) in values.enumerate() {
        if index > 0 { text = text + "," }
        text = text + f"{value}"
    }
    return text
}

fn load_idx(images_path: str, labels_path: str) -> Dataset {
    let images = read_bytes(images_path)
    let labels = read_bytes(labels_path)
    let samples = read_u32(images, 4)
    let rows = read_u32(images, 8)
    let columns = read_u32(images, 12)
    return Dataset.from_u8(
        images,
        labels,
        samples,
        rows * columns,
        feature_offset: 16,
        label_offset: 8,
        feature_scale: 1.0 / 255.0
    )
}

fn load_packed(images_path: str, labels_path: str, width: int) -> Dataset {
    let images = read_bytes(images_path)
    let labels = read_bytes(labels_path)
    if width <= 0 { throw ValueError("feature width must be positive") }
    if images.len() % width != 0 {
        throw ValueError("packed image bytes are not a multiple of width")
    }
    let samples = images.len() // width
    if labels.len() != samples { throw ValueError("packed label count does not match images") }
    return Dataset.from_u8(images, labels, samples, width, feature_scale: 1.0 / 255.0)
}

fn dataset_of(name: str) -> str {
    if name == "fashion-wide" or name == "fashion-deep" or name == "fashion-xl" { return "fashion" }
    if name == "mnist-deep" or name == "mnist-xl" { return "mnist" }
    if name == "cifar-wide" or name == "cifar-deep" or name == "cifar-bottleneck" {
        return "cifar10"
    }
    if name == "cifar100-wide" or name == "cifar100-deep" { return "cifar100" }
    return name
}

fn ensure_data(name: str) -> void {
    run_checked(["python3", "tests/bench/jaitensor/prepare_data.py", dataset_of(name)])
}

fn load_idx_pair(root: Path) -> tuple[Dataset, Dataset] {
    return (
        load_idx(
            root.join("train-images-idx3-ubyte").to_str(),
            root.join("train-labels-idx1-ubyte").to_str()
        ),
        load_idx(
            root.join("t10k-images-idx3-ubyte").to_str(),
            root.join("t10k-labels-idx1-ubyte").to_str()
        )
    )
}

fn load_workload(name: str) -> tuple[Dataset, Dataset, list[int], int] {
    if name == "fashion" or name == "fashion-wide" or name == "fashion-deep" or name == "fashion-xl" {
        let (train, test) = load_idx_pair(Path("data/fashion-mnist"))
        let hidden = if name == "fashion-wide" {
            [2048]
        } elif name == "fashion-deep" {
            [256, 256, 256]
        } elif name == "fashion-xl" {
            [4096]
        } else {
            [256]
        }
        return (train, test, hidden, 10)
    }
    if name == "mnist" or name == "mnist-deep" or name == "mnist-xl" {
        let (train, test) = load_idx_pair(Path("data/mnist"))
        let hidden = if name == "mnist-deep" {
            [256, 256, 256]
        } elif name == "mnist-xl" {
            [8192]
        } else {
            [256]
        }
        return (train, test, hidden, 10)
    }
    if name == "kmnist" {
        let (train, test) = load_idx_pair(Path("data/kmnist"))
        return (train, test, [256], 10)
    }
    if name == "cifar10" or
        name == "cifar-wide" or
        name == "cifar-deep" or
        name == "cifar-bottleneck" {
        let root = Path("data/cifar-10")
        let hidden = if name == "cifar-wide" {
            [1024]
        } elif name == "cifar-deep" {
            [512, 512, 512]
        } elif name == "cifar-bottleneck" {
            [1024, 256, 1024]
        } else {
            [512]
        }
        return (
            load_packed(
                root.join("train-images.bin").to_str(),
                root.join("train-labels.bin").to_str(),
                3072
            ),
            load_packed(
                root.join("test-images.bin").to_str(),
                root.join("test-labels.bin").to_str(),
                3072
            ),
            hidden,
            10
        )
    }
    if name == "cifar100" or name == "cifar100-wide" or name == "cifar100-deep" {
        let root = Path("data/cifar-100")
        let hidden = if name == "cifar100-wide" {
            [2048]
        } elif name == "cifar100-deep" {
            [512, 512, 512]
        } else {
            [512]
        }
        return (
            load_packed(
                root.join("train-images.bin").to_str(),
                root.join("train-labels.bin").to_str(),
                3072
            ),
            load_packed(
                root.join("test-images.bin").to_str(),
                root.join("test-labels.bin").to_str(),
                3072
            ),
            hidden,
            100
        )
    }
    throw ValueError(f"unknown workload {name}")
}

fn batch_size_of(hidden: list[int], features: int) -> int {
    var widest = 0
    for width in hidden {
        if width > widest { widest = width }
    }
    if widest >= 8192 { return 512 }
    # MPSGraph selects a multi-gigabyte workspace for the 2048x3072x2048
    # training graph. 512 stays below the workspace cliff and is faster on
    # both peers for this shape.
    if features >= 3072 and widest >= 2048 { return 512 }
    if widest >= 1024 or hidden.len() >= 3 or features >= 3072 { return 2048 }
    return 512
}

fn use_amp(hidden: list[int]) -> bool {
    var widest = 0
    for width in hidden {
        if width > widest { widest = width }
    }
    return widest >= 1024
}

fn make_layers(hidden: list[int], classes: int) -> list[Layer] {
    var layers: list[Layer] = []
    for width in hidden { layers.push(Dense(width, activation: Activation.Relu)) }
    layers.push(Dense(classes, activation: Activation.Softmax))
    return layers
}

fn main(args: list[str]) -> int {
    let name = if args.len() > 1 { args[1] } else { env_or("JAITENSOR_WORKLOAD", "fashion") }
    let level = env_or("BENCH_LEVEL", "hard")
    let verbose = env_or("JAITENSOR_VERBOSE", "0") != "0"
    let epochs = if args.len() > 2 {
        int(args[2])
    } elif level == "easy" {
        1
    } elif level == "medium" {
        2
    } else {
        8
    }
    if not is_available() { throw RuntimeError("bench needs a GPU") }
    ensure_data(name)
    let (train, test, hidden, classes) = load_workload(name)
    defer { train.free() }
    # This benchmark only trains. Do not keep a test split GPU-resident while
    # the PyTorch peer leaves its test split on the CPU.
    test.free()
    let batch_size = batch_size_of(hidden, train.feature_count)
    if verbose {
        print(f"framework: jaitensor")
        print(f"workload: {name}")
        print(f"GPU: {device_name()}")
        print(f"samples: {train.samples}")
        print(f"features: {train.feature_count}")
        print(f"epochs: {epochs}")
        print(f"batch_size: {batch_size}")
        print(f"hidden: {hidden}")
        print(f"classes: {classes}")
    }
    let model = Sequential(make_layers(hidden, classes))
    defer { model.free() }
    model.compile(
        optimizer: SGD(learning_rate: 0.08),
        loss: SparseCategoricalCrossentropy(),
        mixed_precision: use_amp(hidden)
    )
    # Warm graph compilation, allocator pools and command submission before
    # collecting steady-state samples, as the microbenchmarks do.
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
    let measured_seconds = started.elapsed().to_seconds()
    if env_or("JAITENSOR_LOSS_TRACE", "0") != "0" { print(f"losses: {history.loss}") }
    if verbose {
        print(f"train_seconds: {measured_seconds:.4f}")
        print(f"samples_per_second: {float(train.samples * epochs) / measured_seconds:.1f}")
        return 0
    }
    let processed_per_epoch = train.samples // batch_size * batch_size
    let processed = processed_per_epoch * epochs
    var macs = 0
    var previous = train.feature_count
    for width in hidden {
        macs += previous * width
        previous = width
    }
    macs += previous * classes
    let flops = 6 * processed * macs
    var valid = history.loss.len() == epochs and history.seconds.len() == epochs
    for value in history.loss {
        if not math.is_finite(value) or value <= 0.0 { valid = false }
    }
    if history.loss.len() > 1 and history.loss[history.loss.len() - 1] > history.loss[0] * 1.05 {
        valid = false
    }
    for value in history.seconds {
        if not math.is_finite(value) or value <= 0.0 { valid = false }
    }
    if not math.is_finite(measured_seconds) or measured_seconds <= 0.0 { valid = false }
    let check = if valid { "ok" } else { "invalid" }
    let final_loss = if history.loss.len() == 0 {
        0.0
    } else {
        history.loss[history.loss.len() - 1]
    }
    print(f"jtb\t{name}\t{measured_seconds:.9f}\t{flops}\t{processed}\t{check}\t{final_loss:.6f}")
    return 0
}

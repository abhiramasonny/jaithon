from jaitensor import Activation, Dataset, Dense, Sequential, SparseCategoricalCrossentropy, SGD
from jaitensor import device_name, is_available
from std.io import Path, read_bytes

const PIXELS = 784
const CLASSES = 10

fn read_u32(data: bytes, offset: int) -> int {
    if offset + 4 > data.len() { throw ValueError("truncated IDX header") }
    return data[offset] << 24 | data[offset + 1] << 16 | data[offset + 2] << 8 | data[offset + 3]
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

fn main(args: list[str]) -> int {
    let epochs = if args.len() > 1 { int(args[1]) } else { 8 }
    let root = Path(if args.len() > 2 { args[2] } else { "data/fashion-mnist" })
    if not is_available() { throw RuntimeError("bench needs a GPU") }
    let train = load_idx(
        root.join("train-images-idx3-ubyte").to_str(),
        root.join("train-labels-idx1-ubyte").to_str()
    )
    let test = load_idx(
        root.join("t10k-images-idx3-ubyte").to_str(),
        root.join("t10k-labels-idx1-ubyte").to_str()
    )
    defer { train.free() }
    defer { test.free() }

    print(f"framework: jaitensor")
    print(f"GPU: {device_name()}")
    print(f"samples: {train.samples}")
    print(f"epochs: {epochs}")
    print(f"batch_size: 512")
    let model = Sequential(
        [
            Dense(256, activation: Activation.Relu),
            Dense(CLASSES, activation: Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(optimizer: SGD(learning_rate: 0.08), loss: SparseCategoricalCrossentropy())
    let history = model.fit(
        train,
        epochs: epochs,
        batch_size: 512,
        test_data: test,
        plot_path: null
    )
    var total = 0.0
    for seconds in history.seconds { total += seconds }
    print(f"train_seconds: {total:.4f}")
    print(f"samples_per_second: {float(train.samples * epochs) / total:.1f}")
    return 0
}

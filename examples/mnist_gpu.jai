#: Train and evaluate a dense MNIST classifier with jaitensor.
#:
#: With the dataset installed in `data/mnist`, run:
#:
#:     jaithon run examples/mnist_gpu.jai
#:
#: You can pass image IDX, label IDX, epoch count, and batch size.
import jaitensor as jt
from jaitensor import Dataset
from std.io import read_bytes

fn read_u32(data: bytes, offset: int) -> int {
    if offset < 0 or offset + 4 > data.len() { throw ValueError("truncated IDX header") }
    return data[offset] << 24 | data[offset + 1] << 16 | data[offset + 2] << 8 | data[offset + 3]
}

fn load_mnist(images_path: str, labels_path: str) -> Dataset {
    let images = read_bytes(images_path)
    let labels = read_bytes(labels_path)
    if read_u32(images, 0) != 2051 or read_u32(labels, 0) != 2049 {
        throw ValueError("expected uncompressed MNIST IDX image and label files")
    }
    let samples = read_u32(images, 4)
    let label_count = read_u32(labels, 4)
    let rows = read_u32(images, 8)
    let columns = read_u32(images, 12)
    if samples != label_count {
        throw ValueError(f"MNIST has {samples} images but {label_count} labels")
    }
    let pixels = rows * columns
    if images.len() != 16 + samples * pixels {
        throw ValueError("MNIST image file is truncated or has trailing data")
    }
    if labels.len() != 8 + samples {
        throw ValueError("MNIST label file is truncated or has trailing data")
    }
    return jt.Dataset.from_u8(
        images,
        labels,
        samples,
        pixels,
        feature_offset: 16,
        label_offset: 8,
        feature_scale: 1.0 / 255.0
    )
}

fn main(args: list[str]) -> int {
    let image_path = if args.len() > 1 { args[1] } else { "data/mnist/train-images-idx3-ubyte" }
    let label_path = if args.len() > 2 { args[2] } else { "data/mnist/train-labels-idx1-ubyte" }
    let epochs = if args.len() > 3 { int(args[3]) } else { 50 }
    let batch_size = if args.len() > 4 { int(args[4]) } else { 512 }

    if not jt.is_available() { throw RuntimeError("MNIST training needs a GPU") }
    print(f"GPU: {jt.device_name()}")
    let train = load_mnist(image_path, label_path)
    defer { train.free() }

    let model = jt.Sequential(
        [
            jt.Dense(256, activation: jt.Activation.Relu),
            jt.Dense(10, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(optimizer: jt.SGD(learning_rate: 0.08), loss: jt.SparseCategoricalCrossentropy())
    # The input width becomes known from the dataset on the first fit.
    model.fit(train, epochs: epochs, batch_size: batch_size)
    model.summary()

    let test_images = "data/mnist/t10k-images-idx3-ubyte"
    let test_labels = "data/mnist/t10k-labels-idx1-ubyte"
    if image_path == "data/mnist/train-images-idx3-ubyte" {
        let test = load_mnist(test_images, test_labels)
        defer { test.free() }
        print(f"test: {model.evaluate(test, batch_size: batch_size)}")
    }
    return 0
}

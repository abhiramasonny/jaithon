from jaitensor import Activation, Dataset, Dense, Sequential, SparseCategoricalCrossentropy, SGD, Tensor
from jaitensor import device_name, is_available
from jaiplot import Color, Figure, Rect, save
from std.io import Path, read_bytes
from std.os import run_checked

const SIDE = 28
const PIXELS = 784
const CLASSES = 10
let NAMES = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Boot",
]
let FILES = [
    (
        "train-images-idx3-ubyte",
        "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz"
    ),
    (
        "train-labels-idx1-ubyte",
        "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz"
    ),
    (
        "t10k-images-idx3-ubyte",
        "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz"
    ),
    (
        "t10k-labels-idx1-ubyte",
        "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz"
    ),
]

fn ensure_idx(dest: Path, url: str) -> void {
    if dest.exists() { return }
    dest.parent().mkdir(parents: true, exist_ok: true)
    let gz = Path(dest.to_str() + ".gz")
    if not gz.exists() {
        print(f"downloading {url}")
        run_checked(["curl", "-L", "--fail", "--retry", "3", "-o", gz.to_str(), url])
    }
    print(f"decompressing {gz}")
    run_checked(["gzip", "-df", gz.to_str()])
}

fn read_u32(data: bytes, offset: int) -> int {
    if offset < 0 or offset + 4 > data.len() { throw ValueError("truncated IDX header") }
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

fn fetch_fashion(root: Path) -> void {
    for (name, url) in FILES {
        ensure_idx(root.join(name), url)
    }
}

fn image_cells(pixels: list[float], start: int) -> list[float] {
    var cells: list[float] = []
    for row in 0..SIDE {
        let source = (SIDE - 1 - row) * SIDE
        for column in 0..SIDE {
            cells.push(pixels[start + source + column])
        }
    }
    return cells
}

fn pick_gallery(labels: list[float], per_class: int) -> list[int] {
    var counts = [0 for _index in 0..CLASSES]
    var chosen: list[int] = []
    for (index, label) in labels.enumerate() {
        let class_id = int(label)
        if class_id < 0 or class_id >= CLASSES { continue }
        if counts[class_id] >= per_class { continue }
        counts[class_id] += 1
        chosen.push(index)
        if chosen.len() == CLASSES * per_class { break }
    }
    return chosen
}

fn draw_gallery(
    output: Path,
    pixels: list[float],
    labels: list[float],
    predicted: list[int],
    chosen: list[int]
) -> void {
    let rows = 4
    let columns = 5
    let figure = Figure(width: 1100, height: 900)
    let panels = figure.subplots(rows, columns)
    let ink = Color.from_hex("#f4f4f4")
    let paper = Color.from_hex("#111118")
    for (slot, index) in chosen.enumerate() {
        let panel = panels[slot // columns][slot % columns]
        let truth = int(labels[index])
        let guess = predicted[slot]
        let title = if guess == truth { NAMES[truth] } else { f"{NAMES[truth]} / {NAMES[guess]}" }
        panel.grid_of(
            image_cells(pixels, index * PIXELS),
            SIDE,
            SIDE,
            Rect(0.0, 0.0, float(SIDE), float(SIDE)),
            low_color: paper,
            high_color: ink
        )
        panel.set_xlim(0.0, float(SIDE))
        panel.set_ylim(0.0, float(SIDE))
        panel.set_aspect()
        panel.set_title(title)
    }
    save(figure, output.join("fashion_mnist.svg").to_str())
    save(figure, output.join("fashion_mnist.png").to_str())
}

fn main(args: list[str]) -> int {
    let epochs = if args.len() > 1 { int(args[1]) } else { 8 }
    let output = Path(if args.len() > 2 { args[2] } else { "data/fashion-mnist" })
    output.mkdir(parents: true, exist_ok: true)
    fetch_fashion(output)

    if not is_available() { throw RuntimeError("fashion-mnist training needs a GPU") }
    let train = load_idx(
        output.join("train-images-idx3-ubyte").to_str(),
        output.join("train-labels-idx1-ubyte").to_str()
    )
    let test = load_idx(
        output.join("t10k-images-idx3-ubyte").to_str(),
        output.join("t10k-labels-idx1-ubyte").to_str()
    )
    defer { train.free() }
    defer { test.free() }

    print(f"GPU: {device_name()}")
    print(f"training on {train.samples} fashion-mnist images")
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
        plot_path: output.join("training_history.svg").to_str()
    )

    let pixels = test.features.download()
    let labels = test.labels.download()
    let chosen = pick_gallery(labels, 2)
    var flat: list[float] = []
    for index in chosen { flat.extend(pixels[index * PIXELS:(index + 1) * PIXELS]) }
    let batch = Tensor([chosen.len(), PIXELS], flat)
    defer { batch.free() }
    let predicted = model.predict_classes(batch)
    draw_gallery(output, pixels, labels, predicted, chosen)

    model.summary()
    print(f"wrote {output.join("fashion_mnist.svg")}")
    print(f"wrote {output.join("fashion_mnist.png")}")
    print(f"wrote {history.plot_path}")
    return 0
}

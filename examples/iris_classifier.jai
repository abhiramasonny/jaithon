from jaitensor import Activation, Adam, Dataset, Dense, History, Sequential, SparseCategoricalCrossentropy
from jaitensor import device_name, is_available
from jaiplot import Color, Figure, Rect, save
from std.io import Path, read_csv
from std.os import run_checked
from std.random import Random

const CLASSES = 3
const IRIS_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
let NAMES = ["setosa", "versicolor", "virginica"]

fn ensure_file(path: Path, url: str) -> void {
    if path.exists() { return }
    path.parent().mkdir(parents: true, exist_ok: true)
    print(f"downloading {url}")
    run_checked(["curl", "-L", "--fail", "--retry", "3", "-o", path.to_str(), url])
}

fn class_id(name: str) -> int {
    if name == "setosa" { return 0 }
    if name == "versicolor" { return 1 }
    if name == "virginica" { return 2 }
    throw ValueError(f"unknown iris species {name}")
}

fn load_iris(path: Path) -> tuple[list[list[float]], list[int]] {
    ensure_file(path, IRIS_URL)
    let table = read_csv(path.to_str())
    var rows: list[list[float]] = []
    var labels: list[int] = []
    for (index, record) in table.enumerate() {
        if record.len() < 5 { continue }
        if index == 0 and record[4] == "species" { continue }
        rows.push([float(record[2]), float(record[3])])
        labels.push(class_id(record[4]))
    }
    if rows.len() == 0 { throw ValueError("iris csv had no samples") }
    return (rows, labels)
}

fn shuffle_together(
    rows: list[list[float]],
    labels: list[int],
    seed: int
) -> tuple[list[list[float]], list[int]] {
    let rng = Random.with_seed(seed)
    var mixed_rows = [row for row in rows]
    var mixed_labels = [label for label in labels]
    var index = mixed_rows.len() - 1
    while index > 0 {
        let swap = rng.int_range(0, index + 1)
        let row = mixed_rows[index]
        mixed_rows[index] = mixed_rows[swap]
        mixed_rows[swap] = row
        let label = mixed_labels[index]
        mixed_labels[index] = mixed_labels[swap]
        mixed_labels[swap] = label
        index -= 1
    }
    return (mixed_rows, mixed_labels)
}

fn split(
    rows: list[list[float]],
    labels: list[int],
    train_count: int,
    validation_count: int
) -> tuple[list[list[float]], list[int], list[list[float]], list[int], list[list[float]], list[int]] {
    let train_rows = rows[0:train_count]
    let train_labels = labels[0:train_count]
    let validation_rows = rows[train_count:train_count + validation_count]
    let validation_labels = labels[train_count:train_count + validation_count]
    let test_rows = rows[train_count + validation_count:rows.len()]
    let test_labels = labels[train_count + validation_count:labels.len()]
    return (train_rows, train_labels, validation_rows, validation_labels, test_rows, test_labels)
}

fn bounds(rows: list[list[float]], axis: int, pad: float) -> tuple[float, float] {
    var low = rows[0][axis]
    var high = rows[0][axis]
    for row in rows {
        if row[axis] < low { low = row[axis] }
        if row[axis] > high { high = row[axis] }
    }
    return (low - pad, high + pad)
}

fn classifier() -> Sequential {
    return Sequential(
        [
            Dense(16, activation: Activation.Tanh),
            Dense(16, activation: Activation.Tanh),
            Dense(CLASSES, activation: Activation.Linear),
        ]
    )
}

fn prediction_grid(
    model: Sequential,
    side: int,
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float
) -> list[float] {
    var rows: list[list[float]] = []
    for row in 0..side {
        let y = y_low + (y_high - y_low) * (float(row) + 0.5) / float(side)
        for column in 0..side {
            let x = x_low + (x_high - x_low) * (float(column) + 0.5) / float(side)
            rows.push([x, y])
        }
    }
    let labels = [0 for _index in 0..rows.len()]
    let grid = Dataset.from_tensor_slices(rows, labels)
    defer { grid.free() }
    return [float(value) for value in model.predict_classes(grid.features, batch_size: 2048)]
}

fn draw_report(
    output: Path,
    model: Sequential,
    history: History,
    rows: list[list[float]],
    labels: list[int],
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float
) -> void {
    let colors = [Color.from_hex("#2463eb"), Color.from_hex("#15956f"), Color.from_hex("#e34b4b")]
    let figure = Figure(width: 1100, height: 480)
    let panels = figure.subplots(1, 2)

    let boundary = panels[0][0]
    let side = 140
    boundary.grid_of(
        prediction_grid(model, side, x_low, x_high, y_low, y_high),
        side,
        side,
        Rect(x_low, y_low, x_high, y_high),
        low_color: Color.from_hex("#e7efff"),
        high_color: Color.from_hex("#ffe8e4")
    )
    for label in 0..CLASSES {
        var xs: list[float] = []
        var ys: list[float] = []
        for (index, row) in rows.enumerate() {
            if labels[index] != label { continue }
            xs.push(row[0])
            ys.push(row[1])
        }
        boundary.scatter(xs, ys, label: NAMES[label], color: colors[label], size: 5.5)
    }
    boundary.set_xlim(x_low, x_high)
    boundary.set_ylim(y_low, y_high)
    boundary.set_aspect()
    boundary.set_title("iris petals")
    boundary.set_xlabel("petal length (cm)")
    boundary.set_ylabel("petal width (cm)")
    boundary.legend()

    let epochs = [float(index + 1) for index in 0..history.accuracy.len()]
    let curves = panels[0][1]
    curves.line(epochs, history.accuracy, label: "training")
    curves.line(epochs, history.val_accuracy, label: "validation", dash: [7.0, 4.0])
    curves.set_ylim(0.0, 100.0)
    curves.set_title("classification accuracy")
    curves.set_xlabel("epoch")
    curves.set_ylabel("accuracy (%)")
    curves.grid(true)
    curves.legend()

    save(figure, output.join("iris_classifier.svg").to_str())
    save(figure, output.join("iris_classifier.png").to_str())
}

fn main(args: list[str]) -> int {
    let epochs = if args.len() > 1 { int(args[1]) } else { 80 }
    let output = Path(if args.len() > 2 { args[2] } else { "data/iris-classifier" })
    output.mkdir(parents: true, exist_ok: true)

    if not is_available() { throw RuntimeError("iris training needs a GPU") }
    let (raw_rows, raw_labels) = load_iris(Path("data/iris/iris.csv"))
    let (rows, labels) = shuffle_together(raw_rows, raw_labels, 41)
    let (train_rows, train_labels, validation_rows, validation_labels, test_rows, test_labels) = split(
        rows,
        labels,
        90,
        30
    )
    let train = Dataset.from_tensor_slices(train_rows, train_labels)
    let validation = Dataset.from_tensor_slices(validation_rows, validation_labels)
    let test = Dataset.from_tensor_slices(test_rows, test_labels)
    defer { train.free() }
    defer { validation.free() }
    defer { test.free() }

    let (x_low, x_high) = bounds(rows, 0, 0.4)
    let (y_low, y_high) = bounds(rows, 1, 0.3)

    print(f"GPU: {device_name()}")
    print(f"training on {train.samples} iris flowers")
    let model = classifier()
    defer { model.free() }
    model.compile(Adam(learning_rate: 0.05), loss: SparseCategoricalCrossentropy(from_logits: true))
    let history = model.fit(
        train,
        epochs: epochs,
        batch_size: 30,
        verbose: true,
        validation_data: validation,
        test_data: test,
        plot_path: output.join("training_history.svg").to_str()
    )

    model.summary()
    draw_report(output, model, history, rows, labels, x_low, x_high, y_low, y_high)
    print(f"wrote {output.join("iris_classifier.svg")}")
    print(f"wrote {output.join("iris_classifier.png")}")
    print(f"wrote {history.plot_path}")
    return 0
}

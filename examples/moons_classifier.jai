import std.math
from jaitensor import Activation, Adam, Dataset, Dense, History, Sequential, SparseCategoricalCrossentropy
from jaitensor import device_name, is_available
from jaiplot import Color, Figure, Rect, save
from std.io import Path
from std.random import Random

const CLASSES = 2
const X_LOW = -1.6
const X_HIGH = 2.6
const Y_LOW = -1.2
const Y_HIGH = 1.6

fn moon_rows(
    samples_per_class: int,
    noise: float,
    seed: int
) -> tuple[list[list[float]], list[int]] {
    let rng = Random.with_seed(seed)
    var rows: list[list[float]] = []
    var labels: list[int] = []
    for index in 0..samples_per_class {
        let t = math.PI * float(index) / float(samples_per_class - 1)
        rows.push([math.cos(t) + rng.gauss(0.0, noise), math.sin(t) + rng.gauss(0.0, noise)])
        labels.push(0)
        rows.push(
            [1.0 - math.cos(t) + rng.gauss(0.0, noise), 0.5 - math.sin(t) + rng.gauss(0.0, noise)]
        )
        labels.push(1)
    }
    return (rows, labels)
}

fn classifier() -> Sequential {
    return Sequential(
        [
            Dense(64, activation: Activation.Tanh),
            Dense(64, activation: Activation.Tanh),
            Dense(CLASSES, activation: Activation.Linear),
        ]
    )
}

fn prediction_grid(model: Sequential, side: int) -> list[float] {
    var rows: list[list[float]] = []
    for row in 0..side {
        let y = Y_LOW + (Y_HIGH - Y_LOW) * (float(row) + 0.5) / float(side)
        for column in 0..side {
            let x = X_LOW + (X_HIGH - X_LOW) * (float(column) + 0.5) / float(side)
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
    validation_rows: list[list[float]],
    validation_labels: list[int]
) -> void {
    let colors = [Color.from_hex("#2463eb"), Color.from_hex("#e34b4b")]
    let figure = Figure(width: 1100, height: 480)
    let panels = figure.subplots(1, 2)

    let boundary = panels[0][0]
    let side = 140
    boundary.grid_of(
        prediction_grid(model, side),
        side,
        side,
        Rect(X_LOW, Y_LOW, X_HIGH, Y_HIGH),
        low_color: Color.from_hex("#e7efff"),
        high_color: Color.from_hex("#ffe8e4")
    )
    for label in 0..CLASSES {
        var xs: list[float] = []
        var ys: list[float] = []
        for (index, row) in validation_rows.enumerate() {
            if validation_labels[index] != label { continue }
            xs.push(row[0])
            ys.push(row[1])
        }
        boundary.scatter(xs, ys, label: f"class {label}", color: colors[label], size: 4.5)
    }
    boundary.set_xlim(X_LOW, X_HIGH)
    boundary.set_ylim(Y_LOW, Y_HIGH)
    boundary.set_aspect()
    boundary.set_title("learned moon regions")
    boundary.set_xlabel("x")
    boundary.set_ylabel("y")
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

    save(figure, output.join("moons_classifier.svg").to_str())
    save(figure, output.join("moons_classifier.png").to_str())
}

fn main(args: list[str]) -> int {
    let epochs = if args.len() > 1 { int(args[1]) } else { 80 }
    let output = Path(if args.len() > 2 { args[2] } else { "data/moons-classifier" })
    output.mkdir(parents: true, exist_ok: true)

    if not is_available() { throw RuntimeError("moon training needs a GPU") }
    let (train_rows, train_labels) = moon_rows(250, 0.12, 41)
    let (validation_rows, validation_labels) = moon_rows(80, 0.14, 92)
    let (test_rows, test_labels) = moon_rows(80, 0.14, 117)
    let train = Dataset.from_tensor_slices(train_rows, train_labels)
    let validation = Dataset.from_tensor_slices(validation_rows, validation_labels)
    let test = Dataset.from_tensor_slices(test_rows, test_labels)
    defer { train.free() }
    defer { validation.free() }
    defer { test.free() }

    print(f"GPU: {device_name()}")
    print(f"training on {train.samples} generated moon points")
    let model = classifier()
    defer { model.free() }
    model.compile(Adam(learning_rate: 0.01), loss: SparseCategoricalCrossentropy(from_logits: true))
    let history = model.fit(
        train,
        epochs: epochs,
        batch_size: 250,
        verbose: true,
        validation_data: validation,
        test_data: test,
        plot_path: output.join("training_history.svg").to_str()
    )

    model.summary()
    draw_report(output, model, history, validation_rows, validation_labels)
    print(f"wrote {output.join("moons_classifier.svg")}")
    print(f"wrote {output.join("moons_classifier.png")}")
    print(f"wrote {history.plot_path}")
    return 0
}

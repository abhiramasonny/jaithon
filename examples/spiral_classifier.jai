#: Train a nonlinear spiral classifier and plot the learned decision regions.
#:
#: The example generates its own data. It exercises Adam, tanh layers, logits,
#: validation history, tensor softmax and argmax operations, batched prediction,
#: weight persistence, and jaiplot. Outputs go under the ignored `data` folder.
#:
#:     jaithon run examples/spiral_classifier.jai
#:     jaithon run examples/spiral_classifier.jai -- 120 data/spiral-demo
import jaiplot
import std.math
from jaitensor import Activation, Adam, Dataset, Dense, History, Sequential, SparseCategoricalCrossentropy, argmax, device_name, is_available, softmax
from jaiplot import Color, Figure, Rect
from std.io import Path
from std.random import Random

const CLASSES = 3
const LIMIT = 3.3

fn spiral_rows(
    samples_per_class: int,
    noise: float,
    seed: int
) -> tuple[list[list[float]], list[int]] {
    let rng = Random.with_seed(seed)
    var rows: list[list[float]] = []
    var labels: list[int] = []
    for label in 0..CLASSES {
        for index in 0..samples_per_class {
            let radius = 0.25 + 2.75 * float(index) / float(samples_per_class - 1)
            let arm = math.TAU * float(label) / float(CLASSES)
            let angle = arm + radius * 2.25 + rng.gauss(0.0, noise)
            rows.push([radius * math.cos(angle), radius * math.sin(angle)])
            labels.push(label)
        }
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

fn accuracy(predictions: list[int], labels: list[int]) -> float {
    var correct = 0
    for (index, prediction) in predictions.enumerate() {
        if prediction == labels[index] { correct += 1 }
    }
    return 100.0 * float(correct) / float(labels.len())
}

fn prediction_grid(model: Sequential, side: int) -> list[float] {
    var rows: list[list[float]] = []
    for row in 0..side {
        let y = -LIMIT + 2.0 * LIMIT * (float(row) + 0.5) / float(side)
        for column in 0..side {
            let x = -LIMIT + 2.0 * LIMIT * (float(column) + 0.5) / float(side)
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
    let colors = [Color.from_hex("#2463eb"), Color.from_hex("#e34b4b"), Color.from_hex("#15956f")]
    let figure = Figure(width: 1100, height: 480)
    let panels = figure.subplots(1, 2)

    let boundary = panels[0][0]
    let side = 120
    boundary.grid_of(
        prediction_grid(model, side),
        side,
        side,
        Rect(-LIMIT, -LIMIT, LIMIT, LIMIT),
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
    boundary.set_xlim(-LIMIT, LIMIT)
    boundary.set_ylim(-LIMIT, LIMIT)
    boundary.set_aspect()
    boundary.set_title("learned spiral regions")
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

    jaiplot.save(figure, output.join("spiral_classifier.svg").to_str())
    jaiplot.save(figure, output.join("spiral_classifier.png").to_str())
}

fn main(args: list[str]) -> int {
    let epochs = if args.len() > 1 { int(args[1]) } else { 100 }
    let output = Path(if args.len() > 2 { args[2] } else { "data/spiral-classifier" })
    output.mkdir(parents: true, exist_ok: true)

    if not is_available() { throw RuntimeError("spiral training needs a GPU") }
    let (train_rows, train_labels) = spiral_rows(300, 0.18, 41)
    let (validation_rows, validation_labels) = spiral_rows(100, 0.22, 92)
    let (test_rows, test_labels) = spiral_rows(100, 0.22, 117)
    let train = Dataset.from_tensor_slices(train_rows, train_labels)
    let validation = Dataset.from_tensor_slices(validation_rows, validation_labels)
    let test = Dataset.from_tensor_slices(test_rows, test_labels)
    defer { train.free() }
    defer { validation.free() }
    defer { test.free() }

    print(f"GPU: {device_name()}")
    print(f"training on {train.samples} generated spiral points")
    let model = classifier()
    defer { model.free() }
    model.compile(Adam(learning_rate: 0.01), loss: SparseCategoricalCrossentropy(from_logits: true))
    let history = model.fit(
        train,
        epochs: epochs,
        batch_size: 300,
        verbose: true,
        validation_data: validation,
        test_data: test,
        plot_path: output.join("training_history.svg").to_str()
    )

    let weights_path = output.join("spiral_classifier.weights.json")
    model.save_weights(weights_path.to_str())
    let restored = classifier()
    defer { restored.free() }
    restored.compile(Adam(), loss: SparseCategoricalCrossentropy(from_logits: true))
    restored.load_weights(weights_path.to_str())

    let logits = restored.predict(validation.features)
    let probabilities = softmax(logits)
    let predicted = argmax(probabilities)
    defer { logits.free() }
    defer { probabilities.free() }
    defer { predicted.free() }
    let classes = [int(value) for value in predicted.download()]
    print(f"restored validation accuracy: {accuracy(classes, validation_labels):.2f}%")
    restored.summary()

    draw_report(output, restored, history, validation_rows, validation_labels)
    print(f"wrote {output.join("spiral_classifier.svg")}")
    print(f"wrote {output.join("spiral_classifier.png")}")
    print(f"wrote {history.plot_path}")
    print(f"wrote {weights_path}")
    return 0
}

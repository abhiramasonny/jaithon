#: Training-history plots rendered through jaiplot.
import jaiplot
from jaiplot import Figure
from std.io import Path

#: Save loss and accuracy curves to an SVG, PNG, or PPM file.
#:
#: Validation values form per-epoch curves. Test values are final aggregates,
#: so the report draws them as horizontal references across the epochs.
pub fn save_history_plot(
    path: str,
    loss: list[float],
    accuracy: list[float],
    val_loss: list[float] = [],
    val_accuracy: list[float] = [],
    test_loss: float? = null,
    test_accuracy: float? = null
) -> void {
    if loss.len() == 0 { throw ValueError("a history plot needs at least one epoch") }
    if accuracy.len() != loss.len() {
        throw ValueError("training loss and accuracy histories must have the same length")
    }
    if val_loss.len() != 0 and val_loss.len() != loss.len() {
        throw ValueError("validation loss must be empty or contain one value per epoch")
    }
    if val_accuracy.len() != 0 and val_accuracy.len() != loss.len() {
        throw ValueError("validation accuracy must be empty or contain one value per epoch")
    }

    let epochs = [float(index + 1) for index in 0..loss.len()]
    let figure = Figure(width: 960, height: 420)
    let panels = figure.subplots(1, 2)

    let losses = panels[0][0]
    losses.line(epochs, loss, label: "train")
    if val_loss.len() > 0 {
        losses.line(epochs, val_loss, label: "validation", dash: [7.0, 4.0])
    }
    if test_loss is not null {
        losses.line(epochs, [test_loss for _epoch in epochs], label: "test", dash: [2.0, 4.0])
    }
    losses.set_title("loss")
    losses.set_xlabel("epoch")
    losses.set_ylabel("cross-entropy")
    losses.grid(true)
    losses.legend()

    let accuracies = panels[0][1]
    accuracies.line(epochs, accuracy, label: "train")
    if val_accuracy.len() > 0 {
        accuracies.line(epochs, val_accuracy, label: "validation", dash: [7.0, 4.0])
    }
    if test_accuracy is not null {
        accuracies.line(
            epochs,
            [test_accuracy for _epoch in epochs],
            label: "test",
            dash: [2.0, 4.0]
        )
    }
    accuracies.set_ylim(0.0, 100.0)
    accuracies.set_title("accuracy")
    accuracies.set_xlabel("epoch")
    accuracies.set_ylabel("accuracy (%)")
    accuracies.grid(true)
    accuracies.legend()

    let parent = Path(path).parent()
    if not parent.is_dir() { parent.mkdir(parents: true, exist_ok: true) }
    jaiplot.save(figure, path)
}

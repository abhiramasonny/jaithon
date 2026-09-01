#: Saving a model's weights and loading them back, in memory and through a file.
import jaitensor as jt
import std.math
import std.os
from jaitensor import Dataset
from std.io import Path
from std.test import assert_eq

fn separable_data(repeats: int = 32) -> Dataset {
    var features: list[float] = []
    var labels: list[int] = []
    for _repeat in 0..repeats {
        features.extend([-2.0, -1.0, -1.0, -2.0, 2.0, 1.0, 1.0, 2.0])
        labels.extend([0, 0, 1, 1])
    }
    return jt.Dataset.from_flat(features, labels, 2)
}

fn test_weights_round_trip_through_memory_and_json() -> void {
    if not jt.is_available() { return }
    let data = separable_data(2)
    let source = jt.Sequential([jt.Dense(2, activation: jt.Activation.Softmax)])
    let restored = jt.Sequential([jt.Dense(2, activation: jt.Activation.Softmax)])
    let path = Path(os.temp_dir()).join("jaithon-jaitensor-weights-test.json")
    defer { data.free() }
    defer { source.free() }
    defer { restored.free() }
    defer { if path.exists() { path.remove() } }
    source.compile(jt.SGD())
    source.build(2)
    let weights = source.get_weights()
    source.set_weights(weights)
    source.save_weights(path.to_str())
    restored.compile(jt.SGD())
    restored.load_weights(path.to_str())
    assert_eq(restored.get_weights(), weights)
    assert_eq(restored.predict_classes(data.features), source.predict_classes(data.features))
}

#: A model with a convolution in it, saved and loaded back.
#:
#: The file recorded how many numbers a sample holds and not what shape they
#: are in, and a convolution cannot be rebuilt from the product: the load
#: went down the flat path, every shape-dependent layer came out the wrong
#: size, and it threw on the first parameter that did not match. So no model
#: with a convolution in it could be reloaded at all -- which is most of the
#: reason to save one.
fn test_a_convolutional_model_survives_a_save_and_a_load() -> void {
    if not jt.is_available() { return }
    let path = Path(os.temp_dir()).join("jaithon-jaitensor-conv-weights.json")
    defer { if path.exists() { path.remove() } }
    var pixels: list[float] = []
    var labels: list[float] = []
    var seed = 9
    for _sample in 0..24 {
        seed = (seed * 1103515245 + 12345) % 2147483648
        for _slot in 0..8 * 8 { pixels.push(float(seed % 100) / 100.0) }
        labels.push(float(seed % 3))
    }
    let data = jt.Dataset(jt.Tensor([24, 8, 8, 1], pixels), jt.Tensor([24], labels))
    defer { data.free() }

    let source = jt.Sequential(
        [
            jt.Conv2d(4, 3, padding: 1, in_channels: 1, activation: jt.Activation.Relu),
            jt.MaxPool2d(2),
            jt.Flatten(),
            jt.Dense(3, activation: jt.Activation.Softmax),
        ]
    )
    defer { source.free() }
    source.compile(optimizer: jt.SGD(learning_rate: 0.05), loss: jt.SparseCategoricalCrossentropy())
    source.fit(data, epochs: 2, batch_size: 8, verbose: false, plot_path: null)
    let flat = jt.Tensor([24, 64], pixels)
    let before = source.predict(flat).download()
    source.save_weights(path.to_str())

    let restored = jt.Sequential(
        [
            jt.Conv2d(
                4,
                3,
                padding: 1,
                in_channels: 1,
                activation: jt.Activation.Relu
            ),
            jt.MaxPool2d(2),
            jt.Flatten(),
            jt.Dense(3, activation: jt.Activation.Softmax),
        ]
    )
    defer { restored.free() }
    restored.load_weights(path.to_str())
    let after = restored.predict(flat).download()
    assert_eq(after.len(), before.len())
    for (index, value) in before.enumerate() {
        assert math.fabs(value - after[index]) < 1e-6
    }
}

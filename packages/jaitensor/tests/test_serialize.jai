#: Saving a model's weights and loading them back, in memory and through a file.
import jaitensor as jt
import std.math
import std.os
from jaitensor import Dataset
from std.io import Path
from std.test import assert_eq, assert_throws, assert_true

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

#: Whether two downloads hold the same float32 values bit for bit, as far as a
#: host float can tell: NaN matches NaN, and a zero keeps its sign.
fn same_bits(a: list[float], b: list[float]) -> bool {
    if a.len() != b.len() { return false }
    for (index, x) in a.enumerate() {
        let y = b[index]
        if math.is_nan(x) or math.is_nan(y) {
            if not (math.is_nan(x) and math.is_nan(y)) { return false }
        } elif x == 0.0 and y == 0.0 {
            if (math.copysign(1.0, x) > 0.0) != (math.copysign(1.0, y) > 0.0) { return false }
        } elif x != y {
            return false
        }
    }
    return true
}

#: A binary checkpoint gives back exactly the bits it was given: every rank,
#: the awkward values JSON text cannot carry, the metadata beside them, and
#: an in-place restore into tensors a model already holds.
fn test_a_binary_checkpoint_round_trips_bit_for_bit() -> void {
    if not jt.is_available() { return }
    let path = Path(os.temp_dir()).join("jaithon-jaitensor-checkpoint-test.bin")
    defer { if path.exists() { path.remove() } }
    let odd = [
        0.0,
        -0.0,
        1.0e-40,
        -1.0e-45,
        3.4028234663852886e38,
        math.INF,
        -math.INF,
        math.NAN,
        1.0 / 3.0,
    ]
    let special = jt.Tensor([9], odd)
    let wide = jt.Tensor([3, 70, 33])
    wide.buffer().fill_uniform(-2.0, 2.0, 17)
    let scalar = jt.Tensor([1], [42.5])
    let deep = jt.Tensor([2, 1, 3, 2], [float(i) - 5.5 for i in 0..12])
    let names = ["special", "blocks/wide", "scalar", "deep é"]
    let tensors = [special, wide, scalar, deep]
    defer { for t in tensors { t.free() } }
    jt.save_checkpoint(
        path.to_str(),
        names,
        tensors,
        meta: {"step": 2000, "dim": 256, "note": "gpt"}
    )
    let loaded = jt.load_checkpoint(path.to_str())
    defer { loaded.free() }
    assert_eq(loaded.names, names)
    assert_eq(loaded.meta["step"], 2000)
    assert_eq(loaded.meta["note"], "gpt")
    for (index, tensor) in tensors.enumerate() {
        let back = loaded.tensors[index]
        assert_eq(back.shape, tensor.shape)
        assert_true(same_bits(back.download(), tensor.download()), names[index])
    }
    let specials = loaded.get("special").download()
    assert_true(math.is_nan(specials[7]) and specials[5] == math.INF and specials[6] == -math.INF)
    assert_true(math.copysign(1.0, specials[1]) < 0.0 and specials[2] > 0.0)
    #: Into existing tensors, in any order and leaving the rest of the file.
    let into_wide = jt.Tensor([3, 70, 33])
    let into_deep = jt.Tensor([2, 1, 3, 2])
    defer { into_wide.free() }
    defer { into_deep.free() }
    let meta = jt.restore_checkpoint(
        path.to_str(),
        ["deep é", "blocks/wide"],
        [into_deep, into_wide]
    )
    assert_eq(meta["dim"], 256)
    assert_true(same_bits(into_wide.download(), wide.download()))
    assert_true(same_bits(into_deep.download(), deep.download()))
    let wrong = jt.Tensor([3, 70, 32])
    defer { wrong.free() }
    assert_throws(ValueError, || jt.restore_checkpoint(path.to_str(), ["blocks/wide"], [wrong]))
    assert_throws(KeyError, || jt.restore_checkpoint(path.to_str(), ["absent"], [wrong]))
    assert_throws(ValueError, || jt.save_checkpoint(path.to_str(), ["a", "a"], [wrong, wrong]))
}

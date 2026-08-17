#: Keras-style sequential models trained by general autograd.
import std.gpu as gpu
import std.json as json
from std.io import read_file, write_file
from std.random import Random
from std.time import Instant
from .data import Dataset
from .layers import Activation, Dense, Layer
from .losses import Loss, SparseCategoricalCrossentropy
from .ops import argmax, softmax
from .optimizers import Optimizer, SGD
from .plots import save_history_plot
from .tensor import Tensor

#: Aggregate loss and accuracy for one pass over a dataset.
pub class Metrics {
    #: Mean sparse cross-entropy per sample.
    pub let loss: float
    #: Percentage of samples classified correctly.
    pub let accuracy: float

    #: Store aggregate loss and percentage accuracy.
    pub fn init(self, loss: float, accuracy: float) {
        self.loss = loss
        self.accuracy = accuracy
    }

    fn __str__(self) -> str { return f"loss={self.loss:.4f}, accuracy={self.accuracy:.2f}%" }
}

#: Per-epoch measurements returned by `Sequential.fit`.
pub class History {
    #: Mean loss after each training epoch.
    pub let loss: list[float]
    #: Percentage accuracy after each training epoch.
    pub let accuracy: list[float]
    #: Wall-clock training duration of each epoch.
    pub let seconds: list[float]
    #: Validation loss after each epoch, empty when no validation data was given.
    pub let val_loss: list[float]
    #: Validation accuracy after each epoch, empty without validation data.
    pub let val_accuracy: list[float]
    #: Final test loss, or null when `fit` received no test dataset.
    pub var test_loss: float?
    #: Final test accuracy, or null when `fit` received no test dataset.
    pub var test_accuracy: float?
    #: The report written by `fit`, or null when plotting was disabled.
    pub var plot_path: str?

    #: Create an empty training history.
    pub fn init(self) {
        self.loss = []
        self.accuracy = []
        self.seconds = []
        self.val_loss = []
        self.val_accuracy = []
        self.test_loss = null
        self.test_accuracy = null
        self.plot_path = null
    }

    #: Save this history through jaitensor's jaiplot dependency.
    pub fn plot(self, path: str) -> void {
        save_history_plot(
            path,
            self.loss,
            self.accuracy,
            self.val_loss,
            self.val_accuracy,
            self.test_loss,
            self.test_accuracy
        )
        self.plot_path = path
    }
}

#: A stack of layers trained by general reverse-mode autodiff.
#:
#: Each batch is one fence: forward, loss, backward, and the optimizer step
#: all queue via `dispatch_async`, then one `gpu.synchronize()` reads back
#: that batch's loss and frees its intermediates before the next batch.
pub class Sequential {
    #: Layers in forward order.
    pub let layers: list[Layer]
    var optimizer_value: Optimizer?
    var loss_value: Loss?
    var input_units: int
    var freed: bool

    #: Create an uncompiled model from a non-empty layer list.
    pub fn init(self, layers: list[Layer]) {
        if layers.len() == 0 { throw ValueError("Sequential needs at least one layer") }
        self.layers = [layer for layer in layers]
        self.optimizer_value = null
        self.loss_value = null
        self.input_units = 0
        self.freed = false
    }

    #: Configure the optimizer and loss used by `fit`/`evaluate`.
    pub fn compile(
        self,
        optimizer: Optimizer = SGD(),
        loss: Loss = SparseCategoricalCrossentropy()
    ) -> void {
        self._check_live()
        if not gpu.is_available() { throw RuntimeError("jaitensor needs a GPU") }
        self.optimizer_value = optimizer
        self.loss_value = loss
    }

    #: Train for `epochs`, returning loss and accuracy histories.
    pub fn fit(
        self,
        dataset: Dataset,
        epochs: int = 1,
        batch_size: int = 2048,
        verbose: bool = true,
        validation_data: Dataset? = null,
        test_data: Dataset? = null,
        plot_path: str? = "data/jaitensor-training.svg"
    ) -> History {
        self._prepare(dataset, epochs, batch_size)
        let history = History()

        for epoch in 1..=epochs {
            let started = Instant.now()
            let metrics = self._run_epoch(dataset, batch_size, true)
            let seconds = started.elapsed().to_seconds()
            history.loss.push(metrics.loss)
            history.accuracy.push(metrics.accuracy)
            history.seconds.push(seconds)
            let validation = if validation_data is null {
                null
            } else {
                self.evaluate(validation_data, batch_size: batch_size)
            }
            if validation is not null {
                history.val_loss.push(validation.loss)
                history.val_accuracy.push(validation.accuracy)
            }
            if verbose {
                let suffix = if validation is null {
                    ""
                } else {
                    f", val_loss={validation.loss:.4f}, " + f"val_accuracy={validation.accuracy:.2f}%"
                }
                print(f"epoch {epoch}/{epochs}: {metrics}{suffix} ({seconds:.3f}s)")
            }
        }
        if test_data is not null {
            let testing = self.evaluate(test_data, batch_size: batch_size)
            history.test_loss = testing.loss
            history.test_accuracy = testing.accuracy
            if verbose { print(f"test: {testing}") }
        }
        if plot_path is not null {
            history.plot(plot_path)
            if verbose { print(f"training plot: {plot_path}") }
        }
        return history
    }

    #: Measure loss and accuracy without updating parameters.
    pub fn evaluate(self, dataset: Dataset, batch_size: int = 2048) -> Metrics {
        self._prepare(dataset, 1, batch_size)
        return self._run_epoch(dataset, batch_size, false)
    }

    #: Build layer parameters for a known feature width before training.
    pub fn build(self, input_units: int) -> void {
        self._check_live()
        self._build(input_units)
    }

    #: Run a feature matrix through the model and return device predictions.
    #:
    #: Runs `features` through in one pass rather than chunking by
    #: `batch_size` — see this plan's Task 6 note on why, and on the same
    #: tradeoff in `_slice_features`/`_slice_labels`. `batch_size` is still
    #: validated for call-site compatibility with code written against
    #: today's chunked version.
    pub fn predict(self, features: Tensor, batch_size: int = 2048) -> Tensor {
        self._check_live()
        if batch_size <= 0 { throw ValueError(f"batch_size must be positive, got {batch_size}") }
        if features.rank() != 2 {
            throw ValueError(
                f"predict features must have shape [samples, features], got {features.shape}"
            )
        }
        self._build(features.shape[1])
        let raw = self._forward(features)
        let intermediates = raw.discard_graph()
        gpu.synchronize()
        for tensor in intermediates { tensor.free() }
        let last = self.layers[self.layers.len() - 1]
        if not last.wants_softmax_output() { return raw }
        let probabilities = softmax(raw)
        raw.free()
        return probabilities
    }

    #: Predict the highest-scoring class for each feature row.
    pub fn predict_classes(self, features: Tensor, batch_size: int = 2048) -> list[int] {
        let probabilities = self.predict(features, batch_size: batch_size)
        defer { probabilities.free() }
        let indices = argmax(probabilities)
        defer { indices.free() }
        return [int(value) for value in indices.download()]
    }

    #: Copy every layer's parameters to host lists, in `parameters()` order per layer.
    pub fn get_weights(self) -> list[list[float]] {
        self._check_live()
        if self.input_units == 0 { throw RuntimeError("build the model before reading weights") }
        var values: list[list[float]] = []
        for layer in self.layers {
            for parameter in layer.parameters() { values.push(parameter.download()) }
        }
        return values
    }

    #: Replace every layer's parameters from the flat list returned by `get_weights`.
    pub fn set_weights(self, values: list[list[float]]) -> void {
        self._check_live()
        if self.input_units == 0 { throw RuntimeError("build the model before setting weights") }
        var index = 0
        for layer in self.layers {
            for parameter in layer.parameters() {
                if index >= values.len() {
                    throw ValueError(f"got {values.len()} weight arrays, expected more")
                }
                parameter.upload(values[index])
                index += 1
            }
        }
        if index != values.len() {
            throw ValueError(f"got {values.len()} weight arrays, expected {index}")
        }
    }

    #: Save model dimensions and weights as portable JSON.
    pub fn save_weights(self, path: str) -> void {
        self._check_live()
        if self.input_units == 0 { throw RuntimeError("build the model before saving weights") }
        let document = {
            "format": "jaitensor.weights.v1",
            "input_units": self.input_units,
            "weights": self.get_weights(),
        }
        write_file(path, json.stringify(document) + "\n")
    }

    #: Load weights written by `save_weights`, building the model if needed.
    pub fn load_weights(self, path: str) -> void {
        self._check_live()
        let document = json.parse(read_file(path))
        if document["format"] != "jaitensor.weights.v1" {
            throw ValueError("unsupported jaitensor weight file")
        }
        self._build(int(document["input_units"]))
        var values: list[list[float]] = []
        for packed in document["weights"] {
            values.push([float(value) for value in packed])
        }
        self.set_weights(values)
    }

    #: Number of trainable values across every layer's parameters.
    pub fn parameter_count(self) -> int {
        var count = 0
        for layer in self.layers {
            if self.input_units == 0 { continue }
            for parameter in layer.parameters() { count += parameter.size }
        }
        return count
    }

    #: Print the layer list and total trainable parameter count.
    pub fn summary(self) -> void {
        print("jaitensor Sequential")
        for (index, layer) in self.layers.enumerate() {
            print(f"  {index}: {layer}")
        }
        print(f"trainable parameters: {self.parameter_count()}")
        let optimizer = self.optimizer_value
        if optimizer is not null { print(f"optimizer: {optimizer.name()}") }
    }

    #: Release every layer's parameters.
    pub fn free(self) -> void {
        if self.freed { return }
        gpu.synchronize()
        for layer in self.layers { layer.free() }
        self.freed = true
    }

    fn _prepare(self, dataset: Dataset, epochs: int, batch_size: int) -> void {
        self._check_live()
        if epochs <= 0 { throw ValueError(f"epochs must be positive, got {epochs}") }
        if batch_size <= 0 { throw ValueError(f"batch_size must be positive, got {batch_size}") }
        if self.optimizer_value is null or self.loss_value is null {
            throw RuntimeError("call model.compile() before fit or evaluate")
        }
        self._build(dataset.feature_count)
        for label in dataset.labels.download() {
            let value = int(label)
            if label != float(value) or value < 0 {
                throw ValueError(f"label {label} is not a non-negative integer class")
            }
        }
    }

    fn _build(self, inputs: int) -> void {
        if inputs <= 0 { throw ValueError(f"input width must be positive, got {inputs}") }
        if self.input_units != 0 {
            if self.input_units != inputs {
                throw ValueError(f"model expects {self.input_units} features, got {inputs}")
            }
            return
        }
        let rng = Random.with_seed(7)
        var width = inputs
        for layer in self.layers {
            layer.build(width, rng)
            width = layer.output_width()
        }
        self.input_units = inputs
    }

    fn _forward(self, x: Tensor) -> Tensor {
        var current = x
        for layer in self.layers { current = layer.forward(current) }
        return current
    }

    fn _run_epoch(self, dataset: Dataset, batch_size: int, training: bool) -> Metrics {
        let loss_fn = self.loss_value
        let optimizer = self.optimizer_value
        if loss_fn is null or optimizer is null { throw RuntimeError("model is not compiled") }

        var total_loss = 0.0
        var total_correct = 0.0
        var start = 0
        while start < dataset.samples {
            let count = min(batch_size, dataset.samples - start)
            let batch_features = self._slice_features(dataset, start, count)
            let batch_labels = self._slice_labels(dataset, start, count)

            let prediction = self._forward(batch_features)
            let loss = loss_fn.forward(prediction, batch_labels)
            let intermediates = if training {
                let graph = loss.backward()
                optimizer.step(self._parameters())
                graph
            } else {
                loss.discard_graph()
            }
            gpu.synchronize()
            total_loss += loss.download()[0] * float(count)
            total_correct += self._count_correct(prediction, batch_labels)

            loss.free()
            batch_features.free()
            batch_labels.free()
            for tensor in intermediates { tensor.free() }
            if training { self._zero_grad() }
            start += count
        }
        return Metrics(
            total_loss / float(dataset.samples),
            100.0 * total_correct / float(dataset.samples)
        )
    }

    fn _slice_features(self, dataset: Dataset, start: int, count: int) -> Tensor {
        let width = dataset.feature_count
        let all_values = dataset.features.download()
        let slice = all_values[start * width:(start + count) * width]
        return Tensor([count, width], slice)
    }

    fn _slice_labels(self, dataset: Dataset, start: int, count: int) -> Tensor {
        let all_values = dataset.labels.download()
        let slice = all_values[start:start + count]
        return Tensor([count], slice)
    }

    fn _count_correct(self, prediction: Tensor, labels: Tensor) -> float {
        let predicted = argmax(prediction)
        defer { predicted.free() }
        let predicted_values = predicted.download()
        let label_values = labels.download()
        var correct = 0.0
        for index in 0..predicted_values.len() {
            if int(predicted_values[index]) == int(label_values[index]) { correct += 1.0 }
        }
        return correct
    }

    fn _parameters(self) -> list[Tensor] {
        var parameters: list[Tensor] = []
        for layer in self.layers { parameters.extend(layer.parameters()) }
        return parameters
    }

    fn _zero_grad(self) -> void {
        for parameter in self._parameters() {
            let gradient = parameter.grad
            if gradient is not null {
                gradient.free()
                parameter.grad = null
            }
        }
    }

    fn _check_live(self) -> void {
        if self.freed { throw RuntimeError("this Sequential model has already been freed") }
    }
}

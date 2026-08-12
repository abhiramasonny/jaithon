#: Keras-style sequential dense models.
import std.gpu as gpu
import std.json as json
from std.gpu import Buffer
from std.io import read_file, write_file
from std.random import Random
from std.time import Instant
from .data import Dataset
from .kernels import Kernels
from .layers import Activation, Dense
from .losses import SparseCategoricalCrossentropy
from .ops import argmax
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

class LayerState {
    pub let weight_first: Buffer
    pub let weight_second: Buffer
    pub let bias_first: Buffer
    pub let bias_second: Buffer
    var freed: bool

    fn init(self, layer: Dense) {
        let weight_count = layer.input_units * layer.units
        self.weight_first = gpu.Buffer.of([0.0 for _index in 0..weight_count])
        self.weight_second = gpu.Buffer.of([0.0 for _index in 0..weight_count])
        self.bias_first = gpu.Buffer.of([0.0 for _index in 0..layer.units])
        self.bias_second = gpu.Buffer.of([0.0 for _index in 0..layer.units])
        self.freed = false
    }

    pub fn free(self) -> void {
        if self.freed { return }
        self.weight_first.free()
        self.weight_second.free()
        self.bias_first.free()
        self.bias_second.free()
        self.freed = true
    }
}

class Workspace {
    pub let activations: list[Buffer]
    pub let deltas: list[Buffer]
    pub let metrics: Buffer
    var freed: bool

    fn init(self, layers: list[Dense], batch_size: int, samples: int) {
        self.activations = []
        self.deltas = []
        for layer in layers {
            self.activations.push(gpu.Buffer(batch_size * layer.units))
            self.deltas.push(gpu.Buffer(batch_size * layer.units))
        }
        self.metrics = gpu.Buffer(samples * 2)
        self.freed = false
    }

    pub fn free(self) -> void {
        if self.freed { return }
        gpu.synchronize()
        for buffer in self.activations { buffer.free() }
        for buffer in self.deltas { buffer.free() }
        self.metrics.free()
        self.freed = true
    }
}

#: A stack of dense layers trained as a sparse softmax classifier.
#:
#: The complete epoch is queued before one GPU fence, so buffers and parameters
#: remain resident while Metal executes forward, backward, and optimizer
#: kernels in order. Dense layers support linear, ReLU, sigmoid, tanh, and
#: softmax activations. Sparse classification accepts a final softmax layer or
#: a linear logits layer when the loss sets `from_logits`.
pub class Sequential {
    #: Dense layers in forward order.
    pub let layers: list[Dense]
    var optimizer_value: Optimizer?
    var loss_value: SparseCategoricalCrossentropy?
    var kernels_value: Kernels?
    var settings_value: Buffer?
    var states: list[LayerState]
    var optimizer_step: int
    var input_units: int
    var freed: bool

    #: Create an uncompiled model from a non-empty layer list.
    pub fn init(self, layers: list[Dense]) {
        if layers.len() == 0 { throw ValueError("Sequential needs at least one layer") }
        self.layers = [layer for layer in layers]
        self.optimizer_value = null
        self.loss_value = null
        self.kernels_value = null
        self.settings_value = null
        self.states = []
        self.optimizer_step = 0
        self.input_units = 0
        self.freed = false
    }

    #: Configure sparse classification training and compile the Metal kernels.
    pub fn compile(
        self,
        optimizer: Optimizer = SGD(),
        loss: SparseCategoricalCrossentropy = SparseCategoricalCrossentropy()
    ) -> void {
        self._check_live()
        if not gpu.is_available() { throw RuntimeError("jaitensor needs a GPU") }
        let old_kernels = self.kernels_value
        let old_settings = self.settings_value
        if old_kernels is not null { old_kernels.free() }
        if old_settings is not null { old_settings.free() }
        for state in self.states { state.free() }
        self.states = []
        self.optimizer_value = optimizer
        self.loss_value = loss
        self.kernels_value = Kernels()
        self.settings_value = gpu.Buffer.of(optimizer.settings())
        self.optimizer_step = 0
    }

    #: Train for `epochs`, returning loss and accuracy histories.
    #:
    #: A larger batch keeps the GPU occupied longer. The default is intended
    #: for medium-sized dense classifiers on Apple silicon. The method plots
    #: the completed history at `data/jaitensor-training.svg`; pass
    #: `plot_path: null` to skip the report.
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
        let workspace = Workspace(self.layers, min(batch_size, dataset.samples), dataset.samples)
        defer { workspace.free() }
        let history = History()

        for epoch in 1..=epochs {
            let started = Instant.now()
            self._enqueue(dataset, workspace, batch_size, true)
            gpu.synchronize()
            let metrics = summarize(workspace.metrics.download(), dataset.samples)
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
        let workspace = Workspace(self.layers, min(batch_size, dataset.samples), dataset.samples)
        defer { workspace.free() }
        self._enqueue(dataset, workspace, batch_size, false)
        gpu.synchronize()
        return summarize(workspace.metrics.download(), dataset.samples)
    }

    #: Build layer parameters for a known feature width before training.
    pub fn build(self, input_units: int) -> void {
        self._check_live()
        self._build(input_units)
    }

    #: Run a feature matrix through the model and return device predictions.
    #:
    #: A final softmax layer returns probabilities. Other final activations
    #: return their direct values. The caller owns the returned tensor.
    pub fn predict(self, features: Tensor, batch_size: int = 2048) -> Tensor {
        self._check_live()
        if batch_size <= 0 { throw ValueError(f"batch_size must be positive, got {batch_size}") }
        if features.rank() != 2 {
            throw ValueError(
                f"predict features must have shape [samples, features], got {features.shape}"
            )
        }
        let kernels = self.kernels_value
        if kernels is null { throw RuntimeError("call model.compile() before predict") }
        self._build(features.shape[1])
        let samples = features.shape[0]
        let last_index = self.layers.len() - 1
        let last = self.layers[last_index]
        var scratch: list[Buffer] = []
        for index in 0..last_index {
            scratch.push(gpu.Buffer(min(batch_size, samples) * self.layers[index].units))
        }
        let raw = Tensor([samples, last.units])

        var start = 0
        while start < samples {
            let count = min(batch_size, samples - start)
            var source = features.buffer()
            var source_offset = start
            for (index, layer) in self.layers.enumerate() {
                let output = if index == last_index { raw.buffer() } else { scratch[index] }
                let output_offset = if index == last_index { start } else { 0 }
                kernels.forward.dispatch_async(
                    [source, layer.weights(), layer.bias(), output],
                    count * layer.units,
                    [
                        count,
                        layer.input_units,
                        layer.units,
                        source_offset,
                        layer.activation_code(),
                        output_offset,
                    ]
                )
                source = output
                source_offset = output_offset
            }
            start += count
        }
        gpu.synchronize()
        for buffer in scratch { buffer.free() }
        if last.activation != Activation.Softmax { return raw }
        let probabilities = Tensor(raw.shape)
        kernels.softmax.dispatch(
            [raw.buffer(), probabilities.buffer()],
            samples,
            [samples, last.units]
        )
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

    #: Copy every layer's weights and biases to host lists.
    #:
    #: Entries alternate between a layer's row-major weights and its bias.
    pub fn get_weights(self) -> list[list[float]] {
        self._check_live()
        if self.input_units == 0 { throw RuntimeError("build the model before reading weights") }
        var values: list[list[float]] = []
        for layer in self.layers {
            let (weights, bias) = layer.get_weights()
            values.push(weights)
            values.push(bias)
        }
        return values
    }

    #: Replace all layer weights from the alternating list returned by `get_weights`.
    pub fn set_weights(self, values: list[list[float]]) -> void {
        self._check_live()
        if self.input_units == 0 { throw RuntimeError("build the model before setting weights") }
        if values.len() != self.layers.len() * 2 {
            throw ValueError(f"got {values.len()} weight arrays, expected {self.layers.len() * 2}")
        }
        for (index, layer) in self.layers.enumerate() {
            layer.set_weights(values[index * 2], values[index * 2 + 1])
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

    #: Number of trainable weights and biases.
    pub fn parameter_count(self) -> int {
        var count = 0
        for layer in self.layers {
            if layer.input_units > 0 { count += layer.input_units * layer.units + layer.units }
        }
        return count
    }

    #: Print the layer shapes and total trainable parameter count.
    pub fn summary(self) -> void {
        print("jaitensor Sequential")
        for (index, layer) in self.layers.enumerate() {
            print(f"  {index}: {layer}")
        }
        print(f"trainable parameters: {self.parameter_count()}")
        let optimizer = self.optimizer_value
        if optimizer is not null { print(f"optimizer: {optimizer.name()}") }
    }

    #: Release model parameters and compiled kernels.
    pub fn free(self) -> void {
        if self.freed { return }
        gpu.synchronize()
        for layer in self.layers { layer.free() }
        let kernels = self.kernels_value
        let settings = self.settings_value
        if kernels is not null { kernels.free() }
        if settings is not null { settings.free() }
        for state in self.states { state.free() }
        self.states = []
        self.kernels_value = null
        self.settings_value = null
        self.freed = true
    }

    fn _prepare(self, dataset: Dataset, epochs: int, batch_size: int) -> void {
        self._check_live()
        if epochs <= 0 { throw ValueError(f"epochs must be positive, got {epochs}") }
        if batch_size <= 0 { throw ValueError(f"batch_size must be positive, got {batch_size}") }
        if self.kernels_value is null or self.optimizer_value is null or self.loss_value is null {
            throw RuntimeError("call model.compile() before fit or evaluate")
        }
        self._build(dataset.feature_count)
        self._ensure_states()
        let loss = self.loss_value
        if loss is null { throw RuntimeError("model loss is not configured") }
        let last = self.layers[self.layers.len() - 1]
        if loss.from_logits and last.activation != Activation.Linear {
            throw ValueError("from_logits needs a final Dense layer with Activation.Linear")
        }
        if not loss.from_logits and last.activation != Activation.Softmax {
            throw ValueError("the final Dense layer must use Activation.Softmax")
        }
        for index in 0..self.layers.len() - 1 {
            if self.layers[index].activation == Activation.Softmax {
                throw ValueError("Activation.Softmax is only supported on the final layer")
            }
        }
        for label in dataset.labels.download() {
            let value = int(label)
            if label != float(value) or value < 0 or value >= last.units {
                throw ValueError(f"label {label} is outside the model's {last.units} classes")
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
            width = layer.units
        }
        self.input_units = inputs
    }

    fn _ensure_states(self) -> void {
        if self.states.len() != 0 { return }
        for layer in self.layers { self.states.push(LayerState(layer)) }
    }

    fn _enqueue(
        self,
        dataset: Dataset,
        workspace: Workspace,
        batch_size: int,
        training: bool
    ) -> void {
        let kernels = self.kernels_value
        let settings = self.settings_value
        let optimizer = self.optimizer_value
        if kernels is null or settings is null or optimizer is null {
            throw RuntimeError("model is not compiled")
        }

        var start = 0
        while start < dataset.samples {
            let count = min(batch_size, dataset.samples - start)
            var source = dataset.features.buffer()
            var source_offset = start
            for (index, layer) in self.layers.enumerate() {
                kernels.forward.dispatch_async(
                    [source, layer.weights(), layer.bias(), workspace.activations[index]],
                    count * layer.units,
                    [
                        count,
                        layer.input_units,
                        layer.units,
                        source_offset,
                        layer.activation_code(),
                        0,
                    ]
                )
                source = workspace.activations[index]
                source_offset = 0
            }

            let last_index = self.layers.len() - 1
            let last = self.layers[last_index]
            kernels.loss.dispatch_async(
                [source, dataset.labels.buffer(), workspace.deltas[last_index], workspace.metrics],
                count,
                [count, last.units, start]
            )

            if training {
                self.optimizer_step += 1
                var index = last_index
                while index >= 0 {
                    let layer = self.layers[index]
                    let state = self.states[index]
                    if index > 0 {
                        let previous = self.layers[index - 1]
                        kernels.backward.dispatch_async(
                            [
                                workspace.deltas[index],
                                layer.weights(),
                                workspace.activations[index - 1],
                                workspace.deltas[index - 1],
                            ],
                            count * layer.input_units,
                            [count, layer.input_units, layer.units, previous.activation_code()]
                        )
                    }

                    let layer_input = if index == 0 {
                        dataset.features.buffer()
                    } else {
                        workspace.activations[index - 1]
                    }
                    let offset = if index == 0 { start } else { 0 }
                    kernels.update.dispatch_async(
                        [
                            layer_input,
                            workspace.deltas[index],
                            layer.weights(),
                            layer.bias(),
                            state.weight_first,
                            state.weight_second,
                            state.bias_first,
                            state.bias_second,
                            settings,
                        ],
                        max(layer.input_units * layer.units, layer.units),
                        [
                            count,
                            layer.input_units,
                            layer.units,
                            offset,
                            optimizer.kind(),
                            self.optimizer_step,
                        ]
                    )
                    index -= 1
                }
            }
            start += count
        }
    }

    fn _check_live(self) -> void {
        if self.freed { throw RuntimeError("this Sequential model has already been freed") }
    }
}

fn summarize(values: list[float], samples: int) -> Metrics {
    var loss = 0.0
    var correct = 0.0
    for sample in 0..samples {
        loss += values[sample * 2]
        correct += values[sample * 2 + 1]
    }
    return Metrics(loss / float(samples), 100.0 * correct / float(samples))
}

export { summarize }

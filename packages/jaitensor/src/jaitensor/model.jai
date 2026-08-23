#: Keras-style sequential models trained by general autograd.
import std.gpu as gpu
import std.json as json
from std.io import read_file, write_file
from std.random import Random
from std.time import Instant
from .data import DataLoader, Dataset
from .layers import Layer
from .ops import add_correct_into, add_scaled_into, argmax, copy_into, softmax, warmup_kernels as warmup_ops
from .optimizers import Optimizer, SGD, reset_state, prepare_state, warmup_kernels as warmup_opt
from .losses import Loss, SparseCategoricalCrossentropy, sparse_ce_with_delta, warmup_kernels as warmup_loss
from .compiled import CompiledStep, compile_step, refusal as compile_refusal
from .plan import ExecutionPlan
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
    #: Wall-clock training duration of each epoch. Fully asynchronous fused
    #: fits report the synchronized total amortized across their epochs.
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
#: Forward, loss, and the optimizer step queue via `dispatch_async`. Each
#: batch is `gpu.flush()`ed so the GPU can run while the next batch is
#: encoded. Metrics stay on device until the epoch ends, so a training
#: epoch pays one fence rather than one per op.
pub class Sequential {
    #: Layers in forward order.
    pub let layers: list[Layer]
    var optimizer_value: Optimizer?
    var loss_value: Loss?
    var input_units: int
    var freed: bool
    var cached_parameters: list[Tensor]?
    var fused_sgd: bool
    var compiled_step: CompiledStep?
    var compiled_for: str
    var fused_mlp_step: bool
    var plan_value: ExecutionPlan?
    var mixed_precision: bool
    var shuffle_batches: bool
    #: How many epochs have been run, which is what makes the next one shuffle
    #: differently from the last.
    var epochs_run: int

    #: Create an uncompiled model from a non-empty layer list.
    pub fn init(self, layers: list[Layer]) {
        if layers.len() == 0 { throw ValueError("Sequential needs at least one layer") }
        self.layers = [layer for layer in layers]
        self.optimizer_value = null
        self.loss_value = null
        self.input_units = 0
        self.freed = false
        self.cached_parameters = null
        self.fused_sgd = false
        self.compiled_step = null
        self.compiled_for = ""
        self.fused_mlp_step = false
        self.plan_value = null
        self.mixed_precision = false
        self.shuffle_batches = false
        self.epochs_run = 0
    }

    #: Configure the optimizer and loss used by `fit`/`evaluate`.
    #:
    #: Recompiling a model that has already been built resets that
    #: optimizer's per-parameter state (momentum, second moment, step
    #: count) — otherwise a later `compile()` with a different optimizer
    #: would silently reinterpret the previous optimizer's state as its
    #: own.
    pub fn compile(
        self,
        optimizer: Optimizer = SGD(),
        loss: Loss = SparseCategoricalCrossentropy(),
        input_units: int = 0,
        batch_size: int = 0,
        mixed_precision: bool = false
    ) -> void {
        self._check_live()
        if not gpu.is_available() { throw RuntimeError("jaitensor needs a GPU") }
        if self.input_units != 0 { reset_state(self._parameters()) }
        self.optimizer_value = optimizer
        self.loss_value = loss
        self.mixed_precision = mixed_precision
        gpu.set_mixed_precision(mixed_precision)
        if input_units > 0 {
            self._build(input_units)
            if batch_size > 0 {
                self._refresh_execution_plan(batch_size)
            } else {
                self.plan_value = null
            }
        }
    }

    #: Return the shape-specialized execution plan, if `compile` traced one.
    pub fn execution_plan(self) -> ExecutionPlan? { return self.plan_value }

    #: Train for `epochs`, returning loss and accuracy histories.
    pub fn fit(
        self,
        dataset: Dataset,
        epochs: int = 1,
        batch_size: int = 2048,
        verbose: bool = true,
        validation_data: Dataset? = null,
        test_data: Dataset? = null,
        plot_path: str? = "data/jaitensor-training.svg",
        shuffle: bool = false
    ) -> History {
        self.shuffle_batches = shuffle
        self._prepare(dataset, epochs, batch_size)
        if not self.fused_sgd { self._ensure_grads() }
        let history = History()

        if self.fused_sgd and not verbose and validation_data is null {
            self._fit_fused_async(dataset, epochs, batch_size, history)
        } else {
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
        self._clear_grad()
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
    #: `batch_size` — chunking would need a device-side slice view this
    #: package doesn't have yet, so the whole input is processed at once.
    #: `batch_size` is still validated for call-site compatibility with
    #: code written against a chunked version. The returned tensor is
    #: detached (see `Tensor.detach`): it never carries a live gradient
    #: graph, since this method already freed that graph's intermediates
    #: before returning.
    pub fn predict(self, features: Tensor, batch_size: int = 2048) -> Tensor {
        self._check_live()
        if batch_size <= 0 { throw ValueError(f"batch_size must be positive, got {batch_size}") }
        if features.rank() != 2 {
            throw ValueError(
                f"predict features must have shape [samples, features], got {features.shape}"
            )
        }
        self._build(features.shape[1])
        let raw = self._forward(features, training: false)
        let intermediates = raw.discard_graph()
        #: Freeing parks the buffer for reuse and carries what is queued
        #: against it along, so the next owner waits rather than this one.
        for tensor in intermediates { tensor.free() }
        raw.detach()
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
        var expected = 0
        for layer in self.layers { expected += layer.parameters().len() }
        if values.len() != expected {
            throw ValueError(f"got {values.len()} weight arrays, expected {expected}")
        }
        var index = 0
        for layer in self.layers {
            for parameter in layer.parameters() {
                parameter.upload(values[index])
                index += 1
            }
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
        if self.input_units == 0 { return 0 }
        var count = 0
        for layer in self.layers {
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
        self.cached_parameters = null
        self.freed = true
    }

    fn _prepare(self, dataset: Dataset, epochs: int, batch_size: int) -> void {
        self._check_live()
        if epochs <= 0 { throw ValueError(f"epochs must be positive, got {epochs}") }
        if batch_size <= 0 { throw ValueError(f"batch_size must be positive, got {batch_size}") }
        let loss = self.loss_value
        if self.optimizer_value is null or loss is null {
            throw RuntimeError("call model.compile() before fit or evaluate")
        }
        self._build_from_dataset(dataset)
        let last = self.layers[self.layers.len() - 1]
        if loss.is_from_logits() and last.wants_softmax_output() {
            throw ValueError("from_logits needs a final layer that does not apply softmax itself")
        }
        if not loss.is_from_logits() and not last.wants_softmax_output() {
            throw ValueError("the final layer must apply softmax, or the loss must set from_logits")
        }
        for index in 0..self.layers.len() - 1 {
            if self.layers[index].wants_softmax_output() {
                throw ValueError("softmax is only supported on the final layer")
            }
        }
        let classes = last.output_width()
        if not gpu.labels_valid(dataset.labels.buffer(), dataset.samples, classes) {
            throw ValueError(f"a label is outside the model's {classes} classes")
        }
        self.fused_sgd = self._can_fuse_sgd()
        self.fused_mlp_step = self._can_fuse_mlp_step()
        self._refresh_execution_plan(batch_size)
        if not self.fused_sgd and not self.fused_mlp_step {
            warmup_ops()
            warmup_loss()
            warmup_opt()
        }
        let optimizer = self.optimizer_value
        if optimizer is not null and not self.fused_sgd {
            prepare_state(self._parameters(), optimizer.settings())
        }
        self._try_compiling(dataset, batch_size, classes)
    }

    #: Build the whole step as one graph when the model allows it.
    #:
    #: Only worth trying for the models the hand-fused paths do not already
    #: cover, and only for plain SGD: momentum carries state from one step to
    #: the next, which would have to live inside the graph too.
    fn _try_compiling(self, dataset: Dataset, batch_size: int, classes: int) -> void {
        if self.fused_sgd or self.fused_mlp_step { return }
        let loss = self.loss_value
        let optimizer = self.optimizer_value
        if loss is null or not loss.is_sparse_categorical() { return }
        if optimizer is null { return }
        let kind = optimizer.kind()
        if kind != 0 and kind != 1 { return }
        let settings = optimizer.settings()
        if kind == 0 and settings.len() < 3 { return }
        if kind == 1 and settings.len() < 4 { return }
        let image = dataset.image_shape
        if image is null or image.len() != 3 { return }
        var shape = [batch_size]
        for dimension in image { shape.push(dimension) }

        #: Kept between calls when nothing about it has changed. Building the
        #: graph is the expensive part -- the platform compiles the whole step,
        #: forward, backward and update together -- and an epoch of a small
        #: dataset is only a few dozen batches to spread that over. Fitting
        #: twice used to pay for it twice.
        let wanted = f"{shape} {classes} {kind} {settings}"
        if self.compiled_step is not null and self.compiled_for == wanted { return }
        let existing = self.compiled_step
        if existing is not null { existing.free() }
        self.compiled_step = if kind == 1 {
            compile_step(self.layers, shape, classes, settings[0], adam: true, beta1: settings[1], beta2: settings[2], epsilon: settings[3])
        } else {
            compile_step(self.layers, shape, classes, settings[0], settings[1], settings[2] > 0.5)
        }
        self.compiled_for = if self.compiled_step is null { "" } else { wanted }
    }

    fn _build_from_dataset(self, dataset: Dataset) -> void {
        let image = dataset.image_shape
        if image is not null {
            self._build_shape(image)
            return
        }
        self._build(dataset.feature_count)
    }

    fn _build_shape(self, shape: list[int]) -> void {
        if shape.len() == 0 { throw ValueError("input shape must not be empty") }
        var n = 1
        for dimension in shape { n *= dimension }
        if self.input_units != 0 {
            if self.input_units != n {
                throw ValueError(f"model expects {self.input_units} features, got {n}")
            }
            return
        }
        let rng = Random.with_seed(7)
        var current = shape
        for layer in self.layers {
            layer.build_from_shape(current, rng)
            current = layer.output_shape()
        }
        self.input_units = n
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

    fn _forward(self, x: Tensor, training: bool = true) -> Tensor {
        var current = x
        for layer in self.layers {
            current = if training { layer.forward(current) } else { layer.eval_forward(current) }
        }
        return current
    }

    #: A loader for one epoch, ordered differently from the epoch before it.
    #:
    #: Each of the three epoch bodies used to build its own with the default
    #: seed, so every epoch of a run drew the same permutation: the batches
    #: were reordered once and then frozen, which is most of the point of
    #: shuffling gone. The loader was not freed either, leaving one order
    #: tensor behind per epoch.
    fn _epoch_loader(self, dataset: Dataset, batch_size: int) -> DataLoader? {
        if not self.shuffle_batches { return null }
        self.epochs_run += 1
        return DataLoader(dataset, batch_size, shuffle: true, seed: self.epochs_run)
    }

    fn _run_epoch(self, dataset: Dataset, batch_size: int, training: bool) -> Metrics {
        let loss_fn = self.loss_value
        let optimizer = self.optimizer_value
        if loss_fn is null or optimizer is null { throw RuntimeError("model is not compiled") }
        if training and self.fused_sgd {
            return self._run_fused_epoch(dataset, batch_size, optimizer.settings()[0])
        }
        if training and self.fused_mlp_step {
            return self._run_fused_bwd_epoch(dataset, batch_size)
        }
        let compiled = self.compiled_step
        if training and compiled is not null and dataset.samples >= batch_size {
            return self._run_compiled_epoch(compiled, dataset, batch_size)
        }
        if self._static_step_active(batch_size) {
            return self._run_static_epoch(dataset, batch_size, training)
        }

        let total_loss = Tensor.zeros([1])
        let total_correct = Tensor.zeros([1])
        defer { total_loss.free() }
        defer { total_correct.free() }
        #: Every batch, not every eighth.
        #:
        #: Holding work back was meant to keep the queue full and it does the
        #: opposite: an unflushed command buffer grows, every intermediate it
        #: names stays alive because nothing has finished with it, and the
        #: encoding cost climbs faster than the saved submissions. Measured on
        #: tests/bench/jaitensor's conv workloads at every level: 2.37s against
        #: 2.42s on conv-small and 3.23s against 3.35s on conv-wide, and 2.7x
        #: against not flushing at all. Never slower at any cadence tried.
        let flush_every = 1
        var start = 0
        var batches = 0
        let loader = self._epoch_loader(dataset, batch_size)
        defer {
            if loader is not null { loader.free() }
        }
        while start < dataset.samples {
            var count = min(batch_size, dataset.samples - start)
            var batch_features: Tensor
            var batch_labels: Tensor
            if loader is not null {
                if not loader.has_next() { break }
                let (features, labels) = loader.next_batch()
                batch_features = features
                batch_labels = labels
                count = batch_features.shape[0]
            } else {
                batch_features = self._slice_features(dataset, start, count)
                batch_labels = self._slice_labels(dataset, start, count)
            }

            let prediction = self._forward(batch_features, training: training)
            let loss = loss_fn.forward(prediction, batch_labels, total_loss, count)
            let intermediates = if training {
                self._zero_grad()
                let graph = loss.backward()
                optimizer.step(self._parameters())
                graph
            } else {
                loss.discard_graph()
            }
            add_correct_into(prediction, batch_labels, total_correct)

            loss.free()
            batch_features.free()
            batch_labels.free()
            for tensor in intermediates { tensor.free() }
            start += count
            batches += 1
            if batches % flush_every == 0 or start >= dataset.samples { gpu.flush() }
        }
        let loss_values = total_loss.download()
        let correct_values = total_correct.download()
        return Metrics(
            loss_values[0] / float(dataset.samples),
            100.0 * correct_values[0] / float(dataset.samples)
        )
    }

    fn _flush_every(self) -> int {
        let widths = self._fused_hidden_widths()
        var widest = 0
        for width in widths {
            if width > widest { widest = width }
        }
        if widest >= 4096 { return 1 }
        if widest >= 1024 { return 2 }
        return 4
    }

    fn _fused_relu_layers(self) -> int {
        if self.layers.len() < 2 { return 0 }
        if not self.layers[self.layers.len() - 1].is_classification_head() { return 0 }
        let hidden = self.layers.len() - 1
        if hidden != 1 and hidden != 3 { return 0 }
        for index in 0..hidden {
            if not self.layers[index].is_relu_dense() { return 0 }
        }
        return hidden
    }

    fn _fused_hidden_widths(self) -> list[int] {
        let hidden = self._fused_relu_layers()
        if hidden == 0 { return [] }
        var widths: list[int] = []
        for index in 0..hidden {
            let width = self.layers[index].output_width()
            if width < 256 { return [] }
            widths.push(width)
        }
        return widths
    }

    fn _fused_uniform_hidden(self) -> int {
        let widths = self._fused_hidden_widths()
        if widths.len() == 0 { return 0 }
        let width = widths[0]
        for index in 1..widths.len() {
            if widths[index] != width { return 0 }
        }
        return width
    }

    fn _can_fuse_sgd(self) -> bool {
        let hidden_layers = self._fused_relu_layers()
        if hidden_layers != 1 and hidden_layers != 3 { return false }
        if self._fused_hidden_widths().len() != hidden_layers { return false }
        let loss = self.loss_value
        if loss is null or not loss.is_sparse_categorical() { return false }
        let optimizer = self.optimizer_value
        if optimizer is null or optimizer.kind() != 0 { return false }
        let settings = optimizer.settings()
        return settings.len() >= 2 and settings[1] == 0.0
    }

    fn _can_fuse_mlp_step(self) -> bool {
        if self._can_fuse_sgd() { return false }
        let hidden_layers = self._fused_relu_layers()
        if hidden_layers != 1 and hidden_layers != 3 { return false }
        if self._fused_hidden_widths().len() != hidden_layers { return false }
        let loss = self.loss_value
        if loss is null or not loss.is_sparse_categorical() { return false }
        let optimizer = self.optimizer_value
        if optimizer is null { return false }
        let kind = optimizer.kind()
        if kind != 0 and kind != 1 { return false }
        return true
    }

    fn _fit_fused_async(
        self,
        dataset: Dataset,
        epochs: int,
        batch_size: int,
        history: History
    ) -> void {
        let optimizer = self.optimizer_value
        if optimizer is null { throw RuntimeError("model is not compiled") }
        let learning_rate = optimizer.settings()[0]
        let total_loss = Tensor.zeros([1])
        let epoch_loss = Tensor.zeros([epochs])
        defer { total_loss.free() }
        defer { epoch_loss.free() }
        var processed = 0
        let started = Instant.now()
        for epoch in 1..=epochs {
            processed = self._fused_train(
                dataset,
                batch_size,
                learning_rate,
                total_loss,
                flush_each_epoch: false
            )
            let loss_slot = epoch_loss.view(epoch - 1, [1])
            copy_into(total_loss, loss_slot)
            loss_slot.free()
            total_loss.zero_()
        }
        let losses = epoch_loss.download()
        let seconds_per_epoch = started.elapsed().to_seconds() / float(epochs)
        for index in 0..epochs {
            history.loss.push(losses[index] / float(processed))
            history.accuracy.push(0.0)
            history.seconds.push(seconds_per_epoch)
        }
    }

    fn _run_fused_epoch(self, dataset: Dataset, batch_size: int, learning_rate: float) -> Metrics {
        let total_loss = Tensor.zeros([1])
        let total_correct = Tensor.zeros([1])
        defer { total_loss.free() }
        defer { total_correct.free() }
        let processed = self._fused_train(
            dataset,
            batch_size,
            learning_rate,
            total_loss,
            total_correct
        )
        let loss_values = total_loss.download()
        let correct_values = total_correct.download()
        return Metrics(
            loss_values[0] / float(processed),
            100.0 * correct_values[0] / float(processed)
        )
    }

    fn _fused_train(
        self,
        dataset: Dataset,
        batch_size: int,
        learning_rate: float,
        total_loss: Tensor,
        total_correct: Tensor? = null,
        flush_each_epoch: bool = false
    ) -> int {
        let hidden_layers = self._fused_relu_layers()
        let head = self.layers[self.layers.len() - 1]
        let inputs = self.input_units
        let widths = self._fused_hidden_widths()
        let classes = head.output_width()
        let parameters = self._parameters()
        let expected = (hidden_layers + 1) * 2
        if parameters.len() != expected {
            throw RuntimeError(
                f"fused SGD needs {hidden_layers + 1} dense layers with weights and bias"
            )
        }
        if widths.len() != hidden_layers {
            throw RuntimeError("fused SGD missing hidden widths")
        }
        let flush_every = if flush_each_epoch { 0x7fffffff } else { self._flush_every() }
        let correct = if total_correct is null { null } else { total_correct.buffer() }
        if dataset.features.rank() == 2 {
            if hidden_layers == 1 {
                return gpu.mlp_sgd_epoch(
                    dataset.features.buffer(),
                    parameters[0].buffer(),
                    parameters[1].buffer(),
                    parameters[2].buffer(),
                    parameters[3].buffer(),
                    dataset.labels.buffer(),
                    total_loss.buffer(),
                    dataset.samples,
                    batch_size,
                    inputs,
                    widths[0],
                    classes,
                    learning_rate,
                    flush_every: flush_every,
                    correct_acc: correct
                )
            }
            return gpu.mlp3_sgd_epoch(
                dataset.features.buffer(),
                parameters[0].buffer(),
                parameters[1].buffer(),
                parameters[2].buffer(),
                parameters[3].buffer(),
                parameters[4].buffer(),
                parameters[5].buffer(),
                parameters[6].buffer(),
                parameters[7].buffer(),
                dataset.labels.buffer(),
                total_loss.buffer(),
                dataset.samples,
                batch_size,
                inputs,
                widths[0],
                classes,
                learning_rate,
                flush_every: flush_every,
                correct_acc: correct,
                hidden2: widths[1],
                hidden3: widths[2]
            )
        }
        var start = 0
        var batches = 0
        var processed = 0
        while start < dataset.samples {
            let count = min(batch_size, dataset.samples - start)
            if count != batch_size and processed > 0 { break }
            let batch_features = self._slice_features(dataset, start, count)
            let batch_labels = self._slice_labels(dataset, start, count)
            let correct = if total_correct is null { null } else { total_correct.buffer() }
            if hidden_layers == 1 {
                gpu.mlp_sgd_step(
                    batch_features.buffer(),
                    parameters[0].buffer(),
                    parameters[1].buffer(),
                    parameters[2].buffer(),
                    parameters[3].buffer(),
                    batch_labels.buffer(),
                    total_loss.buffer(),
                    count,
                    inputs,
                    widths[0],
                    classes,
                    learning_rate,
                    correct_acc: correct
                )
            } else {
                gpu.mlp3_sgd_step(
                    batch_features.buffer(),
                    parameters[0].buffer(),
                    parameters[1].buffer(),
                    parameters[2].buffer(),
                    parameters[3].buffer(),
                    parameters[4].buffer(),
                    parameters[5].buffer(),
                    parameters[6].buffer(),
                    parameters[7].buffer(),
                    batch_labels.buffer(),
                    total_loss.buffer(),
                    count,
                    inputs,
                    widths[0],
                    classes,
                    learning_rate,
                    correct_acc: correct,
                    hidden2: widths[1],
                    hidden3: widths[2]
                )
            }
            batch_features.free()
            batch_labels.free()
            start += count
            processed += count
            batches += 1
            if flush_each_epoch {
                if start >= dataset.samples { gpu.flush() }
            } elif batches % flush_every == 0 or start >= dataset.samples {
                gpu.flush()
            }
        }
        return processed
    }

    fn _run_fused_bwd_epoch(self, dataset: Dataset, batch_size: int) -> Metrics {
        let optimizer = self.optimizer_value
        if optimizer is null { throw RuntimeError("model is not compiled") }
        let total_loss = Tensor.zeros([1])
        let total_correct = Tensor.zeros([1])
        defer { total_loss.free() }
        defer { total_correct.free() }
        let processed = self._fused_bwd_train(
            dataset,
            batch_size,
            optimizer,
            total_loss,
            total_correct
        )
        let loss_values = total_loss.download()
        let correct_values = total_correct.download()
        return Metrics(
            loss_values[0] / float(processed),
            100.0 * correct_values[0] / float(processed)
        )
    }

    fn _fused_bwd_train(
        self,
        dataset: Dataset,
        batch_size: int,
        optimizer: Optimizer,
        total_loss: Tensor,
        total_correct: Tensor? = null
    ) -> int {
        let hidden_layers = self._fused_relu_layers()
        let head = self.layers[self.layers.len() - 1]
        let inputs = self.input_units
        let widths = self._fused_hidden_widths()
        let classes = head.output_width()
        let parameters = self._parameters()
        let expected = (hidden_layers + 1) * 2
        if parameters.len() != expected {
            throw RuntimeError(
                f"fused MLP step needs {hidden_layers + 1} dense layers with weights and bias"
            )
        }
        if widths.len() != hidden_layers {
            throw RuntimeError("fused MLP step missing hidden widths")
        }
        let flush_every = self._flush_every()
        var start = 0
        var batches = 0
        var processed = 0
        while start < dataset.samples {
            let count = min(batch_size, dataset.samples - start)
            if count != batch_size and processed > 0 { break }
            let batch_features = self._slice_features(dataset, start, count)
            let batch_labels = self._slice_labels(dataset, start, count)
            let correct = if total_correct is null { null } else { total_correct.buffer() }
            if hidden_layers == 1 {
                gpu.mlp_bwd_step(
                    batch_features.buffer(),
                    parameters[0].buffer(),
                    parameters[1].buffer(),
                    parameters[2].buffer(),
                    parameters[3].buffer(),
                    batch_labels.buffer(),
                    parameters[0].grad.buffer(),
                    parameters[1].grad.buffer(),
                    parameters[2].grad.buffer(),
                    parameters[3].grad.buffer(),
                    total_loss.buffer(),
                    count,
                    inputs,
                    widths[0],
                    classes,
                    correct_acc: correct
                )
            } else {
                gpu.mlp3_bwd_step(
                    batch_features.buffer(),
                    parameters[0].buffer(),
                    parameters[1].buffer(),
                    parameters[2].buffer(),
                    parameters[3].buffer(),
                    parameters[4].buffer(),
                    parameters[5].buffer(),
                    parameters[6].buffer(),
                    parameters[7].buffer(),
                    batch_labels.buffer(),
                    parameters[0].grad.buffer(),
                    parameters[1].grad.buffer(),
                    parameters[2].grad.buffer(),
                    parameters[3].grad.buffer(),
                    parameters[4].grad.buffer(),
                    parameters[5].grad.buffer(),
                    parameters[6].grad.buffer(),
                    parameters[7].grad.buffer(),
                    total_loss.buffer(),
                    count,
                    inputs,
                    widths[0],
                    classes,
                    correct_acc: correct,
                    hidden2: widths[1],
                    hidden3: widths[2]
                )
            }
            optimizer.step(parameters)
            batch_features.free()
            batch_labels.free()
            start += count
            processed += count
            batches += 1
            if batches % flush_every == 0 or start >= dataset.samples { gpu.flush() }
        }
        return processed
    }

    #: The shape one batch of `dataset` arrives in.
    fn _batch_shape(self, dataset: Dataset, count: int) -> list[int] {
        if dataset.features.rank() == 4 {
            let shape = dataset.image_shape
            if shape is null { throw RuntimeError("NHWC dataset missing image_shape") }
            return [count, shape[0], shape[1], shape[2]]
        }
        return [count, dataset.feature_count]
    }

    fn _slice_features(self, dataset: Dataset, start: int, count: int) -> Tensor {
        let width = dataset.feature_count
        if dataset.features.rank() == 4 {
            let shape = dataset.image_shape
            if shape is null { throw RuntimeError("NHWC dataset missing image_shape") }
            return dataset.features.view(start * width, [count, shape[0], shape[1], shape[2]])
        }
        return dataset.features.view(start * width, [count, width])
    }

    fn _slice_labels(self, dataset: Dataset, start: int, count: int) -> Tensor {
        return dataset.labels.view(start, [count])
    }

    fn _parameters(self) -> list[Tensor] {
        let cached = self.cached_parameters
        if cached is not null { return cached }
        var parameters: list[Tensor] = []
        for layer in self.layers { parameters.extend(layer.parameters()) }
        self.cached_parameters = parameters
        return parameters
    }

    fn _ensure_grads(self) -> void {
        for parameter in self._parameters() {
            if parameter.grad is null {
                let gradient = Tensor(parameter.shape)
                gradient.zero_()
                parameter.grad = gradient
            }
        }
    }

    fn _zero_grad(self) -> void {
        for parameter in self._parameters() {
            let gradient = parameter.grad
            if gradient is not null { gradient.zero_() }
        }
    }

    fn _refresh_execution_plan(self, batch_size: int) -> void {
        if self.input_units == 0 { return }
        let loss = self.loss_value
        var static_step = loss is not null and
            loss.is_sparse_categorical() and
            not loss.is_from_logits()
        if static_step {
            for layer in self.layers {
                if not layer.supports_static() {
                    static_step = false
                    break
                }
            }
        }
        var layer_widths: list[int] = []
        for layer in self.layers { layer_widths.push(layer.output_width()) }
        self.plan_value = ExecutionPlan(self.input_units, batch_size, layer_widths, static_step)
    }

    fn _static_step_active(self, batch_size: int) -> bool {
        let _ = batch_size
        if self.fused_sgd or self.fused_mlp_step { return false }
        let loss = self.loss_value
        if loss is null or not loss.is_sparse_categorical() or loss.is_from_logits() {
            return false
        }
        for layer in self.layers {
            if not layer.supports_static() { return false }
        }
        return true
    }

    @trace
    fn _static_forward(self, x: Tensor) -> list[Tensor] {
        var activations: list[Tensor] = []
        var current = x
        for layer in self.layers {
            current = layer.static_forward(current)
            activations.push(current)
        }
        return activations
    }

    @trace
    fn _static_backward(
        self,
        batch_features: Tensor,
        activations: list[Tensor],
        delta: Tensor
    ) -> void {
        var upstream = delta
        var index = self.layers.len() - 1
        while index >= 0 {
            let layer = self.layers[index]
            let input = if index == 0 { batch_features } else { activations[index - 1] }
            let output = activations[index]
            let d_input = layer.static_backward(input, output, upstream)
            if upstream is not delta { upstream.free() }
            upstream = d_input
            index -= 1
        }
        delta.free()
        upstream.free()
    }

    @trace
    fn _static_train_batch(
        self,
        batch_features: Tensor,
        batch_labels: Tensor,
        total_loss: Tensor,
        count: int
    ) -> list[Tensor] {
        let optimizer = self.optimizer_value
        if optimizer is null { throw RuntimeError("model is not compiled") }
        let activations = self._static_forward(batch_features)
        let prediction = activations[activations.len() - 1]
        self._zero_grad()
        let (loss, delta) = sparse_ce_with_delta(prediction, batch_labels, total_loss, count)
        self._static_backward(batch_features, activations, delta)
        optimizer.step(self._parameters())
        loss.free()
        return activations
    }

    #: One epoch through the compiled step.
    #:
    #: The plan is built for one batch size, so a short last batch is left to
    #: the ordinary path rather than run against a graph that does not fit it.
    fn _run_compiled_epoch(self, step: CompiledStep, dataset: Dataset, batch_size: int) -> Metrics {
        let total_correct = Tensor.zeros([1])
        defer { total_correct.free() }
        let scores = Tensor([batch_size, step.classes])
        defer { scores.free() }
        let batch_loss = Tensor.zeros([1])
        defer { batch_loss.free() }
        let total_loss = Tensor.zeros([1])
        defer { total_loss.free() }
        #: The graph reads a whole buffer, so a batch has to be one. Handing it
        #: a slice of the dataset would feed whatever the slice was cut from.
        let staged_features = Tensor(self._batch_shape(dataset, batch_size))
        defer { staged_features.free() }
        let staged_labels = Tensor([batch_size])
        defer { staged_labels.free() }
        var seen = 0
        var start = 0
        let loader = self._epoch_loader(dataset, batch_size)
        defer {
            if loader is not null { loader.free() }
        }
        while start < dataset.samples {
            var count = min(batch_size, dataset.samples - start)
            var batch_features: Tensor
            var batch_labels: Tensor
            if loader is not null {
                if not loader.has_next() { break }
                let (features, labels) = loader.next_batch()
                batch_features = features
                batch_labels = labels
                count = batch_features.shape[0]
            } else {
                batch_features = self._slice_features(dataset, start, count)
                batch_labels = self._slice_labels(dataset, start, count)
            }
            if count != batch_size {
                batch_features.free()
                batch_labels.free()
                start += count
                continue
            }
            copy_into(batch_features, staged_features)
            copy_into(batch_labels, staged_labels)
            step.step_into(staged_features, staged_labels, scores, batch_loss)
            #: Kept on the device and summed there. Reading the loss here would
            #: wait for the step that produced it, every batch.
            add_scaled_into(batch_loss, total_loss, count)
            seen += count
            add_correct_into(scores, staged_labels, total_correct)
            batch_features.free()
            batch_labels.free()
            start += count
            gpu.flush()
        }
        step.settle()
        let loss_values = total_loss.download()
        let correct_values = total_correct.download()
        let counted = if seen > 0 { float(seen) } else { 1.0 }
        return Metrics(loss_values[0] / counted, 100.0 * correct_values[0] / counted)
    }

    fn _run_static_epoch(self, dataset: Dataset, batch_size: int, training: bool) -> Metrics {
        let loss_fn = self.loss_value
        let optimizer = self.optimizer_value
        if loss_fn is null or optimizer is null { throw RuntimeError("model is not compiled") }

        let total_loss = Tensor.zeros([1])
        let total_correct = Tensor.zeros([1])
        defer { total_loss.free() }
        defer { total_correct.free() }
        let flush_every = 8
        var start = 0
        var batches = 0
        let loader = self._epoch_loader(dataset, batch_size)
        defer {
            if loader is not null { loader.free() }
        }
        while start < dataset.samples {
            var count = min(batch_size, dataset.samples - start)
            var batch_features: Tensor
            var batch_labels: Tensor
            if loader is not null {
                if not loader.has_next() { break }
                let (features, labels) = loader.next_batch()
                batch_features = features
                batch_labels = labels
                count = batch_features.shape[0]
            } else {
                batch_features = self._slice_features(dataset, start, count)
                batch_labels = self._slice_labels(dataset, start, count)
            }
            let activations = if training {
                self._static_train_batch(batch_features, batch_labels, total_loss, count)
            } else {
                self._static_forward(batch_features)
            }
            let prediction = activations[activations.len() - 1]

            if not training {
                let loss = loss_fn.forward(prediction, batch_labels, total_loss, count)
                let graph = loss.discard_graph()
                loss.free()
                for tensor in graph { tensor.free() }
            }
            add_correct_into(prediction, batch_labels, total_correct)

            batch_features.free()
            batch_labels.free()
            for activation in activations { activation.free() }
            start += count
            batches += 1
            if batches % flush_every == 0 or start >= dataset.samples { gpu.flush() }
        }
        let loss_values = total_loss.download()
        let correct_values = total_correct.download()
        return Metrics(
            loss_values[0] / float(dataset.samples),
            100.0 * correct_values[0] / float(dataset.samples)
        )
    }

    fn _clear_grad(self) -> void {
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

#: The pieces `train.jai` and `check.jai` share: the CIFAR-10 loader, the
#: ResNet itself, and one training run with its schedule.
#:
#: The training loop is written out rather than left to `Sequential.fit` so
#: that the learning rate can follow a one-cycle schedule step by step: it is
#: the loop `fit` runs on its autograd route (forward, loss, backward, step),
#: with `optimizer.learning_rate` assigned before each step.
import std.gpu as gpu
from jaitensor import Activate, Activation, AvgPool2d, BatchNorm2d, Conv2d, DataLoader, Dataset, Dense
from jaitensor import Flatten, Layer, ResBlock, SGD, Sequential, SparseCategoricalCrossentropy, Tensor
from jaitensor import add_correct_into, nhwc, one_cycle_lr, random_crop_flip_into, zero_grad
from std.io import Path, read_bytes
from std.random import Random
from std.time import Instant

pub const SIDE = 32
pub const CHANNELS = 3
pub const CLASSES = 10
const PIXELS = SIDE * SIDE * CHANNELS

#: One split of CIFAR-10 from `data/cifar-10/{split}-images.bin` and
#: `-labels.bin`: planar uint8 images, scaled to [0, 1] on upload and turned
#: NHWC once on the device.
pub fn load_cifar(split: str, samples: int = 0) -> Dataset {
    let root = Path("data/cifar-10")
    let images = read_bytes(root.join(f"{split}-images.bin").to_str())
    let labels = read_bytes(root.join(f"{split}-labels.bin").to_str())
    let count = if samples > 0 { samples } else { labels.len() }
    if images.len() < count * PIXELS or labels.len() < count {
        throw ValueError(f"data/cifar-10 {split} split holds fewer than {count} images")
    }
    let planar = Tensor([count, CHANNELS, SIDE, SIDE])
    planar.buffer().upload_u8(images, 0, count * PIXELS, 0, 1.0 / 255.0)
    let features = nhwc(planar)
    planar.free()
    return Dataset(features, Tensor.from_u8(labels, [count]))
}

#: ResNet for CIFAR (He et al. 2016, section 4.2): a 3x3 stem, three stages
#: of `(depth - 2) / 6` basic blocks at `width`, `2 * width` and `4 * width`
#: channels, the second and third halving the resolution, then global
#: average pooling and a linear classifier. Depth 20 at width 16 is ResNet-20
#: (0.27M parameters); depth 8 is the three-block ResNet-8.
pub fn resnet_layers(depth: int, width: int) -> list[Layer] {
    if depth < 8 or (depth - 2) % 6 != 0 { throw ValueError(f"depth must be 6n + 2, got {depth}") }
    let blocks = (depth - 2) // 6
    var layers: list[Layer] = [
        Conv2d(width, 3, padding: 1, in_channels: CHANNELS),
        BatchNorm2d(),
        Activate(Activation.Relu),
    ]
    for stage in 0..3 {
        for block in 0..blocks {
            let stride = if stage > 0 and block == 0 { 2 } else { 1 }
            layers.push(ResBlock(width << stage, stride: stride))
        }
    }
    layers.push(AvgPool2d(SIDE // 4))
    layers.push(Flatten())
    layers.push(Dense(CLASSES, activation: Activation.Softmax))
    return layers
}

#: Build every layer for a CIFAR image with weights drawn from `seed`.
#:
#: `Sequential` builds with a fixed seed of its own; layers built here first
#: keep these weights when it builds them again, so each seed is its own run.
pub fn build_layers(layers: list[Layer], seed: int) -> void {
    let rng = Random.with_seed(seed * 7919 + 7)
    var shape = [SIDE, SIDE, CHANNELS]
    for layer in layers {
        layer.build_from_shape(shape, rng)
        shape = layer.output_shape()
    }
}

#: What one run is asked to do.
pub class Recipe {
    pub var depth: int
    pub var width: int
    pub var epochs: int
    pub var batch: int
    pub var max_lr: float
    pub var pct_start: float
    pub var momentum: float
    pub var weight_decay: float
    pub var padding: int
    pub var seed: int
    pub var verbose: bool

    pub fn init(self) {
        self.depth = 20
        self.width = 16
        self.epochs = 30
        self.batch = 128
        self.max_lr = 0.2
        self.pct_start = 0.25
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.padding = 4
        self.seed = 0
        self.verbose = true
    }

    pub fn describe(self) -> str {
        return f"ResNet-{self.depth} width {self.width}, {self.epochs} epochs, batch {self.batch}, " +
            f"SGD nesterov momentum {self.momentum} wd {self.weight_decay}, one-cycle max lr {self.max_lr} " +
            f"pct_start {self.pct_start}, crop pad {self.padding} + flip, seed {self.seed}"
    }
}

#: What one run measured.
pub class Outcome {
    pub var epoch_seconds: list[float]
    pub var epoch_loss: list[float]
    pub var epoch_accuracy: list[float]
    pub var augment_ms: list[float]
    pub var train_seconds: float
    pub var test_accuracy: float
    pub var test_loss: float
    pub var parameters: int

    pub fn init(self) {
        self.epoch_seconds = []
        self.epoch_loss = []
        self.epoch_accuracy = []
        self.augment_ms = []
        self.train_seconds = 0.0
        self.test_accuracy = 0.0
        self.test_loss = 0.0
        self.parameters = 0
    }

    #: The median epoch time, leaving out the first (kernel compiles, plan
    #: building and the clock ramp land there).
    pub fn steady_epoch(self) -> float {
        if self.epoch_seconds.len() < 2 { return self.epoch_seconds[0] }
        var rest = [self.epoch_seconds[index] for index in 1..self.epoch_seconds.len()]
        rest.sort()
        return rest[rest.len() // 2]
    }
}

#: Train one model on `train` by `recipe` and evaluate it on `test`.
#:
#: Each epoch draws a fresh crop and flip for every image on the GPU, then a
#: shuffled pass in batches of `recipe.batch`, keeping the short last batch.
#: The test set is evaluated with batch statistics at batch 1000 -- see the
#: README for why.
pub fn train_resnet(recipe: Recipe, train: Dataset, test: Dataset) -> Outcome {
    let layers = resnet_layers(recipe.depth, recipe.width)
    build_layers(layers, recipe.seed)
    let optimizer = SGD(
        learning_rate: recipe.max_lr,
        momentum: recipe.momentum,
        nesterov: true,
        weight_decay: recipe.weight_decay
    )
    let loss_fn = SparseCategoricalCrossentropy()
    let model = Sequential(layers)
    model.compile(optimizer: optimizer, loss: loss_fn)
    var parameters: list[Tensor] = []
    for layer in layers { parameters.extend(layer.parameters()) }
    let outcome = Outcome()
    for parameter in parameters { outcome.parameters += parameter.size }

    let augmented = Tensor(train.features.shape)
    let epoch_data = Dataset(augmented, train.labels)
    let steps_per_epoch = (train.samples + recipe.batch - 1) // recipe.batch
    let total_steps = steps_per_epoch * recipe.epochs
    let total_loss = Tensor.zeros([1])
    let total_correct = Tensor.zeros([1])
    var step = 0
    let run_started = Instant.now()
    for epoch in 0..recipe.epochs {
        let started = Instant.now()
        random_crop_flip_into(train.features, augmented, padding: recipe.padding, seed: recipe.seed * 1000003 + epoch)
        if recipe.verbose {
            gpu.synchronize()
            outcome.augment_ms.push(started.elapsed().to_seconds() * 1000.0)
        }
        total_loss.zero_()
        total_correct.zero_()
        let loader = DataLoader(epoch_data, recipe.batch, shuffle: true, seed: recipe.seed * 7919 + epoch + 1)
        while loader.has_next() {
            let (features, labels) = loader.next_batch()
            let count = features.shape[0]
            optimizer.learning_rate = one_cycle_lr(step, total_steps, recipe.max_lr, pct_start: recipe.pct_start)
            var current = features
            for layer in layers { current = layer.forward(current) }
            let loss = loss_fn.forward(current, labels, total_loss, count)
            zero_grad(parameters)
            let graph = loss.backward()
            optimizer.step(parameters)
            add_correct_into(current, labels, total_correct)
            loss.free()
            features.free()
            labels.free()
            for held in graph { held.free() }
            gpu.flush()
            step += 1
        }
        loader.free()
        let summed = total_loss.download()[0]
        let correct = total_correct.download()[0]
        let seconds = started.elapsed().to_seconds()
        outcome.epoch_seconds.push(seconds)
        outcome.epoch_loss.push(summed / float(train.samples))
        outcome.epoch_accuracy.push(100.0 * correct / float(train.samples))
        if recipe.verbose {
            let aug = outcome.augment_ms[outcome.augment_ms.len() - 1]
            print(f"epoch {epoch + 1}/{recipe.epochs}  {seconds:.3f}s  aug {aug:.1f}ms  loss {summed / float(train.samples):.4f}  train {100.0 * correct / float(train.samples):.2f}%  lr {optimizer.learning_rate:.5f}")
        }
    }
    outcome.train_seconds = run_started.elapsed().to_seconds()
    total_loss.free()
    total_correct.free()
    augmented.free()
    let metrics = model.evaluate(test, batch_size: 1000)
    outcome.test_accuracy = metrics.accuracy
    outcome.test_loss = metrics.loss
    model.free()
    return outcome
}

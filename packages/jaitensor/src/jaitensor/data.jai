#: Generic datasets kept resident on the GPU.
from std.random import Random
from .ops import gather_rows
from .tensor import Tensor

#: A feature matrix and sparse integer labels on the GPU.
#:
#: `Dataset` has no knowledge of file formats or named datasets. Load or
#: generate values in application code, then use one of the constructors here
#: to move them to device storage.
pub class Dataset {
    #: Feature matrix with shape `[samples, feature_count]`.
    pub let features: Tensor
    #: Sparse class labels with shape `[samples]`.
    pub let labels: Tensor
    #: Number of examples.
    pub let samples: int
    #: Number of input values per example.
    pub let feature_count: int
    #: Spatial `[H, W, C]` when `features` is NHWC, otherwise `null`.
    pub let image_shape: list[int]?
    var freed: bool

    #: Pair a feature matrix with one label per row.
    pub fn init(self, features: Tensor, labels: Tensor) {
        if features.shape.len() == 2 {
            self.feature_count = features.shape[1]
            self.image_shape = null
        } elif features.shape.len() == 4 {
            self.feature_count = features.shape[1] * features.shape[2] * features.shape[3]
            self.image_shape = [features.shape[1], features.shape[2], features.shape[3]]
        } else {
            throw ValueError(
                f"features must have shape [samples, features] or NHWC, got {features.shape}"
            )
        }
        if labels.shape.len() != 1 or labels.shape[0] != features.shape[0] {
            throw ValueError(
                f"labels shape {labels.shape} does not match features {features.shape}"
            )
        }
        self.features = features
        self.labels = labels
        self.samples = features.shape[0]
        self.freed = false
    }

    #: Build a dataset from a flat row-major feature list and integer labels.
    pub static fn from_flat(
        features: list[float],
        labels: list[int],
        feature_count: int
    ) -> Dataset {
        if feature_count <= 0 {
            throw ValueError(f"feature_count must be positive, got {feature_count}")
        }
        if labels.len() == 0 { throw ValueError("a dataset needs at least one sample") }
        if features.len() != labels.len() * feature_count {
            throw ValueError(
                f"{features.len()} feature values cannot form {labels.len()} rows " +
                    f"of width {feature_count}"
            )
        }
        return Dataset(
            Tensor([labels.len(), feature_count], features),
            Tensor([labels.len()], [float(label) for label in labels])
        )
    }

    #: Build a dataset from feature rows and integer labels.
    pub static fn from_rows(features: list[list[float]], labels: list[int]) -> Dataset {
        if features.len() == 0 { throw ValueError("a dataset needs at least one sample") }
        if features.len() != labels.len() {
            throw ValueError(f"got {features.len()} feature rows and {labels.len()} labels")
        }
        let width = features[0].len()
        if width == 0 { throw ValueError("feature rows must not be empty") }
        var flat: list[float] = []
        for (index, row) in features.enumerate() {
            if row.len() != width {
                throw ValueError(f"feature row {index} has width {row.len()}, expected {width}")
            }
            flat.extend(row)
        }
        return Dataset.from_flat(flat, labels, width)
    }

    #: TensorFlow-style name for constructing a dataset from host rows.
    pub static fn from_tensor_slices(features: list[list[float]], labels: list[int]) -> Dataset {
        return Dataset.from_rows(features, labels)
    }

    #: Expand byte features and byte labels directly into device tensors.
    #:
    #: Offsets allow callers to skip their own headers. `feature_scale` is
    #: applied during upload, which avoids a large intermediate float list.
    pub static fn from_u8(
        feature_data: bytes,
        label_data: bytes,
        samples: int,
        feature_count: int,
        feature_offset: int = 0,
        label_offset: int = 0,
        feature_scale: float = 1.0
    ) -> Dataset {
        if samples <= 0 { throw ValueError(f"samples must be positive, got {samples}") }
        if feature_count <= 0 {
            throw ValueError(f"feature_count must be positive, got {feature_count}")
        }
        if feature_offset < 0 or feature_offset + samples * feature_count > feature_data.len() {
            throw ValueError("feature bytes do not contain the requested matrix")
        }
        if label_offset < 0 or label_offset + samples > label_data.len() {
            throw ValueError("label bytes do not contain the requested samples")
        }
        return Dataset(
            Tensor.from_u8(
                feature_data,
                [samples, feature_count],
                source_offset: feature_offset,
                scale: feature_scale
            ),
            Tensor.from_u8(label_data, [samples], source_offset: label_offset)
        )
    }

    #: Number of minibatches needed to visit every sample once.
    pub fn batch_count(self, batch_size: int) -> int {
        if batch_size <= 0 { throw ValueError(f"batch_size must be positive, got {batch_size}") }
        return (self.samples + batch_size - 1) // batch_size
    }

    #: Release the feature and label buffers. Freeing twice is harmless.
    pub fn free(self) -> void {
        if self.freed { return }
        self.features.free()
        self.labels.free()
        self.freed = true
    }
}

#: Minibatch iterator that can shuffle and gather the next batch on the GPU
#: while the current step is still running.
pub class DataLoader {
    pub let dataset: Dataset
    pub let batch_size: int
    pub let shuffle: bool
    var order: Tensor?
    var cursor: int
    var rng: Random

    pub fn init(self, dataset: Dataset, batch_size: int, shuffle: bool = false, seed: int = 0) {
        if batch_size <= 0 { throw ValueError(f"batch_size must be positive, got {batch_size}") }
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.order = null
        self.cursor = 0
        self.rng = Random.with_seed(seed)
        self.reset()
    }

    #: Rebuild the sample order for a new epoch.
    pub fn reset(self) -> void {
        self.cursor = 0
        let previous = self.order
        if previous is not null { previous.free() }
        if not self.shuffle {
            self.order = null
            return
        }
        var perm: list[float] = []
        for index in 0..self.dataset.samples { perm.push(float(index)) }
        var i = perm.len() - 1
        while i > 0 {
            let j = int(self.rng.next_float() * float(i + 1))
            let bound = if j < 0 { 0 } elif j > i { i } else { j }
            let tmp = perm[i]
            perm[i] = perm[bound]
            perm[bound] = tmp
            i -= 1
        }
        self.order = Tensor([perm.len()], perm)
    }

    #: Whether another minibatch remains in this epoch.
    pub fn has_next(self) -> bool { return self.cursor < self.dataset.samples }

    #: Gather the next minibatch onto new device tensors. Caller frees them.
    pub fn next_batch(self) -> tuple[Tensor, Tensor] {
        if not self.has_next() { throw ValueError("DataLoader is exhausted; call reset()") }
        let count = min(self.batch_size, self.dataset.samples - self.cursor)
        let start = self.cursor
        self.cursor += count
        let order = self.order
        if order is null {
            let width = self.dataset.feature_count
            if self.dataset.features.rank() == 4 {
                let shape = self.dataset.image_shape
                if shape is null { throw RuntimeError("NHWC dataset missing image_shape") }
                return (
                    self.dataset.features.view(
                        start * width,
                        [count, shape[0], shape[1], shape[2]]
                    ),
                    self.dataset.labels.view(start, [count])
                )
            }
            return (
                self.dataset.features.view(start * width, [count, width]),
                self.dataset.labels.view(start, [count])
            )
        }
        let indices = order.view(start, [count])
        let rows = self.dataset.features
        var source = rows
        var wanted: list[int]? = null
        if rows.rank() == 4 {
            let shape = self.dataset.image_shape
            if shape is null { throw RuntimeError("NHWC dataset missing image_shape") }
            source = Tensor([rows.shape[0], self.dataset.feature_count], storage: rows.buffer())
            wanted = [count, shape[0], shape[1], shape[2]]
        }
        let gathered = gather_rows(source, indices, wanted)
        #: The labels have to move with the rows they belong to. Taking them in
        #: sample order instead pairs every image with somebody else's answer.
        let column = self.dataset.labels.view(0, [self.dataset.samples, 1])
        let picked = gather_rows(column, indices, [count])
        column.free()
        return (gathered, picked)
    }
}

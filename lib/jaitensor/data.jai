#: Generic datasets kept resident on the GPU.
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
    var freed: bool

    #: Pair a feature matrix with one label per row.
    pub fn init(self, features: Tensor, labels: Tensor) {
        if features.shape.len() != 2 {
            throw ValueError(f"features must have shape [samples, features], got {features.shape}")
        }
        if labels.shape.len() != 1 or labels.shape[0] != features.shape[0] {
            throw ValueError(
                f"labels shape {labels.shape} does not match features {features.shape}"
            )
        }
        self.features = features
        self.labels = labels
        self.samples = features.shape[0]
        self.feature_count = features.shape[1]
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

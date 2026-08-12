#: GPU-resident tensors.
import std.gpu as gpu
from std.gpu import Buffer

fn element_count(shape: list[int]) -> int {
    if shape.len() == 0 { throw ValueError("a tensor needs at least one dimension") }
    var size = 1
    for dimension in shape {
        if dimension <= 0 { throw ValueError(f"tensor dimensions must be positive, got {shape}") }
        size *= dimension
    }
    return size
}

#: A row-major float32 tensor stored on the GPU.
#:
#: Device memory has explicit ownership. Call `free` when a standalone tensor
#: is no longer needed; models and datasets free the tensors they own.
pub class Tensor {
    #: Dimensions in row-major order.
    pub let shape: list[int]
    #: Number of float32 values in the tensor.
    pub let size: int
    let storage: Buffer
    var freed: bool

    #: Allocate a tensor, optionally initialized from host values.
    pub fn init(self, shape: list[int], values: list[float]? = null) {
        self.shape = [dimension for dimension in shape]
        self.size = element_count(shape)
        self.storage = gpu.Buffer(self.size)
        self.freed = false
        if values is not null {
            if values.len() != self.size {
                self.storage.free()
                self.freed = true
                throw ValueError(f"tensor has {values.len()} values, expected {self.size}")
            }
            self.storage.upload(values)
        }
    }

    #: Allocate a zero-filled tensor.
    pub static fn zeros(shape: list[int]) -> Tensor {
        let size = element_count(shape)
        return Tensor(shape, [0.0 for _index in 0..size])
    }

    #: Allocate a tensor filled with ones.
    pub static fn ones(shape: list[int]) -> Tensor {
        let size = element_count(shape)
        return Tensor(shape, [1.0 for _index in 0..size])
    }

    #: Allocate a tensor with every element set to `value`.
    pub static fn full(shape: list[int], value: float) -> Tensor {
        let size = element_count(shape)
        return Tensor(shape, [value for _index in 0..size])
    }

    #: Expand unsigned bytes directly into a GPU tensor.
    pub static fn from_u8(
        data: bytes,
        shape: list[int],
        source_offset: int = 0,
        scale: float = 1.0
    ) -> Tensor {
        let tensor = Tensor(shape)
        tensor.storage.upload_u8(data, source_offset, tensor.size, scale: scale)
        return tensor
    }

    #: The underlying device buffer, for kernels and advanced users.
    pub fn buffer(self) -> Buffer {
        self._check_live()
        return self.storage
    }

    #: Replace all tensor values from the host.
    pub fn upload(self, values: list[float]) -> void {
        self._check_live()
        if values.len() != self.size {
            throw ValueError(f"tensor upload has {values.len()} values, expected {self.size}")
        }
        self.storage.upload(values)
    }

    #: Copy all values to the host.
    pub fn download(self) -> list[float] {
        self._check_live()
        gpu.synchronize()
        return self.storage.download()
    }

    #: Number of dimensions in the tensor.
    pub fn rank(self) -> int { return self.shape.len() }

    #: Size of one dimension, accepting negative indices from the end.
    pub fn dim(self, axis: int) -> int {
        let resolved = if axis < 0 { self.shape.len() + axis } else { axis }
        if resolved < 0 or resolved >= self.shape.len() {
            throw IndexError(f"axis {axis} is outside a rank-{self.shape.len()} tensor")
        }
        return self.shape[resolved]
    }

    #: Whether this tensor and `other` have identical dimensions.
    pub fn same_shape(self, other: Tensor) -> bool { return self.shape == other.shape }

    #: Release this tensor's device memory. Freeing twice is harmless.
    pub fn free(self) -> void {
        if self.freed { return }
        gpu.synchronize()
        self.storage.free()
        self.freed = true
    }

    #: Whether `free` has released the device storage.
    pub fn is_freed(self) -> bool { return self.freed }

    fn _check_live(self) -> void {
        if self.freed { throw RuntimeError("this Tensor has already been freed") }
    }

    fn __str__(self) -> str { return f"Tensor(shape={self.shape}, device={gpu.device_name()})" }
}

export { element_count }

#: Construct a constant GPU tensor from flat row-major values.
pub fn constant(values: list[float], shape: list[int]) -> Tensor { return Tensor(shape, values) }

#: Allocate a zero-filled GPU tensor.
pub fn zeros(shape: list[int]) -> Tensor { return Tensor.zeros(shape) }

#: Allocate a one-filled GPU tensor.
pub fn ones(shape: list[int]) -> Tensor { return Tensor.ones(shape) }

#: Allocate a GPU tensor filled with `value`.
pub fn fill(shape: list[int], value: float) -> Tensor { return Tensor.full(shape, value) }

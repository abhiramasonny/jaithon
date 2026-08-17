#: GPU-resident tensors.
import std.gpu as gpu
from std.gpu import Buffer, Kernel

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
    #: Whether this tensor should accumulate a gradient when `backward()`
    #: reaches it. Set directly on a leaf, or by `attach_creator` when an op
    #: records how to differentiate a value built from one.
    pub var requires_grad: bool
    #: The gradient accumulated by the most recent `backward()` call, or
    #: `null` before one has run.
    pub var grad: Tensor?
    #: The op that produced this tensor, or `null` for a leaf.
    pub var creator: GradNode?
    #: Whether `backward()`'s topological walk has already visited this tensor.
    #: `pub` only so the free function `visit_for_backward` can reach it.
    pub var topo_mark: bool
    #: A process-lifetime-unique identifier assigned from a monotonic
    #: counter at construction and never reused — stable even after this
    #: tensor's GPU buffer handle is freed and recycled by a later
    #: allocation. Used to key state (e.g. per-parameter optimizer state)
    #: that must not silently attach to an unrelated tensor.
    pub let id: int

    #: Allocate a tensor, optionally initialized from host values.
    pub fn init(self, shape: list[int], values: list[float]? = null, requires_grad: bool = false) {
        self.shape = [dimension for dimension in shape]
        self.size = element_count(shape)
        self.storage = gpu.Buffer(self.size)
        self.freed = false
        self.requires_grad = requires_grad
        self.grad = null
        self.creator = null
        self.topo_mark = false
        self.id = next_tensor_id()
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
        let gradient = self.grad
        if gradient is not null and not gradient.freed {
            gradient.free()
            self.grad = null
        }
    }

    #: Return this tensor, no longer part of any autograd graph.
    #:
    #: Clears `.creator` and `.requires_grad`, so an op consuming this
    #: tensor afterward never attaches a graph reaching back through
    #: intermediates that may already be freed.
    pub fn detach(self) -> Tensor {
        self.creator = null
        self.requires_grad = false
        return self
    }

    #: Record how to differentiate this tensor. Every op that builds a
    #: `GradNode` calls this instead of setting `.creator` directly — it
    #: sets `requires_grad` too, so the *next* op downstream sees this
    #: tensor as differentiable and keeps recording instead of the chain
    #: silently stopping one op after the first.
    pub fn attach_creator(self, node: GradNode) -> void {
        self.creator = node
        self.requires_grad = true
    }

    #: Compute gradients of every tensor this scalar tensor depends on.
    #:
    #: Seeds `self.grad` to `ones(self.shape)` first if it isn't already
    #: set, so a plain `loss.backward()` works without the caller manually
    #: pre-seeding the root gradient. Then accumulates into each
    #: dependency's `.grad`. Returns every non-leaf, non-root tensor the
    #: graph built — still valid, not yet freed — so the caller can free
    #: them after its own next `gpu.synchronize()` instead of forcing one
    #: here.
    pub fn backward(self) -> list[Tensor] {
        if self.size != 1 {
            throw ValueError(f"backward needs a scalar tensor, got shape {self.shape}")
        }
        if self.grad is null { self.grad = Tensor.ones(self.shape) }
        let order = self._topo_order()
        var index = order.len() - 1
        while index >= 0 {
            let tensor = order[index]
            let node = tensor.creator
            let upstream = tensor.grad
            if node is not null and upstream is not null {
                let grads = node.apply(upstream)
                for (position, input) in node.inputs.enumerate() {
                    accumulate_grad(input, grads[position])
                }
            }
            index -= 1
        }
        return self._collect_intermediates(order)
    }

    #: Every non-leaf, non-self tensor a forward-only graph built, without
    #: running backward. For a graph nothing will ever call `.backward()`
    #: on (evaluation, prediction outside training) — parameters still have
    #: `requires_grad: true` regardless of training/evaluating, so the graph
    #: gets built either way, and either way something has to free it.
    pub fn discard_graph(self) -> list[Tensor] {
        return self._collect_intermediates(self._topo_order())
    }

    fn _topo_order(self) -> list[Tensor] {
        var order: list[Tensor] = []
        visit_for_backward(self, order)
        return order
    }

    fn _collect_intermediates(self, order: list[Tensor]) -> list[Tensor] {
        var intermediates: list[Tensor] = []
        for tensor in order {
            tensor.topo_mark = false
            if tensor.creator is not null and tensor is not self {
                intermediates.push(tensor)
            }
        }
        return intermediates
    }

    #: Whether `free` has released the device storage.
    pub fn is_freed(self) -> bool { return self.freed }

    fn _check_live(self) -> void {
        if self.freed { throw RuntimeError("this Tensor has already been freed") }
    }

    fn __str__(self) -> str { return f"Tensor(shape={self.shape}, device={gpu.device_name()})" }
}

#: One recorded op in the autograd graph.
#:
#: `inputs` are the tensors an op consumed; `backward_fn` maps an upstream
#: gradient to one gradient per input, same order as `inputs`. Never
#: returns one of its own `inputs` by reference — always a fresh tensor.
pub class GradNode {
    pub let inputs: list[Tensor]
    let backward_fn: fn(Tensor) -> list[Tensor]

    pub fn init(self, inputs: list[Tensor], backward_fn: fn(Tensor) -> list[Tensor]) {
        self.inputs = inputs
        self.backward_fn = backward_fn
    }

    pub fn apply(self, upstream: Tensor) -> list[Tensor] { return self.backward_fn(upstream) }
}

let _GRAD_ADD_SOURCE = """#include <metal_stdlib>
using namespace metal;

kernel void grad_add(device const float* a [[buffer(0)]],
                     device const float* b [[buffer(1)]],
                     device float* output [[buffer(2)]],
                     constant uint& count [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    if (id < count) { output[id] = a[id] + b[id]; }
}
"""

var _grad_add_kernel: Kernel? = null

fn raw_grad_add(a: Tensor, b: Tensor) -> Tensor {
    let output = Tensor(a.shape)
    let existing = _grad_add_kernel
    let kernel = if existing is not null {
        existing
    } else {
        let compiled = Kernel.compile(_GRAD_ADD_SOURCE, "grad_add")
        _grad_add_kernel = compiled
        compiled
    }
    kernel.dispatch_async([a.buffer(), b.buffer(), output.buffer()], a.size, [a.size])
    return output
}

var _next_tensor_id: int = 0

fn next_tensor_id() -> int {
    let id = _next_tensor_id
    _next_tensor_id += 1
    return id
}

fn visit_for_backward(tensor: Tensor, order: list[Tensor]) -> void {
    if tensor.topo_mark { return }
    tensor.topo_mark = true
    let node = tensor.creator
    if node is not null {
        for input in node.inputs { visit_for_backward(input, order) }
    }
    order.push(tensor)
}

fn accumulate_grad(target: Tensor, contribution: Tensor) -> void {
    let existing = target.grad
    if existing is null {
        target.grad = contribution
    } else {
        let combined = raw_grad_add(existing, contribution)
        existing.free()
        contribution.free()
        target.grad = combined
    }
}

export { element_count, GradNode }

#: Construct a constant GPU tensor from flat row-major values.
pub fn constant(values: list[float], shape: list[int]) -> Tensor { return Tensor(shape, values) }

#: Allocate a zero-filled GPU tensor.
pub fn zeros(shape: list[int]) -> Tensor { return Tensor.zeros(shape) }

#: Allocate a one-filled GPU tensor.
pub fn ones(shape: list[int]) -> Tensor { return Tensor.ones(shape) }

#: Allocate a GPU tensor filled with `value`.
pub fn fill(shape: list[int], value: float) -> Tensor { return Tensor.full(shape, value) }

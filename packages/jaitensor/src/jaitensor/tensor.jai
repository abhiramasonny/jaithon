#: GPU-resident tensors.
import std.gpu as gpu
from std.gpu import Buffer, Kernel
from std.os import env_or

#: Whether `backward()` releases each intermediate's gradient, and any tensor
#: only one op's backward read, as soon as the walk is past them. On by
#: default; `JAITENSOR_BACKWARD_FREE=0` keeps everything until the caller
#: frees what `backward()` returns, which is how it behaved before, for an
#: A/B in one binary. Read once.
let _EARLY_FREE = env_or("JAITENSOR_BACKWARD_FREE", "1") != "0"

#: Whether ops record autograd graphs. `no_grad` clears it for a block.
var _grad_enabled = true

#: How many elements a shape holds.
#:
#: A dimension of zero is allowed and makes the whole tensor empty. Exported
#: graphs produce those -- a slice whose range comes up empty is the usual way
#: -- and an empty tensor has to travel through the same code as any other; the
#: loops over it simply run no times. A negative dimension is still a mistake.
fn element_count(shape: list[int]) -> int {
    if shape.len() == 0 { throw ValueError("a tensor needs at least one dimension") }
    var size = 1
    for dimension in shape {
        if dimension < 0 {
            throw ValueError(f"tensor dimensions cannot be negative, got {shape}")
        }
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
    #: Keep `.grad` after `backward()` on a tensor an op produced. Off by
    #: default, as in torch: an intermediate's gradient is released as soon
    #: as the walk has passed it. Leaves always keep theirs. Set it with
    #: `retain_grad()`.
    pub var retains_grad: bool
    #: Whether the storage was handed in rather than allocated here, so it
    #: may belong to, or share memory with, another tensor (`view`, `reshape`).
    pub let borrowed: bool
    #: A gradient this tensor's memory may be a window into, kept alive for
    #: exactly as long as this tensor is. `backward()` sets it when an op
    #: hands back a view of its upstream gradient (`reshape` does), so the
    #: upstream can be released with the view instead of outliving the walk.
    pub var backing: Tensor?
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
    #:
    #: Pass `storage` to wrap an existing buffer (a view). The tensor does
    #: not own that buffer; `free` only releases the view handle.
    pub fn init(
        self,
        shape: list[int],
        values: list[float]? = null,
        requires_grad: bool = false,
        storage: Buffer? = null
    ) {
        self.shape = [dimension for dimension in shape]
        self.size = element_count(shape)
        self.freed = false
        self.requires_grad = requires_grad
        self.grad = null
        self.creator = null
        self.retains_grad = false
        self.backing = null
        self.topo_mark = false
        self.id = next_tensor_id()
        self.borrowed = storage is not null
        if storage is not null {
            if values is not null {
                throw ValueError("a storage-backed tensor cannot also take host values")
            }
            if storage.count != self.size {
                throw ValueError(f"storage has {storage.count} values, expected {self.size}")
            }
            self.storage = storage
            return
        }
        self.storage = gpu.Buffer(self.size)
        if values is not null {
            if values.len() != self.size {
                self.storage.free()
                self.freed = true
                throw ValueError(f"tensor has {values.len()} values, expected {self.size}")
            }
            self.storage.upload(values)
        }
    }

    #: A window into this tensor. Freeing the view does not free the parent.
    pub fn view(self, start: int, shape: list[int]) -> Tensor {
        self._check_live()
        let n = element_count(shape)
        if start < 0 or start + n > self.size {
            throw ValueError(f"view offset {start} and size {n} run past tensor size {self.size}")
        }
        return Tensor(shape, storage: self.storage.view(start, n))
    }

    #: Allocate a zero-filled tensor.
    pub static fn zeros(shape: list[int]) -> Tensor {
        let tensor = Tensor(shape)
        tensor.storage.fill_zero()
        return tensor
    }

    #: Allocate a tensor filled with ones.
    pub static fn ones(shape: list[int]) -> Tensor {
        let tensor = Tensor(shape)
        tensor.storage.fill(1.0)
        return tensor
    }

    #: Allocate a tensor with every element set to `value`.
    pub static fn full(shape: list[int], value: float) -> Tensor {
        let tensor = Tensor(shape)
        tensor.storage.fill(value)
        return tensor
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
    #:
    #: Waits for the queued kernels that write this tensor, so a GPU fill
    #: cannot race the memcpy. Work queued against anything else keeps running.
    pub fn upload(self, values: list[float]) -> void {
        self._check_live()
        if values.len() != self.size {
            throw ValueError(f"tensor upload has {values.len()} values, expected {self.size}")
        }
        self.storage.upload(values)
    }

    #: Copy all values to the host.
    #:
    #: Waits for the work that produced this tensor and no more, so reading one
    #: result does not stall the batch queued behind it.
    pub fn download(self) -> list[float] {
        self._check_live()
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

    #: Set every value to 0 on the device, keeping this tensor's buffer.
    pub fn zero_(self) -> void {
        self._check_live()
        fill_zero_buffer(self)
    }

    #: Release this tensor's device memory. Freeing twice is harmless.
    #:
    #: Does not wait for the GPU. In-flight command buffers keep the Metal
    #: buffer alive until they complete; call `download()` or
    #: `gpu.synchronize()` before reading values back.
    pub fn free(self) -> void {
        if self.freed { return }
        self.storage.free()
        self.freed = true
        let gradient = self.grad
        if gradient is not null and not gradient.freed {
            gradient.free()
            self.grad = null
        }
        let backing = self.backing
        if backing is not null {
            self.backing = null
            backing.free()
        }
    }

    #: Keep this tensor's `.grad` through `backward()` even though an op
    #: produced it. Returns this tensor.
    pub fn retain_grad(self) -> Tensor {
        self.retains_grad = true
        return self
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
    #:
    #: Inside `no_grad` it records nothing: the tensor stays a plain value
    #: that its caller frees, and the tensors the node saved for its backward
    #: are released here, since no backward will ever read them.
    pub fn attach_creator(self, node: GradNode) -> void {
        if not _grad_enabled {
            for extra in node.extras {
                if extra is not self and extra.creator is null { extra.free() }
            }
            return
        }
        self.creator = node
        self.requires_grad = true
    }

    #: Compute gradients of every tensor this scalar tensor depends on.
    #:
    #: Seeds `self.grad` to `ones(self.shape)` first if it isn't already
    #: set, so a plain `loss.backward()` works without the caller manually
    #: pre-seeding the root gradient. Then accumulates into each
    #: dependency's `.grad`, skipping any that never asked for one. Returns
    #: every non-leaf, non-root tensor the graph built — still valid, not yet
    #: freed — so the caller can free them after encoding the rest of the
    #: step. `Tensor.free` no longer fences; `gpu.flush()` or
    #: `gpu.synchronize()` commits the work.
    #:
    #: An intermediate's own `.grad` is released as soon as its op's backward
    #: has read it (unless `retain_grad()` asked to keep it), and so is a
    #: tensor only one op saved for its backward; freeing those again with
    #: the rest of the returned list is harmless. Leaves keep their `.grad`.
    pub fn backward(self) -> list[Tensor] {
        if self.size != 1 {
            throw ValueError(f"backward needs a scalar tensor, got shape {self.shape}")
        }
        #: Shaped like the loss, not merely sized like it. A loss that came
        #: out of a matrix product is `[1, 1]` rather than `[1]`, and seeding
        #: the wrong rank hands the first operator's backward an upstream it
        #: cannot index. The one-element case keeps the shared scalar, which
        #: is every loss a reduction produced and so nearly all of them.
        let seeded = self.grad
        var own_seed: Tensor? = null
        if seeded is null and self.shape != [1] { own_seed = Tensor.ones(self.shape) }
        let made = own_seed
        let root_upstream = if seeded is not null {
            seeded
        } elif made is not null {
            made
        } else {
            unit_scalar()
        }
        let order = self._topo_order()
        let last_reader = if _EARLY_FREE { last_readers(order) } else { {} }
        var index = order.len() - 1
        while index >= 0 {
            let tensor = order[index]
            let node = tensor.creator
            let upstream = if tensor is self { root_upstream } else { tensor.grad }
            if node is not null and upstream is not null {
                let grads = node.apply(upstream)
                #: The root's upstream is shared or the caller's, and a
                #: retained gradient must not change under the input that
                #: would otherwise alias it, so those are copied out.
                let keeps = tensor is self or tensor.retains_grad
                if keeps { own_grads(node, grads, upstream) }
                settle_grads(node, grads, upstream)
                if not keeps {
                    if _EARLY_FREE {
                        release_upstream(tensor, upstream, node, grads)
                    } elif handed_on(node, grads, upstream) {
                        #: Passed straight through: the input owns it now.
                        tensor.grad = null
                    }
                }
                if _EARLY_FREE { release_saved(node, index, last_reader) }
            }
            index -= 1
        }
        let collected = self._collect_intermediates(order)
        let temporary = own_seed
        if temporary is not null { temporary.free() }
        return collected
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
            let node = tensor.creator
            if node is not null {
                if tensor is not self { intermediates.push(tensor) }
                for extra in node.extras { intermediates.push(extra) }
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
#: gradient to one gradient per input, same order as `inputs`. For an input
#: that wants a gradient it returns a fresh tensor, never one of `inputs` by
#: reference. For one that does not (`needs` is false there) it may skip the
#: work and return that input itself as a placeholder, which `backward()`
#: ignores; anything else returned there is freed unused.
pub class GradNode {
    pub let inputs: list[Tensor]
    pub let extras: list[Tensor]
    #: Whether each input wanted a gradient when the op ran, same order as
    #: `inputs`. Only these receive one: data, targets and constants that
    #: never asked for a gradient stay without one. An input some op
    #: produced counts as wanting one, since the walk runs through it.
    pub let needs: list[bool]
    let backward_fn: fn(Tensor) -> list[Tensor]

    pub fn init(
        self,
        inputs: list[Tensor],
        backward_fn: fn(Tensor) -> list[Tensor],
        extras: list[Tensor] = []
    ) {
        self.inputs = inputs
        self.backward_fn = backward_fn
        self.extras = extras
        self.needs = [wants_grad(input) for input in inputs]
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

kernel void grad_add_into(device const float* source [[buffer(0)]],
                          device float* dest [[buffer(1)]],
                          constant uint& count [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    if (id < count) dest[id] += source[id];
}

kernel void grad_copy(device const float* source [[buffer(0)]],
                      device float* dest [[buffer(1)]],
                      constant uint& count [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    if (id < count) dest[id] = source[id];
}

kernel void fill_zero(device float* output [[buffer(0)]],
                      constant uint& count [[buffer(1)]],
                      uint id [[thread_position_in_grid]]) {
    if (id < count) output[id] = 0.0f;
}
"""

var _grad_add_kernel: Kernel? = null
var _grad_add_into_kernel: Kernel? = null
var _fill_zero_kernel: Kernel? = null
var _grad_copy_kernel: Kernel? = null

fn raw_copy_into(dest: Tensor, source: Tensor) -> void {
    let existing = _grad_copy_kernel
    let kernel = if existing is not null {
        existing
    } else {
        let compiled = Kernel.compile(_GRAD_ADD_SOURCE, "grad_copy")
        _grad_copy_kernel = compiled
        compiled
    }
    kernel.dispatch_async([source.buffer(), dest.buffer()], dest.size, [dest.size])
}

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

fn raw_grad_add_into(dest: Tensor, source: Tensor) -> void {
    let existing = _grad_add_into_kernel
    let kernel = if existing is not null {
        existing
    } else {
        let compiled = Kernel.compile(_GRAD_ADD_SOURCE, "grad_add_into")
        _grad_add_into_kernel = compiled
        compiled
    }
    kernel.dispatch_async([source.buffer(), dest.buffer()], dest.size, [dest.size])
}

fn fill_zero_buffer(tensor: Tensor) -> void {
    let existing = _fill_zero_kernel
    let kernel = if existing is not null {
        existing
    } else {
        let compiled = Kernel.compile(_GRAD_ADD_SOURCE, "fill_zero")
        _fill_zero_kernel = compiled
        compiled
    }
    kernel.dispatch_async([tensor.buffer()], tensor.size, [tensor.size])
}

var _next_tensor_id: int = 0
var _unit_scalar: Tensor? = null

fn unit_scalar() -> Tensor {
    let existing = _unit_scalar
    if existing is not null and not existing.is_freed() { return existing }
    let created = Tensor.ones([1])
    _unit_scalar = created
    return created
}

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
    } elif existing is contribution {
        return
    } else {
        raw_grad_add_into(existing, contribution)
        contribution.free()
    }
}

#: Whether a backward should compute `tensor`'s gradient: it asked for one,
#: or an op produced it, so the walk runs through it. Ops building a
#: `GradNode` use this to skip the side of a backward nobody reads.
pub fn wants_grad(tensor: Tensor) -> bool { return tensor.requires_grad or tensor.creator is not null }

#: Whether `grads` holds `candidate` at a position whose input wants it.
fn handed_on(node: GradNode, grads: list[Tensor], candidate: Tensor) -> bool {
    for (position, grad) in grads.enumerate() {
        if grad is candidate and node.needs[position] { return true }
    }
    return false
}

#: Give each input that wants a gradient its share, and drop the rest.
#:
#: A position whose input never asked for a gradient gets nothing: what the
#: op returned there is freed, unless it is the placeholder (the input
#: itself), the upstream, or a tensor also handed to an input that does want
#: it.
fn settle_grads(node: GradNode, grads: list[Tensor], upstream: Tensor) -> void {
    for (position, input) in node.inputs.enumerate() {
        let grad = grads[position]
        if node.needs[position] {
            if grad is not input { accumulate_grad(input, grad) }
        } elif grad is not input and grad is not upstream and not handed_on(node, grads, grad) {
            grad.free()
        }
    }
}

#: Copy out any gradient that is the upstream or a view of it.
#:
#: For the root, whose upstream is shared (the unit scalar every scalar loss
#: starts from) or the caller's seed, and for a tensor keeping its gradient:
#: an op that passed either straight through, or a view of it, would hand an
#: input a gradient that something else owns.
fn own_grads(node: GradNode, grads: list[Tensor], upstream: Tensor) -> void {
    for (position, grad) in grads.enumerate() {
        if not node.needs[position] { continue }
        let input = node.inputs[position]
        if grad is input or grad is input.grad { continue }
        if grad is upstream or grad.borrowed {
            let copy = Tensor(grad.shape)
            raw_copy_into(copy, grad)
            if grad is not upstream { grad.free() }
            grads[position] = copy
        }
    }
}

#: Release an intermediate's gradient once its op's backward has read it.
#:
#: Every consumer has already contributed by now -- they all come later in
#: the forward order, so earlier in this walk -- and nothing but this op's
#: backward reads it. Two things can still hold its memory: an op that
#: passed it straight through as an input's gradient (which then owns it),
#: or one that handed back a view of it (`reshape`), which then keeps it
#: alive as its `backing`. With more than one such holder it is left for the
#: caller to free with the tensor, as before.
fn release_upstream(tensor: Tensor, upstream: Tensor, node: GradNode, grads: list[Tensor]) -> void {
    var holders = 0
    var holder: Tensor? = null
    for (position, grad) in grads.enumerate() {
        if not node.needs[position] or grad.is_freed() { continue }
        if grad is upstream or grad.borrowed {
            holders += 1
            holder = grad
        }
    }
    if holders > 1 { return }
    tensor.grad = null
    let kept = holder
    if kept is null {
        upstream.free()
    } elif kept is not upstream {
        let earlier = kept.backing
        if earlier is null {
            kept.backing = upstream
        } else {
            #: Already backed; keep this one alive behind it.
            upstream.backing = earlier
            kept.backing = upstream
        }
    }
}

#: The walk position of the last op to read each saved tensor, by id.
#:
#: The walk runs from the end of `order` down, so the last reader is the one
#: with the smallest index. A saved tensor any op also takes as an input is
#: never released early: that op reads it too, and it may be the caller's.
fn last_readers(order: list[Tensor]) -> dict[int, int] {
    var last: dict[int, int] = {}
    for (index, tensor) in order.enumerate() {
        let node = tensor.creator
        if node is null { continue }
        for extra in node.extras {
            let known = last.get(extra.id, -1)
            if known < 0 or index < known { last[extra.id] = index }
        }
    }
    if last.len() == 0 { return last }
    for tensor in order {
        let node = tensor.creator
        if node is null { continue }
        for input in node.inputs {
            if last.contains(input.id) { last[input.id] = -1 }
        }
    }
    return last
}

#: Free the tensors `node` saved for its backward once no later op reads them.
fn release_saved(node: GradNode, index: int, last_reader: dict[int, int]) -> void {
    for extra in node.extras {
        if extra.creator is not null { continue }
        if last_reader.get(extra.id, -1) == index { extra.free() }
    }
}

#: Run `body` without recording any autograd graph.
#:
#: Ops inside record nothing: what they return is a plain tensor with no
#: `.creator`, which the caller frees like any other, and nothing needs
#: `discard_graph()`. Parameters keep their `requires_grad`; the block does
#: not touch them. Nests, and restores the previous state even when `body`
#: throws.
pub fn no_grad(body: fn() -> void) -> void {
    let previous = _grad_enabled
    _grad_enabled = false
    defer { _grad_enabled = previous }
    body()
}

#: Whether ops are recording autograd graphs (false inside `no_grad`).
pub fn is_grad_enabled() -> bool { return _grad_enabled }

#: Turn graph recording on or off until changed again; returns the previous
#: setting so a caller can restore it. `no_grad` is the scoped form.
pub fn set_grad_enabled(enabled: bool) -> bool {
    let previous = _grad_enabled
    _grad_enabled = enabled
    return previous
}

export { element_count, GradNode, no_grad, is_grad_enabled, set_grad_enabled, wants_grad }

#: Construct a constant GPU tensor from flat row-major values.
pub fn constant(values: list[float], shape: list[int]) -> Tensor { return Tensor(shape, values) }

#: Allocate a zero-filled GPU tensor.
pub fn zeros(shape: list[int]) -> Tensor { return Tensor.zeros(shape) }

#: Allocate a one-filled GPU tensor.
pub fn ones(shape: list[int]) -> Tensor { return Tensor.ones(shape) }

#: Allocate a GPU tensor filled with `value`.
pub fn fill(shape: list[int], value: float) -> Tensor { return Tensor.full(shape, value) }

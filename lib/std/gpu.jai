#: Compute on the GPU — device buffers, Metal kernels, and the handful of
#: vector operations most programs actually want — with a correct CPU
#: implementation of every operation for when there is no device.
#:
#: `is_available` reports whether a device was found; `Buffer` and `Kernel`
#: raise `RuntimeError` without one, but every operation below picks CPU or
#: GPU itself and never does.
#:
#: A device buffer holds 32-bit floats (Metal's only option), so `upload` and
#: `download` round through that; the CPU path computes in double precision
#: with Neumaier compensation and is *more* accurate, not just different —
#: never compare the two paths for exact equality, only with a tolerance such
#: as `math.approx_eq`.
#:
#: The CPU/GPU crossover is roughly a hundred thousand elements for elementwise
#: work and far lower for `matmul`, whose CPU cost is cubic. `gpu_dispatch` runs
#: its command buffer to completion before returning, so nothing here needs a
#: separate fence.
#:
#: ```
#: vector_add([1.0, 2.0], [3.0, 4.0])    # [4.0, 6.0]
#: dot([1.0, 2.0], [3.0, 4.0])           # 11.0
#: reduce_sum([1.0, 2.0, 3.0])           # 6.0
#: ```
import std.math

#: Inputs shorter than this are computed on the CPU even when a GPU exists.
#:
#: Uploading a buffer, encoding a command and waiting for the queue costs tens
#: of microseconds; below this size the arithmetic does not.
pub const MIN_GPU_ELEMENTS = 4096

# Must equal the Metal kernels' hardcoded threadgroup width (their scratch
# array size); 256 is within every device's guaranteed maximum.
const _REDUCE_GROUP = 256

let _ELEMENTWISE_SOURCE = """#include <metal_stdlib>
using namespace metal;

kernel void vector_add(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* out [[buffer(2)]],
                       constant uint& n [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < n) { out[i] = a[i] + b[i]; }
}

kernel void vector_mul(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* out [[buffer(2)]],
                       constant uint& n [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < n) { out[i] = a[i] * b[i]; }
}
"""

let _REDUCE_SOURCE = """#include <metal_stdlib>
using namespace metal;

constexpr constant uint GROUP = 256;

kernel void reduce_sum(device const float* x [[buffer(0)]],
                       device float* partials [[buffer(1)]],
                       constant uint& n [[buffer(2)]],
                       uint gid [[thread_position_in_grid]],
                       uint lid [[thread_index_in_threadgroup]],
                       uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[GROUP];
    scratch[lid] = (gid < n) ? x[gid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = GROUP / 2; stride > 0; stride >>= 1) {
        if (lid < stride) { scratch[lid] += scratch[lid + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) { partials[wid] = scratch[0]; }
}

kernel void reduce_dot(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* partials [[buffer(2)]],
                       constant uint& n [[buffer(3)]],
                       uint gid [[thread_position_in_grid]],
                       uint lid [[thread_index_in_threadgroup]],
                       uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[GROUP];
    scratch[lid] = (gid < n) ? a[gid] * b[gid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = GROUP / 2; stride > 0; stride >>= 1) {
        if (lid < stride) { scratch[lid] += scratch[lid + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) { partials[wid] = scratch[0]; }
}
"""

let _MATMUL_SOURCE = """#include <metal_stdlib>
using namespace metal;

kernel void matmul(device const float* a [[buffer(0)]],
                   device const float* b [[buffer(1)]],
                   device float* out [[buffer(2)]],
                   constant uint& rows [[buffer(3)]],
                   constant uint& inner [[buffer(4)]],
                   constant uint& columns [[buffer(5)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid >= rows * columns) { return; }
    uint row = gid / columns;
    uint column = gid % columns;
    float total = 0.0f;
    for (uint k = 0; k < inner; ++k) {
        total += a[row * inner + k] * b[k * columns + column];
    }
    out[gid] = total;
}
"""

let _MAP_PREFIX = """#include <metal_stdlib>
using namespace metal;

kernel void apply(device const float* input [[buffer(0)]],
                  device float* output [[buffer(1)]],
                  constant uint& n [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) {
    if (i >= n) { return; }
    float x = input[i];
    output[i] = ("""

let _MAP_SUFFIX = """);
}
"""

#: Report whether a compute device was found.
pub fn is_available() -> bool { return __prim__.gpu_available() }

#: Return the device name, or "none" when there is no device.
#:
#: Meant for logs and benchmark headers, never for a decision — match on
#: `is_available` instead of on a name.
pub fn device_name() -> str {
    if not is_available() { return "none" }
    return __prim__.gpu_device_name()
}

#: A block of 32-bit floats in device memory.
#:
#: Free it when done: device memory isn't garbage collected, so a leak here
#: survives every collection until the process exits.
pub class Buffer {
    #: Number of float slots the buffer holds.
    pub let count: int

    let handle: int
    var freed: bool

    #: Allocate `count` float slots on the device, with undefined contents.
    #:
    #: Raises `RuntimeError` when no device is available.
    pub fn init(self, count: int) {
        if count <= 0 { throw ValueError(f"buffer count must be positive, got {count}") }
        _require_device("allocate a Buffer")
        self.count = count
        self.handle = __prim__.gpu_buffer_new(count)
        self.freed = false
    }

    #: Allocate a buffer holding `values`.
    pub static fn of(values: list[float]) -> Buffer {
        let buffer = Buffer(values.len())
        buffer.upload(values)
        return buffer
    }

    #: Copy `values` into the buffer starting at `offset`.
    #:
    #: Each value is rounded to the nearest 32-bit float. Raises `ValueError`
    #: when the values would run past the end of the buffer.
    pub fn upload(self, values: list[float], offset: int = 0) -> void {
        self._check_live()
        if offset < 0 { throw ValueError(f"offset must be non-negative, got {offset}") }
        if offset + values.len() > self.count {
            throw ValueError(
                f"upload of {values.len()} values at {offset} exceeds capacity {self.count}"
            )
        }
        __prim__.gpu_buffer_upload(self.handle, values, offset)
    }

    #: Read `count` values back, starting at `offset`.
    #:
    #: A null `count` reads to the end of the buffer.
    pub fn download(self, offset: int = 0, count: int? = null) -> list[float] {
        self._check_live()
        if offset < 0 { throw ValueError(f"offset must be non-negative, got {offset}") }
        let wanted = count ?? self.count - offset
        if wanted < 0 or offset + wanted > self.count {
            throw ValueError(f"download of {wanted} values at {offset} exceeds {self.count}")
        }
        return __prim__.gpu_buffer_download(self.handle, offset, wanted)
    }

    #: Release the device memory. Freeing twice is harmless.
    pub fn free(self) -> void {
        if self.freed { return }
        __prim__.gpu_buffer_free(self.handle)
        self.freed = true
    }

    #: Report whether the buffer has been freed.
    pub fn is_freed(self) -> bool { return self.freed }

    #: Return the native device handle, raising `RuntimeError` once freed.
    #:
    #: `Kernel.dispatch` binds it; nothing else has a use for it.
    pub fn handle_of(self) -> int {
        self._check_live()
        return self.handle
    }

    fn _check_live(self) -> void {
        if self.freed { throw RuntimeError("this Buffer has already been freed") }
    }
}

#: One compute kernel, compiled from Metal Shading Language source.
#:
#: A kernel's arguments bind in fixed order: buffers occupy `buffer(0)` upward
#: in the order passed to `dispatch`, then scalars as `constant uint&` — a
#: kernel expecting its element count last must declare it after every buffer.
pub class Kernel {
    #: The entry-point function name this kernel was compiled from.
    pub let entry: str

    let handle: int
    var freed: bool

    #: Adopt an already-compiled native pipeline `handle`.
    #:
    #: `compile` is the only correct caller.
    pub fn init(self, entry: str, handle: int) {
        self.entry = entry
        self.handle = handle
        self.freed = false
    }

    #: Compile `source` and select the kernel function named `entry`.
    #:
    #: Raises `RuntimeError` when there is no device, and `ValueError` with the
    #: compiler's diagnostics when the source does not build.
    pub static fn compile(source: str, entry: str) -> Kernel {
        _require_device(f"compile the kernel {entry}")
        if entry.len() == 0 { throw ValueError("kernel entry name must not be empty") }
        return Kernel(entry, __prim__.gpu_compile(source, entry))
    }

    #: The largest threadgroup this kernel can be dispatched with.
    pub fn max_group_size(self) -> int {
        self._check_live()
        return __prim__.gpu_max_threads_per_group(self.handle)
    }

    #: Run the kernel over `threads` threads and wait for it to finish.
    #:
    #: `group_size` 0 lets the device pick the widest group the kernel supports;
    #: a kernel written against a fixed threadgroup width (any with a
    #: `threadgroup` array) must be dispatched with that exact width instead.
    pub fn dispatch(
        self,
        buffers: list[Buffer],
        threads: int,
        scalars: list[int] = [],
        group_size: int = 0
    ) -> void {
        self._check_live()
        if threads <= 0 { throw ValueError(f"threads must be positive, got {threads}") }
        if group_size < 0 { throw ValueError(f"group_size must be non-negative, got {group_size}") }
        let handles = buffers.map(|buffer| buffer.handle_of())
        let width = if group_size > 0 { group_size } else { self.max_group_size() }
        __prim__.gpu_dispatch(self.handle, handles, scalars, threads, width)
    }

    #: Release the compiled pipeline. Freeing twice is harmless.
    pub fn free(self) -> void {
        if self.freed { return }
        __prim__.gpu_kernel_free(self.handle)
        self.freed = true
    }

    #: Report whether the kernel has been freed.
    pub fn is_freed(self) -> bool { return self.freed }

    fn _check_live(self) -> void {
        if self.freed { throw RuntimeError(f"the kernel {self.entry} has already been freed") }
    }

    fn __str__(self) -> str { return f"Kernel({self.entry})" }
}

# Caches compiled kernels; keyed by entry+source together since `map_kernel`
# reuses the entry name `apply` for many different sources.
var _pipelines: dict[str, Kernel] = {}

#: Add `a` and `b` elementwise.
#:
#: Raises `ValueError` when the two lengths differ.
#:
#: ```
#: vector_add([1.0, 2.0], [3.0, 4.0])    # [4.0, 6.0]
#: ```
pub fn vector_add(a: list[float], b: list[float]) -> list[float] {
    _check_same_length(a, b, "vector_add")
    return __prim__.gpu_vector_add(a, b)
}

#: Multiply `a` and `b` elementwise.
#:
#: Raises `ValueError` when the two lengths differ.
#:
#: ```
#: vector_mul([1.0, 2.0], [3.0, 4.0])    # [3.0, 8.0]
#: ```
pub fn vector_mul(a: list[float], b: list[float]) -> list[float] {
    _check_same_length(a, b, "vector_mul")
    return __prim__.gpu_vector_mul(a, b)
}

#: Return the dot product of `a` and `b`.
#:
#: The CPU path uses Neumaier-compensated summation; the device path sums a
#: threadgroup-partial tree — a different but equally accurate order, so the
#: two agree only to a tolerance.
#:
#: ```
#: dot([1.0, 2.0], [3.0, 4.0])    # 11.0
#: ```
pub fn dot(a: list[float], b: list[float]) -> float {
    _check_same_length(a, b, "dot")
    if not _use_gpu(a.len()) { return _cpu_dot(a, b) }
    let left = Buffer.of(a)
    let right = Buffer.of(b)
    defer { left.free() }
    defer { right.free() }
    return _reduce_with("reduce_dot", [left, right], a.len())
}

#: Return the sum of `values`.
#:
#: ```
#: reduce_sum([1.0, 2.0, 3.0, 4.5])    # 10.5
#: reduce_sum([])                      # 0.0
#: ```
pub fn reduce_sum(values: list[float]) -> float {
    return __prim__.gpu_reduce_sum(values)
}

#: Multiply the `rows`×`inner` matrix `a` by the `inner`×`columns` matrix `b`.
#:
#: Both matrices are row-major and flat: element `(i, j)` of `a` lives at
#: `a[i * inner + j]`. The result is `rows`×`columns`, also flat. Raises
#: `ValueError` when a length does not match its declared shape.
#:
#: The CPU path iterates row, inner, column so that both `b` and the output row
#: are walked forwards — the same arithmetic as the textbook triple loop, but
#: several times faster for anything that does not fit in cache.
#:
#: ```
#: matmul([1.0, 2.0, 3.0, 4.0], [1.0, 0.0, 0.0, 1.0], 2, 2, 2)    # [1.0, 2.0, 3.0, 4.0]
#: matmul([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 1, 3, 1)              # [32.0]
#: ```
pub fn matmul(a: list[float], b: list[float], rows: int, inner: int, columns: int) -> list[float] {
    if rows <= 0 or inner <= 0 or columns <= 0 {
        throw ValueError(f"matmul shape must be positive, got {rows}x{inner}x{columns}")
    }
    if a.len() != rows * inner {
        throw ValueError(f"left matrix has {a.len()} elements, expected {rows * inner}")
    }
    if b.len() != inner * columns {
        throw ValueError(f"right matrix has {b.len()} elements, expected {inner * columns}")
    }
    # Deliberately not __prim__.gpu_matmul: measured 29ms vs its 205ms at
    # 512x512 (its kernel is tiled, this isn't -- gap is dispatch, not arithmetic).
    if not _use_gpu(rows * columns * inner) { return _cpu_matmul(a, b, rows, inner, columns) }
    let left = Buffer.of(a)
    let right = Buffer.of(b)
    let output = Buffer(rows * columns)
    defer { left.free() }
    defer { right.free() }
    defer { output.free() }
    let kernel = _cached(_MATMUL_SOURCE, "matmul")
    kernel.dispatch([left, right, output], rows * columns, scalars: [rows, inner, columns])
    return output.download()
}

#: Apply a Metal expression to every element of `values`.
#:
#: `expression` is a Metal Shading Language expression in the element `x`, of
#: type `float`; `fallback` is the same function written in Jaithon, used when
#: there is no device. **The two must agree** — the whole point is that a
#: caller gets the same answer either way — so keep them on adjacent lines and
#: change them together.
#:
#: The compiled pipeline is cached, so calling this in a loop with the same
#: expression compiles once.
#:
#: ```
#: map_kernel([1.0, 2.0, 3.0], "x * x + 1.0f", |x| x * x + 1.0)    # [2.0, 5.0, 10.0]
#: ```
pub fn map_kernel(
    values: list[float],
    expression: str,
    fallback: fn(float) -> float
) -> list[float] {
    if expression.len() == 0 { throw ValueError("map_kernel expression must not be empty") }
    if not _use_gpu(values.len()) { return values.map(fallback) }
    let input = Buffer.of(values)
    let output = Buffer(values.len())
    defer { input.free() }
    defer { output.free() }
    let kernel = _cached(_MAP_PREFIX + expression + _MAP_SUFFIX, "apply")
    kernel.dispatch([input, output], values.len(), scalars: [values.len()])
    return output.download()
}

# `work` counts multiply-adds, not elements -- matmul passes `rows*inner*columns`,
# since it's the arithmetic that must outweigh the upload, not the output size.
fn _use_gpu(work: int) -> bool {
    return work >= MIN_GPU_ELEMENTS and is_available()
}

fn _require_device(action: str) -> void {
    if not is_available() {
        throw RuntimeError(f"cannot {action}: no GPU device is available")
    }
}

fn _check_same_length(a: list[float], b: list[float], operation: str) -> void {
    if a.len() != b.len() {
        throw ValueError(f"{operation} needs equal lengths, got {a.len()} and {b.len()}")
    }
}

fn _cached(source: str, entry: str) -> Kernel {
    let key = entry + "\n" + source
    let existing = _pipelines.get(key, null)
    if existing is not null { return existing }
    let kernel = Kernel.compile(source, entry)
    _pipelines[key] = kernel
    return kernel
}

fn _elementwise(a: list[float], b: list[float], entry: str) -> list[float] {
    let left = Buffer.of(a)
    let right = Buffer.of(b)
    let output = Buffer(a.len())
    defer { left.free() }
    defer { right.free() }
    defer { output.free() }
    let kernel = _cached(_ELEMENTWISE_SOURCE, entry)
    kernel.dispatch([left, right, output], a.len(), scalars: [a.len()])
    return output.download()
}

# One threadgroup produces one partial sum; there are few enough of them that
# summing them on the host costs nothing and keeps the kernel to a single pass.
fn _reduce_with(entry: str, inputs: list[Buffer], count: int) -> float {
    let groups = (count + _REDUCE_GROUP - 1) // _REDUCE_GROUP
    let partials = Buffer(groups)
    defer { partials.free() }
    let arguments: list[Buffer] = []
    for input in inputs { arguments.push(input) }
    arguments.push(partials)
    let kernel = _cached(_REDUCE_SOURCE, entry)
    kernel.dispatch(arguments, groups * _REDUCE_GROUP, scalars: [count], group_size: _REDUCE_GROUP)
    return _compensated_sum(partials.download())
}

# Neumaier's variant of Kahan summation: keeps the low bits even when the
# running total is smaller than the term being added (plain Kahan loses them).
fn _compensated_sum(values: list[float]) -> float {
    var total = 0.0
    var correction = 0.0
    for value in values {
        let next = total + value
        if math.fabs(total) >= math.fabs(value) {
            correction += total - next + value
        } else {
            correction += value - next + total
        }
        total = next
    }
    return total + correction
}

fn _cpu_dot(a: list[float], b: list[float]) -> float {
    var total = 0.0
    var correction = 0.0
    for index in 0..a.len() {
        let term = a[index] * b[index]
        let next = total + term
        if math.fabs(total) >= math.fabs(term) {
            correction += total - next + term
        } else {
            correction += term - next + total
        }
        total = next
    }
    return total + correction
}

fn _cpu_matmul(a: list[float], b: list[float], rows: int, inner: int, columns: int) -> list[float] {
    let output = [0.0 for _ in 0..rows * columns]
    for row in 0..rows {
        let out_base = row * columns
        let a_base = row * inner
        for k in 0..inner {
            let factor = a[a_base + k]
            if factor == 0.0 { continue }
            let b_base = k * columns
            for column in 0..columns {
                output[out_base + column] += factor * b[b_base + column]
            }
        }
    }
    return output
}

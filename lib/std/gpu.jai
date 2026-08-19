#: Compute on the GPU — device buffers, Metal kernels, and the handful of
#: vector operations most programs actually want — with a correct CPU
#: implementation of every operation for when there is no device.
#:
#: `is_available` reports whether a device was found; `Buffer` and `Kernel`
#: raise `RuntimeError` without one, but every operation below picks CPU or
#: GPU itself and never does.
#:
#: A device buffer holds 32-bit floats. Mixed precision (`set_mixed_precision`)
#: runs GEMM and fused MLP graphs in float16 and writes float32; storage is
#: still float32. `upload` and `download` round through float32. The CPU path
#: computes in double precision with Neumaier compensation and is *more*
#: accurate, not just different — never compare the two paths for exact
#: equality, only with a tolerance such as `math.approx_eq`.
#:
#: Call `set_device` before the first buffer or kernel. Device memory is not
#: garbage collected; `free` a `Buffer` when you are done.
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

let _FILL_SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void fill_bits(device float* output [[buffer(0)]],
                      constant uint& count [[buffer(1)]],
                      constant uint& bits [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    if (id < count) output[id] = as_type<float>(bits);
}

kernel void fill_uniform(device float* output [[buffer(0)]],
                         constant uint& count [[buffer(1)]],
                         constant uint& low_bits [[buffer(2)]],
                         constant uint& high_bits [[buffer(3)]],
                         constant uint& seed_lo [[buffer(4)]],
                         constant uint& seed_hi [[buffer(5)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    ulong seed = ((ulong)seed_hi << 32) | (ulong)seed_lo;
    if (seed == 0ul) seed = 0x9E3779B97F4A7C15ul;
    ulong state = seed + ((ulong)id + 1ul) * 0x9E3779B97F4A7C15ul;
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    float low = as_type<float>(low_bits);
    float high = as_type<float>(high_bits);
    float u = (float)((state >> 40) & 0xFFFFFFul) * (1.0f / 16777216.0f);
    output[id] = low + (high - low) * u;
}
"""

fn _f32_bits(value: float) -> int {
    if value != value { return 0x7FC00000 }
    var mag = value
    var sign = 0
    if mag < 0.0 {
        sign = 1
        mag = -mag
    }
    if mag == 0.0 { return sign << 31 }
    if mag >= 3.4028234663852886e38 { return sign << 31 | 0x7F800000 }
    if mag <= 1.401298464324817e-45 { return sign << 31 }
    let split = __prim__.f64_frexp(mag)
    let biased = split[1] + 126
    if biased <= 0 {
        let frac = int(__prim__.f64_ldexp(mag, 149))
        return sign << 31 | frac & 0x7FFFFF
    }
    if biased >= 255 { return sign << 31 | 0x7F800000 }
    let frac = int((split[0] * 2.0 - 1.0) * 8388608.0 + 0.5)
    return sign << 31 | biased << 23 | frac & 0x7FFFFF
}

#: Inputs shorter than this are computed on the CPU even when a GPU exists.
#:
#: Uploading a buffer, encoding a command and waiting for the queue costs tens
#: of microseconds; below this size the arithmetic does not.
pub const MIN_GPU_ELEMENTS = 4096

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

#: How many Metal devices the process can see.
pub fn device_count() -> int { return __prim__.gpu_device_count() }

#: Select the Metal device used by later allocations. Must be called before
#: the first buffer or kernel is created.
pub fn set_device(index: int) -> void {
    __prim__.gpu_set_device(index)
}

#: Compute GEMM and fused MLP graphs in float16, storing results in float32.
pub fn set_mixed_precision(enabled: bool) -> void {
    __prim__.gpu_set_mixed_precision(enabled)
}

#: Whether mixed-precision compute is currently enabled.
pub fn mixed_precision() -> bool { return __prim__.gpu_mixed_precision() }

#: `@gpu_kernel` / `@mps_graph` function flag (see `jaithon.compile.emit`).
pub const FN_GPU_KERNEL: int = 1024

#: Whether `target` was compiled with `@gpu_kernel` or `@mps_graph`.
pub fn is_gpu_kernel(target: any) -> bool {
    return __prim__.function_has_flag(target, FN_GPU_KERNEL)
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
    #: Raises `RuntimeError` when no device is available. Pass `handle` to
    #: adopt an existing native buffer (used by `view`).
    pub fn init(self, count: int, handle: int = -1) {
        if count <= 0 { throw ValueError(f"buffer count must be positive, got {count}") }
        self.count = count
        self.freed = false
        if handle < 0 {
            _require_device("allocate a Buffer")
            self.handle = __prim__.gpu_buffer_new(count)
        } else {
            self.handle = handle
        }
    }

    #: A window into this buffer. Freeing the view does not free the parent.
    pub fn view(self, offset: int, count: int) -> Buffer {
        self._check_live()
        if offset < 0 or count <= 0 or offset + count > self.count {
            throw ValueError(f"view of {count} values at {offset} exceeds capacity {self.count}")
        }
        return Buffer(count, __prim__.gpu_buffer_view(self.handle, offset, count))
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

    #: Expand `count` unsigned bytes into float slots without a host-side list.
    #:
    #: Each destination value is `float(data[source_offset + i]) * scale`.
    #: This is intended for byte datasets such as images and token ids, where
    #: constructing millions of boxed Jaithon floats would dominate loading.
    pub fn upload_u8(
        self,
        data: bytes,
        source_offset: int = 0,
        count: int? = null,
        offset: int = 0,
        scale: float = 1.0
    ) -> void {
        self._check_live()
        let wanted = count ?? data.len() - source_offset
        __prim__.gpu_buffer_upload_u8(self.handle, data, source_offset, wanted, offset, scale)
    }

    #: Fill every slot with a uniform sample in `[low, high)` on the device.
    pub fn fill_uniform(self, low: float, high: float, seed: int) -> void {
        self._check_live()
        if not low < high {
            throw ValueError(f"fill_uniform expects low < high, got [{low}, {high})")
        }
        let seed_lo = seed & 0xFFFFFFFF
        let seed_hi = seed >> 32 & 0xFFFFFFFF
        let kernel = Kernel.compile(_FILL_SOURCE, "fill_uniform")
        kernel.dispatch_async(
            [self],
            self.count,
            [self.count, _f32_bits(low), _f32_bits(high), seed_lo, seed_hi]
        )
    }

    #: Fill every slot with 0 on the device, without a host list.
    pub fn fill_zero(self) -> void {
        self._check_live()
        let kernel = Kernel.compile(_FILL_SOURCE, "fill_bits")
        kernel.dispatch_async([self], self.count, [self.count, 0])
    }

    #: Fill every slot with `value` on the device, without a host list.
    pub fn fill(self, value: float) -> void {
        self._check_live()
        if value == 0.0 {
            self.fill_zero()
            return
        }
        let kernel = Kernel.compile(_FILL_SOURCE, "fill_bits")
        kernel.dispatch_async([self], self.count, [self.count, _f32_bits(value)])
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
        #: Nested rather than keyed on `entry + source`: building that key
        #: copies and rehashes the whole shader library on every call, which
        #: for a 36 KB one costs more than eighty microseconds -- an order of
        #: magnitude more than encoding the dispatch it precedes.
        var byEntry = _kernel_intern.get(entry, null)
        if byEntry is null {
            byEntry = {}
            _kernel_intern[entry] = byEntry
        }
        let existing = byEntry.get(source, null)
        if existing is not null and not existing.freed { return existing }
        let kernel = Kernel(entry, __prim__.gpu_compile(source, entry))
        byEntry[source] = kernel
        return kernel
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

    #: Queue a kernel without waiting for it to finish.
    #:
    #: Async dispatches execute in submission order. Call `flush` to commit
    #: queued kernels so the GPU can start, and `synchronize` before
    #: downloading. In-flight command buffers retain their Metal resources, so
    #: a buffer may be `free`d after it has been encoded.
    pub fn dispatch_async(
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
        __prim__.gpu_dispatch_async(self.handle, handles, scalars, threads, width)
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

var _kernel_intern: dict[str, dict[str, Kernel]] = {}

#: Commit queued async kernels so the GPU can start them, without waiting.
#:
#: Returns immediately when no work is queued. A bounded number of command
#: buffers may stay in flight; this call waits only when that bound is hit,
#: and only for the oldest already-running buffer.
pub fn flush() -> void {
    _require_device("flush")
    __prim__.gpu_flush()
}

#: Wait until every asynchronously dispatched kernel has completed.
#:
#: Returns immediately when no work is queued. A failed command raises
#: `RuntimeError`; after this returns, its output buffers are safe to read.
pub fn synchronize() -> void {
    _require_device("synchronize")
    __prim__.gpu_synchronize()
}

#: Queue a device-buffer matrix product without waiting.
#:
#: `trans_a` / `trans_b` read `a` or `b` as transposed. Row bytes of each
#: physical matrix must be a multiple of 16 (width divisible by 4). Used by
#: jaitensor for GEMMs large enough that Metal Performance Shaders wins.
#:
#: `half` runs the product in half precision, converting on the way in and back
#: on the way out. Whether that is a gain depends on the shape — the conversion
#: touches the whole of the result, so a product with a large output and a
#: short inner dimension spends more on converting than it saves on
#: multiplying — which is why it is asked for per call rather than taken from
#: the device-wide mixed-precision setting.
pub fn matmul_async(
    a: Buffer,
    b: Buffer,
    output: Buffer,
    rows: int,
    inner: int,
    columns: int,
    trans_a: bool = false,
    trans_b: bool = false,
    half: bool = false
) -> void {
    _require_device("matmul_async")
    if rows < 0 or inner < 0 or columns < 0 {
        throw ValueError(
            f"matmul_async dimensions must be non-negative, got {rows}x{inner}x{columns}"
        )
    }
    __prim__.gpu_matmul_buffers(
        a.handle_of(),
        b.handle_of(),
        output.handle_of(),
        rows,
        inner,
        columns,
        trans_a,
        trans_b,
        half
    )
}

#: Packed multi-head attention. Q/K/V/out are `[seq, heads*hd]`.
#:
#: Returns false when the device has no fused SDPA path so callers can
#: fall back to a Metal kernel.
pub fn mha_async(
    query: Buffer,
    key: Buffer,
    value: Buffer,
    output: Buffer,
    seq: int,
    heads: int,
    hd: int,
    scale: float
) -> bool {
    _require_device("mha_async")
    if seq <= 0 or heads <= 0 or hd <= 0 {
        throw ValueError(f"mha_async dimensions must be positive, got {seq}x{heads}x{hd}")
    }
    if not scale > 0.0 {
        throw ValueError("mha_async scale must be positive")
    }
    return __prim__.gpu_mha_buffers(
        query.handle_of(),
        key.handle_of(),
        value.handle_of(),
        output.handle_of(),
        seq,
        heads,
        hd,
        scale
    )
}

#: Queue an NHWC 2D convolution without waiting.
#:
#: Weights are HWIO `[kh, kw, cin, cout]`. `bias` is optional `cout` values.
pub fn conv2d_async(
    input: Buffer,
    weights: Buffer,
    output: Buffer,
    n: int,
    height: int,
    width: int,
    cin: int,
    cout: int,
    kh: int,
    kw: int,
    stride_h: int = 1,
    stride_w: int = 1,
    pad_h: int = 0,
    pad_w: int = 0,
    bias: Buffer? = null
) -> void {
    _require_device("conv2d_async")
    if n <= 0 or height <= 0 or width <= 0 or cin <= 0 or cout <= 0 or kh <= 0 or kw <= 0 {
        throw ValueError("conv2d_async dimensions must be positive")
    }
    if stride_h <= 0 or stride_w <= 0 {
        throw ValueError("conv2d_async stride must be positive")
    }
    let bias_handle = if bias is null { null } else { bias.handle_of() }
    __prim__.gpu_conv2d_buffers(
        input.handle_of(),
        weights.handle_of(),
        bias_handle,
        output.handle_of(),
        n,
        height,
        width,
        cin,
        cout,
        kh,
        kw,
        stride_h,
        stride_w,
        pad_h,
        pad_w
    )
}

#: The convolution's gradient with respect to its input.
#:
#: `grad` is the gradient arriving at the output, `weights` the filter the
#: forward pass used, and `output` receives a tensor the shape of the input.
#: The dimensions are the FORWARD pass's, so the same numbers describe both
#: directions. False means this device or this shape has no such primitive and
#: the caller should fall back to its own.
pub fn conv2d_data_grad_async(
    grad: Buffer,
    weights: Buffer,
    output: Buffer,
    n: int,
    height: int,
    width: int,
    cin: int,
    cout: int,
    kh: int,
    kw: int,
    stride_h: int = 1,
    stride_w: int = 1,
    pad_h: int = 0,
    pad_w: int = 0
) -> bool {
    _require_device("conv2d_data_grad_async")
    if n <= 0 or height <= 0 or width <= 0 or cin <= 0 or cout <= 0 or kh <= 0 or kw <= 0 {
        throw ValueError("conv2d_data_grad_async dimensions must be positive")
    }
    if stride_h <= 0 or stride_w <= 0 {
        throw ValueError("conv2d_data_grad_async stride must be positive")
    }
    return __prim__.gpu_conv2d_data_grad(
        grad.handle_of(),
        weights.handle_of(),
        output.handle_of(),
        n,
        height,
        width,
        cin,
        cout,
        kh,
        kw,
        stride_h,
        stride_w,
        pad_h,
        pad_w
    )
}

#: The same convolution's gradient with respect to its weights. `output`
#: receives a tensor the shape of the filter.
pub fn conv2d_weights_grad_async(
    grad: Buffer,
    input: Buffer,
    output: Buffer,
    n: int,
    height: int,
    width: int,
    cin: int,
    cout: int,
    kh: int,
    kw: int,
    stride_h: int = 1,
    stride_w: int = 1,
    pad_h: int = 0,
    pad_w: int = 0
) -> bool {
    _require_device("conv2d_weights_grad_async")
    if n <= 0 or height <= 0 or width <= 0 or cin <= 0 or cout <= 0 or kh <= 0 or kw <= 0 {
        throw ValueError("conv2d_weights_grad_async dimensions must be positive")
    }
    if stride_h <= 0 or stride_w <= 0 {
        throw ValueError("conv2d_weights_grad_async stride must be positive")
    }
    return __prim__.gpu_conv2d_weights_grad(
        grad.handle_of(),
        input.handle_of(),
        output.handle_of(),
        n,
        height,
        width,
        cin,
        cout,
        kh,
        kw,
        stride_h,
        stride_w,
        pad_h,
        pad_w
    )
}

#: Fused two-layer ReLU MLP SGD step on device buffers.
#:
#: Updates `w1`/`b1`/`w2`/`b2` in place, adds the batch's total softmax
#: cross-entropy into `loss_acc`, and adds the correct-class count into
#: `correct_acc`.
pub fn mlp_sgd_step(
    x: Buffer,
    w1: Buffer,
    b1: Buffer,
    w2: Buffer,
    b2: Buffer,
    labels: Buffer,
    loss_acc: Buffer,
    batch: int,
    inputs: int,
    hidden: int,
    classes: int,
    learning_rate: float,
    correct_acc: Buffer? = null
) -> void {
    _require_device("mlp_sgd_step")
    if batch <= 0 or inputs <= 0 or hidden <= 0 or classes <= 0 {
        throw ValueError(
            f"mlp_sgd_step dimensions must be positive, got {batch}x{inputs}x{hidden}x{classes}"
        )
    }
    if learning_rate <= 0.0 {
        throw ValueError(f"mlp_sgd_step learning_rate must be positive, got {learning_rate}")
    }
    let correct_handle = if correct_acc is null { null } else { correct_acc.handle_of() }
    __prim__.gpu_mlp_sgd_step(
        x.handle_of(),
        w1.handle_of(),
        b1.handle_of(),
        w2.handle_of(),
        b2.handle_of(),
        labels.handle_of(),
        loss_acc.handle_of(),
        correct_handle,
        batch,
        inputs,
        hidden,
        classes,
        learning_rate
    )
}

#: Run every full batch of a two-layer fused SGD epoch without returning to
#: the VM between steps. Drops a ragged trailing batch. Returns samples used.
pub fn mlp_sgd_epoch(
    x: Buffer,
    w1: Buffer,
    b1: Buffer,
    w2: Buffer,
    b2: Buffer,
    labels: Buffer,
    loss_acc: Buffer,
    samples: int,
    batch: int,
    inputs: int,
    hidden: int,
    classes: int,
    learning_rate: float,
    flush_every: int = 4,
    correct_acc: Buffer? = null
) -> int {
    _require_device("mlp_sgd_epoch")
    if samples <= 0 or batch <= 0 or inputs <= 0 or hidden <= 0 or classes <= 0 or flush_every <= 0 {
        throw ValueError("mlp_sgd_epoch dimensions must be positive")
    }
    if learning_rate <= 0.0 {
        throw ValueError(f"mlp_sgd_epoch learning_rate must be positive, got {learning_rate}")
    }
    let correct_handle = if correct_acc is null { null } else { correct_acc.handle_of() }
    return __prim__.gpu_mlp_sgd_epoch(
        x.handle_of(),
        w1.handle_of(),
        b1.handle_of(),
        w2.handle_of(),
        b2.handle_of(),
        labels.handle_of(),
        loss_acc.handle_of(),
        correct_handle,
        samples,
        batch,
        inputs,
        hidden,
        classes,
        learning_rate,
        flush_every
    )
}

#: Fused two-layer ReLU MLP forward and backward into gradient buffers.
#:
#: Writes batch-mean parameter gradients into `g1`/`gb1`/`g2`/`gb2`, adds
#: the batch's total softmax cross-entropy into `loss_acc`, and optionally
#: adds the correct-class count into `correct_acc`.
pub fn mlp_bwd_step(
    x: Buffer,
    w1: Buffer,
    b1: Buffer,
    w2: Buffer,
    b2: Buffer,
    labels: Buffer,
    g1: Buffer,
    gb1: Buffer,
    g2: Buffer,
    gb2: Buffer,
    loss_acc: Buffer,
    batch: int,
    inputs: int,
    hidden: int,
    classes: int,
    correct_acc: Buffer? = null
) -> void {
    _require_device("mlp_bwd_step")
    if batch <= 0 or inputs <= 0 or hidden <= 0 or classes <= 0 {
        throw ValueError(
            f"mlp_bwd_step dimensions must be positive, got {batch}x{inputs}x{hidden}x{classes}"
        )
    }
    let correct_handle = if correct_acc is null { null } else { correct_acc.handle_of() }
    __prim__.gpu_mlp_bwd_step(
        x.handle_of(),
        w1.handle_of(),
        b1.handle_of(),
        w2.handle_of(),
        b2.handle_of(),
        labels.handle_of(),
        g1.handle_of(),
        gb1.handle_of(),
        g2.handle_of(),
        gb2.handle_of(),
        loss_acc.handle_of(),
        correct_handle,
        batch,
        inputs,
        hidden,
        classes
    )
}

#: Fused three-hidden-layer ReLU MLP SGD step on device buffers.
pub fn mlp3_sgd_step(
    x: Buffer,
    w1: Buffer,
    b1: Buffer,
    w2: Buffer,
    b2: Buffer,
    w3: Buffer,
    b3: Buffer,
    w4: Buffer,
    b4: Buffer,
    labels: Buffer,
    loss_acc: Buffer,
    batch: int,
    inputs: int,
    hidden: int,
    classes: int,
    learning_rate: float,
    correct_acc: Buffer? = null,
    hidden2: int = 0,
    hidden3: int = 0
) -> void {
    _require_device("mlp3_sgd_step")
    let h2 = if hidden2 == 0 { hidden } else { hidden2 }
    let h3 = if hidden3 == 0 { hidden } else { hidden3 }
    if batch <= 0 or inputs <= 0 or hidden <= 0 or h2 <= 0 or h3 <= 0 or classes <= 0 {
        throw ValueError(
            f"mlp3_sgd_step dimensions must be positive, got {batch}x{inputs}x{hidden}/{h2}/{h3}x{classes}"
        )
    }
    if learning_rate <= 0.0 {
        throw ValueError(f"mlp3_sgd_step learning_rate must be positive, got {learning_rate}")
    }
    let correct_handle = if correct_acc is null { null } else { correct_acc.handle_of() }
    __prim__.gpu_mlp3_sgd_step(
        x.handle_of(),
        w1.handle_of(),
        b1.handle_of(),
        w2.handle_of(),
        b2.handle_of(),
        w3.handle_of(),
        b3.handle_of(),
        w4.handle_of(),
        b4.handle_of(),
        labels.handle_of(),
        loss_acc.handle_of(),
        correct_handle,
        batch,
        inputs,
        hidden,
        h2,
        h3,
        classes,
        learning_rate
    )
}

#: Run every full batch of a three-hidden-layer fused SGD epoch in native code.
pub fn mlp3_sgd_epoch(
    x: Buffer,
    w1: Buffer,
    b1: Buffer,
    w2: Buffer,
    b2: Buffer,
    w3: Buffer,
    b3: Buffer,
    w4: Buffer,
    b4: Buffer,
    labels: Buffer,
    loss_acc: Buffer,
    samples: int,
    batch: int,
    inputs: int,
    hidden: int,
    classes: int,
    learning_rate: float,
    flush_every: int = 4,
    correct_acc: Buffer? = null,
    hidden2: int = 0,
    hidden3: int = 0
) -> int {
    _require_device("mlp3_sgd_epoch")
    let h2 = if hidden2 == 0 { hidden } else { hidden2 }
    let h3 = if hidden3 == 0 { hidden } else { hidden3 }
    if samples <= 0 or
        batch <= 0 or
        inputs <= 0 or
        hidden <= 0 or
        h2 <= 0 or
        h3 <= 0 or
        classes <= 0 or
        flush_every <= 0 {
        throw ValueError("mlp3_sgd_epoch dimensions must be positive")
    }
    if learning_rate <= 0.0 {
        throw ValueError(f"mlp3_sgd_epoch learning_rate must be positive, got {learning_rate}")
    }
    let correct_handle = if correct_acc is null { null } else { correct_acc.handle_of() }
    return __prim__.gpu_mlp3_sgd_epoch(
        x.handle_of(),
        w1.handle_of(),
        b1.handle_of(),
        w2.handle_of(),
        b2.handle_of(),
        w3.handle_of(),
        b3.handle_of(),
        w4.handle_of(),
        b4.handle_of(),
        labels.handle_of(),
        loss_acc.handle_of(),
        correct_handle,
        samples,
        batch,
        inputs,
        hidden,
        h2,
        h3,
        classes,
        learning_rate,
        flush_every
    )
}

#: Fused three-hidden-layer ReLU MLP forward and backward into gradient buffers.
pub fn mlp3_bwd_step(
    x: Buffer,
    w1: Buffer,
    b1: Buffer,
    w2: Buffer,
    b2: Buffer,
    w3: Buffer,
    b3: Buffer,
    w4: Buffer,
    b4: Buffer,
    labels: Buffer,
    g1: Buffer,
    gb1: Buffer,
    g2: Buffer,
    gb2: Buffer,
    g3: Buffer,
    gb3: Buffer,
    g4: Buffer,
    gb4: Buffer,
    loss_acc: Buffer,
    batch: int,
    inputs: int,
    hidden: int,
    classes: int,
    correct_acc: Buffer? = null,
    hidden2: int = 0,
    hidden3: int = 0
) -> void {
    _require_device("mlp3_bwd_step")
    let h2 = if hidden2 == 0 { hidden } else { hidden2 }
    let h3 = if hidden3 == 0 { hidden } else { hidden3 }
    if batch <= 0 or inputs <= 0 or hidden <= 0 or h2 <= 0 or h3 <= 0 or classes <= 0 {
        throw ValueError(
            f"mlp3_bwd_step dimensions must be positive, got {batch}x{inputs}x{hidden}/{h2}/{h3}x{classes}"
        )
    }
    let correct_handle = if correct_acc is null { null } else { correct_acc.handle_of() }
    __prim__.gpu_mlp3_bwd_step(
        x.handle_of(),
        w1.handle_of(),
        b1.handle_of(),
        w2.handle_of(),
        b2.handle_of(),
        w3.handle_of(),
        b3.handle_of(),
        w4.handle_of(),
        b4.handle_of(),
        labels.handle_of(),
        g1.handle_of(),
        gb1.handle_of(),
        g2.handle_of(),
        gb2.handle_of(),
        g3.handle_of(),
        gb3.handle_of(),
        g4.handle_of(),
        gb4.handle_of(),
        loss_acc.handle_of(),
        correct_handle,
        batch,
        inputs,
        hidden,
        h2,
        h3,
        classes
    )
}

#: True when every class-index label in the buffer is an integer in `[0, classes)`.
pub fn labels_valid(labels: Buffer, count: int, classes: int) -> bool {
    _require_device("labels_valid")
    if count < 0 or classes <= 0 {
        throw ValueError(f"labels_valid needs a non-negative count and positive classes")
    }
    return __prim__.gpu_labels_valid(labels.handle_of(), count, classes)
}

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

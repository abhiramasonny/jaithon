#: The two device resources: a block of device memory and a compiled Metal
#: pipeline.
from std.gpu.device import _require_device

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
    #: `count` may be zero. A tensor with an empty axis is a real thing -- an
    #: exported model slices one out whenever a range comes up empty -- and it
    #: has to be passable to the same code as any other. One slot is allocated
    #: behind it so that binding it to a kernel is still legal; nothing reads
    #: that slot, because every loop over it runs zero times.
    #:
    #: Raises `RuntimeError` when no device is available. Pass `handle` to
    #: adopt an existing native buffer (used by `view`).
    pub fn init(self, count: int, handle: int = -1) {
        if count < 0 { throw ValueError(f"buffer count cannot be negative, got {count}") }
        self.count = count
        self.freed = false
        if handle < 0 {
            _require_device("allocate a Buffer")
            self.handle = __prim__.gpu_buffer_new(max(count, 1))
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

    #: Read `count` slots back as clamped unsigned bytes.
    #:
    #: The inverse of `upload_u8`: each slot is divided by `scale`, rounded
    #: half up and clamped into `0..255`. `download` is the general reader, but
    #: it builds a list, and a list element costs far more than a byte -- for
    #: image data, where the values are bytes anyway, this moves a 720p frame
    #: in a fraction of a millisecond against eight for the list.
    pub fn download_u8(self, offset: int = 0, count: int? = null, scale: float = 1.0) -> bytes {
        self._check_live()
        if scale == 0.0 { throw ValueError("download_u8 scale must not be zero") }
        let wanted = count ?? self.count - offset
        return __prim__.gpu_buffer_download_u8(self.handle, offset, wanted, scale)
    }

    #: Read `count` slots back as one `0xAARRGGBB` integer per pixel.
    #:
    #: What a window's back buffer holds, in one pass. Narrowing to bytes and
    #: then packing those is two walks over the frame with a byte string in
    #: between that nothing else wants.
    pub fn pack_argb(
        self,
        offset: int = 0,
        count: int? = null,
        channels: int = 3,
        scale: float = 1.0
    ) -> list[int] {
        self._check_live()
        if scale == 0.0 { throw ValueError("pack_argb scale must not be zero") }
        let wanted = count ?? self.count - offset
        return __prim__.gpu_buffer_pack_argb(self.handle, offset, wanted, channels, scale)
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
        if threads < 0 { throw ValueError(f"threads cannot be negative, got {threads}") }
        if group_size < 0 { throw ValueError(f"group_size must be non-negative, got {group_size}") }
        #: Nothing to run over is not an error, it is nothing to run: an empty
        #: tensor reaches the same code an occupied one does.
        if threads == 0 { return }
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

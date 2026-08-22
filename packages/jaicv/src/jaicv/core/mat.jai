#: `Mat` — the dense 2-D array every other module reads and writes.
#:
#: Storage is one float32 device buffer per allocation. A `Mat` is a window
#: onto it: an element offset and a row stride, so `roi`, `row` and `col_range`
#: alias their parent the way OpenCV's do, and writing through one is visible
#: in the other.
#:
#: The declared depth (`CV_8U`, `CV_16S`, ...) governs rounding and saturation,
#: not the width of the storage. Every integer depth up to 24 bits is exact in
#: float32; `CV_32S` is exact to 2^24 and `CV_64F` is a source-compatible alias
#: for float32, not a wider element.
#:
#: ```
#: let image = Mat(480, 640, CV_8UC3)
#: image.set_to(Scalar(255.0, 0.0, 0.0))
#: image.at(10, 20, 0)      # 255.0
#: ```
import std.gpu as gpu
import std.math
from std.gpu import Buffer, Kernel
from std.core import Printable
from .types import CV_32F, CV_MAKETYPE, CV_MAT_CN, CV_MAT_DEPTH
from .types import Rect, Scalar, Size
from .types import depth_is_integer, depth_max, depth_min, type_to_string

let SOURCE = """#include <metal_stdlib>
using namespace metal;

inline uint plane_index(uint id, uint cols, uint cn, uint off, uint step) {
    uint width = cols * cn;
    uint y = id / width;
    uint rest = id - y * width;
    return off + y * step + rest;
}

/* cvRound is round-to-nearest with ties to even, which `rint` is. */
inline float saturate_to(float value, float low, float high, uint integral) {
    float v = value;
    if (integral != 0u) {
        v = rint(v);
        v = clamp(v, low, high);
    }
    return v;
}

kernel void mat_fill(device float* dst [[buffer(0)]],
                     constant uint& count [[buffer(1)]],
                     constant uint& cols [[buffer(2)]],
                     constant uint& cn [[buffer(3)]],
                     constant uint& off [[buffer(4)]],
                     constant uint& step [[buffer(5)]],
                     constant uint& c0 [[buffer(6)]],
                     constant uint& c1 [[buffer(7)]],
                     constant uint& c2 [[buffer(8)]],
                     constant uint& c3 [[buffer(9)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint channel = id % cn;
    float value = as_type<float>(c0);
    if (channel == 1u) value = as_type<float>(c1);
    else if (channel == 2u) value = as_type<float>(c2);
    else if (channel == 3u) value = as_type<float>(c3);
    dst[plane_index(id, cols, cn, off, step)] = value;
}

kernel void mat_fill_masked(device float* dst [[buffer(0)]],
                            device const float* mask [[buffer(1)]],
                            constant uint& count [[buffer(2)]],
                            constant uint& cols [[buffer(3)]],
                            constant uint& cn [[buffer(4)]],
                            constant uint& off [[buffer(5)]],
                            constant uint& step [[buffer(6)]],
                            constant uint& mask_off [[buffer(7)]],
                            constant uint& mask_step [[buffer(8)]],
                            constant uint& c0 [[buffer(9)]],
                            constant uint& c1 [[buffer(10)]],
                            constant uint& c2 [[buffer(11)]],
                            constant uint& c3 [[buffer(12)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint channel = id % cn;
    uint pixel = id / cn;
    if (mask[plane_index(pixel, cols, 1u, mask_off, mask_step)] == 0.0f) return;
    float value = as_type<float>(c0);
    if (channel == 1u) value = as_type<float>(c1);
    else if (channel == 2u) value = as_type<float>(c2);
    else if (channel == 3u) value = as_type<float>(c3);
    dst[plane_index(id, cols, cn, off, step)] = value;
}

kernel void mat_copy(device const float* src [[buffer(0)]],
                     device float* dst [[buffer(1)]],
                     constant uint& count [[buffer(2)]],
                     constant uint& cols [[buffer(3)]],
                     constant uint& cn [[buffer(4)]],
                     constant uint& src_off [[buffer(5)]],
                     constant uint& src_step [[buffer(6)]],
                     constant uint& dst_off [[buffer(7)]],
                     constant uint& dst_step [[buffer(8)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    dst[plane_index(id, cols, cn, dst_off, dst_step)] =
        src[plane_index(id, cols, cn, src_off, src_step)];
}

kernel void mat_copy_masked(device const float* src [[buffer(0)]],
                            device float* dst [[buffer(1)]],
                            device const float* mask [[buffer(2)]],
                            constant uint& count [[buffer(3)]],
                            constant uint& cols [[buffer(4)]],
                            constant uint& cn [[buffer(5)]],
                            constant uint& src_off [[buffer(6)]],
                            constant uint& src_step [[buffer(7)]],
                            constant uint& dst_off [[buffer(8)]],
                            constant uint& dst_step [[buffer(9)]],
                            constant uint& mask_off [[buffer(10)]],
                            constant uint& mask_step [[buffer(11)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint pixel = id / cn;
    if (mask[plane_index(pixel, cols, 1u, mask_off, mask_step)] == 0.0f) return;
    dst[plane_index(id, cols, cn, dst_off, dst_step)] =
        src[plane_index(id, cols, cn, src_off, src_step)];
}

kernel void mat_convert(device const float* src [[buffer(0)]],
                        device float* dst [[buffer(1)]],
                        constant uint& count [[buffer(2)]],
                        constant uint& cols [[buffer(3)]],
                        constant uint& cn [[buffer(4)]],
                        constant uint& src_off [[buffer(5)]],
                        constant uint& src_step [[buffer(6)]],
                        constant uint& dst_off [[buffer(7)]],
                        constant uint& dst_step [[buffer(8)]],
                        constant uint& alpha_bits [[buffer(9)]],
                        constant uint& beta_bits [[buffer(10)]],
                        constant uint& low_bits [[buffer(11)]],
                        constant uint& high_bits [[buffer(12)]],
                        constant uint& integral [[buffer(13)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float value = src[plane_index(id, cols, cn, src_off, src_step)];
    value = value * as_type<float>(alpha_bits) + as_type<float>(beta_bits);
    dst[plane_index(id, cols, cn, dst_off, dst_step)] =
        saturate_to(value, as_type<float>(low_bits), as_type<float>(high_bits), integral);
}

kernel void mat_extract_channel(device const float* src [[buffer(0)]],
                                device float* dst [[buffer(1)]],
                                constant uint& count [[buffer(2)]],
                                constant uint& cols [[buffer(3)]],
                                constant uint& cn [[buffer(4)]],
                                constant uint& src_off [[buffer(5)]],
                                constant uint& src_step [[buffer(6)]],
                                constant uint& dst_off [[buffer(7)]],
                                constant uint& dst_step [[buffer(8)]],
                                constant uint& channel [[buffer(9)]],
                                uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;
    dst[dst_off + y * dst_step + x] = src[src_off + y * src_step + x * cn + channel];
}

kernel void mat_insert_channel(device const float* src [[buffer(0)]],
                               device float* dst [[buffer(1)]],
                               constant uint& count [[buffer(2)]],
                               constant uint& cols [[buffer(3)]],
                               constant uint& cn [[buffer(4)]],
                               constant uint& src_off [[buffer(5)]],
                               constant uint& src_step [[buffer(6)]],
                               constant uint& dst_off [[buffer(7)]],
                               constant uint& dst_step [[buffer(8)]],
                               constant uint& channel [[buffer(9)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint y = id / cols;
    uint x = id - y * cols;
    dst[dst_off + y * dst_step + x * cn + channel] = src[src_off + y * src_step + x];
}

kernel void mat_transpose(device const float* src [[buffer(0)]],
                          device float* dst [[buffer(1)]],
                          constant uint& count [[buffer(2)]],
                          constant uint& rows [[buffer(3)]],
                          constant uint& cols [[buffer(4)]],
                          constant uint& cn [[buffer(5)]],
                          constant uint& src_off [[buffer(6)]],
                          constant uint& src_step [[buffer(7)]],
                          constant uint& dst_off [[buffer(8)]],
                          constant uint& dst_step [[buffer(9)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint channel = id % cn;
    uint pixel = id / cn;
    uint y = pixel / cols;
    uint x = pixel - y * cols;
    dst[dst_off + x * dst_step + y * cn + channel] =
        src[src_off + y * src_step + x * cn + channel];
}

kernel void mat_flip(device const float* src [[buffer(0)]],
                     device float* dst [[buffer(1)]],
                     constant uint& count [[buffer(2)]],
                     constant uint& rows [[buffer(3)]],
                     constant uint& cols [[buffer(4)]],
                     constant uint& cn [[buffer(5)]],
                     constant uint& src_off [[buffer(6)]],
                     constant uint& src_step [[buffer(7)]],
                     constant uint& dst_off [[buffer(8)]],
                     constant uint& dst_step [[buffer(9)]],
                     constant uint& mode [[buffer(10)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint channel = id % cn;
    uint pixel = id / cn;
    uint y = pixel / cols;
    uint x = pixel - y * cols;
    uint sy = (mode == 0u || mode == 2u) ? (rows - 1u - y) : y;
    uint sx = (mode == 1u || mode == 2u) ? (cols - 1u - x) : x;
    dst[dst_off + y * dst_step + x * cn + channel] =
        src[src_off + sy * src_step + sx * cn + channel];
}
"""

var _kernels: dict[str, Kernel] = {}

#: The compiled pipeline for one entry point in the core shader library.
pub fn cached_kernel(entry: str) -> Kernel {
    let existing = _kernels.get(entry, null)
    if existing is not null { return existing }
    let kernel = Kernel.compile(SOURCE, entry)
    _kernels[entry] = kernel
    return kernel
}

#: Reinterpret a float's bits as an integer, so a kernel can take it as a
#: `constant uint&` and read it back with `as_type<float>`.
pub fn f32_bits(value: float) -> int {
    if value != value { return 0x7FC00000 }
    var magnitude = value
    var sign = 0
    if magnitude < 0.0 {
        sign = 1
        magnitude = -magnitude
    }
    if magnitude == 0.0 { return sign << 31 }
    if magnitude >= 3.4028234663852886e38 { return sign << 31 | 0x7F800000 }
    if magnitude <= 1.401298464324817e-45 { return sign << 31 }
    let split = __prim__.f64_frexp(magnitude)
    let biased = split[1] + 126
    if biased <= 0 {
        let fraction = int(__prim__.f64_ldexp(magnitude, 149))
        return sign << 31 | fraction & 0x7FFFFF
    }
    if biased >= 255 { return sign << 31 | 0x7F800000 }
    let fraction = int((split[0] * 2.0 - 1.0) * 8388608.0 + 0.5)
    return sign << 31 | biased << 23 | fraction & 0x7FFFFF
}

var _generations: dict[int, int] = {}

#: Bump the revision of an allocation, so every window onto it reloads its host
#: mirror before the next read.
#:
#: The mirror cannot live on the `Mat`: `roi` hands out several windows onto one
#: allocation and a write through any of them invalidates all of them. Keyed on
#: the buffer handle, which every window of an allocation shares.
pub fn touch(storage: Buffer) -> void {
    let key = storage.handle_of()
    _generations[key] = _generations.get(key, 0) + 1 & 0x3FFFFFFF
}

fn generation_of(storage: Buffer) -> int { return _generations.get(storage.handle_of(), 0) }

fn require_device(what: str) -> void {
    if not gpu.is_available() {
        throw RuntimeError(f"jaicv needs a GPU to {what}")
    }
}

#: A dense two-dimensional array of pixels.
pub class Mat: Printable {
    #: Number of rows.
    pub let rows: int
    #: Number of columns.
    pub let cols: int
    #: Type code, as `CV_MAKETYPE(depth, channels)` builds.
    pub let flags: int
    #: The allocation this matrix is a window onto, shared with every other
    #: window of it.
    pub let storage: Buffer
    #: Element offset of the first pixel inside `storage`.
    pub let offset: int
    #: Elements between one row and the next inside `storage`.
    pub let step_: int
    var host: list[float]?
    var host_generation: int
    var pending: bool
    var freed: bool

    #: Allocate `rows x cols` pixels of `type_code`, uninitialised.
    #:
    #: Pass `storage`, `offset` and `step` to build a window onto an existing
    #: allocation instead; `roi` and `row` are the callers that do.
    pub fn init(
        self,
        rows: int = 0,
        cols: int = 0,
        type_code: int = 0,
        storage: Buffer? = null,
        offset: int = 0,
        step: int = 0
    ) {
        if rows < 0 or cols < 0 {
            throw ValueError(f"a Mat cannot have {rows} rows and {cols} columns")
        }
        self.rows = rows
        self.cols = cols
        self.flags = type_code
        self.offset = offset
        self.host = null
        self.host_generation = -1
        self.pending = false
        self.freed = false
        let channels = CV_MAT_CN(type_code)
        self.step_ = if step > 0 { step } else { cols * channels }
        if storage is not null {
            self.storage = storage
            return
        }
        require_device("allocate a Mat")
        let count = max(1, rows * cols * channels)
        self.storage = Buffer(count)
    }

    #: A matrix of zeros.
    pub static fn zeros(rows: int, cols: int, type_code: int) -> Mat {
        let mat = Mat(rows, cols, type_code)
        mat.storage.fill_zero()
        return mat
    }

    #: A matrix of ones.
    pub static fn ones(rows: int, cols: int, type_code: int) -> Mat {
        let mat = Mat(rows, cols, type_code)
        mat.storage.fill(1.0)
        return mat
    }

    #: A matrix filled with `value`.
    pub static fn full(rows: int, cols: int, type_code: int, value: float) -> Mat {
        let mat = Mat(rows, cols, type_code)
        mat.storage.fill(value)
        return mat
    }

    #: An identity matrix, ones down the leading diagonal.
    pub static fn eye(rows: int, cols: int, type_code: int) -> Mat {
        let channels = CV_MAT_CN(type_code)
        var values: list[float] = []
        for _index in 0..rows * cols * channels { values.push(0.0) }
        let limit = min(rows, cols)
        for index in 0..limit {
            for channel in 0..channels { values[(index * cols + index) * channels + channel] = 1.0 }
        }
        return Mat.from_list(rows, cols, type_code, values)
    }

    #: Build from row-major values, one per channel per pixel.
    pub static fn from_list(rows: int, cols: int, type_code: int, values: list[float]) -> Mat {
        let channels = CV_MAT_CN(type_code)
        if values.len() != rows * cols * channels {
            throw ValueError(
                f"{values.len()} values cannot fill {rows}x{cols} with {channels} channels"
            )
        }
        let mat = Mat(rows, cols, type_code)
        mat.storage.upload(values)
        return mat
    }

    #: Build from packed unsigned bytes, the layout a camera or a decoder hands
    #: over. `scale` multiplies each byte on the way in.
    pub static fn from_bytes(
        rows: int,
        cols: int,
        type_code: int,
        data: bytes,
        source_offset: int = 0,
        scale: float = 1.0
    ) -> Mat {
        let channels = CV_MAT_CN(type_code)
        let count = rows * cols * channels
        if source_offset + count > data.len() {
            throw ValueError(
                f"{data.len()} bytes cannot fill {rows}x{cols}x{channels} from {source_offset}"
            )
        }
        let mat = Mat(rows, cols, type_code)
        mat.storage.upload_u8(data, source_offset, count, 0, scale)
        return mat
    }

    #: A 1 x n matrix holding `values`, the shape OpenCV uses for point lists.
    pub static fn row_vector(values: list[float], channels: int = 1) -> Mat {
        let count = values.len() // channels
        return Mat.from_list(1, count, CV_MAKETYPE(CV_32F, channels), values)
    }

    #: Number of channels per pixel.
    pub fn channels(self) -> int { return CV_MAT_CN(self.flags) }

    #: Element depth, one of the `CV_8U`-style constants.
    pub fn depth(self) -> int { return CV_MAT_DEPTH(self.flags) }

    #: Full type code.
    pub fn type_(self) -> int { return self.flags }

    #: Number of pixels, channels not counted.
    pub fn total(self) -> int { return self.rows * self.cols }

    #: Extent as a `Size`, width first.
    pub fn size(self) -> Size { return Size(self.cols, self.rows) }

    #: Whether the matrix holds no pixels.
    pub fn empty(self) -> bool { return self.rows == 0 or self.cols == 0 }

    #: Whether rows follow each other without a gap.
    pub fn is_continuous(self) -> bool { return self.step_ == self.cols * self.channels() }

    #: Elements between one row and the next in the underlying allocation.
    pub fn step(self) -> int { return self.step_ }

    #: Element offset of the first pixel in the underlying allocation.
    pub fn data_offset(self) -> int { return self.offset }

    #: The allocation this matrix windows. Shared with every view of it.
    #: The device buffer, with any host-side edits pushed to it first.
    #:
    #: Flushing HERE rather than at the end of every edit is what makes
    #: drawing usable: a `line` or a `put_text` writes into the host mirror,
    #: and eagerly uploading after each one cost a `gpu.synchronize()` plus a
    #: whole-image upload per primitive -- 1.04 ms for a single line on a
    #: 640x480 frame. Deferring it means a run of primitives shares one
    #: upload, taken the moment something actually reads the device. `flush`
    #: returns immediately when nothing is pending, so this costs a bool test
    #: on the path that has no edits to push.
    pub fn buffer(self) -> Buffer {
        self.flush()
        return self.storage
    }

    #: Number of elements the underlying allocation holds.
    pub fn buffer_count(self) -> int { return self.storage.count }

    #: Whether the pixels are already on the host, so that reading or writing
    #: them there costs nothing.
    #:
    #: A caller with both a host and a device way of doing its work uses this
    #: to pick: building the mirror for a frame that has none costs a download
    #: and a list of one boxed number per channel per pixel, which for a 720p
    #: frame is 44 MB and about 2 ms.
    pub fn has_host_copy(self) -> bool {
        let cached = self.host
        return cached is not null and self.host_generation == generation_of(self.storage)
    }

    #: Whether host-side edits are waiting to be pushed to the device.
    pub fn has_pending_edits(self) -> bool { return self.pending }

    #: Whether `other` has the same shape and type.
    pub fn same_kind(self, other: Mat) -> bool {
        return self.rows == other.rows and self.cols == other.cols and self.flags == other.flags
    }

    fn check_live(self) -> void {
        if self.freed { throw RuntimeError("this Mat has already been released") }
    }

    #: Drop every window's host mirror after a device write.
    pub fn invalidate(self) -> void {
        self.host = null
        touch(self.storage)
    }

    fn ensure_host(self) -> list[float] {
        self.check_live()
        let cached = self.host
        if cached is not null and self.host_generation == generation_of(self.storage) {
            return cached
        }
        #: The download waits for the work that wrote this buffer and leaves
        #: everything queued behind it running. Draining the queue here instead
        #: would make reading one frame wait for the next frame's network, so a
        #: loop that queues ahead would get no overlap at all.
        let width = self.cols * self.channels()
        var values: list[float] = []
        if self.is_continuous() and self.offset == 0 and self.storage.count == self.rows * width {
            values = self.storage.download()
        } else {
            for y in 0..self.rows {
                for value in self.storage.download(self.offset + y * self.step_, width) {
                    values.push(value)
                }
            }
        }
        self.host = values
        self.host_generation = generation_of(self.storage)
        return values
    }

    #: Every element in row-major order, channels interleaved.
    #: A COPY of every element, in row-major order.
    #:
    #: The copy is not free -- at VGA it is five milliseconds against one and a
    #: third for the download it protects -- so anything inside this library
    #: that only reads should call `values` instead. This stays copying because
    #: it is the public accessor and a caller who mutates what it hands back
    #: would corrupt the mirror every later read comes from.
    pub fn to_list(self) -> list[float] { return [value for value in self.ensure_host()] }

    #: The live host mirror itself, without the copy. READ ONLY.
    #:
    #: Writing through this corrupts the cache that every later read shares --
    #: `edit` is the way to change pixels. It exists because copying is most of
    #: what reading an image cost: a 640x480 `to_list` was seven milliseconds,
    #: of which five and a half were the copy, and every host-side algorithm in
    #: this library begins with one.
    pub fn values(self) -> list[float] { return self.ensure_host() }

    #: Every element clamped and rounded into bytes, the layout an encoder or a
    #: display expects.
    pub fn to_bytes(self, scale: float = 1.0) -> bytes {
        #: One primitive rather than a loop. Written here it was a multiply, a
        #: NaN test, a `math.floor`, an `int` and two compares an element, and
        #: the call to `math.floor` is what stopped the loop compiling: a
        #: 640x480 frame took thirty-nine milliseconds, which put a
        #: twenty-five frame ceiling on writing video before a single frame was
        #: encoded. The same work in C is a third of a millisecond.
        self.check_live()
        let cached = self.host
        if cached is not null and self.host_generation == generation_of(self.storage) {
            return __prim__.bytes_quantise(cached, scale)
        }
        #: Nothing on the host to narrow, so the device narrows on the way out
        #: instead of handing over floats to be narrowed here. `ensure_host`
        #: would build a list first, and a list element costs far more than the
        #: byte it ends up as: a 720p frame is 2.8 million of them, eight
        #: milliseconds of boxing for data that was bytes to begin with.
        #:
        #: `download_u8` divides where `bytes_quantise` multiplies -- it is the
        #: inverse of `upload_u8`, which is the pair a reader of `std.gpu`
        #: expects -- so the reciprocal goes in.
        if self.is_continuous() and self.offset == 0 {
            let wanted = self.rows * self.cols * self.channels()
            if self.storage.count == wanted {
                return self.storage.download_u8(0, wanted, 1.0 / scale)
            }
        }
        return __prim__.bytes_quantise(self.ensure_host(), scale)
    }

    #: One channel of one pixel.
    pub fn at(self, y: int, x: int, channel: int = 0) -> float {
        let cn = self.channels()
        if y < 0 or y >= self.rows or x < 0 or x >= self.cols or channel < 0 or channel >= cn {
            throw IndexError(f"({y}, {x}, {channel}) is outside a {self.rows}x{self.cols} Mat")
        }
        return self.ensure_host()[(y * self.cols + x) * cn + channel]
    }

    #: Every channel of one pixel.
    pub fn at_scalar(self, y: int, x: int) -> Scalar {
        let cn = self.channels()
        let base = (y * self.cols + x) * cn
        let values = self.ensure_host()
        var out = [0.0, 0.0, 0.0, 0.0]
        for channel in 0..min(cn, 4) { out[channel] = values[base + channel] }
        return Scalar(out[0], out[1], out[2], out[3])
    }

    #: Write one channel of one pixel. Buffered until `flush` or the next read.
    pub fn set_at(self, y: int, x: int, value: float, channel: int = 0) -> void {
        let cn = self.channels()
        if y < 0 or y >= self.rows or x < 0 or x >= self.cols or channel < 0 or channel >= cn {
            throw IndexError(f"({y}, {x}, {channel}) is outside a {self.rows}x{self.cols} Mat")
        }
        let values = self.ensure_host()
        values[(y * self.cols + x) * cn + channel] = value
        self.pending = true
    }

    #: Write every channel of one pixel.
    pub fn set_scalar(self, y: int, x: int, colour: Scalar) -> void {
        let cn = self.channels()
        for channel in 0..min(cn, 4) { self.set_at(y, x, colour[channel], channel: channel) }
    }

    #: The live host mirror, for a caller that writes many pixels at once.
    #:
    #: Writes through the returned list land in the mirror and reach the device
    #: on the next `flush`. Rasterising a shape through `set_at` would re-check
    #: the mirror once per pixel; the drawing module takes the list once.
    #: Say that the host mirror has been written to without pushing it now.
    #:
    #: `edit` already sets this, so a caller that went through it and then
    #: wrote is covered; this exists for a caller that wants to say so
    #: explicitly after finishing a run of edits.
    pub fn mark_edited(self) -> void {
        self.check_live()
        self.pending = true
    }

    pub fn edit(self) -> list[float] {
        let values = self.ensure_host()
        self.pending = true
        return values
    }

    #: Push buffered `set_at` writes back to the device.
    pub fn flush(self) -> void {
        if not self.pending { return }
        let values = self.host
        if values is null {
            self.pending = false
            return
        }
        #: An upload lands immediately while queued kernels have not run, so
        #: anything already encoded against this buffer would overwrite it. The
        #: upload waits for exactly that and nothing else.
        let width = self.cols * self.channels()
        if self.is_continuous() and self.offset == 0 and self.storage.count == self.rows * width {
            self.storage.upload(values)
        } else {
            for y in 0..self.rows {
                var row: list[float] = []
                for index in 0..width { row.push(values[y * width + index]) }
                self.storage.upload(row, self.offset + y * self.step_)
            }
        }
        self.pending = false
        touch(self.storage)
        self.host_generation = generation_of(self.storage)
    }

    #: An independent copy.
    pub fn clone(self) -> Mat {
        let out = Mat(self.rows, self.cols, self.flags)
        self.copy_to(out)
        return out
    }

    #: Copy into `dst`, which is reshaped when it does not already match.
    #:
    #: With a `mask`, only the pixels it marks non-zero are written.
    pub fn copy_to(self, dst: Mat, mask: Mat? = null) -> Mat {
        self.flush()
        if not dst.same_kind(self) {
            throw ValueError(
                f"copy_to needs a {self.rows}x{self.cols} {type_to_string(self.flags)} destination"
            )
        }
        let count = self.rows * self.cols * self.channels()
        if count == 0 { return dst }
        if mask is null {
            let kernel = cached_kernel("mat_copy")
            kernel.dispatch_async(
                [self.storage, dst.storage],
                count,
                [count, self.cols, self.channels(), self.offset, self.step_, dst.offset, dst.step_]
            )
        } else {
            mask.flush()
            let kernel = cached_kernel("mat_copy_masked")
            kernel.dispatch_async(
                [self.storage, dst.storage, mask.storage],
                count,
                [
                    count,
                    self.cols,
                    self.channels(),
                    self.offset,
                    self.step_,
                    dst.offset,
                    dst.step_,
                    mask.offset,
                    mask.step_,
                ]
            )
        }
        dst.invalidate()
        return dst
    }

    #: Set every element, or every element the `mask` marks, to `colour`.
    pub fn set_to(self, colour: Scalar, mask: Mat? = null) -> Mat {
        self.flush()
        let count = self.rows * self.cols * self.channels()
        if count == 0 { return self }
        let bits = [f32_bits(colour[index]) for index in 0..4]
        if mask is null {
            let kernel = cached_kernel("mat_fill")
            kernel.dispatch_async(
                [self.storage],
                count,
                [
                    count,
                    self.cols,
                    self.channels(),
                    self.offset,
                    self.step_,
                    bits[0],
                    bits[1],
                    bits[2],
                    bits[3],
                ]
            )
        } else {
            mask.flush()
            let kernel = cached_kernel("mat_fill_masked")
            kernel.dispatch_async(
                [self.storage, mask.storage],
                count,
                [
                    count,
                    self.cols,
                    self.channels(),
                    self.offset,
                    self.step_,
                    mask.offset,
                    mask.step_,
                    bits[0],
                    bits[1],
                    bits[2],
                    bits[3],
                ]
            )
        }
        self.invalidate()
        return self
    }

    #: Convert to `type_code`, scaling by `alpha` and offsetting by `beta`.
    #:
    #: An integer destination rounds to nearest and saturates, as OpenCV's
    #: `convertTo` does.
    pub fn convert_to(self, type_code: int = -1, alpha: float = 1.0, beta: float = 0.0) -> Mat {
        self.flush()
        let target = if type_code < 0 { self.flags } else { type_code }
        if CV_MAT_CN(target) != self.channels() {
            throw ValueError("convert_to cannot change the channel count")
        }
        let out = Mat(self.rows, self.cols, target)
        let count = self.rows * self.cols * self.channels()
        if count == 0 { return out }
        let depth = CV_MAT_DEPTH(target)
        let integral = if depth_is_integer(depth) { 1 } else { 0 }
        let kernel = cached_kernel("mat_convert")
        kernel.dispatch_async(
            [self.storage, out.storage],
            count,
            [
                count,
                self.cols,
                self.channels(),
                self.offset,
                self.step_,
                out.offset,
                out.step_,
                f32_bits(alpha),
                f32_bits(beta),
                f32_bits(depth_min(depth)),
                f32_bits(depth_max(depth)),
                integral,
            ]
        )
        return out
    }

    #: A window onto the pixels `rect` covers. Writing through it writes here.
    pub fn roi(self, rect: Rect) -> Mat {
        if rect.x < 0 or
            rect.y < 0 or
            rect.x + rect.width > self.cols or
            rect.y + rect.height > self.rows {
            throw ValueError(f"{rect} does not fit inside a {self.rows}x{self.cols} Mat")
        }
        self.flush()
        return Mat(
            rect.height,
            rect.width,
            self.flags,
            storage: self.storage,
            offset: self.offset + rect.y * self.step_ + rect.x * self.channels(),
            step: self.step_
        )
    }

    #: A window onto one row.
    pub fn row(self, y: int) -> Mat { return self.roi(Rect(0, y, self.cols, 1)) }

    #: A window onto one column.
    pub fn col(self, x: int) -> Mat { return self.roi(Rect(x, 0, 1, self.rows)) }

    #: A window onto the rows in `[start, end)`.
    pub fn row_range(self, start: int, end: int) -> Mat {
        return self.roi(Rect(0, start, self.cols, end - start))
    }

    #: A window onto the columns in `[start, end)`.
    pub fn col_range(self, start: int, end: int) -> Mat {
        return self.roi(Rect(start, 0, end - start, self.rows))
    }

    #: Reinterpret the same elements with a different channel count and row
    #: count, without moving anything. Only a continuous matrix can reshape.
    pub fn reshape(self, channels: int = 0, rows: int = 0) -> Mat {
        if not self.is_continuous() {
            throw ValueError("reshape needs a continuous Mat; clone it first")
        }
        self.flush()
        let cn = if channels <= 0 { self.channels() } else { channels }
        let total = self.rows * self.cols * self.channels()
        if total % cn != 0 {
            throw ValueError(f"{total} elements do not divide into {cn} channels")
        }
        let pixels = total // cn
        let new_rows = if rows <= 0 { self.rows } else { rows }
        if pixels % new_rows != 0 {
            throw ValueError(f"{pixels} pixels do not divide into {new_rows} rows")
        }
        let new_cols = pixels // new_rows
        return Mat(
            new_rows,
            new_cols,
            CV_MAKETYPE(self.depth(), cn),
            storage: self.storage,
            offset: self.offset,
            step: new_cols * cn
        )
    }

    #: A continuous copy, or `self` when it already is one.
    pub fn continuous(self) -> Mat {
        if self.is_continuous() and self.offset == 0 { return self }
        return self.clone()
    }

    #: Rows become columns.
    pub fn transpose(self) -> Mat {
        self.flush()
        let out = Mat(self.cols, self.rows, self.flags)
        let count = self.rows * self.cols * self.channels()
        if count == 0 { return out }
        let kernel = cached_kernel("mat_transpose")
        kernel.dispatch_async(
            [self.storage, out.storage],
            count,
            [
                count,
                self.rows,
                self.cols,
                self.channels(),
                self.offset,
                self.step_,
                out.offset,
                out.step_,
            ]
        )
        return out
    }

    #: Split into one single-channel matrix per channel.
    pub fn split(self) -> list[Mat] {
        self.flush()
        let cn = self.channels()
        var planes: list[Mat] = []
        let kernel = cached_kernel("mat_extract_channel")
        let count = self.rows * self.cols
        for channel in 0..cn {
            let plane = Mat(self.rows, self.cols, CV_MAKETYPE(self.depth(), 1))
            if count > 0 {
                kernel.dispatch_async(
                    [self.storage, plane.storage],
                    count,
                    [
                        count,
                        self.cols,
                        cn,
                        self.offset,
                        self.step_,
                        plane.offset,
                        plane.step_,
                        channel,
                    ]
                )
            }
            planes.push(plane)
        }
        return planes
    }

    #: Release the device memory. A window does not own its allocation, so
    #: releasing one leaves the parent alone.
    pub fn release(self) -> void {
        if self.freed { return }
        self.freed = true
        self.host = null
        if self.offset == 0 and self.is_continuous() { self.storage.free() }
    }

    #: Human-readable shape and type.
    pub fn to_str(self) -> str {
        return f"Mat({self.rows}x{self.cols}, {type_to_string(self.flags)})"
    }

    fn __str__(self) -> str { return self.to_str() }
}

#: Combine single-channel matrices into one multi-channel matrix.
pub fn merge(planes: list[Mat]) -> Mat {
    if planes.len() == 0 { throw ValueError("merge needs at least one plane") }
    let first = planes[0]
    for plane in planes {
        if plane.channels() != 1 { throw ValueError("merge needs single-channel planes") }
        if plane.rows != first.rows or plane.cols != first.cols {
            throw ValueError("merge needs planes of one shape")
        }
    }
    let cn = planes.len()
    let out = Mat(first.rows, first.cols, CV_MAKETYPE(first.depth(), cn))
    let kernel = cached_kernel("mat_insert_channel")
    let count = first.rows * first.cols
    for (channel, plane) in planes.enumerate() {
        plane.flush()
        if count > 0 {
            kernel.dispatch_async(
                [plane.storage, out.storage],
                count,
                [count, first.cols, cn, plane.offset, plane.step_, out.offset, out.step_, channel]
            )
        }
    }
    return out
}

#: Mirror around an axis: 0 flips vertically, 1 horizontally, -1 both.
pub fn flip(src: Mat, flip_code: int) -> Mat {
    src.flush()
    let out = Mat(src.rows, src.cols, src.flags)
    let count = src.rows * src.cols * src.channels()
    if count == 0 { return out }
    let mode = if flip_code == 0 { 0 } elif flip_code > 0 { 1 } else { 2 }
    let kernel = cached_kernel("mat_flip")
    kernel.dispatch_async(
        [src.storage, out.storage],
        count,
        [
            count,
            src.rows,
            src.cols,
            src.channels(),
            src.offset,
            src.step_,
            out.offset,
            out.step_,
            mode,
        ]
    )
    return out
}

#: Rows become columns.
pub fn transpose(src: Mat) -> Mat { return src.transpose() }

#: Split into one single-channel matrix per channel.
pub fn split(src: Mat) -> list[Mat] { return src.split() }

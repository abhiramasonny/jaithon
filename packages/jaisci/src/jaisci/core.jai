#: `Array`, the kernel cache, and the numeric floor the rest of jaisci stands on.
#:
#: Two policies, and every module in this package obeys both.
#:
#: **Host arithmetic is f64, device arithmetic is f32.** A Jaithon `float` — and
#: so a `list[float]` — is a double; a `std.gpu.Buffer` holds singles. A number
#: that goes to the device and comes back has been rounded to about seven
#: digits, which is fine for a distance matrix and ruinous for a pivot. That is
#: why the `dense_*` factorisations below take and return `list[float]` and
#: never touch a buffer: small dense work of a few hundred rows stays on the
#: host, exactly where jaicv's `core/linalg.jai` already keeps it, and for the
#: same reason — at that size a kernel launch costs more than the arithmetic and
#: the extra digits are the whole point.
#:
#: **Only O(n^2) and up becomes a kernel.** Distance matrices, convolutions,
#: sparse products, Jacobian assembly, a distribution evaluated over an array:
#: those are kernels. A Newton step on twelve parameters is not. Between the two
#: sits the rule that catches the common mistake — a host loop over more than a
#: few thousand elements is a bug in this repository, not a slow path. It costs
#: seconds. Everything here that could grow with the data therefore either
#: dispatches or bounds its host loop by construction: the reductions fold at
#: most `GROUPS` partials, the sorts hand the whole array to the device above
#: `gpu.MIN_GPU_ELEMENTS`, and `arange` and `linspace` are a ramp kernel rather
#: than a `push` per element.
#:
#: FOLLOWUP: `Array` and the free ops below are `jainum`'s charter, not this
#: package's. They are defined here because nothing in the repository yet offers
#: a plain numeric array — `jaitensor.Tensor` carries autograd and a training
#: shape, `jaicv.Mat` is a two-dimensional image with depth and channels. When
#: jainum grows an equivalent, jaisci should re-export it and delete this half
#: of the file.
#:
#: ```
#: let x = Array.linspace(0.0, 1.0, 4096)
#: defer { x.free() }
#: let y = multiply(x, x)
#: defer { y.free() }
#: print(sum(y))
#: ```
import std.gpu as gpu
import std.math
from std.core import Printable
from std.gpu import Buffer, Kernel

#: Threads in a reduction group, and the most groups a reduction will use.
#:
#: A fixed grid rather than one thread per element: a million-element sum the
#: other way round is four thousand groups whose threads each read one float
#: and then spend eight barriers folding it. Here at most a thousand groups
#: walk the array in strides and accumulate in registers, and the host folds
#: what is left — a thousand numbers, which is the largest host loop anything
#: in this file runs.
const GROUP = 256
const GROUPS = 1024

let SOURCE = """#include <metal_stdlib>
using namespace metal;

constexpr constant uint GROUP = 256;

/* op: 0 add, 1 subtract, 2 multiply, 3 divide, 4 maximum, 5 minimum.
 *
 * One entry point with a selector rather than six. The branch is uniform
 * across the grid and disappears next to the two loads and the store, while
 * six entry points would be six pipelines to compile the first time any of
 * them is touched. */
kernel void core_binary(device const float* a [[buffer(0)]],
                        device const float* b [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        constant uint& count [[buffer(3)]],
                        constant uint& op [[buffer(4)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float x = a[id];
    float y = b[id];
    float r = x + y;
    if (op == 1u) r = x - y;
    else if (op == 2u) r = x * y;
    else if (op == 3u) r = x / y;
    else if (op == 4u) r = fmax(x, y);
    else if (op == 5u) r = fmin(x, y);
    out[id] = r;
}

/* op: 0 negate, 1 exp, 2 log, 3 sqrt, 4 absolute, 5 raise to `param`. */
kernel void core_unary(device const float* src [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       constant uint& op [[buffer(3)]],
                       constant uint& param [[buffer(4)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float x = src[id];
    float r = -x;
    if (op == 1u) r = exp(x);
    else if (op == 2u) r = log(x);
    else if (op == 3u) r = sqrt(x);
    else if (op == 4u) r = fabs(x);
    else if (op == 5u) r = pow(x, as_type<float>(param));
    out[id] = r;
}

/* Scale and shift together, which is also how one buffer is copied into
 * another: factor 1, shift 0. Having the copy be an existing kernel is worth
 * more than the fma it wastes -- padding a sort, moving a scan result home
 * and cloning an array all go through here. */
kernel void core_affine(device const float* src [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant uint& count [[buffer(2)]],
                        constant uint& factor [[buffer(3)]],
                        constant uint& shift [[buffer(4)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    out[id] = fma(src[id], as_type<float>(factor), as_type<float>(shift));
}

kernel void core_clamp(device const float* src [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       constant uint& low [[buffer(3)]],
                       constant uint& high [[buffer(4)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    out[id] = clamp(src[id], as_type<float>(low), as_type<float>(high));
}

/* 0, 1, 2, ... as floats: the ramp `arange` scales and the identity
 * permutation an argsort starts from. */
kernel void core_iota(device float* out [[buffer(0)]],
                      constant uint& count [[buffer(1)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    out[id] = float(id);
}

/* Every `stride`-th slot, for the diagonal of a square matrix. */
kernel void core_diagonal(device float* out [[buffer(0)]],
                          constant uint& count [[buffer(1)]],
                          constant uint& stride [[buffer(2)]],
                          constant uint& value [[buffer(3)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    out[id * stride] = as_type<float>(value);
}

/* Sums, and dot products when `paired` is set. `span` is the whole grid, so
 * the loop strides by it and every group sees a contiguous-ish run.
 *
 * `b` is bound even for a plain sum -- the caller passes `a` twice -- because
 * a kernel argument cannot be left unbound and a second entry point that
 * differed by one multiply is not worth compiling. */
kernel void core_reduce(device const float* a [[buffer(0)]],
                        device const float* b [[buffer(1)]],
                        device float* partials [[buffer(2)]],
                        constant uint& count [[buffer(3)]],
                        constant uint& span [[buffer(4)]],
                        constant uint& paired [[buffer(5)]],
                        uint gid [[thread_position_in_grid]],
                        uint lid [[thread_index_in_threadgroup]],
                        uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[GROUP];
    float total = 0.0f;
    for (uint at = gid; at < count; at += span) {
        total += paired != 0u ? a[at] * b[at] : a[at];
    }
    scratch[lid] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = GROUP / 2u; stride > 0u; stride >>= 1u) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) partials[wid] = scratch[0];
}

/* The extremes and where they are, four numbers a group: low, high, and the
 * index of each.
 *
 * A thread that saw nothing keeps INFINITY in all four, so it loses both
 * comparisons without a "did I see anything" test. Ties go to the smaller
 * index, which is what numpy's argmin reports. */
kernel void core_extrema(device const float* src [[buffer(0)]],
                         device float* partials [[buffer(1)]],
                         constant uint& count [[buffer(2)]],
                         constant uint& span [[buffer(3)]],
                         uint gid [[thread_position_in_grid]],
                         uint lid [[thread_index_in_threadgroup]],
                         uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float lows[GROUP];
    threadgroup float highs[GROUP];
    threadgroup float low_at[GROUP];
    threadgroup float high_at[GROUP];
    float low = INFINITY;
    float high = -INFINITY;
    float seen_low = INFINITY;
    float seen_high = INFINITY;
    for (uint at = gid; at < count; at += span) {
        float value = src[at];
        if (value < low) { low = value; seen_low = float(at); }
        if (value > high) { high = value; seen_high = float(at); }
    }
    lows[lid] = low;
    highs[lid] = high;
    low_at[lid] = seen_low;
    high_at[lid] = seen_high;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = GROUP / 2u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            uint other = lid + stride;
            bool lower = lows[other] < lows[lid]
                || (lows[other] == lows[lid] && low_at[other] < low_at[lid]);
            if (lower) {
                lows[lid] = lows[other];
                low_at[lid] = low_at[other];
            }
            bool upper = highs[other] > highs[lid]
                || (highs[other] == highs[lid] && high_at[other] < high_at[lid]);
            if (upper) {
                highs[lid] = highs[other];
                high_at[lid] = high_at[other];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid != 0u) return;
    partials[wid * 4u + 0u] = lows[0];
    partials[wid * 4u + 1u] = highs[0];
    partials[wid * 4u + 2u] = low_at[0];
    partials[wid * 4u + 3u] = high_at[0];
}

/* One doubling of an inclusive scan. Hillis-Steele, so log2(n) dispatches of
 * n threads: more arithmetic than a work-efficient blocked scan, but no host
 * round trip between levels and no second-level pass to write. */
kernel void core_scan_step(device const float* src [[buffer(0)]],
                           device float* dst [[buffer(1)]],
                           constant uint& count [[buffer(2)]],
                           constant uint& reach [[buffer(3)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float value = src[id];
    if (id >= reach) value += src[id - reach];
    dst[id] = value;
}

/* The fallback product, for shapes MPS will not take. */
kernel void core_matmul(device const float* a [[buffer(0)]],
                        device const float* b [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        constant uint& rows [[buffer(3)]],
                        constant uint& inner [[buffer(4)]],
                        constant uint& columns [[buffer(5)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= rows * columns) return;
    uint row = id / columns;
    uint column = id - row * columns;
    float total = 0.0f;
    for (uint k = 0u; k < inner; k++) {
        total = fma(a[row * inner + k], b[k * columns + column], total);
    }
    out[id] = total;
}

/* Indexed by the destination, so the writes coalesce and the reads scatter --
 * the cheaper way round on a unified-memory device. */
kernel void core_transpose(device const float* src [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           constant uint& rows [[buffer(2)]],
                           constant uint& columns [[buffer(3)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= rows * columns) return;
    uint row = id / rows;
    uint column = id - row * rows;
    out[id] = src[column * columns + row];
}

/* One part of a concatenation, placed at `start` along the joined axis.
 * `inner` is the product of the axes after it, so the same kernel serves any
 * rank and any axis. */
kernel void core_concat(device const float* src [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant uint& count [[buffer(2)]],
                        constant uint& inner [[buffer(3)]],
                        constant uint& extent [[buffer(4)]],
                        constant uint& joined [[buffer(5)]],
                        constant uint& start [[buffer(6)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint block = inner * extent;
    uint outer = id / block;
    uint rest = id - outer * block;
    uint along = rest / inner;
    uint within = rest - along * inner;
    out[(outer * joined + start + along) * inner + within] = src[id];
}

/* Gather. `inner` is 1 for a flat take and the row length for `axis: 0`.
 * Negative indices wrap; anything still out of range reads zero, because a
 * kernel has no way to raise. */
kernel void core_take(device const float* src [[buffer(0)]],
                      device const float* index [[buffer(1)]],
                      device float* out [[buffer(2)]],
                      constant uint& count [[buffer(3)]],
                      constant uint& inner [[buffer(4)]],
                      constant uint& limit [[buffer(5)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint slot = id / inner;
    uint within = id - slot * inner;
    int at = int(index[slot]);
    if (at < 0) at += int(limit);
    if (at < 0 || at >= int(limit)) {
        out[id] = 0.0f;
        return;
    }
    out[id] = src[uint(at) * inner + within];
}

/* One compare-exchange stage of a bitonic sort, keys carrying a payload.
 *
 * Bitonic because it is the network that maps onto a dispatch per stage with
 * no host round trip between them: log2(n)*(log2(n)+1)/2 of them, 210 for a
 * million elements, each a single pass. */
kernel void core_bitonic(device float* keys [[buffer(0)]],
                         device float* payload [[buffer(1)]],
                         constant uint& count [[buffer(2)]],
                         constant uint& run [[buffer(3)]],
                         constant uint& gap [[buffer(4)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint partner = id ^ gap;
    if (partner <= id) return;
    bool ascending = (id & run) == 0u;
    float mine = keys[id];
    float theirs = keys[partner];
    if ((mine > theirs) == ascending) {
        keys[id] = theirs;
        keys[partner] = mine;
        float held = payload[id];
        payload[id] = payload[partner];
        payload[partner] = held;
    }
}

/* Where each query would be inserted to keep `sorted` sorted. `right` picks
 * the far end of a run of equals, which is what makes average-tie ranking
 * two of these and an add. */
kernel void core_searchsorted(device const float* sorted [[buffer(0)]],
                              device const float* queries [[buffer(1)]],
                              device float* out [[buffer(2)]],
                              constant uint& count [[buffer(3)]],
                              constant uint& length [[buffer(4)]],
                              constant uint& right [[buffer(5)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float value = queries[id];
    uint low = 0u;
    uint high = length;
    while (low < high) {
        uint middle = (low + high) >> 1;
        bool before = right != 0u ? sorted[middle] <= value : sorted[middle] < value;
        if (before) low = middle + 1u;
        else high = middle;
    }
    out[id] = float(low);
}
"""

#: The compiled pipelines of one shader library, one entry point at a time.
#:
#: Every file in jaicv repeats a `let SOURCE`, a `dict[str, Kernel]` and a
#: `kernel_for`; thirteen more copies of it in this package would be thirteen
#: more chances to get the double-compile wrong. `Kernel.compile` interns on
#: the source text too, but reaching that intern means hashing the whole
#: library — around eighty microseconds for a 36 KB one, an order of magnitude
#: more than encoding the dispatch that follows — so the entry-name dict in
#: front of it is what makes a hot loop cheap.
#:
#: ```
#: let kernels = KernelCache(MY_SOURCE)
#: kernels.get("my_entry").dispatch_async([input, output], count, [count])
#: ```
pub class KernelCache {
    let source: str
    var compiled: dict[str, Kernel]

    #: Hold `source`. Nothing compiles until `get` asks for an entry point, so
    #: a module that is imported and never used costs nothing.
    pub fn init(self, source: str) {
        self.source = source
        self.compiled = {}
    }

    #: The pipeline for `entry`, compiled on first use.
    pub fn get(self, entry: str) -> Kernel {
        let existing = self.compiled.get(entry, null)
        if existing is not null { return existing }
        let kernel = Kernel.compile(self.source, entry)
        self.compiled[entry] = kernel
        return kernel
    }
}

#: Reinterpret a float's bits as an integer, so a kernel can take it as a
#: `constant uint&` and read it back with `as_type<float>`.
#:
#: `Kernel.dispatch` only carries `list[int]`, so every float a kernel needs —
#: a tolerance, an exponent, a fill value — travels this way. Lifted from
#: `jaicv/core/mat.jai`, which needed it first.
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

let _kernels = KernelCache(SOURCE)
let _ONE = f32_bits(1.0)
let _ZERO = f32_bits(0.0)

#: How many elements a shape holds.
#:
#: A zero dimension is allowed and makes the whole array empty; every loop
#: over it simply runs no times. A negative one is still a mistake.
pub fn element_count(shape: list[int]) -> int {
    if shape.len() == 0 { throw ValueError("an Array needs at least one dimension") }
    var size = 1
    for extent in shape {
        if extent < 0 { throw ValueError(f"Array dimensions cannot be negative, got {shape}") }
        size *= extent
    }
    return size
}

fn copy_run(src: Buffer, dst: Buffer, count: int) -> void {
    if count <= 0 { return }
    _kernels.get("core_affine").dispatch_async([src, dst], count, [count, _ONE, _ZERO])
}

#: A dense row-major float32 array on the device.
#:
#: One `std.gpu.Buffer` per allocation, no strides and no offset: a `view` is a
#: window on the parent's buffer and a `reshape` is the same buffer read with
#: different extents, so both alias and neither copies. Device memory is not
#: collected — call `free`, usually from a `defer`.
#:
#: Modelled on `jaicv.Mat`'s ownership and `jaitensor.Tensor`'s shape handling,
#: minus the parts jaisci does not need: no channels, no depth, no autograd.
pub class Array: Printable {
    #: Extents in row-major order.
    pub let shape: list[int]
    #: Number of float32 values the array holds.
    pub let size: int
    let storage: Buffer
    var freed: bool

    #: Allocate `shape`, optionally filled from `values`.
    #:
    #: Pass `storage` to wrap a buffer that someone else owns; `view` and
    #: `reshape` are the callers that do, and freeing what they return releases
    #: the window rather than the parent.
    pub fn init(self, shape: list[int], values: list[float]? = null, storage: Buffer? = null) {
        self.shape = [extent for extent in shape]
        self.size = element_count(shape)
        self.freed = false
        if storage is not null {
            if values is not null {
                throw ValueError("a storage-backed Array cannot also take host values")
            }
            if storage.count != self.size {
                throw ValueError(f"storage holds {storage.count} values, not {self.size}")
            }
            self.storage = storage
            return
        }
        self.storage = Buffer(self.size)
        if values is not null {
            if values.len() != self.size {
                self.storage.free()
                self.freed = true
                throw ValueError(f"{values.len()} values cannot fill {shape}")
            }
            self.storage.upload(values)
        }
    }

    #: An array of zeros.
    pub static fn zeros(shape: list[int]) -> Array {
        let out = Array(shape)
        out.storage.fill_zero()
        return out
    }

    #: An array of ones.
    pub static fn ones(shape: list[int]) -> Array { return Array.full(shape, 1.0) }

    #: An array with every element set to `value`.
    pub static fn full(shape: list[int], value: float) -> Array {
        let out = Array(shape)
        out.storage.fill(value)
        return out
    }

    #: Build from row-major host values. A null `shape` makes a vector.
    pub static fn from_list(values: list[float], shape: list[int]? = null) -> Array {
        return Array(shape ?? [values.len()], values: values)
    }

    #: `start`, `start + step`, ... up to but not including `stop`.
    #:
    #: A ramp kernel and a scale, not a `push` per element: `arange` over a
    #: million points is exactly the host loop this package forbids.
    pub static fn arange(start: float, stop: float, step: float = 1.0) -> Array {
        if step == 0.0 { throw ValueError("arange needs a non-zero step") }
        let count = max(0, int(math.ceil((stop - start) / step)))
        let out = Array([count])
        if count == 0 { return out }
        _kernels.get("core_iota").dispatch_async([out.storage], count, [count])
        _kernels.get("core_affine").dispatch_async(
            [out.storage, out.storage],
            count,
            [count, f32_bits(step), f32_bits(start)]
        )
        return out
    }

    #: `count` points from `start` to `stop`, both ends included.
    #:
    #: The last point is `start + (count - 1) * step` computed in float32, so it
    #: lands within an ulp of `stop` rather than exactly on it. Code that needs
    #: the endpoint exactly should write it rather than read it back.
    pub static fn linspace(start: float, stop: float, count: int) -> Array {
        if count < 0 { throw ValueError(f"linspace needs a non-negative count, got {count}") }
        let out = Array([count])
        if count == 0 { return out }
        let step = if count > 1 { (stop - start) / float(count - 1) } else { 0.0 }
        _kernels.get("core_iota").dispatch_async([out.storage], count, [count])
        _kernels.get("core_affine").dispatch_async(
            [out.storage, out.storage],
            count,
            [count, f32_bits(step), f32_bits(start)]
        )
        return out
    }

    #: The identity, `rows` by `cols` (square when `cols` is null).
    pub static fn eye(rows: int, cols: int? = null) -> Array {
        let columns = cols ?? rows
        let out = Array.zeros([rows, columns])
        let diagonal = min(rows, columns)
        if diagonal == 0 { return out }
        _kernels.get("core_diagonal").dispatch_async(
            [out.storage],
            diagonal,
            [diagonal, columns + 1, _ONE]
        )
        return out
    }

    #: Every element on the host, row-major.
    #:
    #: A download and a boxed list — four megabytes and a few milliseconds for
    #: a million elements — and the f32 storage becomes f64 on the way. This is
    #: the last step of a computation, never a step in the middle of one.
    pub fn values(self) -> list[float] {
        self.check_live()
        if self.size == 0 { return [] }
        return self.storage.download()
    }

    #: Replace every element from a host list.
    pub fn upload(self, values: list[float]) -> void {
        self.check_live()
        if values.len() != self.size {
            throw ValueError(f"{values.len()} values cannot fill {self.shape}")
        }
        if self.size == 0 { return }
        self.storage.upload(values)
    }

    #: A window onto this array's storage, starting at element `start`.
    pub fn view(self, start: int, shape: list[int]) -> Array {
        self.check_live()
        let wanted = element_count(shape)
        if start < 0 or wanted <= 0 or start + wanted > self.size {
            throw ValueError(f"a view of {wanted} at {start} runs past {self.size}")
        }
        return Array(shape, storage: self.storage.view(start, wanted))
    }

    #: The same elements read as `shape`. Shares storage; no copy, no kernel.
    pub fn reshape(self, shape: list[int]) -> Array {
        self.check_live()
        if element_count(shape) != self.size {
            throw ValueError(f"{self.shape} cannot be read as {shape}")
        }
        if self.size == 0 { return Array(shape) }
        return Array(shape, storage: self.storage.view(0, self.size))
    }

    #: An independent copy.
    pub fn copy(self) -> Array {
        let out = Array(self.shape)
        copy_run(self.buffer(), out.storage, self.size)
        return out
    }

    #: How many dimensions the array has.
    pub fn rank(self) -> int { return self.shape.len() }

    #: One extent, counting from the end when `axis` is negative.
    pub fn dim(self, axis: int) -> int {
        let index = if axis < 0 { axis + self.shape.len() } else { axis }
        if index < 0 or index >= self.shape.len() {
            throw IndexError(f"axis {axis} is outside {self.shape}")
        }
        return self.shape[index]
    }

    #: Rows of a matrix; the length of a vector.
    pub fn rows(self) -> int { return if self.rank() >= 2 { self.shape[0] } else { self.size } }

    #: Columns of a matrix; one for a vector.
    pub fn cols(self) -> int { return if self.rank() >= 2 { self.shape[1] } else { 1 } }

    #: The device buffer. Kernels bind this.
    pub fn buffer(self) -> Buffer {
        self.check_live()
        return self.storage
    }

    #: Release the device memory. Freeing twice is harmless.
    pub fn free(self) -> void {
        if self.freed { return }
        self.storage.free()
        self.freed = true
    }

    #: Whether `free` has already run.
    pub fn is_freed(self) -> bool { return self.freed }

    fn check_live(self) -> void {
        if self.freed { throw RuntimeError("this Array has already been freed") }
    }

    pub fn to_str(self) -> str { return f"Array(shape={self.shape})" }

    fn __str__(self) -> str { return self.to_str() }
}

fn same_shape(a: Array, b: Array, what: str) -> void {
    if a.shape != b.shape {
        throw ValueError(f"{what} needs matching shapes, got {a.shape} and {b.shape}")
    }
}

fn binary(a: Array, b: Array, op: int, what: str) -> Array {
    same_shape(a, b, what)
    let out = Array(a.shape)
    if a.size == 0 { return out }
    _kernels.get("core_binary").dispatch_async(
        [a.buffer(), b.buffer(), out.buffer()],
        a.size,
        [a.size, op]
    )
    return out
}

fn unary(a: Array, op: int, param: int) -> Array {
    let out = Array(a.shape)
    if a.size == 0 { return out }
    _kernels.get("core_unary").dispatch_async(
        [a.buffer(), out.buffer()],
        a.size,
        [a.size, op, param]
    )
    return out
}

fn affine(a: Array, factor: float, shift: float) -> Array {
    let out = Array(a.shape)
    if a.size == 0 { return out }
    _kernels.get("core_affine").dispatch_async(
        [a.buffer(), out.buffer()],
        a.size,
        [a.size, f32_bits(factor), f32_bits(shift)]
    )
    return out
}

#: Elementwise sum. Shapes must match — nothing here broadcasts.
pub fn add(a: Array, b: Array) -> Array { return binary(a, b, 0, "add") }

#: Elementwise difference.
pub fn subtract(a: Array, b: Array) -> Array { return binary(a, b, 1, "subtract") }

#: Elementwise product.
pub fn multiply(a: Array, b: Array) -> Array { return binary(a, b, 2, "multiply") }

#: Elementwise quotient. Division by zero gives an infinity, as float32 does.
pub fn divide(a: Array, b: Array) -> Array { return binary(a, b, 3, "divide") }

#: Elementwise larger of two arrays.
pub fn maximum(a: Array, b: Array) -> Array { return binary(a, b, 4, "maximum") }

#: Elementwise smaller of two arrays.
pub fn minimum(a: Array, b: Array) -> Array { return binary(a, b, 5, "minimum") }

#: Multiply every element by `factor`.
pub fn scale(a: Array, factor: float) -> Array { return affine(a, factor, 0.0) }

#: Add `shift` to every element.
pub fn offset(a: Array, shift: float) -> Array { return affine(a, 1.0, shift) }

#: Elementwise negation.
pub fn neg(a: Array) -> Array { return unary(a, 0, 0) }

#: Elementwise `e ** x`.
pub fn exp(a: Array) -> Array { return unary(a, 1, 0) }

#: Elementwise natural logarithm.
pub fn log(a: Array) -> Array { return unary(a, 2, 0) }

#: Elementwise square root.
pub fn sqrt(a: Array) -> Array { return unary(a, 3, 0) }

#: Elementwise absolute value.
pub fn abs(a: Array) -> Array { return unary(a, 4, 0) }

#: Every element raised to `exponent`.
#:
#: A scalar exponent rather than a second array: distances, moments and kernel
#: weights all raise by a constant, and an array-valued exponent is `exp` of a
#: `multiply` of a `log` for the two callers a year that want it.
pub fn power(a: Array, exponent: float) -> Array { return unary(a, 5, f32_bits(exponent)) }

#: Every element held between `low` and `high`.
pub fn clamp(a: Array, low: float, high: float) -> Array {
    let out = Array(a.shape)
    if a.size == 0 { return out }
    _kernels.get("core_clamp").dispatch_async(
        [a.buffer(), out.buffer()],
        a.size,
        [a.size, f32_bits(low), f32_bits(high)]
    )
    return out
}

fn reduce_groups(count: int) -> int {
    return max(1, min(GROUPS, (count + GROUP - 1) // GROUP))
}

fn folded(a: Array, b: Array, paired: int) -> float {
    let groups = reduce_groups(a.size)
    let partials = Buffer(groups)
    defer { partials.free() }
    _kernels.get("core_reduce").dispatch_async(
        [a.buffer(), b.buffer(), partials],
        groups * GROUP,
        [a.size, groups * GROUP, paired],
        group_size: GROUP
    )
    var total = 0.0
    for value in partials.download() { total += value }
    return total
}

#: Sum of every element.
#:
#: The groups accumulate in float32 and the host folds their partials in
#: float64, so the error grows with the length of one group's stride rather
#: than with the whole array. A sum that has to be exact to the last bit still
#: belongs on the host.
pub fn sum(a: Array) -> float {
    if a.size == 0 { return 0.0 }
    return folded(a, a, 0)
}

#: Arithmetic mean of every element.
pub fn mean(a: Array) -> float {
    if a.size == 0 { throw ValueError("an empty Array has no mean") }
    return sum(a) / float(a.size)
}

#: Inner product. Shapes must match.
pub fn dot(a: Array, b: Array) -> float {
    same_shape(a, b, "dot")
    if a.size == 0 { return 0.0 }
    return folded(a, b, 1)
}

#: Smallest, largest, and where each of them first occurs.
#:
#: The positions travel through the kernel as floats, so they are exact up to
#: 2^24 elements and approximate above it — sixteen million, well past the
#: point where a caller wanting an index should be asking a different question.
fn extremes(a: Array) -> tuple[float, float, int, int] {
    if a.size == 0 { throw ValueError("an empty Array has no extremes") }
    let groups = reduce_groups(a.size)
    let partials = Buffer(groups * 4)
    defer { partials.free() }
    _kernels.get("core_extrema").dispatch_async(
        [a.buffer(), partials],
        groups * GROUP,
        [a.size, groups * GROUP],
        group_size: GROUP
    )
    let folded_values = partials.download()
    var low = folded_values[0]
    var high = folded_values[1]
    var low_at = int(folded_values[2])
    var high_at = int(folded_values[3])
    #: Group order is not index order once the array is longer than the grid --
    #: group 0's second stride sits past group 1's first -- so the tie has to be
    #: broken on the position rather than on which partial arrived first.
    for group in 1..groups {
        let base = group * 4
        let candidate_low = int(folded_values[base + 2])
        let candidate_high = int(folded_values[base + 3])
        let takes_low = folded_values[base] == low and candidate_low < low_at
        if folded_values[base] < low or takes_low {
            low = folded_values[base]
            low_at = candidate_low
        }
        let takes_high = folded_values[base + 1] == high and candidate_high < high_at
        if folded_values[base + 1] > high or takes_high {
            high = folded_values[base + 1]
            high_at = candidate_high
        }
    }
    return (low, high, low_at, high_at)
}

#: Smallest element.
pub fn amin(a: Array) -> float {
    let (low, _high, _low_at, _high_at) = extremes(a)
    return low
}

#: Largest element.
pub fn amax(a: Array) -> float {
    let (_low, high, _low_at, _high_at) = extremes(a)
    return high
}

#: Index of the smallest element, earliest on a tie.
pub fn argmin(a: Array) -> int {
    let (_low, _high, low_at, _high_at) = extremes(a)
    return low_at
}

#: Index of the largest element, earliest on a tie.
pub fn argmax(a: Array) -> int {
    let (_low, _high, _low_at, high_at) = extremes(a)
    return high_at
}

#: Running total, inclusive, over the flattened array.
#:
#: Every partial sum is a float32, so a long run of same-signed values loses
#: digits the way any single-precision accumulation does. The scan is
#: log2(n) dispatches over the whole array.
pub fn cumsum(a: Array) -> Array {
    let out = Array(a.shape)
    if a.size == 0 { return out }
    copy_run(a.buffer(), out.buffer(), a.size)
    if a.size == 1 { return out }
    let scratch = Buffer(a.size)
    defer { scratch.free() }
    let kernel = _kernels.get("core_scan_step")
    var source = out.buffer()
    var target = scratch
    var reach = 1
    while reach < a.size {
        kernel.dispatch_async([source, target], a.size, [a.size, reach])
        let held = source
        source = target
        target = held
        reach = reach * 2
    }
    #: An odd number of doublings leaves the answer in the scratch buffer.
    if source is not out.buffer() { copy_run(source, out.buffer(), a.size) }
    return out
}

#: Matrix product of two two-dimensional arrays.
#:
#: Metal Performance Shaders takes the product when every row of every operand
#: is a multiple of sixteen bytes and the result is big enough to pay for the
#: encode; otherwise the fallback kernel runs, which is a thread per output and
#: a loop over the inner dimension.
pub fn matmul(a: Array, b: Array) -> Array {
    if a.rank() != 2 or b.rank() != 2 {
        throw ValueError(f"matmul needs two matrices, got {a.shape} and {b.shape}")
    }
    let rows = a.shape[0]
    let inner = a.shape[1]
    let columns = b.shape[1]
    if b.shape[0] != inner {
        throw ValueError(f"cannot multiply {a.shape} by {b.shape}")
    }
    let out = Array.zeros([rows, columns])
    if rows * columns == 0 or inner == 0 { return out }
    if inner % 4 == 0 and columns % 4 == 0 and rows * columns >= gpu.MIN_GPU_ELEMENTS {
        gpu.matmul_async(a.buffer(), b.buffer(), out.buffer(), rows, inner, columns)
        return out
    }
    _kernels.get("core_matmul").dispatch_async(
        [a.buffer(), b.buffer(), out.buffer()],
        rows * columns,
        [rows, inner, columns]
    )
    return out
}

#: Exchange the two axes of a matrix.
pub fn transpose(a: Array) -> Array {
    if a.rank() != 2 { throw ValueError(f"transpose needs a matrix, got {a.shape}") }
    let rows = a.shape[0]
    let columns = a.shape[1]
    let out = Array([columns, rows])
    if out.size == 0 { return out }
    _kernels.get("core_transpose").dispatch_async(
        [a.buffer(), out.buffer()],
        out.size,
        [rows, columns]
    )
    return out
}

#: Join arrays along `axis`. Every other extent must agree.
pub fn concat(parts: list[Array], axis: int = 0) -> Array {
    if parts.len() == 0 { throw ValueError("concat needs at least one Array") }
    let first = parts[0]
    let rank = first.rank()
    let along = if axis < 0 { axis + rank } else { axis }
    if along < 0 or along >= rank { throw IndexError(f"axis {axis} is outside {first.shape}") }
    var joined = 0
    for part in parts {
        if part.rank() != rank {
            throw ValueError(f"concat needs equal ranks, got {first.shape} and {part.shape}")
        }
        for index in 0..rank {
            if index != along and part.shape[index] != first.shape[index] {
                throw ValueError(
                    f"concat cannot join {first.shape} and {part.shape} on axis {along}"
                )
            }
        }
        joined += part.shape[along]
    }
    var shape = [extent for extent in first.shape]
    shape[along] = joined
    let out = Array(shape)
    var inner = 1
    let after = along + 1
    for index in after..rank { inner *= first.shape[index] }
    var start = 0
    let kernel = _kernels.get("core_concat")
    for part in parts {
        if part.size > 0 {
            kernel.dispatch_async(
                [part.buffer(), out.buffer()],
                part.size,
                [part.size, inner, part.shape[along], joined, start]
            )
        }
        start += part.shape[along]
    }
    return out
}

#: Gather by index.
#:
#: A null `axis` reads the flattened array and gives back the shape of
#: `indices`; `axis: 0` reads whole rows. Indices are floats because `Array` is
#: the only array this package has; negative ones wrap, and anything still out
#: of range reads zero rather than raising, because the check happens in a
#: kernel.
pub fn take(a: Array, indices: Array, axis: int? = null) -> Array {
    let along = axis ?? 0
    if along != 0 { throw ValueError("take reads the flattened Array, or rows with axis 0") }
    let by_row = axis is not null
    var inner = 1
    var limit = a.size
    var shape = [extent for extent in indices.shape]
    if by_row {
        if indices.rank() != 1 {
            throw ValueError(f"take on axis 0 needs a vector of indices, got {indices.shape}")
        }
        limit = a.shape[0]
        for index in 1..a.rank() { inner *= a.shape[index] }
        shape = [indices.size]
        for index in 1..a.rank() { shape.push(a.shape[index]) }
    }
    let out = Array(shape)
    if out.size == 0 { return out }
    _kernels.get("core_take").dispatch_async(
        [a.buffer(), indices.buffer(), out.buffer()],
        out.size,
        [out.size, inner, limit]
    )
    return out
}

let _INFINITY = 1.0e308 * 10.0

#: Run the bitonic network over a padded power-of-two run.
fn bitonic(keys: Buffer, payload: Buffer, padded: int) -> void {
    let kernel = _kernels.get("core_bitonic")
    var run = 2
    while run <= padded {
        var gap = run // 2
        while gap > 0 {
            kernel.dispatch_async([keys, payload], padded, [padded, run, gap])
            gap = gap // 2
        }
        run = run * 2
    }
}

#: Keys and their original positions, sorted ascending on the device.
#:
#: The padding is `+inf` so the pad sorts to the end and the first `a.size`
#: slots are the answer. Both buffers belong to the caller.
fn device_sorted(a: Array) -> tuple[Buffer, Buffer] {
    let padded = max(2, math.next_power_of_two(a.size))
    let keys = Buffer(padded)
    let order = Buffer(padded)
    keys.fill(_INFINITY)
    _kernels.get("core_iota").dispatch_async([order], padded, [padded])
    copy_run(a.buffer(), keys, a.size)
    bitonic(keys, order, padded)
    return (keys, order)
}

#: Every element in ascending order.
#:
#: Below `gpu.MIN_GPU_ELEMENTS` the host sort wins outright: the device path is
#: two allocations, a fill and log2(n)*(log2(n)+1)/2 dispatches, which for four
#: thousand elements is more encode time than the whole sort is worth.
#:
#: NaNs compare false against everything, so they stay wherever the network
#: leaves them. Filter them out first if their position matters.
pub fn sort_values(a: Array) -> Array {
    if a.size < 2 { return a.copy() }
    if a.size < gpu.MIN_GPU_ELEMENTS {
        return Array.from_list(a.values().sort(), a.shape)
    }
    let (keys, order) = device_sorted(a)
    defer { keys.free() }
    defer { order.free() }
    let out = Array(a.shape)
    copy_run(keys, out.buffer(), a.size)
    return out
}

#: The permutation that would sort the array, as float indices.
pub fn argsort(a: Array) -> Array {
    if a.size < 2 { return Array.arange(0.0, float(a.size)) }
    if a.size < gpu.MIN_GPU_ELEMENTS {
        let host = a.values()
        var order: list[int] = []
        for index in 0..a.size { order.push(index) }
        return Array.from_list([float(index) for index in order.sort(key: |index| host[index])])
    }
    let (keys, order) = device_sorted(a)
    defer { keys.free() }
    defer { order.free() }
    let out = Array([a.size])
    copy_run(order, out.buffer(), a.size)
    return out
}

fn bisect(sorted_values: list[float], value: float, right: bool) -> int {
    var low = 0
    var high = sorted_values.len()
    while low < high {
        let middle = (low + high) // 2
        let at = sorted_values[middle]
        let before = if right { at <= value } else { at < value }
        if before { low = middle + 1 } else { high = middle }
    }
    return low
}

#: Where each of `queries` would be inserted to keep `sorted_values` sorted.
#:
#: `right` picks the far end of a run of equals, matching numpy's `side`.
pub fn searchsorted(sorted_values: Array, queries: Array, right: bool = false) -> Array {
    let out = Array(queries.shape)
    if queries.size == 0 { return out }
    if queries.size < gpu.MIN_GPU_ELEMENTS {
        let haystack = sorted_values.values()
        var found: list[float] = []
        for value in queries.values() { found.push(float(bisect(haystack, value, right))) }
        out.upload(found)
        return out
    }
    _kernels.get("core_searchsorted").dispatch_async([sorted_values.buffer(), queries.buffer(), out.buffer()], queries.size, [queries.size, sorted_values.size, if right {
        1
    } else {
        0
    }])
    return out
}

#: One-based ranks, ties sharing their average — scipy's `rankdata`.
#:
#: Ranking looks sequential and is not. For a value with `c` copies and `L`
#: values strictly below it, the average rank is `L + (c + 1) / 2`, and
#: `searchsorted` on the sorted array gives `L` on the left and `L + c` on the
#: right. So the whole thing is a sort, two searches and an affine — every step
#: already a kernel, and nothing walks the array on the host.
pub fn rank_data(a: Array) -> Array {
    if a.size == 0 { return Array(a.shape) }
    let ordered = sort_values(a)
    defer { ordered.free() }
    let low = searchsorted(ordered, a, right: false)
    defer { low.free() }
    let high = searchsorted(ordered, a, right: true)
    defer { high.free() }
    let both = add(low, high)
    defer { both.free() }
    let halved = scale(both, 0.5)
    defer { halved.free() }
    return offset(halved, 0.5)
}

#: LU with partial pivoting, in float64 on the host.
#:
#: The whole `dense_*` family below takes a row-major `list[float]` and its
#: dimensions rather than an `Array`, and that is the precision policy showing
#: through: a pivot divided in float32 loses half its digits, and these run at
#: sizes — a spline system, a Jacobian, a covariance — where the device would
#: lose the launch overhead race anyway. jaicv's `core/linalg.jai` made the same
#: call for the same reason.
#:
#: Returns the combined factors (`L` below the diagonal with its implied unit
#: diagonal, `U` on and above), the row permutation, and the sign of that
#: permutation — `1` or `-1`, and `0` when a column was exactly zero and the
#: matrix is singular. `dense_lu_solve` consumes all three.
pub fn dense_lu(a: list[float], n: int) -> tuple[list[float], list[int], int] {
    if a.len() != n * n { throw ValueError(f"{a.len()} values are not an {n}x{n} matrix") }
    var lu = [value for value in a]
    var pivots: list[int] = []
    for index in 0..n { pivots.push(index) }
    var sign = 1
    for column in 0..n {
        var best = column
        var largest = math.fabs(lu[column * n + column])
        let after = column + 1
        for row in after..n {
            let size = math.fabs(lu[row * n + column])
            if size > largest {
                largest = size
                best = row
            }
        }
        if largest == 0.0 {
            sign = 0
            continue
        }
        if best != column {
            for index in 0..n {
                let held = lu[column * n + index]
                lu[column * n + index] = lu[best * n + index]
                lu[best * n + index] = held
            }
            let held = pivots[column]
            pivots[column] = pivots[best]
            pivots[best] = held
            sign = -sign
        }
        let diagonal = lu[column * n + column]
        for row in after..n {
            let factor = lu[row * n + column] / diagonal
            lu[row * n + column] = factor
            for index in after..n { lu[row * n + index] -= factor * lu[column * n + index] }
        }
    }
    return (lu, pivots, sign)
}

#: Solve with the factors `dense_lu` produced. `b` is row-major, `nrhs` wide.
pub fn dense_lu_solve(
    lu: list[float],
    pivots: list[int],
    b: list[float],
    n: int,
    nrhs: int = 1
) -> list[float] {
    if lu.len() != n * n { throw ValueError(f"{lu.len()} values are not an {n}x{n} factor") }
    if b.len() != n * nrhs { throw ValueError(f"{b.len()} values are not {n}x{nrhs}") }
    var x: list[float] = []
    for row in 0..n {
        for column in 0..nrhs { x.push(b[pivots[row] * nrhs + column]) }
    }
    for row in 0..n {
        for column in 0..nrhs {
            var total = x[row * nrhs + column]
            for k in 0..row { total -= lu[row * n + k] * x[k * nrhs + column] }
            x[row * nrhs + column] = total
        }
    }
    var row = n - 1
    while row >= 0 {
        let diagonal = lu[row * n + row]
        if diagonal == 0.0 { throw ValueError("the matrix is singular to working precision") }
        let after = row + 1
        for column in 0..nrhs {
            var total = x[row * nrhs + column]
            for k in after..n { total -= lu[row * n + k] * x[k * nrhs + column] }
            x[row * nrhs + column] = total / diagonal
        }
        row -= 1
    }
    return x
}

#: Cholesky factor `L`, lower triangular, with `L L^T = a`.
#:
#: Raises when a pivot is not positive, which is the cheapest test there is for
#: "this matrix is not positive definite" and the one every caller wants: a
#: covariance that has gone indefinite, a Hessian that is not a minimum.
pub fn dense_cholesky(a: list[float], n: int) -> list[float] {
    if a.len() != n * n { throw ValueError(f"{a.len()} values are not an {n}x{n} matrix") }
    var lower: list[float] = []
    for _index in 0..n * n { lower.push(0.0) }
    for row in 0..n {
        let through = row + 1
        for column in 0..through {
            var total = a[row * n + column]
            for k in 0..column { total -= lower[row * n + k] * lower[column * n + k] }
            if row != column {
                lower[row * n + column] = total / lower[column * n + column]
                continue
            }
            if total <= 0.0 {
                throw ValueError(
                    f"dense_cholesky needs a positive definite matrix, pivot {row} is {total}"
                )
            }
            lower[row * n + column] = math.sqrt(total)
        }
    }
    return lower
}

#: Thin QR by Householder reflections: `q` is `rows x cols`, `r` is `cols x cols`.
#:
#: Householder rather than Gram-Schmidt because the columns a least-squares fit
#: hands over are usually close to dependent — a polynomial design matrix, a
#: Jacobian near a plateau — and modified Gram-Schmidt loses orthogonality
#: exactly there.
pub fn dense_qr(a: list[float], rows: int, cols: int) -> tuple[list[float], list[float]] {
    if a.len() != rows * cols { throw ValueError(f"{a.len()} values are not {rows}x{cols}") }
    if rows < cols { throw ValueError(f"dense_qr needs rows >= cols, got {rows}x{cols}") }
    var upper = [value for value in a]
    var q: list[float] = []
    for row in 0..rows {
        for column in 0..rows { q.push(if row == column { 1.0 } else { 0.0 }) }
    }
    var reflector: list[float] = []
    for _index in 0..rows { reflector.push(0.0) }
    for column in 0..cols {
        var norm = 0.0
        for row in column..rows {
            let value = upper[row * cols + column]
            norm += value * value
        }
        norm = math.sqrt(norm)
        if norm == 0.0 { continue }
        let head = upper[column * cols + column]
        #: Reflect away from the head rather than towards it: subtracting two
        #: nearly equal numbers here is the classic way to lose the column.
        let alpha = if head >= 0.0 { -norm } else { norm }
        for row in 0..rows { reflector[row] = 0.0 }
        reflector[column] = head - alpha
        let below = column + 1
        for row in below..rows { reflector[row] = upper[row * cols + column] }
        var scale_ = 0.0
        for row in column..rows { scale_ += reflector[row] * reflector[row] }
        if scale_ == 0.0 { continue }
        for index in column..cols {
            var projection = 0.0
            for row in column..rows { projection += reflector[row] * upper[row * cols + index] }
            let factor = 2.0 * projection / scale_
            for row in column..rows { upper[row * cols + index] -= factor * reflector[row] }
        }
        for row in 0..rows {
            var projection = 0.0
            for k in column..rows { projection += q[row * rows + k] * reflector[k] }
            let factor = 2.0 * projection / scale_
            for k in column..rows { q[row * rows + k] -= factor * reflector[k] }
        }
    }
    var thin: list[float] = []
    for row in 0..rows {
        for column in 0..cols { thin.push(q[row * rows + column]) }
    }
    var triangle: list[float] = []
    for row in 0..cols {
        for column in 0..cols {
            triangle.push(if row <= column { upper[row * cols + column] } else { 0.0 })
        }
    }
    return (thin, triangle)
}

#: Solve a triangular system by substitution.
#:
#: `unit` reads the diagonal as ones without looking at it, which is how the
#: `L` inside `dense_lu`'s combined factor is stored.
pub fn dense_triangular(
    a: list[float],
    b: list[float],
    n: int,
    lower: bool,
    unit: bool = false
) -> list[float] {
    if a.len() != n * n { throw ValueError(f"{a.len()} values are not an {n}x{n} matrix") }
    if b.len() != n { throw ValueError(f"{b.len()} values are not a length-{n} vector") }
    var x = [value for value in b]
    if lower {
        for row in 0..n {
            var total = x[row]
            for k in 0..row { total -= a[row * n + k] * x[k] }
            x[row] = if unit { total } else { total / a[row * n + row] }
        }
        return x
    }
    var row = n - 1
    while row >= 0 {
        var total = x[row]
        let after = row + 1
        for k in after..n { total -= a[row * n + k] * x[k] }
        x[row] = if unit { total } else { total / a[row * n + row] }
        row -= 1
    }
    return x
}

#: Thomas's algorithm: an `n x n` tridiagonal solve in linear time.
#:
#: `lower` and `upper` are the off-diagonals, `n - 1` long each. No pivoting —
#: the systems that reach here are the diagonally dominant ones a spline or a
#: heat equation produces, where pivoting never triggers, and a caller who
#: cannot promise that should assemble a band and call `dense_banded`.
pub fn dense_tridiagonal(
    lower: list[float],
    diagonal: list[float],
    upper: list[float],
    rhs: list[float]
) -> list[float] {
    let n = diagonal.len()
    if n == 0 { return [] }
    if lower.len() != n - 1 or upper.len() != n - 1 {
        throw ValueError(f"a {n}-row tridiagonal needs {n - 1} off-diagonal values on each side")
    }
    if rhs.len() != n { throw ValueError(f"{rhs.len()} values are not a length-{n} vector") }
    var sweep = [value for value in upper]
    var solution = [value for value in rhs]
    var pivot = diagonal[0]
    if pivot == 0.0 { throw ValueError("dense_tridiagonal met a zero pivot") }
    if n > 1 { sweep[0] = upper[0] / pivot }
    solution[0] = rhs[0] / pivot
    for row in 1..n {
        pivot = diagonal[row] - lower[row - 1] * sweep[row - 1]
        if pivot == 0.0 { throw ValueError(f"dense_tridiagonal met a zero pivot at row {row}") }
        if row < n - 1 { sweep[row] = upper[row] / pivot }
        solution[row] = (rhs[row] - lower[row - 1] * solution[row - 1]) / pivot
    }
    var row = n - 2
    while row >= 0 {
        solution[row] -= sweep[row] * solution[row + 1]
        row -= 1
    }
    return solution
}

#: Solve a banded system held in LAPACK's `ab` layout, as `scipy.linalg.solve_banded`.
#:
#: `ab` has `lower + upper + 1` rows and `n` columns, with `A[i][j]` at
#: `ab[upper + i - j][j]`. Elimination pivots inside the band, so `U` can spread
#: `lower` columns further right than `A` did; the working store carries that
#: fill-in and nothing else, which keeps the cost `O(n * (lower + upper)^2)`
#: instead of the `O(n^3)` a dense factorisation would spend on zeros.
pub fn dense_banded(
    lower: int,
    upper: int,
    ab: list[float],
    n: int,
    b: list[float]
) -> list[float] {
    if lower < 0 or upper < 0 { throw ValueError("band widths cannot be negative") }
    if ab.len() != (lower + upper + 1) * n {
        throw ValueError(f"{ab.len()} values are not a {lower + upper + 1}x{n} band")
    }
    if b.len() != n { throw ValueError(f"{b.len()} values are not a length-{n} vector") }
    let width = 2 * lower + upper + 1
    var work: list[float] = []
    for _index in 0..n * width { work.push(0.0) }
    for row in 0..n {
        let first = max(0, row - lower)
        let past = min(n, row + upper + 1)
        for column in first..past {
            work[row * width + column - row + lower] = ab[(upper + row - column) * n + column]
        }
    }
    var solution = [value for value in b]
    for column in 0..n {
        let last = min(n - 1, column + lower)
        var best = column
        var largest = math.fabs(work[column * width + lower])
        let below = column + 1
        for row in below..last + 1 {
            let size = math.fabs(work[row * width + column - row + lower])
            if size > largest {
                largest = size
                best = row
            }
        }
        if largest == 0.0 { throw ValueError(f"the band is singular at column {column}") }
        let reach = min(n - 1, column + upper + lower)
        if best != column {
            for index in column..reach + 1 {
                let held = work[column * width + index - column + lower]
                work[column * width + index - column + lower] = work[best * width + index - best + lower]
                work[best * width + index - best + lower] = held
            }
            let held = solution[column]
            solution[column] = solution[best]
            solution[best] = held
        }
        let pivot = work[column * width + lower]
        for row in below..last + 1 {
            let factor = work[row * width + column - row + lower] / pivot
            if factor == 0.0 { continue }
            for index in column..reach + 1 {
                work[row * width + index - row + lower] -= factor *
                    work[column * width + index - column + lower]
            }
            solution[row] -= factor * solution[column]
        }
    }
    var row = n - 1
    while row >= 0 {
        let reach = min(n - 1, row + upper + lower)
        var total = solution[row]
        let after = row + 1
        for index in after..reach + 1 {
            total -= work[row * width + index - row + lower] * solution[index]
        }
        solution[row] = total / work[row * width + lower]
        row -= 1
    }
    return solution
}

#: The Givens rotation that zeros `b` against `a`, as `(cosine, sine, radius)`.
#:
#: `hypot` rather than `sqrt(a*a + b*b)`: the QR updates that use this run on
#: residuals that can be tiny, and squaring them underflows long before the
#: rotation is meaningless.
pub fn givens(a: float, b: float) -> tuple[float, float, float] {
    if b == 0.0 { return (1.0, 0.0, a) }
    if a == 0.0 { return (0.0, 1.0, b) }
    let magnitude = math.hypot(a, b)
    let radius = if math.fabs(a) > math.fabs(b) {
        math.copysign(magnitude, a)
    } else {
        math.copysign(magnitude, b)
    }
    return (a / radius, b / radius, radius)
}

#: What every minimiser in this package returns, named as scipy names it.
#:
#: `status` is `0` for a converged run and an algorithm-specific code
#: otherwise; `message` says the same thing in words. `jac` is empty when the
#: method never formed a gradient.
pub class OptimizeResult: Printable {
    #: The parameter vector at the point the search stopped.
    pub var x: list[float]
    #: The objective there.
    pub var fun: float
    #: The gradient there, or empty when none was formed.
    pub var jac: list[float]
    #: Whether the stopping test was convergence rather than a limit.
    pub var success: bool
    #: Zero on success, an algorithm-specific code otherwise.
    pub var status: int
    #: The same verdict in words.
    pub var message: str
    #: Iterations taken.
    pub var nit: int
    #: Objective evaluations taken.
    pub var nfev: int

    pub fn init(
        self,
        x: list[float],
        fun: float,
        success: bool = true,
        status: int = 0,
        message: str = "",
        nit: int = 0,
        nfev: int = 0,
        jac: list[float] = []
    ) {
        self.x = x
        self.fun = fun
        self.jac = jac
        self.success = success
        self.status = status
        self.message = message
        self.nit = nit
        self.nfev = nfev
    }

    pub fn to_str(self) -> str {
        return f"OptimizeResult(fun={self.fun}, nit={self.nit}, success={self.success})"
    }

    fn __str__(self) -> str { return self.to_str() }
}

#: What a scalar root finder returns.
pub class RootResult: Printable {
    #: The root, or the best estimate when the search gave up.
    pub var root: float
    #: Iterations taken.
    pub var iterations: int
    #: Function evaluations taken.
    pub var function_calls: int
    #: Whether the bracket or the step closed to tolerance.
    pub var converged: bool
    #: A short reason: "converged", "max iterations", "no sign change".
    pub var flag: str

    pub fn init(
        self,
        root: float,
        iterations: int = 0,
        function_calls: int = 0,
        converged: bool = true,
        flag: str = "converged"
    ) {
        self.root = root
        self.iterations = iterations
        self.function_calls = function_calls
        self.converged = converged
        self.flag = flag
    }

    pub fn to_str(self) -> str {
        return f"RootResult(root={self.root}, converged={self.converged})"
    }

    fn __str__(self) -> str { return self.to_str() }
}

#: What an initial value problem solver returns.
#:
#: `y[i]` is the whole state at `t[i]` — the transpose of scipy's `(states,
#: points)` layout, and the one a stepper produces without a rearrangement.
#: Said out loud here because it is the sort of thing every caller would
#: otherwise guess differently.
pub class OdeResult: Printable {
    #: The times the solution was recorded at.
    pub var t: list[float]
    #: The state at each of those times.
    pub var y: list[list[float]]
    #: Whether the integration reached the end of the span.
    pub var success: bool
    #: Zero on success, an algorithm-specific code otherwise.
    pub var status: int
    #: The same verdict in words.
    pub var message: str
    #: Right-hand-side evaluations taken.
    pub var nfev: int

    pub fn init(
        self,
        t: list[float],
        y: list[list[float]],
        success: bool = true,
        status: int = 0,
        message: str = "",
        nfev: int = 0
    ) {
        self.t = t
        self.y = y
        self.success = success
        self.status = status
        self.message = message
        self.nfev = nfev
    }

    pub fn to_str(self) -> str {
        return f"OdeResult(points={self.t.len()}, success={self.success})"
    }

    fn __str__(self) -> str { return self.to_str() }
}

#: What every hypothesis test returns: the statistic and its p-value.
pub class TestResult: Printable {
    #: The test statistic.
    pub var statistic: float
    #: The probability of a statistic at least this extreme under the null.
    pub var pvalue: float

    pub fn init(self, statistic: float, pvalue: float) {
        self.statistic = statistic
        self.pvalue = pvalue
    }

    pub fn to_str(self) -> str {
        return f"TestResult(statistic={self.statistic}, pvalue={self.pvalue})"
    }

    fn __str__(self) -> str { return self.to_str() }
}

#: What `linregress` returns, field for field as scipy has it.
pub class LinregressResult: Printable {
    #: Slope of the fitted line.
    pub var slope: float
    #: Intercept of the fitted line.
    pub var intercept: float
    #: Pearson correlation between the two samples.
    pub var rvalue: float
    #: Two-sided p-value for a null hypothesis of zero slope.
    pub var pvalue: float
    #: Standard error of the slope.
    pub var stderr: float
    #: Standard error of the intercept.
    pub var intercept_stderr: float

    pub fn init(
        self,
        slope: float,
        intercept: float,
        rvalue: float = 0.0,
        pvalue: float = 0.0,
        stderr: float = 0.0,
        intercept_stderr: float = 0.0
    ) {
        self.slope = slope
        self.intercept = intercept
        self.rvalue = rvalue
        self.pvalue = pvalue
        self.stderr = stderr
        self.intercept_stderr = intercept_stderr
    }

    pub fn to_str(self) -> str {
        return f"LinregressResult(slope={self.slope}, intercept={self.intercept}, r={self.rvalue})"
    }

    fn __str__(self) -> str { return self.to_str() }
}

#: Forward-difference gradient of `f` at `x`.
#:
#: The step is `eps` scaled by the size of the coordinate rather than scipy's
#: flat absolute step: a parameter of order a thousand and one of order a
#: thousandth turn up in the same vector all the time — a rate beside an
#: amplitude — and one absolute step cannot serve both. The default `eps` is
#: the square root of float64 epsilon, which is where truncation error and
#: cancellation error meet for a first difference.
#:
#: `n + 1` evaluations. A method that can afford `2n` should use central
#: differences instead; this is the one every optimiser falls back to.
pub fn approx_fprime(
    x: list[float],
    f: fn(list[float]) -> float,
    eps: float = 1.4901161193847656e-08
) -> list[float] {
    let base = f(x)
    var out: list[float] = []
    for index in 0..x.len() {
        var shifted = [value for value in x]
        let size = math.fabs(x[index])
        let step = eps * (if size > 1.0 { size } else { 1.0 })
        shifted[index] = x[index] + step
        out.push((f(shifted) - base) / step)
    }
    return out
}

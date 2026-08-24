#: `NDArray` — the strided n-dimensional array every other jainum module reads
#: and writes, and the Metal prelude they all compile against.
#:
#: Storage is one float32 device buffer per allocation. An `NDArray` is a window
#: onto it: an element offset and one stride per axis, so a slice, a transpose,
#: a flip and a diagonal all alias their parent and writing through one is
#: visible in the other. Strides are counted in float slots and may be negative,
#: which is what lets `a[::-1]` be a view rather than a copy.
#:
#: The declared dtype (`BOOL`, `INT32`, `COMPLEX64`, ...) governs rounding,
#: saturation and result type, not the width of the storage — jaicv's `Mat`
#: depth model, for the same reason. Every integer up to 2^24 is exact in
#: float32, `FLOAT64` is a source-compatible alias for float32 rather than a
#: wider element, and `COMPLEX64` is interleaved re/im with two float slots per
#: element, the way jaicv spells `CV_32FC2`.
#:
#: Indexing does not use slice syntax, and this will surprise everyone who
#: comes from numpy. `a[1:3]` reaches `OP_GET_SLICE` in the VM, which throws
#: for anything that is not a list, str or tuple; the checker's "a user type
#: needs `__getitem__`" is not backed by the runtime. So an index is an int, a
#: `Slice`, a first-class `range`, `NEWAXIS`, `ELLIPSIS`, or several of those
#: separated by commas — which the VM hands over as one tuple.
#:
#: ```
#: let a = NDArray([4, 5])
#: a.upload([float(index) for index in 0..20])
#: a[1]                              #: a view of shape [5]
#: a[1, 0..3]                        #: row 1, columns 0 1 2
#: a[ELLIPSIS, sl(null, null, -1)]   #: the last axis reversed, still a view
#: ```
#:
#: A `range` is the nicest spelling where the bounds are known and positive;
#: `sl` covers the open ends and the negative steps a `range` cannot write.
#:
#: This module is the root of jainum's import DAG: it may not import any other
#: jainum module, and everything else imports it.
import std.gpu as gpu
from std.core import Printable
from std.gpu import Buffer, Kernel

#: Elements per threadgroup in every kernel here that folds through
#: threadgroup memory. 256 is eight simdgroups, which is what the scan's
#: per-warp table is sized for.
pub const GROUP = 256

#: Largest whole number float32 represents exactly. An integer dtype is a
#: promise about rounding, not about width, so a count past this silently
#: stops being a count.
pub const MAX_EXACT_INT = 16777216

#: dtype: 0.0 or 1.0, nothing else. A comparison produces one of these.
pub const BOOL = 0
#: dtype: whole numbers, exact to `MAX_EXACT_INT`.
pub const INT32 = 1
#: dtype: whole numbers. Storage is float32, so this is exact to
#: `MAX_EXACT_INT` too — an alias that keeps source compatible, not a wider
#: element.
pub const INT64 = 2
#: dtype: 32-bit float, the storage everything actually is.
pub const FLOAT32 = 3
#: dtype: 64-bit float. Storage is 32-bit, so this is an alias that keeps
#: source compatible rather than a wider element.
pub const FLOAT64 = 4
#: dtype: interleaved real and imaginary float32, two slots per element.
pub const COMPLEX64 = 5

#: Arithmetic selector shared by the four kernels the operators own.
pub const OP_ADD = 0
#: Arithmetic selector: subtraction.
pub const OP_SUB = 1
#: Arithmetic selector: multiplication.
pub const OP_MUL = 2
#: Arithmetic selector: true division.
pub const OP_DIV = 3

#: Conversion mode for the copy kernels: take the value as it stands.
pub const CAST_NONE = 0
#: Conversion mode: round to nearest even, then clamp — what an integer dtype
#: means.
pub const CAST_ROUND = 1
#: Conversion mode: anything non-zero becomes 1.0.
pub const CAST_BOOL = 2

#: The name numpy prints for a dtype.
pub fn dtype_name(dtype: int) -> str {
    let names = ["bool", "int32", "int64", "float32", "float64", "complex64"]
    if dtype < 0 or dtype >= names.len() { return f"dtype{dtype}" }
    return names[dtype]
}

#: Float slots one element of `dtype` occupies: 2 for complex, 1 for the rest.
pub fn dtype_lanes(dtype: int) -> int { return if dtype == COMPLEX64 { 2 } else { 1 } }

#: Whether `dtype` stores whole numbers, and therefore rounds and saturates on
#: the way in.
pub fn dtype_is_integer(dtype: int) -> bool { return dtype <= INT64 }

#: Whether `dtype` carries an imaginary part.
pub fn dtype_is_complex(dtype: int) -> bool { return dtype == COMPLEX64 }

#: Whether `dtype` is the two-valued one.
pub fn dtype_is_bool(dtype: int) -> bool { return dtype == BOOL }

#: Lowest value `dtype` holds, for a saturating cast.
pub fn dtype_min(dtype: int) -> float {
    if dtype == BOOL { return 0.0 }
    return -2147483648.0
}

#: Highest value `dtype` holds, for a saturating cast.
#:
#: 2147483520 and not 2147483647: a float32 near 2^31 steps in 128s, so the
#: true int32 ceiling rounds *up* to 2^31 and a clamp against it lets through
#: a value the type cannot hold. This is the largest float32 below 2^31.
#:
#: `INT64` saturates there too rather than at 2^63. Past `MAX_EXACT_INT`
#: neither bound is representable one integer at a time, and a bound that lies
#: is worse than a bound that is merely narrow.
pub fn dtype_max(dtype: int) -> float {
    if dtype == BOOL { return 1.0 }
    return 2147483520.0
}

#: The dtype an operation on `a` and `b` produces.
#:
#: A plain lattice — `BOOL < INT32 < INT64 < FLOAT32 < FLOAT64 < COMPLEX64` —
#: because the constants are ordered to make it one. numpy's value-based
#: promotion is not reproduced: with float32 storage underneath, the only
#: distinctions this can honour are integer-versus-float and real-versus-
#: complex, and pretending otherwise would promise widths that do not exist.
pub fn promote(a: int, b: int) -> int { return max(a, b) }

#: Reject a complex array at the door of a kernel that has no lanes.
#:
#: Complex is supported by `fft` and by the arithmetic here; every other
#: module calls this first and is then free to assume one float per element.
pub fn require_real(a: NDArray, op: str) -> void {
    if dtype_is_complex(a.dtype) {
        throw TypeError(f"{op} does not accept a complex array; take .real() or .imag() first")
    }
}

#: Elements a shape holds. An empty shape is one element, not none: a 0-d
#: array is a scalar living on the device.
pub fn element_count(shape: list[int]) -> int {
    var total = 1
    for extent in shape {
        if extent < 0 { throw ValueError(f"a shape cannot contain {extent}") }
        total *= extent
    }
    return total
}

#: Turn a possibly negative axis into an index into `0..ndim`.
pub fn normalize_axis(axis: int, ndim: int, name: str = "axis") -> int {
    let at = if axis < 0 { axis + ndim } else { axis }
    if at < 0 or at >= ndim {
        throw ValueError(f"{name} {axis} is out of bounds for an array of {ndim} dimensions")
    }
    return at
}

#: `normalize_axis` over a list, rejecting a repeat.
pub fn normalize_axes(axes: list[int], ndim: int, name: str = "axis") -> list[int] {
    var seen: dict[int, bool] = {}
    var out: list[int] = []
    for axis in axes {
        let at = normalize_axis(axis, ndim, name)
        if seen.get(at, false) { throw ValueError(f"{name} {axis} appears twice") }
        seen[at] = true
        out.push(at)
    }
    return out
}

#: Row-major strides for `shape`, in float slots.
pub fn contiguous_strides(shape: list[int], lanes: int) -> list[int] {
    var strides = [0 for _axis in shape]
    var running = lanes
    var axis = shape.len() - 1
    while axis >= 0 {
        strides[axis] = running
        running *= shape[axis]
        axis -= 1
    }
    return strides
}

#: Whether `strides` walks `shape` in row-major order with nothing between the
#: elements.
#:
#: An axis of one element is skipped: its stride is never used, so a view that
#: kept a stale one is still as dense as a fresh allocation, and refusing it
#: would send `reshape` through a copy for nothing.
pub fn is_contiguous(shape: list[int], strides: list[int], lanes: int) -> bool {
    if element_count(shape) == 0 { return true }
    var expected = lanes
    var axis = shape.len() - 1
    while axis >= 0 {
        let extent = shape[axis]
        if extent != 1 {
            if strides[axis] != expected { return false }
            expected *= extent
        }
        axis -= 1
    }
    return true
}

#: The shape `a` and `b` broadcast to, aligned from the right.
#:
#: The surviving extent is the one that is not 1, which is `max` everywhere
#: except against a zero-length axis: numpy answers `(3, 0)` for `(3, 0)`
#: against `(3, 1)` and against `()`, because a length-one axis stretches to
#: whatever the other side is and an empty one stays empty. `max` answered 1
#: there and handed the caller an output with three elements it then could
#: not fill from an empty operand.
pub fn broadcast_shapes(a: list[int], b: list[int]) -> list[int] {
    let rank = max(a.len(), b.len())
    var out = [0 for _axis in 0..rank]
    for axis in 0..rank {
        let left = if axis < rank - a.len() { 1 } else { a[axis - (rank - a.len())] }
        let right = if axis < rank - b.len() { 1 } else { b[axis - (rank - b.len())] }
        if left != right and left != 1 and right != 1 {
            throw ValueError(f"shapes {a} and {b} do not broadcast together")
        }
        out[axis] = if left == 1 { right } else { left }
    }
    return out
}

#: Strides that read `shape` as if it were `target`.
#:
#: A new leading axis and a stretched length-one axis both get stride 0, so
#: every thread of the broadcast axis reads the same slot and no memory is
#: copied to make the shapes agree.
pub fn broadcast_strides(shape: list[int], strides: list[int], target: list[int]) -> list[int] {
    let pad = target.len() - shape.len()
    if pad < 0 { throw ValueError(f"shape {shape} cannot broadcast to the shorter {target}") }
    var out = [0 for _axis in target]
    for axis in 0..target.len() {
        if axis < pad { continue }
        let extent = shape[axis - pad]
        if extent == target[axis] {
            out[axis] = strides[axis - pad]
        } elif extent != 1 {
            throw ValueError(f"shape {shape} does not broadcast to {target}")
        }
    }
    return out
}

#: Two views of one common shape, neither of them a copy.
pub fn broadcast_pair(a: NDArray, b: NDArray) -> (NDArray, NDArray) {
    let shape = broadcast_shapes(a.shape, b.shape)
    return (a.broadcast_to(shape), b.broadcast_to(shape))
}

#: A view of the `offset`-th diagonal of the plane `axis1` x `axis2`.
#:
#: The diagonal is an axis whose stride is the sum of the two it replaces,
#: which is why this is a view and not a gather. The other axes keep their
#: order and the diagonal is appended last, as numpy's `diagonal` does.
pub fn diagonal_view(a: NDArray, offset: int = 0, axis1: int = 0, axis2: int = 1) -> NDArray {
    let rank = a.ndim()
    let first = normalize_axis(axis1, rank, "axis1")
    let second = normalize_axis(axis2, rank, "axis2")
    if first == second { throw ValueError(f"axis1 and axis2 are both {first}") }
    var start = a.offset
    var length = 0
    if offset >= 0 {
        length = max(0, min(a.shape[first], a.shape[second] - offset))
        start += offset * a.strides[second]
    } else {
        length = max(0, min(a.shape[first] + offset, a.shape[second]))
        start += -offset * a.strides[first]
    }
    var shape: list[int] = []
    var strides: list[int] = []
    for axis in 0..rank {
        if axis == first or axis == second { continue }
        shape.push(a.shape[axis])
        strides.push(a.strides[axis])
    }
    shape.push(length)
    strides.push(a.strides[first] + a.strides[second])
    return a.view(shape, strides, start, a.dtype)
}

#: A half-open range of one axis, with an optional step.
#:
#: `null` means "the end the step runs from": `sl(null, null, -1)` reverses.
#: Negative bounds count back from the end, as they do in numpy. Prefer a
#: `range` literal — `a[2..7]` — where the bounds are known and positive; this
#: exists for the open ends and the negative steps a `range` cannot spell.
pub class Slice: Printable {
    #: First index, or `null` for the natural start.
    pub let start: int?
    #: One past the last index, or `null` for the natural end.
    pub let stop: int?
    #: Distance between the indices taken. Never zero.
    pub let step: int

    pub fn init(self, start: int? = null, stop: int? = null, step: int = 1) {
        if step == 0 { throw ValueError("a slice step cannot be zero") }
        self.start = start
        self.stop = stop
        self.step = step
    }

    #: First index and count, clamped against an axis of `extent` elements.
    pub fn resolve(self, extent: int) -> (int, int) {
        let step = self.step
        var first = 0
        var last = 0
        if step > 0 {
            first = _clamp_bound(self.start ?? 0, extent, 0, extent)
            last = _clamp_bound(self.stop ?? extent, extent, 0, extent)
        } else {
            first = _clamp_bound(self.start ?? extent - 1, extent, -1, extent - 1)
            last = _clamp_bound(self.stop ?? -extent - 1, extent, -1, extent - 1)
        }
        let span = last - first
        if span == 0 or (span > 0) != (step > 0) { return (first, 0) }
        return (first, (span + step - _sign(step)) // step)
    }

    pub fn to_str(self) -> str {
        let start = if self.start is null { "" } else { f"{self.start}" }
        let stop = if self.stop is null { "" } else { f"{self.stop}" }
        return f"sl({start}, {stop}, {self.step})"
    }

    fn __str__(self) -> str { return self.to_str() }
}

fn _sign(value: int) -> int { return if value < 0 { -1 } else { 1 } }

fn _clamp_bound(value: int, extent: int, low: int, high: int) -> int {
    let shifted = if value < 0 { value + extent } else { value }
    return max(low, min(shifted, high))
}

#: `Slice(start, stop, step)`, spelled the way numpy's `s_` reads.
pub fn sl(start: int? = null, stop: int? = null, step: int = 1) -> Slice {
    return Slice(start, stop, step)
}

#: One of the two index sentinels. There is no reason to build another.
pub class IndexToken: Printable {
    pub let name: str

    pub fn init(self, name: str) { self.name = name }

    pub fn to_str(self) -> str { return self.name }

    fn __str__(self) -> str { return self.name }
}

#: Insert an axis of length one at this position, as `np.newaxis` does.
pub let NEWAXIS = IndexToken("newaxis")

#: Stand in for as many full axes as the key leaves over, as `...` does.
pub let ELLIPSIS = IndexToken("Ellipsis")

#: Reinterpret a float's bits as an integer, so a kernel can take it as a
#: `constant uint&` and read it back with `as_type<float>`.
#:
#: A float scalar that carries no layout — a fill value, a divisor — travels
#: this way. Shapes and strides cannot: they are signed and there are many of
#: them, which is what `layout_buffer` is for.
pub fn f32_bits(value: float) -> int {
    if value != value { return 0x7FC00000 }
    var magnitude = value
    var sign = 0
    if magnitude < 0.0 {
        sign = 1
        magnitude = -magnitude
    }
    #: A negative zero does not compare less than zero, so `copysign` is what
    #: reads its sign. Worth the call: a fill value travels through here, and
    #: -0.0 is a value a caller can mean.
    if magnitude == 0.0 {
        return if __prim__.f64_copysign(1.0, value) < 0.0 { 1 << 31 } else { 0 }
    }
    if magnitude > 3.4028234663852886e38 { return sign << 31 | 0x7F800000 }
    if magnitude <= 1.401298464324817e-45 { return sign << 31 }
    let split = __prim__.f64_frexp(magnitude)
    let biased = split[1] + 126
    if biased <= 0 {
        let fraction = int(__prim__.f64_ldexp(magnitude, 149))
        return sign << 31 | fraction & 0x7FFFFF
    }
    if biased >= 255 { return sign << 31 | 0x7F800000 }
    var fraction = int((split[0] * 2.0 - 1.0) * 8388608.0 + 0.5)
    var exponent = biased
    #: Rounding the mantissa can carry out of it -- 2147483647.0 rounds to
    #: 2^31 -- and masking the carry away instead of adding it to the exponent
    #: turns the largest int32 into 2^30. That is not academic: it is the
    #: saturation bound every integer cast is dispatched with.
    if fraction >= 0x800000 {
        fraction = 0
        exponent += 1
        if exponent >= 255 { return sign << 31 | 0x7F800000 }
    }
    return sign << 31 | exponent << 23 | fraction
}

#: Read a float back out of the bits `f32_bits` made.
pub fn f32_from_bits(bits: int) -> float {
    let sign = if bits & 0x80000000 != 0 { -1.0 } else { 1.0 }
    let biased = bits >> 23 & 0xFF
    let fraction = bits & 0x7FFFFF
    if biased == 0 { return sign * __prim__.f64_ldexp(float(fraction), -149) }
    if biased == 255 {
        if fraction != 0 { return 0.0 / 0.0 }
        return sign * 1.0e308 * 10.0
    }
    return sign * __prim__.f64_ldexp(float(fraction | 0x800000), biased - 150)
}

#: The Metal every jainum kernel is compiled on top of: the index walk, the
#: conversion and arithmetic rules, and the universal passes.
#:
#: A module writes `let SOURCE = PRELUDE + """ ... """` and asks `kernel_of`
#: for its entry points.
pub let PRELUDE = """#include <metal_stdlib>
using namespace metal;

constexpr constant uint ND_GROUP = 256u;
constexpr constant uint ND_WARPS = ND_GROUP / 32u;

/* One float block per dispatch, as `layout_buffer` writes it:
 *
 *   [0]                              rank of the iteration shape
 *   [1 .. rank]                      the iteration shape
 *   [1 + rank + k*(rank+1)]          operand k's element offset
 *   [2 + rank + k*(rank+1) + axis]   operand k's stride along `axis`
 *
 * Floats and not `constant uint&` scalars for two reasons. A stride is signed
 * here -- a flip and a negative-step slice are views, not copies -- and
 * `Kernel.dispatch` binds its trailing scalars as unsigned. And there are
 * rank * operands of them, which is not a fixed argument list. Exact to 2^24,
 * which is also where an element count stops being exact, so nothing that
 * fits in an array fails to fit in its metadata.
 *
 * Operand 0 is the destination, by convention everywhere below. */
inline int nd_rank(device const float* meta) { return int(meta[0]); }

/* A linear element index turned into one operand's float slot.
 *
 * The divmod per axis is the real cost of the general path, which is why
 * every bulk operation here also has a contiguous entry point that takes a
 * flat count and skips this entirely. Most calls take that one. */
inline int nd_offset(device const float* meta, uint operand, uint linear) {
    int rank = int(meta[0]);
    int base = 1 + rank + int(operand) * (rank + 1);
    int at = int(meta[base]);
    uint rest = linear;
    for (int axis = rank - 1; axis >= 0; axis--) {
        uint extent = uint(meta[1 + axis]);
        at += int(rest % extent) * int(meta[base + 1 + axis]);
        rest /= extent;
    }
    return at;
}

/* What an integer or boolean dtype means on float32 storage: rounding to
 * nearest even -- which `rint` is -- and a clamp, or a flattening to 0 and 1.
 * NaN is truthy, as it is in numpy. */
inline float nd_convert(float value, uint mode, float low, float high) {
    if (mode == 1u) return clamp(rint(value), low, high);
    if (mode == 2u) return value != 0.0f ? 1.0f : 0.0f;
    return value;
}

inline float nd_apply(float a, float b, uint op) {
    if (op == 0u) return a + b;
    if (op == 1u) return a - b;
    if (op == 2u) return a * b;
    return a / b;
}

/* Complex arithmetic on an interleaved pair. Division is the naive formula
 * rather than Smith's: the inputs are float32 and the intermediates are
 * float32, so the range Smith's rescaling protects is already gone by the
 * time the values arrive. */
inline float2 nd_apply2(float2 a, float2 b, uint op) {
    if (op == 0u) return a + b;
    if (op == 1u) return a - b;
    if (op == 2u) return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
    float divisor = b.x * b.x + b.y * b.y;
    return float2((a.x * b.x + a.y * b.y) / divisor, (a.y * b.x - a.x * b.y) / divisor);
}

/* A real scalar meets a complex array the way numpy has it: it is a complex
 * number with a zero imaginary part, so it lands on the real lane for a sum
 * and on both lanes for a product. */
inline float2 nd_apply2_scalar(float2 a, float s, uint op) {
    if (op == 0u) return float2(a.x + s, a.y);
    if (op == 1u) return float2(a.x - s, a.y);
    if (op == 2u) return a * s;
    return a / s;
}

/* Exclusive prefix sum of `value` over a 256-thread group, handing every
 * thread its running position and the group's total.
 *
 * `warp_totals` must hold ND_WARPS uints. Barriered at both ends so the same
 * scratch survives a second call. */
inline uint nd_group_scan(
    uint value,
    threadgroup uint* warp_totals,
    uint lane,
    uint warp,
    thread uint* group_total
) {
    uint within = simd_prefix_exclusive_sum(value);
    uint warp_sum = simd_sum(value);
    if (lane == 0u) warp_totals[warp] = warp_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = 0u;
    uint total = 0u;
    for (uint j = 0u; j < ND_WARPS; j++) {
        if (j < warp) base += warp_totals[j];
        total += warp_totals[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    *group_total = total;
    return base + within;
}

/* Copy and cast, dense. `count` is float slots, so lanes need no mention. */
kernel void nd_copy(device float* dst [[buffer(0)]],
                    device const float* src [[buffer(1)]],
                    constant uint& count [[buffer(2)]],
                    constant uint& mode [[buffer(3)]],
                    constant uint& low_bits [[buffer(4)]],
                    constant uint& high_bits [[buffer(5)]],
                    uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    dst[id] = nd_convert(src[id], mode, as_type<float>(low_bits), as_type<float>(high_bits));
}

/* Copy, cast and materialise through the layout buffer, and the one pass that
 * changes lane count: a real source fills a complex destination's imaginary
 * lane with zero, and a complex source dropped into a real destination loses
 * it. Every other kernel in the package takes one lane count for both sides. */
kernel void nd_copy_strided(device float* dst [[buffer(0)]],
                            device const float* src [[buffer(1)]],
                            device const float* meta [[buffer(2)]],
                            constant uint& count [[buffer(3)]],
                            constant uint& dst_lanes [[buffer(4)]],
                            constant uint& src_lanes [[buffer(5)]],
                            constant uint& mode [[buffer(6)]],
                            constant uint& low_bits [[buffer(7)]],
                            constant uint& high_bits [[buffer(8)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    int to = nd_offset(meta, 0u, id);
    int from = nd_offset(meta, 1u, id);
    float low = as_type<float>(low_bits);
    float high = as_type<float>(high_bits);
    for (uint lane = 0u; lane < dst_lanes; lane++) {
        float value = lane < src_lanes ? src[from + int(lane)] : 0.0f;
        dst[to + int(lane)] = nd_convert(value, mode, low, high);
    }
}

/* Fill, dense. `count` is elements: a complex fill takes a real value, so the
 * imaginary lane goes to zero rather than to a copy of it. */
kernel void nd_fill(device float* dst [[buffer(0)]],
                    constant uint& count [[buffer(1)]],
                    constant uint& lanes [[buffer(2)]],
                    constant uint& bits [[buffer(3)]],
                    uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint at = id * lanes;
    dst[at] = as_type<float>(bits);
    for (uint lane = 1u; lane < lanes; lane++) dst[at + lane] = 0.0f;
}

kernel void nd_fill_strided(device float* dst [[buffer(0)]],
                            device const float* meta [[buffer(1)]],
                            constant uint& count [[buffer(2)]],
                            constant uint& lanes [[buffer(3)]],
                            constant uint& bits [[buffer(4)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    int at = nd_offset(meta, 0u, id);
    dst[at] = as_type<float>(bits);
    for (uint lane = 1u; lane < lanes; lane++) dst[at + int(lane)] = 0.0f;
}

/* `a[mask] = value`. Operand 1 is the mask, which broadcasts like any other
 * operand, so a whole row can be switched on by a column of flags. */
kernel void nd_fill_masked(device float* dst [[buffer(0)]],
                           device const float* mask [[buffer(1)]],
                           device const float* meta [[buffer(2)]],
                           constant uint& count [[buffer(3)]],
                           constant uint& lanes [[buffer(4)]],
                           constant uint& bits [[buffer(5)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    if (mask[nd_offset(meta, 1u, id)] == 0.0f) return;
    int at = nd_offset(meta, 0u, id);
    dst[at] = as_type<float>(bits);
    for (uint lane = 1u; lane < lanes; lane++) dst[at + int(lane)] = 0.0f;
}

/* The four arithmetic ops, dense. `count` is elements, because a complex
 * product is not a per-slot operation.
 *
 * `mode`, `low` and `high` are the destination dtype's conversion, the same
 * three `nd_copy` takes: a sum of two int32 arrays that leaves the int32
 * range saturates here rather than being stored as a float the dtype cannot
 * hold. Only the single-lane path converts -- a complex destination is never
 * an integer dtype, so its mode is always CAST_NONE. */
kernel void nd_binary(device float* dst [[buffer(0)]],
                      device const float* a [[buffer(1)]],
                      device const float* b [[buffer(2)]],
                      constant uint& count [[buffer(3)]],
                      constant uint& lanes [[buffer(4)]],
                      constant uint& op [[buffer(5)]],
                      constant uint& mode [[buffer(6)]],
                      constant uint& low_bits [[buffer(7)]],
                      constant uint& high_bits [[buffer(8)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    if (lanes == 1u) {
        float value = nd_apply(a[id], b[id], op);
        dst[id] = nd_convert(value, mode, as_type<float>(low_bits), as_type<float>(high_bits));
        return;
    }
    uint at = id * 2u;
    float2 value = nd_apply2(float2(a[at], a[at + 1u]), float2(b[at], b[at + 1u]), op);
    dst[at] = value.x;
    dst[at + 1u] = value.y;
}

kernel void nd_binary_strided(device float* dst [[buffer(0)]],
                              device const float* a [[buffer(1)]],
                              device const float* b [[buffer(2)]],
                              device const float* meta [[buffer(3)]],
                              constant uint& count [[buffer(4)]],
                              constant uint& lanes [[buffer(5)]],
                              constant uint& op [[buffer(6)]],
                              constant uint& mode [[buffer(7)]],
                              constant uint& low_bits [[buffer(8)]],
                              constant uint& high_bits [[buffer(9)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    int to = nd_offset(meta, 0u, id);
    int left = nd_offset(meta, 1u, id);
    int right = nd_offset(meta, 2u, id);
    if (lanes == 1u) {
        float one = nd_apply(a[left], b[right], op);
        dst[to] = nd_convert(one, mode, as_type<float>(low_bits), as_type<float>(high_bits));
        return;
    }
    float2 value = nd_apply2(float2(a[left], a[left + 1]), float2(b[right], b[right + 1]), op);
    dst[to] = value.x;
    dst[to + 1] = value.y;
}

kernel void nd_binary_scalar(device float* dst [[buffer(0)]],
                             device const float* a [[buffer(1)]],
                             constant uint& count [[buffer(2)]],
                             constant uint& lanes [[buffer(3)]],
                             constant uint& op [[buffer(4)]],
                             constant uint& bits [[buffer(5)]],
                             constant uint& mode [[buffer(6)]],
                             constant uint& low_bits [[buffer(7)]],
                             constant uint& high_bits [[buffer(8)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float scalar = as_type<float>(bits);
    if (lanes == 1u) {
        float one = nd_apply(a[id], scalar, op);
        dst[id] = nd_convert(one, mode, as_type<float>(low_bits), as_type<float>(high_bits));
        return;
    }
    uint at = id * 2u;
    float2 value = nd_apply2_scalar(float2(a[at], a[at + 1u]), scalar, op);
    dst[at] = value.x;
    dst[at + 1u] = value.y;
}

kernel void nd_binary_scalar_strided(device float* dst [[buffer(0)]],
                                     device const float* a [[buffer(1)]],
                                     device const float* meta [[buffer(2)]],
                                     constant uint& count [[buffer(3)]],
                                     constant uint& lanes [[buffer(4)]],
                                     constant uint& op [[buffer(5)]],
                                     constant uint& bits [[buffer(6)]],
                                     constant uint& mode [[buffer(7)]],
                                     constant uint& low_bits [[buffer(8)]],
                                     constant uint& high_bits [[buffer(9)]],
                                     uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    int to = nd_offset(meta, 0u, id);
    int from = nd_offset(meta, 1u, id);
    float scalar = as_type<float>(bits);
    if (lanes == 1u) {
        float one = nd_apply(a[from], scalar, op);
        dst[to] = nd_convert(one, mode, as_type<float>(low_bits), as_type<float>(high_bits));
        return;
    }
    float2 value = nd_apply2_scalar(float2(a[from], a[from + 1]), scalar, op);
    dst[to] = value.x;
    dst[to + 1] = value.y;
}

/* dst[i] = src[index[i]], both sides dense. A negative index counts from the
 * end; anything still outside is clamped, because a kernel cannot raise. The
 * caller validates -- `take` does it with one max reduction, which is cheaper
 * than a per-thread flag buffer nobody reads on the happy path. */
kernel void nd_gather(device float* dst [[buffer(0)]],
                      device const float* src [[buffer(1)]],
                      device const float* index [[buffer(2)]],
                      constant uint& count [[buffer(3)]],
                      constant uint& lanes [[buffer(4)]],
                      constant uint& limit [[buffer(5)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    int at = int(index[id]);
    if (at < 0) at += int(limit);
    at = clamp(at, 0, int(limit) - 1);
    for (uint lane = 0u; lane < lanes; lane++) {
        dst[id * lanes + lane] = src[uint(at) * lanes + lane];
    }
}

/* dst[index[i]] = src[i * src_step], optionally gated by mask[i].
 *
 * The gate is what turns the scan below into a compaction: scan a mask into
 * offsets, then scatter the source through those offsets with the same mask,
 * and the survivors land packed. `src_step` of zero broadcasts one value
 * across every target. When `use_mask` is zero the mask argument is bound to
 * the index buffer, which is read-only and already resident. */
kernel void nd_scatter(device float* dst [[buffer(0)]],
                       device const float* index [[buffer(1)]],
                       device const float* src [[buffer(2)]],
                       device const float* mask [[buffer(3)]],
                       constant uint& count [[buffer(4)]],
                       constant uint& lanes [[buffer(5)]],
                       constant uint& limit [[buffer(6)]],
                       constant uint& use_mask [[buffer(7)]],
                       constant uint& src_step [[buffer(8)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    if (use_mask != 0u && mask[id] == 0.0f) return;
    int at = int(index[id]);
    if (at < 0) at += int(limit);
    at = clamp(at, 0, int(limit) - 1);
    for (uint lane = 0u; lane < lanes; lane++) {
        dst[uint(at) * lanes + lane] = src[id * src_step + lane];
    }
}

/* Pass one of the exclusive scan: how many non-zeros each group's chunk holds.
 *
 * A group owns `span` contiguous elements and each of its threads owns `per`
 * contiguous elements inside that, the same partition pass three re-walks --
 * a prefix has to be built in element order, so the strided read pattern a
 * pure total would prefer is not available to the pass that needs it, and
 * matching partitions is worth more than coalescing one of the two. */
kernel void nd_scan_totals(device const float* src [[buffer(0)]],
                           device float* partials [[buffer(1)]],
                           constant uint& count [[buffer(2)]],
                           constant uint& span [[buffer(3)]],
                           constant uint& per [[buffer(4)]],
                           uint group [[threadgroup_position_in_grid]],
                           uint lid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint warp_totals[ND_WARPS];
    uint start = group * span + lid * per;
    uint mine = 0u;
    for (uint k = 0u; k < per; k++) {
        uint at = start + k;
        if (at < count && src[at] != 0.0f) mine++;
    }
    uint total = 0u;
    nd_group_scan(mine, warp_totals, lane, warp, &total);
    if (lid == 0u) partials[group] = float(total);
}

/* Pass two: one group turns the per-group totals into per-group bases, and
 * reports the grand total. Serial in the number of groups, which is why the
 * grid is capped rather than one group per 256 elements. */
kernel void nd_scan_bases(device float* partials [[buffer(0)]],
                          device float* totals [[buffer(1)]],
                          constant uint& groups [[buffer(2)]],
                          constant uint& per [[buffer(3)]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint warp_totals[ND_WARPS];
    uint start = lid * per;
    uint mine = 0u;
    for (uint k = 0u; k < per; k++) {
        uint at = start + k;
        if (at < groups) mine += uint(partials[at]);
    }
    uint grand = 0u;
    uint running = nd_group_scan(mine, warp_totals, lane, warp, &grand);
    for (uint k = 0u; k < per; k++) {
        uint at = start + k;
        if (at >= groups) break;
        uint held = uint(partials[at]);
        partials[at] = float(running);
        running += held;
    }
    if (lid == 0u) totals[0] = float(grand);
}

/* Pass three: each element's exclusive offset, which is where it would land
 * if the non-zeros were packed. */
kernel void nd_scan_write(device float* dst [[buffer(0)]],
                          device const float* src [[buffer(1)]],
                          device const float* partials [[buffer(2)]],
                          constant uint& count [[buffer(3)]],
                          constant uint& span [[buffer(4)]],
                          constant uint& per [[buffer(5)]],
                          uint group [[threadgroup_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint warp_totals[ND_WARPS];
    uint start = group * span + lid * per;
    uint mine = 0u;
    for (uint k = 0u; k < per; k++) {
        uint at = start + k;
        if (at < count && src[at] != 0.0f) mine++;
    }
    uint ignored = 0u;
    uint running = uint(partials[group]) + nd_group_scan(mine, warp_totals, lane, warp, &ignored);
    for (uint k = 0u; k < per; k++) {
        uint at = start + k;
        if (at >= count) break;
        dst[at] = float(running);
        if (src[at] != 0.0f) running++;
    }
}
"""

let SOURCE = PRELUDE

var _kernels: dict[str, Kernel] = {}

#: The compiled pipeline for one entry point, kept for the life of the process.
#:
#: Keyed on the entry name alone. `Kernel.compile` interns on the source text
#: as well, and hashing a thirty-kilobyte shader costs more than encoding the
#: dispatch that follows it; every entry point in jainum carries its module's
#: prefix, so the names cannot collide and the source need not be consulted.
pub fn kernel_of(source: str, entry: str) -> Kernel {
    let existing = _kernels.get(entry, null)
    if existing is not null { return existing }
    let kernel = Kernel.compile(source, entry)
    _kernels[entry] = kernel
    return kernel
}

fn require_device(what: str) -> void {
    if not gpu.is_available() { throw RuntimeError(f"jainum needs a GPU to {what}") }
}

#: The metadata block a strided dispatch reads: rank, iteration shape, then
#: each operand's offset and strides. Operand 0 is the destination.
#:
#: Every array must already have the iteration rank — call `broadcast_to`
#: first. Allocated fresh for each dispatch and freed straight after encoding:
#: an in-flight command buffer retains it, and a recycled pool of these would
#: need a host-write fence that this repo has been bitten by before.
pub fn layout_buffer(arrays: list[NDArray]) -> Buffer {
    if arrays.len() == 0 { throw ValueError("a layout needs at least one operand") }
    let rank = arrays[0].ndim()
    var values = [float(rank)]
    for extent in arrays[0].shape { values.push(float(extent)) }
    for operand in arrays {
        if operand.ndim() != rank {
            let got = operand.ndim()
            throw ValueError(f"a layout of rank {rank} got rank {got}; broadcast_to it first")
        }
        values.push(float(operand.offset))
        for stride in operand.strides { values.push(float(stride)) }
    }
    let meta = Buffer(values.len())
    meta.upload(values)
    return meta
}

#: A dense n-dimensional array of float32 slots, with a dtype that says how to
#: read them.
pub class NDArray: Printable {
    #: Elements along each axis. Empty for a 0-d array.
    pub let shape: list[int]
    #: Float slots between neighbouring elements along each axis. May be
    #: negative, and is zero on a broadcast axis.
    pub let strides: list[int]
    #: Float slot of the first element inside `storage`.
    pub let offset: int
    #: Semantic tag: rounding, saturation and result type. Not a width.
    pub let dtype: int
    #: The allocation this array is a window onto, shared with every other
    #: window of it.
    pub let storage: Buffer
    #: The array that owns `storage`, or null when this one does.
    pub let base: NDArray?
    var freed: bool

    #: Allocate `shape` elements of `dtype`, uninitialised.
    #:
    #: Pass `storage`, `strides` and `offset` to build a window onto an
    #: existing allocation instead; `view` and `__getitem__` are the callers
    #: that do.
    pub fn init(
        self,
        shape: list[int],
        dtype: int = FLOAT32,
        storage: Buffer? = null,
        strides: list[int]? = null,
        offset: int = 0,
        base: NDArray? = null
    ) {
        if dtype < BOOL or dtype > COMPLEX64 { throw ValueError(f"unknown dtype {dtype}") }
        let lanes = dtype_lanes(dtype)
        self.shape = [extent for extent in shape]
        self.strides = strides ?? contiguous_strides(shape, lanes)
        if self.strides.len() != self.shape.len() {
            throw ValueError(f"{self.strides.len()} strides for {self.shape.len()} axes")
        }
        self.dtype = dtype
        self.offset = offset
        self.base = base
        self.freed = false
        if storage is not null {
            self.storage = storage
            return
        }
        require_device("allocate an NDArray")
        self.storage = Buffer(max(1, element_count(self.shape) * lanes))
    }

    #: A window onto the same allocation with a shape of its own.
    #:
    #: `dtype` of -1 keeps this array's. The window points at whichever array
    #: owns the storage, so a view of a view still frees correctly.
    pub fn view(
        self,
        shape: list[int],
        strides: list[int],
        offset: int,
        dtype: int = -1
    ) -> NDArray {
        let tag = if dtype < 0 { self.dtype } else { dtype }
        return NDArray(shape, tag, self.storage, strides, offset, self.base ?? self)
    }

    #: Elements the array holds.
    pub fn size(self) -> int { return element_count(self.shape) }

    #: Number of axes.
    pub fn ndim(self) -> int { return self.shape.len() }

    #: Float slots one element occupies.
    pub fn lanes(self) -> int { return dtype_lanes(self.dtype) }

    #: Bytes one element occupies, as numpy's `itemsize` reports it. Storage is
    #: float32, so this is 4 per lane whatever the dtype claims to be.
    pub fn itemsize(self) -> int { return 4 * self.lanes() }

    #: The device allocation. Shared with every other window of it.
    pub fn buffer(self) -> Buffer { return self.storage }

    #: Whether the elements run in row-major order with nothing between them.
    pub fn is_contiguous(self) -> bool {
        return is_contiguous(self.shape, self.strides, self.lanes())
    }

    #: A view of the real parts of a complex array.
    #:
    #: No stride arithmetic: strides are counted in float slots, so a complex
    #: array's strides already step over both lanes and the real view keeps
    #: them unchanged. That is most of why they are counted that way.
    pub fn real(self) -> NDArray {
        if not dtype_is_complex(self.dtype) { return self }
        return self.view(self.shape, self.strides, self.offset, FLOAT32)
    }

    #: A view of the imaginary parts of a complex array, or zeros for a real
    #: one — which, being an allocation, is not a view.
    pub fn imag(self) -> NDArray {
        if not dtype_is_complex(self.dtype) {
            let zeros = NDArray(self.shape, self.dtype)
            fill(zeros, 0.0)
            return zeros
        }
        return self.view(self.shape, self.strides, self.offset + 1, FLOAT32)
    }

    #: A view reading this array as if it had `shape`, stretching every
    #: length-one and every missing leading axis.
    pub fn broadcast_to(self, shape: list[int]) -> NDArray {
        if shape == self.shape { return self }
        return self.view(shape, broadcast_strides(self.shape, self.strides, shape), self.offset)
    }

    #: The one value a one-element array holds.
    pub fn item(self) -> float {
        if self.size() != 1 { throw ValueError(f"item() needs one element, not {self.size()}") }
        if dtype_is_complex(self.dtype) {
            throw TypeError("item() has no complex reading; take .real() or .imag()")
        }
        let (dense, owned) = _dense(self)
        defer { if owned { dense.free() } }
        return dense.storage.download(dense.offset, 1)[0]
    }

    #: Every float slot in element order, materialising first when strided.
    #:
    #: A complex array hands back interleaved re/im, twice as many floats as
    #: elements.
    pub fn to_list(self) -> list[float] {
        let (dense, owned) = _dense(self)
        defer { if owned { dense.free() } }
        let count = dense.size() * dense.lanes()
        if count == 0 { return [] }
        return dense.storage.download(dense.offset, count)
    }

    #: The values as nested lists, one level per axis.
    pub fn to_nested(self) -> any {
        return _nest(self.to_list(), self.shape, 0, 0, self.lanes())[0]
    }

    #: Copy `values` in, in element order. The array must be contiguous and the
    #: count must match exactly, lanes included.
    pub fn upload(self, values: list[float]) -> void {
        self._check_live()
        if not self.is_contiguous() {
            throw ValueError("upload needs a contiguous array; call contiguous() first")
        }
        let wanted = self.size() * self.lanes()
        if values.len() != wanted {
            throw ValueError(f"{values.len()} values cannot fill {wanted} slots")
        }
        if wanted == 0 { return }
        self.storage.upload(values, self.offset)
    }

    #: A fresh contiguous array holding the same values.
    pub fn copy(self) -> NDArray {
        let out = NDArray(self.shape, self.dtype)
        copy_into(out, self)
        return out
    }

    #: This array if it is already dense at offset zero, a copy otherwise.
    #:
    #: The offset matters: a caller that reached for this wants to hand
    #: `storage` straight to a kernel that indexes from slot zero.
    pub fn contiguous(self) -> NDArray {
        if self.offset == 0 and self.is_contiguous() { return self }
        return self.copy()
    }

    #: A copy read as `dtype`, rounding and saturating on the way if the target
    #: is an integer dtype, and flattening to 0 and 1 if it is `BOOL`.
    pub fn astype(self, dtype: int) -> NDArray {
        let out = NDArray(self.shape, dtype)
        copy_into(out, self)
        return out
    }

    #: Release the device memory. A view does not own its allocation, so
    #: releasing one leaves the parent alone.
    pub fn free(self) -> void {
        if self.freed { return }
        self.freed = true
        if self.base is null { self.storage.free() }
    }

    #: Whether `free` has been called on this array.
    pub fn is_freed(self) -> bool { return self.freed }

    fn _check_live(self) -> void {
        if self.freed { throw RuntimeError("this NDArray has already been freed") }
    }

    fn __len__(self) -> int {
        if self.ndim() == 0 { throw TypeError("a 0-d array has no length") }
        return self.shape[0]
    }

    #: Shape, dtype, and as much of the values as is cheap to look at.
    #:
    #: A big array shows its leading corner and says so. Printing is a debug
    #: aid and a debug aid that downloads a hundred megabytes is a trap; the
    #: corner costs one small strided copy.
    pub fn to_str(self) -> str {
        if self.freed { return f"NDArray(freed, dtype={dtype_name(self.dtype)})" }
        let head = f"dtype={dtype_name(self.dtype)}"
        if self.size() <= _PRINT_LIMIT {
            return f"array({_render(self.to_nested())}, {head})"
        }
        var corner = self
        for axis in 0..self.ndim() {
            if corner.shape[axis] > _PRINT_EDGE {
                corner = _slice_axis(corner, axis, 0, _PRINT_EDGE, 1)
            }
        }
        return f"array(shape={self.shape}, {head}, corner={_render(corner.to_nested())})"
    }

    fn __str__(self) -> str { return self.to_str() }

    #: A view selected by a basic key, or a packed copy for the two flat fancy
    #: keys.
    #:
    #: Basic keys are an int, a `Slice`, a `range`, `NEWAXIS`, `ELLIPSIS`, or
    #: several of those comma-separated -- which arrives as one tuple -- and
    #: always produce a view. An int drops its axis, so
    #: `a[0]` on a 1-d array is a 0-d array rather than a float — `.item()`
    #: turns it into one, and keeping it an array is what stops every index
    #: expression from being a device sync.
    #:
    #: Fancy keys are an `NDArray` of `BOOL` shaped exactly like this one,
    #: which selects into a 1-d array, and a 1-d integer `NDArray`, which
    #: gathers from a 1-d array. `take` in `index` covers the axis-wise cases.
    fn __getitem__(self, key: any) -> any {
        self._check_live()
        if key is NDArray { return _fancy_get(self, key) }
        return _basic_view(self, if key is tuple { [item for item in key] } else { [key] })
    }

    #: Write through a key.
    #:
    #: A basic key writes through the view it selects, broadcasting an array
    #: value and filling a scalar one. A `BOOL` mask with a scalar value is a
    #: masked fill, and a 1-d integer key scatters.
    fn __setitem__(self, key: any, value: any) -> void {
        self._check_live()
        if key is NDArray { return _fancy_set(self, key, value) }
        let target = _basic_view(self, if key is tuple { [item for item in key] } else { [key] })
        if value is NDArray { copy_into(target, value) } else { fill(target, _as_float(value)) }
    }

    #: Elementwise sum with an array or a scalar.
    #:
    #: There is deliberately no `__eq__`. An elementwise `==` returning an
    #: array breaks `assert_eq`, every `is not null` guard that follows it, and
    #: membership in a dict or a set — all of which need a bool. `equal` and
    #: `array_equal` in `ufunc` are the spellings that work; do not "fix" this.
    #:
    #: A scalar only reads on the right: the VM has no reflected dunder, so
    #: `2.0 - a` raises where `a - 2.0` works. `subtract` in `ufunc` takes both
    #: orders.
    fn __add__(self, other: any) -> NDArray { return binary_arith(self, other, OP_ADD) }

    fn __sub__(self, other: any) -> NDArray { return binary_arith(self, other, OP_SUB) }

    fn __mul__(self, other: any) -> NDArray { return binary_arith(self, other, OP_MUL) }

    fn __div__(self, other: any) -> NDArray { return binary_arith(self, other, OP_DIV) }

    fn __neg__(self) -> NDArray { return _binary_scalar(self, -1.0, self.dtype, OP_MUL) }
}

const _PRINT_LIMIT = 256
const _PRINT_EDGE = 3

fn _render(value: any) -> str {
    if value is list {
        return "[" + ", ".join([_render(item) for item in value]) + "]"
    }
    return f"{value}"
}

fn _nest(values: list[float], shape: list[int], axis: int, at: int, lanes: int) -> (any, int) {
    if axis == shape.len() {
        if lanes == 2 { return (f"{values[at]}{_signed(values[at + 1])}j", at + 2) }
        return (values[at], at + 1)
    }
    var out: list[any] = []
    var cursor = at
    for _index in 0..shape[axis] {
        let (item, next) = _nest(values, shape, axis + 1, cursor, lanes)
        out.push(item)
        cursor = next
    }
    return (out, cursor)
}

fn _signed(value: float) -> str { return if value < 0.0 { f"{value}" } else { f"+{value}" } }

fn _as_float(value: any) -> float {
    if value is float { return value }
    if value is int { return float(value) }
    if value is bool { return if value { 1.0 } else { 0.0 } }
    throw TypeError(f"expected a number, got {type_of(value)}")
}

#: A view of one axis narrowed to `count` elements from `first`, stepping.
fn _slice_axis(a: NDArray, axis: int, first: int, count: int, step: int) -> NDArray {
    var shape = [extent for extent in a.shape]
    var strides = [stride for stride in a.strides]
    shape[axis] = count
    strides[axis] = a.strides[axis] * step
    return a.view(shape, strides, a.offset + first * a.strides[axis])
}

#: A `range` as a `Slice`.
#:
#: `range` exposes start, stop and step but not whether it was written `..` or
#: `..=`, and the two disagree by one. Its length does know, so an exclusive
#: length that differs from the real one is exactly the inclusive case.
fn _slice_of_range(key: range) -> Slice {
    let start = key.start()
    let step = key.step()
    var stop = key.stop()
    let span = stop - start
    var exclusive = 0
    if (span > 0) == (step > 0) and span != 0 { exclusive = (span + step - _sign(step)) // step }
    if key.len() != exclusive { stop += _sign(step) }
    return Slice(start, stop, step)
}

fn _basic_view(a: NDArray, items: list[any]) -> NDArray {
    var expanded: list[any] = []
    var ellipses = 0
    var consumed = 0
    for item in items {
        if item is IndexToken and item.name == ELLIPSIS.name {
            ellipses += 1
        } elif not item is IndexToken {
            consumed += 1
        }
    }
    if ellipses > 1 { throw ValueError("a key may hold one ELLIPSIS at most") }
    for item in items {
        if item is IndexToken and item.name == ELLIPSIS.name {
            for _axis in 0..a.ndim() - consumed { expanded.push(Slice()) }
        } else {
            expanded.push(item)
        }
    }
    if consumed > a.ndim() {
        throw IndexError(f"a key of {consumed} axes on an array of {a.ndim()}")
    }

    var shape: list[int] = []
    var strides: list[int] = []
    var offset = a.offset
    var axis = 0
    for item in expanded {
        if item is IndexToken {
            shape.push(1)
            strides.push(0)
            continue
        }
        let extent = a.shape[axis]
        let stride = a.strides[axis]
        axis += 1
        if item is int {
            let at = if item < 0 { item + extent } else { item }
            if at < 0 or at >= extent {
                throw IndexError(f"index {item} is out of bounds for an axis of {extent}")
            }
            offset += at * stride
            continue
        }
        if not item is Slice and not item is range {
            throw TypeError(f"an index may not be a {type_of(item)}")
        }
        let cut = if item is range { _slice_of_range(item) } else { item }
        let (first, count) = cut.resolve(extent)
        offset += first * stride
        shape.push(count)
        strides.push(stride * cut.step)
    }
    while axis < a.ndim() {
        shape.push(a.shape[axis])
        strides.push(a.strides[axis])
        axis += 1
    }
    return a.view(shape, strides, offset)
}

fn _fancy_get(a: NDArray, key: NDArray) -> NDArray {
    if dtype_is_bool(key.dtype) {
        if key.shape != a.shape {
            throw IndexError(f"a mask of {key.shape} on an array of {a.shape}")
        }
        let (offsets, total) = exclusive_scan(key)
        defer { offsets.free() }
        let out = NDArray([total], a.dtype)
        scatter_flat(out, offsets, a, key)
        return out
    }
    if a.ndim() != 1 {
        throw IndexError(
            f"an integer array indexes a 1-d array; use take() for {a.ndim()} dimensions"
        )
    }
    return gather_flat(a, key)
}

fn _fancy_set(a: NDArray, key: NDArray, value: any) -> void {
    if dtype_is_bool(key.dtype) {
        if value is NDArray {
            throw TypeError("a masked write takes a scalar; use where() for an array")
        }
        fill(a, _as_float(value), key)
        return
    }
    if a.ndim() != 1 { throw IndexError("an integer array writes into a 1-d array") }
    let source = if value is NDArray { value } else { _scalar_array(_as_float(value), a.dtype) }
    scatter_flat(a, key, source)
    if not value is NDArray { source.free() }
}

fn _scalar_array(value: float, dtype: int) -> NDArray {
    let out = NDArray([], dtype)
    fill(out, value)
    return out
}

#: `a` as a dense array starting at slot zero, and whether that cost an
#: allocation the caller now owns.
#:
#: `contiguous()` returns the array itself when it can, so freeing its result
#: unconditionally would free the caller's array. The flag is the only honest
#: way to say which of the two happened.
fn _dense(a: NDArray) -> (NDArray, bool) {
    if a.offset == 0 and a.is_contiguous() { return (a, false) }
    return (a.copy(), true)
}

#: The conversion mode a destination dtype asks of the copy kernels.
fn _cast_mode(dtype: int) -> int {
    if dtype == BOOL { return CAST_BOOL }
    if dtype_is_integer(dtype) { return CAST_ROUND }
    return CAST_NONE
}

#: Copy `src` into `dst`, broadcasting and converting as the shapes and dtypes
#: require.
pub fn copy_into(dst: NDArray, src: NDArray) -> void {
    let count = dst.size()
    if count == 0 { return }
    let source = src.broadcast_to(dst.shape)
    let mode = _cast_mode(dst.dtype)
    let low = f32_bits(dtype_min(dst.dtype))
    let high = f32_bits(dtype_max(dst.dtype))
    let lanes = dst.lanes()
    let dense_pair = dst.offset == 0 and dst.is_contiguous() and source.offset == 0
    if dense_pair and source.is_contiguous() and lanes == source.lanes() {
        kernel_of(SOURCE, "nd_copy").dispatch_async(
            [dst.storage, source.storage],
            count * lanes,
            [count * lanes, mode, low, high]
        )
        return
    }
    let meta = layout_buffer([dst, source])
    defer { meta.free() }
    kernel_of(SOURCE, "nd_copy_strided").dispatch_async(
        [dst.storage, source.storage, meta],
        count,
        [count, lanes, source.lanes(), mode, low, high]
    )
}

#: Set every element of `a` to `value`, or only those where `mask` is non-zero.
#:
#: A complex array takes the value on its real lane and zero on its imaginary
#: one, which is what assigning a real number to a complex array means.
pub fn fill(a: NDArray, value: float, mask: NDArray? = null) -> void {
    let count = a.size()
    if count == 0 { return }
    let lanes = a.lanes()
    let bits = f32_bits(value)
    if mask is not null {
        let gate = mask.broadcast_to(a.shape)
        let meta = layout_buffer([a, gate])
        defer { meta.free() }
        kernel_of(SOURCE, "nd_fill_masked").dispatch_async(
            [a.storage, gate.storage, meta],
            count,
            [count, lanes, bits]
        )
        return
    }
    if a.offset == 0 and a.is_contiguous() {
        kernel_of(SOURCE, "nd_fill").dispatch_async([a.storage], count, [count, lanes, bits])
        return
    }
    let meta = layout_buffer([a])
    defer { meta.free() }
    kernel_of(SOURCE, "nd_fill_strided").dispatch_async(
        [a.storage, meta],
        count,
        [count, lanes, bits]
    )
}

#: `left` op `right`, where `right` is an array or a scalar.
#:
#: The five operators on `NDArray` are the only callers that matter; `ufunc`
#: re-exports them under numpy's names rather than compiling the arithmetic a
#: second time. A class method cannot live in another file and the import DAG
#: forbids a cycle, so this is where they have to be.
pub fn binary_arith(left: NDArray, right: any, op: int) -> NDArray {
    if right is NDArray { return _binary_arrays(left, right, op) }
    return _binary_scalar(left, _as_float(right), _scalar_dtype(right), op)
}

#: The dtype a bare Jaithon number contributes to a promotion.
#:
#: Its own type and not its value: numpy's value-based rule would make
#: `ints + 1.0` a float array and `ints + 1` an int one, and reading that off
#: `1.0 == trunc(1.0)` instead gets the first case wrong.
fn _scalar_dtype(value: any) -> int {
    if value is bool { return BOOL }
    if value is int { return INT32 }
    return FLOAT32
}

#: The dtype an arithmetic result carries.
#:
#: Two bools give an int rather than numpy's or-for-add, because half of these
#: operators have no boolean reading at all and one rule beats four. True
#: division always leaves the integers, as it does in numpy.
#:
#: An integer result saturates on the way out, the same rule `astype` and the
#: `ufunc` kernels follow: 2000000000 + 2000000000 in an int32 array answers
#: the int32 ceiling. numpy wraps there instead, which is a reading of a
#: 32-bit register this package does not have -- the value lives in float32
#: storage, where the wrapped pattern was never computed.
fn _result_dtype(a: int, b: int, op: int) -> int {
    var out = promote(a, b)
    if out == BOOL { out = INT32 }
    if op == OP_DIV and dtype_is_integer(out) { out = FLOAT32 }
    return out
}

fn _binary_arrays(left: NDArray, right: NDArray, op: int) -> NDArray {
    let dtype = _result_dtype(left.dtype, right.dtype, op)
    let (a, b) = broadcast_pair(left, right)
    let lanes = dtype_lanes(dtype)

    #: One side is complex and the other is not. Widening both first is a real
    #: materialisation, but it happens only on that mix, and it keeps every
    #: kernel below on a single lane count.
    if lanes != a.lanes() or lanes != b.lanes() {
        let widened_a = a.astype(dtype)
        let widened_b = b.astype(dtype)
        defer { widened_a.free() }
        defer { widened_b.free() }
        return _binary_arrays(widened_a, widened_b, op)
    }
    let out = NDArray(a.shape, dtype)
    let count = out.size()
    if count == 0 { return out }
    let mode = _cast_mode(dtype)
    let low = f32_bits(dtype_min(dtype))
    let high = f32_bits(dtype_max(dtype))
    if a.offset == 0 and a.is_contiguous() and b.offset == 0 and b.is_contiguous() {
        kernel_of(SOURCE, "nd_binary").dispatch_async(
            [out.storage, a.storage, b.storage],
            count,
            [count, lanes, op, mode, low, high]
        )
        return out
    }
    let meta = layout_buffer([out, a, b])
    defer { meta.free() }
    kernel_of(SOURCE, "nd_binary_strided").dispatch_async(
        [out.storage, a.storage, b.storage, meta],
        count,
        [count, lanes, op, mode, low, high]
    )
    return out
}

fn _binary_scalar(a: NDArray, value: float, scalar_dtype: int, op: int) -> NDArray {
    let dtype = _result_dtype(a.dtype, scalar_dtype, op)
    let out = NDArray(a.shape, dtype)
    let count = out.size()
    if count == 0 { return out }
    let lanes = out.lanes()
    let bits = f32_bits(value)
    let mode = _cast_mode(dtype)
    let low = f32_bits(dtype_min(dtype))
    let high = f32_bits(dtype_max(dtype))
    if a.offset == 0 and a.is_contiguous() and lanes == a.lanes() {
        kernel_of(SOURCE, "nd_binary_scalar").dispatch_async(
            [out.storage, a.storage],
            count,
            [count, lanes, op, bits, mode, low, high]
        )
        return out
    }
    let meta = layout_buffer([out, a.broadcast_to(out.shape)])
    defer { meta.free() }
    kernel_of(SOURCE, "nd_binary_scalar_strided").dispatch_async(
        [out.storage, a.storage, meta],
        count,
        [count, lanes, op, bits, mode, low, high]
    )
    return out
}

#: `source` read at each flat index in `indices`, shaped like `indices`.
#:
#: A negative index counts from the end. One outside the array is clamped
#: rather than raised on — a kernel cannot throw — so a caller that owes numpy
#: an `IndexError` checks the extremes first.
pub fn gather_flat(source: NDArray, indices: NDArray) -> NDArray {
    require_real(indices, "gather_flat")
    let (dense, owns_dense) = _dense(source)
    let (keys, owns_keys) = _dense(indices)
    defer { if owns_dense { dense.free() } }
    defer { if owns_keys { keys.free() } }
    let out = NDArray(indices.shape, source.dtype)
    let count = out.size()
    if count == 0 { return out }
    kernel_of(SOURCE, "nd_gather").dispatch_async(
        [out.storage, dense.storage, keys.storage],
        count,
        [count, out.lanes(), max(1, dense.size())]
    )
    return out
}

#: Write `values` into `dst` at each flat index in `indices`.
#:
#: `values` is either as long as `indices` or a single element broadcast over
#: all of them. Two indices naming the same slot race, and which one lands is
#: undefined — numpy's "the last one wins" is a promise a thread grid cannot
#: keep. A non-null `mask`, as long as `indices`, drops the writes where
#: it is zero — which is what makes a compaction out of `exclusive_scan`:
#: scan a mask into offsets, then scatter through those offsets under the same
#: mask, and the survivors land packed. `index`, `sorting` and `stats` all
#: build on that pair.
pub fn scatter_flat(
    dst: NDArray,
    indices: NDArray,
    values: NDArray,
    mask: NDArray? = null
) -> void {
    if not dst.is_contiguous() or dst.offset != 0 {
        throw ValueError("scatter_flat writes into a contiguous array")
    }
    let (keys, owns_keys) = _dense(indices)
    let (source, owns_source) = _dense(values)
    defer { if owns_keys { keys.free() } }
    defer { if owns_source { source.free() } }
    let count = keys.size()
    if count == 0 { return }
    if source.size() != count and source.size() != 1 {
        throw ValueError(f"{source.size()} values for {count} indices")
    }
    if mask is not null and mask.size() != count {
        throw ValueError(f"a mask of {mask.size()} for {count} indices")
    }
    let lanes = dst.lanes()
    let step = if source.size() == 1 { 0 } else { lanes }
    let gated = if mask is null { 0 } else { 1 }
    #: An unused gate still has to be bound, and the index buffer is the one
    #: read-only allocation already resident for this dispatch.
    let (gate, owns_gate) = _dense(mask ?? keys)
    defer { if owns_gate { gate.free() } }
    kernel_of(SOURCE, "nd_scatter").dispatch_async(
        [dst.storage, keys.storage, source.storage, gate.storage],
        count,
        [count, lanes, max(1, dst.size()), gated, step]
    )
}

#: Where each element would land if the non-zeros of `mask` were packed, plus
#: how many there are.
#:
#: Three dispatches over a fixed grid: per-group totals, one group turning
#: those into bases, then the per-element offsets. The grid is capped so the
#: middle pass stays serial in a number of groups it can hold, and the outer
#: two partition their chunks identically so the prefix is built in element
#: order.
#:
#: The returned count is the one host-visible value, and reading it costs a
#: synchronise — compaction cannot size its output without it.
pub fn exclusive_scan(mask: NDArray) -> (NDArray, int) {
    require_real(mask, "exclusive_scan")
    let (dense, owned) = _dense(mask)
    defer { if owned { dense.free() } }
    let count = dense.size()
    let offsets = NDArray([count], INT32)
    if count == 0 { return (offsets, 0) }

    let per = max(1, (count + _SCAN_GROUPS * GROUP - 1) // (_SCAN_GROUPS * GROUP))
    let span = per * GROUP
    let groups = (count + span - 1) // span
    let partials = Buffer(groups)
    let totals = Buffer(1)
    defer { partials.free() }
    defer { totals.free() }

    kernel_of(SOURCE, "nd_scan_totals").dispatch_async(
        [dense.storage, partials],
        groups * GROUP,
        [count, span, per],
        GROUP
    )
    kernel_of(SOURCE, "nd_scan_bases").dispatch_async(
        [partials, totals],
        GROUP,
        [groups, (groups + GROUP - 1) // GROUP],
        GROUP
    )
    kernel_of(SOURCE, "nd_scan_write").dispatch_async(
        [offsets.storage, dense.storage, partials],
        groups * GROUP,
        [count, span, per],
        GROUP
    )
    #: The download is the sync point -- it waits on the batch that wrote this
    #: buffer, and only on that -- so no global synchronise is needed here.
    return (offsets, int(totals.download(0, 1)[0]))
}

#: Groups the scan runs over at most. 1024 of them is a million elements at
#: one per thread before the per-thread chunk grows, and 1024 partials is four
#: per thread in the single group that scans them.
const _SCAN_GROUPS = 1024

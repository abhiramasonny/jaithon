#: Array creation — everything that makes an array out of something that is
#: not already one.
#:
#: The rule this module exists to enforce: a generator writes on the device.
#: `arange`, `linspace`, `logspace` and `geomspace` are one ramp kernel;
#: `eye` and `tri` are one index-predicate kernel. Building a `list[float]` of
#: a million values on the host and uploading it is the mistake this file is
#: here to avoid — a host loop over a million elements costs seconds in this
#: language, and the upload costs the transfer on top. The only entry point
#: allowed to upload is `array(nested)`, whose input was already a host list,
#: and `from_bytes`, which hands the bytes straight to `Buffer.upload_u8`
#: without boxing a float per element.
#:
#: ```
#: let x = linspace(0.0, 1.0, 1024)      #: one dispatch, no host list
#: let grid = fromfunction("i0 * i1", [512, 512])
#: let mask = tri(8, 8, 0)
#: ```
#:
#: Two rules worth reading first, because both are places numpy's behaviour is
#: easy to guess wrong:
#:
#: - **`linspace` endpoint.** With `endpoint` true the gap is
#:   `(stop - start) / (num - 1)` and the last sample *is* `stop`, written in
#:   exactly rather than computed — the ramp is `start + i * delta`, which
#:   lands beside the endpoint and not on it. With `endpoint` false the gap is
#:   `(stop - start) / num` and `stop` is not in the array at all. `num` of 1
#:   gives `[start]` under either flag.
#: - **`arange` with a float step.** The count is numpy's,
#:   `ceil((stop - start) / step)` floored at zero, and not "keep adding until
#:   you pass stop". The two part company whenever the division rounds up:
#:   `(1.3 - 1.0) / 0.1` is 3.0000000000000004, so `arange(1.0, 1.3, 0.1)` has
#:   four samples and the fourth is the stop the half-open interval was
#:   supposed to exclude.
#:
#: `fromfunction` takes a Metal expression rather than a host callback. It is
#: compiled into a kernel with `i0`, `i1`, ... in scope as the float
#: coordinates of the element being written, so it runs once per thread on the
#: device; a Jaithon closure called a million times from the host would be the
#: same seconds this module exists to avoid.
#:
#: Not here: random creation (`random`), anything that reshapes or joins
#: arrays that already exist (`manipulate`), and reading files.
import std.math
from std.gpu import Kernel
from .array import BOOL, CAST_BOOL, CAST_NONE, CAST_ROUND, COMPLEX64, FLOAT32, INT32
from .array import MAX_EXACT_INT, NDArray, PRELUDE, contiguous_strides, copy_into
from .array import diagonal_view, dtype_is_complex, dtype_is_integer, dtype_lanes
from .array import dtype_max, dtype_min, element_count, f32_bits, fill, kernel_of
from .array import layout_buffer, promote

let SOURCE = PRELUDE + """

/* One sample of a ramp: `base + index * delta`, optionally read as an
 * exponent.
 *
 * Every generator in this module is that line. `logspace` raises `root` to
 * it; `geomspace` is the same with root e and the sign carried outside the
 * exponent, because a progression through negative numbers has no logarithm.
 * One flag covers both, and neither needs a kernel of its own.
 *
 * Computed from the index and not accumulated: a running sum gives element i
 * the rounding of i additions, so the far end of a million-point linspace
 * drifts off the endpoint it was asked for. `fma` holds this to one. */
inline float nc_sample(float index, float base, float delta,
                       uint curve, float root, float sign) {
    float x = fma(index, delta, base);
    return curve == 0u ? x : sign * pow(root, x);
}

/* `nd_convert` a quad at a time. rint, clamp and select are all
 * component-wise in Metal, so the vector form is the scalar one with the
 * types widened and nothing else changed. */
inline float4 nc_convert4(float4 value, uint mode, float low, float high) {
    if (mode == 1u) return clamp(rint(value), float4(low), float4(high));
    if (mode == 2u) return select(float4(0.0f), float4(1.0f), value != 0.0f);
    return value;
}

/* The ramp, and the only generator dispatch this module makes.
 *
 * `wide` picks the write path. When the run divides by four a thread owns a
 * quad and stores through a float4 view of the destination -- always whole
 * and always aligned, because the destination is a fresh dense allocation and
 * this kernel is its only writer -- and otherwise a thread owns one element
 * and `count` is elements rather than quads. The branch is uniform across the
 * grid, so it is a predicted jump and not a divergence. */
kernel void nc_ramp(device float* dst [[buffer(0)]],
                    constant uint& count [[buffer(1)]],
                    constant uint& base_bits [[buffer(2)]],
                    constant uint& delta_bits [[buffer(3)]],
                    constant uint& curve [[buffer(4)]],
                    constant uint& root_bits [[buffer(5)]],
                    constant uint& sign_bits [[buffer(6)]],
                    constant uint& last_bits [[buffer(7)]],
                    constant uint& pin [[buffer(8)]],
                    constant uint& mode [[buffer(9)]],
                    constant uint& low_bits [[buffer(10)]],
                    constant uint& high_bits [[buffer(11)]],
                    constant uint& wide [[buffer(12)]],
                    uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float base = as_type<float>(base_bits);
    float delta = as_type<float>(delta_bits);
    float root = as_type<float>(root_bits);
    float sign = as_type<float>(sign_bits);
    float low = as_type<float>(low_bits);
    float high = as_type<float>(high_bits);
    bool endmost = pin != 0u && id == count - 1u;
    if (wide == 0u) {
        float value = nc_sample(float(id), base, delta, curve, root, sign);
        if (endmost) value = as_type<float>(last_bits);
        dst[id] = nd_convert(value, mode, low, high);
        return;
    }
    float first = float(id * 4u);
    float4 value = float4(nc_sample(first, base, delta, curve, root, sign),
                          nc_sample(first + 1.0f, base, delta, curve, root, sign),
                          nc_sample(first + 2.0f, base, delta, curve, root, sign),
                          nc_sample(first + 3.0f, base, delta, curve, root, sign));
    if (endmost) value.w = as_type<float>(last_bits);
    ((device float4*)dst)[id] = nc_convert4(value, mode, low, high);
}

/* Whether (row, column) is inside the band, given only their difference.
 *
 * Three comparisons is all four of `eye`, `tri`, `tril` and `triu` are, which
 * is why they share a kernel: `gate` 0 is the single diagonal `eye` marks, 1
 * is the at-or-below half `tri` and `tril` keep, 2 is the at-or-above half
 * `triu` keeps. `k` arrives biased by `rows` so it can travel as the unsigned
 * scalar a dispatch binds; the host has already clamped it into
 * `[-rows, cols]`, where every larger magnitude gives the same answer. */
inline float nc_band(int diagonal, int k, uint gate) {
    bool inside = gate == 0u ? diagonal == k : (gate == 1u ? diagonal <= k : diagonal >= k);
    return inside ? 1.0f : 0.0f;
}

/* `eye` and `tri`.
 *
 * The row costs one division and the column one subtraction. A `/` and a `%`
 * apiece would be two integer divisions on a device that has none, which for
 * a kernel this cheap is the whole of its cost. `wide` is the same split the
 * ramp makes: a thread owns four columns of one row when the row divides by
 * four, and `cols` arrives already divided. */
kernel void nc_band_write(device float* dst [[buffer(0)]],
                          constant uint& count [[buffer(1)]],
                          constant uint& cols [[buffer(2)]],
                          constant uint& shift [[buffer(3)]],
                          constant uint& rows [[buffer(4)]],
                          constant uint& gate [[buffer(5)]],
                          constant uint& wide [[buffer(6)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint r = id / cols;
    uint c = id - r * cols;
    int k = int(shift) - int(rows);
    if (wide == 0u) {
        dst[id] = nc_band(int(c) - int(r), k, gate);
        return;
    }
    int diagonal = int(c * 4u) - int(r);
    ((device float4*)dst)[id] =
        float4(nc_band(diagonal, k, gate), nc_band(diagonal + 1, k, gate),
               nc_band(diagonal + 2, k, gate), nc_band(diagonal + 3, k, gate));
}

/* `tril` and `triu`, over the copy the host has already made: one division
 * per level -- the plane, then the row -- and a subtraction for each
 * remainder.
 *
 * No quad path here, and the reason is the opposite of the one above. A
 * thread that keeps its element writes nothing at all, so the scalar kernel
 * touches memory only on the half it is clearing; a float4 store covering
 * four columns would have to read them back first wherever the band cuts
 * through the quad, turning a one-sided write into a read and a write for
 * every element of the array. */
kernel void nc_band_zero(device float* dst [[buffer(0)]],
                         constant uint& count [[buffer(1)]],
                         constant uint& plane [[buffer(2)]],
                         constant uint& cols [[buffer(3)]],
                         constant uint& shift [[buffer(4)]],
                         constant uint& rows [[buffer(5)]],
                         constant uint& gate [[buffer(6)]],
                         constant uint& lanes [[buffer(7)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint sheet = id / plane;
    uint within = id - sheet * plane;
    uint r = within / cols;
    uint c = within - r * cols;
    if (nc_band(int(c) - int(r), int(shift) - int(rows), gate) != 0.0f) return;
    uint at = id * lanes;
    for (uint lane = 0u; lane < lanes; lane++) dst[at + lane] = 0.0f;
}
"""

#: Band selector: the single diagonal itself, which is what `eye` marks.
const GATE_DIAGONAL = 0
#: Band selector: the diagonal and everything below it — `tri` and `tril`.
const GATE_BELOW = 1
#: Band selector: the diagonal and everything above it — `triu`.
const GATE_ABOVE = 2

#: Ramp shape: the sample is the linear value itself.
const CURVE_LINEAR = 0
#: Ramp shape: the sample is `root` raised to the linear value, signed.
const CURVE_POWER = 1

#: `shape` elements of `dtype`, holding whatever the allocator left behind.
pub fn empty(shape: list[int], dtype: int = FLOAT32) -> NDArray {
    return NDArray(shape, dtype)
}

#: `shape` elements of `dtype`, all zero.
pub fn zeros(shape: list[int], dtype: int = FLOAT32) -> NDArray {
    let out = NDArray(shape, dtype)
    fill(out, 0.0)
    return out
}

#: `shape` elements of `dtype`, all one.
pub fn ones(shape: list[int], dtype: int = FLOAT32) -> NDArray {
    let out = NDArray(shape, dtype)
    fill(out, 1.0)
    return out
}

#: `shape` elements of `dtype`, all `value`, converted the way `dtype` says.
pub fn full(shape: list[int], value: float, dtype: int = FLOAT32) -> NDArray {
    let out = NDArray(shape, dtype)
    fill(out, _cast_scalar(value, dtype))
    return out
}

#: An uninitialised array shaped and typed like `a`. A `dtype` of -1 keeps
#: `a`'s, which is what every `_like` here means by the default.
pub fn empty_like(a: NDArray, dtype: int = -1) -> NDArray {
    return NDArray(a.shape, _or_dtype(dtype, a))
}

#: Zeros shaped and typed like `a`.
pub fn zeros_like(a: NDArray, dtype: int = -1) -> NDArray {
    return zeros(a.shape, _or_dtype(dtype, a))
}

#: Ones shaped and typed like `a`.
pub fn ones_like(a: NDArray, dtype: int = -1) -> NDArray {
    return ones(a.shape, _or_dtype(dtype, a))
}

#: `value` everywhere, shaped and typed like `a`.
pub fn full_like(a: NDArray, value: float, dtype: int = -1) -> NDArray {
    return full(a.shape, value, _or_dtype(dtype, a))
}

#: Evenly spaced values over a half-open interval, as numpy's `arange`.
#:
#: One argument is the stop and the start is zero, two are start and stop, and
#: `step` may be negative. The dtype, when it is not given, is the promotion of
#: whatever the three arguments were: `arange(5)` is `INT32` and
#: `arange(5.0)` is `FLOAT32`, on their argument types and never on their
#: values.
#:
#: The count is numpy's `ceil((stop - start) / step)`, floored at zero, and
#: not "add until you pass stop"; the two differ whenever the division rounds
#: up. A span needing more than `MAX_EXACT_INT` elements is refused rather
#: than allocated, because past there the index itself stops being exact in
#: float32 and the ramp would start repeating values.
pub fn arange(start: any, stop: any = null, step: any = 1, dtype: int = -1) -> NDArray {
    let bare = stop is null
    let first = if bare { 0.0 } else { _number(start) }
    let last = if bare { _number(start) } else { _number(stop) }
    let delta = _number(step)
    if delta == 0.0 { throw ValueError("arange needs a step other than zero") }

    let span = (last - first) / delta
    if span != span { throw ValueError(f"arange cannot count from {first} to {last} by {delta}") }
    if span > float(MAX_EXACT_INT) {
        throw ValueError(f"arange of {span} elements is past MAX_EXACT_INT")
    }
    let count = if span <= 0.0 { 0 } else { int(math.ceil(span)) }

    var tag = promote(_number_dtype(start), _number_dtype(step))
    if not bare { tag = promote(tag, _number_dtype(stop)) }
    if dtype >= 0 { tag = dtype }
    return _linear(count, first, delta, tag)
}

#: `num` evenly spaced samples from `start`, as numpy's `linspace`.
#:
#: With `endpoint` the gap is `(stop - start) / (num - 1)` and the last sample
#: is `stop` exactly — written in rather than computed, because
#: `start + i * delta` lands beside the endpoint and not on it. Without it the
#: gap is `(stop - start) / num` and `stop` is not in the array. `num` of 1
#: gives `[start]` either way.
pub fn linspace(
    start: float,
    stop: float,
    num: int = 50,
    endpoint: bool = true,
    dtype: int = FLOAT32
) -> NDArray {
    _require_count(num, "linspace")
    let delta = linspace_step(start, stop, num, endpoint)
    return _ramp(num, start, delta, CURVE_LINEAR, 0.0, 1.0, endpoint and num > 1, stop, dtype)
}

#: The gap between two neighbouring `linspace` samples — numpy's `retstep`.
#:
#: A function of its own and not a second return value, which under a flag
#: would make every caller unpack a shape it did not ask for. `num` of 1 has
#: no gap to report and this answers 0.0 where numpy answers NaN; the array is
#: `[start]` under both.
pub fn linspace_step(start: float, stop: float, num: int, endpoint: bool = true) -> float {
    let divisions = if endpoint { num - 1 } else { num }
    if divisions <= 0 { return 0.0 }
    return (stop - start) / float(divisions)
}

#: `num` samples spaced evenly on a log scale — `base ** linspace(start, stop)`
#: — with the same endpoint rule `linspace` has.
#:
#: The endpoint is pinned to `base ** stop` computed here rather than to the
#: device's `pow` of the last sample, which is both exact at the end and
#: cheaper than making the kernel more careful everywhere else.
pub fn logspace(
    start: float,
    stop: float,
    num: int = 50,
    endpoint: bool = true,
    base: float = 10.0,
    dtype: int = FLOAT32
) -> NDArray {
    _require_count(num, "logspace")
    let delta = linspace_step(start, stop, num, endpoint)
    let pin = endpoint and num > 1
    return _ramp(num, start, delta, CURVE_POWER, base, 1.0, pin, base ** stop, dtype)
}

#: `num` samples in a geometric progression from `start` to `stop`.
#:
#: A progression through zero does not exist and one that changes sign is not
#: geometric, so both are refused. A wholly negative one is fine: the ramp runs
#: over the logarithms of the magnitudes and the sign rides outside the
#: exponent.
pub fn geomspace(
    start: float,
    stop: float,
    num: int = 50,
    endpoint: bool = true,
    dtype: int = FLOAT32
) -> NDArray {
    _require_count(num, "geomspace")
    if start == 0.0 or stop == 0.0 { throw ValueError("geomspace endpoints cannot be zero") }
    if (start < 0.0) != (stop < 0.0) {
        throw ValueError(f"geomspace endpoints {start} and {stop} do not share a sign")
    }
    let sign = if start < 0.0 { -1.0 } else { 1.0 }
    let low = math.ln(math.fabs(start))
    let high = math.ln(math.fabs(stop))
    let delta = linspace_step(low, high, num, endpoint)
    let pin = endpoint and num > 1
    return _ramp(num, low, delta, CURVE_POWER, math.E, sign, pin, stop, dtype)
}

#: A `rows` x `cols` array with ones on the `k`-th diagonal and zeros
#: elsewhere. `cols` defaults to `rows`.
pub fn eye(rows: int, cols: int? = null, k: int = 0, dtype: int = FLOAT32) -> NDArray {
    return _band(rows, cols ?? rows, k, GATE_DIAGONAL, dtype, "eye")
}

#: The `n` x `n` identity.
pub fn identity(n: int, dtype: int = FLOAT32) -> NDArray { return eye(n, n, 0, dtype) }

#: A `rows` x `cols` array with ones at and below the `k`-th diagonal.
pub fn tri(rows: int, cols: int? = null, k: int = 0, dtype: int = FLOAT32) -> NDArray {
    return _band(rows, cols ?? rows, k, GATE_BELOW, dtype, "tri")
}

#: A copy of `a` with everything above the `k`-th diagonal zeroed, over the
#: last two axes of a stack of any rank.
pub fn tril(a: NDArray, k: int = 0) -> NDArray { return _triangle(a, k, GATE_BELOW, "tril") }

#: A copy of `a` with everything below the `k`-th diagonal zeroed.
pub fn triu(a: NDArray, k: int = 0) -> NDArray { return _triangle(a, k, GATE_ABOVE, "triu") }

#: A 1-d array laid on the `k`-th diagonal of a square one, or that diagonal
#: read back out of a 2-d array.
#:
#: The extraction is a copy and not the view `diagonal_view` hands back. numpy
#: returns a read-only view there and jainum has no read-only array, so a view
#: would be a write path into the source that numpy never offered; reach for
#: `diagonal_view` directly when the aliasing is what you want.
pub fn diag(v: NDArray, k: int = 0) -> NDArray {
    if v.ndim() == 1 {
        let side = v.shape[0] + (if k < 0 { -k } else { k })
        let out = zeros([side, side], v.dtype)
        copy_into(diagonal_view(out, k), v)
        return out
    }
    if v.ndim() != 2 { throw ValueError(f"diag takes one or two axes, got {v.ndim()}") }
    return diagonal_view(v, k).copy()
}

#: `diag` of the flattened input, whatever its rank.
pub fn diagflat(v: NDArray, k: int = 0) -> NDArray {
    let dense = v.contiguous()
    let flat = dense.view([dense.size()], [dense.lanes()], dense.offset)
    let out = diag(flat, k)
    if not dense == v { dense.free() }
    return out
}

#: An array from nested Jaithon lists, with the shape and dtype read off them.
#:
#: The one entry point here that uploads, and the only one that may: its input
#: is already a host list, so there is nothing on the device to generate from.
#: The dtype comes from the leaves' types and not their values — all `bool` is
#: `BOOL`, all `int` is `INT32`, anything with a `float` in it is `FLOAT32` —
#: and an empty input is `FLOAT32`, as numpy's is float64. A ragged input is
#: refused rather than padded.
pub fn array(values: any, dtype: int = -1) -> NDArray {
    if values is NDArray { return values.astype(if dtype < 0 { values.dtype } else { dtype }) }
    var shape: list[int] = []
    _measure(values, shape)
    var flat: list[float] = []
    var found = _flatten(values, shape, 0, flat)
    if element_count(shape) == 0 { found = FLOAT32 }
    let tag = if dtype < 0 { found } else { dtype }
    if not dtype_is_complex(tag) { return _upload(flat, shape, tag) }
    #: A host number has no imaginary part to carry, so a complex request is
    #: the real array widened — the same copy `astype` already runs.
    let staged = _upload(flat, shape, FLOAT32)
    defer { staged.free() }
    return staged.astype(tag)
}

#: `values` as an array, without copying one that already is.
#:
#: The result may alias the argument, which is the whole difference from
#: `array`: freeing it can free the caller's array. Take `array` when the
#: result outlives the input.
pub fn asarray(values: any, dtype: int = -1) -> NDArray {
    if values is NDArray and (dtype < 0 or dtype == values.dtype) { return values }
    return array(values, dtype)
}

#: A flat `list[float]` read as `shape`, in element order.
#:
#: The count must match exactly, lanes included: a `COMPLEX64` array of n
#: elements takes 2n values, interleaved real and imaginary.
pub fn from_list(values: list[float], shape: list[int], dtype: int = FLOAT32) -> NDArray {
    return _upload(values, shape, dtype)
}

#: An array of `shape` filled from packed unsigned bytes, each multiplied by
#: `scale` on the way in.
#:
#: The path a dataset takes. `Buffer.upload_u8` expands the bytes on the
#: native side, so a million-pixel image costs one transfer and no boxed
#: Jaithon floats at all — building the equivalent `list[float]` first is the
#: several seconds this module is written to avoid.
#:
#: `scale` can make a fractional value out of a byte, and an integer dtype
#: does not round it back afterwards; `astype` is the pass that does.
pub fn from_bytes(
    data: bytes,
    shape: list[int],
    dtype: int = FLOAT32,
    source_offset: int = 0,
    scale: float = 1.0
) -> NDArray {
    let wanted = element_count(shape) * dtype_lanes(dtype)
    if source_offset < 0 or source_offset + wanted > data.len() {
        throw ValueError(f"{data.len()} bytes cannot fill {shape} from {source_offset}")
    }
    let out = NDArray(shape, dtype)
    if wanted > 0 { out.storage.upload_u8(data, source_offset, wanted, 0, scale) }
    return out
}

#: `from_bytes` into a 1-d array, as numpy's `frombuffer`.
#:
#: `count` is elements and `offset` is bytes, which for `COMPLEX64` are not
#: the same number. A null `count` takes everything left after `offset`.
pub fn frombuffer(
    data: bytes,
    count: int? = null,
    offset: int = 0,
    dtype: int = FLOAT32,
    scale: float = 1.0
) -> NDArray {
    let elements = count ?? (data.len() - offset) // dtype_lanes(dtype)
    return from_bytes(data, [max(0, elements)], dtype, offset, scale)
}

#: Coordinate arrays from a list of 1-d ones, as numpy's `meshgrid`.
#:
#: `"ij"` gives input i varying along axis i. `"xy"` — the default, and
#: Cartesian order — exchanges the first two axes, so x runs along the columns
#: and the results are shaped `[len(y), len(x), ...]`. `sparse` leaves every
#: axis but the input's own at length one, which broadcasts to the same thing
#: for a fraction of the memory.
#:
#: Every result owns its allocation: the grid is written by a strided copy
#: through a stride-0 view of the input, so the caller frees what it was
#: handed and the inputs are untouched.
pub fn meshgrid(
    arrays: list[NDArray],
    indexing: str = "xy",
    sparse: bool = false
) -> list[NDArray] {
    if indexing != "xy" and indexing != "ij" {
        throw ValueError(f"meshgrid indexing is \"xy\" or \"ij\", got {indexing}")
    }
    let rank = arrays.len()
    var extents: list[int] = []
    for a in arrays {
        if a.ndim() != 1 { throw ValueError(f"meshgrid takes 1-d arrays, got {a.ndim()} axes") }
        extents.push(a.shape[0])
    }
    var axis_of = [index for index in 0..rank]
    if indexing == "xy" and rank >= 2 {
        axis_of[0] = 1
        axis_of[1] = 0
        let held = extents[0]
        extents[0] = extents[1]
        extents[1] = held
    }
    var out: list[NDArray] = []
    for index in 0..rank {
        let axis = axis_of[index]
        let source = arrays[index]
        let shape = [if sparse and other != axis { 1 } else { extents[other] } for other in 0..rank]
        var strides = [0 for _other in 0..rank]
        strides[axis] = source.strides[0]
        let grid = NDArray(shape, source.dtype)
        copy_into(grid, source.view(shape, strides, source.offset))
        out.push(grid)
    }
    return out
}

#: The coordinate of every element of `shape`, as numpy's `indices`.
#:
#: The result is shaped `[rank] + shape`, and sheet d holds the index along
#: axis d. Each sheet is one ramp of `shape[d]` values read through a stride-0
#: view of the full shape, so nothing walks the whole array on the host.
pub fn indices(shape: list[int], dtype: int = INT32) -> NDArray {
    let rank = shape.len()
    var full = [rank]
    for extent in shape { full.push(extent) }
    let out = NDArray(full, dtype)
    let plane = element_count(shape)
    if plane == 0 or rank == 0 { return out }
    let lanes = out.lanes()
    let strides = contiguous_strides(shape, lanes)
    for axis in 0..rank {
        let ramp = _linear(shape[axis], 0.0, 1.0, dtype)
        var stretched = [0 for _other in shape]
        stretched[axis] = lanes
        copy_into(out.view(shape, strides, axis * plane * lanes), ramp.view(shape, stretched, 0))
        ramp.free()
    }
    return out
}

#: An array of `shape` whose every element is a Metal expression of its own
#: coordinates.
#:
#: `expr` is Metal source, not a Jaithon closure, and it is compiled into a
#: kernel that runs once per thread: `fromfunction("i0 * 2.0 + i1", [3, 4])` is
#: one dispatch where a host callback would be twelve interpreted calls — and,
#: at the sizes this package is for, several seconds. `i0` through
#: `i{rank-1}` are in scope as floats, slowest axis first, and everything the
#: Metal standard library offers is available around them.
#:
#: The generated source is interned by `Kernel.compile` on its own text rather
#: than through `kernel_of`, which keys on the entry name alone: every
#: expression here shares one entry name and they must not collide. Only the
#: rank is baked in — the extents arrive through the layout buffer — so a
#: repeated expression recompiles nothing whatever shape it is asked for.
pub fn fromfunction(expr: str, shape: list[int], dtype: int = FLOAT32) -> NDArray {
    if dtype_is_complex(dtype) {
        throw TypeError("fromfunction writes one float per slot; complex has two")
    }
    let out = NDArray(shape, dtype)
    let count = out.size()
    if count == 0 { return out }
    let meta = layout_buffer([out])
    defer { meta.free() }
    Kernel.compile(_function_source(expr, shape.len()), "nc_fromfunction").dispatch_async(
        [out.storage, meta],
        count,
        [count, _cast_mode(dtype), f32_bits(dtype_min(dtype)), f32_bits(dtype_max(dtype))]
    )
    return out
}

#: The conversion mode a destination dtype asks of a kernel that writes it.
fn _cast_mode(dtype: int) -> int {
    if dtype == BOOL { return CAST_BOOL }
    if dtype_is_integer(dtype) { return CAST_ROUND }
    return CAST_NONE
}

fn _or_dtype(dtype: int, a: NDArray) -> int { return if dtype < 0 { a.dtype } else { dtype } }

fn _require_count(num: int, what: str) -> void {
    if num < 0 { throw ValueError(f"{what} needs a non-negative count, got {num}") }
}

#: What `nd_convert` does, done on the host for the one value `fill` carries.
#:
#: `fill` writes the bits it is handed, so an integer dtype has to be honoured
#: before the dispatch instead of inside it. Same operations in the same order
#: as the kernel — round to nearest even, then clamp — so that
#: `full(shape, 2.5, INT32)` and `full(shape, 2.5).astype(INT32)` agree.
fn _cast_scalar(value: float, dtype: int) -> float {
    if dtype == BOOL { return if value != 0.0 { 1.0 } else { 0.0 } }
    if not dtype_is_integer(dtype) { return value }
    return math.clamp(_rint(value), dtype_min(dtype), dtype_max(dtype))
}

#: Round half to even. `math.round` is half away from zero, which disagrees
#: with the device on exactly the values a fill is most likely to carry.
fn _rint(value: float) -> float {
    let down = math.floor(value)
    let fraction = value - down
    if fraction > 0.5 { return down + 1.0 }
    if fraction < 0.5 { return down }
    return if math.floor(down * 0.5) * 2.0 == down { down } else { down + 1.0 }
}

fn _number(value: any) -> float {
    if value is bool { return if value { 1.0 } else { 0.0 } }
    if value is int { return float(value) }
    if value is float { return value }
    throw TypeError(f"an array element must be a number, got {type_of(value)}")
}

#: The dtype a bare Jaithon number contributes, on its type and not its value,
#: for the same reason `array.jai` reads a scalar operand that way.
fn _number_dtype(value: any) -> int {
    if value is bool { return BOOL }
    if value is int { return INT32 }
    if value is float { return FLOAT32 }
    throw TypeError(f"an array element must be a number, got {type_of(value)}")
}

#: A dense array of `shape` holding `values`, converted the way `dtype` says.
#:
#: An integer dtype stages through `FLOAT32` and casts, so the rounding is the
#: kernel's and the host never gets a second opinion about what an integer
#: dtype means. Every other dtype takes the values as written.
fn _upload(values: list[float], shape: list[int], dtype: int) -> NDArray {
    if not dtype_is_integer(dtype) {
        let out = NDArray(shape, dtype)
        out.upload(values)
        return out
    }
    let staged = NDArray(shape, FLOAT32)
    staged.upload(values)
    defer { staged.free() }
    return staged.astype(dtype)
}

#: The shape a nested list describes, taken from the first element at each
#: level. `_flatten` walks every element against it and is what catches a
#: ragged input.
fn _measure(value: any, shape: list[int]) -> void {
    if not value is list { return }
    let items = [item for item in value]
    shape.push(items.len())
    if items.len() == 0 { return }
    _measure(items[0], shape)
}

#: Every leaf in element order into `out`, and the dtype that holds them all.
#:
#: One walk and not two: the input is a host list, and walking it twice is the
#: one cost this function can actually avoid.
fn _flatten(value: any, shape: list[int], axis: int, out: list[float]) -> int {
    if axis == shape.len() {
        if value is list { throw ValueError("a nested list must be rectangular") }
        out.push(_number(value))
        return _number_dtype(value)
    }
    if not value is list { throw ValueError(f"a nested list must be rectangular") }
    let items = [item for item in value]
    if items.len() != shape[axis] {
        throw ValueError(f"a row of {items.len()} where axis {axis} is {shape[axis]}")
    }
    var found = BOOL
    for item in items { found = promote(found, _flatten(item, shape, axis + 1, out)) }
    return found
}

#: A ramp with no curve — `arange`, and the coordinate sheets of `indices`.
fn _linear(count: int, base: float, delta: float, dtype: int) -> NDArray {
    return _ramp(count, base, delta, CURVE_LINEAR, 0.0, 1.0, false, 0.0, dtype)
}

#: The one dispatch every generator in this module reaches.
fn _ramp(
    count: int,
    base: float,
    delta: float,
    curve: int,
    root: float,
    sign: float,
    pin: bool,
    last: float,
    dtype: int
) -> NDArray {
    if dtype_is_complex(dtype) {
        #: The ramp writes one float a slot. A complex ramp is that ramp on the
        #: real lane and zero on the imaginary one, which is exactly the
        #: widening copy `astype` already runs — one extra dispatch, on a path
        #: numpy itself calls unusual.
        let real = _ramp(count, base, delta, curve, root, sign, pin, last, FLOAT32)
        defer { real.free() }
        return real.astype(dtype)
    }
    let out = NDArray([count], dtype)
    if count == 0 { return out }
    let wide = count % 4 == 0
    let threads = if wide { count // 4 } else { count }
    kernel_of(SOURCE, "nc_ramp").dispatch_async([out.storage], threads, [threads, f32_bits(base), f32_bits(delta), curve, f32_bits(root), f32_bits(sign), f32_bits(last), if pin {
        1
    } else {
        0
    }, _cast_mode(dtype), f32_bits(dtype_min(dtype)), f32_bits(dtype_max(dtype)), if wide {
        1
    } else {
        0
    }])
    return out
}

#: `eye` and `tri`, which differ only by the comparison the kernel makes.
fn _band(rows: int, cols: int, k: int, gate: int, dtype: int, what: str) -> NDArray {
    if rows < 0 or cols < 0 {
        throw ValueError(f"{what} needs non-negative extents, got {rows}x{cols}")
    }
    if dtype_is_complex(dtype) {
        let real = _band(rows, cols, k, gate, FLOAT32, what)
        defer { real.free() }
        return real.astype(dtype)
    }
    let out = NDArray([rows, cols], dtype)
    let count = rows * cols
    if count == 0 { return out }
    let wide = cols % 4 == 0
    let threads = if wide { count // 4 } else { count }
    kernel_of(SOURCE, "nc_band_write").dispatch_async([out.storage], threads, [threads, if wide {
        cols // 4
    } else {
        cols
    }, _band_shift(k, rows, cols), rows, gate, if wide {
        1
    } else {
        0
    }])
    return out
}

#: `k` clamped into `[-rows, cols]` and biased by `rows`.
#:
#: The bias is because a dispatch binds its scalars as unsigned and `k` is
#: signed. The clamp is what makes the bias safe, and costs nothing: past
#: either corner the band misses the array entirely, and every `k` beyond one
#: of them gives the answer the corner already gives.
fn _band_shift(k: int, rows: int, cols: int) -> int {
    return max(-rows, min(k, cols)) + rows
}

#: `tril` and `triu`: the copy, then one pass that clears the other half.
fn _triangle(a: NDArray, k: int, gate: int, what: str) -> NDArray {
    if a.ndim() < 2 { throw ValueError(f"{what} needs two axes, got {a.ndim()}") }
    let out = a.copy()
    let count = out.size()
    if count == 0 { return out }
    let rows = a.shape[a.ndim() - 2]
    let cols = a.shape[a.ndim() - 1]
    kernel_of(SOURCE, "nc_band_zero").dispatch_async(
        [out.storage],
        count,
        [count, rows * cols, cols, _band_shift(k, rows, cols), rows, gate, out.lanes()]
    )
    return out
}

#: The kernel `fromfunction` compiles: the coordinate walk, then the caller's
#: expression.
#:
#: The walk goes fastest axis first so each level costs one division and one
#: subtraction, taking the extents from the layout buffer rather than baking
#: them in — a shape in the source would mean a fresh Metal compile per shape,
#: which is milliseconds against the microseconds of a dispatch.
fn _function_source(expr: str, rank: int) -> str {
    var walk = ""
    var axis = rank - 1
    while axis >= 0 {
        let slot = 1 + axis
        walk = walk + f"    uint e{axis} = uint(meta[{slot}]);\n"
        walk = walk + f"    uint q{axis} = rest / e{axis};\n"
        walk = walk + f"    float i{axis} = float(rest - q{axis} * e{axis});\n"
        walk = walk + f"    rest = q{axis};\n"
        axis -= 1
    }
    return PRELUDE + """

kernel void nc_fromfunction(device float* dst [[buffer(0)]],
                            device const float* meta [[buffer(1)]],
                            constant uint& count [[buffer(2)]],
                            constant uint& mode [[buffer(3)]],
                            constant uint& low_bits [[buffer(4)]],
                            constant uint& high_bits [[buffer(5)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint rest = id;
""" + walk + "    float value = " + expr + """;
    dst[id] = nd_convert(value, mode, as_type<float>(low_bits), as_type<float>(high_bits));
}
"""
}

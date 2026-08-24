#: Fancy indexing, gather and scatter — the selection numpy spells with an
#: index array rather than a slice, and the searches that produce one.
#:
#: `array.jai`'s `__getitem__` covers basic slicing and the two flat fancy
#: keys: a full-shape `BOOL` mask, and a 1-d integer array on a 1-d array.
#: Everything else is here. `take` and `put` name an axis, `take_along_axis`
#: and `put_along_axis` carry an index per output position, `choose` picks
#: between whole arrays, and `fancy_index` takes numpy's multi-axis key.
#:
#: The rule the whole module is built around: **no index array is ever
#: downloaded**. An index array lives on the device, and a host loop that
#: reads one back to drive a copy per element is the single most expensive
#: thing this file could do — a million-element gather would cost seconds
#: instead of microseconds. So bool-mask selection is `exclusive_scan` and a
#: scatter, both already in the prelude, and integer selection is one gather
#: dispatch with the keys resident.
#:
#: What does reach the host is a count: how many elements survived a mask, and
#: how long a `repeat_by_counts` result is. Neither can be allocated without
#: knowing it, and each costs one synchronise. `mode` of `"wrap"` or
#: `"clip"` skips the bounds check that `"raise"` owes numpy, which is the
#: other place a synchronise hides.
#:
#: ```
#: let rows = take(image, wanted, axis: 0)
#: let hits = nonzero(greater(image, 0.5))
#: let packed = mask_select(image, greater(image, 0.5))
#: put_along_axis(scores, best, prize, axis: 1)
#: ```
#:
#: One kernel shape does `take`, `take_along_axis`, `choose`, `compress` and
#: the multi-axis key: the source is viewed at the output's shape with a
#: stride of zero on the axis being indexed, the index array is viewed at the
#: same shape with a stride of zero on every axis it does not supply, and the
#: kernel adds `index * step` to the walk. They differ only in which axes get
#: the zeros, which is host-side bookkeeping over a handful of ints.
#:
#: Excluded on purpose: basic slicing and the flat fancy keys, which
#: `array.jai` owns and these functions call rather than reimplement;
#: `searchsorted`, which needs a sort and lives in `sorting`; and
#: `where(cond, x, y)`, which is a selection between values rather than
#: positions and lives in `ufunc`.
from std.gpu import Buffer

from .array import BOOL, INT32, ELLIPSIS, IndexToken, NDArray, PRELUDE, Slice
from .array import OP_ADD, OP_MUL, binary_arith, broadcast_shapes, broadcast_strides
from .array import contiguous_strides, copy_into, diagonal_view, dtype_is_bool
from .array import element_count, exclusive_scan, fill, kernel_of, layout_buffer
from .array import normalize_axis, promote, require_real, scatter_flat
from .ufunc import clip, greater, maximum, negative, remainder, subtract, where
from .reduce import amax, amin, cumsum

#: Out-of-range rule: raise, which numpy makes the default. Validated on the
#: host before the dispatch, since a kernel cannot throw.
pub const MODE_RAISE = 0
#: Out-of-range rule: fold the index back into the axis, so -1 is the last
#: element and `n` is the first.
pub const MODE_WRAP = 1
#: Out-of-range rule: clamp to the ends. numpy's `clip` deliberately does not
#: read a negative index as counting from the end, and neither does this.
pub const MODE_CLIP = 2

#: Internal rule: the index is already a signed slot offset and needs no
#: resolution. Only the multi-axis key uses it, having folded its index arrays
#: into one offset on the way in.
const MODE_OFFSET = 3

let SOURCE = PRELUDE + """
/* One index resolved against an axis of `limit` elements.
 *
 * `MODE_RAISE` is checked on the host before the dispatch -- a kernel cannot
 * raise -- so the clamp on that path is a guard that never fires rather than
 * a rule. `MODE_CLIP` cannot share the branch: numpy's clip does not read a
 * negative index as counting from the end, which is the whole difference
 * between it and the other two. */
inline int ix_resolve(int at, int limit, uint mode) {
    if (mode == 3u) return at;
    if (mode == 1u) {
        /* A C remainder lands in (-limit, limit), so one addition brings it
         * home. Two modulos would be the obvious spelling and twice the cost
         * on a device with no integer divide unit. */
        int folded = at % limit;
        return folded < 0 ? folded + limit : folded;
    }
    if (mode == 2u) return clamp(at, 0, limit - 1);
    if (at < 0) at += limit;
    return clamp(at, 0, limit - 1);
}

/* The signed step `_index_layout` appends after `operands` operand blocks.
 *
 * A step is a stride, a stride is signed here -- a reversed view is still a
 * view -- and `Kernel.dispatch` binds its trailing scalars as unsigned. So it
 * travels in the float block beside the other strides rather than beside the
 * element count. */
inline int ix_step(device const float* meta, uint operands) {
    int rank = int(meta[0]);
    return int(meta[1 + rank + int(operands) * (rank + 1)]);
}

/* Two operands' slots and the innermost coordinate, from one pass of divides.
 *
 * `nd_offset` re-walks the shape once per operand and these kernels want two
 * at a time, so sharing the walk halves the divides. The coordinate falls out
 * of it for free and is what the float4 path below tests. */
inline void ix_walk2(device const float* meta,
                     uint linear,
                     thread int* first,
                     thread int* second,
                     thread int* inner) {
    int rank = int(meta[0]);
    int a_base = 1 + rank;
    int b_base = a_base + rank + 1;
    int at = int(meta[a_base]);
    int bt = int(meta[b_base]);
    uint rest = linear;
    *inner = 0;
    for (int axis = rank - 1; axis >= 0; axis--) {
        uint extent = uint(meta[1 + axis]);
        uint next = rest / extent;
        /* One divide per level and a subtraction for the remainder. */
        int coord = int(rest - next * extent);
        rest = next;
        if (axis == rank - 1) *inner = coord;
        at += coord * int(meta[a_base + 1 + axis]);
        bt += coord * int(meta[b_base + 1 + axis]);
    }
    *first = at;
    *second = bt;
}

/* dst[k] = src[walk(k) + resolve(index[k]) * step].
 *
 * Operand 0 is the source viewed at the output's shape with a stride of zero
 * on the axis being indexed, operand 1 the index array viewed at the same
 * shape with a stride of zero on every axis it does not supply. `take`,
 * `take_along_axis`, `choose`, `compress` and the multi-axis key all reach
 * here; they differ only in which axes carry the zeros.
 *
 * The destination is always a fresh contiguous allocation, so its slot is the
 * linear index times the lane count and it needs no layout of its own. */
kernel void ix_gather_axis(device float* dst [[buffer(0)]],
                           device const float* src [[buffer(1)]],
                           device const float* index [[buffer(2)]],
                           device const float* meta [[buffer(3)]],
                           constant uint& count [[buffer(4)]],
                           constant uint& lanes [[buffer(5)]],
                           constant uint& limit [[buffer(6)]],
                           constant uint& mode [[buffer(7)]],
                           uint id [[thread_position_in_grid]]) {
    uint start = id * 4u;
    if (start >= count) return;
    uint span = min(4u, count - start);
    int rank = int(meta[0]);
    int step = ix_step(meta, 2u);
    int from = 0;
    int key = 0;
    int inner = 0;
    ix_walk2(meta, start, &from, &key, &inner);
    int at = from + ix_resolve(int(index[key]), int(limit), mode) * step;

    /* Four neighbouring outputs read four neighbouring sources exactly when
     * the innermost axis is one the source walks with unit stride and the
     * index does not walk at all -- which is every trailing axis a `take`
     * carries through, and where a gather spends nearly all of its bandwidth.
     * The scalar tail re-walks, which costs what the plain kernel would. */
    if (span == 4u && lanes == 1u && rank > 0 && (at & 3) == 0) {
        bool dense_run = int(meta[1 + 2 * rank]) == 1 && int(meta[2 + 3 * rank]) == 0;
        if (dense_run && inner + 4 <= int(meta[rank])) {
            *((device float4*)(dst + start)) = *((device const float4*)(src + at));
            return;
        }
    }
    for (uint k = 0u; k < span; k++) {
        int slot = at;
        if (k != 0u) {
            ix_walk2(meta, start + k, &from, &key, &inner);
            slot = from + ix_resolve(int(index[key]), int(limit), mode) * step;
        }
        uint to = (start + k) * lanes;
        for (uint lane = 0u; lane < lanes; lane++) dst[to + lane] = src[slot + int(lane)];
    }
}

/* dst[walk(k) + resolve(index[k]) * step] = src[k], the gather run backwards.
 *
 * Operand 0 is the array being written, operand 1 the index array and operand
 * 2 the values, all viewed at the iteration shape. Two indices naming one
 * slot race and which lands is undefined -- the promise `nd_scatter` already
 * makes, and one a thread grid cannot improve on.
 *
 * The values get `nd_offset` rather than a place in the shared walk: a
 * scatter is bound by its random writes, not by the third set of divides. */
kernel void ix_scatter_axis(device float* dst [[buffer(0)]],
                            device const float* index [[buffer(1)]],
                            device const float* src [[buffer(2)]],
                            device const float* meta [[buffer(3)]],
                            constant uint& count [[buffer(4)]],
                            constant uint& lanes [[buffer(5)]],
                            constant uint& limit [[buffer(6)]],
                            constant uint& mode [[buffer(7)]],
                            uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    int to = 0;
    int key = 0;
    int inner = 0;
    ix_walk2(meta, id, &to, &key, &inner);
    to += ix_resolve(int(index[key]), int(limit), mode) * ix_step(meta, 3u);
    int from = nd_offset(meta, 2u, id);
    for (uint lane = 0u; lane < lanes; lane++) dst[to + int(lane)] = src[from + int(lane)];
}

/* a.flat[resolve(index[k])] = values[k % supply], for an `a` that need not be
 * dense.
 *
 * The layout's iteration shape is the destination's own, so `nd_offset` turns
 * a flat position into a strided slot -- which is what lets `put` write
 * through a transpose or a slice without materialising it first. The grid
 * runs over the indices and not over `a`, so the walk is driven by the index
 * value rather than by the thread id. */
kernel void ix_put_flat(device float* dst [[buffer(0)]],
                        device const float* index [[buffer(1)]],
                        device const float* src [[buffer(2)]],
                        device const float* meta [[buffer(3)]],
                        constant uint& count [[buffer(4)]],
                        constant uint& lanes [[buffer(5)]],
                        constant uint& limit [[buffer(6)]],
                        constant uint& supply [[buffer(7)]],
                        constant uint& mode [[buffer(8)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    int at = ix_resolve(int(index[id]), int(limit), mode);
    int to = nd_offset(meta, 0u, uint(at));
    uint from = (id % supply) * lanes;
    for (uint lane = 0u; lane < lanes; lane++) dst[to + int(lane)] = src[from + lane];
}

/* Both masked writes numpy has, which differ only in what indexes the values.
 *
 * `putmask` takes them by flat position, so every thread already knows its
 * own; `place` takes them by rank among the survivors, which is what the
 * exclusive scan hands over. When `by_rank` is zero the rank buffer is bound
 * to the mask, which is read-only and already resident. */
kernel void ix_place(device float* dst [[buffer(0)]],
                     device const float* mask [[buffer(1)]],
                     device const float* rank_of [[buffer(2)]],
                     device const float* values [[buffer(3)]],
                     device const float* meta [[buffer(4)]],
                     constant uint& count [[buffer(5)]],
                     constant uint& lanes [[buffer(6)]],
                     constant uint& supply [[buffer(7)]],
                     constant uint& by_rank [[buffer(8)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    if (mask[id] == 0.0f) return;
    uint pick = by_rank != 0u ? uint(rank_of[id]) : id;
    uint from = (pick % supply) * lanes;
    int to = nd_offset(meta, 0u, id);
    for (uint lane = 0u; lane < lanes; lane++) dst[to + int(lane)] = values[from + lane];
}

/* One row of coordinates per non-zero, at the row the scan assigned it.
 *
 * The mask is dense in element order, so a thread's linear id is both where
 * it reads and the multi-index it decomposes into -- `nonzero`, `argwhere`
 * and `flatnonzero` are this kernel at three ranks. Most threads return at
 * the first branch, which is what a compaction looks like when the survivors
 * are sparse. */
kernel void ix_where_write(device float* dst [[buffer(0)]],
                           device const float* mask [[buffer(1)]],
                           device const float* offsets [[buffer(2)]],
                           device const float* meta [[buffer(3)]],
                           constant uint& count [[buffer(4)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    if (mask[id] == 0.0f) return;
    int rank = int(meta[0]);
    uint base = uint(offsets[id]) * uint(rank);
    uint rest = id;
    for (int axis = rank - 1; axis >= 0; axis--) {
        uint extent = uint(meta[1 + axis]);
        uint next = rest / extent;
        dst[base + uint(axis)] = float(rest - next * extent);
        rest = next;
    }
}
"""

#: `layout_buffer`'s block with the indexed axis's step appended.
#:
#: The step is a stride and a stride is signed, so it cannot ride along as a
#: trailing scalar the way the element count does. `ix_step` reads it back at
#: `1 + rank + operands * (rank + 1)`, which is why each kernel knows how many
#: operands it was given.
fn _index_layout(operands: list[NDArray], step: int) -> Buffer {
    let rank = operands[0].ndim()
    var values = [float(rank)]
    for extent in operands[0].shape { values.push(float(extent)) }
    for operand in operands {
        if operand.ndim() != rank {
            throw ValueError(f"a layout of rank {rank} got rank {operand.ndim()}")
        }
        values.push(float(operand.offset))
        for stride in operand.strides { values.push(float(stride)) }
    }
    values.push(float(step))
    let meta = Buffer(values.len())
    meta.upload(values)
    return meta
}

#: `a` as a dense array starting at slot zero, and whether that cost an
#: allocation the caller now owns.
fn _dense(a: NDArray) -> (NDArray, bool) {
    if a.offset == 0 and a.is_contiguous() { return (a, false) }
    return (a.copy(), true)
}

#: A 1-d view of every element of `a` in element order, plus the allocation
#: that had to be made for it, or null.
#:
#: The view is returned rather than the copy because a view does not own its
#: storage: freeing the view would leave the copy leaked, and freeing the copy
#: is what actually releases it.
fn _flattened(a: NDArray) -> (NDArray, NDArray?) {
    if a.offset == 0 and a.is_contiguous() {
        return (a.view([a.size()], [a.lanes()], 0), null)
    }
    let owned = a.copy()
    return (owned.view([owned.size()], [owned.lanes()], 0), owned)
}

#: `values` read as `dtype`, and whether that cost a copy.
#:
#: The write kernels move float slots and do not convert: an integer
#: destination has to round and saturate on the way in, and a complex one has
#: two slots per element where a real source has one. Doing it once here beats
#: a branch per thread.
fn _as_dtype(values: NDArray, dtype: int) -> (NDArray, bool) {
    if values.dtype == dtype { return (values, false) }
    return (values.astype(dtype), true)
}

#: A view of slot `at` along `axis`, that axis dropped.
fn _slot(a: NDArray, axis: int, at: int) -> NDArray {
    var shape: list[int] = []
    var strides: list[int] = []
    for other in 0..a.ndim() {
        if other == axis { continue }
        shape.push(a.shape[other])
        strides.push(a.strides[other])
    }
    return a.view(shape, strides, a.offset + at * a.strides[axis])
}

#: The rule a `mode` string names.
fn _mode_of(mode: str) -> int {
    if mode == "raise" { return MODE_RAISE }
    if mode == "wrap" { return MODE_WRAP }
    if mode == "clip" { return MODE_CLIP }
    throw ValueError(f"mode is 'raise', 'wrap' or 'clip', not '{mode}'")
}

#: An axis argument that must be an int.
fn _as_axis(axis: any, name: str = "axis") -> int {
    if axis is int { return axis }
    throw TypeError(f"{name} takes an int, not {type_of(axis)}")
}

#: Raise unless every index in `keys` addresses an axis of `limit` elements.
#:
#: One max reduction and one download, not a per-thread flag buffer nobody
#: reads on the happy path. `max(i, -i - 1)` is how far an index reaches past
#: either end — -limit lands on limit - 1 and -limit - 1 on limit — so a
#: single extreme decides both sides at once.
#:
#: The download is a synchronise, and it is the price of the `"raise"` mode
#: numpy defaults to. `"wrap"` and `"clip"` have no out-of-range case and skip
#: this entirely, which is the reason to reach for them in a loop.
fn _require_in_range(keys: NDArray, limit: int, what: str) -> void {
    require_real(keys, "an index array")
    if keys.size() == 0 { return }
    let mirrored = subtract(negative(keys), 1)
    defer { mirrored.free() }
    let reach = maximum(keys, mirrored)
    defer { reach.free() }
    let worst = amax(reach)
    defer { worst.free() }
    let seen = int(worst.item())
    if seen >= limit {
        throw IndexError(f"an index reaching {seen} is out of bounds for {what} of {limit}")
    }
}

#: The one gather every selection here dispatches.
#:
#: `source` and `keys` are already viewed at the output's shape, `step` is the
#: stride of the axis being indexed and `limit` its extent.
fn _gather(
    source: NDArray,
    keys: NDArray,
    step: int,
    limit: int,
    dtype: int,
    mode: int
) -> NDArray {
    let out = NDArray(source.shape, dtype)
    let count = out.size()
    if count == 0 { return out }
    let meta = _index_layout([source, keys], step)
    defer { meta.free() }
    kernel_of(SOURCE, "ix_gather_axis").dispatch_async(
        [out.storage, source.storage, keys.storage, meta],
        (count + 3) // 4,
        [count, out.lanes(), max(1, limit), mode]
    )
    return out
}

#: The one scatter, with `target` the destination already viewed at the
#: iteration shape and its indexed axis carrying a stride of zero.
fn _scatter(
    target: NDArray,
    keys: NDArray,
    values: NDArray,
    step: int,
    limit: int,
    mode: int
) -> void {
    let count = target.size()
    if count == 0 { return }
    let source = values.broadcast_to(target.shape)
    let meta = _index_layout([target, keys, source], step)
    defer { meta.free() }
    kernel_of(SOURCE, "ix_scatter_axis").dispatch_async(
        [target.storage, keys.storage, source.storage, meta],
        count,
        [count, target.lanes(), max(1, limit), mode]
    )
}

#: The views `take` needs: the index array's own axes spliced into `a`'s in
#: place of `axis`, with each side reading zero along the axes it does not
#: own.
fn _take_axis(a: NDArray, keys: NDArray, axis: int, mode: int) -> NDArray {
    let extent = a.shape[axis]
    if extent == 0 and keys.size() > 0 {
        throw IndexError(f"cannot take from axis {axis}, which has no elements")
    }
    var shape: list[int] = []
    var reading: list[int] = []
    var walking: list[int] = []
    for before in 0..axis {
        shape.push(a.shape[before])
        reading.push(a.strides[before])
        walking.push(0)
    }
    for own in 0..keys.ndim() {
        shape.push(keys.shape[own])
        reading.push(0)
        walking.push(keys.strides[own])
    }
    for after in axis + 1..a.ndim() {
        shape.push(a.shape[after])
        reading.push(a.strides[after])
        walking.push(0)
    }
    let source = a.view(shape, reading, a.offset)
    let index = keys.view(shape, walking, keys.offset)
    return _gather(source, index, a.strides[axis], extent, a.dtype, mode)
}

#: The views the `_along_axis` pair need: `a` and `indices` broadcast against
#: each other everywhere except `axis`, where the index array sets the extent
#: and `a` reads zero.
fn _along_axis(a: NDArray, indices: NDArray, axis: int, who: str) -> (NDArray, NDArray) {
    if indices.ndim() != a.ndim() {
        throw ValueError(f"{who} wants an index of {a.ndim()} dimensions, got {indices.ndim()}")
    }
    var flattened = [extent for extent in a.shape]
    var carried = [extent for extent in indices.shape]
    flattened[axis] = 1
    carried[axis] = 1
    var shape = broadcast_shapes(flattened, carried)
    shape[axis] = indices.shape[axis]
    let source = a.view(shape, broadcast_strides(flattened, a.strides, shape), a.offset)
    let keys = indices.view(
        shape,
        broadcast_strides(indices.shape, indices.strides, shape),
        indices.offset
    )
    return (source, keys)
}

#: `a`'s elements at `indices` along `axis`, or along the flattening when
#: `axis` is null.
#:
#: The index array keeps its own shape and is spliced into `a`'s in place of
#: `axis`, so a 2-d key on a 3-d array gives a 4-d result — numpy's rule. A
#: negative index counts from the end under `"raise"` and `"wrap"`, and does
#: not under `"clip"`, which is numpy's inconsistency and not one to fix here.
pub fn take(a: NDArray, indices: NDArray, axis: any = null, mode: str = "raise") -> NDArray {
    require_real(indices, "take")
    let rule = _mode_of(mode)
    if axis is null {
        let (flat, owner) = _flattened(a)
        defer { if owner is not null { owner.free() } }
        if rule == MODE_RAISE { _require_in_range(indices, flat.shape[0], "an array") }
        return _take_axis(flat, indices, 0, rule)
    }
    let at = normalize_axis(_as_axis(axis), a.ndim())
    if rule == MODE_RAISE { _require_in_range(indices, a.shape[at], f"axis {at}") }
    return _take_axis(a, indices, at, rule)
}

#: One element of `a` per position of `indices`, chosen along `axis`.
#:
#: `indices` matches `a`'s rank and broadcasts against it on every other axis,
#: so the shape of the result is the shape of the index array. This is the
#: partner of `argmin`, `argmax` and `argsort`: their output is exactly the
#: index this consumes.
pub fn take_along_axis(a: NDArray, indices: NDArray, axis: int, mode: str = "raise") -> NDArray {
    require_real(indices, "take_along_axis")
    let at = normalize_axis(axis, a.ndim())
    let rule = _mode_of(mode)
    if rule == MODE_RAISE { _require_in_range(indices, a.shape[at], f"axis {at}") }
    let (source, keys) = _along_axis(a, indices, at, "take_along_axis")
    return _gather(source, keys, a.strides[at], a.shape[at], a.dtype, rule)
}

#: Write `values` into `a` at the flat positions `indices` names.
#:
#: In place, and `a` need not be contiguous: the kernel walks `a`'s own shape
#: to turn a flat position into a slot, so a transpose or a slice is written
#: through rather than materialised. `values` shorter than `indices` repeats,
#: as numpy's does. Two indices naming one slot race.
pub fn put(a: NDArray, indices: NDArray, values: NDArray, mode: str = "raise") -> void {
    require_real(indices, "put")
    let rule = _mode_of(mode)
    let limit = a.size()
    let count = indices.size()
    if count == 0 { return }
    if limit == 0 { throw IndexError("cannot put into an array with no elements") }
    if rule == MODE_RAISE { _require_in_range(indices, limit, "an array") }
    let (keys, owns_keys) = _dense(indices)
    defer { if owns_keys { keys.free() } }
    let (cast, owns_cast) = _as_dtype(values, a.dtype)
    defer { if owns_cast { cast.free() } }
    let (supply, owns_supply) = _dense(cast)
    defer { if owns_supply { supply.free() } }
    if supply.size() == 0 { throw ValueError("put() needs at least one value") }
    let meta = layout_buffer([a])
    defer { meta.free() }
    kernel_of(SOURCE, "ix_put_flat").dispatch_async(
        [a.storage, keys.storage, supply.storage, meta],
        count,
        [count, a.lanes(), limit, supply.size(), rule]
    )
}

#: Write `values` into `a` along `axis` at the positions `indices` names, the
#: inverse of `take_along_axis`.
pub fn put_along_axis(
    a: NDArray,
    indices: NDArray,
    values: NDArray,
    axis: int,
    mode: str = "raise"
) -> void {
    require_real(indices, "put_along_axis")
    let at = normalize_axis(axis, a.ndim())
    let rule = _mode_of(mode)
    if rule == MODE_RAISE { _require_in_range(indices, a.shape[at], f"axis {at}") }
    let (target, keys) = _along_axis(a, indices, at, "put_along_axis")
    let (cast, owns_cast) = _as_dtype(values, a.dtype)
    defer { if owns_cast { cast.free() } }
    _scatter(target, keys, cast, a.strides[at], a.shape[at], rule)
}

#: The two masked writes, which differ only in what indexes the values.
fn _masked_write(a: NDArray, mask: NDArray, values: NDArray, by_rank: bool, who: str) -> void {
    require_real(mask, who)
    let count = a.size()
    if count == 0 { return }
    if mask.size() != count {
        throw ValueError(f"{who}() wants a mask of {count} elements, got {mask.size()}")
    }
    let (gate, owns_gate) = _dense(mask)
    defer { if owns_gate { gate.free() } }
    let (cast, owns_cast) = _as_dtype(values, a.dtype)
    defer { if owns_cast { cast.free() } }
    let (supply, owns_supply) = _dense(cast)
    defer { if owns_supply { supply.free() } }
    if supply.size() == 0 { throw ValueError(f"{who}() needs at least one value") }
    let (ranks, owns_ranks) = _ranks_of(gate, by_rank)
    defer { if owns_ranks { ranks.free() } }
    let meta = layout_buffer([a])
    defer { meta.free() }
    kernel_of(SOURCE, "ix_place").dispatch_async([a.storage, gate.storage, ranks.storage, supply.storage, meta], count, [count, a.lanes(), supply.size(), if by_rank {
        1
    } else {
        0
    }])
}

#: Each element's rank among the survivors, or the mask itself when nobody
#: asked for one — an unused buffer still has to be bound, and the mask is the
#: read-only allocation already resident.
#:
#: `exclusive_scan` also hands back a total, and reading it is a synchronise
#: `place` does not need. That cost is most of why numpy has `putmask` too.
fn _ranks_of(gate: NDArray, by_rank: bool) -> (NDArray, bool) {
    if not by_rank { return (gate, false) }
    let (offsets, _total) = exclusive_scan(gate)
    return (offsets, true)
}

#: Write `values` into the positions of `a` where `mask` is non-zero, taking
#: them in order and cycling when they run out.
#:
#: The n-th survivor takes the n-th value, which is what needs the scan.
#: `putmask` is the cheaper one that indexes by flat position instead.
pub fn place(a: NDArray, mask: NDArray, values: NDArray) -> void {
    _masked_write(a, mask, values, true, "place")
}

#: Set `a.flat[n]` to `values[n % values.size()]` wherever `mask.flat[n]` is
#: non-zero.
#:
#: The value follows the position rather than the rank among the survivors,
#: which is the whole difference from `place` and the reason this needs no
#: scan and no synchronise.
pub fn putmask(a: NDArray, mask: NDArray, values: NDArray) -> void {
    _masked_write(a, mask, values, false, "putmask")
}

#: `a`'s elements where `mask` is non-zero, packed into a 1-d array.
#:
#: `exclusive_scan` gives every element the position it would land in, and one
#: scatter under the same mask puts the survivors there. The scan's total is
#: the result's length, and it is the one value this has to read back.
pub fn mask_select(a: NDArray, mask: NDArray) -> NDArray {
    require_real(mask, "mask_select")
    if broadcast_shapes(a.shape, mask.shape) != a.shape {
        throw IndexError(f"a mask of {mask.shape} does not cover an array of {a.shape}")
    }
    let (gate, owns_gate) = _dense(mask.broadcast_to(a.shape))
    defer { if owns_gate { gate.free() } }
    let (dense, owns_dense) = _dense(a)
    defer { if owns_dense { dense.free() } }
    let (offsets, total) = exclusive_scan(gate)
    defer { offsets.free() }
    let out = NDArray([total], a.dtype)
    if total == 0 { return out }
    scatter_flat(out, offsets, dense, gate)
    return out
}

#: Where `mask` is non-zero, as one row of coordinates per survivor.
#:
#: `flat` asks for the rank-1 reading, `[total]` rather than `[total, 1]` —
#: the same block of memory named two ways rather than a second pass over it.
fn _positions(mask: NDArray, flat: bool) -> NDArray {
    require_real(mask, "nonzero")
    let (dense, owned) = _dense(mask)
    defer { if owned { dense.free() } }
    let rank = dense.ndim()
    let (offsets, total) = exclusive_scan(dense)
    defer { offsets.free() }
    let out = NDArray(if flat { [total] } else { [total, rank] }, INT32)
    let count = dense.size()
    if count == 0 or total == 0 { return out }
    let meta = layout_buffer([dense])
    defer { meta.free() }
    kernel_of(SOURCE, "ix_where_write").dispatch_async(
        [out.storage, dense.storage, offsets.storage, meta],
        count,
        [count]
    )
    return out
}

#: The flat positions of the non-zeros, in element order.
pub fn flatnonzero(a: NDArray) -> NDArray {
    let (flat, owner) = _flattened(a)
    defer { if owner is not null { owner.free() } }
    return _positions(flat, true)
}

#: The coordinates of the non-zeros, one row per survivor and one column per
#: axis — numpy's `argwhere`, and `transpose(nonzero(a))`.
pub fn argwhere(a: NDArray) -> NDArray {
    if a.ndim() == 0 { throw ValueError("argwhere() has no reading for a 0-d array") }
    return _positions(a, false)
}

#: One index array per axis, together naming every non-zero — the tuple numpy
#: returns and the key its own fancy indexing consumes.
#:
#: Each array is its own allocation rather than a strided column of one, so
#: freeing them frees the memory. The columns come from a single `argwhere`
#: pass and one strided copy each, `ndim` copies of a short array.
pub fn nonzero(a: NDArray) -> list[NDArray] {
    let rank = a.ndim()
    if rank == 0 { throw ValueError("nonzero() has no reading for a 0-d array") }
    let found = argwhere(a)
    defer { found.free() }
    let total = found.shape[0]
    var out: list[NDArray] = []
    for axis in 0..rank {
        let column = NDArray([total], INT32)
        if total > 0 { copy_into(column, found.view([total], [rank], axis)) }
        out.push(column)
    }
    return out
}

#: `a`'s elements where `condition` is non-zero, both read flat — numpy's
#: `extract`, which is `a[condition]` for a mask of the same size.
pub fn extract(condition: NDArray, a: NDArray) -> NDArray {
    if condition.size() != a.size() {
        throw ValueError(
            f"extract() wants a condition of {a.size()} elements, got {condition.size()}"
        )
    }
    let (flat, owner) = _flattened(a)
    defer { if owner is not null { owner.free() } }
    let (gate, gate_owner) = _flattened(condition)
    defer { if gate_owner is not null { gate_owner.free() } }
    return mask_select(flat, gate)
}

#: The slices of `a` along `axis` that `condition` selects, or the flat
#: elements when `axis` is null.
#:
#: A shorter condition selects from the front and leaves the rest out, as
#: numpy's does. Two dispatches and one synchronise: the condition compacts
#: into a list of positions, and those drive the same gather `take` uses.
pub fn compress(condition: NDArray, a: NDArray, axis: any = null) -> NDArray {
    require_real(condition, "compress")
    if condition.ndim() != 1 {
        throw ValueError(f"compress() takes a 1-d condition, got {condition.shape}")
    }
    if axis is null {
        let (flat, owner) = _flattened(a)
        defer { if owner is not null { owner.free() } }
        return _compress_axis(flat, condition, 0)
    }
    return _compress_axis(a, condition, normalize_axis(_as_axis(axis), a.ndim()))
}

fn _compress_axis(a: NDArray, condition: NDArray, axis: int) -> NDArray {
    let extent = a.shape[axis]
    if condition.size() > extent {
        throw ValueError(f"a condition of {condition.size()} on an axis of {extent}")
    }
    let keys = flatnonzero(condition)
    defer { keys.free() }
    return _take_axis(a, keys, axis, MODE_CLIP)
}

#: `choices[a[k]][k]` at every position, with `a` and the choices broadcast
#: together first.
#:
#: The choices are stacked into one array so the gather has a single source to
#: stride through: a kernel cannot bind a list of buffers whose length is a
#: runtime value. That stack is `len(choices)` copies of already-broadcast
#: data, which is what numpy pays here too.
pub fn choose(a: NDArray, choices: list[NDArray], mode: str = "raise") -> NDArray {
    require_real(a, "choose")
    if choices.len() == 0 { throw ValueError("choose() needs at least one choice") }
    let rule = _mode_of(mode)
    var shape = [extent for extent in a.shape]
    var dtype = choices[0].dtype
    for choice in choices {
        shape = broadcast_shapes(shape, choice.shape)
        dtype = promote(dtype, choice.dtype)
    }
    if rule == MODE_RAISE { _require_in_range(a, choices.len(), "a choice") }
    var stacked_shape = [choices.len()]
    for extent in shape { stacked_shape.push(extent) }
    let stacked = NDArray(stacked_shape, dtype)
    defer { stacked.free() }
    for index in 0..choices.len() {
        copy_into(_slot(stacked, 0, index), choices[index].broadcast_to(shape))
    }
    var reading: list[int] = []
    for axis in 1..stacked.ndim() { reading.push(stacked.strides[axis]) }
    let source = stacked.view(shape, reading, stacked.offset)
    let keys = a.broadcast_to(shape)
    return _gather(source, keys, stacked.strides[0], choices.len(), dtype, rule)
}

#: The first choice whose condition holds, elementwise, and `otherwise` where
#: none does.
#:
#: Folded from the last condition to the first so the earliest true one is the
#: one still standing — one `where` dispatch per pair, and no host branch over
#: a condition it would have to download to read.
pub fn select(
    condlist: list[NDArray],
    choicelist: list[NDArray],
    otherwise: float = 0.0
) -> NDArray {
    if condlist.len() != choicelist.len() {
        throw ValueError(f"select() got {condlist.len()} conditions and {choicelist.len()} choices")
    }
    if condlist.len() == 0 { throw ValueError("select() needs at least one condition") }
    var shape: list[int] = []
    var dtype = choicelist[0].dtype
    for condition in condlist { shape = broadcast_shapes(shape, condition.shape) }
    for choice in choicelist {
        shape = broadcast_shapes(shape, choice.shape)
        dtype = promote(dtype, choice.dtype)
    }
    var out = NDArray(shape, dtype)
    fill(out, otherwise)
    var index = condlist.len() - 1
    while index >= 0 {
        let folded = where(condlist[index], choicelist[index], out)
        out.free()
        out = folded
        index -= 1
    }
    return out
}

#: `a` with element `k` of `axis` repeated `counts[k]` times — numpy's
#: `repeat` with a per-element count rather than a single one.
#:
#: The index array is built on the device out of two running totals and a
#: scatter, never a host loop over the counts: the start of each run is the
#: exclusive sum of the counts, a mark at each start turns into a running
#: segment number, and one less than that is the source position for every
#: output element. Four dispatches, then the same gather `take` uses.
pub fn repeat_by_counts(a: NDArray, counts: NDArray, axis: any = null) -> NDArray {
    require_real(counts, "repeat_by_counts")
    if axis is null {
        let (flat, owner) = _flattened(a)
        defer { if owner is not null { owner.free() } }
        return _repeat_axis(flat, counts, 0)
    }
    return _repeat_axis(a, counts, normalize_axis(_as_axis(axis), a.ndim()))
}

fn _repeat_axis(a: NDArray, counts: NDArray, axis: int) -> NDArray {
    let extent = a.shape[axis]
    if counts.ndim() != 1 or counts.size() != extent {
        throw ValueError(
            f"repeat_by_counts wants one count per element of axis {axis} ({extent})," +
                f" got {counts.shape}"
        )
    }
    let ends = cumsum(counts, null, INT32)
    defer { ends.free() }
    let smallest = amin(counts)
    defer { smallest.free() }
    let (total, lowest) = _probe(ends, smallest, extent)
    if lowest < 0.0 { throw ValueError("a repeat count cannot be negative") }

    var shape = [size for size in a.shape]
    shape[axis] = total
    if total == 0 { return NDArray(shape, a.dtype) }

    let starts = subtract(ends, counts)
    defer { starts.free() }
    let occupied = greater(counts, 0)
    defer { occupied.free() }
    let marks = NDArray([total], INT32)
    defer { marks.free() }
    fill(marks, 0.0)
    let one = NDArray([], INT32)
    defer { one.free() }
    fill(one, 1.0)
    scatter_flat(marks, starts, one, occupied)
    let segments = cumsum(marks, null, INT32)
    defer { segments.free() }
    let keys = subtract(segments, 1)
    defer { keys.free() }
    return _take_axis(a, keys, axis, MODE_CLIP)
}

#: The output length and the smallest count, read back together.
#:
#: One synchronise and not two. Both are single values the host needs before
#: it can allocate, and a download waits on the batch that wrote its buffer —
#: so staging them side by side costs two strided copies and saves a stall.
fn _probe(ends: NDArray, smallest: NDArray, extent: int) -> (int, float) {
    if extent == 0 { return (0, 0.0) }
    let staged = NDArray([2], INT32)
    defer { staged.free() }
    copy_into(
        staged.view([1], [1], 0),
        ends.view([], [], ends.offset + (extent - 1) * ends.strides[0])
    )
    copy_into(staged.view([1], [1], 1), smallest)
    let readings = staged.to_list()
    return (int(readings[0]), readings[1])
}

#: Set the diagonal of `a` to `value`, in place.
#:
#: A diagonal is a view — an axis whose stride is the sum of the ones it
#: replaces — so this is one fill and no gather at all. `wrap` continues the
#: diagonal past the bottom of a tall 2-d array the way numpy's does, which
#: needs the elements laid out flat and therefore a contiguous array; numpy
#: ignores `wrap` above two dimensions, and so does this.
pub fn fill_diagonal(a: NDArray, value: float, wrap: bool = false) -> void {
    let rank = a.ndim()
    if rank < 2 {
        throw ValueError(f"fill_diagonal() needs at least 2 dimensions, got {rank}")
    }
    if rank == 2 and wrap {
        if not a.is_contiguous() {
            throw ValueError("a wrapped diagonal needs a contiguous array")
        }
        let step = a.shape[1] + 1
        let count = (a.size() + step - 1) // step
        fill(a.view([count], [step * a.lanes()], a.offset), value)
        return
    }
    if rank == 2 {
        fill(diagonal_view(a), value)
        return
    }
    var step = 0
    for axis in 0..rank {
        if a.shape[axis] != a.shape[0] {
            throw ValueError(f"a diagonal of {rank} dimensions needs equal axes, got {a.shape}")
        }
        step += a.strides[axis]
    }
    fill(a.view([a.shape[0]], [step], a.offset), value)
}

#: A key checked once and handed back in the list form `fancy_index` takes.
#:
#: numpy's `np.index_exp[...]` exists because slice syntax cannot be passed
#: around as a value. Here the problem is the other way up — a key is already
#: an ordinary tuple — so this normalises a bare index into a one-entry key
#: and rejects the entries that are not indices at all, which is what a caller
#: assembling a key in a loop actually needs.
pub fn index_exp(key: any) -> list[any] {
    let items = _key_items(key)
    for item in items {
        if item is NDArray or item is int or item is Slice { continue }
        if item is range or item is IndexToken { continue }
        throw TypeError(f"an index may not be a {type_of(item)}")
    }
    return items
}

#: The open mesh numpy's `ix_` builds: n one-dimensional arrays reshaped so
#: that axis k of the selection comes from vector k, and the key they form
#: picks out their cross product.
#:
#: Pure view work — each vector keeps its storage and gains length-one axes —
#: so nothing is copied until `fancy_index` broadcasts them in the kernel. A
#: `BOOL` vector becomes its own non-zero positions first, as numpy's does.
pub fn ix_(vectors: list[NDArray]) -> list[NDArray] {
    let rank = vectors.len()
    var out: list[NDArray] = []
    for index in 0..rank {
        var vector = vectors[index]
        if dtype_is_bool(vector.dtype) { vector = flatnonzero(vector) }
        if vector.ndim() != 1 { throw ValueError(f"ix_ takes 1-d arrays, got {vector.shape}") }
        var shape = [1 for _axis in 0..rank]
        var strides = [0 for _axis in 0..rank]
        shape[index] = vector.shape[0]
        strides[index] = vector.strides[0]
        out.push(vector.view(shape, strides, vector.offset))
    }
    return out
}

fn _key_items(key: any) -> list[any] {
    if key is tuple { return [item for item in key] }
    if key is list { return [item for item in key] }
    return [key]
}

#: A key with `ELLIPSIS` expanded and the trailing axes filled in, so it has
#: exactly one entry per axis of `a`.
fn _expand_key(a: NDArray, items: list[any]) -> list[any] {
    var consumed = 0
    var ellipses = 0
    for item in items {
        if item is IndexToken {
            if item.name != ELLIPSIS.name {
                throw ValueError("fancy_index has no reading for NEWAXIS; add the axis after")
            }
            ellipses += 1
        } else {
            consumed += 1
        }
    }
    if ellipses > 1 { throw ValueError("a key may hold one ELLIPSIS at most") }
    if consumed > a.ndim() {
        throw IndexError(f"a key of {consumed} axes on an array of {a.ndim()}")
    }
    var out: list[any] = []
    for item in items {
        if item is IndexToken {
            for _axis in 0..a.ndim() - consumed { out.push(Slice()) }
        } else {
            out.push(item)
        }
    }
    while out.len() < a.ndim() { out.push(Slice()) }
    return out
}

#: Whether a basic entry takes its whole axis, which is the only basic entry
#: this key form accepts.
fn _covers(item: any, extent: int) -> bool {
    if item is Slice {
        let (first, count) = item.resolve(extent)
        return item.step == 1 and first == 0 and count == extent
    }
    if item is range { return item.start() == 0 and item.step() == 1 and item.len() == extent }
    return false
}

#: An index array folded into the axis it indexes, ready to be summed with the
#: others into one slot offset.
#:
#: `remainder` does the negative-from-the-end fold and the `"wrap"` mode in a
#: single dispatch — numpy's remainder takes the sign of the divisor, which is
#: exactly the resolution both modes want — so a validated `"raise"` key costs
#: the same as a wrapped one from here on.
fn _folded(keys: NDArray, extent: int, stride: int, mode: int) -> NDArray {
    let resolved = if mode == MODE_CLIP {
        clip(keys, 0, extent - 1)
    } else {
        remainder(keys, extent)
    }
    defer { resolved.free() }
    return binary_arith(resolved, stride, OP_MUL)
}

#: Whether the entries an index array or a plain int occupies form one run.
#:
#: numpy decides where the broadcast shape lands in the result on exactly this
#: question, and counts a plain int as one of the run even though it
#: contributes no axis. Adjacent puts the shape where the run was; separated
#: puts it first.
fn _adjacent(positions: list[int]) -> bool {
    if positions.len() == 0 { return true }
    let span = positions[positions.len() - 1] - positions[0]
    return span == positions.len() - 1
}

#: numpy's multi-axis integer key: one index array per axis, broadcast against
#: each other, with the axes nobody indexed carried through.
#:
#: A key entry is an integer `NDArray`, a plain int, a full slice or `range`,
#: or `ELLIPSIS`; fewer entries than axes leaves the trailing ones whole. The
#: index arrays broadcast into a shape `B`. Where the entries they occupy are
#: adjacent — counting a plain int as one of them, as numpy does — `B` lands
#: in the result at that position; where a carried axis separates them, numpy
#: puts `B` first instead, and so does this. `a[i, j]`, `a[:, i]`, `a[0, i]`
#: and `a[i, :, :]` are all the first case.
#:
#: Refused, because a subtly wrong answer is worse than none: a partial or
#: stepped slice, and `NEWAXIS`. Basic slicing is free — it is a view — so
#: `fancy_index(a[2..6], [rows, columns])` is the spelling, and it copies
#: nothing the general form would not have. A `BOOL` array is refused too:
#: `mask_select` and `nonzero` are what a mask means, and reading one as
#: integers would quietly select rows 0 and 1.
#:
#: One dispatch whatever the rank of the key. The index arrays are folded into
#: a single signed slot offset per position first — a remainder and a multiply
#: each, over `B` and not over the output — because a kernel cannot bind one
#: buffer per axis when the axis count is a runtime value.
pub fn fancy_index(a: NDArray, key: any, mode: str = "raise") -> NDArray {
    let rule = _mode_of(mode)
    let items = _expand_key(a, _key_items(key))
    var advanced: list[int] = []
    var kept: list[int] = []
    var arrays: list[NDArray] = []
    var axes: list[int] = []
    var base = a.offset
    for position in 0..items.len() {
        let item = items[position]
        let extent = a.shape[position]
        if item is NDArray {
            if dtype_is_bool(item.dtype) {
                throw ValueError(
                    f"a BOOL key on axis {position} is a mask; use mask_select or nonzero"
                )
            }
            require_real(item, "fancy_index")
            if rule == MODE_RAISE { _require_in_range(item, extent, f"axis {position}") }
            advanced.push(position)
            arrays.push(item)
            axes.push(position)
        } elif item is int {
            advanced.push(position)
            let at = if item < 0 { item + extent } else { item }
            if at < 0 or at >= extent {
                throw IndexError(f"index {item} is out of bounds for an axis of {extent}")
            }
            base += at * a.strides[position]
        } elif _covers(item, extent) {
            kept.push(position)
        } else {
            throw ValueError(
                f"a partial slice on axis {position} cannot mix with an index array;" +
                    " slice first, which is a view, then index the result"
            )
        }
    }

    var mesh: list[int] = []
    for array in arrays { mesh = broadcast_shapes(mesh, array.shape) }
    let adjacent = _adjacent(advanced)
    var before: list[int] = []
    var after: list[int] = []
    for position in kept {
        if adjacent and advanced.len() > 0 and position < advanced[0] {
            before.push(position)
        } else {
            after.push(position)
        }
    }

    var shape: list[int] = []
    var reading: list[int] = []
    var walking: list[int] = []
    for position in before {
        shape.push(a.shape[position])
        reading.push(a.strides[position])
        walking.push(0)
    }
    let mesh_strides = contiguous_strides(mesh, 1)
    for axis in 0..mesh.len() {
        shape.push(mesh[axis])
        reading.push(0)
        walking.push(mesh_strides[axis])
    }
    for position in after {
        shape.push(a.shape[position])
        reading.push(a.strides[position])
        walking.push(0)
    }

    let source = a.view(shape, reading, base)
    if arrays.len() == 0 { return source }
    let offsets = _combined(arrays, axes, a, mesh, rule)
    defer { offsets.free() }
    return _gather(source, offsets.view(shape, walking, 0), 1, 1, a.dtype, MODE_OFFSET)
}

#: The index arrays summed into one signed slot offset per position of `mesh`.
fn _combined(
    arrays: list[NDArray],
    axes: list[int],
    a: NDArray,
    mesh: list[int],
    mode: int
) -> NDArray {
    var total = _folded(arrays[0].broadcast_to(mesh), a.shape[axes[0]], a.strides[axes[0]], mode)
    for slot in 1..arrays.len() {
        let axis = axes[slot]
        let term = _folded(arrays[slot].broadcast_to(mesh), a.shape[axis], a.strides[axis], mode)
        defer { term.free() }
        let summed = binary_arith(total, term, OP_ADD)
        total.free()
        total = summed
    }
    if total.shape == mesh and total.offset == 0 and total.is_contiguous() { return total }
    let packed = total.copy()
    total.free()
    return packed
}

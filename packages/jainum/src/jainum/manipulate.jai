#: Shape manipulation, joining and splitting — numpy's `reshape`, `transpose`,
#: `concatenate`, `split`, `flip`, `tile`, `roll` and `pad`.
#:
#: Almost everything here is free. A transpose, a flip, a diagonal, a squeeze
#: and most reshapes only rewrite shape, strides and offset, so they cost one
#: small allocation on the host and touch no device memory at all. That is why
#: `array.jai` counts strides in signed float slots: a flip is a negated
#: stride and a walk from the far end, not a pass over the data.
#:
#: The ones that cannot be a view — joining, tiling, padding, rolling — go
#: through `copy_into`, which is `array.jai`'s strided copy kernel. One
#: dispatch writes one whole part into a slice of the destination; nothing here
#: builds a host-side list of values and uploads it, because a host loop over a
#: million elements costs seconds in this language.
#:
#: `tile` and `repeat` are worth reading for how little they need. A repeat
#: along an axis is that axis split in two, the new half stretched by a
#: stride-0 broadcast, and one strided copy into the result; a tile is the same
#: trick with the halves the other way round. Neither compiles a kernel.
#:
#: The one kernel this module owns is `nm_pad_map`, for the padding modes whose
#: border is a function of the source rather than a constant. Constant padding
#: does not need it: a fill and one copy of the interior are two dispatches of
#: kernels `array.jai` already has.
#:
#: ```
#: let a = reshape(arange_like, [3, 4])
#: let t = transpose(a)                    #: a view, no copy
#: let joined = concatenate([a, a], 0)     #: two dispatches
#: let bordered = pad(a, [[1, 1], [2, 2]], "reflect")
#: ```
from std.gpu import Buffer

from .array import NDArray, PRELUDE, broadcast_shapes, contiguous_strides, copy_into
from .array import diagonal_view, element_count, fill, kernel_of, normalize_axes
from .array import normalize_axis, promote

#: The padding rules whose border reads from the array: clamp to the edge.
pub const PAD_EDGE = 0
#: Padding rule: tile the axis, so the left border continues from the right end.
pub const PAD_WRAP = 1
#: Padding rule: fold about the first and last element without repeating them,
#: which is what numpy calls `reflect`.
pub const PAD_REFLECT = 2

let SOURCE = PRELUDE + """
/* Pad by mapping every destination position back onto a source one.
 *
 * The metadata block is this module's own rather than `layout_buffer`'s: the
 * walk needs each axis's coordinate before it can apply the border rule, and
 * `nd_offset` only hands back a finished slot.
 *
 *   [0]                  rank
 *   [1 .. rank]          destination shape
 *   [1 + rank ..]        source shape
 *   [1 + 2*rank ..]      pad before, per axis
 *   [1 + 3*rank]         source element offset
 *   [2 + 3*rank ..]      source strides, signed
 *
 * The destination is a fresh contiguous allocation, so its slot is the linear
 * index times the lane count and it carries no strides here. */
inline int nm_pad_index(int pos, int limit, uint mode) {
    if (mode == 0u) return clamp(pos, 0, limit - 1);
    if (mode == 1u) {
        /* One modulo and a single correction: a C remainder lands in
         * (-limit, limit), so at most one addition brings it home. Two
         * modulos would be the obvious spelling and twice the cost. */
        int m = pos % limit;
        return m < 0 ? m + limit : m;
    }
    /* A reflection that does not repeat the edge has period 2*(limit-1), and
     * the far half of that period folds back. A single column has no period
     * at all and is its own reflection. */
    if (limit == 1) return 0;
    int period = 2 * (limit - 1);
    int m = pos % period;
    if (m < 0) m += period;
    return m <= limit - 1 ? m : period - m;
}

inline int nm_pad_source(
    device const float* meta,
    int rank,
    uint mode,
    uint linear,
    thread int* inner
) {
    int at = int(meta[1 + 3 * rank]);
    uint rest = linear;
    for (int axis = rank - 1; axis >= 0; axis--) {
        uint extent = uint(meta[1 + axis]);
        uint next = rest / extent;
        /* One divide per level and a subtraction for the remainder. The
         * device has no integer divide unit, so a division and a modulo
         * apiece would double the cost of the whole walk. */
        int coord = int(rest - next * extent);
        rest = next;
        if (axis == rank - 1) *inner = coord;
        int pos = coord - int(meta[1 + 2 * rank + axis]);
        at += nm_pad_index(pos, int(meta[1 + rank + axis]), mode)
            * int(meta[2 + 3 * rank + axis]);
    }
    return at;
}

/* A thread to four destination elements. Inside the innermost axis's interior
 * the border rule is the identity, so a run of four reads four neighbouring
 * floats and moves as one float4 -- which is where a pad spends nearly all of
 * its bandwidth, the border being a surface on a volume. The scalar tail
 * re-walks the index, which costs the same as the plain kernel would have. */
kernel void nm_pad_map(device float* dst [[buffer(0)]],
                       device const float* src [[buffer(1)]],
                       device const float* meta [[buffer(2)]],
                       constant uint& count [[buffer(3)]],
                       constant uint& lanes [[buffer(4)]],
                       constant uint& mode [[buffer(5)]],
                       uint id [[thread_position_in_grid]]) {
    uint start = id * 4u;
    if (start >= count) return;
    uint span = min(4u, count - start);
    int rank = int(meta[0]);
    int inner = 0;
    int at = nm_pad_source(meta, rank, mode, start, &inner);

    if (span == 4u && lanes == 1u && rank > 0 && (at & 3) == 0) {
        int before = int(meta[1 + 3 * rank - 1]);
        int limit = int(meta[1 + 2 * rank - 1]);
        int step = int(meta[2 + 4 * rank - 1]);
        if (step == 1 && inner >= before && inner + 4 <= before + limit) {
            *((device float4*)(dst + start)) = *((device const float4*)(src + at));
            return;
        }
    }
    for (uint k = 0u; k < span; k++) {
        int from = k == 0u ? at : nm_pad_source(meta, rank, mode, start + k, &inner);
        uint to = (start + k) * lanes;
        for (uint lane = 0u; lane < lanes; lane++) dst[to + lane] = src[from + int(lane)];
    }
}
"""

#: A view of one axis narrowed to `count` elements from `first`.
fn _window(a: NDArray, axis: int, first: int, count: int) -> NDArray {
    var shape = [extent for extent in a.shape]
    let strides = [stride for stride in a.strides]
    shape[axis] = count
    return a.view(shape, strides, a.offset + first * a.strides[axis])
}

#: A view of slot `at` along `axis`, that axis dropped — what an int index
#: gives, spelled without going through a key tuple.
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

#: A view of `a` with leading length-one axes until it has `rank` of them.
#:
#: The added strides are zero, which nothing reads: a length-one axis never
#: advances, and `is_contiguous` skips it for exactly that reason.
fn _lift(a: NDArray, rank: int) -> NDArray {
    let pad = rank - a.ndim()
    if pad <= 0 { return a }
    var shape = [1 for _axis in 0..pad]
    var strides = [0 for _axis in 0..pad]
    for axis in 0..a.ndim() {
        shape.push(a.shape[axis])
        strides.push(a.strides[axis])
    }
    return a.view(shape, strides, a.offset)
}

#: An axis argument that is an int, a list of ints, or absent.
fn _axis_list(value: any, rank: int, name: str = "axis") -> list[int] {
    if value is int { return [normalize_axis(value, rank, name)] }
    if value is list {
        let given: list[int] = [item for item in value]
        return normalize_axes(given, rank, name)
    }
    throw TypeError(f"{name} takes an int or a list of ints, not {type_of(value)}")
}

#: A fresh contiguous array of `target` holding `a`'s elements in element
#: order, through one strided copy.
#:
#: The destination is allocated with the shape the caller wants and then
#: *viewed* as the source's shape for the copy, never the other way round: a
#: view of a copy does not own its storage, so handing one back would leak the
#: allocation the first `free()` was supposed to release.
fn _materialise(a: NDArray, target: list[int]) -> NDArray {
    let out = NDArray(target, a.dtype)
    copy_into(out.view(a.shape, contiguous_strides(a.shape, a.lanes()), 0), a)
    return out
}

#: `a` as a 1-d array, and whether that cost an allocation the caller owns.
fn _flat(a: NDArray) -> (NDArray, bool) {
    let total = a.size()
    let strides = _reshape_strides(a.shape, a.strides, [total], a.lanes())
    if strides is not null { return (a.view([total], strides, a.offset), false) }
    return (_materialise(a, [total]), true)
}

#: `target` with its one `-1` resolved against `size`.
fn _resolve_shape(target: list[int], size: int) -> list[int] {
    var out: list[int] = []
    var unknown = -1
    var known = 1
    for axis in 0..target.len() {
        let extent = target[axis]
        if extent == -1 {
            if unknown >= 0 { throw ValueError("only one axis of a reshape may be -1") }
            unknown = axis
            out.push(1)
        } elif extent < 0 {
            throw ValueError(f"a shape cannot contain {extent}")
        } else {
            known *= extent
            out.push(extent)
        }
    }
    if unknown < 0 {
        if known != size { throw ValueError(f"cannot reshape {size} elements into {target}") }
        return out
    }
    if known == 0 or size % known != 0 {
        throw ValueError(f"cannot reshape {size} elements into {target}")
    }
    out[unknown] = size // known
    return out
}

#: Strides that read `shape`/`strides` as `target`, or null when no such
#: strides exist and the caller has to materialise.
#:
#: numpy's `_attempt_nocopy_reshape`, and worth having in full rather than
#: falling back on "contiguous or copy": a reshape that only splits or merges
#: axes the strides already run through consecutively is a view even when the
#: array as a whole is not contiguous, which is the common case after a slice
#: of the leading axis.
fn _reshape_strides(
    shape: list[int],
    strides: list[int],
    target: list[int],
    lanes: int
) -> list[int]? {
    if element_count(shape) == 0 { return contiguous_strides(target, lanes) }

    #: A length-one axis carries a stride nothing reads, so it must not be
    #: allowed to fail the consecutiveness test below.
    var olddims: list[int] = []
    var oldstrides: list[int] = []
    for axis in 0..shape.len() {
        if shape[axis] == 1 { continue }
        olddims.push(shape[axis])
        oldstrides.push(strides[axis])
    }
    let oldnd = olddims.len()
    let newnd = target.len()
    var out = [lanes for _axis in 0..newnd]

    var oi = 0
    var oj = 1
    var ni = 0
    var nj = 1
    while ni < newnd and oi < oldnd {
        var want = target[ni]
        var have = olddims[oi]
        #: Grow whichever side is behind until the two groups cover the same
        #: elements. Equal totals guarantee this meets before either runs out.
        while want != have {
            if want < have {
                want *= target[nj]
                nj += 1
            } else {
                have *= olddims[oj]
                oj += 1
            }
        }
        var ok = oi
        while ok < oj - 1 {
            if oldstrides[ok] != olddims[ok + 1] * oldstrides[ok + 1] { return null }
            ok += 1
        }
        out[nj - 1] = oldstrides[oj - 1]
        var nk = nj - 1
        while nk > ni {
            out[nk - 1] = out[nk] * target[nk]
            nk -= 1
        }
        ni = nj
        nj += 1
        oi = oj
        oj += 1
    }
    return out
}

#: `a` read as `shape` — a view when the strides allow it, a fresh copy when
#: they do not.
#:
#: Which of the two you get is observable: a write through a view lands in the
#: parent and a write through a copy does not. A contiguous array always gives
#: a view; so does any reshape that only splits or merges axes whose strides
#: already run consecutively. A transpose reshaped across its permuted axes is
#: the ordinary case that has to copy. One axis may be `-1`.
pub fn reshape(a: NDArray, shape: list[int]) -> NDArray {
    let target = _resolve_shape(shape, a.size())
    if target == a.shape { return a }
    let strides = _reshape_strides(a.shape, a.strides, target, a.lanes())
    if strides is not null { return a.view(target, strides, a.offset) }
    return _materialise(a, target)
}

#: `a` as one long axis, a view where the strides allow it.
pub fn ravel(a: NDArray) -> NDArray { return reshape(a, [a.size()]) }

#: `a` as one long axis, always a fresh copy — numpy's `flatten`, and the
#: spelling to reach for when the result must not alias the input.
pub fn flatten(a: NDArray) -> NDArray { return _materialise(a, [a.size()]) }

#: A view with the axes reordered, reversed by default. Always free.
pub fn transpose(a: NDArray, axes: list[int]? = null) -> NDArray {
    let rank = a.ndim()
    let wanted = axes ?? [rank - 1 - axis for axis in 0..rank]
    if wanted.len() != rank {
        throw ValueError(f"{wanted.len()} axes for an array of {rank} dimensions")
    }
    let order = normalize_axes(wanted, rank, "axes")
    var shape: list[int] = []
    var strides: list[int] = []
    for axis in order {
        shape.push(a.shape[axis])
        strides.push(a.strides[axis])
    }
    return a.view(shape, strides, a.offset)
}

#: `transpose` under the array API's name, where the permutation is required.
pub fn permute_dims(a: NDArray, axes: list[int]) -> NDArray { return transpose(a, axes) }

#: A view with two axes exchanged.
pub fn swapaxes(a: NDArray, axis1: int, axis2: int) -> NDArray {
    let rank = a.ndim()
    let first = normalize_axis(axis1, rank, "axis1")
    let second = normalize_axis(axis2, rank, "axis2")
    var order = [axis for axis in 0..rank]
    order[first] = second
    order[second] = first
    return transpose(a, order)
}

#: A view with axes moved to new positions, every other axis keeping its order.
pub fn moveaxis(a: NDArray, source: any, destination: any) -> NDArray {
    let rank = a.ndim()
    let sources = _axis_list(source, rank, "source")
    let targets = _axis_list(destination, rank, "destination")
    if sources.len() != targets.len() {
        throw ValueError(f"{sources.len()} sources and {targets.len()} destinations")
    }
    var order: list[int] = []
    for axis in 0..rank {
        var moved = false
        for at in sources {
            if at == axis { moved = true }
        }
        if not moved { order.push(axis) }
    }
    #: The moved axes go in by ascending destination, so each insertion sees
    #: the positions the earlier ones have already claimed. An insertion sort
    #: because this list is as long as the array has axes.
    var pairs: list[list[int]] = []
    for k in 0..targets.len() { pairs.push([targets[k], sources[k]]) }
    for i in 1..pairs.len() {
        var j = i
        while j > 0 and pairs[j - 1][0] > pairs[j][0] {
            let hold = pairs[j - 1]
            pairs[j - 1] = pairs[j]
            pairs[j] = hold
            j -= 1
        }
    }
    for pair in pairs { order.insert(pair[0], pair[1]) }
    return transpose(a, order)
}

#: numpy's older `rollaxis`: one axis pulled out and reinserted so that it ends
#: up *before* the axis that was at `start`.
pub fn rollaxis(a: NDArray, axis: int, start: int = 0) -> NDArray {
    let rank = a.ndim()
    let at = normalize_axis(axis, rank)
    var to = if start < 0 { start + rank } else { start }
    if to < 0 or to > rank { throw ValueError(f"start {start} is out of bounds") }
    #: Removing the axis first shifts every later position down by one, which
    #: is the off-by-one that separates this from `moveaxis`.
    if at < to { to -= 1 }
    if at == to { return a }
    var order = [other for other in 0..rank]
    order.remove_at(at)
    order.insert(to, at)
    return transpose(a, order)
}

#: A view with a length-one axis inserted at `axis`.
pub fn expand_dims(a: NDArray, axis: int) -> NDArray {
    let rank = a.ndim()
    let at = normalize_axis(axis, rank + 1)
    var shape: list[int] = []
    var strides: list[int] = []
    for other in 0..rank {
        if other == at {
            shape.push(1)
            strides.push(0)
        }
        shape.push(a.shape[other])
        strides.push(a.strides[other])
    }
    if at == rank {
        shape.push(1)
        strides.push(0)
    }
    return a.view(shape, strides, a.offset)
}

#: A view with the length-one axes dropped — all of them, or the named ones.
pub fn squeeze(a: NDArray, axis: any = null) -> NDArray {
    let rank = a.ndim()
    var drop: dict[int, bool] = {}
    if axis is null {
        for other in 0..rank {
            if a.shape[other] == 1 { drop[other] = true }
        }
    } else {
        for at in _axis_list(axis, rank) {
            if a.shape[at] != 1 {
                let extent = a.shape[at]
                throw ValueError(f"axis {at} has {extent} elements and cannot be squeezed")
            }
            drop[at] = true
        }
    }
    var shape: list[int] = []
    var strides: list[int] = []
    for other in 0..rank {
        if drop.get(other, false) { continue }
        shape.push(a.shape[other])
        strides.push(a.strides[other])
    }
    return a.view(shape, strides, a.offset)
}

#: A view of at least one dimension: a 0-d array becomes a 1-d one.
pub fn atleast_1d(a: NDArray) -> NDArray {
    if a.ndim() >= 1 { return a }
    return a.view([1], [a.lanes()], a.offset)
}

#: A view of at least two dimensions. A 1-d array becomes one row, which is
#: what makes `vstack` of vectors stack them as rows.
pub fn atleast_2d(a: NDArray) -> NDArray {
    if a.ndim() >= 2 { return a }
    return _lift(atleast_1d(a), 2)
}

#: A view of at least three dimensions, the new axis trailing rather than
#: leading — a 1-d array of n becomes `[1, n, 1]`, as numpy's does, so that
#: `dstack` lays vectors along the depth axis.
pub fn atleast_3d(a: NDArray) -> NDArray {
    let rank = a.ndim()
    if rank >= 3 { return a }
    let plane = _lift(atleast_1d(a), 2)
    return expand_dims(plane, 2)
}

#: A view of `a` stretched to `shape`, every length-one and every missing
#: leading axis reading one slot from every thread.
pub fn broadcast_to(a: NDArray, shape: list[int]) -> NDArray {
    return a.broadcast_to(shape)
}

#: Every array as a view of their common shape. No copies: this is
#: `array.jai`'s broadcasting rule applied across a list instead of a pair.
pub fn broadcast_arrays(arrays: list[NDArray]) -> list[NDArray] {
    if arrays.len() == 0 { return [] }
    var shape = [extent for extent in arrays[0].shape]
    for part in arrays { shape = broadcast_shapes(shape, part.shape) }
    return [part.broadcast_to(shape) for part in arrays]
}

#: A view of the `offset`-th diagonal of the `axis1` x `axis2` plane.
#:
#: A view and not a gather, because the diagonal's stride is the sum of the two
#: it replaces. numpy returns a read-only copy here; this returns the window,
#: so a write to it lands on the diagonal of the parent.
pub fn diagonal(a: NDArray, offset: int = 0, axis1: int = 0, axis2: int = 1) -> NDArray {
    return diagonal_view(a, offset, axis1, axis2)
}

#: The dtype a joined result carries, folded over the parts.
fn _joined_dtype(arrays: list[NDArray]) -> int {
    var dtype = arrays[0].dtype
    for part in arrays { dtype = promote(dtype, part.dtype) }
    return dtype
}

#: The parts laid end to end along an existing axis.
#:
#: One dispatch per part, each writing straight into its slice of the
#: destination through the strided copy kernel. A part that is a transpose or a
#: flip needs no materialising first: the copy reads it where it lies.
pub fn concatenate(arrays: list[NDArray], axis: int = 0) -> NDArray {
    if arrays.len() == 0 { throw ValueError("concatenate needs at least one array") }
    let rank = arrays[0].ndim()
    if rank == 0 { throw ValueError("a 0-d array has no axis to concatenate along") }
    let at = normalize_axis(axis, rank)

    var total = 0
    for part in arrays {
        if part.ndim() != rank {
            throw ValueError(f"a part of rank {part.ndim()} among parts of rank {rank}")
        }
        for other in 0..rank {
            if other != at and part.shape[other] != arrays[0].shape[other] {
                throw ValueError(f"shapes {part.shape} and {arrays[0].shape} differ off axis {at}")
            }
        }
        total += part.shape[at]
    }
    var shape = [extent for extent in arrays[0].shape]
    shape[at] = total
    let out = NDArray(shape, _joined_dtype(arrays))

    var cursor = 0
    for part in arrays {
        let width = part.shape[at]
        if width > 0 { copy_into(_window(out, at, cursor, width), part) }
        cursor += width
    }
    return out
}

#: The parts on a new axis, all of them the same shape.
#:
#: Not `concatenate` of expanded views: writing each part into the slot it
#: owns is the same one dispatch per part with no intermediate views to build.
pub fn stack(arrays: list[NDArray], axis: int = 0) -> NDArray {
    if arrays.len() == 0 { throw ValueError("stack needs at least one array") }
    let rank = arrays[0].ndim()
    let at = normalize_axis(axis, rank + 1)
    for part in arrays {
        if part.shape != arrays[0].shape {
            throw ValueError(f"stack needs one shape; got {part.shape} and {arrays[0].shape}")
        }
    }
    var shape: list[int] = []
    for other in 0..rank {
        if other == at { shape.push(arrays.len()) }
        shape.push(arrays[0].shape[other])
    }
    if at == rank { shape.push(arrays.len()) }

    let out = NDArray(shape, _joined_dtype(arrays))
    for index in 0..arrays.len() { copy_into(_slot(out, at, index), arrays[index]) }
    return out
}

#: The parts stacked as rows. A 1-d part becomes one row first, which is why
#: `vstack` of vectors gives a matrix rather than a longer vector.
pub fn vstack(arrays: list[NDArray]) -> NDArray {
    return concatenate([atleast_2d(part) for part in arrays], 0)
}

#: The parts laid side by side: along axis 1, or along the only axis 1-d parts
#: have.
pub fn hstack(arrays: list[NDArray]) -> NDArray {
    if arrays.len() == 0 { throw ValueError("hstack needs at least one array") }
    let lifted = [atleast_1d(part) for part in arrays]
    return concatenate(lifted, if lifted[0].ndim() == 1 { 0 } else { 1 })
}

#: The parts laid along the depth axis, every part lifted to three dimensions.
pub fn dstack(arrays: list[NDArray]) -> NDArray {
    return concatenate([atleast_3d(part) for part in arrays], 2)
}

#: The 1-d parts as columns of a matrix; parts that already have two or more
#: dimensions are joined along axis 1 unchanged.
pub fn column_stack(arrays: list[NDArray]) -> NDArray {
    var columns: list[NDArray] = []
    for part in arrays {
        columns.push(if part.ndim() <= 1 { expand_dims(atleast_1d(part), 1) } else { part })
    }
    return concatenate(columns, 1)
}

#: The first index of each part, closed with `extent`.
fn _cut_points(extent: int, sections: any, even: bool) -> list[int] {
    if sections is int {
        if sections <= 0 {
            throw ValueError(f"a split needs a positive section count, not {sections}")
        }
        if even and extent % sections != 0 {
            throw ValueError(f"an even split of {extent} into {sections} does not divide")
        }
        #: The remainder goes one element each to the leading parts, which is
        #: how `array_split` differs from a plain floor division.
        let base = extent // sections
        let extra = extent % sections
        var points = [0]
        var at = 0
        for k in 0..sections {
            at += base + (if k < extra { 1 } else { 0 })
            points.push(at)
        }
        return points
    }
    if sections is list {
        let cuts: list[int] = [item for item in sections]
        var points = [0]
        for value in cuts {
            let shifted = if value < 0 { value + extent } else { value }
            points.push(max(0, min(extent, shifted)))
        }
        points.push(extent)
        return points
    }
    throw TypeError("a split takes a section count or a list of cut indices")
}

fn _split_views(a: NDArray, sections: any, axis: int, even: bool) -> list[NDArray] {
    let at = normalize_axis(axis, a.ndim())
    let points = _cut_points(a.shape[at], sections, even)
    var out: list[NDArray] = []
    for k in 0..points.len() - 1 {
        out.push(_window(a, at, points[k], max(0, points[k + 1] - points[k])))
    }
    return out
}

#: `a` cut into `sections` parts along an axis, or at the given cut indices.
#:
#: The parts are views, so this costs nothing and a write to a part lands in
#: `a`. An int section count must divide the axis exactly; `array_split` is
#: the one that tolerates a remainder.
pub fn split(a: NDArray, sections: any, axis: int = 0) -> list[NDArray] {
    return _split_views(a, sections, axis, true)
}

#: `split` without the even-division rule: a remainder is spread one element
#: each over the leading parts.
pub fn array_split(a: NDArray, sections: any, axis: int = 0) -> list[NDArray] {
    return _split_views(a, sections, axis, false)
}

#: `split` along axis 0, on an array of at least two dimensions.
pub fn vsplit(a: NDArray, sections: any) -> list[NDArray] {
    if a.ndim() < 2 { throw ValueError("vsplit needs at least two dimensions") }
    return split(a, sections, 0)
}

#: `split` along axis 1, or along the only axis of a 1-d array.
pub fn hsplit(a: NDArray, sections: any) -> list[NDArray] {
    if a.ndim() == 0 { throw ValueError("hsplit needs at least one dimension") }
    return split(a, sections, if a.ndim() == 1 { 0 } else { 1 })
}

#: `split` along axis 2, on an array of at least three dimensions.
pub fn dsplit(a: NDArray, sections: any) -> list[NDArray] {
    if a.ndim() < 3 { throw ValueError("dsplit needs at least three dimensions") }
    return split(a, sections, 2)
}

#: `a` repeated `reps` times along each axis.
#:
#: One dispatch, whatever the repeat counts. The destination is viewed with
#: every axis split into (repeat, extent) — a contiguous split is always a
#: view — and the source is read with a length-one axis in each repeat slot,
#: which broadcasts to stride 0. The copy kernel then writes every tile from
#: the same source elements without any of them being materialised twice.
pub fn tile(a: NDArray, reps: list[int]) -> NDArray {
    let rank = max(a.ndim(), reps.len())
    var counts = [1 for _axis in 0..rank]
    let lead = rank - reps.len()
    for k in 0..reps.len() {
        if reps[k] < 0 { throw ValueError(f"a repeat count cannot be negative: {reps}") }
        counts[lead + k] = reps[k]
    }
    let source = _lift(a, rank)

    var shape: list[int] = []
    var split_shape: list[int] = []
    var source_shape: list[int] = []
    var source_strides: list[int] = []
    for axis in 0..rank {
        shape.push(counts[axis] * source.shape[axis])
        split_shape.push(counts[axis])
        split_shape.push(source.shape[axis])
        source_shape.push(1)
        source_shape.push(source.shape[axis])
        source_strides.push(0)
        source_strides.push(source.strides[axis])
    }
    let out = NDArray(shape, a.dtype)
    if out.size() == 0 { return out }
    copy_into(
        out.view(split_shape, contiguous_strides(split_shape, out.lanes()), 0),
        source.view(source_shape, source_strides, source.offset)
    )
    return out
}

#: Each element repeated along an axis, or along the flattened array when
#: `axis` is absent — numpy's `repeat`.
#:
#: A single count is one dispatch and no kernel: split the destination axis
#: into (extent, count), give the source a length-one axis to broadcast over,
#: and the strided copy writes `[a, a, b, b, c, c]` in one pass. A list of
#: counts takes one dispatch per source element instead, because the parts are
#: ragged; that list is a host-side argument, so its length is a bound the
#: caller has already paid for once.
pub fn repeat(a: NDArray, repeats: any, axis: int? = null) -> NDArray {
    var source = a
    var owned = false
    var at = 0
    if axis is null {
        let (flat, made) = _flat(a)
        source = flat
        owned = made
    } else {
        at = normalize_axis(axis, a.ndim())
    }
    defer { if owned { source.free() } }

    let extent = source.shape[at]
    let counts = _repeat_counts(repeats, extent)
    if counts.len() == 1 { return _repeat_uniform(source, at, counts[0]) }

    var total = 0
    for count in counts { total += count }
    var shape = [item for item in source.shape]
    shape[at] = total
    let out = NDArray(shape, a.dtype)

    var cursor = 0
    for index in 0..extent {
        let count = counts[index]
        if count > 0 {
            copy_into(_window(out, at, cursor, count), _window(source, at, index, 1))
        }
        cursor += count
    }
    return out
}

#: The repeat counts, as a single-entry list when every element shares one.
fn _repeat_counts(repeats: any, extent: int) -> list[int] {
    if repeats is int {
        if repeats < 0 { throw ValueError(f"a repeat count cannot be negative: {repeats}") }
        return [repeats]
    }
    if not repeats is list { throw TypeError("repeats takes an int or a list of ints") }
    let counts: list[int] = [item for item in repeats]
    if counts.len() != extent {
        throw ValueError(f"{counts.len()} repeat counts for an axis of {extent}")
    }
    var uniform = true
    for count in counts {
        if count < 0 { throw ValueError(f"a repeat count cannot be negative: {counts}") }
        if count != counts[0] { uniform = false }
    }
    #: An axis of one element is already uniform, and a uniform list is worth
    #: detecting: it turns `extent` dispatches back into one.
    if uniform and extent > 0 { return [counts[0]] }
    return counts
}

fn _repeat_uniform(source: NDArray, at: int, count: int) -> NDArray {
    var shape = [item for item in source.shape]
    shape[at] = source.shape[at] * count
    let out = NDArray(shape, source.dtype)
    if out.size() == 0 { return out }

    var split_shape: list[int] = []
    var source_shape: list[int] = []
    var source_strides: list[int] = []
    for axis in 0..source.ndim() {
        source_shape.push(source.shape[axis])
        source_strides.push(source.strides[axis])
        split_shape.push(source.shape[axis])
        if axis == at {
            split_shape.push(count)
            source_shape.push(1)
            source_strides.push(0)
        }
    }
    copy_into(
        out.view(split_shape, contiguous_strides(split_shape, out.lanes()), 0),
        source.view(source_shape, source_strides, source.offset)
    )
    return out
}

#: `a`'s elements cycled to fill `shape` — numpy's `resize`, which repeats
#: rather than zero-filling.
#:
#: Filled by doubling: the first copy lays the source down once, and each
#: further dispatch copies everything written so far. That is log2 of the
#: growth factor in dispatches instead of one per source-length block, which
#: matters when the target is many times the source.
pub fn resize(a: NDArray, shape: list[int]) -> NDArray {
    let total = element_count(shape)
    let out = NDArray(shape, a.dtype)
    if total == 0 { return out }
    if a.size() == 0 { throw ValueError("cannot resize an empty array to a non-empty shape") }

    let (source, owned) = _flat(a)
    defer { if owned { source.free() } }
    let target = out.view([total], contiguous_strides([total], out.lanes()), 0)

    var filled = min(source.size(), total)
    copy_into(_window(target, 0, 0, filled), _window(source, 0, 0, filled))
    while filled < total {
        #: Source and destination are the same allocation but never the same
        #: slots, and async dispatches run in submission order, so the block
        #: being read has landed before the block being written is touched.
        let take = min(filled, total - filled)
        copy_into(_window(target, 0, filled, take), _window(target, 0, 0, take))
        filled += take
    }
    return out
}

#: A view of `a` with the named axes reversed — every axis by default.
#:
#: Free: a flip negates a stride and walks from the far end. Nothing is copied,
#: and a write through the result lands in the parent, reversed.
pub fn flip(a: NDArray, axis: any = null) -> NDArray {
    let rank = a.ndim()
    let axes = if axis is null { [other for other in 0..rank] } else { _axis_list(axis, rank) }
    var strides = [stride for stride in a.strides]
    var offset = a.offset
    for at in axes {
        if a.shape[at] == 0 { continue }
        offset += (a.shape[at] - 1) * a.strides[at]
        strides[at] = -a.strides[at]
    }
    return a.view(a.shape, strides, offset)
}

#: A view with the columns reversed.
pub fn fliplr(a: NDArray) -> NDArray {
    if a.ndim() < 2 { throw ValueError("fliplr needs at least two dimensions") }
    return flip(a, 1)
}

#: A view with the rows reversed.
pub fn flipud(a: NDArray) -> NDArray {
    if a.ndim() < 1 { throw ValueError("flipud needs at least one dimension") }
    return flip(a, 0)
}

fn _roll_into(dst: NDArray, source: NDArray, axis: int, shift: int) -> void {
    let extent = source.shape[axis]
    if extent == 0 { return }
    var by = shift % extent
    if by < 0 { by += extent }
    if by == 0 {
        copy_into(dst, source)
        return
    }
    copy_into(_window(dst, axis, by, extent - by), _window(source, axis, 0, extent - by))
    copy_into(_window(dst, axis, 0, by), _window(source, axis, extent - by, by))
}

fn _shift_list(shift: any, wanted: int) -> list[int] {
    if shift is int { return [shift for _k in 0..wanted] }
    if shift is list {
        let given: list[int] = [item for item in shift]
        if given.len() != wanted { throw ValueError(f"{given.len()} shifts for {wanted} axes") }
        return given
    }
    throw TypeError("shift takes an int or a list of ints")
}

#: The elements moved along, wrapping round the end.
#:
#: Two dispatches per axis — the part that stays and the part that wraps — each
#: a slice of the source written into a slice of the destination. Several axes
#: roll one after another because rolls along different axes commute; that
#: costs an intermediate allocation per axis after the first, which is cheaper
#: than the 2^k parts a single pass would have to cut the array into.
pub fn roll(a: NDArray, shift: any, axis: any = null) -> NDArray {
    if axis is null {
        let out = NDArray(a.shape, a.dtype)
        if a.size() == 0 { return out }
        let (source, owned) = _flat(a)
        defer { if owned { source.free() } }
        let target = out.view([out.size()], contiguous_strides([out.size()], out.lanes()), 0)
        _roll_into(target, source, 0, _shift_list(shift, 1)[0])
        return out
    }
    let axes = _axis_list(axis, a.ndim())
    let shifts = _shift_list(shift, axes.len())
    var current = a
    var owned = false
    for k in 0..axes.len() {
        let out = NDArray(a.shape, a.dtype)
        _roll_into(out, current, axes[k], shifts[k])
        if owned { current.free() }
        current = out
        owned = true
    }
    if not owned { return a.copy() }
    return current
}

#: A view rotated by `k` right angles in the plane of two axes.
#:
#: Free, all four ways: a rotation is a transpose and a flip, and both of those
#: are stride rewrites. numpy's decomposition exactly.
pub fn rot90(a: NDArray, k: int = 1, axes: list[int]? = null) -> NDArray {
    let rank = a.ndim()
    if rank < 2 { throw ValueError("rot90 needs at least two dimensions") }
    let plane = normalize_axes(axes ?? [0, 1], rank, "axes")
    if plane.len() != 2 { throw ValueError(f"rot90 turns in a plane of two axes, not {plane}") }
    var turns = k % 4
    if turns < 0 { turns += 4 }
    if turns == 0 { return a }

    var order = [axis for axis in 0..rank]
    order[plane[0]] = plane[1]
    order[plane[1]] = plane[0]
    if turns == 1 { return transpose(flip(a, plane[1]), order) }
    if turns == 2 { return flip(flip(a, plane[0]), plane[1]) }
    return flip(transpose(a, order), plane[1])
}

fn _pad_mode(mode: str) -> int {
    if mode == "edge" { return PAD_EDGE }
    if mode == "wrap" { return PAD_WRAP }
    if mode == "reflect" { return PAD_REFLECT }
    throw ValueError(f"unknown pad mode {mode}; try constant, edge, wrap or reflect")
}

fn _pad_meta(out: NDArray, a: NDArray, before: list[int]) -> Buffer {
    var values = [float(a.ndim())]
    for extent in out.shape { values.push(float(extent)) }
    for extent in a.shape { values.push(float(extent)) }
    for width in before { values.push(float(width)) }
    values.push(float(a.offset))
    for stride in a.strides { values.push(float(stride)) }
    let meta = Buffer(values.len())
    meta.upload(values)
    return meta
}

#: A border added around `a`, one `[before, after]` pair per axis.
#:
#: `constant` is a fill and one strided copy of the interior — two dispatches
#: of kernels `array.jai` already owns, and no reason to compile another.
#: `edge`, `wrap` and `reflect` go through this module's one kernel, which maps
#: each destination position back onto a source one; `reflect` folds about the
#: first and last element without repeating them, as numpy's does.
pub fn pad(
    a: NDArray,
    pad_width: list[list[int]],
    mode: str = "constant",
    constant_value: float = 0.0
) -> NDArray {
    let rank = a.ndim()
    if pad_width.len() != rank {
        throw ValueError(f"{pad_width.len()} pad widths for an array of {rank} dimensions")
    }
    var before: list[int] = []
    var shape: list[int] = []
    for axis in 0..rank {
        let pair = pad_width[axis]
        if pair.len() != 2 {
            throw ValueError(f"a pad width is a [before, after] pair, not {pair}")
        }
        if pair[0] < 0 or pair[1] < 0 {
            throw ValueError(f"a pad width cannot be negative: {pair}")
        }
        before.push(pair[0])
        shape.push(pair[0] + a.shape[axis] + pair[1])
    }
    let out = NDArray(shape, a.dtype)

    if mode == "constant" {
        fill(out, constant_value)
        if a.size() > 0 {
            var offset = 0
            for axis in 0..rank { offset += before[axis] * out.strides[axis] }
            copy_into(out.view(a.shape, out.strides, offset), a)
        }
        return out
    }

    let selector = _pad_mode(mode)
    for axis in 0..rank {
        #: A border that reads from the array has nothing to read when the
        #: axis is empty, and the kernel's index rule would divide by zero.
        if a.shape[axis] == 0 and shape[axis] > 0 {
            throw ValueError(f"{mode} padding cannot extend the empty axis {axis}")
        }
    }
    let count = out.size()
    if count == 0 { return out }
    let meta = _pad_meta(out, a, before)
    defer { meta.free() }
    kernel_of(SOURCE, "nm_pad_map").dispatch_async(
        [out.storage, a.storage, meta],
        (count + 3) // 4,
        [count, out.lanes(), selector]
    )
    return out
}

#: Sorting, searching and set operations.
#:
#: The sort is a bitonic network on the device, run over the fast axis of the
#: array. Each segment is padded to the next power of two, the network runs one
#: dispatch per (stage, pass), and an index buffer rides alongside the values so
#: `sort` and `argsort` come out of the same network rather than two.
#:
#: Why not the host: jaicv's `core/misc.jai` sorts a matrix by downloading it
#: and handing each row to `list.sort`. That is the right call for an image row
#: -- a few thousand values, once -- and the wrong one for a million-element
#: array, where the download is a synchronise and the loop that reassembles the
#: rows is interpreted. A host path survives here only below
#: `HOST_SORT_LIMIT`, where dispatch latency is the whole bill.
#:
#: Everything else in the module is built on that sort. `unique` sorts, flags
#: the first element of each run and compacts through `array.jai`'s
#: `exclusive_scan`; the set operations are `unique` plus a membership test;
#: `searchsorted` and `isin` bisect the sorted array with one thread per query.
#: `lexsort` composes one stable `argsort` per key.
#:
#: ```
#: let a = NDArray([2, 4])
#: a.upload([3.0, 1.0, 2.0, 0.0, 9.0, 7.0, 8.0, 6.0])
#: sort(a).to_nested()      #: [[0, 1, 2, 3], [6, 7, 8, 9]]
#: argsort(a).to_nested()   #: [[3, 1, 2, 0], [3, 1, 2, 0]]
#: ```
#:
#: Not here: `median` and `percentile`, which `stats` builds on `partition`;
#: histograms; and `nonzero`, which `index` owns.
from .array import BOOL, FLOAT32, INT32, GROUP, NDArray, PRELUDE
from .array import contiguous_strides, copy_into, exclusive_scan, f32_from_bits
from .array import fill, gather_flat, kernel_of, normalize_axis, promote
from .array import require_real, scatter_flat

#: Elements below which a sort runs on the host rather than on the device.
#:
#: The network costs one dispatch per (stage, pass) -- fifty-five of them for a
#: 1024-wide segment, two hundred and ten for a million -- and each one is a
#: fixed encode plus a launch. Below some amount of work that latency is the
#: entire cost and a download, a `list.sort` per segment and an upload win.
#:
#: 4096 is a guess. Nothing in this module has been measured; the number is
#: written down so it can be tuned against a real machine, not because a real
#: machine produced it.
pub const HOST_SORT_LIMIT = 4096

let SOURCE = PRELUDE + """

/* Ascending, NaN last -- which is the order numpy sorts into -- with the
 * descending flag turning the whole thing over, NaN with it.
 *
 * `carry` breaks a tie by the slot the element started in, which is what makes
 * `argsort` stable. It also pins the padding down: a padded slot's index is at
 * least the segment length, so it loses every tie to a real element and cannot
 * end up inside the first n slots the harvest reads back.
 *
 * A plain sort carries no indices, so ties do get swapped. That is harmless:
 * the only values a pad ever ties with are values equal to it, and swapping
 * two equal floats leaves the segment's contents alone. The one visible
 * consequence is that -0.0 and 0.0, which compare equal, come out in an
 * unspecified order where numpy would keep them apart. */
inline bool nds_after(float a, float b, uint ia, uint ib, uint descending, uint carry) {
    bool a_nan = isnan(a);
    bool b_nan = isnan(b);
    if (a_nan || b_nan) {
        if (a_nan && b_nan) return carry != 0u && ia > ib;
        return descending != 0u ? !a_nan : a_nan;
    }
    if (a != b) return descending != 0u ? a < b : a > b;
    return carry != 0u && ia > ib;
}

/* Whether one pair is the wrong way round for the direction this stage of the
 * network wants of it. `up` says the sub-sequence ending here runs upward. */
inline bool nds_swaps(float a, float b, float ia, float ib,
                      bool up, uint descending, uint carry) {
    return nds_after(a, b, uint(ia), uint(ib), descending, carry) == up;
}

/* Strictly before, in the same order the network sorts into. Two NaNs are not
 * ordered against each other; `nds_same` is what asks whether they match. */
inline bool nds_below(float a, float b) {
    if (isnan(b)) return !isnan(a);
    if (isnan(a)) return false;
    return a < b;
}

/* One run of `unique`, one match of `isin`: NaN equals NaN here, because a
 * `unique` that handed back every NaN in the input is not a unique. */
inline bool nds_same(float a, float b) {
    return a == b || (isnan(a) && isnan(b));
}

/* Whether the bisection steps past `value` looking for where `key` belongs.
 * The left side stops at the first equal element, the right side steps past
 * the last. A NaN key lands at the end, matching where the sort puts one. */
inline bool nds_precedes(float value, float key, uint right) {
    if (right != 0u) return !nds_below(key, value);
    return nds_below(value, key);
}

/* Each slot's own position within its segment, which the network then permutes
 * into the answer `argsort` returns. A `wrap` of all-ones makes this a plain
 * iota, which is what the run-start scatter in `unique` wants. */
kernel void nds_iota(device float* dst [[buffer(0)]],
                     constant uint& count [[buffer(1)]],
                     constant uint& wrap [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    dst[id] = float(id & wrap);
}

/* One stage-and-pass of a bitonic network over the fast axis of a
 * [rows, padded] workspace, one thread per compare-exchange.
 *
 * Every extent involved is a power of two and the device has no integer
 * divide, so the widths arrive as logarithms: the row is a shift, the position
 * inside the row is that shifted back and subtracted, and the pair's low index
 * is another shift plus a mask. A `/` and a `%` apiece would cost more than
 * the exchange they address.
 *
 * `keys` is bound to `values` when nothing is being carried -- one already
 * resident allocation rather than a second binding nobody reads -- so every
 * touch of it is behind the `carry` test. */
kernel void nds_bitonic(device float* values [[buffer(0)]],
                        device float* keys [[buffer(1)]],
                        constant uint& pairs [[buffer(2)]],
                        constant uint& half_log [[buffer(3)]],
                        constant uint& j_log [[buffer(4)]],
                        constant uint& stage [[buffer(5)]],
                        constant uint& descending [[buffer(6)]],
                        constant uint& carry [[buffer(7)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= pairs) return;
    uint row = id >> half_log;
    uint slot = id - (row << half_log);
    uint j = 1u << j_log;
    uint low = ((slot >> j_log) << (j_log + 1u)) + (slot & (j - 1u));
    uint left = (row << (half_log + 1u)) + low;
    uint right = left + j;
    bool up = (low & stage) == 0u;

    float a = values[left];
    float b = values[right];
    float ka = carry != 0u ? keys[left] : 0.0f;
    float kb = carry != 0u ? keys[right] : 0.0f;
    if (!nds_swaps(a, b, ka, kb, up, descending, carry)) return;
    values[left] = b;
    values[right] = a;
    if (carry != 0u) {
        keys[left] = kb;
        keys[right] = ka;
    }
}

/* The same pass, four pairs to a thread.
 *
 * Usable once `j` reaches four: the four consecutive low indices then sit
 * inside one `j` block and below one stage boundary, so they share a direction
 * and each side of the exchange is a single aligned float4. Both sides stay
 * 16-byte aligned because `low` and `j` are multiples of four and a row starts
 * at a multiple of `padded`. */
kernel void nds_bitonic4(device float* values [[buffer(0)]],
                         device float* keys [[buffer(1)]],
                         constant uint& quads [[buffer(2)]],
                         constant uint& quad_log [[buffer(3)]],
                         constant uint& j_log [[buffer(4)]],
                         constant uint& stage [[buffer(5)]],
                         constant uint& descending [[buffer(6)]],
                         constant uint& carry [[buffer(7)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= quads) return;
    uint row = id >> quad_log;
    uint slot = (id - (row << quad_log)) << 2u;
    uint j = 1u << j_log;
    uint low = ((slot >> j_log) << (j_log + 1u)) + (slot & (j - 1u));
    uint left = (row << (quad_log + 3u)) + low;
    uint right = left + j;
    bool up = (low & stage) == 0u;

    device float4* wide = (device float4*)values;
    device float4* wide_keys = (device float4*)keys;
    float4 a = wide[left >> 2u];
    float4 b = wide[right >> 2u];
    float4 ka = float4(0.0f);
    float4 kb = float4(0.0f);
    if (carry != 0u) {
        ka = wide_keys[left >> 2u];
        kb = wide_keys[right >> 2u];
    }
    bool4 swap;
    swap.x = nds_swaps(a.x, b.x, ka.x, kb.x, up, descending, carry);
    swap.y = nds_swaps(a.y, b.y, ka.y, kb.y, up, descending, carry);
    swap.z = nds_swaps(a.z, b.z, ka.z, kb.z, up, descending, carry);
    swap.w = nds_swaps(a.w, b.w, ka.w, kb.w, up, descending, carry);
    if (!any(swap)) return;
    wide[left >> 2u] = select(a, b, swap);
    wide[right >> 2u] = select(b, a, swap);
    if (carry != 0u) {
        wide_keys[left >> 2u] = select(ka, kb, swap);
        wide_keys[right >> 2u] = select(kb, ka, swap);
    }
}

/* One thread per query, bisecting a sorted array.
 *
 * The top of every query's search visits the same handful of positions, so the
 * group reads an evenly spaced sample of the array into threadgroup memory
 * once and bisects that first. Eight threadgroup probes then narrow the query
 * to a window of `stride` elements before its first global load, which is
 * where a search over a large array otherwise spends all of its misses.
 *
 * `stride` of zero turns the sample off for an array too short to pay for it.
 * The grid is rounded up to whole groups so the sample is always filled by a
 * full group, which is also why the bounds test comes after the barrier. */
kernel void nds_searchsorted(device float* dst [[buffer(0)]],
                             device const float* sorted [[buffer(1)]],
                             device const float* queries [[buffer(2)]],
                             constant uint& count [[buffer(3)]],
                             constant uint& length [[buffer(4)]],
                             constant uint& stride [[buffer(5)]],
                             constant uint& right [[buffer(6)]],
                             uint id [[thread_position_in_grid]],
                             uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float sample[ND_GROUP];
    if (stride != 0u) sample[lid] = sorted[lid * stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (id >= count) return;

    float key = queries[id];
    uint lo = 0u;
    uint hi = length;
    if (stride != 0u) {
        uint coarse_lo = 0u;
        uint coarse_hi = ND_GROUP;
        while (coarse_lo < coarse_hi) {
            uint mid = (coarse_lo + coarse_hi) >> 1u;
            if (nds_precedes(sample[mid], key, right)) coarse_lo = mid + 1u;
            else coarse_hi = mid;
        }
        /* Every sample below `coarse_lo` precedes the key and the one at it
         * does not, so the answer lies in (coarse_lo - 1, coarse_lo] samples --
         * and past the last sample it is the tail of the array. */
        lo = coarse_lo == 0u ? 0u : (coarse_lo - 1u) * stride;
        hi = coarse_lo == ND_GROUP ? length : coarse_lo * stride;
    }
    while (lo < hi) {
        uint mid = (lo + hi) >> 1u;
        if (nds_precedes(sorted[mid], key, right)) lo = mid + 1u;
        else hi = mid;
    }
    dst[id] = float(lo);
}

/* `isin`, given the insertion point each query would take: the value is
 * present exactly when the array holds an equal one right there. */
kernel void nds_membership(device float* dst [[buffer(0)]],
                           device const float* sorted [[buffer(1)]],
                           device const float* at [[buffer(2)]],
                           device const float* queries [[buffer(3)]],
                           constant uint& count [[buffer(4)]],
                           constant uint& length [[buffer(5)]],
                           constant uint& invert [[buffer(6)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    uint slot = uint(at[id]);
    bool found = slot < length && nds_same(sorted[slot], queries[id]);
    dst[id] = found != (invert != 0u) ? 1.0f : 0.0f;
}

/* The first element of each run of equal values in a sorted array, which the
 * scan then turns into packed positions.
 *
 * Neighbouring threads overlap by exactly one element, which the cache already
 * serves; staging a tile for a one-element halo would cost a barrier and buy
 * nothing. */
kernel void nds_unique_flags(device float* dst [[buffer(0)]],
                             device const float* sorted [[buffer(1)]],
                             constant uint& count [[buffer(2)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    dst[id] = id == 0u || !nds_same(sorted[id], sorted[id - 1u]) ? 1.0f : 0.0f;
}

/* How long each run is, from where the runs start. The last one reaches the
 * end of the sorted array, which is the only length not a difference. */
kernel void nds_run_counts(device float* dst [[buffer(0)]],
                           device const float* starts [[buffer(1)]],
                           constant uint& runs [[buffer(2)]],
                           constant uint& total [[buffer(3)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= runs) return;
    uint here = uint(starts[id]);
    uint next = id + 1u < runs ? uint(starts[id + 1u]) : total;
    dst[id] = float(next - here);
}
"""

#: Smallest power of two at or above `n`.
fn _next_power_of_two(n: int) -> int {
    var value = 1
    while value < n { value *= 2 }
    return value
}

#: The exponent of a power of two, which is what every bitonic width travels
#: as so the kernel can shift instead of divide.
fn _log2(value: int) -> int {
    var bits = 0
    var rest = value
    while rest > 1 {
        rest = rest // 2
        bits += 1
    }
    return bits
}

#: What the slots past the end of a segment hold.
#:
#: NaN when ascending, because NaN is the one value that sorts after +inf and a
#: padding of +inf would let a real NaN slip past it and out of the first n
#: slots the harvest reads. Descending turns the order over, so the last place
#: belongs to -inf instead. Either way a pad ties only with a value equal to
#: it, which is what lets a plain sort carry no indices.
fn _pad_value(descending: bool) -> float {
    if descending { return f32_from_bits(0xFF800000) }
    return f32_from_bits(0x7FC00000)
}

#: A view of `a` with `axis` moved to the end and the other axes in order,
#: which is the layout every kernel here iterates.
fn _axis_last(a: NDArray, axis: int) -> NDArray {
    if axis == a.ndim() - 1 { return a }
    var shape: list[int] = []
    var strides: list[int] = []
    for at in 0..a.ndim() {
        if at == axis { continue }
        shape.push(a.shape[at])
        strides.push(a.strides[at])
    }
    shape.push(a.shape[axis])
    strides.push(a.strides[axis])
    return a.view(shape, strides, a.offset)
}

#: Strides that read a [rows, padded] workspace as if its rows were only the
#: `shape` the caller asked for. The fast axis is dense; everything above it
#: steps in whole padded rows.
fn _padded_strides(shape: list[int], padded: int) -> list[int] {
    var strides = [0 for _axis in shape]
    var axis = shape.len() - 1
    strides[axis] = 1
    var running = padded
    axis -= 1
    while axis >= 0 {
        strides[axis] = running
        running *= shape[axis]
        axis -= 1
    }
    return strides
}

#: `a` as a dense 1-d array at slot zero, and whether that cost an allocation
#: the caller now owns. The flag is the only honest way to say which happened.
fn _flat_1d(a: NDArray) -> (NDArray, bool) {
    if a.ndim() == 1 and a.offset == 0 and a.is_contiguous() { return (a, false) }
    let out = NDArray([a.size()], a.dtype)
    copy_into(out.view(a.shape, contiguous_strides(a.shape, a.lanes()), 0), a)
    return (out, true)
}

#: The axis a sort will run along, having rejected what cannot be sorted.
fn _sortable_axis(a: NDArray, axis: int, what: str) -> int {
    require_real(a, what)
    if a.ndim() == 0 { throw ValueError(f"{what} needs an axis and a 0-d array has none") }
    return normalize_axis(axis, a.ndim())
}

#: Reject a `kth` that names no position, the way numpy does before it starts.
fn _check_kth(kth: int, extent: int) -> void {
    let at = if kth < 0 { kth + extent } else { kth }
    if at < 0 or at >= extent {
        throw ValueError(f"kth {kth} is out of bounds for an axis of {extent}")
    }
}

#: The bitonic network itself, over an already seeded [rows, padded] workspace.
#:
#: Stage `k` sorts runs of `k` alternately up and down; the passes inside it
#: halve the compare distance from `k / 2` to one. That is log2(padded) *
#: (log2(padded) + 1) / 2 dispatches -- fifty-five at 1024 wide, two hundred
#: and ten at a million -- and every one of them is over the whole workspace,
#: which is why the host path below the threshold exists at all.
fn _run_network(
    work: NDArray,
    slots: NDArray,
    rows: int,
    padded: int,
    descending: bool,
    carry: bool
) -> void {
    let half = padded // 2
    let half_log = _log2(half)
    let flags = [if descending { 1 } else { 0 }, if carry { 1 } else { 0 }]
    var stage = 2
    while stage <= padded {
        var width = stage // 2
        while width >= 1 {
            let j_log = _log2(width)
            if width >= 4 {
                let quads = rows * (half // 4)
                kernel_of(SOURCE, "nds_bitonic4").dispatch_async(
                    [work.storage, slots.storage],
                    quads,
                    [quads, half_log - 2, j_log, stage, flags[0], flags[1]]
                )
            } else {
                let pairs = rows * half
                kernel_of(SOURCE, "nds_bitonic").dispatch_async(
                    [work.storage, slots.storage],
                    pairs,
                    [pairs, half_log, j_log, stage, flags[0], flags[1]]
                )
            }
            width = width // 2
        }
        stage *= 2
    }
}

#: One segment's positions in order, on the host.
#:
#: `list.sort` is a stable merge whose radix path bails out on a NaN and whose
#: fallback has no defined order for one, so the NaNs are pulled out here and
#: put back where this module says they belong: last, or first once the
#: descending flag has turned the order over.
fn _host_order(values: list[float], base: int, n: int, descending: bool) -> list[int] {
    var finite: list[int] = []
    var absent: list[int] = []
    for slot in 0..n {
        let value = values[base + slot]
        if value != value { absent.push(slot) } else { finite.push(slot) }
    }
    let ranked = finite.sort(key: |slot| values[base + slot], reverse: descending)
    var out: list[int] = []
    if descending {
        for slot in absent { out.push(slot) }
    }
    for slot in ranked { out.push(slot) }
    if not descending {
        for slot in absent { out.push(slot) }
    }
    return out
}

#: The whole sort on the host, for an array small enough that the network's
#: dispatch latency would dominate it. See `HOST_SORT_LIMIT`.
fn _host_sort(
    source: NDArray,
    target: NDArray,
    ranks: NDArray?,
    rows: int,
    n: int,
    descending: bool
) -> void {
    let plane = NDArray(source.shape, FLOAT32)
    copy_into(plane, source)
    let values = plane.to_list()
    plane.free()

    var ordered = [0.0 for _slot in 0..rows * n]
    var places = [0.0 for _slot in 0..rows * n]
    for row in 0..rows {
        let base = row * n
        let ranked = _host_order(values, base, n, descending)
        for slot in 0..n {
            ordered[base + slot] = values[base + ranked[slot]]
            places[base + slot] = float(ranked[slot])
        }
    }

    let held = NDArray(source.shape, FLOAT32)
    held.upload(ordered)
    copy_into(target, held)
    held.free()
    if ranks is null { return }
    let order = NDArray(source.shape, INT32)
    order.upload(places)
    copy_into(ranks, order)
    order.free()
}

#: Sort `a` along `axis` into `out`, and the permutation that did it into
#: `order` when one is asked for.
#:
#: The axis is moved to the end and the array copied dense, so the network
#: always walks a contiguous fast axis; the harvest copies back through the
#: same moved view, which is where the axis returns to where it came from.
fn _sort_into(a: NDArray, at: int, descending: bool, out: NDArray, order: NDArray?) -> void {
    let n = a.shape[at]
    if a.size() == 0 or n <= 1 {
        copy_into(out, a)
        if order is not null { fill(order, 0.0) }
        return
    }
    let rows = a.size() // n
    let moved = _axis_last(a, at)
    let target = _axis_last(out, at)
    var ranks: NDArray? = null
    if order is not null { ranks = _axis_last(order, at) }
    if rows * n <= HOST_SORT_LIMIT {
        _host_sort(moved, target, ranks, rows, n, descending)
        return
    }

    let carry = ranks is not null
    let padded = _next_power_of_two(n)
    let strides = _padded_strides(moved.shape, padded)
    let work = NDArray([rows, padded], FLOAT32)
    defer { work.free() }
    fill(work, _pad_value(descending))
    copy_into(work.view(moved.shape, strides, 0), moved)

    #: With nothing to carry the network binds the values twice rather than a
    #: second allocation, and never writes through the alias.
    var slots = work
    if carry {
        slots = NDArray([rows, padded], INT32)
        kernel_of(SOURCE, "nds_iota").dispatch_async(
            [slots.storage],
            rows * padded,
            [rows * padded, padded - 1]
        )
    }
    _run_network(work, slots, rows, padded, descending, carry)
    copy_into(target, work.view(moved.shape, strides, 0))
    if ranks is not null {
        copy_into(ranks, slots.view(moved.shape, strides, 0))
        slots.free()
    }
}

#: A sorted copy of `a` along `axis`.
#:
#: Ascending unless `descending`, stable in both directions, and NaN last --
#: which `descending` turns over with everything else, so a descending sort
#: leads with the NaNs.
pub fn sort(a: NDArray, axis: int = -1, descending: bool = false) -> NDArray {
    let at = _sortable_axis(a, axis, "sort")
    let out = NDArray(a.shape, a.dtype)
    _sort_into(a, at, descending, out, null)
    return out
}

#: The indices that would sort `a` along `axis`, stably.
#:
#: Equal elements keep the order they arrived in, in both directions, which is
#: what makes `lexsort` and `unique`'s first-occurrence index work.
pub fn argsort(a: NDArray, axis: int = -1, descending: bool = false) -> NDArray {
    let at = _sortable_axis(a, axis, "argsort")
    let out = NDArray(a.shape, a.dtype)
    defer { out.free() }
    let order = NDArray(a.shape, INT32)
    _sort_into(a, at, descending, out, order)
    return order
}

#: `a` rearranged so position `kth` along `axis` holds the element a full sort
#: would put there, with nothing greater before it and nothing smaller after.
#:
#: This is that full sort, which satisfies the contract exactly -- `kth` is
#: checked and then not used. A device selection network only starts beating a
#: bitonic sort at sizes where the sort is already hundreds of dispatches deep,
#: and it is a piece of work of its own. `stats` builds median and percentile
#: on this, and gets a stronger guarantee than it asked for.
pub fn partition(a: NDArray, kth: int, axis: int = -1) -> NDArray {
    let at = _sortable_axis(a, axis, "partition")
    _check_kth(kth, a.shape[at])
    return sort(a, at)
}

#: The indices that would `partition` `a` -- which, for the same reason, are
#: the indices that would sort it.
pub fn argpartition(a: NDArray, kth: int, axis: int = -1) -> NDArray {
    let at = _sortable_axis(a, axis, "argpartition")
    _check_kth(kth, a.shape[at])
    return argsort(a, at)
}

#: Encode the bisection, one thread to a query.
fn _searchsorted_into(
    out: NDArray,
    sorted: NDArray,
    keys: NDArray,
    count: int,
    right: bool
) -> void {
    let length = sorted.size()
    #: One sample per thread of the group, or none at all when the array is
    #: too short for the coarse pass to narrow anything.
    let stride = if length >= GROUP { length // GROUP } else { 0 }
    let grid = (count + GROUP - 1) // GROUP * GROUP
    kernel_of(SOURCE, "nds_searchsorted").dispatch_async([out.storage, sorted.storage, keys.storage], grid, [count, length, stride, if right {
        1
    } else {
        0
    }], GROUP)
}

#: Where each element of `v` would go in the sorted 1-d array `a`.
#:
#: `side` is `"left"` for the first position that keeps `a` sorted and
#: `"right"` for the last. `a` is assumed sorted and is not checked; sorting it
#: first is the caller's job, and numpy's `sorter` argument is spelled here as
#: sorting `a` yourself.
pub fn searchsorted(a: NDArray, v: NDArray, side: str = "left") -> NDArray {
    require_real(a, "searchsorted")
    require_real(v, "searchsorted")
    if a.ndim() != 1 {
        throw ValueError(f"searchsorted takes a 1-d array, not {a.ndim()} dimensions")
    }
    if side != "left" and side != "right" {
        throw ValueError(f"side is left or right, not {side}")
    }
    let out = NDArray(v.shape, INT32)
    let count = out.size()
    if count == 0 { return out }
    let (sorted, owns_sorted) = _flat_1d(a)
    let (keys, owns_keys) = _flat_1d(v)
    defer { if owns_sorted { sorted.free() } }
    defer { if owns_keys { keys.free() } }
    _searchsorted_into(out, sorted, keys, count, side == "right")
    return out
}

#: Whether each element of `element` appears anywhere in `test_elements`,
#: shaped like `element`.
#:
#: Sort the pool once, then one bisection per query. NaN matches NaN, which is
#: the same call `unique` makes when it folds them into one run.
pub fn isin(element: NDArray, test_elements: NDArray, invert: bool = false) -> NDArray {
    require_real(element, "isin")
    require_real(test_elements, "isin")
    let out = NDArray(element.shape, BOOL)
    let count = out.size()
    if count == 0 { return out }
    let (keys, owns_keys) = _flat_1d(element)
    defer { if owns_keys { keys.free() } }

    let (pool, owns_pool) = _flat_1d(test_elements)
    let sorted = sort(pool, 0)
    if owns_pool { pool.free() }
    defer { sorted.free() }

    let at = NDArray([count], INT32)
    defer { at.free() }
    _searchsorted_into(at, sorted, keys, count, false)
    kernel_of(SOURCE, "nds_membership").dispatch_async([out.storage, sorted.storage, at.storage, keys.storage], count, [count, sorted.size(), if invert {
        1
    } else {
        0
    }])
    return out
}

#: `isin` over the flattened arrays, which is numpy's older spelling of it.
pub fn in1d(element: NDArray, test_elements: NDArray, invert: bool = false) -> NDArray {
    let (flat, owned) = _flat_1d(element)
    defer { if owned { flat.free() } }
    return isin(flat, test_elements, invert)
}

#: What `unique` hands back.
#:
#: numpy varies its return arity with its flags, which a typed language cannot
#: spell; the fields nobody asked for are null instead.
pub class UniqueResult {
    #: The distinct values, ascending, with the NaNs folded into one.
    pub let values: NDArray
    #: Where each distinct value first appeared in the input, or null.
    pub let indices: NDArray?
    #: Which distinct value each input element is, shaped like the input, or
    #: null. `values` gathered through this reconstructs the input.
    pub let inverse: NDArray?
    #: How many times each distinct value appeared, or null.
    pub let counts: NDArray?

    pub fn init(
        self,
        values: NDArray,
        indices: NDArray? = null,
        inverse: NDArray? = null,
        counts: NDArray? = null
    ) {
        self.values = values
        self.indices = indices
        self.inverse = inverse
        self.counts = counts
    }

    #: Release every array in the result. A field nobody asked for is null and
    #: costs nothing here.
    pub fn free(self) -> void {
        self.values.free()
        if self.indices is not null { self.indices.free() }
        if self.inverse is not null { self.inverse.free() }
        if self.counts is not null { self.counts.free() }
    }
}

#: The distinct values of `a`, ascending, over the flattened array.
#:
#: Sort, flag the first element of each run, and compact through the exclusive
#: scan `array.jai` built the masked scatter around. The index and the inverse
#: fall straight out of the stable order: the run's first element carries the
#: smallest original position, and the scan value at each sorted slot already
#: is that slot's run number.
pub fn unique(
    a: NDArray,
    return_index: bool = false,
    return_inverse: bool = false,
    return_counts: bool = false
) -> UniqueResult {
    require_real(a, "unique")
    let (flat, owned) = _flat_1d(a)
    defer { if owned { flat.free() } }
    let n = flat.size()

    let sorted = NDArray([n], a.dtype)
    defer { sorted.free() }
    var order: NDArray? = null
    if return_index or return_inverse { order = NDArray([n], INT32) }
    _sort_into(flat, 0, false, sorted, order)

    let flags = NDArray([n], BOOL)
    defer { flags.free() }
    if n > 0 {
        kernel_of(SOURCE, "nds_unique_flags").dispatch_async(
            [flags.storage, sorted.storage],
            n,
            [n]
        )
    }
    let (offsets, total) = exclusive_scan(flags)
    defer { offsets.free() }
    let values = NDArray([total], a.dtype)
    if total > 0 { scatter_flat(values, offsets, sorted, flags) }

    var indices: NDArray? = null
    var inverse: NDArray? = null
    if order is not null {
        if return_index {
            let first = NDArray([total], INT32)
            if total > 0 { scatter_flat(first, offsets, order, flags) }
            indices = first
        }
        if return_inverse {
            let back = NDArray(a.shape, INT32)
            if n > 0 { scatter_flat(back, order, offsets) }
            inverse = back
        }
        order.free()
    }
    var counts: NDArray? = null
    if return_counts { counts = _run_lengths(offsets, flags, n, total) }
    return UniqueResult(values, indices, inverse, counts)
}

#: How long each run is, from where the runs start in the sorted array.
fn _run_lengths(offsets: NDArray, flags: NDArray, n: int, total: int) -> NDArray {
    let tally = NDArray([total], INT32)
    if total == 0 { return tally }
    let places = NDArray([n], INT32)
    let starts = NDArray([total], INT32)
    kernel_of(SOURCE, "nds_iota").dispatch_async([places.storage], n, [n, 0xFFFFFFFF])
    scatter_flat(starts, offsets, places, flags)
    kernel_of(SOURCE, "nds_run_counts").dispatch_async(
        [tally.storage, starts.storage],
        total,
        [total, n]
    )
    places.free()
    starts.free()
    return tally
}

#: The two arrays laid end to end as one 1-d array of the promoted dtype.
fn _concat_flat(a: NDArray, b: NDArray) -> NDArray {
    require_real(a, "a set operation")
    require_real(b, "a set operation")
    let out = NDArray([a.size() + b.size()], promote(a.dtype, b.dtype))
    if a.size() > 0 { copy_into(out.view([a.size()], [1], 0), _reshaped(a)) }
    if b.size() > 0 { copy_into(out.view([b.size()], [1], a.size()), _reshaped(b)) }
    return out
}

#: `a` seen as 1-d, without a copy when it is already dense.
fn _reshaped(a: NDArray) -> NDArray {
    let (flat, owned) = _flat_1d(a)
    if owned { return flat }
    return flat
}

#: The elements of `a` where `keep` is set, packed into a 1-d array.
fn _compact(a: NDArray, keep: NDArray) -> NDArray {
    let (offsets, total) = exclusive_scan(keep)
    defer { offsets.free() }
    let out = NDArray([total], a.dtype)
    if total > 0 { scatter_flat(out, offsets, a, keep) }
    return out
}

#: The distinct values of either array, ascending.
pub fn union1d(a: NDArray, b: NDArray) -> NDArray {
    let joined = _concat_flat(a, b)
    defer { joined.free() }
    let found = unique(joined)
    return found.values
}

#: The distinct values in both arrays, ascending.
pub fn intersect1d(a: NDArray, b: NDArray) -> NDArray {
    let left = unique(a)
    let right = unique(b)
    defer { left.free() }
    defer { right.free() }
    let keep = isin(left.values, right.values)
    defer { keep.free() }
    return _compact(left.values, keep)
}

#: The distinct values in `a` that are not in `b`, ascending.
pub fn setdiff1d(a: NDArray, b: NDArray) -> NDArray {
    let left = unique(a)
    let right = unique(b)
    defer { left.free() }
    defer { right.free() }
    let keep = isin(left.values, right.values, true)
    defer { keep.free() }
    return _compact(left.values, keep)
}

#: The distinct values in exactly one of the arrays, ascending.
#:
#: The two differences are each unique and disjoint from the other, so laying
#: them end to end and sorting is already the answer -- no second `unique`.
pub fn setxor1d(a: NDArray, b: NDArray) -> NDArray {
    let left = setdiff1d(a, b)
    let right = setdiff1d(b, a)
    defer { left.free() }
    defer { right.free() }
    let joined = _concat_flat(left, right)
    defer { joined.free() }
    return sort(joined, 0)
}

#: The indices that sort by several keys at once, the last key first.
#:
#: numpy's ordering: `keys[-1]` decides, `keys[-2]` breaks its ties, and so on
#: down to `keys[0]`. That falls out of one stable `argsort` per key applied
#: from the least significant upward, each one composed onto the order so far
#: -- the same reason a radix sort works from the bottom digit.
pub fn lexsort(keys: list[NDArray]) -> NDArray {
    if keys.len() == 0 { throw ValueError("lexsort needs at least one key") }
    let n = keys[0].size()
    for key in keys {
        require_real(key, "lexsort")
        if key.ndim() != 1 {
            throw ValueError(f"lexsort takes 1-d keys, not {key.ndim()} dimensions")
        }
        if key.size() != n {
            throw ValueError(f"lexsort got keys of {n} and {key.size()} elements")
        }
    }
    var order = argsort(keys[0], 0)
    for level in 1..keys.len() {
        let ranked = gather_flat(keys[level], order)
        let inner = argsort(ranked, 0)
        let composed = gather_flat(order, inner)
        ranked.free()
        inner.free()
        order.free()
        order = composed
    }
    return order
}

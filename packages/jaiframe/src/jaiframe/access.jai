#: Every way of reading or writing a subset of a frame: `iloc` by position,
#: `loc` by label, boolean masks, the scalar fast paths, and item assignment.
#:
#: All five selection shapes — one row, a list, a slice with a step, a boolean
#: mask, everything — are one class, `Pick` for positions and `Labels` for
#: labels, rather than one function apiece. pandas spells them all `df.iloc[]`
#: and dispatches on the Python type of the subscript; a named constructor per
#: shape says the same thing without the guessing, and it is what lets `iloc`,
#: `loc`, `set_iloc` and `set_loc` share one resolver.
#:
#: A selection resolves to a positions lane on the device and then to one
#: `take` per column. That is one dispatch per column whatever the selection
#: was, and no step anywhere walks the rows on the host. The mask path scans
#: once for the whole frame rather than calling `Column.compress` per column,
#: which would repeat the scan `width` times over the same flags.
#:
#: The one exception is `query`, whose predicate is a Jaithon callback and so
#: has to run once per row on the host. It is here because a predicate is
#: sometimes the only way to say what you mean; `query_mask` is the same
#: selection driven by a `Bool` column, and that is the path to reach for.
#:
#: ```
#: let top = head(frame, 3)
#: let adults = query_mask(frame, between(frame.column("age"), Value.of_int(18), Value.of_int(65)))
#: let one = iat(frame, 0, 1)
#: ```
from std.gpu import Buffer, Kernel
from std.math import NAN, is_nan
from std.random import Random
from std.re import Regex

from .column import Column, DataFrame, Dtype, Index, Series, Value
from .column import NA, MAX_ROWS
from .column import concat_columns, f32_bits, iota, positions_buffer, prefix_sum, wide_pack
from .index import IndexLookup

#: Threadgroup width of the tiled `isin` scan, and how many wanted values one
#: pass of it stages.
const WANTED_TILE = 256

let SOURCE = """#include <metal_stdlib>
using namespace metal;

constexpr constant uint WANTED_TILE = 256;
constexpr constant long WIDE_SCALE = 16777216L;

/* Recombined in `long`, never in `float`: the low 24 bits are the whole
 * reason the second lane exists. */
inline long wide_value(float low, float high) {
    return (long)high * WIDE_SCALE + (long)low;
}

kernel void acc_keep_flags(device const float* values [[buffer(0)]],
                           device const float* valid [[buffer(1)]],
                           device float* flags [[buffer(2)]],
                           constant uint& count [[buffer(3)]],
                           constant uint& has_valid [[buffer(4)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    bool keep = values[gid] != 0.0f;
    if (has_valid != 0u && valid[gid] == 0.0f) keep = false;
    flags[gid] = keep ? 1.0f : 0.0f;
}

/* The same, four rows at a time, for a mask whose length divides by four.
 * This kernel does nothing but move sixteen bytes in and four out, so the
 * quartered thread count is very nearly a quartered cost. */
kernel void acc_keep_flags4(device const float4* values [[buffer(0)]],
                            device const float4* valid [[buffer(1)]],
                            device float4* flags [[buffer(2)]],
                            constant uint& quads [[buffer(3)]],
                            constant uint& has_valid [[buffer(4)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= quads) return;
    float4 keep = select(float4(0.0f), float4(1.0f), values[gid] != 0.0f);
    if (has_valid != 0u) keep = select(float4(0.0f), keep, valid[gid] != 0.0f);
    flags[gid] = keep;
}

/* The scatter that turns a scanned flag lane into the positions it marks.
 * `offsets` is the exclusive scan, so a set row already knows where it lands
 * and no two threads collide. */
kernel void acc_take_positions(device const float* flags [[buffer(0)]],
                               device const float* offsets [[buffer(1)]],
                               device float* out [[buffer(2)]],
                               constant uint& count [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (flags[gid] == 0.0f) return;
    out[(uint)offsets[gid]] = (float)gid;
}

kernel void acc_span_positions(device float* out [[buffer(0)]],
                               constant uint& count [[buffer(1)]],
                               constant int& first [[buffer(2)]],
                               constant int& step [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = (float)(first + (int)gid * step);
}

/* Four positions at a time, from one multiply: a strided slice is arithmetic
 * and nothing else, so the write width is the whole cost. */
kernel void acc_span_positions4(device float4* out [[buffer(0)]],
                                constant uint& quads [[buffer(1)]],
                                constant int& first [[buffer(2)]],
                                constant int& step [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= quads) return;
    int base = first + (int)(gid << 2u) * step;
    out[gid] = float4((float)base,
                      (float)(base + step),
                      (float)(base + 2 * step),
                      (float)(base + 3 * step));
}

/* Dispatched over the listed positions, not over the rows: dropping ten rows
 * from a million costs ten threads and leaves the rest of the lane alone. */
kernel void acc_mark_positions(device float* flags [[buffer(0)]],
                               device const float* positions [[buffer(1)]],
                               constant uint& count [[buffer(2)]],
                               constant uint& length [[buffer(3)]],
                               constant uint& mark [[buffer(4)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    int at = (int)positions[gid];
    if (at < 0 || at >= (int)length) return;
    flags[at] = (mark != 0u) ? 1.0f : 0.0f;
}

kernel void acc_floor_positions(device float* lane [[buffer(0)]],
                                constant uint& count [[buffer(1)]],
                                constant uint& length [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float at = floor(lane[gid]);
    float last = (float)(length - 1u);
    if (at < 0.0f) at = 0.0f;
    if (at > last) at = last;
    lane[gid] = at;
}

/* The permutation that lifts the appended rows back to `at`. Three straight
 * ranges and no division at all, which is what makes an insert one gather
 * rather than three takes and two concatenations per column. */
kernel void acc_insert_order(device float* lane [[buffer(0)]],
                             constant uint& total [[buffer(1)]],
                             constant uint& held [[buffer(2)]],
                             constant uint& at [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= total) return;
    uint added = total - held;
    if (gid < at) {
        lane[gid] = (float)gid;
        return;
    }
    if (gid < at + added) {
        lane[gid] = (float)(held + (gid - at));
        return;
    }
    lane[gid] = (float)(gid - added);
}

kernel void acc_copy_span(device const float* src [[buffer(0)]],
                          device float* dst [[buffer(1)]],
                          constant uint& count [[buffer(2)]],
                          constant uint& dst_offset [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    dst[dst_offset + gid] = src[gid];
}

/* Membership for a dictionary column, and for anything else whose answer
 * depends only on the level: the host decides once per distinct value, which
 * there are few of, and the rows are a single gather from that verdict lane.
 * `between` over text goes through here too, because the codes are in
 * first-appearance order and comparing them would not be comparing strings. */
kernel void acc_codes_in_set(device const float* codes [[buffer(0)]],
                             device const float* valid [[buffer(1)]],
                             device const float* wanted [[buffer(2)]],
                             device float* out [[buffer(3)]],
                             constant uint& count [[buffer(4)]],
                             constant uint& levels [[buffer(5)]],
                             constant uint& has_valid [[buffer(6)]],
                             constant uint& want_missing [[buffer(7)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (has_valid != 0u && valid[gid] == 0.0f) {
        out[gid] = (want_missing != 0u) ? 1.0f : 0.0f;
        return;
    }
    int code = (int)codes[gid];
    out[gid] = (code >= 0 && code < (int)levels) ? wanted[code] : 0.0f;
}

/* Every thread in the group compares against the same wanted values, so they
 * are staged in threadgroup memory a tile at a time rather than re-read from
 * device memory once per row. The tile loop is uniform across the group —
 * `wanted_count` is a scalar — which is what makes the barriers legal; the
 * out-of-range threads are held with a flag instead of returning early for
 * the same reason. */
kernel void acc_isin_values(device const float* values [[buffer(0)]],
                            device const float* wide [[buffer(1)]],
                            device const float* valid [[buffer(2)]],
                            device const float* wanted_low [[buffer(3)]],
                            device const float* wanted_high [[buffer(4)]],
                            device float* out [[buffer(5)]],
                            constant uint& count [[buffer(6)]],
                            constant uint& wanted_count [[buffer(7)]],
                            constant uint& has_wide [[buffer(8)]],
                            constant uint& has_valid [[buffer(9)]],
                            constant uint& want_missing [[buffer(10)]],
                            uint gid [[thread_position_in_grid]],
                            uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float tile_low[WANTED_TILE];
    threadgroup float tile_high[WANTED_TILE];
    bool live = gid < count;
    bool absent = live && has_valid != 0u && valid[gid] == 0.0f;
    bool present = live && !absent;
    float own_low = present ? values[gid] : 0.0f;
    float own_high = (present && has_wide != 0u) ? wide[gid] : 0.0f;
    bool hit = false;
    for (uint base = 0u; base < wanted_count; base += WANTED_TILE) {
        uint span = min(WANTED_TILE, wanted_count - base);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < span) {
            tile_low[lid] = wanted_low[base + lid];
            tile_high[lid] = wanted_high[base + lid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (present && !hit) {
            for (uint i = 0u; i < span; ++i) {
                bool same = tile_low[i] == own_low;
                if (same && has_wide != 0u) same = tile_high[i] == own_high;
                if (same) {
                    hit = true;
                    break;
                }
            }
        }
    }
    if (!live) return;
    if (absent) {
        out[gid] = (want_missing != 0u) ? 1.0f : 0.0f;
        return;
    }
    out[gid] = hit ? 1.0f : 0.0f;
}

kernel void acc_between(device const float* values [[buffer(0)]],
                        device const float* valid [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        constant uint& count [[buffer(3)]],
                        constant uint& has_valid [[buffer(4)]],
                        constant uint& low_bits [[buffer(5)]],
                        constant uint& high_bits [[buffer(6)]],
                        constant uint& closed [[buffer(7)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (has_valid != 0u && valid[gid] == 0.0f) {
        out[gid] = 0.0f;
        return;
    }
    float v = values[gid];
    bool above = ((closed & 1u) != 0u) ? (v >= as_type<float>(low_bits))
                                       : (v > as_type<float>(low_bits));
    bool below = ((closed & 2u) != 0u) ? (v <= as_type<float>(high_bits))
                                       : (v < as_type<float>(high_bits));
    out[gid] = (above && below) ? 1.0f : 0.0f;
}

/* The bounds arrive as their two halves rather than as floats, so a timestamp
 * range compares at full 48-bit width instead of at the 24 a float32 bound
 * would have rounded them to. */
kernel void acc_between_wide(device const float* values [[buffer(0)]],
                             device const float* wide [[buffer(1)]],
                             device const float* valid [[buffer(2)]],
                             device float* out [[buffer(3)]],
                             constant uint& count [[buffer(4)]],
                             constant uint& has_valid [[buffer(5)]],
                             constant int& low_low [[buffer(6)]],
                             constant int& low_high [[buffer(7)]],
                             constant int& high_low [[buffer(8)]],
                             constant int& high_high [[buffer(9)]],
                             constant uint& closed [[buffer(10)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (has_valid != 0u && valid[gid] == 0.0f) {
        out[gid] = 0.0f;
        return;
    }
    long v = wide_value(values[gid], wide[gid]);
    long low = (long)low_high * WIDE_SCALE + (long)low_low;
    long high = (long)high_high * WIDE_SCALE + (long)high_low;
    bool above = ((closed & 1u) != 0u) ? (v >= low) : (v > low);
    bool below = ((closed & 2u) != 0u) ? (v <= high) : (v < high);
    out[gid] = (above && below) ? 1.0f : 0.0f;
}

kernel void acc_where(device const float* values [[buffer(0)]],
                      device const float* wide [[buffer(1)]],
                      device const float* valid [[buffer(2)]],
                      device const float* cond [[buffer(3)]],
                      device const float* cond_valid [[buffer(4)]],
                      device float* out_values [[buffer(5)]],
                      device float* out_wide [[buffer(6)]],
                      device float* out_valid [[buffer(7)]],
                      constant uint& count [[buffer(8)]],
                      constant uint& has_wide [[buffer(9)]],
                      constant uint& has_valid [[buffer(10)]],
                      constant uint& cond_has_valid [[buffer(11)]],
                      constant uint& out_has_valid [[buffer(12)]],
                      constant uint& invert [[buffer(13)]],
                      constant uint& other_low [[buffer(14)]],
                      constant uint& other_high [[buffer(15)]],
                      constant uint& other_missing [[buffer(16)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    bool keep = cond[gid] != 0.0f;
    if (cond_has_valid != 0u && cond_valid[gid] == 0.0f) keep = false;
    if (invert != 0u) keep = !keep;
    if (keep) {
        out_values[gid] = values[gid];
        if (has_wide != 0u) out_wide[gid] = wide[gid];
        if (out_has_valid != 0u) out_valid[gid] = (has_valid != 0u) ? valid[gid] : 1.0f;
        return;
    }
    out_values[gid] = (other_missing != 0u) ? 0.0f : as_type<float>(other_low);
    if (has_wide != 0u) {
        out_wide[gid] = (other_missing != 0u) ? 0.0f : as_type<float>(other_high);
    }
    if (out_has_valid != 0u) out_valid[gid] = (other_missing != 0u) ? 0.0f : 1.0f;
}

/* Writes into the column's own lanes. Assignment is the one operation here
 * that is not allowed to hand back a new column: a frame's readers are
 * holding the buffers, not the frame. */
kernel void acc_assign(device float* values [[buffer(0)]],
                       device float* wide [[buffer(1)]],
                       device float* valid [[buffer(2)]],
                       device const float* flags [[buffer(3)]],
                       constant uint& count [[buffer(4)]],
                       constant uint& has_wide [[buffer(5)]],
                       constant uint& has_valid [[buffer(6)]],
                       constant uint& low_bits [[buffer(7)]],
                       constant uint& high_bits [[buffer(8)]],
                       constant uint& missing [[buffer(9)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (flags[gid] == 0.0f) return;
    if (missing != 0u) {
        if (has_valid != 0u) valid[gid] = 0.0f;
        return;
    }
    values[gid] = as_type<float>(low_bits);
    if (has_wide != 0u) wide[gid] = as_type<float>(high_bits);
    if (has_valid != 0u) valid[gid] = 1.0f;
}

/* Narrows a flag lane to the rows whose level equals one label, and only ever
 * clears — so chaining one call per level leaves the rows that agree on all
 * of them, which is how a multi-index label is matched without a join. */
kernel void acc_label_match(device const float* values [[buffer(0)]],
                            device const float* wide [[buffer(1)]],
                            device const float* valid [[buffer(2)]],
                            device float* flags [[buffer(3)]],
                            constant uint& count [[buffer(4)]],
                            constant uint& has_wide [[buffer(5)]],
                            constant uint& has_valid [[buffer(6)]],
                            constant uint& low_bits [[buffer(7)]],
                            constant uint& high_bits [[buffer(8)]],
                            constant uint& want_missing [[buffer(9)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    if (flags[gid] == 0.0f) return;
    bool present = !(has_valid != 0u && valid[gid] == 0.0f);
    if (want_missing != 0u) {
        flags[gid] = present ? 0.0f : 1.0f;
        return;
    }
    if (!present) {
        flags[gid] = 0.0f;
        return;
    }
    bool same = values[gid] == as_type<float>(low_bits);
    if (same && has_wide != 0u) same = wide[gid] == as_type<float>(high_bits);
    flags[gid] = same ? 1.0f : 0.0f;
}
"""

var _kernels: dict[str, Kernel] = {}

fn kernel_for(entry: str) -> Kernel {
    let existing = _kernels.get(entry, null)
    if existing is not null { return existing }
    let compiled = Kernel.compile(SOURCE, entry)
    _kernels[entry] = compiled
    return compiled
}

fn flag(value: bool) -> int { return if value { 1 } else { 0 } }

fn u32(value: int) -> int { return value & 0xFFFFFFFF }

#: The three lane accessors every dispatch here binds through.
#:
#: A kernel has to be handed a buffer for every slot it declares even when the
#: lane does not exist, so an absent `wide` or `valid` binds the value lane and
#: a `has_wide` or `has_valid` scalar tells the kernel not to read it. Going
#: through these three rather than reaching for the fields keeps that
#: substitution in one place, where the dtype rule it depends on is written
#: down.
fn value_lane(column: Column) -> Buffer { return column.values }

fn wide_lane(column: Column) -> Buffer { return column.wide ?? column.values }

fn valid_lane(column: Column) -> Buffer { return column.valid ?? column.values }

#: Which of the five shapes a selection is.
#:
#: One enum for both `Pick` and `Labels`, because the shapes are the same
#: whether the subscript names positions or labels — and because `iloc` and
#: `loc` then differ only in how a shape resolves, not in what shapes exist.
pub enum SelectKind {
    #: A single row.
    One,
    #: A list of rows, in the order given, repeats allowed.
    Many,
    #: A half-open range with a step, as Python's slice is.
    Span,
    #: A `Bool` column, one flag per row.
    Mask,
    #: Every row, in order.
    Every,
}

#: A selection by position: one row, a list, a slice with a step, a boolean
#: mask, or everything.
#:
#: Negative positions count from the end, as pandas' do. A `Pick` borrows the
#: mask it was built from and never frees it.
pub class Pick {
    #: Which shape this is.
    pub let kind: SelectKind
    #: The row, for `One`.
    pub let one: int
    #: The rows, for `Many`.
    pub let many: list[int]
    #: First row of a `Span`; `null` is the near end for the step's direction.
    pub let start: int?
    #: One past the last row of a `Span`; `null` is the far end.
    pub let stop: int?
    #: Distance between the rows of a `Span`, negative to walk backwards.
    pub let step: int
    #: The flags, for `Mask`.
    pub let flags: Column?

    #: An empty shell. The named constructors are the ones to reach for.
    pub fn init(self) {
        self.kind = SelectKind.Every
        self.one = 0
        self.many = []
        self.start = null
        self.stop = null
        self.step = 1
        self.flags = null
    }

    #: One row, which `iloc` still returns as a one-row frame. `iat` is the
    #: call that returns the cell.
    pub static fn one_at(position: int) -> Pick {
        let pick = Pick()
        pick.kind = SelectKind.One
        pick.one = position
        return pick
    }

    #: The listed rows, in that order. Repeats are kept, which is what makes
    #: this the primitive an upsampling reindex is built on.
    pub static fn of(positions: list[int]) -> Pick {
        let pick = Pick()
        pick.kind = SelectKind.Many
        pick.many = [row for row in positions]
        return pick
    }

    #: A slice, with Python's bounds: `stop` is excluded, a null end runs to
    #: the edge, and a negative step walks backwards.
    pub static fn span(start: int? = null, stop: int? = null, step: int = 1) -> Pick {
        if step == 0 { throw ValueError("a slice of step 0 never advances") }
        let pick = Pick()
        pick.kind = SelectKind.Span
        pick.start = start
        pick.stop = stop
        pick.step = step
        return pick
    }

    #: The rows a `Bool` column marks. A null flag drops its row, the same way
    #: `Column.compress` treats one.
    pub static fn masked(flags: Column) -> Pick {
        let pick = Pick()
        pick.kind = SelectKind.Mask
        pick.flags = flags
        return pick
    }

    #: Every row, in order — the `:` of a two-axis subscript.
    pub static fn every() -> Pick {
        return Pick()
    }
}

#: A selection by label, in the same five shapes `Pick` has.
#:
#: A label spans every level of the index, so each entry is a list of `Value`
#: one per level; `one` and `of` are the single-level spellings. A label that
#: matches several rows selects all of them, as pandas' `loc` does.
pub class Labels {
    #: Which shape this is.
    pub let kind: SelectKind
    #: The labels, one list of level values apiece, for `One` and `Many`.
    pub let keys: list[list[Value]]
    #: First label of a `Span`; `null` is the start of the index.
    pub let start: Value?
    #: Last label of a `Span` — included, as pandas' label slices are.
    pub let stop: Value?
    #: The flags, for `Mask`.
    pub let flags: Column?

    #: An empty shell. The named constructors are the ones to reach for.
    pub fn init(self) {
        self.kind = SelectKind.Every
        self.keys = []
        self.start = null
        self.stop = null
        self.flags = null
    }

    #: One label on a single-level index.
    pub static fn one_of(label: Value) -> Labels {
        return Labels.tuples([[label]])
    }

    #: Several labels on a single-level index, in the order given.
    pub static fn of(labels: list[Value]) -> Labels {
        return Labels.tuples(labels.map(|label| [label]))
    }

    #: Labels on a multi-index: one list of level values per label.
    pub static fn tuples(labels: list[list[Value]]) -> Labels {
        let picked = Labels()
        picked.kind = if labels.len() == 1 { SelectKind.One } else { SelectKind.Many }
        picked.keys = labels.map(|key| [part for part in key])
        return picked
    }

    #: A label range, both ends included.
    #:
    #: pandas closes the right end of a label slice and leaves a positional one
    #: open, on the grounds that a label the caller wrote down is one they
    #: meant. The ends are located by their first and last occurrence, so a
    #: repeated label still selects a contiguous stretch.
    pub static fn span(start: Value? = null, stop: Value? = null) -> Labels {
        let picked = Labels()
        picked.kind = SelectKind.Span
        picked.start = start
        picked.stop = stop
        return picked
    }

    #: The rows a `Bool` column marks — the same selection `Pick.masked` makes,
    #: spelled for a `loc` call site.
    pub static fn masked(flags: Column) -> Labels {
        let picked = Labels()
        picked.kind = SelectKind.Mask
        picked.flags = flags
        return picked
    }

    #: Every row, in index order.
    pub static fn every() -> Labels {
        return Labels()
    }
}

fn resolve_at(position: int, length: int) -> int {
    let at = if position < 0 { length + position } else { position }
    if at < 0 or at >= length {
        throw IndexError(f"row {position} is outside a frame of {length}")
    }
    return at
}

fn clamp_at(raw: int, length: int, low: int, high: int) -> int {
    var at = if raw < 0 { length + raw } else { raw }
    if at < low { at = low }
    if at > high { at = high }
    return at
}

#: The first row of a slice and how many rows it has, with Python's bounds.
fn span_bounds(start: int?, stop: int?, step: int, length: int) -> (int, int) {
    if step == 0 { throw ValueError("a slice of step 0 never advances") }
    if step > 0 {
        let first = clamp_at(start ?? 0, length, 0, length)
        let limit = clamp_at(stop ?? length, length, 0, length)
        let span = limit - first
        return (first, if span <= 0 { 0 } else { (span + step - 1) // step })
    }
    let first = clamp_at(start ?? length - 1, length, -1, length - 1)
    let limit = if stop is null { -1 } else { clamp_at(stop ?? 0, length, -1, length - 1) }
    let span = first - limit
    let back = -step
    return (first, if span <= 0 { 0 } else { (span + back - 1) // back })
}

#: The positions a strided slice names, written straight onto the device.
fn span_positions(start: int?, stop: int?, step: int, length: int) -> (Buffer, int) {
    let bounds = span_bounds(start, stop, step, length)
    let first = bounds[0]
    let count = bounds[1]
    let lane = Buffer(count)
    if count == 0 { return (lane, 0) }
    if count % 4 == 0 {
        kernel_for("acc_span_positions4").dispatch_async(
            [lane],
            count // 4,
            [count // 4, u32(first), u32(step)]
        )
        return (lane, count)
    }
    kernel_for("acc_span_positions").dispatch_async([lane], count, [count, u32(first), u32(step)])
    return (lane, count)
}

#: A `Bool` column as a lane of 0/1 flags, with a null flag counting as false.
fn keep_flags(mask: Column, length: int) -> Buffer {
    if mask.dtype != Dtype.Bool {
        throw TypeError(f"a selection mask must be bool, not {mask.dtype.name()}")
    }
    if mask.len() != length {
        throw ValueError(f"a mask of {mask.len()} cannot select from {length} rows")
    }
    let flags = Buffer(length)
    if length == 0 { return flags }
    let has_valid = flag(mask.has_validity())
    if length % 4 == 0 {
        kernel_for("acc_keep_flags4").dispatch_async(
            [value_lane(mask), valid_lane(mask), flags],
            length // 4,
            [length // 4, has_valid]
        )
        return flags
    }
    kernel_for("acc_keep_flags").dispatch_async(
        [value_lane(mask), valid_lane(mask), flags],
        length,
        [length, has_valid]
    )
    return flags
}

#: The positions a flag lane marks, and how many there are.
#:
#: One scan and one scatter for the whole frame. Every masked selection here
#: goes through this rather than through `Column.compress` per column, which
#: would flag and scan the same rows once per column.
fn compact_flags(flags: Buffer, length: int) -> (Buffer, int) {
    let scanned = prefix_sum(flags, length)
    let offsets = scanned[0]
    let kept = scanned[1]
    let lane = Buffer(kept)
    if kept > 0 {
        kernel_for("acc_take_positions").dispatch_async([flags, offsets, lane], length, [length])
    }
    offsets.free()
    return (lane, kept)
}

fn mask_positions(mask: Column, length: int) -> (Buffer, int) {
    let flags = keep_flags(mask, length)
    let found = compact_flags(flags, length)
    flags.free()
    return found
}

#: The positions a `Pick` names, as a device lane the caller owns.
fn pick_positions(pick: Pick, length: int) -> (Buffer, int) {
    if pick.kind == SelectKind.Every { return (iota(length), length) }
    if pick.kind == SelectKind.One { return (positions_buffer([resolve_at(pick.one, length)]), 1) }
    if pick.kind == SelectKind.Many {
        let rows = pick.many.map(|row| resolve_at(row, length))
        return (positions_buffer(rows), rows.len())
    }
    if pick.kind == SelectKind.Span {
        return span_positions(pick.start, pick.stop, pick.step, length)
    }
    let flags = pick.flags
    if flags is null { throw ValueError("a masked selection carries no mask") }
    return mask_positions(flags, length)
}

#: The positions a `Pick` names, as a host list.
#:
#: For the column axis only, where the count is the frame's width and a
#: hundred columns is a wide frame. The row axis never comes through here.
fn pick_host(pick: Pick, length: int) -> list[int] {
    if pick.kind == SelectKind.Every { return [at for at in 0..length] }
    if pick.kind == SelectKind.One { return [resolve_at(pick.one, length)] }
    if pick.kind == SelectKind.Many { return pick.many.map(|at| resolve_at(at, length)) }
    if pick.kind == SelectKind.Span {
        let bounds = span_bounds(pick.start, pick.stop, pick.step, length)
        return [bounds[0] + step_of * pick.step for step_of in 0..bounds[1]]
    }
    let flags = pick.flags
    if flags is null { throw ValueError("a masked selection carries no mask") }
    if flags.dtype != Dtype.Bool {
        throw TypeError(f"a selection mask must be bool, not {flags.dtype.name()}")
    }
    if flags.len() != length {
        throw ValueError(f"a mask of {flags.len()} cannot select from {length} columns")
    }
    var out: list[int] = []
    for (at, kept) in flags.to_bools().enumerate() {
        if kept { out.push(at) }
    }
    return out
}

const MATCH_VALUE = 0
const MATCH_MISSING = 1
const MATCH_NONE = 2

#: How to look for `label` in a level, and the bits to look for.
#:
#: A text label that the level's dictionary has never seen answers
#: `MATCH_NONE`, and the caller skips the dispatch entirely: the codes cannot
#: hold a value the table does not name. That short-circuit is the point of
#: resolving the scalar on the host.
fn label_match_bits(level: Column, label: Value) -> (int, int, int) {
    if label.is_na() { return (0, 0, MATCH_MISSING) }
    if level.dtype.is_dictionary() {
        let code = level.level_code(label.to_str())
        if code is null { return (0, 0, MATCH_NONE) }
        return (f32_bits(float(code ?? 0)), 0, MATCH_VALUE)
    }
    if level.dtype.is_wide() {
        let split = wide_pack(label.to_int())
        return (f32_bits(split[0]), f32_bits(split[1]), MATCH_VALUE)
    }
    return (f32_bits(label.to_float()), 0, MATCH_VALUE)
}

#: A flag lane holding 1.0 for the rows whose label is `key`.
#:
#: One dispatch per level over a lane that starts full and is only ever
#: cleared, so a two-level label costs two passes and no intermediate.
fn stored_label_flags(index: Index, key: list[Value]) -> Buffer {
    if key.len() != index.nlevels() {
        throw KeyError(f"a label of {key.len()} parts cannot address {index.nlevels()} levels")
    }
    let length = index.len()
    let flags = Buffer(length)
    if length == 0 { return flags }
    flags.fill(1.0)
    for (at, part) in key.enumerate() {
        let level = index.levels[at]
        let found = label_match_bits(level, part)
        if found[2] == MATCH_NONE {
            flags.fill_zero()
            return flags
        }
        kernel_for("acc_label_match").dispatch_async(
            [value_lane(level), wide_lane(level), valid_lane(level), flags],
            length,
            [
                length,
                flag(level.dtype.is_wide()),
                flag(level.has_validity()),
                found[0],
                found[1],
                flag(found[2] == MATCH_MISSING),
            ]
        )
    }
    return flags
}

#: Where a label sits on a range index, or `null` when it is not on the ramp.
#:
#: Arithmetic, not a lookup: the labels of a range index are not stored, and
#: materialising the ramp to search it would cost a whole column to answer a
#: question one division answers.
fn range_label_position(index: Index, label: Value) -> int? {
    if label.is_na() { return null }
    if label.dtype.is_dictionary() {
        throw KeyError(f"a range index labels rows with integers, not {label.to_str()}")
    }
    let wanted = label.to_int()
    let at = (wanted - index.start) // index.step
    if index.start + at * index.step != wanted { return null }
    if at < 0 or at >= index.len() { return null }
    return at
}

fn key_positions(index: Index, key: list[Value]) -> (Buffer, int) {
    if index.is_range() {
        if key.len() != 1 {
            throw KeyError(f"a label of {key.len()} parts cannot address a range index")
        }
        let at = range_label_position(index, key[0])
        if at is null { return (Buffer(0), 0) }
        return (positions_buffer([at ?? 0]), 1)
    }
    let flags = stored_label_flags(index, key)
    let found = compact_flags(flags, index.len())
    flags.free()
    return found
}

fn label_text(key: list[Value]) -> str {
    if key.len() == 1 { return key[0].to_str() }
    return "(" + ", ".join(key.map(|part| part.to_str())) + ")"
}

#: The lanes stacked end to end into one, which the caller owns.
fn stack_positions(parts: list[Buffer], counts: list[int]) -> (Buffer, int) {
    var total = 0
    for count in counts { total += count }
    let lane = Buffer(total)
    var at = 0
    for (which, part) in parts.enumerate() {
        if counts[which] == 0 { continue }
        kernel_for("acc_copy_span").dispatch_async([part, lane], counts[which], [counts[which], at])
        at += counts[which]
    }
    return (lane, total)
}

fn label_span_positions(index: Index, start: Value?, stop: Value?) -> (Buffer, int) {
    let length = index.len()
    let first = if start is null { 0 } else { edge_position(index, start ?? NA, true) }
    let last = if stop is null { length - 1 } else { edge_position(index, stop ?? NA, false) }
    if first > last { return (Buffer(0), 0) }
    return span_positions(first, last + 1, 1, length)
}

#: The first or last row a slice endpoint names.
#:
#: Two one-element downloads rather than a download of the matches, because a
#: label that repeats a million times still has only two interesting rows.
fn edge_position(index: Index, label: Value, want_first: bool) -> int {
    if index.is_range() {
        let at = range_label_position(index, label)
        if at is null { throw KeyError(f"no row labelled {label.to_str()}") }
        return at ?? 0
    }
    let found = key_positions(index, [label])
    let lane = found[0]
    defer { lane.free() }
    if found[1] == 0 { throw KeyError(f"no row labelled {label.to_str()}") }
    let slot = if want_first { 0 } else { found[1] - 1 }
    return int(lane.download(slot, 1)[0])
}

#: The dtype of a level, `Int64` for the implied labels of a range index.
fn level_dtype_at(index: Index, level: int) -> Dtype {
    if index.is_range() { return Dtype.Int64 }
    return index.levels[level].dtype
}

#: One level's worth of asked-for labels as a column of the level's own dtype.
#:
#: Built at the level's dtype rather than at the `Value`'s so that `locate`
#: probes without a cast; it reconciles the two dictionaries itself, so a text
#: key needs no level table of its own here.
fn label_column(dtype: Dtype, cells: list[Value]) -> Column {
    let nulls = cells.map(|cell| cell.is_na())
    if dtype.is_dictionary() {
        return Column.from_strs(cells.map(|cell| if cell.is_na() {
            null
        } else {
            cell.to_str()
        }), dtype)
    }
    if dtype.is_wide() {
        return Column.from_ints(cells.map(|cell| if cell.is_na() {
            0
        } else {
            cell.to_int()
        }), nulls, dtype)
    }
    if dtype == Dtype.Bool {
        return Column.from_bools(cells.map(|cell| if cell.is_na() {
            false
        } else {
            cell.to_bool()
        }), nulls)
    }
    return Column.from_floats(cells.map(|cell| if cell.is_na() {
        NAN
    } else {
        cell.to_float()
    }), nulls)
}

#: The asked-for labels as an index to probe `index` with.
fn key_index(index: Index, keys: list[list[Value]]) -> Index {
    let levels = index.nlevels()
    for key in keys {
        if key.len() != levels {
            throw KeyError(f"a label of {key.len()} parts cannot address {levels} levels")
        }
    }
    var columns: list[Column] = []
    for at in 0..levels {
        try {
            columns.push(label_column(level_dtype_at(index, at), keys.map(|key| key[at])))
        } catch error: Error {
            for column in columns { column.free() }
            throw error
        }
    }
    return Index.of(columns)
}

#: The rows the listed labels sit at, through one hash table rather than one
#: scan apiece.
#:
#: `IndexLookup` costs one pass over the index to build and then answers every
#: label in a single probe, so asking for `k` labels out of `n` rows is
#: `n + k` rather than the `k * n` a flag scan per label would be. The lookup
#: reports one row per label, which is the whole answer only while no label
#: repeats — `label_positions` checks that before coming here.
#:
#: The misses come back on the host because the caller wrote the labels down
#: on the host to begin with, and because naming the first absent one in the
#: error needs the `Value` anyway. That download is `k` long, not `n`.
fn probe_positions(lookup: IndexLookup, index: Index, keys: list[list[Value]]) -> (Buffer, int) {
    let target = key_index(index, keys)
    defer { target.free() }
    let count = keys.len()
    let lane = lookup.locate(target)
    if count == 0 { return (lane, 0) }
    for (at, found) in lane.download(0, count).enumerate() {
        if found < 0.0 {
            lane.free()
            throw KeyError(f"no row labelled {label_text(keys[at])}")
        }
    }
    return (lane, count)
}

#: The rows the listed labels sit at when the index repeats a label.
#:
#: One flag scan per label, stacked. A repeat has to expand to every row it
#: names — that is what separates `loc` from a lookup — and a table that
#: reports one representative row cannot say how many rows a label covers.
fn scanned_positions(index: Index, keys: list[list[Value]]) -> (Buffer, int) {
    var parts: list[Buffer] = []
    var counts: list[int] = []
    for key in keys {
        let found = key_positions(index, key)
        if found[1] == 0 {
            for part in parts { part.free() }
            found[0].free()
            throw KeyError(f"no row labelled {label_text(key)}")
        }
        parts.push(found[0])
        counts.push(found[1])
    }
    let stacked = stack_positions(parts, counts)
    for part in parts { part.free() }
    return stacked
}

#: The positions a `Labels` names, as a device lane the caller owns.
fn label_positions(index: Index, labels: Labels) -> (Buffer, int) {
    if labels.kind == SelectKind.Every { return (iota(index.len()), index.len()) }
    if labels.kind == SelectKind.Mask {
        let flags = labels.flags
        if flags is null { throw ValueError("a masked selection carries no mask") }
        return mask_positions(flags, index.len())
    }
    if labels.kind == SelectKind.Span {
        return label_span_positions(index, labels.start, labels.stop)
    }
    let lookup = IndexLookup.of(index)
    defer { lookup.free() }
    if not lookup.is_unique() { return scanned_positions(index, labels.keys) }
    return probe_positions(lookup, index, labels.keys)
}

fn gather_frame(frame: DataFrame, lane: Buffer, count: int) -> DataFrame {
    return frame._replace(
        columns: frame.columns.map(|column| column.take(lane, count)),
        index: frame.index.take(lane, count)
    )
}

fn gather_series(series: Series, lane: Buffer, count: int) -> Series {
    return Series(
        series.values.take(lane, count),
        index: series.index.take(lane, count),
        name: series.name
    )
}

fn narrow_columns(frame: DataFrame, at: list[int]) -> DataFrame {
    return frame._replace(
        names: at.map(|which| frame.names[which]),
        columns: at.map(|which| frame.columns[which])
    )
}

#: The rows and columns a position selection names.
#:
#: One `take` per selected column and one for the index, whatever shape the
#: selection was — a scalar, a list, a strided slice and a mask all arrive at
#: the same positions lane. A scalar row still gives a one-row frame, as
#: `df.iloc[[3]]` does rather than `df.iloc[3]`; `iat` is the call that
#: returns a cell.
#:
#: The columns are fresh and the caller owns them.
pub fn iloc(frame: DataFrame, rows: Pick, columns: Pick? = null) -> DataFrame {
    let narrowed = if columns is null {
        frame
    } else {
        narrow_columns(frame, pick_host(columns ?? Pick.every(), frame.width()))
    }
    let found = pick_positions(rows, frame.len())
    let lane = found[0]
    defer { lane.free() }
    return gather_frame(narrowed, lane, found[1])
}

#: The rows a position selection names, for a series.
pub fn iloc_series(series: Series, rows: Pick) -> Series {
    let found = pick_positions(rows, series.len())
    let lane = found[0]
    defer { lane.free() }
    return gather_series(series, lane, found[1])
}

#: The rows a label selection names, and the named columns.
#:
#: A label that matches several rows selects all of them, in index order, and
#: a list of labels concatenates their matches in the order the labels were
#: given. A label with no row raises `KeyError`, which is what separates `loc`
#: from a reindex: `loc` asks for rows it believes are there.
#:
#: The columns are fresh and the caller owns them.
pub fn loc(frame: DataFrame, rows: Labels, columns: list[str]? = null) -> DataFrame {
    let narrowed = if columns is null {
        frame
    } else {
        narrow_columns(frame, (columns ?? []).map(|name| frame.column_index(name)))
    }
    let found = label_positions(frame.index, rows)
    let lane = found[0]
    defer { lane.free() }
    return gather_frame(narrowed, lane, found[1])
}

#: The rows a label selection names, for a series.
pub fn loc_series(series: Series, rows: Labels) -> Series {
    let found = label_positions(series.index, rows)
    let lane = found[0]
    defer { lane.free() }
    return gather_series(series, lane, found[1])
}

#: One cell by position. A device round trip; reading many cells goes through
#: `Column.to_floats` and friends.
pub fn iat(frame: DataFrame, row: int, column: int) -> Value {
    let at = resolve_at(column, frame.width())
    return frame.columns[at].get(resolve_at(row, frame.len()))
}

#: One cell by label. The label must name exactly one row.
pub fn at(frame: DataFrame, label: Value, column: str) -> Value {
    return frame.column(column).get(unique_label_position(frame.index, [label]))
}

#: One cell of a series by position.
pub fn iat_series(series: Series, position: int) -> Value {
    return series.values.get(position)
}

#: One cell of a series by label. The label must name exactly one row.
pub fn at_series(series: Series, label: Value) -> Value {
    return series.values.get(unique_label_position(series.index, [label]))
}

fn unique_label_position(index: Index, key: list[Value]) -> int {
    let found = key_positions(index, key)
    let lane = found[0]
    defer { lane.free() }
    if found[1] == 0 { throw KeyError(f"no row labelled {label_text(key)}") }
    if found[1] > 1 {
        throw KeyError(f"{label_text(key)} labels {found[1]} rows, so it names no single cell")
    }
    return int(lane.download(0, 1)[0])
}

#: The rows a `Bool` column marks.
#:
#: The flags are scanned once for the whole frame and turned into a positions
#: lane, then every column is one `take` off that lane. Calling
#: `Column.compress` per column instead would redo the flagging and the scan
#: once per column over the same mask, which is the same answer for `width`
#: times the dispatches.
pub fn filter_rows(frame: DataFrame, mask: Column) -> DataFrame {
    let found = mask_positions(mask, frame.len())
    let lane = found[0]
    defer { lane.free() }
    return gather_frame(frame, lane, found[1])
}

#: The rows a `Bool` column marks, for a series.
#:
#: `Column.compress` is the one-column spelling of this, and it is not used
#: here: the index has to come off the same selection, and `compress` scans
#: the mask privately and hands back no positions to share. One scan and two
#: gathers is a dispatch cheaper than a compress and a scan.
pub fn filter_rows_series(series: Series, mask: Column) -> Series {
    let found = mask_positions(mask, series.len())
    let lane = found[0]
    defer { lane.free() }
    return gather_series(series, lane, found[1])
}

#: The listed rows, or the listed columns when `axis` is 1.
#:
#: Positions may repeat and may be negative. Selecting columns shares them
#: with `frame` rather than copying, the way `DataFrame._replace` does;
#: selecting rows builds fresh ones.
pub fn take(frame: DataFrame, positions: list[int], axis: int = 0) -> DataFrame {
    if axis == 1 {
        return narrow_columns(frame, positions.map(|at| resolve_at(at, frame.width())))
    }
    if axis != 0 { throw ValueError(f"a frame has axes 0 and 1, not {axis}") }
    return iloc(frame, Pick.of(positions))
}

#: The listed rows of a series.
pub fn take_series(series: Series, positions: list[int]) -> Series {
    return iloc_series(series, Pick.of(positions))
}

fn edge_count(n: int, length: int) -> int {
    if n >= 0 { return if n < length { n } else { length } }
    let kept = length + n
    return if kept > 0 { kept } else { 0 }
}

#: The first `n` rows. A negative `n` drops the last `n` instead, as pandas'.
pub fn head(frame: DataFrame, n: int = 5) -> DataFrame {
    return iloc(frame, Pick.span(0, edge_count(n, frame.len())))
}

#: The last `n` rows. A negative `n` drops the first `n` instead.
pub fn tail(frame: DataFrame, n: int = 5) -> DataFrame {
    let kept = edge_count(n, frame.len())
    return iloc(frame, Pick.span(frame.len() - kept, frame.len()))
}

#: The first `n` rows of a series.
pub fn head_series(series: Series, n: int = 5) -> Series {
    return iloc_series(series, Pick.span(0, edge_count(n, series.len())))
}

#: The last `n` rows of a series.
pub fn tail_series(series: Series, n: int = 5) -> Series {
    let kept = edge_count(n, series.len())
    return iloc_series(series, Pick.span(series.len() - kept, series.len()))
}

#: One row as a one-row frame. Negative positions count from the end.
#:
#: `head` and `tail` are the many-row calls; these three are the ones that
#: keep the row's label, which `iat` does not.
pub fn nth(frame: DataFrame, position: int) -> DataFrame {
    return iloc(frame, Pick.one_at(position))
}

#: The first row as a one-row frame.
pub fn first(frame: DataFrame) -> DataFrame { return nth(frame, 0) }

#: The last row as a one-row frame.
pub fn last(frame: DataFrame) -> DataFrame { return nth(frame, -1) }

fn draw_distinct(generator: Random, length: int, count: int) -> list[int] {
    var seen: dict[int, bool] = {}
    var out: list[int] = []
    while out.len() < count {
        let at = generator.int_range(0, length)
        if seen.get(at, false) { continue }
        seen[at] = true
        out.push(at)
    }
    return out
}

#: How many rows `n` and `frac` ask for.
fn sample_count(length: int, n: int?, frac: float?) -> int {
    if n is not null and frac is not null {
        throw ValueError("sample takes a count or a fraction, not both")
    }
    if frac is not null {
        let share = frac ?? 0.0
        if share < 0.0 { throw ValueError(f"cannot sample a {share} fraction") }
        return int(share * float(length) + 0.5)
    }
    let wanted = n ?? 1
    if wanted < 0 { throw ValueError(f"cannot sample {wanted} rows") }
    return wanted
}

#: The positions a `sample` draws.
#:
#: With replacement the draw is a uniform lane the device fills and floors,
#: so a million-row sample costs two dispatches and no host work at all.
#: Without replacement there is no such shortcut short of a sort: distinctness
#: is a property of the whole draw. What there is instead is a choice of which
#: side to draw — the kept rows, or the dropped ones — so the host does
#: `min(count, length - count)` draws and never more than half the frame.
fn sample_positions(
    length: int,
    n: int?,
    frac: float?,
    replace: bool,
    seed: int?
) -> (Buffer, int) {
    let count = sample_count(length, n, frac)
    if count == 0 { return (Buffer(0), 0) }
    if length == 0 { throw ValueError(f"cannot sample {count} rows from an empty frame") }
    let generator = if seed is null { Random() } else { Random.with_seed(seed ?? 0) }
    if replace {
        let lane = Buffer(count)
        lane.fill_uniform(0.0, float(length), seed ?? generator.next_int())
        kernel_for("acc_floor_positions").dispatch_async([lane], count, [count, length])
        return (lane, count)
    }
    if count > length {
        throw ValueError(f"cannot draw {count} distinct rows from {length} without replacement")
    }
    if count * 2 <= length {
        return (positions_buffer(draw_distinct(generator, length, count)), count)
    }
    let dropped = positions_buffer(draw_distinct(generator, length, length - count))
    defer { dropped.free() }
    let flags = Buffer(length)
    flags.fill(1.0)
    defer { flags.free() }
    kernel_for("acc_mark_positions").dispatch_async(
        [flags, dropped],
        length - count,
        [length - count, length, 0]
    )
    return compact_flags(flags, length)
}

#: `n` rows drawn at random, or a `frac` fraction of them.
#:
#: A drawn sample is in draw order, not index order, which is what pandas
#: does and what makes a sample usable as a shuffle.
pub fn sample(
    frame: DataFrame,
    n: int? = null,
    frac: float? = null,
    replace: bool = false,
    seed: int? = null
) -> DataFrame {
    let found = sample_positions(frame.len(), n, frac, replace, seed)
    let lane = found[0]
    defer { lane.free() }
    return gather_frame(frame, lane, found[1])
}

#: `n` rows of a series drawn at random, or a `frac` fraction of them.
pub fn sample_series(
    series: Series,
    n: int? = null,
    frac: float? = null,
    replace: bool = false,
    seed: int? = null
) -> Series {
    let found = sample_positions(series.len(), n, frac, replace, seed)
    let lane = found[0]
    defer { lane.free() }
    return gather_series(series, lane, found[1])
}

#: A `Bool` column saying, per row, what a per-level verdict says.
#:
#: The host decides once per distinct level and the device gathers the verdict
#: per row. That is what keeps every text predicate here — `isin`, `between`,
#: `filter_names` on the index — off a host walk over the rows: a dictionary
#: column has as many distinct values as it has levels, however tall it is.
fn codes_mask(level: Column, verdict: list[float], want_missing: bool) -> Column {
    let out = Column.allocate(Dtype.Bool, level.len())
    if level.len() == 0 { return out }
    let wanted = Buffer(verdict.len())
    defer { wanted.free() }
    if verdict.len() > 0 { wanted.upload(verdict) }
    kernel_for("acc_codes_in_set").dispatch_async(
        [value_lane(level), valid_lane(level), wanted, out.values],
        level.len(),
        [level.len(), verdict.len(), flag(level.has_validity()), flag(want_missing)]
    )
    return out
}

fn name_kept(name: str, items: list[str]?, like: str?, regex: Regex?) -> bool {
    if items is not null { return (items ?? []).contains(name) }
    if like is not null { return name.contains(like ?? "") }
    let pattern = regex
    if pattern is null { return true }
    return pattern.is_match(name)
}

#: The columns whose name matches, or the rows whose label does.
#:
#: Exactly one of `items`, `like` and `regex` is given, as pandas requires.
#: On `axis` 1 the surviving columns are shared with `frame` rather than
#: copied; on `axis` 0 the rows are gathered and fresh.
#:
#: `axis` 0 needs a text index, because the match is over the label's
#: spelling. It costs one host test per distinct label and one dispatch over
#: the rows, not one test per row.
pub fn filter_names(
    frame: DataFrame,
    items: list[str]? = null,
    like: str? = null,
    regex: str? = null,
    axis: int = 1
) -> DataFrame {
    var given = 0
    if items is not null { given += 1 }
    if like is not null { given += 1 }
    if regex is not null { given += 1 }
    if given != 1 {
        throw ValueError(f"filter takes exactly one of items, like and regex, got {given}")
    }
    let pattern = if regex is null { null } else { Regex(regex ?? "") }
    if axis == 1 {
        if items is not null {
            return narrow_columns(
                frame,
                (items ?? []).filter(|name| frame.has(name)).map(
                    |name| frame.column_index(name)
                )
            )
        }
        var kept: list[int] = []
        for (at, name) in frame.names.enumerate() {
            if name_kept(name, items, like, pattern) { kept.push(at) }
        }
        return narrow_columns(frame, kept)
    }
    if axis != 0 { throw ValueError(f"a frame has axes 0 and 1, not {axis}") }
    if frame.index.is_range() or frame.index.nlevels() != 1 {
        throw TypeError("filtering row labels by text needs a single-level text index")
    }
    let level = frame.index.levels[0]
    if not level.dtype.is_dictionary() {
        throw TypeError(f"cannot match {level.dtype.name()} labels against a name pattern")
    }
    let verdict = (level.levels ?? []).map(|name| if name_kept(name, items, like, pattern) {
        1.0
    } else {
        0.0
    })
    let mask = codes_mask(level, verdict, false)
    defer { mask.free() }
    return filter_rows(frame, mask)
}

fn wants_missing(dtype: Dtype, value: Value) -> bool {
    if value.is_na() { return true }
    if dtype.is_dictionary() { return false }
    return is_nan(value.to_float())
}

#: True where a row holds one of `values`.
#:
#: A dictionary column answers off its levels, so a text `isin` costs one host
#: test per distinct string however tall the column is. Everything else stages
#: the wanted values in threadgroup memory and compares in one pass: every
#: thread in a group reads the same wanted lane, so reading it from device
#: memory once per row would be the whole cost of the kernel.
#:
#: `NA` in `values` matches the missing rows, and so does a NaN — pandas
#: treats `isin([nan])` as a null test, which is the only reading that is ever
#: useful, since NaN is not equal to itself.
pub fn isin(column: Column, values: list[Value]) -> Column {
    var want_missing = false
    for value in values {
        if wants_missing(column.dtype, value) { want_missing = true }
    }
    if column.dtype.is_dictionary() {
        var wanted: dict[str, bool] = {}
        for value in values {
            if not value.is_na() { wanted[value.to_str()] = true }
        }
        let verdict = (column.levels ?? []).map(|name| if wanted.get(name, false) {
            1.0
        } else {
            0.0
        })
        return codes_mask(column, verdict, want_missing)
    }
    var low: list[float] = []
    var high: list[float] = []
    for value in values {
        if wants_missing(column.dtype, value) { continue }
        if column.dtype.is_wide() {
            let split = wide_pack(value.to_int())
            low.push(split[0])
            high.push(split[1])
            continue
        }
        low.push(value.to_float())
        high.push(0.0)
    }
    let out = Column.allocate(Dtype.Bool, column.len())
    if column.len() == 0 { return out }
    let wanted_low = Buffer(low.len())
    let wanted_high = Buffer(high.len())
    defer {
        wanted_low.free()
        wanted_high.free()
    }
    if low.len() > 0 {
        wanted_low.upload(low)
        wanted_high.upload(high)
    }
    let groups = (column.len() + WANTED_TILE - 1) // WANTED_TILE
    kernel_for("acc_isin_values").dispatch_async(
        [
            value_lane(column),
            wide_lane(column),
            valid_lane(column),
            wanted_low,
            wanted_high,
            out.values,
        ],
        groups * WANTED_TILE,
        [
            column.len(),
            low.len(),
            flag(column.dtype.is_wide()),
            flag(column.has_validity()),
            flag(want_missing),
        ],
        group_size: WANTED_TILE
    )
    return out
}

#: A `Bool` column per column of the frame, true where a row holds one of
#: `values`.
#:
#: The index is the frame's own, not a copy, which is what `_replace` does
#: with anything an operation did not change — so freeing the answer frees the
#: index out from under the frame it came from.
pub fn frame_isin(frame: DataFrame, values: list[Value]) -> DataFrame {
    return frame._replace(columns: frame.columns.map(|column| isin(column, values)))
}

fn closed_bits(inclusive: str) -> int {
    return match inclusive {
        "both" => 3,
        "left" => 1,
        "right" => 2,
        "neither" => 0,
        _ => throw ValueError(f"inclusive is both, left, right or neither, not {inclusive}"),
    }
}

#: True where a row falls between `low` and `high`.
#:
#: `inclusive` is pandas' — `both`, `left`, `right` or `neither`. A missing row
#: is false, whichever end it would have fallen on.
#:
#: Text compares by its spelling, resolved level by level on the host: the
#: dictionary codes are in first-appearance order, so comparing them would
#: order the strings by when they were first seen rather than alphabetically.
#: An `Int64` or `Datetime` range compares at the full 48 bits, because the
#: bounds cross to the device as their two halves rather than as one float.
pub fn between(column: Column, low: Value, high: Value, inclusive: str = "both") -> Column {
    let closed = closed_bits(inclusive)
    if low.is_na() or high.is_na() {
        throw ValueError("a between range needs two bounds, not NA")
    }
    if column.dtype.is_dictionary() {
        let after = low.to_str()
        let before = high.to_str()
        var verdict: list[float] = []
        for name in column.levels ?? [] {
            let above = if closed & 1 != 0 { name >= after } else { name > after }
            let below = if closed & 2 != 0 { name <= before } else { name < before }
            verdict.push(if above and below { 1.0 } else { 0.0 })
        }
        return codes_mask(column, verdict, false)
    }
    let out = Column.allocate(Dtype.Bool, column.len())
    if column.len() == 0 { return out }
    if column.dtype.is_wide() {
        let after = wide_pack(low.to_int())
        let before = wide_pack(high.to_int())
        kernel_for("acc_between_wide").dispatch_async(
            [value_lane(column), wide_lane(column), valid_lane(column), out.values],
            column.len(),
            [
                column.len(),
                flag(column.has_validity()),
                u32(int(after[0])),
                u32(int(after[1])),
                u32(int(before[0])),
                u32(int(before[1])),
                closed,
            ]
        )
        return out
    }
    kernel_for("acc_between").dispatch_async(
        [value_lane(column), valid_lane(column), out.values],
        column.len(),
        [
            column.len(),
            flag(column.has_validity()),
            f32_bits(low.to_float()),
            f32_bits(high.to_float()),
            closed,
        ]
    )
    return out
}

#: The bits of `value` as this column would store it, interning text into
#: `column`'s dictionary when it is not there yet.
fn scalar_bits(column: Column, value: Value) -> (int, int) {
    if value.is_na() { return (0, 0) }
    if column.dtype.is_dictionary() {
        return (f32_bits(float(column.intern_level(value.to_str()))), 0)
    }
    if column.dtype.is_wide() {
        let split = wide_pack(value.to_int())
        return (f32_bits(split[0]), f32_bits(split[1]))
    }
    return (f32_bits(value.to_float()), 0)
}

fn blend(column: Column, cond: Column, other: Value, invert: bool) -> Column {
    if cond.dtype != Dtype.Bool {
        throw TypeError(f"a condition must be bool, not {cond.dtype.name()}")
    }
    if cond.len() != column.len() {
        throw ValueError(f"a condition of {cond.len()} cannot cover {column.len()} rows")
    }
    let out = Column.allocate(
        column.dtype,
        column.len(),
        with_validity: column.has_validity() or other.is_na()
    )
    if column.dtype.is_dictionary() {
        out.set_levels([name for name in column.levels ?? []])
    }
    let bits = scalar_bits(out, other)
    if column.len() == 0 { return out }
    kernel_for("acc_where").dispatch_async(
        [
            value_lane(column),
            wide_lane(column),
            valid_lane(column),
            value_lane(cond),
            valid_lane(cond),
            out.values,
            wide_lane(out),
            valid_lane(out),
        ],
        column.len(),
        [
            column.len(),
            flag(column.dtype.is_wide()),
            flag(column.has_validity()),
            flag(cond.has_validity()),
            flag(out.has_validity()),
            flag(invert),
            bits[0],
            bits[1],
            flag(other.is_na()),
        ]
    )
    return out
}

#: The column's own value where `cond` is true, `other` everywhere else.
#:
#: A null condition counts as false and takes `other`, which is what makes
#: `where` usable against a mask that came from a column with nulls in it.
pub fn where(column: Column, cond: Column, other: Value = NA) -> Column {
    return blend(column, cond, other, false)
}

#: `other` where `cond` is true, the column's own value everywhere else — the
#: complement of `where`.
pub fn mask(column: Column, cond: Column, other: Value = NA) -> Column {
    return blend(column, cond, other, true)
}

#: `where` over every column, with one row condition shared by all of them.
#:
#: The index is shared with `frame`, as `frame_isin`'s is.
pub fn frame_where(frame: DataFrame, cond: Column, other: Value = NA) -> DataFrame {
    return frame._replace(columns: frame.columns.map(|column| where(column, cond, other)))
}

#: `mask` over every column, with one row condition shared by all of them.
#:
#: The index is shared with `frame`, as `frame_isin`'s is.
pub fn frame_mask(frame: DataFrame, cond: Column, other: Value = NA) -> DataFrame {
    return frame._replace(columns: frame.columns.map(|column| mask(column, cond, other)))
}

#: One row of a frame, as a predicate sees it.
#:
#: The lanes are downloaded once, when the view is built, and the view is then
#: pointed at each row in turn rather than rebuilt — a frame of a million rows
#: allocates one of these, not a million. That still leaves a boxed `Value`
#: per cell the predicate touches, which is why `query` is the slow path.
pub class Row {
    #: Which row the view is pointed at. `query` moves it; a predicate reads it.
    pub var at: int
    var lookup: dict[str, int]
    var dtypes: list[Dtype]
    var numbers: list[list[float]]
    var integers: list[list[int]]
    var texts: list[list[str?]]
    var absent: list[list[bool]]

    #: An empty view. `over` is the constructor.
    pub fn init(self) {
        self.at = 0
        self.lookup = {}
        self.dtypes = []
        self.numbers = []
        self.integers = []
        self.texts = []
        self.absent = []
    }

    #: A view over `frame`, downloading each column once.
    #:
    #: One round trip per column, not per row. A column with no validity lane
    #: contributes no missing list at all, which is the difference between one
    #: host allocation per nullable column and one per column.
    pub static fn over(frame: DataFrame) -> Row {
        let row = Row()
        for (at, name) in frame.names.enumerate() { row.lookup[name] = at }
        row.dtypes = frame.dtypes()
        for column in frame.columns {
            row.absent.push(if column.has_validity() { column.null_mask() } else { [] })
            if column.dtype.is_dictionary() {
                row.texts.push(column.to_strs())
                row.numbers.push([])
                row.integers.push([])
                continue
            }
            row.texts.push([])
            if column.dtype.is_wide() {
                row.integers.push(column.to_ints())
                row.numbers.push([])
                continue
            }
            row.integers.push([])
            row.numbers.push(column.to_floats())
        }
        return row
    }

    #: How a named column is stored.
    pub fn dtype(self, name: str) -> Dtype { return self.dtypes[self.slot(name)] }

    #: Whether the cell in a named column is missing.
    pub fn is_na(self, name: str) -> bool {
        let at = self.slot(name)
        let flags = self.absent[at]
        if flags.len() > 0 and flags[self.at] { return true }
        if self.dtypes[at].is_dictionary() { return self.texts[at][self.at] is null }
        if self.dtypes[at].is_wide() { return false }
        return is_nan(self.numbers[at][self.at])
    }

    #: The cell in a named column as a float; NaN when it is missing.
    pub fn number(self, name: str) -> float {
        let at = self.slot(name)
        if self.is_na(name) { return NAN }
        if self.dtypes[at].is_dictionary() {
            throw TypeError(f"{name} is {self.dtypes[at].name()} and has no numeric payload")
        }
        if self.dtypes[at].is_wide() { return float(self.integers[at][self.at]) }
        return self.numbers[at][self.at]
    }

    #: The cell in a named column as an integer.
    pub fn integer(self, name: str) -> int {
        let at = self.slot(name)
        if self.dtypes[at].is_wide() { return self.integers[at][self.at] }
        return int(self.number(name))
    }

    #: The cell in a named column as text, `null` when it is missing.
    pub fn text(self, name: str) -> str? {
        let at = self.slot(name)
        if self.dtypes[at].is_dictionary() { return self.texts[at][self.at] }
        if self.is_na(name) { return null }
        return self.value(name).to_str()
    }

    #: The cell in a named column, boxed.
    pub fn value(self, name: str) -> Value {
        let at = self.slot(name)
        let dtype = self.dtypes[at]
        if self.is_na(name) { return Value.null_of(dtype) }
        if dtype.is_dictionary() { return Value.of_str(self.texts[at][self.at] ?? "", dtype) }
        if dtype.is_wide() { return Value(dtype, integer: self.integers[at][self.at]) }
        if dtype == Dtype.Bool { return Value.of_bool(self.numbers[at][self.at] != 0.0) }
        return Value.of_float(self.numbers[at][self.at])
    }

    fn slot(self, name: str) -> int {
        let at = self.lookup.get(name, -1)
        if at < 0 { throw KeyError(f"no column named {name}") }
        return at
    }
}

#: A `Bool` column holding what `predicate` said about each row.
#:
#: One download per column, then one host call per row. That per-row call is
#: the cost the vectorised path does not pay: a mask built from `arith.jai`
#: comparisons is a dispatch per comparison and nothing per row, so a
#: predicate that could have been written as one belongs there instead. This
#: is here for the ones that could not.
pub fn query_flags(frame: DataFrame, predicate: fn(Row) -> bool) -> Column {
    let row = Row.over(frame)
    var flags: list[bool] = []
    for at in 0..frame.len() {
        row.at = at
        flags.push(predicate(row))
    }
    return Column.from_bools(flags)
}

#: The rows a predicate accepts.
#:
#: The callback runs once per row on the host. `query_mask` is the same
#: selection driven by a `Bool` column and is the one to reach for whenever
#: the predicate can be written as a comparison.
pub fn query(frame: DataFrame, predicate: fn(Row) -> bool) -> DataFrame {
    let mask = query_flags(frame, predicate)
    defer { mask.free() }
    return filter_rows(frame, mask)
}

#: The rows a `Bool` column marks — the vectorised counterpart of `query`.
#:
#: Named so that the fast path is findable from the slow one; it is
#: `filter_rows`, which is what a mask selection has been all along.
pub fn query_mask(frame: DataFrame, mask: Column) -> DataFrame {
    return filter_rows(frame, mask)
}

fn selection_flags(length: int, lane: Buffer, count: int) -> Buffer {
    let flags = Buffer(length)
    if length == 0 { return flags }
    flags.fill_zero()
    if count > 0 {
        kernel_for("acc_mark_positions").dispatch_async([flags, lane], count, [count, length, 1])
    }
    return flags
}

fn full_flags(length: int) -> Buffer {
    let flags = Buffer(length)
    if length > 0 { flags.fill(1.0) }
    return flags
}

#: A flag lane for the rows a `Pick` names.
#:
#: Assignment wants flags, not positions: it writes in place, so the row a
#: thread owns is the row it was given and there is nothing to compact. A
#: masked selection is therefore one dispatch, with neither the scan nor the
#: scatter a `take` would have needed.
fn assignment_flags(pick: Pick, length: int) -> Buffer {
    if pick.kind == SelectKind.Every { return full_flags(length) }
    if pick.kind == SelectKind.Mask {
        let flags = pick.flags
        if flags is null { throw ValueError("a masked selection carries no mask") }
        return keep_flags(flags, length)
    }
    let found = pick_positions(pick, length)
    let lane = found[0]
    defer { lane.free() }
    return selection_flags(length, lane, found[1])
}

fn label_assignment_flags(index: Index, labels: Labels) -> Buffer {
    if labels.kind == SelectKind.Every { return full_flags(index.len()) }
    if labels.kind == SelectKind.Mask {
        let flags = labels.flags
        if flags is null { throw ValueError("a masked selection carries no mask") }
        return keep_flags(flags, index.len())
    }
    let found = label_positions(index, labels)
    let lane = found[0]
    defer { lane.free() }
    return selection_flags(index.len(), lane, found[1])
}

fn assign_into(column: Column, flags: Buffer, value: Value) -> void {
    if column.len() == 0 { return }
    if value.is_na() { let _ = column.materialize_validity() }
    let bits = scalar_bits(column, value)
    kernel_for("acc_assign").dispatch_async(
        [value_lane(column), wide_lane(column), valid_lane(column), flags],
        column.len(),
        [
            column.len(),
            flag(column.dtype.is_wide()),
            flag(column.has_validity()),
            bits[0],
            bits[1],
            flag(value.is_na()),
        ]
    )
}

fn assignment_targets(frame: DataFrame, columns: list[str]?) -> list[Column] {
    if columns is null { return frame.columns }
    return (columns ?? []).map(|name| frame.column(name))
}

#: Write `value` into the selected rows of the selected columns, in place.
#:
#: In place is the whole point: a frame's readers hold the column buffers, not
#: the frame, so an assignment that built new columns would be invisible to
#: them. Every frame sharing these columns sees the write.
#:
#: A scalar row is one upload and no dispatch. Anything wider is one dispatch
#: per target column over a flag lane built once.
pub fn set_iloc(frame: DataFrame, rows: Pick, value: Value, columns: list[str]? = null) -> void {
    let targets = assignment_targets(frame, columns)
    if rows.kind == SelectKind.One {
        let at = resolve_at(rows.one, frame.len())
        for column in targets { column.set(at, value) }
        return
    }
    let flags = assignment_flags(rows, frame.len())
    defer { flags.free() }
    for column in targets { assign_into(column, flags, value) }
}

#: Write `value` into the rows a label selection names, in place.
pub fn set_loc(frame: DataFrame, rows: Labels, value: Value, columns: list[str]? = null) -> void {
    let targets = assignment_targets(frame, columns)
    let flags = label_assignment_flags(frame.index, rows)
    defer { flags.free() }
    for column in targets { assign_into(column, flags, value) }
}

#: Put `values` in the named column's place, and hand back what was there.
#:
#: The frame adopts `values`; the displaced column is returned rather than
#: freed, because `DataFrame._replace` shares columns and this frame is
#: probably not the only one holding it. Free it when you know it is.
pub fn set_column_values(frame: DataFrame, name: str, values: Column) -> Column {
    let at = frame.column_index(name)
    if values.len() != frame.len() {
        throw ValueError(f"a column of {values.len()} rows does not fit {frame.len()}")
    }
    let displaced = frame.columns[at]
    frame.columns[at] = values
    return displaced
}

fn insert_order(total: int, held: int, at: int) -> Buffer {
    let order = Buffer(total)
    if total > 0 {
        kernel_for("acc_insert_order").dispatch_async([order], total, [total, held, at])
    }
    return order
}

fn insert_index(frame: DataFrame, rows: DataFrame, order: Buffer, total: int) -> Index {
    if frame.index.is_range() and rows.index.is_range() { return Index.range(total) }
    let left = frame.index.materialize()
    let right = rows.index.materialize()
    defer {
        for level in left { level.free() }
        for level in right { level.free() }
    }
    if left.len() != right.len() {
        throw ValueError(
            f"an index of {right.len()} levels cannot be inserted into one of {left.len()}"
        )
    }
    var levels: list[Column] = []
    for (at, level) in left.enumerate() {
        let stacked = concat_columns([level, right[at]])
        levels.push(stacked.take(order, total))
        stacked.free()
    }
    return Index.of(levels, [name for name in frame.index.names])
}

#: `rows` spliced in at `position`, defaulting to the end.
#:
#: The parts are stacked and then permuted, rather than the frame being cut in
#: two and reassembled: a cut would be two takes and a concatenation per
#: column, and the permutation is one gather off a lane that costs no division
#: to build. Columns are matched by name, so `rows` may carry them in any
#: order. Negative positions count from the end.
pub fn insert_rows(frame: DataFrame, rows: DataFrame, position: int? = null) -> DataFrame {
    if rows.width() != frame.width() {
        throw ValueError(f"{rows.width()} columns cannot be inserted into {frame.width()}")
    }
    let held = frame.len()
    let total = held + rows.len()
    if total > MAX_ROWS {
        throw ValueError(f"{total} rows exceeds the {MAX_ROWS} a float32 position lane addresses")
    }
    let at = clamp_at(position ?? held, held, 0, held)
    let order = insert_order(total, held, at)
    defer { order.free() }
    var columns: list[Column] = []
    for (which, name) in frame.names.enumerate() {
        let stacked = concat_columns([frame.columns[which], rows.column(name)])
        columns.push(stacked.take(order, total))
        stacked.free()
    }
    return frame._replace(columns: columns, index: insert_index(frame, rows, order, total))
}

#: The frame without the listed rows.
#:
#: Dispatched over the dropped positions and then compacted, so dropping ten
#: rows from a million is ten threads plus the scan every filter pays. A
#: position listed twice is dropped once.
pub fn drop_rows(frame: DataFrame, positions: list[int]) -> DataFrame {
    let length = frame.len()
    let dropped = positions_buffer(positions.map(|at| resolve_at(at, length)))
    defer { dropped.free() }
    let flags = full_flags(length)
    defer { flags.free() }
    if positions.len() > 0 {
        kernel_for("acc_mark_positions").dispatch_async(
            [flags, dropped],
            positions.len(),
            [positions.len(), length, 0]
        )
    }
    let found = compact_flags(flags, length)
    let lane = found[0]
    defer { lane.free() }
    return gather_frame(frame, lane, found[1])
}

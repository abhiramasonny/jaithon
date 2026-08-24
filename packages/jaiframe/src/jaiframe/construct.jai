#: Building frames from host values, and reshaping their column list.
#:
#: Two jobs in one module because they are the two halves of the same one:
#: getting host data onto the device, and settling what the resulting frame's
#: columns are called and what order they are in. Neither reads a row's values
#: once they are down there — that is every other module's business.
#:
#: The host side of construction is a loop over boxed values, because that is
#: what a `list[list[any]]` is and something has to unbox it. What it must
#: never be is a loop over dispatches: every constructor here builds one host
#: `list[float]` per lane and hands the whole lane to `Buffer.upload` in one
#: call. Reaching for `Column.set` a row at a time instead would be a device
#: round trip per cell, and a hundred thousand of those is the difference
#: between construction that is free and construction that is the program.
#:
#: Dtype inference and the null sentinels live here too, since they are the
#: questions a host value has to answer before it can become a slot. A value is
#: missing when it is `null`, a NaN, the empty string, or one of the
#: `na_values` the caller named.
#:
#: The column-list edits — `insert`, `rename`, `reorder` and the rest — are
#: free functions rather than methods, because a frame is immutable here:
#: every one of them builds a new `DataFrame` over the columns that did not
#: change. Nothing copies a lane that a relabel could have shared, so read the
#: ownership note on each one before freeing both halves.
#:
#: ```
#: let frame = frame_from_dict({"city": ["austin", "dallas"], "high": [38.0, 41.0]})
#: defer { frame.free() }
#: frame.dtypes()                      # [str, float64]
#: let wider = add_prefix(frame, "tx_")
#: wider.names                         # ["tx_city", "tx_high"]
#: ```
from std.gpu import Buffer, Kernel
from std.math import NAN, is_nan
from .column import Column, DataFrame, Dtype, Index, Series, Value
from .column import unify_levels, wide_pack

const USAGE_GROUP = 256
const DIFF_GROUP = 256
const DIFF_GROUPS = 256

let SOURCE = """#include <metal_stdlib>
using namespace metal;

constexpr constant uint USAGE_GROUP = 256;
constexpr constant uint DIFF_GROUP = 256;
constexpr constant uint DIFF_GROUPS = 256;

/* Two NaNs in the same slot are the same table cell. `equals` asks whether two
 * frames hold the same data, not whether their arithmetic agrees, and `x != x`
 * would otherwise make a column of NaNs unequal to itself. */
inline bool slot_equal(float a, float b) {
    return a == b || (isnan(a) && isnan(b));
}

/* Which dictionary codes a present row actually uses.
 *
 * Neighbouring threads collide on the same slots — a column of ten million
 * rows over eight levels would write ten million times into eight addresses —
 * so the marks are staged in threadgroup memory and merged out once per group,
 * which turns that into eight stores a group. Levels past the staging array go
 * straight to global; `levels` is a dispatch constant, so the branch is
 * uniform and the barriers below it are reached by every thread or by none.
 *
 * The stores race, and are meant to: every writer writes the same 1.0f. */
kernel void frame_level_usage(device const float* codes [[buffer(0)]],
                              device const float* valid [[buffer(1)]],
                              device float* used [[buffer(2)]],
                              constant uint& count [[buffer(3)]],
                              constant uint& levels [[buffer(4)]],
                              constant uint& has_valid [[buffer(5)]],
                              uint gid [[thread_position_in_grid]],
                              uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float seen[USAGE_GROUP];
    bool staged = levels <= USAGE_GROUP;
    if (staged) {
        for (uint at = lid; at < levels; at += USAGE_GROUP) seen[at] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (gid < count) {
        bool present = (has_valid == 0u) || valid[gid] != 0.0f;
        int code = present ? (int)codes[gid] : -1;
        if (code >= 0 && code < (int)levels) {
            if (staged) {
                seen[code] = 1.0f;
            } else {
                used[code] = 1.0f;
            }
        }
    }
    if (!staged) return;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint at = lid; at < levels; at += USAGE_GROUP) {
        if (seen[at] != 0.0f) used[at] = 1.0f;
    }
}

/* Codes through a level mapping, where a negative entry means the level is
 * gone and the row with it. Writing the validity lane here rather than copying
 * it afterwards is what makes dropping a category one dispatch. */
kernel void frame_recode(device const float* src [[buffer(0)]],
                         device const float* src_valid [[buffer(1)]],
                         device const float* mapping [[buffer(2)]],
                         device float* dst [[buffer(3)]],
                         device float* dst_valid [[buffer(4)]],
                         constant uint& count [[buffer(5)]],
                         constant uint& levels [[buffer(6)]],
                         constant uint& has_valid [[buffer(7)]],
                         constant uint& write_valid [[buffer(8)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    bool present = (has_valid == 0u) || src_valid[gid] != 0.0f;
    int code = (int)src[gid];
    float mapped = (code >= 0 && code < (int)levels) ? mapping[code] : -1.0f;
    bool keep = present && mapped >= 0.0f;
    dst[gid] = keep ? mapped : 0.0f;
    if (write_valid != 0u) dst_valid[gid] = keep ? 1.0f : 0.0f;
}

/* How many rows two columns disagree on.
 *
 * A fixed grid striding the columns and one threadgroup fold at the end, the
 * shape `col_count_nulls` uses and for the same reason: the fold costs eight
 * barriers, and paying that per row rather than per group would make the
 * comparison slower than the read it is comparing. */
kernel void frame_diff_count(device const float* a_values [[buffer(0)]],
                             device const float* a_wide [[buffer(1)]],
                             device const float* a_valid [[buffer(2)]],
                             device const float* b_values [[buffer(3)]],
                             device const float* b_wide [[buffer(4)]],
                             device const float* b_valid [[buffer(5)]],
                             device float* partials [[buffer(6)]],
                             constant uint& count [[buffer(7)]],
                             constant uint& has_wide [[buffer(8)]],
                             constant uint& a_has_valid [[buffer(9)]],
                             constant uint& b_has_valid [[buffer(10)]],
                             uint gid [[thread_position_in_grid]],
                             uint lid [[thread_index_in_threadgroup]],
                             uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[DIFF_GROUP];
    float total = 0.0f;
    for (uint i = gid; i < count; i += DIFF_GROUP * DIFF_GROUPS) {
        bool a_null = (a_has_valid != 0u) && a_valid[i] == 0.0f;
        bool b_null = (b_has_valid != 0u) && b_valid[i] == 0.0f;
        if (a_null || b_null) {
            if (a_null != b_null) total += 1.0f;
            continue;
        }
        bool same = slot_equal(a_values[i], b_values[i]);
        if (same && has_wide != 0u) same = a_wide[i] == b_wide[i];
        if (!same) total += 1.0f;
    }
    scratch[lid] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = DIFF_GROUP / 2u; stride > 0u; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) partials[wid] = scratch[0];
}

/* The same fold four rows at a time, for one lane a side and no mask on
 * either — the shape of a plain float or bool column, and the one where the
 * comparison is nothing but bandwidth. Sixteen bytes a load instead of four.
 * The compare stays scalar because it is free next to the read. */
kernel void frame_diff_count4(device const float4* a_values [[buffer(0)]],
                              device const float4* b_values [[buffer(1)]],
                              device float* partials [[buffer(2)]],
                              constant uint& quads [[buffer(3)]],
                              uint gid [[thread_position_in_grid]],
                              uint lid [[thread_index_in_threadgroup]],
                              uint wid [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[DIFF_GROUP];
    float total = 0.0f;
    for (uint i = gid; i < quads; i += DIFF_GROUP * DIFF_GROUPS) {
        float4 a = a_values[i];
        float4 b = b_values[i];
        int agree = (slot_equal(a.x, b.x) ? 1 : 0) + (slot_equal(a.y, b.y) ? 1 : 0)
                  + (slot_equal(a.z, b.z) ? 1 : 0) + (slot_equal(a.w, b.w) ? 1 : 0);
        total += (float)(4 - agree);
    }
    scratch[lid] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = DIFF_GROUP / 2u; stride > 0u; stride >>= 1) {
        if (lid < stride) scratch[lid] += scratch[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) partials[wid] = scratch[0];
}
"""

var _kernels: dict[str, Kernel] = {}

fn compiled(entry: str) -> Kernel {
    let existing = _kernels.get(entry, null)
    if existing is not null { return existing }
    let kernel = Kernel.compile(SOURCE, entry)
    _kernels[entry] = kernel
    return kernel
}

fn flag(value: bool) -> int { return if value { 1 } else { 0 } }

#: The dtype a list of host values wants.
#:
#: Missing entries do not vote, so a column of numbers with a gap in it is
#: still numeric. What is left is promoted along one short ladder: all `Bool`
#: stays `Bool`, `Bool` beside `Int64` is `Int64`, anything with a float in it
#: is `Float64`, and every other mixture is `Str` — which is this package's
#: stand-in for pandas' `object`, and is why `[1, "a"]` infers as text rather
#: than raising.
#:
#: A `Value` entry votes with its own dtype, which is the only way host data
#: can ask for `Datetime`. A count of milliseconds is indistinguishable from
#: any other large integer, so nothing here guesses at one.
#:
#: Nothing to vote gives `Float64`, matching what pandas hands back for an
#: all-null column.
pub fn infer_dtype(
    values: list[any],
    na_values: list[str]? = null,
    keep_default_na: bool = true
) -> Dtype {
    return inferred(values, na_lookup(na_values), keep_default_na)
}

#: Whether a host value stands for a missing cell.
#:
#: `null` always is: there is nothing else it could mean. A NaN always is too,
#: because a float lane spells missing that way and a "present NaN" is a
#: distinction no reader of the column could observe. The empty string and the
#: `na_values` words are conventions rather than facts, so `keep_default_na`
#: turns the first off, the way pandas' reader does.
pub fn is_na_value(value: any, na_values: list[str]? = null, keep_default_na: bool = true) -> bool {
    return is_missing(value, na_lookup(na_values), keep_default_na)
}

#: One column from host values, inferring the dtype unless told.
#:
#: This is the bulk upload path every constructor in this module goes through.
#: The loop over `values` runs on the host because that is where the values
#: are, but it produces a whole `list[float]` per lane and each lane reaches
#: the device in a single `Buffer.upload`. Two passes, then: one to settle the
#: dtype and one to fill the lanes. Settling it first is what lets the second
#: pass write into the lanes the dtype actually has.
pub fn column_from_values(
    values: list[any],
    dtype: Dtype? = null,
    na_values: list[str]? = null,
    keep_default_na: bool = true
) -> Column {
    let words = na_lookup(na_values)
    let target = dtype ?? inferred(values, words, keep_default_na)
    if target.is_dictionary() { return text_column(values, target, words, keep_default_na) }
    if target.is_wide() { return wide_column(values, target, words, keep_default_na) }
    return plain_column(values, target, words, keep_default_na)
}

#: An index over host labels, one level.
#:
#: The labels go through the same inference the values do, so `["a", "b"]`
#: labels a frame with text and `[10, 20]` labels it with integers.
pub fn index_from_list(
    labels: list[any],
    name: str? = null,
    dtype: Dtype? = null,
    na_values: list[str]? = null,
    keep_default_na: bool = true
) -> Index {
    let level = column_from_values(labels, dtype, na_values, keep_default_na)
    let names: list[str?] = [name]
    return Index.of([level], names)
}

#: A frame over columns that are already on the device.
#:
#: The frame adopts them; nothing is copied and nothing is inferred. This is
#: the constructor a module that has just produced its columns wants, and the
#: one the others end at.
pub fn frame_from_columns(
    names: list[str],
    columns: list[Column],
    index: Index? = null
) -> DataFrame {
    return DataFrame(names, columns, index: index)
}

#: A frame from host rows, one list per row.
#:
#: The rows are read column-wise, which is a pass over every cell: a table
#: given a row at a time has to be turned the other way round before any of it
#: can be uploaded, and there is no device that can help with data still on the
#: host. What that pass costs is one unboxing per cell, not one dispatch per
#: cell.
#:
#: Unnamed columns are numbered from zero, as pandas numbers them. `dtypes`
#: overrides inference column by column, which is what a reader that already
#: parsed a header wants.
pub fn frame_from_rows(
    rows: list[list[any]],
    columns: list[str]? = null,
    index: Index? = null,
    dtypes: list[Dtype]? = null,
    na_values: list[str]? = null,
    keep_default_na: bool = true
) -> DataFrame {
    let width = if columns is null {
        if rows.len() > 0 {
            rows[0].len()
        } else {
            0
        }
    } else {
        columns.len()
    }
    let names = columns ?? numbered(width)
    #: Every shape complaint is raised before the first allocation, so a
    #: malformed call cannot leave half a frame's worth of device buffers with
    #: nobody holding them.
    check_widths(rows, width)
    check_dtypes(dtypes, width)
    check_index(index, rows.len())
    var lanes: list[list[any]] = []
    for _at in 0..width { lanes.push([]) }
    for row in rows {
        for (at, value) in row.enumerate() { lanes[at].push(value) }
    }
    var built: list[Column] = []
    for (at, lane) in lanes.enumerate() {
        built.push(column_from_values(lane, dtype_at(dtypes, at), na_values, keep_default_na))
    }
    return DataFrame(names, built, index: index)
}

#: A frame from one host list per column.
#:
#: Column order is the dict's insertion order, so the frame reads the way the
#: literal was written.
pub fn frame_from_dict(
    data: dict[str, list[any]],
    index: Index? = null,
    dtypes: dict[str, Dtype]? = null,
    na_values: list[str]? = null,
    keep_default_na: bool = true
) -> DataFrame {
    var names: list[str] = []
    var lanes: list[list[any]] = []
    var length = -1
    for (name, values) in data.items() {
        if length < 0 { length = values.len() }
        if values.len() != length {
            throw ValueError(
                f"column {name} has {values.len()} values where the frame has {length} rows"
            )
        }
        names.push(name)
        lanes.push(values)
    }
    check_index(index, if length < 0 { 0 } else { length })
    var built: list[Column] = []
    for (at, lane) in lanes.enumerate() {
        let asked = if dtypes is null { null } else { dtypes.get(names[at], null) }
        built.push(column_from_values(lane, asked, na_values, keep_default_na))
    }
    return DataFrame(names, built, index: index)
}

#: A frame from host records, one dict per row.
#:
#: The columns are the union of the keys in first-appearance order, so a record
#: that omits a key contributes a missing cell there rather than shifting the
#: row. Naming `columns` fixes the set instead, and then a key no column asked
#: for is dropped.
pub fn frame_from_records(
    records: list[dict[str, any]],
    columns: list[str]? = null,
    index: Index? = null,
    dtypes: list[Dtype]? = null,
    na_values: list[str]? = null,
    keep_default_na: bool = true
) -> DataFrame {
    let names = columns ?? record_names(records)
    check_dtypes(dtypes, names.len())
    check_index(index, records.len())
    var lanes: list[list[any]] = []
    for _at in 0..names.len() { lanes.push([]) }
    for record in records {
        for (at, name) in names.enumerate() { lanes[at].push(record.get(name, null)) }
    }
    var built: list[Column] = []
    for (at, lane) in lanes.enumerate() {
        built.push(column_from_values(lane, dtype_at(dtypes, at), na_values, keep_default_na))
    }
    return DataFrame(names, built, index: index)
}

#: A series from host values.
pub fn series_from_list(
    values: list[any],
    index: Index? = null,
    name: str? = null,
    dtype: Dtype? = null,
    na_values: list[str]? = null,
    keep_default_na: bool = true
) -> Series {
    check_index(index, values.len())
    return Series(
        column_from_values(values, dtype, na_values, keep_default_na),
        index: index,
        name: name
    )
}

#: A series whose labels are the dict's keys and whose values are its values.
#:
#: Insertion order again, which is the order pandas keeps too.
pub fn series_from_dict(
    data: dict[str, any],
    name: str? = null,
    dtype: Dtype? = null,
    na_values: list[str]? = null,
    keep_default_na: bool = true
) -> Series {
    var labels: list[any] = []
    var values: list[any] = []
    for (key, value) in data.items() {
        labels.push(key)
        values.push(value)
    }
    return Series(
        column_from_values(values, dtype, na_values, keep_default_na),
        index: index_from_list(labels),
        name: name
    )
}

#: A frame of no rows, with the columns and dtypes named.
#:
#: The columns are allocated rather than faked, so every operation that walks a
#: frame's columns walks these too and none of them needs a case for a frame
#: that has not been filled in yet.
pub fn empty_frame(names: list[str]? = null, dtypes: list[Dtype]? = null) -> DataFrame {
    let labels = names ?? []
    check_dtypes(dtypes, labels.len())
    var built: list[Column] = []
    for at in 0..labels.len() {
        built.push(Column.allocate(dtype_at(dtypes, at) ?? Dtype.Float64, 0))
    }
    return DataFrame(labels, built, index: Index.range(0))
}

#: A frame of `length` rows where every cell is `value`.
#:
#: The fill happens on the device, one call per column, so the host never
#: builds the values it is asking for. `NA` gives a frame of nulls.
pub fn full(names: list[str], length: int, value: Value, index: Index? = null) -> DataFrame {
    check_index(index, length)
    var built: list[Column] = []
    for _name in names { built.push(Column.full(value.dtype, length, value)) }
    return DataFrame(names, built, index: index ?? Index.range(length))
}

#: The frame with `column` inserted at `position` under `name`.
#:
#: A negative position counts from the end the way a list insert does, so `-1`
#: puts the column second from the right. The result shares every column with
#: `frame`, including the one just handed in — free one frame or the other, not
#: both, and do not read either after the other has gone.
pub fn insert(frame: DataFrame, position: int, name: str, column: Column) -> DataFrame {
    let width = frame.width()
    let at = if position < 0 { width + position } else { position }
    if at < 0 or at > width {
        throw IndexError(f"position {position} is outside a frame of {width} columns")
    }
    var names: list[str] = []
    var columns: list[Column] = []
    for (held, label) in frame.names.enumerate() {
        if held == at {
            names.push(name)
            columns.push(column)
        }
        names.push(label)
        columns.push(frame.columns[held])
    }
    if at == width {
        names.push(name)
        columns.push(column)
    }
    return frame._replace(names: names, columns: columns)
}

#: The frame with `name` set to `column`, replacing it or appending it.
#:
#: pandas' `assign`, less the callable form. A replaced column is not freed:
#: whoever built it still owns it, and this call cannot know whether they are
#: done with it.
pub fn assign(frame: DataFrame, name: str, column: Column) -> DataFrame {
    if not frame.has(name) { return insert(frame, frame.width(), name, column) }
    var columns = [existing for existing in frame.columns]
    columns[frame.column_index(name)] = column
    return frame._replace(columns: columns)
}

#: The frame without the named columns.
#:
#: The dropped columns are returned to nobody and freed by nobody — they are
#: still `frame`'s. `missing_ok` is pandas' `errors="ignore"`.
pub fn drop_columns(frame: DataFrame, names: list[str], missing_ok: bool = false) -> DataFrame {
    var doomed: dict[str, bool] = {}
    for name in names {
        if not frame.has(name) and not missing_ok { throw KeyError(f"no column named {name}") }
        doomed[name] = true
    }
    var kept: list[str] = []
    var columns: list[Column] = []
    for (at, name) in frame.names.enumerate() {
        if doomed.get(name, false) { continue }
        kept.push(name)
        columns.push(frame.columns[at])
    }
    return frame._replace(names: kept, columns: columns)
}

#: The frame without one column, and that column.
#:
#: pandas' `pop` mutates and hands back a `Series`; a frame here is immutable,
#: so both halves come back instead. The column is the same allocation `frame`
#: still holds, not a copy — it outlives the returned frame but not the one it
#: came out of. Wrap it with `to_series` if it wants labels.
pub fn pop(frame: DataFrame, name: str) -> (DataFrame, Column) {
    let taken = frame.column(name)
    return (drop_columns(frame, [name]), taken)
}

#: The frame with its columns renamed through `mapping`.
#:
#: A column the mapping does not name keeps the name it had, and by default a
#: key that names no column is ignored, which is what pandas does. The columns
#: themselves are untouched, so this costs two host lists and no device work.
pub fn rename(frame: DataFrame, mapping: dict[str, str], missing_ok: bool = true) -> DataFrame {
    if not missing_ok {
        for (old, _to) in mapping.items() {
            if not frame.has(old) { throw KeyError(f"no column named {old}") }
        }
    }
    return frame._replace(names: frame.names.map(|name| mapping.get(name, name)))
}

#: The frame's columns in the order `names` gives them.
#:
#: Every column has to appear exactly once: this is a permutation, not a
#: selection. Dropping and repeating columns by name is what the access module
#: is for, and conflating the two is how a typo silently loses a column.
pub fn reorder(frame: DataFrame, names: list[str]) -> DataFrame {
    if names.len() != frame.width() {
        throw ValueError(f"{names.len()} names cannot reorder {frame.width()} columns")
    }
    var seen: dict[str, bool] = {}
    var columns: list[Column] = []
    for name in names {
        if seen.get(name, false) { throw ValueError(f"column {name} appears twice in the order") }
        seen[name] = true
        columns.push(frame.column(name))
    }
    return frame._replace(names: [name for name in names], columns: columns)
}

#: The frame's columns whose dtype is in `include` and not in `exclude`.
#:
#: Column order is kept, so this reads as a filter rather than a rearrangement.
pub fn select_dtypes(
    frame: DataFrame,
    include: list[Dtype]? = null,
    exclude: list[Dtype]? = null
) -> DataFrame {
    if include is null and exclude is null {
        throw ValueError("select_dtypes needs include, exclude, or both")
    }
    var names: list[str] = []
    var columns: list[Column] = []
    for (at, column) in frame.columns.enumerate() {
        if include is not null and not holds(include, column.dtype) { continue }
        if exclude is not null and holds(exclude, column.dtype) { continue }
        names.push(frame.names[at])
        columns.push(column)
    }
    return frame._replace(names: names, columns: columns)
}

#: The frame with the named columns read as other dtypes.
#:
#: `spec` is either one dtype for the whole frame or a mapping of column name
#: to dtype, and either may spell the dtype as a pandas string. Every column is
#: rebuilt, converted or not, so the result owns all of its lanes and its index
#: — which is what pandas' `astype` does, and what makes it safe to free the
#: original straight after.
pub fn astype(frame: DataFrame, spec: any) -> DataFrame {
    var targets: dict[str, Dtype] = {}
    if isinstance(spec, dict) {
        for (name, wanted) in spec.items() {
            if not frame.has(name) { throw KeyError(f"no column named {name}") }
            targets[name] = as_dtype(wanted)
        }
    } else {
        let wanted = as_dtype(spec)
        for name in frame.names { targets[name] = wanted }
    }
    var columns: list[Column] = []
    for (at, column) in frame.columns.enumerate() {
        columns.push(column.cast(targets.get(frame.names[at], column.dtype)))
    }
    return frame._replace(columns: columns, index: frame.index.copy())
}

#: The series read as another dtype, values and index owned by the result.
pub fn astype_series(series: Series, dtype: any) -> Series {
    return Series(
        series.values.cast(as_dtype(dtype)),
        index: series.index.copy(),
        name: series.name
    )
}

#: An independent frame, or a fresh view onto the same columns.
#:
#: A deep copy duplicates every lane — a dispatch per column, and the only way
#: to get a frame the original cannot be seen through. A shallow one rebuilds
#: nothing but the name list and its lookup table, which is what every other
#: edit in this module returns and costs no device work at all.
pub fn copy(frame: DataFrame, deep: bool = true) -> DataFrame {
    if deep { return frame.copy() }
    return frame._replace(names: [name for name in frame.names])
}

#: The frame with `prefix` on the front of every column name.
pub fn add_prefix(frame: DataFrame, prefix: str) -> DataFrame {
    return frame._replace(names: frame.names.map(|name| prefix + name))
}

#: The frame with `suffix` on the end of every column name.
pub fn add_suffix(frame: DataFrame, suffix: str) -> DataFrame {
    return frame._replace(names: frame.names.map(|name| name + suffix))
}

#: A one-column frame as a series, pandas' `squeeze`.
#:
#: The values and the index are shared with the frame, not copied. Refusing a
#: wider frame is the point: picking one column out of several is a selection,
#: and belongs to the access module where the caller has said which one.
pub fn to_series(frame: DataFrame, name: str? = null) -> Series {
    if frame.width() != 1 {
        throw ValueError(f"a frame of {frame.width()} columns is not one series")
    }
    return Series(frame.columns[0], index: frame.index, name: name ?? frame.names[0])
}

#: A series as a one-column frame, sharing its values and its index.
#:
#: An unnamed series becomes column `0`, the way pandas numbers one.
pub fn to_frame(series: Series, name: str? = null) -> DataFrame {
    return DataFrame([(name ?? series.name) ?? "0"], [series.values], index: series.index)
}

#: The column's dictionary read as a category.
#:
#: The storage of `Str` and `Category` is the same, so this is a relabel of
#: intent: the levels become the domain, and a level no row uses survives a
#: group-by as an empty group instead of vanishing.
#:
#: `ordered` sorts the levels. No flag on a column says it is ordered, and
#: none needs to: what an ordered category means to every operation here is
#: that a code comparison is a value comparison, and sorting the table is that
#: property, stored. The codes are remapped on the device, so ordering a column
#: of any height is one dispatch.
#:
#: Numbers are refused. Turning 3.5 into the category "3.5" needs a formatter,
#: and `column.jai` puts formatting on the display side of the line for the
#: same reason it puts parsing on the reader's.
pub fn categorize(column: Column, ordered: bool = false) -> Column {
    if not column.dtype.is_dictionary() {
        throw TypeError(
            f"{column.dtype.name()} has no levels to categorize; format it as text first"
        )
    }
    if not ordered { return column.cast(Dtype.Category) }
    return set_categories(column, sorted([name for name in column.levels ?? []]))
}

#: The column's levels, in code order.
pub fn categories(column: Column) -> list[str] {
    if not column.dtype.is_dictionary() {
        throw TypeError(f"{column.dtype.name()} has no levels")
    }
    return [name for name in column.levels ?? []]
}

#: The column over a new level table, in the order given.
#:
#: A row whose level is not in the new table becomes missing, which is what
#: makes this the way to narrow a domain as well as to reorder one. The result
#: is a `Category`: naming the domain is what the dtype means.
pub fn set_categories(column: Column, categories: list[str]) -> Column {
    if not column.dtype.is_dictionary() {
        throw TypeError(f"{column.dtype.name()} has no levels")
    }
    var lookup: dict[str, int] = {}
    for (code, name) in categories.enumerate() {
        if lookup.get(name, -1) >= 0 { throw ValueError(f"category {name} is listed twice") }
        lookup[name] = code
    }
    var mapping: list[float] = []
    var drops = false
    for name in column.levels ?? [] {
        let code = lookup.get(name, -1)
        if code < 0 { drops = true }
        mapping.push(float(code))
    }
    return recoded(column, mapping, [name for name in categories], Dtype.Category, drops)
}

#: The column without the levels no present row uses.
#:
#: Which levels are live is counted on the device — a mark per row into a
#: threadgroup-staged table, merged out once a group — so the host only ever
#: sees one flag per level. No row changes value, so nothing becomes missing.
pub fn remove_unused_categories(column: Column) -> Column {
    if not column.dtype.is_dictionary() {
        throw TypeError(f"{column.dtype.name()} has no levels")
    }
    let used = level_usage(column)
    var levels: list[str] = []
    var mapping: list[float] = []
    for (code, name) in (column.levels ?? []).enumerate() {
        if not used[code] {
            mapping.push(-1.0)
            continue
        }
        mapping.push(float(levels.len()))
        levels.push(name)
    }
    if levels.len() == column.level_count() { return column.copy() }
    return recoded(column, mapping, levels, column.dtype, false)
}

#: Whether two frames, series, columns or indexes hold the same data.
#:
#: pandas' `equals`: same shape, same dtypes, same values, and two missing
#: cells in the same place count as the same cell. Comparing is done on the
#: device and reduced to one number, so a frame of any height costs a dispatch
#: a column rather than a download of the whole table.
#:
#: Dtypes have to match — an `Int64` column of ones is not a `Float64` column
#: of ones, as in pandas. Names do not: a series' name and an index's names are
#: labels on the data, not the data.
pub fn equals(left: any, right: any) -> bool {
    if isinstance(left, DataFrame) and isinstance(right, DataFrame) {
        return frames_equal(left, right)
    }
    if isinstance(left, Series) and isinstance(right, Series) {
        return indexes_equal(left.index, right.index) and columns_equal(left.values, right.values)
    }
    if isinstance(left, Column) and isinstance(right, Column) {
        return columns_equal(left, right)
    }
    if isinstance(left, Index) and isinstance(right, Index) {
        return indexes_equal(left, right)
    }
    return false
}

fn frames_equal(left: DataFrame, right: DataFrame) -> bool {
    if left.names != right.names { return false }
    if not indexes_equal(left.index, right.index) { return false }
    for (at, column) in left.columns.enumerate() {
        if not columns_equal(column, right.columns[at]) { return false }
    }
    return true
}

fn indexes_equal(left: Index, right: Index) -> bool {
    if left.length != right.length { return false }
    if left.nlevels() != right.nlevels() { return false }
    if left.length == 0 { return true }
    #: Two range indexes agree on every label the moment they agree on the
    #: start and the step, the labels having already agreed on how many there
    #: are. Not having to look at them is the whole point of not storing them.
    if left.is_range() and right.is_range() {
        return left.start == right.start and left.step == right.step
    }
    let ours = left.materialize()
    let theirs = right.materialize()
    defer {
        for level in ours { level.free() }
        for level in theirs { level.free() }
    }
    for (at, level) in ours.enumerate() {
        if not columns_equal(level, theirs[at]) { return false }
    }
    return true
}

fn columns_equal(left: Column, right: Column) -> bool {
    if left.dtype != right.dtype { return false }
    if left.length != right.length { return false }
    if left.length == 0 { return true }
    if not left.dtype.is_dictionary() { return lane_diff_count(left, right) == 0 }
    #: Codes only mean the same thing on one table, so the two dictionaries are
    #: merged before the codes are compared. Levels nobody uses do not make two
    #: columns differ, because no row reads them.
    #:
    #: Two columns out of the same frame almost always carry the same table
    #: already, and comparing two dictionaries on the host is cheaper than the
    #: two remaps unifying them would cost.
    if (left.levels ?? []) == (right.levels ?? []) {
        return lane_diff_count(left, right) == 0
    }
    let unified = unify_levels([left, right])
    defer { for part in unified { part.free() } }
    return lane_diff_count(unified[0], unified[1]) == 0
}

fn lane_diff_count(left: Column, right: Column) -> int {
    let count = left.length
    let partials = Buffer(DIFF_GROUPS)
    let plain = not left.dtype.is_wide() and not left.has_validity() and not right.has_validity()
    if plain and count % 4 == 0 {
        compiled("frame_diff_count4").dispatch_async(
            [left.values, right.values, partials],
            DIFF_GROUPS * DIFF_GROUP,
            [count // 4],
            group_size: DIFF_GROUP
        )
    } else {
        compiled("frame_diff_count").dispatch_async(
            [
                left.values,
                left.wide ?? left.values,
                left.valid ?? left.values,
                right.values,
                right.wide ?? right.values,
                right.valid ?? right.values,
                partials,
            ],
            DIFF_GROUPS * DIFF_GROUP,
            [
                count,
                flag(left.dtype.is_wide()),
                flag(left.has_validity()),
                flag(right.has_validity()),
            ],
            group_size: DIFF_GROUP
        )
    }
    var total = 0.0
    for part in partials.download() { total += part }
    partials.free()
    return int(total)
}

fn level_usage(column: Column) -> list[bool] {
    let levels = column.level_count()
    if levels == 0 { return [] }
    let used = Buffer(levels)
    used.fill_zero()
    if column.length > 0 {
        let groups = (column.length + USAGE_GROUP - 1) // USAGE_GROUP
        compiled("frame_level_usage").dispatch_async(
            [column.values, column.valid ?? column.values, used],
            groups * USAGE_GROUP,
            [column.length, levels, flag(column.has_validity())],
            group_size: USAGE_GROUP
        )
    }
    let marks = used.download()
    used.free()
    return marks.map(|mark| mark != 0.0)
}

fn recoded(
    column: Column,
    mapping: list[float],
    levels: list[str],
    dtype: Dtype,
    drops: bool
) -> Column {
    if mapping.len() == 0 {
        let empty = Column.nulls(dtype, column.length)
        empty.set_levels([name for name in levels])
        return empty
    }
    #: A validity lane appears when a level might be dropped, even if the rows
    #: that used it turn out not to exist. Knowing better would mean a usage
    #: pass this call does not otherwise need, and an all-ones mask is a lane,
    #: not a wrong answer.
    let out = Column.allocate(dtype, column.length, with_validity: drops or column.has_validity())
    out.set_levels([name for name in levels])
    if column.length == 0 { return out }
    let lane = Buffer(mapping.len())
    lane.upload(mapping)
    compiled("frame_recode").dispatch_async(
        [column.values, column.valid ?? column.values, lane, out.values, out.valid ?? out.values],
        column.length,
        [column.length, mapping.len(), flag(column.has_validity()), flag(out.has_validity())]
    )
    lane.free()
    return out
}

fn text_column(
    values: list[any],
    dtype: Dtype,
    words: dict[str, bool],
    keep_default: bool
) -> Column {
    var levels: list[str] = []
    var intern: dict[str, int] = {}
    var codes: list[float] = []
    var present: list[bool] = []
    var any_missing = false
    for value in values {
        if is_missing(value, words, keep_default) {
            codes.push(0.0)
            present.push(false)
            any_missing = true
            continue
        }
        let text = as_text(value)
        var code = intern.get(text, -1)
        if code < 0 {
            code = levels.len()
            levels.push(text)
            intern[text] = code
        }
        codes.push(float(code))
        present.push(true)
    }
    let column = Column.allocate(dtype, values.len(), with_validity: any_missing)
    #: One call for the whole lane. This is the line the module exists to keep
    #: outside a loop.
    if values.len() > 0 { column.values.upload(codes) }
    upload_mask(column, present, any_missing)
    column.set_levels(levels)
    return column
}

fn wide_column(
    values: list[any],
    dtype: Dtype,
    words: dict[str, bool],
    keep_default: bool
) -> Column {
    var low: list[float] = []
    var high: list[float] = []
    var present: list[bool] = []
    var any_missing = false
    for value in values {
        if is_missing(value, words, keep_default) {
            low.push(0.0)
            high.push(0.0)
            present.push(false)
            any_missing = true
            continue
        }
        let split = wide_pack(as_int(value))
        low.push(split[0])
        high.push(split[1])
        present.push(true)
    }
    let column = Column.allocate(dtype, values.len(), with_validity: any_missing)
    if values.len() > 0 {
        column.values.upload(low)
        (column.wide ?? column.values).upload(high)
    }
    upload_mask(column, present, any_missing)
    return column
}

fn plain_column(
    values: list[any],
    dtype: Dtype,
    words: dict[str, bool],
    keep_default: bool
) -> Column {
    #: A missing float slot carries NaN rather than zero, so a caller that
    #: drops the mask and reads the lane still sees a gap instead of a reading.
    let absent = if dtype == Dtype.Bool { 0.0 } else { NAN }
    var slots: list[float] = []
    var present: list[bool] = []
    var any_missing = false
    for value in values {
        if is_missing(value, words, keep_default) {
            slots.push(absent)
            present.push(false)
            any_missing = true
            continue
        }
        slots.push(if dtype == Dtype.Bool {
            if as_bool(value) {
                1.0
            } else {
                0.0
            }
        } else {
            as_float(value)
        })
        present.push(true)
    }
    let column = Column.allocate(dtype, values.len(), with_validity: any_missing)
    if values.len() > 0 { column.values.upload(slots) }
    upload_mask(column, present, any_missing)
    return column
}

fn upload_mask(column: Column, present: list[bool], any_missing: bool) -> void {
    if not any_missing { return }
    let mask = column.valid
    if mask is null or present.len() == 0 { return }
    mask.upload(present.map(|slot| if slot { 1.0 } else { 0.0 }))
}

fn na_lookup(na_values: list[str]?) -> dict[str, bool] {
    var words: dict[str, bool] = {}
    for word in na_values ?? [] { words[word] = true }
    return words
}

fn is_missing(value: any, words: dict[str, bool], keep_default: bool) -> bool {
    if value is null { return true }
    if isinstance(value, Value) { return value.is_na() }
    if isinstance(value, float) and is_nan(value) { return true }
    if not isinstance(value, str) { return false }
    if keep_default and value.len() == 0 { return true }
    return words.get(value, false)
}

fn inferred(values: list[any], words: dict[str, bool], keep_default: bool) -> Dtype {
    var found: Dtype? = null
    for value in values {
        if is_missing(value, words, keep_default) { continue }
        let one = family_of(value)
        found = if found is null { one } else { promote(found ?? one, one) }
        if found == Dtype.Str { return Dtype.Str }
    }
    return found ?? Dtype.Float64
}

fn family_of(value: any) -> Dtype {
    if isinstance(value, Value) { return value.dtype }
    if isinstance(value, bool) { return Dtype.Bool }
    if isinstance(value, int) { return Dtype.Int64 }
    if isinstance(value, float) { return Dtype.Float64 }
    return Dtype.Str
}

fn promote(left: Dtype, right: Dtype) -> Dtype {
    if left == right { return left }
    if not left.is_numeric() or not right.is_numeric() { return Dtype.Str }
    if left == Dtype.Float64 or right == Dtype.Float64 { return Dtype.Float64 }
    return Dtype.Int64
}

fn as_float(value: any) -> float {
    if isinstance(value, Value) { return value.to_float() }
    if isinstance(value, bool) { return if value { 1.0 } else { 0.0 } }
    if isinstance(value, int) { return float(value) }
    if isinstance(value, float) { return value }
    throw TypeError(f"{repr(value)} is not a number; parsing text is the reader's job")
}

fn as_int(value: any) -> int {
    if isinstance(value, Value) { return value.to_int() }
    if isinstance(value, bool) { return if value { 1 } else { 0 } }
    if isinstance(value, int) { return value }
    if isinstance(value, float) { return int(value) }
    throw TypeError(f"{repr(value)} is not an integer; parsing text is the reader's job")
}

fn as_bool(value: any) -> bool {
    if isinstance(value, Value) { return value.to_bool() }
    if isinstance(value, bool) { return value }
    if isinstance(value, str) { return value.len() > 0 }
    return as_float(value) != 0.0
}

#: Host values become text here, where the value is already an object with a
#: spelling. `Column.cast` refuses the same conversion because there it would
#: mean a formatter running on the device, which is a different problem.
fn as_text(value: any) -> str {
    if isinstance(value, Value) { return value.to_str() }
    if isinstance(value, str) { return value }
    return f"{value}"
}

fn as_dtype(spec: any) -> Dtype {
    if isinstance(spec, Dtype) { return spec }
    if isinstance(spec, str) { return Dtype.from_name(spec) }
    throw TypeError(f"{repr(spec)} does not name a dtype")
}

fn holds(dtypes: list[Dtype], dtype: Dtype) -> bool {
    for candidate in dtypes {
        if candidate == dtype { return true }
    }
    return false
}

fn numbered(width: int) -> list[str] {
    var names: list[str] = []
    for at in 0..width { names.push(f"{at}") }
    return names
}

fn record_names(records: list[dict[str, any]]) -> list[str] {
    var names: list[str] = []
    var seen: dict[str, bool] = {}
    for record in records {
        for name in record.keys() {
            if seen.get(name, false) { continue }
            seen[name] = true
            names.push(name)
        }
    }
    return names
}

fn dtype_at(dtypes: list[Dtype]?, at: int) -> Dtype? {
    let listed = dtypes ?? []
    if at >= listed.len() { return null }
    return listed[at]
}

fn check_widths(rows: list[list[any]], width: int) -> void {
    for (at, row) in rows.enumerate() {
        if row.len() != width {
            throw ValueError(f"row {at} has {row.len()} values where the frame is {width} wide")
        }
    }
}

fn check_dtypes(dtypes: list[Dtype]?, width: int) -> void {
    if dtypes is null { return }
    if dtypes.len() != width {
        throw ValueError(f"{dtypes.len()} dtypes cannot describe {width} columns")
    }
}

fn check_index(index: Index?, length: int) -> void {
    if index is null { return }
    if index.len() != length {
        throw ValueError(f"an index of {index.len()} cannot label {length} rows")
    }
}

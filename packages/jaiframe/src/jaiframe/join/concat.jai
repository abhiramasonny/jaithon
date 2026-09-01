#: Stacking frames along either axis, and telling two of them apart.
#:
#: `concat` on rows matches columns by name and stacks the lanes; on columns it
#: aligns the parts on their row labels, which is the same pairing a merge does
#: with the index as the key. `compare` is the other whole-frame operation that
#: takes matching shapes rather than a pair of key columns.
from std.gpu import Buffer
from ..column import Column, DataFrame, Dtype, Index, Value
from ..column import concat_columns, iota, prefix_sum, unify_levels
from .kernels import flag, kernel_for, require_device
from .pairing import copy_lane, key_order, merged_key_column, pair_rows, permute_lane
from .types import How

fn range_like(index: Index, other: Index) -> bool {
    if not index.is_range() or not other.is_range() { return false }
    return index.length == other.length and index.start == other.start and index.step == other.step
}

#: The column names `axis: 0` stacking produces, and where each part has them.
fn concat_names(objects: list[DataFrame], inner: bool, sort: bool) -> list[str] {
    var out: list[str] = []
    var seen: dict[str, int] = {}
    for frame in objects {
        for name in frame.names {
            if seen.get(name, -1) >= 0 { continue }
            if inner {
                var everywhere = true
                for other in objects {
                    if not other.has(name) { everywhere = false }
                }
                if not everywhere { continue }
            }
            seen[name] = out.len()
            out.push(name)
        }
    }
    return if sort { sorted(out) } else { out }
}

#: The dtype the parts agree on for `name`, and a refusal when they do not.
fn concat_dtype(objects: list[DataFrame], name: str) -> Dtype {
    var found: Dtype? = null
    for frame in objects {
        if not frame.has(name) { continue }
        let dtype = frame.column(name).dtype
        let earlier = found
        if earlier is not null and earlier != dtype {
            throw TypeError(
                f"column {name} is {earlier.name()} in one part and {dtype.name()} in another"
            )
        }
        found = dtype
    }
    let dtype = found
    if dtype is null { throw KeyError(f"no part has a column named {name}") }
    return dtype
}

#: The index levels of the parts stacked, with `keys` as an outer level.
fn concat_index(
    objects: list[DataFrame],
    keys: list[str]?,
    names: list[str?]?,
    total: int
) -> Index {
    let levels = objects[0].index.nlevels()
    for frame in objects {
        if frame.index.nlevels() != levels {
            throw ValueError(
                f"an index of {frame.index.nlevels()} levels cannot stack onto one of {levels}"
            )
        }
    }
    var stacked: list[Column] = []
    var labels: list[str?] = []
    let tags = keys
    if tags is not null {
        if tags.len() != objects.len() {
            throw ValueError(f"{tags.len()} keys cannot label {objects.len()} parts")
        }
        var parts: list[Column] = []
        for (at, frame) in objects.enumerate() {
            parts.push(Column.full(Dtype.Str, frame.len(), Value.of_str(tags[at])))
        }
        stacked.push(concat_columns(parts))
        for part in parts { part.free() }
        labels.push(null)
    }
    for level in 0..levels {
        var parts: list[Column] = []
        var owned: list[Column] = []
        for frame in objects {
            let materialised = frame.index.materialize()
            for extra in materialised { owned.push(extra) }
            parts.push(materialised[level])
        }
        stacked.push(concat_columns(parts))
        for part in owned { part.free() }
        labels.push(objects[0].index.names[level])
    }
    if total != stacked[0].length {
        throw ValueError(f"the stacked index has {stacked[0].length} labels for {total} rows")
    }
    let named = names
    if named is not null {
        if named.len() != labels.len() {
            throw ValueError(f"{named.len()} names cannot name {labels.len()} index levels")
        }
        labels = [name for name in named]
    }
    return Index.of(stacked, labels)
}

fn concat_rows(
    objects: list[DataFrame],
    inner: bool,
    keys: list[str]?,
    names: list[str?]?,
    ignore_index: bool,
    sort: bool
) -> DataFrame {
    var total = 0
    for frame in objects { total += frame.len() }
    let wanted = concat_names(objects, inner, sort)
    var columns: list[Column] = []
    for name in wanted {
        let dtype = concat_dtype(objects, name)
        var parts: list[Column] = []
        var filler: list[Column] = []
        for frame in objects {
            if frame.has(name) {
                parts.push(frame.column(name))
                continue
            }
            #: A part with no such column contributes missing rows, which is
            #: what an outer concatenation means and what pandas fills in.
            let gap = Column.nulls(dtype, frame.len())
            filler.push(gap)
            parts.push(gap)
        }
        columns.push(concat_columns(parts))
        for gap in filler { gap.free() }
    }
    let index = if ignore_index {
        Index.range(total)
    } else {
        concat_index(objects, keys, names, total)
    }
    return DataFrame(wanted, columns, index: index)
}

fn concat_columns_axis(
    objects: list[DataFrame],
    inner: bool,
    keys: list[str]?,
    names: list[str?]?,
    ignore_index: bool,
    sort: bool
) -> DataFrame {
    let how = if inner { How.Inner } else { How.Outer }
    var lanes: list[Buffer] = []
    var labels: list[Column] = objects[0].index.materialize()
    var rows = objects[0].len()
    lanes.push(iota(rows))

    for at in 1..objects.len() {
        let frame = objects[at]
        #: Two frames already labelled 0..n-1 the same way need no pairing, and
        #: side-by-side concatenation of default-indexed frames is the common
        #: case by a wide margin.
        if range_like(objects[0].index, frame.index) and rows == frame.len() {
            lanes.push(iota(rows))
            continue
        }
        let next = frame.index.materialize()
        let pairing = pair_rows(labels, next, rows, frame.len(), how)
        var merged: list[Column] = []
        for (level, column) in labels.enumerate() {
            merged.push(merged_key_column(column, next[level], pairing))
        }
        var updated: list[Buffer] = []
        for lane in lanes {
            let mapped = Buffer(pairing.rows)
            kernel_for("jn_map_positions").dispatch_async(
                [pairing.left, lane, mapped],
                pairing.rows,
                [pairing.rows]
            )
            updated.push(mapped)
            lane.free()
        }
        let taken = Buffer(pairing.rows)
        copy_lane(pairing.right, taken, pairing.rows)
        updated.push(taken)
        for column in labels { column.free() }
        for column in next { column.free() }
        pairing.free()
        lanes = updated
        labels = merged
        rows = pairing.rows
    }

    if sort and rows > 1 and labels.len() > 0 {
        let perm = key_order(labels, rows)
        var ordered: list[Column] = []
        for column in labels { ordered.push(column.take(perm, rows)) }
        for column in labels { column.free() }
        labels = ordered
        for lane in lanes { permute_lane(lane, perm, rows) }
        perm.free()
    }

    let tags = keys
    if tags is not null and tags.len() != objects.len() {
        throw ValueError(f"{tags.len()} keys cannot label {objects.len()} parts")
    }
    var out_names: list[str] = []
    var out_columns: list[Column] = []
    for (at, frame) in objects.enumerate() {
        for (position, name) in frame.names.enumerate() {
            #: A flat name list cannot hold pandas' two-level column index, so
            #: a key becomes a prefix instead of an outer level.
            let labelled = if tags is null { name } else { f"{tags[at]}_{name}" }
            out_names.push(labelled)
            out_columns.push(frame.columns[position].take(lanes[at], rows, allow_missing: true))
        }
    }
    for lane in lanes { lane.free() }
    let index = if ignore_index {
        for column in labels {
            column.free()
        }
        Index.range(rows)
    } else {
        let named = names
        var level_names: list[str?] = [name for name in objects[0].index.names]
        if named is not null {
            level_names = [name for name in named]
        }
        Index.of(labels, level_names)
    }
    return DataFrame(out_names, out_columns, index: index)
}

#: The parts combined, pandas' `concat`.
#:
#: `axis: 0` stacks rows, matching columns by name — `join: How.Outer` keeps
#: every name and fills the gaps with nulls, `How.Inner` keeps only the names
#: every part has. `axis: 1` puts the parts side by side, aligning them on
#: their row labels, which is the same pairing a merge does with the index as
#: the key.
#:
#: `keys` labels the parts: an outer index level when stacking rows, a prefix
#: on the column names when stacking columns, because a frame's column names
#: are a flat list here and cannot carry pandas' second level.
pub fn concat(
    objects: list[DataFrame],
    axis: int = 0,
    join: How = How.Outer,
    keys: list[str]? = null,
    names: list[str?]? = null,
    ignore_index: bool = false,
    sort: bool = false
) -> DataFrame {
    if objects.len() == 0 { throw ValueError("a concatenation needs at least one frame") }
    if join != How.Outer and join != How.Inner {
        throw ValueError(f"concat joins outer or inner, not {join.name()}")
    }
    let inner = join == How.Inner
    if axis == 0 { return concat_rows(objects, inner, keys, names, ignore_index, sort) }
    if axis == 1 {
        return concat_columns_axis(objects, inner, keys, names, ignore_index, sort)
    }
    throw ValueError(f"a frame has axes 0 and 1, not {axis}")
}

#: `other`'s rows added under `frame`'s, pandas' `DataFrame.append`.
pub fn append(frame: DataFrame, other: DataFrame, ignore_index: bool = false) -> DataFrame {
    return concat([frame, other], axis: 0, ignore_index: ignore_index)
}

#: How many slots of a 0/1 lane are clear.
fn lane_zeros(lane: Buffer, count: int) -> int {
    if count == 0 { return 0 }
    let flags = Buffer(count)
    kernel_for("jn_not_flags").dispatch_async([lane, flags], count, [count])
    let scanned = prefix_sum(flags, count)
    flags.free()
    scanned[0].free()
    return scanned[1]
}

#: The rows a comparison keeps: every row when the shape is kept, otherwise the
#: rows where at least one column disagreed.
fn compared_rows(all_same: Buffer, rows: int, keep_shape: bool) -> (Buffer, int) {
    if keep_shape { return (iota(rows), rows) }
    let flags = Buffer(rows)
    kernel_for("jn_not_flags").dispatch_async([all_same, flags], rows, [rows])
    let scanned = prefix_sum(flags, rows)
    let lane = Buffer(scanned[1])
    kernel_for("jn_compact_positions").dispatch_async([flags, scanned[0], lane], rows, [rows])
    flags.free()
    scanned[0].free()
    return (lane, scanned[1])
}

#: Where two frames disagree, pandas' `DataFrame.compare`.
#:
#: The two frames have to carry the same columns in the same order over the
#: same number of rows: a difference between differently labelled frames is a
#: merge, and asking for one here would hide the mistake. Each surviving column
#: contributes `name_self` and `name_other`, because a flat name list cannot
#: hold pandas' second column level.
#:
#: By default only the differing rows and columns come back and an agreeing
#: cell inside a surviving row reads as null — the shape of the answer is what
#: makes it readable. `keep_shape` keeps every row and column, `keep_equal`
#: keeps the agreeing values instead of nulling them.
#:
#: Equality is `Column.equal_rows`, the same test a join confirms a bucket
#: with, so two frames compare equal exactly when a merge on every column would
#: have paired their rows — nulls included, which agree with each other.
pub fn compare(
    a: DataFrame,
    b: DataFrame,
    keep_shape: bool = false,
    keep_equal: bool = false
) -> DataFrame {
    require_device("compare two frames")
    if a.len() != b.len() {
        throw ValueError(f"a frame of {a.len()} rows cannot be compared with one of {b.len()}")
    }
    if a.width() != b.width() {
        throw ValueError(
            f"a frame of {a.width()} columns cannot be compared with one of {b.width()}"
        )
    }
    for (at, name) in a.names.enumerate() {
        if name != b.names[at] {
            throw ValueError(f"column {at} is {name} in one frame and {b.names[at]} in the other")
        }
    }

    let rows = a.len()
    let positions = iota(rows)
    defer { positions.free() }
    var owned: list[Column] = []
    defer {
        for column in owned { column.free() }
    }
    let all_same = Buffer(rows)
    all_same.fill(1.0)
    defer { all_same.free() }

    var same: list[Buffer] = []
    defer {
        for lane in same { lane.free() }
    }
    for (at, name) in a.names.enumerate() {
        var mine = a.columns[at]
        var theirs = b.columns[at]
        if mine.dtype != theirs.dtype {
            throw TypeError(
                f"column {name} is {mine.dtype.name()} in one frame and {theirs.dtype.name()} in the other"
            )
        }
        #: Two dictionaries only compare once they are on one table, exactly as
        #: two join keys only hash alike once they are.
        if mine.dtype.is_dictionary() {
            let pair = unify_levels([mine, theirs])
            owned.push(pair[0])
            owned.push(pair[1])
            mine = pair[0]
            theirs = pair[1]
        }
        let lane = Buffer(rows)
        lane.fill(1.0)
        mine.equal_rows(positions, theirs, positions, rows, lane)
        mine.equal_rows(positions, theirs, positions, rows, all_same)
        same.push(lane)
    }

    let kept = compared_rows(all_same, rows, keep_shape)
    let sifted = kept[0]
    let kept_rows = kept[1]
    defer { sifted.free() }

    var out_names: list[str] = []
    var out_columns: list[Column] = []
    for (at, name) in a.names.enumerate() {
        if not keep_shape and lane_zeros(same[at], rows) == 0 { continue }
        let lane = Buffer(kept_rows)
        kernel_for("jn_unequal_positions").dispatch_async(
            [same[at], sifted, lane],
            kept_rows,
            [kept_rows, flag(keep_equal)]
        )
        out_names.push(f"{name}_self")
        out_columns.push(a.columns[at].take(lane, kept_rows, allow_missing: true))
        out_names.push(f"{name}_other")
        out_columns.push(b.columns[at].take(lane, kept_rows, allow_missing: true))
        lane.free()
    }
    return DataFrame(out_names, out_columns, index: a.index.take(sifted, kept_rows))
}

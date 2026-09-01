#: Putting two things on one index: the joint index, the reindex that moves
#: rows onto it, and the fill methods that let a missing label reach for a
#: neighbour.
#:
#: `align` is the call `arith.jai` makes before every binary operation, so the
#: shape here is fixed: two things in, two things on one index out, and `how`
#: deciding which labels survive.
from std.gpu import Buffer
from ..column import Column, DataFrame, Index, Series, Value
from ..column import NA, f32_bits, iota, wide_pack
from .kernels import flag, kernel_for, require_device, u32
from .kernels import valid_lane, value_lane, wide_lane
from .labels import borrowed_levels, recoded_like, release_all
from .lookup import IndexLookup, is_monotonic_increasing
from .setops import equals, intersection, union

#: Carry the last label at or before each target label.
pub const METHOD_PAD = "ffill"
#: Carry the first label at or after each target label.
pub const METHOD_BACKFILL = "bfill"
#: Take whichever neighbouring label is closer.
pub const METHOD_NEAREST = "nearest"

#: The index two frames are put on to be combined.
#:
#: `how` is pandas': `outer` takes every label, `inner` the shared ones, `left`
#: and `right` keep one side's index untouched. Two indexes that already agree
#: come back unchanged whatever `how` says, which is the case an arithmetic
#: expression between two columns of one frame hits every time.
pub fn joint_index(left: Index, right: Index, how: str = "outer") -> Index {
    if equals(left, right) { return left.copy() }
    return match how {
        "outer" => union(left, right),
        "inner" => intersection(left, right),
        "left" => left.copy(),
        "right" => right.copy(),
        _ => throw ValueError(f"unknown join {how}"),
    }
}

#: The frame's rows put onto `target`, one take per column.
#:
#: A label of `target` the frame does not have becomes a missing row, or
#: `fill_value` when one is given. `method` fills those holes from a
#: neighbouring label instead: `METHOD_PAD` from the last label at or before,
#: `METHOD_BACKFILL` from the first at or after, `METHOD_NEAREST` from
#: whichever is closer — all three need a monotonic increasing index, since
#: that is what "before" means.
#:
#: The index has to be unique, as pandas' does: with a label on two rows there
#: is no one row to move.
pub fn reindex(
    frame: DataFrame,
    target: Index,
    method: str? = null,
    fill_value: Value? = null
) -> DataFrame {
    let positions = reindex_indexer(frame.index, target, method)
    var columns: list[Column] = []
    for column in frame.columns {
        let taken = column.take(positions, target.len(), allow_missing: true)
        fill_holes(taken, positions, fill_value)
        columns.push(taken)
    }
    positions.free()
    return DataFrame([name for name in frame.names], columns, index: target.copy())
}

#: The series' rows put onto `target`, on the same terms as `reindex`.
pub fn reindex_series(
    series: Series,
    target: Index,
    method: str? = null,
    fill_value: Value? = null
) -> Series {
    let positions = reindex_indexer(series.index, target, method)
    let taken = series.values.take(positions, target.len(), allow_missing: true)
    fill_holes(taken, positions, fill_value)
    positions.free()
    return Series(taken, index: target.copy(), name: series.name)
}

#: Both frames on one index, which is what an operation between them needs.
#:
#: The pair comes back in the order it went in. This is the call `arith.jai`
#: makes before every binary operation, so its shape is fixed: two frames in,
#: two frames on the same index out, and `how` deciding which labels survive.
pub fn align(left: DataFrame, right: DataFrame, how: str = "outer") -> (DataFrame, DataFrame) {
    let joint = joint_index(left.index, right.index, how)
    let a = reindex(left, joint)
    let b = reindex(right, joint)
    joint.free()
    return (a, b)
}

#: Both series on one index.
pub fn align_series(left: Series, right: Series, how: str = "outer") -> (Series, Series) {
    let joint = joint_index(left.index, right.index, how)
    let a = reindex_series(left, joint)
    let b = reindex_series(right, joint)
    joint.free()
    return (a, b)
}

#: A frame and a series on one index, for the row-wise half of frame
#: arithmetic.
pub fn align_frame_series(
    frame: DataFrame,
    series: Series,
    how: str = "outer"
) -> (DataFrame, Series) {
    let joint = joint_index(frame.index, series.index, how)
    let a = reindex(frame, joint)
    let b = reindex_series(series, joint)
    joint.free()
    return (a, b)
}

#: Where each label of `target` sits in `index`, honouring a fill method.
#:
#: Split out because `reindex` and `reindex_series` want the same lane and
#: because a caller reindexing several frames onto one target should compute
#: it once.
pub fn reindex_indexer(index: Index, target: Index, method: str? = null) -> Buffer {
    require_device("reindex")
    if equals(index, target) { return iota(index.len()) }
    let lookup = IndexLookup.of(index)
    defer { lookup.free() }
    if not lookup.is_unique() {
        throw ValueError("cannot reindex onto an index whose labels repeat")
    }
    if method is null { return lookup.locate(target) }
    return method_indexer(index, target, method ?? METHOD_PAD)
}

#: Write `fill` into the rows a take left empty.
#:
#: Only the rows the indexer missed, not every null: a row that was already
#: missing in the source stays missing, which is what pandas' `fill_value`
#: means and what makes a reindex distinguishable from a `fillna`.
fn fill_holes(column: Column, positions: Buffer, fill: Value?) -> void {
    if fill is null { return }
    let cell = fill ?? NA
    if cell.is_na() or column.len() == 0 { return }
    var low = 0
    var high = 0
    if column.dtype.is_dictionary() {
        low = f32_bits(float(column.intern_level(cell.to_str())))
    } elif column.dtype.is_wide() {
        let split = wide_pack(cell.to_int())
        low = f32_bits(split[0])
        high = f32_bits(split[1])
    } else {
        low = f32_bits(cell.to_float())
    }
    kernel_for("idx_fill_holes").dispatch_async(
        [column.values, wide_lane(column), column.materialize_validity(), positions],
        column.len(),
        [column.len(), flag(column.dtype.is_wide()), u32(low), u32(high)]
    )
}

#: Where each label of `target` sits once a fill method has been allowed to
#: reach for a neighbour.
fn method_indexer(index: Index, target: Index, method: str) -> Buffer {
    let mode = match method {
        "ffill" | "pad" => 0,
        "bfill" | "backfill" => 1,
        "nearest" => 2,
        _ => throw ValueError(f"unknown reindex method {method}"),
    }
    if index.nlevels() != 1 or target.nlevels() != 1 {
        throw ValueError("a fill method reads one level, not a whole multi-index")
    }
    if not is_monotonic_increasing(index) {
        throw ValueError(f"reindex with method {method} needs a monotonic increasing index")
    }
    let source = borrowed_levels(index)
    let keys = source[0][0]
    let borrowed = borrowed_levels(target)
    let raw = borrowed[0]
    let probe = recoded_like(raw[0], keys)
    let count = target.len()
    let bisect = Buffer(if count > 0 { count } else { 1 })
    let out = Buffer(if count > 0 { count } else { 1 })
    if count > 0 {
        kernel_for("idx_search").dispatch_async(
            [
                value_lane(keys),
                wide_lane(keys),
                value_lane(probe),
                wide_lane(probe),
                valid_lane(probe),
                bisect,
            ],
            count,
            [count, index.len(), flag(keys.dtype.is_wide()), flag(probe.has_validity()), 0]
        )
        kernel_for("idx_method_pick").dispatch_async(
            [
                value_lane(keys),
                wide_lane(keys),
                value_lane(probe),
                wide_lane(probe),
                valid_lane(probe),
                bisect,
                out,
            ],
            count,
            [count, index.len(), flag(keys.dtype.is_wide()), flag(probe.has_validity()), mode]
        )
    }
    bisect.free()
    if probe != raw[0] { probe.free() }
    release_all(raw, borrowed[1])
    release_all(source[0], source[1])
    return out
}

#: Writing into a selection in place, and splicing rows in or out.
from std.gpu import Buffer

from ..column import Column, DataFrame, Index, Value
from ..column import MAX_ROWS
from ..column import concat_columns, positions_buffer
from .labels import Labels, label_positions
from .lanes import flag, kernel_for, scalar_bits, valid_lane, value_lane, wide_lane
from .pick import Pick, SelectKind, clamp_at, compact_flags, keep_flags
from .pick import pick_positions, resolve_at
from .select import gather_frame

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

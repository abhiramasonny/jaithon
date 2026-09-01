#: An operation run across labelled operands: series against series, frame
#: against frame, and the two entries everything else funnels through.
from ..column import Column, DataFrame, Series, Value
from .align import align_indexes, aligned_column, shared_name
from .cells import column_cells
from .codes import binary_name, unary_name
from .dispatch import column_of, dispatch_unary, plain_binary

fn aligned_series(left: Series, right: Series, op: int, fill: Value?) -> Series {
    let plan = align_indexes(left.index, right.index)
    let a = aligned_column(left.values, plan.left, plan.length)
    let b = aligned_column(right.values, plan.right, plan.length)
    let out = plain_binary(a[0], b[0], op, fill)
    if a[1] { a[0].free() }
    if b[1] { b[0].free() }
    plan.free_positions()
    return Series(out, index: plan.index, name: shared_name(left.name, right.name))
}

fn series_binary(left: any, right: any, op: int, fill: Value?) -> Series {
    if left is Series and right is Series { return aligned_series(left, right, op, fill) }
    if left is Series {
        return Series(
            plain_binary(left.values, right, op, fill),
            index: left.index.copy(),
            name: left.name
        )
    }
    return Series(
        plain_binary(left, right.values, op, fill),
        index: right.index.copy(),
        name: right.name
    )
}

#: The union of two frames' column names, left order first.
fn union_names(a: list[str], b: list[str]) -> list[str] {
    var names: list[str] = []
    var seen: dict[str, int] = {}
    for name in a {
        seen[name] = 1
        names.push(name)
    }
    for name in b {
        if seen.get(name, -1) >= 0 { continue }
        seen[name] = 1
        names.push(name)
    }
    return names
}

fn aligned_frames(left: DataFrame, right: DataFrame, op: int, fill: Value?) -> DataFrame {
    let plan = align_indexes(left.index, right.index)
    let names = union_names(left.names, right.names)
    var columns: list[Column] = []
    for name in names {
        #: A column only one frame has meets a column of nulls, so a
        #: `fill_value` reaches it exactly as it reaches a missing row. That is
        #: what makes `a.add(b, fill_value: 0)` a union rather than an
        #: intersection.
        let dtype = if left.has(name) { left.column(name).dtype } else { right.column(name).dtype }
        let a = if left.has(name) {
            aligned_column(left.column(name), plan.left, plan.length)
        } else {
            (Column.nulls(dtype, plan.length), true)
        }
        let b = if right.has(name) {
            aligned_column(right.column(name), plan.right, plan.length)
        } else {
            (Column.nulls(dtype, plan.length), true)
        }
        columns.push(plain_binary(a[0], b[0], op, fill))
        if a[1] { a[0].free() }
        if b[1] { b[0].free() }
    }
    plan.free_positions()
    return DataFrame(names, columns, index: plan.index)
}

#: A frame against a series, which pandas matches to the column names.
#:
#: One scalar per column, so the series' values come down once and each column
#: meets a one-slot operand. A label the frame has no column for still becomes
#: a column, all null, the way pandas' union does.
fn frame_over_series(
    frame: DataFrame,
    series: Series,
    op: int,
    fill: Value?,
    frame_left: bool
) -> DataFrame {
    if series.index.nlevels() != 1 {
        throw ValueError("a frame lines up against a series of one label per column")
    }
    let levels = series.index.materialize()
    var labels: list[str] = []
    for cell in column_cells(levels[0]) { labels.push(cell.to_str()) }
    for level in levels { level.free() }
    let cells = column_cells(series.values)
    var lookup: dict[str, int] = {}
    for (at, label) in labels.enumerate() { lookup[label] = at }
    var names: list[str] = []
    var columns: list[Column] = []
    for (at, name) in frame.names.enumerate() {
        let found = lookup.get(name, -1)
        let scalar = if found >= 0 { cells[found] } else { Value.null_of(frame.columns[at].dtype) }
        names.push(name)
        columns.push(if frame_left {
            plain_binary(frame.columns[at], scalar, op, fill)
        } else {
            plain_binary(scalar, frame.columns[at], op, fill)
        })
    }
    for (at, label) in labels.enumerate() {
        if frame.has(label) { continue }
        let empty = Column.nulls(cells[at].dtype, frame.len())
        names.push(label)
        columns.push(if frame_left {
            plain_binary(empty, cells[at], op, fill)
        } else {
            plain_binary(cells[at], empty, op, fill)
        })
        empty.free()
    }
    return DataFrame(names, columns, index: frame.index.copy())
}

fn frame_over_scalar(
    frame: DataFrame,
    other: any,
    op: int,
    fill: Value?,
    frame_left: bool
) -> DataFrame {
    if column_of(other) is not null {
        throw TypeError(
            f"{binary_name(op)} between a DataFrame and a Column is ambiguous: " +
                "give the column an index and pass a Series"
        )
    }
    var columns: list[Column] = []
    for column in frame.columns {
        columns.push(if frame_left {
            plain_binary(column, other, op, fill)
        } else {
            plain_binary(other, column, op, fill)
        })
    }
    return frame._replace(columns: columns, index: frame.index.copy())
}

fn frame_binary(left: any, right: any, op: int, fill: Value?) -> DataFrame {
    if left is DataFrame and right is DataFrame { return aligned_frames(left, right, op, fill) }
    if left is DataFrame and right is Series {
        return frame_over_series(left, right, op, fill, true)
    }
    if right is DataFrame and left is Series {
        return frame_over_series(right, left, op, fill, false)
    }
    if left is DataFrame { return frame_over_scalar(left, right, op, fill, true) }
    return frame_over_scalar(right, left, op, fill, false)
}

#: The one entry every binary operation funnels through.
pub fn binary(left: any, right: any, op: int, fill: Value?) -> any {
    if left is DataFrame or right is DataFrame { return frame_binary(left, right, op, fill) }
    if left is Series or right is Series { return series_binary(left, right, op, fill) }
    return plain_binary(left, right, op, fill)
}

#: The one entry every unary operation funnels through.
pub fn unary(target: any, op: int, arg_a: Value?, arg_b: Value?) -> any {
    if target is DataFrame {
        var columns: list[Column] = []
        for column in target.columns { columns.push(dispatch_unary(column, op, arg_a, arg_b)) }
        return target._replace(columns: columns, index: target.index.copy())
    }
    if target is Series {
        return Series(
            dispatch_unary(target.values, op, arg_a, arg_b),
            index: target.index.copy(),
            name: target.name
        )
    }
    if target is Column { return dispatch_unary(target, op, arg_a, arg_b) }
    throw TypeError(f"{unary_name(op)} needs a Series, a DataFrame or a Column")
}

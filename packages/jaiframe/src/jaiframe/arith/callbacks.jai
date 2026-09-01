#: The escape hatch: a user callback over rows, columns or pairs of series.
from ..column import Column, DataFrame, Index, Series, Value
from .align import align_indexes, aligned_column, shared_name
from .cells import column_cells, column_of_cells

#: A `Str` index over a list of names, for a result labelled by column.
fn index_over_names(names: list[str]) -> Index {
    var labels: list[str?] = []
    for name in names { labels.push(name) }
    let level = Column.from_strs(labels)
    let unnamed: list[str?] = [null]
    return Index.of([level], unnamed)
}

#: `f` over every value of a series.
#:
#: A callback costs a host pass: the column comes down, the loop runs one
#: boxed `Value` per row, and the result goes back up. Reach for it when the
#: transform has no vectorised spelling; `add`, `multiply`, `clip`, `round`
#: and the rest of this module stay on the device and are the alternative.
#:
#: ```
#: map(series, |cell| if cell.is_na() { cell } else { Value.of_float(cell.to_float() * 2.0) })
#: ```
pub fn map(series: Series, f: fn(Value) -> Value) -> Series {
    var out: list[Value] = []
    for cell in column_cells(series.values) { out.push(f(cell)) }
    return Series(
        column_of_cells(out, series.dtype()),
        index: series.index.copy(),
        name: series.name
    )
}

#: `f` over each column of a frame, or each row when `axis` is 1.
#:
#: A callback costs a host pass: every column comes down once, the loop runs on
#: boxed `Value`s, and one column goes back up. The vectorised alternative for
#: a column-at-a-time summary is `reduce.jai`, and for an elementwise one it is
#: the arithmetic here.
#:
#: pandas hands the callback a `Series`; this hands it the cells. A `Series`
#: per row would be one device allocation per row, which for a frame of any
#: size is the slowest thing this package could do.
pub fn apply(frame: DataFrame, f: fn(list[Value]) -> Value, axis: int = 0) -> Series {
    if axis != 0 and axis != 1 { throw ValueError(f"axis must be 0 or 1, not {axis}") }
    var out: list[Value] = []
    if axis == 0 {
        for column in frame.columns { out.push(f(column_cells(column))) }
        return Series(column_of_cells(out), index: index_over_names(frame.names))
    }
    var downloaded: list[list[Value]] = []
    for column in frame.columns { downloaded.push(column_cells(column)) }
    for row in 0..frame.len() {
        var cells: list[Value] = []
        for column in downloaded { cells.push(column[row]) }
        out.push(f(cells))
    }
    return Series(column_of_cells(out), index: frame.index.copy())
}

#: `f` over every cell of a frame, keeping its shape.
#:
#: A callback costs a host pass, one per column: down, loop, up. An elementwise
#: transform that this module already spells — `absolute`, `clip`, `log`, a
#: comparison — stays on the device and is the alternative.
pub fn applymap(frame: DataFrame, f: fn(Value) -> Value) -> DataFrame {
    var columns: list[Column] = []
    for column in frame.columns {
        var cells: list[Value] = []
        for cell in column_cells(column) { cells.push(f(cell)) }
        columns.push(column_of_cells(cells, column.dtype))
    }
    return frame._replace(columns: columns, index: frame.index.copy())
}

#: `f(target)`, so a chain of whole-frame steps reads top to bottom.
#:
#: The one callback here that costs no host pass: nothing is downloaded, the
#: object is handed over whole, and whatever `f` does is what it costs. It is
#: in this file because it is the same escape hatch as the others — a place for
#: what the vectorised operations cannot say.
#:
#: ```
#: pipe(frame, |f| clip(f, 0.0))
#: ```
pub fn pipe(target: any, f: fn(any) -> any) -> any { return f(target) }

#: `f` over paired values of two series, aligned on their labels first.
#:
#: A callback costs a host pass: both columns come down, the loop runs on boxed
#: `Value`s, and the result goes back up. Any pairing this module already
#: spells — `add`, `power`, a comparison — is one dispatch instead, and a
#: two-argument callback that only picks a side is `where` in `missing.jai`.
pub fn combine(a: Series, b: Series, f: fn(Value, Value) -> Value) -> Series {
    let plan = align_indexes(a.index, b.index)
    let left = aligned_column(a.values, plan.left, plan.length)
    let right = aligned_column(b.values, plan.right, plan.length)
    let left_cells = column_cells(left[0])
    let right_cells = column_cells(right[0])
    var out: list[Value] = []
    for row in 0..plan.length { out.push(f(left_cells[row], right_cells[row])) }
    if left[1] { left[0].free() }
    if right[1] { right[0].free() }
    plan.free_positions()
    return Series(
        column_of_cells(out, a.dtype()),
        index: plan.index,
        name: shared_name(a.name, b.name)
    )
}

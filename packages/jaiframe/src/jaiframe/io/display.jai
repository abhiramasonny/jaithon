#: The printed table: a sheet of windowed cells, drawn as plain text or as a
#: markdown pipe table.
import std.fmt as fmt

from ..column import Column, DataFrame, Index, Series
from .render import rendered_slice

#: How many rows `to_string` prints before it truncates to a head and a tail.
pub const DEFAULT_MAX_ROWS = 60

#: How many columns `to_string` prints before it elides the middle ones.
pub const DEFAULT_MAX_COLS = 20

#: What stands in for the rows and the columns a truncated view left out.
pub let ELLIPSIS = "..."

#: A laid-out table on its way to text: titles, cells, and which lanes are
#: numeric.
#:
#: Both renderings build one of these and neither builds the other's. The
#: cells are already strings and already windowed, so nothing downstream of
#: here knows how tall the frame was — which is what keeps printing a
#: ten-million-row frame the same cost as printing a sixty-row one.
class Sheet {
    #: One title per lane, the leading `gutter` of them naming index levels.
    pub var titles: list[str]
    #: One list of rendered cells per lane, all the same length.
    pub var lanes: list[list[str]]
    #: Whether a lane's cells are numbers, and so want the right margin.
    pub var numeric: list[bool]
    #: How many leading lanes are index levels rather than data.
    pub var gutter: int
    #: Rows and columns the frame has, which a truncated view still reports.
    pub var shape: (int, int)
    #: Whether rows or columns were left out.
    pub var truncated: bool

    pub fn init(self) {
        self.titles = []
        self.lanes = []
        self.numeric = []
        self.gutter = 0
        self.shape = (0, 0)
        self.truncated = false
    }

    pub fn height(self) -> int {
        return if self.lanes.len() == 0 { 0 } else { self.lanes[0].len() }
    }
}

#: How many rows to show at the top and at the bottom.
#:
#: A non-positive `max_rows` means all of them, which is what a file-bound
#: rendering like `to_markdown` wants and a terminal-bound one does not.
fn window(length: int, max_rows: int) -> (int, int) {
    if max_rows <= 0 or length <= max_rows { return (length, 0) }
    let head = (max_rows + 1) // 2
    return (head, max_rows - head)
}

#: A column's head and tail cells with an ellipsis between them.
fn windowed_cells(
    column: Column,
    head: int,
    tail: int,
    na_rep: str,
    float_format: str?
) -> list[str] {
    var cells = rendered_slice(column, 0, head, na_rep, float_format)
    if tail == 0 { return cells }
    cells.push(ELLIPSIS)
    for cell in rendered_slice(column, column.len() - tail, tail, na_rep, float_format) {
        cells.push(cell)
    }
    return cells
}

#: The labels of a range index, computed rather than read.
#:
#: The point of not storing the ramp is that most frames never ask for it, and
#: a printed frame is the case that would ask for ten million labels to show
#: sixty of them. Arithmetic on the two numbers it is made of costs nothing.
fn range_labels(index: Index, head: int, tail: int) -> list[str] {
    var out: list[str] = []
    for at in 0..head { out.push(f"{index.start + at * index.step}") }
    if tail == 0 { return out }
    out.push(ELLIPSIS)
    let first = index.length - tail
    for at in first..index.length { out.push(f"{index.start + at * index.step}") }
    return out
}

#: Put the index levels into a sheet as its leading lanes.
fn add_gutter(
    sheet: Sheet,
    index: Index,
    head: int,
    tail: int,
    na_rep: str,
    float_format: str?
) -> void {
    if index.is_range() {
        sheet.titles.push(index.names[0] ?? "")
        sheet.lanes.push(range_labels(index, head, tail))
        sheet.numeric.push(false)
        sheet.gutter = 1
        return
    }
    for (at, level) in index.levels.enumerate() {
        sheet.titles.push(index.names[at] ?? "")
        sheet.lanes.push(windowed_cells(level, head, tail, na_rep, float_format))
        sheet.numeric.push(false)
    }
    sheet.gutter = index.levels.len()
}

#: A frame laid out for printing.
#:
#: Columns are elided the way rows are: the leading and trailing ones survive
#: and an ellipsis lane stands where the middle was.
fn sheet_of_frame(
    frame: DataFrame,
    max_rows: int,
    max_cols: int,
    float_format: str?,
    na_rep: str,
    with_index: bool
) -> Sheet {
    let (head, tail) = window(frame.len(), max_rows)
    let (left, right) = window(frame.width(), max_cols)
    let sheet = Sheet()
    sheet.shape = frame.shape()
    sheet.truncated = tail > 0 or right > 0
    if with_index { add_gutter(sheet, frame.index, head, tail, na_rep, float_format) }
    let rows = head + tail + (if tail > 0 { 1 } else { 0 })
    for at in 0..left {
        sheet.titles.push(frame.names[at])
        sheet.lanes.push(windowed_cells(frame.columns[at], head, tail, na_rep, float_format))
        sheet.numeric.push(not frame.columns[at].dtype.is_dictionary())
    }
    if right > 0 {
        sheet.titles.push(ELLIPSIS)
        sheet.lanes.push([ELLIPSIS for _row in 0..rows])
        sheet.numeric.push(false)
        for at in frame.width() - right..frame.width() {
            sheet.titles.push(frame.names[at])
            sheet.lanes.push(windowed_cells(frame.columns[at], head, tail, na_rep, float_format))
            sheet.numeric.push(not frame.columns[at].dtype.is_dictionary())
        }
    }
    return sheet
}

#: One column laid out for printing, under whatever labels it has.
fn sheet_of_column(
    values: Column,
    index: Index,
    name: str?,
    max_rows: int,
    float_format: str?,
    na_rep: str,
    with_index: bool
) -> Sheet {
    let (head, tail) = window(values.len(), max_rows)
    let sheet = Sheet()
    sheet.shape = (values.len(), 1)
    sheet.truncated = tail > 0
    if with_index { add_gutter(sheet, index, head, tail, na_rep, float_format) }
    sheet.titles.push(name ?? "")
    sheet.lanes.push(windowed_cells(values, head, tail, na_rep, float_format))
    sheet.numeric.push(not values.dtype.is_dictionary())
    return sheet
}

fn justified(text: str, width: int, how: str) -> str {
    return match how {
        "left" => fmt.align_left(text, width),
        "right" => fmt.align_right(text, width),
        "center" => fmt.center(text, width),
        _ => throw ValueError(f"justify must be left, right or center, not {repr(how)}"),
    }
}

#: A sheet as a padded plain-text table.
fn draw_sheet(sheet: Sheet, justify: str, headed: bool) -> str {
    var widths: list[int] = []
    for (at, title) in sheet.titles.enumerate() {
        var width = if headed { fmt.display_width(title) } else { 0 }
        for cell in sheet.lanes[at] {
            let measured = fmt.display_width(cell)
            if measured > width { width = measured }
        }
        widths.push(width)
    }
    var lines: list[str] = []
    if headed {
        var titles: list[str] = []
        for (at, title) in sheet.titles.enumerate() {
            titles.push(justified(title, widths[at], if at < sheet.gutter {
                "left"
            } else {
                justify
            }))
        }
        lines.push("  ".join(titles).rstrip())
    }
    for row in 0..sheet.height() {
        var cells: list[str] = []
        for (at, lane) in sheet.lanes.enumerate() {
            #: The index gutter and text columns hang on the left margin, and
            #: numbers on the right, which is where a reader compares them.
            let how = if at < sheet.gutter or not sheet.numeric[at] { "left" } else { "right" }
            cells.push(justified(lane[row], widths[at], how))
        }
        lines.push("  ".join(cells).rstrip())
    }
    return "\n".join(lines)
}

#: A sheet as a GitHub-flavoured pipe table.
fn draw_markdown(sheet: Sheet) -> str {
    var widths: list[int] = []
    for (at, title) in sheet.titles.enumerate() {
        var width = fmt.display_width(title)
        if width < 3 { width = 3 }
        for cell in sheet.lanes[at] {
            let measured = fmt.display_width(cell)
            if measured > width { width = measured }
        }
        widths.push(width)
    }
    var lines: list[str] = []
    var titles: list[str] = []
    var rules: list[str] = []
    for (at, title) in sheet.titles.enumerate() {
        let right = at >= sheet.gutter and sheet.numeric[at]
        titles.push(justified(title, widths[at], if right { "right" } else { "left" }))
        let dashes = "-".repeat(widths[at] - 1)
        rules.push(if right { dashes + ":" } else { ":" + dashes })
    }
    lines.push("| " + " | ".join(titles) + " |")
    lines.push("| " + " | ".join(rules) + " |")
    for row in 0..sheet.height() {
        var cells: list[str] = []
        for (at, lane) in sheet.lanes.enumerate() {
            let right = at >= sheet.gutter and sheet.numeric[at]
            cells.push(justified(lane[row], widths[at], if right { "right" } else { "left" }))
        }
        lines.push("| " + " | ".join(cells) + " |")
    }
    return "\n".join(lines)
}

fn quoted_cells(cells: list[str], quote: bool) -> list[str] {
    if not quote { return cells }
    return cells.map(|cell| if cell == ELLIPSIS { cell } else { "\"" + cell + "\"" })
}

fn index_string(index: Index, max_rows: int, float_format: str?, na_rep: str) -> str {
    let named = index.names[0]
    if index.is_range() {
        let stop = index.start + index.length * index.step
        var out = f"RangeIndex(start={index.start}, stop={stop}, step={index.step}"
        if named is not null { out = out + f", name={repr(named ?? "")}" }
        return out + ")"
    }
    let (head, tail) = window(index.length, max_rows)
    var levels: list[list[str]] = []
    for level in index.levels {
        levels.push(
            quoted_cells(
                windowed_cells(level, head, tail, na_rep, float_format),
                level.dtype.is_dictionary()
            )
        )
    }
    if levels.len() == 1 {
        var out = f"Index([{", ".join(levels[0])}], dtype={repr(index.levels[0].dtype.name())}"
        if named is not null { out = out + f", name={repr(named ?? "")}" }
        return out + ")"
    }
    var labels: list[str] = []
    for row in 0..levels[0].len() {
        var parts: list[str] = []
        for level in levels { parts.push(level[row]) }
        labels.push("(" + ", ".join(parts) + ")")
    }
    var names: list[str] = []
    for name in index.names { names.push(if name is null { "null" } else { repr(name ?? "") }) }
    return f"MultiIndex([{", ".join(labels)}], names=[{", ".join(names)}])"
}

fn frame_string(
    frame: DataFrame,
    max_rows: int,
    max_cols: int,
    float_format: str?,
    justify: str,
    na_rep: str
) -> str {
    if frame.width() == 0 or frame.len() == 0 {
        var lines = ["Empty DataFrame", "Columns: [" + ", ".join(frame.names) + "]"]
        lines.push("Index: " + index_string(frame.index, max_rows, float_format, na_rep))
        return "\n".join(lines)
    }
    let sheet = sheet_of_frame(frame, max_rows, max_cols, float_format, na_rep, true)
    let body = draw_sheet(sheet, justify, true)
    if not sheet.truncated { return body }
    let (rows, columns) = sheet.shape
    return body + f"\n\n[{rows} rows x {columns} columns]"
}

fn column_string(
    values: Column,
    index: Index,
    name: str?,
    max_rows: int,
    float_format: str?,
    justify: str,
    na_rep: str
) -> str {
    var footer: list[str] = []
    if name is not null { footer.push(f"Name: {name ?? ""}") }
    if values.len() == 0 {
        footer.push("Empty")
    }
    let sheet = sheet_of_column(values, index, name, max_rows, float_format, na_rep, true)
    if sheet.truncated { footer.push(f"Length: {values.len()}") }
    footer.push(f"dtype: {values.dtype.name()}")
    let tail = ", ".join(footer)
    if values.len() == 0 { return tail }
    #: A series has no header row: its name belongs in the footer, where the
    #: dtype is, and a header would put it somewhere a reader has to look
    #: twice to attach it to anything.
    return draw_sheet(sheet, justify, false) + "\n" + tail
}

#: A frame, series, column or index as printable text.
#:
#: pandas' truncation: past `max_rows` the head and the tail survive with an
#: ellipsis row between them, and past `max_cols` the leading and trailing
#: columns do with an ellipsis column. A non-positive limit means no limit.
#: Numbers sit against the right margin and text against the left, which is
#: where each is read from; `justify` moves the column titles and takes
#: `"left"`, `"right"` or `"center"`.
#:
#: `float_format` is a `str.format` spec, and without one a float prints as
#: the shortest decimal that reads back as the same float32. A missing cell
#: prints as `na_rep`.
#:
#: Only the rows it shows are read: a truncated view downloads the head slice
#: and the tail slice of each lane and nothing between them.
pub fn to_string(
    target: any,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_cols: int = DEFAULT_MAX_COLS,
    float_format: str? = null,
    justify: str = "right",
    na_rep: str = "NaN"
) -> str {
    if isinstance(target, DataFrame) {
        return frame_string(target, max_rows, max_cols, float_format, justify, na_rep)
    }
    if isinstance(target, Series) {
        return column_string(
            target.values,
            target.index,
            target.name,
            max_rows,
            float_format,
            justify,
            na_rep
        )
    }
    if isinstance(target, Column) {
        return column_string(
            target,
            Index.range(target.len()),
            null,
            max_rows,
            float_format,
            justify,
            na_rep
        )
    }
    if isinstance(target, Index) { return index_string(target, max_rows, float_format, na_rep) }
    throw TypeError(f"{repr(target)} is not a frame, a series, a column or an index")
}

#: A frame, series or column as a GitHub-flavoured markdown table.
#:
#: `max_rows` defaults to no limit: a markdown table goes into a document
#: rather than onto a screen, and a document that silently dropped the middle
#: of its data would be a worse document. The alignment colons follow the
#: dtypes, so a numeric column right-aligns wherever the table is rendered.
pub fn to_markdown(
    target: any,
    index: bool = true,
    max_rows: int = 0,
    float_format: str? = null,
    na_rep: str = ""
) -> str {
    if isinstance(target, DataFrame) {
        return draw_markdown(sheet_of_frame(target, max_rows, 0, float_format, na_rep, index))
    }
    if isinstance(target, Series) {
        return draw_markdown(
            sheet_of_column(
                target.values,
                target.index,
                target.name,
                max_rows,
                float_format,
                na_rep,
                index
            )
        )
    }
    if isinstance(target, Column) {
        return draw_markdown(
            sheet_of_column(
                target,
                Index.range(target.len()),
                null,
                max_rows,
                float_format,
                na_rep,
                index
            )
        )
    }
    throw TypeError(f"{repr(target)} is not a frame, a series or a column")
}

#: What a frame costs: slots per lane, bytes per column, and the `info`
#: summary that reports them.
import std.fmt as fmt
from std.gpu import Buffer

from ..column import Column, DataFrame, Index, Series
from ..column import wide_unpack
from .display import to_string
from .kernels import BYTES_GROUP, BYTES_GROUPS, compiled, flag

#: How many float32 slots a column occupies, over all the lanes it has.
#:
#: This is the honest unit for this package: a column is one contiguous buffer
#: per lane and nothing else, so its footprint is a slot count and everything
#: else is arithmetic on it. `Int64` and `Datetime` pay a second lane for their
#: exactness; a column that has never held a null pays for no mask.
pub fn buffer_slots(column: Column) -> int {
    var lanes = 1
    if column.dtype.is_wide() { lanes += 1 }
    if column.has_validity() { lanes += 1 }
    return column.len() * lanes
}

#: How many slots an index occupies. A range index occupies none: its labels
#: are two integers and the arithmetic between them.
pub fn index_slots(index: Index) -> int {
    var total = 0
    for level in index.levels { total += buffer_slots(level) }
    return total
}

#: The bytes the strings of a dictionary column occupy, counted once per row.
#:
#: What a `Str` column would cost if it were not dictionary-encoded, which is
#: the number `deep` is asking for. Reduced on the device — one gather out of
#: the level table per row and one fold — so a ten-million-row column costs a
#: dispatch and a download of a few hundred partials, not ten million host
#: string lengths.
fn text_bytes(column: Column) -> int {
    let levels = column.levels ?? []
    if column.length == 0 or levels.len() == 0 { return 0 }
    let lengths = Buffer(levels.len())
    lengths.upload(levels.map(|level| float(bytes(level).len())))
    let low = Buffer(BYTES_GROUPS)
    let high = Buffer(BYTES_GROUPS)
    defer {
        lengths.free()
        low.free()
        high.free()
    }
    compiled("io_text_bytes").dispatch_async(
        [column.values, column.valid ?? column.values, lengths, low, high],
        BYTES_GROUPS * BYTES_GROUP,
        [column.length, levels.len(), flag(column.has_validity())],
        group_size: BYTES_GROUP
    )
    let lows = low.download()
    let highs = high.download()
    var total = 0
    for (at, slot) in lows.enumerate() { total += wide_unpack(slot, highs[at]) }
    return total
}

#: How many bytes a column occupies.
pub fn column_bytes(column: Column, deep: bool = false) -> int {
    var total = buffer_slots(column) * 4
    if deep and column.dtype.is_dictionary() { total += text_bytes(column) }
    return total
}

#: What each column of a frame costs, in bytes, as a series over the names.
#:
#: A slot is four bytes, so every number here is `buffer_slots` times four; the
#: slot count is the thing the package actually allocates and the byte count is
#: the thing a reader is asking about. `index` prepends the index's own cost,
#: which for the default range index is zero.
#:
#: `deep` adds what a dictionary column's strings occupy — the size the column
#: would have been without the encoding — counted on the device rather than by
#: walking the rows.
pub fn memory_usage(frame: DataFrame, index: bool = true, deep: bool = false) -> Series {
    var names: list[str?] = []
    var counts: list[int] = []
    if index {
        names.push("Index")
        counts.push(index_slots(frame.index) * 4)
    }
    for (at, column) in frame.columns.enumerate() {
        names.push(frame.names[at])
        counts.push(column_bytes(column, deep))
    }
    let labels: list[str?] = [null]
    return Series(
        Column.from_ints(counts),
        index: Index.of([Column.from_strs(names)], labels),
        name: "memory_usage"
    )
}

#: A summary of a frame: its shape, its index, and every column's dtype and
#: how many of its rows are present.
#:
#: The non-null counts are the device's answer — one reduction per column
#: through `Column.null_count` — so this costs a dispatch a column rather than
#: a download of the whole table.
pub fn info(frame: DataFrame, deep: bool = false) -> str {
    var lines = [f"<DataFrame: {frame.len()} rows x {frame.width()} columns>"]
    lines.push("Index: " + to_string(frame.index, max_rows: 6))
    var rows: list[list[str]] = [["#", "Column", "Non-Null Count", "Dtype"]]
    var tally: dict[str, int] = {}
    for (at, column) in frame.columns.enumerate() {
        let present = column.len() - column.null_count()
        rows.push([f"{at}", frame.names[at], f"{present} non-null", column.dtype.name()])
        let name = column.dtype.name()
        tally[name] = tally.get(name, 0) + 1
    }
    var widths = [0, 0, 0, 0]
    for row in rows {
        for (at, cell) in row.enumerate() {
            let measured = fmt.display_width(cell)
            if measured > widths[at] { widths[at] = measured }
        }
    }
    for row in rows {
        var padded: list[str] = []
        for (at, cell) in row.enumerate() { padded.push(fmt.align_left(cell, widths[at])) }
        lines.push(" ".join(padded).rstrip())
    }
    var kinds = tally.keys()
    kinds.sort()
    var counted: list[str] = []
    for kind in kinds { counted.push(f"{kind}({tally.get(kind, 0)})") }
    lines.push("dtypes: " + ", ".join(counted))
    var slots = index_slots(frame.index)
    var total = slots * 4
    for column in frame.columns {
        slots += buffer_slots(column)
        total += column_bytes(column, deep)
    }
    lines.push(f"memory usage: {total} bytes over {slots} slots")
    return "\n".join(lines)
}

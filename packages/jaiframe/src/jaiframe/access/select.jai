#: The reads themselves: `iloc`, `loc`, the scalars, the ends, and `sample`.
from std.gpu import Buffer
from std.random import Random

from ..column import Column, DataFrame, Index, Series, Value
from ..column import positions_buffer
from .labels import Labels, key_positions, label_positions, label_text
from .lanes import kernel_for
from .pick import Pick, compact_flags, mask_positions, pick_host, pick_positions
from .pick import resolve_at

pub fn gather_frame(frame: DataFrame, lane: Buffer, count: int) -> DataFrame {
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

pub fn narrow_columns(frame: DataFrame, at: list[int]) -> DataFrame {
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

#: What a frame, a series and a bare column look like to the calls here.
from std.gpu import Buffer

from ..column import Column, DataFrame, Dtype, Index, Series
from ..column import f32_bits
from .lanes import flag, kernel_for
from .table import reduce_lane

pub fn columns_of(target: any) -> list[Column] {
    if target is Column { return [target] }
    if target is Series { return [target.values] }
    if target is DataFrame { return [column for column in target.columns] }
    throw TypeError("expected a Column, a Series or a DataFrame")
}

pub fn names_of(target: any) -> list[str?] {
    var out: list[str?] = []
    if target is Series {
        out.push(target.name)
        return out
    }
    if target is DataFrame {
        for name in target.names { out.push(name) }
        return out
    }
    out.push(null)
    return out
}

pub fn label_of(target: any) -> str? {
    if target is Series { return target.name }
    return null
}

pub fn row_count(target: any) -> int {
    if target is Column { return target.len() }
    if target is Series { return target.len() }
    if target is DataFrame { return target.len() }
    throw TypeError("expected a Column, a Series or a DataFrame")
}

pub fn index_of(target: any) -> Index {
    if target is Series { return target.index }
    if target is DataFrame { return target.index }
    throw TypeError("a bare Column carries no labels to sort by")
}

fn named_columns(frame: DataFrame, names: any) -> list[Column] {
    if names is str { return [frame.column(names)] }
    if names is list {
        var out: list[Column] = []
        for name in names { out.push(frame.column(name)) }
        if out.len() == 0 { throw ValueError("an empty list names no columns") }
        return out
    }
    throw TypeError("columns are named by a string or a list of strings")
}

pub fn subset_columns(frame: any, subset: any) -> list[Column] {
    if subset is null { return columns_of(frame) }
    if frame is DataFrame { return named_columns(frame, subset) }
    throw TypeError("a subset names the columns of a frame")
}

pub fn sort_key_columns(target: any, by: any) -> list[Column] {
    if target is DataFrame {
        if by is null { throw ValueError("sorting a frame needs the columns to sort by") }
        return named_columns(target, by)
    }
    if by is not null {
        throw ValueError("a series or a column sorts by its own values, not by a name")
    }
    return columns_of(target)
}

pub fn spread_any(ascending: any, wanted: int) -> list[bool]? {
    if ascending is bool {
        var out: list[bool] = []
        for _at in 0..wanted { out.push(ascending) }
        return out
    }
    if ascending is list {
        var out: list[bool] = []
        for value in ascending { out.push(value) }
        return out
    }
    throw TypeError("ascending is a bool or one bool per key")
}

pub fn duplicate_mode(keep: any) -> int {
    if keep is bool {
        if not keep { return 2 }
        throw ValueError("keep is first, last or false")
    }
    if keep == "first" { return 0 }
    if keep == "last" { return 1 }
    throw ValueError(f"keep must be first, last or false, not {keep}")
}

#: The rows `positions` names, carrying whatever labels the target had.
#:
#: One positions lane serves the columns and the index alike, which is why
#: every reordering here ends in a `take` rather than a filter: `compress`
#: would leave the index behind.
pub fn reorder(target: any, positions: Buffer, length: int, ignore_index: bool) -> any {
    if target is Column { return target.take(positions, length) }
    if target is Series {
        let labels = if ignore_index {
            Index.range(length)
        } else {
            target.index.take(positions, length)
        }
        return Series(target.values.take(positions, length), index: labels, name: target.name)
    }
    if target is DataFrame {
        let labels = if ignore_index {
            Index.range(length)
        } else {
            target.index.take(positions, length)
        }
        return target._replace(
            columns: target.columns.map(|column| column.take(positions, length)),
            index: labels
        )
    }
    throw TypeError("expected a Column, a Series or a DataFrame")
}

fn level_position(index: Index, level: any) -> int {
    if level is int {
        let at = if level < 0 { index.nlevels() + level } else { level }
        if at < 0 or at >= index.nlevels() {
            throw IndexError(f"level {level} is outside an index of {index.nlevels()}")
        }
        return at
    }
    if level is str {
        for (at, name) in index.names.enumerate() {
            if name == level { return at }
        }
        throw KeyError(f"no index level named {level}")
    }
    throw TypeError("a level is named by a position or a name")
}

pub fn resolve_levels(index: Index, level: any) -> list[int] {
    if level is null {
        var all: list[int] = []
        for at in 0..index.nlevels() { all.push(at) }
        return all
    }
    if level is list {
        var chosen: list[int] = []
        for entry in level { chosen.push(level_position(index, entry)) }
        return chosen
    }
    return [level_position(index, level)]
}

pub fn descending_positions(length: int) -> Buffer {
    let lane = Buffer(length)
    if length > 0 { kernel_for("key_descending").dispatch_async([lane], length, [length]) }
    return lane
}

#: An `Int64` column holding `length - 1` down to 0.
#:
#: The tiebreak that turns `keep: last` into an ordinary key: a stable sort
#: already leaves ties in row order, so reversing the row number as the least
#: significant key is the whole of "prefer the later row".
pub fn descending_column(length: int) -> Column {
    let column = Column.allocate(Dtype.Int64, length)
    if length > 0 {
        kernel_for("key_descending").dispatch_async([column.values], length, [length])
        (column.wide ?? column.values).fill_zero()
    }
    return column
}

#: Counts over equal-width intervals of a numeric column.
#:
#: The edges and the labels are worked out on the host, over `bins` of them and
#: never over the rows: the rows are binned by one atomic tally, and the only
#: thing downloaded is the `bins` counters.
pub fn binned_counts(
    column: Column,
    bins: any,
    normalize: bool,
    sort: bool,
    ascending: bool,
    named: str?
) -> Series {
    if bins is not int { throw TypeError("bins is a number of intervals") }
    if bins <= 0 { throw ValueError(f"cannot cut a column into {bins} intervals") }
    if not column.dtype.is_numeric() {
        throw TypeError(f"cannot bin a {column.dtype.name()} column")
    }
    let count = column.len()
    let borrowed = column.borrow_float_lane()
    let values = borrowed[0]
    let owned = borrowed[1]
    let mask = column.valid
    var low = reduce_lane(values, mask, count, 1)
    var high = reduce_lane(values, mask, count, 2)
    #: No present row leaves the reductions at their identities, which are the
    #: two ends of float32 and would produce edges nothing could fall between.
    if high < low {
        low = 0.0
        high = 0.0
    }
    #: pandas widens the first interval by a thousandth of the range so the
    #: smallest value falls inside a right-closed one, which is why its first
    #: label always looks slightly too low.
    let span = high - low
    let pad = if span > 0.0 { span * 0.001 } else { 0.001 }
    let first_edge = low - pad
    let step = (high - first_edge) / float(bins)
    let inverse = if step > 0.0 { 1.0 / step } else { 0.0 }
    let tally = Buffer(bins)
    tally.fill_zero()
    if count > 0 {
        kernel_for("key_bin_tally").dispatch_async(
            [values, mask ?? values, tally],
            count,
            [count, bins, flag(mask is not null), f32_bits(first_edge), f32_bits(inverse)]
        )
    }
    let read = Buffer(bins)
    kernel_for("key_uint_to_float").dispatch_async([tally, read], bins, [bins])
    let counted = read.download()
    read.free()
    tally.free()
    if owned { values.free() }

    var labels: list[str?] = []
    var totals: list[float] = []
    let scale = if normalize and count > 0 { 1.0 / float(count) } else { 1.0 }
    for at in 0..bins {
        let lower = first_edge + float(at) * step
        let upper = if at == bins - 1 { high } else { first_edge + float(at + 1) * step }
        labels.push(f"({lower}, {upper}]")
        totals.push(counted[at] * scale)
    }
    var order: list[int] = []
    for at in 0..bins { order.push(at) }
    #: `bins` is a handful, so ordering them on the host is cheaper than the
    #: dispatch a device sort would cost.
    if sort { order = order.sort(key: |at| totals[at], reverse: not ascending) }
    var final_labels: list[str?] = []
    var final_counts: list[float] = []
    for at in order {
        final_labels.push(labels[at])
        final_counts.push(totals[at])
    }
    let edges = Column.from_strs(final_labels)
    let values_out = if normalize {
        Column.from_floats(final_counts)
    } else {
        Column.from_ints(final_counts.map(|total| int(total + 0.5)))
    }
    return Series(values_out, index: Index.of([edges], [named]), name: named_result(normalize))
}

fn named_result(normalize: bool) -> str {
    return if normalize { "proportion" } else { "count" }
}

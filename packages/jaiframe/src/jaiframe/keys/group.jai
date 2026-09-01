#: Row identity: which rows carry the same key, and what follows from that.
from std.gpu import Buffer

from ..column import Column, Dtype, Index, Series
from ..column import f32_bits
from .encode import encode_keys
from .frames import binned_counts, columns_of, duplicate_mode, label_of
from .frames import names_of, reorder, subset_columns
from .lanes import kernel_for
from .sort import argsort
from .table import group_rows, kept_rows, positions_where, reduce_lane
from .table import tally_groups

#: The rows that carry the same key, numbered by first appearance.
#:
#: The one call group-by and join both make. `codes[row]` is which group the
#: row belongs to, `uniques` is one row per group in code order, and
#: `group_count` is how many there are. `dropna` gives a row whose key is
#: missing the code `NO_GROUP` and keeps it out of the uniques, which is what
#: `groupby(dropna: true)` and pandas' `factorize` both mean by it.
#:
#: A hash table on the device: one probe sequence per row, a compare-and-
#: exchange to claim an empty slot, and a second pass that replaces whichever
#: thread happened to win with the lowest row in the group — because pandas
#: numbers groups by where they first appear and the device's scheduling is
#: not that order. The table is sized at least 1.5x the rows so a probe is a
#: step or two, which costs 8 MB per million rows and is the trade this makes
#: against a longer walk.
#:
#: `sort` renumbers the groups into key order afterwards. It sorts the uniques,
#: not the rows, so it costs a radix sort over the distinct values.
pub fn factorize(
    columns: list[Column],
    sort: bool = false,
    dropna: bool = false
) -> (Buffer, list[Column], int) {
    let keys = encode_keys(columns)
    defer { keys.free() }
    let grouped = group_rows(keys, dropna)
    let codes = grouped[0]
    let heads = grouped[1]
    let group_count = grouped[2]
    defer { heads.free() }

    let uniques = columns.map(|column| column.take(heads, group_count))
    if not sort or group_count == 0 { return (codes, uniques, group_count) }

    let distinct = keys.gather(heads, group_count)
    defer { distinct.free() }
    let order = distinct.order()
    defer { order.free() }
    let renumbered = Buffer(group_count)
    kernel_for("key_invert_order").dispatch_async([order, renumbered], group_count, [group_count])
    let mapped = Buffer(keys.length)
    if keys.length > 0 {
        kernel_for("key_remap_codes").dispatch_async(
            [codes, renumbered, mapped],
            keys.length,
            [keys.length]
        )
    }
    renumbered.free()
    codes.free()
    let sorted_uniques = uniques.map(|column| column.take(order, group_count))
    for column in uniques { column.free() }
    return (mapped, sorted_uniques, group_count)
}

#: The distinct key rows, in order of first appearance.
#:
#: pandas' `Series.unique`, which does not sort — `factorize(sort: true)` is
#: the call that does.
pub fn unique(target: any) -> Column {
    let columns = columns_of(target)
    if columns.len() != 1 {
        throw ValueError(f"unique takes one column, not {columns.len()}")
    }
    let found = factorize(columns, dropna: false)
    found[0].free()
    return found[1][0]
}

#: How many distinct key rows there are.
pub fn nunique(target: any, dropna: bool = true) -> int {
    let columns = columns_of(target)
    let found = factorize(columns, dropna: dropna)
    found[0].free()
    for column in found[1] { column.free() }
    return found[2]
}

#: How often each distinct key row occurs.
#:
#: The result is a `Series` of counts labelled by the values counted, named
#: `count` — or `proportion` when `normalize` divides through by the total, as
#: pandas names it. `bins` instead cuts a numeric column into that many
#: equal-width right-closed intervals and counts those, including the empty
#: ones; the first edge is nudged down by a thousandth of the range so the
#: smallest value falls inside it, which is what pandas does and why its first
#: interval always looks slightly too wide.
pub fn value_counts(
    target: any,
    normalize: bool = false,
    sort: bool = true,
    ascending: bool = false,
    dropna: bool = true,
    bins: any = null
) -> Series {
    let columns = columns_of(target)
    if bins is not null {
        if columns.len() != 1 {
            throw ValueError("bins counts one numeric column, not a whole frame")
        }
        return binned_counts(columns[0], bins, normalize, sort, ascending, label_of(target))
    }
    let keys = encode_keys(columns)
    defer { keys.free() }
    let grouped = group_rows(keys, dropna)
    let codes = grouped[0]
    let heads = grouped[1]
    let group_count = grouped[2]
    defer { codes.free() }
    defer { heads.free() }

    let tally = tally_groups(codes, keys.length, group_count)
    var counts = Column.allocate(Dtype.Int64, group_count)
    if group_count > 0 {
        kernel_for("key_uint_to_float").dispatch_async(
            [tally, counts.values],
            group_count,
            [group_count]
        )
        (counts.wide ?? counts.values).fill_zero()
    }
    tally.free()
    if normalize {
        let total = float(kept_rows(keys, dropna))
        let shares = Column.allocate(Dtype.Float64, group_count)
        if group_count > 0 {
            let scale = if total > 0.0 { 1.0 / total } else { 0.0 }
            kernel_for("key_scale").dispatch_async(
                [counts.values, shares.values],
                group_count,
                [group_count, f32_bits(scale)]
            )
        }
        counts.free()
        counts = shares
    }

    var labels = columns.map(|column| column.take(heads, group_count))
    let named = if normalize { "proportion" } else { "count" }
    if not sort { return counted_series(labels, counts, names_of(target), named) }

    let order = argsort([counts], [ascending])
    defer { order.free() }
    let ranked = labels.map(|column| column.take(order, group_count))
    for column in labels { column.free() }
    let ordered_counts = counts.take(order, group_count)
    counts.free()
    return counted_series(ranked, ordered_counts, names_of(target), named)
}

#: The most frequent value or values, ascending.
#:
#: Several when they tie, as pandas returns. Built on the sorted factorization
#: so the ties come out in value order without a second sort.
pub fn mode(target: any, dropna: bool = true) -> Column {
    let columns = columns_of(target)
    if columns.len() != 1 { throw ValueError("mode takes one column") }
    let keys = encode_keys(columns)
    defer { keys.free() }
    let grouped = group_rows(keys, dropna)
    let codes = grouped[0]
    let heads = grouped[1]
    let group_count = grouped[2]
    defer { codes.free() }
    defer { heads.free() }
    if group_count == 0 { return Column.allocate(columns[0].dtype, 0) }

    let tally = tally_groups(codes, keys.length, group_count)
    let counts = Buffer(group_count)
    kernel_for("key_uint_to_float").dispatch_async([tally, counts], group_count, [group_count])
    tally.free()
    defer { counts.free() }
    let best = reduce_lane(counts, null, group_count, 2)
    let flags = Buffer(group_count)
    kernel_for("key_flag_equal").dispatch_async(
        [counts, flags],
        group_count,
        [group_count, f32_bits(best)]
    )
    defer { flags.free() }
    let picked = positions_where(flags, group_count)
    let winners = picked[0]
    defer { winners.free() }
    let rows = columns[0].take(heads, group_count)
    defer { rows.free() }
    let tied = rows.take(winners, picked[1])
    defer { tied.free() }
    let order = argsort([tied])
    defer { order.free() }
    return tied.take(order, picked[1])
}

#: True where a row repeats one already seen.
#:
#: `keep` is `"first"` for every occurrence after the first, `"last"` for every
#: one before the last, and `false` for every row of a repeated key. Missing
#: keys count as values and as equal to each other, which is what pandas'
#: `duplicated` does and the opposite of what a `dropna` group-by does.
pub fn duplicated(frame: any, subset: any = null, keep: any = "first") -> Column {
    let columns = subset_columns(frame, subset)
    let mode_code = duplicate_mode(keep)
    let keys = encode_keys(columns)
    defer { keys.free() }
    let grouped = group_rows(keys, false)
    let codes = grouped[0]
    let heads = grouped[1]
    let group_count = grouped[2]
    defer { codes.free() }
    defer { heads.free() }

    let out = Column.allocate(Dtype.Bool, keys.length)
    if keys.length == 0 { return out }
    let last = Buffer(if group_count > 0 { group_count } else { 1 })
    let tally = Buffer(if group_count > 0 { group_count } else { 1 })
    last.fill_zero()
    tally.fill_zero()
    kernel_for("key_group_extents").dispatch_async([codes, last, tally], keys.length, [keys.length])
    kernel_for("key_duplicated").dispatch_async(
        [codes, heads, last, tally, out.values],
        keys.length,
        [keys.length, mode_code]
    )
    last.free()
    tally.free()
    return out
}

#: The frame with repeated key rows removed.
#:
#: The survivors are gathered rather than compressed, so the index travels with
#: them: one positions lane feeds `Column.take` and `Index.take` alike.
pub fn drop_duplicates(
    frame: any,
    subset: any = null,
    keep: any = "first",
    ignore_index: bool = false
) -> any {
    let marks = duplicated(frame, subset, keep)
    defer { marks.free() }
    let flags = Buffer(marks.len())
    if marks.len() > 0 {
        kernel_for("key_not").dispatch_async([marks.values, flags], marks.len(), [marks.len()])
    }
    defer { flags.free() }
    let picked = positions_where(flags, marks.len())
    let survivors = picked[0]
    defer { survivors.free() }
    return reorder(frame, survivors, picked[1], ignore_index)
}

fn counted_series(labels: list[Column], counts: Column, names: list[str?], named: str) -> Series {
    return Series(counts, index: Index.of(labels, names), name: named)
}

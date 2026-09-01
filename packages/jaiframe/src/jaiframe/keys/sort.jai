#: Row order: the radix sort, the calls over it, and the rankings.
from std.gpu import Buffer

from ..column import Column, DataFrame, Dtype, Series
from ..column import cached_kernel, iota, prefix_sum
from .encode import KeyEncoding, encode_keys
from .frames import descending_column, descending_positions, index_of, reorder
from .frames import resolve_levels, row_count, sort_key_columns, spread_any
from .lanes import flag, kernel_for
from .table import kept_rows

#: The permutation that sorts the rows by several key columns at once.
#:
#: Stable, and least significant key first — which is not an implementation
#: detail but the reason the sort is simple. Because each pass leaves rows that
#: tie on its own key in the order the previous pass left them, sorting by the
#: last key and then by the first gives exactly the order a comparator over the
#: pair would: the earlier keys break ties the later ones left. So there is no
#: multi-key comparison anywhere, only a chain of eight-bit passes over one
#: digit lane at a time.
#:
#: `ascending` is one flag per column, one flag for all of them, or `null`.
#: `na_position` is absolute, as in pandas: `"last"` means missing rows end up
#: at the end however the columns are sorted.
pub fn argsort(
    columns: list[Column],
    ascending: list[bool]? = null,
    na_position: str = "last"
) -> Buffer {
    let keys = encode_keys(columns)
    defer { keys.free() }
    return keys.order(ascending, na_position)
}

#: The frame, series or column sorted by its key columns.
#:
#: `by` names the columns of a frame, and a series or a column sorts by its own
#: values and refuses one. `ascending` is a bool or one bool per key.
#: `ignore_index` replaces the labels with a fresh range, as pandas' does,
#: rather than carrying the old ones along in their new order.
pub fn sort_values(
    target: any,
    by: any = null,
    ascending: any = true,
    na_position: str = "last",
    ignore_index: bool = false
) -> any {
    let columns = sort_key_columns(target, by)
    let order = argsort(columns, spread_any(ascending, columns.len()), na_position)
    defer { order.free() }
    return reorder(target, order, row_count(target), ignore_index)
}

#: The frame or series sorted by its row labels.
#:
#: `level` picks which levels of a multi-index decide, by position or by name;
#: `null` uses all of them, outermost first. A range index is already in order,
#: so this recognises one instead of materialising the ramp it stands for and
#: sorting that.
pub fn sort_index(
    target: any,
    level: any = null,
    ascending: bool = true,
    na_position: str = "last",
    ignore_index: bool = false
) -> any {
    let index = index_of(target)
    let length = index.len()
    if index.is_range() {
        let forward = (index.step > 0) == ascending
        let order = if forward { iota(length) } else { descending_positions(length) }
        defer { order.free() }
        return reorder(target, order, length, ignore_index)
    }
    var levels: list[Column] = []
    for at in resolve_levels(index, level) { levels.push(index.levels[at]) }
    let order = argsort(levels, [ascending], na_position)
    defer { order.free() }
    return reorder(target, order, length, ignore_index)
}

#: The `n` rows with the largest keys, largest first.
#:
#: `keep` decides the ties at the cutoff: `"first"` takes the rows that came
#: first, `"last"` the ones that came last, and `"all"` widens the result past
#: `n` to take the whole run. A row missing any of its key columns is left out
#: entirely, the way pandas leaves a NaN out of an `nlargest`.
pub fn nlargest(target: any, n: int, columns: any = null, keep: str = "first") -> any {
    return extreme(target, n, columns, keep, true)
}

#: The `n` rows with the smallest keys, smallest first.
pub fn nsmallest(target: any, n: int, columns: any = null, keep: str = "first") -> any {
    return extreme(target, n, columns, keep, false)
}

#: Each row's position in the sorted order, one to `n`.
#:
#: `method` decides what a tie gets: `"average"` the midpoint of the positions
#: the tied rows occupy, `"min"` the first of them, `"max"` the last, `"first"`
#: the position each row actually landed in, and `"dense"` the run's number
#: among distinct values with no gaps after it. `pct` divides through by the
#: number of ranked rows — by the number of distinct values for `"dense"`,
#: which is what makes a dense percentile reach 1.0.
#:
#: `na_option` is `"keep"` for a missing rank on a missing value, `"top"` to
#: rank missing first and `"bottom"` to rank it last. The sort underneath is
#: the same one `argsort` runs; the ties come off the same encoding, so a rank
#: cannot disagree with the order it was read from.
pub fn rank(
    target: any,
    method: str = "average",
    ascending: bool = true,
    pct: bool = false,
    na_option: str = "keep"
) -> any {
    if target is DataFrame {
        let ranked = target.columns.map(
            |column| rank_column(column, method, ascending, pct, na_option)
        )
        return target._replace(columns: ranked)
    }
    if target is Series {
        return Series(
            rank_column(target.values, method, ascending, pct, na_option),
            index: target.index.copy(),
            name: target.name
        )
    }
    if target is Column { return rank_column(target, method, ascending, pct, na_option) }
    throw TypeError("rank takes a Column, a Series or a DataFrame")
}

fn rank_column(column: Column, method: str, ascending: bool, pct: bool, na_option: str) -> Column {
    let code = rank_method(method)
    if na_option != "keep" and na_option != "top" and na_option != "bottom" {
        throw ValueError(f"na_option must be keep, top or bottom, not {na_option}")
    }
    let keeps = na_option == "keep"
    let na_position = if na_option == "top" { "first" } else { "last" }
    let count = column.len()
    let keys = encode_keys([column])
    defer { keys.free() }
    let order = keys.order([ascending], na_position)
    defer { order.free() }
    let out = Column.allocate(Dtype.Float64, count, with_validity: keeps)
    if count == 0 { return out }

    let runs = dense_runs(keys, order, count)
    let dense = runs[0]
    let starts = runs[1]
    let distinct = runs[2]
    defer { dense.free() }
    defer { starts.free() }

    let absent = count - kept_rows(keys, true)
    let present = count - absent
    let dense_total = if keeps and absent > 0 { distinct - 1 } else { distinct }
    var mask = out.valid
    if mask is null { mask = Buffer(count) }
    kernel_for("key_rank_assign").dispatch_async(
        [order, dense, starts, out.values, mask ?? out.values],
        count,
        [count, distinct, code, present, flag(keeps), flag(pct), dense_total]
    )
    if out.valid is null and mask is not null { mask.free() }
    return out
}

fn rank_method(name: str) -> int {
    return match name {
        "average" => 0,
        "min" => 1,
        "max" => 2,
        "first" => 3,
        "dense" => 4,
        _ => throw ValueError(f"unknown rank method {name}"),
    }
}

fn extreme(target: any, n: int, columns: any, keep: str, largest: bool) -> any {
    if n < 0 { throw ValueError(f"cannot take {n} rows") }
    if keep != "first" and keep != "last" and keep != "all" {
        throw ValueError(f"keep must be first, last or all, not {keep}")
    }
    let ranked = sort_key_columns(target, columns)
    let length = row_count(target)
    var wanted = [column for column in ranked]
    var ups = [not largest for _column in ranked]
    var tiebreak: Column? = null
    if keep == "last" {
        let backwards = descending_column(length)
        tiebreak = backwards
        wanted.push(backwards)
        ups.push(true)
    }
    let keys = encode_keys(wanted)
    let order = keys.order(ups, "last")
    let present = kept_rows(keys, true)
    var limit = if n < present { n } else { present }
    if keep == "all" and limit > 0 and limit < present {
        limit = tie_boundary(keys, order, length, limit, present)
    }
    let head = Buffer(limit)
    if limit > 0 {
        cached_kernel("col_copy_lane").dispatch_async([order, head], limit, [limit, 0, 0])
    }
    order.free()
    keys.free()
    let spare = tiebreak
    if spare is not null { spare.free() }
    let taken = reorder(target, head, limit, false)
    head.free()
    return taken
}

#: Where the run of ties straddling the cutoff ends.
#:
#: Two one-element downloads rather than a scan on the host: the dense code at
#: the cutoff says which run it is in, and the run table says where the next
#: run starts.
fn tie_boundary(keys: KeyEncoding, order: Buffer, length: int, limit: int, present: int) -> int {
    let runs = dense_runs(keys, order, length)
    let dense = runs[0]
    let starts = runs[1]
    let distinct = runs[2]
    defer { dense.free() }
    defer { starts.free() }
    let run = int(dense.download(limit - 1, 1)[0])
    if run + 1 >= distinct { return present }
    let after = int(starts.download(run + 1, 1)[0])
    return if after < present { after } else { present }
}

#: Which run of equal keys each sorted position falls in, where each run
#: starts, and how many runs there are.
#:
#: Ranking and a `keep: all` cutoff both need the ties, and both take them off
#: the encoding the order came from rather than comparing the values a second
#: way — two answers read from one sort cannot disagree. The caller owns both
#: lanes.
fn dense_runs(keys: KeyEncoding, positions: Buffer, length: int) -> (Buffer, Buffer, int) {
    let flags = keys.run_flags(positions)
    defer { flags.free() }
    let scanned = prefix_sum(flags, length)
    let offsets = scanned[0]
    let distinct = scanned[1]
    defer { offsets.free() }
    let starts = Buffer(distinct)
    kernel_for("key_positions_where").dispatch_async([flags, offsets, starts], length, [length])
    let dense = Buffer(length)
    kernel_for("key_dense_codes").dispatch_async([offsets, flags, dense], length, [length])
    return (dense, starts, distinct)
}

#: Running totals and lags: the scans, the cumulative forms, and the shifts.
#:
#: A scan turns a column into another of the same length. `scan_into` is the
#: Blelloch sweep the cumulative forms and the window paths both build on, and
#: `dispatch_column_wise` is how any of them is lifted to a series or a frame.
from std.gpu import Buffer
from ..column import Agg, Column, DataFrame, Dtype, Series, Value
from ..column import f32_bits, wide_pack
from .kernels import SCAN_ADD, SCAN_GROUP, SCAN_MAX, SCAN_MIN, SCAN_MUL
from .kernels import flag, kernel_for, require_device, u32

#: An inclusive or exclusive scan of a lane, into a buffer the caller owns.
#:
#: Blocks of 256 swept in threadgroup memory, then the block totals scanned by
#: the same kernel and combined back — so 2^24 rows is three levels of dispatch
#: rather than a host loop over sixty-five thousand block totals. Asking each
#: level for its total would put a device wait between the levels, which is why
#: nothing here downloads anything.
pub fn scan_into(input: Buffer, out: Buffer, length: int, op: int, inclusive: bool) -> void {
    if length <= 0 { return }
    require_device("scan a column")
    let blocks = (length + SCAN_GROUP - 1) // SCAN_GROUP
    let totals = Buffer(blocks)
    kernel_for("scan_block").dispatch_async(
        [input, out, totals],
        blocks * SCAN_GROUP,
        [length, op, flag(inclusive)],
        group_size: SCAN_GROUP
    )
    if blocks > 1 {
        let carried = Buffer(blocks)
        scan_into(totals, carried, blocks, op, false)
        kernel_for("scan_carry").dispatch_async([out, carried], length, [length, op])
        carried.free()
    }
    totals.free()
}

fn scan_op(agg: Agg) -> int {
    return match agg {
        Agg.Sum => SCAN_ADD,
        Agg.Prod => SCAN_MUL,
        Agg.Max => SCAN_MAX,
        Agg.Min => SCAN_MIN,
        _ => throw ValueError(f"{agg.name()} does not scan; a scan needs an associative operator"),
    }
}

#: A running `agg` down one column.
#:
#: The answer is a `Float64` even over an `Int64` column, for the reason the
#: reductions give: the running total is accumulated in a float32 lane. A
#: missing row keeps the running value where it was and stays missing itself;
#: with `skipna` off, it and everything after it are missing, which the scan
#: gets for free by putting a NaN in the lane instead of the operator's
#: identity.
pub fn scan_column(column: Column, agg: Agg, skipna: bool = true) -> Column {
    if not column.dtype.is_numeric() and column.dtype != Dtype.Datetime {
        throw TypeError(f"a running {agg.name()} needs numbers, not {column.dtype.name()}")
    }
    let op = scan_op(agg)
    let rows = column.len()
    let out = Column.allocate(Dtype.Float64, rows, with_validity: true)
    if rows == 0 { return out }
    let terms = Buffer(rows)
    kernel_for("scan_prepare").dispatch_async(
        [column.values, column.wide ?? column.values, column.valid ?? column.values, terms],
        rows,
        [rows, flag(column.dtype.is_wide()), flag(column.valid is not null), op, flag(skipna)]
    )
    scan_into(terms, out.values, rows, op, true)
    terms.free()
    kernel_for("scan_finish").dispatch_async(
        [out.values, column.valid ?? out.values, out.valid ?? out.values],
        rows,
        [rows, flag(column.valid is not null), flag(skipna)]
    )
    return out
}

#: How many rows up to and including each one hold nothing.
#:
#: An inclusive sum-scan of the missing flags, which are exact integers under
#: 2^24 and so exact in a float32 lane. Never missing itself — every row has an
#: answer, including the missing ones.
pub fn cumcount_na_column(column: Column) -> Column {
    let rows = column.len()
    let out = Column.allocate(Dtype.Int64, rows)
    if rows == 0 { return out }
    let flags = Buffer(rows)
    kernel_for("scan_missing_flags").dispatch_async(
        [column.values, column.valid ?? column.values, flags],
        rows,
        [rows, flag(column.valid is not null)]
    )
    let running = Buffer(rows)
    scan_into(flags, running, rows, SCAN_ADD, true)
    kernel_for("red_split_wide").dispatch_async(
        [running, out.values, out.wide ?? out.values],
        rows,
        [rows]
    )
    flags.free()
    running.free()
    return out
}

#: A column-wise transform lifted to whatever the caller handed in.
#:
#: A frame keeps only the columns the transform is defined over — pandas
#: concatenates text under `cumsum` and raises under `cummax`, and neither is
#: a shape a later join can rely on.
pub fn dispatch_column_wise(target: any, body: fn(Column) -> Column, numeric_only: bool) -> any {
    if target is Column { return body(target) }
    if target is Series {
        return Series(body(target.values), index: target.index.copy(), name: target.name)
    }
    if target is not DataFrame { throw TypeError("this takes a Column, a Series or a DataFrame") }
    var names: list[str] = []
    var columns: list[Column] = []
    for (at, column) in target.columns.enumerate() {
        if numeric_only and not column.dtype.is_numeric() and column.dtype != Dtype.Datetime {
            continue
        }
        names.push(target.names[at])
        columns.push(body(column))
    }
    return DataFrame(names, columns, index: target.index.copy())
}

#: The running sum down each column.
pub fn cumsum(target: any, skipna: bool = true) -> any {
    return dispatch_column_wise(target, |column| scan_column(column, Agg.Sum, skipna), true)
}

#: The running product down each column.
pub fn cumprod(target: any, skipna: bool = true) -> any {
    return dispatch_column_wise(target, |column| scan_column(column, Agg.Prod, skipna), true)
}

#: The running largest value down each column.
pub fn cummax(target: any, skipna: bool = true) -> any {
    return dispatch_column_wise(target, |column| scan_column(column, Agg.Max, skipna), true)
}

#: The running smallest value down each column.
pub fn cummin(target: any, skipna: bool = true) -> any {
    return dispatch_column_wise(target, |column| scan_column(column, Agg.Min, skipna), true)
}

#: The running count of missing rows down each column.
pub fn cumcount_na(target: any) -> any {
    return dispatch_column_wise(target, |column| cumcount_na_column(column), false)
}

#: One column moved along its index by `periods` rows.
#:
#: Both value lanes travel, so a shifted `Int64` or `Datetime` column is still
#: exact, and a text column keeps its level table. The rows that fall off the
#: end are `fill_value`, or missing when there is none.
pub fn shift_column(column: Column, periods: int = 1, fill_value: Value? = null) -> Column {
    let rows = column.len()
    let out = Column.allocate(column.dtype, rows, with_validity: true)
    if column.dtype.is_dictionary() {
        out.set_levels([name for name in column.levels ?? []])
    }
    if rows == 0 { return out }
    var low = 0.0
    var high = 0.0
    var has_fill = false
    let fill = fill_value
    if fill is not null and not fill.is_na() {
        has_fill = true
        if column.dtype.is_dictionary() {
            low = float(out.intern_level(fill.to_str()))
        } elif column.dtype.is_wide() {
            let split = wide_pack(fill.to_int())
            low = split[0]
            high = split[1]
        } else {
            low = fill.to_float()
        }
    }
    kernel_for("lag_shift").dispatch_async(
        [
            column.values,
            column.wide ?? column.values,
            column.valid ?? column.values,
            out.values,
            out.wide ?? out.values,
            out.valid ?? out.values,
        ],
        rows,
        [
            rows,
            u32(periods),
            flag(column.dtype.is_wide()),
            flag(column.valid is not null),
            f32_bits(low),
            f32_bits(high),
            flag(has_fill),
        ]
    )
    return out
}

fn change_column(column: Column, periods: int, ratio: bool) -> Column {
    let numeric = column.dtype.is_numeric() or column.dtype == Dtype.Datetime
    if not numeric {
        throw TypeError(f"a difference needs numbers, not {column.dtype.name()}")
    }
    if ratio and column.dtype == Dtype.Datetime {
        throw TypeError("the ratio of two timestamps is not a quantity; take the difference")
    }
    let rows = column.len()
    let out = Column.allocate(Dtype.Float64, rows, with_validity: true)
    if rows == 0 { return out }
    kernel_for("lag_diff").dispatch_async(
        [
            column.values,
            column.wide ?? column.values,
            column.valid ?? column.values,
            out.values,
            out.valid ?? out.values,
        ],
        rows,
        [
            rows,
            u32(periods),
            flag(column.dtype.is_wide()),
            flag(column.valid is not null),
            flag(ratio),
        ]
    )
    return out
}

#: Every value moved `periods` rows later, or earlier when negative.
pub fn shift(target: any, periods: int = 1, fill_value: Value? = null) -> any {
    return dispatch_column_wise(target, |column| shift_column(column, periods, fill_value), false)
}

#: Each row less the row `periods` back. A `Datetime` column gives the gap in
#: milliseconds, because a duration is not a timestamp.
pub fn diff(target: any, periods: int = 1) -> any {
    return dispatch_column_wise(target, |column| change_column(column, periods, false), true)
}

#: The fractional change from the row `periods` back.
pub fn pct_change(target: any, periods: int = 1) -> any {
    return dispatch_column_wise(target, |column| change_column(column, periods, true), true)
}

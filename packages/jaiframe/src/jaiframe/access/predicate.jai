#: Tests over a column that answer as a `Bool` column, and the blends they drive.
from std.gpu import Buffer
from std.math import is_nan
from std.re import Regex

from ..column import Column, DataFrame, Dtype, Value
from ..column import NA
from ..column import f32_bits, wide_pack
from .lanes import WANTED_TILE, flag, kernel_for, scalar_bits, u32, valid_lane
from .lanes import value_lane, wide_lane
from .select import filter_rows, narrow_columns

#: A `Bool` column saying, per row, what a per-level verdict says.
#:
#: The host decides once per distinct level and the device gathers the verdict
#: per row. That is what keeps every text predicate here — `isin`, `between`,
#: `filter_names` on the index — off a host walk over the rows: a dictionary
#: column has as many distinct values as it has levels, however tall it is.
fn codes_mask(level: Column, verdict: list[float], want_missing: bool) -> Column {
    let out = Column.allocate(Dtype.Bool, level.len())
    if level.len() == 0 { return out }
    let wanted = Buffer(verdict.len())
    defer { wanted.free() }
    if verdict.len() > 0 { wanted.upload(verdict) }
    kernel_for("acc_codes_in_set").dispatch_async(
        [value_lane(level), valid_lane(level), wanted, out.values],
        level.len(),
        [level.len(), verdict.len(), flag(level.has_validity()), flag(want_missing)]
    )
    return out
}

fn name_kept(name: str, items: list[str]?, like: str?, regex: Regex?) -> bool {
    if items is not null { return (items ?? []).contains(name) }
    if like is not null { return name.contains(like ?? "") }
    let pattern = regex
    if pattern is null { return true }
    return pattern.is_match(name)
}

#: The columns whose name matches, or the rows whose label does.
#:
#: Exactly one of `items`, `like` and `regex` is given, as pandas requires.
#: On `axis` 1 the surviving columns are shared with `frame` rather than
#: copied; on `axis` 0 the rows are gathered and fresh.
#:
#: `axis` 0 needs a text index, because the match is over the label's
#: spelling. It costs one host test per distinct label and one dispatch over
#: the rows, not one test per row.
pub fn filter_names(
    frame: DataFrame,
    items: list[str]? = null,
    like: str? = null,
    regex: str? = null,
    axis: int = 1
) -> DataFrame {
    var given = 0
    if items is not null { given += 1 }
    if like is not null { given += 1 }
    if regex is not null { given += 1 }
    if given != 1 {
        throw ValueError(f"filter takes exactly one of items, like and regex, got {given}")
    }
    let pattern = if regex is null { null } else { Regex(regex ?? "") }
    if axis == 1 {
        if items is not null {
            return narrow_columns(
                frame,
                (items ?? []).filter(|name| frame.has(name)).map(
                    |name| frame.column_index(name)
                )
            )
        }
        var kept: list[int] = []
        for (at, name) in frame.names.enumerate() {
            if name_kept(name, items, like, pattern) { kept.push(at) }
        }
        return narrow_columns(frame, kept)
    }
    if axis != 0 { throw ValueError(f"a frame has axes 0 and 1, not {axis}") }
    if frame.index.is_range() or frame.index.nlevels() != 1 {
        throw TypeError("filtering row labels by text needs a single-level text index")
    }
    let level = frame.index.levels[0]
    if not level.dtype.is_dictionary() {
        throw TypeError(f"cannot match {level.dtype.name()} labels against a name pattern")
    }
    let verdict = (level.levels ?? []).map(|name| if name_kept(name, items, like, pattern) {
        1.0
    } else {
        0.0
    })
    let mask = codes_mask(level, verdict, false)
    defer { mask.free() }
    return filter_rows(frame, mask)
}

fn wants_missing(dtype: Dtype, value: Value) -> bool {
    if value.is_na() { return true }
    if dtype.is_dictionary() { return false }
    return is_nan(value.to_float())
}

#: True where a row holds one of `values`.
#:
#: A dictionary column answers off its levels, so a text `isin` costs one host
#: test per distinct string however tall the column is. Everything else stages
#: the wanted values in threadgroup memory and compares in one pass: every
#: thread in a group reads the same wanted lane, so reading it from device
#: memory once per row would be the whole cost of the kernel.
#:
#: `NA` in `values` matches the missing rows, and so does a NaN — pandas
#: treats `isin([nan])` as a null test, which is the only reading that is ever
#: useful, since NaN is not equal to itself.
pub fn isin(column: Column, values: list[Value]) -> Column {
    var want_missing = false
    for value in values {
        if wants_missing(column.dtype, value) { want_missing = true }
    }
    if column.dtype.is_dictionary() {
        var wanted: dict[str, bool] = {}
        for value in values {
            if not value.is_na() { wanted[value.to_str()] = true }
        }
        let verdict = (column.levels ?? []).map(|name| if wanted.get(name, false) {
            1.0
        } else {
            0.0
        })
        return codes_mask(column, verdict, want_missing)
    }
    var low: list[float] = []
    var high: list[float] = []
    for value in values {
        if wants_missing(column.dtype, value) { continue }
        if column.dtype.is_wide() {
            let split = wide_pack(value.to_int())
            low.push(split[0])
            high.push(split[1])
            continue
        }
        low.push(value.to_float())
        high.push(0.0)
    }
    let out = Column.allocate(Dtype.Bool, column.len())
    if column.len() == 0 { return out }
    let wanted_low = Buffer(low.len())
    let wanted_high = Buffer(high.len())
    defer {
        wanted_low.free()
        wanted_high.free()
    }
    if low.len() > 0 {
        wanted_low.upload(low)
        wanted_high.upload(high)
    }
    let groups = (column.len() + WANTED_TILE - 1) // WANTED_TILE
    kernel_for("acc_isin_values").dispatch_async(
        [
            value_lane(column),
            wide_lane(column),
            valid_lane(column),
            wanted_low,
            wanted_high,
            out.values,
        ],
        groups * WANTED_TILE,
        [
            column.len(),
            low.len(),
            flag(column.dtype.is_wide()),
            flag(column.has_validity()),
            flag(want_missing),
        ],
        group_size: WANTED_TILE
    )
    return out
}

#: A `Bool` column per column of the frame, true where a row holds one of
#: `values`.
#:
#: The index is the frame's own, not a copy, which is what `_replace` does
#: with anything an operation did not change — so freeing the answer frees the
#: index out from under the frame it came from.
pub fn frame_isin(frame: DataFrame, values: list[Value]) -> DataFrame {
    return frame._replace(columns: frame.columns.map(|column| isin(column, values)))
}

fn closed_bits(inclusive: str) -> int {
    return match inclusive {
        "both" => 3,
        "left" => 1,
        "right" => 2,
        "neither" => 0,
        _ => throw ValueError(f"inclusive is both, left, right or neither, not {inclusive}"),
    }
}

#: True where a row falls between `low` and `high`.
#:
#: `inclusive` is pandas' — `both`, `left`, `right` or `neither`. A missing row
#: is false, whichever end it would have fallen on.
#:
#: Text compares by its spelling, resolved level by level on the host: the
#: dictionary codes are in first-appearance order, so comparing them would
#: order the strings by when they were first seen rather than alphabetically.
#: An `Int64` or `Datetime` range compares at the full 48 bits, because the
#: bounds cross to the device as their two halves rather than as one float.
pub fn between(column: Column, low: Value, high: Value, inclusive: str = "both") -> Column {
    let closed = closed_bits(inclusive)
    if low.is_na() or high.is_na() {
        throw ValueError("a between range needs two bounds, not NA")
    }
    if column.dtype.is_dictionary() {
        let after = low.to_str()
        let before = high.to_str()
        var verdict: list[float] = []
        for name in column.levels ?? [] {
            let above = if closed & 1 != 0 { name >= after } else { name > after }
            let below = if closed & 2 != 0 { name <= before } else { name < before }
            verdict.push(if above and below { 1.0 } else { 0.0 })
        }
        return codes_mask(column, verdict, false)
    }
    let out = Column.allocate(Dtype.Bool, column.len())
    if column.len() == 0 { return out }
    if column.dtype.is_wide() {
        let after = wide_pack(low.to_int())
        let before = wide_pack(high.to_int())
        kernel_for("acc_between_wide").dispatch_async(
            [value_lane(column), wide_lane(column), valid_lane(column), out.values],
            column.len(),
            [
                column.len(),
                flag(column.has_validity()),
                u32(int(after[0])),
                u32(int(after[1])),
                u32(int(before[0])),
                u32(int(before[1])),
                closed,
            ]
        )
        return out
    }
    kernel_for("acc_between").dispatch_async(
        [value_lane(column), valid_lane(column), out.values],
        column.len(),
        [
            column.len(),
            flag(column.has_validity()),
            f32_bits(low.to_float()),
            f32_bits(high.to_float()),
            closed,
        ]
    )
    return out
}

fn blend(column: Column, cond: Column, other: Value, invert: bool) -> Column {
    if cond.dtype != Dtype.Bool {
        throw TypeError(f"a condition must be bool, not {cond.dtype.name()}")
    }
    if cond.len() != column.len() {
        throw ValueError(f"a condition of {cond.len()} cannot cover {column.len()} rows")
    }
    let out = Column.allocate(
        column.dtype,
        column.len(),
        with_validity: column.has_validity() or other.is_na()
    )
    if column.dtype.is_dictionary() {
        out.set_levels([name for name in column.levels ?? []])
    }
    let bits = scalar_bits(out, other)
    if column.len() == 0 { return out }
    kernel_for("acc_where").dispatch_async(
        [
            value_lane(column),
            wide_lane(column),
            valid_lane(column),
            value_lane(cond),
            valid_lane(cond),
            out.values,
            wide_lane(out),
            valid_lane(out),
        ],
        column.len(),
        [
            column.len(),
            flag(column.dtype.is_wide()),
            flag(column.has_validity()),
            flag(cond.has_validity()),
            flag(out.has_validity()),
            flag(invert),
            bits[0],
            bits[1],
            flag(other.is_na()),
        ]
    )
    return out
}

#: The column's own value where `cond` is true, `other` everywhere else.
#:
#: A null condition counts as false and takes `other`, which is what makes
#: `where` usable against a mask that came from a column with nulls in it.
pub fn where(column: Column, cond: Column, other: Value = NA) -> Column {
    return blend(column, cond, other, false)
}

#: `other` where `cond` is true, the column's own value everywhere else — the
#: complement of `where`.
pub fn mask(column: Column, cond: Column, other: Value = NA) -> Column {
    return blend(column, cond, other, true)
}

#: `where` over every column, with one row condition shared by all of them.
#:
#: The index is shared with `frame`, as `frame_isin`'s is.
pub fn frame_where(frame: DataFrame, cond: Column, other: Value = NA) -> DataFrame {
    return frame._replace(columns: frame.columns.map(|column| where(column, cond, other)))
}

#: `mask` over every column, with one row condition shared by all of them.
#:
#: The index is shared with `frame`, as `frame_isin`'s is.
pub fn frame_mask(frame: DataFrame, cond: Column, other: Value = NA) -> DataFrame {
    return frame._replace(columns: frame.columns.map(|column| mask(column, cond, other)))
}

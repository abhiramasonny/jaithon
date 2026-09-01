#: How a grouping is asked for, and how it becomes a lane of codes.
#:
#: `by` and `level` are unpacked here, a callback over the labels is run here,
#: and `factorize_keys` is the single door onto `keys.jai` — nothing past this
#: file knows how a key is hashed. `category_product` is the other door, the
#: one `observed: false` goes through.
from std.gpu import Buffer
from ..column import Column, DataFrame, Dtype, Index, Series, Value
from ..column import MAX_ROWS
from ..keys import factorize
from .kernels import flag, kernel_for

#: The frame a source is grouped as, and the name a `Series` source carried.
pub fn as_frame(source: any) -> (DataFrame, str?) {
    if source is DataFrame { return (source, null) }
    if source is Series {
        let name = source.name ?? "values"
        return (DataFrame([name], [source.values], index: source.index), name)
    }
    throw TypeError("group_by takes a DataFrame or a Series")
}

#: The cells of one index level or one key level, as host values.
#:
#: A download per lane rather than a `get` per row: this is the boundary where
#: a label stops being a device allocation, and the callers — `groups`,
#: `get_group`, the `by` callback — are the calls that already cost one.
pub fn level_values(column: Column) -> list[Value] {
    let missing = column.null_mask()
    var out: list[Value] = []
    if column.dtype.is_dictionary() {
        for text in column.to_strs() {
            out.push(if text is null {
                Value.null_of(column.dtype)
            } else {
                Value.of_str(text, column.dtype)
            })
        }
        return out
    }
    if column.dtype.is_wide() {
        for (at, number) in column.to_ints().enumerate() {
            out.push(if missing[at] {
                Value.null_of(column.dtype)
            } else {
                Value(column.dtype, integer: number)
            })
        }
        return out
    }
    for (at, number) in column.to_floats().enumerate() {
        out.push(if missing[at] {
            Value.null_of(column.dtype)
        } elif column.dtype == Dtype.Bool {
            Value.of_bool(number != 0.0)
        } else {
            Value.of_float(number)
        })
    }
    return out
}

#: One label per row, the levels joined the way `groups` and `get_group` spell
#: a multi-key group.
pub fn render_labels(levels: list[Column], length: int) -> list[str] {
    var out: list[str] = []
    for _row in 0..length { out.push("") }
    for (at, level) in levels.enumerate() {
        for (row, cell) in level_values(level).enumerate() {
            out[row] = if at == 0 { cell.to_str() } else { out[row] + "|" + cell.to_str() }
        }
    }
    return out
}

#: How a grouping was asked for, so `group_by` can name its levels.
enum KeySource {
    #: One or more of the frame's own columns.
    Columns,
    #: One or more levels of the frame's index.
    Levels,
    #: A callback over the index labels.
    Callback,
    #: A column handed in directly, belonging to nobody.
    Loose,
}

#: A column holding host cells, typed by the first one that is not missing.
pub fn column_of_values(cells: list[Value]) -> Column {
    var dtype = Dtype.Float64
    for cell in cells {
        if not cell.is_na() {
            dtype = cell.dtype
            break
        }
    }
    if dtype.is_dictionary() {
        var text: list[str?] = []
        for cell in cells { text.push(if cell.is_na() { null } else { cell.to_str() }) }
        return Column.from_strs(text, dtype)
    }
    var missing: list[bool] = []
    for cell in cells { missing.push(cell.is_na()) }
    if dtype.is_wide() {
        var numbers: list[int] = []
        for cell in cells { numbers.push(if cell.is_na() { 0 } else { cell.to_int() }) }
        return Column.from_ints(numbers, nulls: missing, dtype: dtype)
    }
    if dtype == Dtype.Bool {
        var truths: list[bool] = []
        for cell in cells { truths.push(not cell.is_na() and cell.to_bool()) }
        return Column.from_bools(truths, nulls: missing)
    }
    var numbers: list[float] = []
    for cell in cells { numbers.push(cell.to_float()) }
    return Column.from_floats(numbers, nulls: missing)
}

#: How `groups` and `get_group` spell one group's key.
pub fn label_of(key: any) -> str {
    if key is str { return key }
    if key is Value { return key.to_str() }
    if key is list {
        var parts: list[str] = []
        for part in key { parts.push(label_of(part)) }
        return "|".join(parts)
    }
    throw TypeError("a group key is a label, a Value, or a list of them")
}

#: The index levels `level` names, by position or by name.
pub fn index_levels(index: Index, level: any) -> list[int] {
    if level is list {
        var out: list[int] = []
        for one in level { out.push(index_levels(index, one)[0]) }
        return out
    }
    if level is int {
        let at = if level < 0 { index.nlevels() + level } else { level }
        if at < 0 or at >= index.nlevels() {
            throw IndexError(f"level {level} is outside an index of {index.nlevels()}")
        }
        return [at]
    }
    if level is str {
        for (at, name) in index.names.enumerate() {
            if name == level { return [at] }
        }
        throw KeyError(f"no index level named {level}")
    }
    throw TypeError("a level is a position or a name")
}

#: The key specifications `by` holds, always as a list.
pub fn key_specs(by: any) -> list[any] {
    if by is list { return by }
    return [by]
}

#: A key column from a callback over the index labels.
#:
#: The callback is Jaithon, so it runs once per row on the host — the same
#: bargain `access.jai:query` makes. Hand `by` a `Column` instead when the key
#: can be computed on the device.
pub fn labelled_keys(index: Index, callback: any) -> Column {
    let levels = index.materialize()
    var cells: list[list[Value]] = []
    for level in levels { cells.push(level_values(level)) }
    for level in levels { level.free() }
    var produced: list[Value] = []
    for row in 0..index.len() {
        var label: list[Value] = []
        for column in cells { label.push(column[row]) }
        let answer = if label.len() == 1 { callback(label[0]) } else { callback(label) }
        produced.push(coerce_key(answer))
    }
    return column_of_values(produced)
}

pub fn coerce_key(answer: any) -> Value {
    if answer is Value { return answer }
    if answer is str { return Value.of_str(answer) }
    if answer is int { return Value.of_int(answer) }
    if answer is float { return Value.of_float(answer) }
    if answer is bool { return Value.of_bool(answer) }
    throw TypeError("a group-by callback answers with a label")
}

#: `keys.jai:factorize`, reshaped to the three things this module wants.
#:
#: Every dependency on the key machinery goes through here: the codes lane,
#: one column per key level holding the distinct values, and how many groups
#: there are. Nothing else in this file knows how a key is hashed.
pub fn factorize_keys(
    columns: list[Column],
    sort: bool,
    dropna: bool
) -> (Buffer, list[Column], int) {
    return factorize(columns, sort: sort, dropna: dropna)
}

#: Every combination of the key columns' levels, observed or not.
#:
#: `factorize` numbers the key tuples that rows actually carry, which is
#: `observed: true`. pandas' `observed: false` asks for a group per level
#: combination whether or not a row landed in it, and that is only a question a
#: `Category` can answer — its level table *is* the domain, where a `Str`
#: column's is just what turned up. So a mixed set of keys is refused rather
#: than half-expanded.
pub fn category_product(
    keys: list[Column],
    rows: int,
    dropna: bool
) -> (Buffer, list[Column], int) {
    var total = 1
    for key in keys {
        if key.dtype != Dtype.Category {
            throw TypeError(
                f"observed: false needs every key to be a category, and one is {key.dtype.name()}"
            )
        }
        total *= max(1, key.level_count())
    }
    if not dropna {
        throw ValueError("observed: false has no slot for a missing level; pass dropna: true")
    }
    if total > MAX_ROWS {
        throw ValueError(f"{total} level combinations exceeds the {MAX_ROWS} a code lane holds")
    }
    var strides: list[int] = []
    for _key in keys { strides.push(1) }
    var running = 1
    for step in 0..keys.len() {
        let at = keys.len() - 1 - step
        strides[at] = running
        running *= max(1, keys[at].level_count())
    }
    let codes = Buffer(max(1, rows))
    if rows > 0 { codes.fill_zero() }
    for (at, key) in keys.enumerate() {
        kernel_for("gb_product_codes").dispatch_async(
            [key.values, key.valid ?? key.values, codes],
            rows,
            [rows, strides[at], key.level_count(), flag(key.valid is not null)]
        )
    }
    var uniques: list[Column] = []
    for (at, key) in keys.enumerate() {
        let level = Column.allocate(Dtype.Category, total)
        level.set_levels([name for name in key.levels ?? []])
        kernel_for("gb_product_levels").dispatch_async(
            [level.values],
            total,
            [total, strides[at], max(1, key.level_count())]
        )
        uniques.push(level)
    }
    return (codes, uniques, total)
}

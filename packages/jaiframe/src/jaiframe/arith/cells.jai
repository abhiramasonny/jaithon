#: The one door out to the host: a column as boxed values, and a column built
#: back from them.
from std.math import NAN, is_nan

from ..column import Column, Dtype, Value

#: Every cell of a column, as host `Value`s.
#:
#: One download per lane, which is the whole cost of leaving the device. Every
#: callback in this module and the label matching in `align_indexes` go through
#: here so that there is exactly one place that pays it.
pub fn column_cells(column: Column) -> list[Value] {
    if column.len() == 0 { return [] }
    let dtype = column.dtype
    var out: list[Value] = []
    if dtype.is_dictionary() {
        for text in column.to_strs() {
            out.push(if text is null { Value.null_of(dtype) } else { Value.of_str(text, dtype) })
        }
        return out
    }
    let missing = column.null_mask()
    if dtype.is_wide() {
        for (row, value) in column.to_ints().enumerate() {
            out.push(if missing[row] { Value.null_of(dtype) } else { Value(dtype, integer: value) })
        }
        return out
    }
    for (row, value) in column.to_floats().enumerate() {
        if missing[row] or is_nan(value) {
            out.push(Value.null_of(dtype))
        } elif dtype == Dtype.Bool {
            out.push(Value.of_bool(value != 0.0))
        } else {
            out.push(Value.of_float(value))
        }
    }
    return out
}

#: The dtype a list of host cells should be stored as.
#:
#: A text cell anywhere makes the column text, because a dictionary can hold
#: the printed form of a number and no number can hold a word. Otherwise the
#: widest numeric kind present wins, which is the same ladder `promote` walks.
fn inferred_dtype(cells: list[Value], hint: Dtype?) -> Dtype {
    var saw_text = false
    var saw_float = false
    var saw_int = false
    var saw_time = false
    var saw_any = false
    for cell in cells {
        if cell.is_na() { continue }
        saw_any = true
        if cell.dtype.is_dictionary() {
            saw_text = true
        } elif cell.dtype == Dtype.Float64 {
            saw_float = true
        } elif cell.dtype == Dtype.Int64 {
            saw_int = true
        } elif cell.dtype == Dtype.Datetime {
            saw_time = true
        }
    }
    if not saw_any { return hint ?? Dtype.Float64 }
    if saw_text { return Dtype.Str }
    if saw_time and not saw_float and not saw_int { return Dtype.Datetime }
    if saw_float { return Dtype.Float64 }
    if saw_int or saw_time { return Dtype.Int64 }
    return Dtype.Bool
}

#: Host cells back into a column, in one upload.
pub fn column_of_cells(cells: list[Value], hint: Dtype? = null) -> Column {
    let dtype = inferred_dtype(cells, hint)
    var missing: list[bool] = []
    for cell in cells { missing.push(cell.is_na()) }
    if dtype.is_dictionary() {
        var texts: list[str?] = []
        for cell in cells { texts.push(if cell.is_na() { null } else { cell.to_str() }) }
        return Column.from_strs(texts, dtype)
    }
    if dtype.is_wide() {
        var numbers: list[int] = []
        for cell in cells { numbers.push(if cell.is_na() { 0 } else { cell.to_int() }) }
        return Column.from_ints(numbers, missing, dtype)
    }
    if dtype == Dtype.Bool {
        var flags: list[bool] = []
        for cell in cells { flags.push(not cell.is_na() and cell.to_bool()) }
        return Column.from_bools(flags, missing)
    }
    var numbers: list[float] = []
    for cell in cells { numbers.push(if cell.is_na() { NAN } else { cell.to_float() }) }
    return Column.from_floats(numbers, missing)
}

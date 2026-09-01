#: The two dispatches themselves: how an operand reaches the kernel, and what
#: comes back.
from std.algo.sort import natural_order, sort_by

from ..column import Column, Dtype, Series, Value
from ..column import f32_bits, unify_levels, wide_pack
from .codes import OP_AND, OP_DIVIDE, OP_EQUAL, OP_FLOOR_DIVIDE, OP_GREATER_EQUAL
from .codes import OP_LESS, OP_MODULO, OP_POWER, binary_name
from .codes import UN_LOG, UN_LOG10, UN_LOG2, UN_SQRT
from .kernels import cached_kernel, flag, require_device
from .rules import promote, unary_result

#: Whether the op yields a `Bool` column whatever its operands were.
fn is_comparison(op: int) -> bool { return op >= OP_EQUAL and op <= OP_GREATER_EQUAL }

#: Whether the op orders its operands rather than only testing them equal.
fn is_ordering(op: int) -> bool { return op >= OP_LESS and op <= OP_GREATER_EQUAL }

fn is_logical(op: int) -> bool { return op >= OP_AND }

#: The dtype the result of `op` is stored as.
fn binary_result(op: int, common: Dtype) -> Dtype {
    if is_comparison(op) or is_logical(op) { return Dtype.Bool }
    if common.is_dictionary() {
        throw TypeError(f"{binary_name(op)} is not defined on {common.name()}")
    }
    if common == Dtype.Datetime {
        throw TypeError(f"{binary_name(op)} on two timestamps is a duration, which timeseries owns")
    }
    #: Division and exponentiation leave the integers whatever went in: 7 // 2
    #: is exact, but 7 / 2 is not an integer and 2 ** 40 is past the 48 bits
    #: two lanes hold.
    if op == OP_DIVIDE or op == OP_POWER { return Dtype.Float64 }
    return common
}

#: Whether the operation can turn present operands into a missing result, and
#: so needs a validity lane even when neither input had one.
fn binary_may_null(op: int, exact: bool) -> bool {
    if is_logical(op) { return false }
    if exact { return op == OP_FLOOR_DIVIDE or op == OP_MODULO }
    return op == OP_DIVIDE or op == OP_FLOOR_DIVIDE or op == OP_MODULO or op == OP_POWER
}

fn unary_may_null(op: int) -> bool {
    return op == UN_LOG or op == UN_LOG2 or op == UN_LOG10 or op == UN_SQRT
}

#: One operand of a binary dispatch: a column, and how far the kernel walks it.
#:
#: A stride of 0 is a scalar — a one-row column every thread reads slot zero
#: of. Materialising the scalar to full length instead would cost an allocation
#: and a fill the size of the data, which for a `frame + 1` is the whole frame.
class Side {
    #: The lanes the kernel reads.
    pub var column: Column
    #: 1 for a real column, 0 for a broadcast scalar.
    pub var stride: int
    #: Whether this side allocated its column and must release it.
    pub var owned: bool

    pub fn init(self, column: Column, stride: int, owned: bool) {
        self.column = column
        self.stride = stride
        self.owned = owned
    }

    #: Release the column when this side built it.
    pub fn release(self) -> void {
        if self.owned { self.column.free() }
    }
}

#: The column behind a `Series` or a `Column`, or null for a scalar.
pub fn column_of(value: any) -> Column? {
    if value is Column { return value }
    if value is Series { return value.values }
    return null
}

#: A host scalar as a `Value`, typed against the column it will meet.
pub fn scalar_value(value: any, against: Dtype) -> Value {
    if value is Value { return value }
    if value is bool { return Value.of_bool(value) }
    if value is int { return Value.of_int(value) }
    if value is float { return Value.of_float(value) }
    if value is str {
        return Value.of_str(value, if against.is_dictionary() { against } else { Dtype.Str })
    }
    throw TypeError("an operand must be a Series, a DataFrame, a Column or a scalar")
}

#: The dtype a one-row column for `cell` should carry.
#:
#: An integer written next to a `Datetime` column is milliseconds, not a count,
#: and a string next to a `Category` is a category. Typing the scalar against
#: its neighbour is what keeps `stamps > 1700000000000` from being a promotion
#: error.
fn scalar_dtype(cell: Value, against: Dtype) -> Dtype {
    if against == Dtype.Datetime and cell.dtype == Dtype.Int64 { return Dtype.Datetime }
    if against == Dtype.Category and cell.dtype == Dtype.Str { return Dtype.Category }
    return cell.dtype
}

fn side_of(value: any, against: Dtype) -> Side {
    let existing = column_of(value)
    if existing is not null { return Side(existing, 1, false) }
    let cell = scalar_value(value, against)
    if cell.is_na() { return Side(Column.nulls(against, 1), 0, true) }
    let dtype = scalar_dtype(cell, against)
    return Side(Column.full(dtype, 1, cell), 0, true)
}

#: The two scalars the kernel reads a fill or a bound through.
#:
#: In exact mode they are the halves of the 48-bit integer; in float mode the
#: first is the float and the second is zero, because `wide_value` would read
#: it either way and a stale half would be a silent wrong answer.
fn packed_bits(cell: Value?, exact: bool) -> (int, int) {
    if cell is null { return (0, 0) }
    if exact {
        let split = wide_pack(cell.to_int())
        return (f32_bits(split[0]), f32_bits(split[1]))
    }
    return (f32_bits(cell.to_float()), 0)
}

#: Whether four rows a thread is safe for this shape.
#:
#: The quad path reads and writes whole float4s, so no lane may be the high
#: half of a two-lane integer and the rows must divide by four. Everything else
#: — a scalar operand, a validity lane, an exact comparison of two `Bool`
#: columns — it handles.
fn quad_span(rows: int, wide_anywhere: bool) -> int {
    if wide_anywhere or rows == 0 or rows % 4 != 0 { return 1 }
    return 4
}

#: Run `op` over two operands that already agree on length.
fn dispatch_binary(a: Side, b: Side, rows: int, op: int, fill: Value?) -> Column {
    require_device("evaluate an expression")
    let common = promote(a.column.dtype, b.column.dtype)
    let out_dtype = binary_result(op, common)
    #: Exact means the arithmetic runs in `long`. Everything integral goes
    #: that way, including a `Bool` operand, whose 0 and 1 are exact in either
    #: mode. Dictionary codes stay in float: they are small non-negative
    #: integers and only ever compared.
    let numeric = common != Dtype.Float64 and not common.is_dictionary()
    let exact = numeric and out_dtype != Dtype.Float64
    let out_wide = out_dtype.is_wide()
    let inputs_null = a.column.has_validity() or b.column.has_validity()
    let with_validity = inputs_null or binary_may_null(op, exact)
    let out = Column.allocate(out_dtype, rows, with_validity: with_validity)
    if rows == 0 { return out }
    let fill_bits = packed_bits(fill, exact)
    let wide_anywhere = a.column.dtype.is_wide() or b.column.dtype.is_wide() or out_wide
    let span = quad_span(rows, wide_anywhere)
    cached_kernel("frame_binary").dispatch_async(
        [
            a.column.values,
            a.column.wide ?? a.column.values,
            a.column.valid ?? a.column.values,
            b.column.values,
            b.column.wide ?? b.column.values,
            b.column.valid ?? b.column.values,
            out.values,
            out.wide ?? out.values,
            out.valid ?? out.values,
        ],
        (rows + span - 1) // span,
        [
            rows,
            op,
            span,
            flag(a.column.dtype.is_wide()),
            flag(b.column.dtype.is_wide()),
            flag(out_wide),
            flag(a.column.has_validity()),
            flag(b.column.has_validity()),
            flag(with_validity),
            a.stride,
            b.stride,
            flag(exact),
            flag(fill is not null),
            fill_bits[0],
            fill_bits[1],
        ]
    )
    return out
}

pub fn dispatch_unary(source: Column, op: int, arg_a: Value?, arg_b: Value?) -> Column {
    require_device("evaluate an expression")
    let out_dtype = unary_result(op, source.dtype)
    let exact = out_dtype.is_wide()
    let with_validity = source.has_validity() or unary_may_null(op)
    let out = Column.allocate(out_dtype, source.len(), with_validity: with_validity)
    let rows = source.len()
    if rows == 0 { return out }
    let a_bits = packed_bits(arg_a, exact)
    let b_bits = packed_bits(arg_b, exact)
    let span = quad_span(rows, source.dtype.is_wide() or exact)
    cached_kernel("frame_unary").dispatch_async(
        [
            source.values,
            source.wide ?? source.values,
            source.valid ?? source.values,
            out.values,
            out.wide ?? out.values,
            out.valid ?? out.values,
        ],
        (rows + span - 1) // span,
        [
            rows,
            op,
            span,
            flag(source.dtype.is_wide()),
            flag(source.has_validity()),
            flag(with_validity),
            flag(exact),
            flag(arg_a is not null),
            a_bits[0],
            a_bits[1],
            flag(arg_b is not null),
            b_bits[0],
            b_bits[1],
        ]
    )
    return out
}

#: Every part's dictionary rebuilt onto one table whose codes are ranks.
#:
#: A text comparison is an integer compare on the codes, which only answers
#: `less` if the codes are in the strings' order. So the shared table is seeded
#: with the sorted union before `unify_levels` walks the parts, and the codes
#: it hands back are ranks. The sort is over the distinct values, not the rows:
#: a million-row city column with fifty cities sorts fifty strings.
fn on_one_ordered_table(parts: list[Column]) -> list[Column] {
    var names: list[str] = []
    var seen: dict[str, int] = {}
    for part in parts {
        for name in part.levels ?? [] {
            if seen.get(name, -1) >= 0 { continue }
            seen[name] = 1
            names.push(name)
        }
    }
    let seed = Column.from_strs([], parts[0].dtype)
    seed.set_levels(sort_by(names, natural_order))
    let unified = unify_levels([seed] + parts)
    seed.free()
    unified[0].free()
    return unified[1:unified.len()]
}

#: A one-row dictionary column whose code no level can ever match.
#:
#: `level_code` says a scalar the dictionary has never seen is in it, and this
#: is how that becomes an answer rather than a special case: comparing against
#: -1 gives false on every row in one dispatch, and the kernel carries the
#: source's nulls the way it always does. The alternative is remapping both
#: sides through `unify_levels` to discover the same thing.
fn absent_code_column(dtype: Dtype) -> Column {
    let column = Column.allocate(dtype, 1)
    column.values.fill(-1.0)
    return column
}

#: Whether `side` is a scalar string the other column's dictionary lacks.
fn scalar_is_unseen(side: Side, other: Column) -> bool {
    if side.stride != 0 or side.column.level_count() != 1 { return false }
    if not other.dtype.is_dictionary() or other.level_count() == 0 { return false }
    return other.level_code(side.column.level_at(0)) is null
}

#: Comparison between dictionary-encoded operands.
fn text_binary(a: Side, b: Side, rows: int, op: int) -> Column {
    if not is_comparison(op) {
        throw TypeError(f"{binary_name(op)} is not defined on text")
    }
    if not (a.column.dtype.is_dictionary() and b.column.dtype.is_dictionary()) {
        throw TypeError(f"cannot compare {a.column.dtype.name()} with {b.column.dtype.name()}")
    }
    if not is_ordering(op) {
        if scalar_is_unseen(a, b.column) {
            let sentinel = absent_code_column(a.column.dtype)
            let out = dispatch_binary(Side(sentinel, 0, false), b, rows, op, null)
            sentinel.free()
            return out
        }
        if scalar_is_unseen(b, a.column) {
            let sentinel = absent_code_column(b.column.dtype)
            let out = dispatch_binary(a, Side(sentinel, 0, false), rows, op, null)
            sentinel.free()
            return out
        }
        let unified = unify_levels([a.column, b.column])
        let out = dispatch_binary(
            Side(unified[0], a.stride, false),
            Side(unified[1], b.stride, false),
            rows,
            op,
            null
        )
        for part in unified { part.free() }
        return out
    }
    let ordered = on_one_ordered_table([a.column, b.column])
    let out = dispatch_binary(
        Side(ordered[0], a.stride, false),
        Side(ordered[1], b.stride, false),
        rows,
        op,
        null
    )
    for part in ordered { part.free() }
    return out
}

#: `op` over two columns, or a column and a scalar. No labels are consulted.
pub fn plain_binary(left: any, right: any, op: int, fill: Value?) -> Column {
    let a_column = column_of(left)
    let b_column = column_of(right)
    if a_column is null and b_column is null {
        throw TypeError(f"{binary_name(op)} needs a column on at least one side")
    }
    if a_column is not null and b_column is not null and a_column.len() != b_column.len() {
        throw ValueError(
            f"{a_column.len()} rows against {b_column.len()}: align them before {binary_name(op)}"
        )
    }
    let rows = if a_column is not null { a_column.len() } else { b_column.len() }
    let left_dtype = if a_column is not null { a_column.dtype } else { Dtype.Float64 }
    let right_dtype = if b_column is not null { b_column.dtype } else { Dtype.Float64 }
    let a = side_of(left, right_dtype)
    let b = side_of(right, left_dtype)
    let out = if a.column.dtype.is_dictionary() or b.column.dtype.is_dictionary() {
        text_binary(a, b, rows, op)
    } else {
        dispatch_binary(a, b, rows, op, fill)
    }
    a.release()
    b.release()
    return out
}

#: The named operations, each one an op code handed to `binary` or `unary`.
from ..column import Column, DataFrame, Dtype, Series, Value
from .codes import OP_ADD, OP_AND, OP_DIVIDE, OP_EQUAL, OP_FLOOR_DIVIDE, OP_GREATER
from .codes import OP_GREATER_EQUAL, OP_LESS, OP_LESS_EQUAL, OP_MODULO, OP_MULTIPLY
from .codes import OP_NOT_EQUAL, OP_OR, OP_POWER, OP_SUBTRACT, OP_XOR
from .codes import UN_ABSOLUTE, UN_CEILING, UN_CLIP, UN_EXP, UN_FLOOR, UN_LOG
from .codes import UN_LOG10, UN_LOG2, UN_NEGATE, UN_NOT, UN_RECIPROCAL, UN_ROUND
from .codes import UN_SIGN, UN_SQRT, UN_SQUARE, UN_TRUNCATE
from .dispatch import dispatch_unary, scalar_value
from .frames import binary, unary
from .rules import unary_result

#: Elementwise sum. `fill_value` stands in for a missing operand on one side.
pub fn add(left: any, right: any, fill_value: Value? = null) -> any {
    return binary(left, right, OP_ADD, fill_value)
}

#: Elementwise difference.
pub fn subtract(left: any, right: any, fill_value: Value? = null) -> any {
    return binary(left, right, OP_SUBTRACT, fill_value)
}

#: Elementwise product.
pub fn multiply(left: any, right: any, fill_value: Value? = null) -> any {
    return binary(left, right, OP_MULTIPLY, fill_value)
}

#: Elementwise quotient, always `Float64`.
#:
#: Even over two `Int64` columns: 7 / 2 is not an integer, and pandas' `/` is
#: true division. `floor_divide` is the one that stays exact.
pub fn divide(left: any, right: any, fill_value: Value? = null) -> any {
    return binary(left, right, OP_DIVIDE, fill_value)
}

#: Elementwise `//`: the quotient rounded toward negative infinity.
#:
#: Exact over integers, and a zero divisor there gives a null rather than a
#: wrong number — integer division by zero has no answer, and the device does
#: not fault on one. Over floats it is `floor(a / b)`, so a zero divisor gives
#: the infinity a float division gives.
pub fn floor_divide(left: any, right: any, fill_value: Value? = null) -> any {
    return binary(left, right, OP_FLOOR_DIVIDE, fill_value)
}

#: Elementwise `%`, with the sign of the divisor as Python's has.
pub fn modulo(left: any, right: any, fill_value: Value? = null) -> any {
    return binary(left, right, OP_MODULO, fill_value)
}

#: Elementwise `**`, always `Float64`.
#:
#: `pow` is a float function on the device, and an exact integer power leaves
#: the 48 bits two lanes hold at 6 ** 18, so promoting is the honest answer
#: rather than one that is exact for a while.
pub fn power(left: any, right: any, fill_value: Value? = null) -> any {
    return binary(left, right, OP_POWER, fill_value)
}

#: Elementwise `==`, as a `Bool` column.
#:
#: A null on either side makes the answer null, not false — this module's
#: columns are nullable, so a comparison propagates the way pandas' nullable
#: dtypes do. `fillna(false)` over the result is the numpy-backed behaviour.
#: Text compares as codes on one dictionary, so it is an integer compare.
pub fn equal(left: any, right: any, fill_value: Value? = null) -> any {
    return binary(left, right, OP_EQUAL, fill_value)
}

#: Elementwise `!=`, as a `Bool` column.
pub fn not_equal(left: any, right: any, fill_value: Value? = null) -> any {
    return binary(left, right, OP_NOT_EQUAL, fill_value)
}

#: Elementwise `<`, as a `Bool` column.
#:
#: Over text this is the strings' order, reached by putting both dictionaries
#: on one table whose codes are ranks. The sort behind that is over the
#: distinct values, not the rows.
pub fn less(left: any, right: any, fill_value: Value? = null) -> any {
    return binary(left, right, OP_LESS, fill_value)
}

#: Elementwise `<=`, as a `Bool` column.
pub fn less_equal(left: any, right: any, fill_value: Value? = null) -> any {
    return binary(left, right, OP_LESS_EQUAL, fill_value)
}

#: Elementwise `>`, as a `Bool` column.
pub fn greater(left: any, right: any, fill_value: Value? = null) -> any {
    return binary(left, right, OP_GREATER, fill_value)
}

#: Elementwise `>=`, as a `Bool` column.
pub fn greater_equal(left: any, right: any, fill_value: Value? = null) -> any {
    return binary(left, right, OP_GREATER_EQUAL, fill_value)
}

#: Elementwise conjunction over `Bool` columns.
#:
#: Three-valued, as pandas' nullable booleans are: `false and null` is false,
#: because a present false settles the answer whatever the unknown side holds.
#: Only `null and null` and `true and null` are unknown.
pub fn and_(left: any, right: any) -> any { return binary(left, right, OP_AND, null) }

#: Elementwise disjunction, three-valued: `true or null` is true.
pub fn or_(left: any, right: any) -> any { return binary(left, right, OP_OR, null) }

#: Elementwise exclusive or. Unknown on either side makes it unknown — neither
#: value settles an inequality.
pub fn xor(left: any, right: any) -> any { return binary(left, right, OP_XOR, null) }

#: Elementwise negation of a `Bool` column.
pub fn not_(target: any) -> any { return unary(target, UN_NOT, null, null) }

#: Elementwise `-x`. `Bool` becomes `Int64`, as it does under `+`.
pub fn negate(target: any) -> any { return unary(target, UN_NEGATE, null, null) }

#: Elementwise magnitude.
pub fn absolute(target: any) -> any { return unary(target, UN_ABSOLUTE, null, null) }

#: -1, 0 or 1 by the sign of each value; a null stays null.
pub fn sign(target: any) -> any { return unary(target, UN_SIGN, null, null) }

#: 10 to the `n`, for `n` of either sign.
fn power_of_ten(n: int) -> float {
    var scale = 1.0
    let steps = if n < 0 { -n } else { n }
    for _step in 0..steps { scale *= 10.0 }
    return if n < 0 { 1.0 / scale } else { scale }
}

#: The integer 10 to the `n`, for a non-negative `n`.
fn integer_power_of_ten(n: int) -> int {
    var scale = 1
    for _step in 0..n { scale *= 10 }
    return scale
}

#: Round to `decimals` places, ties to even as numpy rounds.
#:
#: An integer column is already rounded at `decimals` 0 and above; below it the
#: value is snapped to a multiple of ten in `long`, so a `1234.round(-2)` stays
#: exact rather than going through a float.
pub fn round(target: any, decimals: int = 0) -> any {
    let numeric = Value.of_float(power_of_ten(decimals))
    let exact = Value.of_int(if decimals < 0 { integer_power_of_ten(-decimals) } else { 1 })
    if target is DataFrame or target is Series or target is Column {
        return round_over(target, numeric, exact)
    }
    throw TypeError("round needs a Series, a DataFrame or a Column")
}

#: `round` has to pick its argument by the column's dtype, which is per column
#: in a frame, so it dispatches a column at a time rather than through `unary`.
fn round_over(target: any, numeric: Value, exact: Value) -> any {
    if target is DataFrame {
        var columns: list[Column] = []
        for column in target.columns { columns.push(round_column(column, numeric, exact)) }
        return target._replace(columns: columns, index: target.index.copy())
    }
    if target is Series {
        return Series(
            round_column(target.values, numeric, exact),
            index: target.index.copy(),
            name: target.name
        )
    }
    return round_column(target, numeric, exact)
}

fn round_column(column: Column, numeric: Value, exact: Value) -> Column {
    let argument = if unary_result(UN_ROUND, column.dtype).is_wide() { exact } else { numeric }
    return dispatch_unary(column, UN_ROUND, argument, null)
}

#: Elementwise round toward negative infinity.
pub fn floor(target: any) -> any { return unary(target, UN_FLOOR, null, null) }

#: Elementwise round toward positive infinity.
pub fn ceiling(target: any) -> any { return unary(target, UN_CEILING, null, null) }

#: Elementwise round toward zero.
pub fn truncate(target: any) -> any { return unary(target, UN_TRUNCATE, null, null) }

#: Elementwise `e ** x`, always `Float64`.
pub fn exp(target: any) -> any { return unary(target, UN_EXP, null, null) }

#: Elementwise natural log. A non-positive value has none, so it becomes null.
pub fn log(target: any) -> any { return unary(target, UN_LOG, null, null) }

#: Elementwise base-2 log.
pub fn log2(target: any) -> any { return unary(target, UN_LOG2, null, null) }

#: Elementwise base-10 log.
pub fn log10(target: any) -> any { return unary(target, UN_LOG10, null, null) }

#: Elementwise square root. A negative value has none, so it becomes null.
pub fn sqrt(target: any) -> any { return unary(target, UN_SQRT, null, null) }

#: Elementwise `x * x`, keeping an integer column integral.
pub fn square(target: any) -> any { return unary(target, UN_SQUARE, null, null) }

#: Elementwise `1 / x`, always `Float64`. A zero gives an infinity, as pandas'
#: division does, rather than a null.
pub fn reciprocal(target: any) -> any { return unary(target, UN_RECIPROCAL, null, null) }

#: A bound as a `Value`, so a caller may write a bare number.
fn bound_value(bound: any) -> Value? {
    if bound is null { return null }
    if bound is Value { return if bound.is_na() { null } else { bound } }
    return scalar_value(bound, Dtype.Float64)
}

#: Every value held inside `[lower, upper]`. Either bound may be left out.
#:
#: A null row stays null: clipping is about the values that are there.
pub fn clip(target: any, lower: any = null, upper: any = null) -> any {
    return unary(target, UN_CLIP, bound_value(lower), bound_value(upper))
}

#: Elementwise arithmetic, comparison, logic and the host callbacks.
#:
#: Almost nothing here is checked against a number this module printed. What
#: is checked is the definition: that `a` equals `(a // b) * b + (a % b)` for
#: both signs, that exactly one of `<`, `==`, `>` holds on every row, that
#: `exp` undoes `log`, that a comparison agrees with the host's own `<` over
#: the same strings. Golden numbers only appear where the definition is the
#: number — the tie-breaking of `round`, the truth table of a three-valued
#: `and`.
#:
#: Two shapes matter beyond the maths. The kernel takes four rows a thread when
#: the row count divides by four, which is a different code path from the one
#: every small test exercises, so a 1024-row column and a 1022-row column have
#: to agree. And a scalar operand is a one-slot column read with a stride of
#: zero rather than a full-length copy, so it has to give what a full-length
#: operand gives.
import std.gpu as gpu
from std.math import NAN, is_nan
from std.test import assert_eq, assert_false, assert_near, assert_throws, assert_true
from jaiframe.column import Column, DataFrame, Dtype, Index, Series, Value
from jaiframe.arith import absolute, add, and_, ceiling, clip, combine, divide, equal
from jaiframe.arith import exp, floor, floor_divide, greater, greater_equal, less
from jaiframe.arith import less_equal, log, map, modulo, multiply, negate, not_
from jaiframe.arith import not_equal, or_, power, promote, reciprocal, round, sign
from jaiframe.arith import sqrt, square, subtract, truncate, xor
from jaiframe.arith import apply, applymap, pipe

fn test_promote_walks_the_pandas_ladder() -> void {
    assert_eq(promote(Dtype.Int64, Dtype.Int64), Dtype.Int64)
    assert_eq(promote(Dtype.Int64, Dtype.Float64), Dtype.Float64)
    assert_eq(promote(Dtype.Bool, Dtype.Int64), Dtype.Int64)
    assert_eq(promote(Dtype.Bool, Dtype.Bool), Dtype.Int64)
    assert_eq(promote(Dtype.Datetime, Dtype.Datetime), Dtype.Datetime)
    assert_eq(promote(Dtype.Str, Dtype.Category), Dtype.Str)
    assert_throws(TypeError, || promote(Dtype.Str, Dtype.Float64))
    assert_throws(TypeError, || promote(Dtype.Datetime, Dtype.Int64))
}

fn test_arithmetic_matches_the_host_definition() -> void {
    if not gpu.is_available() { return }
    let left = Column.from_floats([1.5, -2.25, 0.0, 7.75])
    defer { left.free() }
    let right = Column.from_floats([2.0, 4.0, -3.5, 0.25])
    defer { right.free() }
    let a = left.to_floats()
    let b = right.to_floats()
    let sum = add(left, right)
    defer { sum.free() }
    let difference = subtract(left, right)
    defer { difference.free() }
    let product = multiply(left, right)
    defer { product.free() }
    let quotient = divide(left, right)
    defer { quotient.free() }
    let summed = sum.to_floats()
    let differenced = difference.to_floats()
    let multiplied = product.to_floats()
    let divided = quotient.to_floats()
    for row in 0..a.len() {
        assert_near(summed[row], a[row] + b[row], tolerance: 1e-6)
        assert_near(differenced[row], a[row] - b[row], tolerance: 1e-6)
        assert_near(multiplied[row], a[row] * b[row], tolerance: 1e-6)
        assert_near(divided[row], a[row] / b[row], tolerance: 1e-6)
    }
}

#: 1700000000123 is past 2^24, so a column that flattened its two lanes into
#: one float would answer this wrong by thousands.
fn test_integer_arithmetic_stays_exact_past_two_to_the_24() -> void {
    if not gpu.is_available() { return }
    let stamps = Column.from_ints([1700000000123, -1700000000123, 8388609])
    defer { stamps.free() }
    let shifted = add(stamps, 7)
    defer { shifted.free() }
    assert_eq(shifted.dtype, Dtype.Int64)
    assert_eq(shifted.to_ints(), [1700000000130, -1700000000116, 8388616])
    let doubled = multiply(stamps, 2)
    defer { doubled.free() }
    assert_eq(doubled.to_ints(), [3400000000246, -3400000000246, 16777218])
}

fn test_division_always_leaves_the_integers() -> void {
    if not gpu.is_available() { return }
    let numbers = Column.from_ints([7, 8, 9])
    defer { numbers.free() }
    let halved = divide(numbers, 2)
    defer { halved.free() }
    assert_eq(halved.dtype, Dtype.Float64)
    assert_near(halved.to_floats()[0], 3.5, tolerance: 1e-6)
    let floored = floor_divide(numbers, 2)
    defer { floored.free() }
    assert_eq(floored.dtype, Dtype.Int64)
    assert_eq(floored.to_ints(), [3, 4, 4])
    let raised = power(numbers, 2)
    defer { raised.free() }
    assert_eq(raised.dtype, Dtype.Float64)
    assert_near(raised.to_floats()[0], 49.0, tolerance: 1e-6)
}

#: The defining identity of a floored quotient and its remainder, over every
#: combination of signs. A truncating divide fails this on three of the four.
fn test_floor_divide_and_modulo_satisfy_their_identity() -> void {
    if not gpu.is_available() { return }
    let left = Column.from_ints([7, -7, 7, -7, 0, 1700000000123])
    defer { left.free() }
    let right = Column.from_ints([3, 3, -3, -3, 5, 1000])
    defer { right.free() }
    let quotient = floor_divide(left, right)
    defer { quotient.free() }
    let remainder = modulo(left, right)
    defer { remainder.free() }
    let scaled = multiply(quotient, right)
    defer { scaled.free() }
    let rebuilt = add(scaled, remainder)
    defer { rebuilt.free() }
    assert_eq(rebuilt.to_ints(), left.to_ints())
    let divisors = right.to_ints()
    for (row, value) in remainder.to_ints().enumerate() {
        assert_true(value == 0 or (value > 0) == (divisors[row] > 0), f"row {row} takes its sign")
        let size = if value < 0 { -value } else { value }
        let limit = if divisors[row] < 0 { -divisors[row] } else { divisors[row] }
        assert_true(size < limit, f"row {row} is smaller than its divisor")
    }
}

fn test_float_modulo_takes_the_divisors_sign() -> void {
    if not gpu.is_available() { return }
    let left = Column.from_floats([7.5, -7.5, 7.5, -7.5])
    defer { left.free() }
    let right = Column.from_floats([3.0, 3.0, -3.0, -3.0])
    defer { right.free() }
    let remainder = modulo(left, right)
    defer { remainder.free() }
    let quotient = floor_divide(left, right)
    defer { quotient.free() }
    let a = left.to_floats()
    let b = right.to_floats()
    let q = quotient.to_floats()
    for (row, value) in remainder.to_floats().enumerate() {
        assert_near(q[row] * b[row] + value, a[row], tolerance: 1e-5)
        assert_true(value == 0.0 or (value > 0.0) == (b[row] > 0.0), f"row {row} takes its sign")
    }
}

fn test_a_zero_divisor_is_a_null_over_integers() -> void {
    if not gpu.is_available() { return }
    let left = Column.from_ints([7, 7])
    defer { left.free() }
    let right = Column.from_ints([0, 2])
    defer { right.free() }
    let quotient = floor_divide(left, right)
    defer { quotient.free() }
    assert_eq(quotient.null_mask(), [true, false])
    assert_eq(quotient.get(1).to_int(), 3)
    let remainder = modulo(left, right)
    defer { remainder.free() }
    assert_eq(remainder.null_mask(), [true, false])
}

#: 0/0 is NaN, which is how a float column spells missing; 1/0 is an infinity,
#: which pandas keeps as a value.
fn test_a_float_nan_becomes_a_null_but_an_infinity_does_not() -> void {
    if not gpu.is_available() { return }
    let left = Column.from_floats([0.0, 1.0, -1.0])
    defer { left.free() }
    let right = Column.from_floats([0.0, 0.0, 0.0])
    defer { right.free() }
    let quotient = divide(left, right)
    defer { quotient.free() }
    assert_eq(quotient.null_mask(), [true, false, false])
    assert_true(quotient.to_floats()[1] > 1e30, "one over zero is an infinity")
}

fn test_nulls_propagate_and_a_fill_value_stands_in() -> void {
    if not gpu.is_available() { return }
    let left = Column.from_floats([1.0, NAN, NAN, 4.0])
    defer { left.free() }
    let right = Column.from_floats([10.0, 20.0, NAN, NAN])
    defer { right.free() }
    let plain = add(left, right)
    defer { plain.free() }
    assert_eq(plain.null_mask(), [false, true, true, true])
    assert_near(plain.to_floats()[0], 11.0, tolerance: 1e-6)
    let filled = add(left, right, fill_value: Value.of_float(100.0))
    defer { filled.free() }
    #: One missing side takes the fill; missing on both stays missing.
    assert_eq(filled.null_mask(), [false, false, true, false])
    assert_near(filled.to_floats()[1], 120.0, tolerance: 1e-6)
    assert_near(filled.to_floats()[3], 104.0, tolerance: 1e-6)
}

#: Exactly one of `<`, `==`, `>` holds on every row, and the non-strict forms
#: are the strict ones or equality. Nothing here reads a number this module
#: produced except through those laws.
fn test_comparisons_are_a_trichotomy() -> void {
    if not gpu.is_available() { return }
    let left = Column.from_floats([1.0, 2.0, 3.0, -4.5])
    defer { left.free() }
    let right = Column.from_floats([2.0, 2.0, 1.0, -4.5])
    defer { right.free() }
    let below = less(left, right)
    defer { below.free() }
    let same = equal(left, right)
    defer { same.free() }
    let above = greater(left, right)
    defer { above.free() }
    let differ = not_equal(left, right)
    defer { differ.free() }
    let at_most = less_equal(left, right)
    defer { at_most.free() }
    let at_least = greater_equal(left, right)
    defer { at_least.free() }
    let a = below.to_bools()
    let b = same.to_bools()
    let c = above.to_bools()
    for row in 0..left.len() {
        let hits = (if a[row] {
            1
        } else {
            0
        }) + (if b[row] {
            1
        } else {
            0
        }) + (if c[row] {
            1
        } else {
            0
        })
        assert_eq(hits, 1, f"row {row} satisfies exactly one ordering")
        assert_eq(differ.to_bools()[row], not b[row])
        assert_eq(at_most.to_bools()[row], a[row] or b[row])
        assert_eq(at_least.to_bools()[row], c[row] or b[row])
    }
}

fn test_a_comparison_propagates_a_null() -> void {
    if not gpu.is_available() { return }
    let values = Column.from_floats([1.0, NAN, 3.0])
    defer { values.free() }
    let hot = greater(values, 2.0)
    defer { hot.free() }
    assert_eq(hot.dtype, Dtype.Bool)
    assert_eq(hot.null_mask(), [false, true, false])
    assert_eq(hot.to_bools(), [false, false, true])
}

fn test_integer_comparison_is_exact_past_two_to_the_24() -> void {
    if not gpu.is_available() { return }
    let stamps = Column.from_ints([1700000000122, 1700000000123, 1700000000124])
    defer { stamps.free() }
    let after = greater(stamps, 1700000000123)
    defer { after.free() }
    assert_eq(after.to_bools(), [false, false, true])
}

#: All nine pairings of true, false and unknown, checked against the Kleene
#: rules written out below rather than against a table read off the kernel.
#: The two that make it three-valued rather than propagating are called out
#: afterwards: a present false settles an `and` and a present true settles an
#: `or`, so neither has to know what the unknown side held.
#:
#: The left operand runs true, false, unknown three times against a right that
#: holds each of the three for three rows, so row 7 is the present false meeting
#: an unknown and row 6 is the present true meeting one. The left side used to
#: be missing on rows 6 and 7 as well, which made three rows unknown against
#: unknown and left the two rows this test names for its point uncovered.
#: Checked against `pandas.array(..., dtype="boolean")`, whose `&` answers
#: false for (false, NA) and `|` answers true for (true, NA).
fn test_three_valued_logic_settles_what_it_can() -> void {
    if not gpu.is_available() { return }
    let values = [true, false, true, true, false, true, true, false, true]
    let missing = [false, false, true, false, false, true, false, false, true]
    let other = [true, true, true, false, false, false, true, false, false]
    let other_missing = [false, false, false, false, false, false, true, true, true]
    let left = Column.from_bools(values, missing)
    defer { left.free() }
    let right = Column.from_bools(other, other_missing)
    defer { right.free() }
    let both = and_(left, right)
    defer { both.free() }
    let either = or_(left, right)
    defer { either.free() }
    let odd = xor(left, right)
    defer { odd.free() }
    let flipped = not_(left)
    defer { flipped.free() }
    let and_mask = both.null_mask()
    let or_mask = either.null_mask()
    let xor_mask = odd.null_mask()
    let and_values = both.to_bools()
    let or_values = either.to_bools()
    let flipped_mask = flipped.null_mask()
    let flipped_values = flipped.to_bools()
    for row in 0..values.len() {
        let a_false = not missing[row] and not values[row]
        let a_true = not missing[row] and values[row]
        let b_false = not other_missing[row] and not other[row]
        let b_true = not other_missing[row] and other[row]
        let unknown = missing[row] or other_missing[row]
        assert_eq(and_mask[row], not (a_false or b_false) and unknown, f"and row {row}")
        assert_eq(and_values[row], a_true and b_true, f"and value row {row}")
        assert_eq(or_mask[row], not (a_true or b_true) and unknown, f"or row {row}")
        assert_eq(or_values[row], a_true or b_true, f"or value row {row}")
        assert_eq(xor_mask[row], unknown, f"xor row {row}")
        assert_eq(flipped_mask[row], missing[row], f"not row {row}")
        assert_eq(flipped_values[row], a_false, f"not value row {row}")
    }
    assert_false(and_mask[7], "a present false settles an and against an unknown")
    assert_false(and_values[7], "and it settles it false")
    assert_false(or_mask[6], "a present true settles an or against an unknown")
    assert_true(or_values[6], "and it settles it true")
}

fn test_unary_inverses_agree() -> void {
    if not gpu.is_available() { return }
    let values = Column.from_floats([0.5, 1.0, 2.0, 7.25])
    defer { values.free() }
    let logged = log(values)
    defer { logged.free() }
    let restored = exp(logged)
    defer { restored.free() }
    let squared = square(values)
    defer { squared.free() }
    let rooted = sqrt(squared)
    defer { rooted.free() }
    let flipped = reciprocal(values)
    defer { flipped.free() }
    let source = values.to_floats()
    for row in 0..source.len() {
        assert_near(restored.to_floats()[row], source[row], tolerance: 1e-5)
        assert_near(rooted.to_floats()[row], source[row], tolerance: 1e-5)
        assert_near(flipped.to_floats()[row] * source[row], 1.0, tolerance: 1e-5)
    }
}

fn test_a_log_of_a_non_positive_value_is_a_null() -> void {
    if not gpu.is_available() { return }
    let values = Column.from_floats([1.0, 0.0, -2.0])
    defer { values.free() }
    let logged = log(values)
    defer { logged.free() }
    #: log(0) is negative infinity, which is a value; log(-2) is not a number.
    assert_eq(logged.null_mask(), [false, false, true])
    let rooted = sqrt(values)
    defer { rooted.free() }
    assert_eq(rooted.null_mask(), [false, false, true])
}

fn test_magnitude_and_sign_reconstruct_a_value() -> void {
    if not gpu.is_available() { return }
    let values = Column.from_floats([-3.5, 0.0, 2.25])
    defer { values.free() }
    let size = absolute(values)
    defer { size.free() }
    let direction = sign(values)
    defer { direction.free() }
    let rebuilt = multiply(size, direction)
    defer { rebuilt.free() }
    let negated = negate(values)
    defer { negated.free() }
    let source = values.to_floats()
    for row in 0..source.len() {
        assert_near(rebuilt.to_floats()[row], source[row], tolerance: 1e-6)
        assert_near(negated.to_floats()[row], -source[row], tolerance: 1e-6)
    }
}

#: floor, ceiling and truncate bracket a value the way their definitions say,
#: and truncate is the one that agrees with floor only above zero.
fn test_rounding_directions_bracket_a_value() -> void {
    if not gpu.is_available() { return }
    let values = Column.from_floats([2.7, -2.7, 4.0, -0.25])
    defer { values.free() }
    let down = floor(values)
    defer { down.free() }
    let up = ceiling(values)
    defer { up.free() }
    let toward_zero = truncate(values)
    defer { toward_zero.free() }
    let source = values.to_floats()
    for row in 0..source.len() {
        assert_true(down.to_floats()[row] <= source[row], f"row {row} floors below")
        assert_true(up.to_floats()[row] >= source[row], f"row {row} ceils above")
        let gap = up.to_floats()[row] - down.to_floats()[row]
        assert_true(gap == 0.0 or gap == 1.0, f"row {row} spans at most one")
        let expected = if source[row] < 0.0 { up.to_floats()[row] } else { down.to_floats()[row] }
        assert_near(toward_zero.to_floats()[row], expected, tolerance: 1e-6)
    }
}

#: Ties to even, which is numpy's rule and not the one a schoolbook gives.
fn test_round_breaks_ties_to_even() -> void {
    if not gpu.is_available() { return }
    let values = Column.from_floats([0.5, 1.5, 2.5, -0.5, -1.5, 2.4])
    defer { values.free() }
    let rounded = round(values)
    defer { rounded.free() }
    assert_eq(rounded.to_ints(), [0, 2, 2, 0, -2, 2])
    let places = Column.from_floats([1.2345, -1.2345])
    defer { places.free() }
    let trimmed = round(places, 2)
    defer { trimmed.free() }
    assert_near(trimmed.to_floats()[0], 1.23, tolerance: 1e-6)
    assert_near(trimmed.to_floats()[1], -1.23, tolerance: 1e-6)
}

#: An integer column rounds in `long`, so the answer is exact where a float
#: round trip would have lost the low bits.
fn test_rounding_an_integer_snaps_to_a_multiple() -> void {
    if not gpu.is_available() { return }
    let values = Column.from_ints([1234, 1250, 1350, -1250, 1700000000123])
    defer { values.free() }
    let kept = round(values, 2)
    defer { kept.free() }
    assert_eq(kept.dtype, Dtype.Int64)
    assert_eq(kept.to_ints(), values.to_ints())
    let hundreds = round(values, -2)
    defer { hundreds.free() }
    assert_eq(hundreds.to_ints(), [1200, 1200, 1400, -1200, 1700000000100])
}

fn test_clip_holds_values_between_the_bounds() -> void {
    if not gpu.is_available() { return }
    let values = Column.from_floats([-5.0, 0.5, 12.0, NAN])
    defer { values.free() }
    let held = clip(values, 0.0, 10.0)
    defer { held.free() }
    assert_eq(held.null_mask(), [false, false, false, true])
    for (row, value) in held.to_floats().enumerate() {
        if is_nan(value) { continue }
        assert_true(value >= 0.0 and value <= 10.0, f"row {row} sits inside the bounds")
    }
    assert_near(held.to_floats()[1], 0.5, tolerance: 1e-6)
    let only_below = clip(values, 0.0)
    defer { only_below.free() }
    assert_near(only_below.to_floats()[2], 12.0, tolerance: 1e-6)
    let stamps = Column.from_ints([1000, 1700000000123], dtype: Dtype.Datetime)
    defer { stamps.free() }
    let window = clip(stamps, Value.of_datetime(2000), Value.of_datetime(1700000000000))
    defer { window.free() }
    assert_eq(window.dtype, Dtype.Datetime)
    assert_eq(window.to_ints(), [2000, 1700000000000])
}

#: The ramp a range index materialises, as floats, built on the device.
fn ramp(rows: int) -> Column {
    let levels = Index.range(rows).materialize()
    let counted = levels[0]
    let out = counted.cast(Dtype.Float64)
    counted.free()
    return out
}

#: The kernel takes four rows a thread when the count divides by four, which
#: is an entirely separate branch from the one every other test here runs. A
#: 1024-row column and a 1022-row one have to give the same answers.
fn test_the_four_wide_path_agrees_with_the_scalar_one() -> void {
    if not gpu.is_available() { return }
    for rows in [1024, 1022] {
        let counted = ramp(rows)
        defer { counted.free() }
        let tripled = multiply(counted, 3.0)
        defer { tripled.free() }
        let scaled = add(tripled, 1.0)
        defer { scaled.free() }
        let values = scaled.to_floats()
        assert_eq(values.len(), rows)
        for row in [0, 1, 2, 3, 4, 7, 500, rows - 1] {
            assert_near(values[row], float(row) * 3.0 + 1.0, tolerance: 1e-3)
        }
        let hot = greater(counted, float(rows - 2))
        defer { hot.free() }
        assert_eq(hot.to_bools()[rows - 1], true)
        assert_eq(hot.to_bools()[rows - 2], false)
    }
}

fn test_the_four_wide_path_carries_the_validity_lane() -> void {
    if not gpu.is_available() { return }
    let counted = ramp(1024)
    defer { counted.free() }
    counted.set(5, Value.null_of(Dtype.Float64))
    counted.set(1023, Value.null_of(Dtype.Float64))
    let shifted = add(counted, 1.0)
    defer { shifted.free() }
    assert_eq(shifted.null_count(), 2)
    let mask = shifted.null_mask()
    assert_true(mask[5] and mask[1023], "the missing rows survive four to a thread")
    assert_false(mask[4] or mask[6], "their neighbours do not")
}

#: A scalar is a one-slot column the kernel reads with a stride of zero. It has
#: to give what a full-length operand of the same value gives.
fn test_a_scalar_broadcasts_without_a_column_of_copies() -> void {
    if not gpu.is_available() { return }
    let values = Column.from_floats([1.0, 2.0, 3.0])
    defer { values.free() }
    let spread = Column.full(Dtype.Float64, 3, Value.of_float(4.0))
    defer { spread.free() }
    let against_scalar = subtract(values, 4.0)
    defer { against_scalar.free() }
    let against_column = subtract(values, spread)
    defer { against_column.free() }
    assert_eq(against_scalar.to_floats(), against_column.to_floats())
    let reversed = subtract(4.0, values)
    defer { reversed.free() }
    assert_eq(reversed.to_floats(), [3.0, 2.0, 1.0])
    let missing = add(values, Value.null_of(Dtype.Float64))
    defer { missing.free() }
    assert_eq(missing.null_mask(), [true, true, true])
}

fn test_text_equality_is_a_code_compare() -> void {
    if not gpu.is_available() { return }
    let cities = Column.from_strs(["austin", "dallas", "austin", null])
    defer { cities.free() }
    let here = equal(cities, "austin")
    defer { here.free() }
    assert_eq(here.dtype, Dtype.Bool)
    assert_eq(here.to_bools(), [true, false, true, false])
    assert_eq(here.null_mask(), [false, false, false, true])
    let elsewhere = not_equal(cities, "austin")
    defer { elsewhere.free() }
    assert_eq(elsewhere.to_bools(), [false, true, false, false])
    #: A name the dictionary never held matches nothing, and that is one
    #: dispatch against an impossible code rather than a remap of both sides.
    let absent = equal(cities, "houston")
    defer { absent.free() }
    assert_eq(absent.to_bools(), [false, false, false, false])
    assert_eq(absent.null_mask(), [false, false, false, true])
    let other = Column.from_strs(["dallas", "dallas", "austin", "austin"])
    defer { other.free() }
    let paired = equal(cities, other)
    defer { paired.free() }
    assert_eq(paired.to_bools(), [false, true, true, false])
}

#: Checked against the host's own ordering of the same strings, not against a
#: table this module wrote.
fn test_text_ordering_follows_the_strings() -> void {
    if not gpu.is_available() { return }
    let left = Column.from_strs(["pear", "apple", "fig", "fig"])
    defer { left.free() }
    let right = Column.from_strs(["fig", "fig", "fig", "apple"])
    defer { right.free() }
    let below = less(left, right)
    defer { below.free() }
    let at_most = less_equal(left, right)
    defer { at_most.free() }
    let a = left.to_strs()
    let b = right.to_strs()
    for row in 0..a.len() {
        assert_eq(below.to_bools()[row], (a[row] ?? "") < (b[row] ?? ""))
        assert_eq(at_most.to_bools()[row], (a[row] ?? "") <= (b[row] ?? ""))
    }
    #: A scalar outside either dictionary still has a place in the order.
    let before_m = less(left, "m")
    defer { before_m.free() }
    for (row, text) in a.enumerate() {
        assert_eq(before_m.to_bools()[row], (text ?? "") < "m")
    }
}

fn test_series_align_on_the_union_of_their_labels() -> void {
    if not gpu.is_available() { return }
    let left = Series(
        Column.from_floats([1.0, 2.0]),
        index: Index.of([Column.from_strs(["x", "y"])]),
        name: "reading"
    )
    defer { left.free() }
    let right = Series(
        Column.from_floats([10.0, 20.0]),
        index: Index.of([Column.from_strs(["y", "z"])]),
        name: "reading"
    )
    defer { right.free() }
    let total = add(left, right)
    defer { total.free() }
    assert_eq(total.len(), 3)
    assert_eq(total.name, "reading")
    assert_eq(total.index.label_at(0)[0].to_str(), "x")
    assert_eq(total.index.label_at(1)[0].to_str(), "y")
    assert_eq(total.index.label_at(2)[0].to_str(), "z")
    assert_eq(total.values.null_mask(), [true, false, true])
    assert_near(total.values.to_floats()[1], 12.0, tolerance: 1e-6)
    let filled = add(left, right, fill_value: Value.of_float(0.0))
    defer { filled.free() }
    assert_eq(filled.values.null_mask(), [false, false, false])
    assert_eq(filled.values.to_floats(), [1.0, 12.0, 20.0])
}

fn test_two_default_indexes_align_without_a_gather() -> void {
    if not gpu.is_available() { return }
    let left = Series(Column.from_floats([1.0, 2.0, 3.0]), name: "a")
    defer { left.free() }
    let right = Series(Column.from_floats([4.0, 5.0, 6.0]), name: "b")
    defer { right.free() }
    let total = add(left, right)
    defer { total.free() }
    assert_eq(total.values.to_floats(), [5.0, 7.0, 9.0])
    assert_true(total.index.is_range(), "two range indexes stay a range index")
    assert_eq(total.name, null, "names that disagree are dropped, as pandas drops them")
}

fn test_frames_align_on_rows_and_columns() -> void {
    if not gpu.is_available() { return }
    let left = DataFrame(
        ["a", "b"],
        [Column.from_floats([1.0, 2.0]), Column.from_floats([3.0, 4.0])],
        index: Index.of([Column.from_strs(["r0", "r1"])])
    )
    defer { left.free() }
    let right = DataFrame(
        ["b", "c"],
        [Column.from_floats([10.0, 20.0]), Column.from_floats([30.0, 40.0])],
        index: Index.of([Column.from_strs(["r1", "r2"])])
    )
    defer { right.free() }
    let total = add(left, right, fill_value: Value.of_float(0.0))
    defer { total.free() }
    assert_eq(total.names, ["a", "b", "c"])
    assert_eq(total.len(), 3)
    #: A row one frame lacks meets the fill; a row neither frame has a value
    #: for stays missing, which is what pandas' fill_value does and does not do.
    assert_eq(total.column("a").null_mask(), [false, false, true])
    assert_near(total.column("a").to_floats()[1], 2.0, tolerance: 1e-6)
    assert_eq(total.column("b").to_floats(), [3.0, 14.0, 20.0])
    assert_eq(total.column("c").null_mask(), [true, false, false])
    assert_near(total.column("c").to_floats()[2], 40.0, tolerance: 1e-6)
}

fn test_a_frame_against_a_series_matches_column_names() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(
        ["a", "b"],
        [Column.from_floats([1.0, 2.0]), Column.from_floats([3.0, 4.0])]
    )
    defer { frame.free() }
    let offsets = Series(
        Column.from_floats([10.0, 100.0]),
        index: Index.of([Column.from_strs(["a", "b"])])
    )
    defer { offsets.free() }
    let shifted = add(frame, offsets)
    defer { shifted.free() }
    assert_eq(shifted.column("a").to_floats(), [11.0, 12.0])
    assert_eq(shifted.column("b").to_floats(), [103.0, 104.0])
}

fn test_a_frame_takes_a_scalar_column_by_column() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(
        ["a", "b"],
        [Column.from_floats([1.0, 2.0]), Column.from_floats([3.0, 4.0])]
    )
    defer { frame.free() }
    let doubled = multiply(frame, 2.0)
    defer { doubled.free() }
    assert_eq(doubled.column("a").to_floats(), [2.0, 4.0])
    assert_eq(doubled.column("b").to_floats(), [6.0, 8.0])
    let held = clip(frame, 2.0, 3.0)
    defer { held.free() }
    assert_eq(held.column("a").to_floats(), [2.0, 2.0])
    assert_eq(held.column("b").to_floats(), [3.0, 3.0])
}

fn test_arithmetic_refuses_what_it_cannot_mean() -> void {
    if not gpu.is_available() { return }
    let cities = Column.from_strs(["austin"])
    defer { cities.free() }
    let stamps = Column.from_ints([1000], dtype: Dtype.Datetime)
    defer { stamps.free() }
    let numbers = Column.from_floats([1.0])
    defer { numbers.free() }
    let pair = Column.from_floats([1.0, 2.0])
    defer { pair.free() }
    assert_throws(TypeError, || add(cities, 1))
    assert_throws(TypeError, || less(cities, 1))
    assert_throws(TypeError, || subtract(stamps, stamps))
    assert_throws(TypeError, || not_(numbers))
    assert_throws(TypeError, || log(cities))
    assert_throws(ValueError, || add(numbers, pair))
}

fn test_a_timestamp_compares_against_its_milliseconds() -> void {
    if not gpu.is_available() { return }
    let stamps = Column.from_ints(
        [1699999999999, 1700000000000, 1700000000001],
        dtype: Dtype.Datetime
    )
    defer { stamps.free() }
    let after = greater_equal(stamps, 1700000000000)
    defer { after.free() }
    assert_eq(after.to_bools(), [false, true, true])
}

fn test_map_runs_a_callback_over_every_value() -> void {
    if not gpu.is_available() { return }
    let series = Series(Column.from_floats([1.0, NAN, 3.0]), name: "v")
    defer { series.free() }
    let doubled = map(series, |cell| if cell.is_na() {
        cell
    } else {
        Value.of_float(cell.to_float() * 2.0)
    })
    defer { doubled.free() }
    assert_eq(doubled.values.null_mask(), [false, true, false])
    assert_near(doubled.values.to_floats()[2], 6.0, tolerance: 1e-6)
    assert_eq(doubled.name, "v")
    let labelled = map(series, |cell| Value.of_str(if cell.is_na() { "none" } else { "some" }))
    defer { labelled.free() }
    assert_eq(labelled.values.to_strs(), ["some", "none", "some"])
}

#: The callback's answers are checked against the same totals worked out from
#: the downloaded columns, so the test does not trust the callback plumbing to
#: have visited the right cells.
fn test_apply_folds_a_column_or_a_row() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(
        ["a", "b"],
        [Column.from_floats([1.0, 2.0, 3.0]), Column.from_floats([10.0, 20.0, 30.0])]
    )
    defer { frame.free() }
    let by_column = apply(frame, |cells| Value.of_float(total_of(cells)))
    defer { by_column.free() }
    assert_eq(by_column.len(), 2)
    assert_eq(by_column.index.label_at(0)[0].to_str(), "a")
    assert_near(by_column.values.to_floats()[0], 6.0, tolerance: 1e-6)
    assert_near(by_column.values.to_floats()[1], 60.0, tolerance: 1e-6)
    let by_row = apply(frame, |cells| Value.of_float(total_of(cells)), axis: 1)
    defer { by_row.free() }
    assert_eq(by_row.len(), 3)
    let a = frame.column("a").to_floats()
    let b = frame.column("b").to_floats()
    for row in 0..3 {
        assert_near(by_row.values.to_floats()[row], a[row] + b[row], tolerance: 1e-6)
    }
    assert_throws(ValueError, || apply(frame, |cells| Value.of_float(total_of(cells)), axis: 2))
}

fn total_of(cells: list[Value]) -> float {
    var sum = 0.0
    for cell in cells {
        if cell.is_na() { continue }
        sum += cell.to_float()
    }
    return sum
}

fn test_applymap_keeps_the_shape() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(
        ["a", "b"],
        [Column.from_floats([1.0, -2.0]), Column.from_floats([-3.0, 4.0])]
    )
    defer { frame.free() }
    let sized = applymap(frame, |cell| if cell.is_na() {
        cell
    } else {
        Value.of_float(if cell.to_float() < 0.0 {
            -cell.to_float()
        } else {
            cell.to_float()
        })
    })
    defer { sized.free() }
    assert_eq(sized.shape(), (2, 2))
    assert_eq(sized.column("a").to_floats(), [1.0, 2.0])
    assert_eq(sized.column("b").to_floats(), [3.0, 4.0])
}

fn test_pipe_hands_the_whole_object_over() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(["a"], [Column.from_floats([-1.0, 5.0])])
    defer { frame.free() }
    let held = pipe(frame, |target| clip(target, 0.0, 2.0))
    defer { held.free() }
    assert_eq(held.column("a").to_floats(), [0.0, 2.0])
}

fn test_combine_aligns_before_it_pairs() -> void {
    if not gpu.is_available() { return }
    let left = Series(
        Column.from_floats([1.0, 5.0]),
        index: Index.of([Column.from_strs(["x", "y"])])
    )
    defer { left.free() }
    let right = Series(
        Column.from_floats([4.0, 2.0]),
        index: Index.of([Column.from_strs(["y", "z"])])
    )
    defer { right.free() }
    let larger = combine(left, right, |a, b| pick_larger(a, b))
    defer { larger.free() }
    assert_eq(larger.len(), 3)
    assert_eq(larger.index.label_at(2)[0].to_str(), "z")
    assert_eq(larger.values.to_floats(), [1.0, 5.0, 2.0])
}

fn pick_larger(a: Value, b: Value) -> Value {
    if a.is_na() { return b }
    if b.is_na() { return a }
    return if a.to_float() >= b.to_float() { a } else { b }
}

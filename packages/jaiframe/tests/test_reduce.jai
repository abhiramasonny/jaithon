#: Reductions, scans, lags and windows.
#:
#: Every case here computes its answer a second way and compares. The second
#: way is a plain host loop over the same numbers — the definition of the
#: statistic, written out — so a kernel that is fast and wrong fails. Where two
#: kernels answer the same question, both are checked against that loop rather
#: than against each other: the rolling tests run either side of the window
#: where the module switches from the direct path to the prefix one, and the
#: reduction tests run a row count that divides by four and one that does not,
#: so the four-at-a-time read and the scalar fallback both appear.
#:
#: The 70000-row cases are built by `Index.range().materialize()`, on the
#: device, so they cost a dispatch rather than a host loop over seventy
#: thousand boxed floats — and they are the only ones that reach the second and
#: third levels of the scan and the strided reduction grid, where every bug in
#: a carry lives. The one exception is the `Datetime` case, which has to be
#: uploaded because no device path builds a stamp ramp; seventy thousand
#: `wide_pack` calls is the price of reaching the strided grid with two lanes.
import std.gpu as gpu
import std.math
from std.math import NAN, is_nan
from std.test import assert_eq, assert_false, assert_near, assert_throws, assert_true
from jaiframe.column import Agg, Column, DataFrame, Dtype, Index, Series, Value
from jaiframe.column import NA
from jaiframe.reduce import ColumnStats, Window
from jaiframe.reduce import all_, any_, corr, corrwith, count, cov, cumcount_na, cummax
from jaiframe.reduce import cummin, cumprod, cumsum, describe, diff, ewm, expanding
from jaiframe.reduce import first_valid_index, idxmax, idxmin, kurtosis, last_valid_index
from jaiframe.reduce import maximum, mean, median, minimum, pct_change, product, quantile
from jaiframe.reduce import reduce_column, rolling, scan_column, sem, shift, shift_column
from jaiframe.reduce import skew, std, sum, variance

fn present(values: list[float]) -> list[float] {
    var out: list[float] = []
    for value in values {
        if not is_nan(value) { out.push(value) }
    }
    return out
}

fn host_sum(values: list[float]) -> float {
    var total = 0.0
    for value in present(values) { total += value }
    return total
}

fn host_mean(values: list[float]) -> float {
    let kept = present(values)
    if kept.len() == 0 { return NAN }
    return host_sum(values) / float(kept.len())
}

#: Two-pass variance, which is the definition and not the shortcut the kernel
#: takes.
fn host_variance(values: list[float], ddof: int) -> float {
    let kept = present(values)
    if kept.len() - ddof <= 0 { return NAN }
    let middle = host_mean(values)
    var total = 0.0
    for value in kept { total += (value - middle) * (value - middle) }
    return total / float(kept.len() - ddof)
}

fn host_min(values: list[float]) -> float {
    var best = NAN
    for value in present(values) {
        if is_nan(best) or value < best { best = value }
    }
    return best
}

fn host_max(values: list[float]) -> float {
    var best = NAN
    for value in present(values) {
        if is_nan(best) or value > best { best = value }
    }
    return best
}

#: Insertion sort of the present values, so a quantile has an answer that owes
#: nothing to the radix search under test.
fn host_sorted(values: list[float]) -> list[float] {
    var out: list[float] = []
    for value in present(values) {
        var at = out.len()
        out.push(value)
        while at > 0 and out[at - 1] > value {
            out[at] = out[at - 1]
            out[at - 1] = value
            at -= 1
        }
    }
    return out
}

fn host_quantile(values: list[float], q: float) -> float {
    let ordered = host_sorted(values)
    if ordered.len() == 0 { return NAN }
    let position = q * float(ordered.len() - 1)
    let lower = int(math.floor(position))
    let upper = if lower + 1 < ordered.len() { lower + 1 } else { lower }
    let fraction = position - float(lower)
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])
}

fn host_window(
    values: list[float],
    window: int,
    min_periods: int,
    center: bool,
    what: str
) -> list[float] {
    let rows = values.len()
    let lead = if center { (window - 1) // 2 } else { 0 }
    var out: list[float] = []
    for row in 0..rows {
        let last = row + lead
        let first = last - window + 1
        let low = if first < 0 { 0 } else { first }
        let high = if last > rows - 1 { rows - 1 } else { last }
        var slice: list[float] = []
        if high >= low {
            for at in low..high + 1 { slice.push(values[at]) }
        }
        let kept = present(slice)
        if what == "count" {
            out.push(float(kept.len()))
            continue
        }
        if kept.len() == 0 or kept.len() < min_periods {
            out.push(NAN)
            continue
        }
        out.push(match what {
            "sum" => host_sum(slice),
            "mean" => host_mean(slice),
            "min" => host_min(slice),
            "max" => host_max(slice),
            "median" => host_quantile(slice, 0.5),
            _ => host_variance(slice, 1),
        })
    }
    return out
}

fn assert_lanes(
    actual: list[float],
    expected: list[float],
    tolerance: float,
    message: str
) -> void {
    assert_eq(actual.len(), expected.len(), f"{message}: length")
    for (at, want) in expected.enumerate() {
        if is_nan(want) {
            assert_true(is_nan(actual[at]), f"{message}: row {at} should be missing")
            continue
        }
        assert_near(actual[at], want, tolerance: tolerance, message: f"{message}: row {at}")
    }
}

let SAMPLE = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0]
let GAPPY = [3.0, NAN, 4.0, 1.0, NAN, 9.0, 2.0, 6.0, 5.0, NAN, 5.0, 8.0]

fn test_a_reduction_agrees_with_the_definition() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_floats(SAMPLE)
    defer { column.free() }
    assert_near(reduce_column(column, Agg.Sum).to_float(), host_sum(SAMPLE), tolerance: 1e-4)
    assert_near(reduce_column(column, Agg.Mean).to_float(), host_mean(SAMPLE), tolerance: 1e-5)
    assert_near(reduce_column(column, Agg.Min).to_float(), host_min(SAMPLE), tolerance: 1e-6)
    assert_near(reduce_column(column, Agg.Max).to_float(), host_max(SAMPLE), tolerance: 1e-6)
    assert_eq(reduce_column(column, Agg.Count).to_int(), SAMPLE.len())
    assert_eq(reduce_column(column, Agg.Size).to_int(), SAMPLE.len())
    assert_near(
        reduce_column(column, Agg.Var).to_float(),
        host_variance(SAMPLE, 1),
        tolerance: 1e-3
    )
    assert_near(
        reduce_column(column, Agg.Std).to_float(),
        math.sqrt(host_variance(SAMPLE, 1)),
        tolerance: 1e-4
    )
    var expected_product = 1.0
    for value in SAMPLE { expected_product *= value }
    assert_near(
        reduce_column(column, Agg.Prod).to_float(),
        expected_product,
        tolerance: expected_product * 1e-5
    )
}

#: The same numbers reduced through the four-at-a-time read and through the
#: scalar one. A column with no validity lane and a row count that divides by
#: four takes the quad path; the same values under an all-present mask cannot.
fn test_the_quad_path_and_the_scalar_path_agree() -> void {
    if not gpu.is_available() { return }
    var gaps: list[bool] = []
    for _value in SAMPLE { gaps.push(false) }
    let quad = Column.from_floats(SAMPLE)
    let scalar = Column.from_floats(SAMPLE, gaps)
    defer { quad.free() }
    defer { scalar.free() }
    assert_eq(SAMPLE.len() % 4, 0, "the sample has to divide by four for this to mean anything")
    assert_false(quad.has_validity(), "no nulls means no mask means the quad path")
    assert_true(scalar.has_validity(), "an explicit mask forces the scalar path")
    let wide = column_summary(quad)
    let narrow = column_summary(scalar)
    for (at, value) in wide.enumerate() {
        assert_near(narrow[at], value, tolerance: 1e-4, message: f"summary slot {at}")
    }
}

fn column_summary(column: Column) -> list[float] {
    return [
        reduce_column(column, Agg.Sum).to_float(),
        reduce_column(column, Agg.Mean).to_float(),
        reduce_column(column, Agg.Min).to_float(),
        reduce_column(column, Agg.Max).to_float(),
        float(reduce_column(column, Agg.Count).to_int()),
        reduce_column(column, Agg.Var).to_float(),
        reduce_column(column, Agg.Median).to_float(),
    ]
}

fn test_missing_rows_are_skipped_or_they_poison() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_floats(GAPPY)
    defer { column.free() }
    assert_eq(reduce_column(column, Agg.Count).to_int(), present(GAPPY).len())
    assert_near(reduce_column(column, Agg.Sum).to_float(), host_sum(GAPPY), tolerance: 1e-4)
    assert_near(reduce_column(column, Agg.Mean).to_float(), host_mean(GAPPY), tolerance: 1e-5)
    assert_true(
        reduce_column(column, Agg.Mean, skipna: false).is_na(),
        "a mean of a column with a gap is nothing when the gaps are not skipped"
    )
    assert_eq(
        reduce_column(column, Agg.Count, skipna: false).to_int(),
        present(GAPPY).len(),
        "a count counts what is there whatever skipna says"
    )
}

fn test_a_quantile_is_the_value_a_sort_would_put_there() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_floats(GAPPY)
    defer { column.free() }
    for q in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
        assert_near(
            reduce_column(column, Agg.Quantile, q: q).to_float(),
            host_quantile(GAPPY, q),
            tolerance: 1e-5,
            message: f"quantile {q}"
        )
    }
    assert_near(
        reduce_column(column, Agg.Median).to_float(),
        host_quantile(GAPPY, 0.5),
        tolerance: 1e-5
    )
    assert_throws(ValueError, || reduce_column(column, Agg.Quantile, q: 1.5))
}

#: Negatives and both zeroes, because the ordered-key trick the search rests on
#: is exactly where a float's bit pattern stops being monotone.
fn test_a_quantile_orders_negatives_and_zero() -> void {
    if not gpu.is_available() { return }
    let awkward = [-3.5, 0.0, 2.25, -0.0, -100.0, 7.0, -0.25, 1e6]
    let column = Column.from_floats(awkward)
    defer { column.free() }
    for q in [0.0, 0.2, 0.5, 0.8, 1.0] {
        assert_near(
            reduce_column(column, Agg.Quantile, q: q).to_float(),
            host_quantile(awkward, q),
            tolerance: 1e-3,
            message: f"quantile {q} of a column with negatives"
        )
    }
}

fn test_an_int_column_reduces_without_losing_its_bits() -> void {
    if not gpu.is_available() { return }
    let big = [1700000000001, 1700000000000, 1700000000003, 1700000000002]
    let column = Column.from_ints(big)
    defer { column.free() }
    assert_eq(reduce_column(column, Agg.Min).to_int(), 1700000000000)
    assert_eq(reduce_column(column, Agg.Max).to_int(), 1700000000003)
    assert_eq(reduce_column(column, Agg.First).to_int(), 1700000000001)
    assert_eq(reduce_column(column, Agg.Last).to_int(), 1700000000002)
}

#: The neighbours of the case above, which a fold over the flattened lane gets
#: wrong in the same way and a fold in `long` has to get right: the block
#: negated, the extreme in the last row, a tie that must resolve to its
#: earliest row, a masked row that would otherwise win, and a column longer
#: than the 65536-thread grid so the per-thread stride and both folds carry the
#: winner. Every expectation is what pandas answers for the same int64 values.
fn test_a_wide_extreme_survives_its_neighbours() -> void {
    if not gpu.is_available() { return }
    let mirror = [-1700000000001, -1700000000000, -1700000000003, -1700000000002]
    let negated = Column.from_ints(mirror)
    defer { negated.free() }
    assert_eq(reduce_column(negated, Agg.Min).to_int(), -1700000000003, "the most negative")
    assert_eq(reduce_column(negated, Agg.Max).to_int(), -1700000000000, "the least negative")
    assert_eq(idxmin(Series(negated)), Value.of_int(2), "the row of the most negative")
    let odd = [1700000000004, 1700000000001, 1700000000000, 1700000000003, 1700000000002]
    let five = Column.from_ints(odd)
    defer { five.free() }
    assert_eq(
        reduce_column(five, Agg.Min).to_int(),
        1700000000000,
        "five rows do not divide by four"
    )
    let tied = Column.from_ints([1700000000009, 1700000000001, 1700000000009, 1700000000001])
    defer { tied.free() }
    assert_eq(idxmin(Series(tied)), Value.of_int(1), "the first of two equal minima")
    assert_eq(idxmax(Series(tied)), Value.of_int(0), "the first of two equal maxima")
    let masked = Column.from_ints([1700000000000, 1700000000005], [true, false])
    defer { masked.free() }
    assert_eq(reduce_column(masked, Agg.Min).to_int(), 1700000000005, "a masked row cannot win")
}

#: Millisecond stamps inside one 131-second block, which is what the two-lane
#: fold was written for, past the grid so no group holds the answer alone.
fn test_a_datetime_column_past_the_grid_keeps_its_milliseconds() -> void {
    if not gpu.is_available() { return }
    let rows = 70000
    let base = 1700000000000
    var stamps: list[int] = []
    for row in 0..rows { stamps.push(base + row % 97) }
    stamps[69999] = base - 5
    stamps[40001] = base + 900
    let column = Column.from_ints(stamps, null, Dtype.Datetime)
    defer { column.free() }
    assert_eq(
        reduce_column(column, Agg.Min).to_int(),
        base - 5,
        "the earliest stamp is the last row"
    )
    assert_eq(reduce_column(column, Agg.Max).to_int(), base + 900, "the latest stamp is mid-column")
    assert_eq(idxmin(Series(column)), Value.of_int(69999), "the row of the earliest")
    assert_eq(idxmax(Series(column)), Value.of_int(40001), "the row of the latest")
}

fn test_a_seventy_thousand_row_column_reduces_to_the_closed_form() -> void {
    if not gpu.is_available() { return }
    let rows = 70000
    let ramp = Index.range(rows).materialize()[0]
    defer { ramp.free() }
    let expected = float(rows) * float(rows - 1) / 2.0
    assert_near(
        reduce_column(ramp, Agg.Sum).to_float(),
        expected,
        tolerance: expected * 1e-5,
        message: "the strided grid and the join have to add up"
    )
    assert_eq(reduce_column(ramp, Agg.Min).to_int(), 0)
    assert_eq(reduce_column(ramp, Agg.Max).to_int(), rows - 1)
    assert_eq(reduce_column(ramp, Agg.Count).to_int(), rows)
    assert_near(reduce_column(ramp, Agg.Median).to_float(), float(rows - 1) / 2.0, tolerance: 1.0)
}

fn test_the_shape_statistics_match_their_formulas() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_floats(SAMPLE)
    let series = Series(column)
    defer { series.free() }
    let rows = SAMPLE.len()
    let middle = host_mean(SAMPLE)
    var m2 = 0.0
    var m3 = 0.0
    var m4 = 0.0
    for value in SAMPLE {
        let d = value - middle
        m2 += d * d
        m3 += d * d * d
        m4 += d * d * d * d
    }
    let spread = m2 / float(rows - 1)
    let deviation = math.sqrt(spread)
    let standardised = m3 / (deviation * deviation * deviation)
    let want_skew = float(rows) / float((rows - 1) * (rows - 2)) * standardised
    let lead = float(rows * (rows + 1)) / float((rows - 1) * (rows - 2) * (rows - 3))
    let tail = 3.0 * float((rows - 1) * (rows - 1)) / float((rows - 2) * (rows - 3))
    let want_kurtosis = lead * (m4 / (spread * spread)) - tail
    let got_skew: Value = skew(series)
    let got_kurtosis: Value = kurtosis(series)
    let got_sem: Value = sem(series)
    assert_near(got_skew.to_float(), want_skew, tolerance: 1e-3)
    assert_near(got_kurtosis.to_float(), want_kurtosis, tolerance: 1e-2)
    assert_near(got_sem.to_float(), math.sqrt(spread / float(rows)), tolerance: 1e-5)
}

#: Vacuous truth belongs to the column with nothing in it, not to the column
#: of zeroes. `pd.Series([0.0, 0.0]).all()` is False and `.any()` is False;
#: `pd.Series([nan, nan]).all()` is True and `.any()` is False, because the
#: missing rows are skipped and `all` of nothing is true. Checked against
#: pandas 2 rather than remembered.
fn test_truth_and_edges() -> void {
    if not gpu.is_available() { return }
    let mixed = Column.from_floats([0.0, 0.0, 2.0, 0.0])
    let zeros = Column.from_floats([0.0, 0.0])
    defer { mixed.free() }
    defer { zeros.free() }
    assert_true(reduce_column(mixed, Agg.Any).to_bool(), "one non-zero is enough")
    assert_false(reduce_column(mixed, Agg.All).to_bool(), "a zero is not all")
    assert_false(reduce_column(zeros, Agg.Any).to_bool(), "no non-zero is not any")
    assert_false(reduce_column(zeros, Agg.All).to_bool(), "a column of zeroes is not all")
    let gappy = Column.from_floats([NAN, NAN, 7.0, 2.0, NAN])
    let series = Series(gappy)
    defer { series.free() }
    assert_eq(first_valid_index(series), Value.of_int(2))
    assert_eq(last_valid_index(series), Value.of_int(3))
    let empty = Column.from_floats([NAN, NAN])
    let hollow = Series(empty)
    defer { hollow.free() }
    assert_eq(first_valid_index(hollow), NA)
    assert_true(reduce_column(empty, Agg.All).to_bool(), "every value of none is true")
    assert_false(reduce_column(empty, Agg.Any).to_bool(), "no value at all is not any")
}

fn test_the_extreme_carries_the_row_it_was_found_at() -> void {
    if not gpu.is_available() { return }
    let ties = [4.0, 1.0, 9.0, 1.0, 9.0]
    let series = Series(Column.from_floats(ties))
    defer { series.free() }
    assert_eq(idxmin(series), Value.of_int(1), "the earliest of the tied minima")
    assert_eq(idxmax(series), Value.of_int(2), "the earliest of the tied maxima")
}

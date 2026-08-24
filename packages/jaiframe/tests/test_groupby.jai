#: Group-by: the codes lane, the two reduction paths, and everything read
#: through them.
#:
#: Every aggregation here is checked against a second, independent answer
#: folded on the host from the same rows — not against a number this code
#: produced earlier. That is the only way to catch the failure these kernels
#: actually have, which is a group boundary being off by one: a wrong `starts`
#: entry gives a plausible-looking total for every group and a correct one for
#: none.
#:
#: The two paths matter as much as the answers. `use_scatter` picks between the
#: atomic scatter and the segmented walk on the shape of the data, so several
#: tests run the same aggregation twice — once on a fresh handle, and once on a
#: handle that has already been made to lay its rows out — and insist the two
#: agree. A tier that only one of them exercises is a tier nobody tests.
#:
#: The larger case is 8192 rows over 137 groups. That is past one threadgroup
#: of the scan behind `starts`, past one tile of the segmented walk, and it
#: divides by neither 4 nor 256, so the quad path's tail and the padded grid
#: are both live.
import std.gpu as gpu
from std.math import NAN, is_nan, sqrt
from std.test import assert_eq, assert_near, assert_throws, assert_true
from jaiframe.column import Agg, Column, DataFrame, Dtype, Index, Series, Value
from jaiframe.groupby import GroupBy, group_by

#: A key per row, deterministic and spread over `groups` distinct values.
fn keys_of(rows: int, groups: int) -> list[int] {
    var out: list[int] = []
    for row in 0..rows { out.push((row * 31 + 7) % groups) }
    return out
}

#: A value per row, deterministic, both signs, and never a round number.
fn values_of(rows: int) -> list[float] {
    var out: list[float] = []
    for row in 0..rows { out.push(float(row * 7919 % 1997) / 8.0 - 124.0) }
    return out
}

#: The rows of each group, in row order, for the group keys the handle chose.
#:
#: Reading the keys back rather than assuming the group order is what makes
#: these tests independent of whether `factorize` sorted or not.
fn folded(handle: GroupBy, keys: list[int], values: list[float]) -> list[list[float]] {
    var out: list[list[float]] = []
    for wanted in handle.uniques[0].to_ints() {
        var mine: list[float] = []
        for (row, key) in keys.enumerate() {
            if key == wanted and not is_nan(values[row]) { mine.push(values[row]) }
        }
        out.push(mine)
    }
    return out
}

#: The label `groups` and `get_group` give each group, in group order.
fn render(handle: GroupBy) -> list[str] {
    var out: list[str] = []
    for label in handle.groups() { out.push(label) }
    return out
}

fn total_of(items: list[float]) -> float {
    var sum = 0.0
    for item in items { sum += item }
    return sum
}

fn sorted_copy(items: list[float]) -> list[float] {
    var out = [item for item in items]
    for at in 1..out.len() {
        let held = out[at]
        var back = at - 1
        while back >= 0 and out[back] > held {
            out[back + 1] = out[back]
            back -= 1
        }
        out[back + 1] = held
    }
    return out
}

#: The linear-interpolation quantile, spelled out from its definition rather
#: than borrowed from the code under test.
fn quantile_of(items: list[float], fraction: float) -> float {
    let ordered = sorted_copy(items)
    let position = fraction * float(ordered.len() - 1)
    let below = int(position)
    let above = if below + 1 < ordered.len() { below + 1 } else { below }
    let weight = position - float(below)
    return ordered[below] + weight * (ordered[above] - ordered[below])
}

#: The distance between a group's first and last value, for `apply` to carry.
fn span_of(piece: DataFrame) -> Value {
    let values = piece.column("value")
    return Value.of_float(values.get(piece.len() - 1).to_float() - values.get(0).to_float())
}

fn frame_of(keys: list[int], values: list[float], nulls: list[bool]? = null) -> DataFrame {
    return DataFrame(
        ["key", "value"],
        [Column.from_ints(keys), Column.from_floats(values, nulls: nulls)]
    )
}

fn test_a_sum_matches_a_host_fold() -> void {
    if not gpu.is_available() { return }
    let keys = keys_of(64, 7)
    let values = values_of(64)
    let frame = frame_of(keys, values)
    defer { frame.free() }
    let handle = group_by(frame, "key")
    defer { handle.free() }
    assert_eq(handle.ngroups(), 7)
    let totals = handle.sum()
    defer { totals.free() }
    let expected = folded(handle, keys, values)
    let produced = totals.column("value").to_floats()
    for (group, mine) in expected.enumerate() {
        assert_near(produced[group], total_of(mine), tolerance: 1e-3)
    }
}

fn test_the_two_paths_reach_the_same_answer() -> void {
    if not gpu.is_available() { return }
    let keys = keys_of(8192, 137)
    let values = values_of(8192)
    let frame = frame_of(keys, values)
    defer { frame.free() }
    let scattered = group_by(frame, "key")
    defer { scattered.free() }
    let by_scatter = scattered.sum()
    defer { by_scatter.free() }

    let walked = group_by(frame, "key")
    defer { walked.free() }
    #: `min` has no atomic, so asking for it first lays the rows out — and
    #: every later reduction on that handle takes the segmented path.
    let extremes = walked.min()
    defer { extremes.free() }
    let by_segment = walked.sum()
    defer { by_segment.free() }

    let expected = folded(scattered, keys, values)
    let one = by_scatter.column("value").to_floats()
    let two = by_segment.column("value").to_floats()
    let smallest = extremes.column("value").to_floats()
    for (group, mine) in expected.enumerate() {
        assert_near(one[group], total_of(mine), tolerance: 1e-2)
        assert_near(two[group], one[group], tolerance: 1e-2)
        var low = mine[0]
        for item in mine { if item < low { low = item } }
        assert_near(smallest[group], low, tolerance: 1e-6)
    }
}

fn test_mean_variance_and_deviation_come_from_their_definitions() -> void {
    if not gpu.is_available() { return }
    let keys = keys_of(300, 11)
    let values = values_of(300)
    let frame = frame_of(keys, values)
    defer { frame.free() }
    let handle = group_by(frame, "key")
    defer { handle.free() }
    let averages = handle.mean()
    defer { averages.free() }
    let spread = handle.variance()
    defer { spread.free() }
    let deviation = handle.std()
    defer { deviation.free() }
    let expected = folded(handle, keys, values)
    let means = averages.column("value").to_floats()
    let variances = spread.column("value").to_floats()
    let deviations = deviation.column("value").to_floats()
    for (group, mine) in expected.enumerate() {
        let average = total_of(mine) / float(mine.len())
        assert_near(means[group], average, tolerance: 1e-3)
        var squares = 0.0
        for item in mine { squares += (item - average) * (item - average) }
        let variance = squares / float(mine.len() - 1)
        assert_near(variances[group], variance, tolerance: variance * 1e-3 + 1e-3)
        assert_near(deviations[group], sqrt(variance), tolerance: 1e-2)
    }
}

fn test_the_row_pickers_pick_the_right_rows() -> void {
    if not gpu.is_available() { return }
    let keys = keys_of(500, 13)
    let values = values_of(500)
    let frame = frame_of(keys, values)
    defer { frame.free() }
    let handle = group_by(frame, "key")
    defer { handle.free() }
    let smallest = handle.min()
    defer { smallest.free() }
    let largest = handle.max()
    defer { largest.free() }
    let earliest = handle.first()
    defer { earliest.free() }
    let latest = handle.last()
    defer { latest.free() }
    let expected = folded(handle, keys, values)
    let lows = smallest.column("value").to_floats()
    let highs = largest.column("value").to_floats()
    let heads = earliest.column("value").to_floats()
    let tails = latest.column("value").to_floats()
    for (group, mine) in expected.enumerate() {
        var low = mine[0]
        var high = mine[0]
        for item in mine {
            if item < low { low = item }
            if item > high { high = item }
        }
        assert_near(lows[group], low, tolerance: 1e-6)
        assert_near(highs[group], high, tolerance: 1e-6)
        assert_near(heads[group], mine[0], tolerance: 1e-6)
        assert_near(tails[group], mine[mine.len() - 1], tolerance: 1e-6)
    }
}

fn test_a_median_agrees_with_a_sorted_group() -> void {
    if not gpu.is_available() { return }
    #: 251 rows over 6 groups leaves groups of both parities, which is the
    #: whole difference between a median and an order statistic.
    let keys = keys_of(251, 6)
    let values = values_of(251)
    let frame = frame_of(keys, values)
    defer { frame.free() }
    let handle = group_by(frame, "key")
    defer { handle.free() }
    let middles = handle.median()
    defer { middles.free() }
    let quarter = handle.quantile(0.25)
    defer { quarter.free() }
    let expected = folded(handle, keys, values)
    let produced = middles.column("value").to_floats()
    let quarters = quarter.column("value").to_floats()
    for (group, mine) in expected.enumerate() {
        assert_near(produced[group], quantile_of(mine, 0.5), tolerance: 1e-4)
        assert_near(quarters[group], quantile_of(mine, 0.25), tolerance: 1e-4)
    }
}

fn test_count_and_size_part_company_over_nulls() -> void {
    if not gpu.is_available() { return }
    let keys = [0, 0, 1, 1, 1, 2]
    let values = [1.0, NAN, 3.0, NAN, NAN, 6.0]
    let frame = frame_of(keys, values)
    defer { frame.free() }
    let handle = group_by(frame, "key")
    defer { handle.free() }
    let sizes = handle.size()
    defer { sizes.free() }
    let counts = handle.count()
    defer { counts.free() }
    assert_eq(sizes.values.to_ints(), [2, 3, 1])
    assert_eq(counts.column("value").to_ints(), [1, 1, 1])
    let averages = handle.mean()
    defer { averages.free() }
    assert_near(averages.column("value").get(1).to_float(), 3.0, tolerance: 1e-6)
}

fn test_an_all_null_group_has_no_mean_but_a_sum_of_zero() -> void {
    if not gpu.is_available() { return }
    let frame = frame_of([0, 0, 1], [NAN, NAN, 4.0])
    defer { frame.free() }
    let handle = group_by(frame, "key")
    defer { handle.free() }
    let averages = handle.mean()
    defer { averages.free() }
    let totals = handle.sum()
    defer { totals.free() }
    assert_true(averages.column("value").get(0).is_na(), "no values, no mean")
    assert_near(totals.column("value").get(0).to_float(), 0.0, tolerance: 1e-9)
    assert_true(handle.variance().column("value").get(0).is_na(), "no values, no variance")
}

fn test_nunique_counts_the_distinct_values() -> void {
    if not gpu.is_available() { return }
    let keys = [0, 0, 0, 1, 1, 1, 1]
    let values = [1.0, 1.0, 2.0, 5.0, 5.0, 5.0, NAN]
    let frame = frame_of(keys, values)
    defer { frame.free() }
    let handle = group_by(frame, "key")
    defer { handle.free() }
    let distinct = handle.nunique()
    defer { distinct.free() }
    assert_eq(distinct.column("value").to_ints(), [2, 1])
}

fn test_any_and_all_read_the_truth_of_a_group() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(
        ["key", "flag"],
        [
            Column.from_ints([0, 0, 1, 1, 2, 2]),
            Column.from_bools([true, false, true, true, false, false]),
        ]
    )
    defer { frame.free() }
    let handle = group_by(frame, "key")
    defer { handle.free() }
    let some = handle.any()
    defer { some.free() }
    let every = handle.all()
    defer { every.free() }
    assert_eq(some.column("flag").to_bools(), [true, true, false])
    assert_eq(every.column("flag").to_bools(), [false, true, false])
}

fn test_cumcount_head_tail_and_nth_are_row_ordered() -> void {
    if not gpu.is_available() { return }
    let rows = 8192
    let keys = keys_of(rows, 137)
    let values = values_of(rows)
    let frame = frame_of(keys, values)
    defer { frame.free() }
    let handle = group_by(frame, "key")
    defer { handle.free() }

    var seen: dict[int, int] = {}
    var expected: list[int] = []
    for key in keys {
        let before = seen.get(key, 0)
        expected.push(before)
        seen[key] = before + 1
    }
    let places = handle.cumcount()
    defer { places.free() }
    assert_eq(places.values.to_ints(), expected)

    let leading = handle.head(2)
    defer { leading.free() }
    var wanted: list[float] = []
    for (row, place) in expected.enumerate() {
        if place < 2 { wanted.push(values[row]) }
    }
    assert_eq(leading.len(), wanted.len())
    let produced = leading.column("value").to_floats()
    for (at, value) in wanted.enumerate() { assert_near(produced[at], value, tolerance: 1e-6) }

    let second = handle.nth(1)
    defer { second.free() }
    assert_eq(second.len(), 137)
    let trailing = handle.tail(1)
    defer { trailing.free() }
    assert_eq(trailing.len(), 137)
}

fn test_groups_and_ngroup_agree_with_the_codes() -> void {
    if not gpu.is_available() { return }
    let keys = keys_of(90, 5)
    let values = values_of(90)
    let frame = frame_of(keys, values)
    defer { frame.free() }
    let handle = group_by(frame, "key")
    defer { handle.free() }
    let membership = handle.ngroup().values.to_ints()
    let listed = handle.groups()
    let chosen = handle.uniques[0].to_ints()
    var counted = 0
    for (group, wanted) in chosen.enumerate() {
        let rows = listed[f"{wanted}"]
        counted += rows.len()
        var previous = -1
        for row in rows {
            assert_eq(keys[row], wanted)
            assert_eq(membership[row], group)
            assert_true(row > previous, "a group lists its rows in row order")
            previous = row
        }
    }
    assert_eq(counted, 90)
    let one = handle.get_group(f"{chosen[2]}")
    defer { one.free() }
    assert_eq(one.len(), listed[f"{chosen[2]}"].len())
}

fn test_transform_broadcasts_the_group_result() -> void {
    if not gpu.is_available() { return }
    let keys = keys_of(400, 9)
    let values = values_of(400)
    let frame = frame_of(keys, values)
    defer { frame.free() }
    let handle = group_by(frame, "key")
    defer { handle.free() }
    let spread = handle.transform(Agg.Mean)
    defer { spread.free() }
    let averages = handle.mean().column("value").to_floats()
    let chosen = handle.uniques[0].to_ints()
    var lookup: dict[int, float] = {}
    for (group, wanted) in chosen.enumerate() { lookup[wanted] = averages[group] }
    let produced = spread.column("value").to_floats()
    assert_eq(spread.len(), 400)
    for (row, key) in keys.enumerate() {
        assert_near(produced[row], lookup[key], tolerance: 1e-3)
    }
}

fn test_filter_drops_whole_groups() -> void {
    if not gpu.is_available() { return }
    let keys = [0, 0, 0, 1, 2, 2]
    let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    let frame = frame_of(keys, values)
    defer { frame.free() }
    let handle = group_by(frame, "key")
    defer { handle.free() }
    let big = handle.filter(Column.from_bools([true, false, true]))
    defer { big.free() }
    assert_eq(big.column("key").to_ints(), [0, 0, 0, 2, 2])
    let callback = handle.filter(|piece| piece.len() > 1)
    defer { callback.free() }
    assert_eq(callback.column("key").to_ints(), [0, 0, 0, 2, 2])
}

fn test_apply_is_the_escape_hatch() -> void {
    if not gpu.is_available() { return }
    let keys = [0, 0, 1, 1, 1]
    let values = [1.0, 3.0, 10.0, 20.0, 30.0]
    let frame = frame_of(keys, values)
    defer { frame.free() }
    let handle = group_by(frame, "key")
    defer { handle.free() }
    let spans = handle.apply(|piece| span_of(piece))
    assert_near(spans.values.get(0).to_float(), 2.0, tolerance: 1e-6)
    assert_near(spans.values.get(1).to_float(), 20.0, tolerance: 1e-6)
    spans.free()
}

fn test_agg_takes_all_four_shapes() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(
        ["key", "units", "price"],
        [
            Column.from_ints([0, 0, 1, 1]),
            Column.from_floats([1.0, 3.0, 5.0, 7.0]),
            Column.from_floats([2.0, 4.0, 6.0, 8.0]),
        ]
    )
    defer { frame.free() }
    let handle = group_by(frame, "key")
    defer { handle.free() }

    let plain = handle.agg(Agg.Sum)
    defer { plain.free() }
    assert_eq(plain.names, ["units", "price"])
    assert_near(plain.column("units").get(0).to_float(), 4.0, tolerance: 1e-6)

    let listed = handle.agg([Agg.Min, Agg.Max])
    defer { listed.free() }
    assert_eq(listed.names, ["units_min", "units_max", "price_min", "price_max"])
    assert_near(listed.column("price_max").get(1).to_float(), 8.0, tolerance: 1e-6)

    let mapped = handle.agg({"units": Agg.Sum, "price": [Agg.Min, Agg.Max]})
    defer { mapped.free() }
    assert_eq(mapped.names, ["units", "price_min", "price_max"])

    let named = handle.agg(named: {"total": ("units", Agg.Sum), "best": ("price", Agg.Max)})
    defer { named.free() }
    assert_eq(named.names, ["total", "best"])
    assert_near(named.column("total").get(1).to_float(), 12.0, tolerance: 1e-6)
    assert_near(named.column("best").get(0).to_float(), 4.0, tolerance: 1e-6)

    assert_eq(handle.agg("mean").names, ["units", "price"])
    assert_throws(KeyError, || handle.agg({"nowhere": Agg.Sum}))
}

fn test_a_multi_key_grouping_indexes_by_the_tuple() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(
        ["city", "quarter", "sales"],
        [
            Column.from_strs(["austin", "austin", "dallas", "dallas", "austin"]),
            Column.from_ints([1, 2, 1, 1, 1]),
            Column.from_floats([10.0, 20.0, 30.0, 40.0, 50.0]),
        ]
    )
    defer { frame.free() }
    let handle = group_by(frame, ["city", "quarter"])
    defer { handle.free() }
    assert_eq(handle.ngroups(), 3)
    let totals = handle.sum()
    defer { totals.free() }
    assert_eq(totals.index.nlevels(), 2)
    assert_eq(totals.names, ["sales"])
    let listed = handle.groups()
    assert_eq(listed["austin|1"], [0, 4])
    let places = totals.column("sales").to_floats()
    var found = 0.0
    for (group, label) in render(handle).enumerate() {
        if label == "austin|1" { found = places[group] }
    }
    assert_near(found, 60.0, tolerance: 1e-6)

    #: A minimum over text would compare dictionary codes, which are the order
    #: the levels were interned in, not alphabetical order. It is refused.
    assert_throws(TypeError, || handle.agg({"city": Agg.Max}))

    let flat = group_by(frame, ["city", "quarter"], as_index: false)
    defer { flat.free() }
    let spread = flat.sum()
    defer { spread.free() }
    assert_eq(spread.names, ["city", "quarter", "sales"])
    assert_true(spread.index.is_range(), "as_index false puts the keys in columns")
}

fn test_a_missing_key_is_dropped_or_kept_as_asked() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(
        ["city", "sales"],
        [
            Column.from_strs(["austin", null, "austin", null]),
            Column.from_floats([1.0, 2.0, 3.0, 4.0]),
        ]
    )
    defer { frame.free() }
    let dropped = group_by(frame, "city")
    defer { dropped.free() }
    assert_eq(dropped.ngroups(), 1)
    let totals = dropped.sum()
    defer { totals.free() }
    assert_near(totals.column("sales").get(0).to_float(), 4.0, tolerance: 1e-6)
    assert_eq(dropped.ngroup().values.to_ints()[1], -1)

    let kept = group_by(frame, "city", dropna: false)
    defer { kept.free() }
    assert_eq(kept.ngroups(), 2)
}

fn test_grouping_by_an_index_level_or_a_callback() -> void {
    if not gpu.is_available() { return }
    let level_names: list[str?] = ["city"]
    let frame = DataFrame(
        ["sales"],
        [Column.from_floats([1.0, 2.0, 3.0, 4.0])],
        index: Index.of([Column.from_strs(["a", "b", "a", "b"])], level_names)
    )
    defer { frame.free() }
    let levelled = group_by(frame, level: 0)
    defer { levelled.free() }
    assert_eq(levelled.ngroups(), 2)
    let totals = levelled.sum()
    defer { totals.free() }
    assert_near(totals.column("sales").get(0).to_float(), 4.0, tolerance: 1e-6)

    let called = group_by(frame, |label| Value.of_str(label.to_str().upper()))
    defer { called.free() }
    assert_eq(called.ngroups(), 2)
    assert_eq(called.uniques[0].to_strs(), ["A", "B"])

    assert_throws(ValueError, || group_by(frame))
    assert_throws(ValueError, || group_by(frame, "sales", level: 0))
}

fn test_idxmax_reports_the_first_row_of_a_tie() -> void {
    if not gpu.is_available() { return }
    let frame = frame_of([0, 0, 0, 1, 1], [5.0, 9.0, 9.0, 2.0, 1.0])
    defer { frame.free() }
    let handle = group_by(frame, "key")
    defer { handle.free() }
    let peaks = handle.idxmax()
    defer { peaks.free() }
    let troughs = handle.idxmin()
    defer { troughs.free() }
    assert_eq(peaks.column("value").to_ints(), [1, 3])
    assert_eq(troughs.column("value").to_ints(), [0, 4])
}

fn test_an_unused_category_level_still_gets_a_group() -> void {
    if not gpu.is_available() { return }
    let city = Column.from_strs(["austin", "dallas", "austin"], Dtype.Category)
    city.intern_level("houston")
    let frame = DataFrame(["city", "sales"], [city, Column.from_floats([1.0, 2.0, 3.0])])
    defer { frame.free() }
    let observed = group_by(frame, "city")
    defer { observed.free() }
    assert_eq(observed.ngroups(), 2)

    let every = group_by(frame, "city", observed: false)
    defer { every.free() }
    assert_eq(every.ngroups(), 3)
    let sizes = every.size().values.to_ints()
    assert_eq(sizes, [2, 1, 0])
    let totals = every.sum()
    defer { totals.free() }
    #: pandas' sum over nothing is zero, and its mean over nothing is missing.
    assert_near(totals.column("sales").get(2).to_float(), 0.0, tolerance: 1e-9)
    assert_true(every.mean().column("sales").get(2).is_na(), "an unused level has no mean")
}

fn test_a_grouped_series_keeps_its_name() -> void {
    if not gpu.is_available() { return }
    let series = Series(Column.from_floats([1.0, 2.0, 3.0, 4.0]), name: "sales")
    defer { series.free() }
    let handle = group_by(series, Column.from_ints([0, 1, 0, 1]))
    defer { handle.free() }
    let totals = handle.sum()
    defer { totals.free() }
    assert_eq(totals.names, ["sales"])
    assert_near(totals.column("sales").get(0).to_float(), 4.0, tolerance: 1e-6)
}

fn test_a_released_handle_refuses_to_answer() -> void {
    if not gpu.is_available() { return }
    let frame = frame_of([0, 1], [1.0, 2.0])
    defer { frame.free() }
    let handle = group_by(frame, "key")
    handle.free()
    assert_true(handle.is_freed(), "free is remembered")
    assert_throws(RuntimeError, || handle.sum())
}

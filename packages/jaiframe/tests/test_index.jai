#: Index lookup, set algebra, alignment and levels.
#:
#: Almost everything here is checked against a second computation rather than
#: against a number this module printed once. A lookup is checked against a
#: host dictionary of first occurrences; a searchsorted against a count of the
#: labels below the value; a set operation against the same operation on host
#: lists; a reindex method against the neighbour a host scan picks. Where the
#: answer is a lane of positions, the check is to take the index by it and ask
#: `equals` whether the labels came back — which reaches the answer through
#: `Column.equal_rows` instead of through the table that produced it.
#:
#: Three lookup plans have to be reached separately, because they share no
#: code: a range index is arithmetic, a monotonic one bisects, and everything
#: else goes through the hash table. Each test below says which one it is on.
import std.gpu as gpu
from std.gpu import Buffer
from std.math import NAN
from std.test import assert_eq, assert_false, assert_near, assert_throws, assert_true
from jaiframe.column import Column, DataFrame, Dtype, Index, Series, Value
from jaiframe.column import NA
from jaiframe.index import IndexLookup, NOT_FOUND, PLAN_HASHED, PLAN_RANGE, PLAN_SORTED
from jaiframe.index import align, align_frame_series, align_series, append, difference
from jaiframe.index import droplevel, equals, from_arrays, from_product, from_tuples
from jaiframe.index import get_indexer, get_indexer_for_index, get_level_values
from jaiframe.index import get_loc, get_loc_tuple
from jaiframe.index import identical, intersection, is_monotonic_decreasing
from jaiframe.index import is_monotonic_increasing, is_unique, joint_index, reindex
from jaiframe.index import reindex_series, rename_axis, reorder_levels, reset_index
from jaiframe.index import searchsorted, set_index, set_names, swaplevel
from jaiframe.index import symmetric_difference, union, unique, xs

#: Where each target first appears among the labels, worked out with a plain
#: dictionary. This is the definition `get_indexer` implements.
fn host_indexer(labels: list[int], targets: list[int]) -> list[int] {
    var first: dict[int, int] = {}
    for (at, label) in labels.enumerate() {
        if first.get(label, -1) < 0 { first[label] = at }
    }
    return targets.map(|target| first.get(target, -1))
}

fn host_text_indexer(labels: list[str], targets: list[str]) -> list[int] {
    var first: dict[str, int] = {}
    for (at, label) in labels.enumerate() {
        if first.get(label, -1) < 0 { first[label] = at }
    }
    return targets.map(|target| first.get(target, -1))
}

fn positions_of(lane: Buffer, count: int) -> list[int] {
    return lane.download(0, count).map(|slot| int(slot))
}

fn int_index(labels: list[int], name: str? = null) -> Index {
    let names: list[str?] = [name]
    return Index.of([Column.from_ints(labels)], names)
}

fn text_index(labels: list[str], name: str? = null) -> Index {
    let names: list[str?] = [name]
    var cells: list[str?] = []
    for label in labels { cells.push(label) }
    return Index.of([Column.from_strs(cells)], names)
}

fn labels_of(index: Index) -> list[int] {
    let levels = index.materialize()
    let values = levels[0].to_ints()
    for level in levels { level.free() }
    return values
}

fn text_labels_of(index: Index) -> list[str?] {
    let levels = index.materialize()
    let values = levels[0].to_strs()
    for level in levels { level.free() }
    return values
}

fn test_a_range_index_is_answered_by_arithmetic() -> void {
    if not gpu.is_available() { return }
    let index = Index.range(5, start: 10, step: 3)
    defer { index.free() }
    let lookup = IndexLookup.of(index)
    defer { lookup.free() }
    assert_eq(lookup.plan, PLAN_RANGE, "an implied ramp needs no table")
    assert_true(lookup.is_unique(), "a ramp of nonzero step repeats nothing")
    assert_eq(get_loc(index, Value.of_int(16)), 2)
    assert_eq(get_loc(index, Value.of_int(10)), 0)
    assert_eq(get_loc(index, Value.of_int(22)), 4)
    #: 17 is between two rungs, 7 is before the first and 25 past the last.
    let probes = [Value.of_int(17), Value.of_int(7), Value.of_int(25), Value.of_int(13)]
    let found = get_indexer(index, probes)
    defer { found.free() }
    assert_eq(positions_of(found, 4), [NOT_FOUND, NOT_FOUND, NOT_FOUND, 1])
    assert_throws(KeyError, || get_loc(index, Value.of_int(11)))
}

fn test_a_descending_range_index_still_inverts() -> void {
    if not gpu.is_available() { return }
    let index = Index.range(4, start: 9, step: -2)
    defer { index.free() }
    assert_eq(labels_of(index), [9, 7, 5, 3])
    assert_eq(get_loc(index, Value.of_int(5)), 2)
    assert_true(is_monotonic_decreasing(index), "a negative step falls")
    assert_false(is_monotonic_increasing(index), "and does not rise")
    assert_throws(ValueError, || searchsorted(index, Column.from_ints([5])))
}

fn test_a_monotonic_index_bisects() -> void {
    if not gpu.is_available() { return }
    let labels = [2, 4, 6, 8, 10, 12]
    let index = int_index(labels)
    defer { index.free() }
    let lookup = IndexLookup.of(index)
    defer { lookup.free() }
    assert_eq(lookup.plan, PLAN_SORTED, "sorted labels bisect instead of hashing")
    let targets = [6, 3, 12, 2, 99]
    let found = get_indexer(index, targets.map(|value| Value.of_int(value)))
    defer { found.free() }
    assert_eq(positions_of(found, targets.len()), host_indexer(labels, targets))
}

fn test_an_unsorted_index_goes_through_the_table() -> void {
    if not gpu.is_available() { return }
    let labels = [40, 10, 90, 30, 70, 20]
    let index = int_index(labels)
    defer { index.free() }
    let lookup = IndexLookup.of(index)
    defer { lookup.free() }
    assert_eq(lookup.plan, PLAN_HASHED, "unsorted labels need the table")
    assert_true(lookup.is_unique(), "these labels are distinct")
    let targets = [90, 40, 11, 20, 70, 0]
    let found = get_indexer(index, targets.map(|value| Value.of_int(value)))
    defer { found.free() }
    assert_eq(positions_of(found, targets.len()), host_indexer(labels, targets))
}

fn test_a_repeated_label_reports_its_first_row() -> void {
    if not gpu.is_available() { return }
    let labels = [5, 7, 5, 9, 7, 5]
    let index = int_index(labels)
    defer { index.free() }
    assert_false(is_unique(index), "5 and 7 are on three and two rows")
    let lookup = IndexLookup.of(index)
    defer { lookup.free() }
    assert_eq(lookup.nunique(), 3)
    let targets = [5, 7, 9]
    let found = get_indexer(index, targets.map(|value| Value.of_int(value)))
    defer { found.free() }
    assert_eq(positions_of(found, 3), host_indexer(labels, targets))
    let deduped = unique(index)
    defer { deduped.free() }
    assert_eq(labels_of(deduped), [5, 7, 9])
}

fn test_a_sorted_index_with_repeats_dedupes_by_its_runs() -> void {
    if not gpu.is_available() { return }
    let labels = [1, 1, 2, 3, 3, 3, 4]
    let index = int_index(labels)
    defer { index.free() }
    let lookup = IndexLookup.of(index)
    defer { lookup.free() }
    assert_eq(lookup.plan, PLAN_SORTED, "a run of equals is still monotonic")
    assert_false(lookup.is_unique(), "but not unique")
    assert_eq(lookup.nunique(), 4)
    let deduped = unique(index)
    defer { deduped.free() }
    assert_eq(labels_of(deduped), [1, 2, 3, 4])
    assert_eq(get_loc(index, Value.of_int(3)), 3)
}

fn test_text_labels_look_up_by_their_code() -> void {
    if not gpu.is_available() { return }
    let labels = ["austin", "dallas", "el paso", "austin"]
    let index = text_index(labels)
    defer { index.free() }
    let targets = ["dallas", "austin", "houston", "el paso"]
    let found = get_indexer(index, targets.map(|value| Value.of_str(value)))
    defer { found.free() }
    assert_eq(positions_of(found, 4), host_text_indexer(labels, targets))
}

fn test_a_label_the_dictionary_never_held_is_a_miss() -> void {
    if not gpu.is_available() { return }
    let index = text_index(["a", "b", "c"])
    defer { index.free() }
    assert_eq(get_indexer_for_index(index, text_index(["z"])).download(0, 1)[0], -1.0)
    assert_throws(KeyError, || get_loc(index, Value.of_str("z")))
}

fn test_a_multi_index_has_to_agree_on_every_level() -> void {
    if not gpu.is_available() { return }
    let names: list[str?] = ["city", "year"]
    let index = from_arrays(
        [
            Column.from_strs(["austin", "austin", "dallas", "dallas"]),
            Column.from_ints([2024, 2025, 2024, 2025]),
        ],
        names
    )
    defer { index.free() }
    assert_eq(index.nlevels(), 2)
    assert_eq(get_loc_tuple(index, [Value.of_str("dallas"), Value.of_int(2024)]), 2)
    assert_eq(get_loc_tuple(index, [Value.of_str("austin"), Value.of_int(2025)]), 1)
    #: Both halves exist, the pair does not.
    assert_throws(KeyError, || get_loc_tuple(index, [Value.of_str("austin"), Value.of_int(2023)]))
    let target = from_tuples(
        [
            [Value.of_str("dallas"), Value.of_int(2025)],
            [Value.of_str("austin"), Value.of_int(2024)],
            [Value.of_str("houston"), Value.of_int(2024)],
        ]
    )
    defer { target.free() }
    let found = get_indexer_for_index(index, target)
    defer { found.free() }
    assert_eq(positions_of(found, 3), [3, 0, NOT_FOUND])
}

fn test_monotonicity_agrees_with_a_host_scan() -> void {
    if not gpu.is_available() { return }
    let cases = [[1, 2, 3, 4], [1, 1, 2, 2], [4, 3, 2, 1], [1, 3, 2, 4], [7, 7, 7, 7]]
    for labels in cases {
        let index = int_index(labels)
        var rising = true
        var falling = true
        for at in 1..labels.len() {
            if labels[at] < labels[at - 1] { rising = false }
            if labels[at] > labels[at - 1] { falling = false }
        }
        assert_eq(is_monotonic_increasing(index), rising, f"rising over {labels}")
        assert_eq(is_monotonic_decreasing(index), falling, f"falling over {labels}")
        index.free()
    }
}

fn test_a_null_label_orders_against_nothing() -> void {
    if not gpu.is_available() { return }
    let names: list[str?] = [null]
    let index = Index.of([Column.from_floats([1.0, NAN, 3.0])], names)
    defer { index.free() }
    assert_false(is_monotonic_increasing(index), "a missing label breaks the order")
    assert_false(is_monotonic_decreasing(index), "in both directions")
    let lookup = IndexLookup.of(index)
    defer { lookup.free() }
    assert_eq(lookup.plan, PLAN_HASHED, "so the table has to take it")
    #: Two missing labels are the same label here, as they are in pandas'
    #: index — this is a lookup, not an arithmetic comparison.
    assert_eq(get_loc(index, NA), 1)
    assert_eq(get_loc(index, Value.of_float(3.0)), 2)
}

fn test_searchsorted_counts_the_labels_below() -> void {
    if not gpu.is_available() { return }
    let labels = [10, 20, 20, 30, 40]
    let index = int_index(labels)
    defer { index.free() }
    let probes = [5, 10, 20, 25, 40, 99]
    let values = Column.from_ints(probes)
    defer { values.free() }
    let left = searchsorted(index, values, "left")
    defer { left.free() }
    let right = searchsorted(index, values, "right")
    defer { right.free() }
    var below: list[int] = []
    var not_above: list[int] = []
    for probe in probes {
        var under = 0
        var upto = 0
        for label in labels {
            if label < probe { under += 1 }
            if label <= probe { upto += 1 }
        }
        below.push(under)
        not_above.push(upto)
    }
    assert_eq(positions_of(left, probes.len()), below)
    assert_eq(positions_of(right, probes.len()), not_above)
}

fn test_searchsorted_on_a_range_index_needs_no_labels() -> void {
    if not gpu.is_available() { return }
    let index = Index.range(5, start: 4, step: 2)
    defer { index.free() }
    let values = Column.from_ints([3, 4, 5, 12, 13])
    defer { values.free() }
    let left = searchsorted(index, values, "left")
    defer { left.free() }
    let right = searchsorted(index, values, "right")
    defer { right.free() }
    let labels = labels_of(index)
    var below: list[int] = []
    var not_above: list[int] = []
    for probe in [3, 4, 5, 12, 13] {
        var under = 0
        var upto = 0
        for label in labels {
            if label < probe { under += 1 }
            if label <= probe { upto += 1 }
        }
        below.push(under)
        not_above.push(upto)
    }
    assert_eq(positions_of(left, 5), below)
    assert_eq(positions_of(right, 5), not_above)
}

fn test_set_operations_are_set_algebra() -> void {
    if not gpu.is_available() { return }
    let left_labels = [3, 1, 4, 1, 5]
    let right_labels = [5, 9, 1]
    let left = int_index(left_labels)
    defer { left.free() }
    let right = int_index(right_labels)
    defer { right.free() }

    var seen: dict[int, bool] = {}
    var expected_union: list[int] = []
    for label in left_labels + right_labels {
        if seen.get(label, false) { continue }
        seen[label] = true
        expected_union.push(label)
    }
    let joined = union(left, right)
    defer { joined.free() }
    assert_eq(labels_of(joined), expected_union)

    var in_right: dict[int, bool] = {}
    for label in right_labels { in_right[label] = true }
    var shared: list[int] = []
    var only_left: list[int] = []
    var picked: dict[int, bool] = {}
    for label in left_labels {
        if picked.get(label, false) { continue }
        picked[label] = true
        if in_right.get(label, false) { shared.push(label) } else { only_left.push(label) }
    }
    let both = intersection(left, right)
    defer { both.free() }
    assert_eq(labels_of(both), shared)
    let apart = difference(left, right)
    defer { apart.free() }
    assert_eq(labels_of(apart), only_left)
    let either = symmetric_difference(left, right)
    defer { either.free() }
    assert_eq(labels_of(either), only_left + [9])
}

fn test_append_stacks_and_puts_text_on_one_table() -> void {
    if not gpu.is_available() { return }
    let left = text_index(["austin", "dallas"], "city")
    defer { left.free() }
    let right = text_index(["dallas", "houston"], "city")
    defer { right.free() }
    let stacked = append(left, right)
    defer { stacked.free() }
    assert_eq(text_labels_of(stacked), ["austin", "dallas", "dallas", "houston"])
    assert_eq(stacked.names[0], "city")
    let deduped = union(left, right)
    defer { deduped.free() }
    assert_eq(text_labels_of(deduped), ["austin", "dallas", "houston"])
}

fn test_a_set_operation_drops_a_name_the_two_disagree_on() -> void {
    if not gpu.is_available() { return }
    let left = int_index([1, 2], "row")
    defer { left.free() }
    let right = int_index([2, 3], "line")
    defer { right.free() }
    let joined = union(left, right)
    defer { joined.free() }
    assert_eq(joined.names[0], null)
}

fn test_equals_reads_labels_and_identical_reads_the_rest() -> void {
    if not gpu.is_available() { return }
    let ramp = Index.range(3)
    defer { ramp.free() }
    let stored = int_index([0, 1, 2], "row")
    defer { stored.free() }
    assert_true(equals(ramp, stored), "a range index is the ramp it stands for")
    assert_false(identical(ramp, stored), "but it is not the same kind of index")
    let named = int_index([0, 1, 2], "row")
    defer { named.free() }
    assert_true(identical(stored, named), "same labels, same name, same kind")
    let renamed = set_names(stored, [null])
    defer { renamed.free() }
    assert_false(identical(stored, renamed), "a name is part of being identical")
    assert_true(equals(stored, renamed), "but not part of being equal")
    let shorter = int_index([0, 1])
    defer { shorter.free() }
    assert_false(equals(stored, shorter), "different lengths are different labels")
    let text = text_index(["0", "1", "2"])
    defer { text.free() }
    assert_false(equals(stored, text), "text is not a number")
}

fn test_reindex_moves_rows_and_fills_the_holes() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(
        ["high"],
        [Column.from_floats([38.0, 41.0, 36.0])],
        index: int_index([2, 4, 6])
    )
    defer { frame.free() }
    let target = int_index([4, 6, 8, 2])
    defer { target.free() }
    let moved = reindex(frame, target)
    defer { moved.free() }
    assert_eq(moved.len(), 4)
    assert_eq(moved.column("high").null_mask(), [false, false, true, false])
    assert_near(moved.column("high").get(0).to_float(), 41.0, tolerance: 1e-6)
    assert_near(moved.column("high").get(3).to_float(), 38.0, tolerance: 1e-6)
    let filled = reindex(frame, target, fill_value: Value.of_float(-1.0))
    defer { filled.free() }
    assert_eq(filled.column("high").null_count(), 0)
    assert_near(filled.column("high").get(2).to_float(), -1.0, tolerance: 1e-6)
}

fn test_reindex_refuses_an_index_whose_labels_repeat() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(["v"], [Column.from_floats([1.0, 2.0, 3.0])], index: int_index([1, 1, 2]))
    defer { frame.free() }
    assert_throws(ValueError, || reindex(frame, int_index([1, 2])))
    #: Reindexing onto the index it already has is not a move at all, so the
    #: repeats do not have to be resolved.
    let same = reindex(frame, int_index([1, 1, 2]))
    defer { same.free() }
    assert_eq(same.column("v").to_ints(), [1, 2, 3])
}

fn test_a_fill_method_reaches_for_a_neighbour() -> void {
    if not gpu.is_available() { return }
    let labels = [10, 20, 40]
    #: The value at row `i` is `i + 1`, so an expected row reads straight out
    #: of the expected value.
    let series = Series(Column.from_floats([1.0, 2.0, 3.0]), index: int_index(labels))
    defer { series.free() }
    let probes = [5, 10, 25, 35, 40, 50]
    let target = int_index(probes)
    defer { target.free() }

    var padded: list[int] = []
    var backfilled: list[int] = []
    var nearest: list[int] = []
    for probe in probes {
        var below = -1
        var above = -1
        for (at, label) in labels.enumerate() {
            if label <= probe { below = at }
            if label >= probe and above < 0 { above = at }
        }
        padded.push(below)
        backfilled.push(above)
        if below < 0 {
            nearest.push(above)
        } elif above < 0 {
            nearest.push(below)
        } else {
            nearest.push(if probe - labels[below] <= labels[above] - probe { below } else { above })
        }
    }

    let by_pad = reindex_series(series, target, method: "ffill")
    defer { by_pad.free() }
    assert_eq(by_pad.values.to_ints(), padded.map(|at| if at < 0 { 0 } else { at + 1 }))
    assert_eq(by_pad.values.null_mask(), padded.map(|at| at < 0))

    let by_backfill = reindex_series(series, target, method: "bfill")
    defer { by_backfill.free() }
    assert_eq(by_backfill.values.null_mask(), backfilled.map(|at| at < 0))
    assert_eq(by_backfill.values.to_ints(), backfilled.map(|at| if at < 0 { 0 } else { at + 1 }))

    let by_nearest = reindex_series(series, target, method: "nearest")
    defer { by_nearest.free() }
    assert_eq(by_nearest.values.to_ints(), nearest.map(|at| at + 1))
}

fn test_a_fill_method_needs_an_ordered_index() -> void {
    if not gpu.is_available() { return }
    let series = Series(Column.from_floats([1.0, 2.0]), index: int_index([9, 3]))
    defer { series.free() }
    assert_throws(ValueError, || reindex_series(series, int_index([3, 9]), method: "ffill"))
    assert_throws(ValueError, || reindex_series(series, int_index([3, 9]), method: "sideways"))
}

fn test_align_puts_two_frames_on_one_index() -> void {
    if not gpu.is_available() { return }
    let left = DataFrame(["a"], [Column.from_floats([1.0, 2.0])], index: int_index([1, 2]))
    defer { left.free() }
    let right = DataFrame(["b"], [Column.from_floats([3.0, 4.0])], index: int_index([2, 3]))
    defer { right.free() }

    let outer = align(left, right, "outer")
    defer { outer[0].free() }
    defer { outer[1].free() }
    assert_eq(labels_of(outer[0].index), [1, 2, 3])
    assert_true(equals(outer[0].index, outer[1].index), "both land on one index")
    assert_eq(outer[0].column("a").null_mask(), [false, false, true])
    assert_eq(outer[1].column("b").null_mask(), [true, false, false])

    let inner = align(left, right, "inner")
    defer { inner[0].free() }
    defer { inner[1].free() }
    assert_eq(labels_of(inner[0].index), [2])
    assert_near(inner[0].column("a").get(0).to_float(), 2.0, tolerance: 1e-6)
    assert_near(inner[1].column("b").get(0).to_float(), 3.0, tolerance: 1e-6)

    let kept = align(left, right, "left")
    defer { kept[0].free() }
    defer { kept[1].free() }
    assert_eq(labels_of(kept[0].index), [1, 2])
}

fn test_aligning_two_matching_indexes_leaves_them_alone() -> void {
    if not gpu.is_available() { return }
    let left = Series(Column.from_floats([1.0, 2.0]), name: "a")
    defer { left.free() }
    let right = Series(Column.from_floats([3.0, 4.0]), name: "b")
    defer { right.free() }
    let pair = align_series(left, right, "inner")
    defer { pair[0].free() }
    defer { pair[1].free() }
    assert_eq(pair[0].values.to_ints(), [1, 2])
    assert_eq(pair[1].values.to_ints(), [3, 4])
    assert_true(pair[0].index.is_range(), "an untouched range index stays implied")
}

fn test_joint_index_names_the_four_joins() -> void {
    if not gpu.is_available() { return }
    let left = int_index([1, 2])
    defer { left.free() }
    let right = int_index([2, 3])
    defer { right.free() }
    for entry in [["outer", "1,2,3"], ["inner", "2"], ["left", "1,2"], ["right", "2,3"]] {
        let joined = joint_index(left, right, entry[0])
        let spelled = labels_of(joined).map(|label| f"{label}").join(",")
        assert_eq(spelled, entry[1], f"joint index for {entry[0]}")
        joined.free()
    }
    assert_throws(ValueError, || joint_index(left, right, "sideways"))
}

fn test_from_product_is_the_cartesian_product() -> void {
    if not gpu.is_available() { return }
    let cities = ["austin", "dallas", "el paso"]
    let years = [2023, 2024]
    var city_cells: list[str?] = []
    for city in cities { city_cells.push(city) }
    let names: list[str?] = ["city", "year"]
    let index = from_product([Column.from_strs(city_cells), Column.from_ints(years)], names)
    defer { index.free() }
    assert_eq(index.len(), cities.len() * years.len())
    var expected_city: list[str?] = []
    var expected_year: list[int] = []
    for city in cities {
        for year in years {
            expected_city.push(city)
            expected_year.push(year)
        }
    }
    let level_city = get_level_values(index, "city")
    defer { level_city.free() }
    let level_year = get_level_values(index, 1)
    defer { level_year.free() }
    assert_eq(level_city.to_strs(), expected_city)
    assert_eq(level_year.to_ints(), expected_year)
}

fn test_from_product_carries_three_levels() -> void {
    if not gpu.is_available() { return }
    let index = from_product(
        [
            Column.from_ints([0, 1]),
            Column.from_ints([0, 1, 2]),
            Column.from_ints([0, 1, 2, 3, 4]),
        ]
    )
    defer { index.free() }
    assert_eq(index.len(), 30)
    #: The digits are a mixed-radix counter, so row `i` reads back as `i`.
    let first = get_level_values(index, 0).to_ints()
    let second = get_level_values(index, 1).to_ints()
    let third = get_level_values(index, 2).to_ints()
    for row in 0..30 {
        assert_eq(first[row] * 15 + second[row] * 5 + third[row], row, f"row {row}")
    }
}

fn test_levels_can_be_dropped_swapped_and_reordered() -> void {
    if not gpu.is_available() { return }
    let names: list[str?] = ["city", "year", "month"]
    let index = from_arrays(
        [
            Column.from_strs(["austin", "dallas"]),
            Column.from_ints([2024, 2025]),
            Column.from_ints([1, 7]),
        ],
        names
    )
    defer { index.free() }
    let without_year = droplevel(index, "year")
    defer { without_year.free() }
    assert_eq(without_year.nlevels(), 2)
    assert_eq(without_year.names, ["city", "month"])

    let swapped = swaplevel(index)
    defer { swapped.free() }
    assert_eq(swapped.names, ["city", "month", "year"])
    assert_eq(get_level_values(swapped, 1).to_ints(), [1, 7])

    let reordered = reorder_levels(index, [2, 0, 1])
    defer { reordered.free() }
    assert_eq(reordered.names, ["month", "city", "year"])
    assert_throws(ValueError, || reorder_levels(index, [0, 0, 1]))
    assert_throws(ValueError, || reorder_levels(index, [0, 1]))
    assert_throws(KeyError, || droplevel(index, "week"))
}

fn test_a_single_level_index_cannot_drop_it() -> void {
    if not gpu.is_available() { return }
    let index = int_index([1, 2])
    defer { index.free() }
    assert_throws(ValueError, || droplevel(index, 0))
}

fn test_xs_takes_a_cross_section() -> void {
    if not gpu.is_available() { return }
    let names: list[str?] = ["city", "year"]
    let frame = DataFrame(
        ["high"],
        [Column.from_floats([38.0, 39.0, 41.0, 42.0])],
        index: from_arrays(
            [
                Column.from_strs(["austin", "austin", "dallas", "dallas"]),
                Column.from_ints([2024, 2025, 2024, 2025]),
            ],
            names
        )
    )
    defer { frame.free() }
    let dallas = xs(frame, Value.of_str("dallas"), "city")
    defer { dallas.free() }
    assert_eq(dallas.len(), 2)
    assert_eq(dallas.index.nlevels(), 1)
    assert_eq(dallas.index.names, ["year"])
    assert_eq(dallas.column("high").to_ints(), [41, 42])
    let in_2024 = xs(frame, Value.of_int(2024), "year")
    defer { in_2024.free() }
    assert_eq(in_2024.column("high").to_ints(), [38, 41])
    let nowhere = xs(frame, Value.of_str("houston"), "city")
    defer { nowhere.free() }
    assert_eq(nowhere.len(), 0)
}

fn test_set_index_and_reset_index_come_back_round() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(
        ["city", "year", "high"],
        [
            Column.from_strs(["austin", "dallas"]),
            Column.from_ints([2024, 2025]),
            Column.from_floats([38.0, 41.0]),
        ]
    )
    defer { frame.free() }
    let keyed = set_index(frame, ["city", "year"])
    defer { keyed.free() }
    assert_eq(keyed.names, ["high"])
    assert_eq(keyed.index.nlevels(), 2)
    assert_eq(keyed.index.names, ["city", "year"])
    assert_eq(get_loc_tuple(keyed.index, [Value.of_str("dallas"), Value.of_int(2025)]), 1)

    let restored = reset_index(keyed)
    defer { restored.free() }
    assert_eq(restored.names, ["city", "year", "high"])
    assert_true(restored.index.is_range(), "with every level moved out the index is a ramp again")
    assert_eq(restored.column("city").to_strs(), ["austin", "dallas"])
    assert_eq(restored.column("year").to_ints(), [2024, 2025])

    let kept = set_index(frame, ["city"], drop: false)
    defer { kept.free() }
    assert_eq(kept.names, ["city", "year", "high"])
    assert_eq(kept.index.nlevels(), 1)
}

fn test_reset_index_can_move_one_level_and_can_throw_them_away() -> void {
    if not gpu.is_available() { return }
    let names: list[str?] = ["city", "year"]
    let frame = DataFrame(
        ["high"],
        [Column.from_floats([38.0, 41.0])],
        index: from_arrays(
            [Column.from_strs(["austin", "dallas"]), Column.from_ints([2024, 2025])],
            names
        )
    )
    defer { frame.free() }
    let partly = reset_index(frame, level: [1])
    defer { partly.free() }
    assert_eq(partly.names, ["year", "high"])
    assert_eq(partly.index.nlevels(), 1)
    assert_eq(partly.index.names, ["city"])
    let bare = reset_index(frame, drop: true)
    defer { bare.free() }
    assert_eq(bare.names, ["high"])
    assert_true(bare.index.is_range(), "dropped levels leave a ramp")
}

fn test_reset_index_names_an_unnamed_level() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(["v"], [Column.from_floats([1.0, 2.0])], index: int_index([7, 8]))
    defer { frame.free() }
    let moved = reset_index(frame)
    defer { moved.free() }
    assert_eq(moved.names, ["index", "v"])
    assert_eq(moved.column("index").to_ints(), [7, 8])
}

fn test_rename_axis_renames_the_index() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(["v"], [Column.from_floats([1.0])])
    defer { frame.free() }
    let named = rename_axis(frame, ["row"])
    assert_eq(named.index.names, ["row"])
    assert_true(named.index.is_range(), "renaming does not materialise the ramp")
    named.index.free()
}

#: The table, the scan behind `unique` and the reduces all behave differently
#: past one threadgroup, and the only way to reach that is with enough labels.
#: The index is built with two `Index.range` calls and an append, so eighty
#: thousand labels cost two dispatches rather than eighty thousand boxed
#: floats — and it is deliberately not monotonic, so the lookup has to take
#: the hash table.
fn test_the_table_holds_past_one_threadgroup() -> void {
    if not gpu.is_available() { return }
    let half = 40000
    let upper = Index.range(half, start: half)
    defer { upper.free() }
    let lower = Index.range(half)
    defer { lower.free() }
    let index = append(upper, lower)
    defer { index.free() }
    assert_eq(index.len(), 2 * half)
    assert_false(is_monotonic_increasing(index), "the join drops from 79999 to 0")

    let lookup = IndexLookup.of(index)
    defer { lookup.free() }
    assert_eq(lookup.plan, PLAN_HASHED, "an unsorted index of this size needs the table")
    assert_true(lookup.is_unique(), "the two halves do not overlap")
    assert_eq(lookup.nunique(), 2 * half)

    let target = Index.range(2 * half)
    defer { target.free() }
    let positions = lookup.locate(target)
    defer { positions.free() }
    #: Take the index by the positions the table produced and ask whether the
    #: labels came back. That reaches the answer through `equal_rows` rather
    #: than through the walk that produced it.
    let echoed = index.take(positions, target.len())
    defer { echoed.free() }
    assert_true(equals(echoed, target), "every label came back from where the table put it")

    let missing = Index.range(8, start: 4 * half)
    defer { missing.free() }
    let absent = lookup.locate(missing)
    defer { absent.free() }
    assert_eq(positions_of(absent, 8), [-1, -1, -1, -1, -1, -1, -1, -1])
}

#: Every repeat of a label has to settle into the one slot its key hashes to,
#: whichever thread won the race for it. Eighty thousand rows over two labels
#: is the extreme of that, and it is the case a design that walked a bucket
#: from the host would spend eighty thousand rounds on.
fn test_the_table_holds_when_almost_every_label_repeats() -> void {
    if not gpu.is_available() { return }
    let rows = 40000
    let names: list[str?] = ["label"]
    let sevens = Index.of([Column.full(Dtype.Int64, rows, Value.of_int(7))], names)
    defer { sevens.free() }
    let threes = Index.of([Column.full(Dtype.Int64, rows, Value.of_int(3))], names)
    defer { threes.free() }
    let index = append(sevens, threes)
    defer { index.free() }
    let lookup = IndexLookup.of(index)
    defer { lookup.free() }
    assert_eq(lookup.plan, PLAN_HASHED, "a falling index is not a rising one")
    assert_false(lookup.is_unique(), "two labels over eighty thousand rows")
    assert_eq(lookup.nunique(), 2)
    assert_eq(get_loc(index, Value.of_int(3)), rows)
    let deduped = unique(index)
    defer { deduped.free() }
    assert_eq(labels_of(deduped), [7, 3])
}

#: The other shape of repetition: every label on exactly two rows, spread over
#: the whole table rather than piled into two slots.
fn test_the_table_holds_when_every_label_appears_twice() -> void {
    if not gpu.is_available() { return }
    let rows = 30000
    let once = Index.range(rows)
    defer { once.free() }
    let index = append(once, once)
    defer { index.free() }
    let lookup = IndexLookup.of(index)
    defer { lookup.free() }
    assert_eq(lookup.plan, PLAN_HASHED, "the second copy breaks the order")
    assert_eq(lookup.nunique(), rows)
    let deduped = unique(index)
    defer { deduped.free() }
    assert_true(equals(deduped, once), "the first copy of each label is what survives")
}

fn test_a_lookup_refuses_to_be_used_after_it_is_freed() -> void {
    if not gpu.is_available() { return }
    let index = int_index([3, 1, 2])
    defer { index.free() }
    let lookup = IndexLookup.of(index)
    lookup.free()
    assert_true(lookup.is_freed(), "freeing is recorded")
    lookup.free()
    assert_throws(RuntimeError, || lookup.locate(index))
}

fn test_an_empty_index_answers_without_a_table() -> void {
    if not gpu.is_available() { return }
    let index = int_index([])
    defer { index.free() }
    assert_true(is_unique(index), "nothing repeats")
    assert_true(is_monotonic_increasing(index), "and nothing falls")
    let target = int_index([1, 2])
    defer { target.free() }
    let found = get_indexer_for_index(index, target)
    defer { found.free() }
    assert_eq(positions_of(found, 2), [NOT_FOUND, NOT_FOUND])
}

#: A single row cannot be out of order, so nothing in the ordering pass rules
#: out an index whose one label is missing — the bisect would then compare
#: against whatever the value lane holds under it.
fn test_an_index_of_one_missing_label_does_not_bisect() -> void {
    if not gpu.is_available() { return }
    let names: list[str?] = [null]
    let index = Index.of([Column.from_floats([NAN])], names)
    defer { index.free() }
    let lookup = IndexLookup.of(index)
    defer { lookup.free() }
    assert_eq(lookup.plan, PLAN_HASHED, "a missing label is not a searchable one")
    assert_eq(get_loc(index, NA), 0)
    let found = get_indexer(index, [Value.of_float(0.0)])
    defer { found.free() }
    assert_eq(positions_of(found, 1), [NOT_FOUND])
}

fn test_labels_of_different_dtypes_are_simply_unequal() -> void {
    if not gpu.is_available() { return }
    let whole = int_index([1, 2, 3])
    defer { whole.free() }
    let names: list[str?] = [null]
    let fractional = Index.of([Column.from_floats([1.0, 2.0, 3.0])], names)
    defer { fractional.free() }
    assert_false(equals(whole, fractional), "an int label is not a float one here")
    #: And a set operation over the pair says so rather than answering wrongly.
    assert_throws(TypeError, || get_indexer_for_index(whole, fractional))
}

fn test_a_text_target_cannot_be_looked_up_in_a_number() -> void {
    if not gpu.is_available() { return }
    let numbers = int_index([1, 2])
    defer { numbers.free() }
    let words = text_index(["1", "2"])
    defer { words.free() }
    assert_throws(TypeError, || get_indexer_for_index(numbers, words))
    assert_throws(TypeError, || get_indexer_for_index(Index.range(2), words))
}

fn test_reindexing_a_range_index_needs_no_labels() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(["v"], [Column.from_floats([0.0, 1.0, 2.0, 3.0, 4.0])])
    defer { frame.free() }
    let target = int_index([3, 1, 9])
    defer { target.free() }
    let moved = reindex(frame, target)
    defer { moved.free() }
    assert_eq(moved.column("v").null_mask(), [false, false, true])
    assert_near(moved.column("v").get(0).to_float(), 3.0, tolerance: 1e-6)
    assert_near(moved.column("v").get(1).to_float(), 1.0, tolerance: 1e-6)
    assert_true(equals(moved.index, target), "the result carries the target's labels")
}

fn test_align_frame_and_series_share_one_index() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(["a"], [Column.from_floats([1.0, 2.0])], index: int_index([1, 2]))
    defer { frame.free() }
    let series = Series(Column.from_floats([9.0]), index: int_index([2]), name: "b")
    defer { series.free() }
    let pair = align_frame_series(frame, series, "outer")
    defer { pair[0].free() }
    defer { pair[1].free() }
    assert_eq(labels_of(pair[0].index), [1, 2])
    assert_eq(pair[1].values.null_mask(), [true, false])
    assert_near(pair[1].values.get(1).to_float(), 9.0, tolerance: 1e-6)
    assert_eq(pair[1].name, "b")
}

fn test_a_product_with_an_empty_level_is_empty() -> void {
    if not gpu.is_available() { return }
    let index = from_product([Column.from_ints([1, 2]), Column.from_ints([])])
    defer { index.free() }
    assert_eq(index.len(), 0)
    assert_eq(index.nlevels(), 2)
}

fn test_searchsorted_over_text_follows_the_dictionary() -> void {
    if not gpu.is_available() { return }
    #: The levels are interned in order here, so code order and alphabetical
    #: order happen to agree; what the test pins is that the answer is the
    #: count of labels below, whatever order that is.
    let index = text_index(["ann", "bob", "cal"])
    defer { index.free() }
    assert_true(is_monotonic_increasing(index), "codes rise with first appearance")
    var probes: list[str?] = ["bob", "cal", "ann"]
    let values = Column.from_strs(probes)
    defer { values.free() }
    let left = searchsorted(index, values, "left")
    defer { left.free() }
    assert_eq(positions_of(left, 3), [1, 2, 0])
}

fn test_a_lookup_answers_many_targets_from_one_table() -> void {
    if not gpu.is_available() { return }
    let index = int_index([50, 20, 80, 10])
    defer { index.free() }
    let lookup = IndexLookup.of(index)
    defer { lookup.free() }
    for entry in [[80, 2], [10, 3], [50, 0], [99, NOT_FOUND]] {
        let target = int_index([entry[0]])
        let found = lookup.locate(target)
        assert_eq(positions_of(found, 1), [entry[1]], f"looking {entry[0]} up")
        found.free()
        target.free()
    }
}

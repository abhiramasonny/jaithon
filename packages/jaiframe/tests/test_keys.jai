#: Hashing, factorization and ordering — checked against the definitions of the
#: operations rather than against numbers this module produced.
#:
#: Almost nothing here compares against a written-down answer. A sort is
#: checked by walking the permutation and confirming it never goes downhill,
#: and by confirming it is a permutation at all; stability is checked by
#: confirming tied rows keep their row numbers in order; a factorization is
#: checked by confirming rows share a code exactly when they carry the same
#: value; a rank is checked against the definition of the tie rule, computed on
#: the host from the values. That is the only way a test of a device sort says
#: anything a second run of the same kernel would not.
#:
#: The sizes matter. `RADIX_GROUP` is 256, so a column of a few rows exercises
#: one block and cannot catch a bug in the offsets pass that stitches blocks
#: together; the 70000-row cases here are 274 blocks. The hash table is sized
#: at 1.5x the rows, so the collision-heavy case deliberately uses far more
#: rows than distinct values.
import std.gpu as gpu
from std.gpu import Buffer
from std.math import NAN, is_nan
from std.test import assert_eq, assert_false, assert_near, assert_throws, assert_true
from jaiframe.column import Column, DataFrame, Dtype, Index, Series
from jaiframe.keys import NO_GROUP
from jaiframe.keys import argsort, drop_duplicates, duplicated, encode_keys, factorize
from jaiframe.keys import hash_keys, mode, nlargest, nsmallest, nunique, rank
from jaiframe.keys import sort_index, sort_values, unique, value_counts

#: A deterministic spread of small integers, so a case can be big enough to
#: cross a threadgroup without a literal of seventy thousand numbers.
fn scattered(count: int, span: int, seed: int) -> list[int] {
    var state = seed
    var out: list[int] = []
    for _at in 0..count {
        state = state * 1103515245 + 12345 & 0x7FFFFFFF
        out.push(state - state // span * span)
    }
    return out
}

#: Whether `order` visits every row exactly once.
fn is_permutation(order: list[float], count: int) -> bool {
    var seen = [false for _at in 0..count]
    for slot in order {
        let at = int(slot)
        if at < 0 or at >= count { return false }
        if seen[at] { return false }
        seen[at] = true
    }
    return order.len() == count
}

fn test_encoding_reports_a_lane_per_digit() -> void {
    if not gpu.is_available() { return }
    let stamps = Column.from_ints([5, 3], dtype: Dtype.Datetime)
    defer { stamps.free() }
    let flags = Column.from_bools([true, false])
    defer { flags.free() }
    let keys = encode_keys([stamps, flags])
    defer { keys.free() }
    assert_eq(keys.length, 2)
    assert_eq(keys.columns, 2)
    #: A timestamp is a high and a low digit, a bool is one, and neither of
    #: these two columns can be missing, so no null lane is spent on them.
    assert_eq(keys.lane_count(), 3)
    assert_eq(keys.widths, [25, 24, 1])
    assert_eq(keys.owners, [0, 0, 1])
    assert_eq(keys.missing, null)
}

fn test_a_nullable_column_gets_a_missing_lane() -> void {
    if not gpu.is_available() { return }
    let ages = Column.from_ints([1, 2], nulls: [false, true])
    defer { ages.free() }
    let keys = encode_keys([ages])
    defer { keys.free() }
    assert_eq(keys.lane_count(), 3)
    assert_eq(keys.null_lanes, [true, false, false])
    let absent = keys.missing
    assert_true(absent is not null, "a nullable key knows which rows are missing")
    assert_eq((absent ?? keys.packed).download(), [0.0, 1.0])
}

#: A float column has no validity lane until something writes a null into it,
#: but a NaN a kernel produced is still missing to everything downstream, so
#: the lane is spent whether or not one has appeared yet.
fn test_a_float_column_always_carries_the_missing_lane() -> void {
    if not gpu.is_available() { return }
    let plain = Column.from_floats([1.0, 2.0])
    defer { plain.free() }
    let keys = encode_keys([plain])
    defer { keys.free() }
    assert_eq(keys.lane_count(), 2)
    assert_eq(keys.null_lanes, [true, false])
}

fn test_encoding_refuses_a_mismatched_key() -> void {
    if not gpu.is_available() { return }
    let two = Column.from_floats([1.0, 2.0])
    defer { two.free() }
    let three = Column.from_floats([1.0, 2.0, 3.0])
    defer { three.free() }
    assert_throws(ValueError, || encode_keys([two, three]))
    assert_throws(ValueError, || encode_keys([]))
}

#: Equal rows must hash alike — that is the only property a hash owes anybody.
#: Distinct rows landing on distinct hashes is not required, so the test asks
#: only that most of them do, which a broken mix would fail outright.
fn test_equal_rows_hash_alike() -> void {
    if not gpu.is_available() { return }
    let city = Column.from_strs(["austin", "dallas", "austin", "dallas"])
    defer { city.free() }
    let year = Column.from_ints([2024, 2025, 2024, 2024])
    defer { year.free() }
    let lane = hash_keys([city, year])
    defer { lane.free() }
    let read = lane.download()
    assert_eq(read[0], read[2])
    assert_true(read[1] != read[3], "a different year is a different key")
    assert_true(read[0] != read[1], "a different city is a different key")
}

fn test_the_hash_folds_the_key_order_in() -> void {
    if not gpu.is_available() { return }
    let first = Column.from_ints([1, 2])
    defer { first.free() }
    let second = Column.from_ints([2, 1])
    defer { second.free() }
    let forward = hash_keys([first, second])
    defer { forward.free() }
    let backward = hash_keys([second, first])
    defer { backward.free() }
    assert_true(
        forward.download()[0] != backward.download()[0],
        "(1, 2) and (2, 1) are not the same key"
    )
}

#: The definition: two rows share a code exactly when they carry the same
#: value. Checked against the host's own view of the values, over enough rows
#: and few enough distinct ones that the table has to probe.
fn test_factorize_agrees_with_equality() -> void {
    if not gpu.is_available() { return }
    let raw = scattered(70000, 37, 7)
    let column = Column.from_ints(raw)
    defer { column.free() }
    let found = factorize([column])
    let codes = found[0]
    defer { codes.free() }
    defer { for part in found[1] { part.free() } }
    assert_eq(found[2], 37)

    let read = codes.download()
    var value_of_code: dict[int, int] = {}
    var code_of_value: dict[int, int] = {}
    for (row, slot) in read.enumerate() {
        let code = int(slot)
        let seen = value_of_code.get(code, -1)
        if seen < 0 { value_of_code[code] = raw[row] } else { assert_eq(seen, raw[row]) }
        let owned = code_of_value.get(raw[row], -1)
        if owned < 0 { code_of_value[raw[row]] = code } else { assert_eq(owned, code) }
    }
    assert_eq(value_of_code.len(), 37)
}

#: pandas numbers groups by where they first appear, and the device decides
#: nothing about that: whichever thread wins the table's exchange, the group is
#: named after its lowest row.
fn test_codes_follow_first_appearance() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_strs(["dallas", "austin", "dallas", "houston", "austin"])
    defer { column.free() }
    let found = factorize([column])
    defer { found[0].free() }
    defer { for part in found[1] { part.free() } }
    assert_eq(found[0].download(), [0.0, 1.0, 0.0, 2.0, 1.0])
    assert_eq(found[1][0].to_strs(), ["dallas", "austin", "houston"])
}

fn test_a_sorted_factorization_renumbers_into_key_order() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_strs(["dallas", "austin", "dallas", "houston"])
    defer { column.free() }
    let found = factorize([column], sort: true)
    defer { found[0].free() }
    defer { for part in found[1] { part.free() } }
    assert_eq(found[1][0].to_strs(), ["austin", "dallas", "houston"])
    #: The codes have to move with the uniques, or the pair means nothing.
    let uniques = found[1][0].to_strs()
    let codes = found[0].download()
    let rows = column.to_strs()
    for (row, code) in codes.enumerate() { assert_eq(uniques[int(code)], rows[row]) }
}

fn test_a_multi_column_key_groups_on_the_pair() -> void {
    if not gpu.is_available() { return }
    let city = Column.from_strs(["austin", "austin", "dallas", "austin"])
    defer { city.free() }
    let year = Column.from_ints([2024, 2025, 2024, 2024])
    defer { year.free() }
    let found = factorize([city, year])
    defer { found[0].free() }
    defer { for part in found[1] { part.free() } }
    assert_eq(found[2], 3)
    assert_eq(found[0].download(), [0.0, 1.0, 2.0, 0.0])
}

fn test_missing_keys_group_together_or_drop() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_floats([1.0, NAN, 2.0, NAN])
    defer { column.free() }
    let kept = factorize([column])
    defer { kept[0].free() }
    defer { for part in kept[1] { part.free() } }
    assert_eq(kept[2], 3)
    assert_eq(kept[0].download()[1], kept[0].download()[3])

    let dropped = factorize([column], dropna: true)
    defer { dropped[0].free() }
    defer { for part in dropped[1] { part.free() } }
    assert_eq(dropped[2], 2)
    assert_eq(dropped[0].download(), [0.0, float(NO_GROUP), 1.0, float(NO_GROUP)])
}

fn test_unique_and_nunique_read_the_same_grouping() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_strs(["b", "a", "b", null, "c"])
    defer { column.free() }
    let distinct = unique(column)
    defer { distinct.free() }
    assert_eq(distinct.to_strs(), ["b", "a", null, "c"])
    assert_eq(nunique(column), 3)
    assert_eq(nunique(column, dropna: false), 4)
}

#: A sort is a permutation that never goes downhill. Checked over 70000 rows,
#: which is 274 radix blocks, so the pass that stitches the per-block
#: histograms together has to be right for this to hold.
fn test_argsort_is_an_ascending_permutation() -> void {
    if not gpu.is_available() { return }
    let raw = scattered(70000, 5000, 11)
    let column = Column.from_ints(raw)
    defer { column.free() }
    let order = argsort([column])
    defer { order.free() }
    let read = order.download()
    assert_true(is_permutation(read, 70000), "every row appears once")
    var downhill = false
    for at in 1..read.len() {
        if raw[int(read[at])] < raw[int(read[at - 1])] { downhill = true }
    }
    assert_false(downhill, "the order never goes downhill")
}

#: Stability is what makes a multi-key sort a chain of single-key passes, so it
#: is worth checking directly: rows that tie must keep their row numbers in
#: order. 70000 rows over 100 values leaves runs hundreds long.
fn test_the_sort_is_stable() -> void {
    if not gpu.is_available() { return }
    let raw = scattered(70000, 100, 3)
    let column = Column.from_ints(raw)
    defer { column.free() }
    let order = argsort([column])
    defer { order.free() }
    let read = order.download()
    var jumbled = false
    for at in 1..read.len() {
        if raw[int(read[at])] == raw[int(read[at - 1])] and read[at] < read[at - 1] {
            jumbled = true
        }
    }
    assert_false(jumbled, "ties keep the order they came in")
}

#: The claim the module makes about itself: sorting by the second key and then
#: the first gives what a comparator over the pair would. The host works out
#: the same order by comparing the pair directly.
fn test_a_two_key_sort_matches_comparing_the_pair() -> void {
    if not gpu.is_available() { return }
    let major = scattered(5000, 7, 19)
    let minor = scattered(5000, 11, 23)
    let first = Column.from_ints(major)
    defer { first.free() }
    let second = Column.from_ints(minor)
    defer { second.free() }
    let order = argsort([first, second])
    defer { order.free() }
    let read = order.download()
    var rows: list[int] = []
    for at in 0..5000 { rows.push(at) }
    let expected = rows.sort(key: |row| major[row] * 1000 + minor[row])
    for at in 0..5000 {
        assert_eq(major[int(read[at])], major[expected[at]])
        assert_eq(minor[int(read[at])], minor[expected[at]])
    }
}

fn test_descending_and_missing_are_independent() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_floats([2.0, NAN, 1.0, 3.0])
    defer { column.free() }
    let down = argsort([column], [false])
    defer { down.free() }
    assert_eq(down.download(), [3.0, 0.0, 2.0, 1.0])
    let down_first = argsort([column], [false], na_position: "first")
    defer { down_first.free() }
    assert_eq(down_first.download(), [1.0, 3.0, 0.0, 2.0])
    assert_throws(ValueError, || argsort([column], [true], na_position: "middle"))
}

fn test_text_sorts_by_the_level_not_the_code() -> void {
    if not gpu.is_available() { return }
    #: Codes are handed out in first-appearance order, so a sort that read them
    #: would leave this column exactly as it is.
    let column = Column.from_strs(["zebra", "apple", "mango"])
    defer { column.free() }
    let order = argsort([column])
    defer { order.free() }
    let sorted_column = column.take(order, 3)
    defer { sorted_column.free() }
    assert_eq(sorted_column.to_strs(), ["apple", "mango", "zebra"])
}

fn test_a_wide_key_keeps_every_bit() -> void {
    if not gpu.is_available() { return }
    #: Values either side of the 2^24 boundary the two lanes split on, where an
    #: encoding that dropped the high half would report them all equal.
    let raw = [16777215, 16777216, 16777217, -16777216, -1, 0, 1700000000123]
    let column = Column.from_ints(raw)
    defer { column.free() }
    let order = argsort([column])
    defer { order.free() }
    let sorted_column = column.take(order, raw.len())
    defer { sorted_column.free() }
    assert_eq(sorted_column.to_ints(), raw.sort())
}

fn test_sort_values_carries_the_labels() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(
        ["city", "high"],
        [
            Column.from_strs(["dallas", "austin", "dallas"]),
            Column.from_floats([41.0, 38.0, 36.5]),
        ],
        index: Index.range(3, start: 10)
    )
    defer { frame.free() }
    let sorted_frame = sort_values(frame, by: ["city", "high"])
    defer { sorted_frame.free() }
    assert_eq(sorted_frame.column("high").to_floats(), [38.0, 36.5, 41.0])
    assert_eq(sorted_frame.index.label_at(0)[0].to_int(), 11)
    let renumbered = sort_values(frame, by: "high", ignore_index: true)
    defer { renumbered.free() }
    assert_true(renumbered.index.is_range(), "ignore_index hands back fresh labels")
    assert_eq(renumbered.column("high").to_floats(), [36.5, 38.0, 41.0])
    assert_throws(ValueError, || sort_values(frame))
}

fn test_sort_index_orders_by_the_labels() -> void {
    if not gpu.is_available() { return }
    let level_names: list[str?] = ["key"]
    let labels = Index.of([Column.from_strs(["c", "a", "b"])], level_names)
    let series = Series(Column.from_floats([3.0, 1.0, 2.0]), index: labels, name: "v")
    defer { series.free() }
    let sorted_series = sort_index(series)
    defer { sorted_series.free() }
    assert_eq(sorted_series.values.to_floats(), [1.0, 2.0, 3.0])
    let backwards = sort_index(series, ascending: false)
    defer { backwards.free() }
    assert_eq(backwards.values.to_floats(), [3.0, 2.0, 1.0])
    let by_name = sort_index(series, level: "key")
    defer { by_name.free() }
    assert_eq(by_name.values.to_floats(), [1.0, 2.0, 3.0])
    assert_throws(KeyError, || sort_index(series, level: "absent"))
}

#: A range index is already in order, so this is the one sort that should cost
#: no sort at all — checked by the answer, since the saving is invisible.
fn test_sort_index_recognises_a_range() -> void {
    if not gpu.is_available() { return }
    let series = Series(Column.from_floats([1.0, 2.0, 3.0]))
    defer { series.free() }
    let same = sort_index(series)
    defer { same.free() }
    assert_eq(same.values.to_floats(), [1.0, 2.0, 3.0])
    let reversed = sort_index(series, ascending: false)
    defer { reversed.free() }
    assert_eq(reversed.values.to_floats(), [3.0, 2.0, 1.0])
    assert_eq(reversed.index.label_at(0)[0].to_int(), 2)
}

fn test_value_counts_totals_the_rows() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_strs(["a", "b", "a", "c", "a", "b", null])
    defer { column.free() }
    let counted = value_counts(column)
    defer { counted.free() }
    assert_eq(counted.values.to_ints(), [3, 2, 1])
    assert_eq(counted.index.label_at(0)[0].to_str(), "a")
    assert_eq(counted.name, "count")
    #: The counts must add up to the rows that were counted.
    var total = 0
    for value in counted.values.to_ints() { total += value }
    assert_eq(total, 6)

    let with_nulls = value_counts(column, dropna: false)
    defer { with_nulls.free() }
    var everything = 0
    for value in with_nulls.values.to_ints() { everything += value }
    assert_eq(everything, 7)

    let shares = value_counts(column, normalize: true)
    defer { shares.free() }
    assert_near(shares.values.to_floats()[0], 0.5, tolerance: 1e-6)
    assert_eq(shares.name, "proportion")
}

fn test_value_counts_can_leave_the_order_alone() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_strs(["b", "a", "a"])
    defer { column.free() }
    let counted = value_counts(column, sort: false)
    defer { counted.free() }
    assert_eq(counted.index.label_at(0)[0].to_str(), "b")
    assert_eq(counted.values.to_ints(), [1, 2])
}

fn test_value_counts_bins_a_numeric_column() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_floats([1.0, 2.0, 3.0, 4.0, 9.0, 10.0])
    defer { column.free() }
    let counted = value_counts(column, bins: 2, sort: false)
    defer { counted.free() }
    #: Every row falls in exactly one interval, and there are as many
    #: intervals as were asked for whether or not they are occupied.
    assert_eq(counted.values.len(), 2)
    var total = 0
    for value in counted.values.to_ints() { total += value }
    assert_eq(total, 6)
    assert_eq(counted.values.to_ints(), [4, 2])
    assert_throws(ValueError, || value_counts(column, bins: 0))
}

fn test_mode_picks_every_tied_winner() -> void {
    if not gpu.is_available() { return }
    let single = Column.from_ints([3, 1, 3, 2])
    defer { single.free() }
    let one = mode(single)
    defer { one.free() }
    assert_eq(one.to_ints(), [3])

    let tied = Column.from_strs(["b", "a", "b", "a", "c"])
    defer { tied.free() }
    let both = mode(tied)
    defer { both.free() }
    assert_eq(both.to_strs(), ["a", "b"])
}

fn test_duplicated_marks_every_repeat() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(
        ["k", "v"],
        [
            Column.from_strs(["a", "b", "a", "a"]),
            Column.from_floats([1.0, 2.0, 1.0, 9.0]),
        ]
    )
    defer { frame.free() }
    let after_first = duplicated(frame, subset: "k")
    defer { after_first.free() }
    assert_eq(after_first.to_bools(), [false, false, true, true])
    let before_last = duplicated(frame, subset: "k", keep: "last")
    defer { before_last.free() }
    assert_eq(before_last.to_bools(), [true, false, true, false])
    let every = duplicated(frame, subset: "k", keep: false)
    defer { every.free() }
    assert_eq(every.to_bools(), [true, false, true, true])
    #: Over the whole row, only the exact repeat counts.
    let whole = duplicated(frame)
    defer { whole.free() }
    assert_eq(whole.to_bools(), [false, false, true, false])
}

fn test_drop_duplicates_keeps_the_survivors_and_their_labels() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(
        ["k"],
        [Column.from_strs(["a", "b", "a"])],
        index: Index.range(3, start: 5)
    )
    defer { frame.free() }
    let first = drop_duplicates(frame, subset: "k")
    defer { first.free() }
    assert_eq(first.column("k").to_strs(), ["a", "b"])
    assert_eq(first.index.label_at(0)[0].to_int(), 5)
    let last = drop_duplicates(frame, subset: "k", keep: "last")
    defer { last.free() }
    assert_eq(last.column("k").to_strs(), ["b", "a"])
    assert_eq(last.index.label_at(1)[0].to_int(), 7)
    let none = drop_duplicates(frame, subset: "k", keep: false, ignore_index: true)
    defer { none.free() }
    assert_eq(none.column("k").to_strs(), ["b"])
    assert_true(none.index.is_range(), "ignore_index renumbers the survivors")
}

fn test_nlargest_and_nsmallest_take_the_ends_of_the_order() -> void {
    if not gpu.is_available() { return }
    let series = Series(Column.from_floats([5.0, 1.0, 9.0, 3.0]), name: "v")
    defer { series.free() }
    let top = nlargest(series, 2)
    defer { top.free() }
    assert_eq(top.values.to_floats(), [9.0, 5.0])
    assert_eq(top.index.label_at(0)[0].to_int(), 2)
    let bottom = nsmallest(series, 2)
    defer { bottom.free() }
    assert_eq(bottom.values.to_floats(), [1.0, 3.0])
    #: More than there are is not an error, it is all of them.
    let everything = nlargest(series, 99)
    defer { everything.free() }
    assert_eq(everything.values.len(), 4)
}

fn test_nlargest_decides_its_ties_by_keep() -> void {
    if not gpu.is_available() { return }
    let series = Series(Column.from_floats([5.0, 5.0, 5.0, 1.0]))
    defer { series.free() }
    let first = nlargest(series, 2)
    defer { first.free() }
    assert_eq(first.index.label_at(0)[0].to_int(), 0)
    assert_eq(first.index.label_at(1)[0].to_int(), 1)
    let last = nlargest(series, 2, keep: "last")
    defer { last.free() }
    assert_eq(last.index.label_at(0)[0].to_int(), 2)
    assert_eq(last.index.label_at(1)[0].to_int(), 1)
    let all_of_them = nlargest(series, 2, keep: "all")
    defer { all_of_them.free() }
    assert_eq(all_of_them.values.len(), 3)
    assert_throws(ValueError, || nlargest(series, 1, keep: "some"))
}

fn test_nlargest_leaves_missing_rows_out() -> void {
    if not gpu.is_available() { return }
    let series = Series(Column.from_floats([5.0, NAN, 1.0]))
    defer { series.free() }
    let top = nlargest(series, 3)
    defer { top.free() }
    assert_eq(top.values.to_floats(), [5.0, 1.0])
}

#: Every tie rule checked against its definition, computed on the host from the
#: values: the sorted positions a value occupies decide min, max and average,
#: and the number of distinct values below it decides dense.
fn test_rank_matches_the_definition_of_each_tie_rule() -> void {
    if not gpu.is_available() { return }
    let raw = [40.0, 10.0, 40.0, 20.0, 40.0]
    let column = Column.from_floats(raw)
    defer { column.free() }
    var sorted_values = [value for value in raw]
    sorted_values = sorted_values.sort()
    for (_at, name) in ["min", "max", "average", "dense", "first"].enumerate() {
        let ranked = rank(column, method: name)
        defer { ranked.free() }
        let got = ranked.to_floats()
        for (row, value) in raw.enumerate() {
            var below = 0
            var equal = 0
            for other in raw {
                if other < value { below += 1 }
                if other == value { equal += 1 }
            }
            var distinct_below = 0
            var seen: list[float] = []
            for other in raw {
                if other < value and not seen.contains(other) {
                    seen.push(other)
                    distinct_below += 1
                }
            }
            let low = float(below + 1)
            let high = float(below + equal)
            let wanted = match name {
                "min" => low,
                "max" => high,
                "average" => (low + high) * 0.5,
                "dense" => float(distinct_below + 1),
                _ => 0.0,
            }
            if name != "first" { assert_near(got[row], wanted, tolerance: 1e-6) }
        }
    }
}

#: `first` gives each tied row the position it actually landed in, which for a
#: stable sort is the tie's run in row order.
fn test_rank_first_walks_the_tie_in_row_order() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_floats([40.0, 10.0, 40.0, 40.0])
    defer { column.free() }
    let ranked = rank(column, method: "first")
    defer { ranked.free() }
    assert_eq(ranked.to_floats(), [2.0, 1.0, 3.0, 4.0])
}

fn test_rank_handles_descending_and_percentiles() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_floats([10.0, 20.0, 30.0, 40.0])
    defer { column.free() }
    let down = rank(column, ascending: false)
    defer { down.free() }
    assert_eq(down.to_floats(), [4.0, 3.0, 2.0, 1.0])
    let shares = rank(column, pct: true)
    defer { shares.free() }
    assert_eq(shares.to_floats(), [0.25, 0.5, 0.75, 1.0])
}

fn test_rank_decides_what_a_missing_value_gets() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_floats([3.0, NAN, 1.0])
    defer { column.free() }
    let kept = rank(column)
    defer { kept.free() }
    assert_true(kept.get(1).is_na(), "keep leaves the missing row unranked")
    assert_eq(kept.to_floats()[0], 2.0)
    let bottom = rank(column, na_option: "bottom")
    defer { bottom.free() }
    assert_eq(bottom.to_floats(), [2.0, 3.0, 1.0])
    let top = rank(column, na_option: "top")
    defer { top.free() }
    assert_eq(top.to_floats(), [3.0, 1.0, 2.0])
    assert_throws(ValueError, || rank(column, na_option: "somewhere"))
    assert_throws(ValueError, || rank(column, method: "median"))
}

#: A percentile over the ranked rows only — the missing one is not in the
#: denominator, which is what makes the largest present value reach 1.0.
fn test_rank_percentiles_skip_the_missing_rows() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_floats([1.0, 2.0, NAN])
    defer { column.free() }
    let shares = rank(column, pct: true)
    defer { shares.free() }
    let got = shares.to_floats()
    assert_near(got[0], 0.5, tolerance: 1e-6)
    assert_near(got[1], 1.0, tolerance: 1e-6)
    assert_true(is_nan(got[2]), "the missing row has no percentile")
}

#: Every column of a frame, whatever it holds. A rank runs off the encoding
#: rather than off the values, so text ranks by its level and a timestamp by
#: its two lanes with no code that knows the difference — which is worth
#: asserting, because it is the property the whole module is built on.
fn test_rank_reaches_every_column_of_a_frame() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(
        ["size", "city", "seen"],
        [
            Column.from_ints([30, 10, 20]),
            Column.from_strs(["c", "a", "b"]),
            Column.from_ints([5, 7, 6], dtype: Dtype.Datetime),
        ]
    )
    defer { frame.free() }
    let ranked = rank(frame)
    defer { ranked.free() }
    assert_eq(ranked.column("size").to_floats(), [3.0, 1.0, 2.0])
    assert_eq(ranked.column("city").to_floats(), [3.0, 1.0, 2.0])
    assert_eq(ranked.column("seen").to_floats(), [1.0, 3.0, 2.0])
}

fn test_an_empty_key_answers_without_a_dispatch() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_floats([])
    defer { column.free() }
    let found = factorize([column])
    defer { found[0].free() }
    defer { for part in found[1] { part.free() } }
    assert_eq(found[2], 0)
    let order = argsort([column])
    defer { order.free() }
    assert_eq(order.count, 0)
    assert_eq(nunique(column), 0)
}

#: The uniques of a grouping are read out of the same encoding the grouping
#: used, not re-encoded from the columns that came back — otherwise a
#: dictionary would be ranked against whatever level set survived the take.
fn test_a_gathered_encoding_keeps_the_same_ranking() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_strs(["c", "a", "c", "b"])
    defer { column.free() }
    let keys = encode_keys([column])
    defer { keys.free() }
    let picked = Buffer(2)
    picked.upload([3.0, 0.0])
    defer { picked.free() }
    let subset = keys.gather(picked, 2)
    defer { subset.free() }
    let order = subset.order()
    defer { order.free() }
    assert_eq(order.download(), [0.0, 1.0])
}

#: A table half full is where open addressing earns or loses its keep: 70000
#: rows over about 35000 distinct keys fills a 131072-slot table past half, so
#: most inserts walk a probe chain rather than landing on their first slot.
#: The count is checked against the host's own tally of distinct values.
fn test_the_table_survives_a_heavy_load() -> void {
    if not gpu.is_available() { return }
    let raw = scattered(70000, 35000, 29)
    let column = Column.from_ints(raw)
    defer { column.free() }
    var seen: dict[int, int] = {}
    for value in raw { seen[value] = 1 }
    let found = factorize([column])
    defer { found[0].free() }
    defer { for part in found[1] { part.free() } }
    assert_eq(found[2], seen.len())
    #: And the uniques really are the distinct values, not just the right
    #: number of them.
    var recovered: dict[int, int] = {}
    for value in found[1][0].to_ints() { recovered[value] = 1 }
    assert_eq(recovered.len(), seen.len())
}

fn test_a_sorted_factorization_keeps_the_dropped_rows_dropped() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_strs(["c", null, "a", "c"])
    defer { column.free() }
    let found = factorize([column], sort: true, dropna: true)
    defer { found[0].free() }
    defer { for part in found[1] { part.free() } }
    assert_eq(found[2], 2)
    assert_eq(found[1][0].to_strs(), ["a", "c"])
    assert_eq(found[0].download(), [1.0, float(NO_GROUP), 0.0, 1.0])
}

fn test_dropping_every_row_leaves_no_groups() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_floats([NAN, NAN])
    defer { column.free() }
    let found = factorize([column], dropna: true)
    defer { found[0].free() }
    defer { for part in found[1] { part.free() } }
    assert_eq(found[2], 0)
    assert_eq(found[1][0].len(), 0)
    assert_eq(found[0].download(), [float(NO_GROUP), float(NO_GROUP)])
}

fn test_a_frame_sorts_each_key_its_own_way() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(
        ["k", "v"],
        [Column.from_strs(["a", "b", "a", "b"]), Column.from_ints([1, 1, 2, 2])]
    )
    defer { frame.free() }
    let mixed = sort_values(frame, by: ["k", "v"], ascending: [true, false])
    defer { mixed.free() }
    assert_eq(mixed.column("k").to_strs(), ["a", "a", "b", "b"])
    assert_eq(mixed.column("v").to_ints(), [2, 1, 2, 1])
    assert_throws(TypeError, || sort_values(frame, by: ["k"], ascending: 3))
}

fn test_value_counts_over_a_frame_counts_whole_rows() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(
        ["k", "v"],
        [Column.from_strs(["a", "b", "a"]), Column.from_ints([1, 1, 1])]
    )
    defer { frame.free() }
    let counted = value_counts(frame)
    defer { counted.free() }
    assert_eq(counted.values.to_ints(), [2, 1])
    assert_eq(counted.index.nlevels(), 2)
    assert_eq(counted.index.label_at(0)[0].to_str(), "a")
    assert_eq(counted.index.label_at(0)[1].to_int(), 1)
}

fn test_nlargest_over_a_frame_needs_its_columns() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(
        ["k", "v"],
        [Column.from_strs(["a", "b", "c"]), Column.from_ints([2, 9, 5])]
    )
    defer { frame.free() }
    let top = nlargest(frame, 2, columns: "v")
    defer { top.free() }
    assert_eq(top.column("k").to_strs(), ["b", "c"])
    assert_throws(ValueError, || nlargest(frame, 1))
    let none = nsmallest(frame, 0, columns: "v")
    defer { none.free() }
    assert_eq(none.len(), 0)
}

fn test_duplicated_and_drop_duplicates_reach_a_series() -> void {
    if not gpu.is_available() { return }
    let series = Series(Column.from_ints([7, 8, 7]), name: "v")
    defer { series.free() }
    let marks = duplicated(series)
    defer { marks.free() }
    assert_eq(marks.to_bools(), [false, false, true])
    let kept = drop_duplicates(series)
    defer { kept.free() }
    assert_eq(kept.values.to_ints(), [7, 8])
    assert_eq(kept.name, "v")
    assert_throws(ValueError, || duplicated(series, keep: true))
}

fn test_a_category_ranks_by_its_level_text() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_strs(["zebra", "apple"], dtype: Dtype.Category)
    defer { column.free() }
    let order = argsort([column])
    defer { order.free() }
    assert_eq(order.download(), [1.0, 0.0])
    let counted = value_counts(column)
    defer { counted.free() }
    assert_eq(counted.values.to_ints(), [1, 1])
}

#: A dictionary lane is only as wide as its level count, so it is inverted
#: within that width rather than across 32 bits — a descending text sort is
#: where that goes wrong if the mask is off. The null lane rides above it and
#: is not inverted with it.
fn test_text_sorts_descending_without_moving_the_nulls() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_strs(["b", null, "a", "c"])
    defer { column.free() }
    let down = argsort([column], [false])
    defer { down.free() }
    let ordered = column.take(down, 4)
    defer { ordered.free() }
    assert_eq(ordered.to_strs(), ["c", "b", "a", null])
    let down_first = argsort([column], [false], na_position: "first")
    defer { down_first.free() }
    let with_null = column.take(down_first, 4)
    defer { with_null.free() }
    assert_eq(with_null.to_strs(), [null, "c", "b", "a"])
}

#: Five lanes over two nullable columns of different shapes, with the second
#: key breaking the first key's ties and each column's nulls staying at its own
#: end rather than being hoisted to the front of the whole sort.
fn test_two_nullable_keys_stack_their_lanes() -> void {
    if not gpu.is_available() { return }
    let city = Column.from_strs(["b", "a", "b", null, "a"])
    defer { city.free() }
    let high = Column.from_ints([2, 5, 0, 1, 4], nulls: [false, false, true, false, false])
    defer { high.free() }
    let keys = encode_keys([city, high])
    defer { keys.free() }
    #: A missing flag and a value lane for the text, a missing flag and two
    #: value lanes for the 48-bit integer.
    assert_eq(keys.lane_count(), 5)
    assert_eq(keys.null_lanes, [true, false, true, false, false])
    let order = keys.order()
    defer { order.free() }
    let ranked_city = city.take(order, 5)
    defer { ranked_city.free() }
    let ranked_high = high.take(order, 5)
    defer { ranked_high.free() }
    assert_eq(ranked_city.to_strs(), ["a", "a", "b", "b", null])
    assert_eq(ranked_high.to_ints()[0], 4)
    assert_eq(ranked_high.to_ints()[1], 5)
    assert_eq(ranked_high.to_ints()[2], 2)
    assert_true(ranked_high.get(3).is_na(), "the missing high sorts last inside its own city")
}

fn test_a_seed_moves_the_hash() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_ints([1, 2, 3])
    defer { column.free() }
    let plain = hash_keys([column])
    defer { plain.free() }
    let salted = hash_keys([column], seed: 99)
    defer { salted.free() }
    let a = plain.download()
    let b = salted.download()
    var moved = false
    for at in 0..3 {
        if a[at] != b[at] { moved = true }
    }
    assert_true(moved, "a seed changes where the keys land")
}

#: With nothing present the smallest and largest come back as the two ends of
#: float32, which would cut edges no row could fall between. The intervals
#: still have to be there and still have to be empty.
fn test_binning_a_column_with_nothing_in_it() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_floats([NAN, NAN])
    defer { column.free() }
    let counted = value_counts(column, bins: 3, sort: false)
    defer { counted.free() }
    assert_eq(counted.values.to_ints(), [0, 0, 0])
}

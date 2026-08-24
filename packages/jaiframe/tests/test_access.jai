#: Selection: `iloc`, `loc`, masks, `query`, and item assignment.
#:
#: Nearly every test here computes the answer a second way and compares. A
#: slice is checked against a brute-force walk rather than against the closed
#: form the module uses; `isin` and `between` against the host predicate they
#: are named for; `query` against the vectorised mask the module claims is
#: equivalent to it. Two of them cross the 256-row threadgroup boundary the
#: scan and the `isin` tile are blocked into, which is where every bug in one
#: of those lives.
import std.gpu as gpu
from std.math import NAN, is_nan
from std.test import assert_eq, assert_false, assert_near, assert_throws, assert_true
from jaiframe.column import Column, DataFrame, Dtype, Index, Series, Value
from jaiframe.column import NA
from jaiframe.access import Labels, Pick, Row, SelectKind
from jaiframe.access import at, at_series, between, drop_rows, filter_names, filter_rows
from jaiframe.access import filter_rows_series, first, frame_isin, frame_mask, frame_where
from jaiframe.access import head, head_series, iat, iat_series, iloc, iloc_series
from jaiframe.access import insert_rows, isin, last, loc, loc_series, mask, nth
from jaiframe.access import query, query_flags, query_mask, sample, sample_series
from jaiframe.access import set_column_values, set_iloc, set_loc, tail, take
from jaiframe.access import take_series, where

#: A name list an `Index` will take: its names are optional, one per level.
fn level_names(names: list[str]) -> list[str?] {
    var out: list[str?] = []
    for name in names { out.push(name) }
    return out
}

#: A frame whose one column holds each row's own position, built on the device
#: so a tall one costs a dispatch rather than a host list.
fn ramp_frame(length: int) -> DataFrame {
    let levels = Index.range(length).materialize()
    return DataFrame(["row"], [levels[0]])
}

#: The positions a slice names, walked one step at a time.
#:
#: The definition of a slice rather than the arithmetic the module uses, so
#: that a wrong count and a wrong stride cannot agree with each other. Every
#: case fed to it keeps both ends inside the frame, since the walk knows
#: nothing about the negative-index and clamping rules a bound goes through
#: first — those are checked separately.
fn walked_span(start: int, stop: int, step: int) -> list[int] {
    var out: list[int] = []
    var at = start
    while step > 0 and at < stop or step < 0 and at > stop {
        out.push(at)
        at += step
    }
    return out
}

fn test_a_slice_names_what_walking_it_names() -> void {
    if not gpu.is_available() { return }
    let length = 11
    let frame = ramp_frame(length)
    defer { frame.free() }
    let cases = [[0, 11, 1], [2, 9, 3], [1, 11, 4], [3, 3, 1], [9, 1, -2], [10, 4, -1]]
    for bounds in cases {
        let picked = iloc(frame, Pick.span(bounds[0], bounds[1], bounds[2]))
        defer { picked.free() }
        assert_eq(picked.column("row").to_ints(), walked_span(bounds[0], bounds[1], bounds[2]))
    }
    let open = iloc(frame, Pick.span(step: -1))
    defer { open.free() }
    assert_eq(open.column("row").to_ints(), walked_span(10, -1, -1))
    assert_throws(ValueError, || Pick.span(0, 4, 0))
}

fn test_a_slice_clamps_ends_that_run_off_the_frame() -> void {
    if not gpu.is_available() { return }
    let frame = ramp_frame(5)
    defer { frame.free() }
    let over = iloc(frame, Pick.span(-99, 99))
    defer { over.free() }
    assert_eq(over.column("row").to_ints(), [0, 1, 2, 3, 4])
    let back = iloc(frame, Pick.span(-2, null))
    defer { back.free() }
    assert_eq(back.column("row").to_ints(), [3, 4])
    let empty = iloc(frame, Pick.span(4, 1))
    defer { empty.free() }
    assert_eq(empty.len(), 0)
}

fn test_iloc_takes_a_scalar_a_list_and_a_mask() -> void {
    if not gpu.is_available() { return }
    let frame = ramp_frame(6)
    defer { frame.free() }
    let one = iloc(frame, Pick.one_at(-2))
    defer { one.free() }
    assert_eq(one.column("row").to_ints(), [4])
    let listed = iloc(frame, Pick.of([5, 0, 5]))
    defer { listed.free() }
    assert_eq(listed.column("row").to_ints(), [5, 0, 5])
    let flags = Column.from_bools([true, false, true, false, false, true])
    defer { flags.free() }
    let marked = iloc(frame, Pick.masked(flags))
    defer { marked.free() }
    assert_eq(marked.column("row").to_ints(), [0, 2, 5])
    assert_throws(IndexError, || iloc(frame, Pick.one_at(6)))
}

#: The mask path is one flag pass, one scan and one scatter. The scalar and
#: the four-at-a-time flag kernels are chosen by whether the length divides by
#: four, so both lengths have to answer the same way.
fn test_a_mask_agrees_with_the_host_at_both_flag_widths() -> void {
    if not gpu.is_available() { return }
    for length in [8, 9] {
        let frame = ramp_frame(length)
        defer { frame.free() }
        var flags: list[bool] = []
        for at in 0..length { flags.push(at % 3 == 1) }
        let mask_column = Column.from_bools(flags)
        defer { mask_column.free() }
        var expected: list[int] = []
        for (at, kept) in flags.enumerate() {
            if kept { expected.push(at) }
        }
        let picked = filter_rows(frame, mask_column)
        defer { picked.free() }
        assert_eq(picked.column("row").to_ints(), expected)
    }
}

#: A null flag drops its row, the way `Column.compress` treats one.
fn test_a_null_flag_drops_its_row() -> void {
    if not gpu.is_available() { return }
    let frame = ramp_frame(4)
    defer { frame.free() }
    let flags = Column.from_bools([true, true, false, true], nulls: [false, true, false, false])
    defer { flags.free() }
    let picked = filter_rows(frame, flags)
    defer { picked.free() }
    assert_eq(picked.column("row").to_ints(), [0, 3])
    assert_throws(TypeError, || filter_rows(frame, frame.column("row")))
}

#: 70000 rows is two levels of the block scan behind `prefix_sum`, which is
#: where a carry bug lives. The count is known in closed form, so nothing here
#: leans on the module having got the same answer twice.
fn test_the_mask_scan_carries_across_threadgroups() -> void {
    if not gpu.is_available() { return }
    let frame = ramp_frame(70000)
    defer { frame.free() }
    let flags = between(frame.column("row"), Value.of_int(1000), Value.of_int(60000))
    defer { flags.free() }
    let picked = filter_rows(frame, flags)
    defer { picked.free() }
    assert_eq(picked.len(), 59001)
    assert_eq(picked.column("row").get(0).to_int(), 1000)
    assert_eq(picked.column("row").get(-1).to_int(), 60000)
}

fn test_iloc_selects_columns_as_well_as_rows() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(
        ["a", "b", "c"],
        [Column.from_ints([1, 2]), Column.from_ints([3, 4]), Column.from_ints([5, 6])]
    )
    defer { frame.free() }
    let picked = iloc(frame, Pick.every(), Pick.of([2, 0]))
    defer { picked.free() }
    assert_eq(picked.names, ["c", "a"])
    assert_eq(picked.column("c").to_ints(), [5, 6])
    let sliced = iloc(frame, Pick.one_at(1), Pick.span(1, 3))
    defer { sliced.free() }
    assert_eq(sliced.names, ["b", "c"])
    assert_eq(sliced.column("b").to_ints(), [4])
}

fn labelled_frame() -> DataFrame {
    return DataFrame(
        ["v"],
        [Column.from_ints([10, 20, 30, 40])],
        index: Index.of([Column.from_strs(["a", "b", "a", "c"])], level_names(["k"]))
    )
}

#: A repeated label selects every row it names, which is what separates `loc`
#: from a lookup. The expectation is scanned off the index's own strings.
fn test_loc_finds_every_row_a_label_names() -> void {
    if not gpu.is_available() { return }
    let frame = labelled_frame()
    defer { frame.free() }
    var expected: list[int] = []
    for (at, name) in frame.index.levels[0].to_strs().enumerate() {
        if name == "a" { expected.push(10 + at * 10) }
    }
    let picked = loc(frame, Labels.one_of(Value.of_str("a")))
    defer { picked.free() }
    assert_eq(picked.column("v").to_ints(), expected)
    let several = loc(frame, Labels.of([Value.of_str("c"), Value.of_str("b")]))
    defer { several.free() }
    assert_eq(several.column("v").to_ints(), [40, 20])
    assert_throws(KeyError, || loc(frame, Labels.one_of(Value.of_str("z"))))
}

#: pandas closes both ends of a label slice, and the ends are located by first
#: and last occurrence — so "a" to "b" reaches the second "a" at row 2.
fn test_a_label_slice_closes_both_ends() -> void {
    if not gpu.is_available() { return }
    let frame = labelled_frame()
    defer { frame.free() }
    let span = loc(frame, Labels.span(Value.of_str("a"), Value.of_str("b")))
    defer { span.free() }
    assert_eq(span.column("v").to_ints(), [10, 20])
    let tail_span = loc(frame, Labels.span(Value.of_str("b"), null))
    defer { tail_span.free() }
    assert_eq(tail_span.column("v").to_ints(), [20, 30, 40])
}

#: A range index answers by arithmetic, so a label off the ramp is a miss
#: rather than a rounding.
fn test_loc_reads_a_range_index_without_materialising_it() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(
        ["v"],
        [Column.from_ints([7, 8, 9])],
        index: Index.range(3, start: 10, step: 5)
    )
    defer { frame.free() }
    let picked = loc(frame, Labels.of([Value.of_int(20), Value.of_int(10)]))
    defer { picked.free() }
    assert_eq(picked.column("v").to_ints(), [9, 7])
    assert_throws(KeyError, || loc(frame, Labels.one_of(Value.of_int(12))))
}

fn test_loc_matches_every_level_of_a_multi_index() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(
        ["v"],
        [Column.from_ints([1, 2, 3, 4])],
        index: Index.of(
            [Column.from_strs(["x", "x", "y", "y"]), Column.from_ints([1, 2, 1, 2])],
            level_names(["outer", "inner"])
        )
    )
    defer { frame.free() }
    let picked = loc(frame, Labels.tuples([[Value.of_str("y"), Value.of_int(1)]]))
    defer { picked.free() }
    assert_eq(picked.column("v").to_ints(), [3])
    assert_throws(KeyError, || loc(frame, Labels.tuples([[Value.of_str("y"), Value.of_int(9)]])))
}

#: A unique index resolves a whole list of labels through one hash table
#: instead of one scan apiece, so this checks the answer the table gives
#: against the one a host dictionary gives. The asks are scrambled and repeat
#: themselves, because `loc` owes the caller their order and their repeats,
#: not the index's; 500 rows puts the probe well past the 256 the scan it
#: replaced was blocked into.
fn test_loc_over_a_unique_index_answers_what_a_host_lookup_does() -> void {
    if not gpu.is_available() { return }
    let length = 500
    var names: list[str?] = []
    var values: list[int] = []
    for at in 0..length {
        names.push(f"k{at}")
        values.push(at * 3)
    }
    let frame = DataFrame(
        ["v"],
        [Column.from_ints(values)],
        index: Index.of([Column.from_strs(names)], level_names(["k"]))
    )
    defer { frame.free() }
    var seat: dict[str, int] = {}
    for (at, name) in frame.index.levels[0].to_strs().enumerate() { seat[name ?? ""] = at }
    var asked: list[Value] = []
    var expected: list[int] = []
    for step in 0..40 {
        let name = f"k{(step * 37) % length}"
        asked.push(Value.of_str(name))
        expected.push(seat[name] * 3)
        if step % 9 == 0 {
            asked.push(Value.of_str(name))
            expected.push(seat[name] * 3)
        }
    }
    let picked = loc(frame, Labels.of(asked))
    defer { picked.free() }
    assert_eq(picked.column("v").to_ints(), expected)
    assert_throws(KeyError, || loc(frame, Labels.of([Value.of_str("k7"), Value.of_str("nope")])))
}

#: The two label paths — the hash table for a unique index, a scan per label
#: for one that repeats — have to agree wherever both are defined. Duplicating
#: every row of a frame leaves each label naming two rows, so the scanned
#: answer is the probed one with each position doubled, and any disagreement
#: between the paths shows up as a row in the wrong place.
fn test_both_label_paths_agree_on_the_rows_they_name() -> void {
    if not gpu.is_available() { return }
    let names = ["p", "q", "r", "s", "t"]
    var doubled: list[str?] = []
    var twice: list[int] = []
    for (at, name) in names.enumerate() {
        doubled.push(name)
        doubled.push(name)
        twice.push(at)
        twice.push(at)
    }
    let unique = DataFrame(
        ["v"],
        [Column.from_ints([at for at in 0..names.len()])],
        index: Index.of([Column.from_strs([name for name in names])], level_names(["k"]))
    )
    defer { unique.free() }
    let repeated = DataFrame(
        ["v"],
        [Column.from_ints(twice)],
        index: Index.of([Column.from_strs(doubled)], level_names(["k"]))
    )
    defer { repeated.free() }
    let asked = [Value.of_str("s"), Value.of_str("p"), Value.of_str("r")]
    let probed = loc(unique, Labels.of(asked))
    defer { probed.free() }
    let scanned = loc(repeated, Labels.of(asked))
    defer { scanned.free() }
    var expected: list[int] = []
    for value in probed.column("v").to_ints() {
        expected.push(value)
        expected.push(value)
    }
    assert_eq(scanned.column("v").to_ints(), expected)
}

fn test_at_and_iat_reach_one_cell() -> void {
    if not gpu.is_available() { return }
    let frame = labelled_frame()
    defer { frame.free() }
    assert_eq(iat(frame, 2, 0).to_int(), 30)
    assert_eq(iat(frame, -1, -1).to_int(), 40)
    assert_eq(at(frame, Value.of_str("b"), "v").to_int(), 20)
    assert_throws(KeyError, || at(frame, Value.of_str("a"), "v"))
    assert_throws(IndexError, || iat(frame, 9, 0))
}

fn test_a_series_selects_the_same_five_shapes() -> void {
    if not gpu.is_available() { return }
    let series = Series(
        Column.from_ints([1, 2, 3, 4]),
        index: Index.of([Column.from_strs(["a", "b", "c", "d"])]),
        name: "v"
    )
    defer { series.free() }
    let sliced = iloc_series(series, Pick.span(1, 4, 2))
    defer { sliced.free() }
    assert_eq(sliced.values.to_ints(), [2, 4])
    let byname = loc_series(series, Labels.of([Value.of_str("d")]))
    defer { byname.free() }
    assert_eq(byname.values.to_ints(), [4])
    assert_eq(iat_series(series, -1).to_int(), 4)
    assert_eq(at_series(series, Value.of_str("c")).to_int(), 3)
    let listed = take_series(series, [3, 3, 0])
    defer { listed.free() }
    assert_eq(listed.values.to_ints(), [4, 4, 1])
    let flags = Column.from_bools([false, true, true, false])
    defer { flags.free() }
    let kept = filter_rows_series(series, flags)
    defer { kept.free() }
    assert_eq(kept.values.to_ints(), [2, 3])
    assert_eq(kept.index.label_at(0)[0].to_str(), "b")
    let short = head_series(series, 2)
    defer { short.free() }
    assert_eq(short.values.to_ints(), [1, 2])
}

fn test_head_and_tail_read_a_negative_count_as_a_trim() -> void {
    if not gpu.is_available() { return }
    let frame = ramp_frame(6)
    defer { frame.free() }
    let front = head(frame, 2)
    defer { front.free() }
    assert_eq(front.column("row").to_ints(), [0, 1])
    let back = tail(frame, 2)
    defer { back.free() }
    assert_eq(back.column("row").to_ints(), [4, 5])
    let trimmed = head(frame, -4)
    defer { trimmed.free() }
    assert_eq(trimmed.column("row").to_ints(), [0, 1])
    let trimmed_back = tail(frame, -4)
    defer { trimmed_back.free() }
    assert_eq(trimmed_back.column("row").to_ints(), [4, 5])
    let over = head(frame, 99)
    defer { over.free() }
    assert_eq(over.len(), 6)
}

fn test_nth_first_and_last_keep_the_row_label() -> void {
    if not gpu.is_available() { return }
    let frame = labelled_frame()
    defer { frame.free() }
    let middle = nth(frame, 1)
    defer { middle.free() }
    assert_eq(middle.column("v").to_ints(), [20])
    assert_eq(middle.index.label_at(0)[0].to_str(), "b")
    let front = first(frame)
    defer { front.free() }
    assert_eq(front.column("v").to_ints(), [10])
    let back = last(frame)
    defer { back.free() }
    assert_eq(back.column("v").to_ints(), [40])
}

fn test_take_repeats_and_counts_from_the_end() -> void {
    if not gpu.is_available() { return }
    let frame = ramp_frame(4)
    defer { frame.free() }
    let picked = take(frame, [-1, 0, -1])
    defer { picked.free() }
    assert_eq(picked.column("row").to_ints(), [3, 0, 3])
    let widened = DataFrame(["a", "b"], [Column.from_ints([1]), Column.from_ints([2])])
    defer { widened.free() }
    let narrowed = take(widened, [1], axis: 1)
    assert_eq(narrowed.names, ["b"])
    assert_throws(ValueError, || take(frame, [0], axis: 2))
}

#: The wanted values are staged 256 at a time, so 300 of them is two tiles and
#: the second one has to see the same rows as the first. The expectation is a
#: host membership test over the column's own values.
fn test_isin_agrees_with_a_host_membership_test() -> void {
    if not gpu.is_available() { return }
    let length = 2000
    let frame = ramp_frame(length)
    defer { frame.free() }
    var wanted: list[Value] = []
    var wanted_set: dict[int, bool] = {}
    for step in 0..300 {
        let value = step * 7
        wanted.push(Value.of_int(value))
        wanted_set[value] = true
    }
    let flags = isin(frame.column("row"), wanted)
    defer { flags.free() }
    var expected: list[bool] = []
    for value in frame.column("row").to_ints() { expected.push(wanted_set.get(value, false)) }
    assert_eq(flags.to_bools(), expected)
}

fn test_isin_reads_text_off_the_levels() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_strs(["austin", "dallas", null, "austin"])
    defer { column.free() }
    let flags = isin(column, [Value.of_str("austin"), Value.of_str("houston")])
    defer { flags.free() }
    assert_eq(flags.to_bools(), [true, false, false, true])
    let with_nulls = isin(column, [NA])
    defer { with_nulls.free() }
    assert_eq(with_nulls.to_bools(), [false, false, true, false])
}

fn test_isin_over_a_frame_answers_per_column() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(["a", "b"], [Column.from_ints([1, 2]), Column.from_ints([2, 3])])
    defer { frame.free() }
    let flags = frame_isin(frame, [Value.of_int(2)])
    defer { flags.free() }
    assert_eq(flags.column("a").to_bools(), [false, true])
    assert_eq(flags.column("b").to_bools(), [true, false])
}

#: All four of pandas' `inclusive` settings, each checked against the host
#: comparison it is named for.
fn test_between_matches_the_comparison_it_names() -> void {
    if not gpu.is_available() { return }
    let values = [1.0, 2.0, 3.0, 4.0, NAN]
    let column = Column.from_floats(values)
    defer { column.free() }
    for setting in ["both", "left", "right", "neither"] {
        let flags = between(column, Value.of_float(2.0), Value.of_float(4.0), inclusive: setting)
        defer { flags.free() }
        var expected: list[bool] = []
        for value in values {
            if is_nan(value) {
                expected.push(false)
                continue
            }
            let above = if setting == "both" or setting == "left" {
                value >= 2.0
            } else {
                value > 2.0
            }
            let below = if setting == "both" or setting == "right" {
                value <= 4.0
            } else {
                value < 4.0
            }
            expected.push(above and below)
        }
        assert_eq(flags.to_bools(), expected)
    }
    assert_throws(ValueError, || between(column, NA, Value.of_float(1.0)))
    assert_throws(
        ValueError,
        || between(column, Value.of_float(0.0), Value.of_float(1.0), inclusive: "inside")
    )
}

#: Text compares by its spelling, not by its dictionary code — the levels here
#: are numbered in first-appearance order, which is the reverse of alphabetical.
fn test_between_over_text_compares_the_strings() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_strs(["delta", "alpha", "charlie", "bravo"])
    defer { column.free() }
    let flags = between(column, Value.of_str("alpha"), Value.of_str("charlie"))
    defer { flags.free() }
    assert_eq(flags.to_bools(), [false, true, true, true])
}

#: A timestamp range is past 2^24 milliseconds, so a bound that crossed as one
#: float would have rounded and taken the wrong rows with it.
fn test_between_keeps_all_48_bits_of_a_timestamp() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_ints(
        [1700000000000, 1700000000001, 1700000000002],
        dtype: Dtype.Datetime
    )
    defer { column.free() }
    let flags = between(column, Value.of_datetime(1700000000001), Value.of_datetime(1700000000002))
    defer { flags.free() }
    assert_eq(flags.to_bools(), [false, true, true])
}

#: `where` and `mask` are complements: every row keeps its own value in
#: exactly one of them.
fn test_where_and_mask_are_complements() -> void {
    if not gpu.is_available() { return }
    let values = [1.0, 2.0, 3.0, 4.0]
    let column = Column.from_floats(values)
    defer { column.free() }
    let cond = Column.from_bools([true, false, true, false])
    defer { cond.free() }
    let kept = where(column, cond, Value.of_float(-1.0))
    defer { kept.free() }
    let flipped = mask(column, cond, Value.of_float(-1.0))
    defer { flipped.free() }
    let kept_values = kept.to_floats()
    let flipped_values = flipped.to_floats()
    for (at, value) in values.enumerate() {
        let held_by_where = kept_values[at] == value
        let held_by_mask = flipped_values[at] == value
        assert_true(held_by_where != held_by_mask, f"row {at} is kept by exactly one")
        assert_near(if held_by_where {
            flipped_values[at]
        } else {
            kept_values[at]
        }, -1.0, tolerance: 1e-6)
    }
}

fn test_where_can_replace_with_a_null() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_ints([10, 20, 30])
    defer { column.free() }
    let cond = Column.from_bools([true, false, true])
    defer { cond.free() }
    let kept = where(column, cond)
    defer { kept.free() }
    assert_eq(kept.null_mask(), [false, true, false])
    assert_eq(kept.get(2).to_int(), 30)
}

fn test_where_reads_a_null_condition_as_false() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_floats([1.0, 2.0])
    defer { column.free() }
    let cond = Column.from_bools([true, true], nulls: [false, true])
    defer { cond.free() }
    let kept = where(column, cond, Value.of_float(0.0))
    defer { kept.free() }
    assert_eq(kept.to_floats(), [1.0, 0.0])
}

fn test_where_over_a_frame_shares_one_condition() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(
        ["a", "b"],
        [Column.from_floats([1.0, 2.0]), Column.from_floats([3.0, 4.0])]
    )
    defer { frame.free() }
    let cond = Column.from_bools([true, false])
    defer { cond.free() }
    let kept = frame_where(frame, cond, Value.of_float(0.0))
    defer { kept.free() }
    assert_eq(kept.column("a").to_floats(), [1.0, 0.0])
    assert_eq(kept.column("b").to_floats(), [3.0, 0.0])
    let flipped = frame_mask(frame, cond, Value.of_float(0.0))
    defer { flipped.free() }
    assert_eq(flipped.column("a").to_floats(), [0.0, 2.0])
}

fn adult(row: Row) -> bool {
    let age = row.integer("age")
    return age >= 18 and age <= 65
}

#: The claim the module makes about `query` — that the callback and a
#: vectorised mask select the same rows — checked directly.
fn test_query_and_query_mask_choose_the_same_rows() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(
        ["age", "city"],
        [
            Column.from_ints([17, 34, 66, 20, 65]),
            Column.from_strs(["austin", "dallas", "austin", null, "houston"]),
        ]
    )
    defer { frame.free() }
    let walked = query(frame, adult)
    defer { walked.free() }
    let vectorised = query_mask(
        frame,
        between(frame.column("age"), Value.of_int(18), Value.of_int(65))
    )
    defer { vectorised.free() }
    assert_eq(walked.column("age").to_ints(), [34, 20, 65])
    assert_eq(walked.column("age").to_ints(), vectorised.column("age").to_ints())
    assert_eq(walked.column("city").to_strs(), vectorised.column("city").to_strs())
}

fn test_a_row_view_reads_every_dtype() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(
        ["n", "i", "t", "b"],
        [
            Column.from_floats([1.5, NAN]),
            Column.from_ints([7, 8]),
            Column.from_strs(["austin", null]),
            Column.from_bools([true, false]),
        ]
    )
    defer { frame.free() }
    let row = Row.over(frame)
    row.at = 0
    assert_near(row.number("n"), 1.5, tolerance: 1e-6)
    assert_eq(row.integer("i"), 7)
    assert_eq(row.text("t"), "austin")
    assert_true(row.value("b").to_bool(), "the bool cell is true")
    assert_false(row.is_na("n"), "row 0 is present")
    row.at = 1
    assert_true(row.is_na("n"), "the NaN row is missing")
    assert_true(row.is_na("t"), "the null string is missing")
    assert_eq(row.text("t"), null)
    assert_eq(row.dtype("i"), Dtype.Int64)
    assert_throws(KeyError, || row.number("absent"))
}

fn test_query_flags_hands_back_a_reusable_mask() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(["age"], [Column.from_ints([10, 30, 70])])
    defer { frame.free() }
    let flags = query_flags(frame, adult)
    defer { flags.free() }
    assert_eq(flags.dtype, Dtype.Bool)
    assert_eq(flags.to_bools(), [false, true, false])
}

#: A draw without replacement has to be distinct and inside the frame, and a
#: seeded one has to repeat. The count is checked on both sides of the
#: half-the-frame line, where the module switches from drawing the kept rows
#: to drawing the dropped ones.
fn test_sample_draws_distinct_rows_and_repeats_under_a_seed() -> void {
    if not gpu.is_available() { return }
    let length = 40
    let frame = ramp_frame(length)
    defer { frame.free() }
    for wanted in [5, 30] {
        let drawn = sample(frame, n: wanted, seed: 2026)
        defer { drawn.free() }
        assert_eq(drawn.len(), wanted)
        var seen: dict[int, bool] = {}
        for row in drawn.column("row").to_ints() {
            assert_true(row >= 0 and row < length, f"row {row} is inside the frame")
            assert_false(seen.get(row, false), f"row {row} was drawn once")
            seen[row] = true
        }
        let again = sample(frame, n: wanted, seed: 2026)
        defer { again.free() }
        assert_eq(again.column("row").to_ints(), drawn.column("row").to_ints())
    }
}

fn test_sample_takes_a_fraction_and_can_replace() -> void {
    if not gpu.is_available() { return }
    let frame = ramp_frame(10)
    defer { frame.free() }
    let half = sample(frame, frac: 0.5, seed: 3)
    defer { half.free() }
    assert_eq(half.len(), 5)
    let over = sample(frame, n: 25, replace: true, seed: 3)
    defer { over.free() }
    assert_eq(over.len(), 25)
    for row in over.column("row").to_ints() {
        assert_true(row >= 0 and row < 10, f"row {row} is inside the frame")
    }
    assert_throws(ValueError, || sample(frame, n: 11))
    assert_throws(ValueError, || sample(frame, n: 1, frac: 0.5))
}

fn test_sample_of_a_series_keeps_its_labels() -> void {
    if not gpu.is_available() { return }
    let series = Series(
        Column.from_ints([1, 2, 3, 4]),
        index: Index.of([Column.from_strs(["a", "b", "c", "d"])])
    )
    defer { series.free() }
    let drawn = sample_series(series, n: 4, seed: 11)
    defer { drawn.free() }
    var labels: list[str?] = []
    for at in 0..4 { labels.push(drawn.index.label_at(at)[0].to_str()) }
    for (at, value) in drawn.values.to_ints().enumerate() {
        assert_eq(labels[at], ["a", "b", "c", "d"][value - 1])
    }
}

fn named_frame() -> DataFrame {
    return DataFrame(
        ["alpha", "beta", "alps"],
        [Column.from_ints([1]), Column.from_ints([2]), Column.from_ints([3])]
    )
}

fn test_filter_names_picks_columns_three_ways() -> void {
    if not gpu.is_available() { return }
    let frame = named_frame()
    defer { frame.free() }
    assert_eq(filter_names(frame, items: ["beta", "gamma"]).names, ["beta"])
    assert_eq(filter_names(frame, like: "alp").names, ["alpha", "alps"])
    assert_eq(filter_names(frame, regex: "a$").names, ["alpha", "beta"])
    assert_throws(ValueError, || filter_names(frame))
    assert_throws(ValueError, || filter_names(frame, like: "a", regex: "a"))
}

fn test_filter_names_reads_row_labels_off_the_levels() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(
        ["v"],
        [Column.from_ints([1, 2, 3])],
        index: Index.of([Column.from_strs(["austin", "dallas", "auckland"])])
    )
    defer { frame.free() }
    let picked = filter_names(frame, like: "au", axis: 0)
    defer { picked.free() }
    assert_eq(picked.column("v").to_ints(), [1, 3])
    let ranged = ramp_frame(2)
    defer { ranged.free() }
    assert_throws(TypeError, || filter_names(ranged, like: "a", axis: 0))
}

#: Assignment writes into the column the frame is already holding, so a frame
#: built from the same columns sees it. That is the property the doc comment
#: claims, and it is the one worth pinning down.
fn test_set_iloc_writes_through_to_every_holder() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(["a", "b"], [Column.from_ints([1, 2, 3]), Column.from_ints([4, 5, 6])])
    defer { frame.free() }
    let shared = frame._replace()
    set_iloc(frame, Pick.one_at(1), Value.of_int(99), columns: ["a"])
    assert_eq(shared.column("a").to_ints(), [1, 99, 3])
    assert_eq(shared.column("b").to_ints(), [4, 5, 6])
    let flags = Column.from_bools([true, false, true])
    defer { flags.free() }
    set_iloc(frame, Pick.masked(flags), Value.of_int(0))
    assert_eq(frame.column("a").to_ints(), [0, 99, 0])
    assert_eq(frame.column("b").to_ints(), [0, 5, 0])
    set_iloc(frame, Pick.span(0, 3), NA, columns: ["b"])
    assert_eq(frame.column("b").null_mask(), [true, true, true])
}

fn test_set_loc_writes_the_rows_a_label_names() -> void {
    if not gpu.is_available() { return }
    let frame = labelled_frame()
    defer { frame.free() }
    set_loc(frame, Labels.one_of(Value.of_str("a")), Value.of_int(0))
    assert_eq(frame.column("v").to_ints(), [0, 20, 0, 40])
    set_loc(frame, Labels.every(), Value.of_int(7))
    assert_eq(frame.column("v").to_ints(), [7, 7, 7, 7])
}

fn test_set_iloc_interns_the_text_it_writes() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(["city"], [Column.from_strs(["austin", "dallas"])])
    defer { frame.free() }
    set_iloc(frame, Pick.one_at(0), Value.of_str("houston"))
    assert_eq(frame.column("city").to_strs(), ["houston", "dallas"])
    let flags = Column.from_bools([false, true])
    defer { flags.free() }
    set_iloc(frame, Pick.masked(flags), Value.of_str("el paso"))
    assert_eq(frame.column("city").to_strs(), ["houston", "el paso"])
}

fn test_set_column_values_hands_back_what_it_displaced() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(["a"], [Column.from_ints([1, 2])])
    defer { frame.free() }
    let replacement = Column.from_floats([9.5, 8.5])
    let displaced = set_column_values(frame, "a", replacement)
    defer { displaced.free() }
    assert_eq(displaced.to_ints(), [1, 2])
    assert_eq(frame.column("a").to_floats(), [9.5, 8.5])
    let wrong = Column.from_ints([1])
    defer { wrong.free() }
    assert_throws(ValueError, || set_column_values(frame, "a", wrong))
    assert_throws(KeyError, || set_column_values(frame, "z", wrong))
}

#: Inserting rows and dropping them again has to leave the frame it started
#: from, which is the round trip that catches an off-by-one in the
#: permutation the insert is built on.
fn test_insert_and_drop_are_inverses() -> void {
    if not gpu.is_available() { return }
    let frame = ramp_frame(6)
    defer { frame.free() }
    let extra = DataFrame(["row"], [Column.from_ints([98, 99])])
    defer { extra.free() }
    let bigger = insert_rows(frame, extra, position: 2)
    defer { bigger.free() }
    assert_eq(bigger.column("row").to_ints(), [0, 1, 98, 99, 2, 3, 4, 5])
    assert_eq(bigger.len(), 8)
    let back = drop_rows(bigger, [2, 3])
    defer { back.free() }
    assert_eq(back.column("row").to_ints(), frame.column("row").to_ints())
    let appended = insert_rows(frame, extra)
    defer { appended.free() }
    assert_eq(appended.column("row").to_ints(), [0, 1, 2, 3, 4, 5, 98, 99])
}

fn test_insert_carries_a_stored_index_through() -> void {
    if not gpu.is_available() { return }
    let frame = labelled_frame()
    defer { frame.free() }
    let extra = DataFrame(
        ["v"],
        [Column.from_ints([50])],
        index: Index.of([Column.from_strs(["z"])], level_names(["k"]))
    )
    defer { extra.free() }
    let bigger = insert_rows(frame, extra, position: 1)
    defer { bigger.free() }
    assert_eq(bigger.column("v").to_ints(), [10, 50, 20, 30, 40])
    assert_eq(bigger.index.levels[0].to_strs(), ["a", "z", "b", "a", "c"])
    let mismatched = DataFrame(["other"], [Column.from_ints([1])])
    defer { mismatched.free() }
    assert_throws(KeyError, || insert_rows(frame, mismatched))
}

fn test_drop_rows_ignores_a_repeat_and_counts_from_the_end() -> void {
    if not gpu.is_available() { return }
    let frame = ramp_frame(5)
    defer { frame.free() }
    let kept = drop_rows(frame, [1, 1, -1])
    defer { kept.free() }
    assert_eq(kept.column("row").to_ints(), [0, 2, 3])
    let untouched = drop_rows(frame, [])
    defer { untouched.free() }
    assert_eq(untouched.len(), 5)
    assert_throws(IndexError, || drop_rows(frame, [5]))
}

fn test_selection_over_an_empty_frame_stays_empty() -> void {
    if not gpu.is_available() { return }
    let frame = ramp_frame(0)
    defer { frame.free() }
    let sliced = iloc(frame, Pick.span(0, 5))
    defer { sliced.free() }
    assert_eq(sliced.len(), 0)
    let flags = Column.from_bools([])
    defer { flags.free() }
    let filtered = filter_rows(frame, flags)
    defer { filtered.free() }
    assert_eq(filtered.len(), 0)
    let none = sample(frame, n: 0)
    defer { none.free() }
    assert_eq(none.len(), 0)
    assert_throws(ValueError, || sample(frame, n: 1))
}

fn test_the_selection_kinds_are_what_the_constructors_say() -> void {
    assert_eq(Pick.one_at(3).kind, SelectKind.One)
    assert_eq(Pick.of([1, 2]).kind, SelectKind.Many)
    assert_eq(Pick.span(0, 2).kind, SelectKind.Span)
    assert_eq(Pick.every().kind, SelectKind.Every)
    assert_eq(Labels.one_of(Value.of_int(1)).kind, SelectKind.One)
    assert_eq(Labels.of([Value.of_int(1), Value.of_int(2)]).kind, SelectKind.Many)
    assert_eq(Labels.span(Value.of_int(1), Value.of_int(2)).kind, SelectKind.Span)
    assert_eq(Labels.every().kind, SelectKind.Every)
}

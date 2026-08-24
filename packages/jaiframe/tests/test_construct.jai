#: Building frames from host values, and the edits to their column list.
#:
#: The interesting claims here are not the shapes, which a reader can check by
#: eye, but the two that need arithmetic. The first is that the bulk upload
#: path puts the right number in the right slot at a size where a mistake in
#: the lane building would not show in the first few rows — the 70000-row case
#: checks a host-built column against the same ramp built on the device, which
#: is a second, independent answer. That size also strides the fixed grid
#: `equals` reduces over, whose stride is 65536 threads.
#:
#: The second is that the operations mean what they say: every `equals` here is
#: cross-checked against a comparison done wholly on the host, out of
#: downloaded values and null masks, so a bug in the comparison kernel cannot
#: pass by agreeing with itself.
import std.gpu as gpu
from std.math import NAN, is_nan
from std.test import assert_eq, assert_false, assert_near, assert_throws, assert_true
from jaiframe.column import Column, DataFrame, Dtype, Index, Series, Value, NA
from jaiframe.column import positions_buffer
from jaiframe.construct import add_prefix, add_suffix, assign, astype, astype_series
from jaiframe.construct import categories, categorize, column_from_values, copy
from jaiframe.construct import drop_columns, empty_frame, equals, frame_from_columns
from jaiframe.construct import frame_from_dict, frame_from_records, frame_from_rows, full
from jaiframe.construct import index_from_list, infer_dtype, insert, is_na_value, pop
from jaiframe.construct import remove_unused_categories, rename, reorder, select_dtypes
from jaiframe.construct import series_from_dict, series_from_list, set_categories
from jaiframe.construct import to_frame, to_series

fn test_inference_promotes_along_one_ladder() -> void {
    let bools: list[any] = [true, false, true]
    let ints: list[any] = [1, 2, 3]
    let mixed_numbers: list[any] = [1, 2.5]
    let with_bool: list[any] = [true, 7]
    let text: list[any] = ["a", "b"]
    let ragged: list[any] = [1, "a"]
    let stamps: list[any] = [Value.of_datetime(1700000000000), Value.of_datetime(0)]
    let nothing: list[any] = [null, null]
    assert_eq(infer_dtype(bools), Dtype.Bool)
    assert_eq(infer_dtype(ints), Dtype.Int64)
    assert_eq(infer_dtype(mixed_numbers), Dtype.Float64)
    assert_eq(infer_dtype(with_bool), Dtype.Int64)
    assert_eq(infer_dtype(text), Dtype.Str)
    assert_eq(infer_dtype(ragged), Dtype.Str)
    assert_eq(infer_dtype(stamps), Dtype.Datetime)
    assert_eq(infer_dtype(nothing), Dtype.Float64)
    assert_eq(infer_dtype([]), Dtype.Float64)
}

#: A missing entry must not vote, or a gap would drag a numeric column into
#: text. Checked by inferring the same list twice, once with the gaps spelled
#: as sentinels and once with them left out entirely.
fn test_missing_entries_do_not_vote() -> void {
    let dense: list[any] = [1, 2, 3]
    let gapped: list[any] = [1, null, 2, "", 3, "n/a"]
    assert_eq(infer_dtype(gapped, ["n/a"]), infer_dtype(dense))
    assert_eq(infer_dtype(gapped), Dtype.Str)
}

fn test_the_null_sentinels_are_the_four_documented_ones() -> void {
    assert_true(is_na_value(null), "null is missing")
    assert_true(is_na_value(NAN), "a NaN is missing")
    assert_true(is_na_value(""), "a blank cell is missing")
    assert_true(is_na_value("-", ["-"]), "a named sentinel is missing")
    assert_true(is_na_value(NA), "an NA cell is missing")
    assert_false(is_na_value(0), "zero is a number")
    assert_false(is_na_value("-"), "a dash is text until the caller says otherwise")
    assert_false(is_na_value("", null, false), "keep_default_na off leaves a blank alone")
    assert_true(is_na_value("", ["", "x"], false), "a blank still counts when named")
}

#: Every dtype the host can spell, read back through a different path than it
#: was written: the codes and lanes go up as one upload and come down through
#: `to_strs` and friends.
fn test_a_column_round_trips_every_dtype() -> void {
    if not gpu.is_available() { return }
    let floats = column_from_values([1.5, null, -2.5])
    defer { floats.free() }
    assert_eq(floats.dtype, Dtype.Float64)
    assert_eq(floats.null_mask(), [false, true, false])
    assert_near(floats.get(0).to_float(), 1.5, tolerance: 1e-6)

    let ints = column_from_values([1700000000123, -5, null])
    defer { ints.free() }
    assert_eq(ints.dtype, Dtype.Int64)
    assert_eq(ints.get(0).to_int(), 1700000000123)
    assert_eq(ints.get(1).to_int(), -5)
    assert_true(ints.get(2).is_na(), "the null row stayed missing")

    let flags = column_from_values([true, false, null])
    defer { flags.free() }
    assert_eq(flags.dtype, Dtype.Bool)
    assert_eq(flags.to_bools(), [true, false, false])
    assert_eq(flags.null_count(), 1)

    let text = column_from_values(["austin", "dallas", "austin", null])
    defer { text.free() }
    assert_eq(text.dtype, Dtype.Str)
    assert_eq(text.level_count(), 2)
    assert_eq(text.to_strs(), ["austin", "dallas", "austin", null])

    let stamps = column_from_values([Value.of_datetime(1700000000000), null])
    defer { stamps.free() }
    assert_eq(stamps.dtype, Dtype.Datetime)
    assert_eq(stamps.get(0).to_int(), 1700000000000)
}

fn test_a_forced_dtype_beats_inference() -> void {
    if not gpu.is_available() { return }
    let stamps = column_from_values([0, 1700000000000], dtype: Dtype.Datetime)
    defer { stamps.free() }
    assert_eq(stamps.dtype, Dtype.Datetime)
    assert_eq(stamps.to_ints(), [0, 1700000000000])
    let words = column_from_values([1, 2.5, true], dtype: Dtype.Str)
    defer { words.free() }
    assert_eq(words.to_strs(), ["1", "2.5", "true"])
    assert_throws(TypeError, || column_from_values(["nine"], dtype: Dtype.Float64))
}

#: The bulk upload path at a size where a lane built wrong would not show in
#: the first few rows. The second answer is the same ramp built on the device
#: by `Index.range`, which shares no code with the host loop under test.
#:
#: 70000 also strides the fixed grid `equals` reduces over, so the comparison's
#: own carry between passes is exercised rather than assumed.
fn test_the_bulk_upload_matches_a_device_built_ramp() -> void {
    if not gpu.is_available() { return }
    var host: list[any] = []
    for at in 0..70000 { host.push(at) }
    let uploaded = column_from_values(host)
    defer { uploaded.free() }
    let levels = Index.range(70000).materialize()
    let ramp = levels[0]
    defer { ramp.free() }
    assert_eq(uploaded.dtype, ramp.dtype)
    assert_true(equals(uploaded, ramp), "the host lane matches the device ramp")
    uploaded.set(69999, Value.of_int(-1))
    assert_false(equals(uploaded, ramp), "one changed row is one difference")
}

#: A table given a row at a time has to come out column-wise. Checked cell by
#: cell against the source lists rather than against a written-out expectation.
fn test_rows_become_columns() -> void {
    if not gpu.is_available() { return }
    let rows: list[list[any]] = [["austin", 38.0, 3], ["dallas", 41.0, 1], ["austin", 36.5, 4]]
    let frame = frame_from_rows(rows, ["city", "high", "days"])
    defer { frame.free() }
    assert_eq(frame.shape(), (3, 3))
    assert_eq(frame.dtypes(), [Dtype.Str, Dtype.Float64, Dtype.Int64])
    let cities = frame.column("city").to_strs()
    let highs = frame.column("high").to_floats()
    let days = frame.column("days").to_ints()
    for (at, row) in rows.enumerate() {
        assert_eq(cities[at], row[0])
        assert_near(highs[at], row[1], tolerance: 1e-6)
        assert_eq(days[at], row[2])
    }
}

fn test_rows_without_names_are_numbered() -> void {
    if not gpu.is_available() { return }
    let frame = frame_from_rows([[1, 2], [3, 4]])
    defer { frame.free() }
    assert_eq(frame.names, ["0", "1"])
    assert_eq(frame.column("1").to_ints(), [2, 4])
}

fn test_a_malformed_call_allocates_nothing() -> void {
    if not gpu.is_available() { return }
    assert_throws(ValueError, || frame_from_rows([[1, 2], [3]], ["a", "b"]))
    assert_throws(ValueError, || frame_from_rows([[1]], ["a"], null, [Dtype.Int64, Dtype.Str]))
    assert_throws(ValueError, || frame_from_rows([[1]], ["a"], Index.range(9)))
    assert_throws(ValueError, || frame_from_dict({"a": [1, 2], "b": [3]}))
}

fn test_a_dict_keeps_the_order_it_was_written_in() -> void {
    if not gpu.is_available() { return }
    let frame = frame_from_dict({"zulu": [1, 2], "alpha": ["x", "y"], "mike": [0.5, 1.5]})
    defer { frame.free() }
    assert_eq(frame.names, ["zulu", "alpha", "mike"])
    assert_eq(frame.dtypes(), [Dtype.Int64, Dtype.Str, Dtype.Float64])
    assert_eq(frame.column("alpha").to_strs(), ["x", "y"])
}

#: A record that omits a key must leave a gap in that column and not shift the
#: row along. The check rebuilds each column from the records a second time.
fn test_ragged_records_leave_gaps() -> void {
    if not gpu.is_available() { return }
    let records: list[dict[str, any]] = [
        {"city": "austin", "high": 38.0},
        {"high": 41.0, "wind": 12.0},
        {"city": "dallas"},
    ]
    let frame = frame_from_records(records)
    defer { frame.free() }
    assert_eq(frame.names, ["city", "high", "wind"])
    let cities = frame.column("city").to_strs()
    let highs = frame.column("high").null_mask()
    for (at, record) in records.enumerate() {
        assert_eq(cities[at], record.get("city", null))
        assert_eq(highs[at], record.get("high", null) is null)
    }
}

fn test_records_can_be_told_which_columns_to_keep() -> void {
    if not gpu.is_available() { return }
    let records: list[dict[str, any]] = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    let frame = frame_from_records(records, ["b"])
    defer { frame.free() }
    assert_eq(frame.names, ["b"])
    assert_eq(frame.column("b").to_ints(), [2, 4])
}

fn test_a_series_from_a_dict_is_labelled_by_its_keys() -> void {
    if not gpu.is_available() { return }
    let series = series_from_dict({"austin": 38.0, "dallas": 41.0}, name: "high")
    defer { series.free() }
    assert_eq(series.len(), 2)
    assert_eq(series.name, "high")
    assert_eq(series.index.label_at(0)[0].to_str(), "austin")
    assert_eq(series.index.label_at(1)[0].to_str(), "dallas")
    assert_near(series.values.get(1).to_float(), 41.0, tolerance: 1e-6)
}

fn test_a_series_from_a_list_takes_the_index_it_is_given() -> void {
    if not gpu.is_available() { return }
    let labels = index_from_list(["a", "b"], name: "key")
    let series = series_from_list([1, 2], index: labels, name: "count")
    defer { series.free() }
    assert_eq(series.dtype(), Dtype.Int64)
    assert_eq(series.index.names[0], "key")
    assert_throws(ValueError, || series_from_list([1, 2, 3], index: Index.range(2)))
}

fn test_an_empty_frame_still_has_columns() -> void {
    if not gpu.is_available() { return }
    let frame = empty_frame(["a", "b"], [Dtype.Int64, Dtype.Str])
    defer { frame.free() }
    assert_eq(frame.shape(), (0, 2))
    assert_eq(frame.dtypes(), [Dtype.Int64, Dtype.Str])
    assert_eq(frame.column("a").to_ints(), [])
    let bare = empty_frame()
    defer { bare.free() }
    assert_eq(bare.shape(), (0, 0))
}

fn test_full_fills_every_cell() -> void {
    if not gpu.is_available() { return }
    let frame = full(["a", "b"], 5, Value.of_float(2.5))
    defer { frame.free() }
    assert_eq(frame.shape(), (5, 2))
    var total = 0.0
    for value in frame.column("b").to_floats() { total += value }
    assert_near(total, 12.5, tolerance: 1e-6)
    let gaps = full(["a"], 3, NA)
    defer { gaps.free() }
    assert_eq(gaps.column("a").null_count(), 3)
}

fn test_a_frame_over_columns_adopts_them() -> void {
    if not gpu.is_available() { return }
    let frame = frame_from_columns(["a"], [Column.from_ints([1, 2])])
    defer { frame.free() }
    assert_eq(frame.column("a").to_ints(), [1, 2])
}

fn test_insert_puts_a_column_where_it_is_asked() -> void {
    if not gpu.is_available() { return }
    let frame = frame_from_dict({"a": [1], "c": [3]})
    defer { frame.free() }
    let middle = Column.from_ints([2])
    defer { middle.free() }
    assert_eq(insert(frame, 1, "b", middle).names, ["a", "b", "c"])
    assert_eq(insert(frame, 0, "b", middle).names, ["b", "a", "c"])
    assert_eq(insert(frame, 2, "b", middle).names, ["a", "c", "b"])
    assert_eq(insert(frame, -1, "b", middle).names, ["a", "b", "c"])
    assert_throws(IndexError, || insert(frame, 3, "b", middle))
    assert_throws(ValueError, || insert(frame, 1, "a", middle))
}

fn test_assign_replaces_or_appends() -> void {
    if not gpu.is_available() { return }
    let frame = frame_from_dict({"a": [1, 2]})
    defer { frame.free() }
    let fresh = Column.from_ints([9, 9])
    defer { fresh.free() }
    let replaced = assign(frame, "a", fresh)
    assert_eq(replaced.names, ["a"])
    assert_eq(replaced.column("a").to_ints(), [9, 9])
    let widened = assign(frame, "b", fresh)
    assert_eq(widened.names, ["a", "b"])
    assert_eq(widened.column("a").to_ints(), [1, 2])
}

fn test_drop_and_pop_take_columns_out() -> void {
    if not gpu.is_available() { return }
    let frame = frame_from_dict({"a": [1], "b": [2], "c": [3]})
    defer { frame.free() }
    assert_eq(drop_columns(frame, ["b"]).names, ["a", "c"])
    assert_eq(drop_columns(frame, ["b", "z"], missing_ok: true).names, ["a", "c"])
    assert_throws(KeyError, || drop_columns(frame, ["z"]))
    let taken = pop(frame, "a")
    assert_eq(taken[0].names, ["b", "c"])
    assert_eq(taken[1].to_ints(), [1])
}

fn test_rename_and_reorder_only_touch_names() -> void {
    if not gpu.is_available() { return }
    let frame = frame_from_dict({"a": [1], "b": [2]})
    defer { frame.free() }
    let renamed = rename(frame, {"a": "alpha", "z": "zulu"})
    assert_eq(renamed.names, ["alpha", "b"])
    assert_eq(renamed.column("alpha").to_ints(), [1])
    assert_throws(KeyError, || rename(frame, {"z": "zulu"}, missing_ok: false))
    let flipped = reorder(frame, ["b", "a"])
    assert_eq(flipped.names, ["b", "a"])
    assert_eq(flipped.column("a").to_ints(), [1])
    assert_throws(ValueError, || reorder(frame, ["a"]))
    assert_throws(ValueError, || reorder(frame, ["a", "a"]))
}

fn test_select_dtypes_filters_and_keeps_order() -> void {
    if not gpu.is_available() { return }
    let frame = frame_from_dict({"n": [1], "s": ["x"], "f": [1.5], "b": [true]})
    defer { frame.free() }
    assert_eq(select_dtypes(frame, [Dtype.Int64, Dtype.Float64]).names, ["n", "f"])
    assert_eq(select_dtypes(frame, exclude: [Dtype.Str]).names, ["n", "f", "b"])
    assert_eq(select_dtypes(frame, [Dtype.Int64, Dtype.Bool], [Dtype.Bool]).names, ["n"])
    assert_throws(ValueError, || select_dtypes(frame))
}

#: Prefixes and suffixes are the same edit read two ways, so applying both and
#: stripping them again has to give the names back.
fn test_prefix_and_suffix_wrap_every_name() -> void {
    if not gpu.is_available() { return }
    let frame = frame_from_dict({"a": [1], "b": [2]})
    defer { frame.free() }
    let wrapped = add_suffix(add_prefix(frame, "tx_"), "_c")
    assert_eq(wrapped.names, ["tx_a_c", "tx_b_c"])
    for (at, name) in wrapped.names.enumerate() {
        assert_eq(name.len(), frame.names[at].len() + 5)
    }
}

fn test_astype_converts_and_owns_the_result() -> void {
    if not gpu.is_available() { return }
    let frame = frame_from_dict({"a": [1, 0, 5], "b": [1.5, 2.5, 3.5]})
    defer { frame.free() }
    let converted = astype(frame, {"a": Dtype.Bool, "b": "int64"})
    defer { converted.free() }
    assert_eq(converted.dtypes(), [Dtype.Bool, Dtype.Int64])
    assert_eq(converted.column("a").to_bools(), [true, false, true])
    assert_eq(converted.column("b").to_ints(), [1, 2, 3])
    let widened = astype(frame, Dtype.Float64)
    defer { widened.free() }
    assert_eq(widened.dtypes(), [Dtype.Float64, Dtype.Float64])
    assert_throws(KeyError, || astype(frame, {"z": Dtype.Bool}))
    assert_throws(TypeError, || astype(frame, 7))
}

fn test_astype_leaves_the_original_readable() -> void {
    if not gpu.is_available() { return }
    let frame = frame_from_dict({"a": [1, 2]})
    defer { frame.free() }
    let converted = astype(frame, Dtype.Float64)
    converted.free()
    assert_eq(frame.column("a").to_ints(), [1, 2])
    assert_eq(frame.len(), 2)
}

fn test_a_series_converts_on_its_own() -> void {
    if not gpu.is_available() { return }
    let series = series_from_list([1, 2, 3], name: "n")
    defer { series.free() }
    let floats = astype_series(series, Dtype.Float64)
    defer { floats.free() }
    assert_eq(floats.dtype(), Dtype.Float64)
    assert_eq(floats.name, "n")
    assert_near(floats.values.get(2).to_float(), 3.0, tolerance: 1e-6)
}

fn test_a_deep_copy_is_independent_and_a_shallow_one_is_not() -> void {
    if not gpu.is_available() { return }
    let frame = frame_from_dict({"a": [1.0, 2.0]})
    defer { frame.free() }
    let deep = copy(frame)
    defer { deep.free() }
    let shallow = copy(frame, deep: false)
    frame.column("a").set(0, Value.of_float(99.0))
    assert_near(deep.column("a").get(0).to_float(), 1.0, tolerance: 1e-6)
    assert_near(shallow.column("a").get(0).to_float(), 99.0, tolerance: 1e-6)
}

fn test_a_one_column_frame_and_a_series_are_the_same_thing() -> void {
    if not gpu.is_available() { return }
    let frame = frame_from_dict({"high": [38.0, 41.0]})
    defer { frame.free() }
    let series = to_series(frame)
    assert_eq(series.name, "high")
    let back = to_frame(series)
    assert_eq(back.names, ["high"])
    assert_true(equals(back, frame), "a round trip through a series changes nothing")
    let wide = frame_from_dict({"a": [1], "b": [2]})
    defer { wide.free() }
    assert_throws(ValueError, || to_series(wide))
}

#: `equals` against a comparison done entirely on the host, so the kernel is
#: never checked against itself.
fn test_equals_agrees_with_a_host_comparison() -> void {
    if not gpu.is_available() { return }
    let cases: list[list[any]] = [
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.5],
        [1.0, null, 3.0, 4.0],
        [null, null, null, null],
        ["a", "b", "c", "d"],
        ["a", "b", "c", null],
        [1, 2, 3, 4],
        [1, 2, 3, 5],
        [true, false, true, false],
    ]
    let base = column_from_values(cases[0])
    defer { base.free() }
    for values in cases {
        let other = column_from_values(values)
        defer { other.free() }
        assert_eq(
            equals(base, other),
            same_by_hand(base, other),
            f"equals matches the host answer for {values}"
        )
        let twin = column_from_values(values)
        defer { twin.free() }
        assert_true(equals(other, twin), f"{values} equals itself")
    }
}

fn test_equals_needs_the_same_dtype_and_length() -> void {
    if not gpu.is_available() { return }
    let ints = Column.from_ints([1, 2])
    defer { ints.free() }
    let floats = Column.from_floats([1.0, 2.0])
    defer { floats.free() }
    let shorter = Column.from_ints([1])
    defer { shorter.free() }
    assert_false(equals(ints, floats), "an int column is not a float column")
    assert_false(equals(ints, shorter), "two lengths are two columns")
    assert_false(equals(ints, "no"), "a column is not a string")
}

#: The four-at-a-time path only runs for one lane a side, no mask, and a length
#: that divides by four, so both it and the scalar fallback need exercising —
#: including a NaN, which the kernel is required to call equal to itself.
fn test_equals_reads_four_slots_at_a_time_when_it_can() -> void {
    if not gpu.is_available() { return }
    let quads = maskless([NAN, 1.0, 2.0, 3.0])
    defer { quads.free() }
    let twin = maskless([NAN, 1.0, 2.0, 3.0])
    defer { twin.free() }
    let differs = maskless([NAN, 1.0, 2.0, 3.5])
    defer { differs.free() }
    assert_true(equals(quads, twin), "a NaN in the same slot is the same cell")
    assert_false(equals(quads, differs), "the last of four still counts")
    let odd = maskless([1.0, 2.0, 3.0, 4.0, 5.0])
    defer { odd.free() }
    let odd_twin = maskless([1.0, 2.0, 3.0, 4.0, 5.0])
    defer { odd_twin.free() }
    let odd_differs = maskless([1.0, 2.0, 3.0, 4.0, 5.5])
    defer { odd_differs.free() }
    assert_true(equals(odd, odd_twin), "five slots take the scalar path")
    assert_false(equals(odd, odd_differs), "and still find the difference")
}

fn test_equals_compares_labels_as_well_as_values() -> void {
    if not gpu.is_available() { return }
    let left = frame_from_dict({"a": [1, 2]})
    defer { left.free() }
    let right = frame_from_dict({"a": [1, 2]})
    defer { right.free() }
    assert_true(equals(left, right), "same names, same labels, same values")
    let relabelled = frame_from_dict({"a": [1, 2]}, index: index_from_list(["x", "y"]))
    defer { relabelled.free() }
    assert_false(equals(left, relabelled), "different labels are a different frame")
    let renamed = rename(left, {"a": "b"})
    assert_false(equals(left, renamed), "a column name is part of the frame")
}

#: A range index and the stored labels it stands for hold the same data, so
#: they have to compare equal even though one of them stores nothing.
fn test_an_implied_index_equals_the_labels_it_stands_for() -> void {
    if not gpu.is_available() { return }
    let implied = Index.range(3)
    defer { implied.free() }
    let stored = index_from_list([0, 1, 2])
    defer { stored.free() }
    assert_true(equals(implied, stored), "0, 1, 2 either way")
    let shifted = Index.range(3, start: 1)
    defer { shifted.free() }
    assert_false(equals(implied, shifted), "1, 2, 3 is not 0, 1, 2")
}

fn test_categorize_relabels_and_can_order_the_levels() -> void {
    if not gpu.is_available() { return }
    let text = column_from_values(["dallas", "austin", "dallas"])
    defer { text.free() }
    let plain = categorize(text)
    defer { plain.free() }
    assert_eq(plain.dtype, Dtype.Category)
    assert_eq(categories(plain), ["dallas", "austin"])
    assert_eq(plain.to_strs(), text.to_strs())
    let ranked = categorize(text, ordered: true)
    defer { ranked.free() }
    assert_eq(categories(ranked), sorted(categories(text)))
    assert_eq(ranked.to_strs(), text.to_strs())
    let numbers = Column.from_ints([1])
    defer { numbers.free() }
    assert_throws(TypeError, || categorize(numbers))
}

#: Ordering is stored as the level order, so the codes of an ordered column
#: have to rank the same way the strings do.
fn test_an_ordered_categorys_codes_rank_like_its_strings() -> void {
    if not gpu.is_available() { return }
    let text = column_from_values(["small", "large", "medium", "large"])
    defer { text.free() }
    let ranked = categorize(text, ordered: true)
    defer { ranked.free() }
    let names = ranked.to_strs()
    let codes = ranked.values.download()
    for (at, name) in names.enumerate() {
        for (other, peer) in names.enumerate() {
            if name is null or peer is null { continue }
            assert_eq(
                codes[at] < codes[other],
                (name ?? "") < (peer ?? ""),
                f"{name} against {peer}"
            )
        }
    }
}

#: The levels that survive have to be exactly the distinct strings the rows
#: use, which the test works out on the host with a dict instead of asking the
#: device again.
fn test_remove_unused_categories_keeps_every_row() -> void {
    if not gpu.is_available() { return }
    let text = column_from_values(["a", "b", "c", "d", null])
    defer { text.free() }
    let narrowed = set_categories(text, ["a", "b", "c", "d", "e"])
    defer { narrowed.free() }
    let used = column_from_values(["a", "c", "a", "c", null])
    defer { used.free() }
    let recoded = set_categories(used, categories(narrowed))
    defer { recoded.free() }
    assert_eq(categories(recoded).len(), 5)
    let trimmed = remove_unused_categories(recoded)
    defer { trimmed.free() }
    var distinct: dict[str, bool] = {}
    var expected: list[str] = []
    for name in recoded.to_strs() {
        if name is null { continue }
        if distinct.get(name, false) { continue }
        distinct[name] = true
        expected.push(name)
    }
    assert_eq(sorted(categories(trimmed)), sorted(expected))
    assert_eq(trimmed.to_strs(), recoded.to_strs())
    assert_eq(trimmed.null_count(), recoded.null_count())
}

#: The usage pass stages its marks in threadgroup memory and merges them once
#: per group, so a column tall enough to fill many groups is the case that
#: would show a merge that dropped a mark.
fn test_level_usage_survives_many_threadgroups() -> void {
    if not gpu.is_available() { return }
    var host: list[any] = []
    for at in 0..70000 { host.push(if at == 69999 { "rare" } else { "common" }) }
    let text = column_from_values(host)
    defer { text.free() }
    let wide = set_categories(text, ["common", "rare", "never"])
    defer { wide.free() }
    let trimmed = remove_unused_categories(wide)
    defer { trimmed.free() }
    assert_eq(sorted(categories(trimmed)), ["common", "rare"])
    assert_eq(trimmed.len(), 70000)
    assert_eq(trimmed.get(69999).to_str(), "rare")
    assert_eq(trimmed.null_count(), 0)
}

fn test_set_categories_narrows_the_domain() -> void {
    if not gpu.is_available() { return }
    let text = column_from_values(["a", "b", "c"])
    defer { text.free() }
    let narrowed = set_categories(text, ["c", "a"])
    defer { narrowed.free() }
    assert_eq(narrowed.dtype, Dtype.Category)
    assert_eq(categories(narrowed), ["c", "a"])
    assert_eq(narrowed.to_strs(), ["a", null, "c"])
    assert_eq(narrowed.null_count(), 1)
    assert_throws(ValueError, || set_categories(text, ["a", "a"]))
    let numbers = Column.from_ints([1])
    defer { numbers.free() }
    assert_throws(TypeError, || categories(numbers))
}

#: A lane of ones and no lane at all say the same thing about missingness, so
#: two columns that differ only in whether they carry a mask are equal.
fn test_a_mask_of_ones_is_the_same_as_no_mask() -> void {
    if not gpu.is_available() { return }
    let bare = Column.from_floats([1.0, 2.0])
    defer { bare.free() }
    let masked = Column.from_floats([1.0, 2.0], [false, false])
    defer { masked.free() }
    assert_false(bare.has_validity(), "nothing missing means no lane")
    assert_true(masked.has_validity(), "spelled-out flags make one")
    assert_true(equals(bare, masked), "the same two numbers either way")
}

#: Codes are only comparable on one table. Two columns holding the same strings
#: under tables built in different orders have to compare equal, which is the
#: case the host-side shortcut for identical tables must not swallow.
fn test_the_same_strings_under_different_tables_are_equal() -> void {
    if not gpu.is_available() { return }
    let reversed = Column.from_strs(["b", "a"])
    defer { reversed.free() }
    let positions = positions_buffer([1, 0])
    defer { positions.free() }
    let taken = reversed.take(positions, 2)
    defer { taken.free() }
    let plain = column_from_values(["a", "b"])
    defer { plain.free() }
    assert_eq(categories(taken), ["b", "a"])
    assert_eq(categories(plain), ["a", "b"])
    assert_eq(taken.to_strs(), plain.to_strs())
    assert_true(equals(taken, plain), "the same strings, differently coded")
}

fn test_a_dictionary_with_no_levels_survives_being_trimmed() -> void {
    if not gpu.is_available() { return }
    let blank = Column.from_strs([null, null])
    defer { blank.free() }
    assert_eq(blank.level_count(), 0)
    let trimmed = remove_unused_categories(blank)
    defer { trimmed.free() }
    assert_eq(trimmed.to_strs(), [null, null])
    let recoded = set_categories(blank, ["a"])
    defer { recoded.free() }
    assert_eq(recoded.null_count(), 2)
    assert_eq(categories(recoded), ["a"])
}

fn maskless(values: list[float]) -> Column {
    let column = Column.allocate(Dtype.Float64, values.len())
    column.values.upload(values)
    return column
}

#: The same question `equals` answers, worked out from downloaded values and
#: null masks instead of on the device.
fn same_by_hand(left: Column, right: Column) -> bool {
    if left.dtype != right.dtype { return false }
    if left.len() != right.len() { return false }
    let missing_left = left.null_mask()
    let missing_right = right.null_mask()
    if left.dtype.is_dictionary() {
        let ours = left.to_strs()
        let theirs = right.to_strs()
        for (at, missing) in missing_left.enumerate() {
            if missing != missing_right[at] { return false }
            if not missing and ours[at] != theirs[at] { return false }
        }
        return true
    }
    let ours = left.to_floats()
    let theirs = right.to_floats()
    for (at, missing) in missing_left.enumerate() {
        if missing != missing_right[at] { return false }
        if missing { continue }
        if is_nan(ours[at]) and is_nan(theirs[at]) { continue }
        if ours[at] != theirs[at] { return false }
    }
    return true
}

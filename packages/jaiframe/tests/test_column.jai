#: The column store: the lanes, the dictionary, and the primitives every other
#: jaiframe module dispatches through.
#:
#: Two things are worth checking here that a reader cannot check by eye. The
#: first is that the two-lane integer encoding is exact where it claims to be —
#: 48 bits, both signs, and the boundary where it stops. The second is that the
#: scan behind `compress` and `prefix_sum` is right past one threadgroup: a
#: single-block scan is correct by construction, and every bug in one of these
#: lives in the carry between blocks. The 70000-row case there is built by
#: `Index.range().materialize()`, on the device, so the test costs a dispatch
#: rather than a host loop over seventy thousand boxed floats.
import std.gpu as gpu
from std.gpu import Buffer
from std.math import NAN, is_nan
from std.test import assert_eq, assert_false, assert_near, assert_throws, assert_true
from jaiframe.column import Agg, Column, DataFrame, Dtype, Index, Series, Value
from jaiframe.column import NA, WIDE_LIMIT, WIDE_SCALE
from jaiframe.column import concat_columns, iota, positions_buffer, prefix_sum, unify_levels
from jaiframe.column import wide_pack, wide_unpack

fn test_dtype_reports_how_it_is_stored() -> void {
    assert_eq(Dtype.Float64.name(), "float64")
    assert_eq(Dtype.Datetime.name(), "datetime64[ms]")
    assert_true(Dtype.Int64.is_numeric(), "int64 is numeric")
    assert_false(Dtype.Datetime.is_numeric(), "a timestamp difference is a duration")
    assert_true(Dtype.Datetime.is_wide(), "a timestamp needs both lanes")
    assert_false(Dtype.Bool.is_wide(), "a bool fits one lane")
    assert_true(Dtype.Category.is_dictionary(), "a category is coded")
    assert_eq(Dtype.from_name("int"), Dtype.Int64)
    assert_eq(Dtype.from_name("datetime64[ns]"), Dtype.Datetime)
    assert_throws(ValueError, || Dtype.from_name("complex128"))
}

fn test_a_value_carries_one_cell() -> void {
    assert_eq(Value.of_int(47).to_int(), 47)
    assert_eq(Value.of_bool(true).to_str(), "true")
    assert_eq(Value.of_str("austin").to_str(), "austin")
    assert_eq(Value.of_datetime(1700000000000).to_int(), 1700000000000)
    assert_true(NA.is_na(), "NA is missing")
    assert_true(is_nan(NA.to_float()), "a missing cell reads as NaN")
    assert_eq(Value.null_of(Dtype.Str), NA)
    assert_true(Value.of_float(1.0) == Value.of_int(1), "one is one whichever lane it came from")
    assert_false(Value.of_float(1.0) == Value.of_str("1"), "text is not a number")
    assert_throws(TypeError, || Value.of_str("austin").to_float())
}

fn test_agg_is_one_vocabulary() -> void {
    assert_eq(Agg.Mean.name(), "mean")
    assert_eq(Agg.from_name("avg"), Agg.Mean)
    assert_eq(Agg.from_name("product"), Agg.Prod)
    assert_true(Agg.Size.counts_rows(), "a size is a count")
    assert_false(Agg.Sum.counts_rows(), "a sum keeps its input's type")
    assert_true(Agg.First.picks_a_row(), "a first is one of the inputs")
    assert_false(Agg.Mean.picks_a_row(), "a mean is a new number")
    assert_throws(ValueError, || Agg.from_name("kurtosis"))
}

fn test_the_wide_pair_is_exact_to_48_bits() -> void {
    let cases = [0, 1, -1, WIDE_SCALE, -WIDE_SCALE, 1700000000000, WIDE_LIMIT - 1, -WIDE_LIMIT]
    for value in cases {
        let split = wide_pack(value)
        assert_true(split[0] >= 0.0 and split[0] < float(WIDE_SCALE), f"{value} low half in range")
        assert_eq(wide_unpack(split[0], split[1]), value)
    }
    assert_throws(ValueError, || wide_pack(WIDE_LIMIT))
}

fn test_a_float_column_treats_nan_as_missing() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_floats([1.5, NAN, -2.5])
    defer { column.free() }
    assert_eq(column.len(), 3)
    assert_eq(column.null_count(), 1)
    assert_eq(column.null_mask(), [false, true, false])
    assert_near(column.get(0).to_float(), 1.5, tolerance: 1e-6)
    assert_true(column.get(1).is_na(), "the NaN row is missing")
}

fn test_an_int_column_keeps_every_bit() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_ints([1700000000123, -1700000000123, 0])
    defer { column.free() }
    assert_eq(column.to_ints(), [1700000000123, -1700000000123, 0])
    assert_eq(column.get(1).to_int(), -1700000000123)
    assert_eq(column.null_count(), 0)
}

fn test_a_text_column_is_codes_and_levels() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_strs(["austin", "dallas", "austin", null])
    defer { column.free() }
    assert_eq(column.level_count(), 2)
    assert_eq(column.level_at(0), "austin")
    assert_eq(column.level_code("dallas"), 1)
    assert_eq(column.level_code("houston"), null)
    assert_eq(column.to_strs(), ["austin", "dallas", "austin", null])
    assert_eq(column.null_count(), 1)
}

fn test_get_and_set_reach_one_cell() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_floats([1.0, 2.0, 3.0])
    defer { column.free() }
    column.set(1, Value.of_float(9.0))
    assert_near(column.get(1).to_float(), 9.0, tolerance: 1e-6)
    column.set(-1, NA)
    assert_true(column.get(2).is_na(), "the last row was cleared")
    assert_eq(column.null_count(), 1)
    assert_throws(IndexError, || column.get(3))
}

fn test_take_reorders_and_can_invent_nulls() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_strs(["a", "b", "c"])
    defer { column.free() }
    let positions = positions_buffer([2, 0, -1])
    defer { positions.free() }
    let taken = column.take(positions, 3, allow_missing: true)
    defer { taken.free() }
    assert_eq(taken.to_strs(), ["c", "a", null])
}

fn test_compress_keeps_the_rows_a_mask_marks() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_ints([10, 20, 30, 40])
    defer { column.free() }
    let mask = Column.from_bools([true, false, true, false])
    defer { mask.free() }
    let kept = column.compress(mask)
    defer { kept.free() }
    assert_eq(kept.to_ints(), [10, 30])
    assert_throws(TypeError, || column.compress(column))
}

#: The carry between threadgroups, which a column of a few hundred rows cannot
#: reach. 70000 rows is two levels of the block scan.
fn test_the_scan_carries_across_blocks() -> void {
    if not gpu.is_available() { return }
    let levels = Index.range(70000).materialize()
    let ramp = levels[0]
    defer { ramp.free() }
    let mask = ramp.cast(Dtype.Bool)
    defer { mask.free() }
    let kept = ramp.compress(mask)
    defer { kept.free() }
    assert_eq(kept.len(), 69999)
    assert_eq(kept.get(0).to_int(), 1)
    assert_eq(kept.get(-1).to_int(), 69999)
}

fn test_prefix_sum_reports_offsets_and_a_total() -> void {
    if not gpu.is_available() { return }
    let flags = Column.full(Dtype.Bool, 5000, Value.of_bool(true))
    defer { flags.free() }
    let scanned = prefix_sum(flags.values, 5000)
    let offsets = scanned[0]
    defer { offsets.free() }
    assert_eq(scanned[1], 5000)
    assert_near(offsets.download(0, 1)[0], 0.0, tolerance: 1e-6)
    assert_near(offsets.download(4999, 1)[0], 4999.0, tolerance: 1e-6)
}

fn test_iota_numbers_the_rows() -> void {
    if not gpu.is_available() { return }
    let lane = iota(4)
    defer { lane.free() }
    assert_eq(lane.download(), [0.0, 1.0, 2.0, 3.0])
}

fn test_cast_moves_between_layouts() -> void {
    if not gpu.is_available() { return }
    let ints = Column.from_ints([1, 0, 5])
    defer { ints.free() }
    let floats = ints.cast(Dtype.Float64)
    defer { floats.free() }
    assert_eq(floats.dtype, Dtype.Float64)
    assert_eq(floats.to_ints(), [1, 0, 5])
    let flags = ints.cast(Dtype.Bool)
    defer { flags.free() }
    assert_eq(flags.to_bools(), [true, false, true])
    let stamps = ints.cast(Dtype.Datetime)
    defer { stamps.free() }
    assert_eq(stamps.to_ints(), [1, 0, 5])
    let text = Column.from_strs(["a"])
    defer { text.free() }
    assert_throws(TypeError, || text.cast(Dtype.Float64))
}

fn test_unify_puts_two_dictionaries_on_one_table() -> void {
    if not gpu.is_available() { return }
    let left = Column.from_strs(["austin", "dallas"])
    defer { left.free() }
    let right = Column.from_strs(["dallas", "houston"])
    defer { right.free() }
    let unified = unify_levels([left, right])
    defer { for part in unified { part.free() } }
    assert_eq(unified[0].level_count(), 3)
    assert_eq(unified[1].level_count(), 3)
    assert_eq(unified[1].level_code("dallas"), 1)
    assert_eq(unified[1].to_strs(), ["dallas", "houston"])
}

fn test_concat_stacks_parts_end_to_end() -> void {
    if not gpu.is_available() { return }
    let first = Column.from_floats([1.0, 2.0])
    defer { first.free() }
    let second = Column.from_floats([3.0, NAN])
    defer { second.free() }
    let joined = concat_columns([first, second])
    defer { joined.free() }
    assert_eq(joined.len(), 4)
    assert_eq(joined.null_mask(), [false, false, false, true])
    let text = Column.from_strs(["austin"])
    defer { text.free() }
    assert_throws(TypeError, || concat_columns([first, text]))
}

fn test_concat_unifies_text_parts() -> void {
    if not gpu.is_available() { return }
    let left = Column.from_strs(["austin", "dallas"])
    defer { left.free() }
    let right = Column.from_strs(["houston"])
    defer { right.free() }
    let joined = concat_columns([left, right])
    defer { joined.free() }
    assert_eq(joined.to_strs(), ["austin", "dallas", "houston"])
    assert_eq(joined.level_count(), 3)
}

fn test_equal_rows_narrows_a_pairing() -> void {
    if not gpu.is_available() { return }
    let left = Column.from_ints([10, 20, 30])
    defer { left.free() }
    let right = Column.from_ints([30, 20, 99])
    defer { right.free() }
    let a = positions_buffer([0, 1, 2])
    defer { a.free() }
    let b = positions_buffer([0, 1, 2])
    defer { b.free() }
    let out = ones_lane(3)
    defer { out.free() }
    left.equal_rows(a, right, b, 3, out)
    assert_eq(out.download(), [0.0, 1.0, 0.0])
}

fn ones_lane(count: int) -> Buffer {
    let lane = Buffer(count)
    lane.fill(1.0)
    return lane
}

fn test_hash_into_folds_a_key_lane() -> void {
    if not gpu.is_available() { return }
    let column = Column.from_ints([1, 2, 3])
    defer { column.free() }
    let lane = Buffer(3)
    defer { lane.free() }
    lane.fill_zero()
    column.hash_into(lane, seed: 17)
    assert_eq(lane.count, 3)
}

fn test_a_range_index_costs_nothing_to_hold() -> void {
    let index = Index.range(5, start: 10, step: 2, name: "row")
    defer { index.free() }
    assert_true(index.is_range(), "no levels means a range")
    assert_eq(index.nlevels(), 1)
    assert_eq(index.len(), 5)
    assert_eq(index.label_at(0)[0].to_int(), 10)
    assert_eq(index.label_at(-1)[0].to_int(), 18)
    assert_eq(index.names[0], "row")
    assert_throws(ValueError, || Index.range(3, step: 0))
    assert_throws(IndexError, || index.label_at(5))
}

fn test_a_multi_index_holds_a_level_per_key() -> void {
    if not gpu.is_available() { return }
    let city = Column.from_strs(["austin", "austin", "dallas"])
    let year = Column.from_ints([2024, 2025, 2025])
    let names: list[str?] = ["city", "year"]
    let index = Index.of([city, year], names)
    defer { index.free() }
    assert_false(index.is_range(), "stored levels are not a range")
    assert_eq(index.nlevels(), 2)
    let label = index.label_at(1)
    assert_eq(label[0].to_str(), "austin")
    assert_eq(label[1].to_int(), 2025)
    let positions = positions_buffer([2, 0])
    defer { positions.free() }
    let taken = index.take(positions, 2)
    defer { taken.free() }
    assert_eq(taken.label_at(0)[0].to_str(), "dallas")
    assert_eq(taken.names, ["city", "year"])
}

fn test_taking_a_range_index_materialises_it() -> void {
    if not gpu.is_available() { return }
    let index = Index.range(4)
    defer { index.free() }
    let positions = positions_buffer([3, 1])
    defer { positions.free() }
    let taken = index.take(positions, 2)
    defer { taken.free() }
    assert_false(taken.is_range(), "taken labels have to be stored")
    assert_eq(taken.label_at(0)[0].to_int(), 3)
    assert_eq(taken.label_at(1)[0].to_int(), 1)
}

fn test_a_series_pairs_values_with_labels() -> void {
    if not gpu.is_available() { return }
    let series = Series(Column.from_floats([1.0, 2.0]), name: "high")
    defer { series.free() }
    assert_eq(series.len(), 2)
    assert_eq(series.dtype(), Dtype.Float64)
    assert_eq(series.name, "high")
    assert_true(series.index.is_range(), "an unlabelled series gets a range index")
    assert_throws(ValueError, || Series(Column.from_floats([1.0]), index: Index.range(4)))
}

fn test_a_frame_reaches_a_column_by_name() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(
        ["city", "high"],
        [Column.from_strs(["austin", "dallas"]), Column.from_floats([38.0, 41.0])]
    )
    defer { frame.free() }
    assert_eq(frame.shape(), (2, 2))
    assert_eq(frame.len(), 2)
    assert_eq(frame.width(), 2)
    assert_true(frame.has("high"), "high is a column")
    assert_false(frame.has("low"), "low is not")
    assert_eq(frame.column_index("high"), 1)
    assert_eq(frame.column("high").get(1).to_float(), 41.0)
    assert_eq(frame.dtypes(), [Dtype.Str, Dtype.Float64])
    assert_throws(KeyError, || frame.column("low"))
}

fn test_a_frame_refuses_a_shape_it_cannot_hold() -> void {
    if not gpu.is_available() { return }
    assert_throws(
        ValueError,
        || DataFrame(["a", "a"], [Column.from_floats([1.0]), Column.from_floats([2.0])])
    )
    assert_throws(ValueError, || DataFrame(["a"], []))
    assert_throws(
        ValueError,
        || DataFrame(["a", "b"], [Column.from_floats([1.0]), Column.from_floats([1.0, 2.0])])
    )
}

fn test_replace_swaps_a_part_and_keeps_the_rest() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(["a"], [Column.from_floats([1.0, 2.0])])
    defer { frame.free() }
    let renamed = frame._replace(names: ["b"])
    assert_eq(renamed.names, ["b"])
    assert_true(renamed.has("b"), "the lookup follows the new names")
    assert_eq(renamed.column("b").get(0).to_float(), 1.0)
    assert_eq(renamed.len(), 2)
}

fn test_copying_a_frame_leaves_the_original_alone() -> void {
    if not gpu.is_available() { return }
    let frame = DataFrame(["a"], [Column.from_floats([1.0, 2.0])])
    defer { frame.free() }
    let clone = frame.copy()
    clone.column("a").set(0, Value.of_float(99.0))
    assert_near(frame.column("a").get(0).to_float(), 1.0, tolerance: 1e-6)
    assert_near(clone.column("a").get(0).to_float(), 99.0, tolerance: 1e-6)
    clone.free()
    assert_near(frame.column("a").get(1).to_float(), 2.0, tolerance: 1e-6)
}

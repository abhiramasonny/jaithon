#: Tests for jainum's array core: the dtype tags, the layout arithmetic, the
#: view rules of `__getitem__`, and the universal kernels every other module
#: builds on.
#:
#: The layout helpers are pure and run without a device. Everything that
#: touches `storage` returns early when there is no GPU, the way jaicv's suite
#: does.
import std.gpu as gpu
from std.test import assert_eq, assert_near, assert_throws, assert_true
from jainum.array import BOOL, COMPLEX64, FLOAT32, FLOAT64, INT32, INT64
from jainum.array import ELLIPSIS, GROUP, MAX_EXACT_INT, NEWAXIS, NDArray, Slice
from jainum.array import broadcast_pair, broadcast_shapes, broadcast_strides
from jainum.array import contiguous_strides, copy_into, diagonal_view, dtype_is_bool
from jainum.array import dtype_is_complex, dtype_is_integer, dtype_lanes, dtype_name
from jainum.array import element_count, exclusive_scan, f32_bits, f32_from_bits, fill
from jainum.array import gather_flat, is_contiguous, layout_buffer, normalize_axes
from jainum.array import normalize_axis, promote, require_real, scatter_flat, sl

fn have_gpu() -> bool { return gpu.is_available() }

fn ramp(shape: list[int], dtype: int = FLOAT32) -> NDArray {
    let a = NDArray(shape, dtype)
    a.upload([float(index) for index in 0..element_count(shape) * dtype_lanes(dtype)])
    return a
}

fn test_dtype_is_a_tag_and_not_a_width() -> void {
    assert_eq(dtype_name(FLOAT32), "float32")
    assert_eq(dtype_name(COMPLEX64), "complex64")
    assert_eq(dtype_lanes(COMPLEX64), 2, "complex is the only multi-lane dtype")
    assert_eq(dtype_lanes(FLOAT64), 1, "float64 is an alias, not a wider element")
    assert_true(dtype_is_integer(INT64))
    assert_true(dtype_is_bool(BOOL))
    assert_true(dtype_is_complex(COMPLEX64))
    assert_eq(MAX_EXACT_INT, 16777216, "the point past which an integer dtype stops being exact")
}

fn test_promotion_is_the_lattice_the_constants_spell() -> void {
    assert_eq(promote(BOOL, INT32), INT32)
    assert_eq(promote(INT64, FLOAT32), FLOAT32)
    assert_eq(promote(FLOAT64, COMPLEX64), COMPLEX64)
    assert_eq(promote(BOOL, BOOL), BOOL)
}

fn test_layout_arithmetic() -> void {
    assert_eq(element_count([]), 1, "a 0-d array holds one element")
    assert_eq(element_count([3, 0, 4]), 0)
    assert_eq(contiguous_strides([2, 3, 4], 1), [12, 4, 1])
    assert_eq(contiguous_strides([2, 3], 2), [6, 2], "complex strides count both lanes")
    assert_true(is_contiguous([2, 3], [3, 1], 1))
    assert_true(not is_contiguous([2, 3], [1, 2], 1))

    #: A length-one axis carries a stride nothing reads, so a stale one must
    #: not cost a reshape its copy.
    assert_true(is_contiguous([2, 1, 3], [3, 99, 1], 1))
}

fn test_axis_names_normalise_and_reject() -> void {
    assert_eq(normalize_axis(-1, 3), 2)
    assert_eq(normalize_axes([-1, 0], 3), [2, 0])
    assert_throws(ValueError, || normalize_axis(3, 3))
    assert_throws(ValueError, || normalize_axes([1, -2], 3), "the same axis twice")
}

fn test_broadcasting_pads_from_the_right() -> void {
    assert_eq(broadcast_shapes([3, 1], [4]), [3, 4])
    assert_eq(broadcast_shapes([], [2, 2]), [2, 2])
    assert_throws(ValueError, || broadcast_shapes([3], [4]))

    #: A stretched axis reads one slot from every thread, which is what makes
    #: broadcasting free.
    assert_eq(broadcast_strides([3, 1], [1, 1], [2, 3, 4]), [0, 1, 0])
}

fn test_slices_resolve_the_way_python_ones_do() -> void {
    let (first, count) = Slice(1, 4).resolve(10)
    assert_eq(first, 1)
    assert_eq(count, 3)

    let (from_end, back) = Slice(null, null, -1).resolve(4)
    assert_eq(from_end, 3, "a reversed slice starts at the last element")
    assert_eq(back, 4)

    let (clipped, over) = Slice(-2, 99).resolve(5)
    assert_eq(clipped, 3)
    assert_eq(over, 2)

    let (_start, empty) = Slice(4, 2).resolve(10)
    assert_eq(empty, 0, "a forward slice that runs backwards is empty")
    assert_throws(ValueError, || Slice(0, 1, 0))
}

fn test_a_range_key_carries_its_own_inclusivity() -> void {
    if not have_gpu() { return }
    let a = ramp([6])
    defer { a.free() }
    assert_eq(a[0..3].to_list(), [0.0, 1.0, 2.0])
    assert_eq(a[0..=3].to_list(), [0.0, 1.0, 2.0, 3.0], "..= takes one more")
    assert_eq(a[4..4].to_list(), [], "an empty range selects nothing")
    assert_eq(a[range(1, 6, 2)].to_list(), [1.0, 3.0, 5.0])
}

fn test_bits_round_trip_through_a_uint_scalar() -> void {
    for value in [0.0, 1.0, -2.5, 3.4028234663852886e38, 1.0e-30] {
        assert_near(f32_from_bits(f32_bits(value)), value, math_scale(value), f"{value}")
    }
    assert_eq(f32_bits(-0.0), 1 << 31, "the sign bit survives a zero")
}

fn math_scale(value: float) -> float {
    let magnitude = if value < 0.0 { -value } else { value }
    return magnitude * 1.0e-6 + 1.0e-30
}

fn test_shape_and_size_report_the_element_view() -> void {
    if not have_gpu() { return }
    let a = NDArray([2, 3], COMPLEX64)
    defer { a.free() }
    assert_eq(a.size(), 6)
    assert_eq(a.ndim(), 2)
    assert_eq(a.lanes(), 2)
    assert_eq(a.itemsize(), 8, "two float32 lanes")
    assert_eq(a.strides, [6, 2], "strides are float slots, not elements")
    assert_eq(a.buffer().count, 12)
}

fn test_values_go_in_and_come_back_nested() -> void {
    if not have_gpu() { return }
    let a = ramp([2, 3])
    defer { a.free() }
    assert_eq(a.to_list(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert_eq(a.to_nested(), [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    assert_eq(a[1, 2].item(), 5.0)
    assert_throws(ValueError, || a.item(), "item() needs exactly one element")
}

fn test_a_basic_key_makes_a_view_and_not_a_copy() -> void {
    if not have_gpu() { return }
    let a = ramp([3, 4])
    defer { a.free() }

    let row = a[1]
    assert_eq(row.shape, [4], "an int index drops its axis")
    assert_eq(row.to_list(), [4.0, 5.0, 6.0, 7.0])

    let block = a[0..2, 1..3]
    assert_eq(block.shape, [2, 2])
    assert_eq(block.to_list(), [1.0, 2.0, 5.0, 6.0])

    #: The point of a view: a write through the window lands in the parent.
    block[0, 0] = 99.0
    assert_eq(a[0, 1].item(), 99.0)
}

fn test_negative_steps_and_sentinels() -> void {
    if not have_gpu() { return }
    let a = ramp([2, 4])
    defer { a.free() }

    let reversed = a[ELLIPSIS, sl(null, null, -1)]
    assert_eq(reversed.to_list(), [3.0, 2.0, 1.0, 0.0, 7.0, 6.0, 5.0, 4.0])
    assert_true(reversed.strides[1] < 0, "a flip is a negative stride, not a copy")

    let lifted = a[NEWAXIS, 0]
    assert_eq(lifted.shape, [1, 4])
    assert_eq(a[-1].to_list(), [4.0, 5.0, 6.0, 7.0])
    assert_throws(IndexError, || a[9])
    assert_throws(IndexError, || a[0, 1, 2], "more axes in the key than the array has")
}

fn test_a_strided_view_still_arithmetics() -> void {
    if not have_gpu() { return }
    let a = ramp([3, 4])
    defer { a.free() }

    #: A transposed view has no dense entry point anywhere, so this is the
    #: general layout-buffer path end to end.
    let transposed = a.view([4, 3], [a.strides[1], a.strides[0]], a.offset)
    let doubled = transposed + transposed
    defer { doubled.free() }
    assert_eq(doubled.shape, [4, 3])
    assert_eq(
        doubled.to_list(),
        [0.0, 8.0, 16.0, 2.0, 10.0, 18.0, 4.0, 12.0, 20.0, 6.0, 14.0, 22.0]
    )
}

fn test_the_operators_broadcast_and_take_scalars() -> void {
    if not have_gpu() { return }
    let a = ramp([2, 3])
    let column = NDArray([2, 1])
    column.upload([10.0, 20.0])
    defer { a.free() }
    defer { column.free() }

    let summed = a + column
    defer { summed.free() }
    assert_eq(summed.to_list(), [10.0, 11.0, 12.0, 23.0, 24.0, 25.0])

    let scaled = a * 2.0
    defer { scaled.free() }
    assert_eq(scaled.to_list(), [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])

    let shifted = a - 1
    defer { shifted.free() }
    assert_eq(shifted.to_list(), [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])

    let negated = -a
    defer { negated.free() }
    assert_eq(negated.to_list()[3], -3.0)

    #: Division leaves the integers even when both sides are whole, as numpy's
    #: true division does.
    let ints = ramp([3], INT32)
    defer { ints.free() }
    let halved = ints / 2
    defer { halved.free() }
    assert_eq(halved.dtype, FLOAT32)
    assert_eq(halved.to_list(), [0.0, 0.5, 1.0])
}

fn test_there_is_no_elementwise_equality() -> void {
    if not have_gpu() { return }
    let a = ramp([2])
    defer { a.free() }

    #: Deliberately absent: an `__eq__` returning an array would break
    #: `assert_eq`, every `is not null` guard, and dict membership. Identity is
    #: what `==` means here, and `equal` in `ufunc` is the elementwise spelling.
    assert_true(a == a)
    assert_true(not a == ramp([2]))
}

fn test_casting_rounds_and_saturates() -> void {
    if not have_gpu() { return }
    let a = NDArray([5])
    a.upload([-3.0e9, -0.6, 0.5, 2.5, 3.0e9])
    defer { a.free() }

    let whole = a.astype(INT32)
    defer { whole.free() }

    #: Round to nearest even, then clamp -- what an integer dtype means on
    #: float32 storage.
    #: The ceiling is the largest float32 under 2^31, not 2^31 - 1: the latter
    #: is not a float32 and rounds up past the type it is meant to bound.
    assert_eq(whole.to_list(), [-2147483648.0, -1.0, 0.0, 2.0, 2147483520.0])

    let flags = a.astype(BOOL)
    defer { flags.free() }
    assert_eq(flags.to_list(), [1.0, 1.0, 1.0, 1.0, 1.0])

    let zeroed = NDArray([2], BOOL)
    zeroed.upload([0.0, 0.0])
    defer { zeroed.free() }
    assert_eq(zeroed.astype(FLOAT32).to_list(), [0.0, 0.0])
}

fn test_complex_lanes_are_views_of_the_same_slots() -> void {
    if not have_gpu() { return }
    let z = NDArray([2], COMPLEX64)
    z.upload([1.0, 2.0, 3.0, -1.0])
    defer { z.free() }

    #: The real and imaginary views keep the parent's strides untouched --
    #: strides count float slots, so a complex stride already steps both lanes.
    assert_eq(z.real().to_list(), [1.0, 3.0])
    assert_eq(z.imag().to_list(), [2.0, -1.0])
    assert_eq(z.real().strides, z.strides)

    let squared = z * z
    defer { squared.free() }
    assert_eq(squared.to_list(), [-3.0, 4.0, 8.0, -6.0])

    let quotient = z / z
    defer { quotient.free() }
    assert_near(quotient.to_list()[0], 1.0, 1.0e-5)
    assert_near(quotient.to_list()[1], 0.0, 1.0e-5)

    #: A real scalar is a complex number with no imaginary part, so it lands on
    #: one lane for a sum and on both for a product.
    let raised = z + 1.0
    defer { raised.free() }
    assert_eq(raised.to_list(), [2.0, 2.0, 4.0, -1.0])

    assert_throws(TypeError, || require_real(z, "a real-only kernel"))
}

fn test_real_to_complex_fills_the_imaginary_lane() -> void {
    if not have_gpu() { return }
    let a = ramp([3])
    defer { a.free() }
    let lifted = a.astype(COMPLEX64)
    defer { lifted.free() }
    assert_eq(lifted.to_list(), [0.0, 0.0, 1.0, 0.0, 2.0, 0.0])
    assert_eq(lifted.astype(FLOAT32).to_list(), [0.0, 1.0, 2.0], "and back, dropping it")
}

fn test_fill_and_copy_into_broadcast() -> void {
    if not have_gpu() { return }
    let a = NDArray([2, 3])
    fill(a, 7.0)
    defer { a.free() }
    assert_eq(a.to_list(), [7.0, 7.0, 7.0, 7.0, 7.0, 7.0])

    let row = NDArray([3])
    row.upload([1.0, 2.0, 3.0])
    defer { row.free() }
    copy_into(a, row)
    assert_eq(a.to_list(), [1.0, 2.0, 3.0, 1.0, 2.0, 3.0])

    #: A fill through a strided window touches only that window.
    fill(a[ELLIPSIS, 0..1], 0.0)
    assert_eq(a.to_list(), [0.0, 2.0, 3.0, 0.0, 2.0, 3.0])
}

fn test_gather_and_scatter_wrap_negative_indices() -> void {
    if not have_gpu() { return }
    let source = ramp([5])
    defer { source.free() }
    let keys = NDArray([3], INT32)
    keys.upload([4.0, 0.0, -1.0])
    defer { keys.free() }

    let taken = gather_flat(source, keys)
    defer { taken.free() }
    assert_eq(taken.to_list(), [4.0, 0.0, 4.0])

    let target = NDArray([5])
    fill(target, 0.0)
    defer { target.free() }
    let values = NDArray([3])
    values.upload([9.0, 8.0, 7.0])
    defer { values.free() }
    #: Distinct slots on purpose: two indices naming one slot race, and this
    #: is a thread grid, not numpy's left-to-right loop.
    let distinct = NDArray([3], INT32)
    distinct.upload([4.0, 0.0, -3.0])
    defer { distinct.free() }
    scatter_flat(target, distinct, values)
    assert_eq(target.to_list(), [8.0, 0.0, 7.0, 0.0, 9.0])
}

fn test_a_bool_mask_compacts_and_a_masked_write_gates() -> void {
    if not have_gpu() { return }
    let a = ramp([3, 4])
    defer { a.free() }
    let mask = NDArray([3, 4], BOOL)
    mask.upload([float(index % 3 == 0) for index in 0..12])
    defer { mask.free() }

    let picked = a[mask]
    defer { picked.free() }
    assert_eq(picked.to_list(), [0.0, 3.0, 6.0, 9.0])

    a[mask] = -1.0
    assert_eq(a.to_list(), [-1.0, 1.0, 2.0, -1.0, 4.0, 5.0, -1.0, 7.0, 8.0, -1.0, 10.0, 11.0])
}

fn test_an_integer_key_gathers_from_a_flat_array() -> void {
    if not have_gpu() { return }
    let a = ramp([5])
    defer { a.free() }
    let keys = NDArray([2], INT32)
    keys.upload([3.0, 1.0])
    defer { keys.free() }
    assert_eq(a[keys].to_list(), [3.0, 1.0])

    a[keys] = 0.0
    assert_eq(a.to_list(), [0.0, 0.0, 2.0, 0.0, 4.0])

    let plane = ramp([2, 2])
    defer { plane.free() }
    assert_throws(IndexError, || plane[keys], "an axis-wise take belongs to index.jai")
}

#: The scan is what every compaction in the package is built on, so it is
#: checked past one threadgroup: 100000 elements is 391 groups, which also puts
#: the middle pass over its own 256 threads.
fn test_the_exclusive_scan_crosses_group_boundaries() -> void {
    if not have_gpu() { return }
    let count = 100000
    let mask = NDArray([count], BOOL)
    mask.upload([float(index % 7 == 0) for index in 0..count])
    defer { mask.free() }

    let (offsets, total) = exclusive_scan(mask)
    defer { offsets.free() }
    assert_eq(total, (count + 6) // 7)

    let seen = offsets.to_list()
    assert_eq(seen[0], 0.0)
    assert_eq(seen[1], 1.0, "the offset is exclusive, so element 1 sits past element 0")
    assert_eq(seen[7], 1.0)
    #: The last element is not itself a hit, so its exclusive offset is the
    #: whole total.
    assert_eq(seen[count - 1], float(total))

    let values = NDArray([count], FLOAT32)
    values.upload([float(index) for index in 0..count])
    defer { values.free() }
    let packed = values[mask]
    defer { packed.free() }
    assert_eq(packed.size(), total)
    assert_eq(packed.to_list()[3], 21.0)
}

fn test_a_diagonal_is_a_view_with_a_summed_stride() -> void {
    if not have_gpu() { return }
    let a = ramp([3, 4])
    defer { a.free() }

    let main = diagonal_view(a)
    assert_eq(main.to_list(), [0.0, 5.0, 10.0])
    assert_eq(main.strides, [5], "the row stride plus the column stride")

    assert_eq(diagonal_view(a, 1).to_list(), [1.0, 6.0, 11.0])
    assert_eq(diagonal_view(a, -1).to_list(), [4.0, 9.0])
    assert_eq(diagonal_view(a, 9).to_list(), [], "an offset past the corner is empty")
    assert_throws(ValueError, || diagonal_view(a, 0, 1, 1))
}

fn test_broadcast_pair_returns_two_views() -> void {
    if not have_gpu() { return }
    let a = ramp([3, 1])
    let b = ramp([4])
    defer { a.free() }
    defer { b.free() }
    let (left, right) = broadcast_pair(a, b)
    assert_eq(left.shape, [3, 4])
    assert_eq(right.shape, [3, 4])
    assert_eq(left.strides[1], 0, "a stretched axis reads one slot")
    assert_eq(right.strides[0], 0)
}

fn test_the_layout_buffer_holds_rank_shape_and_every_operand() -> void {
    if not have_gpu() { return }
    let a = ramp([2, 3])
    defer { a.free() }
    let meta = layout_buffer([a, a])
    defer { meta.free() }

    #: rank, shape, then offset and strides per operand.
    assert_eq(meta.count, 1 + 2 + 2 * 3)
    assert_eq(meta.download(), [2.0, 2.0, 3.0, 0.0, 3.0, 1.0, 0.0, 3.0, 1.0])

    let flipped = a[ELLIPSIS, sl(null, null, -1)]
    let signed = layout_buffer([flipped])
    defer { signed.free() }
    assert_eq(signed.download()[5], -1.0, "a stride is signed, which a uint scalar could not be")
    assert_throws(ValueError, || layout_buffer([a, a[0]]), "operands must share a rank")
}

fn test_a_view_does_not_own_the_allocation() -> void {
    if not have_gpu() { return }
    let a = ramp([4])
    let window = a[1..3]
    window.free()
    assert_true(window.is_freed())
    assert_eq(a.to_list(), [0.0, 1.0, 2.0, 3.0], "the parent survives its view")
    a.free()
    assert_true(a.is_freed())
    a.free()
}

fn test_printing_a_big_array_downloads_a_corner() -> void {
    if not have_gpu() { return }
    let small = ramp([2, 2])
    defer { small.free() }
    assert_eq(f"{small}", "array([[0.0, 1.0], [2.0, 3.0]], dtype=float32)")

    let big = NDArray([64, 64])
    fill(big, 1.0)
    defer { big.free() }
    let rendered = f"{big}"
    assert_true(rendered.contains("shape=[64, 64]"))
    assert_true(rendered.contains("corner="), "a big array never downloads all of itself")
}

fn test_group_is_the_width_every_folding_kernel_expects() -> void {
    assert_eq(GROUP, 256, "eight simdgroups, which the scan's per-warp table is sized for")
}

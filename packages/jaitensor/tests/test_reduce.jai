#: The row reductions, at the widths where a group has to stride and combine partials.
import jaitensor as jt
import std.math
from jaitensor.tensor import Tensor
from std.test import assert_eq, assert_near, assert_throws

#: `amax` is the companion to `argmax`: where `argmax` says which element of a
#: row is largest, `amax` says what it is. A detector wants both, and working
#: the value out on the host afterwards means downloading everything the
#: reduction was there to avoid.
fn test_amax_reduces_the_last_axis_and_the_whole_tensor() -> void {
    if not jt.is_available() { return }
    let values = [3.0, -1.0, 2.5, 0.0, -4.0, -0.5, 7.25, 1.0, 6.0, 6.0, -2.0, 5.5]
    let x = Tensor([3, 4], values)
    let along = jt.amax(x, axis: -1).download()
    assert_eq(along.len(), 3)
    assert_near(along[0], 3.0, 1e-6)
    assert_near(along[1], 7.25, 1e-6)
    assert_near(along[2], 6.0, 1e-6)

    #: A tie keeps the first, the same rule `argmax` follows.
    let where = jt.argmax(x).download()
    assert_eq(int(where[2]), 0)

    let whole = jt.amax(x).download()
    assert_eq(whole.len(), 1)
    assert_near(whole[0], 7.25, 1e-6)

    #: A single column is still a reduction, of one element each.
    let column = Tensor([4, 1], [1.0, 2.0, 3.0, 4.0])
    let each = jt.amax(column, axis: -1).download()
    assert_eq(each, [1.0, 2.0, 3.0, 4.0])

    let _refused = assert_throws(ValueError, || jt.amax(x, axis: 0))
    x.free()
    column.free()
}

#: `max_with_index` agrees with asking for the two separately.
fn test_max_with_index_matches_amax_and_argmax() -> void {
    if not jt.is_available() { return }
    var values: list[float] = []
    for index in 0..64 * 40 {
        #: Every row has its peak somewhere different, and negatives so that a
        #: zero-initialised answer would show.
        values.push(float(index * 37 % 79 - 40) * 0.25)
    }
    let input = Tensor([64, 40], values)
    defer { input.free() }
    let (peaks, wheres) = jt.max_with_index(input)
    defer { peaks.free() }
    defer { wheres.free() }
    let apart_values = jt.amax(input, axis: -1)
    defer { apart_values.free() }
    let apart_indices = jt.argmax(input)
    defer { apart_indices.free() }
    assert_eq(peaks.download(), apart_values.download())
    assert_eq(wheres.download(), apart_indices.download())
}

fn test_max_with_index_refuses_anything_but_rank_two() -> void {
    if not jt.is_available() { return }
    let flat = Tensor([8])
    defer { flat.free() }
    assert_throws(ValueError, || jt.max_with_index(flat))
}

#: Reducing down the columns gives what transposing and reducing along the rows
#: gives, which is the pass it exists to avoid.
fn test_max_with_index_down_the_columns_matches_a_transpose() -> void {
    if not jt.is_available() { return }
    var values: list[float] = []
    for index in 0..24 * 57 { values.push(float(index * 41 % 91 - 45) * 0.125) }
    let input = Tensor([24, 57], values)
    defer { input.free() }
    let (peaks, wheres) = jt.max_with_index(input, axis: 0)
    defer { peaks.free() }
    defer { wheres.free() }
    let flipped = jt.transpose(input)
    defer { flipped.free() }
    let (want_peaks, want_wheres) = jt.max_with_index(flipped, axis: -1)
    defer { want_peaks.free() }
    defer { want_wheres.free() }
    assert_eq(peaks.shape, [57])
    assert_eq(peaks.download(), want_peaks.download())
    assert_eq(wheres.download(), want_wheres.download())
}

fn test_max_with_index_refuses_an_axis_it_has_no_kernel_for() -> void {
    if not jt.is_available() { return }
    let input = Tensor([4, 5])
    defer { input.free() }
    assert_throws(ValueError, || jt.max_with_index(input, axis: 2))
}

#: A row wider than the threadgroup, checked against arithmetic done here.
#:
#: The kernel spreads a row across 256 threads and brings the two sums back
#: through a simd reduction, so a row narrower than that never exercises the
#: striding or the second stage of the reduction. Widths either side of 256 and
#: of a warp are the ones that catch a reduction wired up wrongly.
fn test_layer_norm_matches_arithmetic_over_wide_rows() -> void {
    if not jt.is_available() { return }
    for width in [17, 32, 33, 256, 257, 768] {
        let rows = 3
        var values: list[float] = []
        for index in 0..rows * width {
            values.push(float(index * 31 % 97 - 48) * 0.125)
        }
        var scales: list[float] = []
        var shifts: list[float] = []
        for index in 0..width {
            scales.push(1.0 + float(index % 5) * 0.25)
            shifts.push(float(index % 7) * 0.5 - 1.5)
        }
        let x = jt.Tensor([rows, width], values)
        let gamma = jt.Tensor([width], scales)
        let beta = jt.Tensor([width], shifts)
        let y = jt.layer_norm(x, gamma, beta)
        let got = y.download()
        for row in 0..rows {
            var total = 0.0
            for column in 0..width { total += values[row * width + column] }
            let mean = total / float(width)
            var square = 0.0
            for column in 0..width {
                let gap = values[row * width + column] - mean
                square += gap * gap
            }
            let inverse = 1.0 / math.sqrt(square / float(width) + 1e-5)
            for column in 0..width {
                let centred = values[row * width + column] - mean
                let want = scales[column] * centred * inverse + shifts[column]
                assert_near(
                    got[row * width + column],
                    want,
                    1e-3,
                    f"width {width} at {row},{column}"
                )
            }
        }
        x.free()
        gamma.free()
        beta.free()
        y.free()
    }
}

#: A row wider than the threadgroup that reduces it.
#:
#: These reductions used to run one thread to a row, so a row of any width was
#: one thread's serial walk and the width never mattered. Now the group strides
#: the row, and a row of 1000 takes four passes with a partial per thread to
#: combine -- which is the case a row of four never reaches.
fn test_wide_rows_reduce_the_same_as_narrow_ones() -> void {
    if not jt.is_available() { return }
    let rows = 3
    let columns = 1000
    var values: list[float] = []
    for row in 0..rows {
        for column in 0..columns {
            values.push(float((row * 7 + column * 13) % 101) * 0.01)
        }
    }
    #: One unmistakable peak per row, past where a single group would stop.
    values[0 * columns + 700] = 9.5
    values[1 * columns + 999] = 9.5
    values[2 * columns + 0] = 9.5
    let x = jt.Tensor([rows, columns], values)
    defer { x.free() }

    let totals = jt.sum(x, axis: -1).download()
    let averages = jt.mean(x, axis: -1).download()
    let peaks = jt.amax(x, axis: -1).download()
    let where = jt.argmax(x).download()
    let (also_peaks, also_where) = jt.max_with_index(x)
    defer { also_peaks.free() }
    defer { also_where.free() }
    let peak_values = also_peaks.download()
    let peak_places = also_where.download()

    let wanted = [700.0, 999.0, 0.0]
    for row in 0..rows {
        var total = 0.0
        for column in 0..columns { total += values[row * columns + column] }
        assert_near(totals[row], total, total * 1e-5 + 1e-4, f"row {row} total")
        assert_near(averages[row], total / float(columns), 1e-5, f"row {row} mean")
        assert_near(peaks[row], 9.5, 1e-6, f"row {row} peak")
        assert_near(where[row], wanted[row], 1e-6, f"row {row} peak position")
        assert_near(peak_values[row], 9.5, 1e-6, f"row {row} peak, together")
        assert_near(peak_places[row], wanted[row], 1e-6, f"row {row} position, together")
    }
}

#: Every element equal: the first one wins, whichever thread happened to see it.
fn test_a_wide_row_of_ties_keeps_the_first() -> void {
    if not jt.is_available() { return }
    let columns = 700
    var values: list[float] = []
    for _column in 0..columns { values.push(2.5) }
    let x = jt.Tensor([1, columns], values)
    defer { x.free() }
    assert_near(jt.argmax(x).download()[0], 0.0, 1e-6, "a row of ties should keep the first")
    let (peak, place) = jt.max_with_index(x)
    defer { peak.free() }
    defer { place.free() }
    assert_near(peak.download()[0], 2.5, 1e-6)
    assert_near(place.download()[0], 0.0, 1e-6, "a row of ties should keep the first")
}

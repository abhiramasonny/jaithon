#: Tests for jainum's reductions and scans.
#:
#: Every device answer here is checked against the same fold run a second time
#: on the host, over the floats `to_list` hands back — never against a number
#: this module once printed. That matters more here than elsewhere: the kernel
#: reshapes the work five different ways depending on how long the reduced
#: axis is, and a golden number would pass all five while only one of them was
#: right. `test_the_grid_shape_is_invisible_from_outside` is the test that
#: exists for that, and it is the one to look at first when a fold goes wrong.
#:
#: NaN is asserted with `value != value` rather than `assert_near`, which
#: refuses a NaN on either side by design.
import std.gpu as gpu
import std.math
from std.test import assert_eq, assert_near, assert_throws, assert_true
from jainum.array import BOOL, COMPLEX64, FLOAT32, INT32, INT64, NDArray
from jainum.array import element_count, sl
from jainum.reduce import all, allclose, amax, amin, any, argmax, argmin, array_equal
from jainum.reduce import count_nonzero, cummax, cummin, cumprod, cumsum, mean
from jainum.reduce import nanargmax, nanargmin, nanmax, nanmean, nanmin, nanstd
from jainum.reduce import nansum, nanvar, prod, ptp, std, sum, trace, variance
from jainum.reduce import max as fold_max, min as fold_min

fn have_gpu() -> bool { return gpu.is_available() }

fn is_nan(value: float) -> bool { return value != value }

fn holding(shape: list[int], values: list[float], dtype: int = FLOAT32) -> NDArray {
    let a = NDArray(shape, dtype)
    a.upload(values)
    return a
}

fn ramp(shape: list[int], dtype: int = FLOAT32) -> NDArray {
    return holding(shape, [float(index) + 1.0 for index in 0..element_count(shape)], dtype)
}

#: A pseudo-random spread with no pattern a fold could accidentally agree
#: with. Deterministic, so a failure is reproducible.
#:
#: Every value is a multiple of 1/128, which float32 holds exactly. That is
#: what lets the checks on a picked value — a minimum, a maximum, the element
#: an arg-fold names — run at zero tolerance: the host reference and the
#: uploaded copy are the same number, so any difference is the fold's.
fn spread(count: int) -> list[float] {
    var out = [0.0 for _index in 0..count]
    var state = 12345
    for index in 0..count {
        state = (state * 1103515245 + 12345) % 2147483648
        out[index] = float(state % 2000 - 1000) / 128.0
    }
    return out
}

#: A factor near one, so a product of a few dozen of them neither overflows
#: float32 nor collapses to zero.
fn nudge(value: float) -> float { return 1.0 + value / 256.0 }

#: Where `spot` sits in the C-order flattening of `shape`.
fn flat_of(shape: list[int], spot: list[int]) -> int {
    var at = 0
    for axis in 0..shape.len() { at = at * shape[axis] + spot[axis] }
    return at
}

fn step_of(op: str, acc: float, value: float, seen: bool) -> float {
    if op == "sum" { return acc + value }
    if op == "prod" { return acc * value }
    if op == "count" { return if value != 0.0 { acc + 1.0 } else { acc } }
    if op == "min" { return if not seen or value < acc { value } else { acc } }
    if op == "max" { return if not seen or value > acc { value } else { acc } }
    throw ValueError(f"no host fold named {op}")
}

#: `values` folded over `axes` on the host, in the order the kept axes leave
#: behind. Written from the definition of a reduction and sharing no code with
#: the module under test, which is the whole point of it.
fn host_fold(values: list[float], shape: list[int], axes: list[int], op: str) -> list[float] {
    var reduced = [false for _axis in shape]
    for at in axes { reduced[at] = true }
    var kept: list[int] = []
    for axis in 0..shape.len() {
        if not reduced[axis] { kept.push(shape[axis]) }
    }
    let segments = element_count(kept)
    let start = if op == "prod" { 1.0 } else { 0.0 }
    var out = [start for _seg in 0..segments]
    var seen = [false for _seg in 0..segments]
    for flat in 0..element_count(shape) {
        var rest = flat
        var spot = [0 for _axis in shape]
        var axis = shape.len() - 1
        while axis >= 0 {
            spot[axis] = rest % shape[axis]
            rest = rest // shape[axis]
            axis -= 1
        }
        var seg = 0
        for kept_axis in 0..shape.len() {
            if reduced[kept_axis] { continue }
            seg = seg * shape[kept_axis] + spot[kept_axis]
        }
        out[seg] = step_of(op, out[seg], values[flat], seen[seg])
        seen[seg] = true
    }
    return out
}

fn assert_all_near(actual: list[float], expected: list[float], tolerance: float = 1.0e-4) -> void {
    assert_eq(actual.len(), expected.len(), "the fold produced the wrong number of outputs")
    for at in 0..actual.len() {
        assert_near(actual[at], expected[at], tolerance, f"output {at}")
    }
}

fn test_a_fold_is_the_same_fold_run_on_the_host() -> void {
    if not have_gpu() { return }
    let shape = [3, 4, 5]
    let values = spread(60)
    let a = holding(shape, values)
    defer { a.free() }

    let cases = [[0], [1], [2], [0, 1], [1, 2], [0, 2], [0, 1, 2]]
    for axes in cases {
        assert_all_near(sum(a, axes).to_list(), host_fold(values, shape, axes, "sum"))
        assert_all_near(amin(a, axes).to_list(), host_fold(values, shape, axes, "min"))
        assert_all_near(amax(a, axes).to_list(), host_fold(values, shape, axes, "max"))
        assert_all_near(count_nonzero(a, axes).to_list(), host_fold(values, shape, axes, "count"))
    }
    assert_all_near(sum(a).to_list(), host_fold(values, shape, [0, 1, 2], "sum"))

    #: A product needs values near one, or sixty of them overflow float32.
    let near_one = [nudge(values[index]) for index in 0..60]
    let b = holding(shape, near_one)
    defer { b.free() }
    for axes in [[0], [2], [0, 2]] {
        assert_all_near(prod(b, axes).to_list(), host_fold(near_one, shape, axes, "prod"))
    }
}

#: The test this module exists for.
#:
#: One list of values, laid out six ways and folded. Each layout drives the
#: planner down a different road — a lane per output, a group per output, a
#: float4 run, a scalar strided run, and a single segment split over groups
#: with a second pass to join them — and every one of them owes the same
#: total.
fn test_the_grid_shape_is_invisible_from_outside() -> void {
    if not have_gpu() { return }
    let count = 120000
    let values = spread(count)
    var expected = 0.0
    for value in values { expected += value }

    for shape in [[count], [3, 40000], [40000, 3], [1, count], [count, 1], [300, 400]] {
        let a = holding(shape, values)
        defer { a.free() }
        assert_near(sum(a).item(), expected, 0.5, f"a total should not depend on {shape}")
    }

    #: Segment-wise, not just in total: 40000 outputs of three elements each is
    #: the arrangement a thread-per-output kernel gets right and a
    #: group-per-output one starves on, and the reverse for three of 40000.
    let short = holding([40000, 3], values)
    defer { short.free() }
    assert_all_near(sum(short, 1).to_list(), host_fold(values, [40000, 3], [1], "sum"), 1.0e-3)

    let long_axis = holding([3, 40000], values)
    defer { long_axis.free() }
    assert_all_near(sum(long_axis, 1).to_list(), host_fold(values, [3, 40000], [1], "sum"), 0.1)

    #: Every fourth element, which is unit-strided nowhere and so takes the
    #: scalar path over the same numbers.
    let dense = holding([count], values)
    defer { dense.free() }
    let sparse = dense[sl(0, count, 4)]
    var picked: list[float] = []
    var at = 0
    while at < count {
        picked.push(values[at])
        at += 4
    }
    var apart = 0.0
    for value in picked { apart += value }
    assert_near(sum(sparse).item(), apart, 0.2, "a strided fold reads the stride")
    assert_near(amax(sparse).item(), host_fold(picked, [picked.len()], [0], "max")[0])
}

fn test_keepdims_leaves_the_reduced_axes_as_one() -> void {
    if not have_gpu() { return }
    let a = ramp([2, 3, 4])
    defer { a.free() }
    assert_eq(sum(a, 1).shape, [2, 4])
    assert_eq(sum(a, 1, true).shape, [2, 1, 4])
    assert_eq(sum(a, [0, 2], true).shape, [1, 3, 1])
    assert_eq(sum(a, null, true).shape, [1, 1, 1])
    assert_eq(sum(a).shape, [], "a full reduction is 0-d, not 1-d")
    assert_eq(sum(a, 1, true).to_list(), sum(a, 1).to_list(), "keepdims moves no values")
}

fn test_an_axis_can_be_named_several_ways() -> void {
    if not have_gpu() { return }
    let a = ramp([2, 3, 4])
    defer { a.free() }
    assert_eq(sum(a, -1).to_list(), sum(a, 2).to_list())
    assert_eq(sum(a, [0, 2]).to_list(), sum(a, (0, 2)).to_list())
    assert_eq(sum(a, [-3, -1]).to_list(), sum(a, [0, 2]).to_list())

    #: numpy's `axis=()` reduces nothing, which makes the fold a copy.
    let copied = sum(a, [])
    defer { copied.free() }
    assert_eq(copied.shape, [2, 3, 4])
    assert_eq(copied.to_list(), a.to_list())

    assert_throws(ValueError, || sum(a, 3))
    assert_throws(ValueError, || sum(a, [1, 1]), "the same axis twice")
    assert_throws(TypeError, || sum(a, "middle"))
}

fn test_the_accumulator_dtype_follows_numpy() -> void {
    if not have_gpu() { return }
    let flags = holding([5], [1.0, 0.0, 1.0, 1.0, 0.0], BOOL)
    defer { flags.free() }
    let counted = sum(flags)
    defer { counted.free() }
    assert_eq(counted.dtype, INT64, "a boolean total widens rather than saturating")
    assert_eq(counted.item(), 3.0)
    assert_eq(count_nonzero(flags).dtype, INT64)
    assert_eq(any(flags).dtype, BOOL)
    assert_eq(all(flags).dtype, BOOL)
    assert_eq(any(flags).item(), 1.0)
    assert_eq(all(flags).item(), 0.0)

    let ints = holding([4], [3.0, 1.0, 4.0, 1.0], INT32)
    defer { ints.free() }
    assert_eq(sum(ints).dtype, INT64)
    assert_eq(sum(ints).item(), 9.0)
    assert_eq(sum(ints, null, false, FLOAT32).dtype, FLOAT32, "a named dtype wins")
    assert_eq(mean(ints).dtype, FLOAT32, "an average is never an integer")
    assert_near(mean(ints).item(), 2.25)
    assert_eq(amax(ints).dtype, INT32, "an extreme keeps the input's dtype")

    let reals = holding([3], [1.0, 2.0, 3.0])
    defer { reals.free() }
    assert_eq(sum(reals).dtype, FLOAT32, "a float total stays where it started")
}

fn test_mean_and_variance_are_their_definitions() -> void {
    if not have_gpu() { return }
    let shape = [4, 7]
    let values = spread(28)
    let a = holding(shape, values)
    defer { a.free() }

    for axis in 0..2 {
        let totals = host_fold(values, shape, [axis], "sum")
        let n = float(shape[axis])
        let means = [total / n for total in totals]
        assert_all_near(mean(a, axis).to_list(), means)

        #: The variance a second way: the average squared deviation, summed
        #: from the host means rather than from anything the device produced.
        var squares = [0.0 for _seg in means]
        for flat in 0..28 {
            let row = flat // shape[1]
            let column = flat - row * shape[1]
            let seg = if axis == 0 { column } else { row }
            let gap = values[flat] - means[seg]
            squares[seg] += gap * gap
        }
        assert_all_near(variance(a, axis).to_list(), [square / n for square in squares])
        assert_all_near(
            variance(a, axis, false, -1, 1).to_list(),
            [square / (n - 1.0) for square in squares],
            1.0e-4
        )
        assert_all_near(std(a, axis).to_list(), [math.sqrt(square / n) for square in squares])
    }
}

fn test_ptp_is_the_largest_less_the_smallest() -> void {
    if not have_gpu() { return }
    let shape = [5, 6]
    let values = spread(30)
    let a = holding(shape, values)
    defer { a.free() }
    let highs = host_fold(values, shape, [1], "max")
    let lows = host_fold(values, shape, [1], "min")
    assert_all_near(ptp(a, 1).to_list(), [highs[at] - lows[at] for at in 0..highs.len()])
}

fn test_min_and_max_are_the_names_amin_and_amax_answer_to() -> void {
    if not have_gpu() { return }
    let a = ramp([3, 4])
    defer { a.free() }
    assert_eq(fold_min(a, 1).to_list(), amin(a, 1).to_list())
    assert_eq(fold_max(a, 0).to_list(), amax(a, 0).to_list())
}

fn test_arg_folds_point_at_the_extreme_they_found() -> void {
    if not have_gpu() { return }
    let shape = [6, 9]
    let values = spread(54)
    let a = holding(shape, values)
    defer { a.free() }

    let highs = host_fold(values, shape, [1], "max")
    let spots = argmax(a, 1).to_list()
    assert_eq(spots.len(), 6)
    for row in 0..6 {
        let at = int(spots[row])
        assert_near(values[flat_of(shape, [row, at])], highs[row], 0.0, "argmax names the max")
    }

    let lows = host_fold(values, shape, [0], "min")
    let down = argmin(a, 0).to_list()
    for column in 0..9 {
        let at = int(down[column])
        assert_near(values[flat_of(shape, [at, column])], lows[column])
    }

    #: With no axis the answer is an index into the C-order flattening, which
    #: is what numpy returns too.
    let flat = int(argmax(a).item())
    assert_near(values[flat], host_fold(values, shape, [0, 1], "max")[0])
    assert_eq(argmax(a).dtype, INT64)
}

fn test_a_tie_goes_to_the_lowest_index() -> void {
    if not have_gpu() { return }
    let plateau = holding([2, 5], [1.0, 9.0, 9.0, 9.0, 2.0, 9.0, 0.0, 9.0, 0.0, 9.0])
    defer { plateau.free() }
    assert_eq(argmax(plateau, 1).to_list(), [1.0, 0.0])
    let valley = holding([1, 4], [3.0, 1.0, 1.0, 1.0])
    defer { valley.free() }
    assert_eq(argmin(valley, 1).to_list(), [1.0])

    #: Across a chunk boundary as well: a long segment is split over groups
    #: and the join has to prefer the earlier winner, not the earlier group's
    #: winner or whichever partial landed first.
    let count = 200000
    var values = [0.0 for _index in 0..count]
    values[11] = 5.0
    values[count - 3] = 5.0
    let long_run = holding([count], values)
    defer { long_run.free() }
    assert_eq(argmax(long_run).item(), 11.0, "the join keeps the earlier of two equal peaks")

    var deep = [1.0 for _index in 0..count]
    deep[4] = -7.0
    deep[count // 2] = -7.0
    let dip = holding([count], deep)
    defer { dip.free() }
    assert_eq(argmin(dip).item(), 4.0)
}

fn test_an_all_negative_infinity_slice_still_names_a_slot() -> void {
    if not have_gpu() { return }
    #: -inf never beats the starting key, so a slice made only of it once came
    #: back as the -1 of an empty segment.
    let sunk = holding([2, 3], [-math.INF, -math.INF, -math.INF, 1.0, -math.INF, 2.0])
    defer { sunk.free() }
    assert_eq(argmax(sunk, 1).to_list(), [0.0, 2.0])
    let risen = holding([1, 2], [math.INF, math.INF])
    defer { risen.free() }
    assert_eq(argmin(risen, 1).to_list(), [0.0])
}

fn test_a_plain_fold_carries_a_nan_out() -> void {
    if not have_gpu() { return }
    let holed = holding([2, 4], [1.0, math.NAN, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
    defer { holed.free() }
    let lows = amin(holed, 1).to_list()
    let highs = amax(holed, 1).to_list()
    let totals = sum(holed, 1).to_list()
    assert_true(is_nan(lows[0]), "a minimum does not step over a NaN")
    assert_true(is_nan(highs[0]))
    assert_true(is_nan(totals[0]))
    assert_near(lows[1], 1.0)
    assert_near(highs[1], 4.0)
    assert_near(totals[1], 10.0)

    #: numpy answers with the first NaN it meets rather than with an extreme.
    assert_eq(argmax(holed, 1).to_list(), [1.0, 3.0])
    assert_eq(argmin(holed, 1).to_list(), [1.0, 0.0])

    #: NaN is truthy, so it counts as a non-zero.
    assert_eq(any(holed, 1).to_list(), [1.0, 1.0])
    assert_eq(count_nonzero(holed, 1).to_list(), [4.0, 4.0])
}

fn test_a_nan_reaches_the_answer_through_the_chunk_join() -> void {
    if not have_gpu() { return }
    #: A segment long enough to be split over groups, with its one NaN late in
    #: the run: the chunk that saw it has to outrank every chunk that found a
    #: real maximum, and the join has to prefer the NaN's index over both.
    let count = 200000
    var values = spread(count)
    values[3] = 100.0
    values[count - 11] = math.NAN
    let a = holding([count], values)
    defer { a.free() }
    assert_true(is_nan(amax(a).item()), "one NaN carries the whole fold")
    assert_eq(argmax(a).item(), float(count - 11), "and names its own slot")
    assert_near(nanmax(a).item(), 100.0, 0.0, "the nan-aware pass steps over it")
    assert_eq(nanargmax(a).item(), 3.0)
}

fn test_the_nan_aware_folds_skip_the_holes() -> void {
    if not have_gpu() { return }
    let shape = [3, 8]
    var values = spread(24)
    for at in [1, 5, 9, 17, 22] { values[at] = math.NAN }
    let a = holding(shape, values)
    defer { a.free() }

    #: The same rows with the holes taken out, folded on the host.
    for row in 0..3 {
        var kept: list[float] = []
        for column in 0..8 {
            let value = values[flat_of(shape, [row, column])]
            if not is_nan(value) { kept.push(value) }
        }
        let n = float(kept.len())
        var total = 0.0
        for value in kept { total += value }
        let average = total / n
        var square = 0.0
        for value in kept { square += (value - average) * (value - average) }

        assert_near(nansum(a, 1).to_list()[row], total, 1.0e-4, f"row {row} total")
        assert_near(nanmean(a, 1).to_list()[row], average, 1.0e-4)
        assert_near(nanmin(a, 1).to_list()[row], host_fold(kept, [kept.len()], [0], "min")[0])
        assert_near(nanmax(a, 1).to_list()[row], host_fold(kept, [kept.len()], [0], "max")[0])
        assert_near(nanvar(a, 1).to_list()[row], square / n, 1.0e-4)
        assert_near(nanstd(a, 1).to_list()[row], math.sqrt(square / n), 1.0e-4)
        assert_near(
            nanvar(a, 1, false, -1, 1).to_list()[row],
            square / (n - 1.0),
            1.0e-4,
            "ddof comes off the count of what was kept"
        )

        let high = int(nanargmax(a, 1).to_list()[row])
        assert_near(
            values[flat_of(shape, [row, high])],
            host_fold(kept, [kept.len()], [0], "max")[0]
        )
        let low = int(nanargmin(a, 1).to_list()[row])
        assert_near(
            values[flat_of(shape, [row, low])],
            host_fold(kept, [kept.len()], [0], "min")[0]
        )
    }
}

fn test_a_hole_survives_the_float4_and_the_chunked_road_too() -> void {
    if not have_gpu() { return }
    #: A dropped NaN once became a zero that the centring squared, which only
    #: showed on the four-at-a-time path; and the second pass has to carry the
    #: skip as well as the first.
    let count = 120000
    var values = spread(count)
    var kept: list[float] = []
    for at in 0..count {
        if at % 1000 == 7 {
            values[at] = math.NAN
        } else {
            kept.push(values[at])
        }
    }
    let a = holding([count], values)
    defer { a.free() }

    var total = 0.0
    for value in kept { total += value }
    let average = total / float(kept.len())
    var square = 0.0
    for value in kept { square += (value - average) * (value - average) }

    assert_near(nansum(a).item(), total, 1.0, "a chunked nan-aware total")
    assert_near(nanmean(a).item(), average, 1.0e-3)
    assert_near(nanvar(a).item(), square / float(kept.len()), 1.0e-2)
    assert_near(nanmax(a).item(), host_fold(kept, [kept.len()], [0], "max")[0])
    assert_near(nanmin(a).item(), host_fold(kept, [kept.len()], [0], "min")[0])
}

fn test_an_all_nan_slice_keeps_numpys_answers() -> void {
    if not have_gpu() { return }
    let blank = holding([2, 3], [math.NAN, math.NAN, math.NAN, 1.0, 2.0, math.NAN])
    defer { blank.free() }
    assert_eq(nansum(blank, 1).to_list(), [0.0, 3.0], "nothing to add totals zero")
    assert_true(is_nan(nanmin(blank, 1).to_list()[0]), "no smallest of nothing")
    assert_true(is_nan(nanmax(blank, 1).to_list()[0]))
    assert_true(is_nan(nanmean(blank, 1).to_list()[0]), "zero over zero")
    assert_near(nanmin(blank, 1).to_list()[1], 1.0)

    #: -1 and not a raise: knowing the slice was empty would cost a
    #: synchronise inside every reduction, including the ones that are fine.
    assert_eq(nanargmax(blank, 1).to_list(), [-1.0, 1.0])
    assert_eq(nanargmin(blank, 1).to_list(), [-1.0, 0.0])
}

fn test_a_scan_is_the_running_fold() -> void {
    if not have_gpu() { return }
    let shape = [4, 6]
    let values = spread(24)
    let a = holding(shape, values)
    defer { a.free() }

    var running = [0.0 for _at in 0..24]
    for row in 0..4 {
        var acc = 0.0
        for column in 0..6 {
            acc += values[flat_of(shape, [row, column])]
            running[flat_of(shape, [row, column])] = acc
        }
    }
    assert_all_near(cumsum(a, 1).to_list(), running)

    var down = [0.0 for _at in 0..24]
    for column in 0..6 {
        var acc = 0.0
        for row in 0..4 {
            acc += values[flat_of(shape, [row, column])]
            down[flat_of(shape, [row, column])] = acc
        }
    }
    assert_all_near(cumsum(a, 0).to_list(), down)

    var peaks = [0.0 for _at in 0..24]
    var dips = [0.0 for _at in 0..24]
    for row in 0..4 {
        var high = values[flat_of(shape, [row, 0])]
        var low = high
        for column in 0..6 {
            let value = values[flat_of(shape, [row, column])]
            if value > high { high = value }
            if value < low { low = value }
            peaks[flat_of(shape, [row, column])] = high
            dips[flat_of(shape, [row, column])] = low
        }
    }
    assert_all_near(cummax(a, 1).to_list(), peaks)
    assert_all_near(cummin(a, 1).to_list(), dips)

    let near_one = holding(shape, [nudge(values[at]) for at in 0..24])
    defer { near_one.free() }
    var products = [0.0 for _at in 0..24]
    for row in 0..4 {
        var acc = 1.0
        for column in 0..6 {
            acc *= nudge(values[flat_of(shape, [row, column])])
            products[flat_of(shape, [row, column])] = acc
        }
    }
    assert_all_near(cumprod(near_one, 1).to_list(), products)

    #: A scan keeps the shape it was handed, unlike a fold.
    assert_eq(cumsum(a, 1).shape, shape)
}

fn test_a_scan_with_no_axis_runs_down_the_flattening() -> void {
    if not have_gpu() { return }
    let a = ramp([2, 3])
    defer { a.free() }
    let flat = cumsum(a)
    defer { flat.free() }
    assert_eq(flat.shape, [6], "numpy's answer to a flat scan is 1-d")
    assert_eq(flat.to_list(), [1.0, 3.0, 6.0, 10.0, 15.0, 21.0])

    #: The flattening of a view has to be materialised first, and the copy it
    #: makes is the one that has to be released — the 1-d array laid over it
    #: is a view and freeing that would let the copy leak.
    let m = ramp([4, 4])
    defer { m.free() }
    let window = m[sl(1, 3), sl(1, 3)]
    let scanned = cumsum(window)
    defer { scanned.free() }
    assert_eq(scanned.to_list(), [6.0, 13.0, 23.0, 34.0])
    assert_eq(cumsum(window, 0).to_list(), [6.0, 7.0, 16.0, 18.0], "a scan reads the view")
}

fn test_a_long_scan_joins_its_chunks() -> void {
    if not have_gpu() { return }
    let count = 200000
    let ones = holding([count], [1.0 for _at in 0..count])
    defer { ones.free() }
    let running = cumsum(ones).to_list()
    assert_near(running[0], 1.0)
    assert_near(running[1], 2.0)
    assert_near(running[count // 2], float(count // 2 + 1), 1.0)
    assert_near(running[count - 1], float(count), 1.0, "the bases folded back in")

    #: A running maximum over the same length, whose answer is a staircase no
    #: chunk can produce on its own.
    var values = [0.0 for _at in 0..count]
    for at in 0..count { values[at] = float(at * 37 % 1013) }
    let waves = holding([count], values)
    defer { waves.free() }
    let peaks = cummax(waves).to_list()
    var high = values[0]
    var checked = 0
    for at in 0..count {
        if values[at] > high { high = values[at] }
        if at % 997 == 0 {
            assert_near(peaks[at], high, 0.0, f"running maximum at {at}")
            checked += 1
        }
    }
    assert_true(checked > 100, "the sweep looked at the whole run")
    assert_near(peaks[count - 1], 1012.0)
}

fn test_trace_totals_the_diagonal() -> void {
    if not have_gpu() { return }
    let shape = [4, 5]
    let values = spread(20)
    let a = holding(shape, values)
    defer { a.free() }

    var main = 0.0
    for at in 0..4 { main += values[flat_of(shape, [at, at])] }
    assert_near(trace(a).item(), main, 1.0e-5)

    var upper = 0.0
    for at in 0..3 { upper += values[flat_of(shape, [at, at + 2])] }
    assert_near(trace(a, 2).item(), upper, 1.0e-5)

    var lower = 0.0
    for at in 0..3 { lower += values[flat_of(shape, [at + 1, at])] }
    assert_near(trace(a, -1).item(), lower, 1.0e-5)

    #: A stack of planes traces each of them, the diagonal being appended last.
    let cube = ramp([2, 3, 3])
    defer { cube.free() }
    assert_eq(trace(cube, 0, 1, 2).to_list(), [15.0, 42.0])
}

fn test_allclose_reads_the_tolerance_off_the_second_argument() -> void {
    if not have_gpu() { return }
    let a = holding([3], [1.0, 2.0, 3.0])
    let b = holding([3], [1.0, 2.0, 3.000001])
    defer { a.free() }
    defer { b.free() }
    assert_true(allclose(a, b))
    assert_true(not allclose(a, b, 0.0, 0.0))
    assert_true(allclose(a, b, 0.0, 1.0e-5), "an absolute tolerance alone is enough")

    #: Asymmetric, as numpy's is: the relative part scales the second argument.
    let big = holding([1], [1000.0])
    let small = holding([1], [1.0])
    defer { big.free() }
    defer { small.free() }
    assert_true(allclose(small, big, 1.0, 0.0), "1000 * rtol covers the gap")
    assert_true(not allclose(big, small, 1.0, 0.0), "1 * rtol does not")

    let broadcast = holding([1], [2.0])
    let row = holding([2, 3], [2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    defer { broadcast.free() }
    defer { row.free() }
    assert_true(allclose(row, broadcast), "the two sides broadcast")
}

fn test_an_infinity_is_only_close_to_the_same_infinity() -> void {
    if not have_gpu() { return }
    let up = holding([2], [math.INF, 1.0])
    let same = holding([2], [math.INF, 1.0])
    let down = holding([2], [-math.INF, 1.0])
    let finite = holding([2], [1.0e30, 1.0])
    defer { up.free() }
    defer { same.free() }
    defer { down.free() }
    defer { finite.free() }
    assert_true(allclose(up, same), "|inf - inf| is not a usable distance")
    assert_true(not allclose(up, down))
    assert_true(not allclose(up, finite))
}

fn test_nan_is_close_to_nothing_unless_it_is_told_to_be() -> void {
    if not have_gpu() { return }
    let holed = holding([3], [1.0, math.NAN, 3.0])
    let twin = holding([3], [1.0, math.NAN, 3.0])
    defer { holed.free() }
    defer { twin.free() }
    assert_true(not allclose(holed, twin))
    assert_true(allclose(holed, twin, 1.0e-5, 1.0e-8, true))
    assert_true(not array_equal(holed, twin))
    assert_true(array_equal(holed, twin, true))
}

fn test_array_equal_compares_the_shape_first() -> void {
    if not have_gpu() { return }
    let a = ramp([2, 3])
    let flat = ramp([6])
    defer { a.free() }
    defer { flat.free() }
    assert_true(not array_equal(a, flat), "the same values in another shape are not equal")
    assert_true(array_equal(a, a.copy()))

    let apart = a.copy()
    defer { apart.free() }
    apart[1, 2] = 0.0
    assert_true(not array_equal(a, apart))
    assert_true(not allclose(a, apart), "exactness is the only difference between the two")

    let empty = NDArray([0, 4], FLOAT32)
    let twin = NDArray([0, 4], FLOAT32)
    defer { empty.free() }
    defer { twin.free() }
    assert_true(array_equal(empty, twin), "two empties agree everywhere, vacuously")
}

fn test_an_empty_axis_gets_the_identity_without_a_dispatch() -> void {
    if not have_gpu() { return }
    let empty = NDArray([3, 0], FLOAT32)
    defer { empty.free() }
    assert_eq(sum(empty, 1).to_list(), [0.0, 0.0, 0.0])
    assert_eq(prod(empty, 1).to_list(), [1.0, 1.0, 1.0])
    assert_eq(count_nonzero(empty, 1).to_list(), [0.0, 0.0, 0.0])
    assert_eq(any(empty, 1).to_list(), [0.0, 0.0, 0.0])
    assert_eq(all(empty, 1).to_list(), [1.0, 1.0, 1.0], "every element of nothing qualifies")

    #: 0/0, which numpy also answers with NaN rather than raising.
    for value in mean(empty, 1).to_list() { assert_true(is_nan(value)) }
    for value in variance(empty, 1).to_list() { assert_true(is_nan(value)) }

    #: An output with no elements needs no fold at all.
    assert_eq(sum(empty, 0).shape, [0])
    assert_eq(sum(empty, 0).to_list(), [])
    assert_eq(cumsum(empty, 1).shape, [3, 0])
}

fn test_the_folds_with_no_identity_refuse_an_empty_axis() -> void {
    if not have_gpu() { return }
    let empty = NDArray([3, 0], FLOAT32)
    defer { empty.free() }
    assert_throws(ValueError, || amin(empty, 1), "there is no smallest of nothing")
    assert_throws(ValueError, || amax(empty, 1))
    assert_throws(ValueError, || argmin(empty, 1))
    assert_throws(ValueError, || argmax(empty, 1))
    assert_throws(ValueError, || nanmax(empty, 1))
}

fn test_a_reduction_refuses_a_complex_array() -> void {
    if not have_gpu() { return }
    let wave = NDArray([2], COMPLEX64)
    wave.upload([1.0, 2.0, 3.0, 4.0])
    defer { wave.free() }
    assert_throws(TypeError, || sum(wave))
    assert_throws(TypeError, || argmax(wave))
    assert_throws(TypeError, || cumsum(wave))
    assert_throws(TypeError, || allclose(wave, wave))
}

fn test_a_zero_dimensional_array_folds_to_itself() -> void {
    if not have_gpu() { return }
    let a = ramp([3, 4])
    defer { a.free() }
    let one = a[1, 2]
    assert_eq(one.shape, [])
    assert_near(sum(one).item(), one.item())
    assert_near(amax(one).item(), one.item())
    assert_eq(argmax(one).item(), 0.0, "the only element is at index zero")
    assert_eq(cumsum(one).to_list(), [one.item()])
}

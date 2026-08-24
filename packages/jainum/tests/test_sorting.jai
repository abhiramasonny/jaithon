#: Tests for jainum's sort, search and set operations.
#:
#: Nothing here compares against numbers this module printed once and kept.
#: Every check is either the definition of the operation -- a sort is a
#: non-decreasing permutation, `searchsorted` left is the count of elements
#: strictly below the key, `unique`'s inverse reconstructs the input -- or a
#: second implementation that shares no code with the one under test: the VM's
#: own `list.sort`, and host loops over downloaded values.
#:
#: Sizes are chosen around `HOST_SORT_LIMIT`, because the module has two
#: implementations behind one entry point and a test that only ever sees one of
#: them proves half as much as it looks like it does. `crosses_the_limit` names
#: a size on the far side; the small cases stay on the host path.
#:
#: Data comes from a linear congruential generator seeded per test rather than
#: from hand-picked lists, so a bug that survives one arrangement of ties has
#: several thousand more to survive.
import std.gpu as gpu
import std.math
from std.test import assert_eq, assert_false, assert_throws, assert_true
from jainum.array import COMPLEX64, INT32, NDArray
from jainum.sorting import HOST_SORT_LIMIT, argpartition, argsort, in1d
from jainum.sorting import intersect1d, isin, lexsort, partition, searchsorted
from jainum.sorting import setdiff1d, setxor1d, sort, union1d, unique

fn have_gpu() -> bool { return gpu.is_available() }

#: A size the device network has to handle, whatever the threshold is set to.
fn crosses_the_limit() -> int { return HOST_SORT_LIMIT * 2 + 137 }

#: `n` pseudorandom whole numbers below `span`.
#:
#: An LCG rather than a literal list: the point of a span far smaller than `n`
#: is that every value repeats, which is where a stable sort and a `unique`
#: have something to get wrong.
fn noise(n: int, span: int, seed: int) -> list[float] {
    var state = seed
    var out: list[float] = []
    for _ in 0..n {
        state = (state * 1103515245 + 12345) % 2147483648
        out.push(float(state % span))
    }
    return out
}

fn uploaded(shape: list[int], values: list[float]) -> NDArray {
    let a = NDArray(shape)
    a.upload(values)
    return a
}

#: Whether the list never steps backwards. Together with a matching multiset
#: this is the whole definition of a sorted result.
fn ascends(values: list[float]) -> bool {
    for at in 1..values.len() {
        if values[at] < values[at - 1] { return false }
    }
    return true
}

fn descends(values: list[float]) -> bool {
    for at in 1..values.len() {
        if values[at] > values[at - 1] { return false }
    }
    return true
}

#: Whether two lists hold the same values in some order.
#:
#: `list.sort` is the VM's merge sort in C, which the device network shares no
#: line of code with, so agreeing with it is evidence and not a tautology.
fn same_values(left: list[float], right: list[float]) -> bool {
    return left.copy().sort() == right.copy().sort()
}

#: Whether `order` names each of `n` slots exactly once.
fn is_a_permutation(order: list[float], n: int) -> bool {
    return order.copy().sort() == [float(slot) for slot in 0..n]
}

fn gathered(data: list[float], order: list[float]) -> list[float] {
    return [data[int(slot)] for slot in order]
}

#: How many elements sit strictly below `key`, which is what a left insertion
#: point counts.
fn count_below(values: list[float], key: float) -> int {
    var total = 0
    for value in values {
        if value < key { total += 1 }
    }
    return total
}

#: How many sit at or below it, which is the right insertion point.
fn count_at_most(values: list[float], key: float) -> int {
    var total = 0
    for value in values {
        if value <= key { total += 1 }
    }
    return total
}

fn test_sort_returns_a_non_decreasing_permutation() -> void {
    if not have_gpu() { return }
    for n in [1, 2, 3, 5, 7, 17, 64, 1023, 1024, 1025, crosses_the_limit()] {
        let data = noise(n, 97, 2024)
        let a = uploaded([n], data)
        defer { a.free() }
        let ordered = sort(a)
        defer { ordered.free() }
        let got = ordered.to_list()

        assert_true(ascends(got), f"{n} elements come out in order")
        assert_true(same_values(got, data), f"{n} elements come out unchanged")
        assert_eq(a.to_list(), data, "sort leaves its input alone")
    }
}

fn test_argsort_is_the_permutation_that_sorts() -> void {
    if not have_gpu() { return }
    for n in [1, 2, 5, 33, 1025, crosses_the_limit()] {
        let data = noise(n, 41, 7)
        let a = uploaded([n], data)
        defer { a.free() }
        let order = argsort(a)
        defer { order.free() }
        let places = order.to_list()

        assert_true(is_a_permutation(places, n), f"{n} indices, each once")
        assert_true(ascends(gathered(data, places)), f"{n} indices sort the values")
    }
}

fn test_ties_keep_the_order_they_arrived_in() -> void {
    if not have_gpu() { return }

    #: A span of eleven over thousands of elements means every value is a run
    #: of hundreds, so stability is not something the data could satisfy by
    #: accident.
    for n in [300, crosses_the_limit()] {
        let data = noise(n, 11, 999)
        let a = uploaded([n], data)
        defer { a.free() }

        let up = argsort(a)
        defer { up.free() }
        let down = argsort(a, 0, true)
        defer { down.free() }
        assert_true(stable_in(data, up.to_list(), false), f"{n} ascending")
        assert_true(stable_in(data, down.to_list(), true), f"{n} descending")
    }
}

#: Whether `order` runs the right way and never swaps two equal elements.
#:
#: Descending stability is the case worth stating: the values reverse, the
#: indices behind equal values do not.
fn stable_in(data: list[float], order: list[float], descending: bool) -> bool {
    for at in 1..order.len() {
        let before = data[int(order[at - 1])]
        let now = data[int(order[at])]
        if before == now {
            if order[at - 1] > order[at] { return false }
            continue
        }
        if descending and before < now { return false }
        if not descending and before > now { return false }
    }
    return true
}

fn test_descending_turns_the_whole_order_over() -> void {
    if not have_gpu() { return }
    let n = crosses_the_limit()

    #: Distinct values, so "reversed" is unambiguous -- with ties it would not
    #: be, since a stable descending sort does not reverse them.
    let data = [float(slot * 7919 % n) for slot in 0..n]
    let a = uploaded([n], data)
    defer { a.free() }

    let up = sort(a)
    defer { up.free() }
    let down = sort(a, -1, true)
    defer { down.free() }

    let falling = down.to_list()
    assert_true(descends(falling))
    assert_eq(falling, up.to_list().reversed(), "distinct values reverse exactly")
}

fn test_every_axis_of_a_three_dimensional_array() -> void {
    if not have_gpu() { return }
    let shape = [5, 7, 9]
    let data = noise(315, 60, 31)
    let a = uploaded(shape, data)
    defer { a.free() }

    for axis in [0, 1, 2, -1] {
        let ordered = sort(a, axis)
        defer { ordered.free() }
        let got = ordered.to_list()
        assert_eq(ordered.shape, shape, "a sort keeps the shape it was given")
        assert_true(lanes_agree(data, got, shape, axis), f"axis {axis}")
    }
}

#: Whether every 1-d lane along `axis` of `got` is the sorted lane of `data`.
#:
#: The walk here is written from the shape and a flat index rather than from
#: anything the module computed, so a stride bug in `_axis_last` has nowhere to
#: hide.
fn lanes_agree(data: list[float], got: list[float], shape: list[int], axis: int) -> bool {
    let at = if axis < 0 { axis + shape.len() } else { axis }
    var step = 1
    for other in at + 1..shape.len() { step *= shape[other] }
    let extent = shape[at]
    let outer = data.len() // (extent * step)
    for high in 0..outer {
        for low in 0..step {
            let base = high * extent * step + low
            var lane: list[float] = []
            var mine: list[float] = []
            for slot in 0..extent {
                lane.push(data[base + slot * step])
                mine.push(got[base + slot * step])
            }
            if mine != lane.sort() { return false }
        }
    }
    return true
}

fn test_a_strided_view_sorts_by_value_and_not_by_slot() -> void {
    if not have_gpu() { return }
    let rows = 40
    let cols = 130
    let data = noise(rows * cols, 500, 5)
    let a = uploaded([rows, cols], data)
    defer { a.free() }

    #: A transpose has no dense entry point at all, so this is the copy-in,
    #: copy-out path of `_sort_into` end to end -- and it crosses the threshold,
    #: so the network is what runs.
    let flipped = a.view([cols, rows], [a.strides[1], a.strides[0]], a.offset)
    let ordered = sort(flipped)
    defer { ordered.free() }
    let got = ordered.to_list()
    for column in 0..cols {
        var lane: list[float] = []
        for row in 0..rows { lane.push(data[row * cols + column]) }
        assert_eq(got.slice(column * rows, (column + 1) * rows), lane.sort(), f"column {column}")
    }
}

fn test_the_host_and_device_paths_agree() -> void {
    if not have_gpu() { return }

    #: One row of this stays under `HOST_SORT_LIMIT` and two rows of it do not,
    #: so the same data goes down both paths and the answers have to match.
    let n = HOST_SORT_LIMIT * 3 // 4
    let data = noise(n, 23, 4242)
    let one = uploaded([1, n], data)
    let two = uploaded([2, n], data + data)
    defer { one.free() }
    defer { two.free() }
    assert_true(n <= HOST_SORT_LIMIT and 2 * n > HOST_SORT_LIMIT, "the sizes straddle it")

    let host = sort(one)
    let device = sort(two)
    defer { host.free() }
    defer { device.free() }
    assert_eq(device.to_list().slice(0, n), host.to_list(), "the same values")

    let host_order = argsort(one)
    let device_order = argsort(two)
    defer { host_order.free() }
    defer { device_order.free() }
    assert_eq(
        device_order.to_list().slice(0, n),
        host_order.to_list(),
        "and the same stable permutation"
    )
}

fn test_nan_sorts_last_and_the_infinities_in_their_places() -> void {
    if not have_gpu() { return }
    let a = uploaded([6], [1.0, math.NAN, -math.INF, 0.0, math.INF, math.NAN])
    defer { a.free() }

    let up = sort(a)
    defer { up.free() }
    let rising = up.to_list()
    assert_eq(rising.slice(0, 4), [-math.INF, 0.0, 1.0, math.INF])
    assert_true(math.is_nan(rising[4]) and math.is_nan(rising[5]), "NaN last, as numpy has it")

    let down = sort(a, -1, true)
    defer { down.free() }
    let falling = down.to_list()
    assert_true(math.is_nan(falling[0]) and math.is_nan(falling[1]), "and first once turned over")
    assert_eq(falling.slice(2, 6), [math.INF, 1.0, 0.0, -math.INF])
}

fn test_a_padded_segment_cannot_swallow_a_real_nan() -> void {
    if not have_gpu() { return }

    #: The padding is NaN on the ascending path, which is the case where a real
    #: NaN could be pushed past the end of the segment and lost. A length well
    #: off a power of two maximises the padding it has to survive.
    let n = crosses_the_limit()
    var data: list[float] = []
    for slot in 0..n {
        data.push(if slot % 7 == 0 { math.NAN } else { float(slot * 37 % 101) })
    }
    let a = uploaded([n], data)
    defer { a.free() }
    let ordered = sort(a)
    defer { ordered.free() }

    let got = ordered.to_list()
    let wanted = data.filter(|value| math.is_nan(value)).len()
    assert_eq(got.filter(|value| math.is_nan(value)).len(), wanted, "every NaN survives")
    assert_true(ascends(got.slice(0, got.len() - wanted)), "and none of them landed early")
}

fn test_sort_rejects_what_it_cannot_order() -> void {
    if not have_gpu() { return }
    let scalar = NDArray([])
    defer { scalar.free() }
    assert_throws(ValueError, || sort(scalar), "a 0-d array has no axis to sort")

    let tangled = NDArray([4], COMPLEX64)
    defer { tangled.free() }
    assert_throws(TypeError, || sort(tangled), "complex has no order")
    assert_throws(TypeError, || argsort(tangled))
}

fn test_partition_puts_the_kth_element_where_a_sort_would() -> void {
    if not have_gpu() { return }
    let rows = 8
    let cols = 700
    let data = noise(rows * cols, 300, 88)
    let a = uploaded([rows, cols], data)
    defer { a.free() }

    let split = partition(a, 40, 1)
    defer { split.free() }
    let got = split.to_list()
    for row in 0..rows {
        let lane = data.slice(row * cols, (row + 1) * cols).sort()
        let mine = got.slice(row * cols, (row + 1) * cols)
        assert_eq(mine[40], lane[40], f"row {row} holds the 40th smallest at 40")
        assert_true(mine.slice(0, 40).max() <= mine[40], "nothing greater before it")
        assert_true(mine.slice(41, cols).min() >= mine[40], "nothing smaller after it")
    }

    let order = argpartition(a, 40, 1)
    defer { order.free() }
    let places = order.to_list()
    for row in 0..rows {
        let base = row * cols
        let slot = int(places[base + 40])
        assert_eq(data[base + slot], got[base + 40], "the index names the value")
    }

    assert_throws(ValueError, || partition(a, cols, 1), "kth past the axis")
    assert_throws(ValueError, || argpartition(a, -cols - 1, 1))
}

fn test_searchsorted_counts_what_precedes_the_key() -> void {
    if not have_gpu() { return }

    #: Long enough that the coarse threadgroup sample is switched on, which is
    #: a separate path through the kernel from a short array's plain bisection.
    for n in [3, 100, 5000] {
        let pool = noise(n, 60, 17).sort()
        let a = uploaded([n], pool)
        defer { a.free() }

        let asked = [-5.0, 0.0, 1.0, 30.5, 59.0, 60.0, 1000.0]
        let v = uploaded([asked.len()], asked)
        defer { v.free() }

        let left = searchsorted(a, v)
        let right = searchsorted(a, v, "right")
        defer { left.free() }
        defer { right.free() }

        for at in 0..asked.len() {
            let key = asked[at]
            assert_eq(int(left.to_list()[at]), count_below(pool, key), f"{n}: left of {key}")
            assert_eq(int(right.to_list()[at]), count_at_most(pool, key), f"{n}: right of {key}")
        }
    }
}

fn test_searchsorted_keeps_the_shape_of_its_queries() -> void {
    if not have_gpu() { return }
    let a = uploaded([5], [0.0, 1.5, 4.0, 4.0, 10.0])
    defer { a.free() }
    let v = uploaded([2, 2], [-1.0, 4.0, 5.0, 99.0])
    defer { v.free() }

    let at = searchsorted(a, v)
    defer { at.free() }
    assert_eq(at.shape, [2, 2])
    assert_eq(at.dtype, INT32)
    assert_eq(at.to_nested(), [[0.0, 2.0], [4.0, 5.0]])

    assert_throws(ValueError, || searchsorted(a, v, "middle"), "side is left or right")
    assert_throws(ValueError, || searchsorted(v, a), "the haystack has to be 1-d")
}

fn test_isin_agrees_with_a_membership_loop() -> void {
    if not have_gpu() { return }
    let queried = noise(400, 80, 3)
    let pooled = noise(2000, 50, 11)
    let element = uploaded([20, 20], queried)
    let test_elements = uploaded([2000], pooled)
    defer { element.free() }
    defer { test_elements.free() }

    let found = isin(element, test_elements)
    let missing = isin(element, test_elements, true)
    defer { found.free() }
    defer { missing.free() }
    assert_eq(found.shape, [20, 20], "the answer is shaped like the question")

    let flags = found.to_list()
    let holes = missing.to_list()
    for at in 0..queried.len() {
        let wanted = if pooled.contains(queried[at]) { 1.0 } else { 0.0 }
        assert_eq(flags[at], wanted, f"{queried[at]}")
        assert_eq(holes[at], 1.0 - wanted, "invert is the complement")
    }

    #: NaN matching NaN is the one place `isin` departs from `==`, and it is the
    #: same call `unique` makes when it folds a run of them into one value.
    let odd = uploaded([2], [math.NAN, 1.0])
    let haystack = uploaded([3], [0.0, math.NAN, 2.0])
    defer { odd.free() }
    defer { haystack.free() }
    let matched = isin(odd, haystack)
    defer { matched.free() }
    assert_eq(matched.to_list(), [1.0, 0.0], "NaN is in a pool that holds one")
}

fn test_in1d_flattens_its_left_side() -> void {
    if not have_gpu() { return }
    let element = uploaded([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    let pool = uploaded([2], [2.0, 5.0])
    defer { element.free() }
    defer { pool.free() }

    let flat = in1d(element, pool)
    defer { flat.free() }
    assert_eq(flat.shape, [6], "in1d is the flattened spelling")
    assert_eq(flat.to_list(), [0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
}

fn test_unique_is_each_value_once_ascending() -> void {
    if not have_gpu() { return }
    for n in [1, 9, crosses_the_limit()] {
        let data = noise(n, 37, 65)
        let a = uploaded([n], data)
        defer { a.free() }
        let found = unique(a)
        defer { found.free() }

        let values = found.values.to_list()
        assert_eq(values, data.copy().unique().sort(), f"{n}: the distinct values, in order")
        for at in 1..values.len() {
            assert_true(values[at - 1] < values[at], "strictly increasing, so nothing repeats")
        }
    }
}

fn test_unique_reports_index_inverse_and_counts() -> void {
    if not have_gpu() { return }
    let n = crosses_the_limit()
    let data = noise(n, 29, 1234)
    let a = uploaded([n // 7, 7], data.slice(0, n // 7 * 7))
    defer { a.free() }
    let flat = a.to_list()

    let found = unique(a, true, true, true)
    defer { found.free() }
    let values = found.values.to_list()
    let indices = found.indices.to_list()
    let inverse = found.inverse.to_list()
    let counts = found.counts.to_list()

    assert_eq(found.inverse.shape, a.shape, "the inverse is shaped like the input")
    assert_eq(gathered(values, inverse), flat, "values[inverse] rebuilds the input")
    assert_eq(gathered(flat, indices), values, "each index names its own value")
    for at in 0..values.len() {
        assert_eq(int(indices[at]), flat.index(values[at]), "and names the first one")
        assert_eq(int(counts[at]), flat.count(values[at]), f"{values[at]} is counted")
    }
    assert_eq(counts.sum(), float(flat.len()), "the counts account for every element")
}

fn test_unique_folds_every_nan_into_one_value() -> void {
    if not have_gpu() { return }
    let a = uploaded([6], [math.NAN, 1.0, math.NAN, 0.0, 1.0, math.NAN])
    defer { a.free() }
    let found = unique(a, false, false, true)
    defer { found.free() }

    let values = found.values.to_list()
    assert_eq(values.slice(0, 2), [0.0, 1.0])
    assert_eq(values.len(), 3, "one NaN, not three")
    assert_true(math.is_nan(values[2]), "and it sorts last")
    assert_eq(found.counts.to_list(), [1.0, 2.0, 3.0])
}

fn test_the_set_operations_agree_with_host_sets() -> void {
    if not have_gpu() { return }
    let left = noise(1200, 40, 21)
    let right = noise(900, 55, 22)
    let a = uploaded([1200], left)
    let b = uploaded([900], right)
    defer { a.free() }
    defer { b.free() }

    let both = union1d(a, b)
    let shared = intersect1d(a, b)
    let only_left = setdiff1d(a, b)
    let either = setxor1d(a, b)
    defer { both.free() }
    defer { shared.free() }
    defer { only_left.free() }
    defer { either.free() }

    let ls = left.copy().unique().sort()
    let rs = right.copy().unique().sort()
    assert_eq(both.to_list(), (ls + rs).unique().sort())
    assert_eq(shared.to_list(), ls.filter(|value| rs.contains(value)))
    assert_eq(only_left.to_list(), ls.filter(|value| not rs.contains(value)))
    assert_eq(
        either.to_list(),
        (ls.filter(|value| not rs.contains(value)) + rs.filter(|value| not ls.contains(value))).sort()
    )

    #: setxor is the two differences laid end to end, so it must never hold a
    #: value both sides carry -- the property the implementation leans on when
    #: it skips a second `unique`.
    for value in shared.to_list() {
        assert_false(either.to_list().contains(value), f"{value} is in both, so not in either")
    }
}

fn test_the_set_operations_take_disjoint_and_empty_inputs() -> void {
    if not have_gpu() { return }
    let a = uploaded([3], [1.0, 2.0, 3.0])
    let b = uploaded([2], [8.0, 9.0])
    let nothing = NDArray([0])
    defer { a.free() }
    defer { b.free() }
    defer { nothing.free() }

    let shared = intersect1d(a, b)
    defer { shared.free() }
    assert_eq(shared.to_list(), [], "disjoint sets share nothing")

    let both = union1d(a, nothing)
    defer { both.free() }
    assert_eq(both.to_list(), [1.0, 2.0, 3.0], "the empty set adds nothing")

    let dropped = setdiff1d(nothing, a)
    defer { dropped.free() }
    assert_eq(dropped.to_list(), [])

    let either = setxor1d(nothing, b)
    defer { either.free() }
    assert_eq(either.to_list(), [8.0, 9.0])

    #: A 2-d input is flattened, not rejected: these are set operations over
    #: values and the shape carries no information they use.
    let grid = uploaded([2, 2], [4.0, 1.0, 4.0, 2.0])
    defer { grid.free() }
    let folded = union1d(grid, nothing)
    defer { folded.free() }
    assert_eq(folded.to_list(), [1.0, 2.0, 4.0])
}

fn test_lexsort_lets_the_last_key_decide() -> void {
    if not have_gpu() { return }
    let n = 500
    let low = noise(n, 3, 1)
    let middle = noise(n, 4, 2)
    let high = noise(n, 2, 3)
    let a = uploaded([n], low)
    let b = uploaded([n], middle)
    let c = uploaded([n], high)
    defer { a.free() }
    defer { b.free() }
    defer { c.free() }

    let order = lexsort([a, b, c])
    defer { order.free() }
    let places = order.to_list()
    assert_true(is_a_permutation(places, n), "every row appears once")

    #: The reference is a plain lexicographic comparison of the tuple in the
    #: order numpy specifies -- last key first, ties down to the row number.
    for at in 1..n {
        let before = int(places[at - 1])
        let now = int(places[at])
        var verdict = 0
        if high[before] != high[now] {
            verdict = if high[before] < high[now] { -1 } else { 1 }
        } elif middle[before] != middle[now] {
            verdict = if middle[before] < middle[now] { -1 } else { 1 }
        } elif low[before] != low[now] {
            verdict = if low[before] < low[now] { -1 } else { 1 }
        } else {
            verdict = if before < now { -1 } else { 1 }
        }
        assert_eq(verdict, -1, f"rows {before} and {now} are the wrong way round")
    }
}

fn test_lexsort_rejects_keys_it_cannot_line_up() -> void {
    if not have_gpu() { return }
    let a = uploaded([4], [1.0, 2.0, 3.0, 4.0])
    let short = uploaded([3], [1.0, 2.0, 3.0])
    let square = uploaded([2, 2], [1.0, 2.0, 3.0, 4.0])
    defer { a.free() }
    defer { short.free() }
    defer { square.free() }

    let alone = lexsort([a])
    defer { alone.free() }
    assert_eq(alone.to_list(), [0.0, 1.0, 2.0, 3.0], "one key is just an argsort")

    assert_throws(ValueError, || lexsort([]), "no keys names no order")
    assert_throws(ValueError, || lexsort([a, short]), "keys of different lengths")
    assert_throws(ValueError, || lexsort([a, square]), "a 2-d key")
}

fn test_an_empty_array_survives_every_entry_point() -> void {
    if not have_gpu() { return }
    let nothing = NDArray([0])
    let asked = uploaded([2], [1.0, 2.0])
    defer { nothing.free() }
    defer { asked.free() }

    let ordered = sort(nothing)
    let order = argsort(nothing)
    defer { ordered.free() }
    defer { order.free() }
    assert_eq(ordered.to_list(), [])
    assert_eq(order.to_list(), [])

    let found = unique(nothing, true, true, true)
    defer { found.free() }
    assert_eq(found.values.to_list(), [])
    assert_eq(found.counts.to_list(), [])

    #: Nothing to search means every key belongs at slot zero and none of them
    #: is present -- the answers a length of zero forces, not a special case.
    let at = searchsorted(nothing, asked)
    defer { at.free() }
    assert_eq(at.to_list(), [0.0, 0.0])

    let flags = isin(asked, nothing)
    defer { flags.free() }
    assert_eq(flags.to_list(), [0.0, 0.0])

    let empty_answer = isin(nothing, asked)
    defer { empty_answer.free() }
    assert_eq(empty_answer.to_list(), [])
}

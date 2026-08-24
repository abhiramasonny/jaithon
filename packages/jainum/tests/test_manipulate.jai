#: Tests for jainum's shape manipulation: the view rewrites, the joins, the
#: splits, and the one padding kernel this module owns.
#:
#: Every expected value is computed from the definition of the operation
#: rather than read off a previous run. The trick that makes that cheap is
#: `ramp`, whose every element equals its own row-major index: an operation
#: that only moves elements around is then a mapping from a destination index
#: to a source index, and `mapped` writes that mapping down as a lambda and
#: evaluates it on the host. A golden list would only prove the code still
#: does what it did last week.
#:
#: The pure shape arithmetic runs without a device. Everything that touches
#: `storage` returns early when there is no GPU, the way the array suite does.
import std.gpu as gpu
from std.test import assert_eq, assert_len, assert_ne, assert_throws, assert_true
from jainum.array import COMPLEX64, FLOAT32, INT32, NDArray, element_count
from jainum.manipulate import PAD_EDGE, PAD_REFLECT, PAD_WRAP
from jainum.manipulate import array_split, atleast_1d, atleast_2d, atleast_3d
from jainum.manipulate import broadcast_arrays, broadcast_to, column_stack, concatenate
from jainum.manipulate import diagonal, dsplit, dstack, expand_dims, flatten, flip
from jainum.manipulate import fliplr, flipud, hsplit, hstack, moveaxis, pad
from jainum.manipulate import permute_dims, ravel, repeat, reshape, resize, roll
from jainum.manipulate import rollaxis, rot90, split, squeeze, stack, swapaxes
from jainum.manipulate import tile, transpose, vsplit, vstack

fn have_gpu() -> bool { return gpu.is_available() }

#: An array whose every element is its own row-major index, which is what lets
#: a reference below name a source position and read its value for free.
fn ramp(shape: list[int], dtype: int = FLOAT32) -> NDArray {
    let a = NDArray(shape, dtype)
    let lanes = if dtype == COMPLEX64 { 2 } else { 1 }
    a.upload([float(index) for index in 0..element_count(shape) * lanes])
    return a
}

#: The row-major position of `index` inside `shape` — the definition of
#: element order, and therefore of `ramp`'s values.
fn flat_index(shape: list[int], index: list[int]) -> int {
    var at = 0
    for axis in 0..shape.len() { at = at * shape[axis] + index[axis] }
    return at
}

#: Every multi-index of `shape`, in row-major order.
fn indices_of(shape: list[int]) -> list[list[int]] {
    var out: list[list[int]] = [[]]
    for axis in 0..shape.len() {
        var next: list[list[int]] = []
        for prefix in out {
            for step in 0..shape[axis] {
                var one = [item for item in prefix]
                one.push(step)
                next.push(one)
            }
        }
        out = next
    }
    return out
}

#: What a `ramp` of `source` becomes under a mapping from each destination
#: index to a source index — the flat list an operation that only moves
#: elements must produce.
fn mapped(target: list[int], source: list[int], rule: fn(list[int]) -> list[int]) -> list[float] {
    var out: list[float] = []
    for index in indices_of(target) { out.push(float(flat_index(source, rule(index)))) }
    return out
}

fn test_reshape_is_a_view_exactly_when_the_strides_allow_it() -> void {
    if not have_gpu() { return }
    let a = ramp([3, 4])
    defer { a.free() }

    #: A dense array always gives a window, and the write proves it.
    let split_axes = reshape(a, [2, 2, 3])
    assert_eq(split_axes.shape, [2, 2, 3])
    assert_eq(split_axes.to_list(), a.to_list(), "a reshape keeps element order")
    split_axes[0, 0, 0] = 99.0
    assert_eq(a[0, 0].item(), 99.0, "a reshape that could be a view is one")
    a[0, 0] = 0.0

    #: And so does a slice of the leading axis, whose strides still run
    #: consecutively through the axes being merged. This is the case a plain
    #: "contiguous or copy" rule would send through a needless allocation.
    let rows = a[1..3]
    let merged = reshape(rows, [8])
    merged[0] = 77.0
    assert_eq(a[1, 0].item(), 77.0, "a non-contiguous reshape can still be a view")
    a[1, 0] = 4.0

    #: A transpose reshaped across its permuted axes cannot be, so it copies.
    let turned = transpose(a)
    let flatly = reshape(turned, [12])
    defer { flatly.free() }
    assert_eq(flatly.to_list(), mapped([12], [3, 4], |index| [index[0] % 3, index[0] // 3]))
    flatly[0] = 55.0
    assert_eq(a[0, 0].item(), 0.0, "a reshape that had to copy does not alias")
}

fn test_reshape_resolves_one_unknown_axis_and_rejects_the_rest() -> void {
    if not have_gpu() { return }
    let a = ramp([2, 6])
    defer { a.free() }
    assert_eq(reshape(a, [3, -1]).shape, [3, 4])
    assert_eq(reshape(a, [-1]).shape, [12])
    assert_throws(ValueError, || reshape(a, [-1, -1]), "only one axis may be unknown")
    assert_throws(ValueError, || reshape(a, [5, -1]), "12 is not a multiple of 5")
    assert_throws(ValueError, || reshape(a, [5, 5]), "the element counts must agree")
    assert_throws(ValueError, || reshape(a, [-2]))
}

fn test_ravel_and_flatten_differ_in_who_owns_the_storage() -> void {
    if not have_gpu() { return }
    let a = ramp([2, 3])
    defer { a.free() }

    let raveled = ravel(a)
    raveled[0] = 9.0
    assert_eq(a[0, 0].item(), 9.0, "ravel is a view where it can be")

    let flat = flatten(a)
    defer { flat.free() }
    flat[0] = 1.0
    assert_eq(a[0, 0].item(), 9.0, "flatten is always a copy")

    #: A transposed source has no flat window, so both must materialise the
    #: same values.
    let turned = transpose(a)
    let from_view = ravel(turned)
    defer { from_view.free() }
    assert_eq(from_view.to_list(), flatten(turned).to_list())
}

fn test_transpose_permutes_by_definition() -> void {
    if not have_gpu() { return }
    let a = ramp([2, 3, 4])
    defer { a.free() }

    let turned = transpose(a, [2, 0, 1])
    assert_eq(turned.shape, [4, 2, 3])
    assert_eq(
        turned.to_list(),
        mapped([4, 2, 3], [2, 3, 4], |index| [index[1], index[2], index[0]])
    )

    #: The default reverses, and every one of these is a stride rewrite: the
    #: write below lands in the parent because nothing was copied.
    let reversed = transpose(a)
    assert_eq(reversed.shape, [4, 3, 2])
    reversed[3, 2, 1] = 42.0
    assert_eq(a[1, 2, 3].item(), 42.0, "a transpose is free")

    assert_eq(permute_dims(a, [1, 0, 2]).shape, [3, 2, 4])
    assert_throws(ValueError, || transpose(a, [0, 1]), "the permutation covers every axis")
    assert_throws(ValueError, || transpose(a, [0, 0, 1]), "and covers each one once")
}

fn test_swapaxes_moveaxis_and_rollaxis_are_the_permutations_they_name() -> void {
    if not have_gpu() { return }
    let a = ramp([2, 3, 4])
    defer { a.free() }

    assert_eq(swapaxes(a, 0, 2).to_list(), transpose(a, [2, 1, 0]).to_list())
    assert_eq(swapaxes(a, -1, 0).shape, [4, 3, 2])

    #: moveaxis keeps every other axis in order, so a single move is the
    #: permutation you get by deleting a position and re-inserting it.
    assert_eq(moveaxis(a, 0, 2).to_list(), transpose(a, [1, 2, 0]).to_list())
    assert_eq(moveaxis(a, 2, 0).to_list(), transpose(a, [2, 0, 1]).to_list())
    assert_eq(moveaxis(a, [0, 1], [2, 0]).shape, [3, 4, 2])

    #: rollaxis lands the axis *before* the one that was at `start`, which is
    #: the off-by-one that separates it from moveaxis.
    assert_eq(rollaxis(a, 2, 1).to_list(), transpose(a, [0, 2, 1]).to_list())
    assert_eq(rollaxis(a, 0, 3).to_list(), transpose(a, [1, 2, 0]).to_list())
    assert_true(rollaxis(a, 1, 1) == a, "a roll that moves nothing hands the array back")
    assert_throws(ValueError, || rollaxis(a, 0, 9))
}

fn test_expand_dims_and_squeeze_undo_one_another() -> void {
    if not have_gpu() { return }
    let a = ramp([2, 3])
    defer { a.free() }

    for at in [0, 1, 2, -1] {
        let wider = expand_dims(a, at)
        assert_eq(wider.size(), a.size(), "a length-one axis adds no elements")
        assert_eq(squeeze(wider).shape, [2, 3])
        assert_eq(wider.to_list(), a.to_list())
    }
    assert_eq(expand_dims(a, 1).shape, [2, 1, 3])
    assert_eq(expand_dims(a, -1).shape, [2, 3, 1])

    let padded = ramp([1, 3, 1, 2])
    defer { padded.free() }
    assert_eq(squeeze(padded).shape, [3, 2], "every length-one axis by default")
    assert_eq(squeeze(padded, 0).shape, [3, 1, 2])
    assert_eq(squeeze(padded, [0, 2]).shape, [3, 2])
    assert_throws(ValueError, || squeeze(padded, 1), "axis 1 holds three elements")

    #: Free, like every other rewrite here.
    let wider = expand_dims(a, 0)
    wider[0, 1, 2] = 8.0
    assert_eq(a[1, 2].item(), 8.0)
}

fn test_atleast_nd_lifts_where_numpy_lifts() -> void {
    if not have_gpu() { return }
    let scalar = ramp([])
    let vector = ramp([4])
    let plane = ramp([2, 3])
    defer { scalar.free() }
    defer { vector.free() }
    defer { plane.free() }

    assert_eq(atleast_1d(scalar).shape, [1])
    assert_eq(atleast_1d(vector).shape, [4])
    assert_eq(atleast_2d(vector).shape, [1, 4], "a vector becomes one row")
    assert_eq(atleast_2d(plane).shape, [2, 3])

    #: The third axis trails rather than leads, which is what lets `dstack`
    #: lay vectors along the depth axis.
    assert_eq(atleast_3d(vector).shape, [1, 4, 1])
    assert_eq(atleast_3d(plane).shape, [2, 3, 1])
    assert_eq(atleast_3d(scalar).shape, [1, 1, 1])
    assert_eq(atleast_3d(vector).to_list(), vector.to_list())
}

fn test_broadcast_to_and_broadcast_arrays_stretch_without_copying() -> void {
    if not have_gpu() { return }
    let column = ramp([3, 1])
    defer { column.free() }

    let wide = broadcast_to(column, [2, 3, 4])
    assert_eq(wide.shape, [2, 3, 4])
    assert_eq(wide.to_list(), mapped([2, 3, 4], [3, 1], |index| [index[1], 0]))
    assert_eq(wide.strides[0], 0, "a stretched axis reads one slot from every thread")
    assert_eq(wide.strides[2], 0)

    let row = ramp([4])
    let block = ramp([2, 1, 1])
    defer { row.free() }
    defer { block.free() }
    let agreed = broadcast_arrays([column, row, block])
    assert_len(agreed, 3)
    for part in agreed { assert_eq(part.shape, [2, 3, 4], "one common shape") }
    assert_eq(agreed[1].to_list(), mapped([2, 3, 4], [4], |index| [index[2]]))
    assert_eq(broadcast_arrays([]), [])
    assert_throws(ValueError, || broadcast_arrays([ramp([3]), ramp([4])]))
}

fn test_diagonal_is_a_window_on_the_plane() -> void {
    if not have_gpu() { return }
    let a = ramp([3, 4])
    defer { a.free() }

    assert_eq(diagonal(a).to_list(), mapped([3], [3, 4], |index| [index[0], index[0]]))
    assert_eq(diagonal(a, 1).to_list(), mapped([3], [3, 4], |index| [index[0], index[0] + 1]))
    assert_eq(diagonal(a, -1).to_list(), mapped([2], [3, 4], |index| [index[0] + 1, index[0]]))
    assert_eq(diagonal(a, 9).shape, [0], "a diagonal past the corner is empty")

    #: A view and not a gather, because the diagonal's stride is the sum of
    #: the two it replaces. numpy hands back a read-only copy here.
    let main = diagonal(a)
    main[1] = 31.0
    assert_eq(a[1, 1].item(), 31.0)

    let cube = ramp([2, 3, 4])
    defer { cube.free() }
    assert_eq(diagonal(cube, 0, 0, 2).shape, [3, 2], "the diagonal is appended last")
}

fn test_concatenate_lays_the_parts_end_to_end() -> void {
    if not have_gpu() { return }
    let a = ramp([2, 3])
    let b = ramp([4, 3])
    defer { a.free() }
    defer { b.free() }

    let joined = concatenate([a, b], 0)
    defer { joined.free() }
    assert_eq(joined.shape, [6, 3])
    assert_eq(
        joined.to_list(),
        mapped([2, 3], [2, 3], |index| index) + mapped([4, 3], [4, 3], |index| index)
    )

    let side = concatenate([a, a], 1)
    defer { side.free() }
    assert_eq(side.to_list(), mapped([2, 6], [2, 3], |index| [index[0], index[1] % 3]))

    #: A part that is a transpose or a flip is read where it lies; the strided
    #: copy needs no materialising first.
    let turned = concatenate([transpose(a), flip(transpose(a), 0)], 1)
    defer { turned.free() }
    assert_eq(turned.to_list(), mapped([3, 4], [2, 3], |index| if index[1] < 2 {
        [index[1], index[0]]
    } else {
        [index[1] - 2, 2 - index[0]]
    }))

    assert_throws(ValueError, || concatenate([]))
    assert_throws(ValueError, || concatenate([a, ramp([2, 4])], 0), "the other axes must agree")
    assert_throws(ValueError, || concatenate([a, ramp([2])], 0), "and the ranks")
    assert_throws(ValueError, || concatenate([ramp([])]), "a 0-d array has no axis")
}

fn test_stack_puts_each_part_in_the_slot_it_owns() -> void {
    if not have_gpu() { return }
    let a = ramp([2, 3])
    let b = ramp([2, 3])
    defer { a.free() }
    defer { b.free() }

    let front = stack([a, b], 0)
    defer { front.free() }
    assert_eq(front.shape, [2, 2, 3])
    assert_eq(front.to_list(), mapped([2, 2, 3], [2, 3], |index| [index[1], index[2]]))

    let middle = stack([a, b], 1)
    defer { middle.free() }
    assert_eq(middle.shape, [2, 2, 3])
    assert_eq(middle.to_list(), mapped([2, 2, 3], [2, 3], |index| [index[0], index[2]]))

    let back = stack([a, b], -1)
    defer { back.free() }
    assert_eq(back.shape, [2, 3, 2])
    assert_eq(back.to_list(), mapped([2, 3, 2], [2, 3], |index| [index[0], index[1]]))

    assert_throws(ValueError, || stack([]))
    assert_throws(ValueError, || stack([a, ramp([3, 2])]), "stack needs one shape")
}

fn test_the_named_stacks_lift_before_they_join() -> void {
    if not have_gpu() { return }
    let u = ramp([3])
    let v = ramp([3])
    defer { u.free() }
    defer { v.free() }

    #: A 1-d part becomes one row first, which is why vstack of vectors gives
    #: a matrix and hstack of them gives a longer vector.
    let rows = vstack([u, v])
    defer { rows.free() }
    assert_eq(rows.shape, [2, 3])
    assert_eq(rows.to_list(), mapped([2, 3], [3], |index| [index[1]]))

    let run = hstack([u, v])
    defer { run.free() }
    assert_eq(run.shape, [6])
    assert_eq(run.to_list(), mapped([6], [3], |index| [index[0] % 3]))

    let deep = dstack([u, v])
    defer { deep.free() }
    assert_eq(deep.shape, [1, 3, 2])
    assert_eq(deep.to_list(), mapped([1, 3, 2], [3], |index| [index[1]]))

    let columns = column_stack([u, v])
    defer { columns.free() }
    assert_eq(columns.shape, [3, 2])
    assert_eq(columns.to_list(), mapped([3, 2], [3], |index| [index[0]]))

    let plane = ramp([2, 3])
    defer { plane.free() }
    assert_eq(hstack([plane, plane]).shape, [2, 6], "two dimensions join along axis 1")
    assert_eq(column_stack([plane, plane]).shape, [2, 6], "and are left alone by column_stack")
    assert_throws(ValueError, || hstack([]))
}

fn test_split_returns_views_that_cover_the_axis() -> void {
    if not have_gpu() { return }
    let a = ramp([6, 4])
    defer { a.free() }

    let thirds = split(a, 3, 0)
    assert_len(thirds, 3)
    for part in thirds { assert_eq(part.shape, [2, 4]) }
    assert_eq(
        thirds[0].to_list() + thirds[1].to_list() + thirds[2].to_list(),
        a.to_list(),
        "the parts in order are the whole array"
    )

    #: Views, so a write to a part lands in the parent.
    thirds[1][0, 0] = 88.0
    assert_eq(a[2, 0].item(), 88.0)
    a[2, 0] = 8.0

    let cut = split(a, [1, 4], 0)
    assert_eq([part.shape[0] for part in cut], [1, 3, 2], "a list gives the cut indices")
    assert_eq(split(a, [-2], 0)[0].shape[0], 4, "a negative cut counts from the end")

    assert_throws(ValueError, || split(a, 4, 0), "4 does not divide 6")
    assert_throws(ValueError, || split(a, 0, 0))

    #: array_split spreads the remainder one element each over the leading
    #: parts, which is the whole of its difference from split.
    let ragged = array_split(a, 4, 0)
    assert_eq([part.shape[0] for part in ragged], [2, 2, 1, 1])
    assert_eq([part.shape[0] for part in array_split(ramp([7]), 3)], [3, 2, 2])
}

fn test_the_named_splits_pick_their_axis() -> void {
    if not have_gpu() { return }
    let cube = ramp([4, 6, 8])
    defer { cube.free() }

    assert_eq([part.shape for part in vsplit(cube, 2)], [[2, 6, 8], [2, 6, 8]])
    assert_eq([part.shape for part in hsplit(cube, 3)], [[4, 2, 8], [4, 2, 8], [4, 2, 8]])
    assert_eq([part.shape for part in dsplit(cube, 2)], [[4, 6, 4], [4, 6, 4]])
    assert_eq([part.shape for part in hsplit(ramp([6]), 2)], [[3], [3]], "1-d hsplit cuts axis 0")

    assert_throws(ValueError, || vsplit(ramp([4]), 2))
    assert_throws(ValueError, || dsplit(ramp([4, 4]), 2))
}

fn test_tile_repeats_the_whole_block() -> void {
    if not have_gpu() { return }
    let a = ramp([2, 3])
    defer { a.free() }

    let twice = tile(a, [2, 2])
    defer { twice.free() }
    assert_eq(twice.shape, [4, 6])
    assert_eq(twice.to_list(), mapped([4, 6], [2, 3], |index| [index[0] % 2, index[1] % 3]))

    #: Fewer reps than axes pads from the left, so a bare `[3]` tiles the last
    #: axis only.
    let along = tile(a, [3])
    defer { along.free() }
    assert_eq(along.shape, [2, 9])
    assert_eq(along.to_list(), mapped([2, 9], [2, 3], |index| [index[0], index[1] % 3]))

    #: More reps than axes lifts the source, which is where the new leading
    #: axes come from.
    let lifted = tile(ramp([3]), [2, 2])
    defer { lifted.free() }
    assert_eq(lifted.shape, [2, 6])
    assert_eq(lifted.to_list(), mapped([2, 6], [3], |index| [index[1] % 3]))

    assert_eq(tile(a, [0, 2]).size(), 0)
    assert_throws(ValueError, || tile(a, [-1]))
}

fn test_repeat_repeats_each_element() -> void {
    if not have_gpu() { return }
    let a = ramp([2, 3])
    defer { a.free() }

    let along = repeat(a, 2, 1)
    defer { along.free() }
    assert_eq(along.shape, [2, 6])
    assert_eq(along.to_list(), mapped([2, 6], [2, 3], |index| [index[0], index[1] // 2]))

    let down = repeat(a, 3, 0)
    defer { down.free() }
    assert_eq(down.to_list(), mapped([6, 3], [2, 3], |index| [index[0] // 3, index[1]]))

    #: No axis flattens first, which is the case that separates repeat from
    #: tile most clearly.
    let flatly = repeat(a, 2)
    defer { flatly.free() }
    assert_eq(flatly.shape, [12])
    assert_eq(flatly.to_list(), mapped([12], [2, 3], |index| [index[0] // 6, index[0] % 6 // 2]))

    #: A ragged list is one dispatch per source element; a list that happens
    #: to be uniform must take the single-dispatch path and agree with it.
    let ragged = repeat(a, [1, 2, 3], 1)
    defer { ragged.free() }
    assert_eq(ragged.shape, [2, 6])
    let widths = [0, 1, 1, 2, 2, 2]
    assert_eq(ragged.to_list(), mapped([2, 6], [2, 3], |index| [index[0], widths[index[1]]]))
    assert_eq(repeat(a, [2, 2, 2], 1).to_list(), along.to_list())

    assert_eq(repeat(a, 0, 1).size(), 0)
    assert_throws(ValueError, || repeat(a, [1, 2], 1), "one count per element")
    assert_throws(ValueError, || repeat(a, -1, 1))
}

fn test_resize_cycles_the_source() -> void {
    if not have_gpu() { return }
    let a = ramp([2, 3])
    defer { a.free() }

    let grown = resize(a, [4, 4])
    defer { grown.free() }
    assert_eq(grown.to_list(), mapped([16], [6], |index| [index[0] % 6]))

    #: The doubling fill is the part worth checking at a size where it runs
    #: several rounds: 500 elements out of 6 is seven of them.
    let far = resize(a, [500])
    defer { far.free() }
    assert_eq(far.to_list(), mapped([500], [6], |index| [index[0] % 6]))

    let shrunk = resize(a, [4])
    defer { shrunk.free() }
    assert_eq(shrunk.to_list(), [0.0, 1.0, 2.0, 3.0], "a smaller target truncates")
    assert_eq(resize(a, [0]).size(), 0)
    assert_throws(ValueError, || resize(ramp([0]), [4]))
}

fn test_flip_is_a_negated_stride_and_nothing_else() -> void {
    if not have_gpu() { return }
    let a = ramp([2, 3, 4])
    defer { a.free() }

    let last = flip(a, 2)
    assert_eq(
        last.to_list(),
        mapped([2, 3, 4], [2, 3, 4], |index| [index[0], index[1], 3 - index[2]])
    )
    assert_true(last.strides[2] < 0, "a flip walks the axis from the far end")
    assert_eq(last.storage.count, a.storage.count, "and shares the allocation")

    let both = flip(a, [0, 2])
    assert_eq(
        both.to_list(),
        mapped([2, 3, 4], [2, 3, 4], |index| [1 - index[0], index[1], 3 - index[2]])
    )
    let every = flip(a)
    assert_eq(
        every.to_list(),
        mapped([2, 3, 4], [2, 3, 4], |index| [1 - index[0], 2 - index[1], 3 - index[2]])
    )

    #: Free means a write through the window lands in the parent, reversed.
    last[0, 0, 0] = 66.0
    assert_eq(a[0, 0, 3].item(), 66.0)
    a[0, 0, 3] = 3.0

    let plane = ramp([2, 3])
    defer { plane.free() }
    assert_eq(fliplr(plane).to_list(), flip(plane, 1).to_list())
    assert_eq(flipud(plane).to_list(), flip(plane, 0).to_list())
    assert_throws(ValueError, || fliplr(ramp([3])))
}

fn test_roll_moves_by_definition_and_wraps() -> void {
    if not have_gpu() { return }
    let a = ramp([3, 4])
    defer { a.free() }

    #: The definition: out[i] is a[(i - shift) mod n] along the rolled axis.
    let sideways = roll(a, 1, 1)
    defer { sideways.free() }
    assert_eq(sideways.to_list(), mapped([3, 4], [3, 4], |index| [index[0], (index[1] + 3) % 4]))

    let back = roll(a, -1, 0)
    defer { back.free() }
    assert_eq(back.to_list(), mapped([3, 4], [3, 4], |index| [(index[0] + 1) % 3, index[1]]))

    #: A shift past the extent is the same as its remainder, and a zero shift
    #: is still a copy rather than the array itself.
    let round_trip = roll(a, 4, 1)
    defer { round_trip.free() }
    assert_eq(round_trip.to_list(), a.to_list())
    let unmoved = roll(a, 0, 0)
    defer { unmoved.free() }
    assert_ne(unmoved, a, "roll always allocates")

    #: Rolls along different axes commute, so several axes run one after
    #: another and must agree with the composition.
    let both = roll(a, [1, 2], [0, 1])
    defer { both.free() }
    assert_eq(
        both.to_list(),
        mapped([3, 4], [3, 4], |index| [(index[0] + 2) % 3, (index[1] + 2) % 4])
    )
    assert_eq(both.to_list(), roll(roll(a, 1, 0), 2, 1).to_list())

    #: No axis flattens first, so the roll runs across the row boundary.
    let flatly = roll(a, 5)
    defer { flatly.free() }
    assert_eq(flatly.to_list(), mapped([12], [12], |index| [(index[0] + 7) % 12]))
    assert_throws(ValueError, || roll(a, [1, 2], 0), "one shift per named axis")
}

fn test_rot90_is_a_transpose_and_a_flip() -> void {
    if not have_gpu() { return }
    let a = ramp([2, 3])
    defer { a.free() }

    #: A quarter turn anticlockwise sends (row, col) to (cols - 1 - col, row).
    let once = rot90(a)
    assert_eq(once.shape, [3, 2])
    assert_eq(once.to_list(), mapped([3, 2], [2, 3], |index| [index[1], 2 - index[0]]))

    let twice = rot90(a, 2)
    assert_eq(twice.to_list(), mapped([2, 3], [2, 3], |index| [1 - index[0], 2 - index[1]]))

    let thrice = rot90(a, 3)
    assert_eq(thrice.shape, [3, 2])
    assert_eq(thrice.to_list(), mapped([3, 2], [2, 3], |index| [1 - index[1], index[0]]))
    assert_eq(rot90(a, -1).to_list(), thrice.to_list(), "a negative turn is the same modulo four")
    assert_true(rot90(a, 4) == a, "four turns is the identity")

    #: All four ways are stride rewrites, so the write reaches the parent.
    once[0, 0] = 21.0
    assert_eq(a[0, 2].item(), 21.0)

    let cube = ramp([2, 3, 4])
    defer { cube.free() }
    assert_eq(rot90(cube, 1, [1, 2]).shape, [2, 4, 3])
    assert_throws(ValueError, || rot90(ramp([3])))
    assert_throws(ValueError, || rot90(cube, 1, [0, 0]))
}

#: The edge rule, spelled as the clamp it is.
fn edge_at(pos: int, limit: int) -> int { return max(0, min(pos, limit - 1)) }

#: The wrap rule by repeated addition rather than by a remainder, so the
#: reference does not lean on the kernel's one-modulo trick to check it.
fn wrap_at(pos: int, limit: int) -> int {
    var at = pos
    while at < 0 { at += limit }
    while at >= limit { at -= limit }
    return at
}

#: The reflect rule as an actual bounce: fold about the first element, fold
#: about the last, repeat until inside. The kernel folds one remainder
#: instead, and the two agreeing is the point of writing it this way.
fn reflect_at(pos: int, limit: int) -> int {
    if limit == 1 { return 0 }
    var at = pos
    while at < 0 or at >= limit {
        if at < 0 { at = -at }
        if at >= limit { at = 2 * (limit - 1) - at }
    }
    return at
}

fn pad_at(index: list[int], source: list[int], before: list[int], mode: str) -> list[int] {
    var out: list[int] = []
    for axis in 0..index.len() {
        let pos = index[axis] - before[axis]
        if mode == "edge" {
            out.push(edge_at(pos, source[axis]))
        } elif mode == "wrap" {
            out.push(wrap_at(pos, source[axis]))
        } else {
            out.push(reflect_at(pos, source[axis]))
        }
    }
    return out
}

#: `pad_at` against a source that is itself a window, so the reference has to
#: carry the parent's own offset back into the answer.
fn shifted_pad_at(
    index: list[int],
    source: list[int],
    before: list[int],
    mode: str,
    shift: list[int]
) -> list[int] {
    let inner = pad_at(index, source, before, mode)
    var out: list[int] = []
    for axis in 0..inner.len() { out.push(inner[axis] + shift[axis]) }
    return out
}

#: `pad_at` against a source whose last axis is read backwards.
fn flipped_pad_at(index: list[int], source: list[int], before: list[int], mode: str) -> list[int] {
    var out = pad_at(index, source, before, mode)
    let last = out.len() - 1
    out[last] = source[last] - 1 - out[last]
    return out
}

fn test_constant_padding_is_a_fill_and_one_copy() -> void {
    if not have_gpu() { return }
    let a = ramp([2, 3])
    defer { a.free() }

    let bordered = pad(a, [[1, 2], [2, 1]], "constant", -1.0)
    defer { bordered.free() }
    assert_eq(bordered.shape, [5, 6])

    var expected: list[float] = []
    for index in indices_of([5, 6]) {
        let row = index[0] - 1
        let col = index[1] - 2
        let inside = row >= 0 and row < 2 and col >= 0 and col < 3
        expected.push(if inside { float(flat_index([2, 3], [row, col])) } else { -1.0 })
    }
    assert_eq(bordered.to_list(), expected)

    let zeroed = pad(a, [[0, 0], [1, 1]])
    defer { zeroed.free() }
    assert_eq(zeroed.to_list()[0], 0.0, "the default value is zero")
    assert_eq(pad(a, [[0, 0], [0, 0]]).to_list(), a.to_list(), "no border is a copy")

    #: An empty axis has no interior to copy, but a constant border over it is
    #: still well defined — unlike the modes that read from the array.
    assert_eq(pad(ramp([0, 3]), [[1, 1], [0, 0]], "constant", 7.0).shape, [2, 3])
}

fn test_the_border_modes_that_read_from_the_array() -> void {
    if not have_gpu() { return }
    let a = ramp([3, 4])
    defer { a.free() }
    let before = [2, 3]
    let widths = [[2, 2], [3, 3]]

    for mode in ["edge", "wrap", "reflect"] {
        let bordered = pad(a, widths, mode)
        defer { bordered.free() }
        assert_eq(bordered.shape, [7, 10])
        assert_eq(
            bordered.to_list(),
            mapped([7, 10], [3, 4], |index| pad_at(index, [3, 4], before, mode)),
            mode
        )
    }

    #: A border wider than the axis it folds over is where a reflection has to
    #: bounce more than once, and where a naive single fold would be wrong.
    let deep = pad(ramp([3]), [[7, 8]], "reflect")
    defer { deep.free() }
    assert_eq(deep.to_list(), mapped([18], [3], |index| [reflect_at(index[0] - 7, 3)]))

    #: A single column has no reflection period at all.
    let thin = pad(ramp([1, 2]), [[2, 2], [0, 0]], "reflect")
    defer { thin.free() }
    assert_eq(thin.to_list(), mapped([5, 2], [1, 2], |index| [0, index[1]]))

    assert_throws(ValueError, || pad(a, [[1, 1]], "edge"), "one pair per axis")
    assert_throws(ValueError, || pad(a, [[1], [1, 1]], "edge"))
    assert_throws(ValueError, || pad(a, [[-1, 0], [0, 0]], "edge"))
    assert_throws(ValueError, || pad(a, [[0, 0], [0, 0]], "mirror"), "an unknown mode")
    assert_throws(ValueError, || pad(ramp([0, 3]), [[1, 1], [0, 0]], "wrap"), "nothing to read")
}

#: The kernel moves four destination elements per thread and takes a float4
#: path through the innermost axis's interior. Both halves of that decision
#: need a case, and so does the source layout it refuses to vectorise over.
fn test_the_pad_kernel_agrees_with_itself_on_every_path() -> void {
    if not have_gpu() { return }
    let before = [2, 3]
    let widths = [[2, 3], [3, 4]]

    #: A destination wide enough that most threads land squarely inside the
    #: interior run, which is the float4 path.
    let block = ramp([13, 21])
    defer { block.free() }
    for mode in ["edge", "wrap", "reflect"] {
        let bordered = pad(block, widths, mode)
        defer { bordered.free() }
        assert_eq(bordered.shape, [18, 28])
        assert_eq(
            bordered.to_list(),
            mapped([18, 28], [13, 21], |index| pad_at(index, [13, 21], before, mode)),
            mode
        )
    }

    #: A slice starts at an offset the vector load cannot use, so the same
    #: values have to come back through the scalar tail.
    let parent = ramp([9, 11])
    defer { parent.free() }
    let window = parent[1..8, 1..10]
    let offcut = pad(window, widths, "edge")
    defer { offcut.free() }
    assert_eq(
        offcut.to_list(),
        mapped([12, 16], [9, 11], |index| shifted_pad_at(index, [7, 9], before, "edge", [1, 1]))
    )

    #: A flipped source has a negative innermost stride, which rules the
    #: vector load out for a different reason.
    let backwards = pad(flip(parent, 1), [[1, 1], [1, 1]], "wrap")
    defer { backwards.free() }
    assert_eq(
        backwards.to_list(),
        mapped([11, 13], [9, 11], |index| flipped_pad_at(index, [9, 11], [1, 1], "wrap"))
    )
}

fn test_padding_carries_both_lanes_of_a_complex_array() -> void {
    if not have_gpu() { return }
    let a = ramp([2, 3], COMPLEX64)
    defer { a.free() }

    let bordered = pad(a, [[1, 1], [1, 1]], "edge")
    defer { bordered.free() }
    assert_eq(bordered.shape, [4, 5])
    assert_eq(bordered.dtype, COMPLEX64)

    #: `ramp` numbers the float slots, so the real part of element k is 2k and
    #: the imaginary part 2k + 1. Both must follow the same border rule.
    var expected: list[float] = []
    for index in indices_of([4, 5]) {
        let at = pad_at(index, [2, 3], [1, 1], "edge")
        let element = flat_index([2, 3], at)
        expected.push(float(2 * element))
        expected.push(float(2 * element + 1))
    }
    assert_eq(bordered.to_list(), expected)
}

fn test_the_joins_promote_and_the_views_keep_their_dtype() -> void {
    if not have_gpu() { return }
    let whole = ramp([2, 3], INT32)
    let real = ramp([2, 3], FLOAT32)
    defer { whole.free() }
    defer { real.free() }

    assert_eq(concatenate([whole, whole]).dtype, INT32, "a join folds the promotion lattice")
    assert_eq(concatenate([whole, real]).dtype, FLOAT32)
    assert_eq(stack([whole, real]).dtype, FLOAT32)
    assert_eq(transpose(whole).dtype, INT32, "a view is the same array read differently")
    assert_eq(tile(whole, [2]).dtype, INT32)
    assert_eq(pad(whole, [[1, 1], [1, 1]], "edge").dtype, INT32)
}

fn test_the_mode_constants_are_the_ones_the_kernel_switches_on() -> void {
    assert_eq(PAD_EDGE, 0)
    assert_eq(PAD_WRAP, 1)
    assert_eq(PAD_REFLECT, 2)
}

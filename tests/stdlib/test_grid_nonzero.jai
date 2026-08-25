#: `__prim__.grid_nonzero` — a picture scattered into a padded grid as ones and
#: zeros.
#:
#: `dst[origin + y * stride + x] = src[y * cols + x] != 0 ? 1 : 0`. It exists
#: because the two loops that spell it out are at the language's floor and the
#: floor is too high: 4.95 ms for a 1080p frame against 1.62 ms here, and
#: reading the source as `bytes` instead of `list[float]` measured 4.82, so it
#: is the per-element boxing rather than the read. Every border follower and
#: flood fill starts by building exactly this.
#:
#: What is checked is the ANSWER against the loops it replaces, on both source
#: kinds, plus every way the arguments can be wrong -- a primitive that reads
#: past the end of a list is a crash, not an exception, so the bounds tests
#: here are the ones that matter most.
import std.math
from std.test import assert_eq, assert_true, assert_throws

fn by_hand(src: list[float], rows: int, cols: int, stride: int, origin: int) -> list[int] {
    var out = __prim__.list_filled(origin + rows * stride, 0)
    for y in 0..rows {
        for x in 0..cols {
            out[origin + y * stride + x] = if src[y * cols + x] != 0.0 { 1 } else { 0 }
        }
    }
    return out
}

fn test_it_agrees_with_the_loops_it_replaces() -> void {
    let rows = 7
    let cols = 5
    let stride = cols + 2
    let origin = stride + 1
    var src: list[float] = []
    for index in 0..rows * cols { src.push(float(index % 3) - 1.0) }

    let wanted = by_hand(src, rows, cols, stride, origin)
    var got = __prim__.list_filled(wanted.len(), 0)
    __prim__.grid_nonzero(got, src, rows, cols, stride, origin)
    assert_eq(got, wanted, "the padded grid")
}

#: The ring the caller allocated must come back untouched -- that is the whole
#: point of the padding, and a primitive writing one row too far would still
#: pass a test that only looked at the picture.
fn test_the_ring_is_left_alone() -> void {
    let rows = 3
    let cols = 4
    let stride = cols + 2
    let origin = stride + 1
    var got = __prim__.list_filled(stride * (rows + 2), 9)
    var src: list[float] = []
    for _index in 0..rows * cols { src.push(1.0) }
    __prim__.grid_nonzero(got, src, rows, cols, stride, origin)

    for x in 0..stride { assert_eq(got[x], 9, f"the top ring at {x}") }
    for y in 0..rows {
        assert_eq(got[(y + 1) * stride], 9, f"the left ring on row {y}")
        assert_eq(got[(y + 1) * stride + cols + 1], 9, f"the right ring on row {y}")
        for x in 0..cols { assert_eq(got[origin + y * stride + x], 1, f"the picture") }
    }
    for x in 0..stride {
        assert_eq(got[(rows + 1) * stride + x], 9, f"the bottom ring at {x}")
    }
}

#: bytes and a list of ints are both sources, and must give the same grid.
fn test_every_source_kind_gives_the_same_grid() -> void {
    let rows = 4
    let cols = 4
    let stride = cols + 2
    let origin = stride + 1
    var floats: list[float] = []
    var ints: list[int] = []
    for index in 0..rows * cols {
        floats.push(float(index % 2))
        ints.push(index % 2)
    }
    let raw = __prim__.bytes_quantise(floats, 1.0)

    var a = __prim__.list_filled(stride * (rows + 2), 0)
    var b = __prim__.list_filled(stride * (rows + 2), 0)
    var c = __prim__.list_filled(stride * (rows + 2), 0)
    __prim__.grid_nonzero(a, floats, rows, cols, stride, origin)
    __prim__.grid_nonzero(b, ints, rows, cols, stride, origin)
    __prim__.grid_nonzero(c, raw, rows, cols, stride, origin)
    assert_eq(a, b, "floats and ints")
    assert_eq(a, c, "floats and bytes")
}

#: A negative and a NaN are both non-zero, which is what `!= 0` says and what a
#: mask holding -1 for "set" depends on.
fn test_anything_that_is_not_zero_is_one() -> void {
    let src = [0.0, -1.0, 0.5, -0.0, 1e300, math.NAN]
    var got = __prim__.list_filled(6, 7)
    __prim__.grid_nonzero(got, src, 1, 6, 6, 0)
    #: -0.0 == 0.0, so it is background; a NaN is not equal to zero, so it is
    #: foreground.
    assert_eq(got, [0, 1, 1, 0, 1, 1], "the ones and zeros")
}

fn test_a_destination_too_small_is_refused() -> void {
    assert_throws(
        ValueError,
        fn() -> void {
            #: The last cell is origin 4 + one stride + two columns = index
            #: 9, so ten values are exactly enough and nine are one short.
            var got = __prim__.list_filled(9, 0)
            __prim__.grid_nonzero(got, [1.0, 1.0, 1.0, 1.0], 2, 2, 4, 4)
        },
        "a destination that does not reach the last cell"
    )
}

fn test_a_source_too_small_is_refused() -> void {
    assert_throws(ValueError, fn() -> void {
        var got = __prim__.list_filled(64, 0)
        __prim__.grid_nonzero(got, [1.0, 1.0], 2, 2, 4, 0)
    }, "a source with fewer values than the rectangle")
}

fn test_columns_wider_than_the_stride_are_refused() -> void {
    assert_throws(ValueError, fn() -> void {
        var got = __prim__.list_filled(64, 0)
        __prim__.grid_nonzero(got, [1.0, 1.0, 1.0, 1.0], 2, 2, 1, 0)
    }, "columns that do not fit the stride")
}

fn test_a_non_number_in_the_source_is_refused() -> void {
    assert_throws(TypeError, fn() -> void {
        var got = __prim__.list_filled(64, 0)
        __prim__.grid_nonzero(got, [1.0, "two"], 1, 2, 2, 0)
    }, "a string among the values")
}

fn test_an_empty_picture_writes_nothing() -> void {
    var got = __prim__.list_filled(4, 5)
    __prim__.grid_nonzero(got, [], 0, 0, 0, 0)
    assert_eq(got, [5, 5, 5, 5], "nothing was written")
}

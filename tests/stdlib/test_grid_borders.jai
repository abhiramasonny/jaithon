#: `__prim__.grid_borders` — Suzuki and Abe's border following over a whole
#: picture in one call, the scan and the walk behind jaicv's `find_contours`.
#:
#: The expected borders below are what `cv2.findContours` gives for the same
#: pictures, in discovery order (RETR_LIST hands them back reversed). jaicv's
#: own tests hold the retrieval modes and the hierarchy against OpenCV and
#: against flood fills; this file holds the primitive itself: every source kind
#: gives the same answer, a window of a larger picture reads only its own
#: pixels, and every way the arguments can be wrong is an exception rather than
#: a read past the end of something.
from std.test import assert_eq, assert_true, assert_throws

#: A three-by-three square with its middle cleared, inside a ring of clear.
fn square_with_hole() -> list[float] {
    var values: list[float] = []
    for y in 0..5 {
        for x in 0..5 {
            let inside = y >= 1 and y <= 3 and x >= 1 and x <= 3
            values.push(if inside and (y != 2 or x != 2) { 255.0 } else { 0.0 })
        }
    }
    return values
}

fn borders(
    src: any,
    origin: int,
    step: int,
    rows: int,
    cols: int,
    approx: int,
    outer: int
) -> list[list[int]] {
    var meta: list[int] = []
    var points: list[int] = []
    __prim__.grid_borders(src, origin, step, rows, cols, approx, outer, meta, points)
    return [meta, points]
}

fn test_a_square_with_a_hole_matches_opencv() -> void {
    let got = borders(square_with_hole(), 0, 5, 5, 5, 1, 0)
    #: The outer border, a child of the frame; then the hole, a child of it.
    assert_eq(got[0], [1, 1, 8, 0, 2, 12])
    assert_eq(got[1], [1, 1, 1, 2, 1, 3, 2, 3, 3, 3, 3, 2, 3, 1, 2, 1, 1, 2, 2, 1, 3, 2, 2, 3])
}

fn test_simple_keeps_the_corners() -> void {
    let got = borders(square_with_hole(), 0, 5, 5, 5, 2, 0)
    assert_eq(got[0], [1, 1, 4, 0, 2, 8])
    assert_eq(got[1], [1, 1, 1, 3, 3, 3, 3, 1, 1, 2, 2, 1, 3, 2, 2, 3])
}

#: Outer-only skips the hole and everything in it.
fn test_outer_only_skips_the_hole() -> void {
    let got = borders(square_with_hole(), 0, 5, 5, 5, 1, 1)
    assert_eq(got[0], [1, 1, 8])
    assert_eq(got[1], [1, 1, 1, 2, 1, 3, 2, 3, 3, 3, 3, 2, 3, 1, 2, 1])
}

#: Single pixels in the corners and a square, which touch the picture's edge.
fn test_shapes_on_the_edge() -> void {
    var values: list[int] = []
    for y in 0..4 {
        for x in 0..6 {
            let corner = y == 0 and x == 0 or y == 3 and x == 5
            let block = y >= 1 and y <= 2 and x >= 2 and x <= 3
            values.push(if corner or block { 1 } else { 0 })
        }
    }
    let got = borders(values, 0, 6, 4, 6, 1, 0)
    assert_eq(got[0], [1, 1, 1, 1, 1, 5, 1, 1, 6])
    assert_eq(got[1], [0, 0, 2, 1, 2, 2, 3, 2, 3, 1, 5, 3])
}

#: Floats, ints, boxed numbers and bytes are four ways to say the same picture.
fn test_every_source_kind_agrees() -> void {
    let floats = square_with_hole()
    var ints: list[int] = []
    var boxed: list[any] = []
    var raw: list[int] = []
    for value in floats {
        ints.push(int(value))
        boxed.push(if value != 0.0 { 3 } else { 0.0 })
        raw.push(int(value))
    }
    let want = borders(floats, 0, 5, 5, 5, 1, 0)
    assert_eq(borders(ints, 0, 5, 5, 5, 1, 0), want)
    assert_eq(borders(boxed, 0, 5, 5, 5, 1, 0), want)
    assert_eq(borders(bytes(raw), 0, 5, 5, 5, 1, 0), want)
}

#: A window reads its own pixels and nothing either side of them.
fn test_a_window_reads_only_itself() -> void {
    let inner = square_with_hole()
    var wide: list[float] = []
    for y in 0..7 {
        for x in 0..9 {
            let iy = y - 1
            let ix = x - 3
            let within = iy >= 0 and iy < 5 and ix >= 0 and ix < 5
            wide.push(if within { inner[iy * 5 + ix] } else { 1.0 })
        }
    }
    assert_eq(borders(wide, 9 + 3, 9, 5, 5, 1, 0), borders(inner, 0, 5, 5, 5, 1, 0))
}

fn test_an_empty_picture_has_no_borders() -> void {
    assert_eq(borders([0.0, 0.0, 0.0, 0.0], 0, 2, 2, 2, 2, 0), [[], []])
    assert_eq(borders([], 0, 0, 0, 0, 2, 0), [[], []])
}

fn test_bad_arguments_throw() -> void {
    let src = square_with_hole()
    var meta: list[int] = []
    var points: list[int] = []
    #: Six rows of five need thirty values; there are twenty-five.
    assert_throws(ValueError, fn() -> void {
        __prim__.grid_borders(src, 0, 5, 6, 5, 1, 0, meta, points)
    })
    #: Starting one in, the last row runs one past the end.
    assert_throws(ValueError, fn() -> void {
        __prim__.grid_borders(src, 1, 5, 5, 5, 1, 0, meta, points)
    })
    assert_throws(ValueError, fn() -> void {
        __prim__.grid_borders(src, 0, 4, 5, 5, 1, 0, meta, points)
    })
    assert_throws(ValueError, fn() -> void {
        __prim__.grid_borders(src, 0, 5, -1, 5, 1, 0, meta, points)
    })
    assert_throws(ValueError, fn() -> void {
        __prim__.grid_borders(src, 0, 5, 5, 5, 3, 0, meta, points)
    })
    assert_throws(TypeError, fn() -> void {
        __prim__.grid_borders("picture", 0, 5, 5, 5, 1, 0, meta, points)
    })
    assert_throws(TypeError, fn() -> void {
        __prim__.grid_borders([1.0, "two"], 0, 2, 1, 2, 1, 0, meta, points)
    })
    assert_throws(TypeError, fn() -> void { __prim__.grid_borders(src, 0, 5, 5, 5, 1, 0, meta, 7) })
    assert_true(meta.len() == 0)
}

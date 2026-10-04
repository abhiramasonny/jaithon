#: Where jaicv's contour primitives stop and hand back to the Jaithon.
#:
#: `__prim__.fill_strokes` draws both `draw_contours`' wide outlines and
#: `polylines`' wide runs. The two Jaithon paths it replaces part ways on far
#: points: one squares a segment's length in integers and throws when that
#: overflows, the other squares it in floats and draws. So the primitive
#: declines a far point, drawing nothing, and leaves the answer to whichever
#: path called it; it used to throw for both.
#:
#: `__prim__.points_approx` is Douglas and Peucker, whose recursion nests as
#: deep as a curve has points on a spiral. It used to recurse on the C stack
#: and kill the process with no message on a long enough one.
from std.test import assert_eq, assert_true

class Spot {
    var x: int
    var y: int

    fn init(self, x: int, y: int) {
        self.x = x
        self.y = y
    }
}

fn stroke(contours: list[list[Spot]], thickness: int, closed: bool) -> list[any] {
    var canvas: list[float] = []
    for _ in 0..400 { canvas.push(0.0) }
    let drawn = __prim__.fill_strokes(
        canvas,
        [255.0],
        1,
        0,
        20,
        20,
        20,
        contours,
        0,
        0,
        thickness,
        closed
    )
    var lit = 0
    for value in canvas {
        if value != 0.0 { lit += 1 }
    }
    return [drawn, lit]
}

fn test_a_stroke_whose_length_squared_overflows_is_declined() -> void {
    let far = stroke([[Spot(5, 5), Spot(5, 3100000000)]], 2, false)
    assert_eq(far[0], false)
    assert_eq(far[1], 0)
    let across = stroke([[Spot(-2000000000, 10), Spot(2000000000, 10)]], 3, false)
    assert_eq(across[0], false)
    assert_eq(across[1], 0)
}

fn test_a_far_closed_stroke_is_declined_too() -> void {
    let far = stroke([[Spot(5, 5), Spot(5, 5000000000)]], 5, true)
    assert_eq(far[0], false)
    assert_eq(far[1], 0)
}

#: A point past 2^47 wraps through fixed point, and `polylines` caps it where
#: it wraps to; here that is on the canvas.
fn test_a_point_that_wraps_through_fixed_point_is_declined() -> void {
    let wrapped = (1 << 48) + 5
    let far = stroke([[Spot(wrapped, 5), Spot(wrapped, 12)]], 2, false)
    assert_eq(far[0], false)
    assert_eq(far[1], 0)
}

fn test_a_stroke_on_the_canvas_is_still_drawn() -> void {
    let near = stroke([[Spot(5, 5), Spot(5, 12)]], 2, false)
    assert_eq(near[0], true)
    assert_true(near[1] > 0, "nothing was drawn")
}

#: An octagonal spiral, open, which `points_approx` keeps every point of and
#: nests once a point to do it.
fn spiral_source(count: int) -> str {
    return "class Spot {\n" +
        "    var x: int\n" +
        "    var y: int\n" +
        "    fn init(self, x: int, y: int) {\n" +
        "        self.x = x\n" +
        "        self.y = y\n" +
        "    }\n" +
        "}\n" +
        "let diagonal = 0.7071067811865476\n" +
        "let cosines = [1.0, diagonal, 0.0, -diagonal, -1.0, -diagonal, 0.0, diagonal]\n" +
        "var points: list[Spot] = []\n" +
        f"for k in 0..{count} " +
        "{\n" +
        "    let r = float(10 + k)\n" +
        "    points.push(Spot(int(r * cosines[k % 8]), int(r * cosines[(k + 6) % 8])))\n" +
        "}\n" +
        "let kept = __prim__.points_approx(points, 0.5, false)\n" +
        "print(kept.len())\n"
}

#: Run in a child with a 512 KB stack, where 20,000 C frames do not fit.
fn test_a_curve_nested_deeper_than_the_c_stack_is_simplified() -> void {
    let dir = (__prim__.os_env("TMPDIR") ?? "/tmp") + "/jaithon-deep-curve"
    let made = __prim__.os_spawn(["/bin/mkdir", "-p", dir], {})
    assert_eq(made["exit_code"], 0)
    let path = dir + "/spiral.jai"
    let handle = __prim__.io_open(path, "wb")
    __prim__.io_write(handle, spiral_source(20000))
    __prim__.io_close(handle)

    let binary = __prim__.os_argv()[0]
    let result = __prim__.os_spawn(
        ["/bin/sh", "-c", "ulimit -s 512 && exec \"$0\" run \"$1\"", binary, path],
        {}
    )
    __prim__.io_remove(path)
    assert_eq(result["stderr"], "")
    assert_eq(result["exit_code"], 0)
    assert_eq(result["stdout"], "20000\n")
}

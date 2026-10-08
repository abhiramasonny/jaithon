#: The decoding half of the detector, on a network output made up by hand, so
#: it runs without the weights. A script rather than a `jaithon test` file
#: because a test file is loaded without a package, and the detector's
#: relative imports need this directory to be one:
#:
#:     jaithon run examples/yolo_detect/decode_check.jai
import std.gpu as gpu
import std.math
import jaitensor as jt
from jaitensor import Tensor
from std.test import assert_eq, assert_true
from .detector import decode, pixel_box

#: One cell of a `1 x 84 x cells` output scoring `score` for `class_id`, with
#: the box given by its edges in original-image pixels (the letterbox here is
#: scale 1 and no padding, so letterboxed and original pixels are the same).
fn one_box(
    cells: int,
    left: float,
    top: float,
    right: float,
    bottom: float,
    class_id: int,
    score: float
) -> Tensor {
    var values: list[float] = []
    for _index in 0..84 * cells { values.push(0.0) }
    values[0] = (left + right) * 0.5
    values[cells] = (top + bottom) * 0.5
    values[2 * cells] = right - left
    values[3 * cells] = bottom - top
    values[(4 + class_id) * cells] = score
    return jt.Tensor([1, 84, cells], values)
}

fn test_a_box_keeps_its_fractional_edges() -> void {
    if not gpu.is_available() { return }
    let output = one_box(8, 10.7, 20.2, 50.6, 60.9, 3, 0.9)
    let found = decode(output, 1.0, 0, 0, 640, 480)
    output.free()
    assert_eq(found.len(), 1)
    let hit = found[0]
    assert_eq(hit.class_id, 3)
    #: The network's own numbers, to float32 precision -- not truncated.
    assert_true(math.fabs(hit.x - 10.7) < 0.001)
    assert_true(math.fabs(hit.y - 20.2) < 0.001)
    assert_true(math.fabs(hit.w - 39.9) < 0.001)
    assert_true(math.fabs(hit.h - 40.7) < 0.001)
    #: Each edge rounded to its nearest pixel. Truncating both edges gave
    #: (10, 20, 40, 40): the box slid up and to the left by up to a pixel.
    assert_eq(hit.box.x, 11)
    assert_eq(hit.box.y, 20)
    assert_eq(hit.box.width, 40)
    assert_eq(hit.box.height, 41)
}

fn test_a_box_is_clipped_to_the_image_in_floating_point() -> void {
    if not gpu.is_available() { return }
    let output = one_box(8, -5.25, 470.5, 30.25, 490.0, 0, 0.8)
    let found = decode(output, 1.0, 0, 0, 640, 480)
    output.free()
    assert_eq(found.len(), 1)
    let hit = found[0]
    assert_true(math.fabs(hit.x) < 0.001)
    assert_true(math.fabs(hit.w - 30.25) < 0.001)
    assert_true(math.fabs(hit.y - 470.5) < 0.001)
    assert_true(math.fabs(hit.h - 9.5) < 0.001)
}

fn test_pixel_boxes_round_each_edge() -> void {
    let box = pixel_box(0.49, 0.5, 9.0, 2.2)
    assert_eq(box.x, 0)
    assert_eq(box.y, 1)
    assert_eq(box.width, 9)
    assert_eq(box.height, 2)
}

fn main() -> int {
    test_a_box_keeps_its_fractional_edges()
    test_a_box_is_clipped_to_the_image_in_floating_point()
    test_pixel_boxes_round_each_edge()
    print("decode_check: 3 passed")
    return 0
}

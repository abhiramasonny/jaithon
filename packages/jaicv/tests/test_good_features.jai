#: The strongest corners found by the short route agree with the long one.
#:
#: By default only the strongest `max_corners` times a margin are brought back
#: and ranked, and when tens of thousands tie at the cut only the ones that can
#: reach the answer are. `all_candidates: true` ranks every candidate in the
#: frame instead. Both must give the same corners in the same order, on a frame
#: whose repeating texture makes nearly every candidate a tie -- which is where
#: a ranking that loses track of which strength belongs to which candidate
#: would show.
import std.gpu as gpu
from std.test import assert_eq, assert_true
import jaicv as cv
from jaicv import Mat

fn repeating(rows: int, cols: int) -> Mat {
    var values: list[float] = []
    for index in 0..rows * cols { values.push(float(index * 37 % 251) / 251.0) }
    return Mat.from_list(rows, cols, cv.CV_32FC1, values)
}

fn same_corners(image: Mat, max_corners: int, min_distance: float) -> void {
    let quick = cv.good_features_to_track(image, max_corners, 0.01, min_distance)
    let whole = cv.good_features_to_track(image, max_corners, 0.01, min_distance, all_candidates: true)
    assert_true(quick.len() > 0)
    assert_eq(quick.len(), whole.len())
    for index in 0..quick.len() {
        assert_eq(quick[index].x, whole[index].x)
        assert_eq(quick[index].y, whole[index].y)
    }
}

fn test_a_tied_frame_gives_the_same_corners_either_way() -> void {
    if not gpu.is_available() { return }
    let image = repeating(240, 320)
    same_corners(image, 5, 6.0)
    same_corners(image, 20, 3.0)
    same_corners(image, 60, 10.0)
    image.release()
}

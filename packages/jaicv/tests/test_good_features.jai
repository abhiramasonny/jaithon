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
    let whole = cv.good_features_to_track(
        image,
        max_corners,
        0.01,
        min_distance,
        all_candidates: true
    )
    assert_true(quick.len() > 0)
    assert_eq(quick.len(), whole.len())
    for index in 0..quick.len() {
        assert_eq(quick[index].x, whole[index].x)
        assert_eq(quick[index].y, whole[index].y)
    }
}

#: Strengths that hardly ever tie, so the ranking cannot deal them out by
#: value and has to find the strongest few another way.
fn speckled(rows: int, cols: int) -> Mat {
    var values: list[float] = []
    var state = 12345
    for _index in 0..rows * cols {
        state = (state * 1103515245 + 12345) % 2147483648
        values.push(float(state // 65536 % 10007) / 10007.0)
    }
    return Mat.from_list(rows, cols, cv.CV_32FC1, values)
}

fn test_a_speckled_frame_gives_the_same_corners_either_way() -> void {
    if not gpu.is_available() { return }
    let image = speckled(240, 320)
    same_corners(image, 3, 4.0)
    same_corners(image, 10, 8.0)
    image.release()
}

#: Good features worked out on the host from the same response: every
#: interior pixel at least the floor and no smaller than any neighbour,
#: strongest first with ties in raster order, then spaced out greedily.
fn reference(image: Mat, max_corners: int, quality: float, min_distance: float) -> list[float] {
    let response = cv.corner_min_eigen_val(image, 3, 3)
    let values = response.to_list()
    response.release()
    let rows = image.rows
    let cols = image.cols
    var strongest = values[0]
    for value in values { if value > strongest { strongest = value } }
    if strongest <= 0.0 { return [] }
    let floor_value = strongest * quality
    var found: list[int] = []
    for y in 1..rows - 1 {
        for x in 1..cols - 1 {
            let here = values[y * cols + x]
            if here < floor_value { continue }
            var peak = true
            for dy in -1..2 {
                for dx in -1..2 {
                    if values[(y + dy) * cols + x + dx] > here { peak = false }
                }
            }
            if peak { found.push(y * cols + x) }
        }
    }
    let ordered = found.sort(key: |at| values[at], reverse: true)
    var kept: list[float] = []
    for at in ordered {
        if kept.len() >= max_corners * 2 { break }
        let x = float(at % cols)
        let y = float(at // cols)
        var clear = true
        var index = 0
        while index < kept.len() {
            let dx = kept[index] - x
            let dy = kept[index + 1] - y
            if dx * dx + dy * dy < min_distance * min_distance { clear = false }
            index += 2
        }
        if clear {
            kept.push(x)
            kept.push(y)
        }
    }
    return kept
}

fn matches_reference(image: Mat, max_corners: int, min_distance: float) -> void {
    let got = cv.good_features_to_track(image, max_corners, 0.01, min_distance)
    let want = reference(image, max_corners, 0.01, min_distance)
    assert_eq(got.len() * 2, want.len())
    for index in 0..got.len() {
        assert_eq(got[index].x, want[index * 2])
        assert_eq(got[index].y, want[index * 2 + 1])
    }
}

fn test_corners_match_a_host_reference() -> void {
    if not gpu.is_available() { return }
    let tied = repeating(60, 80)
    let speckle = speckled(60, 80)
    for image in [tied, speckle] {
        matches_reference(image, 4, 5.0)
        matches_reference(image, 15, 3.0)
        matches_reference(image, 40, 2.0)
    }
    tied.release()
    speckle.release()
}

fn test_a_tied_frame_gives_the_same_corners_either_way() -> void {
    if not gpu.is_available() { return }
    let image = repeating(240, 320)
    same_corners(image, 5, 6.0)
    same_corners(image, 20, 3.0)
    same_corners(image, 60, 10.0)
    image.release()
}

#: The survey that finds the strongest response must read an empty frame and a
#: response of NaNs as having no corners, the way the reduction it replaced
#: did. An untouched maximum used to decode as a NaN, and comparing it to zero
#: threw.
fn test_an_empty_frame_has_no_corners() -> void {
    if not gpu.is_available() { return }
    let empty = Mat.zeros(0, 0, cv.CV_8UC1)
    assert_eq(cv.good_features_to_track(empty, 10, 0.01, 1.0).len(), 0)
}

fn test_a_response_that_overflows_has_no_corners() -> void {
    if not gpu.is_available() { return }
    var values: list[float] = []
    for index in 0..1600 { values.push(float(index * 37 % 251) * 1e30) }
    let huge = Mat.from_list(40, 40, cv.CV_32FC1, values)
    assert_eq(cv.good_features_to_track(huge, 10, 0.01, 1.0).len(), 0)
    huge.release()
}

#: Above one, the quality level puts the floor over the strongest corner, so
#: nothing qualifies -- OpenCV answers with no corners rather than an error.
fn test_a_quality_level_above_one_finds_nothing() -> void {
    if not gpu.is_available() { return }
    let image = repeating(40, 40)
    assert_eq(cv.good_features_to_track(image, 10, 8.0, 1.0).len(), 0)
    assert_true(cv.good_features_to_track(image, 10, 0.01, 1.0).len() > 0)
    image.release()
}

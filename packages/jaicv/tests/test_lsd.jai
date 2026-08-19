#: The line segment detector.
#:
#: The images are drawn from formulas rather than by `line` and `rectangle`, so
#: the same pixels can be handed to OpenCV without a codec or a rasteriser in
#: between. What is asserted is of two kinds. Most of it is geometry the image
#: fixes on its own -- a rectangle outline is four segments, they sit on the
#: four edges, rotating the image by a right angle rotates them with it -- and
#: the tolerances are stated where they are used. The rest is the comparison
#: against OpenCV 5.0.0's own `cv2.createLineSegmentDetector`, run on these
#: exact images and written out below.
#:
#: Measured over the images here at all three refinement settings, every
#: segment this finds matches one of OpenCV's with every endpoint inside 0.011
#: of a pixel, and the counts agree except on noise: OpenCV's
#: `LSD_REFINE_NONE` and `LSD_REFINE_STD` skip the a-contrario test and report
#: nine segments on the noise image below where this reports none. OpenCV's own
#: `LSD_REFINE_ADV`, which does run the test, also reports none. The module's
#: own doc comment carries the wider measurement and what is behind it.
import std.math
from std.test import assert_eq, assert_near, assert_throws, assert_true
from jaicv.core.mat import Mat
from jaicv.core.types import CV_8UC1, CV_8UC3, Size
from jaicv.imgproc.lsd import create_line_segment_detector, LineSegmentDetector
from jaicv.imgproc.lsd import LSD_REFINE_ADV, LSD_REFINE_NONE, LSD_REFINE_STD

#: A filled rectangle, `left` to `right` and `top` to `bottom` inclusive.
fn filled_rectangle(rows: int, cols: int, left: int, top: int, right: int, bottom: int) -> Mat {
    var values: list[float] = []
    for y in 0..rows {
        for x in 0..cols {
            let inside = x >= left and x <= right and y >= top and y <= bottom
            values.push(if inside { 255.0 } else { 0.0 })
        }
    }
    return Mat.from_list(rows, cols, CV_8UC1, values)
}

#: A three-pixel band along `y = x`, from column `first` to column `last`.
fn diagonal_band(rows: int, cols: int, first: int, last: int) -> Mat {
    var values: list[float] = []
    for y in 0..rows {
        for x in 0..cols {
            let on = x >= first and x <= last and abs(x - y) <= 1
            values.push(if on { 255.0 } else { 0.0 })
        }
    }
    return Mat.from_list(rows, cols, CV_8UC1, values)
}

#: Two three-pixel bars crossing at (45, 45).
fn cross(rows: int, cols: int) -> Mat {
    var values: list[float] = []
    for y in 0..rows {
        for x in 0..cols {
            let horizontal = y >= 44 and y <= 46 and x >= 10 and x <= 80
            let vertical = x >= 44 and x <= 46 and y >= 10 and y <= 80
            values.push(if horizontal or vertical { 255.0 } else { 0.0 })
        }
    }
    return Mat.from_list(rows, cols, CV_8UC1, values)
}

#: Uniform noise from a congruential generator, so the same image can be
#: rebuilt anywhere -- including in Python, where OpenCV saw it.
fn noise(rows: int, cols: int, seed: int) -> Mat {
    var state = seed
    var values: list[float] = []
    for _index in 0..rows * cols {
        state = (state * 1103515245 + 12345) % 2147483648
        values.push(float(state // 65536 % 256))
    }
    return Mat.from_list(rows, cols, CV_8UC1, values)
}

#: The image turned a quarter turn clockwise, which sends `(x, y)` to
#: `(rows - 1 - y, x)`.
fn quarter_turn(image: Mat) -> Mat {
    let values = image.values()
    var out: list[float] = []
    for y in 0..image.cols {
        for x in 0..image.rows { out.push(values[(image.rows - 1 - x) * image.cols + y]) }
    }
    return Mat.from_list(image.cols, image.rows, CV_8UC1, out)
}

#: How far the worse of a segment's two ends is from `want`, taking the segment
#: either way round: LSD reports an end order that follows the gradient, which
#: is not something a caller should have to know.
fn endpoint_error(segment: list[float], want: list[float]) -> float {
    var forward = 0.0
    var backward = 0.0
    for index in 0..4 {
        forward = max(forward, math.fabs(segment[index] - want[index]))
        backward = max(backward, math.fabs(segment[index] - want[(index + 2) % 4]))
    }
    return min(forward, backward)
}

#: The closest segment in `lines` to `want`, by that measure.
fn closest(lines: list[list[float]], want: list[float]) -> float {
    var best = 1e30
    for segment in lines { best = min(best, endpoint_error(segment, want)) }
    return best
}

fn length_of(segment: list[float]) -> float {
    let dx = segment[2] - segment[0]
    let dy = segment[3] - segment[1]
    return math.sqrt(dx * dx + dy * dy)
}

#: The segment's direction in degrees, folded into `[0, 180)` because a segment
#: has no head or tail.
fn heading_of(segment: list[float]) -> float {
    var degrees = math.degrees(math.atan2(segment[3] - segment[1], segment[2] - segment[0]))
    while degrees < 0.0 { degrees += 180.0 }
    while degrees >= 180.0 { degrees -= 180.0 }
    return degrees
}

#: The four sides of the rectangle `filled_rectangle(80, 80, 15, 20, 60, 55)`
#: draws, as corner-to-corner segments. The edge a detector can see runs
#: between the last background pixel and the first foreground one, which is
#: half a pixel outside the filled block.
let RECTANGLE_SIDES = [
    [14.5, 19.5, 14.5, 55.5],
    [60.5, 19.5, 60.5, 55.5],
    [14.5, 19.5, 60.5, 19.5],
    [14.5, 55.5, 60.5, 55.5],
]

fn test_a_rectangle_outline_is_four_segments() -> void {
    let detector = create_line_segment_detector()
    let (lines, widths, precisions, nfas) = detector.detect(
        filled_rectangle(80, 80, 15, 20, 60, 55)
    )
    assert_eq(lines.len(), 4, f"a rectangle outline is four segments, got {lines.len()}")
    for side in RECTANGLE_SIDES {
        #: A pixel and a half: the ends land within a sixth of a pixel of the
        #: edge across it, and about a pixel short of the corner along it,
        #: because the corner pixel's gradient points diagonally and belongs
        #: to neither side.
        assert_true(closest(lines, side) < 1.5, f"no segment within 1.5px of {side}")
    }
    assert_eq(widths.len(), 4)
    assert_eq(precisions.len(), 4)
    assert_eq(nfas.len(), 4)
    for index in 0..4 {
        assert_near(precisions[index], 0.125, tolerance: 1e-12)
        assert_true(
            widths[index] > 1.0 and widths[index] < 5.0,
            f"a step edge should be a few pixels wide, got {widths[index]}"
        )
        assert_true(
            nfas[index] > 0.0,
            f"an accepted segment has -log10(NFA) > 0, got {nfas[index]}"
        )
    }
}

fn test_a_diagonal_is_the_two_edges_of_the_band() -> void {
    let detector = create_line_segment_detector()
    let (lines, _widths, _precisions, _nfas) = detector.detect(diagonal_band(80, 80, 10, 70))
    assert_eq(lines.len(), 2, f"a band has two edges, got {lines.len()}")
    #: The band covers `|x - y| <= 1` between columns 10 and 70, so its two
    #: edges are `y = x + 1.5` and `y = x - 1.5`, each running the length of
    #: the band.
    let above = [9.5, 11.0, 70.5, 72.0]
    let below = [11.0, 9.5, 72.0, 70.5]
    assert_true(closest(lines, above) < 2.0, "no segment along the upper edge of the band")
    assert_true(closest(lines, below) < 2.0, "no segment along the lower edge of the band")
    for segment in lines {
        assert_near(heading_of(segment), 45.0, tolerance: 1.0)
        assert_near(length_of(segment), 61.0 * math.SQRT_2, tolerance: 3.0)
    }
}

fn test_a_cross_is_eight_segments() -> void {
    let detector = create_line_segment_detector()
    let (lines, _widths, _precisions, _nfas) = detector.detect(cross(90, 90))
    #: Four edges, each cut in two where the other bar crosses it.
    assert_eq(lines.len(), 8, f"a cross of two bars is eight segments, got {lines.len()}")
    var horizontal = 0
    var vertical = 0
    for segment in lines {
        let heading = heading_of(segment)
        if heading < 1.0 or heading > 179.0 { horizontal += 1 }
        if math.fabs(heading - 90.0) < 1.0 { vertical += 1 }
        assert_true(
            length_of(segment) > 25.0,
            f"each half of a bar is over half its length, got {length_of(segment)}"
        )
    }
    assert_eq(horizontal, 4, "four of the eight run along the horizontal bar")
    assert_eq(vertical, 4, "four of the eight run along the vertical bar")
}

fn test_a_blank_image_has_no_segments() -> void {
    let detector = create_line_segment_detector()
    let (lines, _widths, _precisions, _nfas) = detector.detect(Mat.zeros(64, 64, CV_8UC1))
    assert_eq(lines.len(), 0, f"a blank image has no lines, got {lines.len()}")
    let (flat, _w, _p, _n) = detector.detect(Mat.full(64, 64, CV_8UC1, 137.0))
    assert_eq(flat.len(), 0, f"a flat image has no lines either, got {flat.len()}")
}

fn test_noise_produces_almost_nothing() -> void {
    #: This is the a-contrario test doing its job: region growing finds plenty
    #: of aligned runs in noise, and the NFA rejects them because runs that
    #: long are exactly what noise is expected to produce.
    let detector = create_line_segment_detector()
    var found = 0
    for seed in [12345, 999, 2024, 7] {
        let (lines, _widths, _precisions, _nfas) = detector.detect(noise(65, 65, seed))
        found += lines.len()
    }
    assert_true(found <= 2, f"four noise images gave {found} segments between them")
}

fn test_rotating_the_image_rotates_the_segments() -> void {
    let upright = filled_rectangle(80, 80, 15, 20, 60, 55)
    let turned_image = quarter_turn(upright)

    #: At full size the turn is exact. The 2x2 window sits between four
    #: pixels, and the half pixel that puts back survives the turn, so an
    #: endpoint at `(x, y)` has to come back at `(rows - 1 - y, x)` to the last
    #: place the arithmetic keeps.
    let detector = create_line_segment_detector(LSD_REFINE_STD, 1.0)
    let (lines, _w, _p, _n) = detector.detect(upright)
    let (turned, _w2, _p2, _n2) = detector.detect(turned_image)
    assert_eq(turned.len(), lines.len(), "a quarter turn does not change how many there are")
    assert_true(lines.len() > 0, "there is something to turn")
    let edge = float(upright.rows - 1)
    for segment in lines {
        let want = [edge - segment[1], segment[0], edge - segment[3], segment[2]]
        assert_true(
            closest(turned, want) < 0.02,
            f"{segment} did not turn with the image, nearest was {closest(turned, want)}"
        )
    }

    #: At the default scale it holds against the grid the detector actually
    #: worked on, whose last row is 63 and not 79. Writing the turn in the
    #: coordinates of the image as given therefore misses by the quarter pixel
    #: between the two grids, and by nothing else.
    let scaled = create_line_segment_detector()
    let (small, _w3, _p3, _n3) = scaled.detect(upright)
    let (small_turned, _w4, _p4, _n4) = scaled.detect(turned_image)
    assert_eq(small_turned.len(), small.len())
    for segment in small {
        let want = [edge - segment[1], segment[0], edge - segment[3], segment[2]]
        let missed = closest(small_turned, want)
        assert_true(missed > 0.2 and missed < 0.3, f"expected a quarter pixel out, got {missed}")
    }
}

#: What OpenCV 5.0.0 reports for `filled_rectangle(80, 80, 15, 20, 60, 55)`,
#: from `cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV).detect(image)`.
let OPENCV_RECTANGLE = [
    [14.3627, 20.6249, 14.3649, 54.3750],
    [15.6249, 55.4572, 59.3750, 55.4546],
    [59.3750, 19.3683, 15.6250, 19.3666],
    [60.4589, 54.3750, 60.4622, 20.6249],
]

#: And for `cross(90, 90)`, which is the harder case: eight segments that all
#: end where another edge crosses them.
let OPENCV_CROSS = [
    [10.6196, 46.5567, 41.8779, 46.4484],
    [41.8786, 43.3107, 10.6205, 43.2048],
    [43.2048, 10.6205, 43.3107, 41.8786],
    [43.3108, 48.1214, 43.2064, 79.3794],
    [46.4484, 41.8779, 46.5567, 10.6196],
    [46.5548, 79.3803, 46.4489, 48.1222],
    [48.1222, 46.4489, 79.3803, 46.5548],
    [79.3794, 43.2064, 48.1214, 43.3108],
]

fn test_matches_opencv_on_the_rectangle_and_the_cross() -> void {
    let detector = create_line_segment_detector(LSD_REFINE_ADV)
    let (lines, _w, _p, _n) = detector.detect(filled_rectangle(80, 80, 15, 20, 60, 55))
    assert_eq(lines.len(), OPENCV_RECTANGLE.len(), "OpenCV finds four sides here too")
    for side in OPENCV_RECTANGLE {
        #: A fiftieth of a pixel. What is left is OpenCV rounding the region's
        #: principal angle through its own `fastAtan2`, which works in degrees
        #: and in single precision.
        assert_true(closest(lines, side) < 0.05, f"OpenCV's {side} was not matched")
    }
    let (crossed, _w2, _p2, _n2) = detector.detect(cross(90, 90))
    assert_eq(crossed.len(), OPENCV_CROSS.len(), "OpenCV finds eight pieces here too")
    for side in OPENCV_CROSS {
        assert_true(closest(crossed, side) < 0.05, f"OpenCV's {side} was not matched")
    }
}

fn test_every_refinement_setting_finds_the_same_clean_lines() -> void {
    let image = filled_rectangle(80, 80, 15, 20, 60, 55)
    let (plain, _w, _p, _n) = create_line_segment_detector(LSD_REFINE_NONE).detect(image)
    let (standard, _w2, _p2, _n2) = create_line_segment_detector(LSD_REFINE_STD).detect(image)
    let (advanced, _w3, _p3, _n3) = create_line_segment_detector(LSD_REFINE_ADV).detect(image)
    assert_eq(plain.len(), 4)
    assert_eq(standard.len(), 4)
    assert_eq(advanced.len(), 4)
    for segment in plain {
        assert_true(closest(standard, segment) < 0.01, "refinement moved a clean edge")
        assert_true(closest(advanced, segment) < 0.01, "refinement moved a clean edge")
    }
}

fn test_working_at_full_size_finds_the_edges_at_the_half_pixel() -> void {
    #: With no downscaling the edges of a step are exactly halfway between two
    #: pixels, and nothing in the detector can blur that.
    let detector = create_line_segment_detector(LSD_REFINE_STD, 1.0)
    let (lines, _w, _p, _n) = detector.detect(filled_rectangle(80, 80, 15, 20, 60, 55))
    assert_eq(lines.len(), 4, f"four sides at full size, got {lines.len()}")
    for segment in lines {
        for index in 0..4 {
            assert_near(segment[index] - math.floor(segment[index]), 0.5, tolerance: 1e-9)
        }
    }
}

fn test_draw_segments_draws() -> void {
    let detector = create_line_segment_detector()
    let image = filled_rectangle(80, 80, 15, 20, 60, 55)
    let (lines, _w, _p, _n) = detector.detect(image)
    let drawn = detector.draw_segments(image, lines)
    assert_eq(drawn.channels(), 3, "the drawing is in colour so the lines can be seen")
    assert_eq(drawn.rows, image.rows)
    assert_eq(drawn.cols, image.cols)

    let painted = drawn.values()
    let grey = image.values()
    var red = 0
    var untouched = 0
    for index in 0..80 * 80 {
        let blue = painted[index * 3]
        let green = painted[index * 3 + 1]
        let scarlet = painted[index * 3 + 2]
        if blue == 0.0 and green == 0.0 and scarlet == 255.0 {
            red += 1
        } elif blue == grey[index] and green == grey[index] and scarlet == grey[index] {
            untouched += 1
        }
    }
    #: The four sides come to about 160 pixels of ink; what matters is that it
    #: is the sides that got drawn and not the whole image.
    assert_true(red > 120, f"only {red} pixels came out red")
    assert_true(untouched > 5000, f"only {untouched} pixels were left alone")

    let (blank, _w2, _p2, _n2) = create_line_segment_detector().detect(Mat.zeros(80, 80, CV_8UC1))
    let empty = detector.draw_segments(image, blank)
    let plain = empty.values()
    for index in 0..80 * 80 {
        assert_eq(plain[index * 3], grey[index])
        assert_eq(plain[index * 3 + 2], grey[index])
    }
}

fn test_compare_segments_counts_where_two_sets_disagree() -> void {
    let detector = create_line_segment_detector()
    let (lines, _w, _p, _n) = detector.detect(filled_rectangle(80, 80, 15, 20, 60, 55))
    let size = Size(80, 80)

    let (same, _canvas) = detector.compare_segments(size, lines, lines, Mat(0, 0, CV_8UC3))
    assert_eq(same, 0, f"a set compared with itself disagrees nowhere, got {same}")

    let (all, canvas) = detector.compare_segments(size, lines, [], Mat(0, 0, CV_8UC3))
    assert_true(all > 120, f"against nothing, every drawn pixel disagrees, got {all}")
    assert_eq(canvas.rows, 80)
    assert_eq(canvas.channels(), 3)
    let painted = canvas.values()
    var blue = 0
    for index in 0..80 * 80 {
        if painted[index * 3] == 255.0 and painted[index * 3 + 2] == 0.0 { blue += 1 }
    }
    assert_eq(blue, all, "the first set is the blue one")

    #: One segment in common and one in each set alone: the shared one cancels.
    let shared = [10.0, 5.0, 70.0, 5.0]
    let only_first = [10.0, 20.0, 70.0, 20.0]
    let only_second = [10.0, 40.0, 70.0, 40.0]
    let (differing, mixed) = detector.compare_segments(
        size,
        [shared, only_first],
        [shared, only_second],
        Mat(0, 0, CV_8UC3)
    )
    assert_eq(differing, 122, f"two 61-pixel rows differ, got {differing}")
    let colours = mixed.values()
    let common = (5 * 80 + 40) * 3
    assert_eq(colours[common], 255.0, "a pixel both sets cover is blue")
    assert_eq(colours[common + 2], 255.0, "and red, which is to say magenta")
}

fn test_the_parameters_are_checked() -> void {
    assert_throws(ValueError, || create_line_segment_detector(7))
    assert_throws(ValueError, || create_line_segment_detector(LSD_REFINE_STD, 0.0))
    assert_throws(ValueError, || create_line_segment_detector(LSD_REFINE_STD, 0.8, 0.0))
    assert_throws(ValueError, || LineSegmentDetector(ang_th: 0.0))
    assert_throws(ValueError, || LineSegmentDetector(ang_th: 180.0))
    assert_throws(ValueError, || LineSegmentDetector(quant: -1.0))
    assert_throws(ValueError, || LineSegmentDetector(density_th: 1.0))
    assert_throws(ValueError, || LineSegmentDetector(n_bins: 0))
    let detector = create_line_segment_detector()
    assert_throws(ValueError, || detector.detect(Mat.zeros(20, 20, CV_8UC3)))
    assert_throws(ValueError, || detector.draw_segments(Mat.zeros(20, 20, CV_8UC1), [[1.0, 2.0]]))
}

fn test_a_degenerate_image_comes_back_empty() -> void {
    let detector = create_line_segment_detector()
    let (thin, _w, _p, _n) = detector.detect(Mat.zeros(1, 40, CV_8UC1))
    assert_eq(thin.len(), 0, "an image one pixel tall has no 2x2 window to differentiate")
    let (small, _w2, _p2, _n2) = detector.detect(Mat.full(3, 3, CV_8UC1, 255.0))
    assert_eq(small.len(), 0, "and a 3x3 downscales to nothing worth looking at")
}

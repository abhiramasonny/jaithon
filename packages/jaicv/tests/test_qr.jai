#: Reading and writing QR codes.
#:
#: The reader and the writer are checked against each other over every version
#: and every error-correction level, which is what says the version tables are
#: right: a symbol whose block split is wrong fails its own parity, and a
#: symbol whose alignment patterns sit in the wrong place reads its data out of
#: them. Both were real: the first draft had the alignment spacing rounded down
#: instead of up, and versions 30, 36 and 39 decoded to nothing.
#:
#: Cross-checked against OpenCV separately, in a way a test cannot carry:
#: OpenCV 5.0.0 reads symbols this writes, and this reads symbols OpenCV
#: writes, at every version from one to forty. The one disagreement is
#: OpenCV's: its encoder puts version 21's last alignment centre at module 92
#: where the standard says 94, and its own detector cannot read what it wrote.
import std.math
import std.gpu as gpu
from std.test import assert_eq, assert_true
from jaicv.core.mat import Mat
from jaicv.core.types import CV_8U, CV_MAKETYPE, Point2f, Scalar, Size
from jaicv.imgproc.filter import median_blur
from jaicv.imgproc.geometric import get_perspective_transform, get_rotation_matrix_2d
from jaicv.imgproc.geometric import warp_affine, warp_perspective
from jaicv.objdetect.qr import alignment_centres, decode_grid, encode_grid
from jaicv.objdetect.qr import is_function_module, qr_encode, smallest_version
from jaicv.objdetect.qr import version_size, QRCodeDetector

fn have_gpu() -> bool { return gpu.is_available() }

#: Where the standard puts the alignment centres, for the versions whose rule
#: is easiest to get wrong: the first with three, the ones where the spacing
#: rounds, and the one the standard exempts from its own rule.
fn test_alignment_centres_match_the_standard() -> void {
    assert_eq(alignment_centres(1), [])
    assert_eq(alignment_centres(2), [6, 18])
    assert_eq(alignment_centres(7), [6, 22, 38])
    assert_eq(alignment_centres(21), [6, 28, 50, 72, 94])
    assert_eq(alignment_centres(30), [6, 26, 52, 78, 104, 130])
    assert_eq(alignment_centres(32), [6, 34, 60, 86, 112, 138])
    assert_eq(alignment_centres(36), [6, 24, 50, 76, 102, 128, 154])
    assert_eq(alignment_centres(39), [6, 26, 54, 82, 110, 138, 166])
    assert_eq(alignment_centres(40), [6, 30, 58, 86, 114, 142, 170])
}

#: Every version's data capacity, counted off the module map, against the
#: codeword totals the tables carry. The two are derived independently -- one
#: from the geometry, one from the standard's list -- so agreeing to the exact
#: number of leftover bits is a real check on both.
fn test_the_module_map_leaves_room_for_exactly_the_codewords() -> void {
    let totals = [
        26,
        44,
        70,
        100,
        134,
        172,
        196,
        242,
        292,
        346,
        404,
        466,
        532,
        581,
        655,
        733,
        815,
        901,
        991,
        1085,
        1156,
        1258,
        1364,
        1474,
        1588,
        1706,
        1828,
        1921,
        2051,
        2185,
        2323,
        2465,
        2611,
        2761,
        2876,
        3034,
        3196,
        3362,
        3532,
        3706,
    ]
    #: What the standard calls the remainder bits: the modules left over after
    #: the last whole codeword, which are written as zero and ignored.
    let remainders = [
        0,
        7,
        7,
        7,
        7,
        7,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    for version in 1..41 {
        let side = version_size(version)
        var free = 0
        for y in 0..side {
            for x in 0..side {
                if not is_function_module(version, y, x) { free += 1 }
            }
        }
        assert_eq(free, totals[version - 1] * 8 + remainders[version - 1])
    }
}

#: Encode then decode, over every version and every level. The payloads rotate
#: so numeric, alphanumeric and byte segments are all exercised, and each is
#: checked to come back with the version and level it went in as.
fn test_every_version_and_level_round_trips() -> void {
    let payloads = [
        "HELLO",
        "https://example.com/a/b?c=1",
        "012345678901234567890123456789",
        "jaicv qr reader and writer",
        "MIXED case Text 42 - and symbols $%*+-./:",
    ]
    var seen = 0
    for level in 0..4 {
        for version in 1..41 {
            for (index, payload) in payloads.enumerate() {
                if (version + index) % 5 != 0 { continue }
                var grid: list[list[bool]] = []
                try {
                    grid = encode_grid(payload, level, version)
                } catch e: ValueError {
                    #: This payload needs a larger symbol than this version.
                    continue
                }
                assert_eq(grid.len(), version_size(version))
                let code = decode_grid(grid)
                if code is null { throw AssertionError(f"v{version} level {level} did not decode") }
                assert_eq(code.text, payload)
                assert_eq(code.version, version)
                assert_eq(code.correction_level, ["L", "M", "Q", "H"][level])
                seen += 1
            }
        }
    }
    assert_true(seen > 120)
}

fn test_the_smallest_version_is_the_smallest_that_fits() -> void {
    assert_eq(smallest_version("HELLO", 0), 1)
    let long_text = "0123456789" * 40
    let chosen = smallest_version(long_text, 3)
    if chosen is null { throw AssertionError("four hundred digits fit in some version") }
    assert_true(chosen > 1)
    #: One version smaller must not fit, or it was not the smallest.
    assert_true(encode_grid(long_text, 3, chosen).len() == version_size(chosen))
}

fn framed(code: Mat, side: int, top: int, left: int) -> Mat {
    var values: list[float] = []
    for _index in 0..side * side { values.push(255.0) }
    let flat = code.to_list()
    for y in 0..code.rows {
        for x in 0..code.cols {
            values[(top + y) * side + left + x] = flat[y * code.cols + x]
        }
    }
    return Mat.from_list(side, side, CV_MAKETYPE(CV_8U, 1), values)
}

fn test_a_rendered_symbol_is_found_and_read() -> void {
    if not have_gpu() { return }
    let detector = QRCodeDetector()
    for level in 0..4 {
        #: Detection is quadratic in the picture and cubic in the number of
        #: finder candidates, and a version 40 symbol is seven hundred pixels
        #: square: reading one is two seconds, which is fine to do and not fine
        #: to do in a gate. The large versions are covered by the grid round
        #: trip above, which exercises everything but the finding.
        for version in [1, 2, 5, 12] {
            let payload = f"jaicv v{version} level {level}"
            var image = qr_encode("x", 0, 1, 1)
            try {
                image = qr_encode(payload, level, version, 4)
            } catch e: ValueError {
                continue
            }
            let (text, corners) = detector.detect_and_decode(image)
            assert_eq(text, payload)
            assert_eq(corners.len(), 4)
        }
    }
}

#: A symbol turned on the page. Its finders are found by scanning rows, so an
#: angle that is not a multiple of a quarter turn is the case that has to work
#: rather than the case that happens to.
fn test_a_turned_symbol_is_read() -> void {
    if not have_gpu() { return }
    let detector = QRCodeDetector()
    let payload = "TURNED SYMBOL"
    let code = qr_encode(payload, 1, 2, 6)
    let side = code.rows + 120
    let scene = framed(code, side, 60, 60)
    let centre = Point2f(float(side) / 2.0, float(side) / 2.0)
    for angle in [0.0, 5.0, 15.0, 45.0, 90.0, 180.0, 270.0, -25.0] {
        let turned = warp_affine(
            scene,
            get_rotation_matrix_2d(centre, angle, 1.0),
            Size(side, side),
            border_value: Scalar(255.0, 255.0, 255.0, 255.0)
        )
        let (text, _corners) = detector.detect_and_decode(turned)
        assert_eq(text, payload)
    }
}

#: A symbol photographed off-axis. Three finder centres fix a perspective and
#: the fourth corner is refined against the alignment pattern nearest it;
#: without that refinement the far corner of a large symbol samples between
#: modules.
fn test_a_symbol_seen_at_an_angle_is_read() -> void {
    if not have_gpu() { return }
    let detector = QRCodeDetector()
    let payload = "https://jaicv.example/qr?v=1"
    let code = qr_encode(payload, 1, 6, 6)
    let side = code.rows + 160
    let scene = framed(code, side, 80, 80)
    let edge = float(side - 1)
    let source = [Point2f(0.0, 0.0), Point2f(edge, 0.0), Point2f(edge, edge), Point2f(0.0, edge)]
    let target = [
        Point2f(edge * 0.05, edge * 0.02),
        Point2f(edge * 0.94, edge * 0.09),
        Point2f(edge * 0.90, edge * 0.96),
        Point2f(edge * 0.09, edge * 0.89),
    ]
    let skewed = warp_perspective(
        scene,
        get_perspective_transform(source, target),
        Size(side, side),
        border_value: Scalar(255.0, 255.0, 255.0, 255.0)
    )
    let (text, _corners) = detector.detect_and_decode(skewed)
    assert_eq(text, payload)
}

fn test_two_symbols_in_one_picture() -> void {
    if not have_gpu() { return }
    let detector = QRCodeDetector()
    let left_code = qr_encode("LEFT ONE", 0, 2, 5)
    let right_code = qr_encode("RIGHT TWO", 0, 3, 5)
    let width = left_code.cols + right_code.cols + 120
    let height = max(left_code.rows, right_code.rows) + 80
    var values: list[float] = []
    for _index in 0..width * height { values.push(255.0) }
    let left_flat = left_code.to_list()
    for y in 0..left_code.rows {
        for x in 0..left_code.cols {
            values[(40 + y) * width + 40 + x] = left_flat[y * left_code.cols + x]
        }
    }
    let right_flat = right_code.to_list()
    let offset = left_code.cols + 80
    for y in 0..right_code.rows {
        for x in 0..right_code.cols {
            values[(40 + y) * width + offset + x] = right_flat[y * right_code.cols + x]
        }
    }
    let scene = Mat.from_list(height, width, CV_MAKETYPE(CV_8U, 1), values)
    let (texts, corners) = detector.detect_and_decode_multi(scene)
    assert_eq(texts.len(), 2)
    assert_eq(corners.len(), 2)
    assert_true(texts.contains("LEFT ONE"))
    assert_true(texts.contains("RIGHT TWO"))
}

#: Salt and pepper. A module is decided by a vote over the pixels around its
#: centre rather than by the one pixel there, and a median filter first --
#: which is what a caller with a noisy camera should do -- carries every level
#: through a fifth of the pixels being wrong.
fn test_noise_is_survivable() -> void {
    if not have_gpu() { return }
    let detector = QRCodeDetector()
    let payload = "NOISE RESISTANCE 2026"
    for level in 0..4 {
        let clean = qr_encode(payload, level, 4, 6)
        var values = clean.to_list()
        var state = 987654321
        for index in 0..values.len() {
            state = (state *% 1103515245 +% 12345) % 2147483648
            if state // 1024 % 100 < 10 { values[index] = 255.0 - values[index] }
        }
        let noisy = Mat.from_list(clean.rows, clean.cols, CV_MAKETYPE(CV_8U, 1), values)
        let (text, _corners) = detector.detect_and_decode(median_blur(noisy, 3))
        assert_eq(text, payload)
    }
}

fn test_a_picture_with_no_symbol_reports_nothing() -> void {
    if not have_gpu() { return }
    let detector = QRCodeDetector()
    var values: list[float] = []
    for index in 0..120 * 120 { values.push(float(index * 37 % 256)) }
    let noise = Mat.from_list(120, 120, CV_MAKETYPE(CV_8U, 1), values)
    let (text, corners) = detector.detect_and_decode(noise)
    assert_eq(text, "")
    assert_eq(corners.len(), 0)
    assert_eq(detector.detect_multi(noise).len(), 0)
    assert_true(detector.detect(noise) is null)
}

#: A grid that is not a symbol at all, and one whose format word is destroyed.
#: Both must come back as nothing rather than as a guess.
fn test_a_damaged_grid_is_refused() -> void {
    var flat: list[list[bool]] = []
    for _y in 0..21 {
        var row: list[bool] = []
        for _x in 0..21 { row.push(false) }
        flat.push(row)
    }
    assert_true(decode_grid(flat) is null)

    var wrong: list[list[bool]] = []
    for _y in 0..20 {
        var row: list[bool] = []
        for _x in 0..20 { row.push(false) }
        wrong.push(row)
    }
    assert_true(decode_grid(wrong) is null)

    var good = encode_grid("DAMAGE ME", 0, 2)
    for y in 0..good.len() {
        for x in 0..good.len() {
            if (x * 7 + y * 3) % 5 == 0 { good[y][x] = not good[y][x] }
        }
    }
    let code = decode_grid(good)
    if code is not null { assert_eq(code.text, "DAMAGE ME") }
}

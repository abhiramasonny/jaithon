#: Hershey text rendering.
#:
#: These are goldens of jaicv's own output, not comparisons against OpenCV:
#: OpenCV draws from re-derived glyph tables rather than the public-domain
#: ones, so the letterforms differ by design. What is checked against OpenCV
#: is the part porting code depends on -- that a capital is the same number of
#: pixels tall at the same `font_scale` -- and those numbers are written out
#: below rather than computed, so a change to the metrics fails here.
import jaicv as cv
from jaicv import Mat, Point, Scalar
from std.test import assert_eq, assert_true

let FACES = [
    cv.FONT_HERSHEY_SIMPLEX,
    cv.FONT_HERSHEY_PLAIN,
    cv.FONT_HERSHEY_DUPLEX,
    cv.FONT_HERSHEY_COMPLEX,
    cv.FONT_HERSHEY_TRIPLEX,
    cv.FONT_HERSHEY_COMPLEX_SMALL,
    cv.FONT_HERSHEY_SCRIPT_SIMPLEX,
    cv.FONT_HERSHEY_SCRIPT_COMPLEX,
]

#: What OpenCV draws a capital as, in pixels, at `font_scale` 1.0.
let CAP_PIXELS = [20, 11, 20, 20, 20, 16, 18, 18]

fn cap_height(face: int, scale: float) -> int {
    #: `A` has no descender and reaches the cap line, so the ink it leaves is
    #: the cap height outright.
    var image = Mat.zeros(200, 200, cv.CV_8UC1)
    cv.put_text(image, "A", Point(40, 150), face, scale, Scalar(255.0), 1)
    var top = -1
    var bottom = -1
    let values = image.values()
    for y in 0..200 {
        for x in 0..200 {
            if values[y * 200 + x] != 0.0 {
                if top < 0 { top = y }
                bottom = y
            }
        }
    }
    if top < 0 { return 0 }
    return bottom - top + 1
}

fn test_cap_height_matches_opencv() -> void {
    for index in 0..FACES.len() {
        let drawn = cap_height(FACES[index], 1.0)
        #: One pixel of slack: the cap line lands between two rows for some
        #: faces, and which way it rounds is not worth pinning.
        assert_true(
            abs(drawn - CAP_PIXELS[index]) <= 1,
            f"face {FACES[index]} drew a capital {drawn} tall, OpenCV draws {CAP_PIXELS[index]}"
        )
    }
}

fn test_cap_height_scales() -> void {
    let single = cap_height(cv.FONT_HERSHEY_SIMPLEX, 1.0)
    let double = cap_height(cv.FONT_HERSHEY_SIMPLEX, 2.0)
    assert_true(
        abs(double - 2 * single) <= 2,
        f"doubling the scale drew {double} against {single} at one"
    )
}

fn test_text_size_bounds_the_ink() -> void {
    let text = "Agjq 123"
    for face in FACES {
        let (size, baseline) = cv.get_text_size(text, face, 1.0, 1)
        var image = Mat.zeros(200, 400, cv.CV_8UC1)
        let org = Point(20, 120)
        cv.put_text(image, text, org, face, 1.0, Scalar(255.0), 1)
        let values = image.values()
        var left = 400
        var right = -1
        var top = 200
        var bottom = -1
        for y in 0..200 {
            for x in 0..400 {
                if values[y * 400 + x] != 0.0 {
                    if x < left { left = x }
                    if x > right { right = x }
                    if y < top { top = y }
                    if y > bottom { bottom = y }
                }
            }
        }
        assert_true(right >= left, "the text drew nothing")
        #: The reported box runs from `org.y - height + baseline` to
        #: `org.y + baseline`, and the ink has to sit inside it.
        assert_true(
            top >= org.y - size.height + baseline - 1,
            f"face {face} drew above its reported box"
        )
        assert_true(bottom <= org.y + baseline + 1, f"face {face} drew below its reported box")
        assert_true(right <= org.x + size.width + 1, f"face {face} drew past its reported width")
    }
}

fn test_size_grows_with_the_string() -> void {
    let (one, _b1) = cv.get_text_size("M", cv.FONT_HERSHEY_SIMPLEX, 1.0, 1)
    let (two, _b2) = cv.get_text_size("MM", cv.FONT_HERSHEY_SIMPLEX, 1.0, 1)
    let (three, _b3) = cv.get_text_size("MMM", cv.FONT_HERSHEY_SIMPLEX, 1.0, 1)
    assert_eq(
        two.width - one.width,
        three.width - two.width,
        "every copy of a letter should advance the same distance"
    )
    assert_eq(one.height, three.height, "a longer string of capitals is no taller")
}

fn test_empty_string_draws_nothing() -> void {
    var image = Mat.zeros(50, 120, cv.CV_8UC1)
    cv.put_text(image, "", Point(10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255.0), 1)
    assert_eq(int(cv.count_non_zero(image)), 0, "an empty string left ink behind")
    let (size, baseline) = cv.get_text_size("", cv.FONT_HERSHEY_SIMPLEX, 1.0, 1)
    assert_eq(size.width, 0, "an empty string is no wider than nothing")
    assert_eq(baseline, 0, "an empty string has no descender")
}

fn test_descender_only_for_glyphs_that_have_one() -> void {
    let (_size, capitals) = cv.get_text_size("ABC", cv.FONT_HERSHEY_SIMPLEX, 1.0, 1)
    let (_size2, descends) = cv.get_text_size("gjq", cv.FONT_HERSHEY_SIMPLEX, 1.0, 1)
    assert_eq(capitals, 0, "capitals do not descend below the baseline")
    assert_true(descends > 0, "a `g` descends below the baseline")
}

fn test_italic_leans_right_and_is_wider() -> void {
    let (upright, _b) = cv.get_text_size("HHH", cv.FONT_HERSHEY_SIMPLEX, 1.0, 1)
    let (slanted, _b2) = cv.get_text_size("HHH", cv.FONT_HERSHEY_SIMPLEX | cv.FONT_ITALIC, 1.0, 1)
    assert_true(slanted.width > upright.width, "a slanted face is wider than an upright one")
    assert_eq(slanted.height, upright.height, "slanting does not change the height")

    #: The top of a slanted letter sits to the right of its foot.
    var image = Mat.zeros(80, 120, cv.CV_8UC1)
    cv.put_text(
        image,
        "H",
        Point(20, 60),
        cv.FONT_HERSHEY_SIMPLEX | cv.FONT_ITALIC,
        1.5,
        Scalar(255.0),
        1
    )
    #: The left edge of the topmost inked row against the left edge of the
    #: bottom one: on a slanted face the top of the stem stands to the right
    #: of its foot.
    let values = image.values()
    var top_left = -1
    var bottom_left = -1
    for y in 0..80 {
        var row_left = -1
        for x in 0..120 {
            if values[y * 120 + x] != 0.0 and row_left < 0 { row_left = x }
        }
        if row_left >= 0 {
            if top_left < 0 { top_left = row_left }
            bottom_left = row_left
        }
    }
    assert_true(
        top_left > bottom_left,
        f"the top ({top_left}) should lean right of the foot ({bottom_left})"
    )
}

fn test_bottom_left_origin_flips_the_text() -> void {
    var upright = Mat.zeros(120, 200, cv.CV_8UC1)
    var flipped = Mat.zeros(120, 200, cv.CV_8UC1)
    cv.put_text(upright, "Ap", Point(20, 60), cv.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255.0), 1)
    cv.put_text(
        flipped,
        "Ap",
        Point(20, 60),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        Scalar(255.0),
        1,
        cv.LINE_8,
        true
    )
    let a = upright.values()
    let b = flipped.values()
    var a_top = 120
    var b_bottom = -1
    for y in 0..120 {
        for x in 0..200 {
            if a[y * 200 + x] != 0.0 and y < a_top { a_top = y }
            if b[y * 200 + x] != 0.0 and y > b_bottom { b_bottom = y }
        }
    }
    assert_true(a_top < 60, "upright text sits above its origin")
    assert_true(b_bottom > 60, "bottom-left-origin text sits below its origin")
}

fn test_every_printable_character_draws() -> void {
    #: A blank glyph for a character the font defines would mean the table was
    #: decoded short. Space is the only one that legitimately draws nothing.
    for code in 33..127 {
        var image = Mat.zeros(120, 120, cv.CV_8UC1)
        cv.put_text(image, chr(code), Point(30, 80), cv.FONT_HERSHEY_SIMPLEX, 2.0, Scalar(255.0), 1)
        assert_true(
            int(cv.count_non_zero(image)) > 0,
            f"character {code} ({chr(code)}) drew nothing"
        )
    }
    var blank = Mat.zeros(120, 120, cv.CV_8UC1)
    cv.put_text(blank, " ", Point(30, 80), cv.FONT_HERSHEY_SIMPLEX, 2.0, Scalar(255.0), 1)
    assert_eq(int(cv.count_non_zero(blank)), 0, "a space drew ink")
}

fn test_thickness_adds_ink() -> void {
    var thin = Mat.zeros(80, 200, cv.CV_8UC1)
    var thick = Mat.zeros(80, 200, cv.CV_8UC1)
    cv.put_text(thin, "WWW", Point(10, 55), cv.FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255.0), 1)
    cv.put_text(thick, "WWW", Point(10, 55), cv.FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255.0), 3)
    assert_true(
        int(cv.count_non_zero(thick)) > int(cv.count_non_zero(thin)),
        "a thicker stroke should lay down more ink"
    )
}

fn test_rejects_a_scale_or_thickness_of_zero() -> void {
    var image = Mat.zeros(40, 80, cv.CV_8UC1)
    var threw = 0
    try {
        cv.put_text(image, "x", Point(4, 30), cv.FONT_HERSHEY_SIMPLEX, 0.0, Scalar(255.0), 1)
    } catch _e {
        threw += 1
    }
    try {
        cv.put_text(image, "x", Point(4, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255.0), 0)
    } catch _e {
        threw += 1
    }
    assert_eq(threw, 2, "a scale of zero and a thickness of zero both have to be refused")
}

fn test_colour_reaches_every_channel() -> void {
    var image = Mat.zeros(80, 200, cv.CV_8UC3)
    cv.put_text(
        image,
        "RGB",
        Point(10, 55),
        cv.FONT_HERSHEY_SIMPLEX,
        1.2,
        Scalar(20.0, 130.0, 240.0),
        2
    )
    let values = image.values()
    var found = false
    for index in 0..80 * 200 {
        let blue = values[index * 3]
        let green = values[index * 3 + 1]
        let red = values[index * 3 + 2]
        if blue == 20.0 and green == 130.0 and red == 240.0 { found = true }
    }
    assert_true(found, "no pixel came out the colour that was asked for")
}

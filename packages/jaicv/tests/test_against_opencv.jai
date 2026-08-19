#: Differential test: every case OpenCV recorded, replayed through jaicv.
#:
#: `tests/oracle/generate.py` runs the reference implementation and writes each
#: input and its output to `tests/oracle/cases.txt`. This reads that file, runs
#: the same call here, and fails on any element that differs by more than the
#: tolerance the operation earns — exact for integer results that are pure
#: arithmetic, a unit or two where OpenCV rounds a fixed-point intermediate.
#:
#: The file is checked in, so the test runs without OpenCV installed. Regenerate
#: it with `python packages/jaicv/tests/oracle/generate.py` when adding cases.
import std.math
from std.io import Path
from std.test import assert_true
from jaicv.core.mat import Mat
from jaicv.core.types import CV_MAT_CN, CV_MAT_DEPTH, CV_MAKETYPE, type_to_string
from jaicv.imgproc.color import cvt_color
from jaicv.imgproc.color import COLOR_BGR2BGRA, COLOR_BGR2GRAY, COLOR_BGR2HLS, COLOR_BGR2HSV
from jaicv.imgproc.color import COLOR_BGR2Lab, COLOR_BGR2RGB, COLOR_BGR2XYZ, COLOR_BGR2YCrCb
from jaicv.imgproc.color import COLOR_BGR2YUV, COLOR_BGRA2BGR, COLOR_GRAY2BGR, COLOR_HLS2BGR
from jaicv.imgproc.color import COLOR_HSV2BGR, COLOR_Lab2BGR, COLOR_RGB2GRAY, COLOR_XYZ2BGR
from jaicv.imgproc.color import COLOR_YCrCb2BGR, COLOR_YUV2BGR
from jaicv.imgproc.geometric import copy_make_border, get_affine_transform
from jaicv.imgproc.geometric import get_perspective_transform, get_rotation_matrix_2d
from jaicv.imgproc.geometric import pyr_down, pyr_up, remap, resize, warp_affine, warp_perspective
from jaicv.core.types import BORDER_CONSTANT, BORDER_REFLECT, BORDER_REFLECT_101
from jaicv.core.types import BORDER_REPLICATE, BORDER_WRAP
from jaicv.core.types import INTER_AREA, INTER_CUBIC, INTER_LANCZOS4, INTER_LINEAR, INTER_NEAREST
from jaicv.core.types import CV_32F, Point, Point2f, Rect, Scalar, Size
from jaicv.imgproc.filter import blur, box_filter, filter_2d, gaussian_blur
from jaicv.imgproc.filter import get_gaussian_kernel, laplacian, median_blur, scharr, sobel
from jaicv.imgproc.threshold import adaptive_threshold, threshold
from jaicv.imgproc.threshold import ADAPTIVE_THRESH_GAUSSIAN_C, ADAPTIVE_THRESH_MEAN_C
from jaicv.imgproc.threshold import THRESH_BINARY, THRESH_BINARY_INV, THRESH_OTSU
from jaicv.imgproc.threshold import THRESH_TOZERO, THRESH_TOZERO_INV, THRESH_TRUNC
from jaicv.imgproc.morph import dilate, erode, get_structuring_element, morphology_ex
from jaicv.imgproc.morph import MORPH_BLACKHAT, MORPH_CLOSE, MORPH_CROSS, MORPH_ELLIPSE
from jaicv.imgproc.morph import MORPH_GRADIENT, MORPH_OPEN, MORPH_RECT, MORPH_TOPHAT
from jaicv.imgproc.edges import canny
from jaicv.imgproc.drawing import arrowed_line, circle, draw_marker, ellipse, fill_convex_poly
from jaicv.imgproc.drawing import fill_poly, line, polylines, rectangle
from jaicv.imgproc.contours import find_contours, CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE
from jaicv.imgproc.contours import RETR_CCOMP, RETR_EXTERNAL, RETR_LIST, RETR_TREE
from jaicv.imgproc.shape import approx_poly_dp, arc_length, bounding_rect, contour_area
from jaicv.imgproc.shape import convex_hull, fit_ellipse, fit_line, hu_moments
from jaicv.imgproc.shape import is_contour_convex, min_area_rect, min_enclosing_circle
from jaicv.imgproc.shape import moments, point_polygon_test
from jaicv.imgproc.segmentation import connected_components, connected_components_with_stats
from jaicv.imgproc.segmentation import distance_transform, flood_fill, integral
from jaicv.imgproc.segmentation import DIST_C, DIST_L1, DIST_L2
from jaicv.imgproc.hist import calc_back_project, calc_hist, compare_hist, equalize_hist, CLAHE
from jaicv.imgproc.hist import HISTCMP_BHATTACHARYYA, HISTCMP_CHISQR, HISTCMP_CHISQR_ALT
from jaicv.imgproc.hist import HISTCMP_CORREL, HISTCMP_INTERSECT, HISTCMP_KL_DIV
from jaicv.imgproc.features import corner_harris, corner_min_eigen_val, hough_lines
from jaicv.imgproc.features import match_template
from jaicv.imgproc.features import TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED
from jaicv.imgproc.features import TM_SQDIFF, TM_SQDIFF_NORMED
from jaicv.features2d.detect import fast
from jaicv.objdetect.cascade import CascadeClassifier
import std.gpu as gpu

#: One recorded call: the operation, its arguments, the input, and what OpenCV
#: produced.
class Case {
    pub let name: str
    pub let args: list[str]
    pub let source: Mat
    pub let expected: Mat

    pub fn init(self, name: str, args: list[str], source: Mat, expected: Mat) {
        self.name = name
        self.args = args
        self.source = source
        self.expected = expected
    }
}

fn parse_mat(fields: list[str]) -> Mat {
    let rows = int(fields[1])
    let cols = int(fields[2])
    let type_code = int(fields[3])
    var values: list[float] = []
    #: A case with nothing in it writes a trailing space, so the split leaves
    #: an empty field behind; that is the only empty field there ever is.
    for index in 4..fields.len() {
        if fields[index].len() == 0 { continue }
        values.push(float(fields[index]))
    }
    return Mat.from_list(rows, cols, type_code, values)
}

fn load_cases(path: Path) -> list[Case] {
    var cases: list[Case] = []
    var name = ""
    var args: list[str] = []
    var source: Mat? = null
    for line in path.read_text().split("\n") {
        if line.len() == 0 { continue }
        let fields = line.split(" ")
        if fields[0] == "case" {
            name = fields[1]
            args = []
            for index in 2..fields.len() { args.push(fields[index]) }
            source = null
            continue
        }
        if fields[0] == "in" {
            source = parse_mat(fields)
            continue
        }
        if fields[0] == "out" {
            let input = source
            if input is null { throw ValueError("an expected output arrived before its input") }
            cases.push(Case(name, args, input, parse_mat(fields)))
        }
    }
    return cases
}

fn colour_code(name: str) -> int {
    if name == "BGR2GRAY" { return COLOR_BGR2GRAY }
    if name == "RGB2GRAY" { return COLOR_RGB2GRAY }
    if name == "BGR2RGB" { return COLOR_BGR2RGB }
    if name == "BGR2HSV" { return COLOR_BGR2HSV }
    if name == "HSV2BGR" { return COLOR_HSV2BGR }
    if name == "BGR2HLS" { return COLOR_BGR2HLS }
    if name == "HLS2BGR" { return COLOR_HLS2BGR }
    if name == "BGR2Lab" { return COLOR_BGR2Lab }
    if name == "Lab2BGR" { return COLOR_Lab2BGR }
    if name == "BGR2YCrCb" { return COLOR_BGR2YCrCb }
    if name == "YCrCb2BGR" { return COLOR_YCrCb2BGR }
    if name == "BGR2XYZ" { return COLOR_BGR2XYZ }
    if name == "XYZ2BGR" { return COLOR_XYZ2BGR }
    if name == "BGR2YUV" { return COLOR_BGR2YUV }
    if name == "YUV2BGR" { return COLOR_YUV2BGR }
    if name == "GRAY2BGR" { return COLOR_GRAY2BGR }
    if name == "BGR2BGRA" { return COLOR_BGR2BGRA }
    if name == "BGRA2BGR" { return COLOR_BGRA2BGR }
    throw ValueError(f"no jaicv colour code for {name}")
}

#: How far a case is allowed to drift from OpenCV.
#:
#: One unit of an 8-bit result throughout: OpenCV computes most conversions in
#: fixed point and this computes them in float, so the two round the last bit
#: differently. Every recorded case is inside that, including Lab, which goes
#: through a cube root on both sides.
fn tolerance_for(name: str, args: list[str]) -> float {
    #: A transform matrix is float, so it is compared as float.
    if name == "getRotationMatrix2D" or
        name == "getPerspectiveTransform" or
        name == "getAffineTransform" {
        return 1e-4
    }
    if name == "getGaussianKernel" { return 1e-5 }
    if name == "distanceTransform" { return 1e-4 }
    if name == "matchTemplate" or name == "matchTemplateColour" { return 5e-3 }
    if name == "cornerHarris" or name == "cornerMinEigenVal" { return 1e-5 }
    if name == "houghLines" { return 1e-4 }
    if name == "connectedComponentsCentroids" or name == "compareHist" { return 1e-4 }
    if name == "moments" or name == "huMoments" { return 1e-3 }
    if name == "contourArea" or name == "arcLengthClosed" or name == "arcLengthOpen" { return 1e-3 }
    if name == "minAreaRect" or
        name == "minEnclosingCircle" or
        name == "fitEllipse" or
        name == "fitLine" or
        name == "pointPolygonTest" {
        return 1e-3
    }
    #: A mask or a structuring element only ever holds two values, so a
    #: one-unit tolerance would hide a wrong pixel entirely: these are exact.
    if name == "getStructuringElement" or
        name == "threshold" or
        name == "thresholdOtsu" or
        name == "adaptiveThreshold" or
        name == "Canny" or
        name.starts_with("draw") {
        return 0.0
    }
    #: A float derivative is exact arithmetic on both sides; the drift that is
    #: left is float32 association order.
    if name == "Sobel" or name == "Scharr" or name == "Laplacian" { return 1e-3 }
    let _unused = args
    return 1.0
}

fn interpolation_named(name: str) -> int {
    if name == "NEAREST" { return INTER_NEAREST }
    if name == "LINEAR" { return INTER_LINEAR }
    if name == "CUBIC" { return INTER_CUBIC }
    if name == "AREA" { return INTER_AREA }
    if name == "LANCZOS4" { return INTER_LANCZOS4 }
    throw ValueError(f"no jaicv interpolation for {name}")
}

fn border_named(name: str) -> int {
    if name == "CONSTANT" { return BORDER_CONSTANT }
    if name == "REPLICATE" { return BORDER_REPLICATE }
    if name == "REFLECT" { return BORDER_REFLECT }
    if name == "REFLECT_101" { return BORDER_REFLECT_101 }
    if name == "WRAP" { return BORDER_WRAP }
    throw ValueError(f"no jaicv border for {name}")
}

let QUAD_SOURCE = [Point2f(0.0, 0.0), Point2f(9.0, 0.0), Point2f(9.0, 7.0), Point2f(0.0, 7.0)]
let QUAD_TARGET = [Point2f(1.0, 1.0), Point2f(8.0, 0.5), Point2f(9.0, 6.0), Point2f(0.5, 7.0)]

fn rotation_case() -> Mat { return get_rotation_matrix_2d(Point2f(4.5, 3.5), 27.0, 1.3) }

fn shape_named(name: str) -> int {
    if name == "RECT" { return MORPH_RECT }
    if name == "CROSS" { return MORPH_CROSS }
    return MORPH_ELLIPSE
}

fn structuring_case(args: list[str]) -> Mat {
    let size = int(args[1])
    return get_structuring_element(shape_named(args[0]), Size(size, size))
}

fn points_from(args: list[str], at: int, count: int) -> list[Point] {
    var out: list[Point] = []
    for index in 0..count {
        out.push(Point(int(args[at + index * 2]), int(args[at + index * 2 + 1])))
    }
    return out
}

let DRAW_COLOUR = Scalar(10.0, 200.0, 90.0)
let DRAW_GREY = Scalar(200.0, 0.0, 0.0)

fn draw_colour_for(canvas: Mat) -> Scalar {
    return if canvas.channels() == 1 { DRAW_GREY } else { DRAW_COLOUR }
}

fn run_drawing(item: Case) -> Mat {
    let canvas = item.source.clone()
    let colour = draw_colour_for(canvas)
    let args = item.args
    if item.name == "drawLine" {
        return line(
            canvas,
            Point(int(args[0]), int(args[1])),
            Point(int(args[2]), int(args[3])),
            colour,
            int(args[4]),
            int(args[5])
        )
    }
    if item.name == "drawRect" {
        return rectangle(
            canvas,
            Point(int(args[0]), int(args[1])),
            Point(int(args[2]), int(args[3])),
            colour,
            int(args[4]),
            int(args[5])
        )
    }
    if item.name == "drawCircle" {
        return circle(
            canvas,
            Point(int(args[0]), int(args[1])),
            int(args[2]),
            colour,
            int(args[3]),
            int(args[4])
        )
    }
    if item.name == "drawEllipse" {
        return ellipse(
            canvas,
            Point(int(args[0]), int(args[1])),
            Size(int(args[2]), int(args[3])),
            float(args[4]),
            float(args[5]),
            float(args[6]),
            colour,
            int(args[7]),
            int(args[8])
        )
    }
    if item.name == "drawPolylines" {
        let count = int(args[3])
        return polylines(
            canvas,
            [points_from(args, 4, count)],
            int(args[0]) != 0,
            colour,
            int(args[1]),
            int(args[2])
        )
    }
    if item.name == "drawFillPoly" {
        return fill_poly(canvas, [points_from(args, 1, int(args[0]))], colour)
    }
    if item.name == "drawFillConvexPoly" {
        return fill_convex_poly(canvas, points_from(args, 1, int(args[0])), colour)
    }
    if item.name == "drawArrow" {
        return arrowed_line(
            canvas,
            Point(int(args[0]), int(args[1])),
            Point(int(args[2]), int(args[3])),
            colour,
            int(args[4]),
            int(args[5]),
            0,
            float(args[6]) / 100.0
        )
    }
    if item.name == "drawMarker" {
        return draw_marker(
            canvas,
            Point(int(args[0]), int(args[1])),
            colour,
            int(args[2]),
            int(args[3]),
            int(args[4]),
            int(args[5])
        )
    }
    throw ValueError(f"no jaicv binding for {item.name}")
}

fn retrieval_named(name: str) -> int {
    if name == "EXTERNAL" { return RETR_EXTERNAL }
    if name == "LIST" { return RETR_LIST }
    if name == "CCOMP" { return RETR_CCOMP }
    if name == "TREE" { return RETR_TREE }
    throw ValueError(f"no jaicv retrieval mode for {name}")
}

fn chain_named(name: str) -> int {
    if name == "NONE" { return CHAIN_APPROX_NONE }
    if name == "SIMPLE" { return CHAIN_APPROX_SIMPLE }
    throw ValueError(f"no jaicv chain approximation for {name}")
}

fn run_contours(item: Case) -> Mat {
    let mode = retrieval_named(item.args[1])
    let method = chain_named(item.args[2])
    let (contours, hierarchy) = find_contours(item.source, mode, method)
    var values: list[float] = []
    var rows = 0
    if item.name == "findContours" {
        for index in 0..contours.len() {
            let contour = contours[index]
            for at in 0..contour.len() {
                values.push(float(index))
                values.push(float(at))
                values.push(float(contour[at].x))
                values.push(float(contour[at].y))
                rows += 1
            }
        }
    } else {
        for node in hierarchy {
            for field in node.to_list() { values.push(float(field)) }
            rows += 1
        }
    }
    return Mat.from_list(rows, 4, CV_MAKETYPE(4, 1), values)
}

let SHAPE_NAMES = [
    "moments",
    "huMoments",
    "contourArea",
    "arcLengthClosed",
    "arcLengthOpen",
    "boundingRect",
    "isContourConvex",
    "convexHull",
    "approxPolyDP",
    "minAreaRect",
    "minEnclosingCircle",
    "pointPolygonTest",
    "fitEllipse",
    "fitLine",
]

fn contour_of(item: Case) -> list[Point] {
    let (contours, _hierarchy) = find_contours(item.source, RETR_LIST, CHAIN_APPROX_NONE)
    return contours[int(item.args[1])]
}

fn row_of(values: list[float]) -> Mat {
    return Mat.from_list(1, values.len(), CV_MAKETYPE(5, 1), values)
}

fn points_mat(points: list[Point]) -> Mat {
    var values: list[float] = []
    for point in points {
        values.push(float(point.x))
        values.push(float(point.y))
    }
    return Mat.from_list(points.len(), 2, CV_MAKETYPE(4, 1), values)
}

fn run_shape(item: Case) -> Mat {
    let contour = contour_of(item)
    if item.name == "moments" { return row_of(moments(contour).to_list()) }
    if item.name == "huMoments" { return row_of(hu_moments(moments(contour))) }
    if item.name == "contourArea" { return row_of([contour_area(contour)]) }
    if item.name == "arcLengthClosed" { return row_of([arc_length(contour, true)]) }
    if item.name == "arcLengthOpen" { return row_of([arc_length(contour, false)]) }
    if item.name == "boundingRect" {
        let box = bounding_rect(contour)
        return Mat.from_list(
            1,
            4,
            CV_MAKETYPE(4, 1),
            [float(box.x), float(box.y), float(box.width), float(box.height)]
        )
    }
    if item.name == "isContourConvex" {
        let convex = if is_contour_convex(contour) { 1.0 } else { 0.0 }
        return Mat.from_list(1, 1, CV_MAKETYPE(4, 1), [convex])
    }
    if item.name == "convexHull" { return points_mat(convex_hull(contour)) }
    if item.name == "approxPolyDP" {
        return points_mat(approx_poly_dp(contour, float(item.args[2]), true))
    }
    if item.name == "minAreaRect" {
        let box = min_area_rect(contour)
        return row_of([box.center.x, box.center.y, box.size.width, box.size.height, box.angle])
    }
    if item.name == "minEnclosingCircle" {
        let (centre, radius) = min_enclosing_circle(contour)
        return row_of([centre.x, centre.y, radius])
    }
    if item.name == "pointPolygonTest" {
        var values: list[float] = []
        for y in 0..item.source.rows {
            for x in 0..item.source.cols {
                values.push(point_polygon_test(contour, Point2f(float(x), float(y)), true))
            }
        }
        return Mat.from_list(item.source.rows, item.source.cols, CV_MAKETYPE(5, 1), values)
    }
    if item.name == "fitEllipse" {
        let box = fit_ellipse(contour)
        return row_of([box.center.x, box.center.y, box.size.width, box.size.height, box.angle])
    }
    if item.name == "fitLine" { return row_of(fit_line(contour)) }
    throw ValueError(f"no jaicv binding for {item.name}")
}

let SEGMENT_NAMES = [
    "floodFill",
    "floodFillStats",
    "connectedComponents",
    "connectedComponentsStats",
    "connectedComponentsCentroids",
    "distanceTransform",
    "integral",
]

let HIST_NAMES = [
    "calcHist1D",
    "calcHist1DMasked",
    "calcHist2D",
    "equalizeHist",
    "clahe",
    "compareHist",
    "calcBackProject",
]

fn distance_named(name: str) -> int {
    if name == "L1" { return DIST_L1 }
    if name == "L2" { return DIST_L2 }
    if name == "C" { return DIST_C }
    throw ValueError(f"no jaicv distance for {name}")
}

fn run_segmentation(item: Case) -> Mat {
    let args = item.args
    if item.name == "floodFill" or item.name == "floodFillStats" {
        let canvas = item.source.clone()
        let lo = float(args[3])
        let up = float(args[4])
        let (count, box) = flood_fill(
            canvas,
            Point(int(args[1]), int(args[2])),
            Scalar(255.0, 255.0, 255.0),
            Scalar(lo, lo, lo, lo),
            Scalar(up, up, up, up),
            int(args[5])
        )
        if item.name == "floodFill" { return canvas }
        return Mat.from_list(
            1,
            5,
            CV_MAKETYPE(4, 1),
            [float(count), float(box.x), float(box.y), float(box.width), float(box.height)]
        )
    }
    if item.name == "connectedComponents" {
        let (_count, labels) = connected_components(item.source, int(args[1]))
        return labels
    }
    if item.name == "connectedComponentsStats" or item.name == "connectedComponentsCentroids" {
        let (count, _labels, stats, centroids) = connected_components_with_stats(
            item.source,
            int(args[1])
        )
        var values: list[float] = []
        if item.name == "connectedComponentsStats" {
            for entry in stats {
                values.push(float(entry.box.x))
                values.push(float(entry.box.y))
                values.push(float(entry.box.width))
                values.push(float(entry.box.height))
                values.push(float(entry.area))
            }
            return Mat.from_list(count, 5, CV_MAKETYPE(4, 1), values)
        }
        for centre in centroids {
            values.push(centre.x)
            values.push(centre.y)
        }
        return Mat.from_list(count, 2, CV_MAKETYPE(5, 1), values)
    }
    if item.name == "distanceTransform" {
        return distance_transform(item.source, distance_named(args[1]), int(args[2]))
    }
    if item.name == "integral" { return integral(item.source) }
    throw ValueError(f"no jaicv binding for {item.name}")
}

fn histogram_method(name: str) -> int {
    if name == "CORREL" { return HISTCMP_CORREL }
    if name == "CHISQR" { return HISTCMP_CHISQR }
    if name == "INTERSECT" { return HISTCMP_INTERSECT }
    if name == "BHATTACHARYYA" { return HISTCMP_BHATTACHARYYA }
    if name == "CHISQR_ALT" { return HISTCMP_CHISQR_ALT }
    if name == "KL_DIV" { return HISTCMP_KL_DIV }
    throw ValueError(f"no jaicv histogram comparison for {name}")
}

fn run_hist(item: Case) -> Mat {
    let args = item.args
    if item.name == "calcHist1D" {
        return calc_hist([item.source], [0], null, [int(args[1])], [0.0, 256.0])
    }
    if item.name == "calcHist1DMasked" {
        let mask = Mat.zeros(item.source.rows, item.source.cols, item.source.type_())
        mask.roi(Rect(5, 4, 11, 8)).set_to(Scalar(255.0, 0.0, 0.0))
        return calc_hist([item.source], [0], mask, [int(args[1])], [0.0, 256.0])
    }
    if item.name == "calcHist2D" {
        return calc_hist(
            [item.source],
            [0, 1],
            null,
            [int(args[1]), int(args[2])],
            [0.0, 256.0, 0.0, 256.0]
        )
    }
    if item.name == "equalizeHist" { return equalize_hist(item.source) }
    if item.name == "clahe" {
        let tiles = int(args[2])
        return CLAHE(float(args[1]), Size(tiles, tiles)).apply(item.source)
    }
    if item.name == "compareHist" {
        let shifted = Mat.zeros(item.source.rows, item.source.cols, item.source.type_())
        let source = item.source.to_list()
        var rolled: list[float] = []
        for y in 0..item.source.rows {
            for x in 0..item.source.cols {
                rolled.push(
                    source[y * item.source.cols + (x + item.source.cols - 3) % item.source.cols]
                )
            }
        }
        let other = Mat.from_list(item.source.rows, item.source.cols, item.source.type_(), rolled)
        let _unused = shifted
        let a = calc_hist([item.source], [0], null, [32], [0.0, 256.0])
        let b = calc_hist([other], [0], null, [32], [0.0, 256.0])
        return Mat.from_list(
            1,
            1,
            CV_MAKETYPE(5, 1),
            [compare_hist(a, b, histogram_method(args[0]))]
        )
    }
    if item.name == "calcBackProject" {
        let hist = calc_hist([item.source], [0], null, [int(args[1])], [0.0, 180.0])
        return calc_back_project([item.source], [0], hist, [0.0, 180.0], 1.0)
    }
    throw ValueError(f"no jaicv binding for {item.name}")
}

let FEATURE_NAMES = [
    "matchTemplate",
    "matchTemplateColour",
    "cornerHarris",
    "cornerMinEigenVal",
    "houghLines",
]

fn template_method(name: str) -> int {
    if name == "SQDIFF" { return TM_SQDIFF }
    if name == "SQDIFF_NORMED" { return TM_SQDIFF_NORMED }
    if name == "CCORR" { return TM_CCORR }
    if name == "CCORR_NORMED" { return TM_CCORR_NORMED }
    if name == "CCOEFF" { return TM_CCOEFF }
    if name == "CCOEFF_NORMED" { return TM_CCOEFF_NORMED }
    throw ValueError(f"no jaicv template method for {name}")
}

fn run_features(item: Case) -> Mat {
    let args = item.args
    if item.name == "matchTemplate" {
        return match_template(
            item.source,
            item.source.roi(Rect(10, 8, 7, 6)),
            template_method(args[1])
        )
    }
    if item.name == "matchTemplateColour" {
        return match_template(
            item.source,
            item.source.roi(Rect(7, 5, 8, 6)),
            template_method(args[1])
        )
    }
    if item.name == "cornerHarris" {
        return corner_harris(item.source, int(args[0]), int(args[1]), 0.04)
    }
    if item.name == "cornerMinEigenVal" {
        return corner_min_eigen_val(item.source, int(args[0]), int(args[1]))
    }
    if item.name == "houghLines" {
        let lines = hough_lines(item.source, 1.0, math.PI / 180.0, int(args[2]))
        var values: list[float] = []
        for line in lines {
            values.push(line.rho)
            values.push(line.theta)
        }
        return Mat.from_list(lines.len(), 2, CV_MAKETYPE(5, 1), values)
    }
    throw ValueError(f"no jaicv binding for {item.name}")
}

fn run_fast(item: Case) -> Mat {
    let found = fast(item.source, int(item.args[1]), int(item.args[2]) != 0)
    var values: list[float] = []
    for point in found {
        values.push(point.pt.x)
        values.push(point.pt.y)
        values.push(point.response)
    }
    return Mat.from_list(found.len(), 3, CV_MAKETYPE(5, 1), values)
}

var _cascade: CascadeClassifier? = null

fn cascade_for() -> CascadeClassifier {
    let existing = _cascade
    if existing is not null { return existing }
    let made = CascadeClassifier("packages/jaicv/tests/oracle/synthetic_cascade.xml")
    _cascade = made
    return made
}

fn run_cascade(item: Case) -> Mat {
    #: With no minimum neighbour count the grouping step is skipped, so this
    #: compares the raw window-by-window decisions rather than the clustering
    #: on top of them.
    let found = cascade_for().detect_multi_scale(item.source, float(item.args[0]), 0)
    var values: list[float] = []
    for box in found {
        values.push(float(box.x))
        values.push(float(box.y))
        values.push(float(box.width))
        values.push(float(box.height))
    }
    return Mat.from_list(found.len(), 4, CV_MAKETYPE(4, 1), values)
}

fn run_case(item: Case) -> Mat {
    if item.name.starts_with("draw") { return run_drawing(item) }
    if item.name == "cascadeDetect" { return run_cascade(item) }
    if item.name == "fast" { return run_fast(item) }
    if FEATURE_NAMES.contains(item.name) { return run_features(item) }
    if SEGMENT_NAMES.contains(item.name) { return run_segmentation(item) }
    if HIST_NAMES.contains(item.name) { return run_hist(item) }
    if SHAPE_NAMES.contains(item.name) { return run_shape(item) }
    if item.name == "findContours" or item.name == "contourHierarchy" { return run_contours(item) }
    if item.name == "cvtColor" { return cvt_color(item.source, colour_code(item.args[0])) }
    if item.name == "resize" {
        return resize(
            item.source,
            size: Size(int(item.args[1]), int(item.args[2])),
            interpolation: interpolation_named(item.args[0])
        )
    }
    if item.name == "pyrDown" { return pyr_down(item.source) }
    if item.name == "pyrUp" { return pyr_up(item.source) }
    if item.name == "copyMakeBorder" {
        return copy_make_border(
            item.source,
            int(item.args[1]),
            int(item.args[2]),
            int(item.args[3]),
            int(item.args[4]),
            border_mode: border_named(item.args[0])
        )
    }
    if item.name == "warpAffine" {
        return warp_affine(item.source, rotation_case(), Size(10, 8))
    }
    if item.name == "warpPerspective" {
        return warp_perspective(
            item.source,
            get_perspective_transform(QUAD_SOURCE, QUAD_TARGET),
            Size(10, 8)
        )
    }
    if item.name == "getRotationMatrix2D" { return rotation_case() }
    if item.name == "getPerspectiveTransform" {
        return get_perspective_transform(QUAD_SOURCE, QUAD_TARGET)
    }
    if item.name == "getAffineTransform" {
        var source: list[Point2f] = []
        var target: list[Point2f] = []
        for index in 0..3 {
            source.push(QUAD_SOURCE[index])
            target.push(QUAD_TARGET[index])
        }
        return get_affine_transform(source, target)
    }
    if item.name == "blur" {
        return blur(item.source, Size(int(item.args[0]), int(item.args[1])))
    }
    if item.name == "GaussianBlur" {
        return gaussian_blur(
            item.source,
            Size(int(item.args[0]), int(item.args[1])),
            float(item.args[2])
        )
    }
    if item.name == "medianBlur" { return median_blur(item.source, int(item.args[0])) }
    if item.name == "boxFilter" {
        return box_filter(
            item.source,
            -1,
            Size(int(item.args[0]), int(item.args[1])),
            normalize: false
        )
    }
    if item.name == "filter2D" {
        if item.args[0] == "sharpen" {
            let weights = Mat.from_list(
                3,
                3,
                CV_MAKETYPE(CV_32F, 1),
                [0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0]
            )
            return filter_2d(item.source, -1, weights)
        }
        var values: list[float] = []
        for index in 0..25 { values.push((float(index) - 12.0) / 40.0) }
        return filter_2d(item.source, -1, Mat.from_list(5, 5, CV_MAKETYPE(CV_32F, 1), values))
    }
    if item.name == "Sobel" {
        return sobel(
            item.source,
            CV_32F,
            int(item.args[0]),
            int(item.args[1]),
            ksize: int(item.args[2])
        )
    }
    if item.name == "Scharr" {
        return scharr(item.source, CV_32F, int(item.args[0]), int(item.args[1]))
    }
    if item.name == "Laplacian" { return laplacian(item.source, CV_32F, ksize: 1) }
    if item.name == "getGaussianKernel" {
        return get_gaussian_kernel(int(item.args[0]), float(item.args[1]))
    }
    if item.name == "threshold" {
        let modes = {
            "BINARY": THRESH_BINARY,
            "BINARY_INV": THRESH_BINARY_INV,
            "TRUNC": THRESH_TRUNC,
            "TOZERO": THRESH_TOZERO,
            "TOZERO_INV": THRESH_TOZERO_INV,
        }
        let (_level, out) = threshold(
            item.source,
            float(item.args[1]),
            float(item.args[2]),
            modes[item.args[0]]
        )
        return out
    }
    if item.name == "thresholdOtsu" {
        let (_level, out) = threshold(item.source, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU)
        return out
    }
    if item.name == "adaptiveThreshold" {
        let method = if item.args[0] == "MEAN" {
            ADAPTIVE_THRESH_MEAN_C
        } else {
            ADAPTIVE_THRESH_GAUSSIAN_C
        }
        return adaptive_threshold(
            item.source,
            255.0,
            method,
            THRESH_BINARY,
            int(item.args[1]),
            float(item.args[2])
        )
    }
    if item.name == "getStructuringElement" {
        return get_structuring_element(
            shape_named(item.args[0]),
            Size(int(item.args[1]), int(item.args[1]))
        )
    }
    if item.name == "erode" {
        return erode(item.source, element: structuring_case(item.args))
    }
    if item.name == "dilate" {
        return dilate(item.source, element: structuring_case(item.args))
    }
    if item.name == "morphologyEx" {
        let ops = {
            "OPEN": MORPH_OPEN,
            "CLOSE": MORPH_CLOSE,
            "GRADIENT": MORPH_GRADIENT,
            "TOPHAT": MORPH_TOPHAT,
            "BLACKHAT": MORPH_BLACKHAT,
        }
        return morphology_ex(
            item.source,
            ops[item.args[0]],
            element: get_structuring_element(MORPH_RECT, Size(3, 3))
        )
    }
    if item.name == "Canny" {
        return canny(
            item.source,
            float(item.args[0]),
            float(item.args[1]),
            aperture_size: int(item.args[2]),
            l2_gradient: item.args[3] == "1"
        )
    }
    if item.name == "remap" {
        var xs: list[float] = []
        var ys: list[float] = []
        for y in 0..8 {
            for x in 0..10 {
                xs.push(float(9 - x))
                ys.push(float(y))
            }
        }
        return remap(
            item.source,
            Mat.from_list(8, 10, CV_MAKETYPE(5, 1), xs),
            Mat.from_list(8, 10, CV_MAKETYPE(5, 1), ys)
        )
    }
    throw ValueError(f"no jaicv binding for {item.name}")
}

#: How many elements a case may differ in before it counts as a failure.
#:
#: Zero for every shape whose pixels this reproduces exactly, which is all of
#: them except the ones OpenCV rasterises through its sixteen-bit fixed-point
#: convex fill: a stroke wider than one pixel, and a filled ellipse. Those come
#: out in the same place and the same width, but which side of the boundary a
#: pixel falls on differs here and there, because OpenCV's span rounding along
#: the two long edges does not follow from the algorithm as published and the
#: shape of the disagreement changes with the slope.
#:
#: A third of the painted pixels is the budget. The worst case in the recorded
#: set is a thickness-two line clipped at both ends, where the surviving stroke
#: is short and the boundary is most of it: 75 of 288. Everything else is far
#: under — a filled ellipse differs in nine elements out of 867.
fn budget_for(item: Case, painted: int) -> int {
    #: A cascade decides each window by comparing a response against a fixed
    #: threshold, and the response is read out of an image this shrank on the
    #: device while the reference shrank it on the host. Those two agree to
    #: within a grey level, which is enough to move a window sitting right on
    #: its threshold from one side to the other. Over two hundred windows one
    #: or two do.
    if item.name == "cascadeDetect" { return max(painted // 50, 4) }
    if not uses_fixed_point(item) { return 0 }
    return painted // 3
}

fn uses_fixed_point(item: Case) -> bool {
    let thickness = stroke_thickness(item)
    if thickness > 1 { return true }
    return item.name == "drawEllipse" and thickness < 0
}

fn stroke_thickness(item: Case) -> int {
    if not item.name.starts_with("draw") { return 0 }
    let args = item.args
    if item.name == "drawLine" or item.name == "drawRect" { return int(args[4]) }
    if item.name == "drawCircle" { return int(args[3]) }
    if item.name == "drawEllipse" { return int(args[7]) }
    if item.name == "drawPolylines" { return int(args[1]) }
    if item.name == "drawArrow" or item.name == "drawMarker" { return int(args[4]) }
    return 0
}

#: Turn a hull so it starts where OpenCV's does.
#:
#: A convex hull is a ring, and both sides walk it the same way; which vertex
#: is written first follows from the scan that built it, and OpenCV's and
#: jaicv's scans differ. Rotating to the same start compares the rings.
fn rotated_like(expected: Mat, produced: Mat) -> Mat {
    if produced.rows != expected.rows or produced.rows == 0 { return produced }
    let want = expected.to_list()
    let got = produced.to_list()
    let width = produced.cols
    for offset in 0..produced.rows {
        var same = true
        for row in 0..produced.rows {
            for column in 0..width {
                if got[(row + offset) % produced.rows * width + column] !=
                    want[row * width + column] {
                    same = false
                    break
                }
            }
            if not same { break }
        }
        if not same { continue }
        var values: list[float] = []
        for row in 0..produced.rows {
            for column in 0..width {
                values.push(got[(row + offset) % produced.rows * width + column])
            }
        }
        return Mat.from_list(produced.rows, width, produced.type_(), values)
    }
    return produced
}

#: How far a case is allowed to drift, in proportion to the largest value in
#: it rather than to each value on its own.
#:
#: A template-matching sum runs to hundreds of thousands and both sides
#: accumulate it in float32, so what is left over is association order over
#: numbers that size. `TM_CCOEFF` then subtracts two such sums, and the drift
#: rides on what went into the subtraction, not on the small number that comes
#: out — which is why this is measured against the case's own scale.
fn relative_tolerance_for(name: str) -> float {
    if name == "matchTemplate" or name == "matchTemplateColour" { return 1e-5 }
    return 0.0
}

fn case_scale(name: str, want: list[float]) -> float {
    if relative_tolerance_for(name) == 0.0 { return 0.0 }
    var peak = 0.0
    for value in want {
        let size = math.fabs(value)
        if size > peak { peak = size }
    }
    return peak
}

fn compare(item: Case, produced: Mat) -> tuple[float, int] {
    let want = item.expected.to_list()
    let got = produced.to_list()
    if want.len() != got.len() {
        throw AssertionError(
            f"{item.name} {item.args} produced {got.len()} values, OpenCV gave {want.len()}"
        )
    }
    let slack = relative_tolerance_for(item.name) * case_scale(item.name, want)
    var worst = 0.0
    var worst_at = 0
    for index in 0..want.len() {
        let drift = math.fabs(got[index] - want[index]) - slack
        if drift > worst {
            worst = drift
            worst_at = index
        }
    }
    return (max(worst, 0.0), worst_at)
}

fn test_every_recorded_case_matches_opencv() -> void {
    if not gpu.is_available() { return }
    let path = Path("packages/jaicv/tests/oracle/cases.txt")
    if not path.exists() { return }
    let cases = load_cases(path)
    assert_true(cases.len() > 0)
    var failures: list[str] = []
    var compared = 0
    for item in cases {
        compared += 1
        var produced = run_case(item)
        if produced.rows != item.expected.rows or produced.cols != item.expected.cols {
            failures.push(f"{item.name} {item.args}: shape {produced} vs {item.expected}")
            continue
        }
        if produced.channels() != item.expected.channels() {
            failures.push(
                f"{item.name} {item.args}: {type_to_string(produced.type_())} vs " +
                    f"{type_to_string(item.expected.type_())}"
            )
            continue
        }
        if item.name == "convexHull" { produced = rotated_like(item.expected, produced) }
        let (worst, worst_at) = compare(item, produced)
        let allowed = tolerance_for(item.name, item.args)
        if worst > allowed {
            let want = item.expected.to_list()
            let got = produced.to_list()
            let slack = relative_tolerance_for(item.name) * case_scale(item.name, want)
            var differing = 0
            for index in 0..want.len() {
                if math.fabs(got[index] - want[index]) - slack > allowed { differing += 1 }
            }
            var painted = 0
            let source = item.source.to_list()
            for index in 0..want.len() {
                if want[index] != source[index] { painted += 1 }
            }
            let budget = budget_for(item, painted)
            if differing > budget {
                failures.push(
                    f"{item.name} {item.args}: {differing} of {painted} elements differ " +
                        f"(budget {budget}), " +
                        f"worst {worst} at {worst_at} " +
                        f"(got {got[worst_at]}, OpenCV {want[worst_at]}, allowed {allowed})"
                )
            }
        }
    }
    #: A silently skipped case is worse than a failing one, so the count the
    #: file holds is asserted rather than assumed.
    assert_true(compared == cases.len())
    if failures.len() > 0 {
        var report = f"{failures.len()} of {cases.len()} cases disagree with OpenCV:"
        for line in failures { report = report + "\n  " + line }
        throw AssertionError(report)
    }
}

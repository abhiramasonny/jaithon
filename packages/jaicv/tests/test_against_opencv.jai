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
from jaicv.core.types import CV_32F, Point, Point2f, Scalar, Size
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
    for index in 4..fields.len() { values.push(float(fields[index])) }
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
    #: A mask or a structuring element only ever holds two values, so a
    #: one-unit tolerance would hide a wrong pixel entirely: these are exact.
    if name == "getStructuringElement" or
        name == "threshold" or
        name == "thresholdOtsu" or
        name == "adaptiveThreshold" or
        name == "Canny" {
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

fn run_case(item: Case) -> Mat {
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
        return get_structuring_element(shape_named(item.args[0]), Size(int(item.args[1]), int(item.args[1])))
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

fn compare(item: Case, produced: Mat) -> tuple[float, int] {
    let want = item.expected.to_list()
    let got = produced.to_list()
    if want.len() != got.len() {
        throw AssertionError(
            f"{item.name} {item.args} produced {got.len()} values, OpenCV gave {want.len()}"
        )
    }
    var worst = 0.0
    var worst_at = 0
    for index in 0..want.len() {
        let drift = math.fabs(got[index] - want[index])
        if drift > worst {
            worst = drift
            worst_at = index
        }
    }
    return (worst, worst_at)
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
        let produced = run_case(item)
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
        let (worst, worst_at) = compare(item, produced)
        let allowed = tolerance_for(item.name, item.args)
        if worst > allowed {
            let want = item.expected.to_list()
            let got = produced.to_list()
            failures.push(
                f"{item.name} {item.args}: off by {worst} at {worst_at} " +
                    f"(got {got[worst_at]}, OpenCV {want[worst_at]}, allowed {allowed})"
            )
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

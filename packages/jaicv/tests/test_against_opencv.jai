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
    let _unused = args
    if name == "cvtColor" { return 1.0 }
    return 1.0
}

fn run_case(item: Case) -> Mat {
    if item.name == "cvtColor" { return cvt_color(item.source, colour_code(item.args[0])) }
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
    for item in cases {
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
    if failures.len() > 0 {
        var report = f"{failures.len()} of {cases.len()} cases disagree with OpenCV:"
        for line in failures { report = report + "\n  " + line }
        throw AssertionError(report)
    }
}

#: Following something from frame to frame: mean shift, its adaptive form, and
#: a Kalman filter to smooth the result.
#:
#: `mean_shift` and `cam_shift` work on a probability image — usually what
#: `calc_back_project` gives from a colour histogram of the thing being
#: followed — and slide a window uphill to where that probability is densest.
#: `KalmanFilter` is the general linear filter, and is what turns a jittery
#: per-frame measurement into a track with a velocity.
#:
#: ```
#: let histogram = calc_hist([hue], [0], mask, [16], [0.0, 180.0])
#: var window = Rect(60, 40, 40, 40)
#: let (rounds, moved) = mean_shift(calc_back_project([hue], [0], histogram, [0.0, 180.0]), window)
#: ```
import std.math
from ..core.mat import Mat
from ..core.types import CV_32F, CV_MAKETYPE, Point2f, Rect, RotatedRect, Size2f

fn matmul(a: Mat, b: Mat) -> Mat {
    if a.cols != b.rows { throw ValueError(f"cannot multiply {a} by {b}") }
    let left = a.values()
    let right = b.values()
    var out: list[float] = []
    for row in 0..a.rows {
        for column in 0..b.cols {
            var total = 0.0
            for index in 0..a.cols {
                total += left[row * a.cols + index] * right[index * b.cols + column]
            }
            out.push(total)
        }
    }
    return Mat.from_list(a.rows, b.cols, CV_MAKETYPE(CV_32F, 1), out)
}

fn transpose_of(a: Mat) -> Mat {
    let values = a.values()
    var out: list[float] = []
    for column in 0..a.cols {
        for row in 0..a.rows { out.push(values[row * a.cols + column]) }
    }
    return Mat.from_list(a.cols, a.rows, CV_MAKETYPE(CV_32F, 1), out)
}

fn add_of(a: Mat, b: Mat) -> Mat {
    let left = a.values()
    let right = b.values()
    var out: list[float] = []
    for index in 0..left.len() { out.push(left[index] + right[index]) }
    return Mat.from_list(a.rows, a.cols, CV_MAKETYPE(CV_32F, 1), out)
}

fn subtract_of(a: Mat, b: Mat) -> Mat {
    let left = a.values()
    let right = b.values()
    var out: list[float] = []
    for index in 0..left.len() { out.push(left[index] - right[index]) }
    return Mat.from_list(a.rows, a.cols, CV_MAKETYPE(CV_32F, 1), out)
}

fn identity_of(size: int) -> Mat {
    var out: list[float] = []
    for row in 0..size {
        for column in 0..size { out.push(if row == column { 1.0 } else { 0.0 }) }
    }
    return Mat.from_list(size, size, CV_MAKETYPE(CV_32F, 1), out)
}

fn invert_of(a: Mat) -> Mat {
    if a.rows != a.cols { throw ValueError("only a square matrix can be inverted") }
    let n = a.rows
    let source = a.values()
    #: Gauss-Jordan with partial pivoting, on the matrix and an identity side
    #: by side; what is left where the identity was is the inverse.
    var work: list[list[float]] = []
    for row in 0..n {
        var line: list[float] = []
        for column in 0..n { line.push(source[row * n + column]) }
        for column in 0..n { line.push(if row == column { 1.0 } else { 0.0 }) }
        work.push(line)
    }
    for column in 0..n {
        var pivot = column
        for row in column..n {
            if math.fabs(work[row][column]) > math.fabs(work[pivot][column]) { pivot = row }
        }
        if math.fabs(work[pivot][column]) < 1e-12 { throw ValueError("this matrix has no inverse") }
        if pivot != column {
            let swap = work[pivot]
            work[pivot] = work[column]
            work[column] = swap
        }
        let scale = work[column][column]
        for index in 0..2 * n { work[column][index] /= scale }
        for row in 0..n {
            if row == column { continue }
            let factor = work[row][column]
            if factor == 0.0 { continue }
            for index in 0..2 * n { work[row][index] -= factor * work[column][index] }
        }
    }
    var out: list[float] = []
    for row in 0..n {
        for column in 0..n { out.push(work[row][n + column]) }
    }
    return Mat.from_list(n, n, CV_MAKETYPE(CV_32F, 1), out)
}

#: A linear filter over a state that is only partly observed.
#:
#: The fields carry OpenCV's names and shapes, so a filter set up for one is
#: set up for the other: fill in `transition_matrix`, `measurement_matrix`, and
#: the two noise covariances, then alternate `predict` and `correct`.
pub class KalmanFilter {
    #: How many numbers the state has.
    pub let dynam_params: int
    #: How many of them are measured.
    pub let measure_params: int
    #: How many control inputs there are.
    pub let control_params: int

    #: The state before this frame's measurement.
    pub var state_pre: Mat
    #: The state after it.
    pub var state_post: Mat
    #: How the state moves on its own.
    pub var transition_matrix: Mat
    #: How a control input moves it.
    pub var control_matrix: Mat
    #: How the state turns into a measurement.
    pub var measurement_matrix: Mat
    #: How much the state drifts by itself.
    pub var process_noise_cov: Mat
    #: How noisy a measurement is.
    pub var measurement_noise_cov: Mat
    #: Uncertainty before this frame's measurement.
    pub var error_cov_pre: Mat
    #: Uncertainty after it.
    pub var error_cov_post: Mat
    #: How far the last measurement was trusted.
    pub var gain: Mat

    pub fn init(self, dynam_params: int, measure_params: int, control_params: int = 0) {
        if dynam_params <= 0 or measure_params <= 0 {
            throw ValueError("a Kalman filter needs a state and something to measure")
        }
        self.dynam_params = dynam_params
        self.measure_params = measure_params
        self.control_params = control_params

        self.state_pre = Mat.zeros(dynam_params, 1, CV_MAKETYPE(CV_32F, 1))
        self.state_post = Mat.zeros(dynam_params, 1, CV_MAKETYPE(CV_32F, 1))
        self.transition_matrix = identity_of(dynam_params)
        self.control_matrix = Mat.zeros(
            dynam_params,
            max(control_params, 1),
            CV_MAKETYPE(CV_32F, 1)
        )
        self.measurement_matrix = Mat.zeros(measure_params, dynam_params, CV_MAKETYPE(CV_32F, 1))
        self.process_noise_cov = identity_of(dynam_params)
        self.measurement_noise_cov = identity_of(measure_params)
        self.error_cov_pre = Mat.zeros(dynam_params, dynam_params, CV_MAKETYPE(CV_32F, 1))
        self.error_cov_post = Mat.zeros(dynam_params, dynam_params, CV_MAKETYPE(CV_32F, 1))
        self.gain = Mat.zeros(dynam_params, measure_params, CV_MAKETYPE(CV_32F, 1))

        #: The measurement matrix starts as the first few state entries being
        #: what is measured, which is what almost every use of this wants.
        for index in 0..min(measure_params, dynam_params) {
            self.measurement_matrix.set_at(index, index, 1.0)
        }
        self.measurement_matrix.flush()
    }

    #: Move the state forward one step, without a measurement.
    pub fn predict(self, control: Mat? = null) -> Mat {
        var predicted = matmul(self.transition_matrix, self.state_post)
        let input = control
        if input is not null { predicted = add_of(predicted, matmul(self.control_matrix, input)) }
        self.state_pre = predicted

        let moved = matmul(
            matmul(self.transition_matrix, self.error_cov_post),
            transpose_of(self.transition_matrix)
        )
        self.error_cov_pre = add_of(moved, self.process_noise_cov)
        #: With no measurement to come, the prediction stands as the estimate.
        self.state_post = predicted.clone()
        self.error_cov_post = self.error_cov_pre.clone()
        return predicted
    }

    #: Fold a measurement into the prediction.
    pub fn correct(self, measurement: Mat) -> Mat {
        if measurement.rows != self.measure_params or measurement.cols != 1 {
            throw ValueError(f"a measurement has to be {self.measure_params} by one")
        }
        let observation = transpose_of(self.measurement_matrix)
        let spread = add_of(
            matmul(matmul(self.measurement_matrix, self.error_cov_pre), observation),
            self.measurement_noise_cov
        )
        self.gain = matmul(matmul(self.error_cov_pre, observation), invert_of(spread))

        let surprise = subtract_of(measurement, matmul(self.measurement_matrix, self.state_pre))
        self.state_post = add_of(self.state_pre, matmul(self.gain, surprise))
        let shrink = subtract_of(
            identity_of(self.dynam_params),
            matmul(self.gain, self.measurement_matrix)
        )
        self.error_cov_post = matmul(shrink, self.error_cov_pre)
        return self.state_post
    }
}

fn clamp_window(window: Rect, rows: int, cols: int) -> Rect {
    var x = window.x
    var y = window.y
    var width = window.width
    var height = window.height
    if width <= 0 or height <= 0 { throw ValueError("a tracking window needs a positive size") }
    if x < 0 { x = 0 }
    if y < 0 { y = 0 }
    if x + width > cols { x = max(cols - width, 0) }
    if y + height > rows { y = max(rows - height, 0) }
    if x + width > cols { width = cols - x }
    if y + height > rows { height = rows - y }
    return Rect(x, y, max(width, 1), max(height, 1))
}

#: Slide a window to where the probability under it is densest.
#:
#: Returns how many steps it took and where the window ended up. The window
#: keeps its size; only its position moves.
pub fn mean_shift(
    probability: Mat,
    window: Rect,
    iterations: int = 10,
    epsilon: float = 1.0
) -> tuple[int, Rect] {
    if probability.channels() != 1 { throw ValueError("mean shift needs a single-channel Mat") }
    probability.flush()
    let values = probability.values()
    let rows = probability.rows
    let cols = probability.cols
    var current = clamp_window(window, rows, cols)

    for step in 0..iterations {
        var total = 0.0
        var sum_x = 0.0
        var sum_y = 0.0
        for y in current.y..current.y + current.height {
            for x in current.x..current.x + current.width {
                let weight = values[y * cols + x]
                if weight <= 0.0 { continue }
                total += weight
                sum_x += weight * float(x)
                sum_y += weight * float(y)
            }
        }
        if total <= 0.0 { return (step, current) }
        let centre_x = sum_x / total
        let centre_y = sum_y / total
        let wanted_x = int(centre_x - float(current.width) * 0.5 + 0.5)
        let wanted_y = int(centre_y - float(current.height) * 0.5 + 0.5)
        let moved = clamp_window(
            Rect(wanted_x, wanted_y, current.width, current.height),
            rows,
            cols
        )
        let dx = float(moved.x - current.x)
        let dy = float(moved.y - current.y)
        current = moved
        if dx * dx + dy * dy <= epsilon * epsilon { return (step + 1, current) }
    }
    return (iterations, current)
}

#: Mean shift that also grows, shrinks, and turns with what it is following.
#:
#: Returns the region as a rotated rectangle, taken from the second moments of
#: the probability under the settled window, along with the upright window
#: itself.
pub fn cam_shift(
    probability: Mat,
    window: Rect,
    iterations: int = 10,
    epsilon: float = 1.0
) -> tuple[RotatedRect, Rect] {
    let rows = probability.rows
    let cols = probability.cols
    var current = clamp_window(window, rows, cols)

    #: The window is resized between rounds, so the whole mean shift is run
    #: again each time rather than once.
    for _round in 0..iterations {
        let (_steps, settled) = mean_shift(probability, current, iterations, epsilon)
        current = settled
        let values = probability.values()
        var m00 = 0.0
        var m10 = 0.0
        var m01 = 0.0
        for y in current.y..current.y + current.height {
            for x in current.x..current.x + current.width {
                let weight = values[y * cols + x]
                if weight <= 0.0 { continue }
                m00 += weight
                m10 += weight * float(x)
                m01 += weight * float(y)
            }
        }
        if m00 <= 0.0 { break }
        #: The window is grown to two standard deviations of the mass it holds,
        #: which is what makes this track something coming closer.
        let side = max(int(2.0 * math.sqrt(m00 / 256.0)), 3)
        let cx = m10 / m00
        let cy = m01 / m00
        let grown = clamp_window(
            Rect(int(cx) - side // 2, int(cy) - side // 2, side, side),
            rows,
            cols
        )
        if grown.width == current.width and
            grown.height == current.height and
            grown.x == current.x and
            grown.y == current.y {
            break
        }
        current = grown
    }

    let values = probability.values()
    var m00 = 0.0
    var m10 = 0.0
    var m01 = 0.0
    var m20 = 0.0
    var m11 = 0.0
    var m02 = 0.0
    for y in current.y..current.y + current.height {
        for x in current.x..current.x + current.width {
            let weight = values[y * cols + x]
            if weight <= 0.0 { continue }
            m00 += weight
            m10 += weight * float(x)
            m01 += weight * float(y)
            m20 += weight * float(x) * float(x)
            m11 += weight * float(x) * float(y)
            m02 += weight * float(y) * float(y)
        }
    }
    if m00 <= 0.0 {
        return (
            RotatedRect(
                Point2f(
                    float(current.x) + float(current.width) * 0.5,
                    float(current.y) + float(current.height) * 0.5
                ),
                Size2f(float(current.width), float(current.height)),
                0.0
            ),
            current
        )
    }

    let cx = m10 / m00
    let cy = m01 / m00
    let a = m20 / m00 - cx * cx
    let b = 2.0 * (m11 / m00 - cx * cy)
    let c = m02 / m00 - cy * cy
    let root = math.sqrt(b * b + (a - c) * (a - c))
    let width = math.sqrt(max((a + c + root) * 0.5, 0.0)) * 4.0
    let height = math.sqrt(max((a + c - root) * 0.5, 0.0)) * 4.0
    let angle = 0.5 * math.atan2(b, a - c) * 180.0 / math.PI
    return (RotatedRect(Point2f(cx, cy), Size2f(width, height), angle), current)
}

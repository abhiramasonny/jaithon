#: `LineSegmentDetector` -- LSD, the a-contrario line segment detector of von
#: Gioi, Jakubowicz, Morel and Randall (IPOL, 2012).
#:
#: A segment here is grown rather than voted for. Every pixel gets the angle of
#: its level line from a 2x2 forward difference -- not from Sobel, which is
#: wider than the one-pixel structures the detector is looking for -- and
#: neighbouring pixels whose level lines agree to within `ang_th` are joined
#: into a region. A rectangle is fitted to the region through its centre of
#: mass and its principal direction, and then the rectangle has to earn its
#: place: the number of pixels inside it that line up with its own direction is
#: compared against the number a pure noise image of the same size would
#: produce by chance, and the rectangle is kept only when that count would be
#: absurd. That last test is the a-contrario one, and it is what makes the
#: detector need no threshold tuning -- it is also why a noise image yields
#: nothing at all here rather than a scattering of short segments.
#:
#: A segment is `list[float]` of four -- `[x1, y1, x2, y2]` -- which is what
#: OpenCV's `Vec4f` holds. jaicv has no fixed-length vector type, and the two
#: existing line types are unsuitable: `HoughLine` is an infinite line and
#: `HoughSegment` has integer ends, where LSD's whole point is that the ends
#: are found to a fraction of a pixel.
#:
#: ## How closely it matches OpenCV
#:
#: Handed the same pixels, this and OpenCV 5.0.0 compute the same segments.
#: Over seventeen synthetic images -- the ones in `tests/test_lsd.jai` plus a
#: triangle, a ring, a checkerboard, a fan of lines, a staircase, a ramp -- at
#: all three refinement settings, this reports 476 segments and 474 of them
#: are segments `cv2.createLineSegmentDetector` reports as well. The median
#: endpoint difference is 0.007 of a pixel, 454 of the 474 agree to within
#: 0.015, and the two counts are equal in 45 of the 51 cases. Every case where
#: they are not, and every endpoint further out than 0.015, is one of the
#: three differences below. On the three images where the two disagree most,
#: handing both implementations the same downscaled image -- so that only the
#: detector proper is being compared -- brings the worst endpoint
#: disagreement down to 0.00005 of a pixel.
#:
#: Three things differ:
#:
#: - **OpenCV skips the validation unless it is asked for it.** OpenCV runs
#:   the NFA test only when `refine` is `LSD_REFINE_ADV`; at
#:   `LSD_REFINE_NONE` and `LSD_REFINE_STD` it reports every region that
#:   survives region growing. The paper validates every rectangle whatever
#:   refinement is asked for, and so does this -- which is why `log_eps` and
#:   the returned NFA mean something at every setting here and only at
#:   `LSD_REFINE_ADV` in OpenCV. On the 65x65 noise image in the tests OpenCV
#:   at `LSD_REFINE_STD` reports nine segments and this reports none;
#:   OpenCV's own `LSD_REFINE_ADV`, which does validate, also reports none.
#:   It costs something at the other end: a ten-pixel checkerboard edge is a
#:   marginal detection that passes only once the variations have been tried,
#:   so on a checkerboard this reports 23 segments at `LSD_REFINE_STD` and
#:   180 at `LSD_REFINE_ADV`, where OpenCV reports 180 at all three.
#: - **The downscale is a grey level out.** `gaussian_blur` reproduces
#:   OpenCV's exactly. `resize` is within one grey level of OpenCV's
#:   `INTER_LINEAR_EXACT` -- it differs on one to thirty per cent of pixels,
#:   never by more than one -- and it takes its sampling grid from the ratio
#:   of the two sizes where OpenCV takes it from the requested `scale`, so the
#:   two grids coincide only when `scale * cols` and `scale * rows` are whole.
#:   That single grey level is the whole of the 0.015 above. It shows most
#:   where a region has to be regrown to make its rectangle dense, which is
#:   chaotic in the pixel values by nature: on a ring and on a fan of lines
#:   one endpoint each moves by about a pixel, and on a detection sitting on
#:   the acceptance threshold it can decide the detection.
#: - **OpenCV's NFA is not quite the paper's either.** To count the points in
#:   a rectangle OpenCV rounds its four corners to whole pixels and sorts
#:   them, where the paper sweeps the rectangle column by column at full
#:   precision, which is what this does. The counts differ by a point here and
#:   there. On the clean images above that never changes a decision; on a
#:   photograph, where a great many detections sit on the threshold, it does.
#:   On a 300x300 crop of `assets/benchmark.png` at `LSD_REFINE_ADV` this
#:   reports 146 segments to OpenCV's 133 and 118 are common to both, while
#:   the paper's algorithm given OpenCV's own downscaled image reports 143 --
#:   so the rectangle sweep is most of that gap and the grey level is the
#:   rest. At the other two settings, where OpenCV validates nothing, all 107
#:   segments this reports are among the 325 OpenCV reports, with a median
#:   endpoint difference of 0.003 of a pixel.
#:
#: ```
#: let detector = create_line_segment_detector()
#: let (lines, widths, precisions, nfas) = detector.detect(gray)
#: let drawn = detector.draw_segments(gray, lines)
#: ```
import std.math
from ..core.mat import Mat
from ..core.types import CV_8UC1, CV_8UC3, Point, Point2f, Scalar, Size
from .color import cvt_color, COLOR_BGR2GRAY, COLOR_GRAY2BGR
from .drawing import line
from .filter import gaussian_blur
from .geometric import resize

#: Report each region's rectangle as it was fitted.
pub const LSD_REFINE_NONE = 0
#: Cut a rectangle down until its aligned points are dense enough to be a line
#: rather than a smear.
pub const LSD_REFINE_STD = 1
#: Also try variations of the rectangle -- narrower, shifted, a finer angle
#: tolerance -- to rescue one that fails the NFA test as fitted.
pub const LSD_REFINE_ADV = 2

#: The level-line angle of a pixel whose gradient is too weak to have one. The
#: value is only ever compared for equality, so any impossible angle does.
let NOT_DEFINED = -1024.0

let TWO_PI = 2.0 * math.PI
let THREE_HALF_PI = 1.5 * math.PI

#: The number of rectangle variations `LSD_REFINE_ADV` tries, which enters the
#: number of tests the NFA is measured against.
let VARIATIONS = 11.0

#: The rectangle fitted to a region: the segment along its axis, the width it
#: spans, and the two numbers the NFA test needs -- the angle tolerance a pixel
#: has to fall inside to count as aligned, and the probability of that
#: happening by chance.
class Support {
    pub var x1: float
    pub var y1: float
    pub var x2: float
    pub var y2: float
    pub var width: float
    pub var centre_x: float
    pub var centre_y: float
    pub var angle: float
    pub var dx: float
    pub var dy: float
    pub var tolerance: float
    pub var probability: float

    pub fn init(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width: float,
        centre_x: float,
        centre_y: float,
        angle: float,
        dx: float,
        dy: float,
        tolerance: float,
        probability: float
    ) {
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.width = width
        self.centre_x = centre_x
        self.centre_y = centre_y
        self.angle = angle
        self.dx = dx
        self.dy = dy
        self.tolerance = tolerance
        self.probability = probability
    }

    pub fn copy(self) -> Support {
        return Support(
            self.x1,
            self.y1,
            self.x2,
            self.y2,
            self.width,
            self.centre_x,
            self.centre_y,
            self.angle,
            self.dx,
            self.dy,
            self.tolerance,
            self.probability
        )
    }

    #: Length of the segment along the rectangle's axis.
    pub fn length(self) -> float {
        return math.sqrt(
            (self.x2 - self.x1) * (self.x2 - self.x1) +
                (self.y2 - self.y1) * (self.y2 - self.y1)
        )
    }
}

#: Smallest turn from `b` to `a`, in `[0, pi]`.
fn angle_distance(a: float, b: float) -> float {
    return math.fabs(signed_angle_distance(a, b))
}

#: The same turn, keeping which way it goes.
fn signed_angle_distance(a: float, b: float) -> float {
    var difference = a - b
    while difference <= -math.PI { difference += TWO_PI }
    while difference > math.PI { difference -= TWO_PI }
    return difference
}

let LANCZOS_Q = [
    75122.6331530,
    80916.6278952,
    36308.2951477,
    8687.24529705,
    601.642200279,
    21.2699512244,
    -0.0011407480,
]

#: `log(Gamma(x))` by Lanczos' series, which is the accurate one below 15.
fn log_gamma_lanczos(x: float) -> float {
    var a = (x + 0.5) * math.log(x + 5.5) - (x + 5.5)
    var b = 0.0
    for n in 0..7 {
        a -= math.log(x + float(n))
        b += LANCZOS_Q[n] * x ** float(n)
    }
    return a + math.log(b)
}

#: `log(Gamma(x))` by Windschitl's approximation, which is the accurate one
#: above 15 and much the cheaper of the two.
fn log_gamma_windschitl(x: float) -> float {
    return 0.918938533204673 +
        (x - 0.5) * math.log(x) - x +
        0.5 * x * math.log(x * math.sinh(1.0 / x) + 1.0 / (810.0 * x ** 6.0))
}

fn log_gamma(x: float) -> float {
    return if x > 15.0 { log_gamma_windschitl(x) } else { log_gamma_lanczos(x) }
}

#: `-log10(NFA)` for `aligned` alignments among `total` points.
#:
#: NFA is the number of false alarms: the expected number of rectangles as
#: aligned as this one in an image of pure noise, which is the number of
#: rectangles that could be drawn times the binomial tail
#: `B(total, aligned, probability)`. A rectangle is a detection when that
#: expectation is below `eps`, and returning the negative logarithm turns the
#: test into a comparison against `log_eps`.
#:
#: The tail is summed term by term from `aligned` upwards rather than by a
#: library call, because the terms fall off geometrically once the ratio drops
#: below one and the remainder can then be bounded and dropped.
fn number_of_false_alarms(total: int, aligned: int, probability: float, log_nt: float) -> float {
    if total == 0 or aligned == 0 { return -log_nt }
    if total == aligned { return -float(total) * math.log10(probability) - log_nt }

    let ratio = probability / (1.0 - probability)
    let picks = float(aligned)
    let rest = float(total - aligned)
    let choose = log_gamma(float(total) + 1.0) - log_gamma(picks + 1.0) - log_gamma(rest + 1.0)
    let log_first = choose + picks * math.log(probability) + rest * math.log(1.0 - probability)
    var term = math.exp(log_first)
    #: The first term underflowed, so every later one has too. Which side of
    #: the mean it underflowed on decides whether the tail is negligible or
    #: whether the first term is the whole of it.
    if term < 1e-300 {
        if picks > float(total) * probability { return -log_first / math.LN_10 - log_nt }
        return -log_nt
    }

    var tail = term
    for index in aligned + 1..=total {
        let binomial_ratio = float(total - index + 1) / float(index)
        let step = binomial_ratio * ratio
        term *= step
        tail += term
        if binomial_ratio >= 1.0 { continue }
        #: Past here the ratio only shrinks, so what is left of the tail is
        #: under a geometric series and can be bounded outright.
        let error = term * ((1.0 - step ** float(total - index + 1)) / (1.0 - step) - 1.0)
        if error < 1e-10 * tail { break }
    }
    return -math.log10(tail) - log_nt
}

#: One image's level-line field, and the region growing and validation that run
#: over it. One `Scan` belongs to one call of `detect`: it owns the `used` map
#: that stops two regions claiming the same pixel.
class Scan {
    let width: int
    let height: int
    var modgrad: list[float]
    var largest_gradient: float
    pub var angles: list[float]
    pub var used: list[bool]
    #: `log10` of the number of rectangles the NFA is measured against: every
    #: pair of endpoints on the pixel grid, every width, and the variations the
    #: refinement may try.
    pub let log_nt: float

    pub fn init(self, values: list[float], width: int, height: int, threshold: float) {
        self.width = width
        self.height = height
        let count = width * height
        self.angles = [NOT_DEFINED for _index in 0..count]
        self.modgrad = [0.0 for _index in 0..count]
        self.used = [false for _index in 0..count]
        self.largest_gradient = -1.0
        self.log_nt = 5.0 * (math.log10(float(width)) + math.log10(float(height))) / 2.0 +
            math.log10(VARIATIONS)

        #: The 2x2 window sits between four pixels, so the last row and the
        #: last column keep the undefined angle they were built with.
        for y in 0..height - 1 {
            let row = y * width
            let below = row + width
            for x in 0..width - 1 {
                let down = values[below + x + 1] - values[row + x]
                let across = values[row + x + 1] - values[below + x]
                let gx = down + across
                let gy = down - across
                let norm = math.sqrt((gx * gx + gy * gy) / 4.0)
                self.modgrad[row + x] = norm
                if norm <= threshold { continue }
                #: The level line runs across the gradient, which is what the
                #: swapped and negated arguments say.
                self.angles[row + x] = math.atan2(gx, -gy)
                if norm > self.largest_gradient { self.largest_gradient = norm }
            }
        }
    }

    #: Pixels to try as seeds, strongest gradient first.
    #:
    #: Sorting outright would be the obvious thing and the paper explicitly
    #: does not: the order only has to be good enough that a region starts at
    #: its strongest pixel, and `n_bins` buckets gets that in one pass.
    pub fn seed_order(self, n_bins: int) -> list[int] {
        var bins: list[list[int]] = [[] for _index in 0..n_bins]
        let coefficient = if self.largest_gradient > 0.0 {
            float(n_bins - 1) / self.largest_gradient
        } else {
            0.0
        }
        for y in 0..self.height - 1 {
            let row = y * self.width
            for x in 0..self.width - 1 {
                let index = row + x
                bins[min(int(self.modgrad[index] * coefficient), n_bins - 1)].push(index)
            }
        }
        var order: list[int] = []
        for step in 0..n_bins {
            for index in bins[n_bins - 1 - step] { order.push(index) }
        }
        return order
    }

    #: Whether the level line at `index` runs within `tolerance` of `angle`.
    #: The caller has already established that `index` is inside the image.
    fn aligned(self, index: int, angle: float, tolerance: float) -> bool {
        let level = self.angles[index]
        if level == NOT_DEFINED { return false }
        var difference = angle - level
        if difference < 0.0 { difference = -difference }
        if difference > THREE_HALF_PI {
            difference -= TWO_PI
            if difference < 0.0 { difference = -difference }
        }
        return difference <= tolerance
    }

    #: Grow a region out from `seed`, taking every neighbour whose level line
    #: is within `tolerance` of the region's running angle.
    #:
    #: The running angle is the direction of the sum of the unit vectors, not
    #: the mean of the angles: angles wrap and their mean does not survive it.
    pub fn grow(self, seed: int, tolerance: float) -> tuple[list[int], float] {
        var region = [seed]
        var region_angle = self.angles[seed]
        self.used[seed] = true
        var sum_dx = math.cos(region_angle)
        var sum_dy = math.sin(region_angle)
        var at = 0
        while at < region.len() {
            let index = region[at]
            at += 1
            let y = index // self.width
            let x = index - y * self.width
            for ny in max(y - 1, 0)..=min(y + 1, self.height - 1) {
                let row = ny * self.width
                for nx in max(x - 1, 0)..=min(x + 1, self.width - 1) {
                    let neighbour = row + nx
                    if self.used[neighbour] { continue }
                    if not self.aligned(neighbour, region_angle, tolerance) { continue }
                    self.used[neighbour] = true
                    region.push(neighbour)
                    sum_dx += math.cos(self.angles[neighbour])
                    sum_dy += math.sin(self.angles[neighbour])
                    region_angle = math.atan2(sum_dy, sum_dx)
                }
            }
        }
        return (region, region_angle)
    }

    #: The region's principal direction, weighted by gradient magnitude.
    #:
    #: This is the eigenvector of the inertia matrix for the smaller
    #: eigenvalue, which is the direction the region is thinnest across and so
    #: the direction it runs along. The eigenvector fixes the axis and not
    #: which way along it, so the region's own angle decides that.
    fn principal_angle(
        self,
        region: list[int],
        centre_x: float,
        centre_y: float,
        region_angle: float,
        tolerance: float
    ) -> float {
        var ixx = 0.0
        var iyy = 0.0
        var ixy = 0.0
        for index in region {
            let weight = self.modgrad[index]
            let y = index // self.width
            let dx = float(index - y * self.width) - centre_x
            let dy = float(y) - centre_y
            ixx += dy * dy * weight
            iyy += dx * dx * weight
            ixy -= dx * dy * weight
        }
        let smaller = 0.5 * (ixx + iyy - math.sqrt((ixx - iyy) * (ixx - iyy) + 4.0 * ixy * ixy))
        #: Take the row of `I - lambda` that is further from zero, or a region
        #: lying along one of the axes divides by nothing.
        var angle = if math.fabs(ixx) > math.fabs(iyy) {
            math.atan2(smaller - ixx, ixy)
        } else {
            math.atan2(ixy, smaller - iyy)
        }
        if angle_distance(angle, region_angle) > tolerance { angle += math.PI }
        return angle
    }

    #: Fit a rectangle to `region`.
    pub fn fit(
        self,
        region: list[int],
        region_angle: float,
        tolerance: float,
        probability: float
    ) -> Support {
        var centre_x = 0.0
        var centre_y = 0.0
        var total = 0.0
        for index in region {
            let weight = self.modgrad[index]
            let y = index // self.width
            centre_x += float(index - y * self.width) * weight
            centre_y += float(y) * weight
            total += weight
        }
        if total <= 0.0 { throw ValueError("a region with no gradient has no rectangle") }
        centre_x /= total
        centre_y /= total

        let angle = self.principal_angle(region, centre_x, centre_y, region_angle, tolerance)
        let dx = math.cos(angle)
        let dy = math.sin(angle)
        var along_min = 0.0
        var along_max = 0.0
        var across_min = 0.0
        var across_max = 0.0
        for index in region {
            let y = index // self.width
            let offset_x = float(index - y * self.width) - centre_x
            let offset_y = float(y) - centre_y
            let along = offset_x * dx + offset_y * dy
            let across = offset_y * dx - offset_x * dy
            if along > along_max {
                along_max = along
            } elif along < along_min {
                along_min = along
            }
            if across > across_max {
                across_max = across
            } elif across < across_min {
                across_min = across
            }
        }
        return Support(
            centre_x + along_min * dx,
            centre_y + along_min * dy,
            centre_x + along_max * dx,
            centre_y + along_max * dy,
            max(across_max - across_min, 1.0),
            centre_x,
            centre_y,
            angle,
            dx,
            dy,
            tolerance,
            probability
        )
    }

    #: `-log10(NFA)` for the rectangle: how surprising its aligned points are.
    #:
    #: The rectangle is swept column by column, taking the pixels between where
    #: the vertical line at each `x` enters and leaves it. Points outside the
    #: image are not counted at all, neither as aligned nor as total, which is
    #: what keeps a rectangle that runs off the edge from being penalised for
    #: the part that is not there.
    pub fn rect_nfa(self, rect: Support) -> float {
        let half = rect.width * 0.5
        let corner_x = [
            rect.x1 - rect.dy * half,
            rect.x2 - rect.dy * half,
            rect.x2 + rect.dy * half,
            rect.x1 + rect.dy * half,
        ]
        let corner_y = [
            rect.y1 + rect.dx * half,
            rect.y2 + rect.dx * half,
            rect.y2 - rect.dx * half,
            rect.y1 - rect.dx * half,
        ]
        var leftmost = corner_x[0]
        var rightmost = corner_x[0]
        for value in corner_x {
            leftmost = min(leftmost, value)
            rightmost = max(rightmost, value)
        }

        var total = 0
        var aligned = 0
        var x = int(math.ceil(leftmost))
        while float(x) <= rightmost {
            var low = 0.0
            var high = 0.0
            var met = false
            for corner in 0..4 {
                let next = if corner == 3 { 0 } else { corner + 1 }
                let ax = corner_x[corner]
                let ay = corner_y[corner]
                let bx = corner_x[next]
                let by = corner_y[next]
                if float(x) < min(ax, bx) or float(x) > max(ax, bx) { continue }
                #: A vertical side is met along its whole length, and it is
                #: also the side whose crossing cannot be found by dividing.
                let vertical = math.fabs(bx - ax) < 1e-12
                let entry = if vertical {
                    min(ay, by)
                } else {
                    ay + (float(x) - ax) * (by - ay) / (bx - ax)
                }
                let leaving = if vertical { max(ay, by) } else { entry }
                if not met {
                    low = entry
                    high = leaving
                    met = true
                } else {
                    low = min(low, entry)
                    high = max(high, leaving)
                }
            }
            if met and x >= 0 and x < self.width {
                var y = max(int(math.ceil(low)), 0)
                let last = min(int(math.floor(high)), self.height - 1)
                while y <= last {
                    total += 1
                    if self.aligned(y * self.width + x, rect.angle, rect.tolerance) { aligned += 1 }
                    y += 1
                }
            }
            x += 1
        }
        return number_of_false_alarms(total, aligned, rect.probability, self.log_nt)
    }

    #: Region points per pixel of the rectangle's area.
    fn density(self, region: list[int], rect: Support) -> float {
        let area = rect.length() * rect.width
        if area <= 0.0 { return 0.0 }
        return float(region.len()) / area
    }

    #: Cut a region down until its rectangle is densely filled.
    #:
    #: A sparse rectangle is the signature of two lines meeting at a shallow
    #: angle: region growing walks round the corner, and the rectangle fitted
    #: to the result is a wide box containing neither line. Regrowing with the
    #: tolerance the region's own angles suggest usually keeps only one of
    #: them; when it does not, the region is trimmed back towards its seed.
    pub fn refine_region(
        self,
        region: list[int],
        tolerance: float,
        probability: float,
        rect: Support,
        density_th: float
    ) -> tuple[bool, Support] {
        if self.density(region, rect) >= density_th { return (true, rect) }

        let seed = region[0]
        let seed_y = seed // self.width
        let seed_x = seed - seed_y * self.width
        let seed_angle = self.angles[seed]
        var sum = 0.0
        var square_sum = 0.0
        var counted = 0
        for index in region {
            self.used[index] = false
            let y = index // self.width
            let dx = float(index - y * self.width - seed_x)
            let dy = float(y - seed_y)
            if math.sqrt(dx * dx + dy * dy) >= rect.width { continue }
            let difference = signed_angle_distance(self.angles[index], seed_angle)
            sum += difference
            square_sum += difference * difference
            counted += 1
        }
        #: The seed itself is always within the rectangle's own width of
        #: itself, so there is always something to average.
        let mean = sum / float(counted)
        let tau = 2.0 * math.sqrt((square_sum - 2.0 * mean * sum) / float(counted) + mean * mean)

        let (regrown, regrown_angle) = self.grow(seed, tau)
        if regrown.len() < 2 { return (false, rect) }
        var tightened = self.fit(regrown, regrown_angle, tolerance, probability)
        if self.density(regrown, tightened) >= density_th { return (true, tightened) }
        return self.shrink(regrown, regrown_angle, tolerance, probability, tightened, density_th)
    }

    #: Drop the points furthest from the seed, a quarter of the radius at a
    #: time, until what is left fills its rectangle.
    fn shrink(
        self,
        region: list[int],
        region_angle: float,
        tolerance: float,
        probability: float,
        rect: Support,
        density_th: float
    ) -> tuple[bool, Support] {
        var current = rect
        let seed = region[0]
        let seed_y = seed // self.width
        let seed_x = float(seed - seed_y * self.width)
        let from_start = (seed_x - current.x1) * (seed_x - current.x1) +
            (float(seed_y) - current.y1) * (float(seed_y) - current.y1)
        let from_end = (seed_x - current.x2) * (seed_x - current.x2) +
            (float(seed_y) - current.y2) * (float(seed_y) - current.y2)
        var radius_squared = max(from_start, from_end)

        while self.density(region, current) < density_th {
            radius_squared *= 0.5625
            var at = 0
            while at < region.len() {
                let index = region[at]
                let y = index // self.width
                let dx = float(index - y * self.width) - seed_x
                let dy = float(y - seed_y)
                if dx * dx + dy * dy <= radius_squared {
                    at += 1
                    continue
                }
                self.used[index] = false
                region[at] = region[region.len() - 1]
                region.pop()
            }
            if region.len() < 2 { return (false, current) }
            current = self.fit(region, region_angle, tolerance, probability)
        }
        return (true, current)
    }

    #: Try variations of a rectangle that failed the NFA test, keeping the best
    #: one found.
    #:
    #: Each family is tried from the best rectangle so far, and the order
    #: matters: tightening the angle tolerance is what rescues a segment whose
    #: region wandered, and narrowing is what rescues one whose rectangle
    #: caught a second edge alongside it.
    pub fn improve(self, rect: Support, log_eps: float) -> tuple[float, Support] {
        let delta = 0.5
        var best = rect
        var log_nfa = self.rect_nfa(best)
        if log_nfa > log_eps { return (log_nfa, best) }

        var trial = best.copy()
        for _step in 0..5 {
            trial.probability /= 2.0
            trial.tolerance = trial.probability * math.PI
            let value = self.rect_nfa(trial)
            if value > log_nfa {
                log_nfa = value
                best = trial.copy()
            }
        }
        if log_nfa > log_eps { return (log_nfa, best) }

        trial = best.copy()
        for _step in 0..5 {
            if trial.width - delta < 0.5 { continue }
            trial.width -= delta
            let value = self.rect_nfa(trial)
            if value > log_nfa {
                log_nfa = value
                best = trial.copy()
            }
        }
        if log_nfa > log_eps { return (log_nfa, best) }

        for side in 0..2 {
            trial = best.copy()
            let way = if side == 0 { 1.0 } else { -1.0 }
            let step_x = -trial.dy * delta * 0.5 * way
            let step_y = trial.dx * delta * 0.5 * way
            for _step in 0..5 {
                if trial.width - delta < 0.5 { continue }
                trial.x1 += step_x
                trial.y1 += step_y
                trial.x2 += step_x
                trial.y2 += step_y
                trial.width -= delta
                let value = self.rect_nfa(trial)
                if value > log_nfa {
                    log_nfa = value
                    best = trial.copy()
                }
            }
            if log_nfa > log_eps { return (log_nfa, best) }
        }

        trial = best.copy()
        for _step in 0..5 {
            if trial.width - delta < 0.5 { continue }
            trial.probability /= 2.0
            trial.tolerance = trial.probability * math.PI
            let value = self.rect_nfa(trial)
            if value > log_nfa {
                log_nfa = value
                best = trial.copy()
            }
        }
        return (log_nfa, best)
    }
}

#: LSD, with the parameters the paper settled on as its defaults.
pub class LineSegmentDetector {
    #: How far a rectangle is worked on before it is reported.
    pub let refine: int
    #: The fraction of the original size the detector works at.
    pub let scale: float
    #: Width of the Gaussian used before downscaling, as a multiple of the
    #: sampling step.
    pub let sigma_scale: float
    #: Bound on the error in the gradient, in grey levels.
    pub let quant: float
    #: How far a level line may lie from the region's angle, in degrees.
    pub let ang_th: float
    #: A rectangle is a detection when `-log10(NFA)` exceeds this.
    pub let log_eps: float
    #: Least share of a rectangle its aligned points have to fill.
    pub let density_th: float
    #: Buckets used to order the seeds by gradient magnitude.
    pub let n_bins: int

    pub fn init(
        self,
        refine: int = LSD_REFINE_STD,
        scale: float = 0.8,
        sigma_scale: float = 0.6,
        quant: float = 2.0,
        ang_th: float = 22.5,
        log_eps: float = 0.0,
        density_th: float = 0.7,
        n_bins: int = 1024
    ) {
        if refine < LSD_REFINE_NONE or refine > LSD_REFINE_ADV {
            throw ValueError(f"refine must be one of 0, 1, 2, got {refine}")
        }
        if scale <= 0.0 { throw ValueError(f"scale must be positive, got {scale}") }
        if sigma_scale <= 0.0 {
            throw ValueError(f"sigma_scale must be positive, got {sigma_scale}")
        }
        if quant < 0.0 { throw ValueError(f"quant cannot be negative, got {quant}") }
        if ang_th <= 0.0 or ang_th >= 180.0 {
            throw ValueError(f"ang_th must be in (0, 180), got {ang_th}")
        }
        if density_th < 0.0 or density_th >= 1.0 {
            throw ValueError(f"density_th must be in [0, 1), got {density_th}")
        }
        if n_bins <= 0 { throw ValueError(f"n_bins must be positive, got {n_bins}") }
        self.refine = refine
        self.scale = scale
        self.sigma_scale = sigma_scale
        self.quant = quant
        self.ang_th = ang_th
        self.log_eps = log_eps
        self.density_th = density_th
        self.n_bins = n_bins
    }

    #: Blur and downscale, which is what makes the detector insensitive to the
    #: staircase a rasteriser leaves along a shallow line.
    fn downscale(self, image: Mat) -> Mat {
        if self.scale == 1.0 { return image }
        let sigma = if self.scale < 1.0 { self.sigma_scale / self.scale } else { self.sigma_scale }
        #: Cut the kernel where the Gaussian has fallen to a thousandth.
        let radius = int(math.ceil(sigma * math.sqrt(6.0 * math.log(10.0))))
        let size = Size(1 + 2 * radius, 1 + 2 * radius)
        return resize(gaussian_blur(image, size, sigma), null, self.scale, self.scale)
    }

    #: Find the line segments in `image`.
    #:
    #: Returns the segments as `[x1, y1, x2, y2]` in the coordinates of the
    #: image as given, alongside each one's rectangle width, the angle
    #: tolerance it was accepted at, and its `-log10(NFA)`.
    pub fn detect(
        self,
        image: Mat
    ) -> tuple[list[list[float]], list[float], list[float], list[float]] {
        if image.channels() != 1 { throw ValueError("detect needs a single-channel Mat") }
        var lines: list[list[float]] = []
        var widths: list[float] = []
        var precisions: list[float] = []
        var nfas: list[float] = []
        if image.rows < 2 or image.cols < 2 { return (lines, widths, precisions, nfas) }

        let scaled = self.downscale(image)
        if scaled.rows < 2 or scaled.cols < 2 { return (lines, widths, precisions, nfas) }

        let tolerance = math.PI * self.ang_th / 180.0
        let probability = self.ang_th / 180.0
        #: A gradient this weak could point anywhere once the quantisation
        #: error is allowed for, so it is not allowed to point at all.
        let threshold = self.quant / math.sin(tolerance)
        var scan = Scan(scaled.values(), scaled.cols, scaled.rows, threshold)
        #: Below this many points even a perfectly aligned region is not
        #: unlikely enough to be worth testing.
        let smallest_region = int(-scan.log_nt / math.log10(probability))
        #: The gradient sits between four pixels, so everything measured on it
        #: is half a pixel up and to the left of the image it came from, and
        #: it is measured on the downscaled image at that.
        let recover = 1.0 / self.scale

        for seed in scan.seed_order(self.n_bins) {
            if scan.used[seed] { continue }
            if scan.angles[seed] == NOT_DEFINED { continue }
            let (region, region_angle) = scan.grow(seed, tolerance)
            if region.len() < smallest_region { continue }
            var rect = scan.fit(region, region_angle, tolerance, probability)
            if self.refine >= LSD_REFINE_STD {
                let (kept, tightened) = scan.refine_region(
                    region,
                    tolerance,
                    probability,
                    rect,
                    self.density_th
                )
                if not kept { continue }
                rect = tightened
            }
            var log_nfa = 0.0
            if self.refine >= LSD_REFINE_ADV {
                let (improved_nfa, improved) = scan.improve(rect, self.log_eps)
                log_nfa = improved_nfa
                rect = improved
            } else {
                log_nfa = scan.rect_nfa(rect)
            }
            if log_nfa <= self.log_eps { continue }

            lines.push(
                [
                    (rect.x1 + 0.5) * recover,
                    (rect.y1 + 0.5) * recover,
                    (rect.x2 + 0.5) * recover,
                    (rect.y2 + 0.5) * recover,
                ]
            )
            widths.push(rect.width * recover)
            precisions.push(rect.probability)
            nfas.push(log_nfa)
        }
        return (lines, widths, precisions, nfas)
    }

    #: Draw `lines` in red over a grey copy of `image`, as OpenCV's
    #: `drawSegments` does. OpenCV replaces the image it is given; this returns
    #: the drawing and leaves the original alone.
    pub fn draw_segments(self, image: Mat, lines: list[list[float]]) -> Mat {
        if image.channels() != 1 and image.channels() != 3 {
            throw ValueError("draw_segments needs a one- or three-channel Mat")
        }
        let grey = if image.channels() == 1 { image } else { cvt_color(image, COLOR_BGR2GRAY) }
        var canvas = cvt_color(grey, COLOR_GRAY2BGR)
        for segment in lines {
            line(canvas, start_of(segment), end_of(segment), Scalar(0.0, 0.0, 255.0), 1)
        }
        return canvas
    }

    #: Draw two sets of segments over each other and count where they disagree.
    #:
    #: `lines1` goes in blue, `lines2` in red, and a pixel both of them cover
    #: comes out magenta. The count is of the pixels exactly one of them
    #: covers, which is the number to watch when comparing two detectors or two
    #: sets of parameters.
    pub fn compare_segments(
        self,
        size: Size,
        lines1: list[list[float]],
        lines2: list[list[float]],
        image: Mat
    ) -> tuple[int, Mat] {
        let extent = if image.empty() { size } else { image.size() }
        if extent.empty() {
            throw ValueError(f"compare_segments needs a size with area, got {extent}")
        }
        var canvas = if image.empty() {
            Mat.zeros(extent.height, extent.width, CV_8UC3)
        } else {
            image
        }
        if canvas.channels() != 3 {
            throw ValueError("compare_segments draws into a three-channel Mat")
        }

        var first = Mat.zeros(extent.height, extent.width, CV_8UC1)
        var second = Mat.zeros(extent.height, extent.width, CV_8UC1)
        for segment in lines1 { line(first, start_of(segment), end_of(segment), Scalar(255.0), 1) }
        for segment in lines2 { line(second, start_of(segment), end_of(segment), Scalar(255.0), 1) }

        let ink1 = first.values()
        let ink2 = second.values()
        var pixels = canvas.edit()
        var differing = 0
        for index in 0..extent.width * extent.height {
            let in_first = ink1[index] != 0.0
            let in_second = ink2[index] != 0.0
            if in_first != in_second { differing += 1 }
            if not in_first and not in_second { continue }
            pixels[index * 3] = if in_first { 255.0 } else { 0.0 }
            pixels[index * 3 + 1] = 0.0
            pixels[index * 3 + 2] = if in_second { 255.0 } else { 0.0 }
        }
        canvas.flush()
        return (differing, canvas)
    }
}

#: Where a segment starts, rounded to the pixel `line` draws to.
fn start_of(segment: list[float]) -> Point { return end_of_segment(segment, 0) }

#: And where it stops.
fn end_of(segment: list[float]) -> Point { return end_of_segment(segment, 2) }

fn end_of_segment(segment: list[float], at: int) -> Point {
    if segment.len() != 4 {
        throw ValueError(f"a segment is four floats, got {segment.len()}")
    }
    return Point2f(segment[at], segment[at + 1]).to_point()
}

#: Build a detector, as `cv::createLineSegmentDetector` does.
pub fn create_line_segment_detector(
    refine: int = LSD_REFINE_STD,
    scale: float = 0.8,
    sigma_scale: float = 0.6,
    quant: float = 2.0,
    ang_th: float = 22.5,
    log_eps: float = 0.0,
    density_th: float = 0.7,
    n_bins: int = 1024
) -> LineSegmentDetector {
    return LineSegmentDetector(
        refine,
        scale,
        sigma_scale,
        quant,
        ang_th,
        log_eps,
        density_th,
        n_bins
    )
}

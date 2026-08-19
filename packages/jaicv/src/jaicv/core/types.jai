#: The small value types OpenCV passes everywhere: points, sizes, rectangles,
#: scalars, and the flag constants that name a border mode or an interpolation.
#:
#: Names and field order follow OpenCV so a program reads the same in both.
import std.math
from std.core import Printable

#: Element depth: unsigned 8-bit, the type a camera frame arrives in.
pub const CV_8U = 0
#: Element depth: signed 8-bit.
pub const CV_8S = 1
#: Element depth: unsigned 16-bit.
pub const CV_16U = 2
#: Element depth: signed 16-bit.
pub const CV_16S = 3
#: Element depth: signed 32-bit.
pub const CV_32S = 4
#: Element depth: 32-bit float.
pub const CV_32F = 5
#: Element depth: 64-bit float. Storage is 32-bit, so this is an alias that
#: keeps source compatible rather than a wider element.
pub const CV_64F = 6

const _DEPTH_MASK = 7
const _CHANNEL_SHIFT = 3

#: Combine a depth and a channel count into a type code, as `CV_MAKETYPE` does.
pub fn CV_MAKETYPE(depth: int, channels: int) -> int {
    if channels < 1 or channels > 512 {
        throw ValueError(f"channel count must be 1..512, got {channels}")
    }
    return (depth & _DEPTH_MASK) + (channels - 1 << _CHANNEL_SHIFT)
}

#: The depth half of a type code.
pub fn CV_MAT_DEPTH(type_code: int) -> int { return type_code & _DEPTH_MASK }

#: The channel-count half of a type code.
pub fn CV_MAT_CN(type_code: int) -> int { return (type_code >> _CHANNEL_SHIFT) + 1 }

pub let CV_8UC1 = CV_MAKETYPE(CV_8U, 1)
pub let CV_8UC2 = CV_MAKETYPE(CV_8U, 2)
pub let CV_8UC3 = CV_MAKETYPE(CV_8U, 3)
pub let CV_8UC4 = CV_MAKETYPE(CV_8U, 4)
pub let CV_8SC1 = CV_MAKETYPE(CV_8S, 1)
pub let CV_8SC3 = CV_MAKETYPE(CV_8S, 3)
pub let CV_16UC1 = CV_MAKETYPE(CV_16U, 1)
pub let CV_16UC3 = CV_MAKETYPE(CV_16U, 3)
pub let CV_16SC1 = CV_MAKETYPE(CV_16S, 1)
pub let CV_16SC3 = CV_MAKETYPE(CV_16S, 3)
pub let CV_32SC1 = CV_MAKETYPE(CV_32S, 1)
pub let CV_32SC2 = CV_MAKETYPE(CV_32S, 2)
pub let CV_32SC3 = CV_MAKETYPE(CV_32S, 3)
pub let CV_32SC4 = CV_MAKETYPE(CV_32S, 4)
pub let CV_32FC1 = CV_MAKETYPE(CV_32F, 1)
pub let CV_32FC2 = CV_MAKETYPE(CV_32F, 2)
pub let CV_32FC3 = CV_MAKETYPE(CV_32F, 3)
pub let CV_32FC4 = CV_MAKETYPE(CV_32F, 4)
pub let CV_64FC1 = CV_MAKETYPE(CV_64F, 1)
pub let CV_64FC3 = CV_MAKETYPE(CV_64F, 3)

#: Lowest value a depth can hold, for saturating casts.
pub fn depth_min(depth: int) -> float {
    if depth == CV_8S { return -128.0 }
    if depth == CV_16S { return -32768.0 }
    if depth == CV_32S { return -2147483648.0 }
    return 0.0
}

#: Highest value a depth can hold, for saturating casts.
pub fn depth_max(depth: int) -> float {
    if depth == CV_8U { return 255.0 }
    if depth == CV_8S { return 127.0 }
    if depth == CV_16U { return 65535.0 }
    if depth == CV_16S { return 32767.0 }
    if depth == CV_32S { return 2147483647.0 }
    return 0.0
}

#: Whether a depth stores whole numbers, and therefore rounds and saturates.
pub fn depth_is_integer(depth: int) -> bool { return depth <= CV_32S }

#: Human-readable type name, as `cv::typeToString` produces.
pub fn type_to_string(type_code: int) -> str {
    let names = ["8U", "8S", "16U", "16S", "32S", "32F", "64F"]
    let depth = CV_MAT_DEPTH(type_code)
    let name = if depth < names.len() { names[depth] } else { f"?{depth}" }
    return f"CV_{name}C{CV_MAT_CN(type_code)}"
}

#: A point with integer coordinates.
pub class Point: Printable {
    pub var x: int
    pub var y: int

    pub fn init(self, x: int = 0, y: int = 0) {
        self.x = x
        self.y = y
    }

    pub fn __add__(self, other: Point) -> Point { return Point(self.x + other.x, self.y + other.y) }

    pub fn __sub__(self, other: Point) -> Point { return Point(self.x - other.x, self.y - other.y) }

    pub fn __eq__(self, other: any) -> bool {
        if not other is Point { return false }
        return self.x == other.x and self.y == other.y
    }

    #: Dot product with `other`.
    pub fn dot(self, other: Point) -> int { return self.x * other.x + self.y * other.y }

    #: Human-readable rendering.
    pub fn to_str(self) -> str { return f"[{self.x}, {self.y}]" }

    fn __str__(self) -> str { return self.to_str() }
}

#: A point with floating-point coordinates.
pub class Point2f: Printable {
    pub var x: float
    pub var y: float

    pub fn init(self, x: float = 0.0, y: float = 0.0) {
        self.x = x
        self.y = y
    }

    pub fn __add__(self, other: Point2f) -> Point2f {
        return Point2f(self.x + other.x, self.y + other.y)
    }

    pub fn __sub__(self, other: Point2f) -> Point2f {
        return Point2f(self.x - other.x, self.y - other.y)
    }

    pub fn __mul__(self, factor: float) -> Point2f {
        return Point2f(self.x * factor, self.y * factor)
    }

    pub fn __eq__(self, other: any) -> bool {
        if not other is Point2f { return false }
        return self.x == other.x and self.y == other.y
    }

    #: Dot product with `other`.
    pub fn dot(self, other: Point2f) -> float { return self.x * other.x + self.y * other.y }

    #: Distance from the origin.
    pub fn norm(self) -> float { return math.sqrt(self.x * self.x + self.y * self.y) }

    #: Round to integer coordinates.
    pub fn to_point(self) -> Point {
        return Point(int(math.floor(self.x + 0.5)), int(math.floor(self.y + 0.5)))
    }

    #: Human-readable rendering.
    pub fn to_str(self) -> str { return f"[{self.x}, {self.y}]" }

    fn __str__(self) -> str { return self.to_str() }
}

#: A point in three dimensions.
pub class Point3f: Printable {
    pub var x: float
    pub var y: float
    pub var z: float

    pub fn init(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) {
        self.x = x
        self.y = y
        self.z = z
    }

    pub fn __add__(self, other: Point3f) -> Point3f {
        return Point3f(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    pub fn __sub__(self, other: Point3f) -> Point3f {
        return Point3f(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    #: Dot product with `other`.
    pub fn dot(self, other: Point3f) -> float {
        return self.x * other.x + self.y * other.y + self.z * other.z
    }

    #: Cross product with `other`.
    pub fn cross(self, other: Point3f) -> Point3f {
        return Point3f(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    }

    #: Human-readable rendering.
    pub fn to_str(self) -> str { return f"[{self.x}, {self.y}, {self.z}]" }

    fn __str__(self) -> str { return self.to_str() }
}

#: Width and height, in that order — the opposite order from `Mat`'s rows and
#: columns, exactly as in OpenCV.
pub class Size: Printable {
    pub var width: int
    pub var height: int

    pub fn init(self, width: int = 0, height: int = 0) {
        self.width = width
        self.height = height
    }

    #: Number of pixels the size covers.
    pub fn area(self) -> int { return self.width * self.height }

    #: Whether either extent is zero or negative.
    pub fn empty(self) -> bool { return self.width <= 0 or self.height <= 0 }

    pub fn __eq__(self, other: any) -> bool {
        if not other is Size { return false }
        return self.width == other.width and self.height == other.height
    }

    #: Human-readable rendering.
    pub fn to_str(self) -> str { return f"[{self.width} x {self.height}]" }

    fn __str__(self) -> str { return self.to_str() }
}

#: An axis-aligned rectangle: top-left corner plus extent.
pub class Rect: Printable {
    pub var x: int
    pub var y: int
    pub var width: int
    pub var height: int

    pub fn init(self, x: int = 0, y: int = 0, width: int = 0, height: int = 0) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    }

    #: Top-left corner.
    pub fn tl(self) -> Point { return Point(self.x, self.y) }

    #: Bottom-right corner, one past the last pixel on each axis.
    pub fn br(self) -> Point { return Point(self.x + self.width, self.y + self.height) }

    #: Extent as a `Size`.
    pub fn size(self) -> Size { return Size(self.width, self.height) }

    #: Number of pixels covered.
    pub fn area(self) -> int { return self.width * self.height }

    #: Whether either extent is zero or negative.
    pub fn empty(self) -> bool { return self.width <= 0 or self.height <= 0 }

    #: Whether `point` falls inside the rectangle.
    pub fn contains(self, point: Point) -> bool {
        return point.x >= self.x and
            point.x < self.x + self.width and
            point.y >= self.y and
            point.y < self.y + self.height
    }

    #: Overlap with `other`, empty when they do not meet.
    pub fn intersect(self, other: Rect) -> Rect {
        let left = max(self.x, other.x)
        let top = max(self.y, other.y)
        let right = min(self.x + self.width, other.x + other.width)
        let bottom = min(self.y + self.height, other.y + other.height)
        if right <= left or bottom <= top { return Rect(0, 0, 0, 0) }
        return Rect(left, top, right - left, bottom - top)
    }

    #: Smallest rectangle containing both.
    pub fn union_with(self, other: Rect) -> Rect {
        if self.empty() { return other }
        if other.empty() { return self }
        let left = min(self.x, other.x)
        let top = min(self.y, other.y)
        let right = max(self.x + self.width, other.x + other.width)
        let bottom = max(self.y + self.height, other.y + other.height)
        return Rect(left, top, right - left, bottom - top)
    }

    pub fn __eq__(self, other: any) -> bool {
        if not other is Rect { return false }
        return self.x == other.x and
            self.y == other.y and
            self.width == other.width and
            self.height == other.height
    }

    #: Human-readable rendering.
    pub fn to_str(self) -> str {
        return f"[{self.width} x {self.height} from ({self.x}, {self.y})]"
    }

    fn __str__(self) -> str { return self.to_str() }
}

#: A rectangle that carries a rotation, as `minAreaRect` returns.
pub class RotatedRect: Printable {
    #: Centre of the rectangle.
    pub var center: Point2f
    #: Extent before rotation.
    pub var size: Size2f
    #: Rotation in degrees, clockwise.
    pub var angle: float

    pub fn init(self, center: Point2f, size: Size2f, angle: float) {
        self.center = center
        self.size = size
        self.angle = angle
    }

    #: The four corners, in order, starting from the one nearest the top left.
    pub fn points(self) -> list[Point2f] {
        let radians = self.angle * math.PI / 180.0
        let cosine = math.cos(radians)
        let sine = math.sin(radians)
        let half_w = self.size.width * 0.5
        let half_h = self.size.height * 0.5
        var corners: list[Point2f] = []
        for (dx, dy) in [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)] {
            corners.push(
                Point2f(
                    self.center.x + dx * cosine - dy * sine,
                    self.center.y + dx * sine + dy * cosine
                )
            )
        }
        return corners
    }

    #: Axis-aligned rectangle that contains every corner.
    pub fn bounding_rect(self) -> Rect {
        let corners = self.points()
        var left = corners[0].x
        var top = corners[0].y
        var right = corners[0].x
        var bottom = corners[0].y
        for corner in corners {
            if corner.x < left { left = corner.x }
            if corner.y < top { top = corner.y }
            if corner.x > right { right = corner.x }
            if corner.y > bottom { bottom = corner.y }
        }
        let x = int(math.floor(left))
        let y = int(math.floor(top))
        return Rect(x, y, int(math.ceil(right)) - x + 1, int(math.ceil(bottom)) - y + 1)
    }

    #: Human-readable rendering.
    pub fn to_str(self) -> str {
        return f"[{self.size.width} x {self.size.height} at {self.center} rot {self.angle}]"
    }

    fn __str__(self) -> str { return self.to_str() }
}

#: Width and height with floating-point extent.
pub class Size2f: Printable {
    pub var width: float
    pub var height: float

    pub fn init(self, width: float = 0.0, height: float = 0.0) {
        self.width = width
        self.height = height
    }

    #: Area the size covers.
    pub fn area(self) -> float { return self.width * self.height }

    #: Human-readable rendering.
    pub fn to_str(self) -> str { return f"[{self.width} x {self.height}]" }

    fn __str__(self) -> str { return self.to_str() }
}

#: Up to four channel values, the way OpenCV passes a colour or a constant.
pub class Scalar: Printable {
    pub var v0: float
    pub var v1: float
    pub var v2: float
    pub var v3: float

    pub fn init(self, v0: float = 0.0, v1: float = 0.0, v2: float = 0.0, v3: float = 0.0) {
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
    }

    #: Every channel set to the same value.
    pub static fn all(value: float) -> Scalar { return Scalar(value, value, value, value) }

    #: The four values as a list, lowest channel first.
    pub fn to_list(self) -> list[float] { return [self.v0, self.v1, self.v2, self.v3] }

    pub fn __getitem__(self, index: int) -> float {
        if index == 0 { return self.v0 }
        if index == 1 { return self.v1 }
        if index == 2 { return self.v2 }
        if index == 3 { return self.v3 }
        throw IndexError(f"Scalar index {index} is out of range")
    }

    pub fn __eq__(self, other: any) -> bool {
        if not other is Scalar { return false }
        return self.v0 == other.v0 and
            self.v1 == other.v1 and
            self.v2 == other.v2 and
            self.v3 == other.v3
    }

    #: Human-readable rendering.
    pub fn to_str(self) -> str { return f"[{self.v0}, {self.v1}, {self.v2}, {self.v3}]" }

    fn __str__(self) -> str { return self.to_str() }
}

#: A half-open span of indices, as `cv::Range`.
pub class Range: Printable {
    pub var start: int
    pub var end: int

    pub fn init(self, start: int = 0, end: int = 0) {
        self.start = start
        self.end = end
    }

    #: The whole axis, whatever its length turns out to be.
    pub static fn all() -> Range { return Range(-1, -1) }

    #: Whether this range means "the whole axis".
    pub fn is_all(self) -> bool { return self.start == -1 and self.end == -1 }

    #: Number of indices covered.
    pub fn size(self) -> int { return self.end - self.start }

    #: Whether the range covers nothing.
    pub fn empty(self) -> bool { return self.start >= self.end }

    #: Human-readable rendering.
    pub fn to_str(self) -> str { return f"[{self.start}, {self.end})" }

    fn __str__(self) -> str { return self.to_str() }
}

#: Stop after a count, after an epsilon, or after whichever comes first.
pub class TermCriteria: Printable {
    #: Stop once `max_count` iterations have run.
    pub static let COUNT: int = 1
    #: Stop once the change falls below `epsilon`.
    pub static let EPS: int = 2

    pub var type_: int
    pub var max_count: int
    pub var epsilon: float

    pub fn init(self, type_: int = 3, max_count: int = 30, epsilon: float = 0.01) {
        self.type_ = type_
        self.max_count = max_count
        self.epsilon = epsilon
    }

    #: Human-readable rendering.
    pub fn to_str(self) -> str {
        return f"TermCriteria(type={self.type_}, count={self.max_count}, eps={self.epsilon})"
    }

    fn __str__(self) -> str { return self.to_str() }
}

#: Pixels outside the image are the mirror of the inside, edge repeated:
#: `fedcba|abcdefgh|hgfedcb`.
pub const BORDER_REFLECT_101 = 4
#: Alias for `BORDER_REFLECT_101`, the OpenCV default for most filters.
pub const BORDER_DEFAULT = 4
#: Pixels outside the image take a constant value.
pub const BORDER_CONSTANT = 0
#: The edge pixel is repeated: `aaaaaa|abcdefgh|hhhhhhh`.
pub const BORDER_REPLICATE = 1
#: Mirror including the edge pixel: `fedcba|abcdefgh|hgfedcb` with the edge
#: doubled.
pub const BORDER_REFLECT = 2
#: The image tiles: `cdefgh|abcdefgh|abcdefg`.
pub const BORDER_WRAP = 3
#: Do not read outside at all; the caller has guaranteed it will not happen.
pub const BORDER_ISOLATED = 16

#: Nearest-neighbour sampling.
pub const INTER_NEAREST = 0
#: Bilinear sampling.
pub const INTER_LINEAR = 1
#: Bicubic sampling over a 4x4 neighbourhood.
pub const INTER_CUBIC = 2
#: Pixel-area resampling, the right choice for shrinking.
pub const INTER_AREA = 3
#: Lanczos over an 8x8 neighbourhood.
pub const INTER_LANCZOS4 = 4
#: Treat the supplied matrix as the inverse transform.
pub const WARP_INVERSE_MAP = 16

#: Sort each row's values, leaving the rows in place.
pub const SORT_EVERY_ROW = 0
#: Sort each column's values.
pub const SORT_EVERY_COLUMN = 1
#: Sort smallest first.
pub const SORT_ASCENDING = 0
#: Sort largest first.
pub const SORT_DESCENDING = 16

#: Reduce to a single row or column by summing.
pub const REDUCE_SUM = 0
#: Reduce by averaging.
pub const REDUCE_AVG = 1
#: Reduce by taking the largest.
pub const REDUCE_MAX = 2
#: Reduce by taking the smallest.
pub const REDUCE_MIN = 3

#: Sum of absolute differences.
pub const NORM_L1 = 2
#: Euclidean length.
pub const NORM_L2 = 4
#: Squared Euclidean length.
pub const NORM_L2SQR = 5
#: Largest absolute value.
pub const NORM_INF = 1
#: Number of differing elements.
pub const NORM_HAMMING = 6
#: Scale so the values span a given range.
pub const NORM_MINMAX = 32

#: Compare for equality.
pub const CMP_EQ = 0
#: Compare greater than.
pub const CMP_GT = 1
#: Compare greater than or equal.
pub const CMP_GE = 2
#: Compare less than.
pub const CMP_LT = 3
#: Compare less than or equal.
pub const CMP_LE = 4
#: Compare for inequality.
pub const CMP_NE = 5

#: Fill a shape solidly rather than stroking its outline.
pub const FILLED = -1
#: Four-connected Bresenham lines.
pub const LINE_4 = 4
#: Eight-connected Bresenham lines.
pub const LINE_8 = 8
#: Antialiased lines.
pub const LINE_AA = 16

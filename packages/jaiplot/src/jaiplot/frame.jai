#: The mapping a series draws through: a transform plus the two scales.
#:
#: A series needs more than a `Transform` to place a point. On a logarithmic
#: axis some values have no position at all, and asking the transform to map one
#: raises. `Frame` pairs the transform with the scales that made it, so a series
#: can ask `place` and get null for a point that does not belong on the axes
#: rather than an exception it has no way to answer.
#:
#: ```
#: from jaiplot.geom import Rect
#: from jaiplot.scale import LinearScale, LogScale
#: let data = Rect(1.0, 0.0, 100.0, 1.0)
#: let frame = Frame(data, Rect(0.0, 0.0, 50.0, 20.0), LogScale(), LinearScale())
#: frame.place(100.0, 1.0)      # Point(50.0, 0.0)
#: frame.place(0.0, 1.0)        # null
#: ```
from .geom import Point, Rect, Transform
from .scale import Scale

#: A transform together with the scales it was built from.
pub class Frame {
    #: The data rectangle currently in view.
    pub let data: Rect

    #: The device rectangle the data is drawn into.
    pub let device: Rect

    #: The scale governing the horizontal axis.
    pub let x_scale: Scale

    #: The scale governing the vertical axis.
    pub let y_scale: Scale

    #: The composed data-to-device map.
    pub let transform: Transform

    #: Build the frame and the transform that goes with it.
    #:
    #: Raises `ValueError` when a limit does not belong on its scale, or when
    #: either rectangle has no extent, both by way of the transform.
    pub fn init(self, data: Rect, device: Rect, x_scale: Scale, y_scale: Scale) {
        self.data = data
        self.device = device
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.transform = Transform(
            data,
            device,
            x_forward: |value| x_scale.forward(value),
            x_inverse: |value| x_scale.inverse(value),
            y_forward: |value| y_scale.forward(value),
            y_inverse: |value| y_scale.inverse(value)
        )
    }

    #: The device point for `(x, y)`, or null when either scale rejects it.
    #:
    #: A rejected point is one the axes cannot show — a zero on a log axis, a
    #: NaN from a gap in the data. Returning null rather than raising lets a
    #: series break its line there and carry on, which is what a reader expects
    #: to see.
    pub fn place(self, x: float, y: float) -> Point? {
        if not self.x_scale.accepts(x) { return null }
        if not self.y_scale.accepts(y) { return null }
        return self.transform.apply(x, y)
    }

    #: The data coordinate under a device point.
    pub fn locate(self, px: float, py: float) -> tuple[float, float] {
        return self.transform.invert(px, py)
    }

    #: Report whether both scales can place `(x, y)`.
    pub fn accepts(self, x: float, y: float) -> bool {
        return self.x_scale.accepts(x) and self.y_scale.accepts(y)
    }

    fn __str__(self) -> str {
        return f"Frame({self.data} -> {self.device}, {self.x_scale}, {self.y_scale})"
    }
}

#: Split `points` into the runs the frame can place, dropping the gaps.
#:
#: A series with a NaN in the middle, or a zero on a log axis, becomes two
#: polylines rather than one line drawn through a point that does not exist.
#: Runs of a single point are kept, since a marker still belongs there.
#:
#: ```
#: from jaiplot.geom import Rect
#: from jaiplot.scale import LinearScale
#: let box = Rect(0.0, 0.0, 4.0, 4.0)
#: let frame = Frame(box, box, LinearScale(), LinearScale())
#: runs(frame, [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]).len()     # 1
#: ```
pub fn runs(frame: Frame, xs: list[float], ys: list[float]) -> list[list[Point]] {
    if xs.len() != ys.len() {
        throw ValueError(f"x and y must be the same length, got {xs.len()} and {ys.len()}")
    }
    var result = []
    var current = []
    for (index, x) in xs.enumerate() {
        let placed = frame.place(x, ys[index])
        if placed == null {
            if current.len() > 0 {
                result.push(current)
                current = []
            }
        } else {
            current.push(placed)
        }
    }
    if current.len() > 0 { result.push(current) }
    return result
}

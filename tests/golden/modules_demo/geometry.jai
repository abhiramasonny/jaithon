#: A small module demonstrating explicit exports.
import std.math

pub const UNIT = 1.0

#: Area of a circle of radius `r`.
pub fn circle_area(r: float) -> float {
    return math.PI * r ** 2
}

#: Area of a rectangle.
pub fn rect_area(w: float, h: float) -> float {
    return w * h
}

# Not exported: private to this module.
fn _internal_scale(v: float) -> float {
    return v * 2.0
}

pub fn scaled(v: float) -> float {
    return _internal_scale(v)
}

#: How a data value becomes a position along an axis.
#:
#: A scale is two things: a monotone warp that carries a data value into the
#: linear space the axes occupy, and a rule for where to put ticks in it. A
#: linear scale warps nothing; a logarithmic one takes the logarithm. Every
#: other difference between the two — where ticks fall, how they are labelled,
#: what limits are legal — follows from that.
#:
#: Scales hold no limits. An axis owns its limits and passes them in, so there
#: is one copy of that state rather than two that can disagree.
#:
#: ```
#: let scale = LinearScale()
#: scale.forward(2.5)                   # 2.5
#: scale.ticks(0.0, 10.0, 5)            # [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
#: ```
import std.math

from .ticks import format_tick, linear_ticks, log_minor_ticks, log_ticks, nice_bounds, tick_step

#: The warp and the tick rule for one axis.
#:
#: `forward` and `inverse` must be exact inverses over the axis's limits, since
#: `Transform` composes one to place a point and the other to answer where the
#: cursor is.
pub trait Scale {
    #: Carry a data value into the linear space the device rectangle spans.
    fn forward(self, value: float) -> float

    #: Carry a warped value back into data units.
    fn inverse(self, value: float) -> float

    #: Report whether `value` can appear on this axis at all.
    fn accepts(self, value: float) -> bool

    #: Labelled tick positions within `[low, high]`, in data units.
    fn ticks(self, low: float, high: float, count_hint: int) -> list[float]

    #: Unlabelled intermediate positions, empty when the scale has none.
    fn minor_ticks(self, low: float, high: float, count_hint: int) -> list[float]

    #: Limits rounded outward to whole ticks, for an axis ranging itself.
    fn nice(self, low: float, high: float, count_hint: int) -> tuple[float, float]

    #: Write the labels for `values`, as a column written to one precision.
    fn labels(self, values: list[float], low: float, high: float, count_hint: int) -> list[str]
}

#: An axis where equal data intervals occupy equal distances.
#:
#: ```
#: let scale = LinearScale()
#: scale.forward(7.0)                       # 7.0
#: scale.nice(2.3, 97.1, 5)                 # (0.0, 100.0)
#: scale.accepts(-4.0)                      # true
#: ```
pub class LinearScale: Scale {
    #: Create a linear scale.
    pub fn init(self) {
    }

    #: Return `value` unchanged; a linear axis warps nothing.
    pub fn forward(self, value: float) -> float {
        return value
    }

    #: Return `value` unchanged.
    pub fn inverse(self, value: float) -> float {
        return value
    }

    #: Accept every finite value. A linear axis has no forbidden region.
    pub fn accepts(self, value: float) -> bool {
        return math.is_finite(value)
    }

    #: Tick positions at nice round multiples within the limits.
    pub fn ticks(self, low: float, high: float, count_hint: int) -> list[float] {
        return linear_ticks(low, high, count_hint)
    }

    #: No intermediate ticks: the labelled ones are already evenly spaced.
    pub fn minor_ticks(self, low: float, high: float, count_hint: int) -> list[float] {
        return []
    }

    #: Limits extended outward to the enclosing ticks.
    pub fn nice(self, low: float, high: float, count_hint: int) -> tuple[float, float] {
        return nice_bounds(low, high, count_hint)
    }

    #: Labels written to the precision the tick step calls for.
    pub fn labels(
        self,
        values: list[float],
        low: float,
        high: float,
        count_hint: int
    ) -> list[str] {
        if values.len() == 0 { return [] }
        let step = tick_step(low, high, count_hint)
        return [format_tick(value, step) for value in values]
    }

    fn __str__(self) -> str {
        return "LinearScale()"
    }
}

#: An axis where equal ratios occupy equal distances.
#:
#: Only positive values have a position, so a series containing zero or a
#: negative number cannot be drawn on one and `accepts` says so. An axis is
#: expected to drop those points rather than ask this class to invent a
#: position for them.
#:
#: ```
#: let scale = LogScale()
#: scale.forward(100.0)                     # 2.0
#: scale.inverse(3.0)                       # 1000.0
#: scale.accepts(0.0)                       # false
#: scale.ticks(1.0, 1000.0, 4)              # [1.0, 10.0, 100.0, 1000.0]
#: ```
pub class LogScale: Scale {
    #: The ratio that one unit of distance represents.
    pub let base: float

    #: Create a logarithmic scale, base ten unless told otherwise.
    #:
    #: Raises `ValueError` for a base of one or less, which would make the
    #: logarithm constant or undefined.
    pub fn init(self, base: float = 10.0) {
        if not math.is_finite(base) or base <= 1.0 {
            throw ValueError(f"a logarithmic base must be finite and exceed 1, got {base}")
        }
        self.base = base
    }

    #: The logarithm of `value` in this base.
    #:
    #: Raises `ValueError` on a value the axis cannot place, rather than
    #: returning the negative infinity the logarithm of zero would give and
    #: letting it poison every coordinate downstream.
    pub fn forward(self, value: float) -> float {
        if not self.accepts(value) {
            throw ValueError(f"a logarithmic axis has no position for {value}")
        }
        return math.log_base(value, self.base)
    }

    #: This base raised to `value`.
    pub fn inverse(self, value: float) -> float {
        return self.base ** value
    }

    #: Accept the finite positive values, which are the ones with a logarithm.
    pub fn accepts(self, value: float) -> bool {
        return math.is_finite(value) and value > 0.0
    }

    #: Tick positions at whole powers of the base.
    pub fn ticks(self, low: float, high: float, count_hint: int) -> list[float] {
        return log_ticks(low, high, self.base, count_hint)
    }

    #: The intermediate multiples between neighbouring powers.
    pub fn minor_ticks(self, low: float, high: float, count_hint: int) -> list[float] {
        return log_minor_ticks(low, high, self.base)
    }

    #: Limits extended outward to the enclosing powers of the base.
    pub fn nice(self, low: float, high: float, count_hint: int) -> tuple[float, float] {
        if not self.accepts(low) or not self.accepts(high) {
            throw ValueError(f"a logarithmic axis needs positive limits, got {low} to {high}")
        }
        if low == high { return (low, high) }
        let first = math.floor(math.log_base(low, self.base))
        let last = math.ceil(math.log_base(high, self.base))
        return (self.base ** first, self.base ** last)
    }

    #: Labels for powers of the base, each written in its own right.
    #:
    #: A log axis spans orders of magnitude, so one shared precision would
    #: render either the small ticks as zeroes or the large ones as a wall of
    #: digits. Each label is therefore given the precision its own magnitude
    #: needs.
    pub fn labels(
        self,
        values: list[float],
        low: float,
        high: float,
        count_hint: int
    ) -> list[str] {
        return [format_tick(value, _own_step(value)) for value in values]
    }

    fn __str__(self) -> str {
        return f"LogScale({self.base})"
    }
}

#: Build a scale from its name, for the string form of `Axes.set_xscale`.
#:
#: Raises `ValueError` naming the scales that exist, since a misspelling would
#: otherwise silently leave the axis linear.
#:
#: ```
#: scale_named("linear").forward(3.0)     # 3.0
#: scale_named("log").forward(100.0)      # 2.0
#: ```
pub fn scale_named(name: str) -> Scale {
    if name == "linear" { return LinearScale() }
    if name == "log" { return LogScale() }
    if name == "log2" { return LogScale(2.0) }
    throw ValueError(f"unknown scale {name}, expected one of linear, log, log2")
}

#: A step that gives `value` enough decimals to read, used where ticks are
#: decades apart and no single shared step is right for all of them.
fn _own_step(value: float) -> float {
    let magnitude = math.fabs(value)
    if magnitude == 0.0 or not math.is_finite(magnitude) { return 1.0 }
    return 10.0 ** math.floor(math.log10(magnitude))
}

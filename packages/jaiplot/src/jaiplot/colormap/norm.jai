#: How a data value reaches the unit interval a map is indexed by.
import std.math

#: How a data value becomes the unit interval a colormap is indexed by.
#:
#: The split between a norm and a map is matplotlib's, and it is the right one:
#: the map decides what the colours are and the norm decides which value gets
#: which, so a heavy-tailed field can be rescued by swapping the norm without
#: touching the palette.
#:
#: Either end of the range may be pinned by the caller or left to the data.
#: `autoscale` fills in only the ends nobody pinned, which is what lets
#: `set_clim(low: 0.0)` hold the bottom while the top keeps tracking the data.
pub trait Norm {
    #: Carry `value` onto the unit interval, NaN when it has no place there.
    #:
    #: The result is deliberately not clamped: `Colormap.at` clamps, and a
    #: caller that wants to know a value was off the end can still see it.
    fn unit(self, value: float) -> float

    #: The data values that land on 0.0 and 1.0.
    fn limits(self) -> tuple[float, float]

    #: Pin either end of the range; a null end is left alone.
    fn set_limits(self, low: float?, high: float?) -> void

    #: Take the unpinned ends from `values`, ignoring what the norm cannot use.
    fn autoscale(self, values: list[float]) -> void
}

#: Equal data intervals take equal stretches of the ramp.
#:
#: matplotlib's `Normalize`. This is the default everywhere and it is right for
#: a field whose values are spread evenly; it is exactly wrong for one that is
#: not, because the whole ramp then goes to the few cells in the tail.
#:
#: ```
#: let norm = LinearNorm(low: 0.0, high: 4.0)
#: norm.unit(1.0)      # 0.25
#: norm.unit(6.0)      # 1.5
#: ```
pub class LinearNorm: Norm {
    var low: float
    var high: float
    var low_pinned: bool
    var high_pinned: bool

    #: Create a norm over `low` to `high`, taking either end from the data.
    #:
    #: Raises `ValueError` when both ends are pinned the wrong way round.
    pub fn init(self, low: float? = null, high: float? = null) {
        self.low = low ?? 0.0
        self.high = high ?? 1.0
        self.low_pinned = low != null
        self.high_pinned = high != null
        if self.low_pinned and self.high_pinned and self.high < self.low {
            throw ValueError(f"limits must be ordered, got {self.low} to {self.high}")
        }
    }

    #: Where `value` falls between the limits.
    #:
    #: A range of zero width sends everything to 0.0 rather than dividing by it,
    #: matching matplotlib: a constant field paints as the bottom of the map.
    pub fn unit(self, value: float) -> float {
        if not math.is_finite(value) { return math.NAN }
        let span = self.high - self.low
        if span <= 0.0 { return 0.0 }
        return (value - self.low) / span
    }

    #: The data values that land on 0.0 and 1.0.
    pub fn limits(self) -> tuple[float, float] {
        return (self.low, self.high)
    }

    #: Pin either end; a null end is left as it was.
    pub fn set_limits(self, low: float?, high: float?) -> void {
        if low != null {
            self.low = low
            self.low_pinned = true
        }
        if high != null {
            self.high = high
            self.high_pinned = true
        }
        if self.high < self.low {
            throw ValueError(f"limits must be ordered, got {self.low} to {self.high}")
        }
    }

    #: Take the unpinned ends from the finite values in `values`.
    pub fn autoscale(self, values: list[float]) -> void {
        let found = _finite_range(values, false)
        if found == null { return }
        let (low, high) = found ?? (0.0, 1.0)
        if not self.low_pinned { self.low = low }
        if not self.high_pinned { self.high = high }
        if self.high < self.low { self.high = self.low }
    }

    fn __str__(self) -> str {
        return f"LinearNorm({self.low} to {self.high})"
    }
}

#: Equal *ratios* take equal stretches of the ramp.
#:
#: matplotlib's `LogNorm`. This is the norm a heavy-tailed field wants: a hit
#: count that runs 1 to 100,000 gives a linear norm nothing to show below about
#: 1,000, while a log norm gives each decade the same fifth of the map.
#:
#: Zero and negative values have no place on a log axis, so `unit` returns NaN
#: for them and they paint as the map's missing colour. That is louder than
#: quietly clamping them to the bottom, which is the point — a log heatmap of
#: data containing zeros is a chart with a hole in it, and the reader should
#: see the hole.
#:
#: ```
#: let norm = LogNorm(low: 1.0, high: 1000.0)
#: norm.unit(10.0)      # 0.3333333333333333
#: norm.unit(1000.0)    # 1.0
#: ```
pub class LogNorm: Norm {
    var low: float
    var high: float
    var low_pinned: bool
    var high_pinned: bool

    #: Create a norm over `low` to `high`, taking either end from the data.
    #:
    #: Raises `ValueError` on a pinned end at or below zero, which has no
    #: logarithm, and on both ends pinned the wrong way round.
    pub fn init(self, low: float? = null, high: float? = null) {
        self.low = low ?? 1.0
        self.high = high ?? 10.0
        self.low_pinned = low != null
        self.high_pinned = high != null
        if self.low <= 0.0 or self.high <= 0.0 {
            throw ValueError(f"log limits must be positive, got {self.low} to {self.high}")
        }
        if self.low_pinned and self.high_pinned and self.high < self.low {
            throw ValueError(f"limits must be ordered, got {self.low} to {self.high}")
        }
    }

    #: Where `value` falls between the limits, measured in decades.
    pub fn unit(self, value: float) -> float {
        if not math.is_finite(value) or value <= 0.0 { return math.NAN }
        let span = math.log10(self.high) - math.log10(self.low)
        if span <= 0.0 { return 0.0 }
        return (math.log10(value) - math.log10(self.low)) / span
    }

    #: The data values that land on 0.0 and 1.0.
    pub fn limits(self) -> tuple[float, float] {
        return (self.low, self.high)
    }

    #: Pin either end; a null end is left as it was.
    pub fn set_limits(self, low: float?, high: float?) -> void {
        if low != null {
            if low <= 0.0 { throw ValueError(f"log limits must be positive, got {low}") }
            self.low = low
            self.low_pinned = true
        }
        if high != null {
            if high <= 0.0 { throw ValueError(f"log limits must be positive, got {high}") }
            self.high = high
            self.high_pinned = true
        }
        if self.high < self.low {
            throw ValueError(f"limits must be ordered, got {self.low} to {self.high}")
        }
    }

    #: Take the unpinned ends from the positive values in `values`.
    #:
    #: A field of nothing but zeros leaves the limits alone rather than raising:
    #: an empty first frame is ordinary, and the norm still has a usable range.
    pub fn autoscale(self, values: list[float]) -> void {
        let found = _finite_range(values, true)
        if found == null { return }
        let (low, high) = found ?? (1.0, 10.0)
        if not self.low_pinned { self.low = low }
        if not self.high_pinned { self.high = high }
        if self.high < self.low { self.high = self.low }
    }

    fn __str__(self) -> str {
        return f"LogNorm({self.low} to {self.high})"
    }
}

#: Two linear pieces meeting at a centre that lands halfway up the ramp.
#:
#: matplotlib's `TwoSlopeNorm`, and the missing half of every diverging map. An
#: anomaly field running -2 to +8 under a linear norm puts zero a fifth of the
#: way up `coolwarm`, so the neutral grey sits in the middle of the cool side
#: and the whole picture reads as warm. Pinning the centre puts the grey where
#: the reader already thinks it is.
#:
#: The two arms have different slopes on purpose — that is what lets a short
#: negative side and a long positive one both use their full half of the map.
#: The cost is that equal distances along the ramp are no longer equal
#: differences in the data, so this is for a field with a meaningful zero, not
#: for one that merely happens to straddle it.
#:
#: ```
#: let norm = TwoSlopeNorm(0.0, low: -2.0, high: 8.0)
#: norm.unit(0.0)       # 0.5
#: norm.unit(-1.0)      # 0.25
#: norm.unit(4.0)       # 0.75
#: ```
pub class TwoSlopeNorm: Norm {
    var center: float
    var low: float
    var high: float
    var low_pinned: bool
    var high_pinned: bool

    #: Create a norm centred on `center`, taking either end from the data.
    #:
    #: Raises `ValueError` when a pinned end is on the wrong side of the centre,
    #: which would fold one arm of the ramp back on itself.
    pub fn init(self, center: float, low: float? = null, high: float? = null) {
        if not math.is_finite(center) {
            throw ValueError(f"the centre must be finite, got {center}")
        }
        self.center = center
        self.low = low ?? center - 1.0
        self.high = high ?? center + 1.0
        self.low_pinned = low != null
        self.high_pinned = high != null
        self._check()
    }

    #: Where `value` falls, with the centre exactly halfway.
    pub fn unit(self, value: float) -> float {
        if not math.is_finite(value) { return math.NAN }
        if value <= self.center {
            let span = self.center - self.low
            if span <= 0.0 { return 0.5 }
            return 0.5 * (value - self.low) / span
        }
        let span = self.high - self.center
        if span <= 0.0 { return 0.5 }
        return 0.5 + 0.5 * (value - self.center) / span
    }

    #: The data values that land on 0.0 and 1.0.
    pub fn limits(self) -> tuple[float, float] {
        return (self.low, self.high)
    }

    #: Pin either end; a null end is left as it was.
    pub fn set_limits(self, low: float?, high: float?) -> void {
        if low != null {
            self.low = low
            self.low_pinned = true
        }
        if high != null {
            self.high = high
            self.high_pinned = true
        }
        self._check()
    }

    #: Take the unpinned ends from `values`, then widen to keep the centre in.
    #:
    #: A field that happens to be all positive would otherwise leave the centre
    #: outside its own range. matplotlib mirrors the other arm across the centre
    #: in that case, and so does this: the picture keeps a neutral middle and
    #: half the ramp goes unused, which is the honest way to show that the data
    #: only went one way.
    pub fn autoscale(self, values: list[float]) -> void {
        let found = _finite_range(values, false)
        if found == null { return }
        let (low, high) = found ?? (0.0, 1.0)
        if not self.low_pinned { self.low = low }
        if not self.high_pinned { self.high = high }
        if self.low >= self.center { self.low = self.center - (self.high - self.center) }
        if self.high <= self.center { self.high = self.center + (self.center - self.low) }
    }

    fn _check(self) -> void {
        if self.low >= self.center or self.high <= self.center {
            throw ValueError(f"a two-slope norm needs {self.low} < {self.center} < {self.high}")
        }
    }

    fn __str__(self) -> str {
        return f"TwoSlopeNorm({self.low} to {self.high} about {self.center})"
    }
}

#: The finite range of `values`, or null when nothing in it qualifies.
#:
#: `positive_only` is what `LogNorm` needs: a zero in an otherwise positive
#: field must not drag the lower limit onto a value with no logarithm.
fn _finite_range(values: list[float], positive_only: bool) -> tuple[float, float]? {
    var low = 0.0
    var high = 0.0
    var seen = false
    for value in values {
        if not math.is_finite(value) { continue }
        if positive_only and value <= 0.0 { continue }
        if not seen {
            low = value
            high = value
            seen = true
        } else {
            low = min(low, value)
            high = max(high, value)
        }
    }
    if not seen { return null }
    return (low, high)
}

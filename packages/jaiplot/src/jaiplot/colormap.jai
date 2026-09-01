#: Colour maps and the normalisations that feed them: `Colormap`, `Norm`,
#: `LinearNorm`, `LogNorm` and `TwoSlopeNorm`.
#:
#: A field drawn between two colours can only say *more* or *less*, and
#: interpolating a good map's endpoints does not recover the map: viridis runs
#: `#440154` to `#fde725`, and `Color.lerp` between those two passes through an
#: olive brown at the quarter point where viridis is a dark blue-green. So the
#: maps here are tables — 256 levels each, the same numbers matplotlib applies,
#: which is what makes a Jaithon heatmap and a matplotlib one the same picture
#: rather than a similar one.
#:
#: `packages/jaicv` already carries these levels for OpenCV, and the six maps
#: the two share agree byte for byte. They are written out again rather than
#: imported because jaiplot has no dependencies and a plotting package reaching
#: into a computer vision package for a colour table is the wrong direction.
#:
#: The other half of the job is the norm, and it is the half that gets skipped.
#: A linear norm cannot tell an affine rescale apart — a Gaussian `g` and
#: `1 + 9999 * g` come out pixel-identical — and it renders anything with a
#: heavy tail as one flat colour with a dot in it. `LogNorm` spreads the decades
#: and `TwoSlopeNorm` pins a zero to the middle of a diverging map, which
#: between them cover most of what a linear norm ruins.
#:
#: ```
#: let map = Colormap.named("viridis")
#: map.at(0.0).to_hex()                              # "#440154"
#: map.at(1.0).to_hex()                              # "#fde725"
#: Colormap.named("viridis_r").at(0.0).to_hex()      # "#fde725"
#: LinearNorm(low: 0.0, high: 10.0).unit(2.5)        # 0.25
#: ```
import std.math
from .style import Color
from .colormap.tables import COLORMAP_NAMES, _CANONICAL, _lookup, _reverse

export { COLORMAP_NAMES }

#: How many levels a built-in map holds.
#:
#: matplotlib's own resolution. It matters that this is the same number: a map
#: is indexed as `floor(unit * levels)`, so resampling to any other count would
#: quantise the ramp differently and the two libraries would disagree by a level
#: here and there.
pub const COLORMAP_LEVELS: int = 256

#: How many bands a grid's legend swatch is drawn in.
#:
#: Enough that the swatch reads as a ramp rather than a flat block, few enough
#: that a legend with several grids in it is still a handful of polygons.
pub const SWATCH_BANDS: int = 12

#: A ramp from a value on the unit interval to a colour.
#:
#: Immutable, like `Color`: `reversed` and `with_missing` hand back a new map.
#: The levels are looked up rather than interpolated, and the level for `unit`
#: is `floor(unit * levels)` clamped to the last one, which is exactly how
#: matplotlib indexes a `Colormap`. Values outside the unit interval clamp to
#: the ends; a non-finite one takes the missing colour instead.
#:
#: ```
#: let map = Colormap.named("magma")
#: map.levels()                     # 256
#: map.at(-3.0) == map.at(0.0)      # true
#: map.name                         # "magma"
#: ```
pub class Colormap {
    #: What the map is called, for a legend or an error message.
    pub let name: str

    #: The colour a value with no place on the ramp takes.
    #:
    #: matplotlib calls this the *bad* colour and defaults it to transparent,
    #: which is the right default here too: an unobserved cell of an occupancy
    #: grid should show the panel behind it, not a colour that reads as data.
    pub let missing: Color

    let stops: list[Color]

    #: Build a map from its levels, dark end first.
    #:
    #: Raises `ValueError` on an empty list, which would leave `at` with nothing
    #: to return.
    pub fn init(self, stops: list[Color], name: str, missing: Color? = null) {
        if stops.len() == 0 {
            throw ValueError(f"colormap {name} needs at least one level, got an empty list")
        }
        self.stops = [color for color in stops]
        self.name = name
        self.missing = missing ?? Color.from_rgba(0, 0, 0, 0)
    }

    #: One of the built-in maps, by name.
    #:
    #: Case does not matter, and a `_r` suffix reverses the map the way it does
    #: in matplotlib. Raises `ValueError` listing the maps that exist, since a
    #: typo would otherwise silently fall back to something and the reader would
    #: never learn which map they were looking at.
    #:
    #: ```
    #: Colormap.named("Viridis").name               # "viridis"
    #: Colormap.named("gray").at(1.0).to_hex()      # "#ffffff"
    #: Colormap.named("gray_r").at(1.0).to_hex()    # "#000000"
    #: ```
    pub static fn named(name: str) -> Colormap {
        let key = name.lower()
        if key.ends_with("_r") {
            let base = key[:key.len() - 2]
            return Colormap(_reverse(_lookup(base, name)), _CANONICAL[base] + "_r")
        }
        return Colormap(_lookup(key, name), _CANONICAL[key])
    }

    #: The built-in map names, in a fresh list.
    pub static fn names() -> list[str] {
        return [name for name in COLORMAP_NAMES]
    }

    #: A map that is exactly the colours given, one level each.
    #:
    #: matplotlib's `ListedColormap`. Use it for a categorical field — eight
    #: land-cover classes, say — where interpolating between two of the colours
    #: would invent a class that does not exist.
    #:
    #: ```
    #: let map = Colormap.from_colors([Color.from_hex("#000"), Color.from_hex("#fff")])
    #: map.at(0.4).to_hex()      # "#000000"
    #: map.at(0.6).to_hex()      # "#ffffff"
    #: ```
    pub static fn from_colors(colors: list[Color], name: str = "listed") -> Colormap {
        return Colormap(colors, name)
    }

    #: A map that blends through the colours given, evenly spaced.
    #:
    #: matplotlib's `LinearSegmentedColormap.from_list`. The blend runs through
    #: `Color.lerp`, so it happens in linear light and the midpoint of black to
    #: white lands at 188 rather than the 128 an sRGB average would give.
    #:
    #: Raises `ValueError` on an empty list or a level count below two.
    #:
    #: ```
    #: let ramp = Colormap.from_stops([Color.from_hex("#000"), Color.from_hex("#fff")])
    #: ramp.at(0.0).to_hex()      # "#000000"
    #: ramp.at(1.0).to_hex()      # "#ffffff"
    #: ```
    pub static fn from_stops(
        colors: list[Color],
        name: str = "blend",
        levels: int = COLORMAP_LEVELS
    ) -> Colormap {
        if colors.len() == 0 {
            throw ValueError(f"colormap {name} needs at least one stop, got an empty list")
        }
        if levels < 2 {
            throw ValueError(f"a blended colormap needs at least two levels, got {levels}")
        }
        if colors.len() == 1 { return Colormap([colors[0] for _index in 0..levels], name) }

        let last = colors.len() - 1
        var stops: list[Color] = []
        for level in 0..levels {
            let along = float(level) / float(levels - 1) * float(last)
            let lower = min(last - 1, int(math.floor(along)))
            stops.push(colors[lower].lerp(colors[lower + 1], along - float(lower)))
        }
        return Colormap(stops, name)
    }

    #: The colour at `unit`, 0.0 at the dark end and 1.0 at the light one.
    #:
    #: Out-of-range values clamp rather than raise: a norm hands over whatever
    #: the data said, and a cell above the upper limit should paint as the top
    #: of the ramp, not stop the render. A non-finite `unit` — which is how a
    #: norm reports a value it cannot carry, such as a zero under `LogNorm` —
    #: takes the missing colour.
    pub fn at(self, unit: float) -> Color {
        if not math.is_finite(unit) { return self.missing }
        let count = self.stops.len()
        let level = int(math.floor(math.clamp(unit, 0.0, 1.0) * float(count)))
        return self.stops[max(0, min(count - 1, level))]
    }

    #: The colour at `unit`, already packed for `Primitive.Raster`.
    #:
    #: Shading a 200x200 grid is 40,000 of these, and going through `Color` for
    #: each one allocates 40,000 objects that are read once and dropped.
    pub fn packed_at(self, unit: float) -> int {
        return self.at(unit).packed()
    }

    #: How many levels the map holds.
    pub fn levels(self) -> int {
        return self.stops.len()
    }

    #: The same map with its ends swapped.
    pub fn reversed(self) -> Colormap {
        let flipped = if self.name.ends_with("_r") {
            self.name[:self.name.len() - 2]
        } else {
            self.name + "_r"
        }
        return Colormap(_reverse(self.stops), flipped, missing: self.missing)
    }

    #: The same map, painting values it cannot carry in `color`.
    pub fn with_missing(self, color: Color) -> Colormap {
        return Colormap(self.stops, self.name, missing: color)
    }

    #: The levels in order, as a fresh list.
    pub fn to_list(self) -> list[Color] {
        return [color for color in self.stops]
    }

    fn __str__(self) -> str {
        return f"Colormap({self.name}, {self.stops.len()} levels)"
    }
}

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

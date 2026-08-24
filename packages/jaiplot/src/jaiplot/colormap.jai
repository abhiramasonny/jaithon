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

#: The levels of a built-in map, decoded once and shared thereafter.
#:
#: Decoding is 1,536 hex digits a map, which is nothing once but is not nothing
#: when a redrawn field asks for viridis on every frame. The cached lists are
#: never handed out — `Colormap` copies on the way in — so nobody can mutate
#: the copy everyone else is sharing.
var _decoded: dict[str, list[Color]] = {}

#: The decoded levels of the map called `key`, by its lowercased name.
#:
#: `asked` is the caller's spelling, so the error quotes what they wrote rather
#: than what the lookup made of it.
fn _lookup(key: str, asked: str) -> list[Color] {
    if _decoded.has(key) { return _decoded[key] }
    if not _TABLES.has(key) {
        throw ValueError(f"unknown colormap {asked}, expected one of {COLORMAP_NAMES.join(", ")}")
    }
    let levels = _decode(_TABLES[key])
    _decoded[key] = levels
    return levels
}

#: A run of `rrggbb` digits as one colour per six.
fn _decode(digits: str) -> list[Color] {
    var out: list[Color] = []
    for start in range(0, digits.len(), 6) {
        out.push(Color.from_hex("#" + digits[start:start + 6]))
    }
    return out
}

#: A copy of `stops` in the other order.
fn _reverse(stops: list[Color]) -> list[Color] {
    var out: list[Color] = []
    for index in 0..stops.len() { out.push(stops[stops.len() - 1 - index]) }
    return out
}

#: The 256-level tables, sixteen colours to a line, `rrggbb` throughout.
#:
#: Written out rather than computed: matplotlib builds viridis and its
#: siblings from a table of 256 measured colours in the first place, and the
#: maps that do have a formula behind them — `hot`, `coolwarm` — are
#: resampled from control points, so re-deriving the formula lands a level
#: away nearly everywhere. Recorded from matplotlib 3.11.1.
#:
#: Do not edit by hand.

#: matplotlib's default since 2.0: perceptually uniform, monotone in
#: lightness, and legible in greyscale and to every common colour vision
#: deficiency. Start here and only leave for a reason.
const _VIRIDIS = "44015444025645045745055946075a46085c460a5d460b5e470d60470e61471063471164471365481467481668481769" +
    "48186a481a6c481b6d481c6e481d6f481f70482071482173482374482475482576482677482878482979472a7a472c7a" +
    "472d7b472e7c472f7d46307e46327e46337f463480453581453781453882443983443a83443b84433d84433e85423f85" +
    "4240864241864142874144874045884046883f47883f48893e49893e4a893e4c8a3d4d8a3d4e8a3c4f8a3c508b3b518b" +
    "3b528b3a538b3a548c39558c39568c38588c38598c375a8c375b8d365c8d365d8d355e8d355f8d34608d34618d33628d" +
    "33638d32648e32658e31668e31678e31688e30698e306a8e2f6b8e2f6c8e2e6d8e2e6e8e2e6f8e2d708e2d718e2c718e" +
    "2c728e2c738e2b748e2b758e2a768e2a778e2a788e29798e297a8e297b8e287c8e287d8e277e8e277f8e27808e26818e" +
    "26828e26828e25838e25848e25858e24868e24878e23888e23898e238a8d228b8d228c8d228d8d218e8d218f8d21908d" +
    "21918c20928c20928c20938c1f948c1f958b1f968b1f978b1f988b1f998a1f9a8a1e9b8a1e9c891e9d891f9e891f9f88" +
    "1fa0881fa1881fa1871fa28720a38620a48621a58521a68522a78522a88423a98324aa8325ab8225ac8226ad8127ad81" +
    "28ae8029af7f2ab07f2cb17e2db27d2eb37c2fb47c31b57b32b67a34b67935b77937b87838b9773aba763bbb753dbc74" +
    "3fbc7340bd7242be7144bf7046c06f48c16e4ac16d4cc26c4ec36b50c46a52c56954c56856c66758c7655ac8645cc863" +
    "5ec96260ca6063cb5f65cb5e67cc5c69cd5b6ccd5a6ece5870cf5773d05675d05477d1537ad1517cd2507fd34e81d34d" +
    "84d44b86d54989d5488bd6468ed64590d74393d74195d84098d83e9bd93c9dd93ba0da39a2da37a5db36a8db34aadc32" +
    "addc30b0dd2fb2dd2db5de2bb8de29bade28bddf26c0df25c2df23c5e021c8e020cae11fcde11dd0e11cd2e21bd5e21a" +
    "d8e219dae319dde318dfe318e2e418e5e419e7e419eae51aece51befe51cf1e51df4e61ef6e620f8e621fbe723fde725"

#: Uniform like viridis but warmer, purple through orange to yellow. Reads
#: better than viridis on a dark background.
const _PLASMA = "0d088710078813078916078a19068c1b068d1d068e20068f2206902406912605912805922a05932c05942e05952f0596" +
    "31059733059735049837049938049a3a049a3c049b3e049c3f049c41049d43039e44039e46039f48039f4903a04b03a1" +
    "4c02a14e02a25002a25102a35302a35502a45601a45801a45901a55b01a55c01a65e01a66001a66100a76300a76400a7" +
    "6600a76700a86900a86a00a86c00a86e00a86f00a87100a87201a87401a87501a87701a87801a87a02a87b02a87d03a8" +
    "7e03a88004a88104a78305a78405a78606a68707a68808a68a09a58b0aa58d0ba58e0ca48f0da4910ea3920fa39410a2" +
    "9511a19613a19814a099159f9a169f9c179e9d189d9e199da01a9ca11b9ba21d9aa31e9aa51f99a62098a72197a82296" +
    "aa2395ab2494ac2694ad2793ae2892b02991b12a90b22b8fb32c8eb42e8db52f8cb6308bb7318ab83289ba3388bb3488" +
    "bc3587bd3786be3885bf3984c03a83c13b82c23c81c33d80c43e7fc5407ec6417dc7427cc8437bc9447aca457acb4679" +
    "cc4778cc4977cd4a76ce4b75cf4c74d04d73d14e72d24f71d35171d45270d5536fd5546ed6556dd7566cd8576bd9586a" +
    "da5a6ada5b69db5c68dc5d67dd5e66de5f65de6164df6263e06363e16462e26561e26660e3685fe4695ee56a5de56b5d" +
    "e66c5ce76e5be76f5ae87059e97158e97257ea7457eb7556eb7655ec7754ed7953ed7a52ee7b51ef7c51ef7e50f07f4f" +
    "f0804ef1814df1834cf2844bf3854bf3874af48849f48948f58b47f58c46f68d45f68f44f79044f79143f79342f89441" +
    "f89540f9973ff9983ef99a3efa9b3dfa9c3cfa9e3bfb9f3afba139fba238fca338fca537fca636fca835fca934fdab33" +
    "fdac33fdae32fdaf31fdb130fdb22ffdb42ffdb52efeb72dfeb82cfeba2cfebb2bfebd2afebe2afec029fdc229fdc328" +
    "fdc527fdc627fdc827fdca26fdcb26fccd25fcce25fcd025fcd225fbd324fbd524fbd724fad824fada24f9dc24f9dd25" +
    "f8df25f8e125f7e225f7e425f6e626f6e826f5e926f5eb27f4ed27f3ee27f3f027f2f227f1f426f1f525f0f724f0f921"

#: Uniform, black to white through red and orange, so it works as ink on
#: paper and as light on a black panel.
const _INFERNO = "00000401000501010601010802010a02020c02020e03021004031204031405041706041907051b08051d09061f0a0722" +
    "0b07240c08260d08290e092b10092d110a30120a32140b34150b37160b39180c3c190c3e1b0c411c0c431e0c451f0c48" +
    "210c4a230c4c240c4f260c51280b53290b552b0b572d0b592f0a5b310a5c320a5e340a5f3609613809623909633b0964" +
    "3d09653e0966400a67420a68440a68450a69470b6a490b6a4a0c6b4c0c6b4d0d6c4f0d6c510e6c520e6d540f6d550f6d" +
    "57106e59106e5a116e5c126e5d126e5f136e61136e62146e64156e65156e67166e69166e6a176e6c186e6d186e6f196e" +
    "71196e721a6e741a6e751b6e771c6d781c6d7a1d6d7c1d6d7d1e6d7f1e6c801f6c82206c84206b85216b87216b88226a" +
    "8a226a8c23698d23698f24699025689225689326679526679727669827669a28659b29649d29649f2a63a02a63a22b62" +
    "a32c61a52c60a62d60a82e5fa92e5eab2f5ead305dae305cb0315bb1325ab3325ab43359b63458b73557b93556ba3655" +
    "bc3754bd3853bf3952c03a51c13a50c33b4fc43c4ec63d4dc73e4cc83f4bca404acb4149cc4248ce4347cf4446d04545" +
    "d24644d34743d44842d54a41d74b3fd84c3ed94d3dda4e3cdb503bdd513ade5238df5337e05536e15635e25734e35933" +
    "e45a31e55c30e65d2fe75e2ee8602de9612bea632aeb6429eb6628ec6726ed6925ee6a24ef6c23ef6e21f06f20f1711f" +
    "f1731df2741cf3761bf37819f47918f57b17f57d15f67e14f68013f78212f78410f8850ff8870ef8890cf98b0bf98c0a" +
    "f98e09fa9008fa9207fa9407fb9606fb9706fb9906fb9b06fb9d07fc9f07fca108fca309fca50afca60cfca80dfcaa0f" +
    "fcac11fcae12fcb014fcb216fcb418fbb61afbb81dfbba1ffbbc21fbbe23fac026fac228fac42afac62df9c72ff9c932" +
    "f9cb35f8cd37f8cf3af7d13df7d340f6d543f6d746f5d949f5db4cf4dd4ff4df53f4e156f3e35af3e55df2e661f2e865" +
    "f2ea69f1ec6df1ed71f1ef75f1f179f2f27df2f482f3f586f3f68af4f88ef5f992f6fa96f8fb9af9fc9dfafda1fcffa4"

#: Inferno's quieter sibling: the same black-to-white sweep with less
#: saturation, which keeps overlaid markers readable.
const _MAGMA = "00000401000501010601010802010902020b02020d03030f03031204041405041606051806051a07061c08071e090720" +
    "0a08220b09240c09260d0a290e0b2b100b2d110c2f120d31130d34140e36150e38160f3b180f3d19103f1a10421c1044" +
    "1d11471e114920114b21114e22115024125325125527125829115a2a115c2c115f2d11612f1163311165331067341069" +
    "36106b38106c390f6e3b0f703d0f713f0f72400f74420f75440f764510774710784910784a10794c117a4e117b4f127b" +
    "51127c52137c54137d56147d57157e59157e5a167e5c167f5d177f5f187f601880621980641a80651a80671b80681c81" +
    "6a1c816b1d816d1d816e1e81701f81721f817320817521817621817822817922827b23827c23827e2482802582812581" +
    "8326818426818627818827818928818b29818c29818e2a81902a81912b81932b80942c80962c80982d80992d809b2e7f" +
    "9c2e7f9e2f7fa02f7fa1307ea3307ea5317ea6317da8327daa337dab337cad347cae347bb0357bb2357bb3367ab5367a" +
    "b73779b83779ba3878bc3978bd3977bf3a77c03a76c23b75c43c75c53c74c73d73c83e73ca3e72cc3f71cd4071cf4070" +
    "d0416fd2426fd3436ed5446dd6456cd8456cd9466bdb476adc4869de4968df4a68e04c67e24d66e34e65e44f64e55064" +
    "e75263e85362e95462ea5661eb5760ec5860ed5a5fee5b5eef5d5ef05f5ef1605df2625df2645cf3655cf4675cf4695c" +
    "f56b5cf66c5cf66e5cf7705cf7725cf8745cf8765cf9785df9795df97b5dfa7d5efa7f5efa815ffb835ffb8560fb8761" +
    "fc8961fc8a62fc8c63fc8e64fc9065fd9266fd9467fd9668fd9869fd9a6afd9b6bfe9d6cfe9f6dfea16efea36ffea571" +
    "fea772fea973feaa74feac76feae77feb078feb27afeb47bfeb67cfeb77efeb97ffebb81febd82febf84fec185fec287" +
    "fec488fec68afec88cfeca8dfecc8ffecd90fecf92fed194fed395fed597fed799fed89afdda9cfddc9efddea0fde0a1" +
    "fde2a3fde3a5fde5a7fde7a9fde9aafdebacfcecaefceeb0fcf0b2fcf2b4fcf4b6fcf6b8fcf7b9fcf9bbfcfbbdfcfdbf"

#: Built to look the same to a deuteranope as to everyone else, at the cost
#: of the blue-yellow axis being all it has.
const _CIVIDIS = "00224e00234f00245100255300255400265600275800285900285b00295d002a5f002a61002b62002c64002c66002d68" +
    "002e6a002e6c002f6d00306f0030700031700031710132710533710833700c34700f357012357014367016377018376f" +
    "1a386f1c396f1e3a6f203a6f213b6e233c6e243c6e263d6e273e6e293f6e2a3f6d2b406d2d416d2e416d2f426d31436d" +
    "32436d33446d34456c35456c36466c38476c39486c3a486c3b496c3c4a6c3d4a6c3e4b6c3f4c6c404c6c414d6c424e6c" +
    "434e6c444f6c45506c46516c47516c48526c49536c4a536c4b546c4c556c4d556c4e566c4f576c50576c51586d52596d" +
    "535a6d545a6d555b6d555c6d565c6d575d6d585e6d595e6e5a5f6e5b606e5c616e5d616e5e626e5e636f5f636f60646f" +
    "61656f62656f636670646770656870656870666970676a71686a71696b716a6c716b6d726c6d726c6e726d6f726e6f73" +
    "6f70737071737172747272747273747374757474757575757676767777767777777878777979777a7a787b7a787c7b78" +
    "7d7c787e7c787e7d787f7e78807f78817f788280798381798482798582798683798784788885788985788a86788b8778" +
    "8c88788d88788e89788f8a78908b78918b78928c78928d78938e78948e77958f779690779791779892779992779a9376" +
    "9b94769c95769d95769e96769f9775a09875a19975a29975a39a74a49b74a59c74a69c74a79d73a89e73a99f73aaa073" +
    "aba072aca172ada272aea371afa471b0a571b1a570b3a670b4a76fb5a86fb6a96fb7a96eb8aa6eb9ab6dbaac6dbbad6d" +
    "bcae6cbdae6cbeaf6bbfb06bc0b16ac1b26ac2b369c3b369c4b468c5b568c6b667c7b767c8b866c9b965cbb965ccba64" +
    "cdbb63cebc63cfbd62d0be62d1bf61d2c060d3c05fd4c15fd5c25ed6c35dd7c45cd9c55cdac65bdbc75adcc859ddc858" +
    "dec958dfca57e0cb56e1cc55e2cd54e4ce53e5cf52e6d051e7d150e8d24fe9d34eead34cebd44bedd54aeed649efd748" +
    "f0d846f1d945f2da44f3db42f5dc41f6dd3ff7de3ef8df3cf9e03afbe138fce236fde334fee434fee535fee636fee838"

#: Google's replacement for jet. Still a rainbow — it is not monotone in
#: lightness, so it invents boundaries that are not in the data — but it
#: has none of jet's cyan and yellow flares.
const _TURBO = "30123b32154333184a341b51351e5836215f37246638276d392a733a2d793b2f803c32863d358b3e38913f3b973f3e9c" +
    "4040a24143a74146ac4249b1424bb5434eba4451bf4454c34456c74559cb455ccf455ed34661d64664da4666dd4669e0" +
    "466be3476ee64771e94773eb4776ee4778f0477bf2467df44680f64682f84685fa4687fb458afc458cfd448ffe4391fe" +
    "4294ff4196ff4099ff3e9bfe3d9efe3ba0fd3aa3fc38a5fb37a8fa35abf833adf731aff52fb2f42eb4f22cb7f02ab9ee" +
    "28bceb27bee925c0e723c3e422c5e220c7df1fc9dd1ecbda1ccdd81bd0d51ad2d21ad4d019d5cd18d7ca18d9c818dbc5" +
    "18ddc218dec018e0bd19e2bb19e3b91ae4b61ce6b41de7b21fe9af20eaac22ebaa25eca727eea42aefa12cf09e2ff19b" +
    "32f29835f39438f4913cf58e3ff68a43f78746f8844af8804ef97d52fa7a55fa7659fb735dfc6f61fc6c65fd6969fd66" +
    "6dfe6271fe5f75fe5c79fe597dff5680ff5384ff5188ff4e8bff4b8fff4992ff4796fe4499fe429cfe409ffd3fa1fd3d" +
    "a4fc3ca7fc3aa9fb39acfb38affa37b1f936b4f836b7f735b9f635bcf534bef434c1f334c3f134c6f034c8ef34cbed34" +
    "cdec34d0ea34d2e935d4e735d7e535d9e436dbe236dde037dfdf37e1dd37e3db38e5d938e7d739e9d539ebd339ecd13a" +
    "eecf3aefcd3af1cb3af2c93af4c73af5c53af6c33af7c13af8be39f9bc39faba39fbb838fbb637fcb336fcb136fdae35" +
    "fdac34fea933fea732fea431fea130fe9e2ffe9b2dfe992cfe962bfe932afe9029fd8d27fd8a26fc8725fc8423fb8122" +
    "fb7e21fa7b1ff9781ef9751df8721cf76f1af66c19f56918f46617f36315f26014f15d13f05b12ef5811ed5510ec530f" +
    "eb500eea4e0de84b0ce7490ce5470be4450ae2430ae14109df3f08dd3d08dc3b07da3907d83706d63506d43305d23105" +
    "d02f05ce2d04cc2b04ca2a04c82803c52603c32503c12302be2102bc2002b91e02b71d02b41b01b21a01af1801ac1701" +
    "a91601a71401a41301a112019e10019b0f01980e01950d01920b018e0a018b09028808028507028106027e05027a0403"

#: Black to white. The honest choice for an occupancy grid or a photograph,
#: and `gray_r` is the ink-on-paper direction.
const _GRAY = "0000000101010202020303030404040505050606060707070808080909090a0a0a0b0b0b0c0c0c0d0d0d0e0e0e0f0f0f" +
    "1010101111111212121313131414141515151616161717171818181919191a1a1a1b1b1b1c1c1c1d1d1d1e1e1e1f1f1f" +
    "2020202121212222222323232424242525252626262727272828282929292a2a2a2b2b2b2c2c2c2d2d2d2e2e2e2f2f2f" +
    "3030303131313232323333333434343535353636363737373838383939393a3a3a3b3b3b3c3c3c3d3d3d3e3e3e3f3f3f" +
    "4040404141414242424343434444444545454646464747474848484949494a4a4a4b4b4b4c4c4c4d4d4d4e4e4e4f4f4f" +
    "5050505151515252525353535454545555555656565757575858585959595a5a5a5b5b5b5c5c5c5d5d5d5e5e5e5f5f5f" +
    "6060606161616262626363636464646565656666666767676868686969696a6a6a6b6b6b6c6c6c6d6d6d6e6e6e6f6f6f" +
    "7070707171717272727373737474747575757676767777777878787979797a7a7a7b7b7b7c7c7c7d7d7d7e7e7e7f7f7f" +
    "8080808181818282828383838484848585858686868787878888888989898a8a8a8b8b8b8c8c8c8d8d8d8e8e8e8f8f8f" +
    "9090909191919292929393939494949595959696969797979898989999999a9a9a9b9b9b9c9c9c9d9d9d9e9e9e9f9f9f" +
    "a0a0a0a1a1a1a2a2a2a3a3a3a4a4a4a5a5a5a6a6a6a7a7a7a8a8a8a9a9a9aaaaaaabababacacacadadadaeaeaeafafaf" +
    "b0b0b0b1b1b1b2b2b2b3b3b3b4b4b4b5b5b5b6b6b6b7b7b7b8b8b8b9b9b9babababbbbbbbcbcbcbdbdbdbebebebfbfbf" +
    "c0c0c0c1c1c1c2c2c2c3c3c3c4c4c4c5c5c5c6c6c6c7c7c7c8c8c8c9c9c9cacacacbcbcbcccccccdcdcdcecececfcfcf" +
    "d0d0d0d1d1d1d2d2d2d3d3d3d4d4d4d5d5d5d6d6d6d7d7d7d8d8d8d9d9d9dadadadbdbdbdcdcdcdddddddedededfdfdf" +
    "e0e0e0e1e1e1e2e2e2e3e3e3e4e4e4e5e5e5e6e6e6e7e7e7e8e8e8e9e9e9eaeaeaebebebecececedededeeeeeeefefef" +
    "f0f0f0f1f1f1f2f2f2f3f3f3f4f4f4f5f5f5f6f6f6f7f7f7f8f8f8f9f9f9fafafafbfbfbfcfcfcfdfdfdfefefeffffff"

#: Black through red and yellow to white. Monotone in lightness, unlike the
#: rainbows, and the one heat scale readers already know.
const _HOT = "0b00000d00001000001200001500001800001a00001d00002000002200002500002700002a00002d00002f0000320000" +
    "3500003700003a00003c00003f00004200004400004700004a00004c00004f00005100005400005700005900005c0000" +
    "5f00006100006400006600006900006c00006e00007100007400007600007900007b00007e0000810000830000860000" +
    "8900008b00008e00009000009300009600009800009b00009e0000a00000a30000a50000a80000ab0000ad0000b00000" +
    "b30000b50000b80000ba0000bd0000c00000c20000c50000c80000ca0000cd0000cf0000d20000d50000d70000da0000" +
    "dd0000df0000e20000e40000e70000ea0000ec0000ef0000f20000f40000f70000f90000fc0000ff0000ff0200ff0500" +
    "ff0800ff0a00ff0d00ff1000ff1200ff1500ff1700ff1a00ff1d00ff1f00ff2200ff2500ff2700ff2a00ff2c00ff2f00" +
    "ff3200ff3400ff3700ff3a00ff3c00ff3f00ff4100ff4400ff4700ff4900ff4c00ff4f00ff5100ff5400ff5600ff5900" +
    "ff5c00ff5e00ff6100ff6400ff6600ff6900ff6b00ff6e00ff7100ff7300ff7600ff7900ff7b00ff7e00ff8000ff8300" +
    "ff8600ff8800ff8b00ff8e00ff9000ff9300ff9500ff9800ff9b00ff9d00ffa000ffa200ffa500ffa800ffaa00ffad00" +
    "ffb000ffb200ffb500ffb700ffba00ffbd00ffbf00ffc200ffc500ffc700ffca00ffcc00ffcf00ffd200ffd400ffd700" +
    "ffda00ffdc00ffdf00ffe100ffe400ffe700ffe900ffec00ffef00fff100fff400fff600fff900fffc00fffe00ffff03" +
    "ffff07ffff0bffff0fffff13ffff17ffff1bffff1fffff22ffff26ffff2affff2effff32ffff36ffff3affff3effff42" +
    "ffff46ffff4affff4effff52ffff56ffff5affff5effff61ffff65ffff69ffff6dffff71ffff75ffff79ffff7dffff81" +
    "ffff85ffff89ffff8dffff91ffff95ffff99ffff9dffffa0ffffa4ffffa8ffffacffffb0ffffb4ffffb8ffffbcffffc0" +
    "ffffc4ffffc8ffffccffffd0ffffd4ffffd8ffffdcffffdfffffe3ffffe7ffffebffffeffffff3fffff7fffffbffffff"

#: Diverging blue to red through a light grey, with the two arms matched in
#: lightness so neither side looks heavier. Pair it with `TwoSlopeNorm`.
const _COOLWARM = "3b4cc03c4ec23d50c33e51c53f53c64055c84257c94358cb445acc455cce465ecf485fd14961d24a63d34b64d54c66d6" +
    "4e68d84f69d9506bda516ddb536edd5470de5572df5673e05875e15977e35a78e45b7ae55d7ce65e7de75f7fe86180e9" +
    "6282ea6384eb6485ec6687ed6788ee688aef6a8bef6b8df06c8ff16e90f26f92f37093f37295f47396f57597f67699f6" +
    "779af7799cf87a9df87b9ff97da0f97ea1fa80a3fa81a4fb82a6fb84a7fc85a8fc86a9fc88abfd89acfd8badfd8caffe" +
    "8db0fe8fb1fe90b2fe92b4fe93b5fe94b6ff96b7ff97b8ff98b9ff9abbff9bbcff9dbdff9ebeff9fbfffa1c0ffa2c1ff" +
    "a3c2fea5c3fea6c4fea7c5fea9c6fdaac7fdabc8fdadc9fdaec9fcafcafcb1cbfcb2ccfbb3cdfbb5cdfab6cefab7cff9" +
    "b9d0f9bad0f8bbd1f8bcd2f7bed2f6bfd3f6c0d4f5c1d4f4c3d5f4c4d5f3c5d6f2c6d6f1c7d7f0c9d7f0cad8efcbd8ee" +
    "ccd9edcdd9eccedaebcfdaead1dae9d2dbe8d3dbe7d4dbe6d5dbe5d6dce4d7dce3d8dce2d9dce1dadce0dbdcdedcdddd" +
    "dddcdcdedcdbdfdbd9e0dbd8e1dad6e2dad5e3d9d3e4d9d2e5d8d1e6d7cfe7d7cee8d6cce9d5cbead5c9ead4c8ebd3c6" +
    "ecd3c5edd2c3edd1c2eed0c0efcfbfefcebdf0cdbbf1cdbaf1ccb8f2cbb7f2cab5f2c9b4f3c8b2f3c7b1f4c6aff4c5ad" +
    "f5c4acf5c2aaf5c1a9f5c0a7f6bfa6f6bea4f6bda2f7bca1f7ba9ff7b99ef7b89cf7b79bf7b599f7b497f7b396f7b194" +
    "f7b093f7af91f7ad90f7ac8ef7aa8cf7a98bf7a889f7a688f6a586f6a385f6a283f5a081f59f80f59d7ef59c7df49a7b" +
    "f4987af39778f39577f39475f29274f29072f18f71f18d6ff08b6ef08a6cef886bee8669ee8468ed8366ec8165ec7f63" +
    "eb7d62ea7b60e97a5fe9785de8765ce7745be67259e57058e46e56e36c55e36b54e26952e16751e0654fdf634ede614d" +
    "dd5f4bdc5d4ada5a49d95847d85646d75445d65244d55042d44e41d24b40d1493fd0473dcf453ccd423bcc403acb3e38" +
    "ca3b37c83836c73635c53334c43032c32e31c12b30c0282fbe242ebd1f2dbb1b2cba162bb8122ab70d28b50927b40426"

#: ColorBrewer's diverging red-blue, running red at the bottom to blue at
#: the top. Reverse it when high should be hot.
const _RDBU = "67001f6a011f6d02207003207304217605217906227c07227f0823810823840924870a248a0b258d0c25900d26930e26" +
    "960f279910279c11279f1228a21328a51429a81529ab162aae172ab1182bb3192cb41c2db61f2eb72230b82531ba2832" +
    "bb2a34bd2d35be3036bf3338c13639c2383ac43b3cc53e3dc6413ec84440c94741cb4942cc4c44ce4f45cf5246d05548" +
    "d25849d35a4ad55d4cd6604dd7634fd86551da6853db6b55dc6e57dd7059de735cdf765ee17860e27b62e37e64e48066" +
    "e58368e6866ae8896ce98b6eea8e70eb9172ec9374ee9677ef9979f09c7bf19e7df2a17ff3a481f4a683f5a886f5aa89" +
    "f5ac8bf6af8ef6b191f6b394f7b596f7b799f7b99cf8bb9ef8bda1f8bfa4f9c2a7f9c4a9f9c6acfac8affacab1fbccb4" +
    "fbceb7fbd0b9fcd3bcfcd5bffcd7c2fdd9c4fddbc7fddcc9fdddcbfcdecdfcdfcffce0d0fce2d2fbe3d4fbe4d6fbe5d8" +
    "fbe6dafae7dcfae8defae9dffaeae1f9ebe3f9ede5f9eee7f9efe9f9f0ebf8f1edf8f2eff8f3f0f8f4f2f7f5f4f7f6f6" +
    "f6f7f7f5f6f7f3f5f6f2f5f6f0f4f6eff3f5edf2f5ecf2f5eaf1f5e9f0f4e7f0f4e6eff4e4eef4e3edf3e1edf3e0ecf3" +
    "deebf2ddebf2dbeaf2dae9f2d8e9f1d7e8f1d5e7f1d4e6f1d2e6f0d1e5f0cfe4efcce2efcae1eec7e0edc5dfecc2ddec" +
    "c0dcebbddbeabbdaeab8d8e9b6d7e8b3d6e8b1d5e7aed3e6acd2e5a9d1e5a7d0e4a5cee3a2cde3a0cce29dcbe19bc9e0" +
    "98c8e096c7df93c6de90c4dd8dc2dc8ac0db87beda84bcd981bad87eb8d77bb6d678b4d575b2d471b0d36eaed26bacd1" +
    "68abd065a9cf62a7ce5fa5cd5ca3cb59a1ca569fc9529dc84f9bc74c99c64997c54695c44393c34291c2408fc13f8ec0" +
    "3e8cbf3c8abe3b88be3a87bd3885bc3783bb3681ba3480b9337eb8327cb7307ab62f79b52e77b52c75b42b73b32a71b2" +
    "2870b1276eb0266caf246aae2369ad2267ac2065ab1f63a81e61a51d5fa21c5c9f1b5a9c1a5899195696185493175290" +
    "15508d144e8a134c8712498411478110457e0f437b0e41790d3f760c3d730a3b7009386d08366a073467063264053061"

#: Every built-in map, keyed by its lowercased name.
let _TABLES: dict[str, str] = {
    "viridis": _VIRIDIS,
    "plasma": _PLASMA,
    "inferno": _INFERNO,
    "magma": _MAGMA,
    "cividis": _CIVIDIS,
    "turbo": _TURBO,
    "gray": _GRAY,
    "hot": _HOT,
    "coolwarm": _COOLWARM,
    "rdbu": _RDBU,
}

#: The canonical spelling of each map, keyed by its lowercased name.
let _CANONICAL: dict[str, str] = {
    "viridis": "viridis",
    "plasma": "plasma",
    "inferno": "inferno",
    "magma": "magma",
    "cividis": "cividis",
    "turbo": "turbo",
    "gray": "gray",
    "hot": "hot",
    "coolwarm": "coolwarm",
    "rdbu": "RdBu",
}

#: The built-in map names, in the order `names()` reports them.
pub let COLORMAP_NAMES: list[str] = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "turbo",
    "gray",
    "hot",
    "coolwarm",
    "RdBu",
]

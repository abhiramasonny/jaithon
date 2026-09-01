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
from .colormap.norm import LinearNorm, LogNorm, Norm, TwoSlopeNorm

export { COLORMAP_NAMES }
export { LinearNorm, LogNorm, Norm, TwoSlopeNorm }

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

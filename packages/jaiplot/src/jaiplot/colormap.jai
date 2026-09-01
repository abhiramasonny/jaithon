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
from .colormap.tables import COLORMAP_NAMES
from .colormap.norm import LinearNorm, LogNorm, Norm, TwoSlopeNorm
from .colormap.map import COLORMAP_LEVELS, Colormap, SWATCH_BANDS

export { COLORMAP_NAMES }
export { LinearNorm, LogNorm, Norm, TwoSlopeNorm }
export { COLORMAP_LEVELS, Colormap, SWATCH_BANDS }

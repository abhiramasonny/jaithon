#: The contract every plot type answers to.
#:
#: This lives apart from `mod.jai` because the concrete series implement it and
#: `mod.jai` imports them. Putting the trait here lets both sides import it
#: without a cycle.
from ..frame import Frame
from ..geom import Rect
from ..primitive import Primitive
from ..style import Color

#: One plotted layer on a set of axes.
#:
#: `Axes` holds a `list[Series]` and never learns what a bar is. Adding a plot
#: type means implementing these five methods in one new file.
pub trait Series {
    #: The data-space box this series occupies, or null when it has no
    #: drawable point at all.
    #:
    #: An axis unions these to range itself, so returning null keeps an empty
    #: series from dragging the limits somewhere arbitrary.
    fn extent(self) -> Rect?

    #: The primitives that draw this series inside `frame`.
    #:
    #: A sample the frame cannot place is dropped rather than drawn at a made-up
    #: position, so a gap in the data reads as a gap in the picture.
    fn draw(self, frame: Frame) -> list[Primitive]

    #: The legend entry, or null to stay out of the legend.
    fn label(self) -> str?

    #: The primitives filling a legend swatch of `box`.
    fn swatch(self, box: Rect) -> list[Primitive]

    #: Take `color` as this series' colour unless one was set explicitly.
    #:
    #: An axis calls this as a series is added, handing over the next colour
    #: from the theme's cycle. A series given a colour at construction ignores
    #: it, which is what makes an explicit colour stick.
    fn adopt(self, color: Color) -> void
}

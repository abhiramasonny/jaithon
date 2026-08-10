#: The plot types that ship, and the contract they answer to.
#:
#: `Axes` holds a `list[Series]` and never learns what a bar is. Every plot type
#: answers the same five questions: how much data space do you need, what
#: primitives draw you, what should the legend call you, what does your swatch
#: look like, and will you take this colour. Adding a plot type touches one
#: file.
#:
#: ```
#: let line = Line([0.0, 1.0], [0.0, 1.0], label: "a")
#: line.label()        # "a"
#: ```
from .base import Series
from .line import FillBetween, Line, Step, finite_extent
from .scatter import Scatter
from .bar import Bar
from .hist import Hist
from .field import Grid, PoseCloud, Rays, WEIGHT_BUCKETS

export { Series, FillBetween, Line, Step, finite_extent, Scatter, Bar, Hist }
export { Grid, PoseCloud, Rays, WEIGHT_BUCKETS }

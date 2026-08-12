#: The keyed box naming each series.
#:
#: A legend is laid out from the labels it is given: the box is exactly wide
#: enough for the longest entry and tall enough for all of them, then placed in
#: whichever corner of the plot area is asked for. It draws itself over the
#: series, on an opaque panel, because a legend that lets gridlines show through
#: is harder to read than one that hides a little data.
from .geom import Point, Rect
from .metrics import text_height, text_width
from .primitive import Primitive
from .series import Series
from .style import HorizontalAnchor, Paint, Stroke, TextStyle, Theme, VerticalAnchor

#: Padding between the legend frame and its contents, in device pixels.
pub const PADDING: float = 6.0

#: Width of the swatch drawn beside each label.
pub const SWATCH_WIDTH: float = 22.0

#: Vertical gap between two entries.
pub const ROW_GAP: float = 4.0

#: Gap between a swatch and the label it belongs to.
pub const SWATCH_GAP: float = 6.0

#: Which corner of the plot area a legend sits in.
pub enum LegendCorner { TopLeft, TopRight, BottomLeft, BottomRight }

#: Draw a legend for `entries` inside `area`, in `theme`.
#:
#: Entries are the series that returned a label; an unlabelled series is the
#: caller's way of keeping something out of the legend, so it never reaches
#: here. An empty list draws nothing at all rather than an empty frame.
pub fn draw_legend(
    entries: list[Series],
    area: Rect,
    theme: Theme,
    corner: LegendCorner = LegendCorner.TopRight,
    scale: int = 1
) -> list[Primitive] {
    if entries.len() == 0 { return [] }

    var labels = []
    for series in entries { labels.push(series.label() ?? "") }

    let row = text_height("A", scale)
    var widest = 0.0
    for label in labels { widest = max(widest, text_width(label, scale)) }

    let width = PADDING * 2.0 + SWATCH_WIDTH + SWATCH_GAP + widest
    let rows = labels.len().to_float()
    let gaps = (labels.len() - 1).to_float()
    let height = PADDING * 2.0 + row * rows + ROW_GAP * gaps
    let box = _corner_box(area, width, height, corner)

    var out = []
    out.push(_panel(box, theme))
    for (index, series) in entries.enumerate() {
        let top = box.bottom() + PADDING + (row + ROW_GAP) * index.to_float()
        let swatch = Rect(box.left() + PADDING, top, box.left() + PADDING + SWATCH_WIDTH, top + row)
        for primitive in series.swatch(swatch) { out.push(primitive) }
        let text_at = Point(swatch.right() + SWATCH_GAP, top)
        let style = TextStyle(
            theme.text,
            scale: scale,
            horizontal: HorizontalAnchor.Start,
            vertical: VerticalAnchor.Top
        )
        out.push(Primitive.Text(text_at, labels[index], style))
    }
    return out
}

#: The frame and backing panel the entries sit on.
fn _panel(box: Rect, theme: Theme) -> Primitive {
    let corners = [
        Point(box.left(), box.bottom()),
        Point(box.right(), box.bottom()),
        Point(box.right(), box.top()),
        Point(box.left(), box.top()),
    ]
    # Slightly transparent: enough to hide a gridline, little enough that a
    # series running underneath is still visible as context.
    let fill = Paint.solid(theme.background.with_alpha(0.88))
    return Primitive.Polygon(corners, fill, Stroke(theme.spine, width: 1.0))
}

#: Place a box of this size in the named corner of `area`, inset by the padding.
#:
#: Device y grows downward, so `bottom()` of a device rectangle is its visual
#: top. `TopLeft` therefore takes the smaller y.
fn _corner_box(area: Rect, width: float, height: float, corner: LegendCorner) -> Rect {
    let inset = PADDING
    let left = match corner {
        LegendCorner.TopLeft => area.left() + inset,
        LegendCorner.BottomLeft => area.left() + inset,
        LegendCorner.TopRight => area.right() - inset - width,
        LegendCorner.BottomRight => area.right() - inset - width,
    }
    let top = match corner {
        LegendCorner.TopLeft => area.bottom() + inset,
        LegendCorner.TopRight => area.bottom() + inset,
        LegendCorner.BottomLeft => area.top() - inset - height,
        LegendCorner.BottomRight => area.top() - inset - height,
    }
    return Rect(left, top, left + width, top + height)
}

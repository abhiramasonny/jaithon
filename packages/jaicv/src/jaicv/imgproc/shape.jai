#: What a contour is: its moments, its area and perimeter, the shapes that
#: bound it, and the tests that describe it.
#:
#: These take a contour as a list of points — what `find_contours` returns —
#: and they follow OpenCV's formulas rather than the textbook ones wherever the
#: two differ, because the differences are visible. A contour's moments come
#: from Green's theorem over the polygon, not from summing pixels; the area is
#: the shoelace sum and so is smaller than the pixel count by roughly half the
#: perimeter; and `bounding_rect` is inclusive of the far edge.
#:
#: ```
#: let (contours, _) = find_contours(mask, RETR_EXTERNAL)
#: for contour in contours {
#:     if contour_area(contour) < 50.0 { continue }
#:     let box = bounding_rect(contour)
#:     rectangle_rect(frame, box, Scalar(0.0, 255.0, 0.0), 1)
#: }
#: ```
#:
#: The implementation lives in `shape/`; this re-exports it.
from .shape.approx import approx_poly_dp, point_polygon_test
from .shape.enclosing import box_points, min_area_rect, min_enclosing_circle
from .shape.fit import CONTOURS_MATCH_I1, CONTOURS_MATCH_I2, CONTOURS_MATCH_I3
from .shape.fit import DIST_FAIR, DIST_HUBER, DIST_L1, DIST_L2, DIST_WELSCH
from .shape.fit import fit_ellipse, fit_line, match_shapes
from .shape.hull import HULL_COORD_LIMIT, convex_hull, convex_hull_indices
from .shape.hull import is_contour_convex
from .shape.moments import Moments, arc_length, bounding_rect, contour_area
from .shape.moments import hu_moments, moments, moments_f, moments_image

export { approx_poly_dp, point_polygon_test }
export { box_points, min_area_rect, min_enclosing_circle }
export { CONTOURS_MATCH_I1, CONTOURS_MATCH_I2, CONTOURS_MATCH_I3 }
export { DIST_FAIR, DIST_HUBER, DIST_L1, DIST_L2, DIST_WELSCH }
export { fit_ellipse, fit_line, match_shapes }
export { HULL_COORD_LIMIT, convex_hull, convex_hull_indices, is_contour_convex }
export { Moments, arc_length, bounding_rect, contour_area }
export { hu_moments, moments, moments_f, moments_image }

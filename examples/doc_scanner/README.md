# doc_scanner

Flatten a photographed page into a square-on scan. No weights, no downloads —
every step is a `jaicv` call.

```bash
jaithon run examples/doc_scanner/scan.jai photo.jpg scan.png
```

Writes `scan.png` and `scan_outline.png`, the second showing which
quadrilateral was chosen and in what corner order, so a wrong crop is obvious
rather than mysterious.

## How it works

```
grey -> blur -> Canny -> dilate -> contours -> largest convex quadrilateral
     -> perspective transform -> adaptive threshold
```

Two steps carry it. `approx_poly_dp` reduces each contour to its corners and
the page is the largest four-cornered convex one; then the corners are ordered
top-left, top-right, bottom-right, bottom-left, because
`get_perspective_transform` maps corner one to corner one and a rotated
ordering gives a rotated or mirrored scan. The ordering uses the sums and
differences of the coordinates, which holds for any rotation short of 45
degrees and needs no trigonometry.

The dilation after Canny matters more than it looks: a page border broken by a
shadow does not close, and a contour that does not close is never a
quadrilateral.

On a synthetic 800x900 photo of a skewed invoice it recovers the four corners
to within a pixel of where they were warped to, in about 70 ms.

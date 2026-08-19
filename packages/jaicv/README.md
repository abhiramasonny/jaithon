# jaicv

Computer vision with OpenCV's API, on the GPU.

```jai
import jaicv as cv

let camera = cv.VideoCapture(0)
loop {
    let (ok, frame) = camera.read()
    if not ok { break }
    let gray = cv.cvt_color(frame, cv.COLOR_BGR2GRAY)
    let edges = cv.canny(cv.gaussian_blur(gray, cv.Size(5, 5), 1.2), 80.0, 200.0)
    let (contours, _hierarchy) = cv.find_contours(edges, cv.RETR_EXTERNAL)
    for contour in contours {
        if cv.contour_area(contour) < 200.0 { continue }
        cv.rectangle_rect(frame, cv.bounding_rect(contour), cv.Scalar(0.0, 255.0, 0.0), 2)
    }
    cv.imshow("camera", frame)
    if cv.wait_key(1) == 27 { break }
}
cv.destroy_all_windows()
```

Names, argument order, and constants follow OpenCV, in snake case: `cvt_color`
for `cvtColor`, `COLOR_BGR2GRAY` unchanged. A program written against OpenCV
translates line for line.

## What it is

An image is a `Mat`: dense, row-major, channels interleaved, BGR by
convention, and backed by one float32 device buffer. Every filter, warp,
threshold, and morphology runs as a Metal kernel, so a chain of them stays on
the GPU from `imread` to `imwrite` and only comes back when something asks for
a value. The parts that are inherently sequential — border following, flood
fill, the Hough accumulators, cascade evaluation — run on the host and read the
image through a mirror that the device writes keep current.

`jaicv` needs a Metal device. `cv.is_available()` says whether there is one.

## How closely it matches OpenCV

The claim is checked rather than asserted. `tests/oracle/generate.py` runs the
real OpenCV, records every input and output to `tests/oracle/cases.txt`, and
`tests/test_against_opencv.jai` replays each one and compares. Seven hundred
cases pass, most of them exactly.

Where a case is not exact the test says by how much and the reason is written
down next to the tolerance. In short:

- **Exact**: colour conversion, thresholding, morphology, Canny, contour
  following and its hierarchy, moments and every shape measure, connected
  components, distance transform, histograms, CLAHE, FAST, the PNG and BMP
  codecs both directions, drawing at one pixel wide, filled polygons, the
  Fourier transforms, PCA, non-maximum suppression.
- **Within a level**: anything that goes through OpenCV's fixed-point
  resampling — `resize`, `remap`, `warp_affine`, `undistort` — where OpenCV
  rounds coordinates to a fraction of a pixel and this does not.
- **Within a fit**: calibration and pose, which agree on the answer to a
  thousandth of a pixel of reprojection error but split it slightly differently
  between the principal point and the last distortion term.
- **As accurate**: `StereoBM` and `StereoSGBM` recover a known disparity on a
  rectified pair as closely as OpenCV's do, including the same gradual
  transition across a depth step that a square window produces.
- **Better than OpenCV's**: `QRCodeDetector` reads every symbol OpenCV 5.0.0's
  encoder produces from version 1 to 40 — including the ones OpenCV's own
  detector cannot — and OpenCV reads what this writes. The one disagreement is
  OpenCV's: its encoder puts version 21's last alignment centre at module 92
  where the standard says 94.
- **Documented differences**: a stroke wider than one pixel and a filled
  ellipse disagree on a boundary pixel here and there; `LINE_AA` computes
  coverage from distance rather than from OpenCV's slope tables; ORB and BRIEF
  use a generated sampling pattern rather than OpenCV's learned table, so their
  descriptors compare against each other but not against OpenCV's; Farneback
  flow follows the paper rather than OpenCV's tuning.
- **Exact**: `grab_cut` reproduces OpenCV's segmentation pixel for pixel —
  intersection-over-union 1.0000 and every pixel agreeing, on a flat rectangle
  over noise and on a textured ellipse over noise, at three and at four
  iterations. The mixture models, the smoothness term and the min cut all have
  to agree for that to happen, so it is a stronger check than it looks.
- **Within float32**: `emd` finds the same optimum OpenCV's does — the
  transportation problem has one — but not the same digits, because OpenCV
  carries its costs and flows in float32 and this carries them in float64.
  Over sixty random signature pairs across all three ground distances and
  both equal and unequal total weights, the worst disagreement is 4.7e-7
  relative, 5.2e-7 absolute.
- **Exact**: every one of the twenty-two colour maps, on all 256 levels, for
  one- and three-channel input alike. They are OpenCV's own tables, recorded
  from it by `tools/colormaps_to_jai.py` rather than written out as formulas,
  which is what makes them exact — OpenCV interpolates sixty-four control
  points even for the maps that have a formula behind them, so a formula lands
  a level away nearly everywhere and nowhere near at all on `RAINBOW`, `PINK`
  and `HOT`. The one divergence is depth: OpenCV refuses anything but 8-bit,
  where this rounds and clamps whatever it is given into the 0..255 the tables
  are indexed by.
- **Same segments, quieter on noise**: the line segment detector returns the
  same segments OpenCV's does, endpoint for endpoint to a tenth of a pixel, on
  rectangles, diagonals and triangles, and switches from finding nothing to
  finding everything at the same contrast OpenCV does. On an image of pure
  random noise OpenCV returns a dozen segments and this returns none — the
  a-contrario test exists to reject exactly that, so the difference is in this
  one's favour, but it is a difference.
- **Same size, different letters**: `put_text` draws the public-domain Hershey
  fonts. OpenCV draws re-derived tables of its own — measured against every
  font in the Hershey distribution, no face of OpenCV's matches on glyph
  widths, missing by about a unit and a half a glyph with no exact matches at
  all — so the letterforms here are not OpenCV's and cannot be. What does
  match is the part layout code depends on: a capital is the same number of
  pixels tall at the same `font_scale` for every face, so text occupies the
  same band of the image. Strings come out roughly a tenth wider, because
  Hershey's own glyphs are.

## What is here

| Area | What it covers |
| --- | --- |
| `core` | `Mat` with views and ROIs, arithmetic, statistics, `merge`/`split`, linear algebra including SVD, DFT and DCT, k-means, PCA, sorting, random fills |
| `imgproc` | colour conversion, resize and warping, filtering, thresholding, morphology, Canny, drawing, contours, shape analysis, segmentation, histograms, template matching, corners, Hough, colour maps, phase correlation, mean-shift filtering; vector text in the Hershey fonts |
| `imgcodecs` | PNG, BMP, PNM, and JPEG, reading and writing, with `imread`/`imwrite`/`imencode`/`imdecode` |
| `videoio` | camera capture through AVFoundation, and AVI reading and writing, MJPEG or uncompressed |
| `highgui` | `imshow`, `wait_key`, windows, mouse callbacks, trackbars |
| `features2d` | FAST, ORB, BRIEF, blob detection, brute-force matching, keypoint and match drawing |
| `objdetect` | cascade classification, reading OpenCV's own trained XML; histograms of oriented gradients; QR codes, read and written, every version and correction level |
| `calib3d` | Rodrigues, homography, affine estimation, projection, distortion, triangulation, `solve_pnp`, `calibrate_camera`, chessboard detection, fundamental and essential matrices, pose recovery, epipolar lines, stereo rectification, block and semi-global matching, reprojection to 3D |
| `video` | Lucas-Kanade and Farneback optical flow, MOG2 and KNN background subtraction, mean shift, CamShift, Kalman |
| `photo` | non-local means denoising, inpainting, edge-preserving smoothing |
| `ml` | nearest neighbours, naive Bayes, support vector machine, logistic regression, decision trees, random forest, multilayer perceptron |
| `dnn` | `blob_from_image`, non-maximum suppression, and a `Net` that fronts a jaitensor model |

## What is not here

Named so that nobody has to find out by trying:

- **Codecs**: TIFF, WebP, JPEG 2000, EXR, GIF. PNG interlacing.
- **objdetect**: barcodes, micro QR, the DNN face detector, and the
  coefficients of OpenCV's pre-trained people detector — `HOGDescriptor`
  computes the descriptor and takes a detector, but does not ship one. A kanji
  QR segment comes back as its Shift-JIS bytes rather than as text.
- **calib3d**: fisheye and circle-grid patterns. `find_fundamental_mat` and
  `find_essential_mat` fit by the normalised eight-point algorithm inside
  RANSAC, so eight correspondences are needed where OpenCV's seven- and
  five-point minimal solvers need fewer. `stereo_rectify` computes the same
  rotations OpenCV does but keeps the cameras' own principal point, where
  OpenCV recomputes one by fitting the largest usable rectangle inside the
  rectified border — the `alpha` parameter, which is not here.
- **dnn**: reading ONNX or Caffe. A network is built and run in jaitensor;
  `Net` is the adapter, not a runtime.
- **imgproc**: guided filter, superpixels and structured edge detection, which
  are OpenCV's contrib module rather than its core.
- **videoio**: any container other than AVI, and any codec inside one other
  than motion JPEG and the uncompressed forms. `VideoCapture` takes a path as
  well as a camera index, indexes the file when it opens and decodes a frame
  when it is asked for, so seeking by frame is exact and costs nothing.

## Camera access

macOS asks before a program may use the camera, and the ask is made of the
binary that runs it. If `cv.permission()` comes back denied, grant it in System
Settings under Privacy and Security, Camera. `cv.PERMISSION_HELP` carries the
same words for printing.

## Tests

```bash
JAITHON_PATH=lib ./jaithon test packages/jaicv/tests/
```

`test_jaicv.jai` holds what can be checked without OpenCV — `Mat` semantics,
arithmetic, codec round trips, video round trips, clustering.
`test_reachable.jai` calls every exported entry point once and checks the shape
of what comes back; it exists because more than half the library had no caller
in any test, and two functions turned out to raise on their first real use.
`test_against_opencv.jai` replays the recorded cases. Regenerate those with
OpenCV installed:

```bash
python packages/jaicv/tests/oracle/generate.py
```

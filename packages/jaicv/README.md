# jaicv

Computer vision with OpenCV's API, on the GPU.

```jai
import jaicv as cv

let camera = cv.VideoCapture(0)
while true {
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
- **Documented differences**: a stroke wider than one pixel and a filled
  ellipse disagree on a boundary pixel here and there; `LINE_AA` computes
  coverage from distance rather than from OpenCV's slope tables; ORB and BRIEF
  use a generated sampling pattern rather than OpenCV's learned table, so their
  descriptors compare against each other but not against OpenCV's; Farneback
  flow follows the paper rather than OpenCV's tuning; the colour maps written
  as formulas match, the ones OpenCV ships as tables do not.

## What is here

| Area | What it covers |
| --- | --- |
| `core` | `Mat` with views and ROIs, arithmetic, statistics, `merge`/`split`, linear algebra, DFT and DCT, k-means, PCA, sorting, random fills |
| `imgproc` | colour conversion, resize and warping, filtering, thresholding, morphology, Canny, drawing, contours, shape analysis, segmentation, histograms, template matching, corners, Hough, colour maps, phase correlation, mean-shift filtering |
| `imgcodecs` | PNG, BMP, PNM, and JPEG, reading and writing, with `imread`/`imwrite`/`imencode`/`imdecode` |
| `videoio` | camera capture through AVFoundation, and AVI reading and writing, MJPEG or uncompressed |
| `highgui` | `imshow`, `wait_key`, windows, mouse callbacks, trackbars |
| `features2d` | FAST, ORB, BRIEF, blob detection, brute-force matching, keypoint and match drawing |
| `objdetect` | cascade classification, reading OpenCV's own trained XML; histograms of oriented gradients |
| `calib3d` | Rodrigues, homography, affine estimation, projection, distortion, triangulation, `solve_pnp`, `calibrate_camera`, chessboard detection |
| `video` | Lucas-Kanade and Farneback optical flow, MOG2 and KNN background subtraction, mean shift, CamShift, Kalman |
| `photo` | non-local means denoising, inpainting, edge-preserving smoothing |
| `ml` | nearest neighbours, naive Bayes, support vector machine, logistic regression, decision trees, random forest, multilayer perceptron |
| `dnn` | `blob_from_image`, non-maximum suppression, and a `Net` that fronts a jaitensor model |

## What is not here

Named so that nobody has to find out by trying:

- **Codecs**: TIFF, WebP, JPEG 2000, EXR, GIF. PNG interlacing.
- **objdetect**: QR and barcode reading, the DNN face detector, and the
  coefficients of OpenCV's pre-trained people detector — `HOGDescriptor`
  computes the descriptor and takes a detector, but does not ship one.
- **calib3d**: fisheye, stereo rectification and matching, `find_essential_mat`
  and pose recovery, circle-grid patterns.
- **dnn**: reading ONNX or Caffe. A network is built and run in jaitensor;
  `Net` is the adapter, not a runtime.
- **imgproc**: GrabCut, `EMD`, line segment detection, the Viridis family of
  colour maps.
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

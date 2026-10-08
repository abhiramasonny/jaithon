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
- **Runs the same numbers**: `read_net_from_onnx` imports a network exported by
  PyTorch and `run_graph` reproduces PyTorch's own output. A non-square
  residual CNN — convolution, batch norm, ReLU, a skip connection, pooling,
  flatten, a linear layer and softmax on a 9x15 input — agrees to every printed
  digit. Non-square on purpose: a square test cannot catch an NCHW/OIHW
  transposition, which is the failure mode an importer actually has.
  `read_net_from_tensorflow` and `read_net_from_caffe` map onto the same
  operator vocabulary and the same executor, and `read_torch_state_dict` reads
  a `.pt` straight — every tensor of a checkpoint, including non-contiguous
  views, zero-dimensional and empty tensors, and int64, uint8, bool and
  float64, matches what torch reports.
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
| `dnn` | `blob_from_image`, non-maximum suppression, a `Net` that fronts a jaitensor model, a protobuf reader, a graph executor over ~68 ONNX operators, and importers for ONNX, Caffe, TensorFlow frozen graphs and PyTorch checkpoints |

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
- **dnn**: a pretrained model zoo. The importers below read the files; nobody
  ships you the weights.
- **imgproc**: guided filter, superpixels and structured edge detection, which
  are OpenCV's contrib module rather than its core.
- **videoio**: any container other than AVI, and any codec inside one other
  than motion JPEG and the uncompressed forms. `VideoCapture` takes a path as
  well as a camera index, indexes the file when it opens and decodes a frame
  when it is asked for, so seeking by frame is exact and costs nothing.

## Native paths

Contours, the shapes fitted to them and the drawing of both run their inner
loops in runtime primitives (`src/runtime/builtins/media/`), each a step-for-step
copy of the Jaithon it replaces, which is kept as the fallback and as the
reference the primitive is checked against. Each is behind a switch, read once,
on unless set to `0`, so the two can be compared in one binary:

| Switch | What it moves to a primitive |
|---|---|
| `JAICV_NATIVE_BORDERS` | `find_contours`' raster scan and border walk (`grid_borders`) |
| `JAICV_NATIVE_POINTS` | making `find_contours`' `Point`s (`grid_border_points`) |
| `JAICV_NATIVE_STROKES` | `draw_contours` outlines, thick and one pixel wide (off, a hairline goes through `polylines`, so `JAICV_NATIVE_LINES=0` too to take it off the primitive) |
| `JAICV_NATIVE_LINES` | `polylines`, hard-edged and anti-aliased |
| `JAICV_NATIVE_FILL` | `fill_poly`, and so `draw_contours` with `FILLED` |
| `JAICV_NATIVE_HULL` | `convex_hull` (`JAICV_HULL_COLUMNS=0` sorts instead of bucketing) |
| `JAICV_NATIVE_BOX` | `min_area_rect` and `min_enclosing_circle` |
| `JAICV_NATIVE_APPROX` | `approx_poly_dp` |
| `JAICV_NATIVE_MEASURES` | `contour_area`, `moments`, `match_shapes` |
| `JAICV_NATIVE_FIT` | `fit_ellipse`, `fit_line` |

Device paths are switched the same way:

| Switch | What it does when on |
|---|---|
| `JAICV_CC_DEVICE_NUMBERING` | `connected_components` numbers its labels on the device; off, the roots are read back and sorted on the host |
| `JAICV_GF_SELECT` | `good_features_to_track` ranks only the strongest candidates it can use; off, it ranks every one |
| `JAICV_GF_SURVEY` | `good_features_to_track` finds its floor and its cut in one pass and one read back; off, `min_max` and then a count against the floor |
| `JAICV_DEVICE_LINALG` | `core.linalg.float_matmul` at a million multiply-adds or more, `mul_transposed`, `calc_covar_matrix` and `PCA`'s covariance and projections go through jaitensor's GEMM in float32, cut into pieces too small for its tuner to time; off, the host loops in double precision. `core.linalg.matmul`, which geometry's least-squares fits use, sums in double either way |
| `JAICV_KMEANS_STAGED` | `kmeans` assigns samples of eight or more columns with the centres staged in threadgroup memory, reading each sample once; off, once per centre. Same labels bit for bit |
| `JAICV_KMEANS_IN_PLACE` | `kmeans` reads a continuous matrix's samples where they lie; off, through a host list and a fresh upload |
| `JAICV_LOGISTIC_DEVICE` | `LogisticRegression.train` forms the gradient on the device once the samples hold 65,536 values or more; off, on the host |
| `JAICV_LOGISTIC_RESIDENT` | `LogisticRegression.train`'s device descent keeps the weights on the device, in float32, and reads nothing back until the last step, for training sets of up to 2^25 values (it holds four copies of them); off, or above that, each step's gradient comes back and the weights are updated on the host in double |

## Image decoding

The JPEG and PNG decoders are Jaithon, and each has a fast path in front of a
reference decoder. The fast path gives the same bytes as the reference, not a
picture within a unit of it: same float operations, same order. Anything it
cannot reproduce exactly goes to the reference -- a JPEG that reads past a
restart marker or the end of its scan, an odd sampling layout, a damaged
Huffman table; a palette, sub-byte, sixteen-bit or interlaced PNG.

| Switch | What it does when on |
|---|---|
| `JAICV_FAST_JPEG` | `jpeg.scan_fast`: unstuffed data under a 56-bit accumulator with a nine-bit Huffman lookahead, an IDCT that skips zero columns and sums a coefficient's eight outputs side by side, integer planes, 4:2:0 chroma doubled inside a table-driven colour conversion, and the picture packed straight into bytes for `Mat.from_bytes`; off, the reference loop in `jpeg.scan` |
| `JAICV_FAST_PNG` | `png.decode_fast` for eight-bit grey, grey-alpha, RGB and RGBA: rows unfiltered in place into one int list, red and blue swapped in it, packed into bytes for `Mat.from_bytes`; off, the reference, which builds a float list a pixel at a time for `Mat.from_list` |

`jpeg.decode_with(data, fast)` and `png.decode_with(data, fast)` take either
path in one process; `tests/test_imgcodecs.jai` holds them to each other on
made-up images, damaged ones included. Real files were checked by hash, every
image byte for byte against the reference decoder: the first 300 COCO val2017
JPEGs (all 4:4:4, two of them grey), all 120 frames of a 720p 4:2:0 MJPEG AVI,
the first 1000 HASYv2 PNGs, and 192 files written by OpenCV at 4:4:4, 4:2:2,
4:2:0, 4:1:1 and 4:4:0, with and without restart intervals, grey, odd sizes,
and PNGs of every colour type and depth.

Measured on an M2 Max under `scripts/bench/gpu_lock.sh`, bytes already in
memory, median of five rounds after a second of warm-up, against OpenCV 4 with
`cv2.setNumThreads(1)` on the same files:

| | fast | reference | cv2 |
|---|---|---|---|
| COCO JPEG, 640x426 average, 20 images | 17.0 ms | 57.9 ms | 1.61 ms |
| HASY 32x32 RGB PNG, 500 images | 23.3 us | 265.6 us | 10.1 us |
| HASY `imread(path, IMREAD_GRAYSCALE)` | 53.0 us | 300.6 us | 36.2 us |

The PNG is at the per-image floor: the rest of `imread` is the file read and
the device buffer the `Mat` lives in, and `IMREAD_GRAYSCALE` adds a
`cvt_color` dispatch. The whole HASY set (168,233 files) loads in about 9 s,
from 50 s.

The JPEG is not. A decode is now about 40,000 interpreted instructions and a
hundred allocations (the reference: 1.1 million and 4,000), so the JIT runs
nearly all of it, and the time is what the compiled code costs. On a COCO
image about 4 ms of the 17 is the Huffman decode, 7.5 the IDCT, 2 the colour
conversion, 1.2 packing the list into bytes, and the rest unstuffing and
allocation. Each step is held to the reference's arithmetic, so there is no
cheaper transform to switch to -- the IDCT has to add the same terms in the
same order -- and what is left is the compiled code's cost per list access
and per float operation: about 0.8 ns a multiply-add in the best case. The
limits it was written around, all in the JIT:

- a function of more than eight arguments never reaches the function tier, so
  the per-block work is methods on `FastScan` reading fields;
- a body is held to 20,000 instructions: the Huffman decode compiles to about
  18,400, which is why it is one method and the IDCT two;
- a call to a function or method that has not run yet stops the compile of
  what follows it, so rare paths -- long Huffman codes, restarts -- are
  written out in line rather than called;
- a return inside a loop keeps that loop off stack replacement, and a local
  declared inside a loop body is null on the way in, which a loop form will
  not take: every loop here declares its locals first and ends on a flag;
- `bytes` is immutable, so the picture is built as an int list and packed.

## Device memory

Device memory is not garbage collected, so every operation releases the
intermediates it makes, and small constant matrices -- filter taps, a warp's
inverse, the default structuring element -- are made once and shared
(`cached_weights` in `imgproc/common.jai`). The shared matrices are not
growth-free but bounded: the cache keeps two generations of 256, and a call
whose values change every time -- a rotating warp, a new sigma each frame --
fills it until the older generation is released. A matrix handed out is never
in the generation being released, so a call keeps the taps it fetched.
`std.gpu.live_buffers()` counts the buffers still allocated;
`tests/test_releases.jai` holds operations repeated with the same arguments to
zero growth with it, and `BENCH_LEAK_CALLS=100 ./jaithon run
tests/bench/jaicv/imgproc.jai` reports the growth of every bench row.

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

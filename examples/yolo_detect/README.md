# yolo_detect

Real YOLOv8 object detection, running entirely in Jaithon. The ONNX file is
imported by `jaicv.dnn.onnx`, executed by `jaicv.dnn.graph` on jaitensor GPU
tensors, and drawn with `jaicv`. Nothing shells out to Python at run time.

## Setup

```bash
examples/yolo_detect/fetch_model.sh
```

That exports `yolov8n.onnx` (~12 MB) next to the script, using a throwaway `uv`
environment. The weights are not committed: they are Ultralytics' to
distribute, and YOLOv8 is AGPL-3.0 — worth reading before you build on it.

## Bigger models

`fetch_model.sh` takes a model name, and the detector reads the geometry out of
the ONNX file rather than assuming it, so nothing here needs changing to run a
different one:

```bash
examples/yolo_detect/fetch_model.sh yolov8s      # 43 MB
examples/yolo_detect/fetch_model.sh yolov8x      # ~260 MB
examples/yolo_detect/fetch_model.sh yolov8x6     # exported at 1280, not 640
```

```bash
YOLO_MODEL=examples/yolo_detect/yolov8s.onnx \
    jaithon run examples/yolo_detect/detect.jai photo.jpg out.png
```

Two numbers come out of the file: the input size the model was exported at, and
the number of output rows, which is four box coordinates plus one score per
class. That covers the whole YOLOv8 family, the `-p6` variants at 1280, the
`yolo11` models, and anything fine-tuned on a class set other than COCO's 80 —
a model with different classes will report `class 0`, `class 1` and so on,
since the names in `coco.jai` no longer describe it.

Measured on `bus.jpg`, same code path, only `YOLO_MODEL` different:

| model | inference | best person |
| --- | --- | --- |
| yolov8n 640 | 217 ms | 0.891 |
| yolov8s 640 | 246 ms | 0.915 |
| yolov8n 1280 | 312 ms | 0.809 |

Loading the ONNX file now takes about 24 ms for yolov8n and 66 ms for yolov8s
(it was 1.8 s and 6.1 s): the weights are copied in blocks and decoded natively.

A file exported with dynamic axes declares no shape at all; the detector says
so and falls back to 640 and 80 classes rather than guessing quietly.

## Run

```bash
jaithon run examples/yolo_detect/detect.jai photo.jpg out.png
```

```bash
jaithon run examples/yolo_detect/live.jai
```

Arguments to `live.jai` go after `--`, because `jaithon run` claims the ones
before it:

```bash
jaithon run examples/yolo_detect/live.jai -- --every 3 --record out.avi
```

| flag | meaning |
| --- | --- |
| `--camera N` | which device, default 0 |
| `--video FILE` | a recorded clip instead of the camera |
| `--every N` | run the network on one frame in N, drawing the last boxes on the rest |
| `--record FILE` | write an annotated AVI |
| `--headless` | no window, for `--video` with `--record` |

`YOLO_MODEL` overrides where the weights are read from.

## Files

- `detector.jai` — letterbox, decode, draw. The part a model file does not
  contain, and the part that is easy to get subtly wrong.
- `coco.jai` — the eighty class names, in output-row order.
- `detect.jai` / `live.jai` — the two front ends.
- `decode_check.jai` — the box decoding on a network output made up by hand,
  so it runs without the weights: `jaithon run examples/yolo_detect/decode_check.jai`.
  A script rather than a `jaithon test` file, because a test file is loaded
  outside any package and `detector.jai`'s relative imports need this
  directory to be one.

## Boxes

A `Detection` carries the box as the network gave it, in floating point --
`x`, `y`, `w`, `h`, in original-image pixels, clipped to the image -- and
`box`, the same box with each edge rounded to the nearest pixel, for drawing.
Non-maximum suppression works on whole-pixel `Rect`s and is handed the box
with both edges truncated, as it always was.

Until it was measured, the reported box was that truncated one too. On the
first 300 COCO val2017 images at a score threshold of 0.001 (COCO's
convention; `detect_frame(model, image, 0.001)`), scored with pycocotools
against onnxruntime's CPU run of the same model with the same preprocessing:

| boxes reported | mAP50-95 | mAP50 | AP_small |
| --- | --- | --- | --- |
| truncated to whole pixels (before) | 0.4013 | 0.5751 | 0.2392 |
| floating point (now) | 0.4063 | 0.5767 | 0.2454 |
| onnxruntime, float boxes, float suppression | 0.4075 | 0.5779 | 0.2375 |

Handing the suppression rounded rather than truncated boxes was measured as
well and comes out lower, 0.4059, so it was left as it was. The rest of the
gap to onnxruntime is the suppression itself working in whole pixels; there is
no `Rect2d` in jaicv yet.

The 300 images take 7.4 s end to end in Jaithon, decode included (16.8 ms an
`imread` and 7.3 ms of `detect_frame` each), against 20.3 s with the reference
JPEG decoder; the detections are the same either way.

## Verified

The importer's output was checked against `onnxruntime` on the same input:
first six values `6.796971 20.137949 25.368713 31.320854 37.385056 42.245132`
against onnxruntime's `6.796949 20.137953 25.368710 31.320843 37.385036
42.245125`, identical max (636.3883) and mean (9.461120). On Ultralytics'
own `bus.jpg` it finds four people and a bus, which is the published result.

## Speed

On an M2 Max the network costs about 3.2 ms for one 640x640 input, and a 720p
frame end to end -- letterbox, network, decode, drawing, display -- costs about
5.1 ms, so detecting on every frame runs at roughly 197 fps and the camera sets
the pace rather than the detector.

That is with the network and the imaging running at the same time. `live.jai`
hands the GPU frame N and then draws frame N-1 while it works, so a frame costs
the slower of the two halves instead of their sum: the same loop written one
step after another measures about 6.5 ms, or 154 fps. What is on screen is one
frame behind the camera as a result.

The overlap works because the drawing is done on the processor, writing
straight into the frame's own memory -- device storage is shared, so there is
nothing to copy and nothing to queue. Drawing with a kernel instead was tried
and is slightly quicker on its own, but it lands on the same queue as the
network and runs after it, so a frame costs the sum again.

`--every N` is not needed at this rate and is there for a slower machine or a
larger model.

From a file the frames arrive as JPEG and have to be decoded first, and the
decoder is Jaithon (see the jaicv README's "Image decoding"). On a 120-frame
720p 4:2:0 MJPEG clip -- the first 120 COCO val2017 images resized to
1280x720 and written by OpenCV's `VideoWriter` with `MJPG`; not in the repo --
headless, under `scripts/bench/gpu_lock.sh`:

```bash
jaithon run examples/yolo_detect/live.jai -- --video clip720.avi --headless
```

| decoder | steady fps (five runs) | whole run |
| --- | --- | --- |
| reference (`JAICV_FAST_JPEG=0`) | 19.1 | 23.7 s |
| fast | 38.2, 43.4, 46.5, 49.3, 50.1 | 4.1-4.6 s |

The fps is `live.jai`'s own, the mean of the last fifteen frames, which is why
it scatters; the whole-run time includes loading the model. A 720p frame takes
27 ms to decode (118 ms with the reference decoder) against roughly 4 ms of
network, so from a file the loop is still decode-bound, just short of the
50 fps the detector could keep up with.

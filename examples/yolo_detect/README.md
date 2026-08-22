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

## Verified

The importer's output was checked against `onnxruntime` on the same input:
first six values `6.796971 20.137949 25.368713 31.320854 37.385056 42.245132`
against onnxruntime's `6.796949 20.137953 25.368710 31.320843 37.385036
42.245125`, identical max (636.3883) and mean (9.461120). On Ultralytics'
own `bus.jpg` it finds four people and a bus, which is the published result.

## Speed

On an M2 Max the network costs about 3.2 ms for one 640x640 input, and a 720p
frame end to end -- letterbox, network, decode, drawing, display -- costs about
5.6 ms, so detecting on every frame runs at roughly 178 fps and the camera sets
the pace rather than the detector.

That is with the network and the imaging running at the same time. `live.jai`
hands the GPU frame N and then draws frame N-1 while it works, so a frame costs
the slower of the two halves instead of their sum: the same loop written one
step after another measures about 7 ms, or 143 fps. What is on screen is one
frame behind the camera as a result.

The overlap works because the drawing is done on the processor, writing
straight into the frame's own memory -- device storage is shared, so there is
nothing to copy and nothing to queue. Drawing with a kernel instead was tried
and is slightly quicker on its own, but it lands on the same queue as the
network and runs after it, so a frame costs the sum again.

`--every N` is not needed at this rate and is there for a slower machine or a
larger model.

# examples

Every one of these runs from a clean checkout. Where a program needs data it
generates it, so there is nothing to download except the one model that says so.

## Machine learning

| | what it shows |
|---|---|
| [`tiny_lm/`](tiny_lm) | a character language model, trained on a page of text in the source file. `Embedding`, and what memorising looks like from the inside |
| [`digit_trainer/`](digit_trainer) | jaicv draws the training set, jaitensor learns it, jaicv draws the mistakes. 99.2% on held-out digits, nine seconds |
| `fashion_mnist.jai` | a dense classifier on a real dataset |
| `mnist_gpu.jai` | the shortest end-to-end training loop here |
| `iris_classifier.jai` | a CSV read with `std.io.read_csv`, then trained |
| `spiral_classifier.jai` | `save_weights` and `load_weights` |
| `moons_classifier.jai` | jaitensor and jaiplot together |
| `sine_regression.jai` | regression rather than classification |
| `neat_snake.jai` | neuro-evolution, hand-rolled — no jaitensor at all |

## Computer vision

| | what it shows |
|---|---|
| [`motion_tracker/`](motion_tracker) | sparse optical flow over a generated scene: pick corners once, follow them, draw the trails |
| [`doc_scanner/`](doc_scanner) | the classical pipeline end to end — edges, contours, a perspective warp, adaptive threshold. No weights |
| [`yolo_detect/`](yolo_detect) | a real YOLOv8 ONNX model through jaicv's importer onto jaitensor tensors. The one thing here that downloads something |
| `camera.jai` | frames from a camera into a window |

## The language itself

| | |
|---|---|
| `raytracer.jai` | no libraries, just the language |
| `calculator.jai`, `snake.jai` | `std.gui` |
| `plot.jai` | jaiplot figures |
| `hello.jai` | one line |

## Running one

```bash
jaithon run examples/motion_tracker/track.jai
```

The multi-file projects have their own README with what to expect, how long it
takes, and what it writes out.

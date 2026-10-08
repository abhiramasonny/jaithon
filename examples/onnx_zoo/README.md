# onnx_zoo

Twenty-six real architectures -- torchvision's classifiers and segmenters, a
UNet, ViT and Swin, BERT, DistilBERT, GPT-2 and the Whisper encoder from
transformers -- exported to ONNX with random weights and run through
`jaicv.dnn` both ways it can run a model: the interpreter (`run_graph`, one
operator at a time) and the compiled plan (`compile_plan`, the whole model as
one MPSGraph). Every row is held against onnxruntime's CPU answer for the
same input, and timed at batch 1 against onnxruntime CPU and eager torch on
the MPS device.

**25 of the 26 run correctly on both paths**, and so do 44 of the 45
exports, counting the two dynamo exports of each transformer. Correct means
within a relative 1e-5 of onnxruntime, or within twice torch's own distance
from onnxruntime where that is more. With measured BatchNorm statistics (see
below) the two references are themselves 2.1e-5 apart on `resnet50` and
1.5e-4 on `fcn_resnet50`, and jaicv lands 2.1e-5 and 1.1e-4 from onnxruntime
there, the same on both paths. Every other export is within 1e-5 flat, and
in practice within 9e-6. Before this example's
fixes it was 13 of 25, and none of the default-settings dynamo exports
imported at all. The one left is torchvision's `ssdlite`, whose graph carries
its own post-processing (`NonZero`, `TopK`, `NonMaxSuppression`, `If`); the
YOLO route in `../yolo_detect` is the detection path that works.

## Running it

The exports are about 2.8 GB and are written outside the repository, to
`~/.cache/jaithon/onnx_zoo` (or `$ONNX_ZOO_DIR`). They take a couple of minutes
to regenerate and are never committed.

```bash
# 1. export (the overlay on the scratch venv is what supplies transformers)
VIRTUAL_ENV=~/.venvs/scratch uv run --active --offline --no-project \
    --with onnx --with onnxruntime --with onnxscript --with torchvision \
    python examples/onnx_zoo/export.py                 # or name some models

# 2. acceptance: four quick exports, both paths, exits non-zero on a miss
jaithon run examples/onnx_zoo/check.jai

# 3. one model, or all of them: first refusal, rel diff, import/interp/compile/plan ms
jaithon run examples/onnx_zoo/zoo.jai -- resnet18 vit_tiny_dyn1
jaithon run examples/onnx_zoo/zoo.jai -- --check-only          # every export, no timing

# 4. the timed table, under the GPU lock, and the peers beside it
BATCH=15 examples/onnx_zoo/run_zoo.sh
```

`export.py` writes three exports where it can: `name` (the TorchScript
exporter, opset 17), `name_dyn` (`torch.onnx.export(..., dynamo=True)` with
every other setting at its default, which since torch 2.9 puts the weights in
a `.onnx.data` file beside the model) and `name_dyn1` (dynamo with the weights
inline). torchvision's ViT zero-initialises its head, so `export.py`
re-randomises it -- otherwise every output is zero and a comparison proves
nothing -- and the legacy `vit_b_16` export needs
`torch.backends.mha.set_fastpath_enabled(False)`.

For the same reason every BatchNorm gets statistics measured on random input
and a random scale and shift before the export. A fresh BatchNorm's mean of 0
and variance of 1 shrink a deep random network's activations towards zero:
mobilenet_v2's logits came out near 3e-9 and googlenet's features near 2e-11,
so HardSwish, SiLU and sigmoid never left their straight part, and agreeing
with onnxruntime to 1e-5 said little. `check.jai` now refuses a reference
that peaks below 1e-3.

`check.jai` is the acceptance set: `squeezenet1_1` (the control),
`mobilenet_v3_small` (HardSwish and depthwise convolutions), `resnet18_dyn`
(weights in a sidecar) and `vit_tiny_dyn1` (a class token picked by a
rank-zero `Gather`). Export them with
`export.py squeezenet1_1 mobilenet_v3_small resnet18 vit_tiny`.

## The table

| model | status | import ms | interp ms | compile ms | plan ms | torch MPS ms | ORT CPU ms | plan vs MPS | plan vs ORT | rel (interp / plan) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| squeezenet1_1 | ok | 49.00 | 4.14 | 20.03 | 0.75 (0.74-0.75) | 4.11 | 2.47 | 5.50x | 3.31x | 3.8e-07 / 1.9e-07 |
| resnet18 | ok | 91.30 | 2.97 | 21.58 | 1.62 (1.62-1.62) | 6.37 | 10.91 | 3.93x | 6.73x | 1.8e-06 / 1.7e-06 |
| resnet50 | ok | 162.36 | 6.67 | 49.98 | 3.67 (3.66-3.71) | 12.83 | 29.06 | 3.50x | 7.92x | 2.1e-05 / 2.1e-05 |
| googlenet | ok | 72.17 | 4.14 | 26.22 | 2.37 (2.33-2.44) | 10.24 | 8.96 | 4.33x | 3.79x | 9.0e-06 / 9.0e-06 |
| densenet121 | ok | 87.00 | 9.57 | 72.37 | 6.41 (6.32-6.53) | 22.98 | 21.42 | 3.59x | 3.34x | 3.6e-06 / 2.8e-06 |
| mobilenet_v2 | ok | 69.77 | 4.09 | 22.99 | 1.97 (1.29-2.10) | 7.46 | 4.64 | 3.79x | 2.35x | 7.2e-06 / 7.2e-06 |
| mobilenet_v3_small | ok | 54.10 | 5.53 | 24.46 | 1.78 (1.69-1.93) | 6.43 | 1.57 | 3.62x | 0.88x | 1.9e-06 / 1.5e-06 |
| mobilenet_v3_large | ok | 71.69 | 6.20 | 33.67 | 1.93 (1.86-1.98) | 11.49 | 4.22 | 5.94x | 2.18x | 2.5e-06 / 2.6e-06 |
| efficientnet_b0 | ok | 67.10 | 11.88 | 38.61 | 2.84 (2.79-3.30) | 12.34 | 8.14 | 4.35x | 2.87x | 3.1e-06 / 2.7e-06 |
| mnasnet1_0 | ok | 59.49 | 4.25 | 22.57 | 2.31 (1.79-2.54) | 9.87 | 5.06 | 4.26x | 2.19x | 3.8e-06 / 3.8e-06 |
| shufflenet_v2_x1_0 | ok | 55.82 | 8.22 | 36.66 | 3.21 (3.01-3.95) | 8.03 | 1.89 | 2.51x | 0.59x | 4.4e-06 / 4.4e-06 |
| regnet_y_400mf | ok | 62.57 | 11.40 | 35.35 | 2.49 (2.46-2.53) | 10.40 | 4.55 | 4.18x | 1.83x | 4.7e-06 / 4.1e-06 |
| convnext_tiny | ok | 183.23 | 11.21 | 97.44 | 4.22 (4.20-4.25) | 7.06 | 29.32 | 1.67x | 6.94x | 1.4e-06 / 1.3e-06 |
| vit_tiny | ok | 79.99 | 19.68 | 81.76 | 4.92 (4.69-5.14) | 4.72 | 8.30 | 0.96x | 1.69x | 1.7e-06 / 1.9e-06 |
| vit_b_16 | ok | 441.07 | 33.84 | 460.72 | 9.97 (9.93-9.99) | 10.99 | 83.49 | 1.10x | 8.37x | 2.6e-06 / 2.4e-06 |
| swin_t | ok | 258.57 | 94.64 | 552.76 | 5.30 (5.28-5.32) | 10.91 | 28.99 | 2.06x | 5.47x | 8.1e-07 / 7.4e-07 |
| lraspp_mbv3 | ok | 55.38 | 8.55 | 31.73 | 2.31 (2.30-2.36) | 12.21 | 11.77 | 5.27x | 5.08x | 1.8e-06 / 1.7e-06 |
| deeplabv3_mbv3 | ok | 109.61 | 10.78 | 45.16 | 3.80 (3.74-3.82) | 13.51 | 27.49 | 3.56x | 7.24x | 7.1e-06 / 6.6e-06 |
| fcn_resnet50 | ok | 205.68 | 21.61 | 79.06 | 16.68 (16.65-16.84) | 29.40 | 293.42 | 1.76x | 17.59x | 1.1e-04 / 1.1e-04 |
| unet | ok | 70.80 | 29.74 | 45.33 | 4.26 (4.24-4.27) | 8.82 | 68.50 | 2.07x | 16.08x | 1.3e-06 / 1.3e-06 |
| unet_bilinear | ok | 70.85 | 6.66 | 36.67 | 3.73 (3.71-3.75) | 10.13 | 70.08 | 2.72x | 18.81x | 9.0e-06 / 2.2e-06 |
| bert_tiny | ok | 63.13 | 2.44 | 21.15 | 0.69 (0.68-0.85) | 2.04 | 0.62 | 2.95x | 0.89x | 2.8e-07 / 2.2e-07 |
| distilbert | ok | 380.94 | 11.09 | 142.56 | 3.65 (3.43-4.20) | 5.73 | 23.59 | 1.57x | 6.47x | 1.2e-06 / 1.3e-06 |
| gpt2_tiny_dyn1 | ok | 129.01 | 2.76 | 30.51 | 1.82 (1.77-1.84) | 2.80 | 1.49 | 1.54x | 0.82x | 6.3e-07 / 1.1e-06 |
| whisper_tiny_enc | ok | 81.13 | 66.61 | 91.34 | 7.18 (7.16-7.21) | 12.00 | 90.38 | 1.67x | 12.59x | 8.2e-07 / 7.8e-07 |
| ssdlite | refused | - | - | - | - | 149.60 | 167.57 | - | - | - / - |

The same architectures as torch's default export writes them (`dynamo=True`,
weights in `.onnx.data` beside the model), which this reader refused outright
before -- all ten import and run on both paths now:

| model | status | import ms | interp ms | compile ms | plan ms | torch MPS ms | ORT CPU ms | plan vs MPS | plan vs ORT | rel (interp / plan) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| resnet18_dyn | ok | 86.51 | 2.69 | 18.96 | 1.63 (1.62-1.63) | 8.34 | 12.20 | 5.12x | 7.49x | 1.8e-06 / 1.7e-06 |
| efficientnet_b0_dyn | ok | 65.71 | 10.82 | 25.72 | 2.55 (2.50-2.75) | 11.43 | 5.27 | 4.48x | 2.07x | 3.5e-06 / 2.2e-06 |
| mobilenet_v3_large_dyn | ok | 66.23 | 4.93 | 20.78 | 2.02 (1.91-2.10) | 8.49 | 3.80 | 4.19x | 1.88x | 2.6e-06 / 2.1e-06 |
| convnext_tiny_dyn | ok | 171.02 | 8.15 | 70.81 | 4.20 (4.18-4.22) | 6.76 | 30.78 | 1.61x | 7.33x | 1.4e-06 / 1.2e-06 |
| unet_bilinear_dyn | ok | 68.28 | 5.23 | 17.78 | 3.73 (3.71-3.75) | 8.54 | 60.95 | 2.29x | 16.36x | 9.0e-06 / 2.2e-06 |
| vit_tiny_dyn | ok | 71.01 | 14.00 | 46.90 | 4.19 (3.75-5.17) | 4.67 | 8.15 | 1.11x | 1.95x | 1.8e-06 / 1.9e-06 |
| swin_t_dyn | ok | 172.69 | 21.88 | 187.33 | 5.32 (5.29-5.35) | 10.68 | 27.99 | 2.01x | 5.26x | 7.9e-07 / 7.7e-07 |
| bert_tiny_dyn | ok | 59.65 | 1.65 | 13.12 | 0.68 (0.67-0.69) | 2.34 | 0.60 | 3.45x | 0.88x | 2.2e-07 / 2.2e-07 |
| distilbert_dyn | ok | 318.81 | 9.78 | 133.62 | 3.50 (3.45-3.75) | 5.74 | 23.63 | 1.64x | 6.76x | 1.3e-06 / 1.3e-06 |
| gpt2_tiny_dyn | ok | 117.52 | 2.78 | 30.65 | 1.91 (1.87-2.03) | 2.92 | 1.38 | 1.53x | 0.72x | 6.3e-07 / 1.1e-06 |

The torch MPS column needs a caveat. `peer_time.py` sized its batch of calls
from an average over the warm-up, which included the first call, so where
that call took a second or more each sample was a single call. A reviewer
read `mobilenet_v3_large` at 16.2 ms (13.0-18.4) where the table has 11.49,
so its 5.94x over MPS is anywhere from about 4x to 8x. The batch is sized
after the warm-up now; the column has not been re-measured since.

The `rel` column is from the BatchNorm-calibrated exports. Every timing
column predates the calibration, which changed the numbers in the BatchNorm
weights and nothing about the work done.

`status` ok means both paths land within that tolerance; `rel` is the
largest absolute difference over the largest absolute reference value, over
the first output. `plan` is the compiled plan's median with its min-max
across seven samples. `interp` is the interpreter's median. Ratios are the
peer's median over the plan's: above 1x the plan is faster.

Where the plan loses, it is mostly to onnxruntime on the CPU, on the
smallest graphs: `mobilenet_v3_small`, `shufflenet_v2_x1_0`, `bert_tiny` and
`gpt2_tiny` take onnxruntime 0.6-1.9 ms, which is below what one GPU round
trip plus the per-operator encode costs here (`squeezenet1_1`, 83 nodes, is
0.75 ms; `shufflenet_v2_x1_0`, 315, is 3.2). The other loss is the legacy
`vit_tiny` against eager MPS, 4.92 ms to 4.72, and it is the noisiest pair in
the table: the same architecture exported by dynamo runs at 3.81 ms, MPS was
measured at 4.67-7.80 ms across its three rows, and the plan's own spread on
that row is 4.69-5.14, so read it as a 4% loss inside the noise. ViT-B/16 is
1.10x over MPS; every convolutional network is 1.6-5.9x.

The compile column is from before the last commit, which stopped
`compile_plan` reading every `Identity` of a constant back from the device
(torch writes one per duplicated weight, so a random-init ResNet-50 has 47 and
densenet121 245). A second pass after it, under a heavier load, read
densenet121 at 39 ms (72 above), vit_b_16 at 286 (461), resnet50 at 45 (50),
mobilenet_v2 at 18 (23) and squeezenet1_1 at 17 (20); its plan columns were
too noisy to replace these.

## What changed in jaicv.dnn to get here

Found in review, each with a test that failed before and passes after:

- **`Gelu` with `approximate="tanh"` came back NaN on the interpreter from an
  input of about 10.4**, and `Tanh` from about 45. Metal's tanh goes through
  exp(2x) under fast math, which is inf over inf once its argument passes 44,
  and Gelu's cubic gets there early. The argument is clamped at 10, where the
  answer is already 1 to the last bit; the LSTM and GRU kernels had the same
  call. The compiled plan was right throughout, so the two paths disagreed,
  and `run_best` answers its first call on the interpreter. The oracle
  comparison also let NaN pass, because `NaN > worst` is false.
  (`gelu_tanh_wide`, `tanh_wide`, `lstm_saturated`, `gru_saturated`.)
- **Optional outputs listed under empty names** (`["y", "", ""]`, valid ONNX)
  made `LayerNormalization` throw on both paths once it stopped producing the
  statistics nobody named. Only a named output that never came back is a
  fault now.
- **A `Tensor.view` of a batch** whose last axis is not a multiple of four
  floats is refused by MPSGraph, which the new one-operator convolution and
  matrix product routes go through; jaitensor took it before. Such a feed is
  now run again as a contiguous copy. A compiled `Plan` fed such a view still
  throws when it runs, as it did before this change.
- **`compile_plan` read back folded values too large to fold**, such as a
  transposed tied embedding, now that `Transpose` folds; it leaves anything
  over `FOLD_LIMIT` on the device.

Found while building the example, again each with a test that failed before
and passes after:

- **A `Gather` by a rank-zero index kept a width-one axis.** jaitensor has no
  rank-zero tensor, so a scalar arrives as `[1]` either way; the importer now
  records which weights and `Constant`s the file wrote with no dimensions,
  follows them through the shape arithmetic done on them (an element of a
  `Shape`, arithmetic on scalars), and marks the `Gather`s they index -- which
  drop the axis -- and the `Unsqueeze`s they feed, which come out with only the
  axes they insert. ViT's class token, BERT pooling and Swin's window
  partition all went through this. (`onnx_scalar_gather.onnx`,
  `onnx_scalar_shape.onnx`, the `gather_scalar_*` oracle cases.)
- **densenet121 was refused by the plan**: its BatchNorm statistics arrive
  through `Identity`, because torch keeps one copy of identical weights, and
  `Identity` did not fold. It does now, with `Dropout` and the rest of the
  shape arithmetic exporters do on constants. densenet121 runs on the plan.
- **Grouped and depthwise convolutions ran a group at a time**, with every
  group's weight slice memoised on the node: about 7000 jaitensor
  convolutions and 7000 live slices per MobileNetV2 frame, seconds per call,
  slower after the first. They are now one dispatch of a cached one-operator
  graph convolution (shared by every node with the same shapes), which also
  takes dilations, unequal strides and one-dimensional input; grouped
  `ConvTranspose` is one kernel dispatch. Nothing is memoised on the node.
- **`LayerNormalization` leaked two GPU buffers per node per interpreted
  call** (its optional row statistics were reshapes of tensors nobody freed)
  -- fifty a frame on a ViT, and the handle table's linear scan made every
  later allocation slower.
- **The plan's `Gather` ignored negative indices** (`x[:, -1]`), and a
  folded open slice end (`INT64_MAX`) did not convert back to an integer;
  both now compile correctly.

Coverage:

- ONNX external data: the path readers follow `.onnx.data` sidecars by
  `location`/`offset`/`length`, read each file once, and refuse an absolute
  location or one with a `..` component before opening anything. The
  bytes-only reader still refuses, naming the file and saying to read from the
  path. (`onnx_external*.onnx`.)
- `HardSwish`, `Gelu` (exact and tanh), `GreaterOrEqual`, `LessOrEqual`,
  `Expand` (bidirectional, and real expansion in the plan), `GatherElements`,
  `GatherND` (and in the plan, for folded coordinates), `ScatterND`, `Range`,
  and TorchScript's `ConstantOfShape`-of-an-empty-shape scalar.
- A batched `MatMul` is one graph dispatch instead of a loop over pages, and a
  batch times a matrix -- every linear layer of a transformer -- one flat
  product; that took the dynamo ViT-Tiny's interpreter from 205 ms a call to 15.

Two of these are behind switches, read once and on by default:
`JAICV_GRAPH_CONV=0` brings back the per-group loop (without the memo) and
`JAICV_GRAPH_MATMUL=0` the per-page matmul loop.

### Before and after, same binary, same hold

An A/B of the dnn source at the branch point (0f2fe2e5) against this
branch, swapped in and out inside one GPU-lock hold. One process per row, so
"cold" is the first call a fresh process makes, which is what a one-shot
classify sees. ms:

| model | path | before | after |
|---|---|---:|---:|
| mobilenet_v2 | first result (import + cold interpreter) | 932 | 178 |
| | interpreter, warm | 1839 | 4.5 |
| | compile_plan | 2943 | 28 |
| efficientnet_b0 | first result | 1284 | 198 |
| | interpreter, warm | 2788 | 11.7 |
| | compile_plan | 4632 | 37 |
| convnext_tiny | first result | 1027 | 276 |
| | interpreter, warm | 1746 | 11.7 |
| | compile_plan | 2833 | 96 |
| resnet50 (no grouped convs; the control) | interpreter, warm | 11.0 | 7.5 |
| | plan, median of 3 rounds | 3.73 | 3.68 |
| | import, median of 3 rounds | 93.1 | 92.9 |
| yolov8n | plan, median of 3 rounds | 3.04 | 3.08 |
| | import, median of 3 rounds | 14.8 | 14.3 |

The branch point still ran these models slower warm than cold (mobilenet_v2
1839 ms against 872), the per-group slices piling up on the nodes; an older
base (0d101603) measured 4898 ms warm and 7647 ms to compile. The warm
interpreter is now within 2x of resnet50's for all three, and so is
compile_plan; the compile figures here are from before the Identity commit
noted under the table. yolov8n's plan reads
1.5% slower over three alternating rounds, inside the 3% the change was held
to; yolov8n/s/x land within 1.5e-6 of onnxruntime on both paths, and
`ort_check.jai`'s sum is unchanged at 6883170.492.

## Conditions

M2 Max, macOS 27.0.1. torch 2.14.1, torchvision 0.29.1, transformers 5.14.1,
onnx 1.23.2, onnxruntime 1.30.0 (CPU provider, default threads). Every timing
ran under `scripts/bench/gpu_lock.sh`, warmed for 1.5 s by the wall clock,
then seven samples of enough calls to fill about 150 ms, one live result at a
time. jaicv's rows are one model per process, fifteen processes per hold. The
peers' rows were taken in two passes: the TorchScript exports an hour earlier,
one model per hold and alternating with jaicv's row for the same model, and the
dynamo exports in a hold of their own beside jaicv's batch. Other agents were
working on the machine throughout (load average 3-17), so read the spreads as
well as the medians. The torch MPS peer is eager, under `torch.no_grad`, with
`torch.mps.synchronize()` per call; it is the same architecture built with the
same seed, not the ONNX file. A model timed after many others in one process
read up to twice as slow on the interpreter before the LayerNorm leak was
fixed, which is why each row gets a process of its own.

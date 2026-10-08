# cifar_resnet

ResNet-20 trained from scratch on CIFAR-10 in Jaithon on the GPU. The run uses
residual blocks, batch normalisation fused with the residual add and ReLU,
augmentation done on the GPU, SGD with Nesterov momentum and weight decay, and
a one-cycle learning rate. The peer is `cifar_resnet.py`, which runs the same
network, data, augmentation and recipe in PyTorch on MPS (fp32, NCHW).

```bash
jaithon run examples/cifar_resnet/train.jai                 # ResNet-20, 30 epochs, seed 0
jaithon run examples/cifar_resnet/train.jai -- --seed 1     # another seed
jaithon run examples/cifar_resnet/check.jai                 # ResNet-8, 2 epochs: the quick acceptance
~/.venvs/scratch/bin/python examples/cifar_resnet/cifar_resnet.py --seed 0   # the torch peer
```

Both sides read `data/cifar-10/{train,test}-{images,labels}.bin`, the planar
uint8 CIFAR-10 files that `tests/bench/jaitensor/prepare_data.py cifar10`
writes (the repo's `data` is a symlink into `~/Developer/datasets`). `train.jai`
also takes `--epochs`, `--depth` (6n+2), `--width`, `--batch`, `--lr` (the
one-cycle peak), `--pct`, `--wd`, `--normalize 0|1` and `--train N` (the first
N training images); the peer takes the same flags.

## Results

30 epochs, three seeds, on an M2 Max (macOS 27.0.1) against torch 2.13.0 on
MPS. Each run had the GPU to itself under `scripts/bench/gpu_lock.sh`, and the
two sides alternated (Jaithon seed 0, torch seed 0, Jaithon seed 1, and so on)
with load averages of 4 to 6 from other work on the machine. The tree was this
branch with the autograd track (`b258afa4`) and the kernels track (`b586d045`)
merged in.

| seed | Jaithon test | torch test (batch stats) | torch test (running stats) | Jaithon, 30 epochs | torch, 30 epochs | torch / Jaithon |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 90.20% | 90.18% | 90.67% | 128.4 s | 179.4 s | 1.40x |
| 1 | 90.08% | 90.56% | 90.93% | 131.9 s | 184.3 s | 1.40x |
| 2 | 90.42% | 90.11% | 90.68% | 130.0 s | 184.3 s | 1.42x |
| **median** | **90.20%** | **90.18%** | 90.68% | **130.0 s** | **184.3 s** | **1.42x** |

- Accuracy: the Jaithon median is 90.20%, 0.02 points from the peer's 90.18%
  in the same evaluation mode.
- The 30-epoch time covers the augmentation and every training step: from the
  first crop to the last batch. Loading and evaluation are left out on both
  sides.
- An epoch takes 4.36 s against torch's 6.14 s (median of epochs 2-30, then of
  the three seeds).
- Augmenting all 50,000 images takes 22 ms on the Jaithon side (one dispatch)
  and 387 ms on torch's two gathers. Leaving augmentation out entirely, the
  training steps alone are 1.33x faster than torch's.
- Training curves match: final training loss 0.161/0.170/0.163 against torch's
  0.165/0.165/0.161, with both sides at 94.4-94.7% training accuracy.

The commands, one per lock and alternated:

```bash
./scripts/bench/gpu_lock.sh ./jaithon run examples/cifar_resnet/train.jai -- --seed 0
./scripts/bench/gpu_lock.sh ~/.venvs/scratch/bin/python examples/cifar_resnet/cifar_resnet.py --seed 0
# ... seeds 1 and 2 the same way
```

`check.jai` (ResNet-8, two epochs, the same recipe) reaches 68.2% in 3.6 s
including data loading. Its thresholds are 60% and 60 s.

### Before and after the fused batch norm

`ResBlock` uses the kernels track's `BatchNorm2d(relu: true)` and
`forward_residual`, so each block's add and ReLUs run inside the norms' own
passes. `JAITENSOR_BN_FUSED=0` composes them from separate ops again, which
gives an A/B in one binary. The test was ResNet-20 for three epochs, five runs
each way alternated inside one lock:

| | 3 epochs (median, range) | epoch 2 (median) |
| --- | --- | --- |
| composed (`JAITENSOR_BN_FUSED=0`) | 13.40 s (13.24-14.24) | 4.41 s |
| fused (default) | 12.13 s (12.00-12.45) | 4.03 s |

The fused path is 1.10x faster, and the two ranges do not overlap. Scaled by
that ratio, the 30-epoch run would take about 144 s without it: 1.28x faster
than torch rather than 1.42x.

### A leak that slowed every epoch

The first 30-epoch runs got slower as they went. Epochs started at 4.2 s and
reached 12.3 s by the thirtieth, and the run took 228.5 s, while torch's
epochs held flat. The cause was not the GPU.

- The host thread sat at 100% CPU.
- A `sample` profile put the main thread in `nGpuBufferNew` and
  `jaiHandleAdd`: 1,646 samples early in the run and 2,562 late.
- Handles are handed out by a scan from the bottom of one table. Two places
  kept a handle on every step: `DataLoader.next_batch` took a view of the
  shuffle order for each batch's indices and never freed it, and every
  convolution's backward did the same with a flat view of its upstream
  gradient. A view holds no device memory, so nothing that counts buffers
  noticed. ResNet-20 kept one or more handles per convolution per step, plus
  one per batch, so hundreds of thousands by the end of a run, and every
  allocation scanned past all of them.

Freeing the two views (`data.jai`, `ops/conv.jai`) fixed it. The same seed now
runs in 128.4 s with every epoch between 4.0 and 4.4 s, and reaches the same
90.20%, as expected, because the leak never touched a number.
`test_a_shuffled_epoch_keeps_no_handles` and
`test_a_residual_training_step_keeps_no_handles` failed before the fix (15 and
27 handles kept) and pass after. A full-size probe over four epochs of real
training holds the handle frontier at 812.

Every `Sequential.fit(shuffle: true)` and every model with a convolution had
the same leak. A long run of either got slower in the same way.

## The recipe

| | |
| --- | --- |
| model | ResNet-20 (He et al. 2016, section 4.2): 3x3 stem conv + BN + ReLU, 3 stages of 3 basic blocks at 16/32/64 channels, 1x1 stride-2 projection + BN where a stage starts, global average pool, linear head. 272,490 parameters |
| data | the 50,000 training images, scaled to [0, 1] and standardised per channel with the training set's mean and std |
| augmentation | each epoch, every image is shifted by -4..4 pixels in x and y with zeros brought in (pad 4 + random crop) and mirrored half the time |
| batches | a fresh shuffle each epoch, batch 128, the short last batch kept (391 steps an epoch) |
| optimiser | SGD, Nesterov momentum 0.9, weight decay 5e-4 on every parameter |
| schedule | one-cycle (PyTorch's `OneCycleLR`, cosine, momentum not cycled): peak 0.2 at 25% of the run, from 0.2/25 down to 0.2/25/1e4, set every step |
| epochs | 30 |
| init | every weight uniform with variance 1/fan_in, every bias zero, BN gamma 1 and beta 0 |
| evaluation | the 10,000 test images in batches of 1000, BatchNorm on batch statistics |

The peer matches each row. It also computes the same per-image shift and flip
from the same hash of image index and seed (`crop_params`), so both sides see
the same augmented images each epoch. Only the shuffle order, the parameter
draws and the floating-point rounding differ.

## Why batch statistics at evaluation

`BatchNorm2d` in jaitensor keeps no running mean or variance. It normalises
by whatever batch it is given, in prediction as well as in training (see the
comment on the class in `packages/jaitensor/src/jaitensor/layers.jai`). Adding
running statistics would change `predict()` for every BN model that exists,
so that decision is left to the user. This example therefore evaluates in
batches of 1000 on batch statistics, and the peer is reported the same way
(`net.train()` under `no_grad`) so the comparison is like for like. Its usual
running-statistics number is printed beside it.

The two modes are not the same number, and the gap is not noise. Training
batches are augmented crops, up to a quarter of whose pixels can be padding;
test batches are clean images. Batch statistics at test time therefore
normalise the network's activations differently from how it was trained. Running
statistics, gathered on the training batches, do not have that mismatch.

With inputs in [0, 1], torch's seed-0 run scored 88.09% on batch statistics
and 90.33% on running statistics: a 2.2-point gap, with zero padding as the
likely cause. Black borders pull a training batch's statistics away from a
clean one's. Standardising the inputs per channel makes the padded zeros the
average colour instead of black. That narrowed the gap to 0.5 points (90.18%
against 90.67%) and put both sides over 90% under batch statistics, so the
example and the peer both standardise.

## What the package gained for this

- `ResBlock(channels, stride: 1, projection: false)` in `layers.jai`: a basic
  block that works inside `Sequential`. A projection appears when the stride
  or channel count changes, or when asked for. Its convolutions have no bias,
  because the BN after each one would remove it again. Both norms use the
  kernels track's fused API (`BatchNorm2d(relu: true)` and `forward_residual`).
  Its gradients are tested against the same block written out op by op
  (with the fused and the composed norm), and against PyTorch.
- `random_crop_flip(images, padding, seed, flip)` and `random_crop_flip_into`
  in `data.jai`: one GPU dispatch over the whole NHWC dataset, seeded per
  image. About 22 ms for 50,000 CIFAR images, against 387 ms for the
  peer's two gathers.
- In `optimizers.jai`:
  - `SGD(weight_decay:)` and `Adam(weight_decay:)`, PyTorch's coupled L2 form.
  - `AdamW`, with decoupled weight decay.
  - An assignable `learning_rate`, so a schedule no longer builds a new
    optimizer each step.
  - `clip_grad_norm`, computed on the device and returning the pre-clip norm
    as a tensor.
  - `one_cycle_lr`, which equals `torch.optim.lr_scheduler.OneCycleLR`.

  Each of these is tested against PyTorch's values. A decaying optimizer
  reports its own kind, so the fused and compiled training paths (which have
  no decay term) step aside instead of silently dropping the decay.
- The update kernels now take their settings as dispatch arguments instead of
  a per-parameter settings buffer that the host rewrote whenever the learning
  rate changed. This was meant to make a per-step schedule cheap: a host
  write to a buffer that queued work reads has to wait for that work. On
  this loop it measured no difference (13.46 against 13.28 ms a step, n = 3,
  inside the noise; `probes/step_phases.jai` in the worktree), because the
  host never ran more than a step ahead. It stays because it is simpler: the
  settings buffers and their uploads are gone, and kinds 0 and 1 compute
  exactly what they did, which the three-path agreement tests confirm.

## Notes

- The compiled whole-step route refuses `ResBlock`, so training takes the
  autograd route. That is the loop in `resnet.jai`: forward, loss, backward,
  step. It is written out rather than left to `Sequential.fit` so the
  learning rate can be set every step.
- `Sequential` builds layers with a fixed seed. `build_layers` builds them
  first from `--seed`, and `Sequential` keeps weights that are already built,
  so each seed is a different initialisation, crop sequence and shuffle.
- The example writes nothing: no checkpoints, plots or images.

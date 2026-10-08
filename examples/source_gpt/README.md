# source_gpt

A byte-level GPT trained on this repository's own Jaithon source, then sampled
through a KV cache. Nothing is downloaded: the corpus is `lib/` and
`packages/*/src/`, about 7.7 MB of `.jai`.

```bash
jaithon run examples/source_gpt/check.jai                  # the acceptance check, about 20 s
jaithon run examples/source_gpt/train.jai                  # 2,000 steps, about 90 s
jaithon run examples/source_gpt/generate.jai -- "pub fn "  # 600 bytes from the checkpoint
```

Run them from the repository root. `train.jai` saves a binary checkpoint to
`$TMPDIR/source_gpt.ckpt` (or `GPT_CKPT`), which `generate.jai` loads. Nothing
it writes lands in the repository.

## The model

4 layers, width 256, 4 heads, a 128-byte context, vocabulary 256 (bytes), batch
64: 3,252,736 parameters. Pre-LN blocks of causal multi-head attention and a
GELU MLP, learned positions, the output head tied to the byte embedding, Adam
at a constant 1e-3. The held-out set is `packages/jaiyaml` and
`packages/jaitoml` -- two parsers the model never reads in training, cut into
1,957 back-to-back windows -- and progress is reported in bits per byte on it.

Every piece is a jaitensor layer or op; nothing in the example is a kernel.
`model.jai` -- model, corpus reader and batches -- is about 500 lines. What it
needed from jaitensor, and now has:

- `MultiHeadAttention` takes a whole batch, `[batch, seq, dim]`, or
  sequence-first `[seq, batch, dim]` with `batch_first: false`. The batch folds
  into the head axis, so a batch is the same number of dispatches as one
  sequence. Before, the layer took `[seq, dim]` only and a batched model had to
  pull the projections out of `parameters()` and write attention by hand.
- Its KV cache holds a batch of sequences (`start_cache(max_seq, batch: n)`),
  and `forward_add` adds the residual inside the output product when decoding.
- `layer_norm` and `LayerNorm` take any rank of at least two.
- `Embedding.lookup` keeps the index shape, `[batch, seq]` to
  `[batch, seq, dim]`; `forward` keeps the `[batch, seq * dim]` that
  `Sequential` sizes its next layer for.
- `PositionalEmbedding`, learned positions with `forward_at(x, start)` for a
  decoder that is part-way through a sequence.
- `save_checkpoint` / `load_checkpoint` / `restore_checkpoint`: a named tensor
  list as raw float32 with a small JSON header, bit-exact.

### Sequence-first

The activations are sequence-first: row `t * 64 + b` is step `t` of sequence
`b`. Every op except attention works row by row and does not care about the
order, and in this order attention reads the projections where they lie --
`[seq, batch * dim]` already is the packed layout of `batch * heads` heads.
Batch-first has to copy Q, K and V into that layout and the context back out:
0.57 ms of a 1.78 ms attention forward and backward per layer at these shapes
(locked, n=7, two rounds). `GPT_BATCH_FIRST=1` trains batch-first for an A/B.

## Numbers

M2 Max, macOS 27.0.1. Every timing ran under `scripts/bench/gpu_lock.sh`, with
the two sides alternated, warmed by 2 s of wall clock, and the loss downloaded
every step on both sides. The torch peer is `source_gpt.py` (torch MPS, same
shapes, initialisation ranges, Adam eps 1e-7, data and held-out windows). Other
agents were using the CPU throughout (load 3-10), which inflates the host-bound
numbers -- decode most of all -- on both sides.

Two trees. **This branch** is `gpt-on-jaithon-source` on its own (base
0f2fe2e5). **Trial merge** is this branch merged last onto the other five wave-5
branches as they stood at 01:20 (autograd b258afa4, kernels b586d045, zoo
eefcf9a8, decode df274f72, resnet 77f8e7e3) -- the merge order the wave uses.
The merged tree has to be measured again once the real merge lands; `bench.sh`
does that.

| | jai, this branch | jai, trial merge | torch MPS | target |
|---|---|---|---|---|
| step, 2 s warm then 7 x 5 steps, 3 rounds | 46.7 / 46.0 / 48.6 ms | 42.1 / 44.9 / 47.3 ms | 56.4 / 56.7 / 59.9 ms (54.4 / 54.3 / 70.0 beside the merge) | 1.3x |
| step, averaged over the 2,000-step run | **43.1 ms (1.35x)** | 47.1, then **45.5 ms (1.35x)** | 58.2 ms (61.2 beside the merge) | 1.3x |
| 2,000 steps, wall | **87.2 s** | 95.3, then **92.1 s** | 117.2 s (123.3 beside the merge) | 100 s |
| held-out bits per byte at step 2,000 | **1.463** | 1.467, then **1.465** | 1.488 (1.463 beside the merge) | within 0.05 |
| cached batch-1 decode | 1.86 / 1.89 / 1.82 ms a byte, **536 bytes/s** | 1.26 / 2.54 / 1.28 ms, **781 bytes/s** | 1.34 / 1.35 / 1.37 ms, 741 bytes/s (5.70 / 1.74 / 1.62 beside the merge) | 600 bytes/s |
| checkpoint save + load, 3,252,736 parameters | **20 + 11 ms** | 10 + 9, 22 + 13 ms | | 0.3 s |
| first step, fresh process | 69 / 152 / 71 ms | 61 / 62 / 64 ms | 309 / 276 / 1176 ms | report |
| sampled top-level functions `jaithon check` accepts | 0 of 50 (9 closed their body in 600 bytes) | | | report |

Read it this way:

- **Training is 1.35x torch over the real run, on both trees, and 1.21-1.29x
  in the short bench.** The 2,000-step averages are the steadier number: 2,000
  steps on each side, the same protocol, back to back under the lock (the
  merged pair inside one acquisition). Two jai runs of the merged tree a few
  minutes apart read 95.3 and 92.1 s, which is the size of the run-to-run
  noise here. The short bench takes
  35 steps after 2 s of warm-up and jai's spread inside a round is wide
  (37-54 ms a step under the CPU load other agents put on the machine), while
  torch's is not. The step is GPU-bound: the GEMMs are about two thirds of it
  and attention most of the rest (a per-op probe at these shapes, unlocked);
  host encode is 2.8-3.4 ms a step and mostly hidden by `GPT_FLUSH`.
- **The two models learn the same thing.** Held-out bits per byte track each
  other all the way: 2.60 / 2.62 at step 250, 1.98 / 1.99 at 500, 1.70 / 1.69
  at 1,000, 1.46 / 1.49 at 2,000 (jai first; 8 batches until the last, which
  is all 1,920 windows). Torch's own two runs ended 0.025 apart, 1.488 and
  1.463, so the gap between the sides is inside the gap between seeds.
- **Decoding reaches the target on the merged tree, not on this branch.** A
  decoded byte is about 40 dispatches of one row each, and the device -- not
  the host, whose encode is 0.3 ms of it -- spends its time in the one-row
  products, which this branch still runs through 32x32 GEMM tiles. The kernels
  track's thin GEMMs take it from 536 to 781 bytes/s.
- **The checkpoint is effectively free:** 13 MB in 31 ms, against an estimated
  ten seconds for the JSON `save_weights` writes.
- **At 2,000 steps the model writes code-shaped text, not code.** It indents,
  opens `fn` with typed parameters, uses `let`, `self`, `for ... in 0..n` and
  `-> list[...]`, and invents identifiers; none of 50 sampled top-level
  functions type-checks, and most do not close their body within 600 bytes.
  The 128-byte context is shorter than most function bodies.

A sample (`generate.jai -- "pub fn parse_"`, temperature 0.8, top-40, this
branch's checkpoint):

```
pub fn parse_parse_int_type(self) -> list[tuple[T, T, C, C] {
        let path] = self._resv()
        if path.len() { return path[path] }
        let length = self.next
        let start = self.copy()
        length = self.count - total * dense)
        let loss = self.sign(a, b)
        let index = self.capacity(
            after,
            after.len(),
                alpha,
                alpha,
                alpha
        )
        let step_bits = if bits > 0 { -1 + bits } else { bits }
```

### Bugs this turned up

- **GELU and tanh gave NaN on large inputs**, and with them every loss of the
  first 2,000-step run from somewhere between steps 400 and 500. The kernels build with fast math, whose `tanh`
  overflows an exponential: GELU was NaN from an input of 12, tanh from 45.
  Fixed in the elementwise kernels by clamping the argument to +-15, where the
  answer is already 1 to the last bit. The fused GEMM epilogue had the same
  bug; the kernels track clamps it (`gemm_tanh`), and until that merges a
  batch of 50 sampled sequences meets NaN logits, which `generate.jai` reports
  instead of crashing on.
- **A reshaped leaf's gradient pointed at freed memory.** `reshape`'s backward
  handed its input a view of the reshaped tensor's own gradient, which is
  freed with the intermediates `backward()` returns, so reading the leaf's
  `.grad` afterwards crashed the process. Batched attention reshapes whatever
  it folds, so any leaf fed to it as `[batch, seq, dim]` hit it. The gradient
  now takes over the buffer.


## How it is measured

```bash
sh examples/source_gpt/bench.sh step     # step time, 3 alternated rounds of 7 x 5 steps
sh examples/source_gpt/bench.sh train    # the 2,000-step runs, both sides
sh examples/source_gpt/bench.sh decode   # cached batch-1 greedy decoding
```

`bench.sh` takes the lock per command and points `JAITENSOR_GEMM_CACHE` at a
scratch copy of the plan cache. By hand:

```bash
MODE=bench ./scripts/bench/gpu_lock.sh ./jaithon run examples/source_gpt/train.jai
MODE=bench ./scripts/bench/gpu_lock.sh ~/.venvs/scratch/bin/python examples/source_gpt/source_gpt.py
MODE=bench ./scripts/bench/gpu_lock.sh ./jaithon run examples/source_gpt/generate.jai
MODE=gen   ./scripts/bench/gpu_lock.sh ~/.venvs/scratch/bin/python examples/source_gpt/source_gpt.py
MODE=checker SAMPLES=50 ./jaithon run examples/source_gpt/generate.jai
```

Two protocol details that favour one side, stated rather than hidden:

- jaitensor's `download` waits for the work that produced the loss and no
  more. `GPT_FLUSH=2` (the default) commits each block's work as it is encoded
  and the forward before the backward, so the download can return before the
  backward has finished; the next step's forward queues behind it. torch's
  `loss.item()` synchronises the whole stream. `GPT_FLUSH=0` commits once, at
  the download, and is the fully synchronous step.
- The 2,000-step wall times include 7 evaluations of 8 held-out batches on
  both sides (about a second) and exclude reading the corpus.

## Switches

All read once, at start.

| | default | |
|---|---|---|
| `GPT_FLUSH` | 2 | 0 commits at the loss download, 1 also after the forward, 2 also after every block. Step, locked, two alternated rounds: 48.6 / 47.2 ms at 0, 45.9 / 46.7 at 1, 45.6 / 45.0 at 2 |
| `GPT_BATCH_FIRST` | 0 | 1 trains batch-first, which copies Q, K, V and the context per layer: 48.2 / 48.1 ms against 45.4 / 52.5 (the second sequence-first round was disturbed); the attention alone is the 0.57 ms a layer above |
| `GPT_FUSED_RESIDUAL` | 1 | 0 decodes with a separate residual add instead of `forward_add`: 1.91 / 1.87 / 1.94 ms a byte against 1.86 / 1.89 / 1.82, inside the noise |
| `JAITENSOR_POSITION_VIEW` | 1 | jaitensor's: 0 makes a frozen `PositionalEmbedding` gather its rows instead of viewing them |
| `GPT_LAYERS`, `GPT_DIM`, `GPT_HEADS`, `GPT_SEQ`, `GPT_BATCH` | 4, 256, 4, 128, 64 | the shape; `source_gpt.py` reads the same |
| `STEPS`, `LR`, `SEED`, `EVAL_EVERY`, `EVAL_BATCHES` | 2000, 1e-3, 7, 250, 8 | `train.jai` |
| `NEW`, `TEMP`, `TOPK`, `SAMPLES` | 600, 0.8, 40, 50 | `generate.jai` |

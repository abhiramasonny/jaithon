# story_chat

A GPT you can talk to, built and trained end to end in Jaithon on human-written
English: a byte-level BPE tokenizer learned here, a 27.4M-parameter model
pretrained on TinyStories, then fine-tuned on DailyDialog conversations with
the loss on the bot's turns only, and a terminal chat that streams its replies
through the attention cache.

```bash
./jaithon run examples/story_chat/check.jai          # the acceptance check, about 5 s
./examples/story_chat/run_full.sh                    # the whole pipeline: 2 h of pretraining, in locked 5-minute chunks
./jaithon run examples/story_chat/chat.jai           # talk to it
./jaithon run examples/story_chat/generate.jai -- "Once upon a time, a little fox"
```

Run everything from the repository root. Nothing it writes lands in the
repository: token files, tokenizers, checkpoints and logs go to
`~/.cache/jaithon/story_chat/` (`STORY_CACHE`).

**The numbers and samples below come from a 15-minute pretraining run, an
eighth of the real budget.** The two-hour run is `run_full.sh` with its
defaults; it has not been run yet.

## Data

Read-only, never copied into the repository:

- `~/Developer/datasets/tinystories/TinyStoriesV2-GPT4-{train,valid}.txt`
  (`STORY_DATA`): 2.23 GB and 22.5 MB, 2,717,495 and 27,630 short stories in
  the vocabulary of a small child, separated by `<|endoftext|>`.
- `~/Developer/datasets/dailydialog/{train,validation,test}/dialogues_*.txt`
  (`DIALOG_DATA`): 11,118 / 1,000 / 1,000 human-written everyday
  conversations, one a line, turns separated by `__eou__`. The text has a space
  before every punctuation mark (`Say , Jim , how about ...?`) and curly
  apostrophes with spaces round them (`I ’ m`, and 540 lines of `I ’ Ve`);
  `tokenizer.jai`'s `normalise_turn` writes them the way English does.

A missing file is an error that names the path and the variable that moves it.
`prepare.jai` refuses a train file that is not 2,227,753,162 bytes (a
download still in progress).

## The pieces

| file | what it does |
|---|---|
| `tokenizer.jai` | byte-level BPE: GPT-2's piece cutter, merges learned with incremental pair counts, a piece cache for encoding, merges saved as text |
| `prepare.jai` | learns the tokenizer from 64 MB of TinyStories train plus all of DailyDialog train, then encodes both TinyStories splits to uint16 token files, 64 MB at a time |
| `model.jai` | the GPT, the token files and batches, held-out loss and bits per byte, AdamW with decay on the matrices only, optimiser state for checkpoints |
| `pretrain.jai` | pretraining: random windows, warm-up and cosine over a time budget, clipping, held-out evaluation and a sample every 250 steps, `--minutes` chunks and `--resume`; `MODE=bench` times the step, `MODE=eval` evaluates a checkpoint |
| `dialog.jai` | DailyDialog as `<|user|>`/`<|bot|>` turns, the loss mask, length buckets, the chat prompt |
| `finetune.jai` | fine-tuning on the bot's tokens, an epoch per run with `--one-epoch --resume` |
| `sample.jai` | temperature, top-k, top-p and repetition penalty on downloaded logits; cached decoding with a sliding context |
| `chat.jai` | the terminal chat |
| `generate.jai` | story continuation |
| `check.jai` | the fast acceptance check |
| `story_chat.py` | the torch MPS peer, and a Python re-implementation of the BPE that checks the token files |
| `run_full.sh`, `bench.sh`, `compare_vocab.sh` | the pipeline and the two measurements |

## What 15 minutes of pretraining gives

The reduced run that proved `run_full.sh` end to end:

```bash
FRESH=1 PRETRAIN_MINUTES=15 CHUNK_MINUTES=5 FT_EPOCHS=2 ./examples/story_chat/run_full.sh
```

It took 21 minutes of wall clock (the lock came round quickly that time):
15 minutes of pretraining in four locked chunks, 2,728 steps, **22,347,776
tokens at 24,830 tokens a second**; two fine-tuning epochs; the final
evaluation. The fine-tuning was then redone at a better learning rate (below)
from the same pretrained checkpoint, with
`FROM_CKPT=.../run-15m/pretrain.ckpt LR=3e-4` and three `--one-epoch --resume`
runs under the lock; 3e-4 and three epochs are now the defaults.

### Pretraining

Held-out loss on the TinyStories valid split, 256 windows (65,536 tokens)
every 250 steps, and over 2,048 windows (524,288 tokens, 2.08 MB of text) at
the end:

| step | tokens | minutes trained | train loss | held-out loss | bits per byte |
|---|---|---|---|---|---|
| 250 | 2.0M | 1.6 | 3.513 | 3.419 | 1.246 |
| 500 | 4.1M | 2.8 | 2.775 | 2.749 | 1.002 |
| 750 | 6.1M | 4.1 | 2.432 | 2.447 | 0.892 |
| 1000 | 8.2M | 5.4 | 2.238 | 2.248 | 0.819 |
| 1500 | 12.3M | 8.2 | 2.004 | 2.020 | 0.736 |
| 2000 | 16.4M | 11.3 | 1.855 | 1.878 | 0.684 |
| 2500 | 20.5M | 13.9 | 1.792 | 1.810 | 0.659 |
| 2728 | 22.3M | 15.0 | 1.778 | 1.792 | 0.653 |
| final, 2,048 windows | | | | **1.768** | **0.642** (perplexity 5.86) |

Bits per byte counts every byte of the targets' text, `<|endoftext|>` as one;
the tokenizer packs 3.96 bytes into a token on these windows. The first
eval's 1.6 minutes include compiling every kernel.

Three stories from `generate.jai -- "Once upon a time"` (temperature 0.8,
top-k 40, top-p 0.95, seed 1) after the 15 minutes, as printed:

> Once upon a time, there was a little girl named Lily. She was an older girl
> who loved to play with her toys. One day, she saw a long, soft bed in the
> living room. She wanted to play with her toys, but she knew she had to share
> it.
> Lily went to the living room and asked her mom, "Can I play with my toys?"
> Her mom smiled and said, "Yes, you can play with your toys, but first, you
> have to ask before you play with your toys first."
> Lily was very happy to share her toys with her mom. She asked her mom to help
> her keep her toys and her toys. Her mom gave her a big hug and said, "Thank
> you for the fun day, Lily! I love you very much."
> Lily learned that sharing was good and not bossy, and she always played with
> her toys with her friends. The moral of the story is to always share and
> play together.

> Once upon a time, there was a little girl named Lily. She had a big, thick
> tree in her yard. In the tree, she would sit on it and look at the pretty
> flowers. She liked to play with her friends in the park.
> One day, Lily was playing with her friend Tom. Tom had a long arm. He was not
> happy. He said, "Hi, Lily! Do you want to play with me?" Lily looked at Tom
> and said, "Yes, I like to play with you!"
> So, Lily and Tom played together with the long tree. They laughed and had
> fun. They were happy to have each other. And from that day on, Lily and Tom
> were the best of friends.

> Once upon a time, there was a little girl named Mia. Mia had a pet cat named
> Spot. Spot was a happy dog. They loved to play together all day. [...] Then,
> something unexpected happened. Spot found the toys near a big tree. He
> brought them back to Mia. They were very happy. They all played together
> with the toys and became good friends.

Grammatical, on topic, with a beginning, a middle and an end; the slips are
of meaning (a cat that is a happy dog, "her toys and her toys"). Decoding
through the cache ran at 442-554 tokens a second, batch 1, under the lock.

### Fine-tuning

The masked loss -- over the bot's tokens and turn ends only -- on DailyDialog
validation (2,210 rows), from the same 15-minute checkpoint:

| | LR 1e-4, 2 epochs (`run_full.sh` as first run) | LR 3e-4, 3 epochs (now the default) |
|---|---|---|
| before | 5.826 | 5.822 |
| epoch 1 | 3.367 | 3.244 |
| epoch 2 | 3.235 | 2.978 |
| epoch 3 | | **2.918** (train 2.303) |

The pretrained model has never seen `<|user|>` or `<|bot|>` and answers the
first probe by carrying on with a story. 24,832 training rows (3.25M tokens,
1.51M of them scored) make an epoch of 779 steps; an epoch took 137-303 s
depending on what else was using the GPU. The third epoch still helps, with
the gap to the training loss opening.

A scripted conversation (temperature 0.7, top-k 40, top-p 0.9, repetition
penalty 1.1, the defaults), verbatim, from the 3e-4 model:

```
you> Hi! How are you today?
bot> Not bad. I have a bad cold.
you> What did you do this weekend?
bot> Well, my stomach was born in the United States. It was none of my life, but I heard that there were some fluents in my mind and Ir I had a bad headache for my pre-servisors.
you> That sounds fun. Do you like to read?
bot> Yes, I do. It's very easy to get out of the noodles and I think I've heard of it before!
you> What is your favourite book?
bot> My book is on sale today, but I'm a bit tired of it.
you> Thanks, bye!
bot> I guess we can read that before we read a book.
```

And at temperature 0.5:

```
you> Hello, what is your name?
bot> My name is Li.
you> Where are you from?
bot> I'm from Germany.
you> Can you recommend a good restaurant?
bot> Yes, I can.
you> I would like to book a table for two tonight.
bot> What kind of food do you like best?
you> How much does it cost?
bot> I want $ 750.
```

It answers in English sentences, in DailyDialog's register -- names, jobs,
shopping, travel, a waiter's "What kind of food do you like best?" -- and takes
its turn and hands it back. Short exchanges land; long replies wander, and it
does not hold a conversation's thread for long. Fifteen minutes is 4% of an
epoch of TinyStories, and DailyDialog's adult vocabulary (hotels, banks,
interviews) is mostly new to a model raised on children's stories. The
two-hour run is what the example is for.

## The tokenizer

Byte-level BPE, trained in Jaithon. Ids 0-255 are the bytes, 256-258 are
`<|endoftext|>`, `<|user|>` and `<|bot|>`, and 259 onwards are the merges in
the order they were learned. Text is first cut into pieces as GPT-2 cuts it --
a word with the one space before it (an apostrophe between letters is part of
the word), a run of digits, a run of punctuation, other white space -- and no
merge crosses a piece, so a piece's ids depend only on the piece: encoding is a
dictionary lookup for every piece seen before, which in TinyStories is nearly
all of them (75,814 distinct pieces in 2.2 GB).

| | 4096 ids | 8192 ids |
|---|---|---|
| learning the merges (64 MB of stories + DailyDialog train, 35,164 distinct pieces) | 4.4 s counting, 0.2 s merging | 4.5 s, 1.0 s |
| bytes a token, train split | 4.049 | 4.183 |
| encoding the 2.2 GB train split, once | **117.7 s, 18.1 MB/s**, 550,178,804 tokens | 216 s (machine under load), 532,557,079 tokens |
| the Python BPE in `story_chat.py`, from the saved merges, against the token file | all 5,554,294 valid ids equal | all 5,377,642 equal |

The target for encoding the corpus was ten minutes; it takes two, in one
process. The encoder's outer loop does not compile, and that was measured
rather than chased: `JAI_JIT_WHY=1` names `dict.get` with a nullable result
(the piece cache) as having no result kind in the tier, and rewriting it as
`has` plus an index only moves the refusal to a `bytes` type guard on
`data.slice`'s result. At 18 MB/s it is not worth the chain.

**4096 ids, not 8192.** `compare_vocab.sh` measured both at the full model
shape, under the lock:

| | 4096 | 8192 |
|---|---|---|
| parameters | 27,427,840 | 29,524,992 |
| ms a step, 5 x 5 after 3 s warm-up (one acquisition, back to back) | **278.9** (29,372 tokens/s) | 326.1 (25,124 tokens/s) |
| 2 minutes of training: steps, tokens | 416, 3.41M | 377, 3.09M |
| held-out bits per byte after 2 minutes | 1.1145 | **1.0987** |
| 5 minutes of training: steps, tokens (a busy machine: 18,794 and 18,045 tokens/s) | 689, 5.64M | 661, 5.41M |
| held-out bits per byte after 5 minutes | **0.9401** | 0.9471 |

The bigger vocabulary starts ahead -- each of its tokens carries more of the
text -- and falls behind by five minutes: TinyStories is written in a small
vocabulary, so doubling the ids buys only 3.3% more bytes a token (4.183
against 4.049), while the output head and its gradients and the embedding
double and the step costs 17% more. The trend runs the way the two-hour run
goes, so 4096 it is. Each figure is one run; the two 5-minute runs saw the
same contention (their rates are within 4%), the 2-minute ones a quiet GPU.

## The model

8 layers, width 512, 8 heads, a 256-token context, vocabulary 4096, batch 32
(8,192 tokens a step): **27,427,840 parameters**, 2.1M of them the tied
embedding. Pre-LN blocks of causal multi-head attention and a GELU MLP,
learned positions, the output head tied to the token embedding --
`examples/source_gpt`'s model with 8.4x the parameters and 16x the vocabulary.
Weights start at GPT-2's 0.02 (as a uniform range), the two projections that
write the residual stream scaled by 1/sqrt(16). AdamW, betas 0.9/0.95, weight
decay 0.1 on the matrices only, the gradient clipped to norm 1; pretraining at
6e-4 after 200 warm-up steps, cosine to a tenth.

The weights are drawn on the host from one `std.random` generator rather than
by jaitensor's `fill_uniform`: that kernel gives correlated tensors for
different seeds (one xorshift round over a Weyl sequence; the decode track is
fixing it this wave), and a deep model built from it starts with layers that
resemble each other.

## Training speed against torch

`bench.sh` holds the GPU lock once and alternates the sides; each takes 3 s of
wall-clock warm-up then 5 x 5 steps, reading the loss every step. The peer is
`story_chat.py`: torch 2.13.0 on MPS, the same shape, initialisation, AdamW
groups, clipping and token file, scaled-dot-product attention.

Three acquisitions, each alternating the sides, at three levels of load from
the dozen other agents on the machine (load average 3 to 110); the ratio held
where the absolute numbers did not:

| | ms a step (median of 5 x 5), two runs | tokens a second | jai faster by |
|---|---|---|---|
| **jai**, busy (load ~40) | **347.1 / 346.5** | **23,603 / 23,640** | |
| torch MPS, default AdamW | 440.2 / 495.8 | 18,610 / 16,523 | 1.27x / 1.43x |
| jai, `GPT_MIXED=1` (products in float16) | 394.2 / 364.9 | 20,780 / 22,449 | |
| **jai**, quiet (load ~3) | **247.8 / 274.0** | **33,060 / 29,898** | |
| torch MPS, fused AdamW (`PEER_FUSED=1`) | 363.5 / 362.0 | 22,534 / 22,632 | 1.47x / 1.32x |
| **jai**, very busy (load 60-110) | **395.4 / 406.1** | **20,716 / 20,172** | |
| torch MPS, default AdamW | 525.8 / 550.3 | 15,579 / 14,887 | 1.33x / 1.36x |

**Jaithon trains this model 1.27-1.47x as fast as torch MPS a step**, against
torch's default and its fused AdamW alike, and both sides start from the same
loss (8.41 and 8.45; ln 4096 is 8.32). The quiet pair is the cleanest: 33,060
and 29,898 tokens a second against 22,534 and 22,632.

Float16 products (`GPT_MIXED=1`) do not help at this size -- the conversions
cost more than they save -- so they stay off. Over the real run the rate is a
little lower than the bench, because every 250 steps the run evaluates 256
held-out windows and samples a story; `pretrain.jai` reports training-only
tokens a second, which over the 15-minute run was 24,830 on a busy machine.

## The full two-hour run

```bash
./examples/story_chat/run_full.sh
```

That is `PRETRAIN_MINUTES=120` of pretraining -- 24 chunks of 5 minutes, each
its own turn on `scripts/bench/gpu_lock.sh` -- then three fine-tuning epochs
(one turn each) and the final evaluation. At 25,000-30,000 tokens a second two
hours is 180-215M tokens, a third of an epoch of TinyStories, about 24,000
steps. The schedule is measured in training time, so the cosine reaches its
floor when the two hours are up whatever the step costs. The wall time is two
hours, plus evaluation and checkpoints (a few percent), plus however long the
lock keeps it waiting between chunks: with a dozen agents queued, a turn took
anywhere from seconds to over an hour to come round during this work, so the
scripts wait up to a day for it (`GPU_LOCK_WAIT`).

It is restartable at any point: run it again and it carries on from the last
checkpoint (`FRESH=1` starts over). The run writes to
`~/.cache/jaithon/story_chat/run-120m/` -- `pretrain.ckpt` (329 MB: weights and
AdamW moments), `chat.ckpt` (110 MB, weights), `chat.ckpt.state`,
`pretrain_log.tsv`, `finetune_log.tsv`, `run_full.log`, `final_eval.txt` -- and
links the two checkpoints into `~/.cache/jaithon/story_chat/`, where `chat.jai`
and `generate.jai` look.

By hand, the same stages:

```bash
./jaithon run examples/story_chat/prepare.jai     # once: tokenizer and token files, about 2 minutes
./scripts/bench/gpu_lock.sh ./jaithon run examples/story_chat/pretrain.jai -- --budget 120 --minutes 5 --resume  # until "schedule finished"
./scripts/bench/gpu_lock.sh ./jaithon run examples/story_chat/finetune.jai -- --epochs 3 --one-epoch --resume    # until "fine-tuning finished"
./jaithon run examples/story_chat/chat.jai
```

## Switches

All read once, at start.

| | default | |
|---|---|---|
| `STORY_DATA`, `DIALOG_DATA` | `~/Developer/datasets/tinystories`, `.../dailydialog` | the datasets |
| `STORY_CACHE` | `~/.cache/jaithon/story_chat` | token files, tokenizers, linked checkpoints |
| `RUN_DIR` | `STORY_CACHE` (`run_full.sh`: `STORY_CACHE/run-<N>m`) | checkpoints and logs of a run |
| `PRETRAIN_CKPT`, `CHAT_CKPT`, `FROM_CKPT` | `RUN_DIR/pretrain.ckpt`, `RUN_DIR/chat.ckpt`, the pretrain checkpoint | `FROM_CKPT` is what fine-tuning starts from |
| `VOCAB` | 4096 | which tokenizer and token files (`prepare.jai` makes them) |
| `SAMPLE_MB`, `BLOCK_MB` | 64, 64 | `prepare.jai`: the tokenizer's sample, the encoding block |
| `GPT_LAYERS`, `GPT_DIM`, `GPT_HEADS`, `GPT_SEQ`, `GPT_BATCH` | 8, 512, 8, 256, 32 | the shape; `story_chat.py` reads the same |
| `LR`, `WARMUP`, `MIN_LR_FRAC`, `WD`, `CLIP`, `SEED` | 6e-4, 200, 0.1, 0.1, 1.0, 7 | pretraining (fine-tuning: `LR` 3e-4, `WARMUP` 50, `SEED` 11) |
| `EVAL_EVERY`, `EVAL_WINDOWS`, `CKPT_EVERY`, `LOG_EVERY`, `SAMPLE_PROMPT` | 250, 256, 1000, 50, "Once upon a time" | pretraining |
| `GPT_FLUSH` | 2 | as in source_gpt: commit the queued work after every block |
| `GPT_MIXED` | 0 | 1 runs the products in float16; slower here (above), so off |
| `FT_BATCH`, `QUANTUM`, `EPOCHS` | 32, 64, 3 | fine-tuning: rows are padded to a multiple of `QUANTUM` tokens |
| `TEMP`, `TOPK`, `TOPP`, `REP`, `MAX_NEW`, `SEED`, `CHAT_ECHO` | 0.7, 40, 0.9, 1.1, 64, 1, 0 | `chat.jai` (`generate.jai`: 0.8, 40, 0.95, 1.0, `NEW` 300, `SAMPLES` 1, `CKPT`) |
| `PRETRAIN_MINUTES`, `CHUNK_MINUTES`, `FT_EPOCHS`, `FRESH`, `GPU_LOCK_WAIT` | 120, 5, 3, 0, 86400 | `run_full.sh` |
| `PEER_FUSED` | 0 | `story_chat.py`: 1 uses torch's fused AdamW |

## What the example needed, and what it found

Nothing in jaitensor had to change; everything this needed is on main:
batched causal `MultiHeadAttention` with a KV cache and `forward_add`,
`LayerNorm` at any rank, `Embedding.lookup`, `PositionalEmbedding.forward_at`,
`gather_rows` with a gradient, `AdamW`, `clip_grad_norm`, and binary
checkpoints. Two things it works around rather than fixes, because other
tracks own them this wave:

- **Correlated initial weights** from `fill_uniform` (decode track): drawn on
  the host instead, above.
- **The masked loss** without touching `losses.jai` (sleep track): the hidden
  states at the bot's positions are gathered with `gather_rows` before the
  output head, so the loss is the plain mean cross-entropy over just those
  rows -- and the head's product is only as big as the bot's share of the
  batch. `check.jai` confirms that gathering every row gives the plain loss to
  the last bit, and that a dialogue batch scores exactly the bot's tokens and
  the token that ends each of its turns.

`check.jai` also proves the resume path the chunked run depends on: a trainer
restored from a checkpoint -- weights, both AdamW moments, the step count --
takes the same next two steps as the original, to the bit.

Three things it ran into in its own tooling. `gpu_lock.sh` gives up after an
hour by default, which with the machine this busy cancelled a queued
benchmark; the scripts now wait up to a day. A chunk that stops at five
minutes of wall clock leaves the evaluation time it spent as a few seconds of
budget, which used to queue for a turn of its own; a leftover under a fifth
of a chunk is now finished in the chunk. And the disk: a run directory holds
about a gigabyte at its peak (the 329 MB pretraining checkpoint is written
beside the old one before it replaces it), and the disk this ran on filled up
during the vocabulary comparison (346 MB free of 787 GB, shared with every
other agent). The best chat model is now saved as weights alone (110 MB),
and the comparison's checkpoints were deleted; leave a few gigabytes free
before the two-hour run.

# story_chat

A GPT you can talk to, built and trained end to end in Jaithon on human-written
English. A byte-level BPE tokenizer learned here, a 27.4M-parameter model
pretrained on TinyStories, then fine-tuned on DailyDialog conversations with
the loss on the bot's turns only, and a terminal chat that streams its replies
through the attention cache.

```bash
./jaithon run examples/story_chat/check.jai          # the acceptance check, about 5 s
./examples/story_chat/run_full.sh                    # the whole pipeline: ~2 h 15 min of GPU, in locked 5-minute chunks
./jaithon run examples/story_chat/chat.jai           # talk to it
./jaithon run examples/story_chat/generate.jai -- "Once upon a time, a little fox"
```

Run everything from the repository root. Nothing it writes lands in the
repository: token files, the tokenizer, checkpoints and logs go to
`~/.cache/jaithon/story_chat/` (`STORY_CACHE`).

## Data

Read-only, never copied into the repository:

- `~/Developer/datasets/tinystories/TinyStoriesV2-GPT4-{train,valid}.txt`
  (`STORY_DATA`): 2.23 GB and 22.5 MB, 2,717,495 and 27,630 short stories
  written by GPT-4 in the vocabulary of a small child, separated by
  `<|endoftext|>`.
- `~/Developer/datasets/dailydialog/{train,validation,test}/dialogues_*.txt`
  (`DIALOG_DATA`): 11,118 / 1,000 / 1,000 human-written everyday
  conversations, one a line, turns separated by `__eou__`. The text has a space
  before every punctuation mark (`Say , Jim , how about ...?`), which
  `tokenizer.jai`'s `normalise_turn` puts back.

A missing file is an error that names the path and the variable that moves it.

## The pieces

| file | what it does |
|---|---|
| `tokenizer.jai` | byte-level BPE: GPT-2's piece cutter (a word with its leading space, digit runs, punctuation runs, white space), merges learned with incremental pair counts, a piece cache for encoding, merges saved as text |
| `prepare.jai` | learns the tokenizer from 64 MB of TinyStories train plus all of DailyDialog train, then encodes both splits to uint16 token files, 64 MB at a time |
| `model.jai` | the GPT, the token files and batches, held-out loss and bits per byte, AdamW with decay on the matrices only, checkpoints with optimiser state |
| `pretrain.jai` | pretraining: random windows, warm-up and cosine over a time budget, clipping, held-out evaluation and a sample every 250 steps, `--minutes` chunks and `--resume` |
| `dialog.jai` | DailyDialog as `<|user|>`/`<|bot|>` turns, the loss mask, length buckets, the chat prompt |
| `finetune.jai` | fine-tuning on the bot's tokens, an epoch per run with `--one-epoch --resume` |
| `sample.jai` | temperature, top-k, top-p and repetition penalty on downloaded logits; cached decoding with a sliding context |
| `chat.jai` | the terminal chat |
| `generate.jai` | story continuation |
| `check.jai` | the fast acceptance check |
| `story_chat.py` | the torch MPS peer, and a Python re-implementation of the BPE that checks the token files |
| `run_full.sh`, `bench.sh`, `compare_vocab.sh` | the pipeline and the two measurements |

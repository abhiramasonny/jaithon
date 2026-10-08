#!/bin/sh
# Re-measure the README's numbers: jai against the torch MPS peer, each
# command under scripts/bench/gpu_lock.sh, the two sides alternated.
#
#     sh examples/source_gpt/bench.sh [step|decode|train|all]
#
# Run from the repository root. PYTHON is the peer's interpreter (default
# ~/.venvs/scratch/bin/python). The GEMM plan cache is a scratch copy of
# yours, so tuning here never writes ~/.cache.
set -e
WHAT=${1:-all}
PY=${PYTHON:-$HOME/.venvs/scratch/bin/python}
LOCK=./scripts/bench/gpu_lock.sh
SCRATCH=${TMPDIR:-/tmp}/source_gpt_bench
mkdir -p "$SCRATCH"
if [ ! -f "$SCRATCH/plans.tsv" ]; then
  cp "$HOME/.cache/jaithon/gemm-plans.tsv" "$SCRATCH/plans.tsv" 2>/dev/null || : > "$SCRATCH/plans.tsv"
fi
export JAITENSOR_GEMM_CACHE="$SCRATCH/plans.tsv" GPT_CKPT="$SCRATCH/source_gpt.ckpt"

step() {
  # Step time: 2 s of steps by the wall clock, then 7 samples of 5 steps.
  for round in 1 2 3; do
    printf "step jai   r%s " $round
    MODE=bench WARM=2 REPS=7 STEPS=5 $LOCK ./jaithon run examples/source_gpt/train.jai 2>&1 | grep -E "first_step|train_ms" | tr '\n' ' '; echo
    printf "step torch r%s " $round
    MODE=bench WARM=2 REPS=7 STEPS=5 $LOCK "$PY" examples/source_gpt/source_gpt.py 2>&1 | grep -E "first_step|train_ms" | tr '\n' ' '; echo
  done
}

train() {
  # The whole run: 2,000 steps, held-out bits per byte, checkpoint save and load.
  $LOCK ./jaithon run examples/source_gpt/train.jai 2>&1 | grep -E "^(step [0-9]+ held|trained|final|checkpoint|live)"
  $LOCK "$PY" examples/source_gpt/source_gpt.py 2>&1 | grep -E "^(step [0-9]+ held|trained|final)"
}

decode() {
  # Cached batch-1 greedy decoding, 112 bytes after a 16-byte prompt.
  if [ ! -f "$GPT_CKPT" ]; then STEPS=50 ./jaithon run examples/source_gpt/train.jai > /dev/null 2>&1; fi
  for round in 1 2 3; do
    printf "decode jai   r%s " $round
    MODE=bench WARM=2 REPS=7 $LOCK ./jaithon run examples/source_gpt/generate.jai 2>&1 | grep -E "bytes_per_s|per_byte" | tr '\n' ' '; echo
    printf "decode torch r%s " $round
    MODE=gen WARM=2 REPS=7 $LOCK "$PY" examples/source_gpt/source_gpt.py 2>&1 | grep -E "bytes_per_s|per_token" | tr '\n' ' '; echo
  done
}

case "$WHAT" in
  step) step ;;
  decode) decode ;;
  train) train ;;
  all) step; train; decode ;;
  *) echo "usage: sh examples/source_gpt/bench.sh [step|decode|train|all]"; exit 2 ;;
esac

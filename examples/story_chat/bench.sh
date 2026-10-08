#!/bin/sh
# Training step time, Jaithon against the torch MPS peer, same shape and data.
#
#   ./examples/story_chat/bench.sh            # 2 rounds: jai, torch, then the reverse
#   ROUNDS=1 VOCAB=8192 ./examples/story_chat/bench.sh
#
# Run from the repository root after prepare.jai. The whole comparison holds
# the GPU lock once (about three minutes at the default shape); each side
# warms for WARM (3) seconds of wall clock, then times REPS (5) x BENCH_STEPS
# (5) steps, reading the loss every step. GPT_* and VOCAB set the shape on
# both sides. PEER_FUSED=1 gives torch its fused AdamW.
set -e
J="${JAITHON:-./jaithon}"
PY="${PYTHON:-$HOME/.venvs/scratch/bin/python}"
LOCK="${GPU_LOCK:-./scripts/bench/gpu_lock.sh}"
# gpu_lock.sh gives up after an hour by default; with many agents queued a
# turn can take longer than that to come round.
export GPU_LOCK_WAIT="${GPU_LOCK_WAIT:-86400}"
ROUNDS="${ROUNDS:-2}"
export WARM="${WARM:-3}" REPS="${REPS:-5}" BENCH_STEPS="${BENCH_STEPS:-5}"

jai() { echo "== jai"; MODE=bench "$J" run examples/story_chat/pretrain.jai | grep -v '^tokens:'; }
torch() { echo "== torch"; MODE=bench "$PY" examples/story_chat/story_chat.py; }

# ./examples/story_chat/bench.sh quality: both sides train STEPS (400) steps
# from scratch on the same schedule (WARMUP 40, LR 6e-4, cosine over the
# steps) and report the held-out loss on the same 256 windows every 100
# steps -- a check that the two learn the same thing, not only at the same
# speed. Each side takes the lock on its own.
if [ "${1:-}" = quality ]; then
    steps="${STEPS:-400}"
    dir="${STORY_CACHE:-$HOME/.cache/jaithon/story_chat}/quality-$steps"
    rm -rf "$dir"
    echo "== jai, $steps steps"
    RUN_DIR="$dir" WARMUP=40 EVAL_EVERY=100 EVAL_WINDOWS=256 LOG_EVERY=100 \
        "$LOCK" "$J" run examples/story_chat/pretrain.jai -- --steps "$steps" | grep -E "held-out|ran steps"
    rm -f "$dir"/*.ckpt
    echo "== torch, $steps steps"
    MODE=train STEPS="$steps" WARMUP=40 EVAL_EVERY=100 EVAL_WINDOWS=256 \
        "$LOCK" "$PY" examples/story_chat/story_chat.py
    exit 0
fi

if [ "${INSIDE_LOCK:-0}" != 1 ]; then
    INSIDE_LOCK=1 exec "$LOCK" "$0" "$@"
fi
r=0
while [ "$r" -lt "$ROUNDS" ]; do
    if [ $((r % 2)) -eq 0 ]; then jai; torch; else torch; jai; fi
    r=$((r + 1))
done

#!/bin/sh
# 4096 against 8192 ids: step time, and held-out bits per byte after the same
# minutes of training, at the full model shape.
#
#   VOCAB=8192 ./jaithon run examples/story_chat/prepare.jai   # first, once
#   ./examples/story_chat/compare_vocab.sh
#
# Holds the GPU lock once, about six minutes: a step-time bench of each, then
# MINUTES (2) of training of each from scratch on the same time-based
# schedule, each evaluated on the same 512 held-out windows of its own
# encoding of the valid split. Bits per byte is the fair measure across
# tokenizers: the loss a token is not, since a bigger vocabulary packs more
# bytes into each.
set -e
J="${JAITHON:-./jaithon}"
LOCK="${GPU_LOCK:-./scripts/bench/gpu_lock.sh}"
# gpu_lock.sh gives up after an hour by default; with many agents queued a
# turn can take longer than that to come round.
export GPU_LOCK_WAIT="${GPU_LOCK_WAIT:-86400}"
MINUTES="${MINUTES:-2}"
CACHE="${STORY_CACHE:-$HOME/.cache/jaithon/story_chat}"
if [ "${INSIDE_LOCK:-0}" != 1 ]; then
    INSIDE_LOCK=1 exec "$LOCK" "$0" "$@"
fi
for v in 4096 8192; do
    echo "== vocab $v: step time"
    VOCAB=$v MODE=bench WARM=3 REPS=5 BENCH_STEPS=5 "$J" run examples/story_chat/pretrain.jai | grep -v '^tokens:'
done
for v in 4096 8192; do
    dir="$CACHE/compare-vocab-$v"
    rm -rf "$dir"
    echo "== vocab $v: $MINUTES minutes of training"
    VOCAB=$v RUN_DIR="$dir" EVAL_EVERY=100000 EVAL_WINDOWS=512 LOG_EVERY=200 \
        "$J" run examples/story_chat/pretrain.jai -- --budget "$MINUTES" | grep -v '^  sample' | tail -4
done

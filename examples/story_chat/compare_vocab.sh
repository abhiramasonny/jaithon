#!/bin/sh
# 4096 against 8192 ids: step time, and held-out bits per byte after the same
# minutes of training, at the full model shape.
#
#   VOCAB=8192 ./jaithon run examples/story_chat/prepare.jai   # first, once
#   ./examples/story_chat/compare_vocab.sh                     # MINUTES=2
#   BENCH=0 MINUTES=5 ./examples/story_chat/compare_vocab.sh
#
# Each stage takes the GPU lock on its own: a step-time bench of both
# vocabularies (about 1.5 minutes; BENCH=0 skips it), then MINUTES (2) of
# training of each from scratch on the same time-based schedule, each
# evaluated on 512 held-out windows of its own encoding of the valid split.
# Bits per byte is the fair measure across tokenizers: the loss a token is
# not, since a bigger vocabulary packs more bytes into each. Runs go to
# STORY_CACHE/compare-vocab-<V>-<MINUTES>m.
set -e
J="${JAITHON:-./jaithon}"
LOCK="${GPU_LOCK:-./scripts/bench/gpu_lock.sh}"
# gpu_lock.sh gives up after an hour by default; with many agents queued a
# turn can take longer than that to come round.
export GPU_LOCK_WAIT="${GPU_LOCK_WAIT:-86400}"
MINUTES="${MINUTES:-2}"
CACHE="${STORY_CACHE:-$HOME/.cache/jaithon/story_chat}"
VOCABS="${VOCABS:-4096 8192}"

if [ "${BENCH:-1}" = 1 ]; then
    cat > "$CACHE/compare_vocab_bench.sh" <<EOF
#!/bin/sh
for v in $VOCABS; do
    echo "== vocab \$v: step time"
    VOCAB=\$v MODE=bench WARM=3 REPS=5 BENCH_STEPS=5 "$J" run examples/story_chat/pretrain.jai | grep -v '^tokens:'
done
EOF
    "$LOCK" sh "$CACHE/compare_vocab_bench.sh"
fi
for v in $VOCABS; do
    dir="$CACHE/compare-vocab-$v-${MINUTES}m"
    rm -rf "$dir"
    echo "== vocab $v: $MINUTES minutes of training"
    VOCAB=$v RUN_DIR="$dir" EVAL_EVERY=100000 EVAL_WINDOWS=512 LOG_EVERY=1000 \
        "$LOCK" "$J" run examples/story_chat/pretrain.jai -- --budget "$MINUTES" \
        | grep -E "held-out|ran steps|model:"
done

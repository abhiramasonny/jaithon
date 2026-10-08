#!/bin/sh
# Training step time, Jaithon against the torch MPS peer, same shape and data.
#
#   ./examples/story_chat/bench.sh            # 2 rounds: jai, torch, jai fp16, then the reverse
#   ROUNDS=1 VOCAB=8192 ./examples/story_chat/bench.sh
#
# Run from the repository root after prepare.jai. The whole comparison holds
# the GPU lock once (about three minutes at the default shape); each side
# warms for WARM (3) seconds of wall clock, then times REPS (5) x BENCH_STEPS
# (5) steps, reading the loss every step. GPT_* and VOCAB set the shape on
# both sides. PEER_FUSED=1 gives torch its fused AdamW. "jai fp16" is
# GPT_MIXED=1, the large products in float16.
set -e
J="${JAITHON:-./jaithon}"
PY="${PYTHON:-$HOME/.venvs/scratch/bin/python}"
LOCK="${GPU_LOCK:-./scripts/bench/gpu_lock.sh}"
ROUNDS="${ROUNDS:-2}"
export WARM="${WARM:-3}" REPS="${REPS:-5}" BENCH_STEPS="${BENCH_STEPS:-5}"

jai() { echo "== jai"; MODE=bench "$J" run examples/story_chat/pretrain.jai | grep -v '^tokens:'; }
mixed() { echo "== jai GPT_MIXED=1"; GPT_MIXED=1 MODE=bench "$J" run examples/story_chat/pretrain.jai | grep -v '^tokens:'; }
torch() { echo "== torch"; MODE=bench "$PY" examples/story_chat/story_chat.py; }

if [ "${INSIDE_LOCK:-0}" != 1 ]; then
    INSIDE_LOCK=1 exec "$LOCK" "$0" "$@"
fi
r=0
while [ "$r" -lt "$ROUNDS" ]; do
    if [ $((r % 2)) -eq 0 ]; then jai; torch; mixed; else mixed; torch; jai; fi
    r=$((r + 1))
done

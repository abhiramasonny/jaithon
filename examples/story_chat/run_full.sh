#!/bin/sh
# The whole story_chat pipeline: tokenizer, corpus, pretraining, fine-tuning,
# a final evaluation.
#
#   sh examples/story_chat/run_full.sh                       # the full run: 120 minutes of pretraining
#   PRETRAIN_MINUTES=15 FT_EPOCHS=1 sh examples/story_chat/run_full.sh   # a reduced run
#
# Run it from the repository root. Every GPU stage runs under
# scripts/bench/gpu_lock.sh, and pretraining is split into chunks of
# CHUNK_MINUTES (5), each its own lock acquisition, so other measurements on
# the machine get the GPU between chunks. Fine-tuning takes one acquisition
# an epoch. The schedule is PRETRAIN_MINUTES of training time, evaluation
# excluded, so the cosine ends when the time does whatever the step costs.
#
# Between two acquisitions the script waits LOCK_YIELD (5) seconds before it
# asks again. gpu_lock.sh polls every 2 s, so without the wait the next chunk
# took the lock back 25-50 ms after letting it go, ahead of every waiter.
#
# Checkpoints and logs go to RUN_DIR (default
# $STORY_CACHE/run-<PRETRAIN_MINUTES>m), and when the run finishes its two
# checkpoints are linked into $STORY_CACHE, where chat.jai and generate.jai
# look by default. Re-running carries on from wherever the last run stopped;
# FRESH=1 starts the run directory again.
set -e

J="${JAITHON:-./jaithon}"
LOCK="${GPU_LOCK:-./scripts/bench/gpu_lock.sh}"
LOCK_YIELD="${LOCK_YIELD:-5}"
# This directory, as the path the script was started by.
case "$0" in */*) EX="${0%/*}" ;; *) EX=. ;; esac
# gpu_lock.sh gives up after an hour by default; with many agents queued a
# turn can take longer than that to come round.
export GPU_LOCK_WAIT="${GPU_LOCK_WAIT:-86400}"
CACHE="${STORY_CACHE:-$HOME/.cache/jaithon/story_chat}"
PRETRAIN_MINUTES="${PRETRAIN_MINUTES:-120}"
CHUNK_MINUTES="${CHUNK_MINUTES:-5}"
FT_EPOCHS="${FT_EPOCHS:-3}"
RUN_DIR="${RUN_DIR:-$CACHE/run-${PRETRAIN_MINUTES}m}"
export STORY_CACHE="$CACHE" RUN_DIR

if [ ! -x "$J" ] || [ ! -f "$EX/pretrain.jai" ]; then
    echo "run_full.sh: run from the repository root, after make" >&2
    exit 2
fi
if [ "${FRESH:-0}" = 1 ]; then rm -rf "$RUN_DIR"; fi
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/run_full.log"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
# Let the lock go round before asking for it again.
held=0
yield_gpu() {
    if [ "$held" = 1 ]; then sleep "$LOCK_YIELD"; fi
    held=1
}

say "run_full: ${PRETRAIN_MINUTES} min pretraining in ${CHUNK_MINUTES} min chunks, ${FT_EPOCHS} fine-tuning epochs, run dir $RUN_DIR"

# 1. Tokenizer and token files (CPU only; skipped when present).
say "prepare: tokenizer and corpus"
"$J" run "$EX/prepare.jai" 2>&1 | tee -a "$LOG"

# 2. Pretraining, a locked chunk at a time. Exit status 10 means more to do.
chunk=0
while true; do
    chunk=$((chunk + 1))
    yield_gpu
    say "pretrain chunk $chunk (waiting for the GPU lock)"
    set +e
    "$LOCK" "$J" run "$EX/pretrain.jai" -- \
        --budget "$PRETRAIN_MINUTES" --minutes "$CHUNK_MINUTES" --resume 2>&1 | tee -a "$LOG"
    status=$(tail -1 "$LOG")
    set -e
    case "$status" in
        "schedule finished"*) break ;;
        *"run again with --resume"*) ;;
        *) say "pretrain chunk $chunk failed: $status"; exit 1 ;;
    esac
done
say "pretraining finished"

# 3. Fine-tuning on DailyDialog, a locked epoch at a time.
while true; do
    yield_gpu
    say "fine-tune epoch (waiting for the GPU lock)"
    set +e
    "$LOCK" "$J" run "$EX/finetune.jai" -- \
        --epochs "$FT_EPOCHS" --one-epoch --resume 2>&1 | tee -a "$LOG"
    status=$(tail -1 "$LOG")
    set -e
    case "$status" in
        "fine-tuning finished"*|"fine-tuning already finished"*) break ;;
        *"run again with --resume"*) ;;
        *) say "fine-tuning failed: $status"; exit 1 ;;
    esac
done
say "fine-tuning finished"

# 4. Final evaluation: held-out bits per byte over 2048 windows, the chat
#    model on the validation and test dialogues, three
#    stories, and a scripted conversation.
yield_gpu
say "final evaluation (waiting for the GPU lock)"
cat > "$RUN_DIR/final_eval.sh" <<EOF
#!/bin/sh
MODE=eval EVAL_WINDOWS=2048 "$J" run "$EX/pretrain.jai" -- --resume
"$J" run "$EX/finetune.jai" -- --eval
SAMPLES=3 NEW=200 "$J" run "$EX/generate.jai" -- "Once upon a time"
printf 'Hi! How are you today?\nWhat did you do this weekend?\nThat sounds fun. Do you like to read?\nWhat is your favourite book?\nThanks, bye!\n' \
    | CHAT_ECHO=1 "$J" run "$EX/chat.jai"
EOF
"$LOCK" sh "$RUN_DIR/final_eval.sh" 2>&1 | tee "$RUN_DIR/final_eval.txt" | tee -a "$LOG"

ln -sf "$RUN_DIR/pretrain.ckpt" "$CACHE/pretrain.ckpt"
ln -sf "$RUN_DIR/chat.ckpt" "$CACHE/chat.ckpt"
say "done: $CACHE/chat.ckpt -> $RUN_DIR/chat.ckpt"
say "talk to it: $J run $EX/chat.jai"

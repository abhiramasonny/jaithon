#!/bin/sh
# run_full.sh's control flow without a GPU: every stage is a stand-in that
# prints what the real one prints at the end, and the lock is a stand-in that
# logs when it is asked for and when it is let go. It requires that the
# pipeline gets from the first chunk to "done", and that each request for the
# lock comes at least LOCK_YIELD after the previous hold ended -- otherwise
# the next chunk takes the lock back before any waiter polling every 2 s can.
#
#   sh examples/story_chat/check_run_full.sh                 # this run_full.sh
#   sh examples/story_chat/check_run_full.sh path/to/run_full.sh
#
# Run it from the repository root. It takes about 6 seconds.
set -e
case "$0" in */*) EX="${0%/*}" ;; *) EX=. ;; esac
SCRIPT="${1:-$EX/run_full.sh}"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/story_chat_run_full.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
now() { perl -MTime::HiRes=time -e 'printf "%.3f\n", time'; }

cat > "$TMP/jaithon" <<'FAKE'
#!/bin/sh
# Stand-in for ./jaithon run <file> -- <args>: prints each stage's last line.
stage=$(basename "$2" .jai)
count_file="$RUN_DIR/fake-$stage"
n=$(($(cat "$count_file" 2>/dev/null || echo 0) + 1))
echo "$n" > "$count_file"
case "$stage" in
    pretrain)
        if [ "${MODE:-}" = eval ]; then echo "held-out loss 1.0"
        elif [ "$n" -lt 3 ]; then echo "$((n * 33))% of the schedule done; run again with --resume to go on"
        else echo "schedule finished"; fi ;;
    finetune)
        case "$*" in
            *--eval*) echo "chat model evaluated" ;;
            *) if [ "$n" -lt 2 ]; then echo "epoch $n of 2 done; run again with --resume to go on"
               else echo "fine-tuning finished"; fi ;;
        esac ;;
    *) echo "$stage ran" ;;
esac
FAKE
cat > "$TMP/lock" <<FAKE
#!/bin/sh
echo "ask \$(perl -MTime::HiRes=time -e 'printf "%.3f", time')" >> "$TMP/lock.log"
"\$@"
status=\$?
echo "free \$(perl -MTime::HiRes=time -e 'printf "%.3f", time')" >> "$TMP/lock.log"
exit \$status
FAKE
chmod +x "$TMP/jaithon" "$TMP/lock"

yield=1
JAITHON="$TMP/jaithon" GPU_LOCK="$TMP/lock" LOCK_YIELD=$yield STORY_CACHE="$TMP/cache" \
    RUN_DIR="$TMP/cache/run" sh "$SCRIPT" > "$TMP/out.txt" 2>&1 || {
    cat "$TMP/out.txt"
    echo "FAIL  run_full.sh exited with an error"
    exit 1
}
failed=0
if grep -q "done: " "$TMP/out.txt"; then
    echo "PASS  the pipeline ran to the end (3 chunks, 2 epochs, the final evaluation)"
else
    cat "$TMP/out.txt"
    echo "FAIL  the pipeline did not reach the end"
    failed=1
fi
asks=$(grep -c '^ask' "$TMP/lock.log")
gaps=$(awk -v y="$yield" '
    /^free/ { last = $2 }
    /^ask/ && last != "" { gap = $2 - last; if (min == "" || gap < min) min = gap }
    END { printf "%.3f", min }' "$TMP/lock.log")
if [ "$asks" -eq 6 ] && awk -v g="$gaps" -v y="$yield" 'BEGIN { exit !(g >= y) }'; then
    echo "PASS  each of the $asks lock requests waited ${yield} s after the last hold (shortest gap ${gaps} s)"
else
    echo "FAIL  $asks lock requests; shortest gap between a hold ending and the next request ${gaps} s, wanted >= ${yield} s"
    failed=1
fi
exit $failed

#!/bin/sh
# Run a command with exclusive use of the GPU.
#
# Several agents work in this repository at once and any two of them measuring
# at the same time produce numbers that mean nothing: the jaicv suite read
# 1.98x under a load average of 4.5 and 3.10x on the same code with the machine
# quiet, and individual rows moved by a factor of three. Wrap anything you
# intend to believe:
#
#   ./scripts/gpu_lock.sh make bench jaicv
#   ./scripts/gpu_lock.sh ./jaithon probe.jai
#
# mkdir is the mutex because macOS has no flock(1). The lock carries the pid
# that took it, so a lock left behind by a killed process is reclaimed rather
# than blocking everyone for ever.
set -e

lock="${TMPDIR:-/tmp}/jaithon-gpu.lock"
waited=0
limit="${GPU_LOCK_WAIT:-3600}"

while ! mkdir "$lock" 2>/dev/null; do
    held=$(cat "$lock/pid" 2>/dev/null || echo "")
    if [ -n "$held" ] && ! kill -0 "$held" 2>/dev/null; then
        echo "gpu_lock: reclaiming the lock from dead pid $held" >&2
        rm -rf "$lock"
        continue
    fi
    if [ "$waited" -ge "$limit" ]; then
        echo "gpu_lock: gave up after ${limit}s waiting for pid ${held:-unknown}" >&2
        exit 75
    fi
    if [ "$waited" -eq 0 ]; then
        echo "gpu_lock: waiting for pid ${held:-unknown} to finish measuring" >&2
    fi
    sleep 2
    waited=$((waited + 2))
done

echo $$ > "$lock/pid"
trap 'rm -rf "$lock"' EXIT INT TERM

"$@"

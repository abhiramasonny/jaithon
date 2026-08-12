#!/usr/bin/env bash
# sum(jaiOpCounts) must equal vm.instructionCount.
#
# The per-opcode histogram and the total counter are incremented in different
# places -- jaiOpCounts in the dispatch macros, instructionCount in the VM
# struct -- so the two agreeing is the only evidence that no dispatch path
# skips the histogram. VM_NEXT_HINT skipped it for five sites, and loop_sum's
# OP_LOOP reported zero against a true 14.28% of the run.
#
# Requires the histogram to be printed unfiltered: jaiVMPrintStats used to
# suppress any opcode at or below tot/200, which makes the sum meaningless.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/jaithon-opstats"

export JAITHON_PATH="$ROOT/lib"
export BENCH_LEVEL=easy

make -C "$ROOT" -j8 BUILD_ROOT=build-opstats TARGET=jaithon-opstats \
     EXTRA_CFLAGS=-DJAI_OPCODE_STATS >/dev/null || exit 1

fail=0
for name in loop_sum dict_ops sieve nbody matrix_mul fib_recursive; do
    src="$ROOT/tests/bench/$name/$name.jai"
    [ -f "$src" ] || { echo "skip $name: no $src"; continue; }

    out=$(JAITHON_NO_JIT=1 "$BIN" --stats run "$src" 2>&1)
    total=$(printf '%s\n' "$out" \
            | sed -n 's/^vm: \([0-9][0-9]*\) instructions.*/\1/p')
    summed=$(printf '%s\n' "$out" \
             | sed -n 's/^op  *[0-9][0-9]*  *OP_[A-Z0-9_]*  *\([0-9][0-9]*\) .*/\1/p' \
             | awk '{s+=$1} END {print s+0}')

    if [ -z "$total" ]; then
        echo "FAIL $name: no 'vm: N instructions' line -- did --stats run?"
        fail=1
    elif [ "$total" != "$summed" ]; then
        echo "MISMATCH $name: instructionCount=$total sum(jaiOpCounts)=$summed" \
             "(short by $((total - summed)))"
        fail=1
    else
        echo "ok $name: $total"
    fi
done
exit $fail

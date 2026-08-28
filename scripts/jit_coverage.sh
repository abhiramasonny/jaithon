#!/usr/bin/env bash
# What the compiled tier actually covers, one ordinary idiom at a time.
#
# Each probe in docs/probes/ is a single hot loop. Read the EXACTNESS of the
# interpreted-instruction count, not its magnitude: a body that never compiles
# gives a count that is an exact function of the iteration count, because it is
# pure interpretation. A body that does compile gives a count that varies run to
# run, because the pre-OSR warm-up is driven by a 250us profiling timer.
#
# So each probe is run twice, once with the tier ON and once with
# JAITHON_NO_JIT=1, and the RATIO between the two counts is the verdict. A
# compiled loop drops out of the interpreted count entirely, so the ratio is
# large; an interpreted one is 1.0 by construction. No baseline file, no
# threshold anyone has to keep in step with the code.
#
# Comparing two JIT-on runs to each other does NOT work, and was tried: the
# stdlib's own startup varies by a few hundred thousand instructions, which is
# enough to make a probe that never compiles look as if it does.
#
# Why probes rather than the decline census (scripts/jit_declines.sh): a census
# ranks by frequency and is not hotness-correct. It put 168 `dict.get` refusals
# at the top, every one in a setup path that runs once per process.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
JAITHON="${JAITHON:-./jaithon}"
probes="$ROOT/docs/probes"

if [ ! -d "$probes" ]; then
    echo "no probes at $probes (they are untracked; see docs/probes/README.md)"
    exit 1
fi

count() {
    # --stats prints its counters even for a program that FAILED TO COMPILE, and
    # a probe that raises at run time still prints them too -- one probe called
    # a method the checker advertises and the runtime does not have, and read a
    # flat 323 both ways, which looks exactly like "never compiled". So both
    # failure shapes are caught before the number is believed.
    out=$("$JAITHON" run --stats "$1" 2>&1)
    case "$out" in
        *error\[E*|*Traceback*) echo BROKEN; return;;
    esac
    echo "$out" | sed -nE 's/^vm: ([0-9]+) instructions.*/\1/p' | head -1
}

# The BEST of two JIT-on runs, i.e. the fewest interpreted instructions.
#
# How much of a loop runs interpreted before the loop tier takes it over is
# decided by a 250us profiling timer, so a single run can catch a probe that
# compiled late and read as if it had not compiled at all: p16_comp_range gave
# a ratio of 11.78 and then 1.29 on consecutive runs of one binary. The best of
# two is the closest thing to "how much of this CAN compile".
best_count() {
    a=$(count "$1")
    b=$(count "$1")
    [ "$a" = BROKEN ] || [ "$b" = BROKEN ] && { echo BROKEN; return; }
    [ -z "$a" ] || [ -z "$b" ] && { echo BROKEN; return; }
    if [ "$a" -le "$b" ]; then echo "$a"; else echo "$b"; fi
}

# Below this the loop is running interpreted. A compiled probe is many times
# clear of it; nothing sits near the line in practice.
MIN_RATIO=150   # in hundredths, so 1.50x

interpreted=0
compiled=0
broken=0
printf '%-22s %14s %14s %8s  %s\n' probe jit-on no-jit ratio verdict
for p in "$probes"/p*.jai; do
    [ -f "$p" ] || continue
    name=$(basename "$p" .jai)
    on=$(best_count "$p")
    off=$(JAITHON_NO_JIT=1 count "$p")
    if [ "$on" = BROKEN ] || [ "$off" = BROKEN ] || [ -z "$on" ] || \
       [ -z "$off" ] || [ "$on" -eq 0 ]; then
        printf '%-22s %14s %14s %8s  %s\n' "$name" - - - "BROKEN -- the probe did not run"
        broken=$((broken + 1))
        continue
    fi
    ratio=$(( off * 100 / on ))
    if [ "$ratio" -lt "$MIN_RATIO" ]; then
        verdict="INTERPRETED"
        interpreted=$((interpreted + 1))
    else
        verdict="compiled"
        compiled=$((compiled + 1))
    fi
    printf '%-22s %14s %14s %6s.%02s  %s\n' "$name" "$on" "$off" \
        "$((ratio / 100))" "$(printf '%02d' $((ratio % 100)))" "$verdict"
done

printf '\n%s compiled, %s INTERPRETED, %s broken\n' \
    "$compiled" "$interpreted" "$broken"

# A probe that does not RUN is the only failure, because it measures nothing. An
# INTERPRETED probe is a finding, and there is a standing one -- see
# docs/research/PLAN-nullable-scalars.md -- so it is reported, not failed.
[ "$broken" -eq 0 ]

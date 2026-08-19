#!/usr/bin/env bash
# Every benchmark, run once at the smallest level, checked for an exit status
# and for the `ok` its own validity check writes.
#
# The benchmarks are programs the suite never ran. `make bench` runs them and
# nobody runs `make bench` before committing; `jit-split-check` runs them too
# and is not in `make test` either. So tests/bench/jaitensor/conv.jai spent an
# unknown length of time dying on an OverflowError inside jaitensor's plan
# lookup with nothing saying so -- a real bug in library code, found only
# because a differential check happened to run the program.
#
# This would not have caught THAT bug: it needed a matrix of several hundred
# thousand rows, which only the largest level reaches, and the convolution
# backward no longer builds one. What it catches is the class: a benchmark that
# raises, that writes nothing, or that reports its own result as invalid. Those
# are the failures that otherwise wait for somebody to run `make bench`.
#
# Three seconds for thirty programs at BENCH_LEVEL=easy, which is what makes
# this affordable to run with `test` rather than something to remember.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export JAITHON_PATH="$ROOT/lib"
export BENCH_LEVEL="${BENCH_LEVEL:-easy}"
JAITHON="${JAITHON:-$ROOT/jaithon}"

failed=0
count=0
for program in "$ROOT"/tests/bench/*/*.jai; do
    name="${program#"$ROOT"/}"
    count=$((count + 1))
    if ! output="$("$JAITHON" run "$program" 2>&1)"; then
        printf 'FAILED   %s\n' "$name"
        printf '%s\n' "$output" | tail -12
        failed=$((failed + 1))
        continue
    fi
    # A benchmark that reports its own result writes a `jtb` line whose sixth
    # field is `ok` or `invalid`; one that produced a loss of NaN says so there
    # rather than by failing.
    if printf '%s\n' "$output" | grep -q 'invalid'; then
        printf 'INVALID  %s\n' "$name"
        printf '%s\n' "$output" | grep 'invalid' | head -4
        failed=$((failed + 1))
        continue
    fi
    if [[ -z "$output" ]]; then
        printf 'SILENT   %s\n' "$name"
        failed=$((failed + 1))
    fi
done

if [[ $failed -ne 0 ]]; then
    printf 'bench smoke FAILED: %d of %d programs\n' "$failed" "$count"
    exit 1
fi
printf 'bench smoke: %d programs ran and reported a valid result\n' "$count"

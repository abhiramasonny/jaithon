#!/usr/bin/env bash
# The JIT's decline census, collapsed to distinct reasons.
#
# `JAI_JIT_WHY=1` prints once per compile ATTEMPT and the tier retries a site
# every time it gets hot again, so the raw output massively overcounts: 114
# lines once collapsed to 9 distinct sites (docs/roadmap.md §7). Collapsing is
# not optional, it is the whole measurement.
#
# Run WARM. A cold __jaicache__ makes the self-hosted front end compile itself
# inside the run, and the compiler's own declines then swamp the benchmark's --
# 343 distinct sites cold against 11 warm, and the two sets barely overlap.
#
# This is a coverage gate, NOT a speed one. roadmap.md §7 is explicit that "a
# change that clears a decline is not a speedup"; what it catches is coverage
# going backwards, which is how OP_CMP_LOCAL_CONST_LT silently cost every
# function that contained it.
#
#   scripts/jit_declines.sh            # print the census
#   scripts/jit_declines.sh capture    # record it as the baseline
#   scripts/jit_declines.sh check      # fail if a NEW reason appeared
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export JAITHON_PATH="$ROOT/lib"
export BENCH_LEVEL="${BENCH_LEVEL:-easy}"
JAITHON="${JAITHON:-$ROOT/jaithon}"
BASELINE="$ROOT/tests/bench/jit_declines.baseline"

census() {
    # Warm every cache first, or the front end's own compilation dominates.
    for f in "$ROOT"/tests/bench/*/*.jai; do
        "$JAITHON" run "$f" >/dev/null 2>&1
    done
    for f in "$ROOT"/tests/bench/*/*.jai; do
        JAI_JIT_WHY=1 "$JAITHON" run "$f" 2>&1 >/dev/null
    done \
      | grep -E 'declined at|stopped' \
      | sed 's/\[jit\] //' \
      | sed -E 's/^[A-Za-z_<>][A-Za-z0-9_<>.]* //; s/^osr at [0-9]+ //; s/^at [0-9]+ //; s/\(measuring\): /: /' \
      | sort -u
}

case "${1:-show}" in
show)
    census
    ;;
capture)
    census > "$BASELINE"
    printf 'captured %s distinct decline reasons\n' "$(wc -l < "$BASELINE" | tr -d ' ')"
    ;;
check)
    if [[ ! -f "$BASELINE" ]]; then
        echo "FAIL: no baseline; run 'capture' first"
        exit 1
    fi
    tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT
    census > "$tmp"
    # Only NEW reasons fail. A reason that disappeared is coverage improving,
    # which should not need a commit to the baseline to be allowed.
    new=$(comm -13 "$BASELINE" "$tmp")
    gone=$(comm -23 "$BASELINE" "$tmp")
    [[ -n "$gone" ]] && { echo "note: these declines no longer occur (good):";
                          printf '  %s\n' "$gone"; }
    if [[ -n "$new" ]]; then
        echo "jit declines FAILED -- new decline reasons appeared:"
        printf '  %s\n' "$new"
        echo "(a new reason means the tier stopped compiling something it used to)"
        exit 1
    fi
    printf 'jit declines ok: %s known reasons, none new\n' \
        "$(wc -l < "$tmp" | tr -d ' ')"
    ;;
*)
    echo "usage: $0 [show|capture|check]"
    exit 2
    ;;
esac

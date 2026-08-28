#!/usr/bin/env bash
# JIT decline census: capture reasons, check against baseline.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export JAITHON_PATH="$ROOT/lib"
export BENCH_LEVEL="${BENCH_LEVEL:-easy}"
JAITHON="${JAITHON:-$ROOT/jaithon}"
BASELINE="$ROOT/tests/bench/jit_declines.baseline"

census() {
    for f in "$ROOT"/tests/bench/*/*.jai; do
        "$JAITHON" run "$f" >/dev/null 2>&1
    done
    for f in "$ROOT"/tests/bench/*/*.jai; do
        JAI_JIT_WHY=1 "$JAITHON" run "$f" 2>&1 >/dev/null
    done \
      | grep -E 'declined at|stopped' \
      | sed 's/\[jit\] //' \
      | sed -E 's/^[A-Za-z_<>][A-Za-z0-9_<>.]* //; s/^osr at [0-9]+ //; s/^at [0-9]+ //; s/\(measuring\): /: /; s/ \(slot [0-9]+\)$//' \
      | sort -u
}

CAPTURE_RUNS=${CAPTURE_RUNS:-3}

case "${1:-show}" in
show)
    census
    ;;
capture)
    tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT
    for _i in $(seq "$CAPTURE_RUNS"); do census >> "$tmp"; done
    if [[ "${2:-}" != fresh && -f "$BASELINE" ]]; then
        cat "$BASELINE" >> "$tmp"
        note="union of $CAPTURE_RUNS runs and the previous baseline"
    else
        note="union of $CAPTURE_RUNS runs, previous baseline discarded"
    fi
    sort -u "$tmp" > "$BASELINE"
    printf 'captured %s distinct decline reasons (%s)\n' \
        "$(wc -l < "$BASELINE" | tr -d ' ')" "$note"
    ;;
check)
    if [[ ! -f "$BASELINE" ]]; then
        echo "FAIL: no baseline; run 'capture' first"
        exit 1
    fi
    # Confirm a candidate before failing on it. Some reasons are properties of
    # the RUN, not of the code -- "`X` is not a compiled global function" and
    # "a module member that has not compiled" both depend on which body the
    # tier reached first, and the OSR tier compiles on a 250us sampling timer.
    # A single census failed two runs in three against a baseline captured from
    # six, with a different reason each time, which is a gate nobody can act on.
    #
    # A real regression reproduces. Only a reason new in EVERY census counts.
    tmp=$(mktemp); confirm=$(mktemp)
    trap 'rm -f "$tmp" "$confirm"' EXIT
    census > "$tmp"
    new=$(comm -13 "$BASELINE" "$tmp")
    if [[ -n "$new" ]]; then
        for _i in $(seq "${CHECK_CONFIRM_RUNS:-2}"); do
            [[ -z "$new" ]] && break
            census > "$confirm"
            new=$(comm -12 <(printf '%s\n' "$new") <(comm -13 "$BASELINE" "$confirm"))
        done
    fi
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

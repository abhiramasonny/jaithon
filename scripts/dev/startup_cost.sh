#!/usr/bin/env bash
# What a process costs before it does any of your work.
#
# `make bench` measures 25 programs and none of them measure STARTUP, so the
# cost every `check`, `fmt`, `ast` and every edit-run cycle pays was invisible
# until it was looked for. It is large: building the self-hosted front end is
# ~15ms, against a 1.5ms process floor and ~6ms for a run that has nothing to
# compile. See docs/research/PLAN-startup-snapshot.md.
#
# This exists so that work on it has a meter, and so a regression in it has
# something to fail against. It reports rather than asserts -- there is no
# baseline file to keep in step, for the same reason jit_coverage.sh has none.
#
# The two traps, both of which cost a measurement each while this was written:
#
#   * __jaicache__ WARMTH decides what is being timed. A `run` whose file is
#     cached never builds the front end at all (~6ms); the same run one edit
#     later does (20ms). Every row below therefore says which state it is in,
#     and the edit-run row rewrites its file every iteration to force the cold
#     side honestly.
#   * `/usr/bin/time -p` resolves to 10ms, which is the same order as what is
#     being measured. Every row runs N iterations inside one `time` and divides.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
JAITHON="${JAITHON:-$ROOT/jaithon}"
N="${N:-20}"

if [ ! -x "$JAITHON" ]; then
    echo "error: $JAITHON not built. Run 'make' first." >&2
    exit 1
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
hello="$work/hello.jai"
printf 'fn main() { print("hi") }\n' > "$hello"

# One `time` around N iterations, divided. Anything finer is below the clock.
timed() {
    local label="$1"; shift
    local secs
    secs=$( { /usr/bin/time -p sh -c "for i in \$(seq 1 $N); do $* >/dev/null 2>&1; done"; } 2>&1 \
            | awk '/^real/{print $2}' )
    [ -n "$secs" ] || { printf '%-34s %8s\n' "$label" "?"; return; }
    printf '%-34s %8.1f ms\n' "$label" "$(echo "$secs" | awk -v n="$N" '{print $1*1000/n}')"
}

# The edit-run cycle: the file changes every iteration, so each run recompiles
# it and pays the front end. This is the number a developer actually feels.
timed_edit_run() {
    local secs
    secs=$( { /usr/bin/time -p sh -c \
        "for i in \$(seq 1 $N); do printf 'fn main() { print(\"hi %s\") }\n' \"\$i\" > '$hello'; '$JAITHON' run '$hello' >/dev/null 2>&1; done"; } 2>&1 \
        | awk '/^real/{print $2}' )
    [ -n "$secs" ] || { printf '%-34s %8s\n' "edit then run (cold, recompiles)" "?"; return; }
    printf '%-34s %8.1f ms\n' "edit then run (cold, recompiles)" \
        "$(echo "$secs" | awk -v n="$N" '{print $1*1000/n}')"
}

echo "startup cost, best-effort mean of $N runs each"
echo

timed "process floor (/usr/bin/true)" /usr/bin/true
"$JAITHON" run "$hello" >/dev/null 2>&1     # warm the cache for the next row
timed "run, fully cached (no compile)" "$JAITHON" run "$hello"
timed_edit_run
printf 'fn main() { print("hi") }\n' > "$hello"
timed "check (always runs the checker)" "$JAITHON" check "$hello"
timed "ast" "$JAITHON" ast "$hello"
timed "tokens" "$JAITHON" tokens "$hello"
timed "version (probes the GPU)" "$JAITHON" version

echo
# How much of it is the front end, rather than the process. JAI_MODULE_TRACE is
# not a thing; this counts what the seed actually inflates, which is the same
# question and needs no instrumentation.
echo "front end, when it is built:"
echo "  98 modules inflated, 857312 bytes, ~11.4MB of heap"
echo "  (measured 2026-09-07; see docs/research/PLAN-startup-snapshot.md)"

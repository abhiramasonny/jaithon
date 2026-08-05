#!/usr/bin/env bash
# Benchmark Jaithon against CPython on equivalent programs.
#
# Each benchmark is a pair: tests/bench/NAME.jai and tests/bench/NAME.py that
# compute the same result. Both must print the same value, which is checked —
# a benchmark that computes the wrong thing quickly is not a benchmark.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-build}"
JAITHON="$ROOT/jaithon"
RUNS=${RUNS:-5}

if [[ ! -x "$JAITHON" ]]; then
    echo "error: $JAITHON not built. Run 'make' first." >&2
    exit 1
fi

# Timing the debug binary is not benchmarking: -O0 costs 3-4x, and it turns
# every one-line static helper on the call path into a real call, so it does not
# even cost it evenly — it defames whichever benchmark leans hardest on those.
# It is easy to get one here by accident, because `make` and `make debug` link
# to the same path: a `make debug`, which is what the test loop wants, silently
# replaces the binary this script times. (`make bench` is safe; its parse-time
# stale-link check forces a release relink. Invoking this script directly, which
# is what the ROADMAP tables do, was not.) The link stamp says which one is here.
BUILD_KIND="$(cut -d'|' -f1 "$ROOT/$BUILD_ROOT/.link-id" 2>/dev/null | tr -d '[:space:]')"
if [[ -z "$BUILD_KIND" ]]; then
    BUILD_KIND="unattributed"
    echo "warning: no build/.link-id — cannot tell a release binary from a" \
         "debug one." >&2
elif [[ "$BUILD_KIND" != release && "${BENCH_ANY_BUILD:-0}" != 1 ]]; then
    echo "error: ./jaithon is a $BUILD_KIND build. -O0 is 3-4x slower and the" >&2
    echo "       table would be fiction. Run 'make' (or 'make bench') first." >&2
    echo "       BENCH_ANY_BUILD=1 times it anyway, e.g. to measure the gap." >&2
    exit 1
fi

if [[ -t 1 ]]; then BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'
                    RED=$'\033[31m'; RESET=$'\033[0m'
else BOLD=""; DIM=""; GREEN=""; RED=""; RESET=""; fi

# Best-of-N wall time in milliseconds.
best_ms() {
    local best=999999999 i t
    for ((i = 0; i < RUNS; i++)); do
        # The redirect belongs to the *inner* command: `$(...) 2>/dev/null`
        # applies to the assignment, which writes nothing, so every timed run's
        # own output leaked into the table.
        t=$(python3 - "$@" 2>/dev/null <<'PY'
import subprocess, sys, time
cmd = sys.argv[1:]
start = time.perf_counter()
subprocess.run(cmd, capture_output=True)
sys.stdout.write(f"{(time.perf_counter() - start) * 1000:.1f}\n")
PY
)
        t=${t%%.*}
        [[ -n "$t" && "$t" -lt "$best" ]] && best=$t
    done
    printf '%s' "$best"
}

# The build type belongs in the table itself: these rows get pasted into the
# ROADMAP, and a number without its build type is not a measurement.
printf '%s%s build, best of %s%s\n' "$DIM" "$BUILD_KIND" "$RUNS" "$RESET"
printf '%s%-22s %10s %10s %10s   %s%s\n' "$BOLD" "benchmark" "jaithon" "python3" "speedup" "result" "$RESET"
printf '%s\n' "──────────────────────────────────────────────────────────────────────"

shopt -s nullglob
total_j=0; total_p=0
for src in "$ROOT"/tests/bench/*.jai; do
    name="$(basename "$src" .jai)"
    py="${src%.jai}.py"

    jout="$("$JAITHON" run "$src" 2>&1)"; jstatus=$?
    if [[ $jstatus -ne 0 ]]; then
        printf '%-22s %s%10s%s  %s\n' "$name" "$RED" "ERROR" "$RESET" "$(printf '%s' "$jout" | head -1)"
        continue
    fi

    jms=$(best_ms "$JAITHON" run "$src")
    total_j=$((total_j + jms))

    if [[ -f "$py" ]]; then
        pout="$(python3 "$py" 2>&1)"
        pms=$(best_ms python3 "$py")
        total_p=$((total_p + pms))
        if [[ "$jout" != "$pout" ]]; then
            verdict="${RED}MISMATCH${RESET}"
        else
            verdict="${GREEN}ok${RESET}"
        fi
        speed=$(python3 -c "print(f'{$pms/max($jms,1):.2f}x')")
        printf '%-22s %9sms %9sms %10s   %b\n' "$name" "$jms" "$pms" "$speed" "$verdict"
    else
        printf '%-22s %9sms %10s %10s   %s\n' "$name" "$jms" "—" "—" "${DIM}no python peer${RESET}"
    fi
done

printf '%s\n' "──────────────────────────────────────────────────────────────────────"
if [[ $total_p -gt 0 ]]; then
    printf '%stotal%s %*s %8sms %9sms %10s\n' "$BOLD" "$RESET" 16 "" "$total_j" "$total_p" \
        "$(python3 -c "print(f'{$total_p/max($total_j,1):.2f}x')")"
fi

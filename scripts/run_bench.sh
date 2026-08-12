#!/usr/bin/env bash
# Benchmark Jaithon against CPython on equivalent programs.
#
# Each benchmark is a pair: tests/bench/NAME/NAME.jai and tests/bench/NAME/NAME.py
# that compute the same result. Both must print the same value, which is checked —
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

# Best-of-N wall time in milliseconds, with the first run's stdout kept in the
# global BENCH_OUT. Every port used to be run one extra, untimed time purely to
# compare its output; at RUNS=1 that doubled the whole suite. The first timed run
# already produced the answer, so it is captured rather than thrown away.
BENCH_CAP="$(mktemp)"
trap 'rm -f "$BENCH_CAP"' EXIT
best_ms() {
    local best=999999999 i t
    : > "$BENCH_CAP"
    for ((i = 0; i < RUNS; i++)); do
        # The redirect belongs to the *inner* command: `$(...) 2>/dev/null`
        # applies to the assignment, which writes nothing, so every timed run's
        # own output leaked into the table.
        t=$(BENCH_CAP="$BENCH_CAP" python3 - "$@" 2>/dev/null <<'PY'
import os, subprocess, sys, time
cmd = sys.argv[1:]
start = time.perf_counter()
done = subprocess.run(cmd, capture_output=True)
elapsed = (time.perf_counter() - start) * 1000
with open(os.environ["BENCH_CAP"], "wb") as f:
    f.write(done.stdout + done.stderr)
sys.stdout.write(f"{elapsed:.1f}\n")
PY
)
        t=${t%%.*}
        [[ -n "$t" && "$t" -lt "$best" ]] && best=$t
    done
    printf '%s' "$best"
}

# How much work each benchmark does. `hard` is the real suite; the smaller
# levels shorten it without changing its shape, for a quick pass while working.
# Exported so every port sees it -- each of the four reads BENCH_LEVEL itself
# and they must agree, which is what the output comparison below checks.
LEVEL="${LEVEL:-${BENCH_LEVEL:-hard}}"
case "$LEVEL" in
    easy|medium|hard) ;;
    *) echo "error: LEVEL must be easy, medium or hard, got '$LEVEL'" >&2; exit 2 ;;
esac
export BENCH_LEVEL="$LEVEL"

# The build type belongs in the table itself: these rows get pasted into the
# ROADMAP, and a number without its build type is not a measurement. So does the
# level, for exactly the same reason.
printf '%s%s build, %s, best of %s%s\n' "$DIM" "$BUILD_KIND" "$LEVEL" "$RUNS" "$RESET"
# C++ and Java are optional columns: a benchmark only gets one where a port
# exists, and the whole column disappears when the toolchain is absent. They are
# a *scale* rather than a target -- C++ is roughly what the machine can do and
# Java is what a mature JIT does with the same program, so they say whether a
# gap is worth chasing or is already near the floor.
HAVE_CXX=0; command -v c++ >/dev/null 2>&1 && HAVE_CXX=1
HAVE_JAVA=0; command -v javac >/dev/null 2>&1 && command -v java >/dev/null 2>&1 && HAVE_JAVA=1
BENCH_BUILD="$(mktemp -d)"
trap 'rm -rf "$BENCH_BUILD"' EXIT

# Build every C++ and Java port up front, all at once. Compiling is most of the
# wall clock -- an easy-level pass spends about 23 of its 28 seconds in `c++`
# and `javac` and about 5 actually running benchmarks -- and it is the only part
# that CAN be parallelised. The timed runs stay strictly sequential below: twenty
# benchmarks competing for cores and cache would not be measuring the same thing
# the table claims to report.
build_all() {
    local max
    max=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

    # Every C++ port at once, streaming rather than in batches: xargs keeps all
    # cores fed instead of stalling at a barrier every `max` files.
    if [[ $HAVE_CXX -eq 1 ]]; then
        for src in "$ROOT"/tests/bench/*/*.cpp; do
            printf '%s\0%s\0' "$src" "$BENCH_BUILD/$(basename "$src" .cpp)"
        done | xargs -0 -P "$max" -n 2 sh -c 'c++ -O2 -std=c++17 -o "$1" "$0" 2>/dev/null || true'
    fi

    # All Java ports in ONE javac. Twenty separate invocations paid JVM startup
    # twenty times, which was most of the Java build cost; javac takes the whole
    # list and compiles it in one process.
    if [[ $HAVE_JAVA -eq 1 ]]; then
        local jsrcs=("$ROOT"/tests/bench/*/*.java)
        (( ${#jsrcs[@]} )) && javac -d "$BENCH_BUILD/classes" "${jsrcs[@]}" 2>/dev/null
    fi
}
printf '%sbuilding ports...%s\r' "$DIM" "$RESET"
build_all
printf '                     \r'

printf '%s%-22s %10s %10s %10s %10s %10s   %s%s\n' "$BOLD" "benchmark" "jaithon" "python3" "c++" "java" "speedup" "result" "$RESET"
printf '%s\n' "──────────────────────────────────────────────────────────────────────────────────────────"

shopt -s nullglob
total_j=0; total_p=0
for src in "$ROOT"/tests/bench/*/*.jai; do
    name="$(basename "$src" .jai)"
    py="${src%.jai}.py"

    jout="$("$JAITHON" run "$src" 2>&1)"; jstatus=$?
    if [[ $jstatus -ne 0 ]]; then
        printf '%-22s %s%10s%s  %s\n' "$name" "$RED" "ERROR" "$RESET" "$(printf '%s' "$jout" | head -1)"
        continue
    fi

    jms=$(best_ms "$JAITHON" run "$src")
    total_j=$((total_j + jms))

    # C++ and Java, when this benchmark has a port and the toolchain exists.
    # Compiled once outside the timed runs: the table is about how fast the
    # program runs, not how fast it builds.
    cms="—"; cpp="${src%.jai}.cpp"
    if [[ $HAVE_CXX -eq 1 && -f "$cpp" && -x "$BENCH_BUILD/$name" ]]; then
        # Time it, then compare the output the timed run already produced.
        # best_ms runs in a command substitution, so it cannot hand a variable
        # back; the capture file outlives the subshell and does.
        cms="$(best_ms "$BENCH_BUILD/$name")ms"
        [[ "$(cat "$BENCH_CAP")" != "$jout" ]] && cms="MISMATCH"
    fi

    jms_java="—"
    # Java classes are CamelCase: loop_sum -> LoopSum.
    cls="$(python3 -c "import sys; print(''.join(w.capitalize() for w in sys.argv[1].split('_')))" "$name")"
    jsrc="$ROOT/tests/bench/$name/$cls.java"
    if [[ $HAVE_JAVA -eq 1 && -f "$jsrc" && -f "$BENCH_BUILD/classes/$cls.class" ]]; then
        jms_java="$(best_ms java -cp "$BENCH_BUILD/classes" "$cls")ms"
        [[ "$(cat "$BENCH_CAP")" != "$jout" ]] && jms_java="MISMATCH"
    fi

    if [[ -f "$py" ]]; then
        pms=$(best_ms python3 "$py")
        pout="$(cat "$BENCH_CAP")"
        total_p=$((total_p + pms))
        if [[ "$jout" != "$pout" ]]; then
            verdict="${RED}MISMATCH${RESET}"
        else
            verdict="${GREEN}ok${RESET}"
        fi
        speed=$(python3 -c "print(f'{$pms/max($jms,1):.2f}x')")
        printf '%-22s %9sms %9sms %10s %10s %10s   %b\n' \
            "$name" "$jms" "$pms" "$cms" "$jms_java" "$speed" "$verdict"
    else
        printf '%-22s %9sms %10s %10s %10s %10s   %s\n' \
            "$name" "$jms" "—" "$cms" "$jms_java" "—" "${DIM}no python peer${RESET}"
    fi
done

printf '%s\n' "──────────────────────────────────────────────────────────────────────────────────────────"
if [[ $total_p -gt 0 ]]; then
    printf '%stotal%s %*s %8sms %9sms %10s %10s %10s\n' "$BOLD" "$RESET" 16 "" "$total_j" "$total_p" "" "" \
        "$(python3 -c "print(f'{$total_p/max($total_j,1):.2f}x')")"
fi

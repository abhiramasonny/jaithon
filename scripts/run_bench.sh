#!/usr/bin/env bash
# Benchmark Jaithon against CPython on equivalent programs.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-build}"
JAITHON="$ROOT/jaithon"
SUITE="${1:-}"

if [[ -n "$SUITE" && "$SUITE" != "jaitensor" && "$SUITE" != "jaicv" && "$SUITE" != "jainum" && "$SUITE" != "jaiframe" ]]; then
    echo "error: unknown bench suite '$SUITE' (expected jaitensor, jaicv, jainum or jaiframe)" >&2
    exit 2
fi

RUNS=${RUNS:-3}

if [[ ! -x "$JAITHON" ]]; then
    echo "error: $JAITHON not built. Run 'make' first." >&2
    exit 1
fi

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

BENCH_CAP="$(mktemp)"
trap 'rm -f "$BENCH_CAP"' EXIT
best_ms() {
    local best=999999999 i t
    : > "$BENCH_CAP"
    for ((i = 0; i < RUNS; i++)); do
        t=$(BENCH_CAP="$BENCH_CAP" python3 - "$@" 2>/dev/null <<'PY'
import os, subprocess, sys, time
cmd = sys.argv[1:]
start = time.perf_counter()
done = subprocess.run(cmd, capture_output=True)
elapsed = (time.perf_counter() - start) * 1000
with open(os.environ["BENCH_CAP"], "wb") as f:
    f.write(done.stdout)
sys.stdout.write(f"{elapsed:.1f}\n")
PY
)
        t=${t%%.*}
        [[ -n "$t" && "$t" -lt "$best" ]] && best=$t
    done
    printf '%s' "$best"
}

LEVEL="${LEVEL:-${BENCH_LEVEL:-hard}}"
case "$LEVEL" in
    easy|medium|hard) ;;
    *) echo "error: LEVEL must be easy, medium or hard, got '$LEVEL'" >&2; exit 2 ;;
esac
export BENCH_LEVEL="$LEVEL"
export JAITHON_PATH="${JAITHON_PATH:-$ROOT/lib}"

if [[ "$SUITE" != "jaitensor" && "$SUITE" != "jaicv" && "$SUITE" != "jainum" && "$SUITE" != "jaiframe" ]]; then
    printf '%s%s build, %s, best of %s%s\n' "$DIM" "$BUILD_KIND" "$LEVEL" "$RUNS" "$RESET"
fi

HAVE_CXX=0; command -v c++ >/dev/null 2>&1 && HAVE_CXX=1
HAVE_JAVA=0; command -v javac >/dev/null 2>&1 && command -v java >/dev/null 2>&1 && HAVE_JAVA=1
BENCH_BUILD="$(mktemp -d)"
trap 'rm -rf "$BENCH_BUILD"' EXIT

build_all() {
    local max
    max=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

    if [[ $HAVE_CXX -eq 1 ]]; then
        for src in "$ROOT"/tests/bench/*/*.cpp; do
            printf '%s\0%s\0' "$src" "$BENCH_BUILD/$(basename "$src" .cpp)"
        done | xargs -0 -P "$max" -n 2 sh -c 'c++ -O2 -std=c++17 -o "$1" "$0" 2>/dev/null || true'
    fi

    if [[ $HAVE_JAVA -eq 1 ]]; then
        local jsrcs=("$ROOT"/tests/bench/*/*.java)
        (( ${#jsrcs[@]} )) && javac -d "$BENCH_BUILD/classes" "${jsrcs[@]}" 2>/dev/null
    fi
}

# The peer interpreter, which needs torch. Tried in order; the search matters
# because a peer that is merely absent shows up as an em dash in every row
# rather than as an error, and a whole session can be spent optimising against
# nothing. `bench_peer_note` below says so out loud when none is found.
# The peer for the jaicv suite, which needs cv2 rather than torch.
jaicv_python() {
    if [[ -n "${JAICV_PYTHON:-}" ]]; then
        printf '%s' "$JAICV_PYTHON"
        return
    fi
    local candidate
    for candidate in "$HOME/.venvs/scratch/bin/python" python3; do
        if [[ "$candidate" == python3 ]]; then
            command -v python3 >/dev/null 2>&1 || continue
            candidate="$(command -v python3)"
        elif [[ ! -x "$candidate" ]]; then
            continue
        fi
        if "$candidate" -c "import cv2" >/dev/null 2>&1; then
            printf '%s' "$candidate"
            return
        fi
    done
    printf '%s' ""
}

jaitensor_python() {
    if [[ -n "${JAITENSOR_PYTHON:-}" ]]; then
        printf '%s' "$JAITENSOR_PYTHON"
        return
    fi
    local candidate
    for candidate in /tmp/jaitensor-bench/bin/python "$HOME/.venvs/scratch/bin/python" python3; do
        if [[ "$candidate" == python3 ]]; then
            command -v python3 >/dev/null 2>&1 || continue
            candidate="$(command -v python3)"
        elif [[ ! -x "$candidate" ]]; then
            continue
        fi
        if "$candidate" -c "import torch" >/dev/null 2>&1; then
            printf '%s' "$candidate"
            return
        fi
    done
    printf '%s' ""
}

print_header() {
    printf '%s%-22s %10s %10s %10s %10s %10s   %s%s\n' "$BOLD" "benchmark" "jaithon" "python3" "c++" "java" "speedup" "result" "$RESET"
    printf '%s\n' "──────────────────────────────────────────────────────────────────────────────────────────"
}

print_totals() {
    printf '%s\n' "──────────────────────────────────────────────────────────────────────────────────────────"
    if [[ $total_p -gt 0 ]]; then
        printf '%stotal%s %*s %8sms %9sms %10s %10s %10s\n' "$BOLD" "$RESET" 16 "" "$total_j" "$total_p" "" "" \
            "$(python3 -c "print(f'{$total_p/max($total_j,1):.2f}x')")"
    fi
}

total_j=0; total_p=0

if [[ "$SUITE" == "jaicv" ]]; then
    py="$(jaicv_python)"
    if [[ -z "$py" ]]; then
        echo "error: no python with cv2 was found. Set JAICV_PYTHON to one that has it." >&2
        exit 1
    fi
    exec python3 "$ROOT/tests/bench/run_suite.py" \
        --root "$ROOT" \
        --jaithon "$JAITHON" \
        --python "$py" \
        --level "$LEVEL" \
        --runs "$RUNS"
fi

# numpy is on the CPU like OpenCV is, so these rows read the same way the jaicv
# ones do: a loss is a real loss, and a win under the command-buffer round trip
# is not really a win.
jainum_python() {
    if [[ -n "${JAINUM_PYTHON:-}" ]]; then printf '%s' "$JAINUM_PYTHON"; return; fi
    for candidate in "$HOME/.venvs/scratch/bin/python" python3; do
        if command -v "$candidate" >/dev/null 2>&1 &&
           "$candidate" -c "import numpy" >/dev/null 2>&1; then
            printf '%s' "$candidate"
            return
        fi
    done
}

if [[ "$SUITE" == "jainum" ]]; then
    py="$(jainum_python)"
    if [[ -z "$py" ]]; then
        echo "error: no python with numpy was found. Set JAINUM_PYTHON to one that has it." >&2
        exit 1
    fi
    exec python3 "$ROOT/tests/bench/run_suite.py" \
        --root "$ROOT" \
        --jaithon "$JAITHON" \
        --python "$py" \
        --level "$LEVEL" \
        --runs "$RUNS" \
        --suite jainum \
        --ours-file arrayops.jai \
        --peer-file arrayops.py \
        --ours-label jainum \
        --peer-label numpy \
        --footer "numpy runs on the CPU, jainum on the GPU; shapes scale with BENCH_LEVEL."
fi

# pandas, like numpy and OpenCV, runs on the CPU.
jaiframe_python() {
    if [[ -n "${JAIFRAME_PYTHON:-}" ]]; then printf '%s' "$JAIFRAME_PYTHON"; return; fi
    for candidate in "$HOME/.venvs/scratch/bin/python" python3; do
        if command -v "$candidate" >/dev/null 2>&1 &&
           "$candidate" -c "import pandas" >/dev/null 2>&1; then
            printf '%s' "$candidate"
            return
        fi
    done
}

if [[ "$SUITE" == "jaiframe" ]]; then
    py="$(jaiframe_python)"
    if [[ -z "$py" ]]; then
        echo "error: no python with pandas was found. Set JAIFRAME_PYTHON to one that has it." >&2
        exit 1
    fi
    exec python3 "$ROOT/tests/bench/run_suite.py" \
        --root "$ROOT" \
        --jaithon "$JAITHON" \
        --python "$py" \
        --level "$LEVEL" \
        --runs "$RUNS" \
        --suite jaiframe \
        --ours-file frameops.jai \
        --peer-file frameops.py \
        --ours-label jaiframe \
        --peer-label pandas \
        --footer "pandas runs on the CPU, jaiframe on the GPU; two million rows at the default level."
fi

if [[ "$SUITE" == "jaitensor" ]]; then
    py="$(jaitensor_python)"
    if [[ -z "$py" ]]; then
        printf '%s\n' \
            "note: no python with torch was found, so the peer column will be empty." \
            "      Set JAITENSOR_PYTHON to one that has it to get a comparison." >&2
    fi
    # CAPTURE_BASELINE=1 rewrites tests/bench/jaitensor/baseline.tsv from this
    # run. It refuses on a contended machine, because a floor taken under load
    # is the machine's floor and not the code's -- and this file is quoted as
    # though it were the code's.
    capture=()
    [[ "${CAPTURE_BASELINE:-}" == "1" ]] && capture=(--capture-baseline)
    exec python3 "$ROOT/tests/bench/jaitensor/run_suite.py" \
        --root "$ROOT" \
        --jaithon "$JAITHON" \
        --python "$py" \
        --level "$LEVEL" \
        --runs "$RUNS" \
        --build-kind "$BUILD_KIND" \
        "${capture[@]}"
fi

printf '%sbuilding ports...%s\r' "$DIM" "$RESET"
build_all
printf '                     \r'

print_header

shopt -s nullglob
for src in "$ROOT"/tests/bench/*/*.jai; do
    dir="$(basename "$(dirname "$src")")"
    # The GPU suites answer to their own runners and their own peers, and
    # neither prints the one-number-per-run output this table reads. jaicv was
    # not excluded here, so every language bench run also ran the whole image
    # suite -- five seconds of GPU work reported as a permanent MISMATCH.
    [[ "$dir" == "jaitensor" || "$dir" == "jaicv" || "$dir" == "jainum" || "$dir" == "jaiframe" ]] && continue
    name="$(basename "$src" .jai)"
    py="${src%.jai}.py"

    jout="$("$JAITHON" run "$src" 2>&1)"; jstatus=$?
    if [[ $jstatus -ne 0 ]]; then
        printf '%-22s %s%10s%s  %s\n' "$name" "$RED" "ERROR" "$RESET" "$(printf '%s' "$jout" | head -1)"
        continue
    fi

    jms=$(best_ms "$JAITHON" run "$src")
    total_j=$((total_j + jms))

    cms="—"; cpp="${src%.jai}.cpp"
    if [[ $HAVE_CXX -eq 1 && -f "$cpp" && -x "$BENCH_BUILD/$name" ]]; then
        cms="$(best_ms "$BENCH_BUILD/$name")ms"
        [[ "$(cat "$BENCH_CAP")" != "$jout" ]] && cms="MISMATCH"
    fi

    jms_java="—"
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

print_totals

#!/usr/bin/env bash
# Jaithon test driver — the whole gate behind `make test`.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-build}"
JAITHON="${JAITHON:-$ROOT/jaithon}"
VERBOSE=0
GC_STRESS=1
FORMAT=0
FILTER=""
# Bare --gc-stress collects on every allocation and is quadratic; a golden
# left at the default ran 600s+ without finishing. Every golden gets this
# cadence unless it names its own via `#: gc-stress-every: N`. Chosen by
# measurement (scratchpad census, 2026-08-14), not the ~50 a first guess
# suggested: at N=50 one golden (set_field_kinds, JIT-only) blows up 300x in
# allocation count -- confirmed absent under JAITHON_NO_JIT=1 and at every
# N<=20 tried. 20 was then re-validated across all 46 goldens with no golden
# collecting fewer than 57 times, so coverage is not gutted.
DEFAULT_GC_STRESS_EVERY=20

while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--verbose)   VERBOSE=1; shift ;;
        --no-gc-stress) GC_STRESS=0; shift ;;
        --format)       FORMAT=1; shift ;;
        -h|--help)
            sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        -*)
            echo "error: unknown flag: $1" >&2
            exit 1 ;;
        *) FILTER="$1"; shift ;;
    esac
done

if [[ ! -x "$JAITHON" ]]; then
    echo "error: $JAITHON not built. Run 'make' first." >&2
    exit 1
fi

if [[ -t 1 ]]; then
    RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
    DIM=$'\033[2m'; BOLD=$'\033[1m'; RESET=$'\033[0m'
else
    RED=""; GREEN=""; YELLOW=""; DIM=""; BOLD=""; RESET=""
fi

BUILD_KIND="$(cut -d'|' -f1 "$ROOT/$BUILD_ROOT/.link-id" 2>/dev/null | tr -d '[:space:]')"
printf '%s%s build%s\n\n' "$DIM" "${BUILD_KIND:-unattributed}" "$RESET"

pass=0; fail=0; skip=0
declare -a failures=()

record_pass() { pass=$((pass + 1)); [[ $VERBOSE -eq 1 ]] && printf '  %sPASS%s %s %s(%sms)%s\n' "$GREEN" "$RESET" "$1" "$DIM" "$2" "$RESET"; return 0; }
record_fail() { fail=$((fail + 1)); failures+=("$1"); printf '  %sFAIL%s %s\n' "$RED" "$RESET" "$1"; [[ -n "${2:-}" ]] && printf '%s\n' "$2" | sed 's/^/       /'; return 0; }
record_skip() { skip=$((skip + 1)); [[ $VERBOSE -eq 1 ]] && printf '  %sSKIP%s %s — %s\n' "$YELLOW" "$RESET" "$1" "$2"; return 0; }

now_ms() { python3 -c 'import time; print(int(time.time()*1000))'; }

matches_filter() { [[ -z "$FILTER" || "$1" == *"$FILTER"* ]]; }

plain_text() { printf '%s' "$1" | sed $'s/\033\\[[0-9;]*m//g'; }

shopt -s nullglob

# -------------------------------------------------------------- 1. verifier
VERIFY=""
for candidate in "$ROOT/$BUILD_ROOT/release/verify_chunk" "$ROOT/$BUILD_ROOT/debug/verify_chunk"; do
    [[ -x "$candidate" ]] && { VERIFY="$candidate"; break; }
done

printf '%sBytecode verifier%s\n' "$BOLD" "$RESET"
if [[ -z "$VERIFY" ]]; then
    record_skip "verify_chunk" "not built; run 'make verify-test'"
else
    verify_output="$("$VERIFY" 2>&1)"
    while IFS= read -r line; do
        plain="$(plain_text "$line")"
        case "$plain" in
            "  ok   "*)  name="${plain#  ok   }"; name="${name%% -> *}"
                         matches_filter "$name" && record_pass "$name" 0 ;;
            "  FAIL "*)  record_fail "${plain#  FAIL }" "" ;;
            "  SKIP "*)  record_skip "${plain#  SKIP }" "the verifier skipped it" ;;
        esac
    done <<< "$verify_output"
fi

if [[ -x "$ROOT/tests/vm/field_kind_disasm.sh" ]]; then
    kind_output="$(JAITHON="$JAITHON" "$ROOT/tests/vm/field_kind_disasm.sh" 2>&1)"
    while IFS= read -r line; do
        case "$line" in
            "ok "*)   name="field_kind: ${line#ok }"
                      matches_filter "$name" && record_pass "$name" 0 ;;
            "FAIL"*)  record_fail "field_kind: ${line#FAIL: }" "" ;;
        esac
    done <<< "$kind_output"
fi

if [[ -x "$ROOT/tests/vm/sidecar.sh" ]]; then
    sidecar_output="$(JAITHON="$JAITHON" "$ROOT/tests/vm/sidecar.sh" 2>&1)"
    while IFS= read -r line; do
        case "$line" in
            "ok "*)   name="sidecar: ${line#ok }"
                      matches_filter "$name" && record_pass "$name" 0 ;;
            "FAIL "*) record_fail "sidecar: ${line#FAIL }" "" ;;
        esac
    done <<< "$sidecar_output"
fi

if [[ -x "$ROOT/tests/vm/cache_corrupt.sh" ]]; then
    corrupt_output="$(JAITHON="$JAITHON" "$ROOT/tests/vm/cache_corrupt.sh" 2>&1)"
    while IFS= read -r line; do
        case "$line" in
            "ok "*)   name="cache_corrupt: ${line#ok }"
                      matches_filter "$name" && record_pass "$name" 0 ;;
            "FAIL "*) record_fail "cache_corrupt: ${line#FAIL }" "" ;;
        esac
    done <<< "$corrupt_output"
fi

# ---------------------------------------------------------------- 2. golden
printf '%sGolden tests%s\n' "$BOLD" "$RESET"

run_golden() {
    local name="$1" src="$2" expected="$3" flag="${4:-}" envset="${5:-}"
    local start actual status errout elapsed prefix
    prefix="env"
    [[ -n "$envset" ]] && prefix="env $envset"
    start=$(now_ms)
    if [[ -n "$flag" ]]; then
        actual="$($prefix "$JAITHON" run "$flag" "$src" 2>"/tmp/jai_test_err.$$")"
    else
        actual="$($prefix "$JAITHON" run "$src" 2>"/tmp/jai_test_err.$$")"
    fi
    status=$?
    errout="$(cat "/tmp/jai_test_err.$$")"; rm -f "/tmp/jai_test_err.$$"
    elapsed=$(( $(now_ms) - start ))
    if [[ $status -ne 0 ]]; then
        record_fail "$name" "exited $status
$errout"
    elif [[ "$actual" == "$(cat "$expected")" ]]; then
        record_pass "$name" "$elapsed"
    else
        record_fail "$name" "$(diff -u "$expected" <(printf '%s\n' "$actual") | head -40)"
    fi
}

for src in "$ROOT"/tests/golden/*.jai; do
    name="$(basename "$src" .jai)"
    matches_filter "$name" || continue
    expected="${src%.jai}.expected"
    if [[ ! -f "$expected" ]]; then
        record_skip "$name" "no .expected file"
        continue
    fi
    run_golden "$name" "$src" "$expected"
    gc_every="$(sed -n 's/^#: *gc-stress-every: *\([0-9][0-9]*\).*/\1/p' "$src" | head -1)"
    gc_flag="--gc-stress=${gc_every:-$DEFAULT_GC_STRESS_EVERY}"
    [[ $GC_STRESS -eq 1 ]] && run_golden "$name (gc-stress)" "$src" "$expected" "$gc_flag"
    run_golden "$name (deopt-stress)" "$src" "$expected" "" \
        JAITHON_JIT_DEOPT_STRESS=1
    for level in -O0 -O1 -O3; do
        run_golden "$name ($level)" "$src" "$expected" "$level"
    done
done

# ------------------------------------------------------------------ 3. unit
printf '%sUnit tests%s\n' "$BOLD" "$RESET"
unit_args=(test --verbose)
[[ -n "$FILTER" ]] && unit_args+=("--filter=$FILTER")
unit_args+=(
    "$ROOT/tests/lang"
    "$ROOT/tests/stdlib"
    "$ROOT/tests/checker"
    "$ROOT/packages/jaiplot/tests"
    "$ROOT/packages/jaitensor/tests"
)

start=$(now_ms)
unit_output="$("$JAITHON" "${unit_args[@]}" 2>&1)"
unit_status=$?
unit_elapsed=$(( $(now_ms) - start ))
unit_seen=0
unit_failed=0
unit_suite=""
in_failures=0

strip_duration() { local s="$1"; printf '%s' "${s%  *}"; }

while IFS= read -r line; do
    plain="$(plain_text "$line")"
    [[ "$plain" == "FAILURES" ]] && { in_failures=1; continue; }
    [[ $in_failures -eq 1 ]] && continue
    case "$plain" in
        "  pass  "*)
            unit_seen=1
            record_pass "$unit_suite$(strip_duration "${plain#  pass  }")" "$unit_elapsed" ;;
        "  FAIL  "*)
            unit_seen=1; unit_failed=1
            record_fail "$unit_suite$(strip_duration "${plain#  FAIL  }")" "" ;;
        "  ERROR  "*)
            unit_seen=1; unit_failed=1
            record_fail "$unit_suite$(strip_duration "${plain#  ERROR  }")" "" ;;
        "  skip  "*)
            unit_seen=1
            record_skip "$unit_suite$(strip_duration "${plain#  skip  }")" "skipped by the runner" ;;
        "  "*|"") ;;
        *) unit_suite="$plain > " ;;
    esac
done <<< "$unit_output"

if [[ $unit_seen -eq 0 ]]; then
    record_fail "jaithon test" "$unit_output"
elif [[ $unit_status -ne 0 ]]; then
    [[ $unit_failed -eq 0 ]] && record_fail "jaithon test" "exited $unit_status"
    printf '%s\n' "$unit_output" | sed -n '/FAILURES/,$p' | sed 's/^/       /'
fi

# ---------------------------------------------------------- 4. diagnostics
printf '%sDiagnostic tests%s\n' "$BOLD" "$RESET"
for src in "$ROOT"/tests/errors/*.jai; do
    name="$(basename "$src" .jai)"
    matches_filter "$name" || continue
    want="$(head -1 "$src" | sed -n 's/^# *expect: *\([EW][0-9]\{4\}\).*/\1/p')"
    if [[ -z "$want" ]]; then
        record_skip "$name" "no '# expect: Exxxx' header"
        continue
    fi
    output="$("$JAITHON" check "$src" 2>&1)"
    if [[ "$output" == *"$want"* ]]; then
        record_pass "$name" 0
    else
        record_fail "$name" "expected $want, got:
$output"
    fi
done

# ------------------------------------------------------------------ 5. repl
printf '%sREPL tests%s\n' "$BOLD" "$RESET"

repl_normalise() { sed 's/^time: [0-9.]* ms$/time: <duration>/'; }

for src in "$ROOT"/tests/repl/*.repl; do
    name="$(basename "$src" .repl)"
    matches_filter "$name" || continue
    expected="${src%.repl}.expected"
    if [[ ! -f "$expected" ]]; then
        record_skip "$name" "no .expected file"
        continue
    fi
    repl_flags="$(sed -n '1s/^# args: *//p' "$src")"
    declare -a repl_argv=()
    [[ -n "$repl_flags" ]] && eval "repl_argv=($repl_flags)"
    start=$(now_ms)
    (cd "$ROOT/tests/repl" && "$JAITHON" repl ${repl_argv[@]+"${repl_argv[@]}"} \
        < "$src" >"/tmp/jai_repl_out.$$" 2>"/tmp/jai_repl_err.$$")
    status=$?
    actual="$(repl_normalise < "/tmp/jai_repl_out.$$")"
    errout="$(plain_text "$(cat "/tmp/jai_repl_err.$$")")"
    rm -f "/tmp/jai_repl_out.$$" "/tmp/jai_repl_err.$$"
    elapsed=$(( $(now_ms) - start ))

    expected_err="${src%.repl}.expected-err"
    [[ -f "$expected_err" ]] || expected_err="/dev/null"
    expected_exit=0
    [[ -f "${src%.repl}.expected-exit" ]] && expected_exit="$(cat "${src%.repl}.expected-exit")"
    if [[ "$actual" != "$(cat "$expected")" ]]; then
        record_fail "$name" "$(diff -u "$expected" <(printf '%s\n' "$actual") | head -40)"
    elif [[ "$errout" != "$(cat "$expected_err")" ]]; then
        record_fail "$name" "stderr:
$(diff -u "$expected_err" <(printf '%s\n' "$errout") | head -40)"
    elif [[ "$status" != "$expected_exit" ]]; then
        record_fail "$name" "exited $status, expected $expected_exit"
    else
        record_pass "$name" "$elapsed"
    fi
done

# ------------------------------------------------------------ 6. formatting
if [[ $FORMAT -eq 1 ]]; then
    printf '%sFormat check%s\n' "$BOLD" "$RESET"
    start=$(now_ms)
    fmt_output="$("$JAITHON" fmt --check "$ROOT/lib" "$ROOT/tests" "$ROOT/examples" "$ROOT/packages" 2>&1)"
    fmt_status=$?
    if [[ $fmt_status -eq 0 ]]; then
        record_pass "fmt --check" "$(( $(now_ms) - start ))"
    else
        record_fail "fmt --check" "$fmt_output"
    fi
fi

# --------------------------------------------------------------- 7. summary
printf '\n%s' "$BOLD"
printf '%s\n' "────────────────────────────────────────"
printf '%s' "$RESET"
if [[ $fail -eq 0 ]]; then
    printf '%s%d passed%s' "$GREEN" "$pass" "$RESET"
else
    printf '%s%d passed%s, %s%d failed%s' "$GREEN" "$pass" "$RESET" "$RED" "$fail" "$RESET"
fi
[[ $skip -gt 0 ]] && printf ', %s%d skipped%s' "$YELLOW" "$skip" "$RESET"
printf '\n'

if [[ $fail -gt 0 ]]; then
    printf '\n%sFailures:%s\n' "$BOLD" "$RESET"
    for f in "${failures[@]}"; do printf '  %s\n' "$f"; done
    exit 1
fi
exit 0

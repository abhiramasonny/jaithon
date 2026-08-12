#!/usr/bin/env bash
# Jaithon test driver — the whole gate behind `make test`.
#
# Six layers, each of which must pass on its own:
#
#   1. verifier     — build/*/verify_chunk, the C-only bytecode verifier tests;
#                     they feed jaiVerifyChunk malformed bytecode, which no
#                     .jai source can express. Skipped when not built.
#   2. golden       — a .jai file next to a .expected file; stdout must match
#                     at every optimisation level, under --gc-stress, and under
#                     JAITHON_JIT_DEOPT_STRESS=1, since a collection that frees
#                     something still live, a pass that rewrites a jump wrongly
#                     and a guard that hands the interpreter the wrong operand
#                     stack all show up nowhere else
#   3. unit         — core and workspace package tests, run by `jaithon test`
#                     (jaithon.tool.test): every top-level `test_` function
#   4. diagnostics  — tests/errors/*.jai must FAIL with the code named on the
#                     file's first line, proving the checker catches it
#   5. repl         — tests/repl/*.repl, a session fed to `jaithon repl` on
#                     stdin; its stdout must match the .expected beside it, and
#                     its stderr the .expected-err when there is one
#   6. formatting   — `jaithon fmt --check` over lib, tests, examples and packages, the
#                     gate that keeps the tree canonical. Off by default: it
#                     re-parses every file and costs more than the other four
#                     together. `make fmt-check` runs the same thing alone.
#
# All six report into one summary and one exit status: 0 only when every
# layer passed.
#
# Usage: scripts/run_tests.sh [-v] [--no-gc-stress] [--format] [filter]
#
#   -v              name every test as it passes, not just the failures
#   --no-gc-stress  skip the second golden pass, which is the slow half
#   --format        also run `jaithon fmt --check` over the tree
#   filter          substring; only matching cases run, in every layer

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-build}"
# Overridable so a second build can be tested without touching the shared tree:
#   BUILD_ROOT=build-repl JAITHON=$PWD/jaithon-repl ./scripts/run_tests.sh
JAITHON="${JAITHON:-$ROOT/jaithon}"
VERBOSE=0
GC_STRESS=1
FORMAT=0
FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--verbose)   VERBOSE=1; shift ;;
        --no-gc-stress) GC_STRESS=0; shift ;;
        --format)       FORMAT=1; shift ;;
        -h|--help)
            sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
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

# The reverse of run_bench.sh's guard, and not fatal: this suite is meaningful
# against either build, but only the debug one has the assertions and the chunk
# verifier compiled in, so a green run against release proves strictly less.
# Say which was used rather than leave it to be inferred later from a timing.
BUILD_KIND="$(cut -d'|' -f1 "$ROOT/$BUILD_ROOT/.link-id" 2>/dev/null | tr -d '[:space:]')"
printf '%s%s build%s\n\n' "$DIM" "${BUILD_KIND:-unattributed}" "$RESET"

pass=0; fail=0; skip=0
declare -a failures=()

record_pass() { pass=$((pass + 1)); [[ $VERBOSE -eq 1 ]] && printf '  %sPASS%s %s %s(%sms)%s\n' "$GREEN" "$RESET" "$1" "$DIM" "$2" "$RESET"; return 0; }
record_fail() { fail=$((fail + 1)); failures+=("$1"); printf '  %sFAIL%s %s\n' "$RED" "$RESET" "$1"; [[ -n "${2:-}" ]] && printf '%s\n' "$2" | sed 's/^/       /'; return 0; }
record_skip() { skip=$((skip + 1)); [[ $VERBOSE -eq 1 ]] && printf '  %sSKIP%s %s — %s\n' "$YELLOW" "$RESET" "$1" "$2"; return 0; }

now_ms() { python3 -c 'import time; print(int(time.time()*1000))'; }

matches_filter() { [[ -z "$FILTER" || "$1" == *"$FILTER"* ]]; }

# Strip the SGR sequences a tool writes when it thinks it has a terminal.
plain_text() { printf '%s' "$1" | sed $'s/\033\\[[0-9;]*m//g'; }

shopt -s nullglob

# -------------------------------------------------------------- 1. verifier
# `make test` builds this; a bare invocation of this script may not have it.
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

# A truncated or corrupt .jaic must be an ordinary cache miss, never a crash and
# never a load failure. It reads and corrupts real cache files, so it runs here
# rather than as a golden -- a golden only compares stdout.
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

run_golden() {   # name, source, expected, extra jaithon flag ("" for none),
                 # NAME=VALUE for the child's environment ("" for none)
    local name="$1" src="$2" expected="$3" flag="${4:-}" envset="${5:-}"
    local start actual status errout elapsed prefix
    # Words rather than an array: macOS ships bash 3.2, where "${a[@]}" on an
    # empty array is an unbound variable under `set -u` and every golden test
    # fails with no command having run at all. Both halves are literals from
    # this script, so splitting them is what is wanted.
    prefix="env"
    [[ -n "$envset" ]] && prefix="env $envset"
    start=$(now_ms)
    # Compare stdout only. Diagnostics and warnings go to stderr by design, so
    # merging the two here would make every test fail on unrelated noise.
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
    [[ $GC_STRESS -eq 1 ]] && run_golden "$name (gc-stress)" "$src" "$expected" --gc-stress
    # Every guard in compiled code fails at once, so each of these runs the
    # deoptimisation path too -- the record a guard hands back, and the
    # interpreter picking up from it. Nothing else reaches it: a guard fires in
    # ordinary running only when a program changes a type under the compiler,
    # and almost none do. It costs about 2% of this suite.
    run_golden "$name (deopt-stress)" "$src" "$expected" "" \
        JAITHON_JIT_DEOPT_STRESS=1
    # The optimisation level may change how long a program takes, never what it
    # prints. -O2 is the default the line above already covered.
    for level in -O0 -O1 -O3; do
        run_golden "$name ($level)" "$src" "$expected" "$level"
    done
done

# ------------------------------------------------------------------ 3. unit
#
# One `jaithon test` run covers the core and package directories: the runner discovers the
# files, executes each in a namespace of its own, and reports one line per
# test under --verbose. Those lines are counted here so that the summary
# counts tests rather than files; the runner's own summary is dropped, since
# this script owns the total.
printf '%sUnit tests%s\n' "$BOLD" "$RESET"
unit_args=(test --verbose)
[[ -n "$FILTER" ]] && unit_args+=("--filter=$FILTER")
unit_args+=(
    "$ROOT/tests/lang"
    "$ROOT/tests/stdlib"
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

# A per-test line is "  LABEL  name  duration"; the same labels appear again in
# the FAILURES block the runner prints afterwards, so stop at that heading or
# every failure is counted twice.
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
        *) unit_suite="$plain > " ;;      # a bare line is the file's heading
    esac
done <<< "$unit_output"

if [[ $unit_seen -eq 0 ]]; then
    # No test line at all: discovery or loading broke, which is a failure of
    # the layer rather than of any one test.
    record_fail "jaithon test" "$unit_output"
elif [[ $unit_status -ne 0 ]]; then
    # A nonzero status with nothing counted would let a failure through, so
    # the status is authoritative even if the output could not be read.
    [[ $unit_failed -eq 0 ]] && record_fail "jaithon test" "exited $unit_status"
    printf '%s\n' "$unit_output" | sed -n '/FAILURES/,$p' | sed 's/^/       /'
fi

# ---------------------------------------------------------- 4. diagnostics
printf '%sDiagnostic tests%s\n' "$BOLD" "$RESET"
for src in "$ROOT"/tests/errors/*.jai; do
    name="$(basename "$src" .jai)"
    matches_filter "$name" || continue
    # First line looks like:  # expect: E0301
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
#
# One case is a .repl file of lines fed to `jaithon repl` on stdin, next to the
# .expected stdout it must produce. stderr is compared separately, against a
# .expected-err, and only when that file exists: the REPL flushes stdout once
# per input while diagnostics leave as they are made, so a merged transcript
# would interleave by whatever the pipe happened to buffer rather than by
# anything the REPL decides.
#
# The exit status is checked too, because a REPL fed by a pipe is a program
# being run and a shell has nothing else to test. It must be 0 unless an
# .expected-exit beside the case says which status the session is meant to end
# with, so a new case that quietly starts failing is a failure of the suite.
#
# Cases run with tests/repl as the working directory, so a `:load` in one names
# its fixture without a path. A first line of `# args: -O0` passes flags to the
# process; the REPL treats it as a comment, so it costs the case nothing.
printf '%sREPL tests%s\n' "$BOLD" "$RESET"

# `:time` reports a duration that is different every run. Nothing else in a
# transcript varies, so this is the only normalisation.
repl_normalise() { sed 's/^time: [0-9.]* ms$/time: <duration>/'; }

for src in "$ROOT"/tests/repl/*.repl; do
    name="$(basename "$src" .repl)"
    matches_filter "$name" || continue
    expected="${src%.repl}.expected"
    if [[ ! -f "$expected" ]]; then
        record_skip "$name" "no .expected file"
        continue
    fi
    # The header is a flag list rather than one argument, so it is split the
    # way a shell splits a command line: on spaces, but honouring quotes, so a
    # flag whose value contains a space (`--eval=':disasm 1+2'`) is one word.
    # Word splitting alone lost those, and left `[` and `*` in an expression to
    # be globbed away under the `nullglob` this script sets.
    repl_flags="$(sed -n '1s/^# args: *//p' "$src")"
    declare -a repl_argv=()
    [[ -n "$repl_flags" ]] && eval "repl_argv=($repl_flags)"
    start=$(now_ms)
    # Not a pipeline: the status wanted is the REPL's own, and inside a command
    # substitution PIPESTATUS is gone by the time it could be read.
    # `${a[@]+"${a[@]}"}`, not `"${a[@]}"`: bash 3.2 is what /usr/bin/env finds
    # on a stock Mac, and there an empty array under `set -u` is an unbound
    # variable, which would fail every case that passes no flags.
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
# Off by default: re-parsing every file under lib, tests, examples and packages costs
# several times what the other four layers cost together, which is too much to
# pay on every run. `make fmt-check` is the same gate on its own.
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

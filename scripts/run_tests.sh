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

if [[ -x "$ROOT/tests/vm/osr_replace.sh" ]]; then
    replace_output="$(JAITHON="$JAITHON" "$ROOT/tests/vm/osr_replace.sh" 2>&1)"
    while IFS= read -r line; do
        case "$line" in
            "ok "*)   name="osr_replace: ${line#ok }"
                      matches_filter "$name" && record_pass "$name" 0 ;;
            "FAIL "*) record_fail "osr_replace: ${line#FAIL }" "" ;;
        esac
    done <<< "$replace_output"
fi

# ----------------------------------------------------- unit tests, started
#
# The unit run is started here and only READ in section 3. Its wall time is one
# long test file (test_against_opencv, ~20s) with every other core idle for most
# of it, and the goldens are hundreds of short processes: run side by side they
# fill those cores. The report is unchanged -- section 3 still prints where it
# always did. UNIT_OVERLAP=0 starts it in section 3 instead.
unit_args=(test --verbose)
[[ -n "$FILTER" ]] && unit_args+=("--filter=$FILTER")
unit_args+=(
    "$ROOT/tests/lang"
    "$ROOT/tests/stdlib"
    "$ROOT/tests/checker"
    "$ROOT/packages/jaicv/tests"
    "$ROOT/packages/jaiplot/tests"
    "$ROOT/packages/jaitensor/tests"
    "$ROOT/packages/jainum/tests"
    # jaiframe sat out of this list for a while on the reading that its arith,
    # groupby and reduce modules disagreed with pandas about nulls. They did
    # not. All 72 failures were one defect in the tier: a bool returned by a
    # function in another module was loaded eight bytes wide out of the call
    # descriptor, and `BOOL_VAL` fills one. `Column.from_floats` asks
    # `values.map(|v| is_nan(v))` which rows are missing, that came back
    # all-true, and every column arrived fully null -- so the answers were
    # `nan` and zero, which reads exactly like null semantics. See
    # tests/lang/test_jit_bool_call_return.jai.
    "$ROOT/packages/jaiframe/tests"
    "$ROOT/packages/jailearn/tests"
    "$ROOT/packages/jaisci/tests"
    "$ROOT/packages/jaitoml/tests"
    "$ROOT/packages/jaiyaml/tests"
)

unit_pid=""
unit_file=""
unit_start=0
start_unit_tests() {
    unit_file="$(mktemp "${TMPDIR:-/tmp}/jai_unit.XXXXXX")"
    unit_start=$(now_ms)
    "$JAITHON" "${unit_args[@]}" > "$unit_file" 2>&1 &
    unit_pid=$!
}
[[ "${UNIT_OVERLAP:-1}" != "0" ]] && start_unit_tests

# ---------------------------------------------------------------- 2. golden
printf '%sGolden tests%s\n' "$BOLD" "$RESET"

# Every golden runs in up to eight configurations, ~500 `jaithon run`s in all,
# and they used to run one after another: 74s of a 160s gate, on a machine with
# a dozen cores idle. They are independent processes, so they now run
# GOLDEN_JOBS at a time (default: one per core; GOLDEN_JOBS=1 is the old
# serial order) and are RECORDED in the order they were listed, so the report
# reads exactly as before. A job's stdout and stderr go to files, so no job can
# stall on a full pipe. The REPL phase uses the same runner.
#
# A job is its fields joined by \x1f -- the working directory (empty: this
# one), a file to read stdin from (empty: none), KEY=VALUE pairs separated by
# spaces, then the command's own words -- and jobs are separated by \x1e, since
# a word can hold a newline (`--eval=$'...\n...'`). JOB_MERGE_STDERR=1 among
# the pairs sends stderr to the .out file too, as `2>&1` would.
parallel_jobs=()
queue_job() {
    local IFS=$'\x1f'
    parallel_jobs+=("$*")
}

# Run every queued job; job N leaves N.out, N.err and N.status ("code ms") in $1.
run_parallel_jobs() {
    local dir="$1"
    [[ ${#parallel_jobs[@]} -eq 0 ]] && return 0
    printf '%s\x1e' "${parallel_jobs[@]}" > "$dir/jobs"
    python3 - "$dir" "${GOLDEN_JOBS:-0}" <<'PY'
import os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
directory, wanted = sys.argv[1], int(sys.argv[2])
jobs = open(os.path.join(directory, "jobs"), encoding="utf-8", errors="surrogateescape").read().split("\x1e")[:-1]

def run(index):
    fields = jobs[index].split("\x1f")
    cwd, stdin, envset, command = fields[0] or None, fields[1], fields[2], fields[3:]
    env = dict(os.environ)
    merged = False
    for pair in envset.split():
        key, _, value = pair.partition("=")
        if key == "JOB_MERGE_STDERR":
            merged = value == "1"
            continue
        env[key] = value
    base = os.path.join(directory, str(index))
    started = time.time()
    with open(base + ".out", "wb") as out, open(base + ".err", "wb") as err:
        source = open(stdin, "rb") if stdin else None
        try:
            status = subprocess.call(command, cwd=cwd, stdin=source, stdout=out,
                                     stderr=subprocess.STDOUT if merged else err, env=env)
        finally:
            if source is not None:
                source.close()
    elapsed = int((time.time() - started) * 1000)
    with open(base + ".status", "w") as record:
        record.write(f"{status} {elapsed}\n")

workers = wanted if wanted > 0 else (os.cpu_count() or 1)
with ThreadPoolExecutor(max_workers=workers) as pool:
    list(pool.map(run, range(len(jobs))))
PY
}

# What each golden job is checked against, by job index.
golden_names=()
golden_expected=()
queue_golden() {
    local name="$1" src="$2" expected="$3" flag="${4:-}" envset="${5:-}"
    golden_names+=("$name")
    golden_expected+=("$expected")
    if [[ "$name" == "@skip" ]]; then
        queue_job "" "" "" true
    elif [[ -n "$flag" ]]; then
        queue_job "" "" "$envset" "$JAITHON" run "$flag" "$src"
    else
        queue_job "" "" "$envset" "$JAITHON" run "$src"
    fi
}

record_golden_jobs() {
    local dir="$1" index=0 count=${#golden_names[@]} name expected status elapsed actual errout
    while [[ $index -lt $count ]]; do
        name="${golden_names[$index]}"
        expected="${golden_expected[$index]}"
        if [[ "$name" == "@skip" ]]; then
            record_skip "$expected" "no .expected file"
            index=$((index + 1))
            continue
        fi
        status=255
        elapsed=0
        [[ -f "$dir/$index.status" ]] && read -r status elapsed < "$dir/$index.status"
        actual="$(cat "$dir/$index.out")"
        errout="$(cat "$dir/$index.err")"
        if [[ $status -ne 0 ]]; then
            record_fail "$name" "exited $status
$errout"
        elif [[ "$actual" == "$(cat "$expected")" ]]; then
            record_pass "$name" "$elapsed"
        else
            record_fail "$name" "$(diff -u "$expected" <(printf '%s\n' "$actual") | head -40)"
        fi
        index=$((index + 1))
    done
}

for src in "$ROOT"/tests/golden/*.jai; do
    name="$(basename "$src" .jai)"
    matches_filter "$name" || continue
    expected="${src%.jai}.expected"
    if [[ ! -f "$expected" ]]; then
        queue_golden "@skip" "" "$name"
        continue
    fi
    queue_golden "$name" "$src" "$expected"
    gc_every="$(sed -n 's/^#: *gc-stress-every: *\([0-9][0-9]*\).*/\1/p' "$src" | head -1)"
    gc_flag="--gc-stress=${gc_every:-$DEFAULT_GC_STRESS_EVERY}"
    [[ $GC_STRESS -eq 1 ]] && queue_golden "$name (gc-stress)" "$src" "$expected" "$gc_flag"
    queue_golden "$name (deopt-stress)" "$src" "$expected" "" \
        JAITHON_JIT_DEOPT_STRESS=1
    # Every body compiles on its FIRST call instead of its 64th, so the golden
    # actually exercises the compiled tier. Without this 40 of the 61 goldens
    # compile nothing at all, and a differential against JAITHON_NO_JIT=1 over
    # them compares the interpreter with itself -- which is how an empty
    # initializer that returned null instead of the new object shipped. Results
    # must be UNCHANGED; a difference here is a miscompile. See jaiJitThreshold
    # in src/vm/jit/jit.h.
    queue_golden "$name (jit-first-call)" "$src" "$expected" "" \
        JAITHON_JIT_THRESHOLD=1
    # Opt-in, via `#: jit-cold-compile: yes` in the golden's own header, because
    # --no-cache costs a full stdlib recompile per run and 61 of them is a
    # minute of gate for nothing. A golden asks for it when what it checks
    # depends on the COMPILER having run in-process: the tier's fallback paths
    # (compileReturnNull, compileAccessor) are only reached once the shared code
    # arena is full, and a warm cache never fills it. That is the difference
    # between catching an empty initializer that returns null and not.
    if sed -n 's/^#: *jit-cold-compile: *\(yes\).*/\1/p' "$src" | head -1 | grep -q yes; then
        queue_golden "$name (cold-compile, jit-first-call)" "$src" "$expected" \
            "--no-cache" JAITHON_JIT_THRESHOLD=1
    fi
    for level in -O0 -O1 -O3; do
        queue_golden "$name ($level)" "$src" "$expected" "$level"
    done
done
golden_dir="$(mktemp -d "${TMPDIR:-/tmp}/jai_golden.XXXXXX")"
run_parallel_jobs "$golden_dir"
record_golden_jobs "$golden_dir"
rm -rf "$golden_dir"
parallel_jobs=()

# ------------------------------------------------- 2b. fuzzer bug repros
#
# tests/fuzz/found/ holds the programs that caught real miscompiles. Nothing ran
# them: they sat as artifacts, so a regression that reintroduced any of those
# bugs would be caught only by a fuzz run that happened to reinvent the same
# shape. Every one of them is a DIFFERENTIAL -- there is no expected output to
# maintain, because the oracle is the tier's own contract (src/vm/jit/jit.h):
# declining is always allowed, answering differently never is.
#
# The configurations are the fuzzer's, not the golden runner's, and that
# matters: these bugs were found under TICK_US (the OSR loop tier) and
# SPLIT_STRESS (the split operand bank), neither of which any golden runs.
# DEOPT_STRESS and THRESHOLD=1 alone would not have caught them.
#
# A file opts OUT with `#: differential-exempt: yes`, which exactly one does --
# recursion_depth_limit.jai, where the compiled tier deliberately does not
# enforce the interpreter's frame limit. See "The one divergence that is
# deliberate" in src/vm/jit/README.md.
printf '%sFuzzer bug repros%s\n' "$BOLD" "$RESET"
# Six runs per program, all independent: on the parallel job runner too.
fuzz_configs=("" "JAITHON_JIT_TICK_US=50" "JAITHON_JIT_DEOPT_STRESS=1" \
              "JAITHON_JIT_THRESHOLD=1" "JAITHON_JIT_SPLIT_STRESS=1")
fuzz_names=()
fuzz_first=()
for src in "$ROOT"/tests/fuzz/found/*.jai; do
    name="fuzz/$(basename "$src" .jai)"
    matches_filter "$name" || continue
    if sed -n 's/^#: *differential-exempt: *\(yes\).*/\1/p' "$src" | head -1 \
       | grep -q yes; then
        fuzz_names+=("$name")
        fuzz_first+=(-1)
        continue
    fi
    fuzz_names+=("$name")
    fuzz_first+=(${#parallel_jobs[@]})
    queue_job "" "" "JOB_MERGE_STDERR=1 JAITHON_NO_JIT=1" "$JAITHON" run "$src"
    for cfg in "${fuzz_configs[@]}"; do
        queue_job "" "" "JOB_MERGE_STDERR=1 $cfg" "$JAITHON" run "$src"
    done
done
fuzz_dir="$(mktemp -d "${TMPDIR:-/tmp}/jai_fuzz.XXXXXX")"
run_parallel_jobs "$fuzz_dir"
index=0
while [[ $index -lt ${#fuzz_names[@]} ]]; do
    name="${fuzz_names[$index]}"
    first="${fuzz_first[$index]}"
    index=$((index + 1))
    if [[ $first -lt 0 ]]; then
        record_skip "$name" "differential-exempt (deliberate divergence)"
        continue
    fi
    reference="$(cat "$fuzz_dir/$first.out")"
    bad=""
    elapsed=0
    job=$first
    for cfg in "" "${fuzz_configs[@]}"; do
        took=0
        [[ -f "$fuzz_dir/$job.status" ]] && read -r _ took < "$fuzz_dir/$job.status"
        elapsed=$((elapsed + took))
        if [[ $job -ne $first ]]; then
            actual="$(cat "$fuzz_dir/$job.out")"
            if [[ "$actual" != "$reference" ]]; then
                bad+="under ${cfg:-default}:
$(diff <(printf '%s\n' "$reference") <(printf '%s\n' "$actual") | head -20)
"
            fi
        fi
        job=$((job + 1))
    done
    if [[ -z "$bad" ]]; then
        record_pass "$name" "$elapsed"
    else
        record_fail "$name" "$bad"
    fi
done
rm -rf "$fuzz_dir"
parallel_jobs=()

# ------------------------------------------------------------------ 3. unit
printf '%sUnit tests%s\n' "$BOLD" "$RESET"
[[ -z "$unit_pid" ]] && start_unit_tests
wait "$unit_pid"
unit_status=$?
unit_output="$(cat "$unit_file")"
rm -f "$unit_file"
unit_elapsed=$(( $(now_ms) - unit_start ))
unit_seen=0
unit_failed=0
unit_suite=""
in_failures=0

# The duration is the last two-space-separated field of a result line. The
# colour codes come off the whole output in one sed, and the duration by
# parameter expansion: a sed and a subshell per result line, over ~3,000
# lines, was seconds of the phase.
unit_plain="$(plain_text "$unit_output")"
while IFS= read -r plain; do
    [[ "$plain" == "FAILURES" ]] && { in_failures=1; continue; }
    [[ $in_failures -eq 1 ]] && continue
    case "$plain" in
        "  pass  "*)
            unit_seen=1
            test_name="${plain#  pass  }"
            record_pass "$unit_suite${test_name%  *}" "$unit_elapsed" ;;
        "  FAIL  "*)
            unit_seen=1; unit_failed=1
            test_name="${plain#  FAIL  }"
            record_fail "$unit_suite${test_name%  *}" "" ;;
        "  ERROR  "*)
            unit_seen=1; unit_failed=1
            test_name="${plain#  ERROR  }"
            record_fail "$unit_suite${test_name%  *}" "" ;;
        "  skip  "*)
            unit_seen=1
            test_name="${plain#  skip  }"
            record_skip "$unit_suite${test_name%  *}" "skipped by the runner" ;;
        "  "*|"") ;;
        *) unit_suite="$plain > " ;;
    esac
done <<< "$unit_plain"

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

repl_names=()
repl_sources=()
for src in "$ROOT"/tests/repl/*.repl; do
    name="$(basename "$src" .repl)"
    matches_filter "$name" || continue
    expected="${src%.repl}.expected"
    repl_names+=("$name")
    repl_sources+=("$src")
    if [[ ! -f "$expected" ]]; then
        queue_job "" "" "" true
        continue
    fi
    repl_flags="$(sed -n '1s/^# args: *//p' "$src")"
    declare -a repl_argv=()
    [[ -n "$repl_flags" ]] && eval "repl_argv=($repl_flags)"
    queue_job "$ROOT/tests/repl" "$src" "" "$JAITHON" repl ${repl_argv[@]+"${repl_argv[@]}"}
done
repl_dir="$(mktemp -d "${TMPDIR:-/tmp}/jai_repl.XXXXXX")"
run_parallel_jobs "$repl_dir"
index=0
while [[ $index -lt ${#repl_names[@]} ]]; do
    name="${repl_names[$index]}"
    src="${repl_sources[$index]}"
    expected="${src%.repl}.expected"
    if [[ ! -f "$expected" ]]; then
        record_skip "$name" "no .expected file"
        index=$((index + 1))
        continue
    fi
    status=255
    elapsed=0
    [[ -f "$repl_dir/$index.status" ]] && read -r status elapsed < "$repl_dir/$index.status"
    actual="$(repl_normalise < "$repl_dir/$index.out")"
    errout="$(plain_text "$(cat "$repl_dir/$index.err")")"

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
    index=$((index + 1))
done
rm -rf "$repl_dir"
parallel_jobs=()

# ------------------------------------------------- 5b. package importability
#
# Every package's own tests import its FILES -- `from jaisci.core import ...` --
# so none of them ever exercised the facade a user reaches through
# `import jaisci`. That facade named nineteen modules of which one existed, so
# `import jaisci` raised for every caller while the suite stayed green and the
# manifest check called the package valid. One line each, and it would not have.
printf '%sPackage imports%s\n' "$BOLD" "$RESET"
for manifest in "$ROOT"/packages/*/jaithon.package.json; do
    pkg="$(basename "$(dirname "$manifest")")"
    matches_filter "$pkg" || continue
    start=$(now_ms)
    probe="$(mktemp -t jaithon-import-XXXXXX)"
    mv "$probe" "$probe.jai"
    probe="$probe.jai"
    printf 'import %s\nfn main() -> int { return 0 }\n' "$pkg" > "$probe"
    import_output="$("$JAITHON" "$probe" 2>&1)"
    import_status=$?
    rm -f "$probe"
    if [[ $import_status -eq 0 ]]; then
        record_pass "import $pkg" "$(( $(now_ms) - start ))"
    else
        record_fail "import $pkg" "$import_output"
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

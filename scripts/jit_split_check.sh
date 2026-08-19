#!/usr/bin/env bash
# Check for split-operand-bank differential bugs in JIT compilation.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export JAITHON_PATH="$ROOT/lib"
export BENCH_LEVEL="${BENCH_LEVEL:-easy}"
JAITHON="${JAITHON:-$ROOT/jaithon}"
RUNS=${SPLIT_CHECK_RUNS:-3}

# tests/bench/jaitensor/mlp.jai reports its own wall clock in the third field of
# its `jtb` line, and two runs never agree on it. Comparing it asks nothing --
# what this gate is for is whether compiled code computes the same ANSWERS, and
# those are the fields after it. Only that one field is blanked; the loss and
# the checksum beside it are floats too and are compared exactly.
strip_timings() {
    awk -F'\t' 'BEGIN { OFS = "\t" } $1 == "jtb" && NF > 2 { $3 = "TIME" } { print }'
}

fail=0
for f in "$ROOT"/tests/bench/*/*.jai; do
    want=$(JAITHON_NO_JIT=1 "$JAITHON" run "$f" 2>&1 | strip_timings)
    for _i in $(seq "$RUNS"); do
        got=$(JAITHON_JIT_SPLIT_STRESS=1 "$JAITHON" run "$f" 2>&1 | strip_timings)
        if [[ "$got" != "$want" ]]; then
            echo "SPLIT DIFF: ${f#"$ROOT"/}"
            echo "  interpreter: $(head -c 200 <<<"$want")"
            echo "  split bank : $(head -c 200 <<<"$got")"
            fail=1
            break
        fi
    done
    got=$(JAITHON_JIT_SPLIT_STRESS=1 JAITHON_JIT_DEOPT_STRESS=1 \
          "$JAITHON" run "$f" 2>&1 | strip_timings)
    if [[ "$got" != "$want" ]]; then
        echo "SPLIT+DEOPT DIFF: ${f#"$ROOT"/}"
        fail=1
    fi
done

if [[ $fail -ne 0 ]]; then
    echo "jit split check FAILED"
    exit 1
fi
echo "jit split check: all benchmarks agree with the interpreter"

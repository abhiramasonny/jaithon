#!/usr/bin/env bash
# Check for split-operand-bank differential bugs in JIT compilation.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export JAITHON_PATH="$ROOT/lib"
export BENCH_LEVEL="${BENCH_LEVEL:-easy}"
JAITHON="${JAITHON:-$ROOT/jaithon}"
RUNS=${SPLIT_CHECK_RUNS:-3}

fail=0
for f in "$ROOT"/tests/bench/*/*.jai; do
    want=$(JAITHON_NO_JIT=1 "$JAITHON" run "$f" 2>&1)
    for _i in $(seq "$RUNS"); do
        got=$(JAITHON_JIT_SPLIT_STRESS=1 "$JAITHON" run "$f" 2>&1)
        if [[ "$got" != "$want" ]]; then
            echo "SPLIT DIFF: ${f#"$ROOT"/}"
            echo "  interpreter: $(head -c 200 <<<"$want")"
            echo "  split bank : $(head -c 200 <<<"$got")"
            fail=1
            break
        fi
    done
    got=$(JAITHON_JIT_SPLIT_STRESS=1 JAITHON_JIT_DEOPT_STRESS=1 \
          "$JAITHON" run "$f" 2>&1)
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

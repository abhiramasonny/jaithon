#!/usr/bin/env bash
# The split-operand-bank differential.
#
# A body whose operand stack is deeper than it ever is at a call keeps the deep
# half in x0..x8 and the shallow half in the callee-saved bank, so the stack is
# TWO runs of registers rather than one. The failure mode of that is a site that
# adds an index to a base and lands one past the end of the first run: the value
# goes to a register nothing reads, no crash, no diagnostic, wrong answer.
#
# It has happened once already. OP_GET_GLOBAL named its destination with
# `pushReg`, which is one past the CURRENT top rather than the register the next
# push lands in -- the same number until there is a boundary between them. On
# the runs where its loop compiled, bitops printed 68720029766 for 999625.
#
# JAITHON_JIT_SPLIT_STRESS=1 puts the boundary into every OSR body that can take
# one instead of only the ones that pay for it, which is what turns "bitops
# happened to be eligible and to read a global at exactly the boundary" into
# "any benchmark reading anything across it". The interpreter is the oracle.
#
# Each benchmark is run several times because WHICH loops get offered to the
# compiler is driven by a wall-clock sampler, so a single run can miss the body
# that has the boundary in it -- the original bug reproduced on 5 runs in 6.
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
    # ...and with every guard forced to fail, so the deopt stubs write the
    # operand stack out across the boundary too.
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

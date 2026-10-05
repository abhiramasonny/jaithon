#!/usr/bin/env bash
# The early-return arm compiled ahead of the frame (JAITHON_JIT_SHRINK_WRAP).
#
# The switch is off by default -- it wins on some recursions and loses on
# others (see src/vm/jit/README.md) -- so tests/lang/test_jit_early_return.jai
# no longer reaches the arm on its own. This runs that file with the arm on,
# in the modes that compile the most and that deoptimise the most, and checks
# that the arm is actually emitted.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export JAITHON_PATH="$ROOT/lib"
JAITHON="${JAITHON:-$ROOT/jaithon}"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

cat > "$work/fib.jai" <<'JAI'
fn fib(n: int) -> int {
    if n < 2 { return n }
    return fib(n - 1) + fib(n - 2)
}
print(fib(24))
JAI

fail=0
note() { echo "$1 $2"; [ "$1" = "FAIL" ] && fail=1; return 0; }

clear="-u JAITHON_NO_JIT -u JAITHON_JIT_DEOPT_STRESS -u JAITHON_JIT_THRESHOLD -u JAITHON_JIT_TICK_US"
why=$(env $clear JAITHON_JIT_SHRINK_WRAP=1 JAI_JIT_WHY=1 "$JAITHON" run "$work/fib.jai" 2>&1)
if printf '%s\n' "$why" | grep -q "fib returns early before its frame" &&
   printf '%s\n' "$why" | grep -qx "46368"; then
    note ok "the arm is emitted when opted in"
else
    note FAIL "fib with the arm on: $(printf '%s\n' "$why" | tail -1)"
fi

# Off unless asked for: the losing recursions are as plausible as the
# winning ones, and nothing the walk sees tells them apart.
why=$(env $clear -u JAITHON_JIT_SHRINK_WRAP JAI_JIT_WHY=1 "$JAITHON" run "$work/fib.jai" 2>&1)
if printf '%s
' "$why" | grep -q "returns early before its frame"; then
    note FAIL "the arm is emitted with the switch unset"
else
    note ok "the arm stays off by default"
fi

for mode in "" "JAITHON_JIT_THRESHOLD=1" "JAITHON_JIT_DEOPT_STRESS=1" "JAITHON_JIT_TICK_US=50"; do
    out=$(env $clear $mode JAITHON_JIT_SHRINK_WRAP=1 "$JAITHON" test \
          "$ROOT/tests/lang/test_jit_early_return.jai" 2>&1)
    if [ $? -eq 0 ]; then
        note ok "early-return tests pass with the arm on ${mode:-by default}"
    else
        note FAIL "early-return tests with the arm on ${mode:-by default}: $(printf '%s\n' "$out" | tail -3 | tr '\n' ' ')"
    fi
done
exit $fail

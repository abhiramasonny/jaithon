#!/usr/bin/env bash
# A body whose first function-tier attempt declines must still compile later.
#
# Each program below has a callee whose first attempt declines for a reason
# that goes away a moment later: an upvalue that holds null when the walk
# reads it, a static method that has not compiled yet, and a member of
# another module that has not compiled yet. The retry that succeeds is what
# keeps the loop's calls compiled. A negative cache once skipped those
# retries because its print of the body's inputs did not cover upvalue
# contents, static callees or other modules' members: each program then ran
# 2-3x the interpreted instructions with the right answer, so no output check
# could see it. The main loop is a `while` and the timer tick is long, so it
# stays interpreted and the count measures the callee alone.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export JAITHON_PATH="$ROOT/lib"
JAITHON="${JAITHON:-$ROOT/jaithon}"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

cat > "$work/upv.jai" <<'JAI'
fn main() -> int {
    var k: any = null
    var warm = false
    let f = fn(x: int) -> int {
        if warm { return x * 2 + k }
        return x
    }
    var s = 0
    for i in 0..70 { s = s + f(i) }
    k = 5
    warm = true
    var i = 0
    while i < 300000 {
        s = s + f(i)
        i = i + 1
    }
    print(s)
    return 0
}
JAI

cat > "$work/stat.jai" <<'JAI'
class Mth {
    pub static fn twice(v: int) -> int { return v + v }
    pub static fn plus(a: int, b: int) -> int { return a + b }
}

fn through(i: int) -> int { return Mth.twice(i) + Mth.plus(i, 3) }

fn main() -> int {
    var s = 0
    var i = 0
    while i < 300000 {
        s = s + through(i)
        i = i + 1
    }
    print(s)
    return 0
}
JAI

cat > "$work/helpmod.jai" <<'JAI'
pub fn twice(v: int) -> int { return v + v }
pub fn plus(a: int, b: int) -> int { return a + b }
JAI

cat > "$work/modcall.jai" <<'JAI'
import helpmod

fn through(i: int) -> int { return helpmod.twice(i) + helpmod.plus(i, 3) }

fn main() -> int {
    var s = 0
    var i = 0
    while i < 300000 {
        s = s + through(i)
        i = i + 1
    }
    print(s)
    return 0
}
JAI

fail=0
note() { echo "$1 $2"; [ "$1" = "FAIL" ] && fail=1; return 0; }

check() {
    local name="$1" expected="$2" what="$3"
    "$JAITHON" run "$work/$name.jai" >/dev/null 2>&1   # warm the cache
    answer=$("$JAITHON" run "$work/$name.jai" 2>&1)
    if [ "$answer" != "$expected" ]; then
        note FAIL "$what: inherited mode printed '$answer'"
        return
    fi
    # Compiled, the callee leaves ~2.1-2.4M interpreted instructions (the
    # loop itself); interpreted, 4.2-6.3M.
    out=$(env -u JAITHON_NO_JIT -u JAITHON_JIT_DEOPT_STRESS \
          -u JAITHON_JIT_THRESHOLD JAITHON_JIT_TICK_US=100000 \
          "$JAITHON" run --stats "$work/$name.jai" 2>&1)
    answer=$(printf '%s\n' "$out" | grep -E '^[0-9]+$')
    interp=$(printf '%s\n' "$out" | sed -n 's/^vm: \([0-9]*\) instructions.*/\1/p')
    if [ "$answer" != "$expected" ]; then
        note FAIL "$what: printed '$answer'"
    elif [ -z "$interp" ] || [ "$interp" -gt 3200000 ]; then
        note FAIL "$what: the callee never compiled ($interp instructions)"
    else
        note ok "$what compiles on a later attempt ($interp instructions)"
    fi
}

check upv 90001202415 "a closure whose upvalue was null"
check stat 135000450000 "a body calling statics"
check modcall 135000450000 "a body calling another module"
exit $fail

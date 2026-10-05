#!/usr/bin/env bash
# A guard proved once at a loop head must come from a site the loop runs.
#
# JAITHON_JIT_CLOSURE_HOIST and JAITHON_JIT_GLOBAL_TAG_PROOF move a site's
# guard -- the closure's function, a global's tag -- to the head of a loop.
# A miss at the head sends every pass of that loop to the interpreter, where
# the site's own guard would have missed only when the site ran. Hoisted from
# a site in an arm taken once in 100000 passes, each call with a different
# closure or a global of another kind ran wholly interpreted: 180M
# interpreted instructions here instead of ~1M, with the right answer, so no
# output check could see it. Also: an inlined body with a `%` once declined
# its caller's function tier when its closure guard was hoisted.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export JAITHON_PATH="$ROOT/lib"
JAITHON="${JAITHON:-$ROOT/jaithon}"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

cat > "$work/rare_closure.jai" <<'JAI'
fn inc() -> fn(int) -> int {
    return |x| x + 1
}

fn halve() -> fn(int) -> int {
    return |x| x * 2 % 1000
}

fn run(f: fn(int) -> int, n: int) -> int {
    var acc = 0
    var i = 0
    while i < n {
        acc = acc + i % 7
        if i % 100000 == 99999 { acc = f(acc) }
        i += 1
    }
    return acc
}

fn main() -> int {
    let a = inc()
    let b = halve()
    var total = 0
    for k in 0..300 {
        total = (total + run(if k % 2 == 0 { a } else { b }, 100000)) % 1000000007
    }
    print(total)
    return 0
}
JAI

cat > "$work/rare_global.jai" <<'JAI'
var bonus: any = 1

fn run(n: int) -> int {
    var acc = 0
    var i = 0
    while i < n {
        acc = acc + i % 7
        if i % 100000 == 99999 { acc = acc + int(bonus) }
        i += 1
    }
    return acc
}

fn main() -> int {
    var total = 0
    for k in 0..300 {
        bonus = if k % 2 == 0 { 1 } else { 2.5 }
        total = (total + run(100000)) % 1000000007
    }
    print(total)
    return 0
}
JAI

cat > "$work/mod_body.jai" <<'JAI'
fn adder(step: int) -> fn(int) -> int {
    return |x| (x + 1) % 65521 + step
}

fn apply_n(f: fn(int) -> int, start: int, times: int) -> int {
    var acc = start
    var i = 0
    while i < times {
        acc = f(acc)
        i += 1
    }
    return acc
}

fn main() -> int {
    var total = 0
    for k in 0..300 {
        total = (total + apply_n(adder(k), 0, 5000)) % 1000000007
    }
    print(total)
    return 0
}
JAI

fail=0
note() { echo "$1 $2"; [ "$1" = "FAIL" ] && fail=1; return 0; }

# The instruction bound is a property of the compiled tiers, so it is checked
# with the modes that keep a loop interpreted by design cleared.
clear="-u JAITHON_NO_JIT -u JAITHON_JIT_DEOPT_STRESS -u JAITHON_JIT_THRESHOLD -u JAITHON_JIT_TICK_US"
for case in "rare_closure 45147900" "rare_global 89998950"; do
    set -- $case
    "$JAITHON" run "$work/$1.jai" >/dev/null 2>&1   # warm the cache
    out=$(env $clear "$JAITHON" run --stats "$work/$1.jai" 2>&1)
    answer=$(printf '%s\n' "$out" | grep -E '^[0-9]+$')
    interp=$(printf '%s\n' "$out" | sed -n 's/^vm: \([0-9]*\) instructions.*/\1/p')
    if [ "$answer" != "$2" ]; then
        note FAIL "$1 printed '$answer'"
    elif [ -z "$interp" ] || [ "$interp" -gt 20000000 ]; then
        note FAIL "$1: the loop ran interpreted ($interp instructions)"
    else
        note ok "$1 keeps its loop compiled ($interp instructions)"
    fi
done

why=$(env $clear JAI_JIT_WHY=1 "$JAITHON" run "$work/mod_body.jai" 2>&1)
if printf '%s\n' "$why" | grep -q "compiled __main__.apply_n "; then
    note ok "a hoisted closure with a % body keeps the function tier"
else
    note FAIL "apply_n declined: $(printf '%s\n' "$why" | grep 'apply_n stopped' | head -1)"
fi
exit $fail

#!/usr/bin/env bash
# A loop form re-compiled in place must still be entered.
#
# JAITHON_JIT_PIC_UPGRADE re-compiles a loop whose polymorphic call site
# gained ways after the loop was compiled, into the record slot another head's
# form just vacated. That slot once kept the vacated form's `declines` count:
# when it was an outer loop the back edge had already given up on, the new
# inner-loop form was born given up and the loop ran interpreted for the rest
# of the run -- 4.4M interpreted instructions here instead of ~54K, with the
# right answer, so no output check could see it. A fast timer tick is what
# makes the outer form compile early enough to be given up on.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export JAITHON_PATH="$ROOT/lib"
JAITHON="${JAITHON:-$ROOT/jaithon}"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

cat > "$work/reshape.jai" <<'JAI'
trait Step {
    fn run(self, v: int) -> int
}

class Plus: Step {
    pub var d: int
    fn init(self, d: int) { self.d = d }
    pub fn run(self, v: int) -> int { return (v + self.d) % 65521 }
}

class Times: Step {
    pub var d: int
    fn init(self, d: int) { self.d = d }
    pub fn run(self, v: int) -> int { return v * self.d % 65521 }
}

class Twice: Step {
    pub var d: int
    fn init(self, d: int) { self.d = d }
    pub fn run(self, v: int) -> int { return (v * 2 + self.d) % 65521 }
}

class Halve: Step {
    pub var d: int
    fn init(self, d: int) { self.d = d }
    pub fn run(self, v: int) -> int { return (v // 2 + self.d) % 65521 }
}

class Mirror: Step {
    pub var d: int
    fn init(self, d: int) { self.d = d }
    pub fn run(self, v: int) -> int { return (65520 - v + self.d) % 65521 }
}

fn make_steps(n: int) -> list[Step] {
    var out: list[Step] = []
    var seed = 11
    for _ in 0..n {
        seed = (seed * 75 + 74) % 65537
        let d = seed % 31 + 1
        let pick = seed % 5
        if pick == 0 {
            out.push(Plus(d))
        } elif pick == 1 {
            out.push(Times(d))
        } elif pick == 2 {
            out.push(Twice(d))
        } elif pick == 3 {
            out.push(Halve(d))
        } else {
            out.push(Mirror(d))
        }
    }
    return out
}

fn main() -> int {
    let steps = make_steps(300)
    var v = 3
    var total = 0
    for round in 0..3000 {
        v = (v + round) % 65521
        for s in steps { v = s.run(v) }
        total = (total + v) % 65521
    }
    print(v)
    print(total)
    return 0
}
JAI

fail=0
note() { echo "$1 $2"; [ "$1" = "FAIL" ] && fail=1; return 0; }

expected=$'59334\n9017'
"$JAITHON" run "$work/reshape.jai" >/dev/null 2>&1   # warm the cache
for tick in 50 100; do
    out=$(JAITHON_JIT_TICK_US=$tick "$JAITHON" run --stats "$work/reshape.jai" 2>&1)
    answer=$(printf '%s\n' "$out" | grep -E '^[0-9]+$')
    interp=$(printf '%s\n' "$out" | sed -n 's/^vm: \([0-9]*\) instructions.*/\1/p')
    if [ "$answer" != "$expected" ]; then
        note FAIL "tick $tick printed '$answer'"
    elif [ -z "$interp" ] || [ "$interp" -gt 1000000 ]; then
        note FAIL "tick $tick: the loop ran interpreted ($interp instructions)"
    else
        note ok "tick $tick re-compiled loop is entered ($interp instructions)"
    fi
done
exit $fail

#!/usr/bin/env bash
# The warm-up tick (JAITHON_JIT_MAIN_EDGES): once the program's module body and
# then its main() start, the 4096th interpreted back edge is a tick, so the
# first hot loop -- and, through it, the loops around it -- compiles without
# waiting for the timer. The timer is slowed to 100ms here so that nothing
# else could have compiled them.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export JAITHON_PATH="$ROOT/lib"
JAITHON="${JAITHON:-$ROOT/jaithon}"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# Three nested loops, as mandelbrot has: the warm-up tick lands on the inner
# one, and the two around it must follow at their next back edges rather
# than a whole timer interval later.
cat > "$work/nest.jai" <<'JAI'
fn main() -> int {
    var inside = 0
    for py in 0..60 {
        let y0 = float(py) * 2.0 / 60.0 - 1.0
        for px in 0..400 {
            let x0 = float(px) * 3.0 / 400.0 - 2.0
            var x = 0.0
            var y = 0.0
            var i = 0
            while i < 60 {
                let x2 = x * x
                let y2 = y * y
                if x2 + y2 > 4.0 { break }
                y = 2.0 * x * y + y0
                x = x2 - y2 + x0
                i += 1
            }
            if i == 60 { inside += 1 }
        }
    }
    print(inside)
    return 0
}
JAI

# The same loops at module level: the warm-up starts with the module body too.
cat > "$work/top.jai" <<'JAI'
var total = 0
var i = 0
while i < 2000000 {
    total = total + i % 7
    i = i + 1
}
print(total)
JAI

fail=0
note() { echo "$1 $2"; [ "$1" = "FAIL" ] && fail=1; return 0; }
clear="-u JAITHON_NO_JIT -u JAITHON_JIT_DEOPT_STRESS -u JAITHON_JIT_THRESHOLD"

count() {   # file, env options and settings (options first)...
    local f=$1; shift
    env $clear "$@" JAITHON_JIT_TICK_US=100000 "$JAITHON" run --stats "$f" 2>&1
}

for prog in nest top; do
    on=$(count "$work/$prog.jai" -u JAITHON_JIT_MAIN_EDGES)
    off=$(count "$work/$prog.jai" -u JAITHON_JIT_MAIN_EDGES JAITHON_JIT_MAIN_EDGES=0)
    a_on=$(printf '%s\n' "$on" | grep -E '^[0-9]+$')
    a_off=$(printf '%s\n' "$off" | grep -E '^[0-9]+$')
    n_on=$(printf '%s\n' "$on" | sed -n 's/^vm: \([0-9]*\) instructions.*/\1/p')
    n_off=$(printf '%s\n' "$off" | sed -n 's/^vm: \([0-9]*\) instructions.*/\1/p')
    if [ -z "$a_on" ] || [ "$a_on" != "$a_off" ]; then
        note FAIL "$prog: answers differ ('$a_on' with the warm-up, '$a_off' without)"
    # A ratio, not a bound: each run compiles its uncached file through the
    # front end first, ~0.5M interpreted instructions of its own.
    elif [ -z "$n_on" ] || [ -z "$n_off" ] ||
         [ "$n_off" -lt $((n_on * 4)) ]; then
        note FAIL "$prog: warm-up left $n_on interpreted instructions (without it $n_off)"
    else
        note ok "$prog compiles from the warm-up ($n_on interpreted instructions, $n_off without)"
    fi
done
exit $fail

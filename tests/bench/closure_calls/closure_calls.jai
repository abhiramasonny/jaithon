#: Higher-order calls and captured variables.
#:
#: `fib_recursive` measures a direct call; this measures an indirect one through
#: a closure that reads an upvalue, which is the shape most library code takes.
from std.os import env_or

let LEVEL = env_or("BENCH_LEVEL", "hard")
let SCALE = if LEVEL == "easy" { 16 } elif LEVEL == "medium" { 4 } else { 1 }
let REPS = max(1000 // SCALE, 1)

fn adder(step: int) -> fn(int) -> int {
    return |x| x + step
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
    var k = 1
    while k <= REPS {
        let bump = adder(k)
        total = (total + apply_n(bump, 0, 100_000)) % 1000000007
        k += 1
    }
    print(total)
    return 0
}

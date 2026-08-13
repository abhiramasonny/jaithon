#: Exception machinery — entering handlers that never fire, and raising through
#: them when they do.
#:
#: Holdout. Nothing else in the suite enters a `try`, so nothing else prices
#: what a handler costs when it is not used, which is the case that matters:
#: real code wraps work in `try` and almost never throws. The second phase
#: raises on one iteration in eight, so the unwind path is measured too rather
#: than assumed free.
from std.os import env_or

let LEVEL = env_or("BENCH_LEVEL", "hard")
let SCALE = if LEVEL == "easy" { 16 } elif LEVEL == "medium" { 4 } else { 1 }
let CALM = 8_000_000 // SCALE
let STORM = 160_000 // SCALE

fn checked(x: int) -> int {
    if x % 8 == 3 { throw ValueError(f"bad value {x}") }
    return x + 1
}

#: Handlers pushed and popped eight million times, never entered.
fn calm(n: int) -> int {
    var total = 0
    var i = 0
    while i < n {
        try {
            total += i % 13
        } catch e: ValueError {
            total -= 1
        }
        i += 1
    }
    return total
}

#: One raise in eight, each carrying a formatted message.
fn storm(n: int) -> int {
    var ok = 0
    var bad = 0
    var i = 0
    while i < n {
        try {
            ok += checked(i)
        } catch e: ValueError {
            bad += 1
        }
        i += 1
    }
    return ok - bad
}

fn main() -> int {
    print(calm(max(CALM, 1)))
    print(storm(max(STORM, 1)))
    return 0
}

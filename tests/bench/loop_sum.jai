#: Tight integer loop — measures raw dispatch and local-variable access.
from std.os import env_or

let LEVEL = env_or("BENCH_LEVEL", "hard")
let SCALE = if LEVEL == "easy" { 16 } elif LEVEL == "medium" { 4 } else { 1 }
let ITERS = 50_000_000 // SCALE

fn main() -> int {
    var total = 0
    var i = 0
    while i < ITERS {
        total += i % 7
        i += 1
    }
    print(total)
    return 0
}

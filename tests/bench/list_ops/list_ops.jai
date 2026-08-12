#: List growth, indexing, and higher-order iteration.
from std.os import env_or

let LEVEL = env_or("BENCH_LEVEL", "hard")
let SCALE = if LEVEL == "easy" { 16 } elif LEVEL == "medium" { 4 } else { 1 }
let ITERS = 10_000_000 // SCALE
let PROBE = 999_999 // SCALE

fn main() -> int {
    var xs = []
    for i in 0..ITERS { xs.push(i * 3) }

    var total = 0
    for x in xs { total += x % 11 }

    let doubled = xs.map(|x| x * 2)
    print(total)
    print(doubled[PROBE])
    return 0
}

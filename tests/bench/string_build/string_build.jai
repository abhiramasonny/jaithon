#: String construction and formatting.
from std.os import env_or

let LEVEL = env_or("BENCH_LEVEL", "hard")
let SCALE = if LEVEL == "easy" { 16 } elif LEVEL == "medium" { 4 } else { 1 }
let ITERS = 2_000_000 // SCALE

fn main() -> int {
    var parts = []
    for i in 0..ITERS { parts.push(f"item-{i}") }
    let joined = ",".join(parts)
    print(joined.len())
    print(joined.split(",").len())
    return 0
}

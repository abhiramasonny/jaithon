#: Hash table insert, lookup, and iteration.
from std.os import env_or

let LEVEL = env_or("BENCH_LEVEL", "hard")
let SCALE = if LEVEL == "easy" { 16 } elif LEVEL == "medium" { 4 } else { 1 }
let ITERS = 30_000_000 // SCALE

fn main() -> int {
    var counts = {}
    for i in 0..ITERS {
        let key = f"k{i % 10_000}"
        counts[key] = counts.get(key, 0) + 1
    }

    var total = 0
    for (_key, value) in counts.items() { total += value }
    print(counts.len())
    print(total)
    return 0
}

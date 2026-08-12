#: Iterating a dict by key and value at once — `for (k, v) in d.items()`.
#:
#: The shape is everywhere in real code (counting, summing by key, filtering a
#: map) and it is the one iteration form that could not reach the compiled
#: tier: `for x in xs` fuses to OP_FOR_ITER_BIND, but a destructuring loop puts
#: OP_UNPACK between the FOR_ITER and the BINDs, so the peephole's rule never
#: matched and the whole enclosing function declined.
#:
#: `dict_ops` covers dict insert/lookup; this covers iteration, which is a
#: different path — the iterator builds a tuple per entry.
from std.os import env_or

let LEVEL = env_or("BENCH_LEVEL", "hard")
let SCALE = if LEVEL == "easy" { 20 } elif LEVEL == "medium" { 5 } else { 1 }
let KEYS = 20000
let REPS = max(120 // SCALE, 1)

fn build(n: int) -> dict[str, int] {
    var d: dict[str, int] = {}
    var i = 0
    while i < n {
        d[f"k{i}"] = i
        i += 1
    }
    return d
}

fn sum_values(d: dict[str, int]) -> int {
    var total = 0
    for (_k, v) in d.items() {
        total += v
    }
    return total
}

fn count_long_keys(d: dict[str, int]) -> int {
    var seen = 0
    for (k, v) in d.items() {
        if k.len() > 4 and v % 2 == 0 { seen += 1 }
    }
    return seen
}

fn main() -> int {
    let d = build(KEYS)
    var total = 0
    var seen = 0
    var rep = 0
    while rep < REPS {
        total = sum_values(d)
        seen = count_long_keys(d)
        rep += 1
    }
    print(total)
    print(seen)
    return 0
}

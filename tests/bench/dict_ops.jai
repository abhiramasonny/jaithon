#: Hash table insert, lookup, and iteration.
fn main() -> int {
    var counts = {}
    for i in 0..500_000 {
        let key = f"k{i % 10_000}"
        counts[key] = counts.get(key, 0) + 1
    }

    var total = 0
    for (_key, value) in counts.items() { total += value }
    print(counts.len())
    print(total)
    return 0
}

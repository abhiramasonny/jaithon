#: String construction and formatting.
fn main() -> int {
    var parts = []
    for i in 0..2_000_000 { parts.push(f"item-{i}") }
    let joined = ",".join(parts)
    print(joined.len())
    print(joined.split(",").len())
    return 0
}

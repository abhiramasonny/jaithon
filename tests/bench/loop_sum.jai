#: Tight integer loop — measures raw dispatch and local-variable access.
fn main() -> int {
    var total = 0
    var i = 0
    while i < 50_000_000 {
        total += i % 7
        i += 1
    }
    print(total)
    return 0
}

#: List growth, indexing, and higher-order iteration.
fn main() -> int {
    var xs = []
    for i in 0..1_000_000 { xs.push(i * 3) }

    var total = 0
    for x in xs { total += x % 11 }

    let doubled = xs.map(|x| x * 2)
    print(total)
    print(doubled[999_999])
    return 0
}

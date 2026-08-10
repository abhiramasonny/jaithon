#: Sieve of Eratosthenes over a list of booleans.
#:
#: The inner loop strides rather than walking, so it defeats any prefetch the
#: flat loops get, and every write is a bounds-checked store of a bool. Nothing
#: else in the suite writes a list this hard.
fn main() -> int {
    let n = 1_000_000
    var flags: list[bool] = []
    for _i in 0..n + 1 { flags.push(true) }

    var count = 0
    var sum = 0
    var p = 2
    while p * p <= n {
        if flags[p] {
            var q = p * p
            while q <= n {
                flags[q] = false
                q += p
            }
        }
        p += 1
    }

    var i = 2
    while i <= n {
        if flags[i] {
            count += 1
            sum = (sum + i) % 1000000007
        }
        i += 1
    }
    print(count)
    print(sum)
    return 0
}

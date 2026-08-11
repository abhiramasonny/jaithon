#: Bit twiddling: shifts, xor, and, or, and a popcount written out by hand.
#:
#: The other integer benchmarks are add, multiply and compare. This is the rest
#: of the integer instruction set, and it touches no heap at all — no list, no
#: string, no dict — so a number here is arithmetic and the loops around it and
#: nothing else. Every value is kept inside 32 bits, so the ports cannot differ
#: by an overflow.
fn popcount(value: int) -> int {
    var v = value
    var bits = 0
    while v != 0 {
        v = v & v - 1
        bits += 1
    }
    return bits
}

fn main() -> int {
    let mask = 4294967295
    var seed = 7
    var ones = 0
    var checksum = 0
    var i = 0
    while i < 1_000_000 {
        seed = (seed * 1103515245 + 12345) % 2147483648
        var v = seed ^ checksum
        v = v ^ v >> 7
        v = (v ^ v << 3) & mask
        v = v ^ v >> 11
        let bits = popcount(v)
        ones += bits
        checksum = checksum + v + bits & mask
        checksum = (checksum << 1 | checksum >> 31) & mask
        i += 1
    }
    print(ones)
    print(checksum)
    return 0
}

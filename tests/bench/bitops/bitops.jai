#: Bit twiddling: shifts, xor, and, or, and a popcount written out by hand.
#:
#: The other integer benchmarks are add, multiply and compare. This is the rest
#: of the integer instruction set, and it touches no heap at all — no list, no
#: string, no dict — so a number here is arithmetic and the loops around it and
#: nothing else. Every value is kept inside 32 bits, so the ports cannot differ
#: by an overflow.
#:
#: THE C++ PEER DOES NOT RUN THIS LOOP. Measured 2026-09-07: clang -O2
#: recognises the `v & (v - 1)` idiom and emits `cnt.8b` + `uaddlv.8b`, the ARM
#: hardware population count -- two instructions in place of the whole
#: sixteen-iteration loop. So the 3.7x this benchmark reports is idiom
#: recognition, not codegen quality, and "written out by hand" above does NOT
#: make the ports comparable the way it intends.
#:
#: jaithon's side is close to its floor already: the inner loop runs at
#: 0.592 ns/iteration against an empty loop's 0.577, i.e. the bit twiddling is
#: nearly free and only the loop remains. Removing the two never-taken overflow
#: branches from it measures exactly zero, and so does hand-inlining popcount to
#: delete the call. Do not read this row as a jaithon deficiency to optimise.
from std.os import env_or

let LEVEL = env_or("BENCH_LEVEL", "hard")
let SCALE = if LEVEL == "easy" { 16 } elif LEVEL == "medium" { 4 } else { 1 }
let ITERS = 1_000_000 // SCALE

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
    while i < ITERS {
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

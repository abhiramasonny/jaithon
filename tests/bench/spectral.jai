#: Spectral-norm style float kernel: divides, square roots and a reduction.
#:
#: `nbody` is float-heavy but its inner loop is dominated by field loads.
#: This one holds everything in locals, so what it measures is the arithmetic
#: itself and whether the compiled code keeps values in registers.
from std.os import env_or

# The matrix is n^2 work, so the level cuts the rounds instead of the side.
let LEVEL = env_or("BENCH_LEVEL", "hard")
let SCALE = if LEVEL == "easy" { 16 } elif LEVEL == "medium" { 4 } else { 1 }
let ROUNDS = max(6 // SCALE, 1)

fn evalA(i: int, j: int) -> float {
    let s = i + j
    return 1.0 / float(s * (s + 1) // 2 + i + 1)
}

fn timesA(v: list[float], n: int) -> list[float] {
    var out: list[float] = []
    for i in 0..n {
        var sum = 0.0
        for j in 0..n { sum += evalA(i, j) * v[j] }
        out.push(sum)
    }
    return out
}

fn timesAt(v: list[float], n: int) -> list[float] {
    var out: list[float] = []
    for i in 0..n {
        var sum = 0.0
        for j in 0..n { sum += evalA(j, i) * v[j] }
        out.push(sum)
    }
    return out
}

fn main() -> int {
    let n = 550
    var u: list[float] = []
    for _i in 0..n { u.push(1.0) }

    var v: list[float] = []
    for _r in 0..ROUNDS {
        v = timesAt(timesA(u, n), n)
        u = timesAt(timesA(v, n), n)
    }

    var vBv = 0.0
    var vv = 0.0
    for i in 0..n {
        vBv += u[i] * v[i]
        vv += v[i] * v[i]
    }
    print(n)
    print(f"{(vBv / vv) ** 0.5:.9f}")
    return 0
}

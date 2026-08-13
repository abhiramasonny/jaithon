#: A 2D float stencil — Jacobi relaxation on a fixed-boundary plate.
#:
#: Holdout. `matrix_mul` also indexes nested lists of float, but its inner loop
#: strides down a column; this one reads five neighbours of one cell, three of
#: them from rows the loop is not walking. Two buffers are swapped rather than
#: reallocated, so unlike `life` nothing here is allocation-bound and what is
#: left is the read pattern and the arithmetic.
from std.os import env_or

let LEVEL = env_or("BENCH_LEVEL", "hard")
# Work is n^2 per step, so the level cuts the side and the step count together.
let SHRINK = if LEVEL == "easy" { 4 } elif LEVEL == "medium" { 2 } else { 1 }
let SIDE = 500 // SHRINK
let STEPS = max(60 // SHRINK, 1)

fn plate(n: int) -> list[list[float]] {
    var g: list[list[float]] = []
    for i in 0..n {
        var row: list[float] = []
        for j in 0..n {
            # Fixed boundary: the top edge is hot, the other three are cold.
            if i == 0 { row.push(100.0) } else { row.push(0.0) }
        }
        g.push(row)
    }
    return g
}

fn relax(src: list[list[float]], dst: list[list[float]], n: int) -> void {
    for i in 1..n - 1 {
        let up = src[i - 1]
        let mid = src[i]
        let down = src[i + 1]
        var out = dst[i]
        for j in 1..n - 1 {
            out[j] = 0.25 * (up[j] + down[j] + mid[j - 1] + mid[j + 1])
        }
    }
}

fn main() -> int {
    let n = SIDE
    var a = plate(n)
    var b = plate(n)

    var s = 0
    while s < STEPS {
        relax(a, b, n)
        relax(b, a, n)
        s += 2
    }

    var total = 0.0
    for i in 0..n {
        let row = a[i]
        for j in 0..n { total += row[j] }
    }
    print(n)
    print(f"{total:.6f}")
    return 0
}

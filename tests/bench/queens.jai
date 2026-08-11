#: N-queens by backtracking.
#:
#: Deep recursion whose depth changes constantly, a branch that is taken about
#: half the time, and list mutation that is undone on the way back out.
#: `fib_recursive` recurses on one integer; this recurses carrying state.
from std.os import env_or

# The board is exponential work in n, so the level cuts the repeats instead.
let LEVEL = env_or("BENCH_LEVEL", "hard")
let SCALE = if LEVEL == "easy" { 16 } elif LEVEL == "medium" { 4 } else { 1 }
let REPS = max(3 // SCALE, 1)

fn safe(cols: list[int], row: int, col: int) -> bool {
    var r = 0
    while r < row {
        let c = cols[r]
        if c == col { return false }
        if c - r == col - row { return false }
        if c + r == col + row { return false }
        r += 1
    }
    return true
}

fn place(cols: list[int], row: int, n: int) -> int {
    if row == n { return 1 }
    var found = 0
    var col = 0
    while col < n {
        if safe(cols, row, col) {
            cols[row] = col
            found += place(cols, row + 1, n)
        }
        col += 1
    }
    return found
}

fn main() -> int {
    let n = 11
    var cols: list[int] = []
    for _i in 0..n { cols.push(0) }
    var total = 0
    for _rep in 0..REPS { total = place(cols, 0, n) }
    print(n)
    print(total)
    return 0
}

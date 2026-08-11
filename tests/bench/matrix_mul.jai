#: Dense matrix multiply over nested lists.
#:
#: Two levels of indexing in the innermost loop, and a float multiply-add that
#: nothing can strength-reduce away. `loop_sum` measures a counted loop with one
#: array; this measures what happens when the address of every operand has to be
#: computed from two of them.
fn make(n: int, seed: int) -> list[list[float]] {
    var m: list[list[float]] = []
    var s = seed
    for _i in 0..n {
        var row: list[float] = []
        for _j in 0..n {
            s = (s * 1103515245 + 12345) % 2147483648
            row.push(float(s % 1000) / 1000.0)
        }
        m.push(row)
    }
    return m
}

fn main() -> int {
    let n = 320
    let a = make(n, 12345)
    let b = make(n, 67890)

    var c: list[list[float]] = []
    for i in 0..n {
        var row: list[float] = []
        let ai = a[i]
        for j in 0..n {
            var sum = 0.0
            for k in 0..n {
                sum += ai[k] * b[k][j]
            }
            row.push(sum)
        }
        c.push(row)
    }

    var trace = 0.0
    for i in 0..n { trace += c[i][i] }
    print(n)
    print(f"{trace:.6f}")
    return 0
}

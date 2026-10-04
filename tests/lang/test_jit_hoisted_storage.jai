#: A list's storage proved once where its header is hoisted, for a slot the
#: loop around it rebinds.
#:
#: An OSR form pins a list local's storage at entry only when nothing in the
#: form writes the local. A stencil's row locals are rebound by the outer loop,
#: so the inner loop used to dispatch on `ObjList::stg` at every element --
#: three extra instructions and a branch per read. The header hoist now checks
#: the storage once, above the inner loop it was hoisted out of
#: (JAITHON_JIT_HOIST_PIN), and a list arriving with another storage resumes
#: the loop head in the interpreter.
#:
#: So the rows here alternate between storages part-way through: an int pushed
#: into a `list[float]` (through `any`) de-specialises it to boxed Values for
#: good, and popping it again leaves a boxed list of floats. Every expectation is plain
#: arithmetic the interpreter agrees with (`JAITHON_NO_JIT=1`), and the loops
#: are long enough for the OSR tier to compile them under the default tick.
from std.test import assert_eq

fn rows_of(n: int, width: int, boxed_every: int) -> list[list[float]] {
    var rows: list[list[float]] = []
    for i in 0..n {
        var row: list[float] = []
        for j in 0..width { row.push(float((i * 7 + j * 3) % 11)) }
        # Boxed in place, elements unchanged, so nothing below depends on
        # which rows were boxed -- only the storage the loops meet differs.
        if boxed_every > 0 and i % boxed_every == boxed_every - 1 {
            let one: any = 1
            row.push(one)
            let _gone = row.pop()
        }
        rows.push(row)
    }
    return rows
}

fn row_sums(rows: list[list[float]], n: int, width: int) -> float {
    var s = 0.0
    for i in 0..n {
        let row = rows[i]
        for j in 1..width - 1 { s += row[j - 1] + row[j] * 2.0 + row[j + 1] }
    }
    return s
}

fn stencil(src: list[list[float]], dst: list[list[float]], n: int, width: int) -> void {
    for i in 1..n - 1 {
        let up = src[i - 1]
        let mid = src[i]
        let down = src[i + 1]
        var out = dst[i]
        for j in 1..width - 1 {
            out[j] = 0.25 * (up[j] + down[j] + mid[j - 1] + mid[j + 1])
        }
    }
}

fn reference_sums(rows: list[list[float]], n: int, width: int) -> float {
    var s = 0.0
    var i = 0
    while i < n {
        let row = rows[i]
        var j = 1
        while j < width - 1 {
            let a = row[j - 1]
            let b = row[j]
            let c = row[j + 1]
            s = s + (a + b * 2.0 + c)
            j += 1
        }
        i += 1
    }
    return s
}

fn test_rows_of_mixed_storage_sum_the_same() -> void {
    let n = 300
    let width = 400
    for every in [0, 1, 2, 7, 150] {
        let rows = rows_of(n, width, every)
        let want = reference_sums(rows, n, width)
        for _round in 0..10 {
            assert_eq(row_sums(rows, n, width), want, f"boxed every {every}")
        }
    }
}

fn test_a_stencil_whose_rows_change_storage() -> void {
    let n = 200
    let width = 300
    let plain = rows_of(n, width, 0)
    let mixed = rows_of(n, width, 3)
    var a = rows_of(n, width, 0)
    var b = rows_of(n, width, 5)
    for _step in 0..8 {
        stencil(plain, a, n, width)
        stencil(mixed, b, n, width)
    }
    # The same stencil from the interpreter's side, spelled with while loops
    # so neither result came from the form under test.
    var wantA = rows_of(n, width, 0)
    var wantB = rows_of(n, width, 5)
    for _step in 0..8 {
        var i = 1
        while i < n - 1 {
            var j = 1
            while j < width - 1 {
                wantA[i][j] = 0.25 *
                    (plain[i - 1][j] + plain[i + 1][j] + plain[i][j - 1] + plain[i][j + 1])
                wantB[i][j] = 0.25 *
                    (mixed[i - 1][j] + mixed[i + 1][j] + mixed[i][j - 1] + mixed[i][j + 1])
                j += 1
            }
            i += 1
        }
    }
    for i in 0..n {
        for j in 0..width {
            assert_eq(a[i][j], wantA[i][j], "unboxed rows")
            assert_eq(b[i][j], wantB[i][j], "rows with boxed ones among them")
        }
    }
}

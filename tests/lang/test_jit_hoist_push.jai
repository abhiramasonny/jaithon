#: A loop that pushes can still hoist the headers of the lists it reads.
#:
#: The push's grow stub is the one call out of such a loop, and it runs no user
#: code: it keeps the registers, and a hoist over it must prove the list
#: pushed to is another (JAITHON_JIT_GROW_KEEPS). So the shapes here are the ones where a header
#: kept across a push would be wrong: the list pushed to IS a hoisted one, by
#: the same local or through another, and it grows -- its items move -- in the
#: middle of the loop. Every expectation is the interpreter's
#: (`JAITHON_NO_JIT=1`), and the loops are long enough to grow many times.
from std.test import assert_eq

fn read_and_push_same(xs: list[int], n: int) -> int {
    var s = 0
    for i in 0..n {
        s += xs[i]
        xs.push(i % 7)
    }
    return s
}

fn read_and_push_alias(xs: list[int], ys: list[int], n: int) -> int {
    var s = 0
    for i in 0..n {
        s += xs[i] * 2 + xs[i + 1]
        ys.push(xs[i])
    }
    return s
}

fn store_read_push(xs: list[float], n: int) -> float {
    var s = 0.0
    for i in 0..n {
        xs[i] = xs[i] * 0.5
        s += xs[i]
        xs.push(1.0)
    }
    return s
}

fn stencil_rows(rows: list[list[int]], h: int, w: int) -> list[list[int]] {
    var out: list[list[int]] = []
    for r in 1..h - 1 {
        let up = rows[r - 1]
        let mid = rows[r]
        let down = rows[r + 1]
        var row: list[int] = []
        for c in 1..w - 1 {
            row.push(up[c] + mid[c - 1] + mid[c + 1] + down[c])
        }
        out.push(row)
    }
    return out
}

fn reference_sum(xs: list[int], n: int) -> int {
    var s = 0
    var i = 0
    while i < n {
        s = s + xs[i]
        i += 1
    }
    return s
}

fn test_a_list_that_grows_under_its_own_hoisted_header() -> void {
    for size in [10, 1000, 50000, 50000] {
        var xs: list[int] = []
        for k in 0..size { xs.push(k % 13) }
        let want = reference_sum(xs, size)
        assert_eq(read_and_push_same(xs, size), want, "same local")
        assert_eq(xs.len(), size * 2, "pushed every time")
        assert_eq(xs[size + 9], 9 % 7, "the pushed values landed")
    }
}

fn test_a_list_that_grows_through_another_local() -> void {
    for size in [10, 1000, 50000, 50000] {
        var xs: list[int] = []
        for k in 0..size + 1 { xs.push(k % 11) }
        var want = 0
        for i in 0..size { want += xs[i] * 2 + xs[i + 1] }
        assert_eq(read_and_push_alias(xs, xs, size), want, "aliased")
        var other: list[int] = []
        var xs2: list[int] = []
        for k in 0..size + 1 { xs2.push(k % 11) }
        assert_eq(read_and_push_alias(xs2, other, size), want, "distinct")
        assert_eq(other.len(), size, "the other list grew")
    }
}

fn test_stores_reads_and_pushes_on_one_list() -> void {
    for size in [10, 1000, 50000] {
        var xs: list[float] = []
        for k in 0..size { xs.push(float(k % 5)) }
        var want = 0.0
        for k in 0..size { want += float(k % 5) * 0.5 }
        assert_eq(store_read_push(xs, size), want, "one list")
        assert_eq(xs.len(), size * 2, "grew")
    }
}

fn test_a_stencil_that_builds_its_output_by_pushing() -> void {
    var rows: list[list[int]] = []
    for r in 0..120 {
        var row: list[int] = []
        for c in 0..150 { row.push((r * 3 + c * 5) % 9) }
        rows.push(row)
    }
    for _round in 0..20 {
        let out = stencil_rows(rows, 120, 150)
        var t = 0
        for row in out {
            for v in row { t += v }
        }
        var want = 0
        var r = 1
        while r < 119 {
            var c = 1
            while c < 149 {
                want = want + rows[r - 1][c] + rows[r][c - 1] + rows[r][c + 1] + rows[r + 1][c]
                c += 1
            }
            r += 1
        }
        assert_eq(t, want, "stencil")
    }
}

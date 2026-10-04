#: Loops that read some lists and append to another, compiled. The two tiers
#: must agree exactly.
#:
#: An append's grow stub used to count as a call, so a loop holding a `push`
#: hoisted no list header and pinned no storage. It now keeps every
#: caller-saved register (JAITHON_JIT_GROW_KEEPS), so those loops hoist -- and
#: a hoisted header is only sound while nothing moves that list. What this file
#: pins down is every way an append could be to a list a header was hoisted
#: from, and every way a storage proved at a hoist could be wrong:
#:
#: - a stencil over push-built rows, appending to a new row (the common case);
#: - the same with rows of two different storages, so the storage the hoist
#:   proves is refuted on some entries and the loop must deoptimise;
#: - the appended-to list IS a hoisted one under another name, which the
#:   pointer compare at the hoist must catch (a stale `items` here reads freed
#:   memory, not merely a wrong answer);
#: - an append to the very slot being read;
#: - an OP_ELEM_KIND stamping an empty list inside the loop;
#: - floats and many ints live across an append that grows, which the stub
#:   must preserve (realloc's memcpy uses the vector registers).
#:
#: Every loop runs long enough for the sampler to give it a compiled form, and
#: every function is called more than the function tier's threshold.

fn row_ints(n: int, seed: int) -> list[int] {
    var row: list[int] = []
    var s = seed
    for _i in 0..n {
        s = (s * 1103515245 + 12345) % 2147483648
        row.push(s // 65536 % 3)
    }
    return row
}

fn row_boxed(n: int, seed: int) -> list[int] {
    let ints = row_ints(n, seed)
    var row: list[int] = [0, 0]
    row.clear()
    for v in ints { row.push(v) }
    return row
}

fn stencil(rows: list[list[int]], w: int) -> int {
    var total = 0
    for r in 1..rows.len() - 1 {
        let up = rows[r - 1]
        let mid = rows[r]
        let down = rows[r + 1]
        var out: list[int] = []
        for c in 1..w - 1 {
            let n = up[c - 1] + up[c] + up[c + 1] + mid[c - 1] + mid[c + 1] +
                down[c - 1] + down[c] + down[c + 1]
            out.push(n % 5 + mid[c])
        }
        for v in out { total += v }
    }
    return total
}

fn alias_push(src: list[int], n: int) -> int {
    var out = src
    var t = 0
    for i in 0..n {
        t += src[i]
        out.push(src[i] + 1)
    }
    return t + out.len() + src.len()
}

fn self_push(n: int) -> int {
    var xs: list[int] = [1, 2]
    for i in 0..n { xs.push(xs[i] + xs[i + 1] % 7) }
    return xs[n] + xs.len()
}

fn stamp_push(n: int) -> int {
    var a = []
    var t = 0
    for i in 0..n {
        var b: list[int] = a
        a.push(i)
        t += b.len()
    }
    return t + a[n - 1]
}

fn floats_across(n: int) -> float {
    var ys: list[float] = []
    var acc = 0.5
    for i in 0..n {
        let x = float(i) * 0.25
        acc = acc * 0.5 + x * 1.5 + float(ys.push(x).len()) * 0.001
    }
    return acc + ys[n - 1]
}

fn ints_across(xs: list[int], n: int) -> int {
    var out: list[int] = []
    var t = 0
    for i in 0..n {
        let a = xs[i]
        let b = xs[i + 1]
        t = (t + a * 3 + b * 5 + (a - b) * (out.push(a + b).len() % 11)) % 1000003
    }
    return t + out.len()
}

fn main() -> int {
    let w = 400
    var same: list[list[int]] = []
    var mixed: list[list[int]] = []
    for r in 0..60 {
        same.push(row_ints(w, r + 1))
        if r % 3 == 1 {
            mixed.push(row_boxed(w, r + 1))
        } else {
            mixed.push(row_ints(w, r + 1))
        }
    }
    var s1 = 0
    var s2 = 0
    for _k in 0..70 {
        s1 = stencil(same, w)
        s2 = stencil(mixed, w)
    }
    print(s1)
    print(s2)

    var a1 = 0
    for k in 0..70 { a1 = alias_push(row_ints(2000, k), 1500) }
    print(a1)

    var p1 = 0
    for _k in 0..70 { p1 = self_push(3000) }
    print(p1)

    var q1 = 0
    for _k in 0..70 { q1 = stamp_push(1500) }
    print(q1)

    var f1 = 0.0
    for _k in 0..70 { f1 = floats_across(3000) }
    print(f1)

    var i1 = 0
    let src = row_ints(5001, 9)
    for _k in 0..70 { i1 = ints_across(src, 5000) }
    print(i1)
    return 0
}

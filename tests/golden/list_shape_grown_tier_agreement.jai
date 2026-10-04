#: Untyped lists that take unboxed storage when they grow past eight elements
#: of one kind (JAITHON_LIST_SHAPE_GROWN). Every tier must agree exactly.
#:
#: A loop form pins the storage of the lists in its locals once, at entry. A
#: short boxed list can still be unboxed inside the region, by an append
#: through another name that grows it past eight, so no form may pin one: not
#: when it is compiled, and not when a form compiled over a long boxed list is
#: entered with a short one. `pinned_then_short` is that second case; reading
#: `a[1]` at the boxed stride after the alias unboxed it returned `a[3]`.
#:
#: Concatenating a typed list with an untyped one of the same storage must not
#: hand the result the typed list's promise, and records built element by
#: element with mixed kinds must stay as they were.
fn pinned_then_short(a: list, n: int, gate: int) -> int {
    var s = 0
    for i in 0..n {
        var b = a
        if i >= gate { if b.len() == 2 { b.push(2) } else { b.push(i) } }
        if a.len() > 3 { s += a[1] }
    }
    return s
}

fn record(i: int) -> list {
    var r = []
    r.push(i)
    r.push("k")
    r.push(i * 2)
    return r
}

fn grown(n: int, last: any) -> list {
    var xs = []
    for k in 0..n { xs.push(k) }
    xs.push(last)
    return xs
}

fn main() -> int {
    print(pinned_then_short([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 300000, 1000000000))
    var x = []
    print(pinned_then_short(x, 2000000, 1000000))
    print(x.len())

    var a: list[int] = []
    a.push(1)
    var b = []
    for k in 0..12 { b.push(k) }
    var c = a + b
    c.push("s")
    print(c)
    var d: list[float] = []
    d.push(0.5)
    var e = []
    for k in 0..10 { e.push(1.5) }
    var g = d + e
    g.push("x")
    g[0] = "u"
    print(g)
    var h = b + b
    h.push("t")
    print(h)

    var t = 0
    for i in 0..200000 {
        let r = record(i)
        t += r[0] + r[2]
    }
    print(t)
    print(record(7))

    print(grown(8, 2.5))
    print(grown(9, "z"))
    print(grown(12, true))
    var fs = []
    for k in 0..10 { fs.push(float(k) / 4.0) }
    fs.push(3)
    print(fs)
    var bs = []
    for k in 0..11 { bs.push(k % 3 == 0) }
    bs[2] = 7
    print(bs)
    var ys = grown(20, 21)
    ys[0] = "first"
    print(ys.pop())
    print(ys[0])
    print(ys.len())
    return 0
}

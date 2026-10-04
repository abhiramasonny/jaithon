#: list.map and list.filter over compiled lambdas, and untyped lists shaped by
#: their first element. Every tier must agree exactly.
#:
#: map and filter drive a compiled callee over a whole run of the list in one
#: native call (jaiMapPreparedFn1Run, jaiFilterPreparedFn1): the window and the
#: staleness test are hoisted, and a run stops at the first element it cannot
#: take, which the native's ordinary loop then takes. This pins every way a run
#: can stop -- a callee that has not compiled yet, an element of another kind
#: in the middle, a callee that raises part-way, one that returns another kind,
#: one that shrinks the very list being mapped -- for int, float and bool
#: results.
#:
#: An untyped `[]` takes unboxed storage when a push grows it past eight
#: elements of one kind (JAITHON_LIST_SHAPE_GROWN), in the interpreter's push
#: and in the compiled one alike; a later element of another kind
#: de-specialises it. None of that may be visible.
var SRC: list = []

fn shrink(x: int) -> int {
    if x % 4 == 0 { SRC.pop() }
    return x + 1
}

fn shrink_pred(x: int) -> bool {
    if x % 5 == 0 { SRC.pop() }
    return x % 2 == 0
}

fn ints(n: int) -> list[int] {
    var xs: list[int] = []
    for i in 0..n { xs.push(i * 7 - 300) }
    return xs
}

fn untyped(n: int) -> list {
    var xs = []
    for i in 0..n { xs.push(i * 3) }
    return xs
}

fn floats(n: int) -> list[float] {
    var xs: list[float] = []
    for i in 0..n { xs.push(float(i) / 7.0 - 3.0) }
    return xs
}

fn sq(x: float) -> float { return x * x }

fn half(x: int) -> int {
    if x > 4000 { return x // 2 }
    return x * 2
}

fn mixed(n: int) -> list {
    var xs = []
    for i in 0..n {
        if i % 997 == 0 { xs.push(1.5) } else { xs.push(i) }
    }
    return xs
}

fn main() -> int {
    let a = ints(5000)
    let b = untyped(5000)
    let f = floats(4000)
    let m = mixed(3000)
    var t = 0
    var ft = 0.0
    var c = 0
    for r in 0..40 {
        let m1 = a.map(|x| x * 2 + 1)
        let m2 = b.map(|x| x - 3)
        let m3 = a.map(half)
        let m4 = f.map(|x| x * 0.5 + 1.0)
        let m5 = f.map(sq)
        let m6 = a.map(|x| x > 0)
        let m7 = m.map(|x| x + 1)
        let k1 = a.filter(|x| x % 3 == 0)
        let k2 = f.filter(|x| x < 0.0)
        let k3 = m.filter(|x| x > 100)
        t += m1[r] + m2[4999 - r] + m3[r * 100] + m6.len() + k1.len() + k3.len()
        ft += m4[3999] + m5[1000] + float(m7[997]) + float(k2.len())
        for v in m6 { if v { c += 1 } }
    }
    print(t)
    print(ft)
    print(c)

    var caught = 0
    for r in 0..100 {
        try {
            let z = b.map(|x| 100 // (x - 3 * (4000 + r)))
            print(z.len())
        } catch _e {
            caught += 1
        }
        try {
            let z = b.filter(|x| 100 // (x - 3 * (4000 + r)) > 0)
            print(z.len())
        } catch _e {
            caught += 1
        }
    }
    print(caught)

    var odd = 0
    for r in 0..30 {
        let z = a.map(|x| if x == 3000 + r { "s" } else { x })
        odd += z.len()
        if r == 29 { print(z[0]) }
    }
    print(odd)

    for r in 0..60 {
        SRC = untyped(300)
        let g = SRC.map(shrink)
        SRC = untyped(300)
        let h = SRC.filter(shrink_pred)
        if r % 20 == 0 {
            print(g.len())
            print(g[g.len() - 1])
            print(h.len())
        }
    }

    var u = []
    for i in 0..10 { u.push(i) }
    u.push("tail")
    print(u)
    var e = []
    print(e)
    e.push(true)
    e.push(2)
    e.push(2.5)
    print(e)
    let shaped = untyped(6)
    print(shaped == [0, 3, 6, 9, 12, 15])
    var s = untyped(8)
    s.reverse()
    print(s)
    s.sort()
    print(s)
    print(s[1:4])
    return 0
}

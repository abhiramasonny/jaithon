#: The arms compiled code keeps out of line, taken. The tiers must agree.
#:
#: Three corrections that are rare by design now wait after the body behind a
#: branch the common case does not take: the floor fix of `x % k` / `x // k`
#: for a negative dividend (emitColdFixup), the negative-index arm of a bounds
#: check (emitBoundsNormalise), and the boxed arm of an element read that
#: dispatches on storage (OP_GET_INDEX). Each leaves the body and comes back,
#: or deoptimises from a record taken at the instruction. This takes every one
#: of them -- negative dividends and indices, an index past the end that must
#: raise, a boxed list where an unboxed one was predicted, and an element of
#: the wrong kind in it -- in bodies hot enough to compile.
fn mods(n: int) -> int {
    var h = 0
    var i = -n
    while i < n {
        h = (h * 31 + i % 7 + i // 9 + i % 1000 + i // 1000) % 1000003
        i += 1
    }
    return h
}

fn neg(xs: list[int], fs: list[float], bs: list[bool], n: int) -> float {
    var t = 0
    var f = 0.0
    var c = 0
    for i in 0..n {
        t += xs[i - n]
        f += fs[i - n] * 0.5
        if bs[i - n] { c += 1 }
    }
    return float(t) + f + float(c)
}

fn chars(s: str, n: int) -> int {
    var c = 0
    for i in 0..n {
        if s[i - n] == "a" { c += 1 }
        if s[i] == "b" { c += 2 }
    }
    return c
}

fn total(xs: list[int], n: int) -> int {
    var t = 0
    var i = 0
    while i < n {
        t += xs[i]
        i += 1
    }
    return t
}

fn ftotal(xs: list, n: int) -> float {
    var t = 0.0
    var i = 0
    while i < n {
        t += xs[i]
        i += 1
    }
    return t
}

fn main() -> int {
    var m = 0
    for _k in 0..40 { m = mods(20000) }
    print(m)

    var xs: list[int] = []
    var fs: list[float] = []
    var bs: list[bool] = []
    for i in 0..3000 {
        xs.push(i * 5 % 17)
        fs.push(float(i) / 7.0)
        bs.push(i % 5 == 1)
    }
    var t = 0.0
    for _k in 0..300 { t = neg(xs, fs, bs, 3000) }
    print(t)
    var bad = 0.0
    try { bad = neg(xs, fs, bs, 3001) } catch _e { bad = -1.0 }
    print(bad)

    var parts: list[str] = []
    for i in 0..300 { parts.push("ab" + str(i % 10)) }
    let s = "".join(parts)
    var cc = 0
    for _k in 0..200 { cc += chars(s, s.len()) }
    print(cc)
    var badc = 0
    try { badc = chars(s, s.len() + 1) } catch _e { badc = -1 }
    print(badc)

    var pushed: list[int] = []
    for i in 0..300 { pushed.push(i * 3) }
    var lit: list[int] = [5, 6, 7, 8, 9, 10, 11, 12]
    var untyped = []
    untyped.push("s")
    untyped.pop()
    for i in 0..300 { untyped.push(i) }
    var fl: list[float] = []
    for i in 0..300 { fl.push(float(i) * 0.5) }
    var mix = []
    mix.push("s")
    mix.pop()
    for i in 0..300 { mix.push(float(i)) }
    var tt = 0
    var ff = 0.0
    for k in 0..2000 {
        tt += total(pushed, 300) + total(lit, 8) + total(untyped, 300)
        ff += ftotal(fl, 300) + ftotal(mix, 300)
        if k == 1500 { mix[150] = 3 }
    }
    print(tt)
    print(ff)
    return 0
}

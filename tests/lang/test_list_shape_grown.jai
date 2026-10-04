#: An untyped list that grows past eight elements of one kind takes unboxed
#: storage (JAITHON_LIST_SHAPE_GROWN). That makes a list's storage something an
#: append through ANY name can change, and each test here is a program that
#: answered wrongly, or raised, when some part of the tier or the runtime still
#: believed an untyped list stays boxed:
#:
#:   - a loop form pins the storage of the lists in its locals once, at entry.
#:     A form compiled while `a` held a long boxed list and then entered with
#:     a short one must not keep the BOXED pin, or an append through the alias
#:     `b` shapes the list mid-loop and every later `a[1]` is read at the boxed
#:     stride (it returned `a[3]`);
#:   - `typed + untyped` with the same storage must not hand the result the
#:     typed list's promise, or pushing a string onto it raised TypeError;
#:   - a record built element by element with mixed kinds is not unboxed, and
#:     neither is a long one whose body appends ints and then an object.
#:
#: Run with JAITHON_NO_JIT=1, JAITHON_JIT_THRESHOLD=1 and JAITHON_JIT_TICK_US=50
#: as well: none of the answers may move.
from std.test import assert_eq

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

fn long_record(i: int) -> list {
    var r = []
    for k in 0..10 { r.push(i + k) }
    r.push("end")
    return r
}

pub fn test_a_form_pinned_boxed_is_not_entered_with_a_short_list() -> void {
    #: Long and boxed first, for long enough that the sampler compiles the
    #: loop with `a` pinned BOXED.
    let long = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    assert_eq(pinned_then_short(long, 300000, 1000000000), 6 * 300000)
    #: Then a fresh list. The first `gate` iterations push nothing, which is
    #: where the form is entered; the appends after it grow `x` past eight.
    #: From the fourth one on, a[1] is the second value pushed: gate + 1.
    var x = []
    let gate = 200000
    let n = 400000
    assert_eq(pinned_then_short(x, n, gate), (gate + 1) * (n - gate - 3))
    assert_eq(x.len(), n - gate)
    assert_eq(x[1], gate + 1)
    assert_eq(x[2], 2)
}

pub fn test_typed_plus_untyped_stays_untyped() -> void {
    var a: list[int] = []
    a.push(1)
    var b = []
    for k in 0..12 { b.push(k) }
    var c = a + b
    c.push("s")
    assert_eq(c.len(), 14)
    assert_eq(c[13], "s")
    assert_eq(c[12], 11)

    var d: list[float] = []
    d.push(0.5)
    var e = []
    for k in 0..10 { e.push(1.5) }
    var g = d + e
    g.push("x")
    g[0] = "u"
    assert_eq(g[0], "u")
    assert_eq(g[11], "x")
    assert_eq(g[1], 1.5)

    var h = b + b
    h.push("t")
    assert_eq(h.len(), 25)
    assert_eq(h[24], "t")
}

pub fn test_records_built_by_push_keep_their_kinds() -> void {
    var t = 0
    for i in 0..20000 {
        let r = record(i)
        t += r[0] + r[2]
    }
    assert_eq(t, 3 * (19999 * 20000 / 2))
    assert_eq(record(7), [7, "k", 14])

    var lt = 0
    for i in 0..5000 {
        let r = long_record(i)
        lt += r[0] + r[9]
    }
    assert_eq(lt, 2 * (4999 * 5000 / 2) + 9 * 5000)
    assert_eq(long_record(3), [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, "end"])
}

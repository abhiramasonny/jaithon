#: Guards the compiled tier proves once per straight line instead of at every
#: read.
#:
#: Two memos, both retired at a join, at a call out, and at a write of what
#: they describe (src/vm/jit/jit_func.c):
#:
#: - A DYNAMIC local -- one slot the bytecode reuses for two kinds, as
#:   `for i in 0..n` then `for b in bodies` does -- has its tag and class
#:   guarded at the first read on an edge and not again
#:   (JAITHON_JIT_DYN_MEMO).
#: - A field read whose tag guard passed is remembered against the local it was
#:   read through, so the next read of it skips the guard and a store of the
#:   same kind skips writing the tag (JAITHON_JIT_FIELD_READ_MEMO).
#:
#: Every shape here is one where a memo kept too long would read a value at the
#: wrong kind. The expectations are the interpreter's (`JAITHON_NO_JIT=1`).
from std.test import assert_eq

#: A field of no declared kind, so it can really change kind at run time; a
#: `float` field widens an int stored into it, and would hide the mistake.
class Q {
    pub var x: any
    pub var y: float

    fn init(self, x: any, y: float) {
        self.x = x
        self.y = y
    }
}

class P {
    pub var x: float
    pub var y: float
    pub var w: float

    fn init(self, x: float, y: float, w: float) {
        self.x = x
        self.y = y
        self.w = w
    }
}

#: The slot of `i` is reused by `p`, so it is dynamic; the second loop reads
#: `p` six times an iteration.
fn reuse_slot(ps: list[P], n: int) -> float {
    var total = 0.0
    for i in 0..n { total += float(i) }
    for p in ps {
        p.x += p.w * p.y
        p.y += p.w * p.x
        total += p.x + p.y
    }
    return total
}

fn reference_reuse(ps: list[P], n: int) -> float {
    var total = 0.0
    var k = 0
    while k < n {
        total = total + float(k)
        k += 1
    }
    var j = 0
    while j < ps.len() {
        let q = ps[j]
        q.x = q.x + q.w * q.y
        q.y = q.y + q.w * q.x
        total = total + (q.x + q.y)
        j += 1
    }
    return total
}

fn make(n: int) -> list[P] {
    var ps: list[P] = []
    for i in 0..n { ps.push(P(float(i) * 0.5, 1.0, 0.001 * float(i % 5))) }
    return ps
}

fn test_a_reused_slot_reads_the_same_values() -> void {
    var a = make(50)
    var b = make(50)
    for _round in 0..300 {
        assert_eq(reuse_slot(a, 20), reference_reuse(b, 20), "dynamic slot")
    }
}

#: An int stored through an alias changes the field's kind under a local that
#: has just read it as a float. A read memo that survived the store would read
#: the int's bits as a float.
fn alias_store(a: Q, b: Q) -> float {
    let first = a.x + a.y
    b.x = 3
    return first + a.x * 2.0
}

fn test_a_store_through_an_alias_retires_the_memo() -> void {
    for _round in 0..300 {
        let p = Q(1.5, 2.0)
        assert_eq(alias_store(p, p), 1.5 + 2.0 + 3.0 * 2.0, "aliased store")
        let q = Q(1.5, 2.0)
        let r = Q(0.5, 0.0)
        assert_eq(alias_store(q, r), 1.5 + 2.0 + 1.5 * 2.0, "distinct objects")
    }
}

#: The receiver of a compound assignment is pushed before the right-hand side
#: runs; when that side rebinds the local, the store goes to the OLD object, and
#: nothing may be remembered about the new one.
fn rebind_mid_store(a: Q, b: Q, flag: bool) -> float {
    var p = a
    p.y += (if flag {
        p = b
        1.0
    } else { 2.0 })
    return p.x + a.x + p.y
}

fn test_a_receiver_rebound_mid_store() -> void {
    for round in 0..300 {
        let a = Q(1.0, 10.0)
        let b = Q(5, 20.0)
        let flag = round % 2 == 0
        let got = rebind_mid_store(a, b, flag)
        # flag: the store lands on `a` (y 11.0) and `p` is `b`, an int x:
        # 5 + 1.0 + 20.0. Otherwise `p` stays `a`: 1.0 + 1.0 + 12.0.
        assert_eq(got, if flag { 26.0 } else { 14.0 }, "rebound receiver")
    }
}

#: A closure that rebinds a captured dynamic local between two reads of it on
#: the same line: the call retires the proof.
fn closure_rebinds(items: list[P], n: int) -> float {
    var acc = 0
    for i in 0..n { acc += i }
    var total = float(acc)
    for item in items {
        var cur = item
        let other = P(100.0, 0.0, 0.0)
        let swap = fn() -> float {
            cur = other
            return 1.0
        }
        total += cur.x + swap() + cur.x
    }
    return total
}

fn test_a_closure_rebinding_a_captured_local() -> void {
    let items = [P(1.0, 1.0, 1.0), P(2.0, 2.0, 2.0)]
    for _round in 0..200 {
        # Per item: its own x, then the call, then the OTHER object's x.
        assert_eq(closure_rebinds(items, 10), 45.0 + (1.0 + 1.0 + 100.0) + (2.0 + 1.0 + 100.0), "closure write seen")
    }
}

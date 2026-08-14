#: One program per optimiser pass, written so the *checker* rewrites nothing.
#:
#: `make bootstrap` compares the two front ends opcode for opcode, but almost
#: every file in the tree diverges before the optimiser is reached: the C front
#: end has a type checker and `lib/jaithon/compile` does not, so a `const` is
#: inlined on one side and loaded on the other, an inferred `int` picks up an
#: `OP_TYPE_GUARD` the self-hosted emitter never writes, and the guard node
#: wrapping a returned call suppresses the tail call that would otherwise have
#: stopped `OP_INVOKE` from fusing. Those divergences are real and they are the
#: checker's, but they land *first*, and the comparison stops at the first one —
#: so they hide whatever the optimiser did behind them.
#:
#: This file removes them. Every binding that would otherwise be inferred to a
#: concrete type is annotated `any`, there is no `const`, and no function
#: declares a return type. The checker then inserts no cast, folds nothing and
#: leaves the tail calls alone, so the only thing left that can make the two
#: front ends disagree is `lib/jaithon/compile/opt` against
#: `src/codegen/optimize.c`. Between them the functions below reach every rule
#: of all five passes; the file is expected to read `ok` in `make bootstrap`,
#: and a `DIFF` on it names an optimiser divergence and nothing else.
#:
#: It is a golden test as well as a bootstrap input, so the driver runs it at
#: -O0, -O1, -O2 and -O3 and requires the same output from each: byte identity
#: is worth nothing if the bytes are wrong.

# --- pass 6, superinstructions ----------------------------------------------
fn arith(a, b) {
    var t: any = a + b
    t = t + 1
    t += 1
    t -= 1
    var u: any = a
    u = u + a
    var n: any = 0
    while n < 7 { n = n + 1 }
    while a < b { a = a + 1 }
    if a == b { t = t + 2 }
    if a != b { t = t + 3 }
    return t + n + u
}

fn iterate(xs) {
    var total: any = 0
    for x in xs { total = total + x }
    var k: any = 0
    loop {
        k = k + 1
        if k > 3 { break }
    }
    return total + k
}

fn fields(p) {
    var s: any = p.x + p.y
    p.x = s
    p.y = p.y + 1
    return s
}

fn under_three(x) {
    var y: any = x
    return y < 3
}

# --- pass 3, peephole -------------------------------------------------------
fn branches(a, b) {
    var r: any = 0
    if a { r = 1 } else { r = 2 }
    if not a { r = r + 1 }
    if true { r = r + 10 }
    if false { r = r + 100 }
    while true {
        r = r + 1
        if r > 5 { break }
    }
    if a and b { r = r + 1 }
    if a or b { r = r + 1 }
    return r
}

# --- pass 2, dead code ------------------------------------------------------
fn returns_always(a) { return a }

fn falls_off(a) {
    if a { return 1 }
}

fn dead_try(a) {
    if false {
        try {
            a = a + 1
        } catch _e: ValueError {
            a = 0
        }
    }
    return a
}

fn live_try(a) {
    try {
        a = a + 1
    } catch _e: ValueError {
        a = 0
    } finally {
        a = a + 2
    }
    return a
}

fn with_defer(a) {
    var seen: any = a
    defer { seen = 0 }
    return seen
}

# --- pass 4, slot coalescing ------------------------------------------------
fn chain(a) {
    let p: any = a + 1
    let q: any = p + 1
    let r: any = q + 1
    let s: any = r + 1
    return s
}

fn tenants(a) {
    var keep: any = a
    let b1: any = a + 1
    let b2: any = b1 + keep
    let b3: any = b2 + 1
    let b4: any = b3 + 1
    return keep + b4
}

fn captures(a) {
    let x: any = a + 1
    let f: any = fn() { return x }
    let y: any = a + 2
    return f() + y
}

fn guarded(a) {
    let m: any = a + 1
    try {
        let n: any = m + 1
        return n
    } catch _e: ValueError {
        return 0
    }
}

fn loop_carried(xs) {
    let pre: any = 1
    let pre2: any = pre + 1
    var acc: any = pre2
    for v in xs {
        let inner: any = v + 1
        acc = acc + inner
    }
    return acc
}

# --- pass 7, global hoisting ------------------------------------------------
var g1: any = 1
var g2: any = 2
var g3: any = 3
var g4: any = 4
var g5: any = 5
var g6: any = 6
var g7: any = 7
var g8: any = 8
var g9: any = 9

fn many(n) {
    var t: any = 0
    while n > 0 {
        t = t + g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8 + g9
        n = n - 1
    }
    return t
}

fn blocked(n) {
    var t: any = 0
    while n > 0 {
        t = t + g1 + many(0)
        n = n - 1
    }
    return t
}

fn siblings(n) {
    var t: any = 0
    var i: any = n
    while i > 0 {
        t = t + g1
        i = i - 1
    }
    var j: any = n
    while j > 0 {
        t = t + g2
        j = j - 1
    }
    return t
}

fn nested(n) {
    var t: any = 0
    var i: any = n
    while i > 0 {
        var j: any = n
        while j > 0 {
            t = t + g1
            j = j - 1
        }
        i = i - 1
    }
    return t
}

fn writes(n) {
    var t: any = 0
    while n > 0 {
        g1 = g1 + 1
        t = t + g1
        n = n - 1
    }
    return t
}

# --- shapes the passes have to leave alone ----------------------------------
class Point {
    pub var x: any
    pub var y: any

    pub fn init(self, x, y) {
        self.x = x
        self.y = y
    }

    pub fn norm(self) { return self.x * self.x + self.y * self.y }

    pub fn bump(self, by = 1) {
        self.x = self.x + by
        return self.x
    }
}

fn matcher(v) {
    return match v {
        0 => 10,
        1 => 20,
        _ => 30,
    }
}

fn nested_loops(xs) {
    var t: any = 0
    for a in xs {
        for b in xs {
            if b > 2 { continue }
            if a > 5 { break }
            t = t + b
        }
    }
    return t
}

fn defaulted(a, b = 2, c = 3) { return a + b + c }

print(arith(1, 4))
print(under_three(1))
print(under_three(9))
print(iterate([1, 2, 3]))
let point: any = Point(2, 3)
print(fields(point))
print(point.norm())
print(point.bump())
print(point.bump(5))
print(branches(true, false))
print(returns_always(9))
print(falls_off(true))
print(falls_off(false))
print(dead_try(1))
print(live_try(1))
print(with_defer(4))
print(chain(1))
print(tenants(1))
print(captures(1))
print(guarded(1))
print(loop_carried([1, 2, 3]))
print(many(2))
print(blocked(2))
print(siblings(2))
print(nested(2))
print(writes(2))
print(matcher(1))
print(nested_loops([1, 2, 3]))
print(defaulted(1))
print(defaulted(1, 20))
print(defaulted(1, 20, 300))
print("done")

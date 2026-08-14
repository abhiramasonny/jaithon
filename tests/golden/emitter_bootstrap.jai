#: One use of every construct the emitter lowers, written so the *checker*
#: rewrites nothing.
#:
#: The companion of `optimiser_bootstrap.jai`, and the same trick: annotate
#: `any` everywhere a binding would otherwise be inferred to a concrete type,
#: declare no `const` and no return types, and the C checker inserts no
#: `OP_TYPE_GUARD`, folds nothing, and leaves every tail call alone. What
#: `make bootstrap` compares on this file is `lib/jaithon/compile/emit.jai`
#: against `src/codegen/`, with the type checker out of the way — and it is the
#: only file in the tree where that comparison is visible, because everywhere
#: else a checker divergence lands first and the report stops there.
#:
#: This exists because the emitters really can drift. `nested_assign` below is a
#: regression test: `emitAssign` (`src/codegen/codegen_expr.c:896`) interns the
#: assigned field's name *before* it emits the object expression, so `a.b.c = v`
#: puts `c` in the constant pool ahead of `b`. `emit.jai` interned it at the
#: SET_FIELD instead, and produced the same instructions with the two u24
#: operands swapped — invisible to every test in the suite, and a `DIFF` on four
#: files in `make bootstrap`.
#:
#: It is a golden test too, so the driver runs it at -O0, -O1, -O2 and -O3 and
#: requires the same output from each.
enum Colour { Red, Green, Blue }

trait Drawable {
    fn draw(self)
}

class Base {
    pub var n: any

    pub fn init(self, n) { self.n = n }

    pub fn draw(self) { return self.n }
}

class Derived extends Base {
    pub fn init(self, n) {
        super(n)
        self.n = self.n + 1
    }

    pub fn draw(self) { return super.draw() + 1 }
}

class Holder {
    pub var inner: any

    pub fn init(self, inner) { self.inner = inner }
}

fn nested_assign(h, v) {
    h.inner.n = v
    h.inner.n += 1
    return h.inner.n
}

fn spread_call(f, xs) { return f(...xs) }

fn kw_call(f) { return f(1, b: 2, c: 3) }

fn kwargs(a, b = 0, c = 0) { return a + b + c }

fn strings(a) {
    let s: any = f"{a} and {a + 1}"
    return s.upper() + "x"
}

fn comprehensions(xs) {
    let a: any = [v + 1 for v in xs]
    let b: any = {v for v in xs}
    let c: any = {v: v + 1 for v in xs}
    let d: any = [v for v in xs if v > 1]
    return a.len() + b.len() + c.len() + d.len()
}

fn collections(xs) {
    let t: any = (1, 2, 3)
    let (p, q, r) = t
    let sl: any = xs[1:]
    let st: any = xs[:2]
    let sp: any = xs[::2]
    xs[0] = 9
    return p + q + r + sl.len() + st.len() + sp.len()
}

fn optionals(a) {
    let v: any = a?.n
    let w: any = v ?? 7
    return w
}

fn membership(xs, v) {
    if v in xs { return 1 }
    if v is int { return 2 }
    return 3
}

fn ranges(n) {
    var t: any = 0
    for i in 0..n { t = t + i }
    for i in 0..=n { t = t + i }
    return t
}

fn closures(a) {
    var acc: any = a
    let bump: any = fn(d) {
        acc = acc + d
        return acc
    }
    let inner: any = fn() { return fn() { return acc } }
    bump(1)
    return inner()()
}

fn ternaries(a) { return if a { 1 } else { 2 } }

print(Derived(1).draw())
print(nested_assign(Holder(Base(0)), 5))
print(spread_call(kwargs, [1, 2, 3]))
print(kw_call(kwargs))
print(strings(1))
print(comprehensions([1, 2, 3]))
print(collections([1, 2, 3]))
print(optionals(Base(4)))
print(optionals(null))
print(membership([1, 2], 2))
print(ranges(3))
print(closures(1))
print(ternaries(true))
print(Colour.Red)

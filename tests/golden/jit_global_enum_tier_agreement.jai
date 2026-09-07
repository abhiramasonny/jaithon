#: Module-level enum variants read from JIT-warmed bodies. The two tiers must
#: agree exactly.
#:
#: Eighth of the tier-agreement set. `TokenKind.Plus` is OP_GET_GLOBAL followed
#: by OP_GET_FIELD, and the global used to take the BY-ADDRESS path: a JaiEntry
#: load behind a tag guard, plus an Obj.type check and a shapeId check in the
#: enum-variant fold. Four deopt records a site. `lexer._punct_kind` has 51 such
#: sites, needed 316 records against a JIT_MAX_DEOPT of 160, and so did not
#: compile at all. The enum is now resolved at compile time like a class and the
#: fold skips both guards for a pinned receiver, which takes that body to
#: deopt=114 and lets it compile.
#:
#: Baking a value by address is only sound while a rebind retires the form, so
#: the edges here are the ones that could disagree if pinning were wrong: many
#: distinct variants read from one hot body (the shape that overflowed the
#: table), a variant compared against another, a variant carried through a
#: `match`, a variant read from a body that also reads a NON-enum global of the
#: same shape, a LOCAL shadowing the enum's name, and a payload-carrying variant
#: which has arity > 0 and so must NOT take the unit fold.
#:
#: NOTE FOR WHOEVER MEASURES THIS: interpreted-instruction counts are IDENTICAL
#: with the arm on and off (13,153 either way), because these bodies compile
#: both ways -- the arm removes guards from compiled code, it does not change
#: what is interpreted. The coverage proof is the compile shape instead:
#:   JAI_JIT_WHY=1 ./jaithon run <this file>
#: reports `classify` at deopt=0 insts=158 with the arm on and deopt=32
#: insts=1086 with JAITHON_JIT_GLOBAL_ENUM=0 -- 6.9x the code and 32 deopt
#: records for the same function.

enum Colour { Red, Green, Blue, Amber, Violet, Teal, Rose, Slate }

enum Shape { Dot, Line(a: int), Box(a: int, b: int) }

let LIMIT = 4

fn classify(n: int) -> Colour {
    if n % 8 == 0 { return Colour.Red }
    if n % 8 == 1 { return Colour.Green }
    if n % 8 == 2 { return Colour.Blue }
    if n % 8 == 3 { return Colour.Amber }
    if n % 8 == 4 { return Colour.Violet }
    if n % 8 == 5 { return Colour.Teal }
    if n % 8 == 6 { return Colour.Rose }
    return Colour.Slate
}

fn is_warm(c: Colour) -> bool {
    return c == Colour.Red or c == Colour.Amber or c == Colour.Rose
}

fn name_of(c: Colour) -> str {
    match c {
        Colour.Red => { return "red" }
        Colour.Green => { return "green" }
        Colour.Blue => { return "blue" }
        Colour.Amber => { return "amber" }
        Colour.Violet => { return "violet" }
        Colour.Teal => { return "teal" }
        Colour.Rose => { return "rose" }
        Colour.Slate => { return "slate" }
    }
}

#: Reads a module-level non-enum global beside the enum, so the two resolution
#: paths meet in one body.
fn bounded(n: int) -> Colour {
    if n > LIMIT { return Colour.Slate }
    return classify(n)
}

#: The enum's own name bound to a local: the local must win.
fn shadowed() -> str {
    let Colour = "not the enum"
    return Colour
}

fn describe(s: Shape) -> str {
    match s {
        Shape.Dot => { return "dot" }
        Shape.Line(a) => { return "line " + str(a) }
        Shape.Box(a, b) => { return "box " + str(a) + "x" + str(b) }
    }
}

fn main() {
    var warm = 0
    var names = ""
    for i in 0..600 {
        if is_warm(classify(i)) { warm += 1 }
        if bounded(i) == Colour.Slate { warm += 1 }
    }
    for i in 0..8 { names = names + name_of(classify(i)) + " " }
    print("warm " + str(warm))
    print("names " + names)

    var out: list[str] = []
    for i in 0..8 { out.push(name_of(classify(i))) }
    print("classify " + str(out))

    print("bounded " + name_of(bounded(2)) + " " + name_of(bounded(99)))
    print("shadowed " + shadowed())

    # Payload variants: arity > 0, so the unit fold must not apply.
    print("shapes " + describe(Shape.Dot) + " / " + describe(Shape.Line(3))
          + " / " + describe(Shape.Box(2, 5)))

    # Identity: the same variant read twice must be the same value.
    print("same " + str(Colour.Teal == Colour.Teal)
          + " diff " + str(Colour.Teal == Colour.Rose))
}

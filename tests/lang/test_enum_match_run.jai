#: A `match` whose consecutive arms only test variants of one enum lowers as
#: a RUN: one type test, one tag read into a frame slot, then one fused
#: compare-and-branch per alternative (emit/pattern.jai `_enum_run`).
#:
#: Each case pins an answer the per-arm lowering already gave, at the places
#: the run changes control flow: a guard that fails and must fall to the NEXT
#: arm testing the same tag; two runs over different enums in one match; a
#: run broken by a binding arm or a payload pattern and resumed after it; a
#: subject that is not the enum at all; a match nested in another match's arm
#: (two tag slots live at once); and a match inside an expression, where the
#: operand stack is not empty when it starts.
#:
#: Every body is called past JAI_JIT_THRESHOLD, so the compiled tier gives the
#: same answers as the interpreter.
from std.test import assert_eq, assert_throws

let CALLS = 300

enum Op { Add, Sub, Mul, Div, Mod, Pow }

enum Color { Red, Green, Blue }

enum Shape { Dot, Line, Circle(radius: int), Box(w: int, h: int) }

enum Letter { A, B, C, D, E, F, G, H, I, J, K, L }

#: Nine distinct tags, so the run dispatches by a binary search on the tag:
#: gaps (G, I, L) that must miss, and a guarded arm whose tag a later arm
#: names again.
fn bucket(x: Letter, n: int) -> int {
    return match x {
        Letter.A | Letter.B => 1,
        Letter.C if n > 0 => 2,
        Letter.D | Letter.E | Letter.F => 3,
        Letter.C => 4,
        Letter.H => 5,
        Letter.J | Letter.K => 6,
        _ => 8,
    }
}

#: Every variant named and no default: no tag misses, but a subject that is
#: not a Letter still has to.
fn every_letter(v: any) -> int {
    match v {
        Letter.A | Letter.L => { return 1 }
        Letter.B | Letter.K => { return 2 }
        Letter.C | Letter.J => { return 3 }
        Letter.D | Letter.I => { return 4 }
        Letter.E | Letter.H => { return 5 }
        Letter.F | Letter.G => { return 6 }
    }
    return 0
}

fn _letters() -> list[Letter] {
    return [
        Letter.A, Letter.B, Letter.C, Letter.D, Letter.E, Letter.F,
        Letter.G, Letter.H, Letter.I, Letter.J, Letter.K, Letter.L,
    ]
}

fn _ops() -> list[Op] {
    return [Op.Add, Op.Sub, Op.Mul, Op.Div, Op.Mod, Op.Pow]
}

#: A guard on the first arm; when it fails the SAME tag is tested again two
#: arms later, after an arm for a different tag.
fn guarded(o: Op, n: int) -> int {
    return match o {
        Op.Add if n > 0 => 10,
        Op.Sub | Op.Mul => 20,
        Op.Add => 30,
        Op.Div if n > 5 => 40,
        Op.Div | Op.Mod if n > 2 => 50,
        _ => 60,
    }
}

#: Two runs, one per enum, on a subject of unknown type.
fn of_any(v: any) -> str {
    return match v {
        Op.Add | Op.Sub => "additive",
        Op.Pow => "power",
        Color.Red | Color.Blue => "warm or cool",
        Color.Green => "green",
        _ => "neither",
    }
}

#: A run, a binding arm with a guard, and a second run of the same enum.
fn broken(o: Op, limit: int) -> int {
    return match o {
        Op.Add | Op.Sub => 1,
        x if limit < 0 => 2,
        Op.Mul | Op.Div => 3,
        Op.Pow => 4,
        _ => 5,
    }
}

#: Payload patterns end the run; constant variants after them start another.
fn measured(s: Shape) -> int {
    return match s {
        Shape.Dot | Shape.Line => 0,
        Shape.Circle(r) => r * 3,
        Shape.Box(w, h) if w == h => -1,
        Shape.Box(w, h) => w * h,
    }
}

#: A match in an arm of another match: both tag slots are live at once.
fn nested(o: Op, c: Color) -> int {
    match o {
        Op.Add | Op.Sub => {
            match c {
                Color.Red => { return 1 }
                Color.Green | Color.Blue => { return 2 }
            }
        }
        Op.Mul => {
            return match c {
                Color.Red | Color.Green => 3,
                Color.Blue => 4,
            }
        }
        _ => { return 5 }
    }
}

#: Started with two values already on the operand stack.
fn inside_expression(o: Op, c: Color) -> int {
    return 100 * (match o {
        Op.Add | Op.Sub => 1,
        Op.Mul | Op.Div => 2,
        _ => 3,
    }) + (match c {
        Color.Red => 10,
        Color.Green | Color.Blue => 20,
    })
}

#: A statement match in a loop whose arms break, continue and fall through.
fn walk(ops: list[Op]) -> int {
    var total = 0
    for o in ops {
        match o {
            Op.Pow => { break }
            Op.Mod | Op.Div => { continue }
            Op.Add => { total += 1 }
            Op.Sub | Op.Mul => { total += 100 }
        }
        total += 1000
    }
    return total
}

#: Every variant listed, no default: an `any` that is not an Op raises.
fn exhaustive(v: any) -> int {
    return match v {
        Op.Add | Op.Sub | Op.Mul => 1,
        Op.Div | Op.Mod | Op.Pow => 2,
    }
}

pub fn test_a_failed_guard_falls_to_the_next_arm_with_that_tag() -> void {
    var call = 0
    while call < CALLS {
        assert_eq(guarded(Op.Add, 1), 10, "guard true")
        assert_eq(guarded(Op.Add, 0), 30, "guard false: the later Add arm")
        assert_eq(guarded(Op.Sub, 0), 20, "an alternation")
        assert_eq(guarded(Op.Mul, 9), 20, "its second alternative")
        assert_eq(guarded(Op.Div, 9), 40, "first Div guard")
        assert_eq(guarded(Op.Div, 3), 50, "second Div guard")
        assert_eq(guarded(Op.Mod, 3), 50, "Mod under the second guard")
        assert_eq(guarded(Op.Mod, 1), 60, "both guards false")
        assert_eq(guarded(Op.Pow, 9), 60, "no arm names it")
        call += 1
    }
}

pub fn test_two_runs_over_two_enums_and_a_subject_of_neither() -> void {
    var call = 0
    while call < CALLS {
        assert_eq(of_any(Op.Add), "additive", "first run")
        assert_eq(of_any(Op.Sub), "additive", "first run, second alternative")
        assert_eq(of_any(Op.Pow), "power", "first run, second arm")
        assert_eq(of_any(Op.Mul), "neither", "an Op no arm names")
        assert_eq(of_any(Color.Red), "warm or cool", "second run")
        assert_eq(of_any(Color.Blue), "warm or cool", "second run, second alternative")
        assert_eq(of_any(Color.Green), "green", "second run, second arm")
        assert_eq(of_any("Add"), "neither", "a string")
        assert_eq(of_any(1), "neither", "an int, equal to a tag")
        assert_eq(of_any(null), "neither", "null")
        assert_eq(of_any(Shape.Dot), "neither", "a third enum")
        call += 1
    }
}

pub fn test_a_run_broken_by_a_binding_arm() -> void {
    let ops = _ops()
    let want_free = [1, 1, 3, 3, 5, 4]
    var call = 0
    while call < CALLS {
        let index = call % 6
        assert_eq(broken(ops[index], 1), want_free[index], "guard false")
        let guarded_want = if index < 2 { 1 } else { 2 }
        assert_eq(broken(ops[index], -1), guarded_want, "guard true")
        call += 1
    }
}

pub fn test_payload_arms_between_constant_arms() -> void {
    var call = 0
    while call < CALLS {
        assert_eq(measured(Shape.Dot), 0, "constant run")
        assert_eq(measured(Shape.Line), 0, "constant run, second alternative")
        assert_eq(measured(Shape.Circle(4)), 12, "payload")
        assert_eq(measured(Shape.Box(3, 3)), -1, "guarded payload")
        assert_eq(measured(Shape.Box(3, 5)), 15, "payload after a failed guard")
        call += 1
    }
}

pub fn test_a_match_nested_in_an_arm() -> void {
    var call = 0
    while call < CALLS {
        assert_eq(nested(Op.Add, Color.Red), 1, "outer run, inner run")
        assert_eq(nested(Op.Sub, Color.Blue), 2, "inner alternation")
        assert_eq(nested(Op.Mul, Color.Green), 3, "inner expression match")
        assert_eq(nested(Op.Mul, Color.Blue), 4, "its last arm")
        assert_eq(nested(Op.Pow, Color.Red), 5, "outer default")
        call += 1
    }
}

pub fn test_matches_inside_an_expression() -> void {
    var call = 0
    while call < CALLS {
        assert_eq(inside_expression(Op.Add, Color.Red), 110, "first arms")
        assert_eq(inside_expression(Op.Div, Color.Blue), 220, "second arms")
        assert_eq(inside_expression(Op.Pow, Color.Green), 320, "default and alternation")
        call += 1
    }
}

pub fn test_a_statement_match_that_breaks_and_continues() -> void {
    var call = 0
    while call < CALLS {
        assert_eq(walk([Op.Add, Op.Sub, Op.Mod, Op.Mul, Op.Pow, Op.Add]), 3201, "stops at Pow")
        assert_eq(walk([Op.Div, Op.Div]), 0, "every element continues")
        call += 1
    }
}

pub fn test_a_run_dispatched_by_binary_search() -> void {
    let letters = _letters()
    let positive = [1, 1, 2, 3, 3, 3, 8, 5, 8, 6, 6, 8]
    let other = [1, 1, 4, 3, 3, 3, 8, 5, 8, 6, 6, 8]
    let paired = [1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1]
    var call = 0
    while call < CALLS {
        let index = call % 12
        assert_eq(bucket(letters[index], 1), positive[index], "guard true")
        assert_eq(bucket(letters[index], 0), other[index], "guard false")
        assert_eq(every_letter(letters[index]), paired[index], "every variant")
        call += 1
    }
    assert_eq(every_letter(Op.Add), 0, "another enum")
    assert_eq(every_letter(3), 0, "an int equal to a tag")
    assert_eq(every_letter("A"), 0, "a string")
}

pub fn test_an_exhaustive_run_still_raises_for_a_foreign_subject() -> void {
    var call = 0
    while call < CALLS {
        assert_eq(exhaustive(Op.Sub), 1, "first arm")
        assert_eq(exhaustive(Op.Pow), 2, "second arm")
        call += 1
    }
    assert_throws(AssertionError, fn() { exhaustive(Color.Red) }, "a foreign enum")
    assert_throws(AssertionError, fn() { exhaustive("Add") }, "a string")
}

#: A subscript of a list the tier can never have a sample of.
#:
#: `out` is built inside the body and filled by a callee that mutates it
#: through the alias, never reassigning it -- so at the moment the tier looks,
#: the local holds an empty list and no later sample of it can exist. The
#: element kind has to come from the declaration (or, for the undeclared
#: literal below, from the literal's own entries) or the whole body declines.
#:
#: jaicv's `min_area_rect` is this exact shape and was the single largest
#: refusal on that benchmark before the declared route existed.
# jit-compiles: declared_head, literal_head, int_literal_head
from std.test import assert_eq

#: Above JAI_JIT_THRESHOLD entries, so the bodies below reach the compiled tier.
let CALLS = 200

fn fill_declared(into: list[float], n: int) {
    var i = 0
    while i < n {
        into.push(float(i) * 1.5)
        i += 1
    }
}

#: No loop of its own, so only the FUNCTION tier ever looks at it -- and the
#: function tier compiles from the frame as it is at ENTRY, where `out` has
#: not been built yet. That is what makes this the no-sample case: an OSR
#: compile of a loop here would see the filled list and never need the
#: declaration at all, which is why the marked bodies below are loop-free.
fn declared_head(n: int) -> float {
    var out: list[float] = []
    fill_declared(out, n)
    return out[0] + out[1] + out[2]
}

fn literal_head(k: int) -> float {
    var box = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fill_declared(box, k)
    #: Past the six the literal declared, so these are the appended values.
    return box[6] + box[7] + box[8]
}

fn int_literal_head(k: int) -> int {
    var box = [0, 0, 0, 0]
    fill_ints(box, k)
    return box[4] + box[5] + box[6]
}

fn fill_ints(into: list[int], n: int) {
    var i = 0
    while i < n {
        into.push(i * 3)
        i += 1
    }
}

fn declared_route(n: int) -> float {
    var out: list[float] = []
    fill_declared(out, n)
    var total = 0.0
    var i = 0
    while i < out.len() {
        total += out[i]
        i += 1
    }
    return total
}

#: No declared type, so no OP_ELEM_KIND -- the kind comes from the six
#: uniform entries of the literal itself.
fn literal_route(n: int) -> float {
    var box = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    var i = 0
    while i < n {
        box[i % 6] = box[i % 6] + float(i)
        i += 1
    }
    return box[0] + box[1] + box[2] + box[3] + box[4] + box[5]
}

fn int_literal_route(n: int) -> int {
    var box = [0, 0, 0, 0]
    var i = 0
    while i < n {
        box[i % 4] = box[i % 4] + i
        i += 1
    }
    return box[0] + box[1] + box[2] + box[3]
}

#: A wrong prediction must deoptimise, not misread. An undeclared literal has
#: boxed storage, so an append may put anything in it; the read tag-checks.
fn mixed_after_literal(n: int) -> int {
    var box = [1, 2, 3]
    box.push("four")
    var seen = 0
    var i = 0
    while i < n {
        let v = box[i % 3]
        seen += v
        i += 1
    }
    return seen + box[3].len()
}

pub fn test_a_declared_list_keeps_its_element_kind() -> void {
    var last = 0.0
    var r = 0
    while r < CALLS {
        last = declared_route(64)
        r += 1
    }
    assert_eq(last, 3024.0, "sum of a list no sample can exist for")
}

pub fn test_an_undeclared_float_literal_predicts_itself() -> void {
    var last = 0.0
    var r = 0
    while r < CALLS {
        last = literal_route(60)
        r += 1
    }
    assert_eq(last, 1770.0, "sum")
}

pub fn test_an_undeclared_int_literal_predicts_itself() -> void {
    var last = 0
    var r = 0
    while r < CALLS {
        last = int_literal_route(40)
        r += 1
    }
    assert_eq(last, 780, "sum")
}

pub fn test_a_literal_that_later_holds_another_kind_still_reads_right() -> void {
    var last = 0
    var r = 0
    while r < CALLS {
        last = mixed_after_literal(30)
        r += 1
    }
    assert_eq(last, 64, "the prediction deoptimises rather than misreading")
}

pub fn test_the_function_tier_reads_a_list_it_never_sampled() -> void {
    var a = 0.0
    var b = 0.0
    var c = 0
    var r = 0
    while r < CALLS {
        a = declared_head(8)
        b = literal_head(8)
        c = int_literal_head(8)
        r += 1
    }
    assert_eq(a, 4.5, "declared list[float]")
    assert_eq(b, 4.5, "undeclared float literal")
    assert_eq(c, 9, "undeclared int literal")
}

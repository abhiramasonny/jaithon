#: `{k: v}` and `{a, b}`, the two container literals still without an arm.
#:
#: Both ENDED THE WALK. They were top of the partial-walk census over the
#: self-hosted parser -- `_node(kind, span, fields: dict = {})` builds a dict
#: for every AST node it makes, and the walk stopped at OP_BUILD_DICT sixteen
#: times in that one file.
#:
#: Measured in one binary through `JAITHON_JIT_TUPLE`:
#:
#:   {"a": i & 3, "b": i & 7}   38,000,320 ->  7,443    0.21s -> 0.17s   1.2x
#:   {i & 3, i & 7}             34,000,320 -> 24,560    0.46s -> 0.31s   1.4x
#:
#: The clock moves less than the instruction count because allocating the dict
#: is most of what the microbenchmark does -- the arm cannot make jaiDictNew
#: cheaper. What it buys is the body around the literal, which in the parser is
#: most of the function.
#:
#: Keys and values alternate on the stack in the order the interpreter reads
#: them, so the descriptor's args array is passed through as-is; a set is the
#: same with one operand per element. Both hash, so both can raise on an
#: unhashable key, and both come back through the descriptor's threw branch
#: with the interpreter's own message: `{[1, 2]: 1}` reports
#: `TypeError: unhashable type: 'list'` from compiled code and interpreted
#: alike.
#: Giving the literals arms made OP_ELEM_KIND reachable with a non-list on top
#: for the first time, and its arm declined outright -- so `var d: dict[str,
#: int] = {}` in a hot body went from a partial walk to a whole-function
#: decline. The stamp arm now does what the interpreter does: both nibbles onto
#: a dict, nothing at all for any other container.
# jit-compiles: mk, mkset, dup, typed
from std.test import assert_eq, assert_true, assert_throws

#: Above JAI_JIT_THRESHOLD entries, so the bodies below reach the compiled tier.
let CALLS = 300

fn mk(a: int, b: int) -> dict[str, int] {
    return {"a": a, "b": b}
}

fn mkset(a: int, b: int) -> set[int] {
    return {a, b}
}

fn dup(a: int) -> dict[str, int] {
    return {"k": a, "k": a + 1}
}

fn empty() -> dict[str, int] {
    return {}
}

fn typed(rounds: int) -> int {
    var d: dict[str, int] = {}
    #: `{}` is always a dict literal, so a set annotation needs an element.
    var s: set[int] = {0}
    var l: list[int] = []
    var i = 0
    while i < rounds {
        d[str(i)] = i
        l.push(i)
        i += 1
    }
    return len(d) + len(s) + len(l)
}

fn warm() -> void {
    var i = 0
    while i < CALLS {
        let _ = mk(i, i + 1)
        let _s = mkset(i & 3, i & 1)
        let _d = dup(i)
        let _t = typed(2)
        i += 1
    }
}

pub fn test_a_dict_literal_holds_what_it_was_given() -> void {
    warm()
    let d = mk(1, 2)
    assert_eq(d["a"], 1, "first")
    assert_eq(d["b"], 2, "second")
    assert_eq(len(d), 2, "len")
    assert_eq(len(empty()), 0, "the empty literal")
}

pub fn test_a_repeated_key_keeps_the_later_value() -> void {
    warm()
    #: The interpreter sets keys left to right, so the second write wins and
    #: the dict is one entry long. The arm passes the operands through in that
    #: same order, which is the whole reason it can.
    assert_eq(dup(5)["k"], 6, "later wins")
    assert_eq(len(dup(5)), 1, "and adds no entry")
}

pub fn test_a_set_literal_dedups() -> void {
    warm()
    assert_eq(len(mkset(3, 3)), 1, "one member")
    assert_eq(len(mkset(1, 2)), 2, "two members")
    assert_true(3 in mkset(3, 3), "membership")
}

pub fn test_an_annotated_container_is_stamped_the_same_way() -> void {
    warm()
    assert_eq(typed(4), 9, "4 dict + 1 set + 4 list")
    assert_eq(typed(0), 1, "just the set element")
}

pub fn test_an_unhashable_key_throws_the_interpreters_error() -> void {
    assert_throws(TypeError, fn() -> void { let _ = {[1, 2]: 1} }, "a list cannot be a key")
}

#: `a == b` on two heap objects the tier can say nothing else about.
#:
#: The equality chain had four arms above this one and each answers without a
#: call -- two registers, interned string pointers, a folded enum unit, a
#: guarded string leaf call. Anything else declined the WHOLE body for one
#: comparison.
#:
#: The shape that made it worth fixing is `self._kind() == kind` on two
#: TokenKinds, in the self-hosted parser's `_check`. The enum arm above cannot
#: help it: that one needs one side to be a FOLDED constant, and here both are
#: values. `_check` was the SHORTEST CHAIN ON A HOT BODY -- one refusal away
#: from compiling, carrying 3.2% of lexer.jai's interpreted work by exact
#: attribution -- which is the ranking that picks work, not refusal frequency.
#:
#: Measured through `JAITHON_JIT_OBJ_EQ`, four samples each, twice:
#:
#:   check --no-cache lexer.jai     0.440s -> 0.428s     +7 compiled bodies
#:   check --no-cache resolve.jai   0.427s -> 0.418s
#:   check --no-cache parser.jai    0.845s -> 0.838s     (inside the noise)
#:   four bench programs            0.24s  -> 0.24s      (no regression)
#:
#: The call is `jaiValuesEqual`, the interpreter's own, so `__eq__` behaves and
#: a comparison that raises raises the same message. It can allocate and throw,
#: so roots go down and the descriptor's threw branch carries the raise.
#:
#: It is LAST in the chain deliberately. Every arm above is strictly better
#: where it applies, and putting a call in front of a register compare would be
#: a straight loss.
# jit-compiles: same_kind, differ
from std.test import assert_eq, assert_true, assert_false, assert_throws

#: Above JAI_JIT_THRESHOLD entries, so these reach the compiled tier by a count
#: rather than by the OSR sampler.
let CALLS = 200

enum Colour { Red, Green, Blue }

class Boxed {
    pub var v: int

    pub fn init(self, v: int) -> void {
        self.v = v
    }
}

class Weird {
    pub var v: int

    pub fn init(self, v: int) -> void {
        self.v = v
    }

    #: A user equality, so the arm is exercised against something that is NOT
    #: pointer identity. The call is the interpreter's own, so this must be
    #: honoured exactly as it is when interpreted.
    pub fn __eq__(self, other: any) -> bool {
        return self.v % 10 == other.v % 10
    }
}

fn same_kind(a: any, b: any) -> bool {
    return a == b
}

fn differ(a: any, b: any) -> bool {
    return a != b
}

fn warm() -> void {
    let red = Colour.Red
    var i = 0
    while i < CALLS {
        let _ = same_kind(red, red)
        let _n = differ(red, red)
        i += 1
    }
}

pub fn test_two_enum_values_compare_by_value() -> void {
    warm()
    let red = Colour.Red
    let green = Colour.Green
    assert_true(same_kind(red, red), "the same variant")
    assert_false(same_kind(red, green), "a different variant")
    assert_false(differ(red, red), "not-equal is the inverse")
    assert_true(differ(red, green), "and both ways round")
}

pub fn test_the_other_object_kinds() -> void {
    warm()
    assert_true(same_kind("ab", "ab"), "strings by content")
    assert_false(same_kind("ab", "ba"), "and unequal ones")
    assert_true(same_kind([1, 2], [1, 2]), "lists by content")
    assert_false(same_kind([1, 2], [2, 1]), "and ordered")
    assert_true(same_kind({"k": 1}, {"k": 1}), "dicts by content")
    assert_false(same_kind("ab", [1, 2]), "different types are not equal")
}

pub fn test_a_user_equality_is_honoured() -> void {
    warm()
    #: 3 and 13 agree mod ten, so this is true only if `__eq__` actually ran.
    assert_true(same_kind(Weird(3), Weird(13)), "the user rule")
    assert_false(same_kind(Weird(3), Weird(14)), "and it discriminates")
    assert_false(differ(Weird(3), Weird(13)), "not-equal inverts it too")
}

pub fn test_identity_is_still_identity_for_plain_instances() -> void {
    warm()
    let one = Boxed(1)
    let two = Boxed(1)
    assert_true(same_kind(one, one), "an instance equals itself")
    #: No `__eq__` on Boxed, so two distinct instances are not equal even with
    #: the same field -- the interpreter's rule, unchanged.
    assert_false(same_kind(one, two), "and not a twin")
}

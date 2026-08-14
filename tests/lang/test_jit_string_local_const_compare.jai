#: `if c == "{"` where `c` is a string local and the right-hand side is a
#: constant -- the shape every hand-written scanner in the language is made of,
#: and the one the peephole folds into OP_JUMP_IF_CMP_LOCAL_K.
#:
#: The tier compiles it by comparing POINTERS, which is only equality for two
#: interned strings. The constant's interning is settled when the code is
#: emitted; the local's is not, so it is guarded, and so is its object type.
#: Both guards resume at the compare itself, with nothing written.
#:
#: The object-type guard has teeth here: remove it and a list reaches the
#: pointer compare. The interning guard does NOT, and that is worth writing
#: down rather than leaving for the next reader to rediscover: every string
#: construction path in this VM interns anything up to 32 bytes, and a constant
#: over 32 bytes is not interned either, so the arm declines it -- which means
#: a non-interned local can never be content-equal to a constant that reached
#: the pointer compare. Indexing, slicing and concatenation were all tried and
#: all come back interned. The guard stays because that is an invariant of
#: object_string.c and not of this arm, and because the OP_EQ arm beside it
#: makes the same check for the same reason.
#:
#: scripts/jit_compile_check.py fails if `scan` or `is_open` stops reaching
#: the compiled tier under JAI_JIT_WHY=1. `scan_concat` is deliberately NOT in
#: the marker: `let c = a + b` concatenates two strings, and OP_ADD_LOCALS on
#: two dynamic operands has no measuring-pass support at all ("OP_ADD_LOCALS",
#: every attempt) -- the same gap as join_keys/spell in
#: test_jit_dict_items_iter.jai. `test_a_concatenation_equal_by_content_is_still_equal`
#: therefore only ever proves the INTERPRETER's answer for the
#: runtime-built-string case; that is a real hole in this file's compiled-tier
#: coverage, not a heat problem, and no marker or warm-up closes it.
from std.test import assert_eq

#: Interned one-character strings, straight out of the shared table.
fn scan(s: str) -> int {
    var opens = 0
    var i = 0
    while i < s.len() {
        let c = s[i]
        if c == "{" { opens = opens + 1 }
        i = i + 1
    }
    return opens
}

#: The local holds a string this loop built rather than one out of the shared
#: one-byte table, so the compare cannot be answered by the ASCII proof the
#: OP_EQ arm uses and goes through the guards.
fn scan_concat(a: str, b: str, n: int) -> int {
    var hits = 0
    var i = 0
    while i < n {
        let c = a + b
        if c == "{x" { hits = hits + 1 }
        i = i + 1
    }
    return hits
}

#: The local is an object that is not a string. It reaches the compiled body --
#: the entry guard for a raw-object parameter only says "an object" -- so the
#: object-type guard inside is the only thing between a list and being compared
#: as though it were an interned string.
fn is_open(c: any) -> int {
    let x = c
    if x == "{" { return 1 }
    return 0
}

fn warm_scan(s: str) -> void {
    for _r in 0..20000 { scan(s) }
}

fn warm_concat(a: str, b: str) -> void {
    for _r in 0..20000 { scan_concat(a, b, 8) }
}

fn warm_is_open() -> void {
    for _r in 0..20000 { is_open("{") }
}

fn test_interned_characters_compare_by_identity() -> void {
    let s = "{a{b{c}}}"
    warm_scan(s)
    assert_eq(scan(s), 3)
}

fn test_a_concatenation_equal_by_content_is_still_equal() -> void {
    #: `"{" + "x"` holds "{x". Equal to the constant, and not the same object.
    warm_concat("{", "x")
    assert_eq(scan_concat("{", "x", 8), 8)
    assert_eq(scan_concat("{", "y", 8), 0)
}

fn test_a_non_string_object_compares_false() -> void {
    warm_is_open()
    assert_eq(is_open("{"), 1)
    assert_eq(is_open("}"), 0)
    var xs: list = []
    xs.push(1)
    assert_eq(is_open(xs), 0)
    assert_eq(is_open(xs), 0)
}

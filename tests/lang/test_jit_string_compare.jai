#: Comparing two strings whose lengths nothing knows, in a loop hot enough to
#: reach the compiled tier.
#:
#: The tier had two string arms and neither covered it: `==` by POINTER, which
#: is only equality for two interned strings, and an ordering arm that needs
#: one side to be statically ONE BYTE. Everything else declined the whole body,
#: and a comparator, a parser or a dispatch on a name is nothing else. The arm
#: added for it guards `Obj.type == OBJ_STRING` on both operands and then calls
#: `jaiStringOrder` -- the runtime's own comparison, the same function
#: `jaiValueCompare` uses -- as a leaf: two pointers in x0/x1, the sign of the
#: order back in x0, no descriptor and no roots, because a string comparison
#: allocates nothing.
#:
#: Three opcodes carry it. `if a < b` fuses into OP_JUMP_IF_CMP_FALSE;
#: binding the answer to a local leaves the plain OP_LT..OP_NE; and a local
#: against a string CONSTANT is folded again, into OP_JUMP_IF_CMP_LOCAL_K,
#: where the local is the left operand. All three are exercised below, because
#: an arm for one of them leaves the other two declining.
#:
#: NO OPERAND HERE IS ONE BYTE LONG, deliberately, and that is what gives the
#: `jit-compiles:` marker its teeth. The one-byte arm is tried first and is
#: strictly better where it applies -- two byte loads and no call -- so a pair
#: with a `"a"` on either side would compile through THAT arm and the marker
#: would pass with this fix reverted. Verified: with `JAITHON_JIT_STRCMP=0`
#: every marked function stops reaching the tier.
#:
#: What these check is the ANSWER, for every case that can tell one ordering
#: from another:
#:
#:   - equal strings, and a string against itself (the pointer shortcut);
#:   - a prefix against its extension, decided by LENGTH after every shared
#:     byte matched;
#:   - the empty string, where the shared run is zero bytes long;
#:   - multi-byte UTF-8. The order is by BYTE. `s[i]` indexes SCALARS, so the
#:     two could have disagreed; `"é"` against `"ez"` is the pair that pins it
#:     down -- one scalar against two, the same number of BYTES, and the answer
#:     is GREATER because 0xC3 > 0x65;
#:   - a pair built at RUN TIME, so neither operand is interned. This is the
#:     case `==` reaches the arm through, and only because of the second half
#:     of this change -- see below. Note a string LITERAL is interned at any
#:     length (`serialize.c:264` interns every constant as it deserialises);
#:     only a run-time string is subject to JAI_INTERN_MAX;
#:   - a string against a NON-string, which must leave the compiled body and
#:     raise exactly what the interpreter raises;
#:   - all six operators, through each of the three opcodes.
#:
#: THE SECOND HALF: `==` and `!=` do not reach this arm on their own. At all
#: three sites an interned-POINTER arm is tested first, on a condition this arm
#: cannot be reached without, so the four orderings `< <= > >=` were the only
#: operators that ever got here. That mattered more than a coverage gap: for a
#: pair built at run time the pointer arm compiles a `subFlag` guard that MISSES
#: EVERY ITERATION, and the deopt round trip costs more than the instructions it
#: saves. `x == y` over a 46-byte run-time pair ran **2x SLOWER with the tier
#: than with JAITHON_NO_JIT=1** -- a pre-existing loss, not one this change
#: introduced.
#:
#: `preferLeafEquality` makes the pointer arm stand aside when neither operand
#: is KNOWN interned, sending `==`/`!=` here instead. Measured on one binary
#: through `JAITHON_JIT_STRCMP_EQ`, best of five:
#:
#:                        with     without    no tier
#:   rv_eq_bare           0.05      0.31       0.16     3.2x faster than no tier
#:   rv_comparator        0.03      0.11       0.10     3.3x
#:   rv_mixed_regress     0.06      0.18       0.13     2.2x
#:
#: The interned-hot controls do not move, which is the point of predicting off
#: the sample: a wrong "interned" keeps today's deopt, a wrong "not interned"
#: pays a call where a register compare would have done.
#:
#: `--stats` SAYS THE OPPOSITE OF THE CLOCK on these. The unfixed arm shows
#: HALF the interpreted instructions and twice the wall time. Do not accept an
#: instruction count as evidence for this arm in either direction.
#:
#: Run with JAITHON_NO_JIT=1, with JAITHON_JIT_DEOPT_STRESS=1 and under
#: --gc-stress=3; the answers must not move. `JAITHON_JIT_STRCMP=0` restores
#: the decline, and `JAITHON_JIT_STRCMP_EQ=0` the pointer arm, and neither must
#: move them either.
#:
#: One thing the green configurations do NOT prove: under
#: JAITHON_JIT_DEOPT_STRESS=1 the type guard becomes an unconditional branch, so
#: the leaf call is dead code in that mode. That run exercises the deopt path
#: and says nothing about the call path.
# jit-compiles: compare_all, compare_all_unfused, compare_all_const, count_less, first_of
from std.test import assert_eq, assert_true, assert_throws

#: Every one of the six operators, fused into its branch -- which is what `if
#: a < b` compiles to, and the form the probe that found this hits. Packed into
#: one int so a single assertion covers all six.
fn compare_all(a: str, b: str) -> int {
    var acc = 0
    var i = 0
    while i < 40 {
        acc = 0
        if a < b { acc += 1 }
        if a <= b { acc += 2 }
        if a > b { acc += 4 }
        if a >= b { acc += 8 }
        if a == b { acc += 16 }
        if a != b { acc += 32 }
        i += 1
    }
    return acc
}

#: The unfused twin. Binding the answer to a local stops the peephole folding
#: the comparison into the branch, so this is the OP_LT..OP_NE arm where
#: `compare_all` is OP_JUMP_IF_CMP_FALSE. Both had to be written: with only the
#: unfused one, the shape anyone actually types still declined.
fn compare_all_unfused(a: str, b: str) -> int {
    var acc = 0
    var i = 0
    while i < 40 {
        acc = 0
        let lt = a < b
        let le = a <= b
        let gt = a > b
        let ge = a >= b
        let eq = a == b
        let ne = a != b
        if lt { acc += 1 }
        if le { acc += 2 }
        if gt { acc += 4 }
        if ge { acc += 8 }
        if eq { acc += 16 }
        if ne { acc += 32 }
        i += 1
    }
    return acc
}

#: The folded form: a local against a string constant, which is what a parser
#: or a dispatch on a name actually emits. Every one of the six again, so the
#: operand order is pinned -- the local is the LEFT one, and reversing it would
#: be a silent wrong answer for four of them rather than a crash.
fn compare_all_const(a: str) -> int {
    var acc = 0
    var i = 0
    while i < 40 {
        acc = 0
        if a < "middle" { acc += 1 }
        if a <= "middle" { acc += 2 }
        if a > "middle" { acc += 4 }
        if a >= "middle" { acc += 8 }
        if a == "middle" { acc += 16 }
        if a != "middle" { acc += 32 }
        i += 1
    }
    return acc
}

#: The shape the arm exists for: a comparison of two things the loop is
#: carrying, with real work around it. Counts how many of `words` sort before
#: `pivot`.
fn count_less(words: list[str], pivot: str) -> int {
    var hits = 0
    var i = 0
    while i < words.len() {
        if words[i] < pivot { hits += 1 }
        i += 1
    }
    return hits
}

#: A minimum by `<`, so both operands come out of the same list and neither is
#: a constant the emitter could have anchored on.
fn first_of(words: list[str]) -> str {
    var best = words[0]
    var i = 1
    while i < words.len() {
        if words[i] < best { best = words[i] }
        i += 1
    }
    return best
}

#: The operand is an object that is not a string. It reaches the compiled body
#: -- the entry guard for a raw-object parameter only says "an object" -- so
#: the type guard inside is the only thing between a list and being handed to
#: `jaiStringOrder` as a string.
fn less_any(a: any, b: any) -> int {
    if a < b { return 1 }
    return 0
}

#: `less_any` misses the guard at the two STACK arms only. This one is the
#: OP_JUMP_IF_CMP_LOCAL_K shape -- a local against a string CONSTANT -- which
#: has its own type guard, its own resume point and its own operand order, and
#: writes x0 and x1 by hand. Nothing had ever missed that guard, including the
#: case worth worrying about: a local whose register holds something that is
#: not a pointer at all.
fn k_any(a: any) -> int {
    if a < "middle of the road, and long enough" { return 1 }
    return 0
}

fn warm_k_any() -> void {
    for _r in 0..20000 { let _x = k_any("aardvark and then some more text") }
}

fn build_words() -> list[str] {
    var words: list[str] = []
    var i = 0
    while i < 64 {
        words.push("word" + str(i % 13))
        i += 1
    }
    return words
}

#: Longer than the 32 bytes every construction path in this VM interns up to,
#: so the pair is never interned and `==` cannot be answered by the pointer arm.
fn long_string(tail: str) -> str {
    return "the quick brown fox jumps over the lazy dog " + tail
}

fn warm_pair(a: str, b: str) -> void {
    for _r in 0..400 {
        let _x = compare_all(a, b)
        let _y = compare_all_unfused(a, b)
    }
}

fn warm_const(a: str) -> void {
    for _r in 0..400 { let _x = compare_all_const(a) }
}

fn warm_lists(words: list[str]) -> void {
    for _r in 0..400 {
        let _x = count_less(words, "word5")
        let _y = first_of(words)
    }
}

fn warm_any() -> void {
    for _r in 0..20000 { let _x = less_any("aa", "ab") }
}

#: 1 + 2 + 32: less, less-or-equal, not-equal.
let LESS = 35
#: 4 + 8 + 32: greater, greater-or-equal, not-equal.
let GREATER = 44
#: 2 + 8 + 16: less-or-equal, greater-or-equal, equal.
let SAME = 26

#: Both forms of every comparison, checked against each other as well as
#: against the expected answer -- the fused and unfused arms are separate code
#: and could have disagreed.
fn both_forms(a: str, b: str) -> int {
    let fused = compare_all(a, b)
    assert_eq(compare_all_unfused(a, b), fused)
    return fused
}

fn test_equal_strings_are_equal_and_neither_less_nor_greater() -> void {
    warm_pair("hello world", "hello world")
    assert_eq(both_forms("hello world", "hello world"), SAME)
    #: The same object on both sides, which jaiStringOrder answers without
    #: reading a byte.
    let s = "hello world"
    assert_eq(both_forms(s, s), SAME)
}

fn test_a_prefix_sorts_before_its_extension() -> void {
    warm_pair("hello", "hello world")
    assert_eq(both_forms("hello", "hello world"), LESS)
    assert_eq(both_forms("hello world", "hello"), GREATER)
}

fn test_the_empty_string_sorts_before_everything() -> void {
    warm_pair("", "ab")
    assert_eq(both_forms("", "ab"), LESS)
    assert_eq(both_forms("ab", ""), GREATER)
    assert_eq(both_forms("", ""), SAME)
    #: A one-byte right-hand side goes to the one-byte arm instead, so it is
    #: checked here rather than through the marked helpers -- but the answer
    #: still has to be the same one.
    assert_true("" < "a")
    assert_true("a" > "")
}

fn test_multi_byte_utf8_orders_by_byte() -> void {
    warm_pair("é", "ez")
    #: One scalar against two, the same two BYTES: 0xC3 0xA9 against 0x65 0x7A.
    #: Greater, because the order is the bytes' and not the scalars'.
    assert_eq(both_forms("é", "ez"), GREATER)
    assert_eq(both_forms("é", "ee"), GREATER)
    assert_eq(both_forms("héllo", "héllo"), SAME)
    assert_eq(both_forms("héllo", "hémlo"), LESS)
    #: An emoji is four bytes and one scalar, and sorts after every Latin one.
    assert_eq(both_forms("zz", "😀"), LESS)
}

fn test_a_run_time_built_pair_compares_by_content() -> void {
    let a = long_string("one")
    let b = long_string("two")
    let c = long_string("one")
    warm_pair(a, b)
    assert_eq(both_forms(a, b), LESS)
    assert_eq(both_forms(b, a), GREATER)
    #: Equal by content, two distinct objects, and neither is interned -- the
    #: case the pointer arm cannot answer.
    assert_eq(both_forms(a, c), SAME)
}

fn test_a_local_against_a_string_constant() -> void {
    warm_const("apple")
    assert_eq(compare_all_const("apple"), LESS)
    assert_eq(compare_all_const("zebra"), GREATER)
    assert_eq(compare_all_const("middle"), SAME)
    #: A prefix of the constant, so the answer comes from the length.
    assert_eq(compare_all_const("mid"), LESS)
    assert_eq(compare_all_const("middleman"), GREATER)
    assert_eq(compare_all_const(""), LESS)
    #: Multi-byte, and still by byte: 0xC3 is above every ASCII letter.
    assert_eq(compare_all_const("ée"), GREATER)
}

fn test_a_comparison_over_a_list_of_strings() -> void {
    let words = build_words()
    warm_lists(words)
    var expected = 0
    var i = 0
    while i < words.len() {
        if words[i] < "word5" { expected += 1 }
        i += 1
    }
    assert_eq(count_less(words, "word5"), expected)
    assert_eq(first_of(words), "word0")
}

fn test_a_non_string_operand_raises_what_the_interpreter_raises() -> void {
    warm_any()
    assert_eq(less_any("aa", "ab"), 1)
    assert_eq(less_any("ab", "aa"), 0)
    var xs: list = []
    xs.push(1)
    assert_throws(TypeError, || less_any("aa", xs))
    assert_throws(TypeError, || less_any(xs, "aa"))
    #: And the body still works afterwards: the guard deoptimises, it does not
    #: retire the compiled code.
    assert_eq(less_any("aa", "ab"), 1)
}

pub fn test_the_local_against_a_constant_guard_misses_cleanly() -> void {
    warm_k_any()
    assert_eq(k_any("aardvark and then some more text"), 1, "still answers")
    assert_eq(k_any("zebra, which sorts after the constant"), 0, "and both ways")

    #: An int in the local is the one that mattered: its register holds a
    #: number, not a pointer, and a missed guard would dereference it.
    let xs: list[int] = [1, 2]
    assert_throws(TypeError, || k_any(3), "an int in the local")
    assert_throws(TypeError, || k_any(3.5), "a float")
    assert_throws(TypeError, || k_any(xs), "a list")
    assert_throws(TypeError, || k_any(null), "null")
    assert_throws(TypeError, || k_any(true), "a bool")

    #: And the compiled body keeps working afterwards.
    assert_eq(k_any("aardvark and then some more text"), 1, "after the misses")
}

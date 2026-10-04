#: `if c == "{"` and `if c < "0"` against a one-byte literal, compiled.
#:
#: The peephole fuses a local and a literal into OP_JUMP_IF_CMP_LOCAL_K. Against
#: a ONE-BYTE literal the tier now compares the local's length and its single
#: byte with an immediate (JAITHON_JIT_ONE_BYTE_LOCAL_K) instead of calling
#: jaiStringOrder. Equality must stay false -- without deoptimising -- for an
#: empty string, a longer one and a multi-byte scalar, and true for a
#: one-character string that is not the interned singleton; ordering must
#: follow bytes, and a local that is not a string must still reach the
#: interpreter's own comparison.
# jit-compiles: classify, order
from std.test import assert_eq

let CALLS = 400

fn classify(c: str) -> int {
    if c == "{" { return 1 }
    if c != "[" {
        if c == "\0" { return 3 }
        return 0
    }
    return 2
}

fn order(c: str) -> int {
    if c < "0" { return -1 }
    if c > "9" { return 1 }
    return 0
}

fn eq_any(c: any) -> bool {
    var hits = 0
    var i = 0
    while i < 3 {
        if c == "x" { hits += 1 }
        i += 1
    }
    return hits == 3
}

pub fn test_equality_against_one_byte() -> void {
    var warm = 0
    for _i in 0..CALLS { warm += classify("{") + classify("a") }
    assert_eq(warm, CALLS, "warm")
    assert_eq(classify("{"), 1, "the literal")
    assert_eq(classify("["), 2, "the second literal")
    assert_eq(classify(""), 0, "empty")
    assert_eq(classify("{{"), 0, "longer")
    assert_eq(classify("é"), 0, "a two-byte scalar")
    assert_eq(classify("\0"), 3, "a NUL")
    # A one-character string built at run time, not taken from the table.
    let built = "" + "{"
    assert_eq(classify(built), 1, "a built one-character string")
    let sliced = "x{y"[1:2]
    assert_eq(classify(sliced), 1, "a slice")
}

pub fn test_ordering_against_one_byte() -> void {
    var warm = 0
    for _i in 0..CALLS { warm += order("5") + order("a") }
    assert_eq(warm, CALLS, "warm")
    assert_eq(order("/"), -1, "below")
    assert_eq(order("0"), 0, "the low edge")
    assert_eq(order("9"), 0, "the high edge")
    assert_eq(order(":"), 1, "above")
    assert_eq(order("é"), 1, "a two-byte scalar compares by its bytes")
    assert_eq(order(""), -1, "empty sorts first")
    assert_eq(order("05"), 0, "a longer string starting in range")
}

pub fn test_a_local_that_is_not_a_string() -> void {
    for _i in 0..CALLS { assert_eq(eq_any("x"), true, "a string") }
    assert_eq(eq_any(7), false, "an int")
    assert_eq(eq_any(["x"]), false, "a list")
}

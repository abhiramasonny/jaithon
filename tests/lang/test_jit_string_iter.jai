#: `for c in text`, which is what a lexer is made of.
#:
#: The iterator arm took a list or a range and nothing else, so every character
#: loop declined the whole enclosing body. It needed less than it looked: the
#: iterator is the SAME `jaiIterNew` the interpreter builds, so the only new
#: things a string wants are a proof that it IS a string and an element
#: exemplar.
#:
#: It was a ONE-LINK CHAIN on a hot body -- `JAI_JIT_CHAIN=1` said
#: `_scan_identifier` compiles the moment this is cleared -- which is the
#: ranking that picks work, rather than how often a refusal is printed.
#:
#:   check --no-cache parser.jai   58,787,038 -> 58,064,028 interpreted  -1.2%
#:   check --no-cache lexer.jai    0.416s -> 0.401s                      ~3.6%
#:   check --no-cache parser.jai   0.764s -> 0.759s                      flat
#:
#: The lexer moves and the parser does not, which is the honest split: character
#: loops are the lexer's work.
#:
#: THE EXEMPLAR IS AN INTERNED EMPTY STRING, deliberately, not the source's
#: first character. A string yields one-CHARACTER elements and a character is
#: not always one byte; claiming one byte would send multi-byte text down the
#: one-byte arm, which guards the length and would therefore deopt once per
#: iteration. Empty says "a string" and claims nothing else.
#:
#: Fixing this also turned up a latent bug next door: `jitMakeIter` returned 1
#: for anything that was not a list, and 1 means "the callee threw" -- with no
#: exception set. The VM reports that as `internal error: failed operation
#: raised nothing`, which is how the first draft of this arm announced itself:
#: 104 test failures and no usable message. It now raises properly.
# jit-compiles: count_chars, join_back, first_upper
from std.test import assert_eq, assert_true, assert_throws

#: Above JAI_JIT_THRESHOLD entries, so these reach the compiled tier by a count
#: rather than by the OSR sampler.
let CALLS = 200

fn count_chars(s: str) -> int {
    var n = 0
    for c in s {
        n += 1
    }
    return n
}

fn join_back(s: str) -> str {
    var out = ""
    for c in s {
        out = out + c
    }
    return out
}

fn first_upper(s: str) -> int {
    var seen = 0
    for c in s {
        if c >= "A" and c <= "Z" { seen += 1 }
    }
    return seen
}

fn iter_any(v: any) -> int {
    var n = 0
    for c in v {
        n += 1
    }
    return n
}

pub fn test_counting_characters() -> void {
    var total = 0
    var call = 0
    while call < CALLS {
        total += count_chars("abc")
        call += 1
    }
    assert_eq(total, 3 * CALLS, "three per call")
    assert_eq(count_chars(""), 0, "the empty string")
    assert_eq(count_chars("a"), 1, "one character")
}

pub fn test_characters_not_bytes() -> void {
    #: The whole reason the exemplar claims nothing about length. `héllo` is
    #: five characters in six bytes and `日本語` three in nine.
    var call = 0
    while call < CALLS {
        let _ = count_chars("héllo")
        call += 1
    }
    assert_eq(count_chars("héllo"), 5, "five characters, six bytes")
    assert_eq(count_chars("日本語"), 3, "three characters, nine bytes")
}

pub fn test_the_characters_are_the_right_ones() -> void {
    var call = 0
    while call < CALLS {
        let _ = join_back("abc")
        call += 1
    }
    assert_eq(join_back("abc"), "abc", "round trip")
    assert_eq(join_back("héllo"), "héllo", "and multi-byte round trips")
    assert_eq(join_back(""), "", "the empty string")
}

pub fn test_comparing_the_yielded_characters() -> void {
    var total = 0
    var call = 0
    while call < CALLS {
        total += first_upper("aBcDe")
        call += 1
    }
    assert_eq(total, 2 * CALLS, "two upper-case")
    assert_eq(first_upper("abc"), 0, "none")
    assert_eq(first_upper("ABC"), 3, "all")
}

pub fn test_the_same_loop_over_other_sources() -> void {
    #: The guard has to notice when the source is not a string.
    var call = 0
    while call < CALLS {
        let _ = iter_any("ab")
        call += 1
    }
    assert_eq(iter_any("ab"), 2, "a string")
    assert_eq(iter_any([1, 2, 3]), 3, "a list through the same site")
    assert_throws(TypeError, || iter_any(7), "an int is not iterable")
}

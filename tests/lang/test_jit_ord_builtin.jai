#: `ord(c)` in a loop hot enough to reach the compiled tier.
#:
#: The tier emits a call only for builtins whose result kind it knows without
#: running anything, and `ord` was not one of them -- so every loop containing
#: one declined WHOLE. It was the only builtin tests/bench/json_parse declines
#: on, naming it 141 times, and a scanner that reads its input a character at a
#: time and asks what that character is is the shape that pays: the arm is
#: 2.88x on such a loop and 1.075x on json_parse itself.
#:
#: The emitted form is the inverse of the string-index arm and restricted the
#: same way -- one byte, ASCII. Everything else deoptimises to the interpreter,
#: which is what these check: a multi-byte scalar still answers with its
#: codepoint, and the empty and too-long cases still raise.
# jit-compiles: sum_codepoints, count_digits
from std.test import assert_eq, assert_throws

let ROUNDS = 400

#: Above JAI_JIT_THRESHOLD entries. The marker below names the FUNCTION tier,
#: which compiles at a count of entries; a body merely looped over many times
#: can only be caught by the OSR tier's 250us sampling timer, which a fast
#: machine does not trip -- and that is why these markers had silently stopped
#: holding.
let WARM_CALLS = 200

fn sum_codepoints(text: str, rounds: int) -> int {
    var total = 0
    var round = 0
    while round < rounds {
        var index = 0
        while index < text.len() {
            total += ord(text[index])
            index += 1
        }
        round += 1
    }
    return total
}

fn count_digits(text: str, rounds: int) -> int {
    var seen = 0
    var round = 0
    while round < rounds {
        var index = 0
        while index < text.len() {
            let code = ord(text[index])
            if code >= 48 and code <= 57 { seen += 1 }
            index += 1
        }
        round += 1
    }
    return seen
}

pub fn test_ascii_matches_the_interpreter() -> void {
    let text = "abc123"
    #: 97 + 98 + 99 + 49 + 50 + 51
    var call = 0
    while call < WARM_CALLS {
        assert_eq(sum_codepoints(text, 1), 444)
        assert_eq(count_digits(text, 1), 3)
        call += 1
    }
    assert_eq(sum_codepoints(text, ROUNDS), 444 * ROUNDS)
    assert_eq(count_digits(text, ROUNDS), 3 * ROUNDS)
}

pub fn test_the_boundary_byte_is_not_ascii() -> void {
    #: 127 is the last codepoint the emitted form may answer for; 128 is the
    #: first that needs decoding, and both have to give the same answer.
    assert_eq(ord("\u{7F}"), 127)
    assert_eq(ord("\u{80}"), 128)
}

pub fn test_multibyte_scalars_still_decode() -> void {
    var total = 0
    var round = 0
    #: Hot enough to compile, and every iteration takes the bail.
    while round < ROUNDS {
        total += ord("é") + ord("→") + ord("😀")
        round += 1
    }
    assert_eq(total, (233 + 8594 + 128512) * ROUNDS)
}

pub fn test_the_error_cases_still_raise() -> void {
    assert_throws(ValueError, fn() -> void { let _ = ord("") }, "empty")
    assert_throws(ValueError, fn() -> void { let _ = ord("ab") }, "two chars")
}

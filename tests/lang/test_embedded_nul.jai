#: A `str` is length-prefixed, so NUL is an ordinary byte (spec §2.1, §1.7).
#:
#: `"a\0b".len()` answered 1 and `"\0".len()` answered 0, because two places on
#: the path from the lexer to the constant pool measured the cooked text with
#: the C library's idea of a string: `jaiTokenStringValue` returned `strlen` of
#: the arena copy instead of the size it was allocated with, and the parser
#: interned it with a `strndup` that stopped at the first NUL while recording
#: the untruncated length — so `"a\0b"` was stored as two bytes and read back
#: as three, the third coming from whatever followed it in the arena.
#:
#: Everything below is a length, an index or a search, which is exactly the
#: surface those two defects reached.
from std.test import assert_eq, assert_true, assert_false

fn test_len_counts_the_nul() -> void {
    assert_eq("a\0b".len(), 3)
    assert_eq("\0".len(), 1)
    assert_eq("\0\0\0".len(), 3)
    assert_eq(len("a\0b"), 3)
    assert_eq("".len(), 0)
}

fn test_the_bytes_are_the_ones_written() -> void {
    assert_eq("a\0b".bytes(), [97, 0, 98])
    assert_eq("\0z".bytes(), [0, 122])
    assert_eq("a\0".bytes(), [97, 0])
    assert_eq("a\0b".bytes()[2], 98)
}

fn test_indexing_reaches_past_the_nul() -> void {
    let s = "a\0b"
    assert_eq(s[0], "a")
    assert_eq(s[2], "b")
    assert_eq(s[-1], "b")
    assert_eq(s[1].len(), 1)
    assert_eq(s[1:].len(), 2)
    assert_eq(s[:2].len(), 2)
}

fn test_search_and_compare_see_the_whole_string() -> void {
    let s = "a\0b"
    assert_true(s.contains("b"))
    assert_eq(s.find("b"), 2)
    assert_eq(s.find("\0"), 1)
    assert_true(s.starts_with("a\0"))
    assert_true(s.ends_with("\0b"))
    assert_true(s == "a\0b")
    assert_false(s == "a")
    assert_false(s == "a\0")
}

fn test_building_new_strings_keeps_the_length() -> void {
    let s = "a\0b"
    assert_eq((s + "c").len(), 4)
    assert_eq(("c" + s).len(), 4)
    assert_eq(s.upper().len(), 3)
    assert_eq(s.repeat(2).len(), 6)
    assert_eq(f"{s}".len(), 3)
    assert_eq(str(s).len(), 3)
    assert_eq(s.replace("\0", "-"), "a-b")
    assert_eq(s.split("\0"), ["a", "b"])
    assert_eq([c for c in s].len(), 3)
}

fn test_a_nul_string_works_as_a_key() -> void {
    let s = "a\0b"
    let d = {s: 1, "a": 2}
    assert_eq(d[s], 1)
    assert_eq(d["a"], 2)
    assert_eq(d.len(), 2)
    assert_true(s in d)
}

#: Folding happens in the checker, which took its own copy of the characters
#: through the same truncating helper.
fn test_constant_folding_keeps_the_nul() -> void {
    let folded = "a\0" + "b"
    assert_eq(folded.len(), 3)
    assert_eq(folded, "a\0b")
}

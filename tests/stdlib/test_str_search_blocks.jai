#: `find`, `count`, `replace`, `split` and `in` against a reference that never
#: calls the native search.
#:
#: The native search compares sixteen candidate positions at a time on the
#: needle's first and last byte (src/common/memory.c, jaiMemFind) and hands
#: what is left of the haystack to a scalar loop. Every way that can go wrong
#: is a boundary: a match straddling two blocks, one ending on the very last
#: byte, a needle as long as a block or longer, a haystack shorter than one
#: block plus the needle, a first byte that is everywhere, embedded NULs, and
#: multi-byte text, where `find` must still answer in characters. Each case is
#: placed at every offset across two blocks so no alignment is left untried.
from std.test import assert_eq, assert_true

fn ref_find(hay: str, needle: str) -> int {
    let n = hay.len()
    let m = needle.len()
    var i = 0
    while i + m <= n {
        if hay[i:i + m] == needle { return i }
        i += 1
    }
    return -1
}

fn ref_count(hay: str, needle: str) -> int {
    let n = hay.len()
    let m = needle.len()
    var hits = 0
    var i = 0
    while i + m <= n {
        if hay[i:i + m] == needle {
            hits += 1
            i += m
        } else {
            i += 1
        }
    }
    return hits
}

fn check(hay: str, needle: str) -> void {
    assert_eq(hay.find(needle), ref_find(hay, needle), f"find {needle} in {hay.len()}")
    assert_eq(hay.count(needle), ref_count(hay, needle), f"count {needle} in {hay.len()}")
    assert_eq(needle in hay, ref_find(hay, needle) >= 0, f"in {needle}")
}

pub fn test_every_offset_across_two_blocks() -> void {
    for needle in ["ab", "abc", "abcdefghijklmnop", "abcdefghijklmnopq", "xyzzy"] {
        for pad in 0..40 {
            let hay = "." * pad + needle + "." * (40 - pad)
            check(hay, needle)
            check(hay + needle, needle)
            check("." * pad + needle, needle)
        }
    }
}

pub fn test_one_byte_needles() -> void {
    # `count` of one byte is a tally of its own, not the search loop.
    for pad in 0..40 {
        let hay = ",x" * pad + ",,"
        check(hay, ",")
        check(hay, "x")
        check(hay, "q")
        assert_eq(hay.count(",", pad), ref_count(hay[pad:], ","), f"count from {pad}")
    }
}

pub fn test_a_first_byte_that_is_everywhere() -> void {
    let hay = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab" * 5
    check(hay, "aab")
    check(hay, "ab")
    check(hay, "aaaaab")
    check(hay, "ba")
    check(hay, "abc")
    assert_eq(hay.replace("ab", "X").count("X"), 5, "replace")
}

pub fn test_short_haystacks() -> void {
    for n in 0..20 {
        let hay = "xy" * n
        check(hay, "yx")
        check(hay, "xyx")
        check(hay, "yy")
    }
    check("", "ab")
    check("a", "ab")
    check("ab", "ab")
}

pub fn test_embedded_nul() -> void {
    let hay = "abc\0def\0\0ghi" * 4
    check(hay, "\0d")
    check(hay, "\0\0")
    check(hay, "f\0\0g")
    check(hay, "i\0")
    assert_eq(hay.split("\0\0").len(), 5, "split on a NUL pair")
}

pub fn test_multibyte_text_answers_in_characters() -> void {
    let hay = "été 中文 café été " * 3
    assert_eq(hay.find("café"), 7, "find in characters")
    assert_eq(hay.find("中"), 4, "a three-byte scalar")
    assert_eq(hay.count("été"), 6, "count")
    assert_true("中文" in hay, "in")
    assert_true(not ("文中" in hay), "not in")
    assert_eq(hay.replace("é", "e").count("ete"), 6, "replace")
}

pub fn test_split_and_replace_agree_with_count() -> void {
    let hay = "w12 w49 w497 x w49w49 " * 30
    assert_eq(hay.split("w49").len(), hay.count("w49") + 1, "split pieces")
    assert_eq(hay.replace("w49", "").len(), hay.len() - 3 * hay.count("w49"), "replace length")
}

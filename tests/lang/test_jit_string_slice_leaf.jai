#: `s[a:b]` on a string from compiled code, which goes through
#: jaiStringSliceLeaf -- a leaf call answering a one-byte slice from the ASCII
#: table and a short one the intern table already holds -- and falls back to
#: the descriptor call to jitGetSlice for everything else: an empty slice, a
#: long one, a new one, a source not yet known to be ASCII.
#:
#: Each helper is called more than JAI_JIT_THRESHOLD times so the
#: whole-function tier compiles it on call count alone; every answer is the
#: interpreter's, which `JAITHON_NO_JIT=1` checks on the same file.
# jit-compiles: cut_words, cut_range, cut_head, cut_tail
from std.test import assert_eq, assert_true

let CALLS = 100

fn cut_words(text: str) -> list[str] {
    var out = []
    var start = 0
    var at = 0
    let n = text.len()
    while at < n {
        if text[at] == " " {
            if at > start { out.push(text[start:at]) }
            start = at + 1
        }
        at += 1
    }
    if start < n { out.push(text[start:n]) }
    return out
}

fn cut_range(s: str, a: int, b: int) -> str {
    return s[a:b]
}

fn cut_head(s: str, b: int) -> str {
    return s[:b]
}

fn cut_tail(s: str, a: int) -> str {
    return s[a:]
}

pub fn test_words_out_of_a_text() -> void {
    let text = "the cat sat on the mat the end"
    var call = 0
    while call < CALLS {
        let words = cut_words(text)
        assert_eq(words.len(), 8)
        assert_eq(words[0], "the")
        assert_eq(words[7], "end")
        call += 1
    }
    #: Repeated words come back as the one interned string.
    let again = cut_words(text)
    assert_true(again[0] is again[4], "interned")
}

pub fn test_bounds_clamp_like_the_interpreter() -> void {
    let s = "abcdefghij"
    var call = 0
    while call < CALLS {
        assert_eq(cut_range(s, 2, 5), "cde")
        assert_eq(cut_range(s, -3, -1), "hi")
        assert_eq(cut_range(s, -100, 2), "ab")
        assert_eq(cut_range(s, 8, 100), "ij")
        assert_eq(cut_range(s, 5, 5), "")
        assert_eq(cut_range(s, 7, 3), "")
        assert_eq(cut_range(s, 100, 200), "")
        assert_eq(cut_range(s, 4, 5), "e")
        assert_eq(cut_range(s, -1, 10), "j")
        call += 1
    }
}

pub fn test_missing_bounds() -> void {
    let s = "abcdefghij"
    var call = 0
    while call < CALLS {
        assert_eq(cut_head(s, 3), "abc")
        assert_eq(cut_head(s, -8), "ab")
        assert_eq(cut_tail(s, 7), "hij")
        assert_eq(cut_tail(s, -2), "ij")
        assert_eq(cut_tail(s, 20), "")
        call += 1
    }
}

pub fn test_long_and_new_slices() -> void {
    let long = "0123456789abcdefghijklmnopqrstuvwxyz0123456789"
    var call = 0
    while call < CALLS {
        #: Past the intern limit: always the general path.
        assert_eq(cut_range(long, 0, 40).len(), 40)
        #: A slice nobody has made before: a miss, then the general path.
        let fresh = cut_range(f"{long}{call}", 30, 46 + (if call < 10 { 1 } else { 2 }))
        assert_eq(fresh, f"{long}{call}"[30:46 + (if call < 10 { 1 } else { 2 })])
        call += 1
    }
}

pub fn test_non_ascii_sources() -> void {
    let s = "héllo wörld"
    var call = 0
    while call < CALLS {
        assert_eq(cut_range(s, 0, 5), "héllo")
        assert_eq(cut_range(s, 6, 11), "wörld")
        assert_eq(cut_range(s, 1, 2), "é")
        call += 1
    }
    assert_eq(cut_words(s), ["héllo", "wörld"])
}

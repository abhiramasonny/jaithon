#: `upper` and `lower` on ASCII text, at the edges of the letter ranges.
#:
#: The ASCII path maps a byte by `(c - lo) < 26 ? c ^ 0x20 : c`, so the bytes
#: either side of each range -- `@ [ ` { ` and the digits -- are where a wrong
#: bound would show. Checked byte by byte against `ord` arithmetic over every
#: ASCII byte, long enough to cross the sixteen-byte steps the loops take.
from std.test import assert_eq

fn every_ascii() -> str {
    var out = ""
    var round = 0
    while round < 3 {
        var c = 1
        while c < 128 {
            out = out + chr(c)
            c += 1
        }
        round += 1
    }
    return out
}

pub fn test_upper_matches_byte_arithmetic() -> void {
    let text = every_ascii()
    let up = text.upper()
    assert_eq(up.len(), text.len(), "same length")
    var i = 0
    while i < text.len() {
        let c = ord(text[i])
        let want = if c >= 97 and c <= 122 { c - 32 } else { c }
        assert_eq(ord(up[i]), want, f"byte {c}")
        i += 1
    }
}

pub fn test_lower_matches_byte_arithmetic() -> void {
    let text = every_ascii()
    let low = text.lower()
    var i = 0
    while i < text.len() {
        let c = ord(text[i])
        let want = if c >= 65 and c <= 90 { c + 32 } else { c }
        assert_eq(ord(low[i]), want, f"byte {c}")
        i += 1
    }
}

pub fn test_mixed_with_multibyte() -> void {
    assert_eq("abc é XYZ".upper(), "ABC É XYZ", "upper with an accent")
    assert_eq("ABC É xyz".lower(), "abc é xyz", "lower with an accent")
    assert_eq("".upper(), "", "empty")
}

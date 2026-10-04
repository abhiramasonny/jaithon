#: `s.split(c)` for a one-byte separator, against a reference built by hand.
#:
#: The one-byte split scans sixteen bytes at a time and hands what is left to a
#: byte loop (builtins_str_split.c, splitOneByte). Its answers must not depend
#: on where the separators fall relative to those blocks: leading, trailing and
#: back-to-back separators, a separator on either side of every block
#: boundary, NUL as the separator, multi-byte text between separators, and a
#: string with no separator at all.
from std.test import assert_eq

fn ref_split(s: str, sep: str) -> list[str] {
    var out: list[str] = []
    var start = 0
    var i = 0
    let n = s.len()
    while i < n {
        if s[i] == sep {
            out.push(s[start:i])
            start = i + 1
        }
        i += 1
    }
    out.push(s[start:n])
    return out
}

fn check(s: str, sep: str) -> void {
    let got = s.split(sep)
    let want = ref_split(s, sep)
    assert_eq(got.len(), want.len(), f"piece count for {s.len()} bytes")
    var i = 0
    while i < got.len() {
        assert_eq(got[i], want[i], f"piece {i}")
        i += 1
    }
}

pub fn test_every_offset() -> void {
    for pad in 0..40 {
        check("x" * pad + "," + "y" * (40 - pad), ",")
        check("," + "x" * pad + ",", ",")
        check("x" * pad + ",,", ",")
    }
}

pub fn test_edges() -> void {
    check("", ",")
    check(",", ",")
    check(",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,", ",")
    check("no separator here, really none", ";")
    check("a,b,c", ",")
}

pub fn test_nul_and_multibyte() -> void {
    check("ab\0cd\0\0ef\0" * 9, "\0")
    check("été,中文,café," * 7, ",")
    assert_eq("été,中文".split(",")[1], "中文", "a multi-byte field")
}

pub fn test_maxsplit_and_rsplit_unchanged() -> void {
    assert_eq("a,b,c,d".split(",", 2).len(), 3, "maxsplit")
    assert_eq("a,b,c,d".rsplit(",", 1)[0], "a,b,c", "rsplit")
}

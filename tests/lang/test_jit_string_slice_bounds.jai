#: String slices cut by compiled code through the leaf (JAITHON_JIT_SLICE_LEAF).
#:
#: The leaf takes `s[a:b]`, `s[a:]` and `s[:b]` on ASCII text and declines
#: everything else back to the general path: multi-byte text, an empty
#: result, a string whose scalar count was never asked for. Its clamping must
#: be the general path's exactly -- negative bounds, bounds past either end,
#: a stop before the start -- so every case here is checked against the same
#: slice taken in the test body itself, which runs once and stays interpreted.
# jit-compiles: cut, cut_from, cut_to
from std.test import assert_eq

let CALLS = 300

fn cut(s: str, a: int, b: int) -> str { return s[a:b] }

fn cut_from(s: str, a: int) -> str { return s[a:] }

fn cut_to(s: str, b: int) -> str { return s[:b] }

pub fn test_bounds_match_the_general_path() -> void {
    let text = "hello, world"
    var warm = 0
    for _i in 0..CALLS {
        warm += cut(text, 1, 4).len() + cut_from(text, 7).len() + cut_to(text, 5).len()
    }
    assert_eq(warm, CALLS * (3 + 5 + 5), "warm")
    let bounds = [-20, -13, -12, -5, -1, 0, 1, 3, 5, 11, 12, 13, 40]
    for a in bounds {
        assert_eq(cut_from(text, a), text[a:], f"from {a}")
        assert_eq(cut_to(text, a), text[:a], f"to {a}")
        for b in bounds {
            assert_eq(cut(text, a, b), text[a:b], f"{a}:{b}")
        }
    }
}

pub fn test_non_ascii_and_embedded_nul() -> void {
    for _i in 0..CALLS { cut("abcdef", 1, 3) }
    let wide = "aé中bc"
    assert_eq(cut(wide, 1, 3), "é中", "multi-byte, in characters")
    assert_eq(cut(wide, -2, 5), "bc", "negative over multi-byte")
    let nul = "ab\0cd"
    assert_eq(cut(nul, 1, 4), "b\0c", "an embedded NUL")
    assert_eq(cut(nul, 2, 3), "\0", "the NUL alone")
    assert_eq(cut("", 0, 3), "", "empty source")
    assert_eq(cut("x", 0, 1), "x", "one character")
}

pub fn test_results_are_ordinary_strings() -> void {
    var seen: dict[str, int] = {}
    let text = "key1 key2 key1 key3 key1"
    for _r in 0..CALLS {
        var i = 0
        while i + 4 <= text.len() {
            let word = cut(text, i, i + 4)
            if word.starts_with("key") { seen[word] = seen.get(word, 0) + 1 }
            i += 5
        }
    }
    assert_eq(seen["key1"], 3 * CALLS, "dict keys from slices")
    assert_eq(seen.len(), 3, "three distinct keys")
}

#: Loop-invariant string facts and headers, proved once at an OSR loop head.
#:
#: `s[i]` used to prove, per character, that `s` is a string and that every
#: scalar in it is one byte, and reload `chars` and `length`; `c == first`
#: proved per iteration that `first` is an interned string. When `s` and
#: `first` are locals the loop never writes, all of that is a loop invariant:
#: the head proves it once (JAITHON_JIT_STR_FACTS) and, in a loop that calls
#: nothing, keeps the header in registers (JAITHON_JIT_STR_HOIST).
#:
#: What can go wrong, and what each test below pins down: the same compiled
#: loop entered later with a string that is NOT all one byte, with a comparand
#: that is NOT interned, with something that is not a string at all, with
#: embedded NUL bytes, and with the local reassigned inside the loop. Every
#: answer is checked against one computed a different way -- by slicing --
#: so a wrong fast path cannot agree with itself.
from std.test import assert_eq

#: Long enough that the sampler reaches the loop and compiles it.
let REPEAT = 4000

fn make_text(unit: str, times: int) -> str {
    var parts = []
    for _i in 0..times { parts.push(unit) }
    return "".join(parts)
}

#: The shape tests/bench/str_search measures: a call-free scan whose header
#: can be hoisted into registers.
fn count_scan(text: str, needle: str) -> int {
    let n = text.len()
    let m = needle.len()
    if m == 0 { return 0 }
    let first = needle[0]
    var hits = 0
    var i = 0
    while i + m <= n {
        if text[i] == first {
            var k = 1
            while k < m and text[i + k] == needle[k] { k += 1 }
            if k == m { hits += 1 }
        }
        i += 1
    }
    return hits
}

#: The reference: no character indexing at all.
fn count_slices(text: str, needle: str) -> int {
    let n = text.len()
    let m = needle.len()
    if m == 0 { return 0 }
    var hits = 0
    var i = 0
    while i + m <= n {
        if text[i:i + m] == needle { hits += 1 }
        i += 1
    }
    return hits
}

fn note(log: list[int], v: int) -> void { log.push(v) }

#: A loop that CALLS, so nothing can stay in a caller-saved register: only the
#: register-free facts apply.
fn spaces_with_calls(text: str, log: list[int]) -> int {
    var count = 0
    var at = 0
    let n = text.len()
    while at < n {
        if text[at] == " " {
            count += 1
            note(log, at)
        }
        at += 1
    }
    return count
}

#: The local is reassigned inside the loop, so neither may be hoisted.
fn alternating(a: str, b: str, rounds: int) -> int {
    var s = a
    var total = 0
    var r = 0
    while r < rounds {
        var i = 0
        while i < s.len() {
            if s[i] == "x" { total += 1 }
            i += 1
        }
        s = if r % 2 == 0 { b } else { a }
        r += 1
    }
    return total
}

fn scan_any(v: any, c: str) -> int {
    var hits = 0
    var i = 0
    while i < v.len() {
        if v[i] == c { hits += 1 }
        i += 1
    }
    return hits
}

pub fn test_ascii_scan_agrees_with_slicing() -> void {
    let text = make_text("abcdbadcabcd", REPEAT)
    for needle in ["abcd", "a", "dcab", "zz", "abcdbadcabcdabcd"] {
        assert_eq(count_scan(text, needle), count_slices(text, needle), f"needle {needle}")
    }
}

pub fn test_non_ascii_text_after_ascii() -> void {
    let ascii = make_text("abcdbadc", REPEAT)
    let wide = make_text("abécd中abcd", REPEAT)
    # Warm on ASCII so the compiled form is built for one-byte text...
    for _r in 0..3 { assert_eq(count_scan(ascii, "abcd"), count_slices(ascii, "abcd"), "warm") }
    # ...then enter it with text whose scalars are not one byte each.
    assert_eq(count_scan(wide, "abcd"), count_slices(wide, "abcd"), "wide text")
    assert_eq(count_scan(wide, "écd"), count_slices(wide, "écd"), "wide needle")
    assert_eq(count_scan(ascii, "abcd"), count_slices(ascii, "abcd"), "back to ascii")
}

pub fn test_embedded_nul_bytes() -> void {
    let text = make_text("ab\0cd\0\0abcd", REPEAT)
    assert_eq(count_scan(text, "abcd"), count_slices(text, "abcd"), "around NULs")
    assert_eq(count_scan(text, "\0"), count_slices(text, "\0"), "NUL needle")
    assert_eq(count_scan(text, "\0\0a"), count_slices(text, "\0\0a"), "NUL run")
}

pub fn test_empty_and_short() -> void {
    assert_eq(count_scan("", "a"), 0, "empty text")
    assert_eq(count_scan("abc", ""), 0, "empty needle")
    assert_eq(count_scan("ab", "abc"), 0, "needle longer than text")
    assert_eq(count_scan("a", "a"), 1, "one character")
}

pub fn test_a_comparand_that_is_not_interned() -> void {
    # Over the run-time interning length, so this one is never interned.
    let long = "x" * 40
    let text = make_text(long, 200)
    var expect = 0
    var i = 0
    while i + 40 <= text.len() {
        if text[i:i + 40] == long { expect += 1 }
        i += 1
    }
    assert_eq(count_scan(text, long), expect, "a 40-byte needle")
}

pub fn test_facts_in_a_loop_that_calls() -> void {
    let text = make_text("ab cd ", REPEAT)
    var log: list[int] = []
    let got = spaces_with_calls(text, log)
    assert_eq(got, 2 * REPEAT, "two spaces per unit")
    assert_eq(log.len(), got, "one call per space")
    assert_eq(log[1], 5, "second space")
    let wide = make_text("é 中 ", REPEAT)
    var log2: list[int] = []
    assert_eq(spaces_with_calls(wide, log2), 2 * REPEAT, "wide text after warm")
}

pub fn test_reassigned_local() -> void {
    let a = make_text("xax", 300)
    let b = make_text("béx", 300)
    # 600 x per round of `a`, 300 per round of `b`; rounds alternate a, b, a...
    assert_eq(alternating(a, b, 40), 20 * 600 + 20 * 300, "alternating")
}

pub fn test_not_a_string_later() -> void {
    let text = make_text("qxqq", REPEAT)
    assert_eq(scan_any(text, "x"), REPEAT, "a string")
    assert_eq(scan_any(text, "x"), REPEAT, "again")
    let xs = ["x", "y", "x"]
    assert_eq(scan_any(xs, "x"), 2, "a list of strings")
}

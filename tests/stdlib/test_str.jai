#: `std.str` — text transformation and distance. Indexing is by scalar, not by
#: byte, so every case here has a unicode counterpart: a helper that is right
#: for ASCII and wrong for the rest is worse than no helper.
import std.str as str

from std.test import assert_eq, assert_true, assert_false, assert_near, assert_throws

fn test_builtin_string_indexing_counts_scalars_not_bytes() -> void {
    let text = "héllo"
    assert_eq(text.len(), 5)
    assert_eq(text[1], "é")
    assert_eq(text[0:2], "hé")
    assert_eq(text.chars(), ["h", "é", "l", "l", "o"])
    assert_eq(text.to_bytes().len(), 6)  # é is two bytes
}

fn test_repeat_rejects_negative_and_shortcuts_zero() -> void {
    assert_eq(str.repeat("ab", 3), "ababab")
    assert_eq(str.repeat("ab", 0), "")
    assert_eq(str.repeat("", 5), "")
    assert_throws(ValueError, || str.repeat("ab", -1))
}

fn test_wrap_breaks_on_spaces_and_splits_long_words() -> void {
    assert_eq(str.wrap("the quick brown fox", 10), ["the quick", "brown fox"])
    assert_eq(str.wrap("", 10), [])
    assert_eq(str.wrap("   ", 10), [])
    assert_eq(str.wrap("abcdefgh", 3), ["abc", "def", "gh"])
    assert_throws(ValueError, || str.wrap("x", 0))
}

fn test_indent_leaves_blank_lines_alone_by_default() -> void {
    assert_eq(str.indent("a\nb", "  "), "  a\n  b")
    assert_eq(str.indent("a\n\nb", "> "), "> a\n\n> b")
    assert_eq(str.indent("a\n\nb", "> ", true), "> a\n> \n> b")
}

fn test_dedent_removes_only_the_shared_margin() -> void {
    assert_eq(str.dedent("    a\n      b"), "a\n  b")
    assert_eq(str.dedent("a\nb"), "a\nb")
    assert_eq(str.dedent("  a\n\n  b"), "a\n\nb")
    assert_eq(str.dedent(""), "")
}

fn test_common_prefix_and_suffix() -> void {
    assert_eq(str.common_prefix("prefix", "president"), "pre")
    assert_eq(str.common_prefix("abc", ""), "")
    assert_eq(str.common_prefix("abc", "abc"), "abc")
    assert_eq(str.common_suffix("testing", "running"), "ing")
    assert_eq(str.common_suffix("abc", "xyz"), "")
}

fn test_case_conversions_round_trip_through_words() -> void {
    assert_eq(str.to_snake("HTTPServerError"), "http_server_error")
    assert_eq(str.to_kebab("someValue"), "some-value")
    assert_eq(str.to_pascal("some_value"), "SomeValue")
    assert_eq(str.to_camel("some_value"), "someValue")
    assert_eq(str.to_screaming_snake("someValue"), "SOME_VALUE")
    assert_eq(str.to_snake(""), "")
}

fn test_levenshtein_is_a_metric() -> void {
    assert_eq(str.levenshtein("kitten", "sitting"), 3)
    assert_eq(str.levenshtein("", "abc"), 3)
    assert_eq(str.levenshtein("abc", "abc"), 0)
    # Symmetry, and the triangle inequality on a case that would catch a
    # transposed loop bound.
    assert_eq(str.levenshtein("flaw", "lawn"), str.levenshtein("lawn", "flaw"))
    assert_true(str.levenshtein("a", "c") <= str.levenshtein("a", "b") + str.levenshtein("b", "c"))
}

fn test_damerau_counts_a_transposition_as_one_edit() -> void {
    assert_eq(str.levenshtein("ca", "ac"), 2)
    assert_eq(str.damerau_levenshtein("ca", "ac"), 1)
}

fn test_hamming_requires_equal_lengths() -> void {
    assert_eq(str.hamming("karolin", "kathrin"), 3)
    assert_eq(str.hamming("", ""), 0)
    assert_throws(ValueError, || str.hamming("ab", "abc"))
}

fn test_jaro_and_jaro_winkler_agree_at_the_extremes() -> void {
    assert_near(str.jaro("abc", "abc"), 1.0)
    assert_near(str.jaro("abc", "xyz"), 0.0)
    assert_near(str.jaro_winkler("abc", "abc"), 1.0)
    # A shared prefix can only help.
    assert_true(str.jaro_winkler("martha", "marhta") >= str.jaro("martha", "marhta"))
}

fn test_longest_common_subsequence_and_substring_differ() -> void {
    assert_eq(str.longest_common_subsequence("ABCBDAB", "BDCABA"), "BCBA")
    assert_eq(str.longest_common_substring("ABABC", "BABCA"), "BABC")
    assert_eq(str.longest_common_subsequence("", "abc"), "")
    assert_eq(str.longest_common_substring("abc", "xyz"), "")
}

fn test_is_palindrome_honours_its_flags() -> void {
    assert_true(str.is_palindrome("racecar"))
    assert_false(str.is_palindrome("Racecar"))
    assert_true(str.is_palindrome("Racecar", true))
    assert_true(str.is_palindrome(""))
    assert_true(str.is_palindrome("a"))
}

fn test_rotate_is_modular() -> void {
    assert_eq(str.rotate("abcdef", 2), "cdefab")
    assert_eq(str.rotate("abcdef", -2), "efabcd")
    assert_eq(str.rotate("abc", 3), "abc")
    assert_eq(str.rotate("", 4), "")
}

fn test_expand_substitutes_and_can_default() -> void {
    assert_eq(str.expand("hi {name}", {"name": "ada"}), "hi ada")
    assert_eq(str.expand("{a}{b}", {"a": "1", "b": "2"}), "12")
    assert_eq(str.expand("{missing}", {}, "?"), "?")
    assert_throws(KeyError, || str.expand("{missing}", {}))
}

fn test_escape_round_trips_through_unescape() -> void {
    let raw = "tab\there\nnewline \"quoted\" back\\slash"
    assert_eq(str.unescape(str.escape(raw)), raw)
    assert_eq(str.escape("\n"), "\\n")
    assert_eq(str.quote("a\"b"), "\"a\\\"b\"")
}

fn test_escape_html_covers_every_metacharacter() -> void {
    assert_eq(str.escape_html("<a href=\"x\">&</a>"), "&lt;a href=&quot;x&quot;&gt;&amp;&lt;/a&gt;")
    assert_eq(str.escape_html("plain"), "plain")
}

fn test_to_int_returns_null_rather_than_raising() -> void {
    assert_eq(str.to_int("42"), 42)
    assert_eq(str.to_int("-42"), -42)
    assert_eq(str.to_int("ff", 16), 255)
    assert_eq(str.to_int("1010", 2), 10)
    assert_true(str.to_int("") == null)
    assert_true(str.to_int("12x") == null)
    assert_true(str.to_int("2", 2) == null)
}

fn test_to_float_returns_null_rather_than_raising() -> void {
    assert_near(str.to_float("1.5") ?? 0.0, 1.5)
    assert_near(str.to_float("-2e3") ?? 0.0, -2000.0)
    assert_true(str.to_float("") == null)
    assert_true(str.to_float("1.2.3") == null)
}

fn test_ascii_helpers_see_beyond_ascii() -> void {
    assert_true(str.is_ascii("plain"))
    assert_false(str.is_ascii("café"))
    assert_eq(str.to_ascii("café"), "cafe")  # transliterated
    assert_eq(str.to_ascii("日"), "?")        # no ASCII spelling
    assert_eq(str.to_ascii("日", ""), "")
}

fn test_normalize_whitespace_collapses_and_trims() -> void {
    assert_eq(str.normalize_whitespace("  a \t\n b  "), "a b")
    assert_eq(str.normalize_whitespace(""), "")
    assert_eq(str.normalize_whitespace("   "), "")
}

fn test_string_builder_beats_repeated_concatenation() -> void {
    let builder = str.StringBuilder()
    builder.push("a")
    builder.push("b")
    assert_eq(builder.build(), "ab")
    assert_eq(builder.len(), 2)
}

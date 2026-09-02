#: `safe_load` and `safe_load_all` against PyYAML, which is where every
#: expected value came from -- except where the core schema parts from
#: YAML 1.1, and those cases say so.
import std.math
import jaiyaml as yaml
from jaiyaml import ComposerError, ConstructorError, ParserError, ReaderError, ScannerError
from jaiyaml import YAMLError
from std.test import assert_eq, assert_false, assert_near, assert_throws, assert_true

fn test_scalars_resolve_by_the_core_schema() -> void {
    assert_eq(yaml.safe_load("12"), 12)
    assert_eq(type_of(yaml.safe_load("12")), "int")
    assert_eq(yaml.safe_load("-7"), -7)
    assert_eq(yaml.safe_load("+7"), 7)
    assert_eq(yaml.safe_load("0x1F"), 31)
    assert_eq(yaml.safe_load("0o17"), 15)
    assert_eq(yaml.safe_load("010"), 10)
    assert_near(yaml.safe_load("1.5"), 1.5)
    assert_near(yaml.safe_load(".5"), 0.5)
    assert_near(yaml.safe_load("5."), 5.0)
    assert_near(yaml.safe_load("1e3"), 1000.0)
    assert_near(yaml.safe_load("-2.5E-2"), -0.025)
    assert_eq(type_of(yaml.safe_load("1.0")), "float")
    assert_true(yaml.safe_load(".inf") == math.INF)
    assert_true(yaml.safe_load("-.Inf") == -math.INF)
    assert_true(math.is_nan(yaml.safe_load(".NaN")))
    assert_true(yaml.safe_load("true"))
    assert_true(yaml.safe_load("True"))
    assert_false(yaml.safe_load("FALSE"))
    assert_true(yaml.safe_load("null") == null)
    assert_true(yaml.safe_load("~") == null)
    assert_true(yaml.safe_load("Null") == null)
    assert_true(yaml.safe_load("") == null)
    assert_eq(yaml.safe_load("hello"), "hello")
}

#: PyYAML reads YAML 1.1: `yes` is a bool there, `1e3` and `0o17` are
#: strings, `010` is octal 8, `1_000` is an int and `1:30` is ninety. All of
#: those are plain strings under the core schema, except `010` which is ten.
fn test_yaml_1_1_spellings_are_strings() -> void {
    assert_eq(yaml.safe_load("yes"), "yes")
    assert_eq(yaml.safe_load("off"), "off")
    assert_eq(yaml.safe_load("1_000"), "1_000")
    assert_eq(yaml.safe_load("1:30"), "1:30")
    assert_eq(yaml.safe_load("0b101"), "0b101")
    assert_eq(yaml.safe_load("2001-12-14"), "2001-12-14")
    assert_eq(yaml.safe_load("+0x1F"), "+0x1F")
    assert_eq(yaml.safe_load("1e"), "1e")
    assert_eq(yaml.safe_load("."), ".")
    assert_eq(yaml.safe_load("-"), [null])
    assert_eq(yaml.safe_load("- -"), [[null]])
}

fn test_quoted_scalars_are_always_strings() -> void {
    assert_eq(yaml.safe_load("'12'"), "12")
    assert_eq(yaml.safe_load("\"true\""), "true")
    assert_eq(yaml.safe_load("'null'"), "null")
    assert_eq(yaml.safe_load("''"), "")
}

fn test_double_quoted_escapes() -> void {
    assert_eq(yaml.safe_load("\"a\\tb\\nc\""), "a\tb\nc")
    assert_eq(yaml.safe_load("\"\\x41\\u00e9\\U0001F600\""), "Aé😀")
    assert_eq(yaml.safe_load("\"\\0\\a\\b\\v\\f\\r\\e\""), "\u{0}\u{7}\u{8}\u{B}\u{C}\u{D}\u{1B}")
    assert_eq(yaml.safe_load("\"\\\"\\\\\\/\""), "\"\\/")
    assert_eq(yaml.safe_load("\"\\N\\_\\L\\P\""), "\u{85}\u{A0}\u{2028}\u{2029}")
    assert_eq(yaml.safe_load("\"\\ x\""), " x")
    assert_eq(yaml.safe_load("\"a\\\n  b\""), "ab")
}

fn test_single_quoted_doubles_its_quote() -> void {
    assert_eq(yaml.safe_load("'it''s'"), "it's")
    assert_eq(yaml.safe_load("'a\\nb'"), "a\\nb")
}

fn test_quoted_scalars_fold_lines() -> void {
    assert_eq(yaml.safe_load("'a\n  b'"), "a b")
    assert_eq(yaml.safe_load("\"a\n\n  b\""), "a\nb")
    assert_eq(yaml.safe_load("'a \n b'"), "a b")
    assert_eq(yaml.safe_load("\"a\n\n\n  b\""), "a\n\nb")
}

fn test_plain_scalars_fold_lines_and_keep_inner_spaces() -> void {
    assert_eq(yaml.safe_load("a\n b"), "a b")
    assert_eq(yaml.safe_load("a\n\n b"), "a\nb")
    assert_eq(yaml.safe_load("a  b   c"), "a  b   c")
    assert_eq(yaml.safe_load("a: b #comment"), {"a": "b"})
    assert_eq(yaml.safe_load("a#b"), "a#b")
    assert_eq(yaml.safe_load("a:b"), "a:b")
    assert_eq(yaml.safe_load("-a"), "-a")
    assert_eq(yaml.safe_load("http://x.y/z"), "http://x.y/z")
}

fn test_literal_block_scalars() -> void {
    assert_eq(yaml.safe_load("|\n  a\n  b\n"), "a\nb\n")
    assert_eq(yaml.safe_load("|\n  a\n   b\n"), "a\n b\n")
    assert_eq(yaml.safe_load("|-\n  a\n  b\n"), "a\nb")
    assert_eq(yaml.safe_load("|+\n  a\n\n\n"), "a\n\n\n")
    assert_eq(yaml.safe_load("|\n  a\n\n\n"), "a\n")
    assert_eq(yaml.safe_load("|2\n   a\n  b\n"), " a\nb\n")
    assert_eq(yaml.safe_load("|\n\n  a\n"), "\na\n")
    assert_eq(yaml.safe_load("k: |\n  a\n  b\nz: 1"), {"k": "a\nb\n", "z": 1})
    assert_eq(yaml.safe_load("| # comment\n  a\n"), "a\n")
}

fn test_folded_block_scalars() -> void {
    assert_eq(yaml.safe_load(">\n  a\n  b\n"), "a b\n")
    assert_eq(yaml.safe_load(">\n  a\n\n  b\n"), "a\nb\n")
    assert_eq(yaml.safe_load(">\n  a\n   b\n  c\n"), "a\n b\nc\n")
    assert_eq(yaml.safe_load(">-\n  a\n  b\n"), "a b")
    assert_eq(yaml.safe_load(">+\n  a\n\n"), "a\n\n")
    assert_eq(yaml.safe_load(">\n  a\n\n\n  b\n"), "a\n\nb\n")
}

fn test_block_scalar_indicators_are_checked() -> void {
    assert_throws(ScannerError, || yaml.safe_load("|0\n  a\n"))
    assert_throws(ScannerError, || yaml.safe_load("|x\n  a\n"))
    assert_throws(ScannerError, || yaml.safe_load("| x\n  a\n"))
}

fn test_block_sequences_and_mappings() -> void {
    assert_eq(yaml.safe_load("- 1\n- 2\n- 3"), [1, 2, 3])
    assert_eq(yaml.safe_load("a: 1\nb: 2"), {"a": 1, "b": 2})
    assert_eq(yaml.safe_load("a:\n  b:\n    c: 1"), {"a": {"b": {"c": 1}}})
    assert_eq(yaml.safe_load("a:\n- 1\n- 2"), {"a": [1, 2]})
    assert_eq(yaml.safe_load("- a: 1\n  b: 2\n- c"), [{"a": 1, "b": 2}, "c"])
    assert_eq(yaml.safe_load("- - 1\n  - 2\n- - 3"), [[1, 2], [3]])
    assert_eq(yaml.safe_load("a:\nb: 2"), {"a": null, "b": 2})
    assert_eq(yaml.safe_load("- \n- 2"), [null, 2])
    assert_eq(yaml.safe_load("? a\n: 1\n? b\n: 2"), {"a": 1, "b": 2})
    assert_eq(yaml.safe_load("? a\n? b"), {"a": null, "b": null})
}

fn test_flow_sequences_and_mappings() -> void {
    assert_eq(yaml.safe_load("[1, 2, 3]"), [1, 2, 3])
    assert_eq(yaml.safe_load("[]"), [])
    assert_eq(yaml.safe_load("{}"), {})
    assert_eq(yaml.safe_load("{a: 1, b: [2, 3]}"), {"a": 1, "b": [2, 3]})
    assert_eq(yaml.safe_load("[a, {b: c}, [d]]"), ["a", {"b": "c"}, ["d"]])
    assert_eq(yaml.safe_load("[1, 2, ]"), [1, 2])
    assert_eq(yaml.safe_load("{a: 1, }"), {"a": 1})
    assert_eq(yaml.safe_load("[\n  1,\n  2\n]"), [1, 2])
    assert_eq(yaml.safe_load("{a: }"), {"a": null})
    assert_eq(yaml.safe_load("{a}"), {"a": null})
    assert_eq(yaml.safe_load("[a: 1]"), [{"a": 1}])
    assert_eq(yaml.safe_load("[? a : 1]"), [{"a": 1}])
    assert_eq(yaml.safe_load("[a:b]"), ["a:b"])
    assert_eq(yaml.safe_load("{a: 1,b: 2}"), {"a": 1, "b": 2})
}

fn test_mapping_keys_of_every_hashable_kind() -> void {
    assert_eq(yaml.safe_load("1: a\n2.5: b\ntrue: c"), {1: "a", 2.5: "b", true: "c"})
    assert_eq(yaml.safe_load("? 1\n: a"), {1: "a"})
}

fn test_unhashable_keys_raise() -> void {
    let error = assert_throws(ConstructorError, || yaml.safe_load("? [1, 2]\n: pair"))
    assert_true(error.message.contains("found unhashable key"))
    assert_throws(ConstructorError, || yaml.safe_load("? {a: 1}\n: pair"))
}

#: PyYAML gives a mapping a `None` key here; a dict in this language cannot
#: hold null, so the package raises where PyYAML would not.
fn test_null_keys_raise() -> void {
    let error = assert_throws(ConstructorError, || yaml.safe_load("~: 1"))
    assert_true(error.message.contains("null key"))
    assert_eq(error.line, 1)
    assert_eq(error.column, 1)
}

fn test_duplicate_keys_keep_the_last_value() -> void {
    assert_eq(yaml.safe_load("a: 1\na: 2"), {"a": 2})
}

fn test_anchors_and_aliases_share_one_value() -> void {
    let loaded = yaml.safe_load("a: &x [1, 2]\nb: *x")
    assert_eq(loaded, {"a": [1, 2], "b": [1, 2]})
    assert_true(id(loaded["a"]) == id(loaded["b"]))
    assert_eq(yaml.safe_load("- &s text\n- *s"), ["text", "text"])
    assert_eq(yaml.safe_load("- &n 5\n- *n"), [5, 5])
    let recursive = yaml.safe_load("&a [*a]")
    assert_true(id(recursive[0]) == id(recursive))
}

fn test_undefined_and_duplicate_anchors_raise() -> void {
    let missing = assert_throws(ComposerError, || yaml.safe_load("a: *nope"))
    assert_true(missing.message.contains("found undefined alias 'nope'"))
    let twice = assert_throws(ComposerError, || yaml.safe_load("- &x 1\n- &x 2"))
    assert_true(twice.message.contains("found duplicate anchor 'x'; first occurrence"))
}

fn test_merge_keys() -> void {
    let text = "base: &b {x: 1, y: 2}\nleft: &l {y: 20, z: 30}\n" +
        "m1:\n  <<: *b\n  z: 3\nm2:\n  <<: [*b, *l]\nm3:\n  <<: *b\n  x: 9\nm4:\n  z: 0\n  <<: *b"
    let loaded = yaml.safe_load(text)
    assert_eq(loaded["m1"], {"x": 1, "y": 2, "z": 3})
    assert_eq(loaded["m2"], {"x": 1, "y": 2, "z": 30})
    assert_eq(loaded["m3"], {"x": 9, "y": 2})
    assert_eq(loaded["m4"], {"x": 1, "y": 2, "z": 0})
    assert_eq(loaded["m2"].keys(), ["y", "z", "x"])
}

fn test_bad_merges_raise() -> void {
    let error = assert_throws(ConstructorError, || yaml.safe_load("a:\n  <<: 1"))
    assert_true(
        error.message.contains(
            "expected a mapping or list of mappings for merging, but found scalar"
        )
    )
    assert_throws(ConstructorError, || yaml.safe_load("a:\n  <<: [1]"))
}

fn test_comments_and_blank_lines_are_ignored() -> void {
    let text = "# top\n\na: 1 # trailing\n\n# middle\nb: # after key\n  - 2\n"
    assert_eq(yaml.safe_load(text), {"a": 1, "b": [2]})
    assert_eq(yaml.safe_load("#only\n"), null)
}

fn test_multiple_documents() -> void {
    assert_eq(yaml.safe_load_all("a: 1\n---\nb: 2\n---\n- 3\n"), [{"a": 1}, {"b": 2}, [3]])
    assert_eq(yaml.safe_load_all("--- 1\n--- 2\n"), [1, 2])
    assert_eq(yaml.safe_load_all("1\n...\n--- 2\n"), [1, 2])
}

#: After `...` only an explicit `---` may open the next document; PyYAML
#: raises here too.
fn test_a_bare_document_after_an_end_marker_raises() -> void {
    let error = assert_throws(ParserError, || yaml.safe_load_all("1\n...\n2\n"))
    assert_eq(error.problem, "expected '<document start>', but found '<scalar>'")
    assert_eq(error.line, 3)
    assert_eq(error.column, 1)
}

fn test_document_markers() -> void {
    assert_eq(yaml.safe_load("---\na: 1\n...\n"), {"a": 1})
    assert_eq(yaml.safe_load("--- text"), "text")
    assert_eq(yaml.safe_load("---\n"), null)
    assert_eq(yaml.safe_load_all("---\n---\n"), [null, null])
    assert_eq(yaml.safe_load_all("a: 1\n...\n---\nb: 2\n"), [{"a": 1}, {"b": 2}])
    assert_eq(yaml.safe_load_all(""), [])
    assert_throws(ParserError, || yaml.safe_load_all("...\n"))
}

fn test_a_second_document_is_an_error_for_safe_load() -> void {
    let error = assert_throws(ComposerError, || yaml.safe_load("a: 1\n---\nb: 2"))
    assert_true(error.message.contains("expected a single document in the stream"))
    assert_true(error.message.contains("but found another document"))
}

fn test_directives() -> void {
    assert_eq(yaml.safe_load("%YAML 1.2\n---\na: 1"), {"a": 1})
    let tagged = yaml.compose("%TAG !e! tag:example.com,2000:\n---\n!e!foo 1")
    assert_eq((tagged ?? yaml.Node.scalar("", "")).tag, "tag:example.com,2000:foo")
    let error = assert_throws(
        ConstructorError,
        || yaml.safe_load("%TAG !e! tag:example.com,2000:\n---\n!e!foo 1")
    )
    assert_true(error.message.contains("'tag:example.com,2000:foo'"))
}

fn test_bad_directives_raise() -> void {
    let version = assert_throws(ParserError, || yaml.safe_load("%YAML 2.0\n---\na"))
    assert_true(
        version.message.contains("found incompatible YAML document (version 1.* is required)")
    )
    assert_throws(ParserError, || yaml.safe_load("%YAML 1.1\n%YAML 1.2\n---\na"))
    assert_throws(ParserError, || yaml.safe_load("%YAML 1.2\na"))
    assert_throws(ScannerError, || yaml.safe_load("%YAML x\n---\na"))
    assert_throws(ScannerError, || yaml.safe_load("% \n---\na"))
}

fn test_explicit_tags() -> void {
    assert_eq(yaml.safe_load("!!str 12"), "12")
    assert_eq(type_of(yaml.safe_load("!!str 12")), "str")
    assert_eq(yaml.safe_load("!!int '12'"), 12)
    assert_near(yaml.safe_load("!!float 1"), 1.0)
    assert_true(yaml.safe_load("!!bool 'true'"))
    assert_true(yaml.safe_load("!!null 'anything'") == null)
    assert_eq(yaml.safe_load("!!seq [1]"), [1])
    assert_eq(yaml.safe_load("!!map {a: 1}"), {"a": 1})
    assert_eq(yaml.safe_load("!<tag:yaml.org,2002:str> 5"), "5")
    assert_eq(yaml.safe_load("! 12"), 12)
    assert_eq(yaml.safe_load("! '12'"), 12)
    assert_eq(yaml.safe_load("! 'a'"), "a")
    assert_eq(yaml.safe_load("!!binary aGVsbG8="), bytes([104, 101, 108, 108, 111]))
    assert_eq(yaml.safe_load("!!binary |\n  aGVs\n  bG8="), bytes([104, 101, 108, 108, 111]))
}

fn test_sets_and_ordered_maps() -> void {
    let loaded = yaml.safe_load("!!set {a, b}")
    assert_true(isinstance(loaded, set))
    assert_eq(loaded.len(), 2)
    assert_true("a" in loaded)
    assert_eq(yaml.safe_load("!!omap [a: 1, b: 2]"), [("a", 1), ("b", 2)])
    assert_eq(yaml.safe_load("!!pairs [a: 1, a: 2]"), [("a", 1), ("a", 2)])
    assert_throws(ConstructorError, || yaml.safe_load("!!omap [a]"))
    assert_throws(ConstructorError, || yaml.safe_load("!!omap [{a: 1, b: 2}]"))
    assert_throws(ConstructorError, || yaml.safe_load("!!omap {a: 1}"))
}

fn test_unknown_and_mismatched_tags_raise() -> void {
    let unknown = assert_throws(ConstructorError, || yaml.safe_load("!!python/object:x {}"))
    assert_true(
        unknown.message.contains(
            "could not determine a constructor for the tag 'tag:yaml.org,2002:python/object:x'"
        )
    )
    assert_throws(ConstructorError, || yaml.safe_load("!custom 1"))
    assert_throws(ConstructorError, || yaml.safe_load("!!int [1]"))
    assert_throws(ConstructorError, || yaml.safe_load("!!seq 1"))
    assert_throws(ConstructorError, || yaml.safe_load("!!map 1"))
    assert_throws(ConstructorError, || yaml.safe_load("!!int abc"))
    assert_throws(ConstructorError, || yaml.safe_load("!!float abc"))
    assert_throws(ConstructorError, || yaml.safe_load("!!bool maybe"))
    assert_eq(yaml.safe_load("!!int 1_000"), 1000)
    assert_eq(yaml.safe_load("!!int ' 6\t'"), 6)
    assert_near(yaml.safe_load("!!float ' 1_5.5 '"), 15.5)
    assert_throws(ConstructorError, || yaml.safe_load("!!int '1 000'"))
    assert_throws(ConstructorError, || yaml.safe_load("!!binary 'a'"))
    assert_eq(yaml.safe_load("!!binary '!!!'"), bytes([]))
    assert_throws(ParserError, || yaml.safe_load("!e!x 1"))
}

#: PyYAML widens to a Python long; a Jaithon int is 64 bits, so anything
#: past it comes back as a float.
fn test_ints_past_64_bits_become_floats() -> void {
    assert_eq(yaml.safe_load("9223372036854775807"), 9223372036854775807)
    assert_eq(type_of(yaml.safe_load("9223372036854775808")), "float")
    assert_near(yaml.safe_load("9223372036854775808"), 9223372036854775808.0)
    assert_eq(type_of(yaml.safe_load("0xFFFFFFFFFFFFFFFFF")), "float")
}

fn test_unicode_and_line_breaks() -> void {
    assert_eq(yaml.safe_load("héllo: 日本"), {"héllo": "日本"})
    assert_eq(yaml.safe_load("a: 1\r\nb: 2\r\n"), {"a": 1, "b": 2})
    assert_eq(yaml.safe_load("a: 1\rb: 2\r"), {"a": 1, "b": 2})
    assert_eq(yaml.safe_load("\u{FEFF}a: 1"), {"a": 1})
    assert_eq(yaml.safe_load("- 😀"), ["😀"])
    assert_eq(yaml.safe_load("'a\u{2028}b'"), "a\u{2028}b")
    assert_throws(ParserError, || yaml.safe_load("|\n  a\u{2028}b\n"))
}

#: PyYAML skips only spaces between tokens, so a tab where a token should
#: start is an error there too.
fn test_tabs_cannot_precede_tokens() -> void {
    let error = assert_throws(ScannerError, || yaml.safe_load("a:\t1"))
    assert_eq(error.problem, "found character '\\t' that cannot start any token")
    assert_throws(ScannerError, || yaml.safe_load("a:\n\t- 1"))
    assert_throws(ScannerError, || yaml.safe_load("a: b\tc"))
    assert_eq(yaml.safe_load("a: 'b\tc'"), {"a": "b\tc"})
}

fn test_scanner_errors_carry_positions() -> void {
    let error = assert_throws(ScannerError, || yaml.safe_load("a: b: c"))
    assert_eq(error.line, 1)
    assert_eq(error.column, 5)
    assert_eq(error.problem, "mapping values are not allowed here")
    assert_eq(error.context, null)
    assert_true(error.mark() is not null)
    assert_eq(
        error.message,
        "mapping values are not allowed here\n  in \"<unicode string>\", line 1, column 5:\n    a: b: c\n        ^"
    )
}

fn test_parser_errors_carry_both_positions() -> void {
    let error = assert_throws(ParserError, || yaml.safe_load("a:\n  - 1\n c: 2"))
    assert_eq(error.context, "while parsing a block mapping")
    assert_eq(error.problem, "expected <block end>, but found '<block mapping start>'")
    assert_eq(error.line, 3)
    assert_eq(error.column, 2)
    let context_mark = error.context_mark
    assert_true(context_mark is not null)
    if context_mark is not null {
        assert_eq(context_mark.line, 0)
        assert_eq(context_mark.column, 0)
    }
}

fn test_every_error_is_a_yaml_error() -> void {
    assert_throws(YAMLError, || yaml.safe_load("a: [1, 2"))
    assert_throws(YAMLError, || yaml.safe_load("- a\nb: 1"))
    assert_throws(YAMLError, || yaml.safe_load("'unterminated"))
    assert_throws(YAMLError, || yaml.safe_load("\"bad \\q escape\""))
    assert_throws(YAMLError, || yaml.safe_load("\"\\u12\""))
    assert_throws(YAMLError, || yaml.safe_load("a: 1\n  b: 2"))
    assert_throws(YAMLError, || yaml.safe_load("{a: 1"))
    assert_eq(yaml.safe_load("[1 2]"), ["1 2"])
    assert_throws(YAMLError, || yaml.safe_load("&"))
    assert_throws(YAMLError, || yaml.safe_load("*"))
    assert_throws(YAMLError, || yaml.safe_load("@x"))
    assert_throws(YAMLError, || yaml.safe_load("`x"))
    assert_throws(YAMLError, || yaml.safe_load("key: 'a\n---\n'"))
}

fn test_the_reader_refuses_control_characters() -> void {
    let error = assert_throws(ReaderError, || yaml.safe_load("a: \u{1}"))
    assert_eq(error.position, 3)
    assert_eq(error.character, 1)
    assert_eq(error.line, 1)
    assert_eq(error.column, 4)
    assert_eq(
        error.message,
        "unacceptable character #x0001: special characters are not allowed\n  in \"<unicode string>\", position 3"
    )
    assert_throws(ReaderError, || yaml.safe_load("\u{7F}"))
    assert_throws(ReaderError, || yaml.safe_load("\u{FFFE}"))
    assert_eq(yaml.safe_load("a\u{85}b"), "a b")
}

fn test_a_sequence_key_is_unhashable() -> void {
    assert_throws(ConstructorError, || yaml.safe_load("? - a\n  - b\n: 1"))
    let composed = yaml.compose("? - a\n  - b\n: 1") ?? yaml.Node.scalar("", "")
    let (key, value) = composed.pairs[0]
    assert_eq(key.kind, yaml.NodeKind.Sequence)
    assert_eq(value.value, "1")
}

fn test_indentless_sequences_inside_mappings() -> void {
    assert_eq(yaml.safe_load("a:\n- 1\n- 2\nb:\n  - 3"), {"a": [1, 2], "b": [3]})
    assert_eq(yaml.safe_load("- a:\n  - 1\n  b: 2"), [{"a": [1], "b": 2}])
}

fn test_long_plain_scalars_span_lines() -> void {
    let text = "key: first line of a value\n  that continues here\n  and here\nnext: 1"
    assert_eq(
        yaml.safe_load(text),
        {"key": "first line of a value that continues here and here", "next": 1}
    )
}

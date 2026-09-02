#: `safe_dump` and `safe_dump_all` against PyYAML, byte for byte: every
#: expected string here is what `yaml.safe_dump` writes for the same value
#: and options, except where the core schema quotes a string PyYAML would
#: not, and those cases say so.
import std.math
import jaiyaml as yaml
from jaiyaml import RepresenterError
from std.test import assert_eq, assert_throws, assert_true

fn test_scalars_dump_as_pyyaml_writes_them() -> void {
    assert_eq(yaml.safe_dump(1), "1\n...\n")
    assert_eq(yaml.safe_dump(-3), "-3\n...\n")
    assert_eq(yaml.safe_dump(1.5), "1.5\n...\n")
    assert_eq(yaml.safe_dump(1e16), "1.0e+16\n...\n")
    assert_eq(yaml.safe_dump(1e-7), "1.0e-07\n...\n")
    assert_eq(yaml.safe_dump(0.1 + 0.2), "0.30000000000000004\n...\n")
    assert_eq(yaml.safe_dump(math.INF), ".inf\n...\n")
    assert_eq(yaml.safe_dump(-math.INF), "-.inf\n...\n")
    assert_eq(yaml.safe_dump(math.NAN), ".nan\n...\n")
    assert_eq(yaml.safe_dump(true), "true\n...\n")
    assert_eq(yaml.safe_dump(false), "false\n...\n")
    assert_eq(yaml.safe_dump(null), "null\n...\n")
    assert_eq(yaml.safe_dump("text"), "text\n...\n")
    assert_eq(yaml.safe_dump(""), "''\n")
}

fn test_strings_that_would_resolve_otherwise_are_quoted() -> void {
    assert_eq(yaml.safe_dump("12"), "'12'\n")
    assert_eq(yaml.safe_dump("1.5"), "'1.5'\n")
    assert_eq(yaml.safe_dump("true"), "'true'\n")
    assert_eq(yaml.safe_dump("null"), "'null'\n")
    assert_eq(yaml.safe_dump("~"), "'~'\n")
    assert_eq(yaml.safe_dump("0x1F"), "'0x1F'\n")
    assert_eq(yaml.safe_dump(".inf"), "'.inf'\n")
    assert_eq(yaml.safe_dump("<<"), "'<<'\n")
}

#: PyYAML quotes `yes` and `1_000`, because they are a bool and an int in
#: YAML 1.1; the core schema reads them as strings, so they stay plain.
fn test_yaml_1_1_spellings_stay_plain() -> void {
    assert_eq(yaml.safe_dump("yes"), "yes\n...\n")
    assert_eq(yaml.safe_dump("1_000"), "1_000\n...\n")
    assert_eq(yaml.safe_dump("0o17"), "'0o17'\n")
    assert_eq(yaml.safe_dump("1e5"), "'1e5'\n")
}

fn test_strings_needing_quotes() -> void {
    assert_eq(yaml.safe_dump("a: b"), "'a: b'\n")
    assert_eq(yaml.safe_dump("a #b"), "'a #b'\n")
    assert_eq(yaml.safe_dump("- a"), "'- a'\n")
    assert_eq(yaml.safe_dump("[a]"), "'[a]'\n")
    assert_eq(yaml.safe_dump("{a"), "'{a'\n")
    assert_eq(yaml.safe_dump("&a"), "'&a'\n")
    assert_eq(yaml.safe_dump("*a"), "'*a'\n")
    assert_eq(yaml.safe_dump("!a"), "'!a'\n")
    assert_eq(yaml.safe_dump("%a"), "'%a'\n")
    assert_eq(yaml.safe_dump("@a"), "'@a'\n")
    assert_eq(yaml.safe_dump("`a"), "'`a'\n")
    assert_eq(yaml.safe_dump("'a"), "'''a'\n")
    assert_eq(yaml.safe_dump("\"a"), "'\"a'\n")
    assert_eq(yaml.safe_dump("a,b"), "a,b\n...\n")
    assert_eq(yaml.safe_dump(" a"), "' a'\n")
    assert_eq(yaml.safe_dump("a "), "'a '\n")
    assert_eq(yaml.safe_dump("---"), "'---'\n")
    assert_eq(yaml.safe_dump("a\tb"), "\"a\\tb\"\n")
    assert_eq(yaml.safe_dump("a\nb"), "'a\n\n  b'\n")
    assert_eq(yaml.safe_dump("a\n b"), "\"a\\n b\"\n")
    assert_eq(yaml.safe_dump("a \nb"), "\"a \\nb\"\n")
    assert_eq(yaml.safe_dump("\u{7}"), "\"\\a\"\n")
    assert_eq(yaml.safe_dump("é"), "\"\\xE9\"\n")
    assert_eq(yaml.safe_dump("日"), "\"\\u65E5\"\n")
    assert_eq(yaml.safe_dump("😀"), "\"\\U0001F600\"\n")
}

fn test_allow_unicode_writes_characters_themselves() -> void {
    assert_eq(yaml.safe_dump("é", allow_unicode: true), "é\n...\n")
    assert_eq(yaml.safe_dump("日本", allow_unicode: true), "日本\n...\n")
    assert_eq(yaml.safe_dump("😀", allow_unicode: true), "😀\n...\n")
    assert_eq(yaml.safe_dump({"é": "ü"}, allow_unicode: true), "é: ü\n")
}

fn test_block_collections() -> void {
    assert_eq(yaml.safe_dump([1, 2]), "- 1\n- 2\n")
    assert_eq(yaml.safe_dump({"a": 1, "b": 2}), "a: 1\nb: 2\n")
    assert_eq(yaml.safe_dump({"a": [1, 2]}), "a:\n- 1\n- 2\n")
    assert_eq(yaml.safe_dump({"a": {"b": 1}}), "a:\n  b: 1\n")
    assert_eq(yaml.safe_dump([[1], [2]]), "- - 1\n- - 2\n")
    assert_eq(yaml.safe_dump([{"a": 1, "b": 2}]), "- a: 1\n  b: 2\n")
    assert_eq(yaml.safe_dump([]), "[]\n")
    assert_eq(yaml.safe_dump({}), "{}\n")
    assert_eq(yaml.safe_dump({"a": [], "b": {}}), "a: []\nb: {}\n")
    assert_eq(yaml.safe_dump([null, [null]]), "- null\n- - null\n")
}

fn test_flow_styles() -> void {
    assert_eq(yaml.safe_dump([1, [2, 3]], default_flow_style: true), "[1, [2, 3]]\n")
    assert_eq(yaml.safe_dump({"a": {"b": 1}}, default_flow_style: true), "{a: {b: 1}}\n")
    assert_eq(
        yaml.safe_dump({"a": [1, 2], "b": {"c": [3]}}, default_flow_style: null),
        "a: [1, 2]\nb:\n  c: [3]\n"
    )
    assert_eq(yaml.safe_dump([[1, 2], [3]], default_flow_style: null), "- [1, 2]\n- [3]\n")
    assert_eq(yaml.safe_dump({"a": "x\ny"}, default_flow_style: true), "{a: 'x\n\n    y'}\n")
}

fn test_sort_keys() -> void {
    assert_eq(yaml.safe_dump({"b": 1, "a": 2}), "a: 2\nb: 1\n")
    assert_eq(yaml.safe_dump({"b": 1, "a": 2}, sort_keys: false), "b: 1\na: 2\n")
    assert_eq(yaml.safe_dump({2: "b", 10: "a", 1.5: "c"}), "1.5: c\n2: b\n10: a\n")
    assert_eq(yaml.safe_dump({"b": 1, 1: 2}), "b: 1\n1: 2\n")
    assert_eq(yaml.safe_dump({true: 1, false: 2}), "false: 2\ntrue: 1\n")
    assert_eq(yaml.safe_dump({"B": 1, "a": 2, "é": 3}), "B: 1\na: 2\n\"\\xE9\": 3\n")
}

fn test_indent_and_width() -> void {
    assert_eq(yaml.safe_dump({"a": {"b": [1]}}, indent: 4), "a:\n    b:\n    - 1\n")
    assert_eq(yaml.safe_dump({"a": {"b": 1}}, indent: 1), "a:\n  b: 1\n")
    assert_eq(yaml.safe_dump({"a": {"b": 1}}, indent: 10), "a:\n  b: 1\n")
    let long = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen"
    assert_eq(
        yaml.safe_dump(long),
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen\n  sixteen\n...\n"
    )
    assert_eq(
        yaml.safe_dump(long, width: 30),
        "one two three four five six seven\n  eight nine ten eleven twelve thirteen\n  fourteen fifteen sixteen\n...\n"
    )
    assert_eq(
        yaml.safe_dump({"k": long}, width: 5),
        "k: one\n  two three\n  four\n  five\n  six seven\n  eight\n  nine\n  ten eleven\n  twelve\n  thirteen\n  fourteen\n  fifteen\n  sixteen\n"
    )
}

#: A line is only broken at a space once the column is already past the
#: width, which is why a width of five still pairs `two three`.
fn test_a_width_no_larger_than_twice_the_indent_is_ignored() -> void {
    let long = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen"
    assert_eq(
        yaml.safe_dump(long, width: 4),
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen\n  sixteen\n...\n"
    )
    assert_eq(
        yaml.safe_dump(long, width: 5),
        "one two\n  three\n  four\n  five\n  six seven\n  eight\n  nine\n  ten eleven\n  twelve\n  thirteen\n  fourteen\n  fifteen\n  sixteen\n...\n"
    )
}

fn test_default_style() -> void {
    assert_eq(yaml.safe_dump({"a": "b"}, default_style: "\""), "\"a\": \"b\"\n")
    assert_eq(yaml.safe_dump({"a": 1}, default_style: "'"), "'a': !!int '1'\n")
    assert_eq(yaml.safe_dump({"a": "x\ny\n"}, default_style: "|"), "\"a\": |\n  x\n  y\n")
    assert_eq(yaml.safe_dump({"a": "x y\n"}, default_style: ">"), "\"a\": >\n  x y\n")
    assert_eq(yaml.safe_dump(["x"], default_style: "|"), "- |-\n  x\n")
    assert_eq(yaml.safe_dump(["x\n\n"], default_style: "|"), "- |+\n  x\n\n...\n")
    assert_eq(yaml.safe_dump([" x"], default_style: "|"), "- |2-\n   x\n")
    assert_eq(yaml.safe_dump(["a b c\n"], default_style: ">", width: 4), "- >\n  a b c\n")
    assert_eq(
        yaml.safe_dump(["a b c d e f\n"], default_style: ">", width: 5),
        "- >\n  a b c\n  d e f\n"
    )
}

fn test_canonical_and_explicit_markers() -> void {
    assert_eq(
        yaml.safe_dump({"a": [1, "b"]}, canonical: true),
        "---\n!!map {\n  ? !!str \"a\"\n  : !!seq [\n    !!int \"1\",\n    !!str \"b\",\n  ],\n}\n"
    )
    assert_eq(yaml.safe_dump({"a": 1}, explicit_start: true), "---\na: 1\n")
    assert_eq(yaml.safe_dump({"a": 1}, explicit_end: true), "a: 1\n...\n")
    assert_eq(yaml.safe_dump(1, explicit_start: true), "--- 1\n...\n")
    assert_eq(yaml.safe_dump({"a": 1}, version: (1, 2)), "%YAML 1.2\n---\na: 1\n")
    assert_eq(
        yaml.safe_dump({"a": 1}, tags: {"!e!": "tag:example.com,2000:"}),
        "%TAG !e! tag:example.com,2000:\n---\na: 1\n"
    )
    assert_eq(yaml.safe_dump({"a": 1}, line_break: "\r\n"), "a: 1\r\n")
}

fn test_dump_all() -> void {
    assert_eq(yaml.safe_dump_all([1, "two", [3]]), "1\n--- two\n---\n- 3\n")
    assert_eq(yaml.safe_dump_all([{"a": 1}, {"b": 2}]), "a: 1\n---\nb: 2\n")
    assert_eq(yaml.safe_dump_all([]), "")
    assert_eq(yaml.safe_dump_all([null, null]), "null\n--- null\n...\n")
    assert_eq(
        yaml.safe_dump_all([{"a": 1}, {"b": 2}], explicit_start: true),
        "---\na: 1\n---\nb: 2\n"
    )
    assert_eq(yaml.safe_dump_all([[1], [2]], explicit_end: true), "- 1\n...\n---\n- 2\n...\n")
}

fn test_shared_containers_become_anchors_and_aliases() -> void {
    let shared = [1, 2]
    assert_eq(yaml.safe_dump({"a": shared, "b": shared}), "a: &id001\n- 1\n- 2\nb: *id001\n")
    let inner = {"x": 1}
    assert_eq(yaml.safe_dump([inner, inner, [3]]), "- &id001\n  x: 1\n- *id001\n- - 3\n")
    let text = "same"
    assert_eq(yaml.safe_dump([text, text]), "- same\n- same\n")
    let first = [1]
    let second = [2]
    assert_eq(
        yaml.safe_dump([second, first, second, first]),
        "- &id001\n  - 2\n- &id002\n  - 1\n- *id001\n- *id002\n"
    )
}

fn test_tuples_sets_and_bytes() -> void {
    assert_eq(yaml.safe_dump([(1, 2)]), "- - 1\n  - 2\n")
    assert_eq(yaml.safe_dump(bytes([104, 101, 108, 108, 111])), "!!binary |\n  aGVsbG8=\n")
    let single: set[any] = set()
    single.add("a")
    assert_eq(yaml.safe_dump(single), "!!set\na: null\n")
}

fn test_unrepresentable_values_raise() -> void {
    let error = assert_throws(RepresenterError, || yaml.safe_dump(|x| x))
    assert_true(error.message.contains("cannot represent an object"))
    assert_throws(RepresenterError, || yaml.safe_dump({"a": yaml.Mark("n", 0, 0, 0, null, 0)}))
}

fn test_round_trips() -> void {
    let values: list[any] = [
        {
            "a": [1, 2.5, "three", null, true],
            "b": {"c": "multi\nline\n", "d": " spaced "},
        },
        [
            "12",
            12,
            "1.0",
            1.0,
            "~",
            "null",
            "",
            "-",
            "- x",
            "# y",
            "a: b",
            "yes",
        ],
        {"nested": [[[]], [{}], {"k": [{"z": "é 😀"}]}]},
    ]
    for value in values {
        assert_eq(yaml.safe_load(yaml.safe_dump(value)), value)
        assert_eq(yaml.safe_load(yaml.safe_dump(value, default_flow_style: true)), value)
        assert_eq(yaml.safe_load(yaml.safe_dump(value, default_style: "\"")), value)
        assert_eq(yaml.safe_load(yaml.safe_dump(value, canonical: true)), value)
        assert_eq(yaml.safe_load(yaml.safe_dump(value, allow_unicode: true, width: 10)), value)
    }
}

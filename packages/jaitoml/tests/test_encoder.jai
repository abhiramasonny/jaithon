#: `jaitoml.encoder` against `tomli_w` 1.2, which wrote every expected
#: document here. The comparisons are whole strings, newlines included,
#: because the layout — where the blank lines fall, which tables go inline,
#: how arrays indent — is the thing being ported.
import std.math as math

from std.io import open, read_file
from std.test import assert_eq, assert_throws, assert_true
from jaitoml.datetime import Date, DateTime, Time
from jaitoml.decoder import loads
from jaitoml.encoder import MAX_LINE_LENGTH, dump, dumps

fn test_scalars_render_like_python() -> void {
    let doc = dumps(
        {
            "i": 1,
            "n": -17,
            "f": 1.5,
            "e": 5e22,
            "big": 1e16,
            "small": 1e-5,
            "zero": -0.0,
            "inf": math.INF,
            "ninf": -math.INF,
            "nan": math.NAN,
            "t": true,
            "s": "text",
        }
    )
    assert_eq(
        doc,
        "i = 1\nn = -17\nf = 1.5\ne = 5e+22\nbig = 1e+16\nsmall = 1e-05\n" +
            "zero = -0.0\ninf = inf\nninf = -inf\nnan = nan\nt = true\ns = \"text\"\n"
    )
}

fn test_dates_and_times_render_unquoted() -> void {
    let doc = dumps(
        {
            "odt": DateTime(1979, 5, 27, 7, 32, 0, 0, 0),
            "west": DateTime(1979, 5, 27, 0, 32, 0, 999999, -420),
            "ldt": DateTime(1979, 5, 27, 7, 32),
            "ld": Date(1979, 5, 27),
            "lt": Time(7, 32, 0, 500000),
        }
    )
    assert_eq(
        doc,
        "odt = 1979-05-27 07:32:00+00:00\nwest = 1979-05-27 00:32:00.999999-07:00\n" +
            "ldt = 1979-05-27 07:32:00\nld = 1979-05-27\nlt = 07:32:00.500000\n"
    )
}

fn test_strings_escape_what_basic_strings_cannot_hold() -> void {
    assert_eq(dumps({"s": "a\"b\\c"}), "s = \"a\\\"b\\\\c\"\n")
    assert_eq(dumps({"s": "tab\there"}), "s = \"tab\there\"\n")
    assert_eq(dumps({"s": "nl\ncr\rff\x0Cbs\x08"}), "s = \"nl\\ncr\\rff\\fbs\\b\"\n")
    assert_eq(dumps({"s": "\x00\x1f\x7f"}), "s = \"\\u0000\\u001f\\u007f\"\n")
    assert_eq(dumps({"s": "é漢😀"}), "s = \"é漢😀\"\n")
    assert_eq(dumps({"s": ""}), "s = \"\"\n")
}

fn test_multiline_strings_when_asked() -> void {
    assert_eq(dumps({"s": "two\nlines"}, multiline_strings: true), "s = \"\"\"\ntwo\nlines\"\"\"\n")
    assert_eq(dumps({"s": "two\nlines"}), "s = \"two\\nlines\"\n")
    assert_eq(dumps({"s": "one line"}, multiline_strings: true), "s = \"one line\"\n")
    assert_eq(dumps({"s": "a\r\nb"}, multiline_strings: true), "s = \"\"\"\na\nb\"\"\"\n")
    assert_eq(dumps({"s": "a\r\nb"}), "s = \"a\\r\\nb\"\n")
    assert_eq(
        dumps({"s": "q\"\"\"\n"}, multiline_strings: true),
        "s = \"\"\"\nq\\\"\\\"\\\"\n\"\"\"\n"
    )
}

fn test_keys_are_quoted_when_not_bare() -> void {
    let doc = dumps({"bare-key_1": 1, "with space": 2, "": 3, "dot.ted": 4, "q\"uote": 5, "é": 6})
    assert_eq(
        doc,
        "bare-key_1 = 1\n\"with space\" = 2\n\"\" = 3\n\"dot.ted\" = 4\n" +
            "\"q\\\"uote\" = 5\n\"é\" = 6\n"
    )
    assert_eq(dumps({"a b": {"c.d": 1}}), "[\"a b\"]\n\"c.d\" = 1\n")
    assert_eq(dumps({"a": {"b c": {"d": 1}}}), "[a.\"b c\"]\nd = 1\n")
}

fn test_arrays_go_one_element_per_line() -> void {
    assert_eq(dumps({"xs": [1, 2, 3]}), "xs = [\n    1,\n    2,\n    3,\n]\n")
    assert_eq(dumps({"xs": []}), "xs = []\n")
    assert_eq(
        dumps({"xs": [[1], [2, 3]]}),
        "xs = [\n    [\n        1,\n    ],\n    [\n        2,\n        3,\n    ],\n]\n"
    )
    assert_eq(dumps({"xs": [1, 2]}, indent: 2), "xs = [\n  1,\n  2,\n]\n")
    assert_eq(dumps({"xs": [1, [2]]}, indent: 0), "xs = [\n1,\n[\n2,\n],\n]\n")
    assert_eq(dumps({"xs": (1, "a")}), "xs = [\n    1,\n    \"a\",\n]\n")
}

fn test_nested_tables_become_sections() -> void {
    let doc = dumps(
        {
            "title": "T",
            "owner": {"name": "Tom", "pet": {"kind": "cat"}},
            "db": {"on": true},
        }
    )
    assert_eq(
        doc,
        "title = \"T\"\n\n[owner]\nname = \"Tom\"\n\n[owner.pet]\nkind = \"cat\"\n\n[db]\non = true\n"
    )
    assert_eq(dumps({"a": {"b": {"c": {}}}}), "[a.b.c]\n")
    assert_eq(dumps({"a": {}}), "[a]\n")
    assert_eq(dumps({}), "")
    assert_eq(dumps({"a": {"x": 1, "b": {}}}), "[a]\nx = 1\n\n[a.b]\n")
}

fn test_a_table_with_only_subtables_writes_no_header() -> void {
    assert_eq(dumps({"a": {"b": {"x": 1}, "c": {"y": 2}}}), "[a.b]\nx = 1\n\n[a.c]\ny = 2\n")
}

fn test_small_tables_in_arrays_stay_inline() -> void {
    let doc = dumps({"points": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]})
    assert_eq(doc, "points = [\n    { x = 1, y = 2 },\n    { x = 3, y = 4 },\n]\n")
    assert_eq(dumps({"e": [{}]}), "e = [\n    {},\n]\n")
    assert_eq(dumps({"n": [{"a": {"b": 1}}]}), "n = [\n    { a = { b = 1 } },\n]\n")
}

fn test_large_tables_in_arrays_become_array_sections() -> void {
    let long = "x" * (MAX_LINE_LENGTH - 10)
    let doc = dumps({"items": [{"name": long}, {"name": "short"}], "after": 1})
    assert_eq(
        doc,
        "after = 1\n\n[[items]]\nname = \"" + long + "\"\n\n[[items]]\nname = \"short\"\n"
    )
    let with_newline = dumps({"items": [{"s": "a\nb"}, {"t": 1}]}, multiline_strings: true)
    assert_eq(with_newline, "[[items]]\ns = \"\"\"\na\nb\"\"\"\n\n[[items]]\nt = 1\n")
    let nested = dumps({"p": {"q": [{"r": long}]}})
    assert_eq(nested, "[[p.q]]\nr = \"" + long + "\"\n")
    assert_eq(MAX_LINE_LENGTH, 100)
}

fn test_an_array_section_carries_its_own_subtables() -> void {
    let doc = dumps(
        {
            "fruits": [
                {
                    "name": "apple",
                    "physical": {"color": "red"},
                    "varieties": [{"name": "granny smith"}],
                },
                {"name": "banana"},
            ],
        }
    )
    assert_eq(
        doc,
        "[[fruits]]\nname = \"apple\"\nvarieties = [\n    { name = \"granny smith\" },\n]\n\n" +
            "[fruits.physical]\ncolor = \"red\"\n\n[[fruits]]\nname = \"banana\"\n"
    )
}

fn test_mixed_arrays_of_tables_and_scalars_stay_inline() -> void {
    assert_eq(dumps({"m": [{"a": 1}, 2]}), "m = [\n    { a = 1 },\n    2,\n]\n")
}

fn test_unsupported_values_are_refused() -> void {
    let missing = assert_throws(TypeError, fn() -> void { let _ = dumps({"a": null}) }, "null")
    assert_eq(missing.message, "Object of type 'NoneType' is not TOML serializable")
    let set_error = assert_throws(TypeError, fn() -> void { let _ = dumps({"a": {1, 2}}) }, "set")
    assert_eq(set_error.message, "Object of type 'set' is not TOML serializable")
    let int_keyed: any = {1: "x"}
    let key_error = assert_throws(TypeError, fn() -> void { let _ = dumps(int_keyed) }, "int key")
    assert_eq(key_error.message, "Invalid mapping key '1' of type 'int'. A string is required.")
    assert_throws(TypeError, fn() -> void { let _ = dumps({"a": [null]}) }, "null in array")
    assert_throws(TypeError, fn() -> void { let _ = dumps({"a": {"b": null}}) }, "null in table")
    let indent_error = assert_throws(ValueError, fn() -> void {
        let _ = dumps({"a": 1}, indent: -1)
    }, "negative indent")
    assert_eq(indent_error.message, "Indent width must be non-negative")
}

fn test_dump_writes_a_file() -> void {
    let path = "/tmp/jaitoml-test-dump.toml"
    let file = open(path, "w")
    dump({"a": {"b": [1, 2]}}, file, indent: 2)
    file.close()
    assert_eq(read_file(path), "[a]\nb = [\n  1,\n  2,\n]\n")
}

fn test_round_trip_survives_loads() -> void {
    let original = {
        "title": "TOML \"Example\"",
        "owner": {"name": "Tom", "dob": DateTime(1979, 5, 27, 7, 32, 0, 0, -480)},
        "database": {
            "ports": [8000, 8001],
            "data": [["delta", "phi"], [3.14]],
            "temp": {"cpu": 79.5},
        },
        "products": [{"name": "Hammer", "sku": 738594937}, {"name": "Nail"}],
        "times": [Date(2020, 2, 29), Time(0, 0, 0, 1), DateTime(2020, 2, 29, 12)],
        "odd keys": {"": 1, "a.b": 2},
        "text": "line\nline\ttab \\ \"",
    }
    let doc = dumps(original)
    assert_eq(loads(doc), original)
    assert_eq(dumps(loads(doc)), doc)
    let block = dumps(original, multiline_strings: true, indent: 2)
    assert_eq(loads(block), original)
    assert_true(block.contains("\"\"\"\nline\nline"))
}

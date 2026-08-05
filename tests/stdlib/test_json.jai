#: `std.json` — a parser is only as good as what it refuses. Half of these
#: tests are malformed documents that must raise rather than be guessed at, and
#: the rest are the round trip: parse(stringify(x)) == x for everything JSON
#: can express.
import std.json as json

from std.test import assert_eq, assert_true, assert_false, assert_near, assert_throws

fn test_scalars_parse_to_the_matching_type() -> void {
    assert_eq(json.parse("1"), 1)
    assert_eq(type_of(json.parse("1")), "int")
    assert_near(json.parse("1.5"), 1.5)
    assert_eq(type_of(json.parse("1.0")), "float")  # the dot decides, not the value
    assert_eq(json.parse("\"text\""), "text")
    assert_true(json.parse("true"))
    assert_false(json.parse("false"))
    assert_true(json.parse("null") == null)
}

fn test_whitespace_around_a_document_is_insignificant() -> void {
    assert_eq(json.parse("  \n\t [1, 2] \n "), [1, 2])
}

fn test_containers_parse_recursively() -> void {
    assert_eq(json.parse("[]"), [])
    assert_eq(json.parse("{}"), {})
    assert_eq(json.parse("[1, [2, [3]]]"), [1, [2, [3]]])
    assert_eq(json.parse("{\"a\": {\"b\": 1}}"), {"a": {"b": 1}})
}

fn test_string_escapes_decode() -> void {
    assert_eq(json.parse("\"a\\nb\""), "a\nb")
    assert_eq(json.parse("\"\\t\\\\\\\"\""), "\t\\\"")
    assert_eq(json.parse("\"\\u00e9\""), "é")
    assert_eq(json.parse("\"\\ud83d\\ude00\""), "😀")  # a surrogate pair
}

fn test_malformed_documents_raise() -> void {
    assert_throws(Error, || json.parse(""))
    assert_throws(Error, || json.parse("{"))
    assert_throws(Error, || json.parse("[1,]"))
    assert_throws(Error, || json.parse("{\"a\": 1,}"))
    assert_throws(Error, || json.parse("{a: 1}"))  # unquoted name
    assert_throws(Error, || json.parse("'single'"))
    assert_throws(Error, || json.parse("01"))   # leading zero
    assert_throws(Error, || json.parse("1 2"))  # trailing content
    assert_throws(Error, || json.parse("NaN"))
    assert_throws(Error, || json.parse("[1"))
}

fn test_stringify_is_compact_by_default() -> void {
    assert_eq(json.stringify([1, 2]), "[1,2]")
    assert_eq(json.stringify({"a": 1}), "{\"a\":1}")
    assert_eq(json.stringify([]), "[]")
    assert_eq(json.stringify({}), "{}")
    assert_eq(json.stringify(null), "null")
    assert_eq(json.stringify(true), "true")
}

fn test_stringify_indent_and_sort_keys() -> void {
    assert_eq(json.stringify([1, 2], 2), "[\n  1,\n  2\n]")
    assert_eq(json.stringify({"b": 1, "a": 2}, 0, true), "{\"a\":2,\"b\":1}")
    # Without sort_keys the members keep insertion order.
    assert_eq(json.stringify({"b": 1, "a": 2}), "{\"b\":1,\"a\":2}")
}

fn test_stringify_escapes_what_json_requires() -> void {
    assert_eq(json.stringify("a\nb"), "\"a\\nb\"")
    assert_eq(json.stringify("\""), "\"\\\"\"")
    assert_eq(json.stringify("é"), "\"é\"")
    assert_eq(json.stringify("é", 0, false, true), "\"\\u00e9\"")
}

fn test_stringify_refuses_what_json_cannot_express() -> void {
    let infinity = 1e308 * 10.0
    assert_throws(ValueError, || json.stringify(infinity))
    assert_throws(ValueError, || json.stringify(infinity - infinity))
    let cycle: list[any] = []
    cycle.push(cycle)
    assert_throws(ValueError, || json.stringify(cycle))
}

fn test_round_trip_preserves_structure() -> void {
    let document: any = {
        "name": "ada",
        "tags": ["x", "y"],
        "nested": {"n": 1, "f": 1.5, "ok": true, "nil": null},
        "empty_list": [],
        "empty_dict": {},
    }
    assert_eq(json.parse(json.stringify(document)), document)
    assert_eq(json.parse(json.stringify(document, 4)), document)
}

fn test_round_trip_survives_awkward_strings() -> void {
    for text in ["", " ", "\n", "\"", "\\", "é", "😀", "a/b"] {
        assert_eq(json.parse(json.stringify(text)), text)
    }
}

fn test_get_path_probes_without_raising() -> void {
    let document = json.parse("{\"a\": {\"b\": [10, 20]}}")
    assert_eq(json.get_path(document, "a.b[1]"), 20)
    assert_eq(json.get_path(document, "a.b[-1]"), 20)
    assert_eq(json.get_path(document, "a.b[0]"), 10)
    assert_true(json.get_path(document, "a.missing") == null)
    assert_eq(json.get_path(document, "a.missing", 0), 0)
    assert_eq(json.get_path(document, "a.b[9]", "none"), "none")
    assert_eq(json.get_path(document, "a.b.c", "none"), "none")  # wrong kind
}

fn test_deeply_nested_input_is_bounded() -> void {
    # The depth limit exists so that a hostile document cannot exhaust the
    # C stack; one level past it must raise rather than crash.
    let too_deep = "[".repeat(json.MAX_DEPTH + 1) + "]".repeat(json.MAX_DEPTH + 1)
    assert_throws(Error, || json.parse(too_deep))
}

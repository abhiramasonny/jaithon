#: `jaitoml.decoder` against Python 3.13's `tomllib`, which produced every
#: expected value and every expected error message here. The messages are
#: compared whole, position suffix included, because a decoder that raises
#: at the wrong column is a decoder that will send someone to the wrong line
#: of their config file.
import std.math as math

from std.io import open, write_file
from std.test import assert_eq, assert_false, assert_throws, assert_true
from jaitoml.datetime import Date, DateTime, Time
from jaitoml.decoder import TOMLDecodeError, load, loads

fn fails_with(text: str, message: str) -> void {
    let error = assert_throws(TOMLDecodeError, fn() -> void { let _ = loads(text) }, message)
    assert_eq(error.message, message, "message for " + repr(text))
}

fn test_empty_and_comment_only_documents() -> void {
    assert_eq(loads(""), {})
    assert_eq(loads("   \n\t\n"), {})
    assert_eq(loads("# just a comment"), {})
    assert_eq(loads("# c\r\n\r\nk = 1 # trailing\r\n"), {"k": 1})
}

fn test_keys_in_every_spelling() -> void {
    let doc = "bare-key_1 = 1\n\"quoted key\" = 2\n'literal key' = 3\n\"\" = 4\n" +
        "a . b . c = 5\n\"x.y\".z = 6\n1234 = 7\n\"\\u00e9\" = 8"
    let value = loads(doc)
    assert_eq(value["bare-key_1"], 1)
    assert_eq(value["quoted key"], 2)
    assert_eq(value["literal key"], 3)
    assert_eq(value[""], 4)
    assert_eq(value["a"], {"b": {"c": 5}})
    assert_eq(value["x.y"], {"z": 6})
    assert_eq(value["1234"], 7)
    assert_eq(value["é"], 8)
}

fn test_basic_strings_and_every_escape() -> void {
    let doc = "s = \"tab\\t nl\\n cr\\r ff\\f bs\\b q\\\" b\\\\ u\\u00E9 U\\U0001F600 end\""
    assert_eq(loads(doc)["s"], "tab\t nl\n cr\r ff\x0C bs\x08 q\" b\\ ué U😀 end")
    assert_eq(loads("s = \"\"")["s"], "")
    assert_eq(loads("s = \"raw tab\there\"")["s"], "raw tab\there")
    assert_eq(loads("s = \"漢字 😀\"")["s"], "漢字 😀")
}

fn test_literal_strings_keep_backslashes() -> void {
    assert_eq(loads("p = 'C:\\Users\\nodejs'")["p"], "C:\\Users\\nodejs")
    assert_eq(loads("q = '\"quoted\"'")["q"], "\"quoted\"")
    assert_eq(loads("r = '<\\i\\c*\\s*>'")["r"], "<\\i\\c*\\s*>")
}

fn test_multiline_basic_strings() -> void {
    assert_eq(loads("s = \"\"\"\nRoses\nViolets\"\"\"")["s"], "Roses\nViolets")
    assert_eq(loads("s = \"\"\"Roses\nViolets\"\"\"")["s"], "Roses\nViolets")
    let folded = "s = \"\"\"\nThe quick \\\n\n\n  brown \\\n       fox.\"\"\""
    assert_eq(loads(folded)["s"], "The quick brown fox.")
    let trailing_ws = "s = \"\"\"a \\   \n   b\"\"\""
    assert_eq(loads(trailing_ws)["s"], "a b")
    assert_eq(loads("s = \"\"\"here \"\" two\"\"\"")["s"], "here \"\" two")
    assert_eq(loads("s = \"\"\"four\"\"\"\"")["s"], "four\"")
    assert_eq(loads("s = \"\"\"five\"\"\"\"\"")["s"], "five\"\"")
    assert_eq(loads("s = \"\"\"\\u0041\\n\"\"\"")["s"], "A\n")
    assert_eq(loads("s = \"\"\"crlf\r\nline\"\"\"")["s"], "crlf\nline")
}

fn test_multiline_literal_strings() -> void {
    assert_eq(loads("s = '''\nfirst\n  second'''")["s"], "first\n  second")
    assert_eq(loads("s = '''no \\escapes \"here\"'''")["s"], "no \\escapes \"here\"")
    assert_eq(loads("s = ''''one quote''''")["s"], "'one quote'")
    assert_eq(loads("s = '''''two'''''")["s"], "''two''")
    assert_eq(loads("s = ''''''")["s"], "")
}

fn test_integers_in_every_base() -> void {
    let doc = "d = +99\nz = 0\nn = -17\nu = 1_000_000\nh = 0xDEADBEEF\nh2 = 0xdead_beef\n" +
        "o = 0o755\nb = 0b1101_0110\nmax = 9223372036854775807\nmin = -9223372036854775808\n" +
        "nz = -0\npz = +0"
    let value = loads(doc)
    assert_eq(value["d"], 99)
    assert_eq(value["z"], 0)
    assert_eq(value["n"], -17)
    assert_eq(value["u"], 1000000)
    assert_eq(value["h"], 3735928559)
    assert_eq(value["h2"], 3735928559)
    assert_eq(value["o"], 493)
    assert_eq(value["b"], 214)
    assert_eq(value["max"], 9223372036854775807)
    assert_eq(value["min"], -9223372036854775807 - 1)
    assert_eq(value["nz"], 0)
    assert_true(isinstance(value["pz"], int))
}

fn test_floats_in_every_spelling() -> void {
    let doc = "a = +1.0\nb = 3.1415\nc = -0.01\nd = 5e+22\ne = 1e06\nf = -2E-2\n" +
        "g = 6.626e-34\nh = 224_617.445_991_228\ni = inf\nj = +inf\nk = -inf\n" +
        "l = nan\nm = +nan\nn = -nan\no = -0.0\np = 1e400"
    let value = loads(doc)
    assert_eq(value["a"], 1.0)
    assert_true(isinstance(value["a"], float))
    assert_eq(value["b"], 3.1415)
    assert_eq(value["c"], -0.01)
    assert_eq(value["d"], 5e22)
    assert_eq(value["e"], 1000000.0)
    assert_eq(value["f"], -0.02)
    assert_eq(value["g"], 6.626e-34)
    assert_eq(value["h"], 224617.445991228)
    assert_eq(value["i"], math.INF)
    assert_eq(value["j"], math.INF)
    assert_eq(value["k"], -math.INF)
    assert_true(math.is_nan(value["l"]))
    assert_true(math.is_nan(value["m"]))
    assert_true(math.is_nan(value["n"]))
    assert_eq(str(value["o"]), "-0.0")
    assert_eq(value["p"], math.INF)
}

fn test_booleans() -> void {
    let value = loads("t = true\nf = false")
    assert_eq(value["t"], true)
    assert_eq(value["f"], false)
    assert_true(isinstance(value["t"], bool))
}

fn test_offset_datetimes() -> void {
    let doc = "a = 1979-05-27T07:32:00Z\nb = 1979-05-27T00:32:00-07:00\n" +
        "c = 1979-05-27T00:32:00.999999-07:00\nd = 1979-05-27 07:32:00Z\n" +
        "e = 1979-05-27t07:32:00z\nf = 1979-05-27T07:32:00.1234567+05:30"
    let value = loads(doc)
    assert_eq(value["a"], DateTime(1979, 5, 27, 7, 32, 0, 0, 0))
    assert_eq(value["a"].to_iso(), "1979-05-27T07:32:00+00:00")
    assert_eq(value["b"], DateTime(1979, 5, 27, 0, 32, 0, 0, -420))
    assert_eq(value["b"].offset_minutes, -420)
    assert_eq(value["c"].microsecond, 999999)
    assert_eq(value["d"], value["a"])
    assert_eq(value["e"], value["a"])
    assert_eq(value["f"].microsecond, 123456)
    assert_eq(value["f"].offset_minutes, 330)
}

fn test_local_datetimes_dates_and_times() -> void {
    let doc = "ldt = 1979-05-27T07:32:00\nldt2 = 1979-05-27T00:32:00.999999\n" +
        "ld = 1979-05-27\nlt = 07:32:00\nlt2 = 00:32:00.999999\nlt3 = 00:32:00.5"
    let value = loads(doc)
    assert_eq(value["ldt"], DateTime(1979, 5, 27, 7, 32))
    assert_false(value["ldt"].is_aware())
    assert_eq(value["ldt2"].microsecond, 999999)
    assert_eq(value["ld"], Date(1979, 5, 27))
    assert_true(isinstance(value["ld"], Date))
    assert_eq(value["lt"], Time(7, 32, 0))
    assert_eq(value["lt2"], Time(0, 32, 0, 999999))
    assert_eq(value["lt3"], Time(0, 32, 0, 500000))
}

fn test_arrays_of_every_shape() -> void {
    let doc = "ints = [1, 2, 3]\nempty = []\nnested = [[1, 2], [\"a\"]]\n" +
        "mixed = [1, 1.5, \"s\", true, 1979-05-27, [2]]\n" +
        "multi = [\n  1, # one\n  2,\n  # three\n  3,\n]\ntables = [{a = 1}, {b = 2}]"
    let value = loads(doc)
    assert_eq(value["ints"], [1, 2, 3])
    assert_eq(value["empty"], [])
    assert_eq(value["nested"], [[1, 2], ["a"]])
    assert_eq(value["mixed"][0:4], [1, 1.5, "s", true])
    assert_eq(value["mixed"][4], Date(1979, 5, 27))
    assert_eq(value["mixed"][5], [2])
    assert_eq(value["multi"], [1, 2, 3])
    assert_eq(value["tables"], [{"a": 1}, {"b": 2}])
}

fn test_tables_and_dotted_headers() -> void {
    let doc = "[a]\nx = 1\n[a.b]\ny = 2\n[ c . \"d e\" ]\nz = 3\n[f.g.h]\n[i]"
    let value = loads(doc)
    assert_eq(
        value,
        {"a": {"x": 1, "b": {"y": 2}}, "c": {"d e": {"z": 3}}, "f": {"g": {"h": {}}}, "i": {}}
    )
}

fn test_super_table_may_be_defined_after_its_child() -> void {
    let value = loads("[x.y.z.w]\n[x]\na = 1")
    assert_eq(value, {"x": {"y": {"z": {"w": {}}}, "a": 1}})
}

fn test_arrays_of_tables() -> void {
    let doc = "[[products]]\nname = \"Hammer\"\nsku = 738594937\n\n[[products]]\n\n" +
        "[[products]]\nname = \"Nail\"\n[products.detail]\ncolor = \"gray\"\n" +
        "[[fruits]]\nname = \"apple\"\n[fruits.physical]\ncolor = \"red\"\n" +
        "[[fruits.varieties]]\nname = \"red delicious\"\n[[fruits.varieties]]\nname = \"granny smith\"\n" +
        "[[fruits]]\nname = \"banana\"\n[[fruits.varieties]]\nname = \"plantain\""
    let value = loads(doc)
    assert_eq(
        value["products"],
        [{"name": "Hammer", "sku": 738594937}, {}, {"name": "Nail", "detail": {"color": "gray"}}]
    )
    assert_eq(
        value["fruits"],
        [
            {
                "name": "apple",
                "physical": {"color": "red"},
                "varieties": [{"name": "red delicious"}, {"name": "granny smith"}],
            },
            {"name": "banana", "varieties": [{"name": "plantain"}]},
        ]
    )
}

fn test_inline_tables() -> void {
    let doc = "name = { first = \"Tom\", last = \"Preston-Werner\" }\npoint = { x = 1, y = 2 }\n" +
        "animal = { type.name = \"pug\" }\nempty = {}\nnested = { a = { b = [1, { c = 2 }] } }"
    let value = loads(doc)
    assert_eq(value["name"], {"first": "Tom", "last": "Preston-Werner"})
    assert_eq(value["point"], {"x": 1, "y": 2})
    assert_eq(value["animal"], {"type": {"name": "pug"}})
    assert_eq(value["empty"], {})
    assert_eq(value["nested"], {"a": {"b": [1, {"c": 2}]}})
}

fn test_dotted_keys_build_and_extend_tables() -> void {
    let doc = "fruit.apple.color = \"red\"\nfruit.apple.taste.sweet = true\n" +
        "[fruit.orange]\ncolor = \"orange\"\n[tbl]\nnested.a = 1\nnested.b = 2"
    let value = loads(doc)
    assert_eq(
        value["fruit"],
        {"apple": {"color": "red", "taste": {"sweet": true}}, "orange": {"color": "orange"}}
    )
    assert_eq(value["tbl"], {"nested": {"a": 1, "b": 2}})
}

fn test_document_order_is_kept() -> void {
    let value = loads("b = 1\na = 2\n[z]\n[y]\nq = 1")
    assert_eq(value.keys(), ["b", "a", "z", "y"])
    assert_eq(loads("[a.b]\n[a]\nc = 1\n[a.d]").keys(), ["a"])
    assert_eq(loads("[a.b]\n[a]\nc = 1\n[a.d]")["a"].keys(), ["b", "c", "d"])
}

fn test_duplicate_keys_are_refused() -> void {
    fails_with("a = 1\na = 2", "Cannot overwrite a value (at end of document)")
    fails_with("a = 1\na = 2\n", "Cannot overwrite a value (at line 2, column 6)")
    fails_with("a.b = 1\na.b = 2\n", "Cannot overwrite a value (at line 2, column 8)")
    fails_with("[t]\nk = 1\nk = 2\n", "Cannot overwrite a value (at line 3, column 6)")
    fails_with("i = { a = 1, a = 2 }", "Duplicate inline table key 'a' (at line 1, column 19)")
    fails_with("a = 1\n[a]", "Cannot overwrite a value (at line 2, column 3)")
    fails_with("a = 1\na.b = 2", "Cannot overwrite a value (at end of document)")
}

fn test_redefining_tables_is_refused() -> void {
    fails_with("[a]\n[a]", "Cannot declare ('a',) twice (at line 2, column 3)")
    fails_with("[a.b]\n[a.b]\n", "Cannot declare ('a', 'b') twice (at line 2, column 5)")
    fails_with("[[a]]\n[a]\n", "Cannot declare ('a',) twice (at line 2, column 3)")
    fails_with("[a]\n[[a]]\n", "Cannot overwrite a value (at line 2, column 4)")
    fails_with("a.b = 1\n[a]\n", "Cannot declare ('a',) twice (at line 2, column 3)")
    fails_with("[a]\nb.c = 1\n[a.b]\n", "Cannot declare ('a', 'b') twice (at line 3, column 5)")
    fails_with("a = {}\n[a.b]\n", "Cannot declare ('a', 'b') twice (at line 2, column 5)")
    fails_with("a = []\n[[a]]\n", "Cannot mutate immutable namespace ('a',) (at line 2, column 4)")
    fails_with(
        "a = {b = 1}\na.c = 2\n",
        "Cannot mutate immutable namespace ('a',) (at line 2, column 8)"
    )
    fails_with("[a]\nb = 1\n[a]\nc = 2", "Cannot declare ('a',) twice (at line 3, column 3)")
    fails_with("[t]\na.b = 1\n[t]\n", "Cannot declare ('t',) twice (at line 3, column 3)")
    fails_with("[x]\n[x.y]\n[x]\n", "Cannot declare ('x',) twice (at line 3, column 3)")
    fails_with(
        "[a.b]\nc = 1\n[a]\nb.d = 2\n",
        "Cannot redefine namespace ('a', 'b') (at line 4, column 8)"
    )
}

fn test_malformed_values_report_their_position() -> void {
    fails_with("= 1", "Invalid statement (at line 1, column 1)")
    fails_with("k", "Expected '=' after a key in a key/value pair (at end of document)")
    fails_with("k =", "Invalid value (at end of document)")
    fails_with("k = ", "Invalid value (at end of document)")
    fails_with("k = @", "Invalid value (at line 1, column 5)")
    fails_with(
        "k = 1 x",
        "Expected newline or end of document after a statement (at line 1, column 7)"
    )
    fails_with(
        "k = 1__2",
        "Expected newline or end of document after a statement (at line 1, column 6)"
    )
    fails_with(
        "k = 01",
        "Expected newline or end of document after a statement (at line 1, column 6)"
    )
    fails_with(
        "k = 1.",
        "Expected newline or end of document after a statement (at line 1, column 6)"
    )
    fails_with("k = .5", "Invalid value (at line 1, column 5)")
    fails_with(
        "k = 0x",
        "Expected newline or end of document after a statement (at line 1, column 6)"
    )
    fails_with("k = tru", "Invalid value (at line 1, column 5)")
    fails_with("[a\n", "Expected ']' at the end of a table declaration (at line 1, column 3)")
    fails_with("[[a]\n", "Expected ']]' at the end of an array declaration (at line 1, column 4)")
    fails_with("[]\n", "Invalid initial character for a key part (at line 1, column 2)")
    fails_with("a = [1, 2", "Unclosed array (at end of document)")
    fails_with("a = [1 2]", "Unclosed array (at line 1, column 8)")
    fails_with("a = [1,,2]", "Invalid value (at line 1, column 8)")
    fails_with("a = {b = 1,}", "Invalid initial character for a key part (at line 1, column 12)")
    fails_with("a = {b = 1\n}", "Unclosed inline table (at line 1, column 11)")
    fails_with("a = { b = 1 c = 2 }", "Unclosed inline table (at line 1, column 13)")
    fails_with("\"a\nb\" = 1", "Illegal character '\\n' (at line 1, column 3)")
    fails_with(
        "a = 1 # ok\nb = 2 \x01",
        "Expected newline or end of document after a statement (at line 2, column 7)"
    )
    fails_with("a = 1 # bad \x7f\n", "Found invalid character '\\x7f' (at line 1, column 13)")
}

fn test_malformed_strings_report_their_position() -> void {
    fails_with("s = \"abc", "Unterminated string (at end of document)")
    fails_with("s = \"abc\ndef\"", "Illegal character '\\n' (at line 1, column 9)")
    fails_with("s = \"\\q\"", "Unescaped '\\' in a string (at line 1, column 8)")
    fails_with("s = \"\\x41\"", "Unescaped '\\' in a string (at line 1, column 8)")
    fails_with("s = \"\\u12\"", "Invalid hex value (at line 1, column 8)")
    fails_with(
        "s = \"\\uD800\"",
        "Escaped character is not a Unicode scalar value (at line 1, column 12)"
    )
    fails_with(
        "s = \"\\U00110000\"",
        "Escaped character is not a Unicode scalar value (at line 1, column 16)"
    )
    fails_with("s = 'abc", "Expected \"'\" (at end of document)")
    fails_with("s = 'a\x00b'", "Found invalid character '\\x00' (at line 1, column 7)")
    fails_with("s = '''abc", "Expected \"'''\" (at end of document)")
    fails_with("s = \"\"\"abc", "Unterminated string (at end of document)")
    fails_with("s = \"\"\"a \\ b\"\"\"", "Unescaped '\\' in a string (at line 1, column 12)")
    fails_with("s = \"\"\"a\x01\"\"\"", "Illegal character '\\x01' (at line 1, column 9)")
    fails_with("s = \"\\\"", "Unterminated string (at end of document)")
}

fn test_malformed_dates_and_times() -> void {
    fails_with("d = 2020-02-30", "Invalid date or datetime (at line 1, column 5)")
    fails_with("d = 0000-01-01", "Invalid date or datetime (at line 1, column 5)")
    fails_with(
        "d = 2020-13-01",
        "Expected newline or end of document after a statement (at line 1, column 9)"
    )
    fails_with(
        "t = 24:00:00",
        "Expected newline or end of document after a statement (at line 1, column 7)"
    )
    fails_with(
        "t = 23:59:60",
        "Expected newline or end of document after a statement (at line 1, column 7)"
    )
    fails_with(
        "t = 07:32",
        "Expected newline or end of document after a statement (at line 1, column 6)"
    )
    fails_with(
        "d = 2020-01-01T",
        "Expected newline or end of document after a statement (at line 1, column 15)"
    )
    fails_with(
        "d = 2020-01-01T07:32:00+24:00",
        "Expected newline or end of document after a statement (at line 1, column 24)"
    )
    fails_with(
        "d = 2020-01-01T07:32:00.",
        "Expected newline or end of document after a statement (at line 1, column 24)"
    )
}

fn test_integers_beyond_64_bits_are_refused() -> void {
    fails_with(
        "i = 9223372036854775808",
        "Integer is outside the 64-bit range this runtime can hold (at line 1, column 5)"
    )
    fails_with(
        "i = -9223372036854775809",
        "Integer is outside the 64-bit range this runtime can hold (at line 1, column 5)"
    )
    fails_with(
        "i = 0xFFFF_FFFF_FFFF_FFFF",
        "Integer is outside the 64-bit range this runtime can hold (at line 1, column 5)"
    )
    assert_eq(loads("i = 0x7FFF_FFFF_FFFF_FFFF")["i"], 9223372036854775807)
}

fn test_decode_error_carries_the_position_as_numbers() -> void {
    let error = assert_throws(TOMLDecodeError, fn() -> void {
        let _ = loads("a = 1\nb = [1,\n  @]\n")
    }, "a bad array element")
    assert_eq(error.msg, "Invalid value")
    assert_eq(error.lineno, 3)
    assert_eq(error.colno, 3)
    assert_eq(error.pos, 16)
    assert_eq(error.doc, "a = 1\nb = [1,\n  @]\n")
    assert_eq(error.message, "Invalid value (at line 3, column 3)")
    let at_end = assert_throws(TOMLDecodeError, fn() -> void { let _ = loads("a =") }, "eof")
    assert_eq(at_end.lineno, 1)
    assert_eq(at_end.colno, 4)
    assert_true(isinstance(at_end, ValueError))
    let bare = TOMLDecodeError("plain")
    assert_eq(bare.message, "plain (at end of document)")
}

fn test_parse_float_replaces_the_float_type() -> void {
    let value = loads("a = 1.5\nb = inf\nc = 2\nd = -nan", parse_float: |text| "float:" + text)
    assert_eq(value["a"], "float:1.5")
    assert_eq(value["b"], "float:inf")
    assert_eq(value["c"], 2)
    assert_eq(value["d"], "float:-nan")
    assert_eq(loads("h = 2_0.5e1_0", parse_float: |text| text)["h"], "2_0.5e1_0")
    assert_throws(ValueError, fn() -> void {
        let _ = loads("a = 1.5", parse_float: |_text| [1])
    }, "a list from parse_float")
}

fn test_load_reads_a_file() -> void {
    let path = "/tmp/jaitoml-test-load.toml"
    write_file(path, "[section]\nkey = \"value\"\nn = 0x10\n")
    let file = open(path)
    defer { file.close() }
    assert_eq(load(file), {"section": {"key": "value", "n": 16}})
}

fn test_a_real_config_file() -> void {
    let doc = "# This is a TOML document\n\ntitle = \"TOML Example\"\n\n[owner]\n" +
        "name = \"Tom Preston-Werner\"\ndob = 1979-05-27T07:32:00-08:00\n\n[database]\n" +
        "enabled = true\nports = [ 8000, 8001, 8002 ]\ndata = [ [\"delta\", \"phi\"], [3.14] ]\n" +
        "temp_targets = { cpu = 79.5, case = 72.0 }\n\n[servers]\n\n[servers.alpha]\n" +
        "ip = \"10.0.0.1\"\nrole = \"frontend\"\n\n[servers.beta]\nip = \"10.0.0.2\"\nrole = \"backend\"\n"
    let value = loads(doc)
    assert_eq(value["title"], "TOML Example")
    assert_eq(value["owner"]["dob"], DateTime(1979, 5, 27, 7, 32, 0, 0, -480))
    assert_eq(value["database"]["ports"], [8000, 8001, 8002])
    assert_eq(value["database"]["data"], [["delta", "phi"], [3.14]])
    assert_eq(value["database"]["temp_targets"], {"cpu": 79.5, "case": 72.0})
    assert_eq(value["servers"]["beta"]["role"], "backend")
}

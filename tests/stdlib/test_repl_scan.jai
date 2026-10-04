#: The prompt's scan places a line and a column by CHARACTER on a buffer it now
#: reads as bytes.
#:
#: `jaithon.compile.repl` decides whether a buffer is whole, open or wrongly
#: closed, and reports where. The lexer's spans are byte offsets, and the scan
#: used to convert them through a list of the buffer's characters rebuilt on
#: every keystroke-level call. It walks the buffer's bytes now and counts a
#: column for each byte that is not a UTF-8 continuation byte, so a column after
#: a multi-byte character has to come out the same as it did.
from std.test import assert_eq, assert_true, assert_false
from jaithon.compile.repl import scan, OpenKind

fn test_a_closer_after_a_two_byte_character_is_placed_by_character() -> void {
    # `é` is two bytes; the `]` is the 14th character and at byte 14.
    let out = scan("let é = (1, 2]\n")
    assert_true(out.mismatched)
    assert_eq(out.closer, "]")
    assert_eq(out.closer_line, 1)
    assert_eq(out.closer_col, 14)
    assert_eq(out.opener, "(")
    assert_eq(out.opener_line, 1)
}

fn test_a_closer_after_four_byte_characters_on_a_later_line() -> void {
    let out = scan("x = [\n  \"😀😀\",\n  1)\n")
    assert_true(out.mismatched)
    assert_eq(out.closer, ")")
    assert_eq(out.closer_line, 3)
    assert_eq(out.closer_col, 4)
    assert_eq(out.opener, "[")
    assert_eq(out.opener_line, 1)
}

fn test_an_open_bracket_reports_its_line_past_multi_byte_lines() -> void {
    let out = scan("# ünïcödé\nlet ok = 1\nf(\n")
    assert_true(out.incomplete)
    assert_eq(out.open, OpenKind.Bracket)
    assert_eq(out.opener, "(")
    assert_eq(out.opener_line, 3)
}

fn test_a_triple_quoted_string_left_open_keeps_reading() -> void {
    let out = scan("let é = 1\ns = r\"\"\"ünïcödé\n")
    assert_true(out.incomplete)
    assert_eq(out.open, OpenKind.String)
    assert_eq(out.opener_line, 2)
}

fn test_a_one_line_string_left_open_is_not_more_input() -> void {
    let out = scan("s = \"é\n")
    assert_false(out.incomplete)
}

fn test_an_open_block_comment_keeps_reading() -> void {
    let out = scan("x = 1\n#* ünïcödé\n")
    assert_true(out.incomplete)
    assert_eq(out.open, OpenKind.Comment)
    assert_eq(out.opener_line, 2)
}

fn test_a_whole_line_with_multi_byte_text_runs() -> void {
    let out = scan("print(\"日本語\")\n")
    assert_false(out.incomplete)
    assert_false(out.mismatched)
}

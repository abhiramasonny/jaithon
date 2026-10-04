#: A token records byte offsets and a shared `Source`, and computes `span` and
#: `text` from them.
#:
#: `text` used to be a substring the lexer cut for every token, by character
#: index, while `start` and `end` are byte offsets. On a file with a
#: multi-byte character the two disagree after it, so the text a token computes
#: has to be checked against the bytes it covers, not against itself.
from std.test import assert_eq, assert_true
from jaithon.ast import Span
from jaithon.compile.lexer import Lexer
from jaithon.compile.token import Token, TokenKind, keyword_lookup

#: A BOM, multi-byte names, strings, an f-string, comments and a label.
const _SOURCE: str = "\u{feff}let café = \"naïve 😀\"  # ünïcödé\n" +
    "fn größe(日本: int) -> int {\n" +
    "    #* blöck *# let s = f\"{日本}é{café}\"\n" +
    "    'äußer: loop { break 'äußer }\n" +
    "    return 日本 + 1\n" +
    "}\n"

#: The text between two byte offsets, decoded from the bytes themselves.
fn _decoded(bytes: list[int], start: int, end: int) -> str {
    return __prim__.str_decode(__prim__.bytes_new(bytes[start:end]))
}

fn _tokens(file_id: int) -> list[Token] {
    return Lexer(_SOURCE, file_id).tokenize()
}

fn test_text_is_exactly_the_bytes_a_token_covers() -> void {
    let bytes = _SOURCE.bytes()
    var non_ascii = 0
    for token in _tokens(3) {
        let expected = _decoded(bytes, token.start, token.end)
        assert_eq(token.text, expected, f"{token}")
        assert_eq(token.slice_text(), expected, f"{token}")
        if expected.len() != token.end - token.start { non_ascii += 1 }
    }
    assert_true(non_ascii >= 6, "the probe has to cover multi-byte tokens")
}

fn test_span_carries_the_file_and_the_byte_offsets() -> void {
    for token in _tokens(7) {
        let span = token.span
        assert_eq(span, Span(7, token.start, token.end))
        assert_eq(token.make_span(), span)
        assert_eq(span.text(_SOURCE), token.text)
    }
}

fn test_every_word_keeps_its_text_and_nothing_else_does() -> void {
    var names: list[str] = []
    for token in _tokens(0) {
        let is_word = token.kind == TokenKind.Ident or
            token.kind == TokenKind.Underscore or
            keyword_lookup(token.text) == token.kind
        if is_word {
            assert_eq(token.word, token.text)
        } else {
            assert_eq(token.word, "")
        }
        if token.kind == TokenKind.Ident { names.push(token.word) }
    }
    assert_true("café" in names)
    assert_true("größe" in names)
    assert_true("日本" in names)
}

fn test_the_bom_is_not_part_of_the_first_token() -> void {
    let first = _tokens(0)[0]
    assert_eq(first.kind, TokenKind.KwLet)
    assert_eq(first.start, 3)
    assert_eq(first.text, "let")
}

fn test_a_string_keeps_its_cooked_value_beside_its_text() -> void {
    for token in _tokens(0) {
        if token.kind == TokenKind.Str {
            assert_eq(token.text, "\"naïve 😀\"")
            assert_eq(token.string_value(), "naïve 😀")
            return
        }
    }
    assert_true(false, "no string token")
}

fn test_the_last_token_ends_the_file() -> void {
    let tokens = _tokens(0)
    let eof = tokens[tokens.len() - 1]
    assert_eq(eof.kind, TokenKind.Eof)
    assert_eq(eof.start, _SOURCE.bytes().len())
    assert_eq(eof.text, "")
}

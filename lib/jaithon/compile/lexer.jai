#: The Jaithon tokenizer: source text in, `list[Token]` out.
#:
#: Mirrors `src/lang/lexer.c` token-for-token (spec §11); `make bootstrap`
#: diffs both front ends over `lib/` and `tests/` to enforce it. Lexical
#: errors recover locally into a shared `DiagnosticBag`, so the parser always
#: sees a well-formed stream ending in one `TokenKind.Eof`.
#:
#: Token vocabulary (`TokenKind`, `TokenFlags`, `Token`) lives in `token.jai`,
#: and shared diagnostics (`Code`, `Severity`, `Diagnostic`, `DiagnosticBag`)
#: in `diag.jai` — neither carries lexer state, so both are free-standing.
#: This file keeps only the scanner: the `Lexer` state machine that walks
#: `position`/`line`/the bracket stack and turns characters into tokens.
#:
#: The scanner walks the source's UTF-8 bytes, a `list[int]`, and every
#: position it keeps is a byte offset -- which is also what a `Span` records,
#: so a span needs no conversion. It used to walk a `list[str]` of one-character
#: strings with a parallel table of byte offsets: one object per character
#: before the first token, and a loop the JIT could not compile. Only a token's
#: text goes back to `str`, through `_text`.
#:
#: ```
#: let tokens = Lexer("x = 1").tokenize()
#: tokens.len()                    # 4: `x`, `=`, `1`, end of file
#: tokens[0].kind == TokenKind.Ident      # true
#: tokens[2].value                 # 1
#: ```
from jaithon.ast import Span
from std.str import StringBuilder
from jaithon.compile.diag import Code, Diagnostic, DiagnosticBag
from jaithon.compile.token import TokenKind, TokenFlags, Token
from jaithon.compile.token import keyword_lookup, kind_name, kind_text

#: How deep f-string interpolations may nest before `E0009`.
pub const MAX_INTERPOLATION_DEPTH: int = 8

const _BRK_PAREN = 1
const _BRK_BRACKET = 2
const _BRK_BRACE_BLOCK = 3
const _BRK_BRACE_EXPR = 4
const _BRK_INTERP = 8
const _BRK_F_SINGLE = 1
const _BRK_F_TRIPLE = 2
const _BRK_F_RAW = 4

# The bytes the scanner compares against, by name.
const _C_TAB = 9
const _C_LF = 10
const _C_VT = 11
const _C_FF = 12
const _C_CR = 13
const _C_SPACE = 32
const _C_BANG = 33
const _C_DQUOTE = 34
const _C_HASH = 35
const _C_PERCENT = 37
const _C_AMP = 38
const _C_SQUOTE = 39
const _C_LPAREN = 40
const _C_RPAREN = 41
const _C_STAR = 42
const _C_PLUS = 43
const _C_COMMA = 44
const _C_MINUS = 45
const _C_DOT = 46
const _C_SLASH = 47
const _C_ZERO = 48
const _C_NINE = 57
const _C_COLON = 58
const _C_SEMICOLON = 59
const _C_LT = 60
const _C_EQ = 61
const _C_GT = 62
const _C_QUESTION = 63
const _C_AT = 64
const _C_UPPER_A = 65
const _C_UPPER_B = 66
const _C_UPPER_E = 69
const _C_UPPER_F = 70
const _C_UPPER_O = 79
const _C_UPPER_X = 88
const _C_UPPER_Z = 90
const _C_LBRACKET = 91
const _C_BACKSLASH = 92
const _C_RBRACKET = 93
const _C_CARET = 94
const _C_UNDERSCORE = 95
const _C_LOWER_A = 97
const _C_LOWER_B = 98
const _C_LOWER_E = 101
const _C_LOWER_F = 102
const _C_LOWER_O = 111
const _C_LOWER_R = 114
const _C_LOWER_U = 117
const _C_LOWER_X = 120
const _C_LOWER_Z = 122
const _C_LBRACE = 123
const _C_PIPE = 124
const _C_RBRACE = 125
const _C_TILDE = 126

enum _ChunkStop { Closed, Interp, Broken }

class _StrFlavor {
    #: The quote byte, `"` or `'`.
    pub let quote: int
    pub let triple: bool
    pub let raw: bool
    pub let interp: bool

    pub fn init(self, quote: int, triple: bool, raw: bool, interp: bool) {
        self.quote = quote
        self.triple = triple
        self.raw = raw
        self.interp = interp
    }

    pub fn closer(self) -> str {
        let quote = chr(self.quote)
        if self.triple { return quote + quote + quote }
        return quote
    }
}

#: The tokenizer.
#:
#: Construct one per source file; a second call to `tokenize` re-scans from
#: scratch and replaces the token list.
pub class Lexer {
    #: The source text, exactly as handed to `init`.
    pub let source: str
    #: The file id every `Span` this lexer produces refers to.
    pub let file_id: int
    #: Where diagnostics go; shared with the parser.
    pub var diagnostics: DiagnosticBag
    #: True once a lexical error has been reported.
    pub var had_error: bool

    #: The source's UTF-8 bytes. Every position below indexes this.
    let bytes: list[int]
    #: Whether every byte is ASCII, so that a byte offset is also a `str` index.
    let ascii: bool
    #: When the source is not ASCII: the `str` index of each byte offset, and
    #: of the end. Otherwise `[0]` and never read -- but not empty, because an
    #: empty list gives the tier no element to learn from and it then refused
    #: the whole of `_text`, which every token goes through.
    let scalar_of: list[int]
    let length: int
    var tokens: list[Token]
    var pos: int
    var line: int
    var brackets: list[int]
    var interp_depth: int
    var at_line_start: bool

    #: Prepare to tokenize `source`.
    pub fn init(self, source: str, file_id: int = 0, diagnostics: DiagnosticBag? = null) {
        self.source = source
        self.file_id = file_id
        self.diagnostics = diagnostics ?? DiagnosticBag()
        self.had_error = false
        let bytes = source.bytes()
        self.bytes = bytes
        self.length = bytes.len()
        self.ascii = self.length == source.len()
        self.scalar_of = if self.ascii { [0] } else { _scalar_offsets(bytes) }
        self.tokens = []
        self.pos = 0
        self.line = 1
        self.brackets = []
        self.interp_depth = 0
        self.at_line_start = true
    }

    #: Scan the whole source and return its tokens, ending with one `Eof`.
    #:
    #: ```
    #: let tokens = Lexer("f(1)\n").tokenize()
    #: [token.text for token in tokens]     # ["f", "(", "1", ")", "\n", ""]
    #: ```
    pub fn tokenize(self) -> list[Token] {
        self.tokens = []
        self.brackets = []
        self.interp_depth = 0
        self.had_error = false
        self.pos = 0
        self.line = 1
        self.at_line_start = true
        let bytes = self.bytes
        if self.length >= 3 and bytes[0] == 0xEF and bytes[1] == 0xBB and bytes[2] == 0xBF {
            self.pos = 3
        }

        while self.pos < self.length { self._scan_token() }

        if self.interp_depth > 0 {
            self._error(
                Code.E0008_UNTERMINATED_INTERPOLATION,
                self.length,
                self.length,
                "unterminated f-string interpolation at end of file",
                help: "close it with `}` and the string's quote"
            )
            while self.brackets.len() > 0 {
                let marker = self.brackets.pop()
                if marker >= _BRK_INTERP {
                    self._push_empty_string(TokenKind.FStringEnd, self.length)
                }
            }
            self.interp_depth = 0
        }

        self._push(TokenKind.Eof, self.length, self.length)
        return self.tokens
    }

    #: The source text between two byte offsets, as `Span` records them.
    pub fn text_of(self, span: Span) -> str {
        return self._text(self._clamp(span.start), self._clamp(span.end))
    }

    # --- cursor ----------------------------------------------------------
    fn _clamp(self, byte_offset: int) -> int {
        if byte_offset < 0 { return 0 }
        if byte_offset > self.length { return self.length }
        return byte_offset
    }

    #: The source text between two byte offsets.
    fn _text(self, start: int, end: int) -> str {
        if self.ascii { return self.source[start:end] }
        return self.source[self.scalar_of[start]:self.scalar_of[end]]
    }

    #: The byte at `index`, or -1 past either end.
    fn _at(self, index: int) -> int {
        if index < 0 or index >= self.length { return -1 }
        return self.bytes[index]
    }

    fn _peek(self) -> int {
        return self._at(self.pos)
    }

    fn _peek_at(self, offset: int) -> int {
        return self._at(self.pos + offset)
    }

    fn _eat(self, character: int) -> bool {
        if self._peek() != character { return false }
        self.pos += 1
        return true
    }

    #: How many bytes the character starting at `index` takes.
    fn _width_at(self, index: int) -> int {
        let lead = self.bytes[index]
        var width = 1
        if lead >= 0xF0 {
            width = 4
        } elif lead >= 0xE0 {
            width = 3
        } elif lead >= 0xC0 {
            width = 2
        }
        let left = self.length - index
        return if width > left { left } else { width }
    }

    #: The code point of the character starting at `index`.
    fn _code_at(self, index: int) -> int {
        let lead = self.bytes[index]
        let width = self._width_at(index)
        if width == 1 { return lead }
        var code = 0
        if width == 2 {
            code = lead & 0x1F
        } elif width == 3 {
            code = lead & 0x0F
        } else {
            code = lead & 0x07
        }
        var at = index + 1
        while at < index + width {
            code = code << 6 | self.bytes[at] & 0x3F
            at += 1
        }
        return code
    }

    fn _span(self, start: int, end: int) -> Span {
        return Span(self.file_id, start, end)
    }

    fn _error(
        self,
        code: str,
        start: int,
        end: int,
        message: str,
        help: str? = null,
        note: str? = null
    ) -> Diagnostic {
        self.had_error = true
        return self.diagnostics.error(code, self._span(start, end), message, help: help, note: note)
    }

    fn _last_kind(self) -> TokenKind {
        if self.tokens.len() == 0 { return TokenKind.Eof }
        return self.tokens[self.tokens.len() - 1].kind
    }

    #: Record a token that carries no value: punctuation, a keyword, a name.
    #:
    #: Kept apart from `_push_value` because a parameter that is sometimes an
    #: int, sometimes a string and usually null has no kind the tier can give
    #: it, and that alone kept every token push interpreted.
    fn _push(self, kind: TokenKind, start: int, end: int) -> void {
        self._push_text(kind, start, end, self._text(start, end))
    }

    #: `_push` for a caller that already sliced the token's text.
    fn _push_text(self, kind: TokenKind, start: int, end: int, text: str) -> void {
        var flags = 0
        if self.at_line_start { flags = TokenFlags.AFTER_NEWLINE }
        self.tokens.push(Token(kind, Span(self.file_id, start, end), text, flags))
        self.at_line_start = false
    }

    #: Record a literal token and its cooked value.
    fn _push_value(self, kind: TokenKind, start: int, end: int, value: any, flags: int) -> void {
        var all_flags = flags
        if self.at_line_start { all_flags = all_flags | TokenFlags.AFTER_NEWLINE }
        let token = Token(kind, Span(self.file_id, start, end), self._text(start, end), all_flags)
        token.value = value
        self.tokens.push(token)
        self.at_line_start = false
    }

    # --- newlines (§1.3) -------------------------------------------------
    fn _bracket_top(self) -> int {
        if self.brackets.len() == 0 { return 0 }
        return self.brackets[self.brackets.len() - 1]
    }

    fn _bracket_top_is_interp(self) -> bool {
        return self.brackets.len() > 0 and self._bracket_top() >= _BRK_INTERP
    }

    fn _bracket_top_is_expression(self) -> bool {
        return self.brackets.len() > 0 and self._bracket_top() != _BRK_BRACE_BLOCK
    }

    fn _ends_wildcard_import(self) -> bool {
        let count = self.tokens.len()
        if count < 2 { return false }
        return self.tokens[count - 1].kind == TokenKind.Star and
            self.tokens[count - 2].kind == TokenKind.KwImport
    }

    fn _handle_newline(self) -> void {
        let start = self.pos
        self.pos += 1
        self.line += 1
        let last = self._last_kind()
        let significant = self.tokens.len() > 0 and
            not self._bracket_top_is_expression() and
            last != TokenKind.Newline and
            (self._ends_wildcard_import() or not _is_continuation(last))
        if significant { self._push(TokenKind.Newline, start, start + 1) }
        self.at_line_start = true
    }

    fn _brace_opens_expression(self) -> bool {
        let previous = self._last_kind()
        if _is_operator(previous) { return true }
        if previous == TokenKind.LBrace {
            return self._bracket_top_is_expression()
        }
        return match previous {
            TokenKind.Comma | TokenKind.Colon | TokenKind.LParen | TokenKind.LBracket => true,
            TokenKind.QuestionQuestion => true,
            TokenKind.KwReturn | TokenKind.KwThrow | TokenKind.KwYield => true,
            TokenKind.KwIn | TokenKind.KwIs | TokenKind.KwAnd | TokenKind.KwOr => true,
            TokenKind.KwNot | TokenKind.FStringStart | TokenKind.FStringMid => true,
            _ => false,
        }
    }

    # --- comments --------------------------------------------------------
    fn _skip_comment(self) -> void {
        let start = self.pos
        let bytes = self.bytes
        let length = self.length
        if self._peek_at(1) != _C_STAR {
            var at = start
            while at < length and bytes[at] != _C_LF { at += 1 }
            self.pos = at
            return
        }

        self.pos += 2
        var depth = 1
        while self.pos < length {
            let character = bytes[self.pos]
            if character == _C_HASH and self._peek_at(1) == _C_STAR {
                depth += 1
                self.pos += 2
                continue
            }
            if character == _C_STAR and self._peek_at(1) == _C_HASH {
                self.pos += 2
                depth -= 1
                if depth == 0 { return }
                continue
            }
            if character == _C_LF { self.line += 1 }
            self.pos += 1
        }

        let diag = self._error(
            Code.E0002_UNTERMINATED_COMMENT,
            start,
            start + 2,
            "unterminated block comment",
            help: "close it with `*#`"
        )
        if depth > 1 { diag.with_note(f"{depth} nested `#*` comments are still open") }
    }

    # --- numbers (§1.6) --------------------------------------------------
    #: The digits of a numeral, underscores removed; the text is only
    #: meaningful when the flag is true.
    fn _scan_digits(self, base: int) -> tuple[bool, str] {
        let start = self.pos
        var previous_underscore = false
        var any = false
        var underscores = false
        loop {
            let character = self._peek()
            if character == _C_UNDERSCORE {
                if not any or previous_underscore {
                    var message = "numeric literal may not start with `_`"
                    if previous_underscore { message = "repeated `_` in numeric literal" }
                    self._error(
                        Code.E0005_INVALID_NUMBER,
                        self.pos,
                        self.pos + 1,
                        message,
                        help: "digit separators must sit between digits"
                    )
                    while self._peek() == _C_UNDERSCORE or _digit_in_base(self._peek(), base) {
                        self.pos += 1
                    }
                    return (false, "")
                }
                previous_underscore = true
                underscores = true
                self.pos += 1
                continue
            }
            if not _digit_in_base(character, base) { break }
            any = true
            previous_underscore = false
            self.pos += 1
        }
        if previous_underscore {
            self._error(
                Code.E0005_INVALID_NUMBER,
                self.pos - 1,
                self.pos,
                "numeric literal may not end with `_`",
                help: "digit separators must sit between digits"
            )
            return (false, "")
        }
        let digits = self._text(start, self.pos)
        if underscores { return (true, digits.replace("_", "")) }
        return (true, digits)
    }

    fn _scan_number(self) -> void {
        let start = self.pos
        var base = 10
        var is_float = false
        var ok = true
        var text = ""

        let prefix = self._peek_at(1)
        if self._peek() == _C_ZERO and _base_of(prefix) > 0 {
            base = _base_of(prefix)
            self.pos += 2
            let (digits_ok, digits) = self._scan_digits(base)
            ok = digits_ok
            text = digits
            if ok and digits.len() == 0 {
                ok = false
                self._error(
                    Code.E0005_INVALID_NUMBER,
                    start,
                    self.pos,
                    f"missing digits after `0{chr(prefix)}`",
                    help: f"write at least one base-{base} digit"
                )
            }
        } else {
            let (digits_ok, digits) = self._scan_digits(10)
            ok = digits_ok
            text = digits

            if ok and self._peek() == _C_DOT and _is_digit(self._peek_at(1)) {
                is_float = true
                self.pos += 1
                let (fraction_ok, fraction) = self._scan_digits(10)
                ok = fraction_ok
                text = text + "." + fraction
            }
            if ok and (self._peek() == _C_LOWER_E or self._peek() == _C_UPPER_E) {
                let sign = self._peek_at(1)
                let digit_offset = sign == _C_PLUS or sign == _C_MINUS ? 2 : 1
                if _is_digit(self._peek_at(digit_offset)) {
                    is_float = true
                    self.pos += digit_offset
                    let (exponent_ok, exponent) = self._scan_digits(10)
                    ok = exponent_ok
                    text = text + "e" + (digit_offset == 2 ? chr(sign) : "") + exponent
                }
            }
        }

        if ok { ok = self._reject_number_suffix(base, start) }

        if is_float {
            let value = ok ? float(text) : 0.0
            self._push_value(TokenKind.Float, start, self.pos, value, 0)
            return
        }
        var value = 0
        var flags = 0
        if ok {
            value = self._parse_int_text(text, base, start) ?? 0
            if value == _INT_MIN and self._is_negated_literal() {
                flags = TokenFlags.NEG_MAGNITUDE
            }
        }
        self._push_value(TokenKind.Int, start, self.pos, value, flags)
    }

    # Decimal literals must fit signed `int` (except the smallest int's
    # magnitude under a prefix `-`); hex/octal/binary are bit patterns, so the
    # full unsigned range is accepted and reinterpreted two's-complement, as
    # the wrapping operators below produce.
    fn _parse_int_text(self, digits: str, base: int, start: int) -> int? {
        var first = 0
        while first < digits.len() and digits[first] == "0" { first += 1 }
        let significant = digits[first:digits.len()]
        if significant.len() == 0 { return 0 }

        if not _fits_in_int(significant, base, self._is_negated_literal()) {
            self._error(
                Code.E0005_INVALID_NUMBER,
                start,
                self.pos,
                "integer literal does not fit in `int`",
                note: f"`int` is 64-bit signed: {_INT_MIN} to {_INT_MAX}"
            )
            return null
        }
        var value = 0
        for character in significant.bytes() {
            value = value *% base +% _hex_value(character)
        }
        return value
    }

    fn _is_negated_literal(self) -> bool {
        let count = self.tokens.len()
        if count == 0 or self.tokens[count - 1].kind != TokenKind.Minus { return false }
        if count == 1 { return true }
        let before = self.tokens[count - 2].kind
        return before != TokenKind.Minus and not _ends_operand(before)
    }

    fn _reject_number_suffix(self, base: int, start: int) -> bool {
        let character = self._peek()
        if character < 0 { return true }
        if not _is_ident_start(character) and not _is_digit(character) and character < 0x80 {
            return true
        }

        let bad = self.pos
        while self.pos < self.length and (_is_ident_cont(self._peek()) or self._peek() >= 0x80) {
            self.pos += 1
        }
        var help = "Jaithon has no literal suffixes; separate the number from the name"
        if _is_digit(character) { help = f"`{chr(character)}` is not a base-{base} digit" }
        let diag = self._error(
            Code.E0005_INVALID_NUMBER,
            start,
            self.pos,
            "invalid suffix on numeric literal",
            help: help
        )
        diag.label(self._span(bad, self.pos), "not part of a number")
        return false
    }

    # --- strings (§1.6, §1.7) --------------------------------------------
    fn _flush(self, out: StringBuilder, from_index: int, to_index: int) -> void {
        if to_index > from_index { out.push(self._text(from_index, to_index)) }
    }

    fn _scan_escape(self, out: StringBuilder) -> bool {
        let escape_start = self.pos
        self.pos += 1
        if self.pos >= self.length { return false }
        let code = self.bytes[self.pos]
        let width = self._width_at(self.pos)
        let character = if width == 1 { chr(code) } else { self._text(self.pos, self.pos + width) }
        self.pos += width

        if code == _C_LOWER_X { return self._scan_hex_escape(out, escape_start) }
        if code == _C_LOWER_U { return self._scan_unicode_escape(out, escape_start) }
        if character in _SIMPLE_ESCAPES {
            out.push(_SIMPLE_ESCAPES[character])
            return true
        }
        self._error(
            Code.E0004_INVALID_ESCAPE,
            escape_start,
            self.pos,
            f"unknown escape sequence `\\{character}`",
            help: "valid escapes are \\n \\t \\r \\\\ \\\" \\' \\0 \\xNN \\u{...}"
        )
        out.push(character)
        return false
    }

    fn _scan_hex_escape(self, out: StringBuilder, escape_start: int) -> bool {
        var value = 0
        for _digit_index in 0..2 {
            let digit = _hex_value(self._peek())
            if digit < 0 {
                self._error(
                    Code.E0004_INVALID_ESCAPE,
                    escape_start,
                    self.pos,
                    "`\\x` needs exactly two hex digits",
                    help: "for example `\\x1b`"
                )
                return false
            }
            value = value * 16 + digit
            self.pos += 1
        }
        out.push(chr(value))
        return true
    }

    fn _scan_unicode_escape(self, out: StringBuilder, escape_start: int) -> bool {
        if self._peek() != _C_LBRACE {
            self._error(
                Code.E0004_INVALID_ESCAPE,
                escape_start,
                self.pos,
                "`\\u` must be followed by `{...}`",
                help: "for example `\\u{1F600}`"
            )
            return false
        }
        self.pos += 1
        var value = 0
        var digits = 0
        while _hex_value(self._peek()) >= 0 {
            if digits < 6 { value = value * 16 + _hex_value(self._peek()) }
            digits += 1
            self.pos += 1
        }
        let closed = self._eat(_C_RBRACE)
        if digits == 0 or digits > 6 or not closed {
            self._error(
                Code.E0004_INVALID_ESCAPE,
                escape_start,
                self.pos,
                "malformed `\\u{...}` escape",
                help: "write one to six hex digits inside the braces"
            )
            return false
        }
        if value > 0x10FFFF or value >= 0xD800 and value <= 0xDFFF {
            self._error(
                Code.E0004_INVALID_ESCAPE,
                escape_start,
                self.pos,
                f"U+{value:04X} is not a Unicode scalar value",
                note: "scalars are U+0000..U+D7FF and U+E000..U+10FFFF"
            )
            return false
        }
        out.push(chr(value))
        return true
    }

    fn _scan_string_chunk(
        self,
        flavor: _StrFlavor,
        out: StringBuilder,
        open_start: int
    ) -> tuple[_ChunkStop, bool] {
        let bytes = self.bytes
        let length = self.length
        let quote = flavor.quote
        let triple = flavor.triple
        let interp = flavor.interp
        var saw_escape = false
        var run_start = self.pos
        loop {
            # Plain text runs up to the next byte that can mean something.
            var at = self.pos
            while at < length {
                let byte = bytes[at]
                if byte == quote or
                    byte == _C_LF or
                    byte == _C_CR or
                    byte == _C_BACKSLASH or
                    byte == _C_LBRACE or
                    byte == _C_RBRACE {
                    break
                }
                at += 1
            }
            self.pos = at

            if self.pos >= length {
                self._flush(out, run_start, self.pos)
                self._unterminated(flavor, open_start, false)
                return (_ChunkStop.Broken, saw_escape)
            }

            let character = bytes[self.pos]

            if character == quote {
                if not triple {
                    self._flush(out, run_start, self.pos)
                    self.pos += 1
                    return (_ChunkStop.Closed, saw_escape)
                }
                if self._peek_at(1) == quote and self._peek_at(2) == quote {
                    self._flush(out, run_start, self.pos)
                    self.pos += 3
                    return (_ChunkStop.Closed, saw_escape)
                }
                self.pos += 1
                continue
            }

            if character == _C_CR and self._peek_at(1) == _C_LF {
                self._flush(out, run_start, self.pos)
                self.pos += 1
                run_start = self.pos
                continue
            }

            if character == _C_LF {
                if not triple {
                    self._flush(out, run_start, self.pos)
                    self._unterminated(flavor, open_start, true)
                    return (_ChunkStop.Broken, saw_escape)
                }
                self.pos += 1
                self.line += 1
                continue
            }

            if character == _C_BACKSLASH {
                if flavor.raw {
                    self.pos += 1
                    continue
                }
                self._flush(out, run_start, self.pos)
                if self._scan_escape(out) { saw_escape = true }
                run_start = self.pos
                continue
            }

            if interp and (character == _C_LBRACE or character == _C_RBRACE) {
                self._flush(out, run_start, self.pos)
                if self._peek_at(1) == character {
                    out.push(if character == _C_LBRACE { "{" } else { "}" })
                    self.pos += 2
                    run_start = self.pos
                    continue
                }
                if character == _C_LBRACE {
                    self.pos += 1
                    return (_ChunkStop.Interp, saw_escape)
                }
                self._single_close_brace()
                out.push("}")
                self.pos += 1
                run_start = self.pos
                continue
            }

            self.pos += 1
        }
    }

    # The string scanner's error reports, out of its loop.
    fn _unterminated(self, flavor: _StrFlavor, open_start: int, at_end_of_line: bool) -> void {
        if at_end_of_line {
            self._error(
                _unterminated_code(flavor),
                open_start,
                self.pos,
                f"unterminated {_string_noun(flavor)} at end of line",
                help: "use a triple-quoted string to span lines"
            )
            return
        }
        self._error(
            _unterminated_code(flavor),
            open_start,
            self.pos,
            f"unterminated {_string_noun(flavor)}",
            help: f"add a closing {flavor.closer()}"
        )
    }

    fn _single_close_brace(self) -> void {
        self._error(
            Code.E0003_INVALID_CHARACTER,
            self.pos,
            self.pos + 1,
            "single `}` in f-string text",
            help: "write `}}` for a literal brace"
        )
    }

    fn _open_interpolation(self, flavor: _StrFlavor) -> void {
        if self.interp_depth >= MAX_INTERPOLATION_DEPTH {
            self._error(
                Code.E0009_NESTED_INTERPOLATION_TOO_DEEP,
                self.pos - 1,
                self.pos,
                "f-string interpolation nested more than " + f"{MAX_INTERPOLATION_DEPTH} deep",
                help: "compute the value into a binding before formatting it"
            )
        }
        self.interp_depth += 1
        var marker = _BRK_INTERP
        if flavor.quote == _C_SQUOTE { marker = marker | _BRK_F_SINGLE }
        if flavor.triple { marker = marker | _BRK_F_TRIPLE }
        if flavor.raw { marker = marker | _BRK_F_RAW }
        self.brackets.push(marker)
    }

    fn _push_empty_string(self, kind: TokenKind, at: int) -> void {
        self._push_value(kind, at, at, "", 0)
    }

    fn _resume_fstring(self, flavor: _StrFlavor, token_start: int) -> void {
        let out = StringBuilder()
        let (stop, saw_escape) = self._scan_string_chunk(flavor, out, token_start)
        let kind = stop == _ChunkStop.Interp ? TokenKind.FStringMid : TokenKind.FStringEnd
        self._push_value(
            kind,
            token_start,
            self.pos,
            out.build(),
            _string_flags(flavor, saw_escape)
        )
        if stop == _ChunkStop.Interp { self._open_interpolation(flavor) }
    }

    fn _scan_string_literal(self, token_start: int, raw: bool, interp: bool) -> void {
        let quote = self.bytes[self.pos]
        let triple = self._peek_at(1) == quote and self._peek_at(2) == quote
        let flavor = _StrFlavor(quote, triple, raw, interp)
        self.pos += triple ? 3 : 1

        let out = StringBuilder()
        let (stop, saw_escape) = self._scan_string_chunk(flavor, out, token_start)
        let flags = _string_flags(flavor, saw_escape)
        if not interp {
            self._push_value(TokenKind.Str, token_start, self.pos, out.build(), flags)
            return
        }

        self._push_value(TokenKind.FStringStart, token_start, self.pos, out.build(), flags)
        if stop == _ChunkStop.Interp {
            self._open_interpolation(flavor)
        } else {
            self._push_empty_string(TokenKind.FStringEnd, self.pos)
        }
    }

    fn _scan_format_spec(self) -> void {
        let start = self.pos
        let bytes = self.bytes
        var depth = 0
        while self.pos < self.length {
            let character = bytes[self.pos]
            if character == _C_LF { break }
            if character == _C_LBRACE {
                depth += 1
            } elif character == _C_RBRACE {
                if depth == 0 { break }
                depth -= 1
            }
            self.pos += 1
        }
        self._push_value(TokenKind.Str, start, self.pos, self._text(start, self.pos), 0)
    }

    # --- identifiers and labels ------------------------------------------
    fn _ident_run_end(self, start: int) -> int {
        let bytes = self.bytes
        let length = self.length
        var index = start
        while index < length {
            let code = bytes[index]
            if code < 0x80 {
                if code >= _C_LOWER_A and code <= _C_LOWER_Z or
                    code >= _C_UPPER_A and code <= _C_UPPER_Z or
                    code >= _C_ZERO and code <= _C_NINE or
                    code == _C_UNDERSCORE {
                    index += 1
                    continue
                }
                break
            }
            if not _is_unicode_ident(self._code_at(index)) { break }
            index += self._width_at(index)
        }
        return index
    }

    fn _scan_identifier(self) -> void {
        let start = self.pos
        self.pos = self._ident_run_end(start)
        let text = self._text(start, self.pos)

        let next = self._peek()
        if self.pos - start <= 2 and (next == _C_DQUOTE or next == _C_SQUOTE) {
            var raw = false
            var interp = false
            var is_prefix = true
            var index = start
            while index < self.pos {
                let character = self.bytes[index]
                if character == _C_LOWER_R and not raw {
                    raw = true
                } elif character == _C_LOWER_F and not interp {
                    interp = true
                } else {
                    is_prefix = false
                    break
                }
                index += 1
            }
            if is_prefix and (raw or interp) {
                self._scan_string_literal(start, raw, interp)
                return
            }
        }

        var kind = keyword_lookup(text)
        if kind == TokenKind.Ident and text == "_" { kind = TokenKind.Underscore }
        self._push_text(kind, start, self.pos, text)
    }

    fn _at_statement_start(self) -> bool {
        return match self._last_kind() {
            TokenKind.Eof | TokenKind.Newline | TokenKind.Semicolon => true,
            TokenKind.LBrace | TokenKind.RBrace => true,
            _ => false,
        }
    }

    fn _scan_quote_or_label(self) -> void {
        let start = self.pos
        let after = start + 1
        let first = self._at(after)
        var ident_start = false
        if first >= 0 {
            ident_start = first < 0x80 ? _is_ident_start(first) : _is_unicode_ident(self._code_at(after))
        }

        if ident_start {
            let end = self._ident_run_end(after)
            let closed_by_quote = self._at(end) == _C_SQUOTE
            var probe = end
            while self._at(probe) == _C_SPACE or self._at(probe) == _C_TAB { probe += 1 }
            let declaration = self._at(probe) == _C_COLON and self._at_statement_start()
            let usage = self._last_kind() == TokenKind.KwBreak or
                self._last_kind() == TokenKind.KwContinue
            if not closed_by_quote and (declaration or usage) {
                self.pos = end
                self._push(TokenKind.Label, start, end)
                return
            }
        }

        self._scan_string_literal(start, false, false)
    }

    # --- punctuation and operators ---------------------------------------
    fn _pop_bracket(self, expected: int) -> void {
        if self.brackets.len() > 0 and self._bracket_top() == expected {
            self.brackets.pop()
        }
    }

    fn _pop_brace_bracket(self) -> void {
        let top = self._bracket_top()
        if self.brackets.len() > 0 and (top == _BRK_BRACE_BLOCK or top == _BRK_BRACE_EXPR) {
            self.brackets.pop()
        }
    }

    fn _close_brace(self, start: int) -> void {
        if self._bracket_top_is_interp() {
            let last = self._last_kind()
            if last == TokenKind.FStringStart or last == TokenKind.FStringMid {
                self._error(
                    Code.E0102_EXPECTED_EXPRESSION,
                    start,
                    self.pos,
                    "empty interpolation in f-string",
                    help: "write `{{}}` for literal braces"
                )
            }
            let marker = self.brackets.pop()
            if self.interp_depth > 0 { self.interp_depth -= 1 }
            self._resume_fstring(_flavor_from_marker(marker), start)
            return
        }
        if self.tokens.len() > 0 and self._last_kind() == TokenKind.Newline {
            self.tokens.pop()
        }
        self._pop_brace_bracket()
        self._push(TokenKind.RBrace, start, self.pos)
    }

    fn _scan_punct(self) -> void {
        let start = self.pos
        let character = self.bytes[self.pos]
        self.pos += 1

        if character == _C_RBRACE {
            self._close_brace(start)
            return
        }
        if character == _C_COLON {
            self._push(TokenKind.Colon, start, self.pos)
            if self._bracket_top_is_interp() { self._scan_format_spec() }
            return
        }
        if character == _C_BANG {
            if self._eat(_C_EQ) {
                self._push(TokenKind.BangEq, start, self.pos)
                return
            }
            self._error(
                Code.E0003_INVALID_CHARACTER,
                start,
                self.pos,
                "unexpected character `!`",
                help: "Jaithon spells logical negation `not` and inequality `!=`"
            )
            return
        }

        let kind = self._punct_kind(character)
        if kind == null {
            let code = character
            var message = f"unexpected character `{chr(character)}`"
            if code < 0x20 or code == 0x7F {
                message = f"unexpected control character U+{code:04X}"
            }
            self._error(Code.E0003_INVALID_CHARACTER, start, self.pos, message)
            return
        }
        self._push(kind, start, self.pos)
    }

    fn _punct_kind(self, character: int) -> TokenKind? {
        if character == _C_LPAREN {
            self.brackets.push(_BRK_PAREN)
            return TokenKind.LParen
        }
        if character == _C_LBRACKET {
            self.brackets.push(_BRK_BRACKET)
            return TokenKind.LBracket
        }
        if character == _C_LBRACE {
            self.brackets.push(self._brace_opens_expression() ? _BRK_BRACE_EXPR : _BRK_BRACE_BLOCK)
            return TokenKind.LBrace
        }
        if character == _C_RPAREN {
            self._pop_bracket(_BRK_PAREN)
            return TokenKind.RParen
        }
        if character == _C_RBRACKET {
            self._pop_bracket(_BRK_BRACKET)
            return TokenKind.RBracket
        }
        if character == _C_COMMA { return TokenKind.Comma }
        if character == _C_SEMICOLON { return TokenKind.Semicolon }
        if character == _C_AT { return TokenKind.At }
        if character == _C_TILDE { return TokenKind.Tilde }
        if character == _C_DOT {
            if not self._eat(_C_DOT) { return TokenKind.Dot }
            if self._eat(_C_DOT) { return TokenKind.Ellipsis }
            if self._eat(_C_EQ) { return TokenKind.DotDotEq }
            return TokenKind.DotDot
        }
        if character == _C_PLUS {
            if self._eat(_C_EQ) { return TokenKind.PlusEq }
            if self._eat(_C_PERCENT) { return TokenKind.PlusPercent }
            return TokenKind.Plus
        }
        if character == _C_MINUS {
            if self._eat(_C_GT) { return TokenKind.Arrow }
            if self._eat(_C_EQ) { return TokenKind.MinusEq }
            if self._eat(_C_PERCENT) { return TokenKind.MinusPercent }
            return TokenKind.Minus
        }
        if character == _C_STAR {
            if self._eat(_C_STAR) {
                return self._eat(_C_EQ) ? TokenKind.StarStarEq : TokenKind.StarStar
            }
            if self._eat(_C_EQ) { return TokenKind.StarEq }
            if self._eat(_C_PERCENT) { return TokenKind.StarPercent }
            return TokenKind.Star
        }
        if character == _C_SLASH {
            if self._eat(_C_SLASH) {
                return self._eat(_C_EQ) ? TokenKind.SlashSlashEq : TokenKind.SlashSlash
            }
            if self._eat(_C_EQ) { return TokenKind.SlashEq }
            return TokenKind.Slash
        }
        if character == _C_PERCENT {
            return self._eat(_C_EQ) ? TokenKind.PercentEq : TokenKind.Percent
        }
        if character == _C_AMP { return self._eat(_C_EQ) ? TokenKind.AmpEq : TokenKind.Amp }
        if character == _C_PIPE { return self._eat(_C_EQ) ? TokenKind.PipeEq : TokenKind.Pipe }
        if character == _C_CARET {
            return self._eat(_C_EQ) ? TokenKind.CaretEq : TokenKind.Caret
        }
        if character == _C_LT {
            if self._eat(_C_LT) { return self._eat(_C_EQ) ? TokenKind.ShlEq : TokenKind.Shl }
            return self._eat(_C_EQ) ? TokenKind.Le : TokenKind.Lt
        }
        if character == _C_GT {
            if self._eat(_C_GT) { return self._eat(_C_EQ) ? TokenKind.ShrEq : TokenKind.Shr }
            return self._eat(_C_EQ) ? TokenKind.Ge : TokenKind.Gt
        }
        if character == _C_EQ {
            if self._eat(_C_EQ) { return TokenKind.EqEq }
            if self._eat(_C_GT) { return TokenKind.FatArrow }
            return TokenKind.Eq
        }
        if character == _C_QUESTION {
            if self._eat(_C_QUESTION) { return TokenKind.QuestionQuestion }
            if self._eat(_C_DOT) { return TokenKind.QuestionDot }
            return TokenKind.Question
        }
        return null
    }

    fn _line_continuation(self) -> void {
        let start = self.pos
        var next = start + 1
        if self._at(next) == _C_CR { next += 1 }
        if self._at(next) == _C_LF {
            self.pos = next + 1
            self.line += 1
            return
        }
        self.pos += 1
        self._error(
            Code.E0003_INVALID_CHARACTER,
            start,
            self.pos,
            "stray `\\` outside a string literal",
            help: "a backslash continues a line only at end of line"
        )
    }

    #: A character outside ASCII where a token may start. Kept out of
    #: `_scan_token` so that the common path never calls a method whose
    #: result it has not yet seen come back, which stops the tier compiling it.
    fn _scan_non_ascii(self) -> void {
        let code = self._code_at(self.pos)
        let width = self._width_at(self.pos)
        if _is_unicode_space(code) {
            self.pos += width
            return
        }
        if not _is_unicode_ident(code) {
            self._error(
                Code.E0003_INVALID_CHARACTER,
                self.pos,
                self.pos + width,
                f"unexpected character U+{code:04X}",
                help: "only ASCII punctuation is meaningful outside " + "strings and comments"
            )
            self.pos += width
            return
        }
        self._scan_identifier()
    }

    # --- the main loop ---------------------------------------------------
    fn _scan_token(self) -> void {
        let character = self.bytes[self.pos]

        if character == _C_SPACE or
            character == _C_TAB or
            character == _C_CR or
            character == _C_VT or
            character == _C_FF {
            self.pos += 1
            return
        }
        if character == _C_LF {
            self._handle_newline()
            return
        }
        if character == _C_HASH {
            self._skip_comment()
            return
        }
        if character == _C_BACKSLASH {
            self._line_continuation()
            return
        }

        if character >= 0x80 {
            self._scan_non_ascii()
            return
        }

        # Spelled out rather than `_is_digit` and `_is_ident_start`: a call to a
        # function that has not compiled yet ends the compiled walk of this
        # body there, and which one compiles first depends on the file.
        if character >= _C_ZERO and character <= _C_NINE {
            self._scan_number()
            return
        }
        if character >= _C_LOWER_A and character <= _C_LOWER_Z or
            character >= _C_UPPER_A and character <= _C_UPPER_Z or
            character == _C_UNDERSCORE {
            self._scan_identifier()
            return
        }
        if character == _C_DQUOTE {
            self._scan_string_literal(self.pos, false, false)
            return
        }
        if character == _C_SQUOTE {
            self._scan_quote_or_label()
            return
        }
        self._scan_punct()
    }
}

const _INT_MAX = 0x7FFFFFFFFFFFFFFF
const _INT_MIN = (-9223372036854775808)

let _SIMPLE_ESCAPES: dict[str, str] = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "\\": "\\",
    "\"": "\"",
    "'": "'",
    "0": "\0",
}

#: The `str` index of every byte offset of `bytes`, and of its end: the
#: number of characters that start before it.
fn _scalar_offsets(bytes: list[int]) -> list[int] {
    var out: list[int] = []
    var scalar = 0
    for byte in bytes {
        out.push(scalar)
        if byte & 0xC0 != 0x80 { scalar += 1 }
    }
    out.push(scalar)
    return out
}

fn _is_digit(character: int) -> bool {
    return character >= _C_ZERO and character <= _C_NINE
}

fn _is_alpha(character: int) -> bool {
    return character >= _C_LOWER_A and character <= _C_LOWER_Z or
        character >= _C_UPPER_A and character <= _C_UPPER_Z
}

fn _is_ident_start(character: int) -> bool {
    return _is_alpha(character) or character == _C_UNDERSCORE
}

fn _is_ident_cont(character: int) -> bool {
    return _is_ident_start(character) or _is_digit(character)
}

fn _hex_value(character: int) -> int {
    if _is_digit(character) { return character - _C_ZERO }
    if character >= _C_LOWER_A and character <= _C_LOWER_F { return character - _C_LOWER_A + 10 }
    if character >= _C_UPPER_A and character <= _C_UPPER_F { return character - _C_UPPER_A + 10 }
    return -1
}

fn _digit_in_base(character: int, base: int) -> bool {
    let value = _hex_value(character)
    return value >= 0 and value < base
}

fn _base_of(prefix: int) -> int {
    if prefix == _C_LOWER_X or prefix == _C_UPPER_X { return 16 }
    if prefix == _C_LOWER_O or prefix == _C_UPPER_O { return 8 }
    if prefix == _C_LOWER_B or prefix == _C_UPPER_B { return 2 }
    return 0
}

fn _fits_in_int(digits: str, base: int, negated: bool) -> bool {
    let count = digits.len()
    if base == 16 { return count <= 16 }
    if base == 8 { return count < 22 or count == 22 and digits[0] <= "1" }
    if base == 2 { return count <= 64 }
    if count < 19 { return true }
    if count > 19 { return false }
    if digits <= "9223372036854775807" { return true }
    return negated and digits == "9223372036854775808"
}

fn _is_unicode_space(code: int) -> bool {
    if code == 0x00A0 or code == 0x1680 or code == 0x3000 or code == 0xFEFF { return true }
    if code >= 0x2000 and code <= 0x200D { return true }
    return code == 0x2028 or code == 0x2029 or code == 0x202F or code == 0x205F
}

fn _is_unicode_ident(code: int) -> bool {
    if code < 0x80 { return false }
    if _is_unicode_space(code) { return false }
    if code >= 0x2000 and code <= 0x206F { return false }
    if code >= 0xFFF9 and code <= 0xFFFB { return false }
    return true
}

fn _is_operator(kind: TokenKind) -> bool {
    return match kind {
        TokenKind.Plus | TokenKind.Minus | TokenKind.Star | TokenKind.Slash => true,
        TokenKind.SlashSlash | TokenKind.Percent | TokenKind.StarStar => true,
        TokenKind.PlusPercent | TokenKind.MinusPercent | TokenKind.StarPercent => true,
        TokenKind.Amp | TokenKind.Pipe | TokenKind.Caret | TokenKind.Tilde => true,
        TokenKind.Shl | TokenKind.Shr | TokenKind.Eq | TokenKind.EqEq => true,
        #: `@`, the matrix product, was left out when it was added. Being in
        #: this list is what lets a line carry on after an operator and what
        #: lets `{` after one open a set rather than a block, so `a @` at the
        #: end of a wrapped line did not parse and `a @ { 1, 2 }` read the
        #: brace as a block. Found by auditing this table against the operators
        #: the formatter is willing to break a line after (ast.jai's
        #: _OP_SOURCE), which is the set that has to be covered here.
        TokenKind.At => true,
        TokenKind.BangEq | TokenKind.Lt | TokenKind.Le | TokenKind.Gt | TokenKind.Ge => true,
        TokenKind.PlusEq | TokenKind.MinusEq | TokenKind.StarEq | TokenKind.SlashEq => true,
        TokenKind.SlashSlashEq | TokenKind.PercentEq | TokenKind.StarStarEq => true,
        TokenKind.AmpEq | TokenKind.PipeEq | TokenKind.CaretEq => true,
        TokenKind.ShlEq | TokenKind.ShrEq => true,
        _ => false,
    }
}

fn _is_continuation(kind: TokenKind) -> bool {
    if _is_operator(kind) { return true }
    return match kind {
        TokenKind.Comma | TokenKind.Dot | TokenKind.DotDot | TokenKind.DotDotEq => true,
        TokenKind.Colon | TokenKind.Arrow | TokenKind.FatArrow => true,
        TokenKind.LParen | TokenKind.LBracket | TokenKind.LBrace => true,
        TokenKind.QuestionDot | TokenKind.QuestionQuestion => true,
        TokenKind.KwAnd | TokenKind.KwOr | TokenKind.KwNot => true,
        #: `in` and `is` continue a line for the same reason `and` does: they
        #: are binary operators that cannot end an expression, so a newline
        #: after one can only be a wrapped line.
        #:
        #: They were missing, and the formatter is what found it. `fmt` breaks
        #: a long line after a binary operator, so a 101-character
        #: `x = a.text("name") in b[b.len() - 1]` came back as two lines with
        #: the first ending in `in` -- which then did not parse. Formatting a
        #: file until it stops changing is supposed to be a fixpoint, and
        #: `make fmt-roundtrip` caught it as "UNPARSEABLE AFTER". Adding the
        #: two here is the fix on the right side: the formatter's break is
        #: reasonable and the lexer's refusal to continue was the odd one out.
        TokenKind.KwIn | TokenKind.KwIs => true,
        TokenKind.FStringStart | TokenKind.FStringMid => true,
        _ => false,
    }
}

fn _ends_operand(kind: TokenKind) -> bool {
    return match kind {
        TokenKind.Ident | TokenKind.Int | TokenKind.Float | TokenKind.Str => true,
        TokenKind.FStringEnd | TokenKind.Label | TokenKind.Underscore => true,
        TokenKind.RParen | TokenKind.RBracket | TokenKind.RBrace => true,
        TokenKind.KwTrue | TokenKind.KwFalse | TokenKind.KwNull => true,
        TokenKind.KwSelf | TokenKind.KwSuper => true,
        _ => false,
    }
}

fn _flavor_from_marker(marker: int) -> _StrFlavor {
    let quote = marker & _BRK_F_SINGLE != 0 ? _C_SQUOTE : _C_DQUOTE
    return _StrFlavor(quote, marker & _BRK_F_TRIPLE != 0, marker & _BRK_F_RAW != 0, true)
}

fn _string_flags(flavor: _StrFlavor, saw_escape: bool) -> int {
    var flags = 0
    if saw_escape { flags = flags | TokenFlags.HAS_ESCAPE }
    if flavor.raw { flags = flags | TokenFlags.RAW_STRING }
    if flavor.triple { flags = flags | TokenFlags.TRIPLE_STRING }
    return flags
}

fn _unterminated_code(flavor: _StrFlavor) -> str {
    if flavor.interp { return Code.E0008_UNTERMINATED_INTERPOLATION }
    return Code.E0001_UNTERMINATED_STRING
}

fn _string_noun(flavor: _StrFlavor) -> str {
    return flavor.interp ? "f-string" : "string literal"
}

#: The token stream of `source`, in the format `jaithon tokens` prints.
#:
#: A token with no text of its own prints the kind's display form, so end of
#: file reads as `<end of file>` rather than as nothing.
#:
#: ```
#: dump_tokens("x\n").split("\n")[0].contains("TOK_IDENT")     # true
#: ```
pub fn dump_tokens(source: str, file_id: int = 0) -> str {
    let lexer = Lexer(source, file_id, DiagnosticBag())
    let tokens = lexer.tokenize()

    # Lines and columns count characters, so a continuation byte moves neither.
    let bytes = source.bytes()
    let out = StringBuilder()
    var index = 0
    var at = 0
    var line = 1
    var column = 1
    for token in tokens {
        while at < bytes.len() and at < token.span.start {
            let byte = bytes[at]
            if byte == _C_LF {
                line += 1
                column = 1
            } elif byte & 0xC0 != 0x80 {
                column += 1
            }
            at += 1
        }

        out.push(f"{index:>5}  {kind_name(token.kind):<22} {line:>4}:{column:<4} ")
        if token.text.len() == 0 {
            out.push(kind_text(token.kind))
        } else {
            for character in token.text {
                if character == "\n" {
                    out.push("\\n")
                } elif character == "\t" {
                    out.push("\\t")
                } elif character == "\r" {
                    out.push("\\r")
                } else {
                    out.push(character)
                }
            }
        }
        out.push("\n")
        index += 1
    }
    return out.build()
}

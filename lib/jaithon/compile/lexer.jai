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

# Bracket-stack markers; every kind but `_BRK_BRACE_BLOCK` suppresses newlines
# (§1.3), since it alone did not open an expression. An interpolation pushes
# `_BRK_INTERP` or-ed with the string's flavour bits, so a nested f-string
# lexes on this same stack with no recursion needed.
const _BRK_PAREN = 1
const _BRK_BRACKET = 2
const _BRK_BRACE_BLOCK = 3
const _BRK_BRACE_EXPR = 4
const _BRK_INTERP = 8
const _BRK_F_SINGLE = 1
const _BRK_F_TRIPLE = 2
const _BRK_F_RAW = 4

enum _ChunkStop { Closed, Interp, Broken }

class _StrFlavor {
    pub let quote: str
    pub let triple: bool
    pub let raw: bool
    pub let interp: bool

    pub fn init(self, quote: str, triple: bool, raw: bool, interp: bool) {
        self.quote = quote
        self.triple = triple
        self.raw = raw
        self.interp = interp
    }

    pub fn closer(self) -> str {
        if self.triple { return self.quote + self.quote + self.quote }
        return self.quote
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

    let chars: list[str]
    let offsets: list[int]
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
        self.chars = [character for character in source]
        var byte_offset = 0
        var offsets: list[int] = []
        for character in self.chars {
            offsets.push(byte_offset)
            byte_offset += _utf8_length(ord(character))
        }
        offsets.push(byte_offset)
        self.offsets = offsets
        self.length = self.chars.len()
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
        # A leading byte-order mark is not source text.
        if self.length > 0 and self.chars[0] == "\u{FEFF}" { self.pos = 1 }

        while self.pos < self.length { self._scan_token() }

        # An interpolation still open at EOF gets a synthetic empty `FStringEnd`,
        # so the parser always sees a balanced START/END sequence.
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
        return self.source[self._scalar_index(span.start):self._scalar_index(span.end)]
    }

    # --- cursor ----------------------------------------------------------
    fn _at(self, index: int) -> str {
        if index < 0 or index >= self.length { return "" }
        return self.chars[index]
    }

    fn _peek(self) -> str {
        return self._at(self.pos)
    }

    fn _peek_at(self, offset: int) -> str {
        return self._at(self.pos + offset)
    }

    fn _eat(self, character: str) -> bool {
        if self._peek() != character { return false }
        self.pos += 1
        return true
    }

    fn _span(self, start: int, end: int) -> Span {
        return Span(self.file_id, self.offsets[start], self.offsets[end])
    }

    # Byte offsets only ever land on scalar boundaries, so a binary search over
    # the offset table is exact.
    fn _scalar_index(self, byte_offset: int) -> int {
        var low = 0
        var high = self.offsets.len() - 1
        while low < high {
            let middle = (low + high) // 2
            if self.offsets[middle] < byte_offset { low = middle + 1 } else { high = middle }
        }
        return low
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

    fn _push(
        self,
        kind: TokenKind,
        start: int,
        end: int,
        value: any = null,
        flags: int = 0
    ) -> void {
        var all_flags = flags
        if self.at_line_start { all_flags = all_flags | TokenFlags.AFTER_NEWLINE }
        self.tokens.push(
            Token(kind, self._span(start, end), self.source[start:end], value, all_flags)
        )
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

    # `import *` is the one place `*` ends a statement, not an expression —
    # without this, the newline after it is swallowed as an operator
    # continuation and two consecutive `from x import *` lines fail to parse.
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

    # Ambiguity resolves to BLOCK: mistaking a block for an expression swallows
    # its statement-separating newlines, while the reverse only leaves harmless
    # NEWLINE tokens the parser skips. So only an unmistakable expression
    # position opens an expression.
    fn _brace_opens_expression(self) -> bool {
        let previous = self._last_kind()
        if _is_operator(previous) { return true }
        if previous == TokenKind.LBrace {
            # `{{` — inherit from the brace we are already inside.
            return self._bracket_top_is_expression()
        }
        # `?` is deliberately absent (as in `_is_continuation`): it's the
        # optional-type suffix, so including it would misparse
        # `fn f() -> int? { ... }` as an expression brace.
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
        if self._peek_at(1) != "*" {
            while self.pos < self.length and self._peek() != "\n" { self.pos += 1 }
            return
        }

        self.pos += 2
        var depth = 1
        while self.pos < self.length {
            if self._peek() == "#" and self._peek_at(1) == "*" {
                depth += 1
                self.pos += 2
                continue
            }
            if self._peek() == "*" and self._peek_at(1) == "#" {
                self.pos += 2
                depth -= 1
                if depth == 0 { return }
                continue
            }
            if self._peek() == "\n" { self.line += 1 }
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
    fn _scan_digits(self, base: int) -> tuple[bool, str] {
        let digits = StringBuilder()
        var previous_underscore = false
        var any = false
        loop {
            let character = self._peek()
            if character == "_" {
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
                    while self._peek() == "_" or _digit_in_base(self._peek(), base) {
                        self.pos += 1
                    }
                    return (false, digits.build())
                }
                previous_underscore = true
                self.pos += 1
                continue
            }
            if not _digit_in_base(character, base) { break }
            digits.push(character)
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
            return (false, digits.build())
        }
        return (true, digits.build())
    }

    fn _scan_number(self) -> void {
        let start = self.pos
        var base = 10
        var is_float = false
        var ok = true
        var text = ""

        let prefix = self._peek_at(1)
        if self._peek() == "0" and _base_of(prefix) > 0 {
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
                    f"missing digits after `0{prefix}`",
                    help: f"write at least one base-{base} digit"
                )
            }
        } else {
            let (digits_ok, digits) = self._scan_digits(10)
            ok = digits_ok
            text = digits

            # A `.` belongs to the number only when a digit follows it, so
            # `0..10` is INT DOTDOT INT and `1.foo` is INT DOT IDENT.
            if ok and self._peek() == "." and _is_digit(self._peek_at(1)) {
                is_float = true
                self.pos += 1
                let (fraction_ok, fraction) = self._scan_digits(10)
                ok = fraction_ok
                text = text + "." + fraction
            }
            if ok and (self._peek() == "e" or self._peek() == "E") {
                let sign = self._peek_at(1)
                let digit_offset = sign == "+" or sign == "-" ? 2 : 1
                if _is_digit(self._peek_at(digit_offset)) {
                    is_float = true
                    self.pos += digit_offset
                    let (exponent_ok, exponent) = self._scan_digits(10)
                    ok = exponent_ok
                    text = text + "e" + (digit_offset == 2 ? sign : "") + exponent
                }
            }
        }

        if ok { ok = self._reject_number_suffix(base, start) }

        if is_float {
            let value = ok ? float(text) : 0.0
            self._push(TokenKind.Float, start, self.pos, value)
            return
        }
        # A literal that does not fit was already reported; it lexes as 0 so that
        # the parser still sees a well-formed token stream.
        var value = 0
        var flags = 0
        if ok {
            value = self._parse_int_text(text, base, start) ?? 0
            # Only 9223372036854775808 is the magnitude of the smallest `int`;
            # recognising the sign+magnitude pair here saves the parser a
            # second read of the digits.
            if value == _INT_MIN and self._is_negated_literal() {
                flags = TokenFlags.NEG_MAGNITUDE
            }
        }
        self._push(TokenKind.Int, start, self.pos, value, flags)
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
        for character in significant {
            value = value *% base +% _hex_value(character)
        }
        return value
    }

    # The smallest int's magnitude is admitted only right after a prefix `-`,
    # decided by token kind rather than whitespace — keeping `a -1` a
    # subtraction, `- x` a negation, and disqualifying `--9223372036854775808`
    # (not a valid `int`).
    fn _is_negated_literal(self) -> bool {
        let count = self.tokens.len()
        if count == 0 or self.tokens[count - 1].kind != TokenKind.Minus { return false }
        if count == 1 { return true }
        let before = self.tokens[count - 2].kind
        return before != TokenKind.Minus and not _ends_operand(before)
    }

    # A letter or digit glued to the end of a literal is never valid: there are
    # no literal suffixes, so `2f` and `0b102` are typos, not two tokens.
    fn _reject_number_suffix(self, base: int, start: int) -> bool {
        let character = self._peek()
        if character == "" { return true }
        if not _is_ident_start(character) and not _is_digit(character) and ord(character) < 0x80 {
            return true
        }

        let bad = self.pos
        while self.pos < self.length and (_is_ident_cont(self._peek()) or ord(self._peek()) >= 0x80) {
            self.pos += 1
        }
        var help = "Jaithon has no literal suffixes; separate the number from the name"
        if _is_digit(character) { help = f"`{character}` is not a base-{base} digit" }
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
        if to_index > from_index { out.push(self.source[from_index:to_index]) }
    }

    # Returns false when the escape was rejected; the backslash is always
    # consumed regardless, so the caller still makes progress.
    fn _scan_escape(self, out: StringBuilder) -> bool {
        let escape_start = self.pos
        self.pos += 1
        if self.pos >= self.length { return false }
        let character = self.chars[self.pos]
        self.pos += 1

        if character == "x" { return self._scan_hex_escape(out, escape_start) }
        if character == "u" { return self._scan_unicode_escape(out, escape_start) }
        # Checked by membership, not the cooked string's length: `\0` cooks to
        # a NUL-only string, so length would ask the wrong question.
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
        out.push(character)  # recover as if the backslash were absent
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
        out.push(chr(value))  # \xNN names a scalar, encoded as UTF-8
        return true
    }

    fn _scan_unicode_escape(self, out: StringBuilder, escape_start: int) -> bool {
        if self._peek() != "{" {
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
        let closed = self._eat("}")
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
        var saw_escape = false
        var run_start = self.pos
        loop {
            if self.pos >= self.length {
                self._flush(out, run_start, self.pos)
                self._error(
                    _unterminated_code(flavor),
                    open_start,
                    self.pos,
                    f"unterminated {_string_noun(flavor)}",
                    help: f"add a closing {flavor.closer()}"
                )
                return (_ChunkStop.Broken, saw_escape)
            }

            let character = self.chars[self.pos]

            if character == flavor.quote {
                if not flavor.triple {
                    self._flush(out, run_start, self.pos)
                    self.pos += 1
                    return (_ChunkStop.Closed, saw_escape)
                }
                if self._peek_at(1) == flavor.quote and self._peek_at(2) == flavor.quote {
                    self._flush(out, run_start, self.pos)
                    self.pos += 3
                    return (_ChunkStop.Closed, saw_escape)
                }
                self.pos += 1
                continue
            }

            if character == "\r" and self._peek_at(1) == "\n" {
                self._flush(out, run_start, self.pos)  # §1.1: the \r of a CRLF is dropped
                self.pos += 1
                run_start = self.pos
                continue
            }

            if character == "\n" {
                if not flavor.triple {
                    # Leave the newline for the main loop so that the next line
                    # still lexes as statements.
                    self._flush(out, run_start, self.pos)
                    self._error(
                        _unterminated_code(flavor),
                        open_start,
                        self.pos,
                        f"unterminated {_string_noun(flavor)} at end of line",
                        help: "use a triple-quoted string to span lines"
                    )
                    return (_ChunkStop.Broken, saw_escape)
                }
                self.pos += 1
                self.line += 1
                continue
            }

            if character == "\\" {
                if flavor.raw {
                    self.pos += 1  # raw strings keep the backslash
                    continue
                }
                self._flush(out, run_start, self.pos)
                if self._scan_escape(out) { saw_escape = true }
                run_start = self.pos
                continue
            }

            if flavor.interp and (character == "{" or character == "}") {
                self._flush(out, run_start, self.pos)
                if self._peek_at(1) == character {
                    out.push(character)
                    self.pos += 2
                    run_start = self.pos
                    continue
                }
                if character == "{" {
                    self.pos += 1
                    return (_ChunkStop.Interp, saw_escape)
                }
                self._error(
                    Code.E0003_INVALID_CHARACTER,
                    self.pos,
                    self.pos + 1,
                    "single `}` in f-string text",
                    help: "write `}}` for a literal brace"
                )
                out.push("}")
                self.pos += 1
                run_start = self.pos
                continue
            }

            self.pos += 1
        }
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
        if flavor.quote == "'" { marker = marker | _BRK_F_SINGLE }
        if flavor.triple { marker = marker | _BRK_F_TRIPLE }
        if flavor.raw { marker = marker | _BRK_F_RAW }
        self.brackets.push(marker)
    }

    fn _push_empty_string(self, kind: TokenKind, at: int) -> void {
        self._push(kind, at, at, "")
    }

    fn _resume_fstring(self, flavor: _StrFlavor, token_start: int) -> void {
        let out = StringBuilder()
        let (stop, saw_escape) = self._scan_string_chunk(flavor, out, token_start)
        # A broken chunk still ends the f-string, so the parser sees a balanced
        # START ... END sequence and can keep going.
        let kind = stop == _ChunkStop.Interp ? TokenKind.FStringMid : TokenKind.FStringEnd
        self._push(kind, token_start, self.pos, out.build(), _string_flags(flavor, saw_escape))
        if stop == _ChunkStop.Interp { self._open_interpolation(flavor) }
    }

    # The cursor sits on the opening quote; `token_start` is the start of the
    # token, which for a prefixed literal is the `r` or `f`.
    fn _scan_string_literal(self, token_start: int, raw: bool, interp: bool) -> void {
        let quote = self.chars[self.pos]
        let triple = self._peek_at(1) == quote and self._peek_at(2) == quote
        let flavor = _StrFlavor(quote, triple, raw, interp)
        self.pos += triple ? 3 : 1

        let out = StringBuilder()
        let (stop, saw_escape) = self._scan_string_chunk(flavor, out, token_start)
        let flags = _string_flags(flavor, saw_escape)
        if not interp {
            self._push(TokenKind.Str, token_start, self.pos, out.build(), flags)
            return
        }

        self._push(TokenKind.FStringStart, token_start, self.pos, out.build(), flags)
        if stop == _ChunkStop.Interp {
            self._open_interpolation(flavor)
        } else {
            self._push_empty_string(TokenKind.FStringEnd, self.pos)
        }
    }

    # A `:` at an interpolation's top level starts a format spec (§1.7): it's
    # data, not Jaithon source (`.2f` would not lex as a number), so it's
    # captured whole as one string token for `std.fmt`.
    fn _scan_format_spec(self) -> void {
        let start = self.pos
        var depth = 0
        while self.pos < self.length {
            let character = self.chars[self.pos]
            if character == "\n" { break }  # unterminated; reported at the `}`
            if character == "{" {
                depth += 1
            } elif character == "}" {
                if depth == 0 { break }
                depth -= 1
            }
            self.pos += 1
        }
        self._push(TokenKind.Str, start, self.pos, self.source[start:self.pos])
    }

    # --- identifiers and labels ------------------------------------------
    fn _ident_run_end(self, start: int) -> int {
        var index = start
        while index < self.length {
            let character = self.chars[index]
            let code = ord(character)
            if code < 0x80 {
                if not _is_ident_cont(character) { break }
            } elif not _is_unicode_ident(code) {
                break
            }
            index += 1
        }
        return index
    }

    fn _scan_identifier(self) -> void {
        let start = self.pos
        self.pos = self._ident_run_end(start)
        let text = self.source[start:self.pos]

        # String prefixes `r`, `f` and their combination, only when the quote is
        # adjacent, so `r "x"` stays an identifier followed by a string.
        if text.len() <= 2 and (self._peek() == "\"" or self._peek() == "'") {
            var raw = false
            var interp = false
            var is_prefix = true
            for character in text {
                if character == "r" and not raw {
                    raw = true
                } elif character == "f" and not interp {
                    interp = true
                } else {
                    is_prefix = false
                    break
                }
            }
            if is_prefix and (raw or interp) {
                self._scan_string_literal(start, raw, interp)
                return
            }
        }

        # `__foo__` is reserved for the implementation (§1.4), but rejecting it
        # is the parser's call: the standard library declares such names.
        var kind = keyword_lookup(text)
        if kind == TokenKind.Ident and text == "_" { kind = TokenKind.Underscore }
        self._push(kind, start, self.pos)
    }

    fn _at_statement_start(self) -> bool {
        return match self._last_kind() {
            TokenKind.Eof | TokenKind.Newline | TokenKind.Semicolon => true,
            TokenKind.LBrace | TokenKind.RBrace => true,
            _ => false,
        }
    }

    # `'name` is a label, `'text'` a string; the grammar decides: a label is
    # only declared as `'name:` at a statement start, or used after
    # `break`/`continue` — everything else starting with `'` is a string, so
    # `'single quotes'` isn't misread as a label.
    fn _scan_quote_or_label(self) -> void {
        let start = self.pos
        let after = start + 1
        let first = self._at(after)
        var ident_start = false
        if first.len() > 0 {
            let code = ord(first)
            ident_start = code < 0x80 ? _is_ident_start(first) : _is_unicode_ident(code)
        }

        if ident_start {
            let end = self._ident_run_end(after)
            let closed_by_quote = self._at(end) == "'"
            var probe = end
            while self._at(probe) == " " or self._at(probe) == "\t" { probe += 1 }
            let declaration = self._at(probe) == ":" and self._at_statement_start()
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

    # Pop only on a match — a stray closer is the parser's problem (E0107);
    # popping blindly would corrupt the newline-suppression state of
    # enclosing brackets.
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
            # Closes an interpolation (no token emitted); `f"{}"` must be caught
            # here since it's otherwise indistinguishable from no interpolation.
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
        # §1.3: a NEWLINE immediately before `}` is discarded.
        if self.tokens.len() > 0 and self._last_kind() == TokenKind.Newline {
            self.tokens.pop()
        }
        self._pop_brace_bracket()
        self._push(TokenKind.RBrace, start, self.pos)
    }

    fn _scan_punct(self) -> void {
        let start = self.pos
        let character = self.chars[self.pos]
        self.pos += 1

        if character == "}" {
            self._close_brace(start)
            return
        }
        if character == ":" {
            self._push(TokenKind.Colon, start, self.pos)
            if self._bracket_top_is_interp() { self._scan_format_spec() }
            return
        }
        if character == "!" {
            if self._eat("=") {
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
            let code = ord(character)
            var message = f"unexpected character `{character}`"
            if code < 0x20 or code == 0x7F {
                message = f"unexpected control character U+{code:04X}"
            }
            self._error(Code.E0003_INVALID_CHARACTER, start, self.pos, message)
            return
        }
        self._push(kind, start, self.pos)
    }

    # Everything whose kind is decided by the characters alone. `}`, `:` and `!`
    # are handled by the caller because they do more than emit a token.
    fn _punct_kind(self, character: str) -> TokenKind? {
        if character == "(" {
            self.brackets.push(_BRK_PAREN)
            return TokenKind.LParen
        }
        if character == "[" {
            self.brackets.push(_BRK_BRACKET)
            return TokenKind.LBracket
        }
        if character == "{" {
            self.brackets.push(self._brace_opens_expression() ? _BRK_BRACE_EXPR : _BRK_BRACE_BLOCK)
            return TokenKind.LBrace
        }
        if character == ")" {
            self._pop_bracket(_BRK_PAREN)
            return TokenKind.RParen
        }
        if character == "]" {
            self._pop_bracket(_BRK_BRACKET)
            return TokenKind.RBracket
        }
        if character == "," { return TokenKind.Comma }
        if character == ";" { return TokenKind.Semicolon }
        if character == "@" { return TokenKind.At }
        if character == "~" { return TokenKind.Tilde }
        if character == "." {
            if not self._eat(".") { return TokenKind.Dot }
            if self._eat(".") { return TokenKind.Ellipsis }
            if self._eat("=") { return TokenKind.DotDotEq }
            return TokenKind.DotDot
        }
        if character == "+" {
            if self._eat("=") { return TokenKind.PlusEq }
            if self._eat("%") { return TokenKind.PlusPercent }
            return TokenKind.Plus
        }
        if character == "-" {
            if self._eat(">") { return TokenKind.Arrow }
            if self._eat("=") { return TokenKind.MinusEq }
            if self._eat("%") { return TokenKind.MinusPercent }
            return TokenKind.Minus
        }
        if character == "*" {
            if self._eat("*") { return self._eat("=") ? TokenKind.StarStarEq : TokenKind.StarStar }
            if self._eat("=") { return TokenKind.StarEq }
            if self._eat("%") { return TokenKind.StarPercent }
            return TokenKind.Star
        }
        if character == "/" {
            if self._eat("/") {
                return self._eat("=") ? TokenKind.SlashSlashEq : TokenKind.SlashSlash
            }
            if self._eat("=") { return TokenKind.SlashEq }
            return TokenKind.Slash
        }
        if character == "%" { return self._eat("=") ? TokenKind.PercentEq : TokenKind.Percent }
        if character == "&" { return self._eat("=") ? TokenKind.AmpEq : TokenKind.Amp }
        if character == "|" { return self._eat("=") ? TokenKind.PipeEq : TokenKind.Pipe }
        if character == "^" { return self._eat("=") ? TokenKind.CaretEq : TokenKind.Caret }
        if character == "<" {
            if self._eat("<") { return self._eat("=") ? TokenKind.ShlEq : TokenKind.Shl }
            return self._eat("=") ? TokenKind.Le : TokenKind.Lt
        }
        if character == ">" {
            if self._eat(">") { return self._eat("=") ? TokenKind.ShrEq : TokenKind.Shr }
            return self._eat("=") ? TokenKind.Ge : TokenKind.Gt
        }
        if character == "=" {
            if self._eat("=") { return TokenKind.EqEq }
            if self._eat(">") { return TokenKind.FatArrow }
            return TokenKind.Eq
        }
        if character == "?" {
            if self._eat("?") { return TokenKind.QuestionQuestion }
            if self._eat(".") { return TokenKind.QuestionDot }
            return TokenKind.Question
        }
        return null
    }

    fn _line_continuation(self) -> void {
        let start = self.pos
        var next = start + 1
        if self._at(next) == "\r" { next += 1 }
        if self._at(next) == "\n" {
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

    # --- the main loop ---------------------------------------------------
    fn _scan_token(self) -> void {
        let character = self.chars[self.pos]

        if character == " " or
            character == "\t" or
            character == "\r" or
            character == "\u{B}" or
            character == "\u{C}" {
            self.pos += 1  # §1.1: a lone \r is discarded, the \n terminates
            return
        }
        if character == "\n" {
            self._handle_newline()
            return
        }
        if character == "#" {
            self._skip_comment()
            return
        }
        if character == "\\" {
            self._line_continuation()
            return
        }

        let code = ord(character)
        if code >= 0x80 {
            if _is_unicode_space(code) {
                self.pos += 1
                return
            }
            if not _is_unicode_ident(code) {
                self._error(
                    Code.E0003_INVALID_CHARACTER,
                    self.pos,
                    self.pos + 1,
                    f"unexpected character U+{code:04X}",
                    help: "only ASCII punctuation is meaningful outside " + "strings and comments"
                )
                self.pos += 1
                return
            }
            self._scan_identifier()
            return
        }

        if _is_digit(character) {
            self._scan_number()
            return
        }
        if _is_ident_start(character) {
            self._scan_identifier()
            return
        }
        if character == "\"" {
            self._scan_string_literal(self.pos, false, false)
            return
        }
        if character == "'" {
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

fn _is_digit(character: str) -> bool {
    return character >= "0" and character <= "9"
}

fn _is_alpha(character: str) -> bool {
    return character >= "a" and character <= "z" or character >= "A" and character <= "Z"
}

fn _is_ident_start(character: str) -> bool {
    return _is_alpha(character) or character == "_"
}

fn _is_ident_cont(character: str) -> bool {
    return _is_ident_start(character) or _is_digit(character)
}

fn _hex_value(character: str) -> int {
    if _is_digit(character) { return ord(character) - ord("0") }
    if character >= "a" and character <= "f" { return ord(character) - ord("a") + 10 }
    if character >= "A" and character <= "F" { return ord(character) - ord("A") + 10 }
    return -1
}

fn _digit_in_base(character: str, base: int) -> bool {
    let value = _hex_value(character)
    return value >= 0 and value < base
}

fn _base_of(prefix: str) -> int {
    if prefix == "x" or prefix == "X" { return 16 }
    if prefix == "o" or prefix == "O" { return 8 }
    if prefix == "b" or prefix == "B" { return 2 }
    return 0
}

# Assumes `digits`' leading zeros are already stripped: length alone decides
# whether it fits the 64-bit `int`.
fn _fits_in_int(digits: str, base: int, negated: bool) -> bool {
    let count = digits.len()
    if base == 16 { return count <= 16 }
    if base == 8 { return count < 22 or count == 22 and digits[0] <= "1" }
    if base == 2 { return count <= 64 }
    if count < 19 { return true }
    if count > 19 { return false }
    # Same length, so a lexicographic comparison is a numeric one.
    if digits <= "9223372036854775807" { return true }
    return negated and digits == "9223372036854775808"
}

fn _utf8_length(code: int) -> int {
    if code < 0x80 { return 1 }
    if code < 0x800 { return 2 }
    if code < 0x10000 { return 3 }
    return 4
}

# Non-ASCII scalars that behave as separators rather than letters. Source is not
# allowed to smuggle these into identifiers, where they would be invisible.
fn _is_unicode_space(code: int) -> bool {
    if code == 0x00A0 or code == 0x1680 or code == 0x3000 or code == 0xFEFF { return true }
    if code >= 0x2000 and code <= 0x200D { return true }
    return code == 0x2028 or code == 0x2029 or code == 0x202F or code == 0x205F
}

# Approximates XID_Start/XID_Continue (§1.4) rather than using real XID
# tables (~10KB, unneeded here): accepts any scalar >= 0x80 except space-like
# and General Punctuation ranges, still rejecting pasted-in smart quotes.
fn _is_unicode_ident(code: int) -> bool {
    if code < 0x80 { return false }
    if _is_unicode_space(code) { return false }
    if code >= 0x2000 and code <= 0x206F { return false }
    if code >= 0xFFF9 and code <= 0xFFFB { return false }
    return true
}

# Every operator token, `Plus` through `ShrEq` in token.h order.
fn _is_operator(kind: TokenKind) -> bool {
    return match kind {
        TokenKind.Plus | TokenKind.Minus | TokenKind.Star | TokenKind.Slash => true,
        TokenKind.SlashSlash | TokenKind.Percent | TokenKind.StarStar => true,
        TokenKind.PlusPercent | TokenKind.MinusPercent | TokenKind.StarPercent => true,
        TokenKind.Amp | TokenKind.Pipe | TokenKind.Caret | TokenKind.Tilde => true,
        TokenKind.Shl | TokenKind.Shr | TokenKind.Eq | TokenKind.EqEq => true,
        TokenKind.BangEq | TokenKind.Lt | TokenKind.Le | TokenKind.Gt | TokenKind.Ge => true,
        TokenKind.PlusEq | TokenKind.MinusEq | TokenKind.StarEq | TokenKind.SlashEq => true,
        TokenKind.SlashSlashEq | TokenKind.PercentEq | TokenKind.StarStarEq => true,
        TokenKind.AmpEq | TokenKind.PipeEq | TokenKind.CaretEq => true,
        TokenKind.ShlEq | TokenKind.ShrEq => true,
        _ => false,
    }
}

# Tokens after which a newline can't end a statement: §1.3's list plus `..`,
# `..=`, and `{` (harmless for blocks, since the grammar already allows
# `"{" { NEWLINE }`, and it keeps multi-line dict/set literals working). `?`
# is deliberately absent — `let x: int?` ends a statement with it.
fn _is_continuation(kind: TokenKind) -> bool {
    if _is_operator(kind) { return true }
    return match kind {
        TokenKind.Comma | TokenKind.Dot | TokenKind.DotDot | TokenKind.DotDotEq => true,
        TokenKind.Colon | TokenKind.Arrow | TokenKind.FatArrow => true,
        TokenKind.LParen | TokenKind.LBracket | TokenKind.LBrace => true,
        TokenKind.QuestionDot | TokenKind.QuestionQuestion => true,
        TokenKind.KwAnd | TokenKind.KwOr | TokenKind.KwNot => true,
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
    let quote = marker & _BRK_F_SINGLE != 0 ? "'" : "\""
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

    # Tokens are in source order, so one cursor walk computes every position;
    # columns count scalars, not bytes (spec 1.1).
    let chars = [character for character in source]
    var offsets: list[int] = []
    var byte_offset = 0
    for character in chars {
        offsets.push(byte_offset)
        byte_offset += _dump_utf8_length(ord(character))
    }
    offsets.push(byte_offset)

    let out = StringBuilder()
    var index = 0
    var at = 0
    var line = 1
    var column = 1
    for token in tokens {
        while at < chars.len() and offsets[at] < token.span.start {
            if chars[at] == "\n" {
                line += 1
                column = 1
            } else {
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

fn _dump_utf8_length(code_point: int) -> int {
    if code_point < 0x80 { return 1 }
    if code_point < 0x800 { return 2 }
    if code_point < 0x10000 { return 3 }
    return 4
}

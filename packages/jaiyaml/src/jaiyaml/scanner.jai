#: Text to tokens: PyYAML's scanner, method for method.
#:
#: The scanner is a queue of tokens fed on demand. What makes it more than a
#: tokenizer is simple keys: `a: b` has no `?` indicator, so when a scalar
#: starts the scanner remembers that it might be a key, and when a `:`
#: turns up on the same line it inserts a KEY token back before the scalar.
#: The indentation stack turns column changes into BLOCK-END tokens, and
#: `flow_level` switches the rules inside `[` and `{`.
#:
#: ```
#: let scanner = Scanner("a: 1")
#: while scanner.check_token() {
#:     let token = scanner.get_token()
#: }
#: ```
#:
#: Every check the Python source writes as `ch in '...'` is a substring test
#: on the same set here, with `\0` standing for the end of the text, and the
#: error messages are PyYAML's, character reprs included.
from std.str import StringBuilder, to_int

from .error import Mark, ScannerError, python_repr
from .reader import Reader, STRING_NAME
from .tokens import Token, TokenKind

let NUL = "\u{0}"
let BREAKS = "\r\n\u{85}\u{2028}\u{2029}"
let NUL_OR_BREAK = "\u{0}\r\n\u{85}\u{2028}\u{2029}"
let NUL_SPACE_OR_BREAK = "\u{0} \t\r\n\u{85}\u{2028}\u{2029}"
let NUL_BLANK_OR_BREAK = "\u{0} \r\n\u{85}\u{2028}\u{2029}"
let SPACE_OR_BREAK = " \r\n\u{85}\u{2028}\u{2029}"
let HEX_DIGITS = "0123456789ABCDEFabcdef"
let DIGITS = "0123456789"
let URI_CHARS = "-;/?:@&=+$,_.!~*'()[]%"
let PLAIN_STOPPERS = "\u{0} \t\r\n\u{85}\u{2028}\u{2029}-?:,[]{}#&*!|>'\"%@`"
let ANCHOR_STOPPERS = "\u{0} \t\r\n\u{85}\u{2028}\u{2029}?:,]}%@`"

let ESCAPE_REPLACEMENTS: dict[str, str] = {
    "0": "\u{0}",
    "a": "\u{7}",
    "b": "\u{8}",
    "t": "\u{9}",
    "\t": "\u{9}",
    "n": "\u{A}",
    "v": "\u{B}",
    "f": "\u{C}",
    "r": "\u{D}",
    "e": "\u{1B}",
    " ": "\u{20}",
    "\"": "\"",
    "\\": "\\",
    "/": "/",
    "N": "\u{85}",
    "_": "\u{A0}",
    "L": "\u{2028}",
    "P": "\u{2029}",
}

let ESCAPE_CODES: dict[str, int] = {"x": 2, "u": 4, "U": 8}

fn is_word_char(ch: str) -> bool {
    return ch >= "0" and ch <= "9" or
        ch >= "A" and ch <= "Z" or
        ch >= "a" and ch <= "z" or
        ch == "-" or
        ch == "_"
}

fn is_digit(ch: str) -> bool { return ch >= "0" and ch <= "9" }

#: A scalar that might turn out to be a mapping key.
class SimpleKey {
    pub let token_number: int
    pub let required: bool
    pub let index: int
    pub let line: int
    pub let column: int
    pub let mark: Mark

    pub fn init(
        self,
        token_number: int,
        required: bool,
        index: int,
        line: int,
        column: int,
        mark: Mark
    ) {
        self.token_number = token_number
        self.required = required
        self.index = index
        self.line = line
        self.column = column
        self.mark = mark
    }
}

#: Produces the token stream for one text.
pub class Scanner {
    #: The cursor over the text.
    pub let reader: Reader

    var done: bool
    var flow_level: int
    var tokens: list[Token]
    var tokens_taken: int
    var indent: int
    var indents: list[int]
    var allow_simple_key: bool
    var possible_simple_keys: dict[int, SimpleKey]

    #: Start scanning `text`; the STREAM-START token is queued at once.
    pub fn init(self, text: str, name: str = STRING_NAME) {
        self.reader = Reader(text, name)
        self.done = false
        self.flow_level = 0
        self.tokens = []
        self.tokens_taken = 0
        self.indent = -1
        self.indents = []
        self.allow_simple_key = true
        self.possible_simple_keys = {}
        self.fetch_stream_start()
    }

    #: Whether the next token is one of `kinds`, or exists at all when none
    #: are named.
    pub fn check_token(self, ...kinds: TokenKind) -> bool {
        while self.need_more_tokens() { self.fetch_more_tokens() }
        if self.tokens.len() == 0 { return false }
        if kinds.len() == 0 { return true }
        let head = self.tokens[0].kind
        for kind in kinds {
            if head == kind { return true }
        }
        return false
    }

    #: The next token without consuming it, or null at the end.
    pub fn peek_token(self) -> Token? {
        while self.need_more_tokens() { self.fetch_more_tokens() }
        if self.tokens.len() == 0 { return null }
        return self.tokens[0]
    }

    #: Consume and return the next token.
    pub fn get_token(self) -> Token {
        while self.need_more_tokens() { self.fetch_more_tokens() }
        if self.tokens.len() == 0 {
            throw ScannerError(null, null, "no more tokens", self.reader.get_mark())
        }
        self.tokens_taken += 1
        return self.tokens.pop(0)
    }

    fn need_more_tokens(self) -> bool {
        if self.done { return false }
        if self.tokens.len() == 0 { return true }
        self.stale_possible_simple_keys()
        let next = self.next_possible_simple_key()
        return next is not null and (next ?? -1) == self.tokens_taken
    }

    fn fetch_more_tokens(self) -> void {
        self.scan_to_next_token()
        self.stale_possible_simple_keys()
        self.unwind_indent(self.reader.column)
        let ch = self.reader.peek()
        if ch == NUL { return self.fetch_stream_end() }
        if ch == "%" and self.check_directive() { return self.fetch_directive() }
        if ch == "-" and self.check_document_start() { return self.fetch_document_start() }
        if ch == "." and self.check_document_end() { return self.fetch_document_end() }
        if ch == "[" { return self.fetch_flow_collection_start(TokenKind.FlowSequenceStart) }
        if ch == "{" { return self.fetch_flow_collection_start(TokenKind.FlowMappingStart) }
        if ch == "]" { return self.fetch_flow_collection_end(TokenKind.FlowSequenceEnd) }
        if ch == "}" { return self.fetch_flow_collection_end(TokenKind.FlowMappingEnd) }
        if ch == "," { return self.fetch_flow_entry() }
        if ch == "-" and self.check_block_entry() { return self.fetch_block_entry() }
        if ch == "?" and self.check_key() { return self.fetch_key() }
        if ch == ":" and self.check_value() { return self.fetch_value() }
        if ch == "*" { return self.fetch_alias() }
        if ch == "&" { return self.fetch_anchor() }
        if ch == "!" { return self.fetch_tag() }
        if ch == "|" and self.flow_level == 0 { return self.fetch_block_scalar("|") }
        if ch == ">" and self.flow_level == 0 { return self.fetch_block_scalar(">") }
        if ch == "'" { return self.fetch_flow_scalar("'") }
        if ch == "\"" { return self.fetch_flow_scalar("\"") }
        if self.check_plain() { return self.fetch_plain() }
        throw ScannerError(
            "while scanning for the next token",
            null,
            f"found character {python_repr(ch)} that cannot start any token",
            self.reader.get_mark()
        )
    }

    fn next_possible_simple_key(self) -> int? {
        var least: int? = null
        for level in self.possible_simple_keys.keys() {
            let key = self.possible_simple_keys[level]
            if least is null or key.token_number < (least ?? 0) { least = key.token_number }
        }
        return least
    }

    fn stale_possible_simple_keys(self) -> void {
        for level in self.possible_simple_keys.keys() {
            let key = self.possible_simple_keys[level]
            if key.line != self.reader.line or self.reader.index - key.index > 1024 {
                if key.required {
                    throw ScannerError(
                        "while scanning a simple key",
                        key.mark,
                        "could not find expected ':'",
                        self.reader.get_mark()
                    )
                }
                self.possible_simple_keys.remove(level)
            }
        }
    }

    fn save_possible_simple_key(self) -> void {
        let required = self.flow_level == 0 and self.indent == self.reader.column
        if self.allow_simple_key {
            self.remove_possible_simple_key()
            let token_number = self.tokens_taken + self.tokens.len()
            self.possible_simple_keys[self.flow_level] = SimpleKey(
                token_number,
                required,
                self.reader.index,
                self.reader.line,
                self.reader.column,
                self.reader.get_mark()
            )
        }
    }

    fn remove_possible_simple_key(self) -> void {
        if self.flow_level in self.possible_simple_keys {
            let key = self.possible_simple_keys[self.flow_level]
            if key.required {
                throw ScannerError(
                    "while scanning a simple key",
                    key.mark,
                    "could not find expected ':'",
                    self.reader.get_mark()
                )
            }
            self.possible_simple_keys.remove(self.flow_level)
        }
    }

    fn unwind_indent(self, column: int) -> void {
        if self.flow_level > 0 { return }
        while self.indent > column {
            let mark = self.reader.get_mark()
            self.indent = self.indents.pop()
            self.tokens.push(Token(TokenKind.BlockEnd, mark, mark))
        }
    }

    fn add_indent(self, column: int) -> bool {
        if self.indent < column {
            self.indents.push(self.indent)
            self.indent = column
            return true
        }
        return false
    }

    fn fetch_stream_start(self) -> void {
        let mark = self.reader.get_mark()
        self.tokens.push(Token(TokenKind.StreamStart, mark, mark))
    }

    fn fetch_stream_end(self) -> void {
        self.unwind_indent(-1)
        self.remove_possible_simple_key()
        self.allow_simple_key = false
        self.possible_simple_keys = {}
        let mark = self.reader.get_mark()
        self.tokens.push(Token(TokenKind.StreamEnd, mark, mark))
        self.done = true
    }

    fn fetch_directive(self) -> void {
        self.unwind_indent(-1)
        self.remove_possible_simple_key()
        self.allow_simple_key = false
        self.tokens.push(self.scan_directive())
    }

    fn fetch_document_start(self) -> void { self.fetch_document_indicator(TokenKind.DocumentStart) }

    fn fetch_document_end(self) -> void { self.fetch_document_indicator(TokenKind.DocumentEnd) }

    fn fetch_document_indicator(self, kind: TokenKind) -> void {
        self.unwind_indent(-1)
        self.remove_possible_simple_key()
        self.allow_simple_key = false
        let start_mark = self.reader.get_mark()
        self.reader.forward(3)
        let end_mark = self.reader.get_mark()
        self.tokens.push(Token(kind, start_mark, end_mark))
    }

    fn fetch_flow_collection_start(self, kind: TokenKind) -> void {
        self.save_possible_simple_key()
        self.flow_level += 1
        self.allow_simple_key = true
        let start_mark = self.reader.get_mark()
        self.reader.forward()
        let end_mark = self.reader.get_mark()
        self.tokens.push(Token(kind, start_mark, end_mark))
    }

    fn fetch_flow_collection_end(self, kind: TokenKind) -> void {
        self.remove_possible_simple_key()
        self.flow_level -= 1
        self.allow_simple_key = false
        let start_mark = self.reader.get_mark()
        self.reader.forward()
        let end_mark = self.reader.get_mark()
        self.tokens.push(Token(kind, start_mark, end_mark))
    }

    fn fetch_flow_entry(self) -> void {
        self.allow_simple_key = true
        self.remove_possible_simple_key()
        let start_mark = self.reader.get_mark()
        self.reader.forward()
        let end_mark = self.reader.get_mark()
        self.tokens.push(Token(TokenKind.FlowEntry, start_mark, end_mark))
    }

    fn fetch_block_entry(self) -> void {
        if self.flow_level == 0 {
            if not self.allow_simple_key {
                throw ScannerError(
                    null,
                    null,
                    "sequence entries are not allowed here",
                    self.reader.get_mark()
                )
            }
            if self.add_indent(self.reader.column) {
                let mark = self.reader.get_mark()
                self.tokens.push(Token(TokenKind.BlockSequenceStart, mark, mark))
            }
        }
        self.allow_simple_key = true
        self.remove_possible_simple_key()
        let start_mark = self.reader.get_mark()
        self.reader.forward()
        let end_mark = self.reader.get_mark()
        self.tokens.push(Token(TokenKind.BlockEntry, start_mark, end_mark))
    }

    fn fetch_key(self) -> void {
        if self.flow_level == 0 {
            if not self.allow_simple_key {
                throw ScannerError(
                    null,
                    null,
                    "mapping keys are not allowed here",
                    self.reader.get_mark()
                )
            }
            if self.add_indent(self.reader.column) {
                let mark = self.reader.get_mark()
                self.tokens.push(Token(TokenKind.BlockMappingStart, mark, mark))
            }
        }
        self.allow_simple_key = self.flow_level == 0
        self.remove_possible_simple_key()
        let start_mark = self.reader.get_mark()
        self.reader.forward()
        let end_mark = self.reader.get_mark()
        self.tokens.push(Token(TokenKind.Key, start_mark, end_mark))
    }

    fn fetch_value(self) -> void {
        if self.flow_level in self.possible_simple_keys {
            let key = self.possible_simple_keys[self.flow_level]
            self.possible_simple_keys.remove(self.flow_level)
            self.tokens.insert(
                key.token_number - self.tokens_taken,
                Token(TokenKind.Key, key.mark, key.mark)
            )
            if self.flow_level == 0 {
                if self.add_indent(key.column) {
                    self.tokens.insert(
                        key.token_number - self.tokens_taken,
                        Token(TokenKind.BlockMappingStart, key.mark, key.mark)
                    )
                }
            }
            self.allow_simple_key = false
        } else {
            if self.flow_level == 0 {
                if not self.allow_simple_key {
                    throw ScannerError(
                        null,
                        null,
                        "mapping values are not allowed here",
                        self.reader.get_mark()
                    )
                }
            }
            if self.flow_level == 0 {
                if self.add_indent(self.reader.column) {
                    let mark = self.reader.get_mark()
                    self.tokens.push(Token(TokenKind.BlockMappingStart, mark, mark))
                }
            }
            self.allow_simple_key = self.flow_level == 0
            self.remove_possible_simple_key()
        }
        let start_mark = self.reader.get_mark()
        self.reader.forward()
        let end_mark = self.reader.get_mark()
        self.tokens.push(Token(TokenKind.Value, start_mark, end_mark))
    }

    fn fetch_alias(self) -> void {
        self.save_possible_simple_key()
        self.allow_simple_key = false
        self.tokens.push(self.scan_anchor(TokenKind.Alias))
    }

    fn fetch_anchor(self) -> void {
        self.save_possible_simple_key()
        self.allow_simple_key = false
        self.tokens.push(self.scan_anchor(TokenKind.Anchor))
    }

    fn fetch_tag(self) -> void {
        self.save_possible_simple_key()
        self.allow_simple_key = false
        self.tokens.push(self.scan_tag())
    }

    fn fetch_block_scalar(self, style: str) -> void {
        self.allow_simple_key = true
        self.remove_possible_simple_key()
        self.tokens.push(self.scan_block_scalar(style))
    }

    fn fetch_flow_scalar(self, style: str) -> void {
        self.save_possible_simple_key()
        self.allow_simple_key = false
        self.tokens.push(self.scan_flow_scalar(style))
    }

    fn fetch_plain(self) -> void {
        self.save_possible_simple_key()
        self.allow_simple_key = false
        self.tokens.push(self.scan_plain())
    }

    fn check_directive(self) -> bool { return self.reader.column == 0 }

    fn check_document_start(self) -> bool {
        return self.reader.column == 0 and
            self.reader.prefix(3) == "---" and
            self.reader.peek(3) in NUL_SPACE_OR_BREAK
    }

    fn check_document_end(self) -> bool {
        return self.reader.column == 0 and
            self.reader.prefix(3) == "..." and
            self.reader.peek(3) in NUL_SPACE_OR_BREAK
    }

    fn check_block_entry(self) -> bool { return self.reader.peek(1) in NUL_SPACE_OR_BREAK }

    fn check_key(self) -> bool {
        if self.flow_level > 0 { return true }
        return self.reader.peek(1) in NUL_SPACE_OR_BREAK
    }

    fn check_value(self) -> bool {
        if self.flow_level > 0 { return true }
        return self.reader.peek(1) in NUL_SPACE_OR_BREAK
    }

    fn check_plain(self) -> bool {
        let ch = self.reader.peek()
        if ch not in PLAIN_STOPPERS { return true }
        return self.reader.peek(1) not in NUL_SPACE_OR_BREAK and
            (ch == "-" or self.flow_level == 0 and (ch == "?" or ch == ":"))
    }

    fn scan_to_next_token(self) -> void {
        if self.reader.index == 0 and self.reader.peek() == "\u{FEFF}" { self.reader.forward() }
        var found = false
        while not found {
            while self.reader.peek() == " " { self.reader.forward() }
            if self.reader.peek() == "#" {
                while self.reader.peek() not in NUL_OR_BREAK { self.reader.forward() }
            }
            if self.scan_line_break().len() > 0 {
                if self.flow_level == 0 { self.allow_simple_key = true }
            } else {
                found = true
            }
        }
    }

    fn scan_directive(self) -> Token {
        let start_mark = self.reader.get_mark()
        self.reader.forward()
        let name = self.scan_directive_name(start_mark)
        var end_mark = self.reader.get_mark()
        var major = 0
        var minor = 0
        var handle: str? = null
        var prefix: str? = null
        if name == "YAML" {
            let (found_major, found_minor) = self.scan_yaml_directive_value(start_mark)
            major = found_major
            minor = found_minor
            end_mark = self.reader.get_mark()
        } elif name == "TAG" {
            let (found_handle, found_prefix) = self.scan_tag_directive_value(start_mark)
            handle = found_handle
            prefix = found_prefix
            end_mark = self.reader.get_mark()
        } else {
            end_mark = self.reader.get_mark()
            while self.reader.peek() not in NUL_OR_BREAK { self.reader.forward() }
        }
        self.scan_directive_ignored_line(start_mark)
        let token = Token(TokenKind.Directive, start_mark, end_mark)
        token.value = name
        token.major = major
        token.minor = minor
        token.handle = handle
        token.suffix = prefix
        return token
    }

    fn scan_directive_name(self, start_mark: Mark) -> str {
        var length = 0
        var ch = self.reader.peek(length)
        while is_word_char(ch) {
            length += 1
            ch = self.reader.peek(length)
        }
        if length == 0 {
            throw ScannerError(
                "while scanning a directive",
                start_mark,
                f"expected alphabetic or numeric character, but found {python_repr(ch)}",
                self.reader.get_mark()
            )
        }
        let value = self.reader.prefix(length)
        self.reader.forward(length)
        ch = self.reader.peek()
        if ch not in NUL_BLANK_OR_BREAK {
            throw ScannerError(
                "while scanning a directive",
                start_mark,
                f"expected alphabetic or numeric character, but found {python_repr(ch)}",
                self.reader.get_mark()
            )
        }
        return value
    }

    fn scan_yaml_directive_value(self, start_mark: Mark) -> tuple[int, int] {
        while self.reader.peek() == " " { self.reader.forward() }
        let major = self.scan_yaml_directive_number(start_mark)
        if self.reader.peek() != "." {
            throw ScannerError(
                "while scanning a directive",
                start_mark,
                f"expected a digit or '.', but found {python_repr(self.reader.peek())}",
                self.reader.get_mark()
            )
        }
        self.reader.forward()
        let minor = self.scan_yaml_directive_number(start_mark)
        if self.reader.peek() not in NUL_BLANK_OR_BREAK {
            throw ScannerError(
                "while scanning a directive",
                start_mark,
                f"expected a digit or ' ', but found {python_repr(self.reader.peek())}",
                self.reader.get_mark()
            )
        }
        return (major, minor)
    }

    fn scan_yaml_directive_number(self, start_mark: Mark) -> int {
        let ch = self.reader.peek()
        if not is_digit(ch) {
            throw ScannerError(
                "while scanning a directive",
                start_mark,
                f"expected a digit, but found {python_repr(ch)}",
                self.reader.get_mark()
            )
        }
        var length = 0
        while is_digit(self.reader.peek(length)) { length += 1 }
        let value = to_int(self.reader.prefix(length)) ?? 0
        self.reader.forward(length)
        return value
    }

    fn scan_tag_directive_value(self, start_mark: Mark) -> tuple[str, str] {
        while self.reader.peek() == " " { self.reader.forward() }
        let handle = self.scan_tag_directive_handle(start_mark)
        while self.reader.peek() == " " { self.reader.forward() }
        let prefix = self.scan_tag_directive_prefix(start_mark)
        return (handle, prefix)
    }

    fn scan_tag_directive_handle(self, start_mark: Mark) -> str {
        let value = self.scan_tag_handle("directive", start_mark)
        let ch = self.reader.peek()
        if ch != " " {
            throw ScannerError(
                "while scanning a directive",
                start_mark,
                f"expected ' ', but found {python_repr(ch)}",
                self.reader.get_mark()
            )
        }
        return value
    }

    fn scan_tag_directive_prefix(self, start_mark: Mark) -> str {
        let value = self.scan_tag_uri("directive", start_mark)
        let ch = self.reader.peek()
        if ch not in NUL_BLANK_OR_BREAK {
            throw ScannerError(
                "while scanning a directive",
                start_mark,
                f"expected ' ', but found {python_repr(ch)}",
                self.reader.get_mark()
            )
        }
        return value
    }

    fn scan_directive_ignored_line(self, start_mark: Mark) -> void {
        while self.reader.peek() == " " { self.reader.forward() }
        if self.reader.peek() == "#" {
            while self.reader.peek() not in NUL_OR_BREAK { self.reader.forward() }
        }
        let ch = self.reader.peek()
        if ch not in NUL_OR_BREAK {
            throw ScannerError(
                "while scanning a directive",
                start_mark,
                f"expected a comment or a line break, but found {python_repr(ch)}",
                self.reader.get_mark()
            )
        }
        self.scan_line_break()
    }

    fn scan_anchor(self, kind: TokenKind) -> Token {
        let start_mark = self.reader.get_mark()
        let indicator = self.reader.peek()
        let name = indicator == "*" ? "alias" : "anchor"
        self.reader.forward()
        var length = 0
        var ch = self.reader.peek(length)
        while is_word_char(ch) {
            length += 1
            ch = self.reader.peek(length)
        }
        if length == 0 {
            throw ScannerError(
                f"while scanning an {name}",
                start_mark,
                f"expected alphabetic or numeric character, but found {python_repr(ch)}",
                self.reader.get_mark()
            )
        }
        let value = self.reader.prefix(length)
        self.reader.forward(length)
        ch = self.reader.peek()
        if ch not in ANCHOR_STOPPERS {
            throw ScannerError(
                f"while scanning an {name}",
                start_mark,
                f"expected alphabetic or numeric character, but found {python_repr(ch)}",
                self.reader.get_mark()
            )
        }
        let end_mark = self.reader.get_mark()
        let token = Token(kind, start_mark, end_mark)
        token.value = value
        return token
    }

    fn scan_tag(self) -> Token {
        let start_mark = self.reader.get_mark()
        var ch = self.reader.peek(1)
        var handle: str? = null
        var suffix = ""
        if ch == "<" {
            handle = null
            self.reader.forward(2)
            suffix = self.scan_tag_uri("tag", start_mark)
            if self.reader.peek() != ">" {
                throw ScannerError(
                    "while parsing a tag",
                    start_mark,
                    f"expected '>', but found {python_repr(self.reader.peek())}",
                    self.reader.get_mark()
                )
            }
            self.reader.forward()
        } elif ch in NUL_SPACE_OR_BREAK {
            handle = null
            suffix = "!"
            self.reader.forward()
        } else {
            var length = 1
            var use_handle = false
            while ch not in NUL_BLANK_OR_BREAK {
                if ch == "!" {
                    use_handle = true
                    break
                }
                length += 1
                ch = self.reader.peek(length)
            }
            if use_handle {
                handle = self.scan_tag_handle("tag", start_mark)
            } else {
                handle = "!"
                self.reader.forward()
            }
            suffix = self.scan_tag_uri("tag", start_mark)
        }
        ch = self.reader.peek()
        if ch not in NUL_BLANK_OR_BREAK {
            throw ScannerError(
                "while scanning a tag",
                start_mark,
                f"expected ' ', but found {python_repr(ch)}",
                self.reader.get_mark()
            )
        }
        let end_mark = self.reader.get_mark()
        let token = Token(TokenKind.Tag, start_mark, end_mark)
        token.handle = handle
        token.suffix = suffix
        return token
    }

    fn scan_block_scalar(self, style: str) -> Token {
        let folded = style == ">"
        let chunks = StringBuilder()
        let start_mark = self.reader.get_mark()
        self.reader.forward()
        let (chomping, increment) = self.scan_block_scalar_indicators(start_mark)
        self.scan_block_scalar_ignored_line(start_mark)
        var min_indent = self.indent + 1
        if min_indent < 1 { min_indent = 1 }
        var indent = 0
        var breaks: list[str] = []
        var end_mark = self.reader.get_mark()
        if increment is null {
            let (found_breaks, max_indent, found_end) = self.scan_block_scalar_indentation()
            breaks = found_breaks
            end_mark = found_end
            indent = max_indent > min_indent ? max_indent : min_indent
        } else {
            indent = min_indent + (increment ?? 1) - 1
            let (found_breaks, found_end) = self.scan_block_scalar_breaks(indent)
            breaks = found_breaks
            end_mark = found_end
        }
        var line_break = ""
        while self.reader.column == indent and self.reader.peek() != NUL {
            chunks.extend(breaks)
            let leading_non_space = self.reader.peek() != " " and self.reader.peek() != "\t"
            var length = 0
            while self.reader.peek(length) not in NUL_OR_BREAK { length += 1 }
            chunks.push(self.reader.prefix(length))
            self.reader.forward(length)
            line_break = self.scan_line_break()
            let (found_breaks, found_end) = self.scan_block_scalar_breaks(indent)
            breaks = found_breaks
            end_mark = found_end
            if self.reader.column == indent and self.reader.peek() != NUL {
                if folded and
                    line_break == "\n" and
                    leading_non_space and
                    self.reader.peek() != " " and
                    self.reader.peek() != "\t" {
                    if breaks.len() == 0 { chunks.push(" ") }
                } else {
                    chunks.push(line_break)
                }
            } else {
                break
            }
        }
        if chomping is null or (chomping ?? false) { chunks.push(line_break) }
        if chomping is not null and (chomping ?? false) { chunks.extend(breaks) }
        let token = Token(TokenKind.Scalar, start_mark, end_mark)
        token.value = chunks.build()
        token.plain = false
        token.style = style
        return token
    }

    fn scan_block_scalar_indicators(self, start_mark: Mark) -> tuple[bool?, int?] {
        var chomping: bool? = null
        var increment: int? = null
        var ch = self.reader.peek()
        if ch == "+" or ch == "-" {
            chomping = ch == "+"
            self.reader.forward()
            ch = self.reader.peek()
            if ch in DIGITS {
                increment = to_int(ch) ?? 0
                if (increment ?? 0) == 0 {
                    throw ScannerError(
                        "while scanning a block scalar",
                        start_mark,
                        "expected indentation indicator in the range 1-9, but found 0",
                        self.reader.get_mark()
                    )
                }
                self.reader.forward()
            }
        } elif ch in DIGITS {
            increment = to_int(ch) ?? 0
            if (increment ?? 0) == 0 {
                throw ScannerError(
                    "while scanning a block scalar",
                    start_mark,
                    "expected indentation indicator in the range 1-9, but found 0",
                    self.reader.get_mark()
                )
            }
            self.reader.forward()
            ch = self.reader.peek()
            if ch == "+" or ch == "-" {
                chomping = ch == "+"
                self.reader.forward()
            }
        }
        ch = self.reader.peek()
        if ch not in NUL_BLANK_OR_BREAK {
            throw ScannerError(
                "while scanning a block scalar",
                start_mark,
                f"expected chomping or indentation indicators, but found {python_repr(ch)}",
                self.reader.get_mark()
            )
        }
        return (chomping, increment)
    }

    fn scan_block_scalar_ignored_line(self, start_mark: Mark) -> void {
        while self.reader.peek() == " " { self.reader.forward() }
        if self.reader.peek() == "#" {
            while self.reader.peek() not in NUL_OR_BREAK { self.reader.forward() }
        }
        let ch = self.reader.peek()
        if ch not in NUL_OR_BREAK {
            throw ScannerError(
                "while scanning a block scalar",
                start_mark,
                f"expected a comment or a line break, but found {python_repr(ch)}",
                self.reader.get_mark()
            )
        }
        self.scan_line_break()
    }

    fn scan_block_scalar_indentation(self) -> tuple[list[str], int, Mark] {
        let chunks: list[str] = []
        var max_indent = 0
        var end_mark = self.reader.get_mark()
        while self.reader.peek() in SPACE_OR_BREAK {
            if self.reader.peek() != " " {
                chunks.push(self.scan_line_break())
                end_mark = self.reader.get_mark()
            } else {
                self.reader.forward()
                if self.reader.column > max_indent { max_indent = self.reader.column }
            }
        }
        return (chunks, max_indent, end_mark)
    }

    fn scan_block_scalar_breaks(self, indent: int) -> tuple[list[str], Mark] {
        let chunks: list[str] = []
        var end_mark = self.reader.get_mark()
        while self.reader.column < indent and self.reader.peek() == " " { self.reader.forward() }
        while self.reader.peek() in BREAKS {
            chunks.push(self.scan_line_break())
            end_mark = self.reader.get_mark()
            while self.reader.column < indent and self.reader.peek() == " " {
                self.reader.forward()
            }
        }
        return (chunks, end_mark)
    }

    fn scan_flow_scalar(self, style: str) -> Token {
        let double = style == "\""
        let chunks = StringBuilder()
        let start_mark = self.reader.get_mark()
        let quote = self.reader.peek()
        self.reader.forward()
        chunks.extend(self.scan_flow_scalar_non_spaces(double, start_mark))
        while self.reader.peek() != quote {
            chunks.extend(self.scan_flow_scalar_spaces(double, start_mark))
            chunks.extend(self.scan_flow_scalar_non_spaces(double, start_mark))
        }
        self.reader.forward()
        let end_mark = self.reader.get_mark()
        let token = Token(TokenKind.Scalar, start_mark, end_mark)
        token.value = chunks.build()
        token.plain = false
        token.style = style
        return token
    }

    fn scan_flow_scalar_non_spaces(self, double: bool, start_mark: Mark) -> list[str] {
        let chunks: list[str] = []
        loop {
            var length = 0
            while self.reader.peek(length) not in "'\"\\\u{0} \t\r\n\u{85}\u{2028}\u{2029}" {
                length += 1
            }
            if length > 0 {
                chunks.push(self.reader.prefix(length))
                self.reader.forward(length)
            }
            var ch = self.reader.peek()
            if not double and ch == "'" and self.reader.peek(1) == "'" {
                chunks.push("'")
                self.reader.forward(2)
            } elif double and ch == "'" or not double and (ch == "\"" or ch == "\\") {
                chunks.push(ch)
                self.reader.forward()
            } elif double and ch == "\\" {
                self.reader.forward()
                ch = self.reader.peek()
                if ch in ESCAPE_REPLACEMENTS {
                    chunks.push(ESCAPE_REPLACEMENTS[ch])
                    self.reader.forward()
                } elif ch in ESCAPE_CODES {
                    let width = ESCAPE_CODES[ch]
                    self.reader.forward()
                    for k in 0..width {
                        if self.reader.peek(k) not in HEX_DIGITS {
                            throw ScannerError(
                                "while scanning a double-quoted scalar",
                                start_mark,
                                f"expected escape sequence of {width} hexadecimal numbers, but found {python_repr(self.reader.peek(k))}",
                                self.reader.get_mark()
                            )
                        }
                    }
                    let code = to_int(self.reader.prefix(width), 16) ?? 0
                    chunks.push(code_point(code, start_mark, self.reader.get_mark()))
                    self.reader.forward(width)
                } elif ch in BREAKS {
                    self.scan_line_break()
                    chunks.extend(self.scan_flow_scalar_breaks(double, start_mark))
                } else {
                    throw ScannerError(
                        "while scanning a double-quoted scalar",
                        start_mark,
                        f"found unknown escape character {python_repr(ch)}",
                        self.reader.get_mark()
                    )
                }
            } else {
                return chunks
            }
        }
    }

    fn scan_flow_scalar_spaces(self, double: bool, start_mark: Mark) -> list[str] {
        let chunks: list[str] = []
        var length = 0
        while self.reader.peek(length) == " " or self.reader.peek(length) == "\t" { length += 1 }
        let whitespaces = self.reader.prefix(length)
        self.reader.forward(length)
        let ch = self.reader.peek()
        if ch == NUL {
            throw ScannerError(
                "while scanning a quoted scalar",
                start_mark,
                "found unexpected end of stream",
                self.reader.get_mark()
            )
        } elif ch in BREAKS {
            let line_break = self.scan_line_break()
            let breaks = self.scan_flow_scalar_breaks(double, start_mark)
            if line_break != "\n" {
                chunks.push(line_break)
            } elif breaks.len() == 0 {
                chunks.push(" ")
            }
            chunks.extend(breaks)
        } else {
            chunks.push(whitespaces)
        }
        return chunks
    }

    fn scan_flow_scalar_breaks(self, double: bool, start_mark: Mark) -> list[str] {
        let chunks: list[str] = []
        loop {
            let prefix = self.reader.prefix(3)
            if (prefix == "---" or prefix == "...") and self.reader.peek(3) in NUL_SPACE_OR_BREAK {
                throw ScannerError(
                    "while scanning a quoted scalar",
                    start_mark,
                    "found unexpected document separator",
                    self.reader.get_mark()
                )
            }
            while self.reader.peek() == " " or self.reader.peek() == "\t" { self.reader.forward() }
            if self.reader.peek() in BREAKS {
                chunks.push(self.scan_line_break())
            } else {
                return chunks
            }
        }
    }

    fn scan_plain(self) -> Token {
        let chunks = StringBuilder()
        let start_mark = self.reader.get_mark()
        var end_mark = start_mark
        let indent = self.indent + 1
        var spaces: list[str] = []
        loop {
            var length = 0
            if self.reader.peek() == "#" { break }
            loop {
                let ch = self.reader.peek(length)
                if ch in NUL_SPACE_OR_BREAK { break }
                if ch == ":" {
                    let after = self.reader.peek(length + 1)
                    if after in NUL_SPACE_OR_BREAK { break }
                    if self.flow_level > 0 and after in ",[]{}" { break }
                }
                if self.flow_level > 0 and ch in ",?[]{}" { break }
                length += 1
            }
            if length == 0 { break }
            self.allow_simple_key = false
            chunks.extend(spaces)
            chunks.push(self.reader.prefix(length))
            self.reader.forward(length)
            end_mark = self.reader.get_mark()
            let found = self.scan_plain_spaces(indent, start_mark)
            if found is null { break }
            spaces = found ?? []
            if spaces.len() == 0 or
                self.reader.peek() == "#" or
                self.flow_level == 0 and self.reader.column < indent {
                break
            }
        }
        let token = Token(TokenKind.Scalar, start_mark, end_mark)
        token.value = chunks.build()
        token.plain = true
        token.style = null
        return token
    }

    fn scan_plain_spaces(self, indent: int, start_mark: Mark) -> list[str]? {
        let chunks: list[str] = []
        var length = 0
        while self.reader.peek(length) == " " { length += 1 }
        let whitespaces = self.reader.prefix(length)
        self.reader.forward(length)
        let ch = self.reader.peek()
        if ch in BREAKS {
            let line_break = self.scan_line_break()
            self.allow_simple_key = true
            var prefix = self.reader.prefix(3)
            if (prefix == "---" or prefix == "...") and self.reader.peek(3) in NUL_SPACE_OR_BREAK {
                return null
            }
            let breaks: list[str] = []
            while self.reader.peek() in SPACE_OR_BREAK {
                if self.reader.peek() == " " {
                    self.reader.forward()
                } else {
                    breaks.push(self.scan_line_break())
                    prefix = self.reader.prefix(3)
                    if (prefix == "---" or prefix == "...") and
                        self.reader.peek(3) in NUL_SPACE_OR_BREAK {
                        return null
                    }
                }
            }
            if line_break != "\n" {
                chunks.push(line_break)
            } elif breaks.len() == 0 {
                chunks.push(" ")
            }
            chunks.extend(breaks)
        } elif whitespaces.len() > 0 {
            chunks.push(whitespaces)
        }
        return chunks
    }

    fn scan_tag_handle(self, name: str, start_mark: Mark) -> str {
        var ch = self.reader.peek()
        if ch != "!" {
            throw ScannerError(
                f"while scanning a {name}",
                start_mark,
                f"expected '!', but found {python_repr(ch)}",
                self.reader.get_mark()
            )
        }
        var length = 1
        ch = self.reader.peek(length)
        if ch != " " {
            while is_word_char(ch) {
                length += 1
                ch = self.reader.peek(length)
            }
            if ch != "!" {
                self.reader.forward(length)
                throw ScannerError(
                    f"while scanning a {name}",
                    start_mark,
                    f"expected '!', but found {python_repr(ch)}",
                    self.reader.get_mark()
                )
            }
            length += 1
        }
        let value = self.reader.prefix(length)
        self.reader.forward(length)
        return value
    }

    fn scan_tag_uri(self, name: str, start_mark: Mark) -> str {
        let chunks = StringBuilder()
        var length = 0
        var ch = self.reader.peek(length)
        while is_word_char(ch) or ch in URI_CHARS {
            if ch == "%" {
                chunks.push(self.reader.prefix(length))
                self.reader.forward(length)
                length = 0
                chunks.push(self.scan_uri_escapes(name, start_mark))
            } else {
                length += 1
            }
            ch = self.reader.peek(length)
        }
        if length > 0 {
            chunks.push(self.reader.prefix(length))
            self.reader.forward(length)
            length = 0
        }
        if chunks.is_empty() {
            throw ScannerError(
                f"while parsing a {name}",
                start_mark,
                f"expected URI, but found {python_repr(ch)}",
                self.reader.get_mark()
            )
        }
        return chunks.build()
    }

    fn scan_uri_escapes(self, name: str, start_mark: Mark) -> str {
        let codes: list[int] = []
        let mark = self.reader.get_mark()
        while self.reader.peek() == "%" {
            self.reader.forward()
            for k in 0..2 {
                if self.reader.peek(k) not in HEX_DIGITS {
                    throw ScannerError(
                        f"while scanning a {name}",
                        start_mark,
                        f"expected URI escape sequence of 2 hexadecimal numbers, but found {python_repr(self.reader.peek(k))}",
                        self.reader.get_mark()
                    )
                }
            }
            codes.push(to_int(self.reader.prefix(2), 16) ?? 0)
            self.reader.forward(2)
        }
        try {
            return bytes(codes).decode()
        } catch failure: ValueError {
            throw ScannerError(
                f"while scanning a {name}",
                start_mark,
                "invalid UTF-8 in URI escapes",
                mark
            )
        }
    }

    fn scan_line_break(self) -> str {
        let ch = self.reader.peek()
        if ch == "\r" or ch == "\n" or ch == "\u{85}" {
            if self.reader.prefix(2) == "\r\n" {
                self.reader.forward(2)
            } else {
                self.reader.forward()
            }
            return "\n"
        } elif ch == "\u{2028}" or ch == "\u{2029}" {
            self.reader.forward()
            return ch
        }
        return ""
    }
}

fn code_point(code: int, start_mark: Mark, mark: Mark) -> str {
    if code > 0x10FFFF or code >= 0xD800 and code <= 0xDFFF {
        throw ScannerError(
            "while scanning a double-quoted scalar",
            start_mark,
            f"found an escape for code point {code} that is not a character",
            mark
        )
    }
    return chr(code)
}

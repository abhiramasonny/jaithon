#: Read and write JSON, strictly, with positions on every error.
#:
#: The reader implements RFC 8259 and nothing more. Trailing commas, comments,
#: single-quoted strings, unquoted member names, leading zeros, `NaN` and
#: `Infinity` are all rejected, because a parser that quietly accepts them turns
#: a malformed document into a silently different one. `\u` escapes are fully
#: supported, surrogate pairs included.
#:
#: Every syntax error raises `ParseError` carrying the line, the column and an
#: excerpt with a caret under the offending character. A JSON error without a
#: position is useless: real documents are one long minified line, and "expected
#: ',' " tells you nothing about which of the four thousand commas is missing.
#:
#: ```
#: parse("{\"a\": [1, 2]}")           # {"a": [1, 2]}
#: stringify({"a": 1})                # "{\"a\":1}"
#: get_path(parse("{\"a\":[7]}"), "a[0]")   # 7
#: ```
#:
#: `parse` builds the whole document in memory. For inputs too large for that,
#: drive a `Parser` event by event and materialise only the subtrees you want.
from std.math import is_inf, is_nan
from std.str import StringBuilder, repeat, to_float, to_int

#: The deepest nesting the reader and writer accept.
#:
#: Recursion is bounded so that an adversarial document of ten thousand `[`
#: characters raises `ParseError` instead of exhausting the call stack.
pub const MAX_DEPTH: int = 512

const _OBJECT = 0
const _ARRAY = 1

let _HEX = "0123456789abcdef"

# The seven characters JSON gives a two-character escape to.
let _SHORT_ESCAPES: dict[str, str] = {
    "\"": "\\\"",
    "\\": "\\\\",
    "\x08": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\x0C": "\\f",
    "\r": "\\r",
}

#: Convert a value of your own type into something `stringify` can write.
#:
#: `stringify` knows null, bool, int, float, str, list, tuple and dict. A class
#: that implements this trait joins that set: the writer calls `to_json` and
#: serialises whatever comes back.
#:
#: ```
#: class Money: JsonSerializable {
#:     let cents: int
#:     pub fn init(self, cents: int) { self.cents = cents }
#:     pub fn to_json(self) -> any { return self.cents }
#: }
#: stringify([Money(250)])    # "[250]"
#: ```
pub trait JsonSerializable {
    #: Return a value built only from the types `stringify` already knows.
    fn to_json(self) -> any
}

#: What a `Parser` event announces.
#:
#: `Key` carries the member name and `Value` the scalar; the four bracket
#: events carry null. There is no event for the end of the document — the
#: parser returns null instead.
pub enum EventKind {
    #: An object opened. Every following `Key` belongs to it until the
    #: matching `ObjectEnd`.
    ObjectStart,
    #: The object opened by the last unmatched `ObjectStart` closed.
    ObjectEnd,
    #: An array opened. Its elements arrive as bare `Value` events with no
    #: `Key` between them.
    ArrayStart,
    #: The array opened by the last unmatched `ArrayStart` closed.
    ArrayEnd,
    #: A member name inside an object, carried in `value` as a `str`. The
    #: member's own value is the event after it.
    Key,
    #: A scalar: a string, a number, a bool or null, carried in `value` as
    #: itself. A null member is a `Value` whose `value` is null, which is why
    #: `kind` and not `value` is what a reader switches on.
    Value,
}

#: One structural event from a streaming `Parser`.
pub class Event {
    #: Which kind of event this is.
    pub let kind: EventKind

    #: The member name for `Key`, the scalar for `Value`, null otherwise.
    pub let value: any

    #: The 1-based line the event's first character sits on.
    pub let line: int

    #: The 1-based column the event's first character sits on.
    pub let column: int

    #: Record an event at `line` and `column`.
    pub fn init(self, kind: EventKind, value: any, line: int, column: int) {
        self.kind = kind
        self.value = value
        self.line = line
        self.column = column
    }

    fn __str__(self) -> str {
        return f"Event({self.kind}, {repr(self.value)}, {self.line}:{self.column})"
    }
}

#: A pull parser over one JSON document.
#:
#: Two front ends share one scanner. `parse_document` builds the whole tree;
#: `next_event` walks the document one bracket, name or scalar at a time,
#: allocating nothing but the scalars themselves. Mixing them is the point:
#: read events until a `Key` you care about arrives, then call `read_value` to
#: materialise that subtree, or `skip_value` to step over it without building
#: anything.
#:
#: ```
#: let parser = Parser("{\"small\": 1, \"huge\": [0, 1, 2]}")
#: parser.next_event().kind          # EventKind.ObjectStart
#: parser.next_event().value         # "small"
#: parser.read_value()               # 1
#: parser.next_event().value         # "huge"
#: parser.skip_value()
#: parser.next_event().kind          # EventKind.ObjectEnd
#: ```
pub class Parser {
    let text: str
    let length: int
    var index: int
    var line: int
    var line_start: int
    var stack: list[int]
    var counts: list[int]
    var wants_value: bool
    var done: bool

    #: Create a parser positioned at the start of `text`.
    pub fn init(self, text: str) {
        self.text = text
        self.length = text.len()
        self.index = 0
        self.line = 1
        self.line_start = 0
        self.stack = []
        self.counts = []
        self.wants_value = false
        self.done = false
    }

    #: Parse the one value the document holds and require nothing to follow it.
    #:
    #: ```
    #: Parser("[1, 2]").parse_document()    # [1, 2]
    #: ```
    pub fn parse_document(self) -> any {
        let value = self._read(0, true)
        self.done = true
        self._skip_whitespace()
        if self.index < self.length {
            throw self._error("unexpected content after the top-level value")
        }
        return value
    }

    #: Return the next structural event, or null once the document is complete.
    #:
    #: Raises `ParseError` when anything follows the top-level value.
    pub fn next_event(self) -> Event? {
        self._skip_whitespace()
        if self.wants_value {
            self.wants_value = false
            return self._item_event()
        }
        if self.stack.len() == 0 {
            if self.done {
                if self.index < self.length {
                    throw self._error("unexpected content after the top-level value")
                }
                return null
            }
            return self._item_event()
        }
        return self._container_event()
    }

    #: Parse the value the parser is positioned at and return it whole.
    #:
    #: Valid at the start of the document, immediately after a `Key` event, and
    #: at an array element. Raises `ValueError` anywhere else, since there is no
    #: value there to read.
    pub fn read_value(self) -> any {
        self._prepare_value()
        let value = self._read(0, true)
        self._value_done()
        return value
    }

    #: Step over the value the parser is positioned at, building nothing.
    #:
    #: Accepts the same positions as `read_value`. The value is still scanned in
    #: full — a malformed subtree is an error whether or not you wanted it — but
    #: no list or dict is allocated for it.
    pub fn skip_value(self) -> void {
        self._prepare_value()
        self._read(0, false)
        self._value_done()
    }

    #: Report whether only whitespace remains.
    #:
    #: Skips that whitespace, which is insignificant in JSON.
    pub fn at_end(self) -> bool {
        self._skip_whitespace()
        return self.index >= self.length
    }

    #: Return the 1-based line and column the parser is stopped at.
    pub fn position(self) -> tuple[int, int] { return (self.line, self._column()) }

    fn _prepare_value(self) -> void {
        if self.wants_value {
            self.wants_value = false
            return
        }
        if self.stack.len() == 0 {
            if self.done { throw ValueError("the document has already been read to its end") }
            return
        }
        let top = self.stack.len() - 1
        if self.stack[top] == _OBJECT {
            throw ValueError("inside an object, read_value must follow a Key event")
        }
        self._skip_whitespace()
        self._separator(top, "]", "an array element")
        if self._at("]") { throw ValueError("the array has no further element to read") }
    }

    fn _item_event(self) -> Event {
        self._skip_whitespace()
        if self.index >= self.length {
            throw self._error("unexpected end of input: expected a JSON value")
        }
        let line = self.line
        let column = self._column()
        if self._at("{") {
            self._advance()
            self._push(_OBJECT)
            return Event(EventKind.ObjectStart, null, line, column)
        }
        if self._at("[") {
            self._advance()
            self._push(_ARRAY)
            return Event(EventKind.ArrayStart, null, line, column)
        }
        let value = self._scan_scalar()
        self._value_done()
        return Event(EventKind.Value, value, line, column)
    }

    fn _container_event(self) -> Event {
        let top = self.stack.len() - 1
        let is_object = self.stack[top] == _OBJECT
        let closing = is_object ? "}" : "]"
        if self._at(closing) {
            let end_line = self.line
            let end_column = self._column()
            self._advance()
            self._pop()
            let kind = is_object ? EventKind.ObjectEnd : EventKind.ArrayEnd
            return Event(kind, null, end_line, end_column)
        }
        self._separator(top, closing, is_object ? "an object member" : "an array element")
        if not is_object { return self._item_event() }
        let line = self.line
        let column = self._column()
        let key = self._scan_member_name()
        self._expect(":", "expected ':' after a member name")
        self.wants_value = true
        return Event(EventKind.Key, key, line, column)
    }

    # Consume the comma that must separate two elements of the container at
    # `top`, and reject the trailing comma that would otherwise hide behind it.
    fn _separator(self, top: int, closing: str, what: str) -> void {
        if self.counts[top] == 0 { return }
        self._expect(",", f"expected ',' or '{closing}' after {what}")
        self._skip_whitespace()
        if self._at(closing) { throw self._error(f"trailing comma before '{closing}'") }
    }

    fn _push(self, kind: int) -> void {
        if self.stack.len() >= MAX_DEPTH {
            throw self._error(f"nested deeper than the {MAX_DEPTH} level limit")
        }
        self.stack.push(kind)
        self.counts.push(0)
    }

    fn _pop(self) -> void {
        self.stack.pop()
        self.counts.pop()
        self._value_done()
    }

    fn _value_done(self) -> void {
        if self.stack.len() == 0 {
            self.done = true
            return
        }
        self.counts[self.counts.len() - 1] += 1
    }

    # `build` is false when the caller only wants the input validated and the
    # position advanced, so a skipped subtree costs no container allocations.
    fn _read(self, depth: int, build: bool) -> any {
        self._skip_whitespace()
        if self.index >= self.length {
            throw self._error("unexpected end of input: expected a JSON value")
        }
        if self._at("{") { return self._read_object(depth, build) }
        if self._at("[") { return self._read_array(depth, build) }
        return self._scan_scalar()
    }

    fn _read_object(self, depth: int, build: bool) -> any {
        if depth >= MAX_DEPTH {
            throw self._error(f"nested deeper than the {MAX_DEPTH} level limit")
        }
        self._advance()
        let members: dict[str, any] = {}
        self._skip_whitespace()
        if self._at("}") {
            self._advance()
            return members
        }
        loop {
            let key = self._scan_member_name()
            self._expect(":", "expected ':' after a member name")
            let value = self._read(depth + 1, build)
            if build {
                if key in members { throw self._error(f"duplicate member name {repr(key)}") }
                members[key] = value
            }
            if self._tail("}", "an object member") { return members }
        }
    }

    fn _read_array(self, depth: int, build: bool) -> any {
        if depth >= MAX_DEPTH {
            throw self._error(f"nested deeper than the {MAX_DEPTH} level limit")
        }
        self._advance()
        let items: list[any] = []
        self._skip_whitespace()
        if self._at("]") {
            self._advance()
            return items
        }
        loop {
            let value = self._read(depth + 1, build)
            if build { items.push(value) }
            if self._tail("]", "an array element") { return items }
        }
    }

    # After an element, consume either the closing bracket — reporting that the
    # container is finished — or the comma that introduces the next element.
    fn _tail(self, closing: str, what: str) -> bool {
        self._skip_whitespace()
        if self._at(closing) {
            self._advance()
            return true
        }
        self._expect(",", f"expected ',' or '{closing}' after {what}")
        self._skip_whitespace()
        if self._at(closing) { throw self._error(f"trailing comma before '{closing}'") }
        return false
    }

    fn _scan_member_name(self) -> str {
        self._skip_whitespace()
        if self.index >= self.length {
            throw self._error("unexpected end of input: expected a member name")
        }
        if not self._at("\"") {
            throw self._error(
                f"expected a double-quoted member name, got {_show(self.text[self.index])}"
            )
        }
        return self._scan_string()
    }

    fn _scan_scalar(self) -> any {
        let ch = self.text[self.index]
        if ch == "\"" { return self._scan_string() }
        if ch == "-" or _is_digit(ch) { return self._scan_number() }
        if ch == "t" {
            self._expect_word("true")
            return true
        }
        if ch == "f" {
            self._expect_word("false")
            return false
        }
        if ch == "n" {
            self._expect_word("null")
            return null
        }
        if ch == "N" or ch == "I" {
            throw self._error("JSON has no NaN or Infinity literal")
        }
        if ch == "'" { throw self._error("JSON strings are double-quoted, never single-quoted") }
        if ch == "/" or ch == "#" { throw self._error("JSON has no comments") }
        if ch == "+" or ch == "." {
            throw self._error(f"a JSON number may not begin with {_show(ch)}")
        }
        throw self._error(f"unexpected character {_show(ch)}")
    }

    fn _scan_string(self) -> str {
        self._advance()
        let out = StringBuilder()
        loop {
            if self.index >= self.length { throw self._error("unterminated string") }
            let ch = self.text[self.index]
            if ch == "\"" {
                self._advance()
                return out.build()
            }
            if ch == "\\" {
                self._advance()
                out.push(self._scan_escape())
                continue
            }
            if ord(ch) < 0x20 {
                throw self._error(
                    f"a raw control character {_show(ch)} must be escaped inside a string"
                )
            }
            out.push(ch)
            self._advance()
        }
    }

    fn _scan_escape(self) -> str {
        if self.index >= self.length { throw self._error("the string ends with a lone backslash") }
        let marker = self.text[self.index]
        self._advance()
        if marker == "\"" { return "\"" }
        if marker == "\\" { return "\\" }
        if marker == "/" { return "/" }
        if marker == "b" { return "\x08" }
        if marker == "f" { return "\x0C" }
        if marker == "n" { return "\n" }
        if marker == "r" { return "\r" }
        if marker == "t" { return "\t" }
        if marker != "u" { throw self._error(f"unknown escape '\\{marker}'") }
        let first = self._scan_hex4()
        if first < 0xD800 or first > 0xDFFF { return chr(first) }
        if first >= 0xDC00 {
            throw self._error(
                f"lone low surrogate \\u{_hex4(first)}; a pair starts with \\uD800-\\uDBFF"
            )
        }
        let paired = self.index + 1 < self.length and
            self.text[self.index] == "\\" and
            self.text[self.index + 1] == "u"
        if not paired {
            throw self._error(
                f"high surrogate \\u{_hex4(first)} is not followed by \\uDC00-\\uDFFF"
            )
        }
        self._advance()
        self._advance()
        let second = self._scan_hex4()
        if second < 0xDC00 or second > 0xDFFF {
            throw self._error(
                f"high surrogate \\u{_hex4(first)} is followed by \\u{_hex4(second)}, " +
                    "which is not a low surrogate"
            )
        }
        return chr(0x10000 + (first - 0xD800 << 10) + (second - 0xDC00))
    }

    # Accumulated digit by digit rather than through `to_int`, which trims
    # whitespace and accepts a sign: "\u 12f" and "\u+123" would slip through.
    fn _scan_hex4(self) -> int {
        if self.index + 4 > self.length {
            throw self._error("a '\\u' escape needs four hexadecimal digits")
        }
        let digits = self.text[self.index:self.index + 4]
        var value = 0
        for digit in digits {
            let nibble = _hex_value(digit)
            if nibble < 0 {
                throw self._error(f"'\\u{digits}' is not four hexadecimal digits")
            }
            value = value * 16 + nibble
        }
        self.index += 4
        return value
    }

    fn _scan_number(self) -> any {
        let start = self.index
        if self._at("-") { self._advance() }
        if self.index >= self.length {
            throw self._error("a JSON number needs at least one digit")
        }
        let first = self.text[self.index]
        if first == "0" {
            self._advance()
            if self.index < self.length and _is_digit(self.text[self.index]) {
                throw self._error("a JSON number may not have a leading zero")
            }
        } elif _is_digit(first) {
            self._digits()
        } else {
            throw self._error(f"expected a digit, got {_show(first)}")
        }
        var is_float = false
        if self._at(".") {
            is_float = true
            self._advance()
            if not self._at_digit() {
                throw self._error("a JSON number needs at least one digit after '.'")
            }
            self._digits()
        }
        if self._at("e") or self._at("E") {
            is_float = true
            self._advance()
            if self._at("+") or self._at("-") { self._advance() }
            if not self._at_digit() {
                throw self._error("a JSON number needs at least one digit in its exponent")
            }
            self._digits()
        }
        let literal = self.text[start:self.index]
        if not is_float {
            let whole = to_int(literal)
            # An integer too large for 64 bits becomes a float rather than an
            # error: it is valid JSON, and refusing it would reject documents
            # every other implementation accepts.
            if whole is not null { return whole }
        }
        let value = to_float(literal)
        if value is null { throw self._error(f"malformed number '{literal}'") }
        return value
    }

    fn _digits(self) -> void {
        while self._at_digit() { self._advance() }
    }

    fn _at_digit(self) -> bool {
        return self.index < self.length and _is_digit(self.text[self.index])
    }

    fn _at(self, ch: str) -> bool {
        return self.index < self.length and self.text[self.index] == ch
    }

    fn _expect(self, expected: str, message: str) -> void {
        self._skip_whitespace()
        if not self._at(expected) { throw self._error(message) }
        self._advance()
    }

    fn _expect_word(self, word: str) -> void {
        let stop = self.index + word.len()
        if stop > self.length or self.text[self.index:stop] != word {
            throw self._error(f"expected '{word}'")
        }
        self.index = stop
    }

    fn _skip_whitespace(self) -> void {
        while self.index < self.length {
            let ch = self.text[self.index]
            if ch != " " and ch != "\t" and ch != "\n" and ch != "\r" { return }
            self._advance()
        }
    }

    fn _advance(self) -> void {
        if self.text[self.index] == "\n" {
            self.line += 1
            self.line_start = self.index + 1
        }
        self.index += 1
    }

    fn _column(self) -> int { return self.index - self.line_start + 1 }

    # Built only on the error path, so the scan itself pays nothing for it.
    fn _error(self, message: str) -> ParseError {
        let line = self.line
        let column = self._column()
        var stop = self.line_start
        while stop < self.length and self.text[stop] != "\n" { stop += 1 }
        let source = self.text[self.line_start:stop].replace("\t", " ")
        let (excerpt, caret_column) = _window(source, column)
        let number = str(line)
        let gutter = repeat(" ", number.len())
        return ParseError(
            f"invalid JSON at line {line}, column {column}: {message}\n" +
                f"{gutter} |\n" +
                f"{number} | {excerpt}\n" +
                f"{gutter} | " +
                repeat(" ", caret_column - 1) +
                "^"
        )
    }
}

#: Parse `text` as one JSON document and return the value it holds.
#:
#: Objects become `dict[str, any]`, arrays become `list[any]`, and the three
#: literals become null, bool and the numbers. A number with no fraction and no
#: exponent is an `int`; anything else is a `float`.
#:
#: Raises `ParseError`, with the line, column and a caret excerpt, on anything
#: RFC 8259 does not allow.
#:
#: ```
#: parse("[1, 2.5, \"x\", true, null]")   # [1, 2.5, "x", true, null]
#: parse("{\"n\": 1}")["n"]               # 1
#: parse("\"\\uD83D\\uDE00\"")            # "😀"
#: ```
pub fn parse(text: str) -> any { return Parser(text).parse_document() }

#: Render `value` as JSON.
#:
#: With `indent` at 0 the output is compact, with no space anywhere outside a
#: string. A positive `indent` puts every element on its own line, indented by
#: that many spaces per level. `sort_keys` orders every object's members by
#: name, which is what makes two renderings of equal data byte-identical —
#: without it members keep insertion order, which is stable per dict but not
#: across two dicts built differently. `ascii_only` escapes every non-ASCII
#: character as `\u`, using a surrogate pair above U+FFFF.
#:
#: Raises `ValueError` for a NaN or an infinity, neither of which JSON can
#: express, and for a structure that contains itself. Raises `TypeError` for a
#: value of any other unsupported type, and for a non-str member name.
#:
#: ```
#: stringify({"b": 1, "a": 2}, sort_keys: true)   # "{\"a\":2,\"b\":1}"
#: stringify([1, 2], indent: 2)                   # "[\n  1,\n  2\n]"
#: stringify("é", ascii_only: true)               # "\"\\u00e9\""
#: ```
pub fn stringify(
    value: any,
    indent: int = 0,
    sort_keys: bool = false,
    ascii_only: bool = false
) -> str {
    if indent < 0 { throw ValueError(f"indent must be non-negative, got {indent}") }
    let out = StringBuilder()
    _write(out, value, indent, 0, sort_keys, ascii_only, [])
    return out.build()
}

#: Follow a dotted path into a parsed document, or return `default`.
#:
#: A path is member names joined by `.`, with `[i]` for array elements; a
#: negative subscript counts from the end. Every step that is missing, out of
#: range, or applied to the wrong kind of container yields `default` rather than
#: raising, since probing an optional corner of a document is ordinary. A
#: malformed path raises `ValueError`, because that is a bug in the caller.
#:
#: A member name containing `.` or `[` cannot be addressed this way; index the
#: dict directly for those.
#:
#: ```
#: let document = parse("{\"a\": {\"b\": [10, 20]}}")
#: get_path(document, "a.b[1]")           # 20
#: get_path(document, "a.b[-1]")          # 20
#: get_path(document, "a.missing", 0)     # 0
#: ```
pub fn get_path(value: any, path: str, default: any = null) -> any {
    var current = value
    for step in _path_steps(path) {
        if isinstance(step, int) {
            if not isinstance(current, list) { return default }
            var position = step
            if position < 0 { position += current.len() }
            if position < 0 or position >= current.len() { return default }
            current = current[position]
        } else {
            if not isinstance(current, dict) { return default }
            if step not in current { return default }
            current = current[step]
        }
    }
    return current
}

#: Build an instance of `cls` from `data`, checking each field as it goes.
#:
#: Every member of `data` is assigned to the declared field of the same name. A
#: member naming no declared field raises `ValueError` — a typo in a config file
#: should be reported, not ignored — and a field absent from `data` keeps the
#: value its declaration gives it.
#:
#: Keeping the declared value is why the instance is built by calling `cls()`
#: when that is possible: a field initialiser is compiled into `init`, so an
#: instance allocated without running it has null in every field, and both the
#: promise above and the type inference below — which reads the type of the
#: declared default — would be empty. A class whose `init` demands arguments
#: cannot be called that way and is allocated raw, defaults and all left unset.
#:
#: Each value is checked against the expected type before it is stored, and
#: `TypeError` names the field, the expectation and what arrived. The
#: expectation comes from `types[name]` when you supply it — a type such as
#: `int`, or a class — and otherwise from the type of the field's declared
#: default; a field defaulting to null accepts anything, since its declaration
#: leaves nothing to check against. An `int` where a `float` is wanted is
#: promoted, because JSON writes 1.0 as `1`. A type given in `types` that is a
#: class, against a value that is a dict, recurses — so nested objects map onto
#: nested classes.
#:
#: ```
#: class Port {
#:     pub var number: int = 0
#:     pub var host: str = "localhost"
#:     pub fn init(self) { }
#: }
#: let port = to_class(Port, parse("{\"number\": 8080}"))
#: port.number   # 8080
#: port.host     # "localhost"
#: ```
pub fn to_class(cls: any, data: dict[str, any], types: dict[str, any]? = null) -> any {
    let instance = _blank(cls)
    let fields = _field_names(instance)
    if fields.len() == 0 {
        throw TypeError(f"{cls} declares no readable fields to fill from JSON")
    }
    let declared: dict[str, any] = types ?? {}
    for key in data.keys() {
        if key not in fields {
            throw ValueError(f"unknown member {repr(key)}; {cls} declares {fields}")
        }
    }
    for name in fields {
        if name not in data { continue }
        let expected = declared.get(name, null) ?? _type_of(_field_of(instance, name))
        __prim__.obj_set_field(instance, name, _fit(data[name], expected, name))
    }
    return instance
}

#: Convert `instance` into a dict of its fields, ready for `stringify`.
#:
#: Every field `dir` reports whose name does not begin with `_` is included.
#: Nested instances become nested dicts, and lists and dicts are converted
#: element by element, so the result is built only from the types `stringify`
#: knows. Raises `ValueError` on a structure that contains itself.
#:
#: ```
#: class Point {
#:     pub var x: int = 1
#:     pub var y: int = 2
#:     pub fn init(self) { }
#: }
#: stringify(from_class(Point()), sort_keys: true)   # "{\"x\":1,\"y\":2}"
#: ```
pub fn from_class(instance: any) -> dict[str, any] { return _from_class(instance, 0) }

fn _write(
    out: StringBuilder,
    value: any,
    indent: int,
    depth: int,
    sort_keys: bool,
    ascii_only: bool,
    active: list[int]
) -> void {
    if value is null {
        out.push("null")
        return
    }
    if isinstance(value, bool) {
        out.push(value ? "true" : "false")
        return
    }
    if isinstance(value, int) {
        out.push(str(value))
        return
    }
    if isinstance(value, float) {
        if is_nan(value) { throw ValueError("JSON has no representation for NaN") }
        if is_inf(value) { throw ValueError(f"JSON has no representation for {value}") }
        out.push(str(value))
        return
    }
    if isinstance(value, str) {
        out.push(_quote(value, ascii_only))
        return
    }
    for seen in active {
        if seen == id(value) {
            throw ValueError("cannot serialize a structure that contains itself")
        }
    }
    if depth >= MAX_DEPTH {
        throw ValueError(f"nested deeper than the {MAX_DEPTH} level limit")
    }
    active.push(id(value))
    defer { active.pop() }
    if isinstance(value, JsonSerializable) {
        _write(out, value.to_json(), indent, depth, sort_keys, ascii_only, active)
        return
    }
    let is_dict = isinstance(value, dict)
    if not (is_dict or isinstance(value, list) or isinstance(value, tuple)) {
        throw TypeError(f"cannot serialize a value of type {type_of(value)} to JSON")
    }
    let open_pad = indent > 0 ? "\n" + repeat(" ", (depth + 1) * indent) : ""
    let close_pad = indent > 0 ? "\n" + repeat(" ", depth * indent) : ""
    if is_dict {
        var keys = value.keys()
        for key in keys {
            if not isinstance(key, str) {
                throw TypeError(f"a JSON member name must be a str, got {type_of(key)}")
            }
        }
        if keys.len() == 0 {
            out.push("{}")
            return
        }
        if sort_keys { keys = sorted(keys) }
        out.push("{")
        var position = 0
        for key in keys {
            if position > 0 { out.push(",") }
            out.push(open_pad)
            out.push(_quote(key, ascii_only))
            out.push(indent > 0 ? ": " : ":")
            _write(out, value[key], indent, depth + 1, sort_keys, ascii_only, active)
            position += 1
        }
        out.push(close_pad)
        out.push("}")
        return
    }
    if len(value) == 0 {
        out.push("[]")
        return
    }
    out.push("[")
    var count = 0
    for item in value {
        if count > 0 { out.push(",") }
        out.push(open_pad)
        _write(out, item, indent, depth + 1, sort_keys, ascii_only, active)
        count += 1
    }
    out.push(close_pad)
    out.push("]")
}

fn _quote(text: str, ascii_only: bool) -> str {
    let out = StringBuilder()
    out.push("\"")
    for ch in text {
        let short = _SHORT_ESCAPES.get(ch, "")
        if short.len() > 0 {
            out.push(short)
            continue
        }
        let code = ord(ch)
        if code < 0x20 {
            out.push("\\u" + _hex4(code))
        } elif code < 0x7F or not ascii_only {
            out.push(ch)
        } elif code > 0xFFFF {
            let offset = code - 0x10000
            out.push("\\u" + _hex4(0xD800 + (offset >> 10)))
            out.push("\\u" + _hex4(0xDC00 + (offset & 0x3FF)))
        } else {
            out.push("\\u" + _hex4(code))
        }
    }
    out.push("\"")
    return out.build()
}

fn _path_steps(path: str) -> list[any] {
    if path.len() == 0 { throw ValueError("a path must not be empty") }
    let steps: list[any] = []
    let length = path.len()
    var index = 0
    while index < length {
        if path[index] == "[" {
            var close = index + 1
            while close < length and path[close] != "]" { close += 1 }
            if close >= length { throw ValueError(f"unterminated '[' in path {repr(path)}") }
            let digits = path[index + 1:close]
            let subscript = to_int(digits)
            if subscript is null {
                throw ValueError(f"path {repr(path)} has a non-integer subscript {repr(digits)}")
            }
            steps.push(subscript)
            index = close + 1
        } else {
            var stop = index
            while stop < length and path[stop] != "." and path[stop] != "[" { stop += 1 }
            if stop == index { throw ValueError(f"empty segment in path {repr(path)}") }
            steps.push(path[index:stop])
            index = stop
        }
        if index < length and path[index] == "." {
            index += 1
            if index >= length { throw ValueError(f"path {repr(path)} ends with '.'") }
        }
    }
    return steps
}

fn _from_class(instance: any, depth: int) -> dict[str, any] {
    if depth >= MAX_DEPTH {
        throw ValueError(f"nested deeper than the {MAX_DEPTH} level limit")
    }
    let out: dict[str, any] = {}
    for name in _field_names(instance) {
        out[name] = _jsonable(_field_of(instance, name), depth + 1)
    }
    return out
}

fn _jsonable(value: any, depth: int) -> any {
    if _is_scalar(value) { return value }
    if depth >= MAX_DEPTH {
        throw ValueError(f"nested deeper than the {MAX_DEPTH} level limit")
    }
    if isinstance(value, list) or isinstance(value, tuple) {
        return [_jsonable(item, depth + 1) for item in value]
    }
    if isinstance(value, dict) {
        return {key: _jsonable(item, depth + 1) for (key, item) in value.items()}
    }
    if isinstance(value, JsonSerializable) { return _jsonable(value.to_json(), depth + 1) }
    if isinstance(value, set) {
        throw TypeError("JSON has no set type; convert the set to a list first")
    }
    return _from_class(value, depth + 1)
}

# A fresh instance of `cls` with its declared field values in place, or a raw
# one when `init` needs arguments this function has no way to invent.
fn _blank(cls: any) -> any {
    try {
        return cls()
    } catch _error: TypeError {
        return __prim__.obj_new(cls)
    }
}

fn _field_names(value: any) -> list[str] {
    let names: list[str] = []
    for name in dir(value) {
        if name.len() == 0 or name[0] == "_" { continue }
        if callable(_field_of(value, name)) { continue }
        names.push(name)
    }
    return names
}

fn _field_of(value: any, name: str) -> any {
    try {
        return __prim__.obj_get_field(value, name)
    } catch _missing: AttributeError {
        return null
    }
}

fn _type_of(sample: any) -> any {
    if sample is null { return null }
    return type_of(sample)
}

# `expected` arrives either as a type name — which is what `type_of()` on a
# declared default hands back — or as the type itself, when the caller listed it
# in `types`. Both are accepted, and the two are told apart by their own type.
fn _fit(value: any, expected: any, name: str) -> any {
    if expected is null { return value }
    if isinstance(expected, str) {
        if type_of(value) == expected { return value }
        if expected == "float" and isinstance(value, int) { return float(value) }
        throw TypeError(_mismatch(name, expected, value))
    }
    if isinstance(value, expected) { return value }
    # Whether `expected` means float is asked of `expected` rather than assumed,
    # so that a float-like type of the caller's own promotes an int too.
    if isinstance(value, int) and isinstance(0.0, expected) { return float(value) }
    if isinstance(value, dict) { return to_class(expected, value) }
    throw TypeError(_mismatch(name, expected, value))
}

fn _mismatch(name: str, expected: any, value: any) -> str {
    return f"field {repr(name)} expects {expected}, got {type_of(value)} ({repr(value)})"
}

fn _is_scalar(value: any) -> bool {
    if value is null { return true }
    if isinstance(value, bool) or isinstance(value, str) { return true }
    return isinstance(value, int) or isinstance(value, float)
}

fn _is_digit(ch: str) -> bool {
    let code = ord(ch)
    return code >= 0x30 and code <= 0x39
}

fn _hex_value(ch: str) -> int {
    let code = ord(ch)
    if code >= 0x30 and code <= 0x39 { return code - 0x30 }
    if code >= 0x61 and code <= 0x66 { return code - 0x61 + 10 }
    if code >= 0x41 and code <= 0x46 { return code - 0x41 + 10 }
    return -1
}

fn _hex4(code: int) -> str {
    var out = ""
    var shift = 12
    while shift >= 0 {
        out = out + _HEX[code >> shift & 0xF]
        shift -= 4
    }
    return out
}

fn _show(ch: str) -> str {
    let code = ord(ch)
    if code < 0x20 or code == 0x7F { return "U+" + _hex4(code).upper() }
    return "'" + ch + "'"
}

# Minified JSON is one enormous line, so an excerpt of the whole line would be
# unreadable and would push the caret off the screen. Keep a window around the
# offending column instead, and report where the caret lands inside it.
fn _window(source: str, column: int) -> tuple[str, int] {
    let width = 72
    if source.len() <= width { return (source, column) }
    var start = column - 1 - width // 2
    if start < 0 { start = 0 }
    var stop = start + width
    if stop > source.len() {
        stop = source.len()
        start = stop - width
    }
    var piece = source[start:stop]
    var shift = start
    if start > 0 {
        piece = "..." + piece
        shift -= 3
    }
    if stop < source.len() { piece = piece + "..." }
    return (piece, column - shift)
}

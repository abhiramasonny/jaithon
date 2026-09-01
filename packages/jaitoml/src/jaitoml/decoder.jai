#: Read TOML 1.0 documents, following Python's `tomllib` statement for statement.
#:
#: `loads` walks the source one statement at a time — a key/value pair, a
#: `[table]` header, an `[[array]]` header, a comment or a blank line — and
#: fills a nested dict. The rules that make TOML strict are the ones that are
#: easy to get wrong, so they are kept in the same shape `tomllib` keeps them:
#: a `_Flags` tree remembers which tables were declared explicitly and which
#: values (inline tables and arrays) are frozen, and every duplicate key,
#: re-declared table or write into a frozen namespace is refused with the
#: same message `tomllib` gives, at the same position.
#:
#: ```
#: loads("a.b = 1\n[c]\nd = [1, 2]")     # {"a": {"b": 1}, "c": {"d": [1, 2]}}
#: loads("x = 1\nx = 2")                 # TOMLDecodeError: Cannot overwrite a value (at line 2, column 6)
#: ```
#:
#: Values map as they do in Python: tables and inline tables become
#: `dict[str, any]`, arrays become `list[any]`, strings become `str`,
#: integers `int`, floats `float`, booleans `bool`, and the four date-time
#: forms become `DateTime` (with an offset for an offset date-time, without one
#: for a local date-time), `Date` and `Time` from `datetime.jai`.
#:
#: Two deviations from `tomllib`, both forced by the language. A Jaithon `int`
#: is 64-bit, so an integer literal outside that range raises
#: `TOMLDecodeError` where Python would return a big int; TOML itself only
#: promises 64 bits. And `load` reads a `std.io.File` as text, where Python
#: insists on a binary handle.
import std.math as math

from std.io import File
from std.str import StringBuilder, to_int
from .datetime import Date, DateTime, Time

const _FROZEN = 0
const _EXPLICIT_NEST = 1

#: The error `loads` and `load` raise for a document that is not valid TOML.
#:
#: The message ends with the position, in the form Python 3.13 writes:
#: `Invalid value (at line 3, column 7)`, or `(at end of document)` when the
#: source ran out. `lineno` and `colno` carry the same position as numbers,
#: which Python only added in 3.14; both are 1-based and counted in
#: characters, over a source whose `\r\n` pairs were already folded to `\n`.
pub class TOMLDecodeError extends ValueError {
    #: The message without the position suffix.
    pub let msg: str

    #: The source document being parsed.
    pub let doc: str

    #: The character offset the error points at.
    pub let pos: int

    #: The 1-based line of `pos`.
    pub let lineno: int

    #: The 1-based column of `pos`.
    pub let colno: int

    #: Build the error for `msg` at offset `pos` of `doc`.
    pub fn init(self, msg: str, doc: str = "", pos: int = 0) {
        var line = 1
        var line_start = 0
        var index = 0
        let stop = pos < doc.len() ? pos : doc.len()
        while index < stop {
            if doc[index] == "\n" {
                line += 1
                line_start = index + 1
            }
            index += 1
        }
        let column = pos - line_start + 1
        let where = pos >= doc.len() ? "end of document" : f"line {line}, column {column}"
        super(f"{msg} (at {where})")
        self.msg = msg
        self.doc = doc
        self.pos = pos
        self.lineno = line
        self.colno = column
    }
}

#: Parse `s` as a TOML document and return its top-level table.
#:
#: `parse_float`, when given, is called with the source text of every float
#: (underscores included, as in Python) and its result is stored in place of
#: the `float`. It must not return a list or a dict; that raises `ValueError`.
#:
#: Raises `TOMLDecodeError` on anything TOML 1.0 does not allow.
#:
#: ```
#: loads("title = \"TOML\"\n[owner]\nname = \"Tom\"")
#: #     {"title": "TOML", "owner": {"name": "Tom"}}
#: loads("n = 1_000\nf = 6.02e23\nd = 1979-05-27")["d"]    # Date(1979, 5, 27)
#: ```
pub fn loads(s: str, parse_float: any = null) -> dict[str, any] {
    return _Parser(s.replace("\r\n", "\n"), parse_float).parse_document()
}

#: Parse the TOML document `fp` holds and return its top-level table.
#:
#: The whole file is read first; it is text, decoded as UTF-8 by `std.io`.
#:
#: ```
#: let file = open("pyproject.toml")
#: defer { file.close() }
#: load(file)["project"]["name"]
#: ```
pub fn load(fp: File, parse_float: any = null) -> dict[str, any] {
    return loads(fp.read(), parse_float)
}

class _FlagNode {
    pub var flags: list[int]
    pub var recursive_flags: list[int]
    pub var nested: dict[str, _FlagNode]

    fn init(self) {
        self.flags = []
        self.recursive_flags = []
        self.nested = {}
    }
}

#: Which namespaces are explicitly declared and which are frozen.
#:
#: `_EXPLICIT_NEST` marks a table a `[header]` or a dotted key created, so a
#: later `[header]` cannot open it again. `_FROZEN` marks an inline table or
#: an array, recursively: nothing beneath it may be added to from outside.
class _Flags {
    var root: dict[str, _FlagNode]
    var pending: list[list[str]]

    fn init(self) {
        self.root = {}
        self.pending = []
    }

    pub fn add_pending(self, key: list[str]) -> void { self.pending.push(key) }

    pub fn finalize_pending(self) -> void {
        for key in self.pending { self.set(key, _EXPLICIT_NEST, false) }
        self.pending = []
    }

    pub fn unset_all(self, key: list[str]) -> void {
        var cont = self.root
        for index in 0..key.len() - 1 {
            let part = key[index]
            if part not in cont { return }
            cont = cont[part].nested
        }
        cont.remove(key[key.len() - 1])
    }

    pub fn set(self, key: list[str], flag: int, recursive: bool) -> void {
        var cont = self.root
        for index in 0..key.len() - 1 {
            let part = key[index]
            if part not in cont { cont[part] = _FlagNode() }
            cont = cont[part].nested
        }
        let stem = key[key.len() - 1]
        if stem not in cont { cont[stem] = _FlagNode() }
        let target = recursive ? cont[stem].recursive_flags : cont[stem].flags
        if flag not in target { target.push(flag) }
    }

    pub fn has(self, key: list[str], flag: int) -> bool {
        if key.len() == 0 { return false }
        var cont = self.root
        for index in 0..key.len() - 1 {
            let part = key[index]
            if part not in cont { return false }
            let inner = cont[part]
            if flag in inner.recursive_flags { return true }
            cont = inner.nested
        }
        let stem = key[key.len() - 1]
        if stem in cont {
            let node = cont[stem]
            return flag in node.flags or flag in node.recursive_flags
        }
        return false
    }
}

#: The document under construction, addressed by key paths.
class _Nest {
    pub var data: dict[str, any]

    fn init(self) { self.data = {} }

    #: Walk to the table at `key`, creating empty tables along the way.
    #:
    #: With `access_lists`, a list on the path stands for its last element,
    #: which is how a key/value pair lands in the newest `[[table]]`. Raises
    #: `KeyError` when the path runs through a value that is not a table.
    pub fn get_or_create(self, key: list[str], access_lists: bool = true) -> dict[str, any] {
        var cont: any = self.data
        for part in key {
            if part not in cont { cont[part] = {} }
            cont = cont[part]
            if access_lists and isinstance(cont, list) { cont = cont[cont.len() - 1] }
            if not isinstance(cont, dict) { throw KeyError("There is no nest behind this key") }
        }
        return cont
    }

    #: Append a new empty table to the array at `key`, creating the array.
    pub fn append_nest_to_list(self, key: list[str]) -> void {
        let cont = self.get_or_create(key[0:key.len() - 1])
        let last = key[key.len() - 1]
        if last in cont {
            let existing = cont[last]
            if not isinstance(existing, list) {
                throw KeyError("An object other than list found behind this key")
            }
            existing.push({})
        } else {
            cont[last] = [{}]
        }
    }
}

#: What a date-time scan found: the field values and where the match ended.
class _Stamp {
    pub var end: int
    pub var fields: list[int]
    pub var has_time: bool
    pub var offset_minutes: int?

    fn init(self) {
        self.end = 0
        self.fields = []
        self.has_time = false
        self.offset_minutes = null
    }
}

#: One pass over a document whose `\r\n` pairs are already `\n`.
class _Parser {
    let src: str
    let length: int
    let parse_float: any
    let out: _Nest
    let flags: _Flags

    fn init(self, src: str, parse_float: any) {
        self.src = src
        self.length = src.len()
        self.parse_float = parse_float
        self.out = _Nest()
        self.flags = _Flags()
    }

    pub fn parse_document(self) -> dict[str, any] {
        var pos = 0
        var header: list[str] = []
        loop {
            pos = self._skip_ws(pos)
            if pos >= self.length { break }
            let char = self.src[pos]
            if char == "\n" {
                pos += 1
                continue
            }
            if _is_key_initial(char) {
                pos = self._key_value_rule(pos, header)
                pos = self._skip_ws(pos)
            } elif char == "[" {
                self.flags.finalize_pending()
                if pos + 1 < self.length and self.src[pos + 1] == "[" {
                    let (list_pos, list_header) = self._create_list_rule(pos)
                    pos = list_pos
                    header = list_header
                } else {
                    let (dict_pos, dict_header) = self._create_dict_rule(pos)
                    pos = dict_pos
                    header = dict_header
                }
                pos = self._skip_ws(pos)
            } elif char != "#" {
                throw self._error(pos, "Invalid statement")
            }
            pos = self._skip_comment(pos)
            if pos >= self.length { break }
            if self.src[pos] != "\n" {
                throw self._error(pos, "Expected newline or end of document after a statement")
            }
            pos += 1
        }
        return self.out.data
    }

    fn _error(self, pos: int, msg: str) -> TOMLDecodeError {
        return TOMLDecodeError(msg, self.src, pos)
    }

    fn _skip_ws(self, start: int) -> int {
        var pos = start
        while pos < self.length {
            let char = self.src[pos]
            if char != " " and char != "\t" { break }
            pos += 1
        }
        return pos
    }

    fn _skip_ws_and_newlines(self, start: int) -> int {
        var pos = start
        while pos < self.length {
            let char = self.src[pos]
            if char != " " and char != "\t" and char != "\n" { break }
            pos += 1
        }
        return pos
    }

    #: Find `expect` at or after `pos` and return where it starts.
    #:
    #: A control character before it is an error at that character, unless
    #: `allow_newline` admits `\n`. When it is missing, `error_on_eof` decides
    #: between raising and returning the end of the source; the missing
    #: terminator is reported before any bad character, as `tomllib` does.
    fn _skip_until(self, pos: int, expect: str, allow_newline: bool, error_on_eof: bool) -> int {
        var stop = self._find(expect, pos)
        if stop < 0 {
            stop = self.length
            if error_on_eof { throw self._error(stop, f"Expected {_py_repr(expect)}") }
        }
        var index = pos
        while index < stop {
            let char = self.src[index]
            if _is_control(char) and not (allow_newline and char == "\n") {
                throw self._error(index, f"Found invalid character {_py_repr(char)}")
            }
            index += 1
        }
        return stop
    }

    fn _find(self, needle: str, start: int) -> int {
        let first = needle[0]
        let width = needle.len()
        var pos = start
        while pos + width <= self.length {
            if self.src[pos] == first and (width == 1 or self.src[pos:pos + width] == needle) {
                return pos
            }
            pos += 1
        }
        return -1
    }

    fn _skip_comment(self, pos: int) -> int {
        if pos < self.length and self.src[pos] == "#" {
            return self._skip_until(pos + 1, "\n", false, false)
        }
        return pos
    }

    fn _skip_comments_and_array_ws(self, start: int) -> int {
        var pos = start
        loop {
            let before = pos
            pos = self._skip_ws_and_newlines(pos)
            pos = self._skip_comment(pos)
            if pos == before { return pos }
        }
    }

    fn _create_dict_rule(self, start: int) -> tuple[int, list[str]] {
        var pos = self._skip_ws(start + 1)
        let (key_end, key) = self._parse_key(pos)
        pos = key_end
        if self.flags.has(key, _EXPLICIT_NEST) or self.flags.has(key, _FROZEN) {
            throw self._error(pos, f"Cannot declare {_py_key(key)} twice")
        }
        self.flags.set(key, _EXPLICIT_NEST, false)
        try {
            let _ = self.out.get_or_create(key)
        } catch _missing: KeyError {
            throw self._error(pos, "Cannot overwrite a value")
        }
        if not self._at(pos, "]") {
            throw self._error(pos, "Expected ']' at the end of a table declaration")
        }
        return (pos + 1, key)
    }

    fn _create_list_rule(self, start: int) -> tuple[int, list[str]] {
        var pos = self._skip_ws(start + 2)
        let (key_end, key) = self._parse_key(pos)
        pos = key_end
        if self.flags.has(key, _FROZEN) {
            throw self._error(pos, f"Cannot mutate immutable namespace {_py_key(key)}")
        }
        self.flags.unset_all(key)
        self.flags.set(key, _EXPLICIT_NEST, false)
        try {
            self.out.append_nest_to_list(key)
        } catch _missing: KeyError {
            throw self._error(pos, "Cannot overwrite a value")
        }
        if not self._at(pos, "]]") {
            throw self._error(pos, "Expected ']]' at the end of an array declaration")
        }
        return (pos + 2, key)
    }

    fn _key_value_rule(self, start: int, header: list[str]) -> int {
        let (pos, key, value) = self._parse_key_value_pair(start)
        let key_parent = key[0:key.len() - 1]
        let key_stem = key[key.len() - 1]
        let abs_key_parent = header + key_parent
        for count in 1..key.len() {
            let cont_key = header + key[0:count]
            if self.flags.has(cont_key, _EXPLICIT_NEST) {
                throw self._error(pos, f"Cannot redefine namespace {_py_key(cont_key)}")
            }
            self.flags.add_pending(cont_key)
        }
        if self.flags.has(abs_key_parent, _FROZEN) {
            throw self._error(pos, f"Cannot mutate immutable namespace {_py_key(abs_key_parent)}")
        }
        var nest: dict[str, any] = {}
        try {
            nest = self.out.get_or_create(abs_key_parent)
        } catch _missing: KeyError {
            throw self._error(pos, "Cannot overwrite a value")
        }
        if key_stem in nest { throw self._error(pos, "Cannot overwrite a value") }
        if isinstance(value, dict) or isinstance(value, list) {
            self.flags.set(header + key, _FROZEN, true)
        }
        nest[key_stem] = value
        return pos
    }

    fn _parse_key_value_pair(self, start: int) -> tuple[int, list[str], any] {
        let (key_end, key) = self._parse_key(start)
        var pos = key_end
        if not self._at(pos, "=") {
            throw self._error(pos, "Expected '=' after a key in a key/value pair")
        }
        pos = self._skip_ws(pos + 1)
        let (value_end, value) = self._parse_value(pos)
        return (value_end, key, value)
    }

    fn _parse_key(self, start: int) -> tuple[int, list[str]] {
        let (first_end, first) = self._parse_key_part(start)
        let key = [first]
        var pos = self._skip_ws(first_end)
        loop {
            if not self._at(pos, ".") { return (pos, key) }
            pos = self._skip_ws(pos + 1)
            let (part_end, part) = self._parse_key_part(pos)
            key.push(part)
            pos = self._skip_ws(part_end)
        }
    }

    fn _parse_key_part(self, pos: int) -> tuple[int, str] {
        if pos < self.length {
            let char = self.src[pos]
            if _is_bare_key_char(char) {
                var stop = pos
                while stop < self.length and _is_bare_key_char(self.src[stop]) { stop += 1 }
                return (stop, self.src[pos:stop])
            }
            if char == "'" { return self._parse_literal_str(pos) }
            if char == "\"" { return self._parse_basic_str(pos + 1, false) }
        }
        throw self._error(pos, "Invalid initial character for a key part")
    }

    fn _parse_array(self, start: int) -> tuple[int, list[any]] {
        let array: list[any] = []
        var pos = self._skip_comments_and_array_ws(start + 1)
        if self._at(pos, "]") { return (pos + 1, array) }
        loop {
            let (value_end, value) = self._parse_value(pos)
            array.push(value)
            pos = self._skip_comments_and_array_ws(value_end)
            if self._at(pos, "]") { return (pos + 1, array) }
            if not self._at(pos, ",") { throw self._error(pos, "Unclosed array") }
            pos = self._skip_comments_and_array_ws(pos + 1)
            if self._at(pos, "]") { return (pos + 1, array) }
        }
    }

    fn _parse_inline_table(self, start: int) -> tuple[int, dict[str, any]] {
        let nested = _Nest()
        let flags = _Flags()
        var pos = self._skip_ws(start + 1)
        if self._at(pos, "}") { return (pos + 1, nested.data) }
        loop {
            let (pair_end, key, value) = self._parse_key_value_pair(pos)
            pos = pair_end
            let key_parent = key[0:key.len() - 1]
            let key_stem = key[key.len() - 1]
            if flags.has(key, _FROZEN) {
                throw self._error(pos, f"Cannot mutate immutable namespace {_py_key(key)}")
            }
            var nest: dict[str, any] = {}
            try {
                nest = nested.get_or_create(key_parent, false)
            } catch _missing: KeyError {
                throw self._error(pos, "Cannot overwrite a value")
            }
            if key_stem in nest {
                throw self._error(pos, f"Duplicate inline table key {_py_repr(key_stem)}")
            }
            nest[key_stem] = value
            pos = self._skip_ws(pos)
            if self._at(pos, "}") { return (pos + 1, nested.data) }
            if not self._at(pos, ",") { throw self._error(pos, "Unclosed inline table") }
            if isinstance(value, dict) or isinstance(value, list) {
                flags.set(key, _FROZEN, true)
            }
            pos = self._skip_ws(pos + 1)
        }
    }

    fn _parse_basic_str_escape(self, start: int, multiline: bool) -> tuple[int, str] {
        let escape_id = self.src[start:_min(start + 2, self.length)]
        var pos = start + 2
        if multiline and (escape_id == "\\ " or escape_id == "\\\t" or escape_id == "\\\n") {
            if escape_id != "\\\n" {
                pos = self._skip_ws(pos)
                if pos >= self.length { return (pos, "") }
                if self.src[pos] != "\n" { throw self._error(pos, "Unescaped '\\' in a string") }
                pos += 1
            }
            return (self._skip_ws_and_newlines(pos), "")
        }
        if escape_id == "\\u" { return self._parse_hex_char(pos, 4) }
        if escape_id == "\\U" { return self._parse_hex_char(pos, 8) }
        let replacement = _ESCAPE_REPLACEMENTS.get(escape_id, "")
        if replacement.len() == 0 { throw self._error(pos, "Unescaped '\\' in a string") }
        return (pos, replacement)
    }

    fn _parse_hex_char(self, pos: int, hex_len: int) -> tuple[int, str] {
        let hex_str = self.src[pos:_min(pos + hex_len, self.length)]
        if hex_str.len() != hex_len or not _all_hex(hex_str) {
            throw self._error(pos, "Invalid hex value")
        }
        let code = to_int(hex_str, 16) ?? -1
        if not _is_unicode_scalar_value(code) {
            throw self._error(pos + hex_len, "Escaped character is not a Unicode scalar value")
        }
        return (pos + hex_len, chr(code))
    }

    fn _parse_literal_str(self, start: int) -> tuple[int, str] {
        let pos = start + 1
        let stop = self._skip_until(pos, "'", false, true)
        return (stop + 1, self.src[pos:stop])
    }

    fn _parse_multiline_str(self, start: int, literal: bool) -> tuple[int, str] {
        var pos = start + 3
        if self._at(pos, "\n") { pos += 1 }
        var result = ""
        var delim = ""
        if literal {
            delim = "'"
            let stop = self._skip_until(pos, "'''", true, true)
            result = self.src[pos:stop]
            pos = stop + 3
        } else {
            delim = "\""
            let (body_end, body) = self._parse_basic_str(pos, true)
            pos = body_end
            result = body
        }
        if not self._at(pos, delim) { return (pos, result) }
        pos += 1
        if not self._at(pos, delim) { return (pos, result + delim) }
        return (pos + 1, result + delim + delim)
    }

    fn _parse_basic_str(self, start: int, multiline: bool) -> tuple[int, str] {
        let result = StringBuilder()
        var pos = start
        var run_start = start
        loop {
            if pos >= self.length { throw self._error(pos, "Unterminated string") }
            let char = self.src[pos]
            if char == "\"" {
                if not multiline {
                    result.push(self.src[run_start:pos])
                    return (pos + 1, result.build())
                }
                if self._at(pos, "\"\"\"") {
                    result.push(self.src[run_start:pos])
                    return (pos + 3, result.build())
                }
                pos += 1
                continue
            }
            if char == "\\" {
                result.push(self.src[run_start:pos])
                let (escape_end, text) = self._parse_basic_str_escape(pos, multiline)
                result.push(text)
                pos = escape_end
                run_start = pos
                continue
            }
            if _is_control(char) and not (multiline and char == "\n") {
                throw self._error(pos, f"Illegal character {_py_repr(char)}")
            }
            pos += 1
        }
    }

    fn _parse_value(self, pos: int) -> tuple[int, any] {
        let char = pos < self.length ? self.src[pos] : ""
        if char == "\"" {
            if self._at(pos, "\"\"\"") { return self._parse_multiline_str(pos, false) }
            return self._parse_basic_str(pos + 1, false)
        }
        if char == "'" {
            if self._at(pos, "'''") { return self._parse_multiline_str(pos, true) }
            return self._parse_literal_str(pos)
        }
        if char == "t" and self._at(pos, "true") { return (pos + 4, true) }
        if char == "f" and self._at(pos, "false") { return (pos + 5, false) }
        if char == "[" { return self._parse_array(pos) }
        if char == "{" { return self._parse_inline_table(pos) }
        let stamp = self._scan_datetime(pos)
        if stamp is not null {
            try {
                return (stamp.end, _stamp_value(stamp))
            } catch _invalid: ValueError {
                throw self._error(pos, "Invalid date or datetime")
            }
        }
        let clock = self._scan_time(pos)
        if clock is not null {
            let fields = clock.fields
            return (clock.end, Time(fields[0], fields[1], fields[2], fields[3]))
        }
        let number_end = self._scan_number(pos)
        if number_end > pos { return (number_end, self._number_value(pos, number_end)) }
        let first_three = self.src[pos:_min(pos + 3, self.length)]
        if first_three == "inf" or first_three == "nan" {
            return (pos + 3, self._special_float(first_three))
        }
        let first_four = self.src[pos:_min(pos + 4, self.length)]
        if first_four == "-inf" or
            first_four == "+inf" or
            first_four == "-nan" or
            first_four == "+nan" {
            return (pos + 4, self._special_float(first_four))
        }
        throw self._error(pos, "Invalid value")
    }

    fn _special_float(self, text: str) -> any {
        if self.parse_float is not null { return self._safe_parse_float(text) }
        if text.ends_with("nan") { return math.NAN }
        return text[0] == "-" ? -math.INF : math.INF
    }

    fn _safe_parse_float(self, text: str) -> any {
        let value = self.parse_float(text)
        if isinstance(value, dict) or isinstance(value, list) {
            throw ValueError("parse_float must not return dicts or lists")
        }
        return value
    }

    #: Match `tomllib`'s number pattern at `pos` and return where it ends, or
    #: `pos` itself when nothing matches.
    #:
    #: A prefixed form (`0x`, `0o`, `0b`) needs at least one digit after the
    #: prefix; otherwise the plain `0` matches and the letter is left behind,
    #: exactly as the regex alternation falls through.
    fn _scan_number(self, pos: int) -> int {
        if pos >= self.length { return pos }
        let char = self.src[pos]
        if char == "0" and pos + 1 < self.length {
            let marker = self.src[pos + 1]
            var stop = -1
            if marker == "x" {
                stop = self._scan_digits(pos + 2, 16)
            } elif marker == "b" {
                stop = self._scan_digits(pos + 2, 2)
            } elif marker == "o" {
                stop = self._scan_digits(pos + 2, 8)
            }
            if stop > pos + 2 { return stop }
        }
        var index = pos
        if char == "+" or char == "-" { index += 1 }
        if index >= self.length { return pos }
        let first = self.src[index]
        if first == "0" {
            index += 1
        } elif _is_digit(first) {
            index = self._scan_digits(index, 10)
        } else {
            return pos
        }
        if self._at(index, ".") and index + 1 < self.length and _is_digit(self.src[index + 1]) {
            index = self._scan_digits(index + 1, 10)
        }
        if index < self.length and (self.src[index] == "e" or self.src[index] == "E") {
            var exponent = index + 1
            if exponent < self.length and (self.src[exponent] == "+" or self.src[exponent] == "-") {
                exponent += 1
            }
            if exponent < self.length and _is_digit(self.src[exponent]) {
                index = self._scan_digits(exponent, 10)
            }
        }
        return index
    }

    #: Consume `digit(_?digit)*` in `base` from `start`; `start` when none.
    fn _scan_digits(self, start: int, base: int) -> int {
        if start >= self.length or _digit_value(self.src[start]) >= base { return start }
        var pos = start + 1
        loop {
            if pos < self.length and _digit_value(self.src[pos]) < base {
                pos += 1
            } elif pos + 1 < self.length and
                self.src[pos] == "_" and
                _digit_value(self.src[pos + 1]) < base {
                pos += 2
            } else {
                return pos
            }
        }
    }

    fn _number_value(self, start: int, stop: int) -> any {
        let text = self.src[start:stop]
        if text.len() > 2 and text[0] == "0" {
            let marker = text[1]
            var base = 0
            if marker == "x" {
                base = 16
            } elif marker == "o" {
                base = 8
            } elif marker == "b" {
                base = 2
            }
            if base > 0 {
                let value = to_int(text[2:].replace("_", ""), base)
                if value is null { throw self._error(start, _OUT_OF_RANGE) }
                return value
            }
        }
        if text.contains(".") or text.contains("e") or text.contains("E") {
            if self.parse_float is not null { return self._safe_parse_float(text) }
            return float(text.replace("_", ""))
        }
        let value = to_int(text.replace("_", ""))
        if value is null { throw self._error(start, _OUT_OF_RANGE) }
        return value
    }

    #: Match a date, optionally followed by a time and an offset.
    #:
    #: The optional parts follow the regex: a separator with no valid time
    #: after it leaves just the date matched, and a malformed offset leaves a
    #: local date-time.
    fn _scan_datetime(self, pos: int) -> _Stamp? {
        if not self._digits_at(pos, 4) { return null }
        if not self._at(pos + 4, "-") { return null }
        let month = self._two_digits(pos + 5, 1, 12)
        if month < 0 or not self._at(pos + 7, "-") { return null }
        let day = self._two_digits(pos + 8, 1, 31)
        if day < 0 { return null }
        let stamp = _Stamp()
        stamp.end = pos + 10
        stamp.fields = [to_int(self.src[pos:pos + 4]) ?? 0, month, day]
        let separator = stamp.end < self.length ? self.src[stamp.end] : ""
        if separator != "T" and separator != "t" and separator != " " { return stamp }
        let clock = self._scan_time(stamp.end + 1)
        if clock is null { return stamp }
        stamp.has_time = true
        stamp.fields = stamp.fields + clock.fields
        stamp.end = clock.end
        let zone = stamp.end < self.length ? self.src[stamp.end] : ""
        if zone == "Z" or zone == "z" {
            stamp.offset_minutes = 0
            stamp.end += 1
            return stamp
        }
        if zone == "+" or zone == "-" {
            let hours = self._two_digits(stamp.end + 1, 0, 23)
            if hours < 0 or not self._at(stamp.end + 3, ":") { return stamp }
            let minutes = self._two_digits(stamp.end + 4, 0, 59)
            if minutes < 0 { return stamp }
            let magnitude = hours * 60 + minutes
            stamp.offset_minutes = zone == "-" ? -magnitude : magnitude
            stamp.end += 6
        }
        return stamp
    }

    #: Match `HH:MM:SS` with an optional fraction; the first six fractional
    #: digits are the microseconds and the rest are consumed and dropped.
    fn _scan_time(self, pos: int) -> _Stamp? {
        let hour = self._two_digits(pos, 0, 23)
        if hour < 0 or not self._at(pos + 2, ":") { return null }
        let minute = self._two_digits(pos + 3, 0, 59)
        if minute < 0 or not self._at(pos + 5, ":") { return null }
        let second = self._two_digits(pos + 6, 0, 59)
        if second < 0 { return null }
        let stamp = _Stamp()
        stamp.end = pos + 8
        var micros = 0
        if self._at(stamp.end, ".") and
            stamp.end + 1 < self.length and
            _is_digit(self.src[stamp.end + 1]) {
            var index = stamp.end + 1
            var digits = ""
            while index < self.length and _is_digit(self.src[index]) {
                if digits.len() < 6 { digits = digits + self.src[index] }
                index += 1
            }
            while digits.len() < 6 { digits = digits + "0" }
            micros = to_int(digits) ?? 0
            stamp.end = index
        }
        stamp.fields = [hour, minute, second, micros]
        return stamp
    }

    fn _digits_at(self, pos: int, count: int) -> bool {
        if pos + count > self.length { return false }
        for index in pos..pos + count {
            if not _is_digit(self.src[index]) { return false }
        }
        return true
    }

    #: Two digits at `pos` as a number within `low..=high`, else -1.
    fn _two_digits(self, pos: int, low: int, high: int) -> int {
        if not self._digits_at(pos, 2) { return -1 }
        let value = (ord(self.src[pos]) - 0x30) * 10 + ord(self.src[pos + 1]) - 0x30
        if value < low or value > high { return -1 }
        return value
    }

    fn _at(self, pos: int, text: str) -> bool {
        let stop = pos + text.len()
        if stop > self.length { return false }
        if text.len() == 1 { return self.src[pos] == text }
        return self.src[pos:stop] == text
    }
}

const _OUT_OF_RANGE = "Integer is outside the 64-bit range this runtime can hold"

let _ESCAPE_REPLACEMENTS: dict[str, str] = {
    "\\b": "\x08",
    "\\t": "\t",
    "\\n": "\n",
    "\\f": "\x0C",
    "\\r": "\r",
    "\\\"": "\"",
    "\\\\": "\\",
}

fn _stamp_value(stamp: _Stamp) -> any {
    let fields = stamp.fields
    if not stamp.has_time { return Date(fields[0], fields[1], fields[2]) }
    return DateTime(
        fields[0],
        fields[1],
        fields[2],
        fields[3],
        fields[4],
        fields[5],
        fields[6],
        stamp.offset_minutes
    )
}

fn _is_digit(char: str) -> bool {
    let code = ord(char)
    return code >= 0x30 and code <= 0x39
}

fn _digit_value(char: str) -> int {
    let code = ord(char)
    if code >= 0x30 and code <= 0x39 { return code - 0x30 }
    if code >= 0x61 and code <= 0x66 { return code - 0x61 + 10 }
    if code >= 0x41 and code <= 0x46 { return code - 0x41 + 10 }
    return 99
}

fn _all_hex(text: str) -> bool {
    for char in text {
        if _digit_value(char) >= 16 { return false }
    }
    return true
}

fn _is_bare_key_char(char: str) -> bool {
    let code = ord(char)
    if code >= 0x61 and code <= 0x7A { return true }
    if code >= 0x41 and code <= 0x5A { return true }
    if code >= 0x30 and code <= 0x39 { return true }
    return char == "-" or char == "_"
}

fn _is_key_initial(char: str) -> bool {
    return _is_bare_key_char(char) or char == "\"" or char == "'"
}

#: ASCII control characters other than tab, which TOML admits nowhere else.
fn _is_control(char: str) -> bool {
    let code = ord(char)
    return code < 0x20 and code != 0x09 or code == 0x7F
}

fn _is_unicode_scalar_value(code: int) -> bool {
    return code >= 0 and code <= 0xD7FF or code >= 0xE000 and code <= 0x10FFFF
}

fn _min(a: int, b: int) -> int { return a < b ? a : b }

#: A key path as Python prints a tuple of strings, which is how `tomllib`
#: names a namespace in its messages: `('a', 'b')`, and `('a',)` for one part.
fn _py_key(key: list[str]) -> str {
    let parts = [_py_repr(part) for part in key]
    if parts.len() == 1 { return "(" + parts[0] + ",)" }
    return "(" + ", ".join(parts) + ")"
}

#: `repr()` of a string as Python writes it, for messages that quote one.
fn _py_repr(text: str) -> str {
    let quote = text.contains("'") and not text.contains("\"") ? "\"" : "'"
    let out = StringBuilder()
    out.push(quote)
    for char in text {
        let code = ord(char)
        if char == "\\" {
            out.push("\\\\")
        } elif char == quote {
            out.push("\\" + quote)
        } elif char == "\n" {
            out.push("\\n")
        } elif char == "\r" {
            out.push("\\r")
        } elif char == "\t" {
            out.push("\\t")
        } elif code < 0x20 or code == 0x7F {
            out.push("\\x" + _hex2(code))
        } else {
            out.push(char)
        }
    }
    out.push(quote)
    return out.build()
}

fn _hex2(code: int) -> str {
    let digits = "0123456789abcdef"
    return digits[code >> 4 & 0xF] + digits[code & 0xF]
}

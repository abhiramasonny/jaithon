#: Positions in the source text and the error classes every stage raises.
#:
#: `Mark` is PyYAML's `yaml.Mark`: a zero-based `line` and `column` plus the
#: character `index`, and it renders itself the way PyYAML does, with the
#: offending line and a caret under the position. `MarkedYAMLError` carries
#: up to two of them -- where the problem is, and the context it sits in --
#: and its message is PyYAML's multi-line description of both, so a program
#: that prints the error sees the same text it would have seen from Python.
#:
#: ```
#: try {
#:     safe_load("a: [1, 2")
#: } catch error: YAMLError {
#:     print(error.message)
#: }
#: ```
#:
#: One shape is deliberately different from PyYAML. `Error.init` is called
#: directly instead of through `super`, because this language resolves
#: `super` from the object's own class rather than from the class the method
#: was written in, so a grandchild such as `ScannerError` would otherwise call
#: `MarkedYAMLError.init` again from inside itself.
from std.str import repeat

let LINE_ENDS = "\u{0}\r\n\u{85}\u{2028}\u{2029}"

#: A position in the source: the character `index`, and the zero-based `line`
#: and `column`, as PyYAML counts them.
pub class Mark {
    #: The name PyYAML gives a string source: `<unicode string>`.
    pub let name: str

    #: Characters from the start of the text.
    pub let index: int

    #: Zero-based line.
    pub let line: int

    #: Zero-based column, in characters.
    pub let column: int

    let buffer: str?
    let pointer: int

    #: Record a position; `buffer` and `pointer` let `get_snippet` show it.
    pub fn init(self, name: str, index: int, line: int, column: int, buffer: str?, pointer: int) {
        self.name = name
        self.index = index
        self.line = line
        self.column = column
        self.buffer = buffer
        self.pointer = pointer
    }

    #: The source line around the mark with a caret under it, or null when no
    #: buffer was kept.
    pub fn get_snippet(self, indent: int = 4, max_length: int = 75) -> str? {
        if self.buffer is null { return null }
        let buffer = self.buffer ?? ""
        let half = float(max_length) / 2.0 - 1.0
        var head = ""
        var start = self.pointer
        while start > 0 and buffer[start - 1] not in LINE_ENDS {
            start -= 1
            if float(self.pointer - start) > half {
                head = " ... "
                start += 5
                break
            }
        }
        var tail = ""
        var end = self.pointer
        while end < buffer.len() and buffer[end] not in LINE_ENDS {
            end += 1
            if float(end - self.pointer) > half {
                tail = " ... "
                end -= 5
                break
            }
        }
        let snippet = buffer[start:end]
        return repeat(" ", indent) +
            head +
            snippet +
            tail +
            "\n" +
            repeat(" ", indent + self.pointer - start + head.len()) +
            "^"
    }

    #: PyYAML's rendering: `  in "<name>", line L, column C`, one-based, and
    #: the snippet beneath it when there is one.
    pub fn to_str(self) -> str {
        var where = f"  in \"{self.name}\", line {self.line + 1}, column {self.column + 1}"
        let snippet = self.get_snippet()
        if snippet is not null { where = where + ":\n" + (snippet ?? "") }
        return where
    }

    fn __str__(self) -> str { return self.to_str() }

    fn __repr__(self) -> str { return f"Mark({self.line}:{self.column})" }
}

#: The root of every error this package raises.
pub class YAMLError extends Error {
    #: Raise with `message`.
    pub fn init(self, message: str) { Error.init(self, message) }
}

#: An error tied to one or two positions in the source.
#:
#: `problem` says what was found and `problem_mark` where; `context` and
#: `context_mark` say what was being read at the time. Any of them may be
#: null. `line` and `column` are the one-based numbers the message shows,
#: taken from the problem mark or, failing that, the context mark, and are
#: zero when the error has no position at all.
pub class MarkedYAMLError extends YAMLError {
    #: What was being read when the problem appeared.
    pub let context: str?

    #: Where that reading started.
    pub let context_mark: Mark?

    #: What went wrong.
    pub let problem: str?

    #: Where it went wrong.
    pub let problem_mark: Mark?

    #: A hint for the reader, when PyYAML gives one.
    pub let note: str?

    #: One-based line of the problem, or zero without a mark.
    pub let line: int

    #: One-based column of the problem, or zero without a mark.
    pub let column: int

    #: Raise with PyYAML's four-part description.
    pub fn init(
        self,
        context: str? = null,
        context_mark: Mark? = null,
        problem: str? = null,
        problem_mark: Mark? = null,
        note: str? = null
    ) {
        self.context = context
        self.context_mark = context_mark
        self.problem = problem
        self.problem_mark = problem_mark
        self.note = note
        let mark = problem_mark ?? context_mark
        var line = 0
        var column = 0
        if mark is not null {
            let shown = mark ?? Mark("", 0, 0, 0, null, 0)
            line = shown.line + 1
            column = shown.column + 1
        }
        self.line = line
        self.column = column
        Error.init(self, describe_marked(context, context_mark, problem, problem_mark, note))
    }

    #: The mark the error is reported at: the problem's, else the context's.
    pub fn mark(self) -> Mark? { return self.problem_mark ?? self.context_mark }
}

fn describe_marked(
    context: str?,
    context_mark: Mark?,
    problem: str?,
    problem_mark: Mark?,
    note: str?
) -> str {
    let lines: list[str] = []
    if context is not null { lines.push(context ?? "") }
    if context_mark is not null {
        let shown = context_mark ?? Mark("", 0, 0, 0, null, 0)
        var distinct = problem is null or problem_mark is null
        if not distinct {
            let other = problem_mark ?? shown
            distinct = shown.name != other.name or
                shown.line != other.line or
                shown.column != other.column
        }
        if distinct { lines.push(shown.to_str()) }
    }
    if problem is not null { lines.push(problem ?? "") }
    if problem_mark is not null {
        lines.push((problem_mark ?? Mark("", 0, 0, 0, null, 0)).to_str())
    }
    if note is not null { lines.push(note ?? "") }
    return "\n".join(lines)
}

#: A character the reader refuses: anything outside YAML's printable set.
pub class ReaderError extends YAMLError {
    #: The source name.
    pub let name: str

    #: Character offset of the offending character.
    pub let position: int

    #: Its code point.
    pub let character: int

    #: Why it was refused.
    pub let reason: str

    #: One-based line of the character.
    pub let line: int

    #: One-based column of the character.
    pub let column: int

    #: Raise for `character` at `position`; `line` and `column` are one-based.
    pub fn init(
        self,
        name: str,
        position: int,
        character: int,
        reason: str,
        line: int,
        column: int
    ) {
        self.name = name
        self.position = position
        self.character = character
        self.reason = reason
        self.line = line
        self.column = column
        Error.init(
            self,
            f"unacceptable character #x{hex_lower(character, 4)}: {reason}\n" +
                f"  in \"{name}\", position {position}"
        )
    }
}

#: A token could not be scanned.
pub class ScannerError extends MarkedYAMLError { }

#: The tokens do not form a document.
pub class ParserError extends MarkedYAMLError { }

#: An alias has no anchor, or an anchor is defined twice.
pub class ComposerError extends MarkedYAMLError { }

#: A node cannot become a value: an unknown tag, an unhashable key, a bad
#: merge.
pub class ConstructorError extends MarkedYAMLError { }

#: A value has no YAML representation.
pub class RepresenterError extends YAMLError { }

#: The serializer was driven out of order.
pub class SerializerError extends YAMLError { }

#: The emitter received events it cannot write.
pub class EmitterError extends YAMLError { }

#: Lower-case hexadecimal, zero-padded to at least `width` digits.
pub fn hex_lower(value: int, width: int) -> str {
    let digits = "0123456789abcdef"
    var out = ""
    var rest = value
    while rest > 0 {
        out = digits[rest & 0xF] + out
        rest = rest >> 4
    }
    while out.len() < width { out = "0" + out }
    return out
}

#: Upper-case hexadecimal, zero-padded to at least `width` digits.
pub fn hex_upper(value: int, width: int) -> str { return hex_lower(value, width).upper() }

#: Python's `repr` of a string, which PyYAML's messages quote characters with.
#:
#: Single quotes unless the text holds a single quote and no double quote;
#: backslash, the quote and the C escapes are escaped by letter; other
#: control and non-printable characters as `\xNN`, `\uNNNN` or `\UNNNNNNNN`.
#: Printability follows Python only for the ranges YAML itself distinguishes,
#: which is all the scanner ever quotes.
pub fn python_repr(text: str) -> str {
    var quote = "'"
    if "'" in text and "\"" not in text { quote = "\"" }
    var out = quote
    for ch in text {
        let code = ord(ch)
        if ch == quote or ch == "\\" {
            out = out + "\\" + ch
        } elif ch == "\n" {
            out = out + "\\n"
        } elif ch == "\r" {
            out = out + "\\r"
        } elif ch == "\t" {
            out = out + "\\t"
        } elif code < 0x20 or code == 0x7F {
            out = out + "\\x" + hex_lower(code, 2)
        } elif code >= 0x80 and code <= 0xA0 {
            out = out + "\\x" + hex_lower(code, 2)
        } elif code == 0xAD or code == 0x2028 or code == 0x2029 or code == 0xFEFF {
            out = out + "\\u" + hex_lower(code, 4)
        } elif code >= 0xD800 and code <= 0xDFFF {
            out = out + "\\u" + hex_lower(code, 4)
        } elif code >= 0xE000 and code <= 0xF8FF {
            out = out + "\\u" + hex_lower(code, 4)
        } elif code == 0xFFFE or code == 0xFFFF {
            out = out + "\\u" + hex_lower(code, 4)
        } else {
            out = out + ch
        }
    }
    return out + quote
}

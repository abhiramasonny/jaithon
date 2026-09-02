#: The character cursor the scanner reads through.
#:
#: The whole text is held in memory with a NUL appended, as PyYAML's reader
#: holds a string source, so `peek` past the end returns `\0` rather than
#: raising and every scanner check can treat NUL as "end of input". The
#: reader also rejects the characters YAML does not allow anywhere -- most
#: control characters, surrogates, the two non-characters -- before the
#: scanner sees them, with PyYAML's `ReaderError`.
from .error import Mark, ReaderError

#: The name PyYAML gives a string source.
pub const STRING_NAME: str = "<unicode string>"

let NUL = "\u{0}"

#: Whether `code` is a character YAML permits in a stream.
pub fn is_printable(code: int) -> bool {
    if code == 0x09 or code == 0x0A or code == 0x0D { return true }
    if code >= 0x20 and code <= 0x7E { return true }
    if code == 0x85 { return true }
    if code >= 0xA0 and code <= 0xD7FF { return true }
    if code >= 0xE000 and code <= 0xFFFD { return true }
    return code >= 0x10000 and code <= 0x10FFFF
}

#: A cursor over one text, tracking the character index, line and column.
pub class Reader {
    #: The source name.
    pub let name: str

    #: The text with a trailing NUL.
    pub let buffer: str

    #: Characters consumed so far.
    pub var index: int

    #: Zero-based line of the cursor.
    pub var line: int

    #: Zero-based column of the cursor.
    pub var column: int

    let length: int

    #: Wrap `text`, raising `ReaderError` on a character YAML forbids.
    pub fn init(self, text: str, name: str = STRING_NAME) {
        self.name = name
        self.buffer = text + NUL
        self.length = text.len()
        self.index = 0
        self.line = 0
        self.column = 0
        check_printable(text, name)
    }

    #: The character `offset` places ahead, or `\0` past the end.
    pub fn peek(self, offset: int = 0) -> str {
        let at = self.index + offset
        if at >= self.length { return NUL }
        return self.buffer[at]
    }

    #: The next `length` characters, fewer at the end of the text.
    pub fn prefix(self, length: int = 1) -> str {
        var stop = self.index + length
        if stop > self.length { stop = self.length }
        return self.buffer[self.index:stop]
    }

    #: Consume `length` characters, keeping the line and column current.
    #:
    #: A `\r\n` pair counts as one break; a byte order mark advances no
    #: column, as in PyYAML.
    pub fn forward(self, length: int = 1) -> void {
        var remaining = length
        while remaining > 0 {
            if self.index >= self.length { return }
            let ch = self.buffer[self.index]
            self.index += 1
            if ch == "\n" or
                ch == "\u{85}" or
                ch == "\u{2028}" or
                ch == "\u{2029}" or
                ch == "\r" and self.buffer[self.index] != "\n" {
                self.line += 1
                self.column = 0
            } elif ch != "\u{FEFF}" {
                self.column += 1
            }
            remaining -= 1
        }
    }

    #: The cursor's position as a `Mark` that can show its line.
    pub fn get_mark(self) -> Mark {
        return Mark(self.name, self.index, self.line, self.column, self.buffer, self.index)
    }
}

fn check_printable(text: str, name: str) -> void {
    var position = 0
    var line = 0
    var column = 0
    for ch in text {
        let code = ord(ch)
        if not is_printable(code) {
            throw ReaderError(
                name,
                position,
                code,
                "special characters are not allowed",
                line + 1,
                column + 1
            )
        }
        position += 1
        if ch == "\n" {
            line += 1
            column = 0
        } else {
            column += 1
        }
    }
}

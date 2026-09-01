#: `Span`, the half-open range of source text that every node and every
#: sub-structure carries, with the two UTF-8 helpers only it uses.
#:
#: It imports nothing, and that is what earns it a file: `Span` is the piece of
#: this directory that the lexer, the token stream and the diagnostics all want
#: without wanting a tree, and `jaithon.compile.token`, `lexer` and `diag`
#: already import it on its own through the facade.
#:
#: What is NOT here: line and column arithmetic beyond the scalar counting
#: `line_column` needs. Byte offsets go in and 1-based lines and columns come
#: out; laying a caret under one belongs to whoever prints the diagnostic.

#: A half-open range of source text, as the C lexer records it.
#:
#: `start` and `end` are UTF-8 **byte** offsets into the file, so they can be
#: compared with the C compiler's own diagnostics. Lines and columns are 1-based
#: and count Unicode scalar values, which is what §1.1 of the language spec
#: requires and what a caret under an emoji needs.
#:
#: ```
#: Span(0, 4, 5).line_column("let x = 1")   # (1, 5)
#: Span(0, 4, 5).text("let x = 1")          # "x"
#: ```
pub class Span {
    #: Index of the source file in the compilation's file table, or -1.
    pub let file: int
    #: Byte offset of the first character of the span.
    pub let start: int
    #: Byte offset one past the last character of the span.
    pub let end: int

    #: Create a span over `[start, end)` of the file numbered `file`.
    pub fn init(self, file: int, start: int, end: int) {
        if end < start { throw ValueError(f"span end {end} is before its start {start}") }
        self.file = file
        self.start = start
        self.end = end
    }

    #: The span of a synthesised node: file -1, an empty range at offset 0.
    pub static fn none() -> Span {
        return Span(-1, 0, 0)
    }

    #: Whether this span points into a real file.
    pub fn is_known(self) -> bool {
        return self.file >= 0
    }

    #: The number of bytes the span covers.
    pub fn length(self) -> int {
        return self.end - self.start
    }

    #: The smallest span covering both `self` and `other`.
    #:
    #: Mirrors `jaiSpanJoin`: a synthesised span contributes nothing, so joining
    #: a real span with `Span.none()` gives the real one back.
    pub fn join(self, other: Span) -> Span {
        if not other.is_known() { return self }
        if not self.is_known() { return other }
        return Span(self.file, min(self.start, other.start), max(self.end, other.end))
    }

    #: The slice of `source` this span covers.
    pub fn text(self, source: str) -> str {
        return source[_char_index(source, self.start):_char_index(source, self.end)]
    }

    #: The 1-based line and column of `start` in `source`.
    pub fn line_column(self, source: str) -> tuple[int, int] {
        var offset = 0
        var line = 1
        var column = 1
        for character in source {
            if offset >= self.start { return (line, column) }
            offset += _utf8_length(character)
            if character == "\n" {
                line += 1
                column = 1
            } else {
                column += 1
            }
        }
        return (line, column)
    }

    #: The 1-based line of `start` in `source`.
    pub fn line(self, source: str) -> int {
        let (line, _column) = self.line_column(source)
        return line
    }

    #: The 1-based column of `start` in `source`.
    pub fn column(self, source: str) -> int {
        let (_line, column) = self.line_column(source)
        return column
    }

    #: The whole source line containing `start`, without its terminator.
    #:
    #: This is the line a diagnostic prints above its caret.
    pub fn line_text(self, source: str) -> str {
        let lines = source.split("\n")
        let index = self.line(source) - 1
        if index < 0 or index >= lines.len() { return "" }
        return lines[index]
    }

    fn __eq__(self, other: Span) -> bool {
        return self.file == other.file and self.start == other.start and self.end == other.end
    }

    fn __hash__(self) -> int {
        return hash((self.file, self.start, self.end))
    }

    fn __str__(self) -> str {
        return f"Span({self.file}, {self.start}, {self.end})"
    }
}

fn _utf8_length(character: str) -> int {
    let code = ord(character)
    if code < 0x80 { return 1 }
    if code < 0x800 { return 2 }
    if code < 0x10000 { return 3 }
    return 4
}

fn _char_index(source: str, offset: int) -> int {
    var byte = 0
    var index = 0
    for character in source {
        if byte >= offset { return index }
        byte += _utf8_length(character)
        index += 1
    }
    return index
}

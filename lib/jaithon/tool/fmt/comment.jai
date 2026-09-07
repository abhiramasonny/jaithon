#: The comment scanner, and the bookkeeping that places every comment again.
from std.str import StringBuilder
from .text import _dedent_continuations, _skip_string

#: One comment, with everything the printer needs to place it again.
pub class Comment {
    #: The comment as written, including its `#`, `#:` or `#*` opener.
    pub let text: str

    #: The 1-based line the comment starts on.
    pub let line: int

    #: The 1-based line the comment ends on, which differs from `line` only for
    #: a block comment.
    pub let end_line: int

    #: The 1-based column of the opening `#`.
    pub let column: int

    #: True when nothing but whitespace precedes the comment on its line.
    pub let own_line: bool

    #: True when the comment opens with `#:`, making it a doc comment.
    pub let is_doc: bool

    #: Record a comment running from `line` to `end_line`, opening at `column`.
    pub fn init(self, text: str, line: int, end_line: int, column: int, own_line: bool) {
        self.text = text
        self.line = line
        self.end_line = end_line
        self.column = column
        self.own_line = own_line
        self.is_doc = text.starts_with("#:")
    }

    #: Return the comment without its opener and surrounding space.
    #:
    #: ```
    #: Comment("#: Return the mean.", 1, 1, 1, true).body()    # "Return the mean."
    #: Comment("# why", 1, 1, 12, false).body()                # "why"
    #: ```
    pub fn body(self) -> str {
        if self.text.starts_with("#*") {
            let inner = self.text[2:self.text.len()]
            let closed = if inner.ends_with("*#") { inner[0:inner.len() - 2] } else { inner }
            return closed.strip()
        }
        let opener = if self.is_doc { 2 } else { 1 }
        return self.text[opener:self.text.len()].strip()
    }

    fn __str__(self) -> str { return f"Comment({self.line}, {repr(self.text)})" }
}

#: Return every comment in `source`, in source order.
#:
#: Aware of strings (a `#` inside one is text, not a comment) and of nested
#: `#* ... *#`; block-comment continuation lines are dedented to the comment's
#: own column.
#:
#: ```
#: scan_comments("let x = 1  # why\n")[0].body()    # "why"
#: scan_comments("\"# not a comment\"\n").len()     # 0
#: scan_comments("#: doc\n")[0].is_doc              # true
#: ```
pub fn scan_comments(source: str) -> list[Comment] {
    let chars = [character for character in source]
    let length = chars.len()
    let found: list[Comment] = []
    var index = 0
    var line = 1
    var column = 1
    var code_seen = false

    while index < length {
        let character = chars[index]
        if character == "\n" {
            line += 1
            column = 1
            code_seen = false
            index += 1
        } elif character == "#" {
            let start_line = line
            let start_column = column
            let piece = StringBuilder()
            if index + 1 < length and chars[index + 1] == "*" {
                var depth = 0
                while index < length {
                    if chars[index] == "#" and index + 1 < length and chars[index + 1] == "*" {
                        depth += 1
                        piece.push("#*")
                        index += 2
                        column += 2
                    } elif chars[index] == "*" and index + 1 < length and chars[index + 1] == "#" {
                        depth -= 1
                        piece.push("*#")
                        index += 2
                        column += 2
                        if depth == 0 { break }
                    } else {
                        if chars[index] == "\n" {
                            line += 1
                            column = 1
                        } else {
                            column += 1
                        }
                        piece.push(chars[index])
                        index += 1
                    }
                }
            } else {
                while index < length and chars[index] != "\n" {
                    piece.push(chars[index])
                    index += 1
                    column += 1
                }
            }
            found.push(
                Comment(
                    _dedent_continuations(piece.build(), start_column),
                    start_line,
                    line,
                    start_column,
                    not code_seen
                )
            )
            code_seen = true
        } elif character == "\"" or character == "'" {
            let raw = index > 0 and chars[index - 1] == "r"
            let (next, next_line, next_column) = _skip_string(chars, index, line, column, raw)
            index = next
            line = next_line
            column = next_column
            code_seen = true
        } else {
            if character != " " and character != "\t" and character != "\r" { code_seen = true }
            index += 1
            column += 1
        }
    }
    return found
}

pub class _Trivia {
    pub let items: list[Comment]
    pub let used: list[bool]
    pub var cursor: int

    pub fn init(self, items: list[Comment]) {
        self.items = items
        self.used = [false for _item in items]
        self.cursor = 0
    }

    pub fn take_leading(self, line: int) -> list[Comment] {
        let taken: list[Comment] = []
        var index = self.cursor
        while index < self.items.len() and self.items[index].line < line {
            if not self.used[index] and self.items[index].own_line {
                self.used[index] = true
                taken.push(self.items[index])
            }
            index += 1
        }
        self._advance()
        return taken
    }

    pub fn take_trailing(self, line: int) -> Comment? {
        var index = self.cursor
        while index < self.items.len() and self.items[index].line <= line {
            let comment = self.items[index]
            if not self.used[index] and not comment.own_line and comment.line == line {
                self.used[index] = true
                self._advance()
                return comment
            }
            index += 1
        }
        return null
    }

    pub fn consume_within(self, first: int, last: int) -> void {
        var index = self.cursor
        while index < self.items.len() and self.items[index].line <= last {
            if self.items[index].line >= first { self.used[index] = true }
            index += 1
        }
        self._advance()
    }

    pub fn any_within(self, first: int, last: int) -> bool {
        var index = self.cursor
        while index < self.items.len() and self.items[index].line <= last {
            if not self.used[index] and self.items[index].line >= first { return true }
            index += 1
        }
        return false
    }

    pub fn any_inside(self, first: int, last: int) -> bool {
        var index = self.cursor
        while index < self.items.len() and self.items[index].line <= last {
            let comment = self.items[index]
            let is_trailing = comment.line == last and not comment.own_line
            if not self.used[index] and comment.line >= first and not is_trailing {
                return true
            }
            index += 1
        }
        return false
    }

    pub fn all_used(self) -> bool {
        for flag in self.used {
            if not flag { return false }
        }
        return true
    }

    pub fn unused(self) -> list[Comment] {
        return [self.items[index] for index in 0..self.items.len() if not self.used[index]]
    }

    pub fn _advance(self) -> void {
        while self.cursor < self.items.len() and self.used[self.cursor] { self.cursor += 1 }
    }
}

#: The canonical formatter behind `jaithon fmt`.
#:
#: There is one way to lay out Jaithon and this module is it. There are no
#: options — no width flag, no brace style, no tabs-versus-spaces — because a
#: formatter with options is a formatter you argue about. The layout is the one
#: `spec/STYLE.md` §1 fixes: four spaces, the opening brace on the same line,
#: one blank line between top-level declarations, never two blank lines in a
#: row, no trailing whitespace, exactly one newline at the end of the file, and
#: a 100-column soft limit that breaks an over-long argument or element list one
#: item to a line.
#:
#: Pipeline: parse into a `jaithon.ast` tree, `freeze_literals`, `scan_comments`,
#: then print, delegating expressions to `jaithon.ast.to_source` and owning what
#: that can't know: comments, blank lines, one-line blocks, the column limit.
#: Every comment in the input must appear in the output, or the file is left
#: exactly as it was.
#:
#: ```
#: format_source("let x=1+2\n", "demo.jai")         # "let x = 1 + 2\n"
#: format_source("if a{f( 1,2 )}\n", "demo.jai")    # "if a { f(1, 2) }\n"
#: ```
from jaithon.ast import DeclKind, Node, NodeKind, TypeExpr, TypeKind, Visibility
from jaithon.ast import op_text, walk
from jaithon.ast_unparse import to_source
from jaithon.compile.lexer import DiagnosticBag, Severity
from jaithon.compile.parser import parse
from std.io import Path, print_err, read_file, write_file
from std.str import StringBuilder, repeat
from .args import Arguments

#: The column the formatter tries to keep every line inside.
#:
#: A soft limit — a long string literal or operator chain may exceed it. Only
#: argument and element lists are broken to obey it.
pub const LINE_WIDTH: int = 100

#: How many unchanged lines `unified_diff` prints around each change.
pub const DIFF_CONTEXT: int = 3

const _INDENT: str = "    "

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

#: Parse `source` into a `jaithon.ast` tree, or raise `ParseError`.
#:
#: Uses the front end the compiler itself runs, so the formatter sees the same
#: tree the compiler does. The exception message carries the first diagnostic's
#: stable code.
pub fn parse_module(source: str, path: str) -> Node {
    let bag = DiagnosticBag()
    let program = parse(source, 0, bag)
    if program is not null { return program }

    var first = f"{path}: cannot parse"
    for diagnostic in bag.items {
        if diagnostic.severity == Severity.Error {
            first = f"{path}: {diagnostic.code}: {diagnostic.message}"
            break
        }
    }
    throw ParseError(first)
}

#: Pin every literal in `program` to the text it was written as, in place.
#:
#: The tree remembers a literal's value, not its spelling (`0xFF` would
#: re-render as `255`), so each literal node is rewritten into an identifier
#: holding the original source slice. The result is for printing only — never
#: feed it to a later compiler pass.
pub fn freeze_literals(program: Node, source: str) -> Node {
    let text = _Source(source)
    walk(program, fn(current: Node) -> bool {
        if not _is_literal(current.kind) {
            return true
        }
        let spelling = text.slice_span(current.span)
        if spelling.len() == 0 {
            return true
        }
        current.kind = NodeKind.Ident
        current.fields = {"name": spelling}
        return false
    }, fn(_current: Node) -> void {})
    return program
}

#: Format one source file and return the canonical form of it.
#:
#: Raises `ParseError` when the source does not parse. Returns `source`
#: unchanged when the printer found no place for a comment — declining to
#: format always beats losing text.
#:
#: ```
#: format_source("fn f( a:int )->int{return a}\n", "d.jai")
#: #     "fn f(a: int) -> int { return a }\n"
#: ```
pub fn format_source(source: str, path: str) -> str {
    let (formatted, lost) = _format(source, path)
    if lost.len() > 0 { return source }
    return formatted
}

#: Return the comments in `source` the printer found no place for.
#:
#: Empty for every file the formatter can rewrite. A non-empty answer is a bug
#: in this module, not the file — `jaithon fmt` reports it rather than silently
#: absorbing it.
#:
#: ```
#: unplaceable_comments("let x = 1   # why\n", "d.jai").len()    # 0
#: ```
pub fn unplaceable_comments(source: str, path: str) -> list[Comment] {
    let (_formatted, lost) = _format(source, path)
    return lost
}

# The printer's raw result: text and unused comments. Both callers need both
# halves, and the work is too slow to do twice.
fn _format(source: str, path: str) -> tuple[str, list[Comment]] {
    let text = _Source(source)
    let program = freeze_literals(parse_module(source, path), source)
    let trivia = _Trivia(scan_comments(source))
    let printer = _Printer(text, trivia)
    printer.statements(program.child_list("stmts"), text.line_count() + 1, true)
    let lost = trivia.unused()
    let lines = printer.render()
    let formatted = if lines.len() == 0 { "" } else { "\n".join(lines) + "\n" }
    return (formatted, lost)
}

#: Report whether `source` is already in canonical form.
#:
#: ```
#: is_formatted("let x = 1\n", "d.jai")    # true
#: is_formatted("let x=1\n", "d.jai")      # false
#: ```
pub fn is_formatted(source: str, path: str) -> bool {
    return format_source(source, path) == source
}

#: Return the one-line header of a declaration, without its body.
#:
#: Pass a node from a tree `freeze_literals` has been over, so a default value
#: reads the way it was written.
#:
#: ```
#: let source = "pub fn f(a: int = 2) -> str? { return null }\n"
#: let tree = freeze_literals(parse_module(source, "d.jai"), source)
#: render_signature(tree.child_list("stmts")[0])    # "pub fn f(a: int = 2) -> str?"
#: ```
pub fn render_signature(decl: Node) -> str {
    return _Printer(_Source(""), _Trivia([])).signature(decl)
}

#: Return a unified diff from `before` to `after`, or the empty string.
#:
#: Standard format — `---`/`+++` headers, `@@` hunks, `DIFF_CONTEXT` lines of
#: context — so the output pipes into `patch`.
#:
#: ```
#: unified_diff("a\n", "a\n", "f.jai")                   # ""
#: unified_diff("a\n", "b\n", "f.jai").split("\n")[0]    # "--- a/f.jai"
#: ```
pub fn unified_diff(before: str, after: str, path: str) -> str {
    if before == after { return "" }
    let hunks = _hunks(_lines_of(before), _lines_of(after))
    if hunks.len() == 0 { return "" }
    let out: list[str] = [f"--- a/{path}", f"+++ b/{path}"]
    for hunk in hunks { out.extend(hunk) }
    return "\n".join(out) + "\n"
}

#: Return the 1-based number of the first line where `before` and `after`
#: differ, or 0 when they hold the same text.
#:
#: A shared line counts as the same only in full — trailing whitespace or a
#: changed indent counts as a difference too.
#:
#: ```
#: first_difference("a\nb\n", "a\nb\n")    # 0
#: first_difference("a\nb\n", "a\nc\n")    # 2
#: first_difference("a\n", "a\nb\n")       # 2
#: ```
pub fn first_difference(before: str, after: str) -> int {
    let old = _lines_of(before)
    let new = _lines_of(after)
    for index in 0..min(old.len(), new.len()) {
        if old[index] != new[index] { return index + 1 }
    }
    if old.len() == new.len() { return 0 }
    return min(old.len(), new.len()) + 1
}

#: Report where `path` stops being canonical, as one `file:line:` message and
#: the two spellings of the line beneath it.
#:
#: What `--check` prints — naming the line and both spellings means the log
#: alone is enough to see the change coming.
pub fn check_report(before: str, after: str, path: str) -> str {
    let line = first_difference(before, after)
    if line == 0 { return "" }
    let old = _lines_of(before)
    let new = _lines_of(after)
    let head = f"{path}:{line}: not formatted"
    let had = if line <= old.len() { f"\n  have: {old[line - 1]}" } else { "" }
    let want = if line <= new.len() { f"\n  want: {new[line - 1]}" } else { "" }
    return head + had + want
}

#: Rewrite Jaithon 2 source into Jaithon 3, mechanically.
#:
#: The table in `spec/LANGUAGE.md` §16 applied to text (Jaithon 2 source cannot
#: be parsed into a Jaithon 3 tree). What cannot be done mechanically — a
#: `namespace` block — is marked with a `# TODO(migrate2):` comment instead of
#: guessed at. The result is not guaranteed to compile.
#:
#: ```
#: migrate2("print x\n")              # "print(x)\n"
#: migrate2("var n = 2 ^ 3\n")        # "var n = 2 ** 3\n"
#: migrate2("if a eq b then do\n")    # "if a == b {\n"
#: migrate2("end\n")                  # "}\n"
#: ```
pub fn migrate2(source: str) -> str {
    let out: list[str] = []
    for raw in _lines_of(source) {
        let indent = _leading_space(raw)
        let body = raw.strip()
        if body.len() == 0 {
            out.push("")
        } elif body.starts_with("#") {
            out.push(raw)
        } elif body == "end" {
            out.push(indent + "}")
        } else {
            out.push(indent + _migrate_line(body))
        }
    }
    return "\n".join(out) + "\n"
}

#: Expand command-line paths into the `.jai` files they name.
#:
#: A file argument is taken as given (not filtered by `keep`); a directory
#: argument is walked and filtered. Raises `FileNotFoundError` for a missing
#: argument — silently processing nothing is how a misspelled path passes CI.
pub fn expand_paths(inputs: list[str], keep: fn(Path) -> bool = _every_file) -> list[str] {
    let found: list[str] = []
    let seen: set[str] = set()
    for input in inputs {
        let path = Path(input)
        if path.is_dir() {
            for entry in path.walk() {
                if entry.is_dir() or entry.extension() != "jai" { continue }
                if not keep(entry) { continue }
                if entry.text not in seen {
                    seen.add(entry.text)
                    found.push(entry.text)
                }
            }
        } elif path.exists() {
            if path.text not in seen {
                seen.add(path.text)
                found.push(path.text)
            }
        } else {
            throw FileNotFoundError(f"no such file or directory: {input}")
        }
    }
    return found.sorted()
}

#: Drop the leading subcommand or script path from `args`.
#:
#: Shared by the four `jaithon.tool` entry points: the CLI hands a subcommand
#: its own name (or a script its path) in `args[0]`.
#:
#: ```
#: tail_arguments(["fmt", "--check"], "fmt")          # ["--check"]
#: tail_arguments(["--check"], "fmt")                 # ["--check"]
#: tail_arguments(["lib/jaithon/tool/fmt.jai"], "fmt")    # []
#: tail_arguments(["a.jai"], "fmt")                   # ["a.jai"]
#: ```
pub fn tail_arguments(args: list[str], command: str) -> list[str] {
    if args.len() == 0 { return args }
    if args[0] == command { return args[1:args.len()] }
    # Only the tool's own script path is dropped here — any other `.jai`
    # argument is a file the caller asked for.
    if args[0].ends_with("/" + command + ".jai") { return args[1:args.len()] }
    return args
}

#: Run `jaithon fmt`.
#:
#: ```text
#: jaithon fmt [--check] [--diff] [--migrate2] [paths...]
#: ```
#:
#: `--check` reports each unformatted file and exits 1 (the CI gate); `--diff`
#: prints a unified diff instead of writing; `--migrate2` runs the Jaithon 2
#: rewrite first. Exit status: 0 clean, 1 for `--check` findings, 2 on parse
#: failure.
pub fn main(args: list[str]) -> int {
    let options = _Options.parse(tail_arguments(args, "fmt"))
    if options.usage_error is not null {
        print_err(f"jaithon fmt: {options.usage_error}")
        print_err("usage: jaithon fmt [--check] [--diff] [--migrate2] [paths...]")
        return 2
    }

    var failed = 0
    var unformatted = 0
    for path in expand_paths(options.paths) {
        let source = read_file(path)
        let input = if options.migrate { migrate2(source) } else { source }
        var output = input
        try {
            let (formatted, lost) = _format(input, path)
            if lost.len() > 0 {
                # A comment with nowhere to go freezes this file until the
                # printer is taught where it belongs.
                print_err(
                    f"{path}:{lost[0].line}: the formatter cannot place " +
                        f"{lost.len()} comment(s); the file is left unchanged"
                )
            } else {
                output = formatted
            }
        } catch error: ParseError {
            if options.migrate {
                # The rewrite is still worth keeping even if formatting failed.
                print_err(f"{path}: migrated but not formatted: {error.message}")
                if not options.check and not options.diff { write_file(path, input) }
            } else {
                print_err(f"{path}: {error.message}")
            }
            failed += 1
            continue
        }
        if output == source { continue }
        unformatted += 1
        if options.check { print(check_report(source, output, path)) }
        if options.diff { print(unified_diff(source, output, path)) }
        if not options.check and not options.diff {
            write_file(path, output)
            print(path)
        }
    }

    if failed > 0 { return 2 }
    if options.check and unformatted > 0 { return 1 }
    return 0
}

# The default filter of `expand_paths`: skips diagnostic fixtures, which don't
# parse on purpose and would otherwise report as formatter failures.
fn _every_file(path: Path) -> bool { return not is_diagnostic_fixture(path) }

#: Report whether `path` is a diagnostic fixture: a file whose first line reads
#: `# expect: Exxxx`, declaring the error it exists to provoke.
#:
#: Invalid on purpose, so a directory walk skips it; naming one explicitly
#: still processes it.
#:
#: ```
#: is_diagnostic_fixture(Path("tests/errors/unclosed_brace.jai"))    # true
#: is_diagnostic_fixture(Path("lib/std/str.jai"))                    # false
#: ```
pub fn is_diagnostic_fixture(path: Path) -> bool {
    let first = _first_line_of(read_file(path.text))
    if not first.starts_with("#") { return false }
    let body = first[1:first.len()].strip()
    if not body.starts_with("expect:") { return false }
    let code = body[7:body.len()].strip()
    return code.len() >= 5 and (code.starts_with("E") or code.starts_with("W"))
}

fn _first_line_of(source: str) -> str {
    let stop = source.find("\n")
    return if stop < 0 { source } else { source[0:stop] }
}

# --- The source text ---------------------------------------------------------

# A file plus the two indexes the printer needs over it.
#
# `jaithon.ast` spans are byte offsets but Jaithon strings are indexed by
# scalar, so every span goes through `scalar_index` first. These tables answer
# what `Span.line_column` computes in O(n) per call, in O(log n) instead.
class _Source {
    pub let text: str
    pub let offsets: list[int]  # byte offset of each scalar, plus one past the end
    pub let starts: list[int]   # scalar index where each line begins

    pub fn init(self, text: str) {
        var byte = 0
        var index = 0
        pub let offsets: list[int] = []
        pub let starts: list[int] = [0]
        for character in text {
            offsets.push(byte)
            byte += _utf8_length(ord(character))
            index += 1
            if character == "\n" { starts.push(index) }
        }
        offsets.push(byte)
        self.text = text
        self.offsets = offsets
        self.starts = starts
    }

    pub fn line_count(self) -> int { return self.starts.len() }

    pub fn scalar_index(self, byte: int) -> int {
        var low = 0
        var high = self.offsets.len() - 1
        while low < high {
            let middle = (low + high + 1) // 2
            if self.offsets[middle] <= byte { low = middle } else { high = middle - 1 }
        }
        return low
    }

    pub fn line_of(self, byte: int) -> int {
        let target = self.scalar_index(byte)
        var low = 0
        var high = self.starts.len() - 1
        while low < high {
            let middle = (low + high + 1) // 2
            if self.starts[middle] <= target { low = middle } else { high = middle - 1 }
        }
        return low + 1
    }

    pub fn line_text(self, line: int) -> str {
        if line < 1 or line > self.starts.len() { return "" }
        let start = self.starts[line - 1]
        let end = if line < self.starts.len() { self.starts[line] - 1 } else { self.text.len() }
        return self.text[start:max(start, end)]
    }

    pub fn slice_span(self, span: any) -> str {
        if span is null { return "" }
        if span.end <= span.start { return "" }
        let start = self.scalar_index(span.start)
        let end = self.scalar_index(span.end)
        if end <= start { return "" }
        return self.text[start:end]
    }
}

# --- Comment bookkeeping -----------------------------------------------------

# The comment list with a consumption mark: every comment is emitted exactly
# once, and `format_source` can check at the end that none were missed.
class _Trivia {
    pub let items: list[Comment]
    pub let used: list[bool]
    pub var cursor: int

    pub fn init(self, items: list[Comment]) {
        self.items = items
        self.used = [false for _item in items]
        self.cursor = 0
    }

    pub fn take_leading(self, line: int) -> list[Comment] {
        pub let taken: list[Comment] = []
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

    # Consume everything a verbatim-copied statement already carries.
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

    # Like `any_within` but ignores the comment on `last` itself — counting it
    # drove every commented one-line statement down the verbatim path forever.
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

# One physical output line, tagged with the depth that decides which trailing-
# comment alignment run it belongs to.
class _Line {
    pub let text: str
    pub var comment: str?
    pub let depth: int

    pub fn init(self, text: str, depth: int) {
        self.text = text
        self.comment = null
        self.depth = depth
    }
}

# One member of a class or enum body. `node` is a `Node` for a method/getter/
# setter but a field record (dict) for a field — tagged by `role`.
class _Member {
    pub let role: str
    pub let node: any
    pub let start: int

    pub fn init(self, role: str, node: any, start: int) {
        self.role = role
        self.node = node
        self.start = start
    }
}

# --- The printer -------------------------------------------------------------
class _Printer {
    pub let src: _Source
    pub let trivia: _Trivia
    pub let out: list[_Line]
    pub var depth: int
    pub var last_line: int

    pub fn init(self, src: _Source, trivia: _Trivia) {
        self.src = src
        self.trivia = trivia
        self.out = []
        self.depth = 0
        self.last_line = 0
    }

    # --- output ------------------------------------------------------------
    pub fn emit(self, text: str) -> void {
        let prefix = repeat(_INDENT, self.depth)
        for piece in text.split("\n") {
            let trimmed = _strip_end(piece)
            let full = if trimmed.len() == 0 { "" } else { prefix + trimmed }
            self.out.push(_Line(full, self.depth))
        }
    }

    pub fn header(self, text: str, line: int) -> void {
        self.emit(text)
        let trailing = self.trivia.take_trailing(line)
        if trailing is not null { self.attach(trailing) }
    }

    pub fn blank(self) -> void {
        if self.out.len() == 0 { return }
        let previous = self.out[self.out.len() - 1].text
        if previous.len() == 0 or previous.ends_with("{") { return }
        self.out.push(_Line("", self.depth))
    }

    pub fn gap_to(self, line: int) -> void {
        if self.last_line > 0 and line > self.last_line + 1 { self.blank() }
        self.last_line = line
    }

    pub fn attach(self, comment: Comment) -> void {
        if self.out.len() == 0 { return }
        self.out[self.out.len() - 1].comment = comment.text
    }

    pub fn column(self) -> int { return self.depth * _INDENT.len() }

    # Aligns each run of consecutive trailing comments sharing a depth into
    # one column.
    pub fn render(self) -> list[str] {
        pub let text: list[str] = []
        var index = 0
        while index < self.out.len() {
            if self.out[index].comment is null {
                text.push(self.out[index].text)
                index += 1
                continue
            }
            var last = index
            while last + 1 < self.out.len() and
                self.out[last + 1].comment is not null and
                self.out[last + 1].depth == self.out[index].depth {
                last += 1
            }
            var width = 0
            for position in index..=last { width = max(width, self.out[position].text.len()) }
            for position in index..=last {
                let line = self.out[position]
                let padding = repeat(" ", width - line.text.len() + 2)
                text.push(line.text + padding + (line.comment ?? ""))
            }
            index = last + 1
        }
        while text.len() > 0 and text[text.len() - 1].len() == 0 { text.pop() }
        return text
    }

    # --- statement sequences -----------------------------------------------
    pub fn statements(self, stmts: list[Node], close_line: int, top_level: bool) -> void {
        for statement in stmts {
            let start = self.src.line_of(statement.span.start)
            let leading = self.trivia.take_leading(start)
            let first = if leading.len() > 0 { leading[0].line } else { start }
            if top_level and self.out.len() > 0 and _is_declaration(statement) {
                self.blank()
                self.last_line = first
            } else {
                self.gap_to(first)
            }
            self.comments(leading)
            self.last_line = start
            let end = self.src.line_of(statement.span.end)
            # A verbatim copy already carries its trailing comment — asking
            # again would print it twice.
            if not self.statement(statement, start, end) {
                let trailing = self.trivia.take_trailing(end)
                if trailing is not null { self.attach(trailing) }
            }
            self.last_line = end
        }
        self.comments(self.trivia.take_leading(close_line))
    }

    pub fn comments(self, items: list[Comment]) -> void {
        for comment in items {
            self.gap_to(comment.line)
            self.emit(comment.text)
            self.last_line = comment.end_line
        }
    }

    # Prints one statement; returns true when it was copied verbatim (so it
    # already carries the comments in its span).
    pub fn statement(self, node: Node, start: int, end: int) -> bool {
        # A comment inside an expression has nowhere to go when reprinted, so
        # the statement is copied verbatim instead.
        if not _has_block(node) and self.trivia.any_inside(start, end) {
            self.trivia.consume_within(start, end)
            self.emit(_dedent_block(self.src, start, end))
            return true
        }

        let simple = self.simple_statement(node)
        if simple is not null {
            self.emit(simple)
            return false
        }

        # A block written on one line stays on one line (`spec/STYLE.md` §10);
        # exploding it would rewrite most of the library for nothing.
        if start == end {
            let probe = _Printer(self.src, _Trivia([]))
            probe.block_statement(node)
            let collapsed = _collapse(probe.render())
            if collapsed is not null and self.column() + collapsed.len() <= LINE_WIDTH {
                self.emit(collapsed)
                return false
            }
        }
        self.block_statement(node)
        return false
    }

    # Statements with no block of their own; null for everything else.
    pub fn simple_statement(self, node: Node) -> str? {
        let column = self.column()
        match node.kind {
            NodeKind.ExprStmt => { return self.expr(node.child("expr"), column) }
            NodeKind.VarDecl => { return self.var_decl(node, column) }
            NodeKind.Assign => {
                let target = to_source(node.require("target"))
                let operator = if node.flag("isCompound") { op_text(node.op()) + "=" } else { "=" }
                let head = target + " " + operator + " "
                return head + self.expr(node.child("value"), column + head.len())
            }
            NodeKind.Return => {
                let value = node.child("value")
                if value is null { return "return" }
                return "return " + self.expr(value, column + 7)
            }
            NodeKind.Throw => { return "throw " + self.expr(node.child("value"), column + 6) }
            NodeKind.Break => { return "break" + _label_suffix(node.opt_text("label")) }
            NodeKind.Continue => { return "continue" + _label_suffix(node.opt_text("label")) }
            NodeKind.Assert => {
                let condition = self.expr(node.child("cond"), column + 7)
                let message = node.child("message")
                if message is null { return "assert " + condition }
                return "assert " + condition + ", " + self.expr(message, column)
            }
            NodeKind.Import | NodeKind.FromImport | NodeKind.Export | NodeKind.ModuleDecl => {
                return to_source(node)
            }
            NodeKind.TypeDecl => { return self.signature(node) }
            _ => { return null }
        }
    }

    # `head { ... }` on one line, or null when the body does not fit that shape.
    #
    # Printed by a throwaway printer holding no comments — safe because the
    # caller has already established this construct spans one source line.
    pub fn collapse_body(self, head: str, body: Node?) -> str? {
        if body is null { return null }
        let probe = _Printer(self.src, _Trivia([]))
        probe.emit(head + " {")
        probe.body(body)
        probe.emit("}")
        return _collapse(probe.render())
    }

    pub fn block_statement(self, node: Node) -> void {
        let column = self.column()
        let line = self.src.line_of(node.span.start)
        match node.kind {
            NodeKind.Block => {
                self.header("{", line)
                self.body(node)
                self.emit("}")
            }
            NodeKind.If => { self.if_statement(node) }
            NodeKind.While => {
                self.header(
                    _label_prefix(node.opt_text("label")) +
                        "while " +
                        self.expr(node.child("cond"), column + 6) +
                        " {",
                    line
                )
                self.body(node.child("body"))
                self.emit("}")
            }
            NodeKind.Loop => {
                self.header(_label_prefix(node.opt_text("label")) + "loop {", line)
                self.body(node.child("body"))
                self.emit("}")
            }
            NodeKind.For => {
                self.header(
                    _label_prefix(node.opt_text("label")) +
                        "for " +
                        to_source(node.require("pattern")) +
                        " in " +
                        self.expr(node.child("iterable"), column) +
                        " {",
                    line
                )
                self.body(node.child("body"))
                self.emit("}")
            }
            NodeKind.Match => {
                self.header("match " + self.expr(node.child("subject"), column + 6) + " {", line)
                self.depth += 1
                self.arms(node)
                self.depth -= 1
                self.emit("}")
            }
            NodeKind.Try => { self.try_statement(node) }
            NodeKind.Defer => {
                self.header("defer {", line)
                self.body(node.child("body"))
                self.emit("}")
            }
            NodeKind.FnDecl => {
                self.header(self.signature(node) + " {", line)
                self.body(node.child("body"))
                self.emit("}")
            }
            NodeKind.ClassDecl => { self.type_body(node, _class_members(node)) }
            NodeKind.TraitDecl => { self.type_body(node, _method_members(node)) }
            NodeKind.EnumDecl => { self.enum_decl(node) }
            _ => {
                # An unknown shape is copied rather than lost.
                let end = self.src.line_of(node.span.end)
                self.trivia.consume_within(line, end)
                self.emit(_dedent_block(self.src, line, end))
            }
        }
    }

    pub fn body(self, block: Node?) -> void {
        if block is null { return }
        self.depth += 1
        if block.kind == NodeKind.Block {
            let saved = self.last_line
            self.last_line = self.src.line_of(block.span.start)
            self.statements(block.child_list("stmts"), self.src.line_of(block.span.end), false)
            self.last_line = saved
        } else {
            self.emit(self.expr(block, self.column()))
        }
        self.depth -= 1
    }

    pub fn if_statement(self, node: Node) -> void {
        var current = node
        var opener = "if "
        loop {
            let line = self.src.line_of(current.span.start)
            self.header(
                opener + self.expr(current.child("cond"), self.column() + opener.len()) + " {",
                line
            )
            self.body(current.child("thenBranch"))
            let alternative = current.child("elseBranch")
            if alternative is null {
                self.emit("}")
                return
            }
            if alternative.kind == NodeKind.If {
                opener = "} elif "
                current = alternative
                continue
            }
            self.header("} else {", self.src.line_of(alternative.span.start))
            self.body(alternative)
            self.emit("}")
            return
        }
    }

    pub fn try_statement(self, node: Node) -> void {
        self.header("try {", self.src.line_of(node.span.start))
        self.body(node.child("body"))
        for handler in node.records("catches") {
            let name = handler["name"] ?? "_error"
            let caught = [entry.to_source() for entry in handler["types"]]
            let clause = if caught.len() == 0 { "" } else { ": " + " | ".join(caught) }
            self.header(f"}} catch {name}{clause} {{", self.src.line_of(handler["span"].start))
            self.body(handler["body"])
        }
        let ending = node.child("finallyBlock")
        if ending is not null {
            self.header("} finally {", self.src.line_of(ending.span.start))
            self.body(ending)
        }
        self.emit("}")
    }

    # Arms are records rather than statements, so `statements` never sees them;
    # without this, a comment here would drift to the end of the block or be
    # dropped entirely.
    pub fn arms(self, node: Node) -> void {
        self.last_line = self.src.line_of(node.span.start)
        for arm in node.records("arms") {
            let start = self.src.line_of(arm["span"].start)
            let leading = self.trivia.take_leading(start)
            self.gap_to(if leading.len() > 0 { leading[0].line } else { start })
            self.comments(leading)
            self.last_line = start
            self.emit(self.match_arm(arm))
            let end = self.src.line_of(arm["span"].end)
            let trailing = self.trivia.take_trailing(end)
            if trailing is not null { self.attach(trailing) }
            self.last_line = end
        }
        self.comments(self.trivia.take_leading(self.src.line_of(node.span.end)))
    }

    pub fn match_arm(self, arm: dict[str, any]) -> str {
        let guard = arm["guard"]
        let guard_text = if guard is null { "" } else { " if " + to_source(guard) }
        let head = to_source(arm["pattern"]) + guard_text + " => "
        let body = arm["body"]
        if body is not null and body.kind == NodeKind.Block {
            let inner = self.collapse_body("", body)
            if inner is not null and self.column() + head.len() + inner.len() <= LINE_WIDTH {
                return head + inner
            }
            return head + self.nested_block(body)
        }
        return head + self.expr(body, self.column() + head.len()) + ","
    }

    # --- declarations -------------------------------------------------------
    pub fn signature(self, node: Node) -> str {
        match node.kind {
            NodeKind.FnDecl | NodeKind.AnonFn => {
                let returns = _type_source(node.type_field("returnType"))
                let arrow = if returns.len() == 0 { "" } else { " -> " + returns }
                let statics = if node.flag("isStatic") { "static " } else { "" }
                let head = _visibility(node.visibility()) +
                    statics +
                    "fn " +
                    (node.opt_text("name") ?? "") +
                    self.generics(node.records("generics"))
                # The return type and opening ` {` land on the same line as the
                # parameters, so both count toward `LINE_WIDTH` too.
                return head +
                    self.params(node.records("params"), self.column() + head.len(), arrow.len() + 2) +
                    arrow
            }
            NodeKind.ClassDecl => {
                let parent = _type_source(node.type_field("superclass"))
                let extends_clause = if parent.len() == 0 { "" } else { " extends " + parent }
                let traits = [entry.to_source() for entry in node.type_list("traits")]
                let implements = if traits.len() == 0 { "" } else { ": " + ", ".join(traits) }
                return _visibility(node.visibility()) +
                    "class " +
                    node.text("name") +
                    self.generics(node.records("generics")) +
                    extends_clause +
                    implements
            }
            NodeKind.TraitDecl => {
                let supers = [entry.to_source() for entry in node.type_list("supers")]
                let bounds = if supers.len() == 0 { "" } else { ": " + ", ".join(supers) }
                return _visibility(node.visibility()) +
                    "trait " +
                    node.text("name") +
                    self.generics(node.records("generics")) +
                    bounds
            }
            NodeKind.EnumDecl => {
                return _visibility(node.visibility()) +
                    "enum " +
                    node.text("name") +
                    self.generics(node.records("generics"))
            }
            NodeKind.TypeDecl => {
                return _visibility(node.visibility()) +
                    "type " +
                    node.text("name") +
                    self.generics(node.records("generics")) +
                    " = " +
                    _type_source(node.type_field("aliased"))
            }
            NodeKind.VarDecl => { return self.var_decl(node, self.column()) }
            _ => { return to_source(node) }
        }
    }

    pub fn var_decl(self, node: Node, column: int) -> str {
        let word = match node.get("varDeclKind") {
            DeclKind.Let => "let",
            DeclKind.Var => "var",
            DeclKind.Const => "const",
        }
        let declared = _type_source(node.type_field("declaredType"))
        let annotation = if declared.len() == 0 { "" } else { ": " + declared }
        let head = _visibility(node.visibility()) +
            word +
            " " +
            to_source(node.require("pattern")) +
            annotation
        let init = node.child("init")
        if init is null { return head }
        return head + " = " + self.expr(init, column + head.len() + 3)
    }

    pub fn type_body(self, node: Node, entries: list[_Member]) -> void {
        self.header(self.signature(node) + " {", self.src.line_of(node.span.start))
        self.depth += 1
        self.members(node, entries)
        self.depth -= 1
        self.emit("}")
    }

    # A payload-less variant is written bare (`None`, never `None()`), so
    # `params` — which answers "()" for an empty list — can't be reused as is.
    pub fn variant(self, entry: dict[str, any]) -> str {
        let payload = entry["params"]
        if payload.len() == 0 { return entry["name"] }
        return entry["name"] + self.params(payload, self.column())
    }

    pub fn enum_decl(self, node: Node) -> void {
        let variants = [self.variant(entry) for entry in node.records("variants")]
        let methods = node.child_list("methods")
        let open = self.src.line_of(node.span.start)
        let close = self.src.line_of(node.span.end)
        let flat = self.signature(node) + " { " + ", ".join(variants) + " }"
        # Collapsing onto one line has nowhere to put a comment between
        # variants, so a commented enum keeps its lines.
        if methods.len() == 0 and
            self.column() + flat.len() <= LINE_WIDTH and
            not flat.contains("\n") and
            not self.trivia.any_within(open, close) {
            self.emit(flat)
            return
        }
        self.header(self.signature(node) + " {", open)
        self.depth += 1
        self.last_line = open
        for (index, entry) in node.records("variants").enumerate() {
            let start = self.src.line_of(entry["span"].start)
            let leading = self.trivia.take_leading(start)
            self.gap_to(if leading.len() > 0 { leading[0].line } else { start })
            self.comments(leading)
            self.last_line = start
            self.emit(variants[index] + ",")
            let end = self.src.line_of(entry["span"].end)
            let trailing = self.trivia.take_trailing(end)
            if trailing is not null { self.attach(trailing) }
            self.last_line = end
        }
        if methods.len() > 0 {
            self.members(node, _method_members(node))
        } else {
            self.comments(self.trivia.take_leading(close))
        }
        self.depth -= 1
        self.emit("}")
    }

    # Class and enum bodies keep source order, though the tree splits members
    # into one list per kind.
    pub fn members(self, owner: Node, entries: list[_Member]) -> void {
        self.last_line = self.src.line_of(owner.span.start)
        for (index, member) in _by_position(entries).enumerate() {
            let start = self.src.line_of(member.start)
            let leading = self.trivia.take_leading(start)
            let first = if leading.len() > 0 { leading[0].line } else { start }
            if index > 0 and member.role != "field" {
                self.blank()
                self.last_line = first
            } else {
                self.gap_to(first)
            }
            self.comments(leading)
            self.last_line = start
            let end = self.member(member, start)
            let trailing = self.trivia.take_trailing(end)
            if trailing is not null { self.attach(trailing) }
            self.last_line = end
        }
        self.comments(self.trivia.take_leading(self.src.line_of(owner.span.end)))
    }

    # Print one member and return the source line it ended on.
    pub fn member(self, member: _Member, start: int) -> int {
        if member.role == "field" {
            let field = member.node
            let keyword = if field["isLet"] { "let" } else { "var" }
            let statics = if field["isStatic"] { "static " } else { "" }
            let head = _visibility(field["visibility"]) +
                statics +
                keyword +
                " " +
                field["name"] +
                ": " +
                _type_source(field["type"])
            let fallback = field["defaultValue"]
            if fallback is null {
                self.emit(head)
            } else {
                self.emit(head + " = " + self.expr(fallback, self.column() + head.len() + 3))
            }
            return self.src.line_of(field["span"].end)
        }

        let node = member.node
        let head = match member.role {
            "getter" => _visibility(node.visibility()) + "get " + node.text("name") + "() -> " + _type_source(node.type_field("returnType")),
            "setter" => _visibility(node.visibility()) + "set " + node.text("name") + self.params(node.records("params"), self.column()),
            _ => self.signature(node),
        }
        let end = self.src.line_of(node.span.end)
        let body = node.child("body")
        if body is null {
            self.emit(head)
            return end
        }
        if start == end and not self.trivia.any_within(start, end) {
            let inner = self.collapse_body(head, body)
            if inner is not null and self.column() + inner.len() <= LINE_WIDTH {
                self.emit(inner)
                return end
            }
        }
        self.header(head + " {", start)
        self.body(body)
        self.emit("}")
        return end
    }

    pub fn generics(self, generics: list[dict[str, any]]) -> str {
        if generics.len() == 0 { return "" }
        let parts: list[str] = []
        for entry in generics {
            let bound = _type_source(entry["bound"])
            parts.push(entry["name"] + (if bound.len() == 0 { "" } else { ": " + bound }))
        }
        return "[" + ", ".join(parts) + "]"
    }

    # `reserve` is what the caller will print after the closing paren and still
    # on that line, and so has to fit inside the limit too.
    pub fn params(self, params: list[dict[str, any]], column: int, reserve: int = 0) -> str {
        if params.len() == 0 { return "()" }
        let parts: list[str] = []
        for param in params {
            let marker = if param["isVariadic"] {
                "..."
            } elif param["isKwRest"] {
                "**"
            } else {
                ""
            }
            let declared = _type_source(param["type"])
            let annotation = if declared.len() == 0 { "" } else { ": " + declared }
            let fallback = param["defaultValue"]
            let given = if fallback is null { "" } else { " = " + self.expr(fallback, column) }
            parts.push(marker + param["name"] + annotation + given)
        }
        let flat = "(" + ", ".join(parts) + ")"
        if column + flat.len() + reserve <= LINE_WIDTH and not flat.contains("\n") { return flat }
        return "(\n" + ",\n".join([_indent_block(part) for part in parts]) + "\n)"
    }

    # --- expressions --------------------------------------------------------

    # `to_source` is the canonical renderer — it owns precedence, parentheses
    # and spelling. This wrapper only knows about the column limit.
    pub fn expr(self, node: Node?, column: int) -> str {
        if node is null { return "" }
        let flat = to_source(node)
        if flat.contains("\n") {
            # `to_source` always opens a block for an `if`/`match` expression;
            # this wrapper folds it back when it fits (`spec/STYLE.md` §10).
            let folded = _collapse(flat.split("\n"))
            if folded is not null and column + folded.len() <= LINE_WIDTH { return folded }
            return flat
        }
        if column + flat.len() <= LINE_WIDTH { return flat }
        return self.broken(node, column, flat) ?? flat
    }

    # One item per line. `flat` is the same node rendered on one line, so any
    # parentheses a low-precedence callee needs are already resolved there.
    pub fn broken(self, node: Node, column: int, flat: str) -> str? {
        match node.kind {
            NodeKind.Call => {
                let args = node.records("args")
                if args.len() == 0 { return null }
                let parts: list[str] = []
                for argument in args {
                    let spread = if argument["isSpread"] { "..." } else { "" }
                    let name = argument["name"]
                    let label = if name is null { "" } else { name + ": " }
                    parts.push(spread + label + self.expr(argument["value"], column + 4))
                }
                let rendered = to_source(node.require("callee"))
                let callee = if flat.starts_with(rendered + "(") { rendered } else { "(" + rendered + ")" }
                # No trailing comma — the §15 grammar for `args` doesn't admit one.
                return callee + "(\n" + ",\n".join([_indent_block(part) for part in parts]) + "\n)"
            }
            NodeKind.ListLit => { return self.spread_items(node, "[", "]", column) }
            NodeKind.SetLit => { return self.spread_items(node, "{", "}", column) }
            NodeKind.TupleLit => { return self.spread_items(node, "(", ")", column) }
            NodeKind.DictLit => {
                let keys = node.child_list("keys")
                let values = node.child_list("values")
                if keys.len() == 0 { return null }
                let parts: list[str] = []
                for index in 0..keys.len() {
                    parts.push(self.expr(keys[index], column + 4) + ": " + self.expr(values[index], column + 4))
                }
                return "{\n" + ",\n".join([_indent_block(part) for part in parts]) + ",\n}"
            }
            NodeKind.Binary | NodeKind.Logical => { return _broken_chain(node, flat) }
            _ => { return null }
        }
    }

    pub fn spread_items(self, node: Node, open: str, close: str, column: int) -> str? {
        let items = node.child_list("items")
        if items.len() == 0 { return null }
        # `(x,)` is a one-element tuple and `(x)` is not, and the comma cannot
        # survive the break, so a one-element tuple stays on its line.
        if close == ")" and items.len() == 1 { return null }
        let parts = [self.expr(item, column + 4) for item in items]
        let comma = if close == ")" { "" } else { "," }
        return open +
            "\n" +
            ",\n".join([_indent_block(part) for part in parts]) +
            comma +
            "\n" +
            close
    }

    # A block inside an expression, rendered by its own printer so its comments
    # follow the same rules; the lines return relatively indented, which `emit`
    # shifts into place.
    pub fn nested_block(self, block: Node) -> str {
        let inner = _Printer(self.src, self.trivia)
        inner.depth = 1
        inner.last_line = self.src.line_of(block.span.start)
        inner.statements(block.child_list("stmts"), self.src.line_of(block.span.end), false)
        let lines = inner.render()
        if lines.len() == 0 { return "{ }" }
        return "{\n" + "\n".join(lines) + "\n}"
    }
}

# --- Printing helpers --------------------------------------------------------
fn _class_members(node: Node) -> list[_Member] {
    let entries: list[_Member] = []
    for field in node.records("fields") {
        entries.push(_Member("field", field, field["span"].start))
    }
    for method in node.child_list("methods") {
        entries.push(_Member("method", method, method.span.start))
    }
    for getter in node.child_list("getters") {
        entries.push(_Member("getter", getter, getter.span.start))
    }
    for setter in node.child_list("setters") {
        entries.push(_Member("setter", setter, setter.span.start))
    }
    return entries
}

fn _method_members(node: Node) -> list[_Member] {
    return [_Member("method", method, method.span.start) for method in node.child_list("methods")]
}

# Insertion sort: a class body holds a handful of members, the input is four
# already-sorted runs, and stability keeps co-located members in source order.
fn _by_position(entries: list[_Member]) -> list[_Member] {
    let ordered = [entry for entry in entries]
    var index = 1
    while index < ordered.len() {
        let current = ordered[index]
        var position = index - 1
        while position >= 0 and ordered[position].start > current.start {
            ordered[position + 1] = ordered[position]
            position -= 1
        }
        ordered[position + 1] = current
        index += 1
    }
    return ordered
}

# An operator chain broken after each operator, continuations indented a step.
#
# Splits the rendered text, not the tree: re-rendering an operand alone would
# lose the parens that keep `(a+b)/c` different from `a+b/c`.
fn _broken_chain(node: Node, flat: str) -> str? {
    let pieces = _split_operator(flat, op_text(node.op()))
    if pieces.len() < 2 { return null }
    var out = pieces[0]
    for index in 1..pieces.len() {
        out = out + " " + op_text(node.op()) + "\n" + _INDENT + pieces[index]
    }
    return out
}

# Cut `text` at every ` op ` sitting outside brackets and outside a string.
#
# A binary operator is always rendered with a space on each side, and a unary
# one never is, so the spaces alone tell the two apart: `a - -b` holds one ` - `.
fn _split_operator(text: str, operator: str) -> list[str] {
    let needle = " " + operator + " "
    let chars = [character for character in text]
    let pieces: list[str] = []
    var depth = 0
    var start = 0
    var index = 0
    while index < chars.len() {
        let character = chars[index]
        if character == "\"" or character == "'" {
            let raw = index > 0 and chars[index - 1] == "r"
            let (next, _line, _column) = _skip_string(chars, index, 1, 1, raw)
            index = max(next, index + 1)
            continue
        }
        if character == "(" or character == "[" or character == "{" { depth += 1 }
        if character == ")" or character == "]" or character == "}" { depth -= 1 }
        if depth == 0 and character == " " and _matches_at(chars, index, needle) {
            pieces.push("".join(chars[start:index]))
            index += needle.len()
            start = index
            continue
        }
        index += 1
    }
    pieces.push("".join(chars[start:chars.len()]))
    return pieces
}

fn _matches_at(chars: list[str], index: int, needle: str) -> bool {
    if index + needle.len() > chars.len() { return false }
    for offset in 0..needle.len() {
        if chars[index + offset] != needle[offset] { return false }
    }
    return true
}

# Fold a printed block onto one line, or refuse when it can't (a body of more
# than one statement has no separator but the newline).
#
# ["if c {", "    return 1", "}"]  ->  "if c { return 1 }"
# ["if c {", "    a()", "    b()", "}"]  ->  null
fn _collapse(lines: list[str]) -> str? {
    if lines.len() < 2 { return null }
    var out = ""
    for raw in lines {
        let part = raw.strip()
        if part.len() == 0 { return null }
        if out.len() == 0 {
            out = part
        } elif part.starts_with("}") or out.ends_with("{") {
            out = out + " " + part
        } else {
            return null
        }
    }
    return out
}

fn _is_literal(kind: NodeKind) -> bool {
    return kind == NodeKind.IntLit or
        kind == NodeKind.FloatLit or
        kind == NodeKind.StrLit or
        kind == NodeKind.FString
}

fn _is_declaration(node: Node) -> bool {
    let kind = node.kind
    return kind == NodeKind.FnDecl or
        kind == NodeKind.ClassDecl or
        kind == NodeKind.TraitDecl or
        kind == NodeKind.EnumDecl or
        kind == NodeKind.TypeDecl
}

# Statements whose interior comments the printer places by walking a block.
fn _has_block(node: Node) -> bool {
    let kind = node.kind
    return kind == NodeKind.If or
        kind == NodeKind.While or
        kind == NodeKind.Loop or
        kind == NodeKind.For or
        kind == NodeKind.Match or
        kind == NodeKind.Try or
        kind == NodeKind.Defer or
        kind == NodeKind.Block or
        kind == NodeKind.FnDecl or
        kind == NodeKind.ClassDecl or
        kind == NodeKind.TraitDecl or
        kind == NodeKind.EnumDecl
}

# An absent annotation and an `Infer` annotation are the same thing written two
# ways, and both print as nothing.
fn _type_source(type_expr: TypeExpr?) -> str {
    if type_expr is null { return "" }
    if type_expr.kind == TypeKind.Infer { return "" }
    return type_expr.to_source()
}

fn _visibility(value: Visibility) -> str {
    if value == Visibility.Public { return "pub " }
    if value == Visibility.Protected { return "prot " }
    return ""
}

fn _label_prefix(label: str?) -> str {
    if label is null { return "" }
    return f"'{label}: "
}

fn _label_suffix(label: str?) -> str {
    if label is null { return "" }
    return f" '{label}"
}

fn _indent_block(text: str) -> str {
    return _INDENT + text.replace("\n", "\n" + _INDENT)
}

fn _strip_end(text: str) -> str {
    var end = text.len()
    while end > 0 and (text[end - 1] == " " or text[end - 1] == "\t") { end -= 1 }
    return text[0:end]
}

fn _leading_space(text: str) -> str {
    var index = 0
    while index < text.len() and (text[index] == " " or text[index] == "\t") { index += 1 }
    return text[0:index]
}

fn _lines_of(text: str) -> list[str] {
    let lines = text.split("\n")
    if lines.len() > 0 and lines[lines.len() - 1].len() == 0 { lines.pop() }
    return lines
}

# The source of a statement, shifted to column zero, for verbatim copying.
fn _dedent_block(src: _Source, first: int, last: int) -> str {
    let lines: list[str] = []
    for line in first..=last { lines.push(src.line_text(line)) }
    if lines.len() == 0 { return "" }
    let margin = _leading_space(lines[0]).len()
    let shifted: list[str] = []
    for (index, line) in lines.enumerate() {
        if index == 0 {
            shifted.push(line.strip())
        } else {
            let cut = min(margin, _leading_space(line).len())
            shifted.push(_strip_end(line[cut:line.len()]))
        }
    }
    return "\n".join(shifted)
}

fn _dedent_continuations(text: str, column: int) -> str {
    if not text.contains("\n") { return text }
    let lines = text.split("\n")
    let out: list[str] = [lines[0]]
    for index in 1..lines.len() {
        let line = lines[index]
        let cut = min(column - 1, _leading_space(line).len())
        out.push(line[cut:line.len()])
    }
    return "\n".join(out)
}

fn _utf8_length(code_point: int) -> int {
    if code_point < 0x80 { return 1 }
    if code_point < 0x800 { return 2 }
    if code_point < 0x10000 { return 3 }
    return 4
}

# Advance past a string literal, returning the position after its closing quote.
fn _skip_string(
    chars: list[str],
    start: int,
    line: int,
    column: int,
    raw: bool
) -> tuple[int, int, int] {
    let length = chars.len()
    let quote = chars[start]
    var index = start + 1
    var current_line = line
    var current_column = column + 1
    var triple = false
    if index + 1 < length and chars[index] == quote and chars[index + 1] == quote {
        triple = true
        index += 2
        current_column += 2
    }
    while index < length {
        let character = chars[index]
        if character == "\\" and not raw {
            index += 2
            current_column += 2
        } elif character == "\n" {
            current_line += 1
            current_column = 1
            index += 1
        } elif character == quote and not triple {
            return (index + 1, current_line, current_column + 1)
        } elif character == quote and
            triple and
            index + 2 < length and
            chars[index + 1] == quote and
            chars[index + 2] == quote {
            return (index + 3, current_line, current_column + 3)
        } else {
            index += 1
            current_column += 1
        }
    }
    return (length, current_line, current_column)
}

# --- The diff ----------------------------------------------------------------

# One line of a diff, tagged with what happened to it.
class _EditOp {
    pub let tag: str  # " ", "-" or "+"
    pub let text: str
    pub let old_line: int
    pub let new_line: int

    pub fn init(self, tag: str, text: str, old_line: int, new_line: int) {
        self.tag = tag
        self.text = text
        self.old_line = old_line
        self.new_line = new_line
    }
}

# Hunks are the runs of edits, each padded with DIFF_CONTEXT unchanged lines.
fn _hunks(old: list[str], new: list[str]) -> list[list[str]] {
    var head = 0
    while head < old.len() and head < new.len() and old[head] == new[head] { head += 1 }
    var tail = 0
    while tail < old.len() - head and
        tail < new.len() - head and
        old[old.len() - 1 - tail] == new[new.len() - 1 - tail] {
        tail += 1
    }

    let ops = _edit_ops(old[head:old.len() - tail], new[head:new.len() - tail])
    let keep = [false for _op in ops]
    for (index, op) in ops.enumerate() {
        if op.tag == " " { continue }
        for near in max(0, index - DIFF_CONTEXT)..min(ops.len(), index + DIFF_CONTEXT + 1) {
            keep[near] = true
        }
    }

    let hunks: list[list[str]] = []
    var index = 0
    while index < ops.len() {
        if not keep[index] {
            index += 1
            continue
        }
        var last = index
        while last < ops.len() and keep[last] { last += 1 }
        hunks.push(_render_hunk(ops, index, last, head))
        index = last
    }
    return hunks
}

# Longest common subsequence over whole lines. The shared prefix and suffix are
# trimmed by the caller, so the quadratic middle is only the changed region.
fn _edit_ops(left: list[str], right: list[str]) -> list[_EditOp] {
    let n = left.len()
    let m = right.len()
    let ops: list[_EditOp] = []
    # Past this size a whole-file replacement is cheaper to print than the table.
    if n * m > 4_000_000 {
        for (index, text) in left.enumerate() { ops.push(_EditOp("-", text, index, 0)) }
        for (index, text) in right.enumerate() { ops.push(_EditOp("+", text, 0, index)) }
        return ops
    }

    let table = [[0 for _column in 0..=m] for _row in 0..=n]
    var row = n - 1
    while row >= 0 {
        var column = m - 1
        while column >= 0 {
            if left[row] == right[column] {
                table[row][column] = table[row + 1][column + 1] + 1
            } else {
                table[row][column] = max(table[row + 1][column], table[row][column + 1])
            }
            column -= 1
        }
        row -= 1
    }

    var i = 0
    var j = 0
    while i < n and j < m {
        if left[i] == right[j] {
            ops.push(_EditOp(" ", left[i], i, j))
            i += 1
            j += 1
        } elif table[i + 1][j] >= table[i][j + 1] {
            ops.push(_EditOp("-", left[i], i, j))
            i += 1
        } else {
            ops.push(_EditOp("+", right[j], i, j))
            j += 1
        }
    }
    while i < n {
        ops.push(_EditOp("-", left[i], i, j))
        i += 1
    }
    while j < m {
        ops.push(_EditOp("+", right[j], i, j))
        j += 1
    }
    return ops
}

fn _render_hunk(ops: list[_EditOp], first: int, last: int, offset: int) -> list[str] {
    var old_count = 0
    var new_count = 0
    let body: list[str] = []
    for index in first..last {
        let op = ops[index]
        if op.tag != "+" { old_count += 1 }
        if op.tag != "-" { new_count += 1 }
        body.push(op.tag + op.text)
    }
    let old_start = offset + ops[first].old_line + 1
    let new_start = offset + ops[first].new_line + 1
    let out: list[str] = [f"@@ -{old_start},{old_count} +{new_start},{new_count} @@"]
    out.extend(body)
    return out
}

# --- The Jaithon 2 rewrite ---------------------------------------------------

# The word-for-word half of the §16 table.
let _MIGRATE_WORDS: dict[str, str] = {"eq": "==", "func": "fn", "elseif": "elif"}

fn _migrate_line(body: str) -> str {
    if body.starts_with("namespace ") {
        return "# TODO(migrate2): `" +
            body +
            "` becomes a module file or a class of statics (spec §16)"
    }
    if body.contains("static void main") { return "pub fn main() -> int {" }
    if body == "print" { return "print()" }
    if body.starts_with("print ") {
        return "print(" + _rewrite_code(body[6:body.len()].strip()) + ")"
    }
    if body.starts_with("input ") {
        let name = body[6:body.len()].strip()
        return f"let {name} = input()"
    }
    if body.starts_with("system ") {
        return "os.shell(" + body[7:body.len()].strip() + ")"
    }
    if body.starts_with("import ") and body.contains("/") {
        return "import std." + body[7:body.len()].strip().split("/")[0]
    }
    return _migrate_block_words(_rewrite_code(body))
}

# `then do`, `do` and `else` become braces.
fn _migrate_block_words(text: str) -> str {
    var out = text
    if out.ends_with(" then do") {
        out = out[0:out.len() - 8] + " {"
    } elif out.ends_with(" then") {
        out = out[0:out.len() - 5] + " {"
    } elif out.ends_with(" do") {
        out = out[0:out.len() - 3] + " {"
    }
    if out == "else" { return "} else {" }
    if out.starts_with("else ") { return "} else " + out[5:out.len()] }
    if out.starts_with("fn ") and not out.ends_with("{") { return out + " {" }
    return out
}

# Token-level rewrites, applied only outside strings.
fn _rewrite_code(text: str) -> str {
    let out = StringBuilder()
    let chars = [character for character in text]
    var index = 0
    while index < chars.len() {
        let character = chars[index]
        if character == "\"" or character == "'" {
            let raw = index > 0 and chars[index - 1] == "r"
            let (next, _line, _column) = _skip_string(chars, index, 1, 1, raw)
            out.push("".join(chars[index:next]))
            index = next
        } elif character == "^" {
            out.push("**")
            index += 1
        } elif character == "!" and (index + 1 >= chars.len() or chars[index + 1] != "=") {
            let (head, operand) = _split_factorial_operand(out.build())
            out.clear()
            out.push(head + "math.factorial(" + operand + ")")
            index += 1
        } elif _is_ident_start(character) {
            index = _rewrite_word(out, chars, index)
        } else {
            out.push(character)
            index += 1
        }
    }
    return out.build()
}

# Rewrite the identifier starting at `index` and return where to carry on.
fn _rewrite_word(out: StringBuilder, chars: list[str], index: int) -> int {
    var end = index
    while end < chars.len() and _is_ident_part(chars[end]) { end += 1 }
    let word = "".join(chars[index:end])

    if word == "new" {
        var next = end
        while next < chars.len() and chars[next] == " " { next += 1 }
        return next
    }
    if word == "_alen" or word == "arrayLen" {
        let (next, argument) = _call_argument(chars, end)
        if argument.len() > 0 {
            out.push(argument.strip() + ".len()")
            return next
        }
    }
    if word == "_get" or word == "arrayGet" {
        let (next, argument) = _call_argument(chars, end)
        let parts = _split_top_level(argument)
        if parts.len() == 2 {
            out.push(parts[0].strip() + "[" + parts[1].strip() + "]")
            return next
        }
    }
    out.push(_MIGRATE_WORDS.get(word, word))
    return end
}

# The operand a postfix `!` applied to, peeled off the end of what was emitted.
fn _split_factorial_operand(text: str) -> tuple[str, str] {
    let end = text.len()
    if end == 0 { return ("", "0") }
    var index = end
    if text[end - 1] == ")" {
        var depth = 0
        index = end - 1
        while index >= 0 {
            if text[index] == ")" { depth += 1 }
            if text[index] == "(" {
                depth -= 1
                if depth == 0 { break }
            }
            index -= 1
        }
    }
    while index > 0 and (_is_ident_part(text[index - 1]) or text[index - 1] == ".") {
        index -= 1
    }
    return (text[0:index], text[index:end])
}

# The text inside the parentheses starting at `open`, and the index after them.
fn _call_argument(chars: list[str], open: int) -> tuple[int, str] {
    var index = open
    while index < chars.len() and chars[index] == " " { index += 1 }
    if index >= chars.len() or chars[index] != "(" { return (open, "") }
    var depth = 0
    let start = index + 1
    while index < chars.len() {
        if chars[index] == "(" { depth += 1 }
        if chars[index] == ")" {
            depth -= 1
            if depth == 0 { break }
        }
        index += 1
    }
    return (index + 1, "".join(chars[start:index]))
}

fn _split_top_level(text: str) -> list[str] {
    let parts: list[str] = []
    var depth = 0
    var start = 0
    for index in 0..text.len() {
        let character = text[index]
        if character == "(" or character == "[" { depth += 1 }
        if character == ")" or character == "]" { depth -= 1 }
        if character == "," and depth == 0 {
            parts.push(text[start:index])
            start = index + 1
        }
    }
    parts.push(text[start:text.len()])
    return parts
}

fn _is_ident_start(character: str) -> bool {
    return character == "_" or character.lower() != character.upper()
}

fn _is_ident_part(character: str) -> bool {
    return _is_ident_start(character) or character >= "0" and character <= "9"
}

# --- Command line ------------------------------------------------------------
class _Options {
    pub var check: bool
    pub var diff: bool
    pub var migrate: bool
    pub var paths: list[str]
    pub var usage_error: str?

    pub fn init(self) {
        self.check = false
        self.diff = false
        self.migrate = false
        self.paths = []
        self.usage_error = null
    }

    pub static fn parse(args: list[str]) -> _Options {
        let options = _Options()
        let reader = Arguments(args)
        while true {
            let argument = reader.next()
            if argument is null { break }
            if not argument.is_flag() {
                options.paths.push(argument.text)
                continue
            }
            if argument.name == "--check" {
                if reader.is_bare(argument) { options.check = true }
            } elif argument.name == "--diff" {
                if reader.is_bare(argument) { options.diff = true }
            } elif argument.name == "--migrate2" {
                if reader.is_bare(argument) { options.migrate = true }
            } else {
                reader.unknown(argument)
            }
        }
        options.usage_error = reader.error
        if options.paths.len() == 0 { options.paths = ["."] }
        return options
    }
}

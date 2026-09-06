#: The pipeline: parse, freeze the literals, scan the comments, print.
from jaithon.ast import Node, NodeKind
from jaithon.ast import walk
from jaithon.compile.diag import DiagnosticBag, Severity
from jaithon.compile.parser import parse
from .comment import Comment, _Trivia, scan_comments
from .layout import _is_literal
from .printer import _Printer
from .source import _Source

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
        # Slots are positioned per kind, so the new kind brings its own list.
        current.kind = NodeKind.Ident
        current.fields = Node.ident(spelling).fields
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

pub fn _format(source: str, path: str) -> tuple[str, list[Comment]] {
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

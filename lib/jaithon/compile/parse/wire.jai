#: The one module that knows every other module of `parse/` at once: it fills
#: in the seven seams `state.jai` declares, and holds `parse`, the front door
#: of the whole directory.
#:
#: It has to be a module that imports everything and is imported by nothing but
#: the facade, because a seam exists exactly where an import may not go.
#:
#: The wiring runs as a statement at module scope rather than as a call each
#: caller makes. `parser.jai` imports this file, so anything that can name
#: `Parser` has already run it, and the two places that build a `Parser`
#: directly -- `repl.jai` and `mod.jai` -- keep working untouched. The sibling
#: of `check/checker.jai`'s `install`, except that nobody has to remember it.
from jaithon.ast import Node
from jaithon.compile.diag import DiagnosticBag
from jaithon.compile.lexer import Lexer
from jaithon.compile.parse.state import Parser, install_hooks
from jaithon.compile.parse.expr import _parse_expression, _parse_unary
from jaithon.compile.parse.pattern import _parse_pattern
from jaithon.compile.parse.stmt import _parse_if, _parse_match
from jaithon.compile.parse.decl import _parse_declaration, _parse_fn_rest

install_hooks(
    _parse_declaration,
    _parse_expression,
    _parse_pattern,
    _parse_fn_rest,
    _parse_if,
    _parse_match,
    _parse_unary
)

#: Parse `source` into a `NodeKind.Program` node, or null when it had any error.
#:
#: ```
#: let bag = DiagnosticBag()
#: parse("let x = \n", diagnostics: bag)     # null
#: bag.has_errors()                          # true
#: ```
pub fn parse(source: str, file_id: int = 0, diagnostics: DiagnosticBag? = null) -> Node? {
    let bag = diagnostics ?? DiagnosticBag()
    let lexer = Lexer(source, file_id, bag)
    let parser = Parser(lexer)
    let program = parser.parse_program()
    if lexer.had_error { return null }
    return program
}

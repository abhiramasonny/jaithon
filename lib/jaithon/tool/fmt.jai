#: The canonical formatter behind `jaithon fmt`.
#:
#: Facade: imports and re-exports every sibling module in `fmt/`.
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
from .fmt.comment import Comment, scan_comments
from .fmt.diff import DIFF_CONTEXT, check_report, first_difference, unified_diff
from .fmt.driver import expand_paths, is_diagnostic_fixture, main, tail_arguments
from .fmt.format import format_source, freeze_literals, is_formatted, parse_module
from .fmt.format import render_signature, unplaceable_comments
from .fmt.layout import LINE_WIDTH
from .fmt.migrate import _is_ident_part, _is_ident_start, migrate2
from .fmt.text import _utf8_length

export { Comment, scan_comments }
export { DIFF_CONTEXT, check_report, first_difference, unified_diff }
export { expand_paths, is_diagnostic_fixture, main, tail_arguments }
export { format_source, freeze_literals, is_formatted, parse_module }
export { render_signature, unplaceable_comments }
export { LINE_WIDTH }
export { _is_ident_part, _is_ident_start, migrate2 }
export { _utf8_length }

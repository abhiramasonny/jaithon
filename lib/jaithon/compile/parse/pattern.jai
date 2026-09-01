#: Match patterns: the literal, binding, sequence and class forms, the
#: comma-separated list that a bracketed pattern and a class pattern's
#: arguments share, and `_expr_to_pattern`, which reinterprets an
#: already-parsed expression once a later `=` proves it was a destructuring
#: target.
#:
#: Separate from `expr.jai` because a pattern is a different grammar that
#: merely shares a prefix with an expression: `Point(x, y)` is a call in one
#: and a class pattern in the other, and only the position decides which was
#: read.
#:
#: What is NOT here: `match` itself. The statement and the expression are
#: control flow and live in `stmt.jai`; this file parses only what an arm
#: matches against.
from jaithon.ast import Node, NodeKind, OpKind
from jaithon.compile.diag import Code
from jaithon.compile.token import TokenKind
from jaithon.compile.parse.state import Parser
from jaithon.compile.parse.node_build import _join, _node, _nulls, _pat_bind, _pat_class, _pat_seq, _unary
from jaithon.compile.parse.predicate import _starts_pattern_literal
from jaithon.compile.parse.primary import _close_sequence, _parse_primary
from jaithon.compile.parse.expr import _parse_neg_magnitude

pub fn _parse_pattern_list(
    p: Parser,
    closer: TokenKind,
    close_text: str,
    what: str,
    open_index: int
) -> tuple[list[Node], int, bool] {
    var elems: list[Node] = []
    var rest_index = -1
    var saw_comma = false

    p._skip_newlines()
    if not p._check(closer) {
        loop {
            p._skip_newlines()
            if p._check(TokenKind.Ellipsis) {
                let rest_span = p._cur_span()
                p._advance()
                var rest = _node(NodeKind.PatWildcard, rest_span)
                if p._check(TokenKind.Ident) {
                    let name_token = p._advance()
                    rest = _pat_bind(_join(rest_span, name_token.span), name_token.text)
                }
                if rest_index >= 0 {
                    p._error_at(
                        rest.span,
                        Code.E0110_INVALID_PATTERN,
                        "a pattern may bind only one `...` rest element"
                    )
                } else {
                    rest_index = elems.len()
                }
                elems.push(rest)
            } else {
                elems.push(_parse_pattern(p))
            }
            p._skip_newlines()
            if not p._match(TokenKind.Comma) { break }
            saw_comma = true
            p._skip_newlines()
            if p._check(closer) or p._check(TokenKind.Eof) { break }
            if p.panic_mode { break }
        }
    }
    _close_sequence(p, closer, close_text, what, open_index)
    return (elems, rest_index, saw_comma)
}

pub fn _parse_pattern_literal_value(p: Parser) -> Node {
    let start_index = p.current
    if p._check(TokenKind.Minus) {
        let folded = _parse_neg_magnitude(p)
        if folded != null { return folded ?? p._error_expr(p._cur_span()) }
    }
    if not p._match(TokenKind.Minus) { return _parse_primary(p) }
    let inner = _parse_primary(p)
    return _unary(p._span_since(start_index), OpKind.Neg, inner)
}

pub fn _parse_pattern_primary(p: Parser) -> Node {
    let start_index = p.current

    if p._match(TokenKind.Underscore) {
        return _node(NodeKind.PatWildcard, p._span_since(start_index))
    }

    if p._check(TokenKind.LParen) {
        p._advance()
        let (elems, rest_index, saw_comma) = _parse_pattern_list(
            p,
            TokenKind.RParen,
            ")",
            "tuple pattern",
            start_index
        )
        if elems.len() == 1 and rest_index < 0 and not saw_comma { return elems[0] }
        return _pat_seq(NodeKind.PatTuple, p._span_since(start_index), elems, rest_index)
    }

    if p._check(TokenKind.LBracket) {
        p._advance()
        let (elems, rest_index, _saw_comma) = _parse_pattern_list(
            p,
            TokenKind.RBracket,
            "]",
            "list pattern",
            start_index
        )
        return _pat_seq(NodeKind.PatList, p._span_since(start_index), elems, rest_index)
    }

    if _starts_pattern_literal(p._kind()) {
        let low = _parse_pattern_literal_value(p)
        if not p._check(TokenKind.DotDot) and not p._check(TokenKind.DotDotEq) {
            return _node(NodeKind.PatLiteral, p._span_since(start_index), {"value": low})
        }
        let inclusive = p._check(TokenKind.DotDotEq)
        p._advance()
        var high: Node? = null
        if _starts_pattern_literal(p._kind()) { high = _parse_pattern_literal_value(p) }
        return _node(
            NodeKind.PatRange,
            p._span_since(start_index),
            {"lo": low, "hi": high, "inclusive": inclusive}
        )
    }

    if p._check(TokenKind.Ident) {
        let name = p._advance().text

        if p._match(TokenKind.Dot) {
            let variant = p._expect_ident_name("an enum variant name")
            if variant == null {
                return _node(NodeKind.PatWildcard, p._span_since(start_index))
            }
            var subs: list[Node] = []
            if p._check(TokenKind.LParen) {
                p._advance()
                let (elems, _rest, _comma) = _parse_pattern_list(
                    p,
                    TokenKind.RParen,
                    ")",
                    "variant pattern",
                    start_index
                )
                subs = elems
            }
            return _pat_class(
                NodeKind.PatEnum,
                p._span_since(start_index),
                name,
                variant,
                _nulls(subs.len()),
                subs
            )
        }

        if p._check(TokenKind.LBrace) {
            return _parse_class_pattern(p, name, start_index)
        }

        if p._check(TokenKind.LParen) {
            p._advance()
            let (elems, _rest, _comma) = _parse_pattern_list(
                p,
                TokenKind.RParen,
                ")",
                "class pattern",
                start_index
            )
            return _pat_class(
                NodeKind.PatClass,
                p._span_since(start_index),
                name,
                null,
                _nulls(elems.len()),
                elems
            )
        }

        return _pat_bind(p._span_since(start_index), name)
    }

    p._error_at(
        p._cur_span(),
        Code.E0110_INVALID_PATTERN,
        f"expected a pattern, found {p._describe(p._peek())}"
    )
    return _node(NodeKind.PatWildcard, p._cur_span())
}

pub fn _parse_class_pattern(p: Parser, name: str, start_index: int) -> Node {
    let open_index = p.current
    p._advance()
    var fields: list[str?] = []
    var subs: list[Node] = []
    p._skip_newlines()
    if not p._check(TokenKind.RBrace) {
        loop {
            p._skip_newlines()
            let field_token = p._peek()
            let field = p._expect_ident_name("a field name in a class pattern")
            if field == null { break }
            var sub = _pat_bind(field_token.span, field)
            if p._match(TokenKind.Colon) { sub = _parse_pattern(p) }
            fields.push(field)
            subs.push(sub)
            p._skip_newlines()
            if not p._match(TokenKind.Comma) { break }
            p._skip_newlines()
            if p._check(TokenKind.RBrace) or p._check(TokenKind.Eof) { break }
        }
    }
    _close_sequence(p, TokenKind.RBrace, "}", "class pattern", open_index)
    return _pat_class(NodeKind.PatClass, p._span_since(start_index), name, null, fields, subs)
}

pub fn _parse_pattern(p: Parser) -> Node {
    if not p._enter_recursion() {
        p.expr_depth -= 1
        return _node(NodeKind.PatWildcard, p._cur_span())
    }
    defer { p.expr_depth -= 1 }

    let start_index = p.current
    let first = _parse_pattern_primary(p)
    if not p._check(TokenKind.Pipe) { return first }

    var alternatives: list[Node] = [first]
    while p._match(TokenKind.Pipe) {
        alternatives.push(_parse_pattern_primary(p))
        if p.panic_mode or p._check(TokenKind.Eof) { break }
    }
    return _pat_seq(NodeKind.PatOr, p._span_since(start_index), alternatives, -1)
}

pub fn _expr_to_pattern(p: Parser, expr: Node) -> Node? {
    if expr.kind == NodeKind.Ident { return _pat_bind(expr.span, expr.text("name")) }
    if expr.kind != NodeKind.TupleLit and expr.kind != NodeKind.ListLit { return null }

    var elems: list[Node] = []
    for item in expr.child_list("items") {
        let sub = _expr_to_pattern(p, item)
        if sub == null { return null }
        elems.push(sub)
    }
    let kind = expr.kind == NodeKind.TupleLit ? NodeKind.PatTuple : NodeKind.PatList
    return _pat_seq(kind, expr.span, elems, -1)
}

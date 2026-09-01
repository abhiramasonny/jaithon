#: The operand forms the precedence ladder bottoms out on: call arguments,
#: f-strings and their format specs, comprehension clauses, the list, dict, set
#: and tuple literals, the parenthesised expression, lambdas, and the
#: `_parse_primary` dispatch itself.
#:
#: Kept apart from `expr.jai` by concern rather than by dependence. The ladder
#: is precedence plumbing; this is the grammar's vocabulary of operands. The
#: two call each other, and `expr.jai` is the file that imports, so the edges
#: running the other way -- an argument needing a whole expression, `if`,
#: `match` and `fn` in expression position, a comprehension's binding pattern,
#: the operand of `await` -- go through the seams `state.jai` declares.
#:
#: What is NOT here: `_parse_postfix` and `_parse_subscript`. A call, a member
#: access and an index read as operators rather than as operands, so they
#: stayed with the ladder.
from jaithon.ast import CompKind, Node, NodeKind, Span, TypeExpr, Visibility
from jaithon.compile.diag import Code
from jaithon.compile.token import Token, TokenKind
from jaithon.compile.parse.state import Parser, parse_expression, parse_fn_rest, parse_if, parse_match
from jaithon.compile.parse.state import parse_pattern, parse_unary
from jaithon.compile.parse.node_build import _arg, _comp_clause, _dict_lit, _ident, _member, _node, _param
from jaithon.compile.parse.node_build import _sequence, _str_lit, _wrap
from jaithon.compile.parse.predicate import _starts_await_operand
from jaithon.compile.parse.type_expr import _parse_type

# Matches JAI_MAX_ARGS in src/common/common.h: the call instruction encodes the
# argument count in one byte.
const _MAX_ARGS = 255

pub fn _parse_args(p: Parser, open_index: int) -> list[dict[str, any]] {
    var args: list[dict[str, any]] = []
    p._skip_newlines()
    if not p._check(TokenKind.RParen) {
        loop {
            p._skip_newlines()
            let arg_start = p.current
            var name: str? = null
            let is_spread = p._match(TokenKind.Ellipsis)
            if not is_spread and p._check(TokenKind.Ident) and p._kind_at(1) == TokenKind.Colon {
                name = p._advance().text
                p._advance()
            }
            let value = parse_expression(p)
            let span = p._span_since(arg_start)
            args.push(_arg(name, value, is_spread, span))

            if args.len() > _MAX_ARGS {
                p._error_at(
                    span,
                    Code.E0100_UNEXPECTED_TOKEN,
                    f"more than {_MAX_ARGS} arguments in one call"
                )
                break
            }
            p._skip_newlines()
            if not p._match(TokenKind.Comma) { break }
            p._skip_newlines()
            if p._check(TokenKind.RParen) or p._check(TokenKind.Eof) { break }
        }
    }
    if p._check(TokenKind.RParen) {
        p._advance()
    } else {
        p._unclosed(open_index, ")", "argument list")
    }
    return args
}

pub fn _apply_format_spec(p: Parser, value: Node) -> Node {
    if not p._match(TokenKind.Colon) { return value }
    let spec_span = p._cur_span()
    var spec_end = spec_span.start
    while not p._check(TokenKind.FStringMid) and
        not p._check(TokenKind.FStringEnd) and
        not p._check(TokenKind.Eof) {
        spec_end = p._advance().span.end
    }
    if spec_end <= spec_span.start { return value }

    let span = value.span
    let text = p.lexer.text_of(Span(spec_span.file, spec_span.start, spec_end))
    let formatter = _member(NodeKind.Member, span, value, "__format__")
    return _node(
        NodeKind.Call,
        span,
        {"callee": formatter, "args": [_arg(null, _str_lit(span, text), false, span)]}
    )
}

pub fn _parse_fstring(p: Parser) -> Node {
    let start_index = p.current
    var parts: list[Node] = []

    let start_token = p._advance()
    if start_token.string_value().len() > 0 {
        parts.push(_str_lit(start_token.span, start_token.string_value()))
    }
    let complete = start_token.kind == TokenKind.FStringEnd

    while not complete {
        if p._check(TokenKind.Eof) {
            p._error_at_cur(Code.E0107_UNCLOSED_DELIMITER, "unterminated interpolated string")
            break
        }
        if p._check(TokenKind.Colon) {
            let empty = p._error_expr(p._cur_span())
            p._error_at_cur(
                Code.E0102_EXPECTED_EXPRESSION,
                "expected an expression before the format spec"
            )
            parts.push(empty)
        } elif not p._check(TokenKind.FStringMid) and not p._check(TokenKind.FStringEnd) {
            parts.push(_apply_format_spec(p, parse_expression(p)))
        }

        if p._check(TokenKind.FStringMid) {
            let mid = p._advance()
            if mid.string_value().len() > 0 {
                parts.push(_str_lit(mid.span, mid.string_value()))
            }
            continue
        }
        if p._check(TokenKind.FStringEnd) {
            let end = p._advance()
            if end.string_value().len() > 0 {
                parts.push(_str_lit(end.span, end.string_value()))
            }
            break
        }
        if not p.panic_mode {
            p._error_at_cur(
                Code.E0101_EXPECTED_TOKEN,
                f"expected `}}` to close the interpolation, " + f"found {p._describe(p._peek())}"
            )
        }
        break
    }

    let span = p._span_since(start_index)
    if parts.len() == 0 { parts.push(_str_lit(span, "")) }
    return _node(NodeKind.FString, span, {"parts": parts})
}

# --- comprehensions and collection literals --------------------------
pub fn _parse_comp_clauses(p: Parser) -> list[dict[str, any]] {
    var clauses: list[dict[str, any]] = []
    while p._check(TokenKind.KwFor) {
        let clause_start = p.current
        p._advance()
        let pattern = parse_pattern(p)
        p._expect(TokenKind.KwIn, "`in` after the comprehension pattern")
        let iterable = parse_expression(p)

        var conditions: list[Node] = []
        while p._match(TokenKind.KwIf) {
            conditions.push(parse_expression(p))
            if p._check(TokenKind.Eof) { break }
        }
        clauses.push(_comp_clause(pattern, iterable, conditions, p._span_since(clause_start)))
        if p._check(TokenKind.Eof) or p.panic_mode { break }
    }
    return clauses
}

pub fn _finish_comprehension(
    p: Parser,
    start_index: int,
    comp_kind: CompKind,
    key_expr: Node?,
    element: Node,
    closer: TokenKind,
    close_text: str,
    what: str
) -> Node {
    let clauses = _parse_comp_clauses(p)
    p._skip_newlines()
    _close_sequence(p, closer, close_text, what, start_index)

    return _node(
        NodeKind.Comprehension,
        p._span_since(start_index),
        {"compKind": comp_kind, "element": element, "keyExpr": key_expr, "clauses": clauses}
    )
}

pub fn _parse_rest_of_sequence(p: Parser, first: Node, closer: TokenKind) -> list[Node] {
    var items: list[Node] = [first]
    p._skip_newlines()
    while p._match(TokenKind.Comma) {
        p._skip_newlines()
        if p._check(closer) or p._check(TokenKind.Eof) { break }
        items.push(parse_expression(p))
        p._skip_newlines()
        if p.panic_mode { break }
    }
    return items
}

pub fn _close_sequence(
    p: Parser,
    closer: TokenKind,
    close_text: str,
    what: str,
    open_index: int
) -> void {
    if p._check(closer) {
        p._advance()
    } else {
        p._unclosed(open_index, close_text, what)
    }
}

pub fn _parse_list_literal(p: Parser) -> Node {
    let start_index = p.current
    p._advance()
    p._skip_newlines()

    if p._match(TokenKind.RBracket) {
        return _sequence(NodeKind.ListLit, p._span_since(start_index), [])
    }

    let first = parse_expression(p)
    if p._check(TokenKind.KwFor) {
        return _finish_comprehension(
            p,
            start_index,
            CompKind.List,
            null,
            first,
            TokenKind.RBracket,
            "]",
            "list comprehension"
        )
    }
    let items = _parse_rest_of_sequence(p, first, TokenKind.RBracket)
    _close_sequence(p, TokenKind.RBracket, "]", "list literal", start_index)
    return _sequence(NodeKind.ListLit, p._span_since(start_index), items)
}

pub fn _parse_brace_literal(p: Parser) -> Node {
    let start_index = p.current
    p._advance()
    p._skip_newlines()

    if p._match(TokenKind.RBrace) {
        return _dict_lit(p._span_since(start_index), [], [])
    }

    let first = parse_expression(p)

    if p._match(TokenKind.Colon) {
        p._skip_newlines()
        let first_value = parse_expression(p)
        if p._check(TokenKind.KwFor) {
            return _finish_comprehension(
                p,
                start_index,
                CompKind.Dict,
                first,
                first_value,
                TokenKind.RBrace,
                "}",
                "dict comprehension"
            )
        }
        var keys: list[Node] = [first]
        var values: list[Node] = [first_value]
        p._skip_newlines()
        while p._match(TokenKind.Comma) {
            p._skip_newlines()
            if p._check(TokenKind.RBrace) or p._check(TokenKind.Eof) { break }
            keys.push(parse_expression(p))
            p._expect(TokenKind.Colon, "`:` between a dict key and its value")
            p._skip_newlines()
            values.push(parse_expression(p))
            p._skip_newlines()
            if p.panic_mode { break }
        }
        _close_sequence(p, TokenKind.RBrace, "}", "dict literal", start_index)
        return _dict_lit(p._span_since(start_index), keys, values)
    }

    if p._check(TokenKind.KwFor) {
        return _finish_comprehension(
            p,
            start_index,
            CompKind.Set,
            null,
            first,
            TokenKind.RBrace,
            "}",
            "set comprehension"
        )
    }
    let items = _parse_rest_of_sequence(p, first, TokenKind.RBrace)
    _close_sequence(p, TokenKind.RBrace, "}", "set literal", start_index)
    return _sequence(NodeKind.SetLit, p._span_since(start_index), items)
}

pub fn _parse_paren_expr(p: Parser) -> Node {
    let start_index = p.current
    p._advance()
    p._skip_newlines()

    if p._match(TokenKind.RParen) {
        return _sequence(NodeKind.TupleLit, p._span_since(start_index), [])
    }

    let first = parse_expression(p)
    p._skip_newlines()

    if p._check(TokenKind.KwFor) {
        return _finish_comprehension(
            p,
            start_index,
            CompKind.Generator,
            null,
            first,
            TokenKind.RParen,
            ")",
            "generator expression"
        )
    }

    if p._check(TokenKind.Comma) {
        let items = _parse_rest_of_sequence(p, first, TokenKind.RParen)
        _close_sequence(p, TokenKind.RParen, ")", "tuple literal", start_index)
        return _sequence(NodeKind.TupleLit, p._span_since(start_index), items)
    }

    _close_sequence(p, TokenKind.RParen, ")", "parenthesised expression", start_index)
    return first
}

# --- lambdas ---------------------------------------------------------
pub fn _parse_lambda(p: Parser) -> Node {
    let start_index = p.current
    p._advance()

    var params: list[dict[str, any]] = []
    if not p._check(TokenKind.Pipe) {
        loop {
            let name_token = p._peek()
            let name = p._expect_ident_name("a lambda parameter name")
            if name == null { break }
            var declared: TypeExpr? = null
            if p._match(TokenKind.Colon) { declared = _parse_type(p) }
            p._check_duplicate_param(params, name, name_token.span)
            params.push(_param(name, declared, null, name_token.span, false, false))
            if not p._match(TokenKind.Comma) { break }
            if p._check(TokenKind.Eof) { break }
        }
    }
    p._expect(TokenKind.Pipe, "`|` to close the lambda parameter list")

    let saved_yield = p.yield_count
    p.yield_count = 0
    let body = parse_expression(p)
    let is_generator = p.yield_count > 0
    p.yield_count = saved_yield

    return _node(
        NodeKind.Lambda,
        p._span_since(start_index),
        {
            "name": null,
            "generics": [],
            "params": params,
            "returnType": null,
            "body": body,
            "visibility": Visibility.Private,
            "isStatic": false,
            "isExprBody": true,
            "isGenerator": is_generator,
            "isAsync": false,
        }
    )
}

# --- primary ---------------------------------------------------------
pub fn _parse_primary(p: Parser) -> Node {
    let start_index = p.current
    let token = p._peek()
    let kind = token.kind

    if kind == TokenKind.Int {
        p._advance()
        return _node(NodeKind.IntLit, token.span, {"intLit": token.value ?? 0})
    }
    if kind == TokenKind.Float {
        p._advance()
        return _node(NodeKind.FloatLit, token.span, {"floatLit": token.value ?? 0.0})
    }
    if kind == TokenKind.Str {
        p._advance()
        return _str_lit(token.span, token.string_value())
    }
    if kind == TokenKind.FStringStart or kind == TokenKind.FStringEnd {
        return _parse_fstring(p)
    }
    if kind == TokenKind.KwThrow {
        p._advance()
        let operand = parse_expression(p)
        return _wrap(NodeKind.ThrowExpr, p._span_since(start_index), operand)
    }
    if kind == TokenKind.KwTrue or kind == TokenKind.KwFalse {
        p._advance()
        return _node(NodeKind.BoolLit, token.span, {"boolLit": kind == TokenKind.KwTrue})
    }
    if kind == TokenKind.KwNull {
        p._advance()
        return _node(NodeKind.NullLit, token.span)
    }
    if kind == TokenKind.KwSelf {
        p._advance()
        return _node(NodeKind.Self, token.span)
    }
    if kind == TokenKind.KwSuper {
        p._advance()
        return _node(NodeKind.Super, token.span)
    }
    if kind == TokenKind.Underscore {
        p._advance()
        return _ident(token.span, "_")
    }
    if kind == TokenKind.LParen { return _parse_paren_expr(p) }
    if kind == TokenKind.LBracket { return _parse_list_literal(p) }
    if kind == TokenKind.LBrace { return _parse_brace_literal(p) }
    if kind == TokenKind.Pipe { return _parse_lambda(p) }
    if kind == TokenKind.KwIf { return parse_if(p, true) }
    if kind == TokenKind.KwMatch { return parse_match(p, true) }
    if kind == TokenKind.KwFn {
        let anonymous = parse_fn_rest(p, Visibility.Private, false, start_index, false, false)
        return anonymous ?? p._error_expr(token.span)
    }
    if kind == TokenKind.KwYield {
        p._advance()
        p.yield_count += 1
        let operand = p._starts_expression() ? parse_expression(p) : null
        return _node(NodeKind.Yield, p._span_since(start_index), {"operand": operand})
    }
    if kind == TokenKind.Ident { return _parse_ident_primary(p, token, start_index) }

    let diag = p._error_at(
        token.span,
        Code.E0102_EXPECTED_EXPRESSION,
        f"expected an expression, found {p._describe(token)}"
    )
    if diag != null {
        if kind == TokenKind.KwElse or kind == TokenKind.KwElif {
            diag.with_help(f"put `{token.text}` on the same line as the closing `}}`")
        } elif kind == TokenKind.Eq {
            diag.with_help("`==` compares, `=` assigns")
        }
    }
    return p._error_expr(token.span)
}

pub fn _parse_ident_primary(p: Parser, token: Token, start_index: int) -> Node {
    if token.text == "new" and p._kind_at(1) == TokenKind.Ident {
        let class_name = p._token_at(p.current + 1).text
        p._error_recovered(
            token.span,
            Code.E0100_UNEXPECTED_TOKEN,
            "the `new` keyword was removed in Jaithon 3",
            help: f"a class is callable: write `{class_name}(...)`"
        )
        p._advance()
        return _parse_primary(p)
    }
    if token.text == "await" and _starts_await_operand(p._kind_at(1)) {
        p._advance()
        let operand = parse_unary(p)
        return _wrap(NodeKind.Await, p._span_since(start_index), operand)
    }
    p._advance()
    return _ident(token.span, token.text)
}

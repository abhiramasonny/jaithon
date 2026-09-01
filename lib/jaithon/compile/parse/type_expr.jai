#: Written type annotations: the primary type forms, the postfix `?` and `[]`
#: wrappers, and the comma-separated list a `throws` clause or a base list asks
#: for.
#:
#: The one production group of `parse/` that reaches nothing above itself. It
#: asks `state.jai` for tokens and `node_build.jai` for a `TypeExpr`, and stops
#: there, which is why `primary.jai`, `stmt.jai` and `decl.jai` can all call
#: into it without a seam.
#:
#: What is NOT here: an expression. A written type never contains one, so
#: nothing in this file reaches the ladder.
from jaithon.ast import TypeExpr, TypeKind
from jaithon.compile.diag import Code
from jaithon.compile.token import TokenKind
from jaithon.compile.parse.state import Parser
from jaithon.compile.parse.node_build import _type

pub fn _parse_primary_type(p: Parser) -> TypeExpr {
    let start_index = p.current

    if p._check(TokenKind.KwFn) {
        p._advance()
        var params: list[TypeExpr] = []
        p._expect(TokenKind.LParen, "`(` after `fn` in a function type")
        p._skip_newlines()
        if not p._check(TokenKind.RParen) {
            loop {
                p._skip_newlines()
                params.push(_parse_type(p))
                p._skip_newlines()
                if not p._match(TokenKind.Comma) { break }
                if p._check(TokenKind.RParen) or p._check(TokenKind.Eof) { break }
            }
        }
        p._expect(TokenKind.RParen, "`)` to close the parameter list")
        if not p._match(TokenKind.Arrow) {
            p._error_at_cur(
                Code.E0101_EXPECTED_TOKEN,
                "expected `->` and a return type in a function type"
            )
            let missing = _type(TypeKind.Infer, p._cur_span())
            return _type(TypeKind.Fn, p._span_since(start_index), args: params, inner: missing)
        }
        let result = _parse_type(p)
        return _type(TypeKind.Fn, p._span_since(start_index), args: params, inner: result)
    }

    if p._check(TokenKind.LParen) {
        p._advance()
        var members: list[TypeExpr] = []
        var trailing_comma = false
        p._skip_newlines()
        if not p._check(TokenKind.RParen) {
            loop {
                p._skip_newlines()
                members.push(_parse_type(p))
                p._skip_newlines()
                if not p._match(TokenKind.Comma) { break }
                p._skip_newlines()
                if p._check(TokenKind.RParen) {
                    trailing_comma = true
                    break
                }
                if p._check(TokenKind.Eof) { break }
            }
        }
        p._expect(TokenKind.RParen, "`)` to close the tuple type")
        if members.len() == 1 and not trailing_comma { return members[0] }
        return _type(TypeKind.Tuple, p._span_since(start_index), args: members)
    }

    if p._match(TokenKind.Underscore) {
        return _type(TypeKind.Infer, p._span_since(start_index))
    }
    if p._match(TokenKind.KwNull) {
        return _type(TypeKind.Name, p._span_since(start_index), name: "null")
    }
    if p._check(TokenKind.Int) {
        let token = p._advance()
        return _type(TypeKind.Const, p._span_since(start_index), name: token.text)
    }

    if not p._check(TokenKind.Ident) {
        p._error_at(
            p._cur_span(),
            Code.E0105_EXPECTED_TYPE,
            f"expected a type, found {p._describe(p._peek())}"
        )
        return _type(TypeKind.Infer, p._cur_span())
    }

    var name = p._advance().text
    while p._check(TokenKind.Dot) and p._kind_at(1) == TokenKind.Ident {
        p._advance()
        name = name + "." + p._advance().text
    }

    if not p._check(TokenKind.LBracket) {
        return _type(TypeKind.Name, p._span_since(start_index), name: name)
    }

    let open_index = p.current
    p._advance()
    var args: list[TypeExpr] = []
    p._skip_newlines()
    if not p._check(TokenKind.RBracket) {
        loop {
            p._skip_newlines()
            args.push(_parse_type(p))
            p._skip_newlines()
            if not p._match(TokenKind.Comma) { break }
            p._skip_newlines()
            if p._check(TokenKind.RBracket) or p._check(TokenKind.Eof) { break }
        }
    }
    if p._check(TokenKind.RBracket) {
        p._advance()
    } else {
        p._unclosed(open_index, "]", "type argument list")
    }
    return _type(TypeKind.Generic, p._span_since(start_index), name: name, args: args)
}

pub fn _parse_postfix_type(p: Parser) -> TypeExpr {
    let start_index = p.current
    var result = _parse_primary_type(p)
    while p._match(TokenKind.Question) {
        result = _type(TypeKind.Optional, p._span_since(start_index), inner: result)
    }
    return result
}

pub fn _parse_type(p: Parser) -> TypeExpr {
    if not p._enter_recursion() {
        p.expr_depth -= 1
        return _type(TypeKind.Infer, p._cur_span())
    }
    defer { p.expr_depth -= 1 }

    let start_index = p.current
    let first = _parse_postfix_type(p)
    if not p._check(TokenKind.Pipe) { return first }

    var parts: list[TypeExpr] = [first]
    while p._match(TokenKind.Pipe) {
        parts.push(_parse_postfix_type(p))
        if p._check(TokenKind.Eof) { break }
    }
    return _type(TypeKind.Union, p._span_since(start_index), args: parts)
}

pub fn _parse_type_list(p: Parser) -> list[TypeExpr] {
    var types: list[TypeExpr] = []
    loop {
        let parsed = _parse_type(p)
        if parsed.kind == TypeKind.Union {
            for member in parsed.args { types.push(member) }
        } else {
            types.push(parsed)
        }
        if not p._match(TokenKind.Comma) { break }
        if p._check(TokenKind.Eof) or p.panic_mode { break }
    }
    return types
}

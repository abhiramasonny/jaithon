#: Statements and the blocks that hold them: `if`, `while`, `loop`, `for`,
#: `match`, `try`, assignment and expression statements, the lookahead that
#: tells a destructuring `a, b = ...` from an ordinary expression, and the
#: legacy bare-call form.
#:
#: What is NOT here: declarations. `_parse_statement` and `_parse_declaration`
#: call each other, so one of the two had to be the file that is imported and
#: the other the file that imports. Declarations are the larger half and the
#: half that drags in visibility, generics and members, so they became
#: `decl.jai`, and the two edges that run back this way -- a block full of
#: declarations, and the `fn` a legacy call statement turns out to be -- go
#: through the seams `state.jai` declares.
from jaithon.ast import Node, NodeKind, OpKind, TypeExpr, Visibility
from jaithon.compile.diag import Code
from jaithon.compile.token import TokenKind
from jaithon.compile.parse.state import Parser, parse_declaration, parse_fn_rest
from jaithon.compile.parse.node_build import _arg, _arm, _catch, _ident, _match_node, _node
from jaithon.compile.parse.token_class import _assign_op
from jaithon.compile.parse.predicate import _is_assign_target, _starts_bare_argument
from jaithon.compile.parse.type_expr import _parse_type_list
from jaithon.compile.parse.primary import _close_sequence
from jaithon.compile.parse.expr import _parse_expression
from jaithon.compile.parse.pattern import _expr_to_pattern, _parse_pattern

pub fn _parse_block(p: Parser) -> Node {
    let start_index = p.current
    p._expect(TokenKind.LBrace, "`{` to open a block")

    var stmts: list[Node] = []
    p._skip_newlines()
    while not p._check(TokenKind.RBrace) and not p._check(TokenKind.Eof) {
        let before = p.current
        let stmt = parse_declaration(p)
        if stmt != null { stmts.push(stmt) }
        if p.panic_mode { p._synchronize() }
        if p.current == before { p._advance() }
        p._skip_newlines()
    }
    _close_sequence(p, TokenKind.RBrace, "}", "block", start_index)
    return _node(NodeKind.Block, p._span_since(start_index), {"stmts": stmts})
}

pub fn _empty_block(p: Parser) -> Node {
    return _node(NodeKind.Block, p._cur_span(), {"stmts": []})
}

pub fn _expect_block(p: Parser, what: str) -> Node {
    if p._check(TokenKind.LBrace) { return _parse_block(p) }

    if p._ident_text(0, "then") or p._ident_text(0, "do") {
        p._error_at(
            p._cur_span(),
            Code.E0104_EXPECTED_BLOCK,
            f"`{p._peek().text}` is not a Jaithon 3 keyword",
            help: f"blocks are braces: `{what} {{ ... }}`"
        )
        return _empty_block(p)
    }

    p._error_at(
        p._cur_span(),
        Code.E0104_EXPECTED_BLOCK,
        f"expected `{{` to open the {what} body, " + f"found {p._describe(p._peek())}"
    )
    return _empty_block(p)
}

pub fn _parse_if(p: Parser, as_expr: bool) -> Node {
    let start_index = p.current
    p._advance()
    let cond = _parse_expression(p)
    let then_branch = _expect_block(p, "if")
    var else_branch: Node? = null

    if p._check(TokenKind.KwElif) {
        else_branch = _parse_if(p, as_expr)
    } elif p._match(TokenKind.KwElse) {
        if p._check(TokenKind.KwIf) {
            else_branch = _parse_if(p, as_expr)
        } else {
            else_branch = _expect_block(p, "else")
        }
    }

    let kind = as_expr ? NodeKind.IfExpr : NodeKind.If
    return _node(
        kind,
        p._span_since(start_index),
        {"cond": cond, "thenBranch": then_branch, "elseBranch": else_branch}
    )
}

pub fn _parse_while(p: Parser, label: str?, start_index: int) -> Node {
    p._advance()
    let cond = _parse_expression(p)
    let body = _expect_block(p, "while")

    return _node(
        NodeKind.While,
        p._span_since(start_index),
        {"cond": cond, "body": body, "label": label}
    )
}

pub fn _parse_loop(p: Parser, label: str?, start_index: int) -> Node {
    p._advance()
    let body = _expect_block(p, "loop")

    return _node(
        NodeKind.Loop,
        p._span_since(start_index),
        {"cond": null, "body": body, "label": label}
    )
}

pub fn _parse_for(p: Parser, label: str?, start_index: int) -> Node {
    p._advance()
    let pattern = _parse_pattern(p)
    p._expect(TokenKind.KwIn, "`in` after the loop pattern")
    let iterable = _parse_expression(p)
    let body = _expect_block(p, "for")

    return _node(
        NodeKind.For,
        p._span_since(start_index),
        {"pattern": pattern, "iterable": iterable, "body": body, "label": label}
    )
}

pub fn _parse_match(p: Parser, as_expr: bool) -> Node {
    let start_index = p.current
    p._advance()
    let subject = _parse_expression(p)
    let kind = as_expr ? NodeKind.MatchExpr : NodeKind.Match

    var arms: list[dict[str, any]] = []
    let open_index = p.current
    if not p._expect(TokenKind.LBrace, "`{` to open the match body") {
        return _match_node(kind, p._span_since(start_index), subject, arms)
    }

    p._skip_newlines()
    while not p._check(TokenKind.RBrace) and not p._check(TokenKind.Eof) {
        let before = p.current
        let pattern = _parse_pattern(p)
        var guard: Node? = null
        if p._match(TokenKind.KwIf) { guard = _parse_expression(p) }
        p._expect(TokenKind.FatArrow, "`=>` after a match pattern")
        let is_block = p._check(TokenKind.LBrace)
        let body = is_block ? _parse_block(p) : _parse_expression(p)
        arms.push(_arm(pattern, guard, body, p._span_since(before)))

        if p.panic_mode {
            p._synchronize()
        } else {
            p._match(TokenKind.Comma)
        }
        if p.current == before { p._advance() }
        p._skip_newlines()
    }
    _close_sequence(p, TokenKind.RBrace, "}", "match body", open_index)
    return _match_node(kind, p._span_since(start_index), subject, arms)
}

pub fn _parse_try(p: Parser) -> Node {
    let start_index = p.current
    p._advance()
    let body = _expect_block(p, "try")

    var catches: list[dict[str, any]] = []
    while p._check(TokenKind.KwCatch) {
        let catch_start = p.current
        p._advance()
        var name: str? = null
        var types: list[TypeExpr] = []
        if p._check(TokenKind.Ident) or p._check(TokenKind.Underscore) {
            name = p._expect_ident_name("an exception binding name")
        }
        if p._match(TokenKind.Colon) { types = _parse_type_list(p) }
        let handler_body = _expect_block(p, "catch")
        catches.push(_catch(name, types, handler_body, p._span_since(catch_start)))
        if p.panic_mode { break }
    }

    var finally_block: Node? = null
    if p._match(TokenKind.KwFinally) { finally_block = _expect_block(p, "finally") }

    if catches.len() == 0 and finally_block == null {
        p._error_at_cur(
            Code.E0101_EXPECTED_TOKEN,
            "a `try` block needs at least one `catch` or a `finally`"
        )
    }

    return _node(
        NodeKind.Try,
        p._span_since(start_index),
        {"body": body, "catches": catches, "finallyBlock": finally_block}
    )
}

pub fn _looks_like_destructuring(p: Parser) -> bool {
    let open = p._kind()
    if open != TokenKind.LParen and open != TokenKind.LBracket { return false }
    let closer = open == TokenKind.LParen ? TokenKind.RParen : TokenKind.RBracket

    var depth = 0
    var index = p.current
    loop {
        let kind = p._token_at(index).kind
        if kind == TokenKind.Eof { return false }
        if kind == TokenKind.LParen or kind == TokenKind.LBracket or kind == TokenKind.LBrace {
            depth += 1
        } elif kind == TokenKind.RParen or kind == TokenKind.RBracket or kind == TokenKind.RBrace {
            depth -= 1
            if depth == 0 {
                if kind != closer { return false }
                return p._token_at(index + 1).kind == TokenKind.Eq
            }
            if depth < 0 { return false }
        }
        index += 1
    }
}

pub fn _parse_expr_or_assign(p: Parser) -> Node {
    let start_index = p.current
    let destructuring = _looks_like_destructuring(p)
    var target = destructuring ? _parse_pattern(p) : _parse_expression(p)

    let compound_op = _assign_op(p._kind())
    let compound = compound_op != null
    if not compound and not p._check(TokenKind.Eq) {
        p._end_statement()
        return _node(NodeKind.ExprStmt, p._span_since(start_index), {"expr": target})
    }

    let op_span = p._cur_span()
    p._advance()

    if not compound and (target.kind == NodeKind.TupleLit or target.kind == NodeKind.ListLit) {
        let as_pattern = _expr_to_pattern(p, target)
        if as_pattern != null { target = as_pattern }
    }

    if not _is_assign_target(target) {
        let diag = p._error_at(
            target.span,
            Code.E0109_INVALID_ASSIGN_TARGET,
            "this expression cannot be assigned to"
        )
        if diag != null {
            diag.label(op_span, "assignment happens here")
            if target.kind == NodeKind.Call {
                diag.with_help("a call result is a value; assign to a name, field, or index")
            }
        }
    }

    let value = _parse_expression(p)
    p._end_statement()
    return _node(
        NodeKind.Assign,
        p._span_since(start_index),
        {
            "target": target,
            "value": value,
            "op": compound_op ?? OpKind.Add,
            "isCompound": compound,
        }
    )
}

pub fn _parse_legacy_call_statement(p: Parser, name: str) -> Node {
    let start_index = p.current
    let name_token = p._advance()
    let diag = p._error_recovered(
        name_token.span,
        Code.E0100_UNEXPECTED_TOKEN,
        f"`{name}` is a function in Jaithon 3"
    )

    var args: list[dict[str, any]] = []
    let arg_start = p.current
    loop {
        let one = p.current
        let value = _parse_expression(p)
        args.push(_arg(null, value, false, p._span_since(one)))
        if not p._match(TokenKind.Comma) { break }
        if not p._starts_expression() or p.panic_mode { break }
    }
    if diag != null {
        let written = p.lexer.text_of(p._span_since(arg_start))
        diag.with_help(f"write `{name}({written})`")
    }

    let call = _node(
        NodeKind.Call,
        p._span_since(start_index),
        {"callee": _ident(name_token.span, name_token.text), "args": args}
    )
    let stmt = _node(NodeKind.ExprStmt, call.span, {"expr": call})
    p._end_statement()
    return stmt
}

pub fn _check_legacy_statement(p: Parser) -> tuple[bool, Node?] {
    if not p._check(TokenKind.Ident) { return (false, null) }
    let text = p._peek().text

    if text == "print" or text == "input" or text == "system" {
        if not _starts_bare_argument(p._kind_at(1)) { return (false, null) }
        return (true, _parse_legacy_call_statement(p, text))
    }

    if text == "end" {
        let next = p._kind_at(1)
        if next != TokenKind.Newline and next != TokenKind.Eof and next != TokenKind.RBrace {
            return (false, null)
        }
        p._error_at(
            p._cur_span(),
            Code.E0100_UNEXPECTED_TOKEN,
            "`end` is not a Jaithon 3 keyword",
            help: "blocks close with `}`"
        )
        p._advance()
        return (true, null)
    }

    if (text == "func" or text == "def") and p._kind_at(1) == TokenKind.Ident {
        p._error_recovered(
            p._cur_span(),
            Code.E0100_UNEXPECTED_TOKEN,
            f"`{text}` is not a Jaithon 3 keyword",
            help: "functions are declared with `fn`"
        )
        return (true, parse_fn_rest(p, Visibility.Private, false, p.current, true, false))
    }

    return (false, null)
}

pub fn _parse_statement(p: Parser) -> Node? {
    let start_index = p.current
    let kind = p._kind()

    if kind == TokenKind.LBrace { return _parse_block(p) }
    if kind == TokenKind.KwIf { return _parse_if(p, false) }
    if kind == TokenKind.KwWhile { return _parse_while(p, null, start_index) }
    if kind == TokenKind.KwLoop { return _parse_loop(p, null, start_index) }
    if kind == TokenKind.KwFor { return _parse_for(p, null, start_index) }
    if kind == TokenKind.KwMatch { return _parse_match(p, false) }
    if kind == TokenKind.KwTry { return _parse_try(p) }

    if kind == TokenKind.Label {
        let label = p._label_name(p._advance())
        p._expect(TokenKind.Colon, "`:` after a loop label")
        if p._check(TokenKind.KwWhile) { return _parse_while(p, label, start_index) }
        if p._check(TokenKind.KwLoop) { return _parse_loop(p, label, start_index) }
        if p._check(TokenKind.KwFor) { return _parse_for(p, label, start_index) }
        p._error_at(
            p._span_since(start_index),
            Code.E0115_LABEL_NOT_ON_LOOP,
            f"`'{label}` labels something that is not a loop",
            help: "only `while`, `loop`, and `for` may be labelled"
        )
        return null
    }

    if kind == TokenKind.KwReturn {
        p._advance()
        let value = p._starts_expression() ? _parse_expression(p) : null
        p._end_statement()
        return _node(NodeKind.Return, p._span_since(start_index), {"value": value})
    }
    if kind == TokenKind.KwThrow {
        p._advance()
        let value = _parse_expression(p)
        p._end_statement()
        return _node(NodeKind.Throw, p._span_since(start_index), {"value": value})
    }
    if kind == TokenKind.KwBreak or kind == TokenKind.KwContinue {
        p._advance()
        var label: str? = null
        if p._check(TokenKind.Label) { label = p._label_name(p._advance()) }
        p._end_statement()
        let jump = kind == TokenKind.KwBreak ? NodeKind.Break : NodeKind.Continue
        return _node(jump, p._span_since(start_index), {"label": label})
    }
    if kind == TokenKind.KwDefer {
        p._advance()
        let body = _expect_block(p, "defer")
        return _node(NodeKind.Defer, p._span_since(start_index), {"body": body})
    }
    if kind == TokenKind.KwAssert {
        p._advance()
        let cond = _parse_expression(p)
        let message = p._match(TokenKind.Comma) ? _parse_expression(p) : null
        p._end_statement()
        return _node(
            NodeKind.Assert,
            p._span_since(start_index),
            {"cond": cond, "message": message}
        )
    }

    if kind == TokenKind.KwElse or
        kind == TokenKind.KwElif or
        kind == TokenKind.KwCatch or
        kind == TokenKind.KwFinally {
        let word = p._peek().text
        p._error_at(
            p._cur_span(),
            Code.E0100_UNEXPECTED_TOKEN,
            f"`{word}` has no matching statement before it",
            help: f"put `{word}` on the same line as the closing `}}`"
        )
        return null
    }

    if kind == TokenKind.KwImpl {
        p._error_at(
            p._cur_span(),
            Code.E0100_UNEXPECTED_TOKEN,
            "`impl` blocks are not part of Jaithon 3",
            help: "declare methods inside the class body and list traits with `:`"
        )
        return null
    }

    if kind == TokenKind.Ident {
        let (handled, legacy) = _check_legacy_statement(p)
        if handled { return legacy }
    }

    return _parse_expr_or_assign(p)
}

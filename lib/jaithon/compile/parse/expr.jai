#: The precedence ladder of `spec/LANGUAGE.md` 4.1, top rung to bottom:
#: coalesce, `or`, `and`, `not`, the comparison chain, range, the three bitwise
#: levels, shift, additive, multiplicative, unary, power, and the postfix and
#: subscript forms the descent ends on.
#:
#: Almost every function here is one rung that calls the next, so the file
#: reads as a single descent and is best read that way.
#:
#: What is NOT here: the operands the ladder lands on, which are `primary.jai`;
#: and the TokenKind-to-OpKind tables a rung consults, which are
#: `token_class.jai` -- "is this token my operator?" is a question with no
#: parser state in it.
from jaithon.ast import Node, NodeKind, OpKind
from jaithon.compile.diag import Code
from jaithon.compile.token import TokenFlags, TokenKind
from jaithon.compile.parse.state import Parser
from jaithon.compile.parse.node_build import _binary, _join, _member, _node, _unary
from jaithon.compile.parse.token_class import _additive_op, _compare_op, _multiplicative_op
from jaithon.compile.parse.predicate import _is_numeric_literal
from jaithon.compile.parse.primary import _close_sequence, _parse_args, _parse_primary

pub fn _parse_subscript(p: Parser, object: Node, open_index: int) -> Node {
    p._advance()
    var start: Node? = null
    if not p._check(TokenKind.Colon) {
        if p._check(TokenKind.RBracket) {
            p._error_at_cur(Code.E0102_EXPECTED_EXPRESSION, "expected an index expression")
            p._advance()
            return p._error_expr(_join(object.span, p._span_since(open_index)))
        }
        start = _parse_expression(p)
    }

    if p._match(TokenKind.Colon) {
        var stop: Node? = null
        var step: Node? = null
        if not p._check(TokenKind.Colon) and not p._check(TokenKind.RBracket) {
            stop = _parse_expression(p)
        }
        if p._match(TokenKind.Colon) {
            if not p._check(TokenKind.RBracket) { step = _parse_expression(p) }
        }
        _close_sequence(p, TokenKind.RBracket, "]", "slice", open_index)
        return _node(
            NodeKind.Slice,
            _join(object.span, p._span_since(open_index)),
            {"object": object, "start": start, "stop": stop, "step": step}
        )
    }

    if p._check(TokenKind.Comma) {
        var args: list[Node] = [start ?? p._error_expr(p._cur_span())]
        while p._match(TokenKind.Comma) {
            p._skip_newlines()
            if p._check(TokenKind.RBracket) { break }
            args.push(_parse_expression(p))
            if p.panic_mode { break }
        }
        _close_sequence(p, TokenKind.RBracket, "]", "generic argument list", open_index)
        let generic_span = _join(object.span, p._span_since(open_index))
        let tuple = _node(NodeKind.TupleLit, generic_span, {"items": args})
        return _node(NodeKind.Index, generic_span, {"object": object, "index": tuple})
    }

    _close_sequence(p, TokenKind.RBracket, "]", "index expression", open_index)
    let span = _join(object.span, p._span_since(open_index))
    return _node(NodeKind.Index, span, {"object": object, "index": start ?? p._error_expr(span)})
}

pub fn _parse_postfix(p: Parser) -> Node {
    var expr = _parse_primary(p)

    loop {
        if p._check(TokenKind.Dot) or p._check(TokenKind.QuestionDot) {
            let optional = p._check(TokenKind.QuestionDot)
            p._advance()
            let name_token = p._peek()
            let name = p._expect_ident_name("a member name after `.`")
            if name == null { break }
            let kind = optional ? NodeKind.OptMember : NodeKind.Member
            expr = _member(kind, _join(expr.span, name_token.span), expr, name)
        } elif p._check(TokenKind.LParen) {
            let open_index = p.current
            p._advance()
            let args = _parse_args(p, open_index)
            expr = _node(
                NodeKind.Call,
                _join(expr.span, p._span_since(open_index)),
                {"callee": expr, "args": args}
            )
        } elif p._check(TokenKind.LBracket) {
            expr = _parse_subscript(p, expr, p.current)
        } else {
            break
        }
        if p.panic_mode { break }
    }
    return expr
}

# --- the precedence ladder (§4.1) ------------------------------------
pub fn _parse_power(p: Parser) -> Node {
    let left = _parse_postfix(p)
    if not p._match(TokenKind.StarStar) { return left }
    return _binary(NodeKind.Binary, OpKind.Pow, left, _parse_unary(p))
}

pub fn _parse_neg_magnitude(p: Parser) -> Node? {
    let literal = p._token_at(p.current + 1)
    if literal.kind != TokenKind.Int or not literal.has_flag(TokenFlags.NEG_MAGNITUDE) {
        return null
    }
    let start_index = p.current
    p._advance()
    p._advance()
    return _node(NodeKind.IntLit, p._span_since(start_index), {"intLit": literal.value ?? 0})
}

pub fn _parse_unary(p: Parser) -> Node {
    if not p._enter_recursion() {
        p.expr_depth -= 1
        return p._error_expr(p._cur_span())
    }
    defer { p.expr_depth -= 1 }

    let kind = p._kind()
    if kind == TokenKind.Minus {
        let folded = _parse_neg_magnitude(p)
        if folded != null { return folded ?? p._error_expr(p._cur_span()) }
    }
    if kind != TokenKind.Minus and kind != TokenKind.Plus and kind != TokenKind.Tilde {
        return _parse_power(p)
    }
    let start_index = p.current
    p._advance()
    let operand = _parse_unary(p)
    var op = OpKind.Bnot
    if kind == TokenKind.Minus {
        op = OpKind.Neg
    } elif kind == TokenKind.Plus {
        op = OpKind.Pos
    }
    return _unary(p._span_since(start_index), op, operand)
}

pub fn _parse_multiplicative(p: Parser) -> Node {
    var left = _parse_unary(p)
    loop {
        let op = _multiplicative_op(p._kind())
        if op == null { return left }
        p._advance()
        left = _binary(NodeKind.Binary, op ?? OpKind.Mul, left, _parse_unary(p))
        if p.panic_mode { return left }
    }
}

pub fn _parse_additive(p: Parser) -> Node {
    var left = _parse_multiplicative(p)
    loop {
        let op = _additive_op(p._kind())
        if op == null { return left }
        p._advance()
        left = _binary(NodeKind.Binary, op ?? OpKind.Add, left, _parse_multiplicative(p))
        if p.panic_mode { return left }
    }
}

pub fn _parse_shift(p: Parser) -> Node {
    var left = _parse_additive(p)
    loop {
        let kind = p._kind()
        if kind != TokenKind.Shl and kind != TokenKind.Shr { return left }
        p._advance()
        let op = kind == TokenKind.Shl ? OpKind.Shl : OpKind.Shr
        left = _binary(NodeKind.Binary, op, left, _parse_additive(p))
        if p.panic_mode { return left }
    }
}

pub fn _parse_bit_and(p: Parser) -> Node {
    var left = _parse_shift(p)
    while p._match(TokenKind.Amp) {
        left = _binary(NodeKind.Binary, OpKind.Band, left, _parse_shift(p))
        if p.panic_mode { break }
    }
    return left
}

pub fn _parse_bit_xor(p: Parser) -> Node {
    var left = _parse_bit_and(p)
    while p._check(TokenKind.Caret) {
        let op_span = p._cur_span()
        p._advance()
        let right = _parse_bit_and(p)
        if not p.trial_active and _is_numeric_literal(left) and _is_numeric_literal(right) {
            p.diagnostics.warn(
                Code.W0105_DEPRECATED,
                op_span,
                "`^` is bitwise xor in Jaithon 3, not exponentiation",
                note: "the power operator is `**`"
            )
        }
        left = _binary(NodeKind.Binary, OpKind.Bxor, left, right)
        if p.panic_mode { break }
    }
    return left
}

pub fn _parse_bit_or(p: Parser) -> Node {
    var left = _parse_bit_xor(p)
    while p._match(TokenKind.Pipe) {
        left = _binary(NodeKind.Binary, OpKind.Bor, left, _parse_bit_xor(p))
        if p.panic_mode { break }
    }
    return left
}

pub fn _parse_range(p: Parser) -> Node {
    let left = _parse_bit_or(p)
    if not p._check(TokenKind.DotDot) and not p._check(TokenKind.DotDotEq) {
        return left
    }
    let inclusive = p._check(TokenKind.DotDotEq)
    p._advance()
    let stop = p._starts_expression() ? _parse_bit_or(p) : null
    var span = left.span
    if stop != null { span = _join(left.span, stop.span) }
    return _node(NodeKind.Range, span, {"start": left, "stop": stop, "inclusive": inclusive})
}

pub fn _peek_compare_op(p: Parser) -> tuple[OpKind, int, bool]? {
    let kind = p._kind()
    let simple = _compare_op(kind)
    if simple != null { return (simple ?? OpKind.Eq, 1, false) }
    if kind == TokenKind.KwIs {
        if p._kind_at(1) == TokenKind.KwNot { return (OpKind.IsNot, 2, false) }
        return (OpKind.Is, 1, false)
    }
    if kind == TokenKind.KwNot {
        if p._kind_at(1) == TokenKind.KwIn { return (OpKind.NotIn, 2, false) }
        return null
    }
    if kind == TokenKind.Ident {
        let text = p._peek().text
        if text == "eq" { return (OpKind.Eq, 1, true) }
        if text == "ne" or text == "neq" { return (OpKind.Ne, 1, true) }
    }
    return null
}

pub fn _parse_comparison(p: Parser) -> Node {
    let first = _parse_range(p)
    if _peek_compare_op(p) == null { return first }

    var operands: list[Node] = [first]
    var ops: list[OpKind] = []
    loop {
        let found = _peek_compare_op(p)
        if found == null { break }
        let (op, op_tokens, legacy) = found
        if legacy {
            let spelling = op == OpKind.Eq ? "==" : "!="
            p._error_recovered(
                p._cur_span(),
                Code.E0100_UNEXPECTED_TOKEN,
                f"`{p._peek().text}` is not an operator in Jaithon 3",
                help: f"write `{spelling}`"
            )
        }
        for _step in 0..op_tokens { p._advance() }
        ops.push(op)
        operands.push(_parse_range(p))
        if p.panic_mode or p._check(TokenKind.Eof) { break }
    }

    if ops.len() == 1 { return _binary(NodeKind.Binary, ops[0], operands[0], operands[1]) }
    let last = operands[operands.len() - 1]
    return _node(
        NodeKind.CompareChain,
        _join(first.span, last.span),
        {"operands": operands, "ops": ops}
    )
}

pub fn _parse_not(p: Parser) -> Node {
    if not p._check(TokenKind.KwNot) { return _parse_comparison(p) }
    if not p._enter_recursion() {
        p.expr_depth -= 1
        return p._error_expr(p._cur_span())
    }
    defer { p.expr_depth -= 1 }

    let start_index = p.current
    p._advance()
    let operand = _parse_not(p)
    return _unary(p._span_since(start_index), OpKind.Not, operand)
}

pub fn _parse_and(p: Parser) -> Node {
    var left = _parse_not(p)
    while p._match(TokenKind.KwAnd) {
        left = _binary(NodeKind.Logical, OpKind.And, left, _parse_not(p))
        if p.panic_mode { break }
    }
    return left
}

pub fn _parse_or(p: Parser) -> Node {
    var left = _parse_and(p)
    while p._match(TokenKind.KwOr) {
        left = _binary(NodeKind.Logical, OpKind.Or, left, _parse_and(p))
        if p.panic_mode { break }
    }
    return left
}

pub fn _parse_coalesce(p: Parser) -> Node {
    var left = _parse_or(p)
    while p._match(TokenKind.QuestionQuestion) {
        let right = _parse_or(p)
        left = _node(
            NodeKind.Coalesce,
            _join(left.span, right.span),
            {"left": left, "right": right}
        )
        if p.panic_mode { break }
    }
    return left
}

pub fn _parse_expression(p: Parser) -> Node {
    if not p._enter_recursion() {
        p.expr_depth -= 1
        return p._error_expr(p._cur_span())
    }
    defer { p.expr_depth -= 1 }

    let cond = _parse_coalesce(p)
    if not p._match(TokenKind.Question) { return cond }
    let then_expr = _parse_expression(p)
    p._expect(TokenKind.Colon, "`:` in a conditional expression")
    let else_expr = _parse_expression(p)
    return _node(
        NodeKind.Ternary,
        _join(cond.span, else_expr.span),
        {"cond": cond, "thenExpr": then_expr, "elseExpr": else_expr}
    )
}

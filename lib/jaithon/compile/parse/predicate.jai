#: The yes-or-no questions the parser asks before it commits to a production:
#: whether a `TokenKind` can start an expression, a declaration, a pattern
#: literal, an `await` operand or a bare argument, and whether a `Node` is a
#: numeric literal, a legal assignment target, or a reserved dunder name.
#:
#: Grouped by the answer they give rather than by the argument they take, which
#: is why the three that read a `Node` or a `str` sit beside the five that read
#: a `TokenKind`. None of them touches parser state, so every module of
#: `parse/` may ask them.
from jaithon.ast import Node, NodeKind
from jaithon.compile.token import TokenKind

pub fn _starts_expression_kind(kind: TokenKind) -> bool {
    return match kind {
        TokenKind.Int | TokenKind.Float | TokenKind.Str | TokenKind.FStringStart => true,
        TokenKind.Ident | TokenKind.Underscore => true,
        TokenKind.LParen | TokenKind.LBracket | TokenKind.LBrace => true,
        TokenKind.Minus | TokenKind.Plus | TokenKind.Tilde | TokenKind.Pipe => true,
        TokenKind.KwNot | TokenKind.KwTrue | TokenKind.KwFalse | TokenKind.KwNull => true,
        TokenKind.KwSelf | TokenKind.KwSuper | TokenKind.KwFn | TokenKind.KwIf => true,
        TokenKind.KwMatch | TokenKind.KwYield => true,
        TokenKind.KwThrow => true,
        _ => false,
    }
}

pub fn _starts_declaration(kind: TokenKind) -> bool {
    return match kind {
        TokenKind.KwImport | TokenKind.KwFrom | TokenKind.KwExport | TokenKind.KwModule => true,
        TokenKind.KwClass | TokenKind.KwTrait | TokenKind.KwEnum | TokenKind.KwType => true,
        TokenKind.KwFn | TokenKind.KwLet | TokenKind.KwVar | TokenKind.KwConst => true,
        TokenKind.KwPub | TokenKind.KwProt | TokenKind.KwStatic | TokenKind.KwIf => true,
        TokenKind.KwWhile | TokenKind.KwFor | TokenKind.KwLoop | TokenKind.KwMatch => true,
        TokenKind.KwReturn | TokenKind.KwBreak | TokenKind.KwContinue => true,
        TokenKind.KwThrow | TokenKind.KwTry | TokenKind.KwDefer | TokenKind.KwAssert => true,
        _ => false,
    }
}

pub fn _starts_pattern_literal(kind: TokenKind) -> bool {
    return match kind {
        TokenKind.Int | TokenKind.Float | TokenKind.Str | TokenKind.FStringStart => true,
        TokenKind.KwTrue | TokenKind.KwFalse | TokenKind.KwNull | TokenKind.Minus => true,
        _ => false,
    }
}

pub fn _starts_await_operand(kind: TokenKind) -> bool {
    return match kind {
        TokenKind.Ident | TokenKind.KwSelf | TokenKind.Int | TokenKind.Float => true,
        TokenKind.Str | TokenKind.FStringStart => true,
        _ => false,
    }
}

pub fn _starts_bare_argument(kind: TokenKind) -> bool {
    return match kind {
        TokenKind.Ident | TokenKind.Int | TokenKind.Float | TokenKind.Str => true,
        TokenKind.FStringStart | TokenKind.KwTrue | TokenKind.KwFalse => true,
        TokenKind.KwNull | TokenKind.KwSelf => true,
        _ => false,
    }
}

pub fn _is_numeric_literal(node: Node) -> bool {
    return node.kind == NodeKind.IntLit or node.kind == NodeKind.FloatLit
}

pub fn _is_assign_target(node: Node) -> bool {
    return match node.kind {
        NodeKind.Ident | NodeKind.Member | NodeKind.Index | NodeKind.Slice => true,
        NodeKind.PatBind | NodeKind.PatTuple | NodeKind.PatList => true,
        NodeKind.PatWildcard => true,
        _ => false,
    }
}

pub fn _is_reserved_name(name: str) -> bool {
    if name.len() < 5 { return false }
    return name[0:2] == "__" and name[name.len() - 2:name.len()] == "__"
}

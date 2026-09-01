#: Which `OpKind` a `TokenKind` means at a given rung of the precedence ladder:
#: multiplicative, additive, comparison, and the compound assignments.
#:
#: Each answers null for a token that is not an operator at that level, and
#: that null is how a loop in `expr.jai` decides it has just read the last
#: operand of its rung.
#:
#: What is NOT here: a question that answers `bool` rather than an `OpKind`.
#: "Can this token start an expression?" is a different kind of question and
#: lives in `predicate.jai`.
from jaithon.ast import OpKind
from jaithon.compile.token import TokenKind

pub fn _multiplicative_op(kind: TokenKind) -> OpKind? {
    return match kind {
        TokenKind.Star => OpKind.Mul,
        TokenKind.Slash => OpKind.Div,
        TokenKind.SlashSlash => OpKind.FloorDiv,
        TokenKind.Percent => OpKind.Mod,
        TokenKind.At => OpKind.MatMul,
        TokenKind.StarPercent => OpKind.MulWrap,
        _ => null,
    }
}

pub fn _additive_op(kind: TokenKind) -> OpKind? {
    return match kind {
        TokenKind.Plus => OpKind.Add,
        TokenKind.Minus => OpKind.Sub,
        TokenKind.PlusPercent => OpKind.AddWrap,
        TokenKind.MinusPercent => OpKind.SubWrap,
        _ => null,
    }
}

pub fn _compare_op(kind: TokenKind) -> OpKind? {
    return match kind {
        TokenKind.Lt => OpKind.Lt,
        TokenKind.Le => OpKind.Le,
        TokenKind.Gt => OpKind.Gt,
        TokenKind.Ge => OpKind.Ge,
        TokenKind.EqEq => OpKind.Eq,
        TokenKind.BangEq => OpKind.Ne,
        TokenKind.KwIn => OpKind.In,
        _ => null,
    }
}

pub fn _assign_op(kind: TokenKind) -> OpKind? {
    return match kind {
        TokenKind.PlusEq => OpKind.Add,
        TokenKind.MinusEq => OpKind.Sub,
        TokenKind.StarEq => OpKind.Mul,
        TokenKind.SlashEq => OpKind.Div,
        TokenKind.SlashSlashEq => OpKind.FloorDiv,
        TokenKind.PercentEq => OpKind.Mod,
        TokenKind.StarStarEq => OpKind.Pow,
        TokenKind.AmpEq => OpKind.Band,
        TokenKind.PipeEq => OpKind.Bor,
        TokenKind.CaretEq => OpKind.Bxor,
        TokenKind.ShlEq => OpKind.Shl,
        TokenKind.ShrEq => OpKind.Shr,
        _ => null,
    }
}

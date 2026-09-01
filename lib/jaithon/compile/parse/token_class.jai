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

#: The four operator tables are lists indexed by `TokenKind.ordinal()`, built
#: once when the module loads. They were `match` chains: on `check lib/jaithon`
#: those four functions cost 37.3M interpreted instructions, 6.8% of the whole
#: run, because a chain compares one variant at a time and the interpreter
#: cannot do better. An indexed list is one bounds check and one load.
#:
#: `_table` sizes each list by the largest ordinal it maps, so a token kind
#: past the end simply reads as `null`, the same answer the chains gave.
fn _table(pairs: list[tuple[TokenKind, OpKind]]) -> list[OpKind?] {
    var size = 0
    for (kind, _) in pairs {
        if kind.ordinal() + 1 > size { size = kind.ordinal() + 1 }
    }
    var table: list[OpKind?] = []
    for _ in range(size) { table.push(null) }
    for (kind, op) in pairs { table[kind.ordinal()] = op }
    return table
}

fn _lookup(table: list[OpKind?], kind: TokenKind) -> OpKind? {
    let index = kind.ordinal()
    if index >= table.len() { return null }
    return table[index]
}

let _MULTIPLICATIVE_OP: list[OpKind?] = _table(
    [
        (TokenKind.Star, OpKind.Mul),
        (TokenKind.Slash, OpKind.Div),
        (TokenKind.SlashSlash, OpKind.FloorDiv),
        (TokenKind.Percent, OpKind.Mod),
        (TokenKind.At, OpKind.MatMul),
        (TokenKind.StarPercent, OpKind.MulWrap),
    ]
)

let _ADDITIVE_OP: list[OpKind?] = _table(
    [
        (TokenKind.Plus, OpKind.Add),
        (TokenKind.Minus, OpKind.Sub),
        (TokenKind.PlusPercent, OpKind.AddWrap),
        (TokenKind.MinusPercent, OpKind.SubWrap),
    ]
)

let _COMPARE_OP: list[OpKind?] = _table(
    [
        (TokenKind.Lt, OpKind.Lt),
        (TokenKind.Le, OpKind.Le),
        (TokenKind.Gt, OpKind.Gt),
        (TokenKind.Ge, OpKind.Ge),
        (TokenKind.EqEq, OpKind.Eq),
        (TokenKind.BangEq, OpKind.Ne),
        (TokenKind.KwIn, OpKind.In),
    ]
)

let _ASSIGN_OP: list[OpKind?] = _table(
    [
        (TokenKind.PlusEq, OpKind.Add),
        (TokenKind.MinusEq, OpKind.Sub),
        (TokenKind.StarEq, OpKind.Mul),
        (TokenKind.SlashEq, OpKind.Div),
        (TokenKind.SlashSlashEq, OpKind.FloorDiv),
        (TokenKind.PercentEq, OpKind.Mod),
        (TokenKind.StarStarEq, OpKind.Pow),
        (TokenKind.AmpEq, OpKind.Band),
        (TokenKind.PipeEq, OpKind.Bor),
        (TokenKind.CaretEq, OpKind.Bxor),
        (TokenKind.ShlEq, OpKind.Shl),
        (TokenKind.ShrEq, OpKind.Shr),
    ]
)

pub fn _multiplicative_op(kind: TokenKind) -> OpKind? {
    return _lookup(_MULTIPLICATIVE_OP, kind)
}

pub fn _additive_op(kind: TokenKind) -> OpKind? {
    return _lookup(_ADDITIVE_OP, kind)
}

pub fn _compare_op(kind: TokenKind) -> OpKind? {
    return _lookup(_COMPARE_OP, kind)
}

pub fn _assign_op(kind: TokenKind) -> OpKind? {
    return _lookup(_ASSIGN_OP, kind)
}

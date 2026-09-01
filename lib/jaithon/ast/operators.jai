#: The operator vocabulary: `OpKind`, the C `OpKind` name table it is checked
#: against, the source spelling of every operator, and the four accessors over
#: those two tables.
#:
#: A file of its own rather than a section of `kinds.jai` because it shares
#: nothing with it. No operator table is indexed by a `NodeKind` and no kind
#: table by an `OpKind`, the two drift asserts each read only their own pair,
#: and this module therefore imports nothing at all.

#: Every operator code, in the order of the C `OpKind` enum.
#:
#: `AddWrap`, `SubWrap` and `MulWrap` are the wrapping operators `+%`, `-%` and
#: `*%`; `Neg`, `Pos`, `Not` and `Bnot` are the four unary operators.
pub enum OpKind {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
    Pow,
    AddWrap,
    SubWrap,
    MulWrap,
    Band,
    Bor,
    Bxor,
    Shl,
    Shr,
    MatMul,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Is,
    IsNot,
    In,
    NotIn,
    And,
    Or,
    Neg,
    Pos,
    Not,
    Bnot,
}

let _OP_NAMES: list[str] = [
    "OPK_ADD",
    "OPK_SUB",
    "OPK_MUL",
    "OPK_DIV",
    "OPK_FLOORDIV",
    "OPK_MOD",
    "OPK_POW",
    "OPK_ADD_WRAP",
    "OPK_SUB_WRAP",
    "OPK_MUL_WRAP",
    "OPK_BAND",
    "OPK_BOR",
    "OPK_BXOR",
    "OPK_SHL",
    "OPK_SHR",
    "OPK_MATMUL",
    "OPK_EQ",
    "OPK_NE",
    "OPK_LT",
    "OPK_LE",
    "OPK_GT",
    "OPK_GE",
    "OPK_IS",
    "OPK_IS_NOT",
    "OPK_IN",
    "OPK_NOT_IN",
    "OPK_AND",
    "OPK_OR",
    "OPK_NEG",
    "OPK_POS",
    "OPK_NOT",
    "OPK_BNOT",
]

let _OPS: list[OpKind] = [
    OpKind.Add,
    OpKind.Sub,
    OpKind.Mul,
    OpKind.Div,
    OpKind.FloorDiv,
    OpKind.Mod,
    OpKind.Pow,
    OpKind.AddWrap,
    OpKind.SubWrap,
    OpKind.MulWrap,
    OpKind.Band,
    OpKind.Bor,
    OpKind.Bxor,
    OpKind.Shl,
    OpKind.Shr,
    OpKind.MatMul,
    OpKind.Eq,
    OpKind.Ne,
    OpKind.Lt,
    OpKind.Le,
    OpKind.Gt,
    OpKind.Ge,
    OpKind.Is,
    OpKind.IsNot,
    OpKind.In,
    OpKind.NotIn,
    OpKind.And,
    OpKind.Or,
    OpKind.Neg,
    OpKind.Pos,
    OpKind.Not,
    OpKind.Bnot,
]

let _OP_SOURCE: list[str] = [
    "+",
    "-",
    "*",
    "/",
    "//",
    "%",
    "**",
    "+%",
    "-%",
    "*%",
    "&",
    "|",
    "^",
    "<<",
    ">>",
    "@",
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "is",
    "is not",
    "in",
    "not in",
    "and",
    "or",
    "-",
    "+",
    "not",
    "~",
]

let _OP_INDEX: dict[OpKind, int] = {op: index for (index, op) in _OPS.enumerate()}
let _OP_BY_NAME: dict[str, OpKind] = {name: op for (name, op) in zip(_OP_NAMES, _OPS)}

const _FIRST_COMPARISON = 16
const _LAST_COMPARISON = 25

assert _OP_NAMES.len() == _OPS.len(), "the operator tables have different lengths"
assert _OP_SOURCE.len() == _OPS.len(), "the operator source table has the wrong length"

#: Return the C enum name of `op`.
pub fn op_name(op: OpKind) -> str {
    return _OP_NAMES[_OP_INDEX[op]]
}

#: Return the operator whose C name is `name`.
pub fn op_from_name(name: str) -> OpKind {
    let op = _OP_BY_NAME.get(name, null)
    if op is null { throw ValueError(f"unknown operator kind: {name}") }
    return op
}

#: Return the source spelling of `op`.
#:
#: ```
#: op_text(OpKind.Pow)     # "**"
#: op_text(OpKind.IsNot)   # "is not"
#: ```
pub fn op_text(op: OpKind) -> str {
    return _OP_SOURCE[_OP_INDEX[op]]
}

#: Whether `op` is one of the ten comparison operators, which chain.
pub fn op_is_comparison(op: OpKind) -> bool {
    let index = _OP_INDEX[op]
    return index >= _FIRST_COMPARISON and index <= _LAST_COMPARISON
}

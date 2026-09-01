#: The two operator tables: an `OpKind` from the tree to the opcode that
#: performs it.
#:
#: Neither table reports an unknown operator -- `_unary_op` folds `BNot` into
#: its fall-through and `_binary_op` falls back to `Add`. That is the shape
#: both had before the split, and it holds because the checker has settled what
#: every operator means long before the tree reaches this directory.
#:
#: What is NOT here: the stack effect of each opcode, which is `opcode.jai`'s.
from jaithon.ast import OpKind
from jaithon.compile.emit.opcode import Op

pub fn _unary_op(op: OpKind) -> Op {
    if op == OpKind.Neg { return Op.Neg }
    if op == OpKind.Pos { return Op.Pos }
    if op == OpKind.Not { return Op.Not }
    return Op.BNot
}

pub fn _binary_op(op: OpKind) -> Op {
    if op == OpKind.Add { return Op.Add }
    if op == OpKind.Sub { return Op.Sub }
    if op == OpKind.Mul { return Op.Mul }
    if op == OpKind.Div { return Op.Div }
    if op == OpKind.FloorDiv { return Op.FloorDiv }
    if op == OpKind.Mod { return Op.Mod }
    if op == OpKind.Pow { return Op.Pow }
    if op == OpKind.AddWrap { return Op.AddWrap }
    if op == OpKind.SubWrap { return Op.SubWrap }
    if op == OpKind.MulWrap { return Op.MulWrap }
    if op == OpKind.Band { return Op.BAnd }
    if op == OpKind.Bor { return Op.BOr }
    if op == OpKind.Bxor { return Op.BXor }
    if op == OpKind.Shl { return Op.Shl }
    if op == OpKind.Shr { return Op.Shr }
    if op == OpKind.Eq { return Op.Eq }
    if op == OpKind.Ne { return Op.Ne }
    if op == OpKind.Lt { return Op.Lt }
    if op == OpKind.Le { return Op.Le }
    if op == OpKind.Gt { return Op.Gt }
    if op == OpKind.Ge { return Op.Ge }
    if op == OpKind.Is { return Op.Is }
    if op == OpKind.IsNot { return Op.IsNot }
    if op == OpKind.In { return Op.In }
    if op == OpKind.NotIn { return Op.NotIn }
    return Op.Add
}

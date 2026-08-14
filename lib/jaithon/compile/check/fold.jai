#: The constant folder and the compile-time evaluator.
#:
#: A port of `src/sema/check_fold.c`. `const_eval` answers what an expression's
#: value is at compile time, which a `const` initialiser (spec §2.2) and a match
#: arm (§5.3) both need; `try_fold` is the same walk run for its side effect,
#: replacing a node in place with the literal it evaluates to.
#:
#: Arithmetic is overflow-checked rather than wrapping: `2 ** 70` reports
#: `E0900`, `1 // 0` reports `E0901`, and neither folds. Overflow is an error
#: only where a constant is required — the C reads `gJaiCheck.constRequired`,
#: here a `bag` passed in is that flag.
#:
#: A Jaithon `Ident` holds a name and no symbol, so a `const`'s folded
#: initialiser arrives as the `consts` dict, keyed by name, rather than through
#: the C's `symbol->constValue`. Without it `const K = 8` inlines nowhere.
from jaithon.ast import Node, NodeKind, OpKind, Span, op_text
from jaithon.compile.check.assign import assignable
from jaithon.compile.check.kinds import MAX_DECL_DEPTH
from jaithon.compile.check.ty import Type
from jaithon.compile.check.universe import Universe
from jaithon.compile.diag import DiagnosticBag
from std.math import INT_MAX, INT_MIN, floor, fmod
from std.str import StringBuilder

#: An operation whose constant result does not fit an `int` (spec §13).
pub const E0900_CONST_OVERFLOW: str = "E0900"

#: A constant divided or reduced modulo zero.
pub const E0901_CONST_DIVIDE_BY_ZERO: str = "E0901"

#: The C's `ConstKind`: what a constant evaluated to, `None` if it is not one.
pub enum ConstKind { None, Int, Float, Bool, Str, Null }

#: One compile-time value: a kind, and the value that kind implies.
#:
#: There is deliberately no `__str__`: `render_const` is the one spelling both
#: front ends must agree on byte for byte.
pub class ConstValue {
    #: Which of the six kinds this is.
    pub let kind: ConstKind
    #: The value itself: an `int`, `float`, `bool` or `str`, else null.
    pub let value: any

    #: Hold `value` as a constant of `kind`; prefer the constructors below.
    pub fn init(self, kind: ConstKind, value: any = null) {
        self.kind = kind
        self.value = value
    }

    #: The answer for an expression with no compile-time value.
    pub static fn none() -> ConstValue {
        return ConstValue(ConstKind.None)
    }

    #: The integer `i`.
    pub static fn of_int(i: int) -> ConstValue {
        return ConstValue(ConstKind.Int, i)
    }

    #: The float `f`.
    pub static fn of_float(f: float) -> ConstValue {
        return ConstValue(ConstKind.Float, f)
    }

    #: The boolean `b`.
    pub static fn of_bool(b: bool) -> ConstValue {
        return ConstValue(ConstKind.Bool, b)
    }

    #: The string `s`.
    pub static fn of_str(s: str) -> ConstValue {
        return ConstValue(ConstKind.Str, s)
    }

    #: The `null` literal, which is a constant like any other.
    pub static fn of_null() -> ConstValue {
        return ConstValue(ConstKind.Null)
    }

    #: Whether this is the "not a constant" answer.
    pub fn is_none(self) -> bool {
        return self.kind == ConstKind.None
    }

    #: Whether this is an `int` or a `float`, the two kinds arithmetic accepts.
    pub fn is_numeric(self) -> bool {
        return self.kind == ConstKind.Int or self.kind == ConstKind.Float
    }

    #: The integer held here; raises `ValueError` for any other kind.
    pub fn as_int(self) -> int {
        if self.kind != ConstKind.Int { throw ValueError(f"a {self.kind} constant has no int") }
        let out: int = self.value
        return out
    }

    #: The float held here, widening an `int` — the C's `constAsFloat`. The
    #: widening is what makes a mixed pair fold as two floats (§2.5).
    pub fn as_float(self) -> float {
        if self.kind == ConstKind.Int { return float(self.as_int()) }
        if self.kind != ConstKind.Float { throw ValueError(f"a {self.kind} constant has no float") }
        let out: float = self.value
        return out
    }

    #: The boolean held here; raises `ValueError` for any other kind.
    pub fn as_bool(self) -> bool {
        if self.kind != ConstKind.Bool { throw ValueError(f"a {self.kind} constant has no bool") }
        let out: bool = self.value
        return out
    }

    #: The string held here; raises `ValueError` for any other kind.
    pub fn as_str(self) -> str {
        if self.kind != ConstKind.Str { throw ValueError(f"a {self.kind} constant has no str") }
        let out: str = self.value
        return out
    }

    fn __eq__(self, other: any) -> bool {
        if other is not ConstValue { return false }
        if self.kind != other.kind { return false }
        return self.value == other.value
    }
}

#: Evaluate `node` at compile time, or answer `ConstValue.none()`.
#:
#: Passing a `bag` is what says a constant is required here; without one the
#: fold abandons silently and the VM raises at run time instead. `consts` maps a
#: `const` binding's name to the initialiser it folded to.
pub fn const_eval(
    node: Node?,
    bag: DiagnosticBag? = null,
    consts: dict[str, Node] = {}
) -> ConstValue {
    return _eval(node, bag, consts, 0)
}

#: Rewrite `node` in place as the literal for `value`, and answer its type.
#:
#: The node keeps its span; its kind and whole field set are replaced. Answers
#: null, changing nothing, for `none`. The `dump` record is appended *before*
#: the rewrite because the node's own kind is half of it.
pub fn const_replace(u: Universe, node: Node, value: ConstValue) -> Type? {
    if value.is_none() { return null }

    match value.kind {
        ConstKind.Int => {
            _become(node, Node.int_lit(value.as_int()))
            return u.t_int
        }
        ConstKind.Float => {
            _become(node, Node.float_lit(value.as_float()))
            return u.t_float
        }
        ConstKind.Bool => {
            _become(node, Node.bool_lit(value.as_bool()))
            return u.t_bool
        }
        ConstKind.Str => {
            _become(node, Node.str_lit(value.as_str()))
            return u.t_str
        }
        ConstKind.Null => {
            _become(node, Node.null_lit())
            return u.t_null
        }
        ConstKind.None => { return null }
    }
}

#: Fold an operator node in place once its operands have been checked.
#:
#: Answers `(folded, ty)`. `checked` is kept whenever the literal fits it for
#: free, because a union or an alias is more informative than the literal's own
#: type — `1` folded in an `int | str` slot stays an `int | str`. A fit needing
#: a guard means the literal is the more precise answer.
pub fn try_fold(
    u: Universe,
    node: Node,
    checked: Type? = null,
    bag: DiagnosticBag? = null,
    consts: dict[str, Node] = {}
) -> (bool, Type?) {
    if not _is_foldable(node.kind) { return (false, checked) }

    let value = const_eval(node, bag, consts)
    if value.is_none() { return (false, checked) }

    let literal = const_replace(u, node, value)
    if checked is not null {
        let (accepted, guard) = assignable(u, literal, checked)
        if accepted and not guard { return (true, checked) }
    }
    return (true, literal)
}

#: A folded value in the one spelling both front ends must produce for it:
#: integers in decimal, floats through the `%.17g` the constant pool uses.
pub fn render_const(value: ConstValue) -> str {
    match value.kind {
        ConstKind.Int => { return f"{value.as_int()}" }
        ConstKind.Float => { return f"{value.as_float():.17g}" }
        ConstKind.Bool => {
            if value.as_bool() { return "true" }
            return "false"
        }
        ConstKind.Str => { return _quote(value.as_str()) }
        ConstKind.Null => { return "null" }
        ConstKind.None => { return "<none>" }
    }
}

fn _is_foldable(kind: NodeKind) -> bool {
    match kind {
        NodeKind.Unary | NodeKind.Binary | NodeKind.Logical => { return true }
        NodeKind.Ternary | NodeKind.CompareChain | NodeKind.Coalesce => { return true }
        _ => { return false }
    }
}

fn _become(node: Node, literal: Node) -> void {
    node.kind = literal.kind
    node.fields = literal.fields
}

fn _eval(node: Node?, bag: DiagnosticBag?, consts: dict[str, Node], hops: int) -> ConstValue {
    if node is null { return ConstValue.none() }

    match node.kind {
        NodeKind.IntLit => { return ConstValue.of_int(node.number("intLit")) }
        NodeKind.FloatLit => { return ConstValue.of_float(node.get("floatLit")) }
        NodeKind.BoolLit => { return ConstValue.of_bool(node.flag("boolLit")) }
        NodeKind.StrLit => { return ConstValue.of_str(node.opt_text("chars") ?? "") }
        NodeKind.NullLit => { return ConstValue.of_null() }

        NodeKind.Ident => {
            let name = node.opt_text("name")
            if name is null or name not in consts { return ConstValue.none() }
            if hops >= MAX_DECL_DEPTH { return ConstValue.none() }
            return _eval(consts[name], bag, consts, hops + 1)
        }

        NodeKind.Cast => { return _eval(node.child("operand"), bag, consts, hops) }

        NodeKind.Unary => { return _fold_unary(node, bag, consts, hops) }
        NodeKind.Binary => { return _fold_binary(node, bag, consts, hops) }

        NodeKind.Logical => {
            let a = _eval(node.child("left"), bag, consts, hops)
            if a.kind != ConstKind.Bool { return ConstValue.none() }
            let op = node.op()
            if op == OpKind.And and not a.as_bool() { return ConstValue.of_bool(false) }
            if op == OpKind.Or and a.as_bool() { return ConstValue.of_bool(true) }
            let b = _eval(node.child("right"), bag, consts, hops)
            if b.kind == ConstKind.Bool { return b }
            return ConstValue.none()
        }

        NodeKind.Ternary => {
            let cond = _eval(node.child("cond"), bag, consts, hops)
            if cond.kind != ConstKind.Bool { return ConstValue.none() }
            if cond.as_bool() { return _eval(node.child("thenExpr"), bag, consts, hops) }
            return _eval(node.child("elseExpr"), bag, consts, hops)
        }

        NodeKind.Coalesce => {
            let a = _eval(node.child("left"), bag, consts, hops)
            if a.is_none() { return a }
            if a.kind == ConstKind.Null { return _eval(node.child("right"), bag, consts, hops) }
            return a
        }

        NodeKind.CompareChain => { return _fold_chain(node, bag, consts, hops) }

        _ => { return ConstValue.none() }
    }
}

fn _fold_unary(node: Node, bag: DiagnosticBag?, consts: dict[str, Node], hops: int) -> ConstValue {
    let v = _eval(node.child("operand"), bag, consts, hops)
    if v.is_none() { return v }

    match node.op() {
        OpKind.Neg => {
            if v.kind == ConstKind.Int {
                # `-INT_MIN` is the one negation with no `int` answer.
                if v.as_int() == INT_MIN { return _overflow(bag, node.span, OpKind.Neg) }
                return ConstValue.of_int(-v.as_int())
            }
            if v.kind == ConstKind.Float { return ConstValue.of_float(-v.as_float()) }
            return ConstValue.none()
        }
        OpKind.Pos => {
            if v.is_numeric() { return v }
            return ConstValue.none()
        }
        OpKind.Not => {
            if v.kind == ConstKind.Bool { return ConstValue.of_bool(not v.as_bool()) }
            return ConstValue.none()
        }
        OpKind.Bnot => {
            if v.kind == ConstKind.Int { return ConstValue.of_int(~v.as_int()) }
            return ConstValue.none()
        }
        _ => { return ConstValue.none() }
    }
}

fn _fold_binary(node: Node, bag: DiagnosticBag?, consts: dict[str, Node], hops: int) -> ConstValue {
    let a = _eval(node.child("left"), bag, consts, hops)
    if a.is_none() { return a }
    let b = _eval(node.child("right"), bag, consts, hops)
    if b.is_none() { return b }
    return _combine(bag, node.op(), a, b, node.span)
}

fn _combine(
    bag: DiagnosticBag?,
    op: OpKind,
    a: ConstValue,
    b: ConstValue,
    span: Span
) -> ConstValue {
    if a.kind == ConstKind.Str { return _fold_str_binary(op, a, b) }
    if a.kind == ConstKind.Bool and b.kind == ConstKind.Bool {
        # `and`/`or` are `Logical` nodes, so equality is all bools can be asked.
        if op == OpKind.Eq { return ConstValue.of_bool(a.as_bool() == b.as_bool()) }
        if op == OpKind.Ne { return ConstValue.of_bool(a.as_bool() != b.as_bool()) }
        return ConstValue.none()
    }
    if not a.is_numeric() or not b.is_numeric() { return ConstValue.none() }
    if a.kind == ConstKind.Int and b.kind == ConstKind.Int {
        return _fold_int_binary(bag, op, a.as_int(), b.as_int(), span)
    }
    return _fold_float_binary(bag, op, a.as_float(), b.as_float(), span)
}

fn _fold_chain(node: Node, bag: DiagnosticBag?, consts: dict[str, Node], hops: int) -> ConstValue {
    let operands = node.child_list("operands")
    let ops: list[OpKind] = node.get("ops")
    if ops.len() < 1 or operands.len() != ops.len() + 1 { return ConstValue.none() }

    for (index, op) in ops.enumerate() {
        let a = _eval(operands[index], bag, consts, hops)
        let b = _eval(operands[index + 1], bag, consts, hops)
        if a.is_none() or b.is_none() { return ConstValue.none() }
        let link = _combine(bag, op, a, b, node.span)
        if link.kind != ConstKind.Bool { return ConstValue.none() }
        if not link.as_bool() { return ConstValue.of_bool(false) }
    }
    return ConstValue.of_bool(true)
}

fn _fold_int_binary(bag: DiagnosticBag?, op: OpKind, a: int, b: int, span: Span) -> ConstValue {
    match op {
        OpKind.Add => {
            let (bad, sum) = _add_overflow(a, b)
            if bad { return _overflow(bag, span, op) }
            return ConstValue.of_int(sum)
        }
        OpKind.Sub => {
            let (bad, difference) = _sub_overflow(a, b)
            if bad { return _overflow(bag, span, op) }
            return ConstValue.of_int(difference)
        }
        OpKind.Mul => {
            let (bad, product) = _mul_overflow(a, b)
            if bad { return _overflow(bag, span, op) }
            return ConstValue.of_int(product)
        }

        OpKind.AddWrap => { return ConstValue.of_int(a +% b) }
        OpKind.SubWrap => { return ConstValue.of_int(a -% b) }
        OpKind.MulWrap => { return ConstValue.of_int(a *% b) }

        # `/` on two ints yields float (BYTECODE §3.3), so it folds as one.
        OpKind.Div => {
            if b == 0 { return _divide_by_zero(bag, span, op) }
            return ConstValue.of_float(float(a) / float(b))
        }

        OpKind.FloorDiv => {
            if b == 0 { return _divide_by_zero(bag, span, op) }
            if a == INT_MIN and b == -1 { return _overflow(bag, span, op) }
            return ConstValue.of_int(a // b)
        }
        OpKind.Mod => {
            if b == 0 { return _divide_by_zero(bag, span, op) }
            if a == INT_MIN and b == -1 { return ConstValue.of_int(0) }
            return ConstValue.of_int(a % b)
        }

        OpKind.Pow => {
            if b < 0 { return ConstValue.none() }
            let (ok, power) = _ipow(a, b)
            if not ok { return _overflow(bag, span, op) }
            return ConstValue.of_int(power)
        }

        OpKind.Band => { return ConstValue.of_int(a & b) }
        OpKind.Bor => { return ConstValue.of_int(a | b) }
        OpKind.Bxor => { return ConstValue.of_int(a ^ b) }

        # These must fold to exactly what `bitwise()` in the VM computes.
        OpKind.Shl => {
            if b < 0 { return _negative_shift(bag, span, op) }
            if b >= 64 { return ConstValue.of_int(0) }
            return ConstValue.of_int(a << b)
        }
        OpKind.Shr => {
            if b < 0 { return _negative_shift(bag, span, op) }
            if b >= 64 { return ConstValue.of_int(if a < 0 { -1 } else { 0 }) }
            return ConstValue.of_int(a >> b)
        }

        OpKind.Eq => { return ConstValue.of_bool(a == b) }
        OpKind.Ne => { return ConstValue.of_bool(a != b) }
        OpKind.Lt => { return ConstValue.of_bool(a < b) }
        OpKind.Le => { return ConstValue.of_bool(a <= b) }
        OpKind.Gt => { return ConstValue.of_bool(a > b) }
        OpKind.Ge => { return ConstValue.of_bool(a >= b) }
        _ => { return ConstValue.none() }
    }
}

fn _fold_float_binary(
    bag: DiagnosticBag?,
    op: OpKind,
    a: float,
    b: float,
    span: Span
) -> ConstValue {
    match op {
        OpKind.Add => { return ConstValue.of_float(a + b) }
        OpKind.Sub => { return ConstValue.of_float(a - b) }
        OpKind.Mul => { return ConstValue.of_float(a * b) }
        OpKind.Div => {
            if b == 0.0 { return _divide_by_zero(bag, span, op) }
            return ConstValue.of_float(a / b)
        }
        OpKind.FloorDiv => {
            if b == 0.0 { return _divide_by_zero(bag, span, op) }
            return ConstValue.of_float(floor(a / b))
        }
        OpKind.Mod => {
            if b == 0.0 { return _divide_by_zero(bag, span, op) }
            # `fmod` takes the sign of the dividend, `%` that of the divisor.
            var r = fmod(a, b)
            if r != 0.0 and (r < 0.0) != (b < 0.0) { r += b }
            return ConstValue.of_float(r)
        }
        OpKind.Pow => { return ConstValue.of_float(a ** b) }
        OpKind.Eq => { return ConstValue.of_bool(a == b) }
        OpKind.Ne => { return ConstValue.of_bool(a != b) }
        OpKind.Lt => { return ConstValue.of_bool(a < b) }
        OpKind.Le => { return ConstValue.of_bool(a <= b) }
        OpKind.Gt => { return ConstValue.of_bool(a > b) }
        OpKind.Ge => { return ConstValue.of_bool(a >= b) }
        # No float overflow: IEEE 754 answers `inf`, which is a value.
        _ => { return ConstValue.none() }
    }
}

fn _fold_str_binary(op: OpKind, a: ConstValue, b: ConstValue) -> ConstValue {
    if op == OpKind.Add and b.kind == ConstKind.Str {
        return ConstValue.of_str(a.as_str() + b.as_str())
    }
    if b.kind != ConstKind.Str { return ConstValue.none() }
    return _compare_result(op, _compare_str(a.as_str(), b.as_str()))
}

# UTF-8 orders by code point exactly as it orders by byte, so this answers what
# the C's `memcmp` plus byte-length tie-break answers.
fn _compare_str(a: str, b: str) -> int {
    if a == b { return 0 }
    if a < b { return -1 }
    return 1
}

fn _compare_result(op: OpKind, cmp: int) -> ConstValue {
    match op {
        OpKind.Eq => { return ConstValue.of_bool(cmp == 0) }
        OpKind.Ne => { return ConstValue.of_bool(cmp != 0) }
        OpKind.Lt => { return ConstValue.of_bool(cmp < 0) }
        OpKind.Le => { return ConstValue.of_bool(cmp <= 0) }
        OpKind.Gt => { return ConstValue.of_bool(cmp > 0) }
        OpKind.Ge => { return ConstValue.of_bool(cmp >= 0) }
        _ => { return ConstValue.none() }
    }
}

# Every test runs before the operation, so the folder never executes an
# overflowing add. Both subtractions are in range for the `b` that reaches them.
fn _add_overflow(a: int, b: int) -> (bool, int) {
    if b > 0 and a > INT_MAX - b { return (true, 0) }
    if b < 0 and a < INT_MIN - b { return (true, 0) }
    return (false, a + b)
}

fn _sub_overflow(a: int, b: int) -> (bool, int) {
    if b < 0 and a > INT_MAX + b { return (true, 0) }
    if b > 0 and a < INT_MIN + b { return (true, 0) }
    return (false, a - b)
}

fn _mul_overflow(a: int, b: int) -> (bool, int) {
    if a == 0 or b == 0 { return (false, 0) }
    if a == -1 {
        if b == INT_MIN { return (true, 0) }
        return (false, -b)
    }
    if b == -1 {
        if a == INT_MIN { return (true, 0) }
        return (false, -a)
    }
    let wrapped = a *% b
    if wrapped // a != b { return (true, 0) }
    return (false, wrapped)
}

fn _ipow(base: int, exp: int) -> (bool, int) {
    var b = base
    var e = exp
    var result = 1
    while e > 0 {
        if e & 1 != 0 {
            let (bad, product) = _mul_overflow(result, b)
            if bad { return (false, 0) }
            result = product
        }
        e >>= 1
        if e > 0 {
            let (bad, squared) = _mul_overflow(b, b)
            if bad { return (false, 0) }
            b = squared
        }
    }
    return (true, result)
}

fn _fail(bag: DiagnosticBag?, code: str, span: Span, message: str) -> ConstValue {
    if bag is not null { bag.error(code, span, message) }
    return ConstValue.none()
}

fn _overflow(bag: DiagnosticBag?, span: Span, op: OpKind) -> ConstValue {
    return _fail(bag, E0900_CONST_OVERFLOW, span, f"integer overflow in constant `{op_text(op)}`")
}

fn _divide_by_zero(bag: DiagnosticBag?, span: Span, op: OpKind) -> ConstValue {
    return _fail(
        bag,
        E0901_CONST_DIVIDE_BY_ZERO,
        span,
        f"division by zero in constant `{op_text(op)}`"
    )
}

fn _negative_shift(bag: DiagnosticBag?, span: Span, op: OpKind) -> ConstValue {
    return _fail(
        bag,
        E0900_CONST_OVERFLOW,
        span,
        f"negative shift count in constant `{op_text(op)}`"
    )
}

fn _quote(text: str) -> str {
    let out = StringBuilder()
    out.push("\"")
    for character in text {
        if character == "\\" or character == "\"" {
            out.push("\\")
            out.push(character)
        } elif character == "\n" {
            out.push("\\n")
        } else {
            out.push(character)
        }
    }
    out.push("\"")
    return out.build()
}

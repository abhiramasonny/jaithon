#: The constant folder and the compile-time evaluator.
#:
#: A port of `src/sema/check_fold.c`: two jobs that are one implementation.
#: `const_eval` answers "what is this expression's value, if it has one at
#: compile time", which a `const` initialiser (spec §2.2) and a match arm (§5.3)
#: both need. `try_fold` is the same walk run for its side effect, replacing a
#: node in place with the literal it evaluates to.
#:
#: **Arithmetic here is overflow-checked rather than wrapping.** Folding
#: `2 ** 70` reports `E0900` and folds nothing; `1 // 0` reports `E0901`. A
#: folder that wrapped would produce a different program from the one the VM
#: would have run, which is the one failure mode a constant folder must never
#: have — the VM raises `OverflowError` for every operation checked here, so a
#: quiet wrap would turn a run-time error into a wrong answer.
#:
#: **Overflow is only an error where a constant is required.** The C reads a
#: global `gJaiCheck.constRequired` for that; here it is `bag`. Pass one where a
#: value is demanded and the two codes are reported; leave it out and the fold
#: simply gives up, which is what a fold running for optimisation wants — the VM
#: will raise at run time and that is what the language promises.
#:
#: Three things this port spells differently from the C, all because the
#: Jaithon `Node` carries less than the C `AstNode` does:
#:
#: | the C | here |
#: |---|---|
#: | `Checker *c`, for `jaiDiagError` and the arena | `bag: DiagnosticBag?` |
#: | `symbol->constValue` on an `AST_IDENT` | `consts: dict[str, Node]` |
#: | `node->type`, read and written in place | `checked` in, the type out |
#:
#: A Jaithon `Ident` node holds a name and no symbol, so the folded initialiser
#: of a `const` reaches this module as a name-keyed dict rather than through the
#: resolver. That dict is the const-propagation half of the fold family, and
#: without it `const K = 8` inlines nowhere.
#:
#: ```
#: let plus = Node.binary(OpKind.Add, Node.int_lit(2), Node.int_lit(3))
#: render_const(const_eval(plus))                   # "5"
#: const_eval(Node.ident("x")).is_none()            # true
#: const_eval(Node.ident("K"), null, {"K": Node.int_lit(8)}).as_int()   # 8
#: ```
from jaithon.ast import Node, NodeKind, OpKind, Span, kind_name, op_text
from jaithon.compile.check.assign import assignable
from jaithon.compile.check.dump import Record
from jaithon.compile.check.kinds import MAX_DECL_DEPTH
from jaithon.compile.check.ty import Type
from jaithon.compile.check.universe import Universe
from jaithon.compile.lexer import DiagnosticBag
from std.math import INT_MAX, INT_MIN, floor, fmod
from std.str import StringBuilder

#: An operation whose constant result does not fit an `int` (spec §13).
#:
#: This and `E0901` are spelled here rather than in `jaithon.compile.lexer.Code`
#: because that table is the front end's and stops at the parser; the E09xx
#: group belongs to the checker, and this is the first file of it.
pub const E0900_CONST_OVERFLOW: str = "E0900"

#: A constant divided or reduced modulo zero.
pub const E0901_CONST_DIVIDE_BY_ZERO: str = "E0901"

#: What a constant expression evaluated to, or `None` for "not a constant".
#:
#: Mirrors the C `ConstKind`. `None` is not a value: it is the answer for every
#: expression that has none at compile time, including one whose fold was
#: abandoned because it overflowed.
pub enum ConstKind { None, Int, Float, Bool, Str, Null }

#: One compile-time value: a kind, and the value that kind implies.
#:
#: The C's tagged union, with `value` where the union was. Read it through the
#: accessors — each checks the kind, because a folded `int` read back as a `str`
#: is a bug in the caller and not a value to invent a default for.
#:
#: There is deliberately no `__str__`. `render_const` is the spelling both front
#: ends must agree on byte for byte, since it is what a sema dump's `fold`
#: record carries, and a second rendering path is exactly how that contract gets
#: lost.
#:
#: ```
#: ConstValue.of_int(5).as_int()                        # 5
#: ConstValue.of_int(5) == ConstValue.of_float(5.0)     # false
#: ConstValue.none().is_none()                          # true
#: ```
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

    #: The float held here, widening an `int` — the C's `constAsFloat`.
    #:
    #: The widening is the rule that makes a mixed pair fold as two floats,
    #: which is what §2.5 says arithmetic across the two numeric types produces.
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
#: `bag` is where `E0900` and `E0901` go, and passing one is what says a
#: constant is *required* here: without it an overflow or a division by zero
#: abandons the fold silently and leaves the operation for the VM to raise on.
#: Either way the answer is `none`, so a caller never folds a wrong value.
#:
#: `consts` maps the name of a `const` binding to the initialiser it was folded
#: to, which is how a use of one inlines. It stands in for the C's
#: `symbol->constValue`, unreachable from a Jaithon `Ident` node.
#:
#: ```
#: let bag = DiagnosticBag()
#: let over = Node.binary(OpKind.Pow, Node.int_lit(2), Node.int_lit(70))
#: const_eval(over, bag).is_none()    # true
#: bag.items[0].code                  # "E0900"
#: bag.items[0].message               # "integer overflow in constant `**`"
#: const_eval(over).is_none()         # true, and nothing reported
#: ```
pub fn const_eval(
    node: Node?,
    bag: DiagnosticBag? = null,
    consts: dict[str, Node] = {}
) -> ConstValue {
    return _eval(node, bag, consts, 0)
}

#: Rewrite `node` in place as the literal for `value`, and answer its type.
#:
#: The node keeps its span and becomes the literal outright: its kind and its
#: whole field set are replaced, which is the C's `memset` of the union plus the
#: assignment that follows it. Answers null, changing nothing, when `value` is
#: `none`.
#:
#: `dump` collects the sema-dump `fold` record, appended **before** the rewrite
#: because the node's own kind is half of it: that `2 + 3` folded says more than
#: that some node is now `5`. A synthesised node is skipped, as in the C — it
#: has no counterpart in the other front end's tree to be compared against.
#:
#: ```
#: let u = Universe()
#: let node = Node.binary(OpKind.Add, Node.int_lit(1), Node.int_lit(2))
#: const_replace(u, node, ConstValue.of_int(3)) == u.t_int   # true
#: kind_name(node.kind)                                      # "AST_INT_LIT"
#: node.number("intLit")                                     # 3
#: ```
pub fn const_replace(
    u: Universe,
    node: Node,
    value: ConstValue,
    dump: list[Record]? = null
) -> Type? {
    if value.is_none() { return null }

    if dump is not null and node.span.is_known() {
        let at = Record(
            node.span.start,
            node.span.end,
            kind_name(node.kind),
            "fold",
            render_const(value)
        )
        dump.push(at)
    }

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
        # The C copies the characters into the AST arena here, because they may
        # point into a scratch buffer that outlives nothing. A Jaithon string is
        # a value and needs no copy.
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
#: Answers `(folded, ty)`: whether the node was replaced, and the type it
#: carries now. `checked` is the type the checker derived for the node, and it
#: is **kept** whenever the literal still fits it, because a union or an alias
#: is more informative than the literal's own type — `1` folded inside an
#: `int | str` slot stays an `int | str`. It is only kept when the fit is free:
#: a fit that needs a guard means the literal is the more precise answer.
#:
#: Only the six operator kinds fold: `Unary`, `Binary`, `Logical`, `Ternary`,
#: `CompareChain` and `Coalesce`. A literal is already folded, and every other
#: expression either has a run-time effect or is not an expression at all.
#:
#: The C tests `c->foldConstants` first. There is no flag here: a caller that
#: does not want folding does not call this.
#:
#: ```
#: let u = Universe()
#: let node = Node.binary(OpKind.Mul, Node.int_lit(6), Node.int_lit(7))
#: let (folded, ty) = try_fold(u, node, u.t_int)
#: folded                     # true
#: node.number("intLit")      # 42
#: ty == u.t_int              # true
#: try_fold(u, Node.ident("x"), u.t_int)   # (false, u.t_int)
#: ```
pub fn try_fold(
    u: Universe,
    node: Node,
    checked: Type? = null,
    bag: DiagnosticBag? = null,
    consts: dict[str, Node] = {},
    dump: list[Record]? = null
) -> (bool, Type?) {
    if not _is_foldable(node.kind) { return (false, checked) }

    let value = const_eval(node, bag, consts)
    if value.is_none() { return (false, checked) }

    let literal = const_replace(u, node, value, dump)
    if checked is not null {
        let (accepted, guard) = assignable(u, literal, checked)
        if accepted and not guard { return (true, checked) }
    }
    return (true, literal)
}

#: A folded value in the one spelling both front ends must produce for it.
#:
#: Integers in decimal, floats through the `%.17g` the constant pool uses,
#: strings quoted with the three escapes a dump line cannot carry raw, and the
#: two singletons by name. This is what a sema-dump `fold` record holds, so it
#: is compared byte for byte by `make sema-diff` and is not a debug rendering.
#:
#: One difference from the C, in a place no real program reaches: the C renders
#: into a 512-byte buffer and truncates a longer string, an artefact of the
#: buffer rather than a decision, and nothing here truncates.
#:
#: ```
#: render_const(ConstValue.of_int(-5))         # "-5"
#: render_const(ConstValue.of_float(0.5))      # "0.5"
#: render_const(ConstValue.of_str("a\nb"))     # "\"a\\nb\""
#: render_const(ConstValue.none())             # "<none>"
#: ```
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

# The six kinds `jaiChkTryFold` fires for, in the C's order.
fn _is_foldable(kind: NodeKind) -> bool {
    match kind {
        NodeKind.Unary | NodeKind.Binary | NodeKind.Logical => { return true }
        NodeKind.Ternary | NodeKind.CompareChain | NodeKind.Coalesce => { return true }
        _ => { return false }
    }
}

# The node becomes the literal: same span, new kind, and the field set that kind
# declares rather than the one it had. Building the literal through `Node` is
# what guarantees the second half — a hand-written dict would leave the old
# kind's fields behind for `to_json` to trip over.
fn _become(node: Node, literal: Node) -> void {
    node.kind = literal.kind
    node.fields = literal.fields
}

# The recursion proper. `hops` counts *only* the substitutions of one constant
# for another, so `const A = B` through `const B = C` cannot loop forever on a
# `consts` dict that refers to itself; an expression nested more deeply than
# that still folds, which is why the counter is not a general depth.
fn _eval(node: Node?, bag: DiagnosticBag?, consts: dict[str, Node], hops: int) -> ConstValue {
    if node is null { return ConstValue.none() }

    match node.kind {
        NodeKind.IntLit => { return ConstValue.of_int(node.number("intLit")) }
        NodeKind.FloatLit => { return ConstValue.of_float(node.get("floatLit")) }
        NodeKind.BoolLit => { return ConstValue.of_bool(node.flag("boolLit")) }
        NodeKind.StrLit => { return ConstValue.of_str(node.opt_text("chars") ?? "") }
        NodeKind.NullLit => { return ConstValue.of_null() }

        # A `const` binding was folded at its declaration; its uses inline it.
        NodeKind.Ident => {
            let name = node.opt_text("name")
            if name is null or name not in consts { return ConstValue.none() }
            if hops >= MAX_DECL_DEPTH { return ConstValue.none() }
            return _eval(consts[name], bag, consts, hops + 1)
        }

        # A cast of a constant is that constant: the checker has already decided
        # the cast is legal, and the guard it may emit tests a value this walk
        # can see is of the right shape.
        NodeKind.Cast => { return _eval(node.child("operand"), bag, consts, hops) }

        NodeKind.Unary => { return _fold_unary(node, bag, consts, hops) }
        NodeKind.Binary => { return _fold_binary(node, bag, consts, hops) }

        NodeKind.Logical => {
            let a = _eval(node.child("left"), bag, consts, hops)
            if a.kind != ConstKind.Bool { return ConstValue.none() }
            let op = node.op()
            # Short-circuit, so a non-constant right operand still folds when
            # the left one decides the answer on its own.
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
                # `-INT_MIN` is the one negation with no `int` answer, and the
                # VM raises on it too.
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

# `a op b` for two constants of any kind, which is the body both `_fold_binary`
# and one link of a comparison chain need.
fn _combine(
    bag: DiagnosticBag?,
    op: OpKind,
    a: ConstValue,
    b: ConstValue,
    span: Span
) -> ConstValue {
    if a.kind == ConstKind.Str { return _fold_str_binary(op, a, b) }
    if a.kind == ConstKind.Bool and b.kind == ConstKind.Bool {
        # `and`/`or` are `Logical` nodes, so equality is all a pair of bools
        # can be asked here.
        if op == OpKind.Eq { return ConstValue.of_bool(a.as_bool() == b.as_bool()) }
        if op == OpKind.Ne { return ConstValue.of_bool(a.as_bool() != b.as_bool()) }
        return ConstValue.none()
    }
    if not a.is_numeric() or not b.is_numeric() { return ConstValue.none() }
    if a.kind == ConstKind.Int and b.kind == ConstKind.Int {
        return _fold_int_binary(bag, op, a.as_int(), b.as_int(), span)
    }
    # A mixed pair widens to float, which is what comparison across the two
    # numeric types already did.
    return _fold_float_binary(bag, op, a.as_float(), b.as_float(), span)
}

# `a < b <= c` is every link and-ed together, and a false link ends the chain
# without asking the rest — the same short circuit the emitter produces.
fn _fold_chain(node: Node, bag: DiagnosticBag?, consts: dict[str, Node], hops: int) -> ConstValue {
    let operands = node.child_list("operands")
    let ops: list[OpKind] = node.get("ops")
    # The C tests only that there is an operator, because its parser guarantees
    # one more operand than operators. Nothing guarantees that about a tree
    # built by hand, and an index past the end is not a signal this file has.
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

        # The wrapping forms are the ones the source asked to wrap.
        OpKind.AddWrap => { return ConstValue.of_int(a +% b) }
        OpKind.SubWrap => { return ConstValue.of_int(a -% b) }
        OpKind.MulWrap => { return ConstValue.of_int(a *% b) }

        # `/` on two ints yields float (BYTECODE §3.3), so it folds as one.
        OpKind.Div => {
            if b == 0 { return _divide_by_zero(bag, span, op) }
            return ConstValue.of_float(float(a) / float(b))
        }

        # Jaithon's `//` and `%` already floor, which is what the C reaches its
        # own `ifloordiv` and `imod` for: C division truncates. Only the two
        # cases that would trap need a guard of their own.
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
            # A negative exponent has no int result, so the VM decides it.
            if b < 0 { return ConstValue.none() }
            let (ok, power) = _ipow(a, b)
            if not ok { return _overflow(bag, span, op) }
            return ConstValue.of_int(power)
        }

        OpKind.Band => { return ConstValue.of_int(a & b) }
        OpKind.Bor => { return ConstValue.of_int(a | b) }
        OpKind.Bxor => { return ConstValue.of_int(a ^ b) }

        # These must fold to exactly what `bitwise()` in the VM computes, or a
        # constant-folded expression means something different from the same
        # expression behind a variable. Discarded bits are not an overflow; only
        # a negative count is an error, and it is one the VM raises too.
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
            # `fmod` takes the sign of the dividend and `%` takes the sign of
            # the divisor, so the remainder is corrected the way `imod` is.
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

# `+` concatenates and the six comparisons order; a right operand that is not a
# string folds to nothing, because `"ab" * 3` is a repetition this does not do.
fn _fold_str_binary(op: OpKind, a: ConstValue, b: ConstValue) -> ConstValue {
    if op == OpKind.Add and b.kind == ConstKind.Str {
        return ConstValue.of_str(a.as_str() + b.as_str())
    }
    if b.kind != ConstKind.Str { return ConstValue.none() }
    return _compare_result(op, _compare_str(a.as_str(), b.as_str()))
}

# The C compares the common prefix with `memcmp` and breaks a tie on byte
# length. UTF-8 orders by code point exactly as it orders by byte, and a string
# that is a prefix of another is shorter in both units, so the native
# comparison reaches the same answer on the same pairs.
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

# ------------------------------------------------------------------
# Overflow-checked integer arithmetic
# ------------------------------------------------------------------

# Every test is performed before the operation, so an overflowing addition is
# never executed — the VM would raise `OverflowError` from inside the folder,
# and an error raised where a fold merely has to give up is not a failure the
# caller can do anything with. Both subtractions below are in range for the
# sign of `b` that reaches them.
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

# The C tests a multiplication with a division, because it has no wrapping
# multiply to test against. Jaithon does: `*%` is `a * b` truncated to 64 bits,
# so recovering `b` from it is exact when nothing was lost and cannot be when
# something was. The three special cases are the C's, and `a == -1` has to be
# one of them either way: `INT_MIN // -1` traps.
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

# Exponentiation by squaring, giving up on the first multiply that overflows.
# The squaring is guarded by `e > 0` so that the last one, whose result nothing
# reads, cannot report an overflow that did not affect the answer: `2 ** 63`
# would otherwise fail on a square it never used.
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

# ------------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------------

# The C reports through the checker and only when `constRequired` is set. Here
# the presence of a bag is that flag, so the fold is silent wherever the caller
# has nowhere to put an error.
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

# E0900 rather than a code of its own, as in the C: a shift by a negative count
# is reported as the overflow group's "this constant has no value" error.
fn _negative_shift(bag: DiagnosticBag?, span: Span, op: OpKind) -> ConstValue {
    return _fail(
        bag,
        E0900_CONST_OVERFLOW,
        span,
        f"negative shift count in constant `{op_text(op)}`"
    )
}

# The three escapes a dump line cannot carry raw. Everything else, including a
# tab and every non-ASCII scalar, goes through as itself.
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

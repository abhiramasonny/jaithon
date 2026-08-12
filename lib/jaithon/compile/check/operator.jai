#: What an operator makes of its operands, and what makes it not apply. A port of
#: `jaiTypeBinaryResult` and `jaiTypeUnaryResult` in `src/sema/types.c`. Null is
#: the only failure signal: this table never raises, the caller reports.
from jaithon.ast import OpKind
from jaithon.compile.check.kinds import MAX_DECL_DEPTH, TyKind
from jaithon.compile.check.relate import join as join_types
from jaithon.compile.check.ty import MemberInfo, Type, TypeDecl
from jaithon.compile.check.universe import Universe, list_of

#: Whether `op` is one of the ten arithmetic operators.
pub fn is_arithmetic(op: OpKind) -> bool {
    match op {
        OpKind.Add | OpKind.Sub | OpKind.Mul | OpKind.Div => { return true }
        OpKind.FloorDiv | OpKind.Mod | OpKind.Pow => { return true }
        OpKind.AddWrap | OpKind.SubWrap | OpKind.MulWrap => { return true }
        _ => { return false }
    }
}

#: Whether `op` widens an `int` operand against a `float` one (spec §2.5). The
#: wrapping and bitwise forms are excluded: both are 64-bit integer operations
#: with no float form, so a float operand there is still E0406.
pub fn widens_operands(op: OpKind) -> bool {
    match op {
        OpKind.AddWrap | OpKind.SubWrap | OpKind.MulWrap => { return false }
        _ => { return is_arithmetic(op) }
    }
}

#: The dunder that implements `op` for a user type (spec §7.1). `//` and `@` are
#: absent from the C's `arithDunder`; `__neg__` is looked up by name elsewhere.
pub fn arith_dunder(op: OpKind) -> str? {
    match op {
        OpKind.Add => { return "__add__" }
        OpKind.Sub => { return "__sub__" }
        OpKind.Mul => { return "__mul__" }
        OpKind.Div => { return "__div__" }
        OpKind.Mod => { return "__mod__" }
        OpKind.Pow => { return "__pow__" }
        _ => { return null }
    }
}

#: The type of `left op right`, or null when the operator does not apply. A null
#: operand is read as `any`; `join` is the element join `list + list` needs.
pub fn binary_result(
    u: Universe,
    op: OpKind,
    left: Type?,
    right: Type?,
    join: fn(Universe, Type?, Type?) -> Type = join_types
) -> Type? {
    # The C resolves an alias off each operand first. Not ported: nothing in
    # either front end builds an alias type. See `TyKind.Alias` in `kinds.jai`.
    let l = left ?? u.t_any
    let r = right ?? u.t_any

    # Comparability is the caller's `__eq__`/`__lt__` lookup, so these answer
    # before the `any` rule below.
    match op {
        OpKind.Eq | OpKind.Ne | OpKind.Lt | OpKind.Le | OpKind.Gt | OpKind.Ge => { return u.t_bool }
        OpKind.Is | OpKind.IsNot | OpKind.In | OpKind.NotIn => { return u.t_bool }

        # No truthiness (spec §5.1): `if items` is an error, `if items.len() > 0` is not.
        OpKind.And | OpKind.Or => {
            let l_ok = l.is_kind(TyKind.Bool) or l.is_kind(TyKind.Any)
            let r_ok = r.is_kind(TyKind.Bool) or r.is_kind(TyKind.Any)
            if l_ok and r_ok { return u.t_bool }
            return null
        }
        _ => { }
    }

    if l.is_kind(TyKind.Any) or r.is_kind(TyKind.Any) { return u.t_any }

    let both_int = l.is_kind(TyKind.Int) and r.is_kind(TyKind.Int)
    let both_float = l.is_kind(TyKind.Float) and r.is_kind(TyKind.Float)
    # An int widens to float (spec §2.5), so a mixed pair takes the same row as two
    # floats. Not the wrapping or bitwise forms: those are 64-bit int operations.
    let mixed = _is_mixed_numeric(l, r)
    let any_float = both_float or mixed

    match op {
        OpKind.Add => {
            if both_int { return u.t_int }
            if any_float { return u.t_float }
            if l.is_kind(TyKind.Str) and r.is_kind(TyKind.Str) { return u.t_str }
            if l.is_kind(TyKind.Bytes) and r.is_kind(TyKind.Bytes) { return u.t_bytes }
            if l.is_kind(TyKind.List) and r.is_kind(TyKind.List) {
                return list_of(u, join(u, _element(l), _element(r)))
            }
        }

        OpKind.Sub => {
            if both_int { return u.t_int }
            if any_float { return u.t_float }
        }

        OpKind.Mul => {
            if both_int { return u.t_int }
            if any_float { return u.t_float }
            if l.is_kind(TyKind.Str) and r.is_kind(TyKind.Int) { return u.t_str }
            if l.is_kind(TyKind.Int) and r.is_kind(TyKind.Str) { return u.t_str }
            if l.is_kind(TyKind.Bytes) and r.is_kind(TyKind.Int) { return u.t_bytes }
            if l.is_kind(TyKind.Int) and r.is_kind(TyKind.Bytes) { return u.t_bytes }
            if l.is_kind(TyKind.List) and r.is_kind(TyKind.Int) { return l }
            if l.is_kind(TyKind.Int) and r.is_kind(TyKind.List) { return r }
        }

        # `int / int` is float (BYTECODE.md §3.3); `//` is the int division.
        OpKind.Div => { if both_int or any_float { return u.t_float } }

        OpKind.FloorDiv | OpKind.Mod => {
            if both_int { return u.t_int }
            if any_float { return u.t_float }
        }

        OpKind.Pow => {
            # A negative exponent raises ValueError at run time; the type stays int.
            if both_int { return u.t_int }
            if any_float { return u.t_float }
        }

        OpKind.AddWrap | OpKind.SubWrap | OpKind.MulWrap => { if both_int { return u.t_int } }

        OpKind.Band | OpKind.Bor | OpKind.Bxor | OpKind.Shl | OpKind.Shr => {
            # Answering here, not falling out of the match, is what blocks overloading.
            if both_int { return u.t_int }
            return null
        }

        # No core type defines `@`, and `arith_dunder` has no entry for it either.
        OpKind.MatMul => { }

        _ => { return null }
    }

    # Only the left operand: there is no reflected `__radd__`, so `1 + money` fails.
    if l.is_named() { return _overload_result(u, l, arith_dunder(op)) }
    return null
}

#: The type of `op operand`, or null when the operator does not apply. `-` and
#: `~` answer the operand's own type rather than a fixed one.
pub fn unary_result(u: Universe, op: OpKind, operand: Type?) -> Type? {
    let t = operand ?? u.t_any

    match op {
        OpKind.Neg => {
            if t.is_kind(TyKind.Int) or t.is_kind(TyKind.Float) or t.is_kind(TyKind.Any) {
                return t
            }
            return if t.is_named() { _overload_result(u, t, "__neg__") } else { null }
        }

        OpKind.Pos => {
            # `+x` has no dunder: the C front end gives a class no hook for it.
            if t.is_kind(TyKind.Int) or t.is_kind(TyKind.Float) or t.is_kind(TyKind.Any) {
                return t
            }
            return null
        }

        OpKind.Not => {
            if t.is_kind(TyKind.Bool) or t.is_kind(TyKind.Any) { return u.t_bool }
            return null
        }

        OpKind.Bnot => {
            if t.is_kind(TyKind.Int) or t.is_kind(TyKind.Any) { return t }
            return null
        }

        _ => { return null }
    }
}

fn _is_mixed_numeric(a: Type, b: Type) -> bool {
    if a.is_kind(TyKind.Int) and b.is_kind(TyKind.Float) { return true }
    return a.is_kind(TyKind.Float) and b.is_kind(TyKind.Int)
}

# The C's `overloadResult`.
fn _overload_result(u: Universe, t: Type, dunder: str?) -> Type? {
    let name = dunder
    if name == null { return null }
    let decl = t.decl
    if decl == null { return null }
    let member = _find_member(decl, name, 0)
    if member == null { return null }

    # Signature unchecked or not a function: `any` beats an error that vanishes next pass.
    let signature = member.ty
    if signature == null or signature.kind != TyKind.Fn { return u.t_any }
    return signature.ret ?? u.t_any
}

# The C's `declFindMember`, not `TypeDecl.find_member`: it walks the traits and
# the superclass chain as well as the own list, in that order, so an inherited
# overload is found and an override wins. A null `ty` still counts as found.
fn _find_member(d: TypeDecl?, name: str, depth: int) -> MemberInfo? {
    if d == null or depth > MAX_DECL_DEPTH { return null }
    let own = d.find_member(name)
    if own != null { return own }
    for implemented in d.traits {
        let found = _find_member(implemented, name, depth + 1)
        if found != null { return found }
    }
    return _find_member(d.superclass, name, depth + 1)
}

# Null for a hand-built `Type` with no argument, so the join sees null rather
# than an index error. Every list the universe builds has exactly one argument.
fn _element(t: Type) -> Type? {
    return if t.arg_count() > 0 { t.args[0] } else { null }
}

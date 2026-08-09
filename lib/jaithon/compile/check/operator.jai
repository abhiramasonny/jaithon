#: What an operator makes of its operands, and what makes it not apply.
#:
#: A port of `jaiTypeBinaryResult` and `jaiTypeUnaryResult` in
#: `src/sema/types.c`, driven by `jaithon.ast.OpKind` — the same thirty-two
#: codes as the C `OpKind` in `src/lang/ast.h`, in the same order.
#:
#: **Null is the only failure signal.** These functions compute a result type
#: or answer null; they never raise and never report. The caller turns a null
#: into its diagnostic, because only the caller holds the span to hang it on
#: and knows whether the operands were already wrong for some earlier reason.
#: A type table that raised would report one mistake per run.
#:
#: Three rules run through the whole table and are easier to state once than to
#: read off thirty cases:
#:
#: - **`any` is soft-accepted.** An operand the checker could not work out
#:   makes the result `any` rather than a second error, so one unknown does not
#:   cascade through every expression built on it.
#: - **`int` widens to `float`, but only where a float meaning exists.** A
#:   mixed pair lands on the same row as two floats for the arithmetic
#:   operators, and on no row at all for the wrapping and bitwise ones: those
#:   are 64-bit integer operations, so a float operand stays an error.
#: - **Only the left operand is consulted for an overload.** There is no
#:   reflected form, so `1 + money` never asks `Money` anything.
#:
#: ```
#: let u = Universe()
#: binary_result(u, OpKind.Div, u.t_int, u.t_int) == u.t_float   # true
#: binary_result(u, OpKind.Band, u.t_int, u.t_float) == null     # true
#: unary_result(u, OpKind.Not, u.t_bool) == u.t_bool             # true
#: ```
from jaithon.ast import OpKind
from jaithon.compile.check.kinds import JOIN_MAX_MEMBERS, MAX_DECL_DEPTH, TyKind
from jaithon.compile.check.ty import MemberInfo, Type, TypeDecl
from jaithon.compile.check.union import union_of
from jaithon.compile.check.universe import Universe, list_of

#: The dunder that implements `op` for a user type (spec §7.1).
#:
#: These six are every *binary* operator a class can give a meaning to, which
#: is a language rule rather than an omission: a comparison goes through
#: `__eq__`/`__lt__`, and the wrapping and bitwise forms are 64-bit integer
#: operations, so nothing a class could define would give them a meaning.
#: `//` and `@` are simply not overloadable — the C's `arithDunder` does not
#: list them and this is a port of that list.
#:
#: It is not the whole overload surface: `-` is overloadable too, through
#: `__neg__`, which `unary_result` looks up by name rather than through here.
#: `+` has no unary hook at all, for the reason given there.
#:
#: ```
#: arith_dunder(OpKind.Add)       # "__add__"
#: arith_dunder(OpKind.FloorDiv)  # null
#: ```
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

#: The type of `left op right`, or null when the operator does not apply.
#:
#: A null operand is read as `any`: it is one the checker could not work out,
#: and a diagnostic for it has already been reported somewhere else.
#:
#: `join` is the element join that `list + list` needs, and it is a parameter
#: because `relate.jai` does not exist yet — the C calls `jaiTypeJoin`, whose
#: subtype half is a file of its own. The default is `jaiTypeJoin` minus that
#: half, so the two disagree on exactly the inputs that half decides: whenever
#: one element is a subtype of the other, `jaiTypeJoin` answers the wider one
#: and this default keeps both in a union. That is more than the nominal case
#: it is easiest to picture — `list[Derived] + list[Base]` is `list[Base]` in
#: the C and `list[Base | Derived]` here — because the subtype relation also
#: reaches a generic parameter, which accepts anything (`list[int] + list[T]`
#: is `list[T]` there and `list[int | T]` here), a container whose element is
#: `any` (`list[list[str]] + list[list[any]]`), and tuples and function types,
#: which it compares position by position. Everywhere else the two agree, and
#: passing the real join once there is one removes the disagreement outright.
#:
#: ```
#: let u = Universe()
#: binary_result(u, OpKind.Add, u.t_int, u.t_float) == u.t_float   # true
#: binary_result(u, OpKind.Add, u.t_str, u.t_str) == u.t_str       # true
#: binary_result(u, OpKind.Add, u.t_str, u.t_int) == null          # true
#: binary_result(u, OpKind.Lt, u.t_str, u.t_int) == u.t_bool       # true
#: ```
pub fn binary_result(
    u: Universe,
    op: OpKind,
    left: Type?,
    right: Type?,
    join: fn(Universe, Type?, Type?) -> Type = _union_join
) -> Type? {
    # The C resolves an alias off each operand first. That step is not ported
    # because nothing in either front end constructs an alias type, so it is
    # the identity here; see the note on `TyKind.Alias` in `kinds.jai`.
    let l = left ?? u.t_any
    let r = right ?? u.t_any

    # Comparisons always produce bool. Whether the operands are comparable at
    # all depends on `__eq__`/`__lt__`, which is a member lookup the caller
    # does; this table only fixes the result type. That is also why they answer
    # before the `any` rule below: `x == y` is a bool even when neither side
    # has a type yet.
    match op {
        OpKind.Eq | OpKind.Ne | OpKind.Lt | OpKind.Le | OpKind.Gt | OpKind.Ge => { return u.t_bool }
        OpKind.Is | OpKind.IsNot | OpKind.In | OpKind.NotIn => { return u.t_bool }

        # No truthiness (spec §5.1): `and` and `or` take bools and yield a
        # bool, so `if items` is an error and `if items.len() > 0` is not.
        OpKind.And | OpKind.Or => {
            let l_ok = l.is_kind(TyKind.Bool) or l.is_kind(TyKind.Any)
            let r_ok = r.is_kind(TyKind.Bool) or r.is_kind(TyKind.Any)
            if l_ok and r_ok { return u.t_bool }
            return null
        }
        _ => { }
    }

    # Anything mixed with `any` stays dynamic.
    if l.is_kind(TyKind.Any) or r.is_kind(TyKind.Any) { return u.t_any }

    let both_int = l.is_kind(TyKind.Int) and r.is_kind(TyKind.Int)
    let both_float = l.is_kind(TyKind.Float) and r.is_kind(TyKind.Float)
    # One int and one float: the int widens and the result is float (spec
    # §2.5). `any_float` is what the arithmetic rows below test, so a mixed
    # pair lands on the same row as two floats. The checker inserts the
    # conversion into the tree before it gets here; this function only says
    # what the result type is.
    #
    # Deliberately not folded into the wrapping forms or the bitwise
    # operators: both are 64-bit integer operations with no float meaning, so
    # they stay int-only and a float operand is still an error.
    let mixed = _is_mixed_numeric(l, r)
    let any_float = both_float or mixed

    match op {
        OpKind.Add => {
            if both_int { return u.t_int }
            if any_float { return u.t_float }
            if l.is_kind(TyKind.Str) and r.is_kind(TyKind.Str) { return u.t_str }
            if l.is_kind(TyKind.Bytes) and r.is_kind(TyKind.Bytes) { return u.t_bytes }
            # Concatenation, so the element type is the join of the two: a
            # `list[int]` and a `list[str]` concatenate into a `list[int | str]`
            # rather than into nothing.
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
            # Repetition. `int * str` is the same operation with the operands
            # the other way round, and both orders are written out rather than
            # normalised because the list rows answer the operand itself: which
            # one that is depends on the side it was written on.
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
            # `int ** int` stays int. A negative exponent has no int result, so
            # it raises ValueError at run time rather than silently producing a
            # float: widening the static type to float for every power would
            # make `2 ** 8` unusable where an int is required.
            if both_int { return u.t_int }
            if any_float { return u.t_float }
        }

        OpKind.AddWrap | OpKind.SubWrap | OpKind.MulWrap => { if both_int { return u.t_int } }

        OpKind.Band | OpKind.Bor | OpKind.Bxor | OpKind.Shl | OpKind.Shr => {
            # Bitwise operators are int-only; there is no bool arithmetic. They
            # answer here instead of falling out of the match, which is what
            # makes them the one arithmetic-looking group a class cannot
            # overload.
            if both_int { return u.t_int }
            return null
        }

        # No core type defines `@`, so it falls through to the overload lookup
        # with nothing to find: `arith_dunder` has no entry for it either.
        OpKind.MatMul => { }

        # The four unary codes land here when they are handed to the binary
        # table by mistake, and so would any code a later `OpKind` adds.
        _ => { return null }
    }

    # Fall back to an operator overload on the left operand, and only the left.
    # A class that wants to be added to an int defines `__add__` and is written
    # on the left; `1 + money` is an error, because a reflected `__radd__`
    # would make the meaning of `+` depend on a lookup the reader cannot see.
    if l.is_named() { return _overload_result(u, l, arith_dunder(op)) }
    return null
}

#: The type of `op operand`, or null when the operator does not apply.
#:
#: `-` and `~` answer the operand's own type rather than a fixed one, so `-x`
#: on a float is a float and on `any` is `any`.
#:
#: ```
#: let u = Universe()
#: unary_result(u, OpKind.Neg, u.t_float) == u.t_float   # true
#: unary_result(u, OpKind.Bnot, u.t_float) == null       # true
#: unary_result(u, OpKind.Not, u.t_int) == null          # true
#: ```
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
            # `+x` has no dunder. It is a no-op on a number and nothing else,
            # and giving a class a hook for it would invent a language rule the
            # C front end does not have.
            if t.is_kind(TyKind.Int) or t.is_kind(TyKind.Float) or t.is_kind(TyKind.Any) {
                return t
            }
            return null
        }

        OpKind.Not => {
            # No truthiness: `not` takes a bool and gives a bool.
            if t.is_kind(TyKind.Bool) or t.is_kind(TyKind.Any) { return u.t_bool }
            return null
        }

        OpKind.Bnot => {
            if t.is_kind(TyKind.Int) or t.is_kind(TyKind.Any) { return t }
            return null
        }

        # Every binary code, and any code a later `OpKind` adds.
        _ => { return null }
    }
}

# One int and one float, in either order.
fn _is_mixed_numeric(a: Type, b: Type) -> bool {
    if a.is_kind(TyKind.Int) and b.is_kind(TyKind.Float) { return true }
    return a.is_kind(TyKind.Float) and b.is_kind(TyKind.Int)
}

# The C's `overloadResult`. A user type gives an operator a meaning by defining
# the dunder, and the result of the expression is what that method returns.
fn _overload_result(u: Universe, t: Type, dunder: str?) -> Type? {
    let name = dunder
    if name == null { return null }
    let decl = t.decl
    if decl == null { return null }
    let member = _find_member(decl, name, 0)
    if member == null { return null }

    # The method exists but its signature has not been checked yet, or it is
    # not a function: `any` keeps the expression usable instead of raising a
    # bogus error that would vanish on the next pass.
    let signature = member.ty
    if signature == null or signature.kind != TyKind.Fn { return u.t_any }
    return signature.ret ?? u.t_any
}

# The C's `declFindMember`, which is not `TypeDecl.find_member`: it walks the
# traits and the superclass chain as well as the declaration's own list, in
# that order, so an overload inherited from a base class is found and an
# override wins over the member it overrides. `static` in the C and private
# here for the same reason — the operator table rests on the data layer alone,
# and the shared form of this walk belongs with the other nominal relations
# rather than beside the one caller that wants it for a dunder lookup.
#
# A member found with a null `ty` is still found. That is the case
# `_overload_result` answers `any` for, and it is distinct from absence.
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

# The element of a list type, or null when it was interned without one. Every
# list the universe builds has exactly one argument; this is here so that a
# hand-built `Type` reaches the join's null case rather than an index error,
# which is not a signal this file has.
fn _element(t: Type) -> Type? {
    return if t.arg_count() > 0 { t.args[0] } else { null }
}

# The default `join`: `jaiTypeJoin` with its two subtype tests left out,
# because the relation they need is `relate.jai` and this port has not reached
# it. Everything else is the C's, including the cap — a least upper bound that
# keeps growing is worse than no answer, since it slows every later comparison
# and prints a type nobody can read.
fn _union_join(u: Universe, a: Type?, b: Type?) -> Type {
    if a == null { return b ?? u.t_any }
    if b == null { return a }

    let members: list[Type?] = [a, b]
    let joined = union_of(u, members)
    if joined.kind == TyKind.Union and joined.arg_count() > JOIN_MAX_MEMBERS {
        return u.t_any
    }
    return joined
}

#: The five type-shape predicates, a port of `Predicates` in `src/sema/types.c`.
#: `resolveAlias` is not ported: nothing here builds a `TyKind.Alias`.
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.nominal import find_member
from jaithon.compile.check.relate import join as join_types
from jaithon.compile.check.ty import Type
from jaithon.compile.check.universe import Universe

#: Is `t` `int` or `float`? `bool` is not: the language has no bool arithmetic.
pub fn is_numeric(t: Type?) -> bool {
    if t is null { return false }
    return t.kind == TyKind.Int or t.kind == TyKind.Float
}

#: Can a value of `t` be null? `null` and `any` answer true with no union around
#: them. One level of members is enough: `union_of` flattens before interning.
pub fn is_optional(t: Type?) -> bool {
    if t is null { return false }
    if t.kind == TyKind.Null or t.kind == TyKind.Any { return true }
    if t.kind != TyKind.Union { return false }
    for member in t.args {
        if member.kind == TyKind.Null { return true }
    }
    return false
}

#: Can `t` appear before a `(`? A class object is its own constructor (§7.1).
pub fn is_callable(t: Type?) -> bool {
    if t is null { return false }
    match t.kind {
        TyKind.Fn | TyKind.Any | TyKind.Class | TyKind.Enum => { return true }
        TyKind.Trait => { return find_member(t.decl, "__call__") is not null }
        _ => { return false }
    }
}

#: What `for x in e` binds, or null when `e` cannot be looped over. A dict yields
#: its keys. A union yields nothing, even when every member is iterable.
pub fn is_iterable(
    u: Universe,
    t: Type?,
    join: fn(Universe, Type?, Type?) -> Type = join_types
) -> Type? {
    if t is null { return null }
    match t.kind {
        TyKind.List | TyKind.Set => { return t.args[0] }
        TyKind.Dict => { return t.args[0] }
        TyKind.Tuple => { return _join_all(u, t.args, join) }
        TyKind.Str => { return u.t_str }
        TyKind.Bytes | TyKind.Range => { return u.t_int }
        TyKind.Any => { return u.t_any }
        TyKind.Class | TyKind.Trait | TyKind.Enum => {
            if find_member(t.decl, "iter") is not null { return u.t_any }
            if find_member(t.decl, "__iter__") is not null { return u.t_any }
            return null
        }
        _ => { return null }
    }
}

#: What `e[i]` wants for `i` and hands back, or null when `e` cannot be indexed.
#: A set is not indexable, and neither is a union. A tuple joins its members;
#: `checkIndex` in `src/sema/check_expr.c` resolves a constant subscript first.
pub fn is_indexable(
    u: Universe,
    t: Type?,
    join: fn(Universe, Type?, Type?) -> Type = join_types
) -> (Type, Type)? {
    if t is null { return null }
    match t.kind {
        TyKind.List => { return (u.t_int, t.args[0]) }
        TyKind.Dict => { return (t.args[0], t.args[1]) }
        TyKind.Tuple => { return (u.t_int, _join_all(u, t.args, join)) }
        TyKind.Str => { return (u.t_int, u.t_str) }
        TyKind.Bytes | TyKind.Range => { return (u.t_int, u.t_int) }
        TyKind.Any => { return (u.t_any, u.t_any) }
        TyKind.Class | TyKind.Trait | TyKind.Enum => {
            if find_member(t.decl, "__getitem__") is not null { return (u.t_any, u.t_any) }
            return null
        }
        _ => { return null }
    }
}

# `joinAll` in the C. The fold's cap must fire per step, and the `never` seed
# relies on `join`'s `Never` arm: without it a wide-union member trips the cap.
fn _join_all(u: Universe, members: list[Type], join: fn(Universe, Type?, Type?) -> Type) -> Type {
    var result: Type = u.t_never
    for member in members { result = join(u, result, member) }
    return result
}

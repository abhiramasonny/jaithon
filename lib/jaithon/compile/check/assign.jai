#: Assignability: does a value of one type fit a slot of another, and what has
#: to be emitted for it to. A port of `assignableInto`, `jaiTypeWidenTarget`,
#: `jaiTypeAssignable` and `jaiTypeIsSubtype` in `src/sema/types.c`. Generics
#: are erased (spec §6.1), so a `GenericParam` target accepts before the unions
#: are split and a source only after; the tests here are order-sensitive.
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.nominal import is_subclass_of, satisfies_trait
from jaithon.compile.check.ty import Type
from jaithon.compile.check.universe import Universe

#: Whether `source` fits `target`, and whether the fit needs a run-time check:
#: the `any`-to-concrete `OP_TYPE_GUARD` or the `int`-to-`float` `OP_TO_FLOAT`,
#: told apart by `widen_target` (`jaiChkRequireAssignable`, `src/sema/check.c`).
#: `int` widens only here; as a subtype rule `list[int]` would fit `list[float]`.
pub fn assignable(u: Universe, source: Type?, target: Type?) -> (bool, bool) {
    let (accepted, guard) = _assignable_into(source, target)
    if accepted { return (true, guard) }
    if widen_target(u, source, target) is null { return (false, false) }
    return (true, true)
}

#: Whether `source` fits `target` at all, guard or no guard.
pub fn is_assignable(u: Universe, source: Type?, target: Type?) -> bool {
    let (accepted, _guard) = assignable(u, source, target)
    return accepted
}

#: Whether `sub` is a subtype of `base` — `assignable` with the guarded answers
#: rejected, since `any` converts into a concrete type but is not one.
pub fn is_subtype(u: Universe, sub: Type?, base: Type?) -> bool {
    let (accepted, guard) = assignable(u, sub, base)
    return accepted and not guard
}

#: The `float` an `int` has to become to fit `target`, or null when no
#: conversion applies. The arm itself is returned rather than a bool because the
#: caller casts the node to it (`jaiChkWidenToFloat` in `src/sema/check.c`).
pub fn widen_target(u: Universe, source: Type?, target: Type?) -> Type? {
    if source is null or target is null { return null }
    if source.kind != TyKind.Int { return null }
    if target.kind == TyKind.Float { return target }
    if target.kind != TyKind.Union { return null }

    var found: Type? = null
    for member in target.args {
        if member.kind == TyKind.Int or member.kind == TyKind.Any { return null }
        if member.kind == TyKind.Float { found = member }
    }
    return found
}

fn _assignable_into(source: Type?, target: Type?) -> (bool, bool) {
    if source is null or target is null { return (true, false) }
    if source == target { return (true, false) }
    if target.kind == TyKind.Any { return (true, false) }
    if source.kind == TyKind.Never { return (true, false) }
    if source.kind == TyKind.Any { return (true, true) }

    if target.kind == TyKind.GenericParam { return (true, false) }

    if source.kind == TyKind.Union { return _union_source_into(source, target) }
    if target.kind == TyKind.Union { return _into_union_target(source, target) }

    if source.kind == TyKind.GenericParam { return (true, true) }

    match source.kind {
        TyKind.Class | TyKind.Enum => {
            if _is_tensor_type(source) and _is_tensor_type(target) {
                return _tensor_into(source, target)
            }
            if target.kind == TyKind.Class {
                return (is_subclass_of(source.decl, target.decl), false)
            }
            if target.kind == TyKind.Trait {
                return (satisfies_trait(source.decl, target), false)
            }
            return (false, false)
        }

        TyKind.Trait => {
            if target.kind != TyKind.Trait { return (false, false) }
            return (satisfies_trait(source.decl, target), false)
        }

        TyKind.List | TyKind.Set | TyKind.Dict => { return _container_into(source, target) }
        TyKind.Tuple => { return _tuple_into(source, target) }
        TyKind.Fn => { return _fn_into(source, target) }

        _ => { return (false, false) }
    }
}

fn _union_source_into(source: Type, target: Type) -> (bool, bool) {
    var guard = false
    for member in source.args {
        let (accepted, member_guard) = _assignable_into(member, target)
        if not accepted { return (false, false) }
        if member_guard { guard = true }
    }
    return (true, guard)
}

fn _into_union_target(source: Type, target: Type) -> (bool, bool) {
    var guarded = false
    for member in target.args {
        let (accepted, member_guard) = _assignable_into(source, member)
        if not accepted { continue }
        if not member_guard { return (true, false) }
        guarded = true
    }
    return (guarded, guarded)
}

fn _container_into(source: Type, target: Type) -> (bool, bool) {
    if target.kind != source.kind { return (false, false) }
    if source.arg_count() != target.arg_count() { return (false, false) }

    var guard = false
    for (position, element) in source.args.enumerate() {
        let wanted = target.args[position]
        if element == wanted { continue }

        if _is_erased(element) or _is_erased(wanted) {
            if element.kind == TyKind.Any and wanted.kind != TyKind.Any { guard = true }
            continue
        }

        if element.kind == wanted.kind and _is_structural(element.kind) {
            let (accepted, element_guard) = _assignable_into(element, wanted)
            if accepted {
                if element_guard { guard = true }
                continue
            }
        }
        return (false, false)
    }
    return (true, guard)
}

fn _tuple_into(source: Type, target: Type) -> (bool, bool) {
    if target.kind != TyKind.Tuple { return (false, false) }
    if source.arg_count() != target.arg_count() { return (false, false) }

    var guard = false
    for (position, element) in source.args.enumerate() {
        let (accepted, element_guard) = _assignable_into(element, target.args[position])
        if not accepted { return (false, false) }
        if element_guard { guard = true }
    }
    return (true, guard)
}

fn _fn_into(source: Type, target: Type) -> (bool, bool) {
    if target.kind != TyKind.Fn { return (false, false) }
    if source.arg_count() != target.arg_count() { return (false, false) }
    if source.fn_flags != target.fn_flags { return (false, false) }

    var guard = false
    for (position, parameter) in source.args.enumerate() {
        let (parameter_ok, parameter_guard) = _assignable_into(target.args[position], parameter)
        if not parameter_ok { return (false, false) }
        if parameter_guard { guard = true }
    }

    let (return_ok, return_guard) = _assignable_into(source.ret, target.ret)
    if not return_ok { return (false, false) }
    if return_guard { guard = true }
    return (true, guard)
}

fn _is_erased(t: Type) -> bool {
    return t.kind == TyKind.Any or t.kind == TyKind.GenericParam
}

fn _is_tensor_type(t: Type?) -> bool {
    if t is null or t.kind != TyKind.Class { return false }
    let name = t.name
    if name is null { return false }
    return name == "Tensor" or name.ends_with(".Tensor")
}

fn _tensor_into(source: Type, target: Type) -> (bool, bool) {
    if not _is_tensor_type(source) or not _is_tensor_type(target) { return (false, false) }
    if source.arg_count() == 0 or target.arg_count() == 0 { return (true, false) }
    if source.arg_count() != target.arg_count() { return (false, false) }
    for (position, argument) in source.args.enumerate() {
        if argument != target.args[position] { return (false, false) }
    }
    return (true, false)
}

fn _is_structural(kind: TyKind) -> bool {
    match kind {
        TyKind.List | TyKind.Set | TyKind.Dict | TyKind.Tuple | TyKind.Fn => { return true }
        _ => { return false }
    }
}

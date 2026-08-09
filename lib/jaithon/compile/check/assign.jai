#: Assignability: does a value of one type fit a slot of another, and what has
#: to be emitted for it to.
#:
#: A port of `assignableInto`, `jaiTypeWidenTarget`, `jaiTypeAssignable` and
#: `jaiTypeIsSubtype` in `src/sema/types.c`. The rest of this directory answers
#: questions about a type; this file answers one about an *assignment*, and two
#: of its answers put an instruction into the bytecode that the source does not
#: contain. `any` reaching a concrete type is accepted behind an
#: `OP_TYPE_GUARD`, and `int` reaching `float` is accepted behind an
#: `OP_TO_FLOAT`. Both are decided here and nowhere else.
#:
#: **The guard comes back as a value rather than through an out-parameter.**
#: `jaiTypeAssignable` writes it through `bool *needsGuard`, which a caller can
#: read after a false return, where it means nothing. `assignable` answers
#: `(accepted, needs_guard)` and leaves `needs_guard` false whenever `accepted`
#: is, so there is no such reading.
#:
#: **One flag still covers the guard and the conversion**, exactly as the C's
#: one out-parameter does, so a caller that has to tell an `OP_TYPE_GUARD` from
#: an `OP_TO_FLOAT` asks `widen_target` first. That is what
#: `jaiChkRequireAssignable` in `src/sema/check.c` does: a non-null widen target
#: means insert the conversion and stop, and only what is left over reaches the
#: general rule.
#:
#: **The order of the tests that run before the switch on the kind is the
#: algorithm**, and two of them are order-sensitive against the unions:
#:
#: - A `GenericParam` *target* accepts before the unions are taken apart; a
#:   `GenericParam` *source* is tested after them. Generics are erased (spec
#:   §6.1), so a parameter names no run-time shape: as a target it accepts
#:   everything for free, and as a source it crosses the same erased-to-concrete
#:   boundary `any` crosses, which is checked at run time. Testing the source
#:   first would make `T` reaching `T | null` — which is `T` reaching `T`, and
#:   free — report a dynamic conversion.
#: - Identity, `-> any` and `never ->` all answer before `any ->` does, which is
#:   what keeps `any` reaching `any` and `never` reaching `int` free rather than
#:   guarded.
#:
#: `u` is not read by anything here. Nothing in this file builds a type — every
#: answer is one of the arguments or a member of one — and the universe is taken
#: anyway so that a caller need not remember which relations in this directory
#: want one.
#:
#: **The identity test is `source == target` and not a call to `equals`.**
#: `jaiTypeEquals` is `resolveAlias(a) == resolveAlias(b)`, and with no alias to
#: resolve (see `kinds.jai`) that is `a == b`, which is what interning already
#: made it. `relate.jai` does port it, under the C's name, for the callers that
#: want one — but it is `relate.jai` that imports `is_subtype` from here, so
#: importing `equals` back would close the loop into a cycle for an operator
#: this file can already spell.
#:
#: ```
#: let u = Universe()
#: assignable(u, u.t_int, u.t_int)     # (true, false)
#: assignable(u, u.t_any, u.t_int)     # (true, true)
#: assignable(u, u.t_int, u.t_float)   # (true, true)
#: assignable(u, u.t_float, u.t_int)   # (false, false)
#: ```
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.nominal import is_subclass_of, satisfies_trait
from jaithon.compile.check.ty import Type
from jaithon.compile.check.universe import Universe

#: Whether `source` fits `target`, and whether the fit needs a run-time check.
#:
#: `(false, false)` when it does not fit at all; `(true, false)` when it fits
#: statically; `(true, true)` when it fits only with an instruction — either the
#: `any`-to-concrete guard or the `int`-to-`float` conversion. `widen_target`
#: tells those two apart, and only the caller knows which of them it can emit.
#:
#: **`int` widens to `float` at this level and deliberately not inside the
#: recursion.** It is a conversion rather than a subtype relation: the bits that
#: arrive are not the bits that are stored, so something in the tree has to
#: perform it, and only a top-level assignment has a node to hang that on. Were
#: `int` a subtype of `float` instead, `list[int]` would pass where a
#: `list[float]` was wanted with no way to convert the elements, and a
#: `fn() -> int` would pass as a `fn() -> float` with no way to convert a result
#: that does not exist yet. Those two are exactly why the relation stops here.
#:
#: `[1, 2.0]` used to be a third reason and is not any more, which is worth
#: knowing before someone re-derives the rule from it: the checker fuses such a
#: literal before the join runs (`fuseNumeric` in `src/sema/check_expr.c`),
#: converting each int element in place, so the `list[float]` the join reports
#: is one the elements agree with. That is a rewrite of the tree, performed
#: where the per-element nodes exist. It is still not a subtype rule, and
#: nothing in this file knows about it.
#:
#: ```
#: from jaithon.compile.check.union import optional_of
#: from jaithon.compile.check.universe import list_of
#: let u = Universe()
#: assignable(u, u.t_never, u.t_int)                     # (true, false)
#: assignable(u, u.t_null, optional_of(u, u.t_int))      # (true, false)
#: assignable(u, list_of(u, u.t_any), list_of(u, u.t_int))   # (true, true)
#: assignable(u, list_of(u, u.t_int), list_of(u, u.t_float)) # (false, false)
#: ```
pub fn assignable(u: Universe, source: Type?, target: Type?) -> (bool, bool) {
    let (accepted, guard) = _assignable_into(source, target)
    if accepted { return (true, guard) }
    if widen_target(u, source, target) is null { return (false, false) }
    return (true, true)
}

#: Whether `source` fits `target` at all, guard or no guard.
#:
#: `assignable` without the second answer, for the callers that only have to
#: report an error or not.
#:
#: ```
#: let u = Universe()
#: is_assignable(u, u.t_any, u.t_int)     # true
#: is_assignable(u, u.t_int, u.t_float)   # true
#: is_assignable(u, u.t_float, u.t_int)   # false
#: ```
pub fn is_assignable(u: Universe, source: Type?, target: Type?) -> bool {
    let (accepted, _guard) = assignable(u, source, target)
    return accepted
}

#: Whether `sub` is a subtype of `base`.
#:
#: Assignability minus the dynamic escape hatch: `any` is not a subtype of a
#: concrete type, it merely converts into one, and a conversion is precisely
#: what the guard marks. So this is `assignable` with the guarded answers
#: rejected, and `int` reaching `float` is not a subtype either.
#:
#: This is `jaiTypeIsSubtype`, and it is the relation `jaiTypeJoin` and
#: `jaiTypeNarrow` are built on — which is why it lives beside the function it
#: reads and `relate.jai` imports it from here rather than deriving a second
#: copy.
#:
#: ```
#: let u = Universe()
#: is_subtype(u, u.t_int, u.t_int)     # true
#: is_subtype(u, u.t_any, u.t_int)     # false
#: is_subtype(u, u.t_int, u.t_any)     # true
#: is_subtype(u, u.t_int, u.t_float)   # false
#: ```
pub fn is_subtype(u: Universe, sub: Type?, base: Type?) -> bool {
    let (accepted, guard) = assignable(u, sub, base)
    return accepted and not guard
}

#: The `float` an `int` has to become to fit `target`, or null when no
#: conversion applies.
#:
#: Null for every source but `int`, and for every target that already admits an
#: `int`: plain assignability accepted those, so they never needed converting.
#: A union with a `float` arm and no `int` arm — `float?`, most often — is the
#: one shape beyond bare `float` that answers, because a value reaching it still
#: has to become a float to fit.
#:
#: The answer is the `float` type itself rather than a bool, because the caller
#: wraps the node in a cast to it (`jaiChkWidenToFloat` in
#: `src/sema/check.c`), and the arm of a union is the type that cast targets.
#:
#: ```
#: from jaithon.compile.check.union import optional_of, union_of
#: let u = Universe()
#: widen_target(u, u.t_int, u.t_float) == u.t_float                  # true
#: widen_target(u, u.t_int, optional_of(u, u.t_float)) == u.t_float  # true
#: widen_target(u, u.t_int, union_of(u, [u.t_int, u.t_float]))       # null
#: widen_target(u, u.t_float, u.t_int)                               # null
#: ```
pub fn widen_target(u: Universe, source: Type?, target: Type?) -> Type? {
    if source is null or target is null { return null }
    if source.kind != TyKind.Int { return null }
    if target.kind == TyKind.Float { return target }
    if target.kind != TyKind.Union { return null }

    var found: Type? = null
    for member in target.args {
        # A union that already admits `int` needs no conversion, and one that
        # admits `any` cannot be built by `union_of` at all — `any` absorbs the
        # union it is a member of. The second test is the C's and is kept
        # because a union interned directly still reaches here.
        if member.kind == TyKind.Int or member.kind == TyKind.Any { return null }
        if member.kind == TyKind.Float { found = member }
    }
    return found
}

# The recursion proper, and the whole of the C's `assignableInto`. Split from
# `assignable` because the widening above it is not part of the relation: it
# runs once, on the assignment, and must not be reachable from a container
# element or a return type.
#
# Answers `(false, false)` on rejection rather than carrying a guard out with
# it. Nothing is lost: every caller of this function, here and in the C alike,
# returns immediately on a false and never reads the flag beside it.
fn _assignable_into(source: Type?, target: Type?) -> (bool, bool) {
    # An unknown operand means an error was already reported upstream; accepting
    # it keeps one mistake from producing a second diagnostic.
    if source is null or target is null { return (true, false) }
    # Interning: `T -> T` is one pointer comparison, and it is what makes the
    # element loops below able to skip equal elements without recursing.
    if source == target { return (true, false) }
    if target.kind == TyKind.Any { return (true, false) }
    if source.kind == TyKind.Never { return (true, false) }
    # `target` is concrete by now: the dynamic-to-static boundary, checked at
    # run time by a guard.
    if source.kind == TyKind.Any { return (true, true) }

    # Erased, so an unsubstituted parameter names no runtime shape and accepts
    # every value. Without this every call to a generic function would be an
    # arity-shaped type error.
    if target.kind == TyKind.GenericParam { return (true, false) }

    if source.kind == TyKind.Union { return _union_source_into(source, target) }
    if target.kind == TyKind.Union { return _into_union_target(source, target) }

    # The mirror of the target rule above, but only once the unions have been
    # taken apart: `T` reaching `int | null` is `T` reaching `int`, which is the
    # erased-to-concrete boundary `any` also crosses.
    if source.kind == TyKind.GenericParam { return (true, true) }

    match source.kind {
        # A generic instantiation compares as its declaration and never as its
        # arguments, so `Box[int]` and `Box[str]` are mutually assignable. That
        # is the C's rule and not an omission of this port: `declIsSubclassOf`
        # is handed the two `TypeDecl`s, which a `Box` shares with every other
        # `Box`.
        TyKind.Class | TyKind.Enum => {
            if target.kind == TyKind.Class {
                return (is_subclass_of(source.decl, target.decl), false)
            }
            if target.kind == TyKind.Trait {
                return (satisfies_trait(source.decl, target), false)
            }
            return (false, false)
        }

        # A trait value satisfies the traits that trait extends, and nothing
        # else: a trait-typed binding names no class, so no class type can
        # receive it.
        TyKind.Trait => {
            if target.kind != TyKind.Trait { return (false, false) }
            return (satisfies_trait(source.decl, target), false)
        }

        TyKind.List | TyKind.Set | TyKind.Dict => { return _container_into(source, target) }
        TyKind.Tuple => { return _tuple_into(source, target) }
        TyKind.Fn => { return _fn_into(source, target) }

        # `int -> float` is not here: it is a conversion rather than a subtype
        # relation, so it belongs to `widen_target` and not to this recursion.
        _ => { return (false, false) }
    }
}

# Every member has to fit, and one guarded member guards the whole assignment:
# `int | str` reaching `int` is rejected outright, while a union with an erased
# member reaching a concrete type is the same run-time check a bare `any` gets.
fn _union_source_into(source: Type, target: Type) -> (bool, bool) {
    var guard = false
    for member in source.args {
        let (accepted, member_guard) = _assignable_into(member, target)
        if not accepted { return (false, false) }
        if member_guard { guard = true }
    }
    return (true, guard)
}

# Also the `null -> T?` rule, since `T?` is `T | null`.
#
# A member that fits without a guard wins over one that only fits behind a
# run-time check, and the difference is visible whenever the guarded member
# sorts first: `T` reaching `int | T` tries `int` first, which `T` reaches only
# dynamically, and must still answer that `T` reaching `T` is free. Union
# members are sorted by kind ordinal, so which member comes first is not
# something a caller chose.
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

# `list`, `set` and `dict`: invariant between concrete elements, not covariant.
# These containers are mutable, so if `list[Circle]` were assignable to
# `list[Shape]` the callee could push a `Square` through the widened reference
# and the caller would read it back as a `Circle`.
#
# A dynamic element is the exception, because `any` is the one type that
# converts both ways (spec §2.2): `list[any]` — what an untyped comprehension or
# an `.enumerate()` produces — flows into `list[str]` behind the same guard a
# bare `any` gets, and `list[str]` flows back into `list[any]` freely. Equal
# elements are the same object, so `_assignable_into`'s identity test has
# already covered them.
fn _container_into(source: Type, target: Type) -> (bool, bool) {
    if target.kind != source.kind { return (false, false) }
    if source.arg_count() != target.arg_count() { return (false, false) }

    var guard = false
    for (position, element) in source.args.enumerate() {
        let wanted = target.args[position]
        if element == wanted { continue }

        if _is_erased(element) or _is_erased(wanted) {
            # Only `any` reaching something concrete costs a check. Widening to
            # `any` is free, and a generic parameter is erased in both
            # directions — which is why an element that is one never guards,
            # though a top-level `T` does.
            if element.kind == TyKind.Any and wanted.kind != TyKind.Any { guard = true }
            continue
        }

        # A structural element answers the same question one level down:
        # `list[(any, any)]` is what an untyped comprehension of pairs produces
        # and it has to reach `list[(K, V)]`. A nominal element does not
        # recurse — that would be the covariance ruled out above.
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

# Tuples are immutable, so element-wise covariance is sound.
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

# Contravariant in the parameters, covariant in the return type, and exact in
# both the arity and the flags.
#
# The flags are part of it because they change how a call is assembled rather
# than what it means: `fn(a, ...rest)` reads one list off the stack where
# `fn(a, b)` reads two values, so a function accepted for the other's slot would
# be called with a frame it cannot read. Nothing about that is expressible as a
# relation between the parameter types, which is why it is a separate test.
fn _fn_into(source: Type, target: Type) -> (bool, bool) {
    if target.kind != TyKind.Fn { return (false, false) }
    if source.arg_count() != target.arg_count() { return (false, false) }
    if source.fn_flags != target.fn_flags { return (false, false) }

    var guard = false
    # The target's caller passes `target`'s argument types, so `source` has to
    # accept them — the question is asked in the other direction.
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

# `any` and an unsubstituted generic parameter: the two kinds that name no
# run-time shape, and so the two an invariant element position lets past.
fn _is_erased(t: Type) -> bool {
    return t.kind == TyKind.Any or t.kind == TyKind.GenericParam
}

# The kinds an invariant element position will recurse into. A class is not one
# of them, and that is the whole of what makes these containers invariant.
fn _is_structural(kind: TyKind) -> bool {
    match kind {
        TyKind.List | TyKind.Set | TyKind.Dict | TyKind.Tuple | TyKind.Fn => { return true }
        _ => { return false }
    }
}

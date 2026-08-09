#: Generic parameter substitution: `list[T]` under `{T: int}` is `list[int]`.
#:
#: A port of `jaiTypeSubstitute` in `src/sema/types.c`. It walks a type,
#: replaces every `GenericParam` whose name the bindings mention, and re-interns
#: whatever it rebuilds so the answer is a type of this universe rather than a
#: structural twin of one.
#:
#: The mapping is a `dict[str, Type]` rather than the C's parallel `names` and
#: `values` arrays. That closes a case the C has to carry: `values[i]` may be
#: null there, and the C answers the parameter unchanged when it is, whereas a
#: dict value is a `Type` and the situation cannot arise.
#:
#: **A type that mentions no bound parameter comes back as the same object**,
#: and that is checked at every level of the walk rather than only at the top.
#: For a type this universe interned, a rebuild would land on the same object
#: anyway, so what the early-out buys there is not rebuilding and re-hashing a
#: type only to be told nothing changed — substitution runs once per member per
#: trait implementation, and most members mention no parameter at all. For a
#: type this universe did not intern it buys more: the answer is the object the
#: caller passed rather than a lookup, which is what the C's pointer comparison
#: means.
#:
#: `Self` needs no special case. It is a generic parameter spelled with a name
#: the parser forbids any declaration from introducing (E0114), so binding it is
#: `{"Self": implementer}` and nothing else (spec §7.2).
#:
#: One branch of the C is not ported: `TY_ALIAS`, which resolves the alias and
#: answers the substituted target. Nothing in either front end constructs an
#: alias type, so the branch has no input; the reasoning is at the point where
#: the kind would have been handled.
#:
#: ```
#: from jaithon.compile.check.universe import Universe
#: from jaithon.compile.check.universe import generic_param, list_of
#: let u = Universe()
#: let param = generic_param(u, "T")
#: substitute(u, list_of(u, param), {"T": u.t_int}) == list_of(u, u.t_int)  # true
#: mentions_generic_param(list_of(u, param))                                # true
#: mentions_generic_param(list_of(u, u.t_str))                              # false
#: ```
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.ty import Type
from jaithon.compile.check.union import union_of
from jaithon.compile.check.universe import Universe, dict_of, fn_of, instantiate
from jaithon.compile.check.universe import list_of, set_of, tuple_of

#: `t` with every generic parameter named in `bindings` replaced.
#:
#: Answers `t` itself — the same object, not an equal one — when nothing was
#: replaced, and null only when `t` was null.
#:
#: ```
#: from jaithon.compile.check.render import render
#: from jaithon.compile.check.universe import Universe
#: from jaithon.compile.check.universe import dict_of, fn_of, generic_param
#: let u = Universe()
#: let bindings = {"T": u.t_int}
#: let param = generic_param(u, "T")
#: render(substitute(u, dict_of(u, u.t_str, param), bindings))  # "dict[str, int]"
#: render(substitute(u, fn_of(u, [param], param), bindings))    # "fn(int) -> int"
#: ```
pub fn substitute(u: Universe, t: Type?, bindings: dict[str, Type]) -> Type? {
    # An empty mapping is the C's `count <= 0`: there is nothing to look up, so
    # every type is its own answer.
    if t is null or bindings.len() == 0 { return t }
    return _substitute(u, t, bindings)
}

# The walk proper. Split from `substitute` so the recursion never has to ask
# whether its argument is null: only the entry point can be handed one.
fn _substitute(u: Universe, t: Type, bindings: dict[str, Type]) -> Type {
    if t.kind == TyKind.GenericParam {
        let name = t.name
        if name is null { return t }
        if name in bindings { return bindings[name] }
        return t
    }

    # The containers are branches of their own rather than cases of the loop
    # below, which is the C's shape: their arity is fixed, so each compares its
    # one or two elements directly and never builds a list to find out that
    # nothing moved. They rebuild through the module's own constructors so that
    # a container built here is indistinguishable from a written one.
    if t.kind == TyKind.List {
        let element = _substitute(u, t.args[0], bindings)
        return if element == t.args[0] { t } else { list_of(u, element) }
    }
    if t.kind == TyKind.Set {
        let element = _substitute(u, t.args[0], bindings)
        return if element == t.args[0] { t } else { set_of(u, element) }
    }
    if t.kind == TyKind.Dict {
        let key = _substitute(u, t.args[0], bindings)
        let value = _substitute(u, t.args[1], bindings)
        if key == t.args[0] and value == t.args[1] { return t }
        return dict_of(u, key, value)
    }

    # Everything else with parts: tuples, functions, unions, and a generic
    # class, trait or enum instantiation. A primitive and a module have no
    # argument to descend into and are their own answer, which is the C's
    # `default:`.
    #
    # An alias is its own answer here too, and that is the one place this
    # function answers something the C does not. The C's `case TY_ALIAS`
    # descends through `ret` — not through `args`, an alias has none — and
    # answers the *substituted target*, on the grounds that once a parameter
    # inside it has been replaced the name no longer describes the result.
    # `resolveAlias` is not ported anywhere in this directory: `TyKind.Alias`
    # exists for ordinal fidelity and nothing in either front end constructs
    # one, so the branch has no input to act on (see `kinds.jai`, and the same
    # note in `predicate.jai`). `test_an_alias_is_answered_unchanged` pins the
    # decision rather than leaving it to be rediscovered from this comment.
    let rebuilds = t.kind == TyKind.Tuple or
        t.kind == TyKind.Fn or
        t.kind == TyKind.Union or
        t.is_named()
    if not rebuilds { return t }

    var args: list[Type] = []
    var changed = false
    for argument in t.args {
        let replaced = _substitute(u, argument, bindings)
        if replaced != argument { changed = true }
        args.push(replaced)
    }

    # A function is the only kind here whose `ret` is a type of its own — on an
    # alias the field holds the alias target instead — so the C guards on the
    # kind rather than on the field being set, and so does this.
    var returns = t.ret
    if t.kind == TyKind.Fn {
        let written = t.ret
        if written is not null {
            let replaced = _substitute(u, written, bindings)
            if replaced != written { changed = true }
            returns = replaced
        }
    }

    if not changed { return t }

    if t.kind == TyKind.Tuple { return tuple_of(u, args) }
    if t.kind == TyKind.Fn { return fn_of(u, args, returns, t.fn_flags) }
    if t.kind == TyKind.Union {
        # Rebuilt through `union_of` and not `intern`, because substitution is
        # exactly what makes normalisation necessary again: `T | int` under
        # `{T: int}` has to collapse to `int` rather than intern a union with
        # the same member twice.
        #
        # The copy is not busywork. Containers are invariant, so a `list[Type]`
        # is not a `list[Type?]` and `union_of` takes the latter.
        var members: list[Type?] = []
        for member in args { members.push(member) }
        return union_of(u, members)
    }

    # A generic class, trait or enum instantiation keeps its name and its
    # declaration; only the arguments change. `instantiate` interns with a null
    # `ret` and no flags, which is what a nominal type always had: `named` and
    # `instantiate` are its only producers and neither sets either.
    return instantiate(u, t, args)
}

#: Does `t` mention a generic parameter that nothing has substituted away?
#:
#: A port of `mentionsGenericParam` in `src/sema/check.c`, where it decides
#: whether a type guard can be emitted at all. Generics are erased, so such a
#: type has no run-time shape: a guard against `T` would compile to an
#: `OP_TYPE_GUARD` looking for a class named `T`, and there is no such class.
#: The checker's answer is to emit no guard rather than a failing one.
#:
#: The walk descends into `ret` only for a function, which is the C's rule and
#: not an oversight — on an alias `ret` is the target, and whether the target
#: mentions a parameter is a fact about the target rather than about `t`.
#:
#: ```
#: from jaithon.compile.check.universe import Universe
#: from jaithon.compile.check.universe import fn_of, generic_param
#: let u = Universe()
#: mentions_generic_param(u.t_int)                                     # false
#: mentions_generic_param(fn_of(u, [u.t_int], generic_param(u, "T")))  # true
#: mentions_generic_param(null)                                        # false
#: ```
pub fn mentions_generic_param(t: Type?) -> bool {
    if t is null { return false }
    if t.kind == TyKind.GenericParam { return true }
    for argument in t.args {
        if mentions_generic_param(argument) { return true }
    }
    return t.kind == TyKind.Fn and mentions_generic_param(t.ret)
}

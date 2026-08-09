#: Generic parameter substitution: `list[T]` under `{T: int}` is `list[int]`.
#: A port of `jaiTypeSubstitute` in `src/sema/types.c`. `Self` needs no special
#: case: the parser forbids declaring the name (E0114, spec §7.2).
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.ty import Type
from jaithon.compile.check.union import union_of
from jaithon.compile.check.universe import Universe, dict_of, fn_of, instantiate
from jaithon.compile.check.universe import list_of, set_of, tuple_of

#: `t` with every generic parameter named in `bindings` replaced. Answers `t`
#: itself — the same object, not an equal one — when nothing was replaced.
pub fn substitute(u: Universe, t: Type?, bindings: dict[str, Type]) -> Type? {
    if t is null or bindings.len() == 0 { return t }
    return _substitute(u, t, bindings)
}

fn _substitute(u: Universe, t: Type, bindings: dict[str, Type]) -> Type {
    if t.kind == TyKind.GenericParam {
        let name = t.name
        if name is null { return t }
        if name in bindings { return bindings[name] }
        return t
    }

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

    # An alias is its own answer, where the C's `TY_ALIAS` answers the
    # substituted target. Nothing constructs an alias type (`kinds.jai`), so the
    # branch has no input; `test_an_alias_is_answered_unchanged` pins that.
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

    # Only a function's `ret` is a type of its own; on an alias it is the target.
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
        # `union_of`, not `intern`: `T | int` under `{T: int}` becomes `int`.
        var members: list[Type?] = []
        for member in args { members.push(member) }
        return union_of(u, members)
    }

    return instantiate(u, t, args)
}

#: Does `t` mention a generic parameter nothing substituted away? A port of
#: `mentionsGenericParam` in `src/sema/check.c`: generics are erased, so such a
#: type has no run-time shape and the checker emits no type guard for it.
pub fn mentions_generic_param(t: Type?) -> bool {
    if t is null { return false }
    if t.kind == TyKind.GenericParam { return true }
    for argument in t.args {
        if mentions_generic_param(argument) { return true }
    }
    return t.kind == TyKind.Fn and mentions_generic_param(t.ret)
}

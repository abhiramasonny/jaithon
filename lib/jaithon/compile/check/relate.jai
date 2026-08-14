#: Equality, join and narrowing: the relations that answer with a type. A port
#: of `jaiTypeEquals`, `jaiTypeJoin` and `jaiTypeNarrow` in `src/sema/types.c`.
from jaithon.compile.check.assign import is_subtype
from jaithon.compile.check.kinds import JOIN_MAX_MEMBERS, TyKind
from jaithon.compile.check.ty import Type
from jaithon.compile.check.union import union_of
from jaithon.compile.check.universe import Universe

#: Are `a` and `b` the same type? `Universe.intern` makes identity enough.
pub fn equals(a: Type?, b: Type?) -> bool {
    return a == b
}

#: The least type that holds both `a` and `b`. The lattice is the union, not
#: the class hierarchy: `Circle` joined with `Square` is `Circle | Square`.
pub fn join(u: Universe, a: Type?, b: Type?) -> Type {
    if a == null { return b ?? u.t_any }
    if b == null { return a }
    if a == b { return a }

    if a.kind == TyKind.Never { return b }
    if b.kind == TyKind.Never { return a }
    if a.kind == TyKind.Any or b.kind == TyKind.Any { return u.t_any }

    if is_subtype(u, a, b) { return b }
    if is_subtype(u, b, a) { return a }

    let members: list[Type?] = [a, b]
    let joined = union_of(u, members)
    if joined.kind == TyKind.Union and joined.arg_count() > JOIN_MAX_MEMBERS {
        return u.t_any
    }
    return joined
}

#: The join folded over `types` from `never`, so `join`'s cap fires per step.
pub fn join_all(u: Universe, types: list[Type]) -> Type {
    var result: Type = u.t_never
    for t in types { result = join(u, result, t) }
    return result
}

#: What `t` is once `x is by` has been tested, on the branch `positive` names.
#: Positive refines a member to `by`; negative drops one only when it is
#: definitely a `by`, so `x is not Circle` on a `Shape` removes nothing.
pub fn narrow(u: Universe, t: Type?, by: Type?, positive: bool) -> Type? {
    if t == null or by == null { return t }

    if t.kind == TyKind.Union {
        var keep: list[Type?] = []
        for member in t.args {
            if positive {
                let survivor = _narrow_member(u, member, by)
                if survivor != null { keep.push(survivor) }
            } elif not is_subtype(u, member, by) {
                keep.push(member)
            }
        }
        return union_of(u, keep)
    }

    if positive { return _narrow_member(u, t, by) ?? u.t_never }
    return if is_subtype(u, t, by) { u.t_never } else { t }
}

fn _narrow_member(u: Universe, m: Type, by: Type) -> Type? {
    if is_subtype(u, m, by) { return m }
    if is_subtype(u, by, m) { return by }
    return null
}

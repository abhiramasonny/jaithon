#: Union normalisation: what makes `A | B` and `B | A` the same type.
#:
#: A union is interned by its member list, so member order is identity-affecting:
#: it is the order `render` prints, which is the string the sema diff compared
#: against the C. Every step below is a port of `jaiTypeUnion`, not a choice.
from jaithon.compile.check.kinds import TyKind
from jaithon.compile.check.ty import Type
from jaithon.compile.check.universe import Universe

#: Order two types the way the C's `typeCompare` does. Named types are compared
#: by kind and name before their arguments are reached, which is what stops a
#: recursive class — `class Node { next: Node }` — from recursing forever here.
pub fn compare_types(a: Type?, b: Type?) -> int {
    if a == null and b == null { return 0 }
    if a == null { return -1 }
    if b == null { return 1 }
    if a.ord != b.ord { return if a.ord < b.ord { -1 } else { 1 } }

    let a_name = a.name
    let b_name = b.name
    if a_name != b_name {
        if a_name == null { return -1 }
        if b_name == null { return 1 }
        if a_name < b_name { return -1 }
        if b_name < a_name { return 1 }
    }

    if a.arg_count() != b.arg_count() {
        return if a.arg_count() < b.arg_count() { -1 } else { 1 }
    }
    for (position, argument) in a.args.enumerate() {
        let order = compare_types(argument, b.args[position])
        if order != 0 { return order }
    }

    let order = compare_types(a.ret, b.ret)
    if order != 0 { return order }

    if a.fn_flags != b.fn_flags {
        return if a.fn_flags < b.fn_flags { -1 } else { 1 }
    }
    return 0
}

# Flattening is why `(int | str) | float` interns as `int | str | float`.
fn _collect(out: list[Type], t: Type?) -> void {
    if t == null { return }
    if t.kind == TyKind.Union {
        for member in t.args { _collect(out, member) }
        return
    }
    out.push(t)
}

#: The union of `members`, normalised. An empty list is `never`.
pub fn union_of(u: Universe, members: list[Type?]) -> Type {
    if members.len() == 0 { return u.t_never }

    var flat: list[Type] = []
    for member in members { _collect(flat, member) }

    var live: list[Type] = []
    for member in flat {
        if member.kind == TyKind.Any { return u.t_any }
        if member.kind == TyKind.Never { continue }
        live.push(member)
    }

    # Insertion sort, as the C does, so ties break the same way every run.
    var sorted: list[Type] = []
    for member in live {
        var at = sorted.len()
        while at > 0 and compare_types(sorted[at - 1], member) > 0 {
            at -= 1
        }
        sorted.insert(at, member)
    }

    # The sort makes equal members adjacent, so one comparison each is complete.
    var unique: list[Type] = []
    for member in sorted {
        if unique.len() > 0 and compare_types(unique[unique.len() - 1], member) == 0 {
            continue
        }
        unique.push(member)
    }

    if unique.len() == 0 { return u.t_never }
    if unique.len() == 1 { return unique[0] }
    return u.intern(TyKind.Union, null, unique)
}

#: `t | null`, which is what a written `T?` means.
pub fn optional_of(u: Universe, t: Type?) -> Type {
    if t == null { return u.t_any }
    return union_of(u, [t, u.t_null])
}
